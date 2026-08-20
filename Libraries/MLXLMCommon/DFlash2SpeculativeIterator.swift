// Copyright © 2026 Apple Inc.
//
// DFlash2SpeculativeIterator.swift
// mlx-swift-lm
//
// Speculative decoding with a DFlash2 block-parallel drafter
// (https://inco.ai/blog/dflash2/), ported from the reference implementation
// (z-lab/dflash, dflash/model_mlx.py `_stream_generate`).
//
// Each speculation round:
//   1. PROPOSE — the drafter consumes `[anchor, MASK, ...]` plus the target
//      hidden states of the tokens committed since the last round, and
//      predicts the whole block in one parallel pass. DFlash2's selector then
//      traces one path through the per-position top-K candidates.
//   2. VERIFY — the target evaluates `[anchor, draft_1, ..., draft_k]` in a
//      single forward pass, emitting per-position logits, the captured layer
//      hidden states (next round's drafter context), and — on hybrid
//      attention/recurrent models — the per-GDN-layer captures a rollback
//      needs.
//   3. ACCEPT — greedy: longest prefix matching the target's argmax; sampled:
//      lossless rejection sampling against the target's processed
//      distribution (`q` from the selector's candidate probabilities).
//   4. RECONCILE — trim rejected rows from trimmable (attention) caches and
//      replay-restore recurrent (gated-delta) state; feed the drafter's
//      context cache exactly the committed tokens' hidden states.
//
// The output is provably identical to non-speculative decoding of the same
// processed distribution: greedy acceptance compares argmaxes, and rejection
// sampling restores the target distribution exactly.

import Foundation
import MLX

// MARK: - Sampling helpers (ports of model_mlx.py `_sampling_probs` / `_rejection_sample`)

/// Temperature + top-k + top-p probability distribution over logits, in
/// float32. Mirrors the reference `_sampling_probs` step for step.
public func dflash2SamplingProbs(
    _ logits: MLXArray, temperature: Float, topP: Float, topK: Int
) -> MLXArray {
    var scores = logits.asType(.float32) / temperature
    let vocab = scores.dim(-1)
    var indices: MLXArray? = nil
    if topK > 0, topK < vocab {
        let kth = vocab - topK
        let idx = MLX.argPartition(scores, kth: kth, axis: -1)[.ellipsis, kth...]
        scores = MLX.takeAlong(scores, idx, axis: -1)
        indices = idx
    }

    var probs = MLX.softmax(scores, axis: -1)
    if topP < 1.0 {
        let order = MLX.argSort(-probs, axis: -1)
        let sortedProbs = MLX.takeAlong(probs, order, axis: -1)
        let keep = MLX.cumsum(sortedProbs, axis: -1) - sortedProbs .< topP
        let masked = MLX.where(keep, sortedProbs, MLXArray(0, dtype: .float32))
        probs = putAlong(MLX.zeros(like: probs), order, values: masked, axis: -1)
        probs = probs / probs.sum(axis: -1, keepDims: true)
    }

    if let indices {
        return putAlong(MLX.zeros(logits.shape, dtype: .float32), indices, values: probs, axis: -1)
    }
    return probs
}

/// Lossless speculative acceptance under sampling (the reference
/// `_rejection_sample` with `draft_indices` — the DFlash2 selector path).
///
/// - Parameters:
///   - draftTokens: proposed tokens, `[gamma]`.
///   - targetProbs: processed target distributions over the verify input,
///     `[bs, V]` — row i predicts verify-input position i+1.
///   - draftProbs: selector probabilities over the top-K candidates,
///     `[gamma, K]`.
///   - draftCandidates: those candidates, `[gamma, K]`.
/// - Returns: (accepted prefix length 0...gamma, emitted extra token — the
///   residual sample on rejection, the target's free sample when all accept).
func dflash2RejectionSample(
    draftTokens: MLXArray,
    targetProbs: MLXArray,
    draftProbs: MLXArray,
    draftCandidates: MLXArray
) -> (accepted: Int, bonus: Int) {
    let gamma = draftTokens.dim(0)
    // p[i] = targetProbs[i, draftTokens[i]] — gather per row.
    let p = MLX.takeAlong(
        targetProbs[0 ..< gamma, 0...], draftTokens[0..., .newAxis], axis: -1
    )[0..., 0]
    // q per position: probability the selector assigned the drafted token.
    let isDraft = draftCandidates .== draftTokens[0..., .newAxis]  // [gamma, K] bool
    let q = (draftProbs * isDraft.asType(.float32)).sum(axis: -1)  // [gamma]

    let accept = (uniform(0 ..< 1, [gamma]) * q) .< p
    let accepted = MLX.cumprod(accept.asType(.int32), axis: -1).sum().item(Int.self)

    if accepted == gamma {
        let bonusProbs = targetProbs[gamma, 0...]
        return (gamma, categorical(log(bonusProbs)).item(Int.self))
    }

    // Residual distribution: p - q (q scattered at the candidate slots),
    // clamped, normalized; falls back to p when the residual degenerates.
    let candidateSlots = draftCandidates[accepted, 0...]
    let draftRow = draftProbs[accepted, 0...]
    let updated = MLX.takeAlong(targetProbs[accepted, 0...], candidateSlots, axis: -1) - draftRow
    var residual = putAlong(
        targetProbs[accepted, 0...], candidateSlots, values: updated, axis: -1)
    residual = MLX.maximum(residual, MLXArray(0, dtype: .float32))
    let total = residual.sum()
    eval(residual, total)
    if total.item(Float.self) > 0 {
        residual = residual / MLX.maximum(total, MLXArray(Float(1e-30)))
    } else {
        residual = targetProbs[accepted, 0...]
    }
    return (accepted, categorical(log(residual)).item(Int.self))
}

// MARK: - Iterator

/// Token iterator driving DFlash2 block-parallel speculative decoding.
///
/// Unlike ``MTPSpeculativeTokenIterator`` there is no shared-K/V contract and
/// no staged rounds: the drafter owns a private sliding-window context cache
/// fed by the target's hidden states, and the target's hybrid cache is
/// rewound in place (attention trim + recurrent replay). The iterator talks
/// to the target only through ``LMOutput/State`` keys
/// (``dflash2CaptureLayerIdsKey``, ``dflash2GDNCaptureContextKey``), so any
/// target model that answers those keys pairs with any
/// ``DFlash2DrafterModel``.
public struct DFlash2SpeculativeTokenIterator: TokenIteratorProtocol {
    /// The anchor: last emitted token, not yet committed to the target cache.
    var y: LMInput.Text

    let mainModel: any LanguageModel
    let drafter: any DFlash2DrafterModel

    var mainCache: [KVCache]
    var draftCache: [DFlash2ContextCache]
    let gdnCapture = GDNCaptureContext()

    var processor: LogitProcessor?
    let sampler: LogitSampler
    let temperature: Float
    let topP: Float
    let topK: Int

    public let maxTokens: Int?
    /// Total tokens per verify pass: 1 anchor + (blockSize - 1) drafts.
    public let blockSize: Int

    public private(set) var promptPrefillTime: TimeInterval = 0.0

    private var pendingTokens = [Int]()
    private var pendingIndex = 0
    /// Pending tokens committed to `mainCache` by the round that produced
    /// them (`accepted + 1`, independent of any maxTokens clamping).
    /// `finalizeGeneration` rewinds whatever the consumer did not drain.
    private var committedPendingTokenCount = 0

    /// Target positions committed to `mainCache` (prompt + verified tokens).
    private var processedTokens = 0

    /// Hidden rows waiting to feed the next proposal: the target's captured
    /// layer-stack for the tokens committed since the last proposal
    /// (`[B, S, nLayers * H]`). Set by `prepare` and each round.
    private var pendingContextHidden: MLXArray!

    /// Greedy when `temperature == 0` (reference: `_sample_logits` argmaxes).
    private var greedy: Bool { temperature <= 0 }

    private var telemetry = SpeculativeDecodingTelemetry()
    public var speculativeDecodingTelemetry: SpeculativeDecodingTelemetry? {
        telemetry.roundCount > 0 ? telemetry : nil
    }
    public private(set) var acceptedCount = 0
    public private(set) var proposedCount = 0

    private var passthrough = false

    /// Per-phase wall-clock accumulators in seconds (DFLASH2_PROFILE=1). MLX
    /// defers GPU work to sync points, so each profiled phase ends in an
    /// explicit `eval` — the numbers attribute GPU time to the phase that
    /// enqueued it, at the cost of a few extra syncs while profiling.
    public private(set) var profilePhaseSeconds: [String: Double] = [:]
    public private(set) var profileRoundCount = 0
    private let profileEnabled = ProcessInfo.processInfo.environment["DFLASH2_PROFILE"] != nil

    private mutating func profileMark(_ phase: String, since start: ContinuousClock.Instant) {
        let elapsed = ContinuousClock.now - start
        profilePhaseSeconds[phase, default: 0] +=
            Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) / 1e18
    }

    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        drafter: any DFlash2DrafterModel,
        mainCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        blockSize: Int? = nil,
        components: GenerationComponents = .init()
    ) throws {
        let cache = try mainCache ?? mainModel.newCache(parameters: parameters)
        self.y = input.text
        self.mainModel = mainModel
        self.drafter = drafter
        self.mainCache = cache
        self.draftCache = drafter.makeDFlashContextCaches()
        self.sampler = parameters.sampler()
        self.temperature = parameters.temperature
        self.topP = parameters.topP
        self.topK = parameters.topK
        try components.validate(parameters: parameters)
        self.processor = components.logitProcessor(parameters: parameters)
        self.maxTokens = parameters.maxTokens
        self.blockSize = Swift.max(2, blockSize ?? drafter.dflashBlockSize)

        let prefillStart = Date.timeIntervalSinceReferenceDate
        drafter.bindDFlashTarget(mainModel)
        try prepare(input: input, prefill: parameters.prefill)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart
    }

    // MARK: Prefill

    /// Chunked prompt prefill capturing the target's per-layer hidden states
    /// into a rolling window (the drafter's first context). Mirrors
    /// `_prefill_target`: the final prompt position is always its own chunk,
    /// and non-final chunks flush memory.
    mutating func prepare(input: LMInput, prefill: PrefillParameters = .init()) throws {
        processor?.prompt(input.text.tokens)
        let promptTokens = input.text.tokens
        let promptLength = promptTokens.dim(0)
        precondition(promptLength > 0, "DFlash2 iterator requires a non-empty prompt")

        let stepSize = Swift.max(1, prefill.stepSize ?? 2048)
        let keepCount = drafter.dflashContextKeepCount

        var hiddenWindow: MLXArray? = nil
        var hiddenOffset = 0
        var start = 0
        var lastLogits: MLXArray? = nil
        while start < promptLength {
            let remaining = promptLength - start
            let end = start + (remaining == 1 ? 1 : Swift.min(stepSize, remaining - 1))

            var chunkState = LMOutput.State()
            chunkState[dflash2CaptureLayerIdsKey] = drafter.dflashTargetLayerIds
            let chunk = LMInput.Text(tokens: promptTokens[start ..< end])
            let result = mainModel(chunk[text: .newAxis], cache: mainCache, state: chunkState)
            processedTokens += end - start

            guard let captured = result.state?[dflash2CapturedHiddenStatesKey],
                captured.count == drafter.dflashTargetLayerIds.count
            else {
                // Target cannot emit hidden captures: run the rest of the
                // stream without speculation rather than corrupting output.
                finishPrefillWithoutCapture(
                    input: input, from: end, lastLogits: lastLogits ?? result.logits,
                    prefill: prefill)
                return
            }

            var chunkHidden =
                captured.count == 1 ? captured[0] : concatenated(captured, axis: -1)
            if let keepCount {
                if let existing = hiddenWindow {
                    chunkHidden = concatenated([existing, chunkHidden], axis: 1)
                }
                let rows = chunkHidden.dim(1)
                if rows > keepCount {
                    hiddenOffset += rows - keepCount
                    chunkHidden = chunkHidden[0..., (rows - keepCount)..., 0...]
                }
                hiddenWindow = chunkHidden
            } else {
                hiddenWindow =
                    hiddenWindow.map { concatenated([$0, chunkHidden], axis: 1) } ?? chunkHidden
            }

            lastLogits = result.logits
            if end < promptLength {
                eval(mainCache.flatMap { $0.innerState() })
                eval(hiddenWindow!)
                Memory.clearCache()
            }
            prefill.progress?(end, promptLength)
            start = end
        }

        // The drafter's context timeline starts at the first retained row.
        for cache in draftCache {
            cache.offset = hiddenOffset
        }
        pendingContextHidden = hiddenWindow

        // First token from the prompt's final logits.
        var logits = lastLogits![0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: logits)
        processor?.didSample(token: token)
        eval(token)
        y = .init(tokens: token)
        pendingTokens.append(token.item(Int.self))
    }

    /// Capture-incapable target: finish prefill conventionally and mark the
    /// iterator passthrough (single-token decode only from here on).
    private mutating func finishPrefillWithoutCapture(
        input: LMInput, from processed: Int, lastLogits: MLXArray,
        prefill: PrefillParameters
    ) {
        passthrough = true
        let promptTokens = input.text.tokens
        let promptLength = promptTokens.dim(0)
        var start = processed
        var logits = lastLogits
        while start < promptLength {
            let end = Swift.min(promptLength, start + Swift.max(1, prefill.stepSize ?? 2048))
            let chunk = LMInput.Text(tokens: promptTokens[start ..< end])
            let result = mainModel(chunk[text: .newAxis], cache: mainCache, state: nil)
            processedTokens += end - start
            logits = result.logits
            start = end
        }
        var row = logits[0..., -1, 0...]
        row = processor?.process(logits: row) ?? row
        let token = sampler.sample(logits: row)
        processor?.didSample(token: token)
        eval(token)
        y = .init(tokens: token)
        pendingTokens.append(token.item(Int.self))
    }

    // MARK: Rounds

    /// One speculation round: propose → verify → accept → reconcile.
    mutating func speculateRound() {
        guard !passthrough else { return }

        // Round width: blockSize, clamped to the remaining output budget.
        // Reference: `bs = min(block_size, max_tokens - n + 1)` — the verify
        // pass always covers the anchor plus up to `remaining` drafts; the
        // commit is clamped to `remaining` tokens below.
        let numDraft: Int
        if let maxTokens {
            let remaining = maxTokens - tokenCount
            guard remaining > 0 else { return }
            numDraft = Swift.min(blockSize - 1, remaining)
        } else {
            numDraft = blockSize - 1
        }

        let anchorToken = y.tokens
        let anchor = anchorToken.item(Int.self)
        let maskId = drafter.dflashMaskTokenId
        let blockIds = MLXArray([anchor] + Array(repeating: maskId, count: numDraft))
            .expandedDimensions(axis: 0)  // [1, bs]

        let proposeStart = ContinuousClock.now

        // 1. Propose (one parallel draft pass + selector path).
        let (draftTokens, draftCandidates, draftProbs) = drafter.dflashPropose(
            blockIds,
            targetHidden: pendingContextHidden,
            cache: draftCache,
            temperature: temperature,
            logitsStart: 1)

        // Defensive reconcile (mirrors the reference): after the proposal the
        // draft cache timeline must sit at the count of processed positions.
        let draftTrim = draftCache.first.map { $0.offset - processedTokens } ?? 0
        if draftTrim > 0 {
            for cache in draftCache { cache.trimNewest(draftTrim) }
        }
        asyncEval(draftTokens)
        if profileEnabled {
            eval(draftTokens)
            profileMark("propose", since: proposeStart)
        }

        // 2. Verify: one target pass over [anchor, drafts...], capturing the
        // layer hidden states and (hybrid targets) the GDN rollback inputs.
        let verifyStart = ContinuousClock.now
        gdnCapture.clear()
        var verifyState = LMOutput.State()
        verifyState[dflash2CaptureLayerIdsKey] = drafter.dflashTargetLayerIds
        verifyState[dflash2GDNCaptureContextKey] = gdnCapture
        let verifyTokens = concatenated([anchorToken, draftTokens.flattened()])
        let verifyInput = LMInput.Text(tokens: verifyTokens)
        let mainResult = mainModel(
            verifyInput[text: .newAxis], cache: mainCache, state: verifyState)
        let mainLogits = mainResult.logits  // [B, bs, V]: row i predicts position i+1

        guard let captured = mainResult.state?[dflash2CapturedHiddenStatesKey],
            !captured.isEmpty
        else {
            // The verify pass already ran: degrade to an accepted=0 round —
            // keep the anchor's cache rows, emit the target's row-0 sample as
            // a plain decode step, and stop speculating.
            singleTokenFallbackRound(logits: mainLogits, gamma: numDraft)
            return
        }
        let verifyHidden =
            captured.count == 1 ? captured[0] : concatenated(captured, axis: -1)
        if profileEnabled {
            eval(mainLogits)
            profileMark("verify", since: verifyStart)
        }

        // 3. Accept.
        let acceptStart = ContinuousClock.now
        eval(draftTokens)
        let draftList = draftTokens.asArray(Int.self)

        // 3. Accept.
        let accepted: Int
        let bonusTokenValue: Int
        let gamma = draftList.count
        if greedy {
            var processed = mainLogits[0, 0 ..< (gamma + 1), 0...]  // [bs, V]
            processed = processor?.process(logits: processed) ?? processed
            let targetTokens = argMax(processed, axis: -1)  // [bs]
            eval(targetTokens)
            let targetList = targetTokens.asArray(Int.self)

            var acceptedCount = 0
            while acceptedCount < gamma, targetList[acceptedCount] == draftList[acceptedCount] {
                acceptedCount += 1
            }
            accepted = acceptedCount
            bonusTokenValue = targetList[accepted]
            for token in draftList.prefix(accepted) {
                processor?.didSample(token: MLXArray(token))
            }
            processor?.didSample(token: MLXArray(bonusTokenValue))
        } else {
            var processed = mainLogits[0, 0 ..< (gamma + 1), 0...]  // [bs, V]
            processed = processor?.process(logits: processed) ?? processed
            let targetProbs = dflash2SamplingProbs(
                processed, temperature: temperature, topP: topP, topK: topK)
            guard let selectorProbs = draftProbs else {
                preconditionFailure("DFlash2 selector returned no probabilities at T>0")
            }
            let result = dflash2RejectionSample(
                draftTokens: draftTokens.flattened(),
                targetProbs: targetProbs,
                draftProbs: selectorProbs[0],
                draftCandidates: draftCandidates[0])
            accepted = result.accepted
            bonusTokenValue = result.bonus
            for token in draftList.prefix(accepted) {
                processor?.didSample(token: MLXArray(token))
            }
            processor?.didSample(token: MLXArray(bonusTokenValue))
        }
        if profileEnabled { profileMark("accept", since: acceptStart) }

        // 4. Reconcile caches: `accepted + 1` verify positions stay committed
        // (anchor + accepted drafts); the rest are rolled back.
        let reconcileStart = ContinuousClock.now
        let rejected = gamma - accepted
        processedTokens += accepted + 1
        if rejected > 0 {
            rollbackSpeculativeHybridCaches(
                mainCache, context: gdnCapture, accepted: accepted, rejected: rejected)
        }

        // The drafter's next context: hidden states of the committed verify
        // positions (anchor + accepted drafts).
        pendingContextHidden = verifyHidden[0..., 0 ... accepted, 0...]

        var newTokens = [Int](draftList.prefix(accepted))
        newTokens.append(bonusTokenValue)
        // Clamp to the output budget. The cache committed the unclamped
        // count; `finalizeGeneration` rewinds the unemitted remainder.
        if let maxTokens {
            let remaining = maxTokens - tokenCount
            if newTokens.count > remaining {
                newTokens = Array(newTokens.prefix(remaining))
            }
        }
        committedPendingTokenCount = accepted + 1
        pendingTokens.append(contentsOf: newTokens)

        proposedCount += gamma
        acceptedCount += accepted
        telemetry.recordRound(
            drafted: gamma, accepted: accepted, targetVerified: gamma + 1, draftModelCalls: 1)

        if profileEnabled {
            eval(pendingContextHidden)
            profileMark("reconcile", since: reconcileStart)
            profileRoundCount += 1
        }
        y = .init(tokens: MLXArray([bonusTokenValue]))
    }

    /// A verify pass whose target emitted no captures: salvage it as a plain
    /// single-token step (anchor committed, row-0 sample emitted), roll the
    /// speculative tail back, and stop speculating for this stream.
    private mutating func singleTokenFallbackRound(logits: MLXArray, gamma: Int) {
        var row = logits[0..., 0, 0...]
        row = processor?.process(logits: row) ?? row
        let token = sampler.sample(logits: row)
        processor?.didSample(token: token)
        eval(token)
        let value = token.item(Int.self)

        processedTokens += 1
        if gamma > 0 {
            rollbackSpeculativeHybridCaches(
                mainCache, context: gdnCapture, accepted: 0, rejected: gamma)
        }
        pendingTokens.append(value)
        committedPendingTokenCount = 1
        y = .init(tokens: token)
        switchToPassthrough()
    }

    private mutating func switchToPassthrough() {
        passthrough = true
    }

    /// One plain target step (passthrough mode only; a passthrough target is
    /// capture-incapable, so the step is not rewindable).
    private mutating func passthroughStep() -> Int? {
        if let maxTokens, tokenCount >= maxTokens { return nil }
        let result = mainModel(y[text: .newAxis], cache: mainCache, state: nil)
        processedTokens += 1
        var logits = result.logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: logits)
        processor?.didSample(token: token)
        eval(token)
        let value = token.item(Int.self)
        y = .init(tokens: token)
        return value
    }

    // MARK: TokenIteratorProtocol

    public var tokenCount: Int { telemetry.emittedTokenCount }

    public mutating func discardGeneratedToken() {
        telemetry.discardGeneratedToken()
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        if pendingIndex < pendingTokens.count {
            let token = pendingTokens[pendingIndex]
            pendingIndex += 1
            telemetry.recordGeneratedToken()
            return token
        }

        if passthrough {
            if let token = passthroughStep() {
                telemetry.recordGeneratedToken()
                return token
            }
            return nil
        }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        committedPendingTokenCount = 0
        autoreleasepool { speculateRound() }

        if pendingTokens.isEmpty {
            if passthrough, let token = passthroughStep() {
                telemetry.recordGeneratedToken()
                return token
            }
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        telemetry.recordGeneratedToken()
        return token
    }
}

extension DFlash2SpeculativeTokenIterator: GenerationFinalizingTokenIterator {
    public mutating func finalizeGeneration() {
        // Rewind verify-pass tokens the consumer never drained. The consumer
        // drained `pendingIndex` of this round's `committedPendingTokenCount`
        // tokens; pending token j covers verify-input position j + 1, so the
        // cache must retain `consumed + 1` verify positions (the anchor plus
        // the drained tokens).
        let consumed = Swift.min(pendingIndex, committedPendingTokenCount)
        let lookahead = committedPendingTokenCount - consumed
        guard lookahead > 0 else { return }
        // Rows to drop = committed (accepted+1) minus retained (consumed+1).
        let rewind = lookahead - 1
        guard rewind > 0 else { return }
        if gdnCapture.captures.isEmpty {
            // Uncaptured step (pure-attention target, or a passthrough step
            // that could not record): trim-only rollback.
            for cache in mainCache where cache.isTrimmable {
                cache.trim(rewind)
            }
        } else {
            rollbackSpeculativeHybridCaches(
                mainCache, context: gdnCapture,
                accepted: consumed, rejected: rewind)
        }
        processedTokens -= rewind
    }
}

extension DFlash2SpeculativeTokenIterator: MTPStatsCollecting {
    public var proposedDraftTokens: Int { proposedCount }
    public var acceptedDraftTokens: Int { acceptedCount }
    public var passthroughReason: String? {
        passthrough ? "target cannot emit DFlash2 capture state" : nil
    }
}
