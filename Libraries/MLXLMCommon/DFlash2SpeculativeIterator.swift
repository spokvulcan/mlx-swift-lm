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
    /// The anchor's token id on the host. The accept step already lands the
    /// bonus token host-side, so tracking it here removes a per-round D2H
    /// sync at proposal build time. Kept in lockstep with `y` everywhere
    /// `y` is assigned.
    private var dflash2AnchorValue: Int = -1

    let mainModel: any LanguageModel
    let drafter: any DFlash2DrafterModel

    var mainCache: [KVCache]
    var draftCache: [DFlash2ContextCache]
    let gdnCapture = GDNCaptureContext()

    /// Stream n-gram index feeding the selector's path advisor (prompt +
    /// committed tokens). The advisor forces a drafted candidate when the
    /// stream's own longest-suffix continuation is long and dominant enough
    /// — self-similar content (code edits, quoting summaries, structured
    /// output) resolves the near-ties the learned selector misses. Drafting
    /// signal only; verification still gates every emitted token.
    private let ngramIndex = DFlash2NGramIndex()
    /// Trailing committed tokens (anchor last) — the advisor's context head.
    private var ngramTail: [Int] = []
    /// `DFLASH2_SELECTOR=advised` opts into the host-side walk + n-gram
    /// force; `host` runs the host walk with the force disabled. Default is
    /// the on-GPU walk: the 2026-08-22 A/B (ledger R21) measured the
    /// advised path acceptance-NEUTRAL on canonical content (early-death
    /// conversions get absorbed by phase-shifted deaths deeper in the
    /// chain) while its mid-round sync costs real round time — kept as an
    /// experimental hatch, not the production path.
    private static let advisedSelectorEnabled: Bool = {
        let mode = ProcessInfo.processInfo.environment["DFLASH2_SELECTOR"]
        return mode == "advised" || mode == "host"
    }()
    private static let ngramForceEnabled =
        ProcessInfo.processInfo.environment["DFLASH2_SELECTOR"] == "advised"
    /// Force thresholds: the continuation must come from an order-4+ match
    /// and carry >= 60% of the context's observed continuations.
    private static let ngramMinOrder = 4
    private static let ngramMinShare = 0.6

    private mutating func ngramCommit(_ tokens: [Int]) {
        ngramIndex.extend(tokens)
        ngramTail.append(contentsOf: tokens)
        if ngramTail.count > 8 {
            ngramTail.removeFirst(ngramTail.count - 8)
        }
    }

    var processor: LogitProcessor?
    let sampler: LogitSampler
    let temperature: Float
    let topP: Float
    let topK: Int

    public let maxTokens: Int?
    /// Total tokens per verify pass at the current policy width: 1 anchor +
    /// (width - 1) drafts. The init ``blockSize`` is the CAP (the draft
    /// checkpoint's trained width); the policy narrows on measured speed.
    public let blockSize: Int

    /// Adaptive-width bandit (the reference runs an acceptance-threshold
    /// policy; on this stack per-width cost varies enough that the objective
    /// has to be measured speed, not acceptance): the candidate shortlist is
    /// {3, 4, cap} — mma8 makes verify near-flat across 5..8, so the optimum
    /// is bracketed by the floor, the mid, and the cap. An 8-round window per
    /// width (~0.5 s) scores decode tok/s; the stream settles on the argmax
    /// with 3% hysteresis. After 24 settled windows every score is dropped so
    /// the shortlist re-walks — that drift re-sweep tracks content drift.
    /// Exploration is once per stream (~16 rounds) plus ~2 windows per
    /// re-sweep.
    private var roundWidth: Int = 0
    private var widthWindowRounds = 0
    private var widthWindowTokens = 0
    private var widthWindowStart: ContinuousClock.Instant?
    private var widthScores = [Int: Double]()
    private var windowsSinceSettle = 0
    private let adaptiveWidth: Bool

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

    // MARK: Accept-log instrumentation (DFLASH2_ACCEPT_LOG=1)

    /// Measurement mode for the greedy path: per-position acceptance,
    /// selector-candidate coverage, and a host-phase wall-clock timeline.
    /// Unlike DFLASH2_PROFILE it adds NO evals — the candidate ids ride the
    /// round's existing packed D2H transfer, and the timeline stamps are
    /// pure host clock reads — so round structure matches production.
    private let acceptLogEnabled =
        ProcessInfo.processInfo.environment["DFLASH2_ACCEPT_LOG"] != nil

    /// DFLASH2_HOST_PROFILE=1: the accept-log wall-clock timeline WITHOUT the
    /// candidate instrumentation, so the compiled selector walk stays engaged
    /// (DFLASH2_ACCEPT_LOG forces the eager walk for rank capture) and the
    /// round structure is exactly production. Pure clock reads, zero evals.
    private let hostProfileEnabled =
        ProcessInfo.processInfo.environment["DFLASH2_HOST_PROFILE"] != nil
    private var hostTimelineEnabled: Bool { acceptLogEnabled || hostProfileEnabled }
    private var hostTimelineRounds = 0

    // MARK: Pipelined propose (DFLASH2_PIPELINE=1)

    /// Build the next round's proposal during this round's verify sync: the
    /// accept-dependent inputs ride as lazy arrays (see
    /// ``DFlash2PipelinedDrafter``), so the propose host cost (graph splice +
    /// schedule, ~11 ms/round measured) leaves the round's critical path and
    /// the draft pass queues right behind the verify on the GPU. Greedy-only;
    /// measurement modes that reshape the round (accept-log, lattice dump,
    /// advised selector) fall back to the synchronous propose. Default ON;
    /// `DFLASH2_PIPELINE=0` restores the synchronous round.
    private static let pipelineEnabled =
        ProcessInfo.processInfo.environment["DFLASH2_PIPELINE"] != "0"
    private static let prebuiltEagerSchedule =
        ProcessInfo.processInfo.environment["DFLASH2_PIPELINE_SCHED"] != "0"
    /// Accept-invariant reconcile prebuild (stage 2a): build the GDN rollback
    /// replay as a masked full-width graph in the same sync window, so the
    /// reconcile splice (~2.9 ms/round measured) also leaves the critical
    /// path. Default ON; `DFLASH2_ROLLBACK_PREBUILD=0` restores the
    /// synchronous accepted-prefix replay.
    private static let rollbackPrebuildEnabled =
        ProcessInfo.processInfo.environment["DFLASH2_ROLLBACK_PREBUILD"] != "0"
    /// Accept-invariant verify prebuild (stage 2): build the NEXT round's
    /// whole verify pass (and its packed sync array) inside this round's sync
    /// window and schedule it behind the in-flight verify, so the GPU never
    /// drains between rounds. Every accept-dependent input is lazy: tokens
    /// from the prebuilt draft, KV write offset via a dynamic slice update,
    /// RoPE as a `[1]` array, GDN initial states from the stage-2a replay,
    /// SDPA visibility as one lazy bool mask. Greedy, `processor == nil`
    /// rounds only. Default ON; `DFLASH2_VERIFY_PREBUILD=0` restores the
    /// synchronous verify build.
    private static let verifyPrebuildEnabled =
        ProcessInfo.processInfo.environment["DFLASH2_VERIFY_PREBUILD"] != "0"

    /// A verify pass built one round ahead (stage 2): everything the accept
    /// and reconcile of that round will need, all lazy, already scheduled.
    private struct DFlash2PrebuiltVerify {
        var logits: MLXArray
        var hidden: MLXArray
        var capture: GDNCaptureContext
        var packed: MLXArray
        var targetTokens: MLXArray
        var width: Int
    }

    private var prebuiltVerify: DFlash2PrebuiltVerify?
    /// Capture context of the round whose rows the caches currently hold —
    /// `finalizeGeneration` replays it to rewind undrained tokens. For
    /// synchronous rounds this is the shared `gdnCapture`; stage-2 rounds
    /// carry their own per-build context.
    private var lastRoundCapture: GDNCaptureContext?
    private var prebuiltNext: DFlash2PipelinedProposal?
    /// RoPE/timeline position of the CURRENT round's anchor token, once the
    /// pipelined chain is running (tracked host-side; the cache write cursor
    /// stops advancing on pipelined rounds).
    private var pipelineAnchorPos = 0
    /// Accept count of the previous round — resolves the staged rows'
    /// validity when a prebuilt proposal is adopted.
    private var lastAccepted = -1
    private var alRounds = 0
    private var alAcceptHist: [Int: Int] = [:]
    private var alPosMatch: [Int: Int] = [:]
    private var alPosSurvive: [Int: Int] = [:]
    private var alPosCover: [Int: Int] = [:]
    private var alPosCover2: [Int: Int] = [:]
    private var alPosCover4: [Int: Int] = [:]
    private var alPosDenom: [Int: Int] = [:]
    private var alFirstMissCovered = 0
    private var alFirstMissRounds = 0
    /// Rank (0-based, score-ordered) of the target token at the first-miss
    /// position; k = not in the candidate set.
    private var alFirstMissRank: [Int: Int] = [:]
    private var alTimelineMs: [String: Double] = [:]
    private var alLastRoundEnd: ContinuousClock.Instant?
    private var alSummaryEmitted = false

    private static func alMs(_ duration: Duration) -> Double {
        Double(duration.components.seconds) * 1e3
            + Double(duration.components.attoseconds) / 1e15
    }

    /// DFLASH2_LATTICE_DUMP companion stream: per greedy round, the verify
    /// targets and accept outcome (joined offline with the selector's
    /// lattice lines by round index).
    private static let latticeDumpPath =
        ProcessInfo.processInfo.environment["DFLASH2_LATTICE_DUMP"]
    nonisolated(unsafe) private static var latticeAcceptRound = 0

    private static func dumpAccept(
        to file: String, targets: [Int], drafts: [Int], accepted: Int
    ) {
        var line = "{\"r\":\(latticeAcceptRound),\"targets\":\(targets)"
        line += ",\"drafts\":\(drafts),\"accepted\":\(accepted)}\n"
        latticeAcceptRound += 1
        dumpLine(to: file, line: line)
    }

    private static func dumpLine(to file: String, line: String) {
        if let handle = FileHandle(forWritingAtPath: file) {
            handle.seekToEndOfFile()
            handle.write(Data(line.utf8))
            try? handle.close()
        } else {
            FileManager.default.createFile(
                atPath: file, contents: Data(line.utf8))
        }
    }

    private mutating func alStamp(_ phase: String, since start: ContinuousClock.Instant) {
        alTimelineMs[phase, default: 0] += Self.alMs(ContinuousClock.now - start)
    }

    /// Fold one greedy round into the accept log. `candidates` is the
    /// selector's flattened top-K id list (`gamma * K`), from the packed
    /// transfer.
    private mutating func alRecordRound(
        gamma: Int, accepted: Int, drafts: [Int], targets: [Int], candidates: [Int]
    ) {
        alRounds += 1
        alAcceptHist[accepted, default: 0] += 1
        let k = gamma > 0 ? candidates.count / gamma : 0
        for i in 0 ..< gamma {
            alPosDenom[i, default: 0] += 1
            if targets[i] == drafts[i] { alPosMatch[i, default: 0] += 1 }
            if i < accepted { alPosSurvive[i, default: 0] += 1 }
            if k > 0 {
                // Candidates arrive score-ordered (selector measurement
                // mode), so the index of the target IS its selector rank.
                let row = candidates[(i * k) ..< ((i + 1) * k)]
                if let r = row.firstIndex(of: targets[i]) {
                    let rank = r - i * k
                    alPosCover[i, default: 0] += 1
                    if rank < 2 { alPosCover2[i, default: 0] += 1 }
                    if rank < 4 { alPosCover4[i, default: 0] += 1 }
                }
            }
        }
        if accepted < gamma, k > 0 {
            alFirstMissRounds += 1
            let i = accepted
            let row = candidates[(i * k) ..< ((i + 1) * k)]
            if let r = row.firstIndex(of: targets[i]) {
                alFirstMissCovered += 1
                alFirstMissRank[r - i * k, default: 0] += 1
            } else {
                alFirstMissRank[k, default: 0] += 1
            }
        }
    }

    private mutating func alEmitSummaryOnce() {
        guard hostTimelineEnabled, !alSummaryEmitted, hostTimelineRounds > 0 else { return }
        alSummaryEmitted = true
        func emit(_ line: String) {
            FileHandle.standardOutput.write(Data((line + "\n").utf8))
        }
        let phases = [
            "gap", "propose", "p-splice", "p-sched", "build", "prebuild", "pb-vsched",
            "pb-graph", "pb-vbuild", "pb-dsched", "sync", "accept", "reconcile",
        ]
        let perRound = phases.map { phase in
            String(
                format: "%@ %.2f", phase, (alTimelineMs[phase] ?? 0) / Double(hostTimelineRounds))
        }.joined(separator: " | ")
        emit("[dflash2-accept] timeline ms/round: \(perRound) (n=\(hostTimelineRounds))")
        guard acceptLogEnabled, alRounds > 0 else { return }
        let maxGamma = (alPosDenom.keys.max() ?? -1) + 1
        let hist = (0 ... maxGamma)
            .map { "\($0):\(alAcceptHist[$0] ?? 0)" }.joined(separator: " ")
        emit("[dflash2-accept] rounds=\(alRounds) hist \(hist)")
        func rates(_ table: [Int: Int]) -> String {
            (0 ..< maxGamma).map { i in
                let d = alPosDenom[i] ?? 0
                return d > 0
                    ? String(format: "%.2f", Double(table[i] ?? 0) / Double(d)) : "-"
            }.joined(separator: "/")
        }
        emit("[dflash2-accept] match \(rates(alPosMatch))")
        emit("[dflash2-accept] survive \(rates(alPosSurvive))")
        emit("[dflash2-accept] cover2 \(rates(alPosCover2))")
        emit("[dflash2-accept] cover4 \(rates(alPosCover4))")
        emit("[dflash2-accept] cover16 \(rates(alPosCover))")
        emit("[dflash2-accept] first-miss-covered \(alFirstMissCovered)/\(alFirstMissRounds)")
        if !alFirstMissRank.isEmpty {
            let ranks = alFirstMissRank.keys.sorted()
                .map { "r\($0):\(alFirstMissRank[$0]!)" }.joined(separator: " ")
            emit("[dflash2-accept] first-miss-rank \(ranks)")
        }
    }

    /// - Parameters:
    ///   - input: the full prompt to decode from (text tokens only).
    ///   - mainModel: the target model whose output the speculation must
    ///     reproduce; it also supplies the hidden states the drafter reads.
    ///   - drafter: the DFlash2 draft model bound to `mainModel`'s family.
    ///   - mainCache: an existing target cache to decode over. Pass a warm
    ///     cache (a restored prefix-cache checkpoint, or a caller-side
    ///     chunked prefill) together with `prefilledPrefixTokens` to start
    ///     speculation without re-processing the prefix.
    ///   - prefilledPrefixTokens: leading positions of `input` that
    ///     `mainCache` already holds. The capture prefill runs only over the
    ///     remaining suffix, so the drafter's context window starts with the
    ///     suffix rows alone (hidden states are never stored with a KV
    ///     prefix); RoPE positions stay absolute, so the math for the suffix
    ///     and every decode round is identical to a cold run of the same
    ///     timeline. The logit processor is still primed with the full
    ///     prompt.
    ///   - parameters: sampling and generation limits.
    ///   - blockSize: round-width cap (1 anchor + `blockSize - 1` drafts);
    ///     `nil` uses the drafter's trained block size.
    ///   - adaptiveWidth: narrow the round width on rejection, re-widen on
    ///     acceptance, under the `blockSize` cap.
    ///   - components: caller-supplied generation hooks (e.g. an app logit
    ///     processor replacing the parameter-built penalty processor).
    public init(
        input: LMInput,
        mainModel: any LanguageModel,
        drafter: any DFlash2DrafterModel,
        mainCache: [KVCache]? = nil,
        prefilledPrefixTokens: Int = 0,
        parameters: GenerateParameters,
        blockSize: Int? = nil,
        adaptiveWidth: Bool = true,
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
        self.roundWidth = self.blockSize
        self.adaptiveWidth = adaptiveWidth

        let prefillStart = Date.timeIntervalSinceReferenceDate
        drafter.bindDFlashTarget(mainModel)
        try prepare(
            input: input, prefilledPrefixTokens: prefilledPrefixTokens,
            prefill: parameters.prefill)
        self.promptPrefillTime = Date.timeIntervalSinceReferenceDate - prefillStart
    }

    // MARK: Prefill

    /// Chunked prompt prefill capturing the target's per-layer hidden states
    /// into a rolling window (the drafter's first context). Mirrors
    /// `_prefill_target`: the final prompt position is always its own chunk,
    /// and non-final chunks flush memory.
    ///
    /// With `prefilledPrefixTokens > 0` the loop starts past the positions
    /// `mainCache` already holds (a warm prefix-cache restore, or the
    /// caller's own checkpoint-capturing prefill): only the suffix is
    /// forwarded and captured, and the drafter's context timeline anchors at
    /// the suffix start — the same absolute positions a cold run would give
    /// those rows.
    mutating func prepare(
        input: LMInput, prefilledPrefixTokens: Int = 0,
        prefill: PrefillParameters = .init()
    ) throws {
        processor?.prompt(input.text.tokens)
        let promptTokens = input.text.tokens
        let promptLength = promptTokens.dim(0)
        precondition(
            prefilledPrefixTokens >= 0 && prefilledPrefixTokens < promptLength,
            "DFlash2 iterator requires at least one unprefilled prompt token")
        if prefilledPrefixTokens > 0,
            let attention = mainCache.first(where: { $0.isTrimmable })
        {
            precondition(
                attention.offset == prefilledPrefixTokens,
                """
                DFlash2 warm start: cache offset \(attention.offset) does not \
                match prefilledPrefixTokens \(prefilledPrefixTokens)
                """)
        }
        if let dumpPath = Self.latticeDumpPath {
            let ids = promptTokens.asArray(Int32.self).map(Int.init)
            Self.dumpLine(
                to: dumpPath + ".accept", line: "{\"r\":-1,\"prompt\":\(ids)}\n")
        }
        if Self.advisedSelectorEnabled {
            ngramCommit(promptTokens.asArray(Int32.self).map(Int.init))
        }

        let stepSize = Swift.max(1, prefill.stepSize ?? 2048)
        let keepCount = drafter.dflashContextKeepCount

        var hiddenWindow: MLXArray? = nil
        var hiddenOffset = prefilledPrefixTokens
        var start = prefilledPrefixTokens
        processedTokens = prefilledPrefixTokens
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
        let firstValue = token.item(Int.self)
        dflash2AnchorValue = firstValue
        pendingTokens.append(firstValue)
        if Self.advisedSelectorEnabled {
            ngramCommit([firstValue])
        }
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
        let firstValue = token.item(Int.self)
        dflash2AnchorValue = firstValue
        pendingTokens.append(firstValue)
    }

    // MARK: Rounds

    /// One speculation round: propose → verify → accept → reconcile.
    mutating func speculateRound() {
        guard !passthrough else { return }

        let alRoundStart = ContinuousClock.now
        if hostTimelineEnabled, let last = alLastRoundEnd {
            alTimelineMs["gap", default: 0] += Self.alMs(alRoundStart - last)
        }

        // Round width: the policy's current width, clamped to the remaining
        // output budget. Reference: `bs = min(block_size, max_tokens - n + 1)`
        // — the verify pass always covers the anchor plus up to `remaining`
        // drafts; the commit is clamped to `remaining` tokens below.
        let numDraft: Int
        if let maxTokens {
            let remaining = maxTokens - tokenCount
            guard remaining > 0 else { return }
            numDraft = Swift.min(roundWidth - 1, remaining)
        } else {
            numDraft = roundWidth - 1
        }

        // Host-side anchor (no D2H): the block is built entirely from host
        // values; the anchor's MLXArray for the verify pass is an H2D upload.
        let anchor = dflash2AnchorValue
        precondition(anchor >= 0, "DFlash2 round ran before prefill set an anchor")
        let anchorToken = MLXArray([Int32(anchor)])
        let maskId = drafter.dflashMaskTokenId

        let proposeStart = ContinuousClock.now

        // 1. Propose. A prebuilt (pipelined) proposal from the previous
        // round's sync window is used whenever present — its accept-dependent
        // inputs were lazy, so it is correct for the accept outcome that
        // materialized; adopting the staged caches commits the appended
        // verify rows, whose validity resolves with the now-known accept
        // count. A width switched by the bandit takes effect one round late
        // (the prebuilt width governs this round).
        let (draftTokens, draftCandidates, draftProbs): (MLXArray, MLXArray, MLXArray?)
        var roundGamma = numDraft
        var usedPrebuilt = false
        if let pre = prebuiltNext {
            prebuiltNext = nil
            for (cache, clone) in zip(draftCache, pre.stagedCaches) {
                cache.adopt(clone)
                cache.resolveNewestValidity(
                    newestCount: pre.appendedRows, validCount: lastAccepted + 1)
                cache.offset = pipelineAnchorPos
            }
            roundGamma = pre.width - 1
            (draftTokens, draftCandidates, draftProbs) = (pre.tokens, pre.candidates, nil)
            usedPrebuilt = true
        } else if Self.advisedSelectorEnabled, greedy {
            let index = ngramIndex
            let tail = ngramTail
            let forceEnabled = Self.ngramForceEnabled
            let advisor: DFlash2PathAdvisor = { chosen, candidates in
                guard
                    forceEnabled,
                    let prediction = index.predict(context: tail + chosen),
                    prediction.order >= Self.ngramMinOrder,
                    prediction.share >= Self.ngramMinShare,
                    candidates.contains(prediction.token)
                else { return nil }
                return prediction.token
            }
            let blockIds = MLXArray([anchor] + Array(repeating: maskId, count: numDraft))
                .expandedDimensions(axis: 0)  // [1, bs]
            (draftTokens, draftCandidates, draftProbs) = drafter.dflashProposeAdvised(
                blockIds,
                targetHidden: pendingContextHidden,
                cache: draftCache,
                temperature: temperature,
                logitsStart: 1,
                pathAdvisor: advisor)
        } else {
            let blockIds = MLXArray([anchor] + Array(repeating: maskId, count: numDraft))
                .expandedDimensions(axis: 0)  // [1, bs]
            (draftTokens, draftCandidates, draftProbs) = drafter.dflashPropose(
                blockIds,
                targetHidden: pendingContextHidden,
                cache: draftCache,
                temperature: temperature,
                logitsStart: 1)
        }

        // Defensive reconcile (mirrors the reference): after the proposal the
        // draft cache timeline must sit at the count of processed positions.
        // Prebuilt rounds skip it — their appended rows are placeholders the
        // validity mask governs, and the tokens were scheduled last round.
        if !usedPrebuilt {
            let draftTrim = draftCache.first.map { $0.offset - processedTokens } ?? 0
            if draftTrim > 0 {
                for cache in draftCache { cache.trimNewest(draftTrim) }
            }
            if hostTimelineEnabled { alStamp("p-splice", since: proposeStart) }
            let scheduleStart = ContinuousClock.now
            asyncEval(draftTokens)
            if hostTimelineEnabled { alStamp("p-sched", since: scheduleStart) }
        }
        if profileEnabled {
            eval(draftTokens)
            profileMark("propose", since: proposeStart)
        }
        if hostTimelineEnabled { alStamp("propose", since: alRoundStart) }
        let alBuildStart = ContinuousClock.now

        // 2. Verify: one target pass over [anchor, drafts...], capturing the
        // layer hidden states and (hybrid targets) the GDN rollback inputs.
        // A stage-2 prebuilt verify (built and scheduled in the previous
        // round's sync window from lazy accept-dependent inputs) replaces the
        // whole build; it is correct for whatever accept materialized.
        let verifyStart = ContinuousClock.now
        let mainLogits: MLXArray
        let verifyHidden: MLXArray
        let roundCapture: GDNCaptureContext
        let consumedVerify = prebuiltVerify
        prebuiltVerify = nil
        if let pv = consumedVerify {
            mainLogits = pv.logits
            verifyHidden = pv.hidden
            roundCapture = pv.capture
        } else {
            gdnCapture.clear()
            var verifyState = LMOutput.State()
            verifyState[dflash2CaptureLayerIdsKey] = drafter.dflashTargetLayerIds
            verifyState[dflash2GDNCaptureContextKey] = gdnCapture
            let verifyTokens = concatenated([anchorToken, draftTokens.flattened()])
            let verifyInput = LMInput.Text(tokens: verifyTokens)
            let mainResult = mainModel(
                verifyInput[text: .newAxis], cache: mainCache, state: verifyState)
            // [B, bs, V]: row i predicts position i+1
            let logits = mainResult.logits

            guard let captured = mainResult.state?[dflash2CapturedHiddenStatesKey],
                !captured.isEmpty
            else {
                // The verify pass already ran: degrade to an accepted=0 round —
                // keep the anchor's cache rows, emit the target's row-0 sample
                // as a plain decode step, and stop speculating.
                singleTokenFallbackRound(logits: logits, gamma: roundGamma)
                return
            }
            mainLogits = logits
            verifyHidden =
                captured.count == 1 ? captured[0] : concatenated(captured, axis: -1)
            roundCapture = gdnCapture
        }
        if profileEnabled {
            eval(mainLogits)
            profileMark("verify", since: verifyStart)
        }

        // 3. Accept.
        let acceptStart = ContinuousClock.now

        // 3. Accept.
        let accepted: Int
        let bonusTokenValue: Int
        let gamma = roundGamma
        let draftList: [Int]
        var alAcceptFrom = acceptStart
        // This round's anchor position (the RoPE position of `anchorToken` in
        // the target timeline) — the base the pipelined prebuild appends the
        // verify rows at. Old-path rounds read it from the just-advanced
        // cache write cursor; prebuilt rounds track it explicitly.
        let roundAnchorPos = usedPrebuilt ? pipelineAnchorPos : (draftCache.first?.offset ?? 0)
        var nextPrebuilt: DFlash2PipelinedProposal? = nil
        var prebuiltRollback: [(state: MLXArray, conv: MLXArray)]? = nil
        var nextVerify: DFlash2PrebuiltVerify? = nil
        if greedy {
            let targetTokens: MLXArray
            let packed: MLXArray
            if let pv = consumedVerify {
                // Prebuilt round: the packed transfer (and the argmax inside
                // it) were built AND scheduled in the previous sync window.
                targetTokens = pv.targetTokens
                packed = pv.packed
            } else {
                var processed = mainLogits[0, 0 ..< (gamma + 1), 0...]  // [bs, V]
                processed = processor?.process(logits: processed) ?? processed
                targetTokens = argMax(processed, axis: -1).asType(.int32)  // [bs]

                // ONE D2H sync per round: draft ids and target argmax ids ship
                // in a single packed transfer (the reference's single-sync
                // cycle). Accept-log mode appends the selector's top-K
                // candidate ids to the same transfer (coverage measurement,
                // no extra sync).
                var packedParts = [draftTokens.flattened().asType(.int32), targetTokens]
                if acceptLogEnabled {
                    packedParts.append(draftCandidates.flattened().asType(.int32))
                }
                packed = concatenated(packedParts)
            }
            if hostTimelineEnabled { alStamp("build", since: alBuildStart) }

            // Pipelined prebuild: start the verify pass on the GPU, then use
            // its idle host window to build the NEXT round's proposal with
            // the accept-dependent values as lazy arrays (accepted count via
            // cumulative-product prefix match, bonus anchor via a lazy take).
            // Scheduling the prebuilt tokens queues the draft pass right
            // behind the verify on the GPU stream.
            if Self.pipelineEnabled, !acceptLogEnabled, !Self.advisedSelectorEnabled,
                Self.latticeDumpPath == nil,
                let pipelined = drafter as? DFlash2PipelinedDrafter
            {
                let prebuildStart = ContinuousClock.now
                // A prebuilt round's packed transfer was scheduled (with its
                // whole verify pass) in the previous window.
                if consumedVerify == nil { asyncEval(packed) }
                // Sub-stamps discriminate throttled waiting (the encode thread
                // paces to the GPU past MAX_ACTIVE_TASKS in-flight command
                // buffers) from genuine host graph-build work in this window.
                if hostTimelineEnabled { alStamp("pb-vsched", since: prebuildStart) }
                let pbGraphStart = ContinuousClock.now
                let eq = (draftTokens.flattened().asType(.int32) .== targetTokens[0 ..< gamma])
                    .asType(.int32)
                let acceptedLazy = cumprod(eq).sum()
                let validCountLazy = acceptedLazy + 1
                let bonusLazy = takeAlong(targetTokens, acceptedLazy.reshaped([1]), axis: 0)
                let nextWidth = roundWidth
                let blockIdsLazy = concatenated([
                    bonusLazy.asType(.int32),
                    MLXArray(Array(repeating: Int32(maskId), count: nextWidth - 1)),
                ]).expandedDimensions(axis: 0)
                let blockPosOffset =
                    MLXArray([Int32(roundAnchorPos)]) + validCountLazy.asType(.int32)
                nextPrebuilt = pipelined.dflashProposePipelined(
                    blockIds: blockIdsLazy,
                    targetHidden: verifyHidden,
                    validCount: validCountLazy,
                    contextPositionBase: roundAnchorPos,
                    blockPositionOffset: blockPosOffset,
                    caches: draftCache)
                // Accept-invariant rollback for THIS round's reconcile: the
                // masked replay equals the accepted-prefix replay for every
                // outcome, so its graph splices here in the sync window.
                // Left unscheduled — an all-accepted round drops it unrun.
                if Self.rollbackPrebuildEnabled {
                    prebuiltRollback = prebuildSpeculativeRollback(
                        context: roundCapture, validCount: validCountLazy)
                }
                if hostTimelineEnabled { alStamp("pb-graph", since: pbGraphStart) }
                let pbVBuildStart = ContinuousClock.now
                // Stage-2: build the NEXT round's verify pass here, entirely
                // from lazy accept-dependent inputs, so its graph (and the
                // sync transfer) can be scheduled behind the in-flight verify
                // and the GPU never drains between rounds.
                if Self.verifyPrebuildEnabled, processor == nil,
                    let np = nextPrebuilt, let pr = prebuiltRollback
                {
                    // GDN initial states for the next pass = the
                    // accept-invariant replay (on full acceptance bitwise the
                    // committed states — same kernel, same captured inputs;
                    // rejected>0 rounds already take exactly this path).
                    applyGDNRollback(mainCache, prebuilt: pr)
                    let nextCapture = GDNCaptureContext()
                    let nextWidthS = np.width
                    let worstLen = roundAnchorPos + (gamma + 1) + nextWidthS
                    // One lazy comparison encodes in-block causality, history
                    // visibility, and stale-row exclusion: col j is visible
                    // to row i iff j < start + i + 1.
                    let cols = MLXArray(Int32(0) ..< Int32(worstLen))
                        .expandedDimensions(axis: 0)
                    let rowIdx = MLXArray(Int32(0) ..< Int32(nextWidthS))
                        .reshaped([nextWidthS, 1])
                    let attnMask = cols .< (blockPosOffset + rowIdx + 1)
                    let plan = DFlash2PipelinedVerifyPlan(
                        kvStart: blockPosOffset, worstLen: worstLen, attnMask: attnMask)
                    var vState = LMOutput.State()
                    vState[dflash2CaptureLayerIdsKey] = drafter.dflashTargetLayerIds
                    vState[dflash2GDNCaptureContextKey] = nextCapture
                    vState[dflash2PipelinedVerifyPlanKey] = plan
                    let nextVerifyTokens = concatenated(
                        [bonusLazy.asType(.int32), np.tokens.flattened().asType(.int32)])
                    let vInput = LMInput.Text(tokens: nextVerifyTokens)
                    let vResult = mainModel(
                        vInput[text: .newAxis], cache: mainCache, state: vState)
                    if let capturedNext = vResult.state?[dflash2CapturedHiddenStatesKey],
                        !capturedNext.isEmpty
                    {
                        let hiddenNext =
                            capturedNext.count == 1
                            ? capturedNext[0] : concatenated(capturedNext, axis: -1)
                        let gammaNext = nextWidthS - 1
                        let targetNext = argMax(
                            vResult.logits[0, 0 ..< (gammaNext + 1), 0...], axis: -1
                        ).asType(.int32)
                        let packedNext = concatenated(
                            [np.tokens.flattened().asType(.int32), targetNext])
                        nextVerify = DFlash2PrebuiltVerify(
                            logits: vResult.logits, hidden: hiddenNext,
                            capture: nextCapture, packed: packedNext,
                            targetTokens: targetNext, width: nextWidthS)
                    }
                }
                if hostTimelineEnabled { alStamp("pb-vbuild", since: pbVBuildStart) }
                let pbDschedStart = ContinuousClock.now
                // Eager scheduling wins (measured): deferring to the next
                // round's build stream puts the draft's schedule cost on the
                // GPU-chase path and reopens the packed sync (40.8 vs 46.7
                // tok/s warm). `DFLASH2_PIPELINE_SCHED=0` defers for probes.
                if Self.prebuiltEagerSchedule, let np = nextPrebuilt {
                    asyncEval(np.tokens, np.candidates)
                }
                // Schedule the prebuilt verify (and its packed transfer)
                // behind the draft pass: the GPU pipeline for the next round
                // is fully committed before this round's accept resolves.
                if let nv = nextVerify {
                    asyncEval(nv.packed)
                }
                if hostTimelineEnabled {
                    alStamp("pb-dsched", since: pbDschedStart)
                    alStamp("prebuild", since: prebuildStart)
                }
            }
            let alSyncStart = ContinuousClock.now
            eval(packed)
            if hostTimelineEnabled { alStamp("sync", since: alSyncStart) }
            alAcceptFrom = ContinuousClock.now
            let both = packed.asArray(Int32.self)
            let greedyDrafts = both[0 ..< gamma].map(Int.init)
            let targetList = both[gamma ..< (2 * gamma + 1)].map(Int.init)
            draftList = greedyDrafts

            var acceptedCount = 0
            while acceptedCount < gamma, targetList[acceptedCount] == greedyDrafts[acceptedCount] {
                acceptedCount += 1
            }
            accepted = acceptedCount
            bonusTokenValue = targetList[accepted]
            if acceptLogEnabled {
                alRecordRound(
                    gamma: gamma, accepted: accepted, drafts: greedyDrafts,
                    targets: targetList, candidates: both[(2 * gamma + 1)...].map(Int.init))
            }
            if let dumpPath = Self.latticeDumpPath {
                Self.dumpAccept(
                    to: dumpPath + ".accept", targets: targetList,
                    drafts: greedyDrafts, accepted: accepted)
            }
            for token in greedyDrafts.prefix(accepted) {
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
            eval(draftTokens)
            let sampledDrafts = draftTokens.asArray(Int.self)
            draftList = sampledDrafts
            let result = dflash2RejectionSample(
                draftTokens: draftTokens.flattened(),
                targetProbs: targetProbs,
                draftProbs: selectorProbs[0],
                draftCandidates: draftCandidates[0])
            accepted = result.accepted
            bonusTokenValue = result.bonus
            for token in sampledDrafts.prefix(accepted) {
                processor?.didSample(token: MLXArray(token))
            }
            processor?.didSample(token: MLXArray(bonusTokenValue))
        }
        if profileEnabled { profileMark("accept", since: acceptStart) }
        if hostTimelineEnabled { alStamp("accept", since: alAcceptFrom) }

        // Pipelined bookkeeping: next round's anchor position, and the
        // accept count that resolves the staged rows' validity at adopt.
        lastAccepted = accepted
        pipelineAnchorPos = roundAnchorPos + accepted + 1
        if nextPrebuilt == nil, usedPrebuilt {
            // The pipelined chain broke (drafter escape hatch): restore the
            // write cursor so a synchronous propose appends at the right
            // positions next round.
            for cache in draftCache { cache.offset = roundAnchorPos }
        }
        prebuiltNext = nextPrebuilt
        prebuiltVerify = nextVerify
        let alReconcileStart = ContinuousClock.now

        // 4. Reconcile caches: `accepted + 1` verify positions stay committed
        // (anchor + accepted drafts); the rest are rolled back.
        let reconcileStart = ContinuousClock.now
        let rejected = gamma - accepted
        processedTokens += accepted + 1
        if consumedVerify != nil {
            // Stage-2 round: the KV rows were written at the true (lazily
            // resolved) offset on the GPU — only the committed count moves.
            // Rejected rows need no cleanup; the next write overwrites them.
            for cache in mainCache where cache.isTrimmable {
                (cache as! KVCacheSimple).commitPipelined(
                    rows: accepted + 1, anchor: roundAnchorPos)
            }
            // GDN: a continuing chain re-assigns the replay states in every
            // window; on a break, restore the slots for a synchronous
            // successor round.
            if nextVerify == nil, rejected > 0 {
                guard let prebuilt = prebuiltRollback else {
                    preconditionFailure(
                        "pipelined verify round without a prebuilt rollback")
                }
                applyGDNRollback(mainCache, prebuilt: prebuilt)
            }
        } else if rejected > 0 {
            if let prebuilt = prebuiltRollback {
                // The masked replay built in the sync window equals the
                // accepted-prefix replay for this outcome; only trims and
                // cache-slot assignment remain on the host here.
                applySpeculativeRollback(mainCache, prebuilt: prebuilt, rejected: rejected)
            } else {
                rollbackSpeculativeHybridCaches(
                    mainCache, context: roundCapture, accepted: accepted, rejected: rejected)
            }
        }
        lastRoundCapture = roundCapture

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

        // Adaptive width bandit: fold the round into the window; act every 8.
        if adaptiveWidth, blockSize > 3 {
            if widthWindowRounds == 0 { widthWindowStart = ContinuousClock.now }
            widthWindowRounds += 1
            widthWindowTokens += accepted + 1
            if widthWindowRounds >= 8, let windowStart = widthWindowStart {
                let elapsed = ContinuousClock.now - windowStart
                let secs =
                    Double(elapsed.components.seconds)
                    + Double(elapsed.components.attoseconds) / 1e18
                if secs > 0 {
                    widthScores[roundWidth] =
                        Double(widthWindowTokens) / secs
                }
                widthWindowRounds = 0
                widthWindowTokens = 0

                // Candidate shortlist {3, 4, cap}: bracket the optimum.
                var candidates = Set([3, 4, blockSize])
                candidates = candidates.filter { $0 >= 3 && $0 <= blockSize }
                // Unscored candidates first (initial sweep), then the argmax
                // with 3% hysteresis; every 24th settled window drops all
                // scores so the shortlist re-walks (content-drift re-sweep).
                if let next = candidates.filter({ widthScores[$0] == nil })
                    .min(by: { abs($0 - roundWidth) < abs($1 - roundWidth) }),
                    next != roundWidth
                {
                    roundWidth = next
                    windowsSinceSettle = 0
                } else if let best = widthScores.max(by: { $0.value < $1.value }),
                    let current = widthScores[roundWidth],
                    best.key != roundWidth && best.value > current * 1.03
                {
                    roundWidth = best.key
                    windowsSinceSettle = 0
                } else {
                    windowsSinceSettle += 1
                    if windowsSinceSettle >= 24 {
                        // Drift re-sweep: drop every score so the next windows
                        // re-walk the shortlist (~2 windows every few minutes
                        // of decode).
                        widthScores.removeAll(keepingCapacity: true)
                        windowsSinceSettle = 0
                    }
                }
            }
        }

        if profileEnabled {
            eval(pendingContextHidden)
            profileMark("reconcile", since: reconcileStart)
            profileRoundCount += 1
        }
        y = .init(tokens: MLXArray([Int32(bonusTokenValue)]))
        dflash2AnchorValue = bonusTokenValue
        if Self.advisedSelectorEnabled {
            ngramCommit([Int](draftList.prefix(accepted)) + [bonusTokenValue])
        }
        if hostTimelineEnabled {
            alStamp("reconcile", since: alReconcileStart)
            alLastRoundEnd = ContinuousClock.now
            hostTimelineRounds += 1
        }
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
        dflash2AnchorValue = value

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
        dflash2AnchorValue = value
        return value
    }

    // MARK: TokenIteratorProtocol

    public var tokenCount: Int { telemetry.emittedTokenCount }

    public mutating func discardGeneratedToken() {
        telemetry.discardGeneratedToken()
    }

    public mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            alEmitSummaryOnce()
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
        let finalCapture = lastRoundCapture ?? gdnCapture
        if finalCapture.captures.isEmpty {
            // Uncaptured step (pure-attention target, or a passthrough step
            // that could not record): trim-only rollback.
            for cache in mainCache where cache.isTrimmable {
                cache.trim(rewind)
            }
        } else {
            rollbackSpeculativeHybridCaches(
                mainCache, context: finalCapture,
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
