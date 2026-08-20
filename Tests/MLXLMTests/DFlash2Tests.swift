// Copyright © 2026 Apple Inc.
//
// DFlash2Tests.swift
// mlx-swift-lm
//
// Synthetic tests for the DFlash2 block-parallel drafter and its speculative
// iterator: context-cache semantics, block masks, the two-tap dynamic conv,
// the candidate selector, hybrid-cache rollback, rejection sampling, and an
// end-to-end iterator run against mocks. Real-weights parity with the Python
// reference lives in the integration tests (they need the checkpoints).

import Foundation
import MLX
import MLXNN
import Testing

@_spi(Testing) @testable import MLXLMCommon
@testable import MLXLLM

// MARK: - Config

@Test
func testDFlash2ConfigurationDecodesReleaseCheckpoint() throws {
    // Mirrors incoai/Qwen3.8-27B-DFlash2's config.json.
    let json = """
        {
          "architectures": ["DFlash2DraftModel"],
          "is_causal": false,
          "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 248070,
            "selector_rank": 256,
            "selector_top_k": 16,
            "target_layer_ids": [5, 19, 33, 47, 61]
          },
          "head_dim": 128,
          "hidden_size": 5120,
          "intermediate_size": 17408,
          "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
          "max_position_embeddings": 262144,
          "model_type": "qwen3",
          "num_attention_heads": 32,
          "num_hidden_layers": 5,
          "num_key_value_heads": 8,
          "num_target_layers": 64,
          "rms_norm_eps": 1e-06,
          "rope_parameters": { "rope_theta": 10000000, "rope_type": "default" },
          "sliding_window": 2048,
          "tie_word_embeddings": false,
          "vocab_size": 248320
        }
        """
    let config = try JSONDecoder.json5().decode(
        DFlash2Configuration.self, from: Data(json.utf8))
    #expect(config.hiddenSize == 5120)
    #expect(config.hiddenLayers == 5)
    #expect(config.kvHeads == 8)
    #expect(config.headDim == 128)
    #expect(config.isCausal == false)
    #expect(config.ropeTheta == 10_000_000)
    #expect(config.slidingWindow == 2048)
    #expect(config.contextKeepCount == 2047)
    #expect(config.dflash.blockSize == 8)
    #expect(config.dflash.maskTokenId == 248070)
    #expect(config.dflash.selectorTopK == 16)
    #expect(config.dflash.targetLayerIds == [5, 19, 33, 47, 61])
    #expect(config.dflash.convKernelSize == 2)
    #expect(config.dflash.convGroupSize == 16)
}

private func tinyConfig(
    hiddenSize: Int = 64, layers: Int = 2, heads: Int = 4, kvHeads: Int = 2,
    headDim: Int = 16, intermediate: Int = 96, vocab: Int = 64,
    window: Int = 8, topK: Int = 4, rank: Int = 8, kernel: Int = 2, group: Int = 16
) -> DFlash2Configuration {
    let json = """
        {
          "architectures": ["DFlash2DraftModel"],
          "is_causal": false,
          "dflash_config": {
            "block_size": 5,
            "conv_group_size": \(group),
            "conv_kernel_size": \(kernel),
            "mask_token_id": \(vocab - 2),
            "selector_rank": \(rank),
            "selector_top_k": \(topK),
            "target_layer_ids": [1, 3]
          },
          "head_dim": \(headDim),
          "hidden_size": \(hiddenSize),
          "intermediate_size": \(intermediate),
          "layer_types": [\(Array(repeating: "\"sliding_attention\"", count: layers).joined(separator: ", "))],
          "max_position_embeddings": 1024,
          "model_type": "qwen3",
          "num_attention_heads": \(heads),
          "num_hidden_layers": \(layers),
          "num_key_value_heads": \(kvHeads),
          "num_target_layers": 4,
          "rms_norm_eps": 1e-06,
          "rope_parameters": { "rope_theta": 10000.0, "rope_type": "default" },
          "sliding_window": \(window),
          "vocab_size": \(vocab)
        }
        """
    return try! JSONDecoder.json5().decode(DFlash2Configuration.self, from: Data(json.utf8))
}

// MARK: - Context cache

@Test
func testDFlash2ContextCacheAppendAndTrim() {
    let cache = DFlash2ContextCache(maxSize: 4)

    // First insert: stored verbatim even if it exceeds maxSize (the caller
    // pre-trims incoming context; mirror of RotatingKVCache._update_concat).
    let k1 = MLXArray.ones([1, 1, 5, 2])
    let v1 = MLXArray.full([1, 1, 5, 2], values: MLXArray(2))
    let (keys, _) = cache.append(keys: k1, values: v1)
    #expect(keys.dim(2) == 5)
    #expect(cache.offset == 5)
    #expect(cache.count == 5)

    // Second append: stored tail front-trimmed to maxSize - 1 first.
    let k2 = MLXArray.ones([1, 1, 3, 2]) * 3
    let (keys2, values2) = cache.append(keys: k2, values: k2)
    // stored 5 -> trimmed to 3 -> +3 = 6
    #expect(keys2.dim(2) == 6)
    #expect(cache.offset == 8)
    // front-trimmed: first 3 rows are the OLDEST surviving (value 1), then 3s
    let keyValues = values2[0, 0, 0..., 0].asArray(Float.self)
    #expect(keyValues == [2, 2, 2, 3, 3, 3])

    // trimNewest drops trailing rows and rewinds the write position.
    cache.trimNewest(2)
    #expect(cache.count == 4)
    #expect(cache.offset == 6)
    let afterTrim = cache.values![0, 0, 0..., 0].asArray(Float.self)
    #expect(afterTrim == [2, 2, 2, 3])

    // Trimming more than stored is clamped; the offset keeps the count of
    // front-trimmed (never stored) rows.
    cache.trimNewest(99)
    #expect(cache.count == 0)
    #expect(cache.offset == 2)
}

// MARK: - Mask

@Test
func testDFlash2BlockMaskNonCausalSliding() {
    // L=3 queries, ctxLen=4, window=3, non-causal (block fully visible).
    let mask = DFlash2Attention.makeMask(
        queryLen: 3, contextLen: 4, isCausal: false, slidingWindow: 3)
    // kv axis: [ctx 0..3, block 4..6]; query i sits at concat position 4+i.
    let expected: [[Bool]] = [
        // ctx: (query - key) < 3; block (keys ≥ 4): all visible
        [false, false, true, true, true, true, true],  // q=4: ctx keys 2..3
        [false, false, false, true, true, true, true],  // q=5: ctx key 3
        [false, false, false, false, true, true, true],  // q=6: no ctx keys
    ]
    #expect(mask.shape == [3, 7])
    let actual = mask.asArray(Bool.self)
    #expect(actual == expected.flatMap { $0 })
}

@Test
func testDFlash2BlockMaskCausalSliding() {
    let mask = DFlash2Attention.makeMask(
        queryLen: 3, contextLen: 4, isCausal: true, slidingWindow: 3)
    let expected: [[Bool]] = [
        [false, false, true, true, true, false, false],  // q=4: ctx 2..3 + block ≤4
        [false, false, false, true, true, true, false],  // q=5: ctx 3 + block ≤5
        [false, false, false, false, true, true, true],  // q=6: block ≤6
    ]
    #expect(mask.asArray(Bool.self) == expected.flatMap { $0 })
}

// MARK: - Dynamic conv

@Test
func testDFlash2DynamicConvMatchesNaive() throws {
    let (hiddenSize, kernelSize, groupSize) = (16, 2, 4)
    let conv = DFlash2DynamicConv(
        hiddenSize: hiddenSize, kernelSize: kernelSize, groupSize: groupSize)

    MLXRandom.seed(7)
    let base = MLXRandom.normal([2, kernelSize, hiddenSize]).asType(.bfloat16)
    let proj = MLXRandom.normal([2 * kernelSize * (hiddenSize / groupSize), hiddenSize])
        .asType(.bfloat16)
    try conv.update(
        parameters: ModuleParameters.unflattened([
            ("base_kernel", base), ("kernel_projection.weight", proj),
        ]),
        verify: [])

    let hidden = MLXRandom.normal([1, 5, hiddenSize]).asType(.bfloat16)
    let (prepared, dyn) = conv.prepare(hidden)
    let finished = conv.finish(hidden, dynamic: dyn)
    eval(prepared, finished)

    // Slot-0 dynamic kernel (drives `prepare`); `dyn` is slot 1.
    let fullDyn = conv.kernelProjection(hidden).reshaped(
        1, 5, 2, kernelSize, hiddenSize / groupSize)
    let dyn0 = fullDyn[0..., 0..., 0, 0..., 0...]
    eval(dyn0)

    // Naive reference in plain loops (float32 for comparison slack).
    func naive(_ x: MLXArray, baseRow: Int, dynamic: MLXArray) -> [[Float]] {
        let xF = x.asType(.float32)
        let groups = hiddenSize / groupSize
        let dynF = dynamic.asType(.float32)  // [1, L, K, groups]
        let baseF = base.asType(.float32)
        var out = [[Float]](repeating: [Float](repeating: 0, count: hiddenSize), count: 5)
        for t in 0 ..< 5 {
            for tap in 0 ..< kernelSize {
                let src = t - tap
                guard src >= 0 else { continue }
                for g in 0 ..< groups {
                    let d = dynF[0, t, tap, g].item(Float.self)
                    for c in 0 ..< groupSize {
                        let ch = g * groupSize + c
                        let kVal = baseF[baseRow, tap, ch].item(Float.self) + d
                        out[t][ch] += kVal * xF[0, src, ch].item(Float.self)
                    }
                }
            }
        }
        return out
    }

    // bf16 accumulation-order slack: the reference chain rounds in bf16 while
    // the naive loop accumulates in f32; at |values| ~15 that is ~0.06.
    let expectedPre = naive(hidden, baseRow: 0, dynamic: dyn0)
    let preparedF = prepared.asType(.float32)
    for t in 0 ..< 5 {
        for ch in 0 ..< hiddenSize {
            let got = preparedF[0, t, ch].item(Float.self)
            #expect(abs(got - expectedPre[t][ch]) < 0.15, "prepared[\(t)][\(ch)]")
        }
    }

    // finish uses base row 1 with the same dynamic kernel.
    let expectedFin = naive(hidden, baseRow: 1, dynamic: dyn)
    let finishedF = finished.asType(.float32)
    for t in 0 ..< 5 {
        for ch in 0 ..< hiddenSize {
            let got = finishedF[0, t, ch].item(Float.self)
            #expect(abs(got - expectedFin[t][ch]) < 0.15, "finished[\(t)][\(ch)]")
        }
    }
}

// MARK: - Selector

@Test
func testDFlash2SelectorGreedyPrefersCoherentPath() throws {
    // vocab 12, topK 3, rank 4, hidden 8. Position 0's top-1 candidate is a
    // decoy; the selector must switch to the candidate the anchor points at.
    var config = tinyConfig(hiddenSize: 8, layers: 1, heads: 1, kvHeads: 1, headDim: 8,
                            intermediate: 8, vocab: 12, window: 4, topK: 3, rank: 4,
                            kernel: 2, group: 4)
    let selector = DFlash2CandidateSelector(config)

    // hidden_projection = identity on the first `rank` dims.
    var proj = [Float](repeating: 0, count: 4 * 8)
    for i in 0 ..< 4 { proj[i * 8 + i] = 1 }
    try selector.hiddenProjection.update(
        parameters: ModuleParameters.unflattened([("weight", MLXArray(proj, [4, 8]))]),
        verify: [])

    // Codebooks: token 9 (anchor) strongly matches candidate token 5 via
    // one-hot dims; everything else ~0.
    var pred = [Float](repeating: 0, count: 12 * 4)
    pred[9 * 4 + 0] = 1
    var succ = [Float](repeating: 0, count: 12 * 4)
    succ[5 * 4 + 0] = 50  // huge edge weight anchor 9 -> candidate 5
    try selector.predecessorCodebook.update(
        parameters: ModuleParameters.unflattened([("weight", MLXArray(pred, [12, 4]))]),
        verify: [])
    try selector.successorCodebook.update(
        parameters: ModuleParameters.unflattened([("weight", MLXArray(succ, [12, 4]))]),
        verify: [])
    eval(selector.parameters())

    // Logits over 2 positions; candidates by top-3.
    // pos 0: raw top-1 = 7 (logit 10), but token 5 (logit 9) wins via edge.
    // pos 1: no edge (token 5's pred row is 0): raw top-1 = 2 stands.
    var logits = [Float](repeating: -30, count: 2 * 12)
    logits[0 * 12 + 7] = 10
    logits[0 * 12 + 5] = 9
    logits[0 * 12 + 1] = 8
    logits[1 * 12 + 2] = 10
    logits[1 * 12 + 3] = 9
    logits[1 * 12 + 4] = 8
    let logitsArray = MLXArray(logits, [1, 2, 12])
    // Ones: the identity-projected gate passes the predecessor embedding
    // through (zeros would zero every edge score).
    let hidden = MLXArray.ones([1, 2, 8])

    let (path, candidates, probs) = selector.select(
        hidden: hidden, logits: logitsArray, anchorIds: MLXArray([Int32(9)]), temperature: 0)
    eval(path, candidates)
    #expect(probs == nil)
    #expect(path.asArray(Int32.self) == [5, 2])
}

// MARK: - GDN rollback

@Test
func testGDNCaptureRollbackMatchesPrefixReplay() throws {
    // Synthesize one GDN layer's verify-pass inputs, run the full pass, then
    // roll back to an accepted prefix and compare against a direct prefix
    // computation. Mirrors _GDNStateCapture.rollback in the reference.
    let (B, S, Hk, Dk, Hv, Dv) = (1, 6, 2, 32, 4, 16)
    let K = 4  // conv kernel size
    let convDim = 2 * Hk * Dk + Hv * Dv

    MLXRandom.seed(11)
    let q = MLXRandom.normal([B, S, Hk, Dk]).asType(.bfloat16)
    let k = MLXRandom.normal([B, S, Hk, Dk]).asType(.bfloat16)
    let v = MLXRandom.normal([B, S, Hv, Dv]).asType(.bfloat16)
    let a = MLXRandom.normal([B, S, Hv]).asType(.bfloat16)
    let b = MLXRandom.normal([B, S, Hv]).asType(.bfloat16)
    let aLog = MLXRandom.normal([Hv]).asType(.float32)
    let dtBias = MLXRandom.normal([Hv]).asType(.float32)
    let initState = MLXRandom.normal([B, Hv, Dv, Dk]).asType(.float32)
    let convInput = MLXRandom.normal([B, K - 1 + S, convDim]).asType(.bfloat16)

    let capture = GDNCapture(
        convInput: convInput, q: q, k: k, v: v, a: a, b: b,
        aLog: aLog, dtBias: dtBias, initialState: initState, mask: nil,
        convKernelSize: K)
    let context = GDNCaptureContext()
    context.record(capture)

    // The cache as the full S-position verify pass left it.
    let (_, fullState) = gatedDeltaUpdate(
        q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias, state: initState)
    let mamba = MambaCache()
    mamba[0] = convInput[0..., S..., 0...]  // last K-1 rows after the full pass
    mamba[1] = fullState
    let attention = KVCacheSimple()
    let kvKeys = MLXRandom.normal([1, 1, 20, 4]).asType(.bfloat16)
    _ = attention.update(keys: kvKeys, values: kvKeys)

    let accepted = 3  // keep anchor + 3 drafts = 4 of S=6 verify rows
    let rejected = S - accepted - 1
    rollbackSpeculativeHybridCaches(
        [attention, mamba], context: context, accepted: accepted, rejected: rejected)

    // Expected: prefix replay over n = accepted + 1 positions.
    let (_, expectedState) = gatedDeltaUpdate(
        q: q[0..., ..<4, 0..., 0...], k: k[0..., ..<4, 0..., 0...],
        v: v[0..., ..<4, 0..., 0...], a: a[0..., ..<4, 0...], b: b[0..., ..<4, 0...],
        aLog: aLog, dtBias: dtBias, state: initState)
    let gotState = mamba[1]!
    eval(gotState, expectedState)
    let diff = (gotState - expectedState).abs().max().item(Float.self)
    #expect(diff < 1e-3, "recurrent state replay diverged: \(diff)")

    // Conv state = convInput rows [accepted+1, accepted+K).
    let expectedConv = convInput[0..., 4 ..< 4 + K - 1, 0...]
    let gotConv = mamba[0]!
    eval(gotConv, expectedConv)
    #expect((gotConv - expectedConv).abs().max().item(Float.self) == 0)

    // Attention cache trimmed the rejected rows.
    #expect(attention.offset == 20 - rejected)
}

// MARK: - Rejection sampling

@Test
func testDFlash2RejectionSampleFullAgreementAcceptsAll() {
    // q puts all mass on the drafted token at every position; p agrees.
    // Acceptance must be total and the bonus must come from the last row.
    let (gamma, K, V) = (3, 2, 10)
    let drafts = MLXArray([Int32(2), 3, 4])
    var candidates = [[Int32]](repeating: [0, 0], count: gamma)
    var probs = [[Float]](repeating: [0, 0], count: gamma)
    var target = [[Float]](repeating: [Float](repeating: 0, count: V), count: gamma + 1)
    for i in 0 ..< gamma {
        let token = drafts[i].item(Int32.self)
        candidates[i] = [token, (token + 1) % Int32(V)]
        probs[i] = [1.0, 0.0]
        target[i][Int(token)] = 1.0
    }
    target[gamma][7] = 1.0  // bonus row: token 7 certain

    let result = dflash2RejectionSample(
        draftTokens: drafts,
        targetProbs: MLXArray(target.flatMap { $0 }, [gamma + 1, V]),
        draftProbs: MLXArray(probs.flatMap { $0 }, [gamma, K]),
        draftCandidates: MLXArray(candidates.flatMap { $0 }, [gamma, K]))
    #expect(result.accepted == gamma)
    #expect(result.bonus == 7)
}

@Test
func testDFlash2RejectionSampleZeroProbabilityRejectsFirst() {
    let (gamma, K, V) = (3, 2, 10)
    let drafts = MLXArray([Int32(2), 3, 4])
    let candidates = MLXArray([Int32(2), 5, 3, 5, 4, 5], [gamma, K])
    let probs = MLXArray([Float(1), 0, 1, 0, 1, 0], [gamma, K])
    var target = [[Float]](repeating: [Float](repeating: 0, count: V), count: gamma + 1)
    // Position 0: target assigns zero to the drafted token 2, all mass to 6.
    target[0][6] = 1.0
    let result = dflash2RejectionSample(
        draftTokens: drafts,
        targetProbs: MLXArray(target.flatMap { $0 }, [gamma + 1, V]),
        draftProbs: probs,
        draftCandidates: candidates)
    #expect(result.accepted == 0)
    #expect(result.bonus == 6)  // residual = p (deterministic)
}

@Test
func testDFlash2SamplingProbsTopKTopP() {
    // temperature 1, top-k 2, top-p on: distribution confined to the top-2.
    let logits = MLXArray([Float(0), 1, 2, 3], [1, 4])
    let probs = dflash2SamplingProbs(logits, temperature: 1, topP: 0.9, topK: 2)
    let p = probs.asArray(Float.self)
    #expect(p[0] == 0 && p[1] == 0)
    #expect(abs(p[2] + p[3] - 1) < 1e-6)
    // top-2 softmax shares: e^2/(e^2+e^3)
    let e2 = exp(Float(2))
    let e3 = exp(Float(3))
    #expect(abs(p[2] - e2 / (e2 + e3)) < 1e-5)
}

// MARK: - Iterator end-to-end with mocks

/// Script-driven mock of the DFlash2 drafter contract.
private final class MockDFlash2Drafter: Module, DFlash2DrafterModel {
    /// Tokens proposed per round (cycles if the run has more rounds).
    var script: [[Int32]]
    private(set) var proposeCalls = 0
    private(set) var receivedBlockShapes: [[Int]] = []
    private(set) var receivedContextRows: [Int] = []

    init(script: [[Int32]]) {
        self.script = script
        super.init()
    }

    var dflashBlockSize: Int { 4 }
    var dflashMaskTokenId: Int { 98 }
    var dflashTargetLayerIds: [Int] { [1, 2] }
    var dflashContextKeepCount: Int? { 16 }
    var dflashNumTargetLayers: Int { 4 }
    func bindDFlashTarget(_ target: any LanguageModel) {}
    func makeDFlashContextCaches() -> [DFlash2ContextCache] {
        [DFlash2ContextCache(maxSize: 16)]
    }
    func dflashPropose(
        _ inputs: MLXArray, targetHidden: MLXArray, cache: [DFlash2ContextCache],
        temperature: Float, logitsStart: Int
    ) -> (tokens: MLXArray, candidates: MLXArray, probabilities: MLXArray?) {
        let round = min(proposeCalls, script.count - 1)
        proposeCalls += 1
        receivedBlockShapes.append(inputs.shape)
        receivedContextRows.append(targetHidden.dim(1))
        let tokens = script[round]
        // Fabricate consistent top-K metadata: candidates = token + junk.
        let K = 4
        var candidateRows: [[Int32]] = []
        var probRows: [[Float]] = []
        for token in tokens {
            candidateRows.append([token, 90, 91, 92])
            probRows.append([0.9, 0.05, 0.025, 0.025])
        }
        return (
            MLXArray(tokens).expandedDimensions(axis: 0),
            MLXArray(candidateRows.flatMap { $0 }, [1, tokens.count, K]),
            MLXArray(probRows.flatMap { $0 }, [1, tokens.count, K])
        )
    }
}

/// Mock hybrid target: one trimmable "attention" cache + one MambaCache,
/// one-hot scripted logits, DFlash2 capture emission.
private final class MockDFlash2Target: Module, LanguageModel, KVCacheDimensionProvider {
    var kvHeads: [Int] { [1] }
    /// Planned argmax token per forward position, in call order.
    var tokenScript: [Int32]
    private var scriptIndex = 0
    private(set) var forwardCalls = 0

    static let vocab = 100
    static let hidden = 8

    init(tokenScript: [Int32]) {
        self.tokenScript = tokenScript
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        makeLogits(positions: inputs.dim(-1))
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        forwardCalls += 1
        let positions = input.tokens.dim(-1)
        let logits = makeLogits(positions: positions)

        if let cache {
            for entry in cache {
                if entry is MambaCache {
                    // no-op: state written by the capture emulation below
                } else if let simple = entry as? KVCacheSimple {
                    let kv = MLXArray.zeros([1, 1, positions, 4])
                    _ = simple.update(keys: kv, values: kv)
                }
            }
        }

        var outState = state ?? LMOutput.State()
        if let layerIds = state?[dflash2CaptureLayerIdsKey] {
            outState[dflash2CapturedHiddenStatesKey] = layerIds.map { _ in
                MLXArray.zeros([1, positions, Self.hidden])
            }
        }
        if let context = state?[dflash2GDNCaptureContextKey] {
            // Record one self-consistent capture (the mock has one GDN layer).
            let (Hk, Dk, Hv, Dv) = (1, 32, 2, 16)
            let convDim = 2 * Hk * Dk + Hv * Dv
            MLXRandom.seed(UInt64(1000 + forwardCalls))
            let initState =
                (cache?.first(where: { $0 is MambaCache }) as? MambaCache)?[1]
            context.record(
                GDNCapture(
                    convInput: MLXRandom.normal([1, 3 + positions, convDim]).asType(.bfloat16),
                    q: MLXRandom.normal([1, positions, Hk, Dk]).asType(.bfloat16),
                    k: MLXRandom.normal([1, positions, Hk, Dk]).asType(.bfloat16),
                    v: MLXRandom.normal([1, positions, Hv, Dv]).asType(.bfloat16),
                    a: MLXRandom.normal([1, positions, Hv]).asType(.bfloat16),
                    b: MLXRandom.normal([1, positions, Hv]).asType(.bfloat16),
                    aLog: MLXRandom.normal([Hv]).asType(.float32),
                    dtBias: MLXRandom.normal([Hv]).asType(.float32),
                    initialState: initState ?? MLXArray.zeros([1, Hv, Dv, Dk], dtype: .float32),
                    mask: nil, convKernelSize: 4))
        }
        return LMOutput(logits: logits, state: outState)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let mamba = MambaCache()
        mamba[0] = MLXArray.zeros([1, 3, 96], dtype: .bfloat16)
        mamba[1] = MLXArray.zeros([1, 2, 16, 32], dtype: .float32)
        return [KVCacheSimple(), mamba]
    }

    private func makeLogits(positions: Int) -> MLXArray {
        var data = [Float](repeating: 0, count: positions * Self.vocab)
        for i in 0 ..< positions {
            let idx = scriptIndex + i
            let token = idx < tokenScript.count ? Int(tokenScript[idx]) : 0
            data[i * Self.vocab + token] = 100
        }
        scriptIndex += positions
        return MLXArray(data, [1, positions, Self.vocab])
    }
}

@Test
func testDFlash2IteratorEndToEndAcceptanceAndRollback() throws {
    // Target script (per forward position, in call order). Prefill of the
    // 3-token prompt runs in two chunks (2 + 1 — the final position is always
    // its own chunk), so the first sampled token comes from script[2].
    let target = MockDFlash2Target(tokenScript: [
        1, 1, 10,  // prefill; script[2] = first emitted token (anchor 10)
        20, 21, 55, 0,  // round 1 verify rows: accept 20, 21; correct 99 → 55
        30, 77, 0, 0,  // round 2 verify rows: accept 30; correct 31 → 77
        77, 77, 77, 77,  // round 3 verify rows: accept all, bonus 77
        77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,  // filler
    ])
    let drafter = MockDFlash2Drafter(script: [
        [20, 21, 99],  // round 1: two hits, one miss
        [30, 31, 32],  // round 2: one hit, then miss
        [77, 77, 77],  // round 3
    ])
    let input = LMInput(tokens: MLXArray([Int32(1), 2, 3]))
    var parameters = GenerateParameters(maxTokens: 20)
    parameters.temperature = 0
    var iterator = try DFlash2SpeculativeTokenIterator(
        input: input, mainModel: target, drafter: drafter,
        parameters: parameters, blockSize: 4)

    var produced: [Int] = []
    while let token = iterator.next() {
        produced.append(token)
    }

    // Expected stream: 10 (prefill bonus) | 20 21 55 | 30 77 | 77 77 77 77...
    #expect(produced.prefix(7) == [10, 20, 21, 55, 30, 77, 77])
    #expect(drafter.proposeCalls >= 2)
    // Round 1: accepted 2 of 3 → 1 rejected row trimmed from attention cache.
    #expect(iterator.speculativeDecodingTelemetry != nil)
    let telemetry = iterator.speculativeDecodingTelemetry!
    #expect(telemetry.draftTokenCount >= 3)
    #expect(telemetry.acceptedDraftTokenCount >= 2)
}
