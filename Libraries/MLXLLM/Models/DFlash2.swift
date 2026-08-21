// Copyright © 2026 Apple Inc.
//
// DFlash2.swift
// mlx-swift-lm
//
// Port of the DFlash 2 block-diffusion drafter for speculative decoding:
// https://inco.ai/blog/dflash2/ — reference implementation:
// https://github.com/z-lab/dflash (dflash/model_mlx.py, same weights).
//
// A DFlash2 drafter predicts a whole block of tokens in ONE parallel pass
// (the block positions are bidirectional, seeded with a mask token), keeps
// the top-K candidates at every position, and traces one coherent path
// through them with a lightweight pairwise selector. Its backbone inserts a
// two-tap dynamic depthwise convolution before and after every attention and
// MLP sublayer, so local within-block dependencies do not depend on the
// attention budget. Context enters as a cross-attention: the drafter keeps
// its own sliding-window KV cache over projections of the TARGET model's
// concatenated per-layer hidden states (`fc` + `hidden_norm`), so every
// speculation round only feeds the drafter the hidden states of the tokens
// accepted since the previous round.
//
// Weight contract (incoai/Qwen3.8-27B-DFlash2):
//   fc.weight, hidden_norm.weight, norm.weight,
//   layers.{i}.{input_layernorm,post_attention_layernorm}.weight
//   layers.{i}.self_attn.{q,k,v,o}_proj.weight, layers.{i}.self_attn.{q,k}_norm.weight
//   layers.{i}.mlp.{gate,up,down}_proj.weight
//   layers.{i}.{attention,mlp}_conv.{base_kernel, kernel_projection.weight}
//   candidate_selector.{hidden_projection.weight,
//                       predecessor_codebook, successor_codebook}
// The checkpoint has NO embed_tokens / lm_head: both are borrowed from the
// target model at `bind(target:)` time (quantized or not — the borrowing is
// by module, so a `QuantizedEmbedding`/`QuantizedLinear` target head works).

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// `dflash_config` sub-object of the draft checkpoint's config.json.
public struct DFlash2SpecConfiguration: Codable, Sendable {
    public var blockSize: Int = 16
    public var convGroupSize: Int = 16
    public var convKernelSize: Int = 2
    public var maskTokenId: Int = 0
    public var selectorRank: Int = 256
    public var selectorTopK: Int = 16
    public var targetLayerIds: [Int] = []
    public var inputEmbeddingScale: Float = 1.0
    public var outputMultiplier: Float = 1.0
    public var finalLogitSoftcapping: Float?

    enum CodingKeys: String, CodingKey {
        case blockSize = "block_size"
        case convGroupSize = "conv_group_size"
        case convKernelSize = "conv_kernel_size"
        case maskTokenId = "mask_token_id"
        case selectorRank = "selector_rank"
        case selectorTopK = "selector_top_k"
        case targetLayerIds = "target_layer_ids"
        case inputEmbeddingScale = "input_embedding_scale"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        blockSize = try container.decodeIfPresent(Int.self, forKey: .blockSize) ?? 16
        convGroupSize = try container.decodeIfPresent(Int.self, forKey: .convGroupSize) ?? 16
        convKernelSize = try container.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 2
        maskTokenId = try container.decodeIfPresent(Int.self, forKey: .maskTokenId) ?? 0
        selectorRank = try container.decodeIfPresent(Int.self, forKey: .selectorRank) ?? 256
        selectorTopK = try container.decodeIfPresent(Int.self, forKey: .selectorTopK) ?? 16
        targetLayerIds =
            try container.decodeIfPresent([Int].self, forKey: .targetLayerIds) ?? []
        inputEmbeddingScale =
            try container.decodeIfPresent(Float.self, forKey: .inputEmbeddingScale) ?? 1.0
        outputMultiplier =
            try container.decodeIfPresent(Float.self, forKey: .outputMultiplier) ?? 1.0
        finalLogitSoftcapping =
            try container.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping)
    }
}

/// Configuration of a DFlash/DFlash2 draft checkpoint (`model_type: qwen3`,
/// `architectures: [DFlash2DraftModel]`). Only the fields the inference port
/// needs are decoded.
public struct DFlash2Configuration: Decodable, Sendable {
    public var hiddenSize: Int = 5120
    public var hiddenLayers: Int = 5
    public var attentionHeads: Int = 32
    public var kvHeads: Int = 8
    public var headDim: Int = 128
    public var intermediateSize: Int = 17408
    public var vocabularySize: Int = 248320
    public var rmsNormEps: Float = 1e-6
    public var maxPositionEmbeddings: Int = 262144
    public var slidingWindow: Int? = nil
    public var layerTypes: [String] = []
    /// `is_causal` in the checkpoint: nil means "causal iff sliding". The
    /// released DFlash2 drafters ship `false` (bidirectional block).
    public var isCausal: Bool? = nil
    public var ropeTheta: Float = 10_000_000
    public var ropeScaling: [String: StringOrNumber]? = nil
    public var numTargetLayers: Int = 64
    public var architectures: [String] = []
    public var dflash: DFlash2SpecConfiguration = .init()

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case isCausal = "is_causal"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case numTargetLayers = "num_target_layers"
        case architectures
        case dflash = "dflash_config"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 5120
        hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 5
        attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hiddenSize / attentionHeads)
        intermediateSize =
            try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 17408
        vocabularySize =
            try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248320
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
        slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        layerTypes = try container.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        isCausal = try container.decodeIfPresent(Bool.self, forKey: .isCausal)
        numTargetLayers =
            try container.decodeIfPresent(Int.self, forKey: .numTargetLayers) ?? 64
        architectures =
            try container.decodeIfPresent([String].self, forKey: .architectures) ?? []
        dflash =
            try container.decodeIfPresent(
                DFlash2SpecConfiguration.self, forKey: .dflash) ?? .init()

        var ropeParameters =
            try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)
            ?? container.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        if ropeParameters?["type"] == nil, let ropeType = ropeParameters?["rope_type"] {
            ropeParameters?["type"] = ropeType
        }
        let topLevelTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = ropeParameters?["rope_theta"]?.asFloat() ?? topLevelTheta ?? 10_000_000
        ropeScaling = ropeParameters

        if layerTypes.isEmpty {
            layerTypes = Array(repeating: "full_attention", count: hiddenLayers)
        }
        guard layerTypes.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: container,
                debugDescription: "layer_types count \(layerTypes.count) != num_hidden_layers \(hiddenLayers)")
        }
    }

    /// Context entries the draft cache retains: the sliding window minus the
    /// current block (every block query keeps a full window of context).
    public var contextKeepCount: Int? {
        slidingWindow.map { $0 - 1 }
    }
}

// The draft context cache (`DFlash2ContextCache`) lives in MLXLMCommon
// (DFlash2Support.swift), next to the iterator that owns it.

// MARK: - Two-tap dynamic depthwise convolution

/// Conv_k(x)_t = k_{t,0} ⊙ x_t + k_{t,1} ⊙ x_{t-1}, applied independently to
/// `groupSize`-channel blocks: a learned per-channel base kernel plus a
/// per-group dynamic correction projected from the current hidden state.
///
/// `prepare` convolves the sublayer INPUT (kernel slot 0) and returns the
/// dynamic kernel for slot 1; `finish` applies slot 1 to the sublayer OUTPUT.
/// A shift of one position zero-pads at the front: the block's first position
/// reads nothing from the past through the conv (the residual stream already
/// carries the verified anchor), matching the reference implementation.
final class DFlash2DynamicConv: Module {
    let kernelSize: Int
    let groupSize: Int

    @ParameterInfo(key: "base_kernel") var baseKernel: MLXArray
    @ModuleInfo(key: "kernel_projection") var kernelProjection: Linear

    init(hiddenSize: Int, kernelSize: Int, groupSize: Int) {
        self.kernelSize = kernelSize
        self.groupSize = groupSize
        let groups = hiddenSize / groupSize
        precondition(hiddenSize % groupSize == 0, "conv groups must divide hidden size")
        _baseKernel.wrappedValue = MLXArray.zeros([2, kernelSize, hiddenSize])
        _kernelProjection.wrappedValue = Linear(hiddenSize, 2 * kernelSize * groups, bias: false)
        super.init()
    }

    /// hidden [B, L, H], dynamic [B, L, K, groups], base [K, H].
    private static func convolve(
        _ hidden: MLXArray, dynamic: MLXArray, base: MLXArray, groupSize: Int
    ) -> MLXArray {
        let (b, l, h) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        let groups = h / groupSize
        let blocks = hidden.reshaped(b, l, groups, groupSize)
        var output = MLXArray.zeros(like: blocks)
        for tap in 0 ..< base.dim(0) {
            let values: MLXArray
            if tap == 0 {
                values = blocks
            } else {
                values = concatenated(
                    [MLXArray.zeros(like: blocks[0..., ..<tap]), blocks[0..., ..<(l - tap)]],
                    axis: 1)
            }
            // base: per-channel; dynamic: per-group broadcast over the block.
            let baseKernel = base[tap].asType(hidden.dtype).reshaped(1, 1, groups, groupSize)
            output = output + baseKernel * values
            output = output + dynamic[0..., 0..., tap, 0..., .newAxis] * values
        }
        return output.reshaped(hidden.shape)
    }

    /// Returns (convolved input with kernel slot 0, dynamic kernel for slot 1).
    func prepare(_ hidden: MLXArray) -> (MLXArray, MLXArray) {
        let groups = hidden.dim(-1) / groupSize
        let dynamic = kernelProjection(hidden).reshaped(
            hidden.dim(0), hidden.dim(1), 2, kernelSize, groups)
        return (
            Self.convolve(
                hidden, dynamic: dynamic[0..., 0..., 0, 0..., 0...],
                base: baseKernel[0], groupSize: groupSize),
            dynamic[0..., 0..., 1, 0..., 0...]
        )
    }

    func finish(_ hidden: MLXArray, dynamic: MLXArray) -> MLXArray {
        Self.convolve(hidden, dynamic: dynamic, base: baseKernel[1], groupSize: groupSize)
    }
}

// MARK: - Attention

/// Per-round memo for the draft attention mask. Every sliding layer builds
/// the identical [queryLen, contextLen + queryLen] mask (all-sliding drafts
/// share isCausal/window and the layer caches append in lockstep), so the
/// first layer's mask — built from the true post-append context length,
/// compaction included — serves the remaining layers.
final class DFlash2SharedMask {
    var mask: MLXArray?
}

/// Draft attention: block queries attend to the cached context K/V (target
/// hidden projections) plus the block's own K/V. For `sliding_attention`
/// layers the context is distance-windowed against each block query; the
/// block itself is fully visible when the drafter is non-causal
/// (block-diffusion) and causally chained otherwise.
final class DFlash2Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let isSliding: Bool
    let slidingWindow: Int?
    let isCausal: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: DFlash2Configuration, layerIdx: Int) {
        numHeads = config.attentionHeads
        numKVHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(config.headDim), -0.5)
        isSliding = config.layerTypes[layerIdx] == "sliding_attention"
        slidingWindow = isSliding ? config.slidingWindow : nil
        // Reference: `self.is_causal = self.is_sliding if config.is_causal is None`
        isCausal = config.isCausal ?? isSliding

        _qProj.wrappedValue = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        context xCtx: MLXArray,
        rope: RoPELayer,
        cache: DFlash2ContextCache,
        sharedMask: DFlash2SharedMask? = nil
    ) -> MLXArray {
        let b = x.dim(0)
        let l = x.dim(1)
        var xCtx = xCtx
        var s = xCtx.dim(1)

        // An oversized incoming context keeps only its newest `maxSize` rows;
        // the skipped rows are already represented (or deliberately dropped)
        // in the cache timeline, so the write position still advances.
        if isSliding, let keep = slidingWindow.map({ $0 - 1 }), s > keep {
            let skip = s - keep
            xCtx = xCtx[0..., skip..., 0...]
            s = keep
            cache.offset += skip
        }

        var queries = qProj(x).reshaped(b, l, numHeads, headDim)
        var ctxKeys = kProj(xCtx).reshaped(b, s, numKVHeads, headDim)
        var ctxValues = vProj(xCtx).reshaped(b, s, numKVHeads, headDim)
        var propKeys = kProj(x).reshaped(b, l, numKVHeads, headDim)
        var propValues = vProj(x).reshaped(b, l, numKVHeads, headDim)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        ctxKeys = kNorm(ctxKeys).transposed(0, 2, 1, 3)
        ctxValues = ctxValues.transposed(0, 2, 1, 3)
        propKeys = kNorm(propKeys).transposed(0, 2, 1, 3)
        propValues = propValues.transposed(0, 2, 1, 3)

        let ctxOffset = cache.offset
        queries = applyRotaryPosition(rope, to: queries, offset: .scalar(ctxOffset + s))
        ctxKeys = applyRotaryPosition(rope, to: ctxKeys, offset: .scalar(ctxOffset))
        propKeys = applyRotaryPosition(rope, to: propKeys, offset: .scalar(ctxOffset + s))

        let (keys0, values0) = cache.append(keys: ctxKeys, values: ctxValues)
        let ctxLen = keys0.dim(2)
        let keys = concatenated([keys0, propKeys], axis: 2)
        let values = concatenated([values0, propValues], axis: 2)

        let mask: MLXArray
        if let sharedMask {
            if let memoized = sharedMask.mask {
                mask = memoized
            } else {
                let built = Self.makeMask(
                    queryLen: l, contextLen: ctxLen, isCausal: isCausal,
                    slidingWindow: isSliding ? slidingWindow : nil)
                sharedMask.mask = built
                mask = built
            }
        } else {
            mask = Self.makeMask(
                queryLen: l, contextLen: ctxLen, isCausal: isCausal,
                slidingWindow: isSliding ? slidingWindow : nil)
        }
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(b, l, numHeads * headDim))
    }

    /// Bool mask [queryLen, contextLen + queryLen]: `true` = attend.
    /// Positions are indices into the concatenated [context, block] axis —
    /// distance-based windowing matches the reference mask exactly.
    static func makeMask(
        queryLen l: Int, contextLen ctxLen: Int, isCausal: Bool, slidingWindow: Int?
    ) -> MLXArray {
        let kvLen = ctxLen + l
        let query = (MLXArray(Int32(0) ..< Int32(l)) + Int32(ctxLen)).expandedDimensions(axis: 1)
        let key = MLXArray(Int32(0) ..< Int32(kvLen)).expandedDimensions(axis: 0)
        var visible: MLXArray
        if let window = slidingWindow {
            let contextVisible = (key .< Int32(ctxLen)) & ((query - key) .< Int32(window))
            var blockVisible = key .>= Int32(ctxLen)
            if isCausal {
                blockVisible = blockVisible & (key .<= query)
            }
            visible = contextVisible | blockVisible
        } else {
            // Full attention over context; the block is bidirectional unless
            // the drafter is causal.
            visible = MLXArray.ones([l, kvLen], dtype: .bool)
            if isCausal {
                visible = key .<= query
            }
        }
        return visible
    }
}

// MARK: - Decoder layer

/// Pre-norm decoder layer. DFlash2 inserts a two-tap dynamic conv before AND
/// after each sublayer (`prepare` on the normed input, `finish` on the
/// sublayer output inside the residual add).
final class DFlash2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DFlash2Attention
    @ModuleInfo(key: "mlp") var mlp: DFlash2MLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "attention_conv") var attentionConv: DFlash2DynamicConv
    @ModuleInfo(key: "mlp_conv") var mlpConv: DFlash2DynamicConv

    init(_ config: DFlash2Configuration, layerIdx: Int) {
        _selfAttn.wrappedValue = DFlash2Attention(config, layerIdx: layerIdx)
        _mlp.wrappedValue = DFlash2MLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attentionConv.wrappedValue = DFlash2DynamicConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize)
        _mlpConv.wrappedValue = DFlash2DynamicConv(
            hiddenSize: config.hiddenSize,
            kernelSize: config.dflash.convKernelSize,
            groupSize: config.dflash.convGroupSize)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        context xCtx: MLXArray,
        rope: RoPELayer,
        cache: DFlash2ContextCache,
        sharedMask: DFlash2SharedMask? = nil
    ) -> MLXArray {
        var residual = x
        var (h, kernel) = attentionConv.prepare(inputLayerNorm(x))
        h = selfAttn(h, context: xCtx, rope: rope, cache: cache, sharedMask: sharedMask)
        var out = residual + attentionConv.finish(h, dynamic: kernel)
        residual = out
        (h, kernel) = mlpConv.prepare(postAttentionLayerNorm(out))
        out = residual + mlpConv.finish(mlp(h), dynamic: kernel)
        return out
    }
}

/// Qwen3-style SwiGLU MLP (`gate_proj`/`up_proj`/`down_proj`, no biases).
final class DFlash2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: DFlash2Configuration) {
        _gateProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Candidate selector

/// The DFlash2 path selector. Keeps the top-K draft candidates per position
/// and scores every adjacent pair with a low-rank bilinear match gated by the
/// position's hidden state: S_t(a, b) = U_t(b) + <A(a) ⊙ H(h_t), B(b)>. The
/// only sequential work is the walk over precomputed scores (greedy follows
/// the best successor; sampling draws from softmax(scores / T)).
final class DFlash2CandidateSelector: Module {
    let topK: Int

    @ModuleInfo(key: "predecessor_codebook") var predecessorCodebook: Embedding
    @ModuleInfo(key: "successor_codebook") var successorCodebook: Embedding
    @ModuleInfo(key: "hidden_projection") var hiddenProjection: Linear

    init(_ config: DFlash2Configuration) {
        topK = config.dflash.selectorTopK
        _predecessorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.dflash.selectorRank)
        _successorCodebook.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.dflash.selectorRank)
        _hiddenProjection.wrappedValue = Linear(
            config.hiddenSize, config.dflash.selectorRank, bias: false)
        super.init()
    }

    /// - Parameters:
    ///   - hidden: post-norm block hidden states [B, L, H] (logitsStart already applied).
    ///   - logits: matching draft logits [B, L, V].
    ///   - anchorIds: the last verified token, [B].
    ///   - temperature: 0 → greedy path; > 0 → categorical over softmax(scores / T).
    /// - Returns: (selected tokens [B, L], top-K candidate ids [B, L, K],
    ///   per-position selection probabilities [B, L, K] when sampling).
    func select(
        hidden: MLXArray, logits: MLXArray, anchorIds: MLXArray, temperature: Float
    ) -> (MLXArray, MLXArray, MLXArray?) {
        let L = logits.dim(1)
        let vocab = logits.dim(-1)
        let kth = vocab - topK
        let candidates = MLX.argPartition(logits, kth: kth, axis: -1)[.ellipsis, kth...]
        let unary = MLX.takeAlong(logits, candidates, axis: -1)
        let projected = hiddenProjection(hidden)  // [B, L, R]

        // Batched bigram edge scores for every boundary at once: boundary t
        // scores predecessor candidate a (position t-1) against successor
        // candidate b (position t) as <A(a) ⊙ H(h_t), B(b)> — the gate is the
        // SUCCESSOR position's hidden. Boundary 0's predecessor is the anchor
        // (not a candidate).
        let succEmbs = successorCodebook(candidates)  // [B, L, K, R]
        let predEmbs = predecessorCodebook(candidates)  // [B, L, K, R]
        let edges0 =
            ((predecessorCodebook(anchorIds) * projected[0..., 0])[0..., .newAxis, 0...]
            * succEmbs[0..., 0, 0..., 0...]).sum(axis: -1)  // [B, K]
        // edges[t][a][b] for boundaries 1...L-1, batched: [B, L-1, K, K]
        let gatedPrev = predEmbs[0..., 0 ..< (L - 1), 0..., 0...]
            * projected[0..., 1..., .newAxis, 0...]  // A(cand_{t-1}) ⊙ H(h_t)
        let edges = MLX.matmul(
            gatedPrev,
            succEmbs[0..., 1..., 0..., 0...].transposed(0, 1, 3, 2))

        var path: [MLXArray] = []
        var qRows: [MLXArray] = []
        var selectedIndex: MLXArray? = nil
        for position in 0 ..< L {
            let scores: MLXArray
            if position == 0 {
                scores = unary[0..., 0] + edges0
            } else {
                // The walked predecessor's edge row at this boundary.
                let aRow = takeAlong(
                    edges[0..., position - 1, 0..., 0...],
                    selectedIndex![0..., .newAxis, .newAxis], axis: 1
                )[0..., 0, 0...]
                scores = unary[0..., position] + aRow
            }

            let selected: MLXArray
            if temperature > 0 {
                let q = MLX.softmax(scores.asType(.float32) / temperature, axis: -1)
                qRows.append(q)
                selected = categorical(log(q))
            } else {
                selected = argMax(scores, axis: -1)
            }
            selectedIndex = selected
            path.append(
                MLX.takeAlong(
                    candidates[0..., position], selected[.newAxis, 0...], axis: -1
                )[0..., 0])
        }
        return (
            MLX.stacked(path, axis: 1), candidates,
            qRows.isEmpty ? nil : MLX.stacked(qRows, axis: 1)
        )
    }
}

// MARK: - Draft model

/// DFlash2 draft model. Not a `LanguageModel`: its I/O contract is
/// block-parallel drafting from target hidden states, so it conforms to
/// `BaseLanguageModel` (weight loading) + `DFlash2DrafterModel` (the
/// iterator-facing drafter protocol from MLXLMCommon).
public final class DFlash2DraftModel: Module, DFlash2DrafterModel {
    public let config: DFlash2Configuration

    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [DFlash2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "candidate_selector") var candidateSelector: DFlash2CandidateSelector

    let rope: RoPELayer

    /// Borrowed from the target at ``bind(target:)``; never loaded from the
    /// draft checkpoint (it has neither tensor).
    public private(set) var embedTokens: Embedding?
    public private(set) var lmHead: Linear?

    public init(_ config: DFlash2Configuration) {
        self.config = config
        let concatDim = config.dflash.targetLayerIds.count * config.hiddenSize
        _fc.wrappedValue = Linear(concatDim, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            DFlash2DecoderLayer(config, layerIdx: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _candidateSelector.wrappedValue = DFlash2CandidateSelector(config)
        rope = initializeRope(
            dims: config.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
        super.init()
    }

    // MARK: DFlash2DrafterModel

    public var dflashBlockSize: Int { config.dflash.blockSize }
    public var dflashMaskTokenId: Int { config.dflash.maskTokenId }
    public var dflashTargetLayerIds: [Int] { config.dflash.targetLayerIds }
    public var dflashContextKeepCount: Int? {
        // The hidden-state window the iterator must retain from the target:
        // bounded only when every draft layer is sliding (mirror of the
        // reference's `hidden_limit`).
        layerTypesAllSliding ? config.contextKeepCount : nil
    }
    public var dflashNumTargetLayers: Int { config.numTargetLayers }

    private var layerTypesAllSliding: Bool {
        config.layerTypes.allSatisfy { $0 == "sliding_attention" }
    }

    /// Point the drafter at its target's embedding table and LM head.
    /// Supports the Qwen3.5 family (dense text, VLM-wrapped text, MoE).
    public func bindDFlashTarget(_ target: any LanguageModel) {
        switch target {
        case let model as Qwen35TextModel:
            embedTokens = model.model.embedTokens
            lmHead = model.lmHead
        case let model as Qwen35Model:
            embedTokens = model.languageModel.model.embedTokens
            lmHead = model.languageModel.lmHead
        default:
            break
        }
    }

    /// Bind directly to an embedding table + LM head (test fixtures, exotic
    /// targets the switch doesn't know).
    public func bindDFlashTarget(embedding: Embedding, head: Linear?) {
        embedTokens = embedding
        lmHead = head
    }

    /// Fresh per-stream context caches (one per layer).
    public func makeDFlashContextCaches() -> [DFlash2ContextCache] {
        layers.map { _ in
            DFlash2ContextCache(maxSize: config.contextKeepCount ?? Int.max)
        }
    }

    /// Block hidden states: embed the [anchor, MASK...] block, attend to the
    /// cached context plus the newly projected `targetHidden` rows, norm.
    ///
    /// - Parameters:
    ///   - inputs: block token ids [B, L] (position 0 is the verified anchor).
    ///   - targetHidden: concatenated target layer outputs [B, S, nLayers*H]
    ///     for the context rows accepted since the previous round (the whole
    ///     prompt window on the first round).
    ///   - logitsStart: drop this many leading positions from the result
    ///     (1 during drafting — the anchor's own prediction is unused).
    public func hiddenStates(
        _ inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [DFlash2ContextCache],
        logitsStart: Int = 0
    ) -> MLXArray {
        guard let embedTokens else {
            fatalError("DFlash2DraftModel used before bind(target:)")
        }
        var h = embedTokens(inputs) * config.dflash.inputEmbeddingScale
        let hCtx = hiddenNorm(fc(targetHidden))
        // All-sliding drafters build the identical attention mask in every
        // layer; memoize the first layer's (built from the true post-append
        // context length) and reuse it for the rest.
        let sharedMask = layerTypesAllSliding ? DFlash2SharedMask() : nil
        for (layer, layerCache) in zip(layers, cache) {
            h = layer(h, context: hCtx, rope: rope, cache: layerCache, sharedMask: sharedMask)
        }
        if logitsStart > 0 {
            h = h[0..., logitsStart..., 0...]
        }
        return norm(h)
    }

    /// Draft logits for hidden states produced by ``hiddenStates``.
    public func computeLogits(_ hidden: MLXArray) -> MLXArray {
        var logits: MLXArray
        if let lmHead {
            logits = lmHead(hidden)
        } else if let embedTokens {
            logits = embedTokens.asLinear(hidden)
        } else {
            fatalError("DFlash2DraftModel used before bind(target:)")
        }
        logits = logits * config.dflash.outputMultiplier
        if let cap = config.dflash.finalLogitSoftcapping, cap > 0 {
            logits = tanh(logits / cap) * cap
        }
        return logits
    }

    /// One-pass block proposal: draft logits over positions 1..., then the
    /// selector's path through the top-K candidates.
    public func dflashPropose(
        _ inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [DFlash2ContextCache],
        temperature: Float,
        logitsStart: Int
    ) -> (tokens: MLXArray, candidates: MLXArray, probabilities: MLXArray?) {
        let hidden = hiddenStates(
            inputs, targetHidden: targetHidden, cache: cache, logitsStart: logitsStart)
        return candidateSelector.select(
            hidden: hidden, logits: computeLogits(hidden),
            anchorIds: inputs[0..., 0], temperature: temperature)
    }

    /// Checkpoint layout fix-ups. The safetensors stores the selector
    /// codebooks as bare tensors (`candidate_selector.predecessor_codebook`),
    /// which the `Embedding` modules expect under a `.weight` suffix.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        for name in ["predecessor_codebook", "successor_codebook"] {
            let bare = "candidate_selector.\(name)"
            if let value = weights.removeValue(forKey: bare) {
                weights["\(bare).weight"] = value
            }
        }
        return weights
    }
}

// MARK: - Loader

/// Load a DFlash/DFlash2 draft checkpoint from a local directory (config.json
/// + safetensors). The returned model still needs ``DFlash2DraftModel/bind(target:)``.
///
/// `quantization` post-quantizes the draft in place (reference:
/// `nn.quantize(draft, group_size: 64, bits: 4)` — every `Linear` and the
/// selector's `Embedding` codebooks). The draft is re-run every round, so the
/// 4-bit variant is what production wants; parity fixtures keep bf16.
public func loadDFlash2Draft(
    from directory: URL,
    quantization: (groupSize: Int, bits: Int)? = nil
) throws -> DFlash2DraftModel {
    let configURL = directory.appendingPathComponent("config.json")
    let configData = try Data(contentsOf: configURL)
    let config = try JSONDecoder.json5().decode(DFlash2Configuration.self, from: configData)
    let model = DFlash2DraftModel(config)
    try loadWeights(modelDirectory: directory, model: model)
    if let quantization {
        quantize(model: model, groupSize: quantization.groupSize, bits: quantization.bits)
        eval(model)
    }
    return model
}
