//
//  LFM3.swift
//  mlx-swift-lm
//

// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/lfm3

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// The operator of one LFM3 layer.
public enum LFM3LayerType: String, Codable, Sendable {
    case conv
    case fullAttention = "full_attention"
    case slidingAttention = "sliding_attention"

    public var isAttention: Bool { self != .conv }
}

public struct LFM3Configuration: Codable, Sendable {
    public var modelType: String
    public var vocabularySize: Int
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var hiddenLayers: Int
    public var attentionHeads: Int
    public var kvHeads: Int
    public var headDim: Int
    public var normEps: Float
    public var convBias: Bool
    public var convLCache: Int
    public var layerTypes: [LFM3LayerType]
    public var slidingWindow: Int?
    public var useSandwichNorm: Bool
    public var tieWordEmbeddings: Bool
    /// RoPE parameters keyed by attention layer type (`full_attention`, `sliding_attention`).
    public var ropeParameters: [String: [String: StringOrNumber]]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case normEps = "norm_eps"
        case convBias = "conv_bias"
        case convLCache = "conv_L_cache"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
        case useSandwichNorm = "use_sandwich_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
    }

    public init(
        modelType: String = "lfm3",
        vocabularySize: Int = 128_000,
        hiddenSize: Int = 2048,
        intermediateSize: Int = 7680,
        hiddenLayers: Int = 24,
        attentionHeads: Int = 16,
        kvHeads: Int = 4,
        headDim: Int? = nil,
        normEps: Float = 1e-5,
        convBias: Bool = false,
        convLCache: Int = 7,
        layerTypes: [LFM3LayerType]? = nil,
        slidingWindow: Int? = 256,
        useSandwichNorm: Bool = true,
        tieWordEmbeddings: Bool = true,
        ropeParameters: [String: [String: StringOrNumber]]? = nil
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDim = headDim ?? hiddenSize / attentionHeads
        self.normEps = normEps
        self.convBias = convBias
        self.convLCache = convLCache
        self.layerTypes = layerTypes ?? Self.defaultLayerTypes(count: hiddenLayers)
        self.slidingWindow = slidingWindow
        self.useSandwichNorm = useSandwichNorm
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeParameters = ropeParameters ?? Self.defaultRoPEParameters
    }

    private enum AliasKeys: String, CodingKey {
        case tieEmbedding = "tie_embedding"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        // As in `Lfm3Config`, the `tie_embedding` alias wins.
        let tieEmbedding = try decoder.container(keyedBy: AliasKeys.self)
            .decodeIfPresent(Bool.self, forKey: .tieEmbedding)
        let hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        let attentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16

        // A flat dictionary applies to every attention layer type.
        var ropeParameters = try? container.decodeIfPresent(
            [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
        if ropeParameters == nil,
            let flat = try? container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)
        {
            ropeParameters = [
                LFM3LayerType.fullAttention.rawValue: flat,
                LFM3LayerType.slidingAttention.rawValue: flat,
            ]
        }

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3",
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? 128_000,
            hiddenSize: hiddenSize,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? 7680,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24,
            attentionHeads: attentionHeads,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim),
            normEps: try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5,
            convBias: try container.decodeIfPresent(Bool.self, forKey: .convBias) ?? false,
            convLCache: try container.decodeIfPresent(Int.self, forKey: .convLCache) ?? 7,
            layerTypes: try container.decodeIfPresent([LFM3LayerType].self, forKey: .layerTypes),
            slidingWindow: try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 256,
            useSandwichNorm: try container.decodeIfPresent(Bool.self, forKey: .useSandwichNorm)
                ?? true,
            tieWordEmbeddings: try tieEmbedding
                ?? container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true,
            ropeParameters: ropeParameters)
    }

    /// The reference default: full attention every fourth layer from the third, sliding
    /// attention between them, gated short convolutions elsewhere.
    static func defaultLayerTypes(count: Int) -> [LFM3LayerType] {
        (0 ..< count).map { i in
            if i % 4 == 2 { return .fullAttention }
            if i % 4 == 0 && i > 0 { return .slidingAttention }
            return .conv
        }
    }

    static let defaultRoPEParameters: [String: [String: StringOrNumber]] = [
        LFM3LayerType.fullAttention.rawValue: [
            "rope_type": .string("default"), "rope_theta": .float(1_000_000),
            "partial_rotary_factor": .float(0.5),
        ],
        LFM3LayerType.slidingAttention.rawValue: [
            "rope_type": .string("default"), "rope_theta": .float(10_000),
        ],
    ]
}

extension LFM3Configuration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard layerTypes.count == hiddenLayers else {
            throw ModelFactoryError.invalidConfiguration(
                "LFM3: layer_types has \(layerTypes.count) entries but num_hidden_layers is \(hiddenLayers)"
            )
        }
        if layerTypes.contains(.slidingAttention), (slidingWindow ?? 0) <= 0 {
            throw ModelFactoryError.invalidConfiguration(
                "LFM3: sliding_attention layers need a positive sliding_window")
        }
        for layerType in [LFM3LayerType.fullAttention, .slidingAttention]
        where layerTypes.contains(layerType) {
            let context = "LFM3 rope_parameters.\(layerType.rawValue)"
            guard let parameters = ropeParameters[layerType.rawValue] else {
                throw ModelFactoryError.invalidConfiguration("\(context) is missing")
            }
            try validateRoPEConfiguration(parameters, context: context, supportedTypes: ["default"])
            let factor = parameters["partial_rotary_factor"]?.asFloat() ?? 1
            guard factor > 0, factor <= 1 else {
                throw ModelFactoryError.invalidConfiguration(
                    "\(context): partial_rotary_factor must be in (0, 1]")
            }
        }
    }
}

/// LFM3's partial RoPE keeps the frequency progression of the full head and leaves the
/// tail of each half unrotated (the `proportional` layout of Gemma 4). One fast RoPE call
/// does it, where `ProportionalRoPE` splits and concatenates the halves around one: the tail
/// pairs get infinite periods, so they rotate by zero.
final class LFM3PartialRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dimensions: Int
    let _freqs: MLXArray

    init(dimensions: Int, base: Float, partialRotaryFactor: Float) {
        self.dimensions = dimensions
        let angles = Int(partialRotaryFactor * Float(dimensions)) / 2
        let periods = (0 ..< dimensions / 2).map {
            $0 < angles ? pow(base, 2 * Float($0) / Float(dimensions)) : Float.infinity
        }
        self._freqs = MLXArray(periods)
    }

    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dimensions, traditional: false, base: nil, scale: 1, offset: offset,
            freqs: _freqs)
    }

    func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dimensions, traditional: false, base: nil, scale: 1, offset: offset,
            freqs: _freqs)
    }
}

func makeLFM3RoPE(headDim: Int, parameters: [String: StringOrNumber]?) -> RoPELayer {
    let base = parameters?["rope_theta"]?.asFloat() ?? 1_000_000
    let factor = parameters?["partial_rotary_factor"]?.asFloat() ?? 1
    guard factor < 1 else {
        return RoPE(dimensions: headDim, traditional: false, base: base)
    }
    return LFM3PartialRoPE(dimensions: headDim, base: base, partialRotaryFactor: factor)
}

/// Attention with QK RMSNorm and a sigmoid output gate per head.
final class LFM3Attention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    init(_ args: LFM3Configuration, layerType: LFM3LayerType) {
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDim
        self.scale = pow(Float(headDim), -0.5)
        self.rope = makeLFM3RoPE(
            headDim: headDim, parameters: args.ropeParameters[layerType.rawValue])

        let dim = args.hiddenSize
        _qProj.wrappedValue = Linear(dim, attentionHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(attentionHeads * headDim, dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.normEps)
        _gateProj.wrappedValue = Linear(dim, attentionHeads, bias: false)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(qProj(x).reshaped(B, L, attentionHeads, headDim))
            .transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)

        let gate = sigmoid(gateProj(x))[.ellipsis, .newAxis]
        return oProj((output * gate).reshaped(B, L, -1))
    }
}

/// LFM2's gated short convolution, here with a width-7 kernel.
final class LFM3ShortConv: Module {
    let kernelSize: Int

    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ args: LFM3Configuration) {
        self.kernelSize = args.convLCache
        _conv.wrappedValue = Conv1d(
            inputChannels: args.hiddenSize, outputChannels: args.hiddenSize,
            kernelSize: kernelSize, groups: args.hiddenSize, bias: args.convBias)
        _inProj.wrappedValue = Linear(args.hiddenSize, 3 * args.hiddenSize, bias: args.convBias)
        _outProj.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: args.convBias)
    }

    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let parts = inProj(x).split(parts: 3, axis: -1)
        let (b, c, h) = (parts[0], parts[1], parts[2])
        var bx = b * h

        let state =
            cache?[0]
            ?? MLXArray.zeros([bx.dim(0), kernelSize - 1, bx.dim(2)], dtype: bx.dtype)
        bx = concatenated([state, bx], axis: -2)
        if let cache {
            cache[0] = contiguous(bx[0..., (bx.dim(1) - (kernelSize - 1))..., 0...])
            cache.advance(x.dim(1))
        }

        return outProj(c * conv(bx))
    }
}

final class LFM3MLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _w1.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _w2.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _w3.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

/// One LFM3 block: an attention or short-convolution operator and a SwiGLU feed-forward,
/// each normalized on input and, with the sandwich norm, on output.
public final class LFM3DecoderLayer: Module {
    public let layerType: LFM3LayerType

    @ModuleInfo(key: "self_attn") var attention: LFM3Attention?
    @ModuleInfo(key: "conv") var conv: LFM3ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm?

    public init(_ args: LFM3Configuration, layerIndex: Int) {
        self.layerType = args.layerTypes[layerIndex]
        if layerType.isAttention {
            _attention.wrappedValue = LFM3Attention(args, layerType: layerType)
        } else {
            _conv.wrappedValue = LFM3ShortConv(args)
        }
        _feedForward.wrappedValue = LFM3MLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        if args.useSandwichNorm {
            _postOperatorNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.normEps)
            _postFFNNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
        }
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var h = operatorNorm(x)
        if let attention {
            h = attention(h, mask: mask, cache: cache)
        } else {
            h = conv!(h, cache: cache as? MambaCache)
        }
        if let postOperatorNorm {
            h = postOperatorNorm(h)
        }
        let r = x + h

        var f = feedForward(ffnNorm(r))
        if let postFFNNorm {
            f = postFFNNorm(f)
        }
        return r + f
    }
}

/// The LFM3 trunk: token embedding, layers and the final norm (named `embedding_norm`).
///
/// Trunks that read embeddings only (the LFM3-Audio codec decoder) have no token embedding.
public final class LFM3ModelInner: Module {
    let args: LFM3Configuration
    let firstFullAttention: Int?
    let firstSlidingAttention: Int?

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding?
    public let layers: [LFM3DecoderLayer]
    @ModuleInfo(key: "embedding_norm") public var embeddingNorm: RMSNorm

    public init(_ args: LFM3Configuration, embedsTokens: Bool = true) {
        self.args = args
        self.firstFullAttention = args.layerTypes.firstIndex(of: .fullAttention)
        self.firstSlidingAttention = args.layerTypes.firstIndex(of: .slidingAttention)
        if embedsTokens {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        }
        self.layers = (0 ..< args.hiddenLayers).map { LFM3DecoderLayer(args, layerIndex: $0) }
        _embeddingNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    /// Causal forward over `inputs` (or `inputEmbeddings`), returning normalized hidden states.
    public func callAsFunction(
        _ inputs: MLXArray?, cache: [KVCache]?, inputEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        guard let embedded = inputEmbeddings ?? inputs.flatMap({ embedTokens?($0) }) else {
            fatalError("LFM3ModelInner needs token ids with an embedding, or input embeddings")
        }
        var h = embedded

        let fullMask = firstFullAttention.map { createAttentionMask(h: h, cache: cache?[$0]) }
        let slidingMask = firstSlidingAttention.map {
            createAttentionMask(h: h, cache: cache?[$0], windowSize: args.slidingWindow)
        }
        for (i, layer) in layers.enumerated() {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                switch layer.layerType {
                case .fullAttention: fullMask ?? .none
                case .slidingAttention: slidingMask ?? .none
                case .conv: .none
                }
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return embeddingNorm(h)
    }

    /// Caches by layer type: short-convolution state, a ring of `sliding_window` keys, or a
    /// full attention cache that honors the requested capacity.
    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try args.layerTypes.map { layerType in
            switch layerType {
            case .conv:
                MambaCache()
            case .slidingAttention:
                try makeSlidingWindowKVCache(
                    parameters: parameters, window: args.slidingWindow ?? 0)
            case .fullAttention:
                try makeAttentionKVCache(parameters: parameters)
            }
        }
    }
}

public final class LFM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let configuration: LFM3Configuration

    public let model: LFM3ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ args: LFM3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = args.layerTypes.map { $0.isAttention ? args.kvHeads : 0 }
        self.model = LFM3ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(model(inputs, cache: cache))
    }

    /// Logits over `inputs`, or over `inputEmbeddings` when a multimodal wrapper supplies them.
    public func callAsFunction(
        _ inputs: MLXArray?, cache: [KVCache]?, inputEmbeddings: MLXArray?
    ) -> MLXArray {
        logits(model(inputs, cache: cache, inputEmbeddings: inputEmbeddings))
    }

    /// The output head over normalized hidden states.
    public func logits(_ hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens!.asLinear(hiddenStates)
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        // mlx-vlm conversions keep the backbone under `language_model`.
        var rules: [CheckpointNameMapping.Rule] = [.replacePrefix("language_model", with: "")]
        if configuration.tieWordEmbeddings {
            rules.append(.excludeModule("lm_head"))
        }
        var checkpoint = try checkpoint.mapNames(using: .init(rules))
        checkpoint.weights = Self.sanitizeConvWeights(checkpoint.weights)
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try model.newCache(parameters: parameters)
    }

    /// PyTorch stores the depthwise short-convolution kernels as `[channels, 1, width]`;
    /// MLX wants `[channels, width, 1]`.
    public static func sanitizeConvWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        for (key, value) in weights
        where key.hasSuffix("conv.conv.weight") && value.ndim == 3 && value.dim(2) > value.dim(1) {
            weights[key] = value.transposed(0, 2, 1)
        }
        return weights
    }
}

extension LFM3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}

// MARK: - Chat conventions

extension ReasoningConfig {
    /// LFM3's `<think>` protocol. The tags are added tokens but not special ones, and a
    /// tool call can follow the thinking directly.
    public static let lfm3 = ReasoningConfig(
        startDelimiter: "<think>", endDelimiter: "</think>",
        promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: true),
        isSpecialToken: false,
        implicitEndDelimiters: ["<tool_call>"])
}

extension LFM3Model {
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }
    public var reasoningConfig: ReasoningConfig? { .lfm3 }
}
