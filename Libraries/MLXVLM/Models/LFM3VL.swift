// Copyright © 2026 Apple Inc.

// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/lfm3_vl

import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Vision

private enum LFM3Vision {

    final class MLP: Module, UnaryLayer {
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

    /// Bidirectional grouped-query attention over the patches of each frame, with RMSNorm on
    /// the query and key heads.
    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear
        @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

        init(_ config: LFM3VLConfiguration.VisionConfiguration) {
            self.heads = config.attentionHeads
            self.kvHeads = config.kvHeads
            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)
            let dim = config.hiddenSize
            _qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
            _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _outProj.wrappedValue = Linear(heads * headDim, dim, bias: false)
            _qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            _kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        }

        /// `x` is `[frames, patches, hidden]`; `cos` and `sin` are `[patches, 1, headDim]`.
        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            let (frames, patches) = (x.dim(0), x.dim(1))
            let q = qLayerNorm(qProj(x).reshaped(frames, patches, heads, headDim))
            let k = kLayerNorm(kProj(x).reshaped(frames, patches, kvHeads, headDim))
            let v = vProj(x).reshaped(frames, patches, kvHeads, headDim)

            // Like the reference, rotate in float32.
            func rotate(_ t: MLXArray) -> MLXArray {
                let t32 = t.asType(.float32)
                return (t32 * cos + QwenVL.rotateHalf(t32) * sin).asType(t.dtype)
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: rotate(q).transposed(0, 2, 1, 3),
                keys: rotate(k).transposed(0, 2, 1, 3),
                values: v.transposed(0, 2, 1, 3),
                scale: scale, mask: .none)
            return outProj(output.transposed(0, 2, 1, 3).reshaped(frames, patches, -1))
        }
    }

    /// Gated depthwise 2D convolution over the patch grid of each frame.
    final class Conv: Module {
        let mergeSize: Int

        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "spatial_conv") var spatialConv: Conv2d
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: LFM3VLConfiguration.VisionConfiguration) {
            self.mergeSize = config.spatialMergeSize
            let dim = config.hiddenSize
            _inProj.wrappedValue = Linear(dim, 3 * dim, bias: false)
            _spatialConv.wrappedValue = Conv2d(
                inputChannels: dim, outputChannels: dim,
                kernelSize: .init(config.convKernelSize),
                padding: .init(config.convKernelSize / 2), groups: dim, bias: false)
            _outProj.wrappedValue = Linear(dim, dim, bias: false)
        }

        /// `x` is `[frames, height * width, hidden]` in merge-block order.
        func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
            let parts = inProj(x).split(parts: 3, axis: -1)
            let (b, c, v) = (parts[0], parts[1], parts[2])
            let (frames, dim, m) = (x.dim(0), x.dim(2), mergeSize)

            // merge-block order -> raster grid, convolve, back
            let grid = (b * v)
                .reshaped(frames, height / m, width / m, m, m, dim)
                .transposed(0, 1, 3, 2, 4, 5)
                .reshaped(frames, height, width, dim)
            let convolved = spatialConv(grid)
                .reshaped(frames, height / m, m, width / m, m, dim)
                .transposed(0, 1, 3, 2, 4, 5)
                .reshaped(frames, height * width, dim)
            return outProj(c * convolved)
        }
    }

    final class Layer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention?
        @ModuleInfo(key: "conv") var conv: Conv?
        @ModuleInfo(key: "feed_forward") var feedForward: MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ config: LFM3VLConfiguration.VisionConfiguration, isAttention: Bool) {
            if isAttention {
                _attention.wrappedValue = Attention(config)
            } else {
                _conv.wrappedValue = Conv(config)
            }
            _feedForward.wrappedValue = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            _operatorNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
            _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(
            _ x: MLXArray, height: Int, width: Int, cos: MLXArray, sin: MLXArray
        ) -> MLXArray {
            let h = operatorNorm(x)
            let operatorOutput =
                if let attention {
                    attention(h, cos: cos, sin: sin)
                } else {
                    conv!(h, height: height, width: width)
                }
            let r = x + operatorOutput
            return r + feedForward(ffnNorm(r))
        }
    }

    /// Gated convolution and bidirectional attention layers over the patches of each image or
    /// video frame, embedded by a linear patchifier.
    final class VisionModel: Module {
        let config: LFM3VLConfiguration.VisionConfiguration

        @ModuleInfo(key: "patchifier") var patchifier: Linear
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm
        let layers: [Layer]

        init(_ config: LFM3VLConfiguration.VisionConfiguration) {
            self.config = config
            let patchDimensions =
                config.inChannels * config.temporalPatchSize * config.patchSize * config.patchSize
            _patchifier.wrappedValue = Linear(patchDimensions, config.hiddenSize, bias: false)
            _embeddingNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            self.layers = config.layerTypes.map {
                Layer(config, isAttention: $0 == "full_attention")
            }
        }

        /// Axial RoPE: the same frequencies over half the head for each axis, rotating the
        /// whole head. Patches are in merge-block order.
        private func rotaryAngles(height: Int, width: Int) -> (MLXArray, MLXArray) {
            let m = config.spatialMergeSize
            var rows = [Float]()
            var columns = [Float]()
            rows.reserveCapacity(height * width)
            columns.reserveCapacity(height * width)
            for blockRow in 0 ..< height / m {
                for blockColumn in 0 ..< width / m {
                    for row in 0 ..< m {
                        for column in 0 ..< m {
                            rows.append(Float(blockRow * m + row))
                            columns.append(Float(blockColumn * m + column))
                        }
                    }
                }
            }
            let spatialDim = config.headDim / 2
            let exponents =
                MLXArray(stride(from: 0, to: spatialDim, by: 2)).asType(.float32)
                / Float(spatialDim)
            let inverseFrequencies = 1 / pow(MLXArray(config.ropeTheta), exponents)
            let angles = concatenated(
                [
                    MLXArray(rows)[0..., .newAxis] * inverseFrequencies,
                    MLXArray(columns)[0..., .newAxis] * inverseFrequencies,
                ], axis: -1)
            let emb = concatenated([angles, angles], axis: -1)[0..., .newAxis, 0...]
            return (cos(emb), sin(emb))
        }

        /// `pixels` is `[patches, channels * temporal * patch * patch]`, one item after another
        /// in the order of `grids`. Returns one feature row per patch.
        func callAsFunction(_ pixels: MLXArray, grids: [THW]) throws -> MLXArray {
            var outputs = [MLXArray]()
            var offset = 0
            for grid in grids {
                let patches = grid.h * grid.w
                guard offset + grid.t * patches <= pixels.dim(0) else {
                    throw VLMError.processing("Pixel patch count does not match the image grids")
                }
                let (cos, sin) = rotaryAngles(height: grid.h, width: grid.w)
                // Each temporal frame attends and convolves on its own.
                var x = pixels[offset ..< offset + grid.t * patches]
                    .reshaped(grid.t, patches, -1)
                    .asType(embeddingNorm.weight.dtype)
                x = embeddingNorm(patchifier(x))
                for layer in layers {
                    x = layer(x, height: grid.h, width: grid.w, cos: cos, sin: sin)
                }
                outputs.append(x.reshaped(grid.t * patches, -1))
                offset += grid.t * patches
            }
            guard offset == pixels.dim(0) else {
                throw VLMError.processing("Pixel patch count does not match the image grids")
            }
            return concatenated(outputs, axis: 0)
        }
    }
}

/// Merges each block of `downsample_factor²` vision tokens into one language token.
private final class LFM3VLProjector: Module, UnaryLayer {
    let inputDimensions: Int

    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: LFM3VLConfiguration) {
        let visionHidden = config.visionConfiguration.hiddenSize
        self.inputDimensions = visionHidden * config.downsampleFactor * config.downsampleFactor
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: visionHidden, eps: config.projectorLayerNormEps)
        _linear1.wrappedValue = Linear(
            inputDimensions, config.projectorHiddenSize, bias: config.projectorBias)
        _linear2.wrappedValue = Linear(
            config.projectorHiddenSize, config.textConfiguration.hiddenSize,
            bias: config.projectorBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(layerNorm(x).reshaped(-1, inputDimensions))))
    }
}

// MARK: - Model

/// LFM3-VL: the LFM3 language model with an in-house convolution and attention vision encoder.
///
/// The language model is MLXLLM's ``LFM3Model``, so text-only prompts take its prefill.
public final class LFM3VL: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: LFM3Vision.VisionModel
    @ModuleInfo(key: "multi_modal_projector") private var projector: LFM3VLProjector
    @ModuleInfo(key: "language_model") public var languageModel: LFM3Model

    public let config: LFM3VLConfiguration

    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }

    public init(_ config: LFM3VLConfiguration) {
        self.config = config
        _visionTower.wrappedValue = LFM3Vision.VisionModel(config.visionConfiguration)
        _projector.wrappedValue = LFM3VLProjector(config)
        var text = config.textConfiguration
        text.tieWordEmbeddings = config.tieWordEmbeddings
        _languageModel.wrappedValue = LFM3Model(text)
    }

    /// Projected features of `pixels`: one language embedding per merge block.
    func features(_ pixels: MLXArray, grids: [THW]) throws -> MLXArray {
        projector(try visionTower(pixels, grids: grids))
    }

    /// Token embeddings with the image and video features scattered into their placeholders.
    func inputEmbeddings(_ input: LMInput) throws -> MLXArray {
        let tokens = input.text.tokens.ndim == 1 ? input.text.tokens[.newAxis] : input.text.tokens
        guard tokens.dim(0) == 1 else {
            throw VLMError.processing("LFM3-VL prepares one prompt with media at a time")
        }
        let embeddings = languageModel.model.embedTokens!(tokens)
        let ids = tokens[0].asArray(Int32.self)

        func scatter(_ media: (pixels: MLXArray, frames: [THW]?)?, tokenId: Int) throws {
            let positions = ids.indices.filter { ids[$0] == Int32(tokenId) }
            guard let (pixels, frames) = media else {
                if !positions.isEmpty {
                    throw VLMError.processing("Found \(positions.count) media tokens but no media")
                }
                return
            }
            guard let frames else {
                throw VLMError.processing("LFM3-VL needs the patch grid of every image and video")
            }
            let values = try features(pixels, grids: frames)
            guard values.dim(0) == positions.count else {
                throw VLMError.processing(
                    "Media features and tokens do not match: tokens \(positions.count), features \(values.dim(0))"
                )
            }
            embeddings[0..., MLXArray(positions.map(Int32.init)), 0...] =
                values.asType(embeddings.dtype)[.newAxis]
        }

        try scatter(input.image.map { ($0.pixels, $0.frames) }, tokenId: config.imageTokenId)
        try scatter(input.video.map { ($0.pixels, $0.frames) }, tokenId: config.videoTokenId)
        return embeddings
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.image != nil || input.video != nil else {
            return try languageModel.prepare(
                input.unbatchedText, cache: cache, state: state, prefill: prefill)
        }

        let embeddings = try inputEmbeddings(input)
        return try withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            let total = embeddings.dim(1)
            let processed = try prefill.forEachChunk(total: total) { range in
                _ = languageModel.model(nil, cache: cache, inputEmbeddings: embeddings[0..., range])
                asyncEval(cache)
            }
            if processed > 0 {
                eval(cache)
            }
            let hidden = languageModel.model(
                nil, cache: cache, inputEmbeddings: embeddings[0..., processed...])
            prefill.progress?(total, total)
            return .logits(LMOutput(logits: languageModel.logits(hidden[0..., (-1)...])))
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        // HF checkpoints nest the towers under `model.`; mlx-vlm conversions already match.
        var rules: [CheckpointNameMapping.Rule] = [
            .replacePrefix("model.language_model", with: "language_model.model"),
            .replacePrefix("model.vision_tower", with: "vision_tower"),
            .replacePrefix("model.multi_modal_projector", with: "multi_modal_projector"),
            .replacePrefix("lm_head", with: "language_model.lm_head"),
        ]
        if config.tieWordEmbeddings {
            rules.append(.excludeModule("lm_head"))
        }
        var checkpoint = try checkpoint.mapNames(using: .init(rules))
        var weights = LFM3Model.sanitizeConvWeights(checkpoint.weights)
        // PyTorch's depthwise Conv2d kernel is [out, 1, kh, kw]; MLX wants [out, kh, kw, 1].
        for (key, value) in weights
        where key.hasSuffix("spatial_conv.weight") && value.ndim == 4 && value.dim(1) == 1
            && value.dim(3) > 1
        {
            weights[key] = value.transposed(0, 2, 3, 1)
        }
        checkpoint.weights = weights
        return checkpoint
    }

    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        try prepareCheckpoint(.init(weights: weights)).weights
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.newCache(parameters: parameters)
    }
}

// MARK: - Processor

/// Images at native resolution in 32-pixel steps, as Qwen-VL lays them out: 16-pixel patches
/// grouped in 2 × 2 merge blocks, each image repeated over the two temporal slots.
public struct LFM3VLProcessor: UserInputProcessor {
    private let config: LFM3VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    /// Language tokens in the default image budget, as in the Qwen VL processors. The
    /// checkpoint allows 16 MP (16,384 tokens), whose attention over 65,536 patches takes
    /// seconds; `processing.maxPixels` overrides the budget.
    static let defaultImageTokenBudget = 1280

    public init(_ config: LFM3VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    /// The reference `smart_resize`: sides rounded to `factor` (half to even, as Python
    /// rounds), then scaled into `[minPixels, maxPixels]` keeping the aspect ratio.
    static func targetSize(
        height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int
    ) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0 else {
            throw VLMError.imageProcessingFailure("Image has no pixels: \(width) × \(height)")
        }
        guard max(height, width) / min(height, width) <= 200 else {
            throw VLMError.imageProcessingFailure(
                "Absolute aspect ratio must be smaller than 200: \(width) × \(height)")
        }
        let (h, w, f) = (Double(height), Double(width), Double(factor))
        var hBar = Int((h / f).rounded(.toNearestOrEven)) * factor
        var wBar = Int((w / f).rounded(.toNearestOrEven)) * factor
        if hBar * wBar > maxPixels {
            let beta = (h * w / Double(maxPixels)).squareRoot()
            hBar = max(factor, Int((h / beta / f).rounded(.down)) * factor)
            wBar = max(factor, Int((w / beta / f).rounded(.down)) * factor)
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / (h * w)).squareRoot()
            hBar = Int((h * beta / f).rounded(.up)) * factor
            wBar = Int((w * beta / f).rounded(.up)) * factor
        }
        return (hBar, wBar)
    }

    func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (MLXArray, THW) {
        let settings = config.imageProcessor
        let image = MediaProcessing.apply(image, processing: processing)
        let factor = settings.patchSize * settings.mergeSize
        let budget = Self.defaultImageTokenBudget * factor * factor
        let size = try Self.targetSize(
            height: Int(image.extent.height.rounded()), width: Int(image.extent.width.rounded()),
            factor: factor,
            minPixels: processing?.minPixels ?? settings.minPixels,
            maxPixels: processing?.maxPixels ?? min(settings.maxPixels, budget))

        // Resample in the sRGB tone-curve space: CoreImage works in linear light, and the
        // reference resizes the gamma-encoded bytes.
        let resized = MediaProcessing.resampleBicubic(
            MediaProcessing.inSRGBToneCurveSpace(image),
            to: CGSize(width: size.width, height: size.height))
        let pixels = MediaProcessing.asMLXArray(
            MediaProcessing.normalize(
                resized, mean: settings.imageMeanTuple, std: settings.imageStdTuple))
        return try QwenVL.patchify(
            images: [pixels], mergeSize: settings.mergeSize, patchSize: settings.patchSize,
            temporalPatchSize: settings.temporalPatchSize)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        if !input.videos.isEmpty {
            throw VLMError.processing("LFM3-VL video input is not supported yet")
        }
        let messages = LFM3VLMessageGenerator().generate(
            from: input.removingSpecialTokenLabels(using: tokenizer))
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        guard !input.images.isEmpty else {
            return LMInput(tokens: MLXArray(promptTokens).expandedDimensions(axis: 0))
        }

        let images = try input.images.map {
            try preprocess(image: $0.asCIImage(), processing: input.processing)
        }
        let frames = images.map(\.1)
        promptTokens = try QwenVL.replacePaddingTokens(
            in: promptTokens, frames: frames, paddingToken: "<|image|>",
            mergeSize: config.imageProcessor.mergeSize, tokenizer: tokenizer)

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray).asType(.int8)),
            image: .init(pixels: concatenated(images.map(\.0)), frames: frames))
    }
}

/// Images and videos before the text of each message, as the chat template expects, with an
/// assistant turn's `<think>` block passed as `reasoning_content`.
public struct LFM3VLMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        var message = message
        var reasoning: String?
        if message.role == .assistant {
            let start = ReasoningConfig.lfm3.startDelimiter
            let end = ReasoningConfig.lfm3.endDelimiter
            let body: Substring? =
                if message.prefilledReasoningStartDelimiter == start {
                    message.content[...]
                } else if message.content.hasPrefix(start) {
                    message.content.dropFirst(start.count)
                } else {
                    nil
                }
            if let body, let range = body.range(of: end) {
                reasoning = String(body[..<range.lowerBound])
                message.content = String(body[range.upperBound...])
            }
        }

        var dictionary: MLXLMCommon.Message = [
            "role": message.role.rawValue,
            "content": contentParts(for: message, layout: .imagesThenVideosThenText),
        ]
        if let reasoning {
            dictionary["reasoning_content"] = reasoning
        }
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

// MARK: - Configuration

public struct LFM3VLConfiguration: Codable, Sendable {

    public struct VisionConfiguration: Codable, Sendable {
        public var hiddenSize: Int
        public var intermediateSize: Int
        public var hiddenLayers: Int
        public var attentionHeads: Int
        public var kvHeads: Int
        public var inChannels: Int
        public var patchSize: Int
        public var temporalPatchSize: Int
        public var spatialMergeSize: Int
        public var convKernelSize: Int
        public var normEps: Float
        public var layerTypes: [String]
        public var ropeParameters: [String: StringOrNumber]

        public var headDim: Int { hiddenSize / attentionHeads }
        public var ropeTheta: Float { ropeParameters["rope_theta"]?.asFloat() ?? 1_000_000 }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case inChannels = "in_channels"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case convKernelSize = "conv_kernel_size"
            case normEps = "norm_eps"
            case layerTypes = "layer_types"
            case ropeParameters = "rope_parameters"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
            self.intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2560
            self.hiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 14
            self.attentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            self.inChannels = try container.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
            self.patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            self.temporalPatchSize =
                try container.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
            self.spatialMergeSize =
                try container.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
            self.convKernelSize =
                try container.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 3
            self.normEps = try container.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
            self.layerTypes =
                try container.decodeIfPresent([String].self, forKey: .layerTypes)
                ?? (0 ..< hiddenLayers).map { $0 % 2 == 0 ? "conv" : "full_attention" }
            self.ropeParameters =
                try container.decodeIfPresent(
                    [String: StringOrNumber].self, forKey: .ropeParameters)
                ?? ["rope_type": .string("axial"), "rope_theta": .float(1_000_000)]
        }
    }

    public var textConfiguration: LFM3Configuration
    public var visionConfiguration: VisionConfiguration
    public var modelType: String
    public var downsampleFactor: Int
    public var projectorHiddenSize: Int
    public var projectorHiddenAct: String
    public var projectorBias: Bool
    public var projectorLayerNormEps: Float
    public var imageTokenId: Int
    public var videoTokenId: Int
    public var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case downsampleFactor = "downsample_factor"
        case projectorHiddenSize = "projector_hidden_size"
        case projectorHiddenAct = "projector_hidden_act"
        case projectorBias = "projector_bias"
        case projectorLayerNormEps = "projector_layer_norm_eps"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.textConfiguration = try container.decode(
            LFM3Configuration.self, forKey: .textConfiguration)
        self.visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)
        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3_vl"
        self.downsampleFactor =
            try container.decodeIfPresent(Int.self, forKey: .downsampleFactor) ?? 2
        self.projectorHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 4096
        self.projectorHiddenAct =
            try container.decodeIfPresent(String.self, forKey: .projectorHiddenAct) ?? "gelu"
        self.projectorBias =
            try container.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? true
        self.projectorLayerNormEps =
            try container.decodeIfPresent(Float.self, forKey: .projectorLayerNormEps) ?? 1e-6
        self.imageTokenId =
            try container.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 125_030
        self.videoTokenId =
            try container.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 125_029
        // The composite's flag governs the head, as in `Lfm3VlForConditionalGeneration`.
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? textConfiguration.tieWordEmbeddings
    }
}

extension LFM3VLConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        try textConfiguration.validateModelConfiguration()
        let vision = visionConfiguration
        guard vision.layerTypes.count == vision.hiddenLayers,
            vision.layerTypes.allSatisfy({ $0 == "conv" || $0 == "full_attention" })
        else {
            throw ModelFactoryError.invalidConfiguration(
                "LFM3-VL: vision layer_types needs num_hidden_layers entries of conv or full_attention"
            )
        }
        if case .string(let ropeType) = vision.ropeParameters["rope_type"], ropeType != "axial" {
            throw ModelFactoryError.invalidConfiguration(
                "LFM3-VL: the vision encoder supports axial RoPE only, not \(ropeType)")
        }
        guard downsampleFactor == vision.spatialMergeSize else {
            throw ModelFactoryError.invalidConfiguration(
                "LFM3-VL: downsample_factor must equal the vision spatial_merge_size")
        }
        guard projectorHiddenAct == "gelu" else {
            throw ModelFactoryError.invalidConfiguration(
                "LFM3-VL: unsupported projector activation \(projectorHiddenAct)")
        }
    }
}

/// `processor_config.json`: the image settings under `image_processor` (or at the top level).
public struct LFM3VLProcessorConfiguration: Codable, Sendable {

    public struct ImageProcessor: Codable, Sendable {
        public struct Size: Codable, Sendable {
            public var shortestEdge: Int?
            public var longestEdge: Int?

            enum CodingKeys: String, CodingKey {
                case shortestEdge = "shortest_edge"
                case longestEdge = "longest_edge"
            }
        }

        public var imageMean: [CGFloat]
        public var imageStd: [CGFloat]
        public var patchSize: Int
        public var temporalPatchSize: Int
        public var mergeSize: Int
        public var size: Size?
        private var _minPixels: Int?
        private var _maxPixels: Int?

        public var minPixels: Int { _minPixels ?? size?.shortestEdge ?? 65_536 }
        public var maxPixels: Int { _maxPixels ?? size?.longestEdge ?? 16_777_216 }
        public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
            (imageMean[0], imageMean[1], imageMean[2])
        }
        public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
            (imageStd[0], imageStd[1], imageStd[2])
        }

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case mergeSize = "merge_size"
            case size
            case _minPixels = "min_pixels"
            case _maxPixels = "max_pixels"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.imageMean =
                try container.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [
                    0.5, 0.5, 0.5,
                ]
            self.imageStd =
                try container.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
            self.patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            self.temporalPatchSize =
                try container.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
            self.mergeSize = try container.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
            self.size = try container.decodeIfPresent(Size.self, forKey: .size)
            self._minPixels = try container.decodeIfPresent(Int.self, forKey: ._minPixels)
            self._maxPixels = try container.decodeIfPresent(Int.self, forKey: ._maxPixels)
        }
    }

    public var imageProcessor: ImageProcessor

    enum CodingKeys: String, CodingKey {
        case imageProcessor = "image_processor"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.imageProcessor =
            try container.decodeIfPresent(ImageProcessor.self, forKey: .imageProcessor)
            ?? ImageProcessor(from: decoder)
    }
}

extension LMInput {
    /// Text-only input as an LLM's `prepare` takes it. The processors batch the tokens, and the
    /// iterator adds a batch axis to whatever `prepare` hands back for it to step.
    var unbatchedText: LMInput {
        guard text.tokens.ndim == 2, text.tokens.dim(0) == 1 else { return self }
        return LMInput(text: .init(tokens: text.tokens[0], mask: text.mask?[0]))
    }
}

extension LFM3VL: PreparedInputSplitting {
    /// Each image encodes on its own and positions are plain offsets, so a warm cache can
    /// prefill the suffix with only the images in it.
    public func splitPreparedInput(_ input: LMInput, droppingFirst prefixTokenCount: Int)
        -> LMInput?
    {
        QwenVL.splitPreparedInput(
            input, droppingFirst: prefixTokenCount, imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId,
            mergeSize: config.visionConfiguration.spatialMergeSize)
    }
}

// MARK: - Chat conventions

extension LFM3VL {
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }
    public var reasoningConfig: ReasoningConfig? { .lfm3 }
}
