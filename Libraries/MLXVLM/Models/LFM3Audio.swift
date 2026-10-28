// Copyright © 2026 Apple Inc.

// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/lfm3_audio

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Features

/// The log-mel features the LFM3-Audio speech encoder reads: 128 Slaney mel bins of 16 kHz
/// mono audio every 10 ms, without centering, `(max(log10(mel), -12) + 4) / 4`.
public struct LFM3AudioFeatureExtractor: Sendable {
    public let sampleRate: Int
    public let fftLength: Int
    public let hopLength: Int
    public let melBins: Int
    public let logFloor: Float
    public let normalizationOffset: Float
    public let normalizationDivisor: Float
    /// Feature frames per encoder token.
    public let encoderStride: Int

    private let window: [Float]
    /// `[fftLength / 2 + 1, melBins]`, row-major.
    private let melFilters: [Float]

    public init(_ configuration: LFM3AudioProcessorConfiguration = .init()) {
        self.sampleRate = configuration.samplingRate
        self.fftLength = configuration.fftLength
        self.hopLength = configuration.hopLength
        self.melBins = configuration.featureSize
        self.logFloor = configuration.logFloor
        self.normalizationOffset = configuration.normalizationOffset
        self.normalizationDivisor = configuration.normalizationDivisor
        self.encoderStride = configuration.encoderStride
        // Periodic Hann window over the whole FFT.
        self.window = (0 ..< fftLength).map {
            Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(configuration.fftLength)))
        }
        self.melFilters = Self.slaneyMelFilters(
            bins: fftLength / 2 + 1, mels: melBins, sampleRate: sampleRate)
    }

    /// Feature frames of a clip: it is right-padded to `fftLength + k * hopLength` samples.
    public func frameCount(samples: Int) -> Int {
        1 + max(0, (samples - fftLength + hopLength - 1) / hopLength)
    }

    /// Encoder tokens (and `<|audio|>` placeholders) for `frames` feature frames.
    public func tokenCount(frames: Int) -> Int {
        (frames + encoderStride - 1) / encoderStride
    }

    /// `[melBins, frames]` features of one mono clip at ``sampleRate``.
    public func features(samples: MLXArray) -> MLXArray {
        let count = samples.dim(0)
        let frames = frameCount(samples: count)
        let padding = fftLength + (frames - 1) * hopLength - count
        let padded = concatenated([samples.asType(.float32), MLXArray.zeros([padding])])
        let windows = asStrided(padded, [frames, fftLength], strides: [hopLength, 1])
        let power = square(abs(MLXFFT.rfft(windows * MLXArray(window), axis: -1)))
        let mel = matmul(power, MLXArray(melFilters, [fftLength / 2 + 1, melBins]))
        let logMel = maximum(log10(mel), MLXArray(logFloor))
        return ((logMel + normalizationOffset) / normalizationDivisor).transposed()
    }

    /// Triangular filters on the Slaney mel scale with Slaney (area) normalization, from 0 Hz
    /// to Nyquist, as `transformers.audio_utils.mel_filter_bank` builds them.
    static func slaneyMelFilters(bins: Int, mels: Int, sampleRate: Int) -> [Float] {
        let linearStep = 200.0 / 3
        let logStep = log(6.4) / 27
        func mel(_ hertz: Double) -> Double {
            hertz < 1000 ? hertz / linearStep : 15 + log(hertz / 1000) / logStep
        }
        func hertz(_ mel: Double) -> Double {
            mel < 15 ? mel * linearStep : 1000 * exp(logStep * (mel - 15))
        }
        let nyquist = Double(sampleRate) / 2
        let maxMel = mel(nyquist)
        let edges = (0 ... mels + 1).map { hertz(maxMel * Double($0) / Double(mels + 1)) }
        var filters = [Float](repeating: 0, count: bins * mels)
        for bin in 0 ..< bins {
            let frequency = nyquist * Double(bin) / Double(bins - 1)
            for filter in 0 ..< mels {
                let rising = (frequency - edges[filter]) / (edges[filter + 1] - edges[filter])
                let falling =
                    (edges[filter + 2] - frequency) / (edges[filter + 2] - edges[filter + 1])
                let area = 2 / (edges[filter + 2] - edges[filter])
                filters[bin * mels + filter] = Float(max(0, min(rising, falling)) * area)
            }
        }
        return filters
    }
}

// MARK: - Speech encoder

private enum LFM3AudioEncoder {

    /// Torch evaluates activations of bfloat16 inputs in float32 and rounds once.
    static func float32(_ activation: (MLXArray) -> MLXArray, _ x: MLXArray) -> MLXArray {
        activation(x.asType(.float32)).asType(x.dtype)
    }

    /// Three stride-2 convolutions over the (mel, time) plane: 8 frames of 128 bins become one
    /// vector.
    final class Downsample: Module {
        @ModuleInfo(key: "conv_in") var convIn: Conv2d
        @ModuleInfo(key: "conv_mid") var convMid: Conv2d
        @ModuleInfo(key: "lin_mid") var linMid: Conv2d
        @ModuleInfo(key: "conv_end") var convEnd: Conv2d
        @ModuleInfo(key: "lin_end") var linEnd: Conv2d
        @ModuleInfo(key: "lin_out") var linOut: Linear

        init(_ config: LFM3AudioConfiguration.EncoderConfiguration) {
            let c = config.convChannels
            _convIn.wrappedValue = Conv2d(
                inputChannels: 1, outputChannels: c, kernelSize: 3, stride: 2, padding: 1)
            _convMid.wrappedValue = Conv2d(
                inputChannels: c, outputChannels: c, kernelSize: 3, stride: 2, padding: 1,
                groups: c)
            _linMid.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
            _convEnd.wrappedValue = Conv2d(
                inputChannels: c, outputChannels: c, kernelSize: 3, stride: 2, padding: 1,
                groups: c)
            _linEnd.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 1)
            _linOut.wrappedValue = Linear(config.melBins * c / 8, config.hiddenSize)
        }

        /// `[melBins, frames]` features of one clip -> `[1, ceil(frames / 8), hidden]`.
        func callAsFunction(_ features: MLXArray) -> MLXArray {
            // NHWC with the mel bins as height and time as width.
            var x = features[.newAxis, 0..., 0..., .newAxis].asType(convIn.weight.dtype)
            x = float32(silu, convIn(x))
            x = float32(silu, linMid(convMid(x)))
            x = float32(silu, linEnd(convEnd(x)))
            let (batch, bins, frames, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            // The reference flattens (channel, mel) channel-major.
            return linOut(x.transposed(0, 2, 3, 1).reshaped(batch, frames, channels * bins))
        }
    }

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let rope: RoPE

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear
        @ModuleInfo(key: "q_layernorm") var qLayerNorm: RMSNorm
        @ModuleInfo(key: "k_layernorm") var kLayerNorm: RMSNorm

        init(_ config: LFM3AudioConfiguration.EncoderConfiguration.Trunk) {
            self.heads = config.attentionHeads
            self.kvHeads = config.kvHeads
            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)
            self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
            let dim = config.hiddenSize
            _qProj.wrappedValue = Linear(dim, heads * headDim, bias: false)
            _kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _outProj.wrappedValue = Linear(heads * headDim, dim, bias: false)
            _qLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
            _kLayerNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            let q = qLayerNorm(qProj(x).reshaped(B, L, heads, headDim)).transposed(0, 2, 1, 3)
            let k = kLayerNorm(kProj(x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
            let v = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            let output = MLXFast.scaledDotProductAttention(
                queries: rope(q), keys: rope(k), values: v, scale: scale, mask: mask)
            return outProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    /// LFM2's gated short convolution over a whole clip: causal, zero history.
    final class ShortConv: Module {
        let kernelSize: Int

        @ModuleInfo(key: "conv") var conv: Conv1d
        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: LFM3AudioConfiguration.EncoderConfiguration.Trunk) {
            self.kernelSize = config.convLCache
            let dim = config.hiddenSize
            _conv.wrappedValue = Conv1d(
                inputChannels: dim, outputChannels: dim, kernelSize: kernelSize, groups: dim,
                bias: config.convBias)
            _inProj.wrappedValue = Linear(dim, 3 * dim, bias: config.convBias)
            _outProj.wrappedValue = Linear(dim, dim, bias: config.convBias)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let parts = inProj(x).split(parts: 3, axis: -1)
            let bx = padded(parts[0] * parts[2], widths: [0, .init((kernelSize - 1, 0)), 0])
            return outProj(parts[1] * conv(bx))
        }
    }

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

    final class Layer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention?
        @ModuleInfo(key: "conv") var conv: ShortConv?
        @ModuleInfo(key: "feed_forward") var feedForward: MLP
        @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm

        init(_ config: LFM3AudioConfiguration.EncoderConfiguration.Trunk, isAttention: Bool) {
            if isAttention {
                _attention.wrappedValue = Attention(config)
            } else {
                _conv.wrappedValue = ShortConv(config)
            }
            _feedForward.wrappedValue = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.feedForwardDimensions)
            _operatorNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
            _ffnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode
        ) -> MLXArray {
            let h = operatorNorm(x)
            let r = x + (attention.map { $0(h, mask: mask) } ?? conv!(h))
            return r + feedForward(ffnNorm(r))
        }
    }

    /// The LFM2 trunk: causal attention over a window of `sliding_window` encoder tokens.
    final class Trunk: Module {
        let layers: [Layer]
        @ModuleInfo(key: "embedding_norm") var embeddingNorm: RMSNorm

        init(_ config: LFM3AudioConfiguration.EncoderConfiguration.Trunk) {
            self.layers = config.layerTypes.map {
                Layer(config, isAttention: $0 == "full_attention")
            }
            _embeddingNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
        }

        func callAsFunction(_ x: MLXArray, window: Int) -> MLXArray {
            let mask = makeAttentionMask(n: x.dim(1), cache: nil, windowSize: window)
            var h = x
            for layer in layers {
                h = layer(h, mask: mask)
            }
            return embeddingNorm(h)
        }
    }

    final class Encoder: Module {
        let window: Int

        @ModuleInfo(key: "downsample") var downsample: Downsample
        @ModuleInfo(key: "lfm") var lfm: Trunk

        init(_ config: LFM3AudioConfiguration.EncoderConfiguration) {
            self.window = config.slidingWindow
            _downsample.wrappedValue = Downsample(config)
            _lfm.wrappedValue = Trunk(config.trunk)
        }

        /// `[melBins, frames]` -> `[ceil(frames / 8), hidden]`.
        func callAsFunction(_ features: MLXArray) -> MLXArray {
            lfm(downsample(features), window: window)[0]
        }
    }
}

private final class LFM3AudioProjector: Module, UnaryLayer {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputDimensions: Int, outputDimensions: Int) {
        _layerNorm.wrappedValue = LayerNorm(dimensions: inputDimensions, eps: 1e-5)
        _linear1.wrappedValue = Linear(inputDimensions, outputDimensions)
        _linear2.wrappedValue = Linear(outputDimensions, outputDimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(LFM3AudioEncoder.float32(gelu, linear1(layerNorm(x))))
    }
}

// MARK: - Depth decoder

/// RMSNorm that scales by the weight in float32 and rounds once, as the trained heads do.
final class LFM3AudioHeadNorm: Module, UnaryLayer {
    let eps: Float
    let weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self.weight = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps)
            .asType(x.dtype)
    }
}

/// The input embedding and tied output head of one codebook.
final class LFM3AudioCodebook: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "embedding_norm") var embeddingNorm: LFM3AudioHeadNorm

    init(codebookSize: Int, dimensions: Int, eps: Float) {
        _embedding.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: dimensions)
        _embeddingNorm.wrappedValue = LFM3AudioHeadNorm(dimensions: dimensions, eps: eps)
    }

    func logits(_ x: MLXArray) -> MLXArray {
        embedding.asLinear(embeddingNorm(x))
    }
}

/// One speech frame (a code per codebook) from a backbone hidden state: a bidirectional
/// LFM3 trunk over one slot per codebook, revealed by MaskGIT over a few passes.
final class LFM3AudioDepthDecoder: Module {
    let codebooks: Int
    let hiddenSize: Int

    @ModuleInfo(key: "depth_linear") var depthLinear: Linear
    @ParameterInfo(key: "mask_embedding") var maskEmbedding: MLXArray
    @ModuleInfo(key: "model") var model: LFM3ModelInner
    @ModuleInfo(key: "depth_embeddings") var depthEmbeddings: [LFM3AudioCodebook]

    init(_ config: LFM3Configuration, inputDimensions: Int, codebooks: Int, codebookSize: Int) {
        self.codebooks = codebooks
        self.hiddenSize = config.hiddenSize
        _depthLinear.wrappedValue = Linear(inputDimensions, codebooks * config.hiddenSize)
        _maskEmbedding.wrappedValue = MLXArray.zeros([config.hiddenSize])
        _model.wrappedValue = LFM3ModelInner(config, embedsTokens: false)
        _depthEmbeddings.wrappedValue = (0 ..< codebooks).map { _ in
            LFM3AudioCodebook(
                codebookSize: codebookSize, dimensions: config.hiddenSize, eps: config.normEps)
        }
    }

    /// The per-codebook slots of a frame, `[B, codebooks, hidden]`, from `[B, input]`.
    func slots(_ hidden: MLXArray) -> MLXArray {
        depthLinear(hidden).reshaped(hidden.dim(0), codebooks, hiddenSize)
    }

    /// Logits `[B, codebooks, codebookSize]` given the codes revealed so far; `masked` is true
    /// where a code is still hidden.
    func callAsFunction(_ slots: MLXArray, codes: MLXArray, masked: MLXArray) -> MLXArray {
        let embeddings = stacked(
            (0 ..< codebooks).map { depthEmbeddings[$0].embedding(codes[0..., $0]) }, axis: 1)
        var x =
            slots
            + MLX.where(
                masked[.ellipsis, .newAxis], maskEmbedding.asType(embeddings.dtype), embeddings)
        for layer in model.layers {
            x = layer(x, mask: .none, cache: nil)
        }
        x = model.embeddingNorm(x)
        return stacked(
            (0 ..< codebooks).map { depthEmbeddings[$0].logits(x[0..., $0, 0...]) }, axis: 1)
    }
}

// MARK: - Model

/// LFM3-Audio: the LFM3 language model, a speech encoder for input, and a depth decoder that
/// writes speech frames of 12 codebook codes for output.
///
/// Through the standard ``LanguageModel`` interface the model reads text and speech and writes
/// text: speech markers are suppressed. ``LFM3AudioGenerator`` writes speech and interleaved
/// text and speech, and ``LFM3AudioCodec`` turns the codes into 24 kHz audio.
public final class LFM3Audio: Module, VLMModel, KVCacheDimensionProvider, SuppressedTokensProviding
{
    @ModuleInfo(key: "language_model") public var languageModel: LFM3Model
    @ModuleInfo(key: "audio_tower") private var audioTower: LFM3AudioEncoder.Encoder
    @ModuleInfo(key: "audio_projector") private var audioProjector: LFM3AudioProjector
    @ModuleInfo(key: "codebook_embedding") var codebookEmbedding: Embedding
    @ModuleInfo(key: "depth_decoder") var depthDecoder: LFM3AudioDepthDecoder

    public let config: LFM3AudioConfiguration
    public var suppressedTokenIds: Set<Int>

    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }

    public init(_ config: LFM3AudioConfiguration) {
        self.config = config
        let hidden = config.textConfiguration.hiddenSize
        _languageModel.wrappedValue = LFM3Model(config.textConfiguration)
        _audioTower.wrappedValue = LFM3AudioEncoder.Encoder(config.encoderConfiguration)
        _audioProjector.wrappedValue = LFM3AudioProjector(
            inputDimensions: config.encoderConfiguration.hiddenSize, outputDimensions: hidden)
        _codebookEmbedding.wrappedValue = Embedding(
            embeddingCount: config.codebooks * config.audioVocabularySize, dimensions: hidden)
        _depthDecoder.wrappedValue = LFM3AudioDepthDecoder(
            config.depthConfiguration, inputDimensions: hidden, codebooks: config.codebooks,
            codebookSize: config.audioVocabularySize)
        // Text output never samples the speech placeholders or starts to speak.
        self.suppressedTokenIds = [
            config.audioTokenId, config.audioOutTokenId, config.audioStartTokenId,
            config.mixedStartTokenId,
        ]
    }

    /// Backbone embeddings of speech frames, `[frames, codebooks]` codes -> `[frames, hidden]`:
    /// the sum of each frame's codebook embeddings.
    func frameEmbeddings(_ codes: MLXArray) -> MLXArray {
        let offsets = MLXArray(
            (0 ..< config.codebooks).map { Int32($0 * config.audioVocabularySize) })
        return codebookEmbedding(codes.asType(.int32) + offsets).sum(axis: -2)
    }

    /// Projected encoder features of the clips in `audio`: one backbone embedding per token.
    func speechFeatures(_ audio: LMInput.ProcessedAudio) -> MLXArray {
        let features = audio.features.ndim == 2 ? audio.features[.newAxis] : audio.features
        let lengths =
            audio.mask.map { $0.asType(.int32).sum(axis: -1).asArray(Int.self) }
            ?? Array(repeating: features.dim(-1), count: features.dim(0))
        let clips = lengths.enumerated().map { index, length in
            audioProjector(audioTower(features[index, 0..., ..<length]))
        }
        return concatenated(clips, axis: 0)
    }

    /// Token embeddings with speech features scattered into the `<|audio|>` placeholders.
    func inputEmbeddings(_ tokens: MLXArray, audio: LMInput.ProcessedAudio?) throws -> MLXArray {
        let tokens = tokens.ndim == 1 ? tokens[.newAxis] : tokens
        guard tokens.dim(0) == 1 else {
            throw VLMError.processing("LFM3-Audio prepares one prompt at a time")
        }
        let embeddings = languageModel.model.embedTokens!(tokens)
        let ids = tokens[0].asArray(Int32.self)

        let positions = ids.indices.filter { ids[$0] == Int32(config.audioTokenId) }
        let features = audio.map(speechFeatures)
        let count = features?.dim(0) ?? 0
        guard positions.count == count else {
            throw VLMError.processing(
                "Audio placeholders (\(positions.count)) and speech features (\(count)) do not match"
            )
        }
        if let features, count > 0 {
            embeddings[0..., MLXArray(positions.map(Int32.init)), 0...] =
                features.asType(embeddings.dtype)[.newAxis]
        }
        return embeddings
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?, prefill: PrefillParameters
    ) throws -> PrepareResult {
        guard input.audio != nil else {
            return try languageModel.prepare(input, cache: cache, state: state, prefill: prefill)
        }
        let embeddings = try inputEmbeddings(input.text.tokens, audio: input.audio)
        return try withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            let hidden = try prefillHidden(embeddings, cache: cache, prefill: prefill)
            return .logits(LMOutput(logits: languageModel.logits(hidden)))
        }
    }

    /// Prefills `embeddings` in chunks and returns the last hidden state, `[1, 1, hidden]`.
    func prefillHidden(_ embeddings: MLXArray, cache: [any KVCache], prefill: PrefillParameters)
        throws -> MLXArray
    {
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
        return hidden[0..., (-1)...]
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        // The original checkpoint keeps the training export's names; mlx-vlm conversions
        // already match the modules.
        var checkpoint = try checkpoint.mapNames(
            using: .init([
                .replacePrefix("lfm.model", with: "language_model.model"),
                .replacePrefix("encoder.downsample", with: "audio_tower.downsample"),
                .replacePrefix("encoder.lfm.lfm", with: "audio_tower.lfm"),
                .replacePrefix("mlp.model.0", with: "audio_projector.layer_norm"),
                .replacePrefix("mlp.model.1", with: "audio_projector.linear_1"),
                .replacePrefix("mlp.model.3", with: "audio_projector.linear_2"),
                .replacePrefix("depthformer.model", with: "depth_decoder.model"),
                .replacePrefix("depth_linear", with: "depth_decoder.depth_linear"),
                .replacePrefix("mask_embedding", with: "depth_decoder.mask_embedding"),
                .replacePrefix("depth_embeddings", with: "depth_decoder.depth_embeddings"),
                // A stored copy of each tied codebook embedding.
                .excludeModule("to_logits"),
                .excludeModule("lm_head"),
            ]))
        var weights = LFM3Model.sanitizeConvWeights(checkpoint.weights)
        for (key, value) in weights where key.hasPrefix("audio_tower.downsample.") {
            if value.ndim == 4, value.dim(2) == value.dim(3) {
                // [out, in, kh, kw] -> [out, kh, kw, in]; the kernels are 3 x 3 or 1 x 1.
                weights[key] = value.transposed(0, 2, 3, 1)
            } else if value.ndim == 3 {
                // `lin_out` is a width-1 Conv1d in the original checkpoint.
                weights[key] = value[0..., 0..., 0]
            }
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

extension LFM3Audio {
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }
}

// MARK: - Processor

/// Speech clips become `<|audio|>` runs, one placeholder per encoder token.
public struct LFM3AudioProcessor: UserInputProcessor {
    private let extractor: LFM3AudioFeatureExtractor
    private let tokenizer: any Tokenizer

    public init(_ config: LFM3AudioProcessorConfiguration, tokenizer: any Tokenizer) {
        self.extractor = LFM3AudioFeatureExtractor(config)
        self.tokenizer = tokenizer
    }

    /// `[melBins, frames]` features of one clip; `.array` sources must be 16 kHz mono already.
    public func features(_ audio: UserInput.Audio) async throws -> MLXArray {
        let samples: MLXArray
        switch audio.source {
        case .array(let array):
            samples = array
        case .url(let url):
            var processing = UserInput.AudioProcessing()
            processing.sampleRate = Double(extractor.sampleRate)
            processing.channels = 1
            samples = try await UserInput.Audio.url(url).asMLXArray(processing: processing)
        }
        guard samples.ndim == 1 else {
            throw VLMError.processing("LFM3-Audio reads mono audio, got shape \(samples.shape)")
        }
        let features = extractor.features(samples: samples)
        // Evaluate before an async return can move to another thread.
        try checkedEval(features)
        return features
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = LFM3AudioMessageGenerator().generate(from: input)
        var tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        guard !input.audios.isEmpty else {
            return LMInput(tokens: MLXArray(tokens).expandedDimensions(axis: 0))
        }

        var clips = [MLXArray]()
        for audio in input.audios {
            clips.append(try await features(audio))
        }
        let audioTokenId = tokenizer.convertTokenToId("<|audio|>") ?? 125_031
        tokens = try Self.expandPlaceholders(
            tokens, tokenId: audioTokenId,
            counts: clips.map { extractor.tokenCount(frames: $0.dim(-1)) })

        // Zero-padded to the longest clip, with the valid frames of each in the mask.
        let frames = clips.map { $0.dim(-1) }
        let longest = frames.max() ?? 0
        let features = stacked(
            clips.map { padded($0, widths: [0, .init((0, longest - $0.dim(-1)))]) }, axis: 0)
        let mask = stacked(
            frames.map { MLXArray(0 ..< Int32(longest)) .< Int32($0) }, axis: 0
        ).asType(.int32)

        let promptArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray).asType(.int8)),
            audio: .init(features: features, mask: mask))
    }

    /// Each occurrence of `tokenId`, in order, repeated `counts[i]` times.
    static func expandPlaceholders(_ tokens: [Int], tokenId: Int, counts: [Int]) throws -> [Int] {
        let found = tokens.filter { $0 == tokenId }.count
        guard found == counts.count else {
            throw VLMError.processing(
                "Found \(found) audio placeholders in the prompt but received \(counts.count) clips"
            )
        }
        var result = [Int]()
        result.reserveCapacity(tokens.count + counts.reduce(0, +))
        var next = counts.makeIterator()
        for token in tokens {
            if token == tokenId, let count = next.next() {
                result.append(contentsOf: repeatElement(tokenId, count: count))
            } else {
                result.append(token)
            }
        }
        return result
    }
}

/// Speech clips before the text of each message, as `{"type": "audio"}` parts.
public struct LFM3AudioMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        var dictionary: MLXLMCommon.Message = ["role": message.role.rawValue]
        if message.audios.isEmpty {
            dictionary["content"] = message.content
        } else {
            var parts: [[String: String]] = message.audios.map { _ in ["type": "audio"] }
            if !message.content.isEmpty {
                parts.append(["type": "text", "text": message.content])
            }
            dictionary["content"] = parts
        }
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

// MARK: - Configuration

public struct LFM3AudioConfiguration: Codable, Sendable {

    /// The speech encoder: strided convolutions, then an LFM2 trunk.
    public struct EncoderConfiguration: Codable, Sendable {

        public struct Trunk: Codable, Sendable {
            public var hiddenSize: Int
            public var intermediateSize: Int
            public var hiddenLayers: Int
            public var attentionHeads: Int
            public var kvHeads: Int
            public var normEps: Float
            public var convBias: Bool
            public var convLCache: Int
            public var layerTypes: [String]
            public var ropeParameters: [String: StringOrNumber]?
            public var blockMultipleOf: Int
            public var blockFFNDimMultiplier: Float?
            public var blockAutoAdjustFFDim: Bool

            public var headDim: Int { hiddenSize / attentionHeads }
            public var ropeTheta: Float { ropeParameters?["rope_theta"]?.asFloat() ?? 1_000_000 }

            /// LFM2's MLP width: two thirds of `intermediate_size`, scaled, rounded up.
            public var feedForwardDimensions: Int {
                guard blockAutoAdjustFFDim else { return intermediateSize }
                var width = Int(Float(2 * intermediateSize) / 3)
                if let blockFFNDimMultiplier {
                    width = Int(blockFFNDimMultiplier * Float(width))
                }
                return blockMultipleOf * ((width + blockMultipleOf - 1) / blockMultipleOf)
            }

            enum CodingKeys: String, CodingKey {
                case hiddenSize = "hidden_size"
                case intermediateSize = "intermediate_size"
                case hiddenLayers = "num_hidden_layers"
                case attentionHeads = "num_attention_heads"
                case kvHeads = "num_key_value_heads"
                case normEps = "norm_eps"
                case convBias = "conv_bias"
                case convLCache = "conv_L_cache"
                case layerTypes = "layer_types"
                case ropeParameters = "rope_parameters"
                case blockMultipleOf = "block_multiple_of"
                case blockFFNDimMultiplier = "block_ffn_dim_multiplier"
                case blockAutoAdjustFFDim = "block_auto_adjust_ff_dim"
            }

            public init(from decoder: Decoder) throws {
                let c = try decoder.container(keyedBy: CodingKeys.self)
                self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
                self.intermediateSize =
                    try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2560
                self.hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 17
                self.attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
                self.kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
                self.normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
                self.convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
                self.convLCache = try c.decodeIfPresent(Int.self, forKey: .convLCache) ?? 3
                self.layerTypes = try c.decode([String].self, forKey: .layerTypes)
                self.ropeParameters = try? c.decodeIfPresent(
                    [String: StringOrNumber].self, forKey: .ropeParameters)
                self.blockMultipleOf =
                    try c.decodeIfPresent(Int.self, forKey: .blockMultipleOf) ?? 256
                self.blockFFNDimMultiplier = try c.decodeIfPresent(
                    Float.self, forKey: .blockFFNDimMultiplier)
                self.blockAutoAdjustFFDim =
                    try c.decodeIfPresent(Bool.self, forKey: .blockAutoAdjustFFDim) ?? true
            }
        }

        public var melBins: Int
        public var convChannels: Int
        public var hiddenSize: Int
        public var slidingWindow: Int
        public var trunk: Trunk

        enum CodingKeys: String, CodingKey {
            case melBins = "dim_in"
            case convChannels = "dim_conv"
            case hiddenSize = "dim_out"
            case slidingWindow = "sliding_window"
            case trunk = "lfm"
        }
    }

    public var modelType: String
    public var textConfiguration: LFM3Configuration
    public var encoderConfiguration: EncoderConfiguration
    public var depthConfiguration: LFM3Configuration
    public var codebooks: Int
    public var audioVocabularySize: Int
    public var maskgitSteps: Int
    /// Text tokens, then speech frames, per block of an interleaved answer.
    public var interleavedTextTokens: Int
    public var interleavedAudioFrames: Int
    public var audioTokenId: Int
    public var audioOutTokenId: Int
    public var audioStartTokenId: Int
    public var textEndTokenId: Int
    public var mixedStartTokenId: Int

    /// The end-of-speech code, one past the codec's codes.
    public var endOfSpeechCode: Int { audioVocabularySize - 1 }

    /// Codebooks MaskGIT reveals in each pass.
    public var maskgitSchedule: [Int] {
        var schedule = Array(repeating: codebooks / maskgitSteps, count: maskgitSteps)
        schedule[maskgitSteps - 1] += codebooks % maskgitSteps
        return schedule
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "lfm"
        case encoderConfiguration = "encoder"
        case depthConfiguration = "depthformer"
        case codebooks
        case audioVocabularySize = "audio_vocab_size"
        case maskgitSteps = "maskgit_steps"
        case interleavedTextTokens = "interleaved_n_text"
        case interleavedAudioFrames = "interleaved_n_audio"
        case audioTokenId = "audio_token_id"
        case audioOutTokenId = "audio_out_token_id"
        case audioStartTokenId = "audio_start_token_id"
        case textEndTokenId = "text_end_token_id"
        case mixedStartTokenId = "mixed_start_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "lfm3_audio"
        self.textConfiguration = try c.decode(LFM3Configuration.self, forKey: .textConfiguration)
        self.encoderConfiguration = try c.decode(
            EncoderConfiguration.self, forKey: .encoderConfiguration)
        self.depthConfiguration = try c.decode(LFM3Configuration.self, forKey: .depthConfiguration)
        self.codebooks = try c.decodeIfPresent(Int.self, forKey: .codebooks) ?? 12
        self.audioVocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .audioVocabularySize) ?? 2017
        self.maskgitSteps = try c.decodeIfPresent(Int.self, forKey: .maskgitSteps) ?? 3
        self.interleavedTextTokens =
            try c.decodeIfPresent(Int.self, forKey: .interleavedTextTokens) ?? 6
        self.interleavedAudioFrames =
            try c.decodeIfPresent(Int.self, forKey: .interleavedAudioFrames) ?? 12
        // The checkpoint's config does not carry the token ids; these are the tokenizer's.
        self.audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 125_031
        self.audioOutTokenId = try c.decodeIfPresent(Int.self, forKey: .audioOutTokenId) ?? 125_032
        self.audioStartTokenId =
            try c.decodeIfPresent(Int.self, forKey: .audioStartTokenId) ?? 125_012
        self.textEndTokenId = try c.decodeIfPresent(Int.self, forKey: .textEndTokenId) ?? 125_014
        self.mixedStartTokenId =
            try c.decodeIfPresent(Int.self, forKey: .mixedStartTokenId) ?? 125_015
    }
}

extension LFM3AudioConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        try textConfiguration.validateModelConfiguration()
        try depthConfiguration.validateModelConfiguration()
        let trunk = encoderConfiguration.trunk
        guard trunk.layerTypes.count == trunk.hiddenLayers,
            trunk.layerTypes.allSatisfy({ $0 == "conv" || $0 == "full_attention" })
        else {
            throw ModelFactoryError.invalidConfiguration(
                "LFM3-Audio: encoder layer_types needs num_hidden_layers entries of conv or full_attention"
            )
        }
        guard depthConfiguration.layerTypes.allSatisfy({ $0 == .fullAttention }) else {
            throw ModelFactoryError.invalidConfiguration(
                "LFM3-Audio: the depth decoder has attention layers only")
        }
        guard maskgitSteps > 0, maskgitSteps <= codebooks, encoderConfiguration.melBins % 8 == 0
        else {
            throw ModelFactoryError.invalidConfiguration(
                "LFM3-Audio: maskgit_steps must be in 1...codebooks and dim_in a multiple of 8")
        }
    }
}

/// The log-mel settings of `preprocessor_config.json` (or of its `feature_extractor`).
public struct LFM3AudioProcessorConfiguration: Codable, Sendable {
    public var samplingRate: Int = 16_000
    public var featureSize: Int = 128
    public var fftLength: Int = 400
    public var hopLength: Int = 160
    public var logFloor: Float = -12
    public var normalizationOffset: Float = 4
    public var normalizationDivisor: Float = 4
    public var encoderStride: Int = 8

    enum CodingKeys: String, CodingKey {
        case samplingRate = "sampling_rate"
        case featureSize = "feature_size"
        case fftLength = "n_fft"
        case hopLength = "hop_length"
        case winLength = "win_length"
        case logFloor = "log_floor"
        case normalizationOffset = "normalization_offset"
        case normalizationDivisor = "normalization_divisor"
        case encoderStride = "encoder_stride"
        case featureExtractor = "feature_extractor"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let outer = try decoder.container(keyedBy: CodingKeys.self)
        let c =
            outer.contains(.featureExtractor)
            ? try outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .featureExtractor)
            : outer
        self.samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 16_000
        self.featureSize = try c.decodeIfPresent(Int.self, forKey: .featureSize) ?? 128
        self.fftLength = try c.decodeIfPresent(Int.self, forKey: .fftLength) ?? 400
        self.hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? 160
        self.logFloor = try c.decodeIfPresent(Float.self, forKey: .logFloor) ?? -12
        self.normalizationOffset =
            try c.decodeIfPresent(Float.self, forKey: .normalizationOffset) ?? 4
        self.normalizationDivisor =
            try c.decodeIfPresent(Float.self, forKey: .normalizationDivisor) ?? 4
        self.encoderStride = try c.decodeIfPresent(Int.self, forKey: .encoderStride) ?? 8
        let window = try c.decodeIfPresent(Int.self, forKey: .winLength) ?? fftLength
        guard window == fftLength, hopLength > 0, fftLength > 0, featureSize > 0 else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "LFM3-Audio features need win_length == n_fft"))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(samplingRate, forKey: .samplingRate)
        try c.encode(featureSize, forKey: .featureSize)
        try c.encode(fftLength, forKey: .fftLength)
        try c.encode(hopLength, forKey: .hopLength)
        try c.encode(logFloor, forKey: .logFloor)
        try c.encode(normalizationOffset, forKey: .normalizationOffset)
        try c.encode(normalizationDivisor, forKey: .normalizationDivisor)
        try c.encode(encoderStride, forKey: .encoderStride)
    }
}
