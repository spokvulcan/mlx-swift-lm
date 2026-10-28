// Copyright © 2026 Apple Inc.

// Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/lfm3_audio/codec.py

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// The decoder half of LFM3-Audio's Liquid codec: speech frames of 12 FSQ codes, 12.5 per
/// second, to 24 kHz audio.
///
/// The codec ships with the model in its `audio_tokenizer/` folder (`audio_detokenizer/` in
/// earlier checkpoints). Load it with ``load(modelDirectory:)``, then ``decode(_:)`` whole
/// answers or feed a ``Stream`` the frames as they are generated.
public final class LFM3AudioCodec: Module {

    public struct Configuration: Codable, Sendable {
        public var encoderDim: Int
        public var quantizerDim: Int
        public var fsqLevels: [Int]
        public var codebooks: Int
        public var fftLength: Int
        public var hopLength: Int
        public var windowLength: Int
        public var frameStack: Int
        public var decoderWindow: Int
        public var sampleRate: Int

        /// Codes of one codebook: the product of the FSQ levels.
        public var codebookSize: Int { fsqLevels.reduce(1, *) }
        /// Audio samples per speech frame.
        public var frameSamples: Int { frameStack * hopLength }

        enum CodingKeys: String, CodingKey {
            case encoderDim = "encoder_dim"
            case quantizerDim = "quantizer_dim"
            case fsqLevels = "fsq_levels"
            case codebooks = "n_q"
            case fftLength = "stft_n_fft"
            case hopLength = "stft_hop_length"
            case windowLength = "stft_win_length"
            case frameStack = "stft_frame_stack"
            case decoderWindow = "decoder_window"
            case sampleRate = "sampling_rate"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.encoderDim = try c.decodeIfPresent(Int.self, forKey: .encoderDim) ?? 512
            self.quantizerDim = try c.decodeIfPresent(Int.self, forKey: .quantizerDim) ?? 256
            self.fsqLevels = try c.decodeIfPresent([Int].self, forKey: .fsqLevels) ?? [8, 7, 6, 6]
            self.codebooks = try c.decodeIfPresent(Int.self, forKey: .codebooks) ?? 12
            self.fftLength = try c.decodeIfPresent(Int.self, forKey: .fftLength) ?? 1280
            self.hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? 320
            self.windowLength = try c.decodeIfPresent(Int.self, forKey: .windowLength) ?? 1280
            self.frameStack = try c.decodeIfPresent(Int.self, forKey: .frameStack) ?? 6
            self.decoderWindow = try c.decodeIfPresent(Int.self, forKey: .decoderWindow) ?? 30
            self.sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24_000
            guard windowLength == fftLength, fftLength % hopLength == 0,
                (fftLength - hopLength) % 2 == 0
            else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: decoder.codingPath,
                        debugDescription:
                            "The Liquid codec needs stft_win_length == stft_n_fft, a multiple of the hop"
                    ))
            }
        }

        /// The decoder trunk. Its shape is fixed by the reference config class, not the JSON.
        var decoderTrunk: LFM3Configuration {
            LFM3Configuration(
                vocabularySize: 1, hiddenSize: encoderDim, intermediateSize: 2304,
                hiddenLayers: 8, attentionHeads: 8, kvHeads: 2, headDim: 64,
                layerTypes: (0 ..< 8).map { [2, 4, 6].contains($0) ? .slidingAttention : .conv },
                slidingWindow: decoderWindow, tieWordEmbeddings: false,
                ropeParameters: [
                    LFM3LayerType.slidingAttention.rawValue: ["rope_theta": .float(1_000_000)]
                ])
        }
    }

    public enum CodecError: LocalizedError {
        case missingCodec(URL)
        case invalidCodes(String)
        case finishedStream

        public var errorDescription: String? {
            switch self {
            case .missingCodec(let directory):
                "No Liquid codec (audio_tokenizer/) in \(directory.path)"
            case .invalidCodes(let message): message
            case .finishedStream: "This codec stream is finished; start a new one"
            }
        }
    }

    /// FSQ dequantization: each code splits into one digit per level.
    final class Quantizer: Module {
        let levels: [Int]
        let basis: [Int]

        @ModuleInfo(key: "project_out") var projectOut: Linear

        init(_ config: Configuration) {
            self.levels = config.fsqLevels
            self.basis = config.fsqLevels.indices.map { config.fsqLevels[..<$0].reduce(1, *) }
            _projectOut.wrappedValue = Linear(
                config.codebooks * config.fsqLevels.count, config.quantizerDim)
        }

        /// `[codebooks, frames]` codes -> `[1, frames, quantizerDim]`.
        func callAsFunction(_ codes: MLXArray) -> MLXArray {
            let digits = remainder(
                floorDivide(
                    codes.transposed()[.ellipsis, .newAxis], MLXArray(basis.map(Int32.init))),
                MLXArray(levels.map(Int32.init)))
            let half = MLXArray(levels.map { Float($0 / 2) })
            let latents = ((digits.asType(.float32) - half) / half).reshaped(1, codes.dim(1), -1)
            return projectOut(latents.asType(projectOut.weight.dtype))
        }
    }

    final class ISTFT: Module {
        /// The synthesis window, kept in float32 like the reference.
        let window: MLXArray

        init(_ config: Configuration) {
            self.window = MLXArray.zeros([config.windowLength], dtype: .float32)
        }
    }

    final class AudioDecoder: Module {
        let frameStack: Int

        @ModuleInfo(key: "lfm") var lfm: LFM3ModelInner
        @ModuleInfo(key: "lin") var lin: Linear
        @ModuleInfo(key: "istft") var istft: ISTFT

        init(_ config: Configuration) {
            self.frameStack = config.frameStack
            _lfm.wrappedValue = LFM3ModelInner(config.decoderTrunk, embedsTokens: false)
            _lin.wrappedValue = Linear(config.encoderDim, config.fftLength + 2)
            _istft.wrappedValue = ISTFT(config)
        }

        /// STFT frames of `embeddings` `[1, frames, encoderDim]`: `[stack * frames, bins]`
        /// complex spectra.
        func spectra(_ embeddings: MLXArray, cache: [KVCache]?) -> MLXArray {
            let hidden = lfm(
                nil, cache: cache, inputEmbeddings: repeated(embeddings, count: frameStack, axis: 1)
            )
            let parts = lin(hidden)[0].asType(.float32).split(parts: 2, axis: -1)
            let magnitude = exp(parts[0])
            return (magnitude * cos(parts[1])).asType(.complex64)
                + (magnitude * sin(parts[1])).asImaginary()
        }
    }

    public let config: Configuration

    @ModuleInfo(key: "quantizer") var quantizer: Quantizer
    @ModuleInfo(key: "from_quantizer") var fromQuantizer: Linear
    @ModuleInfo(key: "decoder") var decoder: AudioDecoder

    public init(_ config: Configuration) {
        self.config = config
        _quantizer.wrappedValue = Quantizer(config)
        _fromQuantizer.wrappedValue = Linear(config.quantizerDim, config.encoderDim)
        _decoder.wrappedValue = AudioDecoder(config)
    }

    /// Loads the codec from the `audio_tokenizer/` (or `audio_detokenizer/`) folder of an
    /// LFM3-Audio checkpoint. Weights stay in their stored dtype; the window stays float32.
    public static func load(modelDirectory: URL) throws -> LFM3AudioCodec {
        guard
            let folder = ["audio_tokenizer", "audio_detokenizer"]
                .map({ modelDirectory.appendingPathComponent($0) })
                .first(where: {
                    FileManager.default.fileExists(
                        atPath: $0.appendingPathComponent("config.json").path)
                })
        else {
            throw CodecError.missingCodec(modelDirectory)
        }
        let config = try JSONDecoder().decode(
            Configuration.self,
            from: Data(contentsOf: folder.appendingPathComponent("config.json")))
        var weights = [String: MLXArray]()
        let files = try FileManager.default.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: nil
        )
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        for file in files {
            weights.merge(try loadArrays(url: file)) { _, new in new }
        }
        let codec = LFM3AudioCodec(config)
        try codec.update(
            parameters: ModuleParameters.unflattened(sanitize(weights)), verify: [.all])
        eval(codec)
        return codec
    }

    /// Keeps the decoder half and moves the PyTorch conv layout to MLX's.
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in weights {
            let key = key.replacingOccurrences(of: "quantizer.fsq.", with: "quantizer.")
            guard
                key.hasPrefix("decoder.") || key.hasPrefix("from_quantizer.")
                    || key.hasPrefix("quantizer.project_out.")
            else { continue }
            result[key] = value
        }
        return LFM3Model.sanitizeConvWeights(result)
    }

    private func validate(_ codes: MLXArray) throws {
        guard codes.ndim == 2, codes.dim(0) == config.codebooks else {
            throw CodecError.invalidCodes(
                "Expected codes [\(config.codebooks), frames], got \(codes.shape)")
        }
        // The language model's end-of-speech code would map to a valid latent.
        if codes.size > 0,
            codes.min().item(Int.self) < 0 || codes.max().item(Int.self) >= config.codebookSize
        {
            throw CodecError.invalidCodes(
                "Codes must be in 0 ..< \(config.codebookSize); drop end-of-speech frames")
        }
    }

    /// Overlap-adds `[frames, fftLength]` STFT frames: `[(frames - 1) * hop + fftLength]`.
    private func overlapAdd(_ frames: MLXArray) -> MLXArray {
        let (count, hop) = (frames.dim(0), config.hopLength)
        let k = config.fftLength / hop
        let blocks = frames.reshaped(count, k, hop)
        var output = MLXArray.zeros([count + k - 1, hop], dtype: frames.dtype)
        for j in 0 ..< k {
            output = output + padded(blocks[0..., j], widths: [.init((j, k - 1 - j)), 0])
        }
        return output.reshaped(-1)
    }

    /// The windowed inverse FFT of `[1, frames, encoderDim]` embeddings, before the
    /// normalization: overlap-added audio and window envelope.
    private func synthesize(_ embeddings: MLXArray, cache: [KVCache]?) -> (MLXArray, MLXArray) {
        let spectra = decoder.spectra(embeddings, cache: cache)
        let window = decoder.istft.window
        let frames = MLXFFT.irfft(spectra, n: config.fftLength, axis: -1) * window
        let envelope = overlapAdd(broadcast(square(window), to: frames.shape))
        return (overlapAdd(frames), envelope)
    }

    private func embeddings(_ codes: MLXArray) -> MLXArray {
        fromQuantizer(quantizer(codes.asType(.int32)))
    }

    /// 24 kHz float32 audio, `frames * 1920` samples, of `[codebooks, frames]` codes.
    public func decode(_ codes: MLXArray) throws -> MLXArray {
        try validate(codes)
        guard codes.dim(1) > 0 else { return MLXArray.zeros([0], dtype: .float32) }
        let (audio, envelope) = synthesize(embeddings(codes), cache: nil)
        // "Same" padding: trim half the overlap from each end.
        let trim = (config.fftLength - config.hopLength) / 2
        let end = audio.dim(0) - trim
        return audio[trim ..< end] / envelope[trim ..< end]
    }

    /// A stream of frames decoded as they arrive, to the same audio as ``decode(_:)`` of all
    /// of them. Each call returns the samples no later frame changes.
    public final class Stream {
        private let codec: LFM3AudioCodec
        private let cache: [KVCache]
        private var tail: (audio: MLXArray, envelope: MLXArray)?
        private var started = false
        public private(set) var isFinished = false

        init(codec: LFM3AudioCodec) throws {
            self.codec = codec
            self.cache = try codec.decoder.lfm.newCache(parameters: nil)
        }

        /// Decodes the next `[codebooks, frames]` codes. With `isFinal` the stream ends and the
        /// held-back samples are returned too.
        public func decode(_ codes: MLXArray, isFinal: Bool = false) throws -> MLXArray {
            guard !isFinished else { throw CodecError.finishedStream }
            try codec.validate(codes)
            let config = codec.config
            let trim = (config.fftLength - config.hopLength) / 2
            let held = 2 * trim
            if codes.dim(1) == 0, !(isFinal && started) {
                if isFinal { isFinished = true }
                return MLXArray.zeros([0], dtype: .float32)
            }

            var (audio, envelope) =
                codes.dim(1) > 0
                ? codec.synthesize(codec.embeddings(codes), cache: cache)
                : (MLXArray.zeros([held], dtype: .float32), MLXArray.zeros([held], dtype: .float32))
            if let tail {
                audio = concatenated([audio[..<held] + tail.audio, audio[held...]])
                envelope = concatenated([envelope[..<held] + tail.envelope, envelope[held...]])
            }
            if !started {
                audio = audio[trim...]
                envelope = envelope[trim...]
                started = true
            }
            if isFinal {
                let end = audio.dim(0) - trim
                audio = audio[..<end]
                envelope = envelope[..<end]
                tail = nil
                isFinished = true
            } else {
                let end = audio.dim(0) - held
                tail = (audio[end...], envelope[end...])
                audio = audio[..<end]
                envelope = envelope[..<end]
            }
            let samples = audio / envelope
            eval(samples, cache)
            return samples
        }
    }

    /// A new incremental decoder.
    public func makeStream() throws -> Stream {
        try Stream(codec: self)
    }
}
