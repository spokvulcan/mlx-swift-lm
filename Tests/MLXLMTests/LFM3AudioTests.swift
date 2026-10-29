// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

// Offline checks of the LFM3-Audio port on a tiny random model, plus an opt-in parity check of
// a local checkpoint against a fixture from mlx-vlm's `lfm3_audio`.

private func tinyLFM3AudioJSON() -> String {
    """
    {
      "model_type": "lfm3_audio", "codebooks": 4, "audio_vocab_size": 9, "maskgit_steps": 3,
      "interleaved_n_text": 2, "interleaved_n_audio": 3,
      "audio_token_id": 60, "audio_out_token_id": 61, "audio_start_token_id": 62,
      "text_end_token_id": 63, "mixed_start_token_id": 59,
      "lfm": {
        "model_type": "lfm3", "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
        "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 8, "sliding_window": 4,
        "layer_types": ["conv", "sliding_attention", "full_attention"]
      },
      "encoder": {
        "dim_in": 16, "dim_conv": 8, "dim_out": 24, "sliding_window": 3,
        "lfm": {
          "model_type": "lfm2", "hidden_size": 24, "intermediate_size": 48,
          "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
          "conv_L_cache": 3, "block_multiple_of": 8, "block_auto_adjust_ff_dim": true,
          "layer_types": ["conv", "full_attention"],
          "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"}
        }
      },
      "depthformer": {
        "model_type": "lfm3", "vocab_size": 1, "hidden_size": 16, "intermediate_size": 24,
        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 8, "use_sandwich_norm": false, "layer_types": ["full_attention"],
        "rope_parameters": {"full_attention": {"rope_theta": 1000000.0}}
      }
    }
    """
}

private func tinyModel() throws -> LFM3Audio {
    let config = try JSONDecoder().decode(
        LFM3AudioConfiguration.self, from: Data(tinyLFM3AudioJSON().utf8))
    try config.validateModelConfiguration()
    let model = LFM3Audio(config)
    eval(model)
    return model
}

/// Leaves only `token` at the first sampled step.
private struct OnlyFirstToken: LogitProcessor {
    let token: Int
    var sampled = false

    mutating func prompt(_ prompt: MLXArray) {}

    func process(logits: MLXArray) -> MLXArray {
        guard !sampled else { return logits }
        let only = MLXArray(0 ..< Int32(logits.dim(-1))) .== Int32(token)
        return MLX.where(only, logits, MLXArray(-Float.infinity))
    }

    mutating func didSample(token: MLXArray) {
        sampled = true
    }
}

/// Features of a tiny clip for the tiny model: 16 mel bins.
private func tinyFeatures(frames: Int) -> MLXArray {
    MLXRandom.normal([16, frames], key: MLXRandom.key(3))
}

struct LFM3AudioTests {

    // MARK: - Configuration

    @Test("The configuration derives the encoder width and the MaskGIT schedule")
    func configuration() throws {
        let config = try JSONDecoder().decode(
            LFM3AudioConfiguration.self, from: Data(tinyLFM3AudioJSON().utf8))
        // int(2 * 48 / 3) = 32, already a multiple of 8.
        #expect(config.encoderConfiguration.trunk.feedForwardDimensions == 32)
        #expect(config.maskgitSchedule == [1, 1, 2])
        #expect(config.endOfSpeechCode == 8)
        #expect(!config.depthConfiguration.useSandwichNorm)

        // LFM3-Audio-2B: intermediate_size 2560 -> 1792.
        let released = try JSONDecoder().decode(
            LFM3AudioConfiguration.EncoderConfiguration.Trunk.self,
            from: Data(
                #"{"intermediate_size": 2560, "layer_types": [], "block_ffn_dim_multiplier": 1.0}"#
                    .utf8))
        #expect(released.feedForwardDimensions == 1792)
    }

    @Test("The processor configuration reads flat or nested feature extractor settings")
    func processorConfiguration() throws {
        let flat = try JSONDecoder().decode(
            LFM3AudioProcessorConfiguration.self,
            from: Data(#"{"feature_size": 64, "hop_length": 80, "n_fft": 200}"#.utf8))
        #expect(flat.featureSize == 64 && flat.hopLength == 80 && flat.fftLength == 200)
        let nested = try JSONDecoder().decode(
            LFM3AudioProcessorConfiguration.self,
            from: Data(#"{"feature_extractor": {"sampling_rate": 8000}}"#.utf8))
        #expect(nested.samplingRate == 8000 && nested.featureSize == 128)
        #expect(throws: DecodingError.self) {
            try JSONDecoder().decode(
                LFM3AudioProcessorConfiguration.self,
                from: Data(#"{"n_fft": 400, "win_length": 320}"#.utf8))
        }
    }

    // MARK: - Features

    @Test("Frame and token counts follow the reference padding")
    func frameCounts() {
        let extractor = LFM3AudioFeatureExtractor()
        // From the reference `num_frames`.
        let samples = [0, 1, 399, 400, 401, 560, 561, 16000, 74683]
        #expect(samples.map(extractor.frameCount) == [1, 1, 1, 1, 2, 2, 3, 99, 466])
        #expect(extractor.tokenCount(frames: 99) == 13)
        #expect(extractor.tokenCount(frames: 8) == 1)
    }

    @Test("The Slaney filters match the reference filter bank")
    func melFilters() {
        let filters = LFM3AudioFeatureExtractor.slaneyMelFilters(
            bins: 201, mels: 128, sampleRate: 16000)
        func value(_ mel: Int, _ bin: Int) -> Float { filters[bin * 128 + mel] }
        // From mlx_audio's `mel_filters(..., norm="slaney", mel_scale="slaney")` in float64.
        for (mel, bin, expected) in [
            (0, 1, 0.012373986653983593), (10, 6, 0.011289183981716633),
            (30, 18, 0.03386755287647247), (64, 43, 0.018091516569256783),
            (90, 80, 0.012934654951095581), (127, 195, 0.005041603464633226),
        ] {
            #expect(abs(value(mel, bin) - Float(expected)) < 1e-7)
        }
        #expect(value(127, 200) == 0)
        #expect(abs(filters.reduce(0, +) - 3.190985679626465) < 1e-4)
    }

    @Test("Silence sits at the log floor")
    func silenceFeatures() {
        let features = LFM3AudioFeatureExtractor().features(samples: MLXArray.zeros([1000]))
        #expect(features.shape == [128, 5])
        #expect(features.max().item(Float.self) == (-12 + 4) / 4)
    }

    // MARK: - Processor

    @Test("Each audio placeholder expands to its clip's tokens")
    func placeholderExpansion() throws {
        let expanded = try LFM3AudioProcessor.expandPlaceholders(
            [1, 60, 2, 60, 3], tokenId: 60, counts: [2, 3])
        #expect(expanded == [1, 60, 60, 2, 60, 60, 60, 3])
        #expect(throws: VLMError.self) {
            try LFM3AudioProcessor.expandPlaceholders([1, 60], tokenId: 60, counts: [1, 1])
        }
    }

    @Test("Clips come before the text as audio parts")
    func messageGenerator() {
        let generator = LFM3AudioMessageGenerator()
        let speech = generator.generate(
            message: .user("Transcribe.", audios: [.array(MLXArray.zeros([10]))]))
        let parts = speech["content"] as? [[String: String]]
        #expect(parts?.map { $0["type"] } == ["audio", "text"])
        #expect(
            generator.generate(message: .system("Perform ASR."))["content"] as? String
                == "Perform ASR.")
    }

    // MARK: - Model

    @Test("A speech prompt prefills to next-token logits")
    func speechPrefill() throws {
        let model = try tinyModel()
        // 17 frames -> 3 encoder tokens.
        let features = tinyFeatures(frames: 17)
        let ids = MLXArray([Int32(1), 60, 60, 60, 2])[.newAxis]
        let input = LMInput(
            text: .init(tokens: ids, mask: ones(like: ids).asType(.int8)),
            audio: .init(features: features[.newAxis], mask: ones([1, 17]).asType(.int32)))
        #expect(model.speechFeatures(input.audio!).shape == [3, 32])

        let cache = try model.newCache(parameters: nil)
        guard
            case .logits(let output) = try model.prepare(
                input, cache: cache, state: nil, prefill: .init(stepSize: 2))
        else {
            Issue.record("Expected logits from a speech prefill")
            return
        }
        let embeddings = try model.inputEmbeddings(ids, audio: input.audio)
        let expected = model.languageModel(nil, cache: nil, inputEmbeddings: embeddings)
        expectClose(output.logits[0, -1], expected[0, -1], atol: 1e-4, reducedAtol: 2e-2)
        #expect(model.suppressedTokenIds == [60, 61, 62, 59])
    }

    @Test("A text prompt batched as the processor batches it decodes as the bare tokens do")
    func textOnlyDecode() throws {
        let model = try tinyModel()
        let ids = MLXArray([Int32(1), 2, 3, 4, 5])
        var parameters = GenerateParameters(maxTokens: 3, temperature: 0)
        parameters.prefill = .init(stepSize: 2)
        var batched = try TokenIterator(
            input: LMInput(tokens: ids[.newAxis]), model: model, parameters: parameters)
        var bare = try TokenIterator(
            input: LMInput(tokens: ids), model: model, parameters: parameters)
        for _ in 0 ..< 3 {
            let token = batched.next()
            #expect(token != nil)
            #expect(token == bare.next())
        }
    }

    @Test("Padded clips use only their valid frames")
    func paddedClips() throws {
        let model = try tinyModel()
        let short = tinyFeatures(frames: 9)
        let padded = concatenated([short, MLXArray.zeros([16, 7])], axis: 1)
        let mask = (MLXArray(0 ..< Int32(16)) .< Int32(9)).asType(.int32)[.newAxis]
        let fromPadded = model.speechFeatures(.init(features: padded[.newAxis], mask: mask))
        let alone = model.speechFeatures(.init(features: short[.newAxis]))
        #expect(fromPadded.shape == [2, 32])
        expectClose(fromPadded, alone, atol: 1e-5, reducedAtol: 1e-3)
    }

    @Test(
        "The generator follows the requested output",
        arguments: [
            LFM3AudioOutput.text, .speech, .interleaved,
        ])
    func generatorModes(output: LFM3AudioOutput) throws {
        let model = try tinyModel()
        let input = LMInput(tokens: MLXArray([Int32(1), 2, 3])[.newAxis])
        let slots = Array(
            try LFM3AudioGenerator(
                model: model, input: input, output: output, audio: .init(temperature: 0),
                maxSlots: 12, endOfTurnTokenIds: []))
        // A random model may end its turn early.
        #expect((1 ... 12).contains(slots.count))
        let speech = slots.filter {
            if case .speech = $0 { true } else if case .endOfSpeech = $0 { true } else { false }
        }
        switch output {
        case .text:
            #expect(speech.isEmpty)
            #expect(!slots.contains(.text(60)) && !slots.contains(.text(62)))
        case .speech:
            #expect(slots.first.map { if case .text = $0 { false } else { true } } == true)
        case .interleaved, .automatic:
            break
        }
        for case .speech(let codes) in slots {
            #expect(codes.count == 4 && codes.allSatisfy { (0 ..< 8).contains($0) })
        }
    }

    @Test("A custom logit processor picks the first token")
    func customProcessor() throws {
        let model = try tinyModel()
        let input = LMInput(tokens: MLXArray([Int32(1), 2, 3])[.newAxis])
        for first in [5, 7] {
            let components = GenerationComponents().appendingLogitProcessor {
                OnlyFirstToken(token: first)
            }
            var generator = try LFM3AudioGenerator(
                model: model, input: input, maxSlots: 4, endOfTurnTokenIds: [],
                components: components)
            #expect(generator.next() == .text(first))
        }
    }

    @Test("Opening with `<|mixed_start|>` answers interleaved")
    func openingInterleaved() throws {
        let model = try tinyModel()
        let input = LMInput(tokens: MLXArray([Int32(1), 2, 3])[.newAxis])
        let components = GenerationComponents().appendingLogitProcessor {
            OnlyFirstToken(token: 59)
        }
        let slots = Array(
            try LFM3AudioGenerator(
                model: model, input: input, audio: .init(temperature: 0), maxSlots: 6,
                endOfTurnTokenIds: [], components: components))
        #expect(slots.first == .text(59))
        // Two text tokens, then speech.
        #expect(
            slots.dropFirst().prefix(3).contains {
                if case .text = $0 { false } else { true }
            })
    }

    @Test("Original checkpoints map their export names and torch layouts")
    func prepareCheckpoint() throws {
        let model = try tinyModel()
        let prepared = try model.prepareCheckpoint(
            .init(weights: [
                "lfm.model.layers.0.conv.conv.weight": MLXArray.zeros([32, 1, 7]),
                "encoder.downsample.conv_in.weight": MLXArray.zeros([8, 1, 3, 3]),
                "encoder.downsample.lin_mid.weight": MLXArray.zeros([8, 8, 1, 1]),
                "encoder.downsample.lin_out.weight": MLXArray.zeros([24, 16, 1]),
                "encoder.lfm.lfm.layers.0.conv.conv.weight": MLXArray.zeros([24, 1, 3]),
                "mlp.model.0.weight": MLXArray.zeros([24]),
                "mlp.model.3.bias": MLXArray.zeros([32]),
                "depthformer.model.embedding_norm.weight": MLXArray.zeros([16]),
                "depth_linear.bias": MLXArray.zeros([64]),
                "mask_embedding": MLXArray.zeros([16]),
                "depth_embeddings.0.embedding.weight": MLXArray.zeros([9, 16]),
                "depth_embeddings.0.to_logits.weight": MLXArray.zeros([9, 16]),
            ]))
        let weights = prepared.weights
        #expect(weights["language_model.model.layers.0.conv.conv.weight"]?.shape == [32, 7, 1])
        #expect(weights["audio_tower.downsample.conv_in.weight"]?.shape == [8, 3, 3, 1])
        #expect(weights["audio_tower.downsample.lin_mid.weight"]?.shape == [8, 1, 1, 8])
        #expect(weights["audio_tower.downsample.lin_out.weight"]?.shape == [24, 16])
        #expect(weights["audio_tower.lfm.layers.0.conv.conv.weight"]?.shape == [24, 3, 1])
        #expect(weights["audio_projector.layer_norm.weight"] != nil)
        #expect(weights["audio_projector.linear_2.bias"] != nil)
        #expect(weights["depth_decoder.model.embedding_norm.weight"] != nil)
        #expect(weights["depth_decoder.depth_linear.bias"] != nil)
        #expect(weights["depth_decoder.mask_embedding"] != nil)
        #expect(weights["depth_decoder.depth_embeddings.0.embedding.weight"] != nil)
        #expect(weights.count == 11)
    }

    // MARK: - Codec

    @Test("Streamed decoding gives the audio of one decode")
    func codecStreaming() throws {
        let config = try JSONDecoder().decode(
            LFM3AudioCodec.Configuration.self, from: Data("{}".utf8))
        let codec = LFM3AudioCodec(config)
        // A Hann window keeps the overlap-add envelope away from zero.
        let window = (0 ..< 1280).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / 1280)) }
        try codec.update(
            parameters: ModuleParameters.unflattened([
                "decoder.istft.window": MLXArray(window)
            ]), verify: .none)
        eval(codec)

        let codes = MLXRandom.randInt(0 ..< 2016, [12, 9], key: MLXRandom.key(4)).asType(.int32)
        let whole = try codec.decode(codes)
        #expect(whole.shape == [9 * 1920])

        let stream = try codec.makeStream()
        var pieces = [MLXArray]()
        for range in [0 ..< 2, 2 ..< 3, 3 ..< 7] {
            pieces.append(try stream.decode(codes[0..., range]))
        }
        pieces.append(try stream.decode(codes[0..., 7...], isFinal: true))
        #expect(stream.isFinished)
        expectClose(concatenated(pieces), whole, atol: 1e-3, reducedAtol: 1e-2)
        #expect(throws: LFM3AudioCodec.CodecError.self) { try stream.decode(codes) }
        #expect(throws: LFM3AudioCodec.CodecError.self) {
            try codec.decode(MLXArray(Array(repeating: Int32(2016), count: 12)).reshaped(12, 1))
        }
    }

    @Test("FSQ digits dequantize to the level grid")
    func fsqDequantization() throws {
        let config = try JSONDecoder().decode(
            LFM3AudioCodec.Configuration.self, from: Data("{}".utf8))
        let quantizer = LFM3AudioCodec.Quantizer(config)
        // Identity projection of the 48 latents into the first 48 of 256 outputs.
        let projection = concatenated([MLXArray.eye(48), MLXArray.zeros([208, 48])], axis: 0)
        try quantizer.update(
            parameters: ModuleParameters.unflattened([
                "project_out.weight": projection, "project_out.bias": MLXArray.zeros([256]),
            ]), verify: .none)
        // 2015 = 7 + 6 * 8 + 5 * 56 + 5 * 336: digits (7, 6, 5, 5) of levels (8, 7, 6, 6).
        let codes = MLXArray(Array(repeating: Int32(2015), count: 12)).reshaped(12, 1)
        let latents = quantizer(codes)[0, 0, ..<4]
        expectClose(latents, [0.75, 1, 2.0 / 3, 2.0 / 3], atol: 1e-6, reducedAtol: 1e-6)
    }

    // MARK: - Local checkpoint

    /// Compares a local checkpoint with a fixture made by mlx-vlm's `lfm3_audio`: log-mel
    /// features, speech features, greedy ASR, greedy TTS with the fixture's MaskGIT orders and
    /// the codec's audio. Set `MLX_LFM3_AUDIO_MODEL` to the checkpoint and
    /// `MLX_LFM3_AUDIO_FIXTURE` to the fixture path without its extension.
    @Test(
        "A local checkpoint matches the mlx-vlm reference",
        .enabled(if: ProcessInfo.processInfo.environment["MLX_LFM3_AUDIO_FIXTURE"] != nil))
    func localCheckpointParity() async throws {
        let environment = ProcessInfo.processInfo.environment
        let directory = URL(fileURLWithPath: try #require(environment["MLX_LFM3_AUDIO_MODEL"]))
        let prefix = try #require(environment["MLX_LFM3_AUDIO_FIXTURE"])
        let fixture = try JSONDecoder().decode(
            AudioFixture.self, from: Data(contentsOf: URL(fileURLWithPath: prefix + ".json")))
        let tensors = try loadArrays(url: URL(fileURLWithPath: prefix + ".safetensors"))
        let exact = fixture.dtype == "float32"

        // Features
        let clip = try #require(tensors["clip"])
        let referenceFeatures = try #require(tensors["input_features"])[0]
        let features = LFM3AudioFeatureExtractor().features(samples: clip)
        #expect(features.shape == referenceFeatures.shape)
        // The fixture's features are float64. The checkpoint's own float32 extractor differs
        // from them by 3e-6 on average and 3.7e-3 at most, in near-silent bins; ours by 4e-6.
        let featureError = abs(features - referenceFeatures)
        #expect(featureError.mean().item(Float.self) < 1e-4)
        #expect(featureError.max().item(Float.self) < 0.02)

        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = LFM3Audio(try JSONDecoder().decode(LFM3AudioConfiguration.self, from: data))
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        let codec = try LFM3AudioCodec.load(modelDirectory: directory)
        if exact {
            for module in [model as Module, codec] {
                try module.update(
                    parameters: module.parameters().mapValues {
                        $0.dtype == .bfloat16 ? $0.asType(.float32) : $0
                    }, verify: .none)
            }
        }

        // Speech features of the reference log-mel input
        let audio = LMInput.ProcessedAudio(features: referenceFeatures[.newAxis])
        let speech = model.speechFeatures(audio).asType(.float32)
        let reference = try #require(tensors["speech_features"])
        let speechError = (norm(speech - reference) / norm(reference)).item(Float.self)
        #expect(speechError < (exact ? 1e-4 : 0.05), "speech features differ by \(speechError)")

        // ASR
        let ids = MLXArray(fixture.asrIds.map(Int32.init))[.newAxis]
        let asrInput = LMInput(
            text: .init(tokens: ids, mask: ones(like: ids).asType(.int8)), audio: audio)
        guard
            case .logits(let output) = try model.prepare(
                asrInput, cache: try model.newCache(parameters: nil), state: nil,
                prefill: .init())
        else {
            Issue.record("Expected logits from a speech prefill")
            return
        }
        let last = output.logits[0, -1].asType(.float32)
        #expect(argMax(last).item(Int.self) == fixture.asrLastTopIds.first)
        let top = MLXArray(fixture.asrLastTopLogits)
        let logitError =
            (abs(last[MLXArray(fixture.asrLastTopIds.map(Int32.init))] - top).max()
            / abs(top).max()).item(Float.self)
        #expect(logitError < (exact ? 1e-4 : 0.03), "last logits differ by \(logitError)")

        let asr = try LFM3AudioGenerator(
            model: model, input: asrInput, output: .text, maxSlots: fixture.asrTokens.count
        ).map { slot -> Int in
            if case .text(let token) = slot { token } else { -1 }
        }
        let asrAgreeing = zip(asr, fixture.asrTokens).prefix { $0 == $1 }.count
        #expect(
            asrAgreeing >= (exact ? fixture.asrTokens.count : 8),
            "ASR tokens agree for \(asrAgreeing) of \(fixture.asrTokens.count)")

        // ASR from the features computed here
        let ownInput = LMInput(
            text: .init(tokens: ids, mask: ones(like: ids).asType(.int8)),
            audio: .init(features: features[.newAxis]))
        let ownASR = try LFM3AudioGenerator(
            model: model, input: ownInput, output: .text, maxSlots: fixture.asrTokens.count
        ).map { slot -> Int in
            if case .text(let token) = slot { token } else { -1 }
        }
        let ownAgreeing = zip(ownASR, fixture.asrTokens).prefix { $0 == $1 }.count
        #expect(
            ownAgreeing >= 8, "ASR tokens agree for \(ownAgreeing) of \(fixture.asrTokens.count)")

        // TTS with greedy codes and the reference reveal order
        let tts = Array(
            try LFM3AudioGenerator(
                model: model,
                input: LMInput(tokens: MLXArray(fixture.ttsIds.map(Int32.init))[.newAxis]),
                output: .speech, audio: .init(temperature: 0), maxSlots: fixture.ttsSlots.count,
                frameOrders: fixture.ttsFrameOrder))
        let expected = fixture.ttsSlots.map { codes -> LFM3AudioSlot? in
            guard let codes else { return nil }
            return codes.first == 2016 ? .endOfSpeech : .speech(codes.map(Int32.init))
        }
        let ttsAgreeing = zip(tts, expected).prefix { actual, expected in
            if let expected {
                actual == expected
            } else if case .text = actual {
                true
            } else {
                false
            }
        }.count
        #expect(
            ttsAgreeing >= (exact ? expected.count : 1),
            "TTS slots agree for \(ttsAgreeing) of \(expected.count)")

        // Codec audio of the reference frames
        let frames = fixture.ttsSlots.compactMap { $0 }.filter { $0.first != 2016 }
        let codes = LFM3AudioCodec.codes(frames: frames.map { $0.map(Int32.init) })
        let waveform = try codec.decode(codes)
        let referenceWaveform = try #require(tensors["waveform"])
        #expect(waveform.shape == referenceWaveform.shape)
        let waveformError =
            (norm(waveform - referenceWaveform) / norm(referenceWaveform)).item(Float.self)
        #expect(waveformError < (exact ? 1e-4 : 0.05), "audio differs by \(waveformError)")

        let stream = try codec.makeStream()
        var pieces = [MLXArray]()
        var start = 0
        while start < codes.dim(1) {
            let end = min(start + 5, codes.dim(1))
            pieces.append(
                try stream.decode(codes[0..., start ..< end], isFinal: end == codes.dim(1)))
            start = end
        }
        let streamed = concatenated(pieces)
        #expect(streamed.shape == waveform.shape)
        // In bfloat16 the chunked attention rounds apart from the whole pass.
        let streamError = (norm(streamed - waveform) / norm(waveform)).item(Float.self)
        #expect(streamError < (exact ? 1e-4 : 0.02), "streamed audio differs by \(streamError)")
    }
}

private struct AudioFixture: Decodable {
    let dtype: String?
    let asrIds: [Int]
    let asrTokens: [Int]
    let asrLastTopIds: [Int]
    let asrLastTopLogits: [Float]
    let ttsIds: [Int]
    let ttsFrameOrder: [[Int]]
    let ttsSlots: [[Int]?]

    enum CodingKeys: String, CodingKey {
        case dtype
        case asrIds = "asr_ids"
        case asrTokens = "asr_tokens"
        case asrLastTopIds = "asr_last_top_ids"
        case asrLastTopLogits = "asr_last_top_logits"
        case ttsIds = "tts_ids"
        case ttsFrameOrder = "tts_frame_order"
        case ttsSlots = "tts_slots"
    }
}
