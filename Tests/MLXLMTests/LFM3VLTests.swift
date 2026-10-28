// Copyright © 2026 Apple Inc.

import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

// Offline checks of the LFM3-VL port on a tiny random model, plus an opt-in parity check of a
// local checkpoint against a fixture from mlx-vlm's `lfm3_vl`.

private func tinyLFM3VLJSON(_ extras: String = "") -> String {
    """
    {
      "model_type": "lfm3_vl", "image_token_id": 60, "video_token_id": 61,
      "downsample_factor": 2, "projector_hidden_size": 32, "projector_bias": true,
      "projector_hidden_act": "gelu", "projector_layer_norm_eps": 1e-6,
      "tie_word_embeddings": true,
      "text_config": {
        "model_type": "lfm3", "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
        "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 8, "sliding_window": 4,
        "layer_types": ["conv", "sliding_attention", "full_attention"]
      },
      "vision_config": {
        "model_type": "lfm3_vl_vision", "hidden_size": 16, "intermediate_size": 24,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "in_channels": 3, "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2,
        "conv_kernel_size": 3, "layer_types": ["conv", "full_attention"],
        "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "axial"}
      }\(extras)
    }
    """
}

private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
    try JSONDecoder().decode(type, from: Data(json.utf8))
}

private func tinyModel() throws -> LFM3VL {
    let config = try decode(LFM3VLConfiguration.self, tinyLFM3VLJSON())
    try config.validateModelConfiguration()
    let model = LFM3VL(config)
    eval(model)
    return model
}

/// The tiny model's ids for the media tokens, and `1000 +` the code point of any other
/// character. The template is the media markup of the messages between ids 5 and 6.
private struct PlaceholderTokenizer: Tokenizer {
    static let special = [
        "<|image|>": 60, "<|video|>": 61, "<|vision_start|>": 62, "<|vision_end|>": 63,
    ]

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokens = [Int]()
        var rest = Substring(text)
        while let character = rest.first {
            if let (token, id) = Self.special.first(where: { rest.hasPrefix($0.key) }) {
                tokens.append(id)
                rest = rest.dropFirst(token.count)
            } else {
                tokens.append(1000 + Int(character.unicodeScalars.first!.value))
                rest = rest.dropFirst()
            }
        }
        return tokens
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { id in
            if let token = convertIdToToken(id) { return token }
            return id >= 1000 ? String(Character(UnicodeScalar(UInt32(id - 1000))!)) : ""
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? { Self.special[token] }
    func convertIdToToken(_ id: Int) -> String? { Self.special.first { $0.value == id }?.key }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let markup = messages.flatMap { $0["content"] as? [[String: String]] ?? [] }.map {
            switch $0["type"] {
            case "image": "<|vision_start|><|image|><|vision_end|>"
            case "video": "<|vision_start|><|video|><|vision_end|>"
            default: ""
            }
        }
        return [5] + encode(text: markup.joined(), addSpecialTokens: false) + [6]
    }
}

struct LFM3VLTests {

    // MARK: - Configuration

    @Test("The released LFM3-VL configuration decodes and validates")
    func releasedConfiguration() throws {
        let config = try decode(LFM3VLConfiguration.self, tinyLFM3VLJSON())
        #expect(config.visionConfiguration.headDim == 8)
        #expect(config.visionConfiguration.ropeTheta == 1_000_000)
        #expect(config.textConfiguration.layerTypes == [.conv, .slidingAttention, .fullAttention])
        #expect(config.imageTokenId == 60)

        var wrongAct = config
        wrongAct.projectorHiddenAct = "relu"
        #expect(throws: ModelFactoryError.self) { try wrongAct.validateModelConfiguration() }

        var wrongMerge = config
        wrongMerge.downsampleFactor = 3
        #expect(throws: ModelFactoryError.self) { try wrongMerge.validateModelConfiguration() }

        var wrongLayer = config
        wrongLayer.visionConfiguration.layerTypes = ["conv", "sliding_attention"]
        #expect(throws: ModelFactoryError.self) { try wrongLayer.validateModelConfiguration() }
    }

    @Test("processor_config.json is read nested or flat")
    func processorConfiguration() throws {
        let nested = try decode(
            LFM3VLProcessorConfiguration.self,
            """
            {"processor_class": "Lfm3VlProcessor", "image_processor": {
              "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5], "merge_size": 2,
              "patch_size": 16, "temporal_patch_size": 2,
              "size": {"longest_edge": 16777216, "shortest_edge": 65536}}}
            """)
        #expect(nested.imageProcessor.minPixels == 65_536)
        #expect(nested.imageProcessor.maxPixels == 16_777_216)
        #expect(nested.imageProcessor.patchSize == 16)

        let flat = try decode(
            LFM3VLProcessorConfiguration.self, #"{"min_pixels": 1024, "max_pixels": 4096}"#)
        #expect(flat.imageProcessor.minPixels == 1024)
        #expect(flat.imageProcessor.maxPixels == 4096)
        #expect(flat.imageProcessor.mergeSize == 2)
    }

    @Test(
        "Resizing follows the reference smart_resize, rounding half to even",
        arguments: [
            // (height, width, maxPixels) -> expected, from the checkpoint's Python function
            (480, 640, 16_777_216, 480, 640),
            (100, 100, 16_777_216, 256, 256),
            (80, 1648, 16_777_216, 64, 1664),
            (3024, 4032, 1_310_720, 960, 1312),
            (10, 1000, 16_777_216, 32, 2560),
            (720, 1296, 1_310_720, 704, 1280),
            (1, 1, 16_777_216, 256, 256),
        ])
    func targetSize(sizes: (Int, Int, Int, Int, Int)) throws {
        let size = try LFM3VLProcessor.targetSize(
            height: sizes.0, width: sizes.1, factor: 32, minPixels: 65_536, maxPixels: sizes.2)
        #expect(size.height == sizes.3)
        #expect(size.width == sizes.4)
    }

    // MARK: - Video

    @Test("The video settings decode from video_processor, with the reference defaults")
    func videoProcessorConfiguration() throws {
        let video = try decode(
            LFM3VLProcessorConfiguration.self,
            """
            {"image_processor": {"patch_size": 16}, "video_processor": {
              "fps": 1.5, "min_frames": 2, "max_frames": 64, "patch_size": 14,
              "size": {"longest_edge": 1000000, "shortest_edge": 2048}}}
            """
        ).videoProcessor
        #expect(video.fps == 1.5)
        #expect(video.minFrames == 2 && video.maxFrames == 64)
        #expect(video.patchSize == 14 && video.mergeSize == 2 && video.temporalPatchSize == 2)
        #expect(video.minPixels == 2048 && video.maxPixels == 1_000_000)

        let defaults = try decode(LFM3VLProcessorConfiguration.self, "{}").videoProcessor
        #expect(defaults.fps == 2 && defaults.minFrames == 4 && defaults.maxFrames == 768)
        #expect(defaults.minPixels == 4_096 && defaults.maxPixels == 25_165_824)
    }

    @Test(
        "Frames are sampled at 2 fps by index, rounded half to even as numpy rounds",
        arguments: [
            // (total, fps) -> indices, from the reference rule
            (75, 30.0, [0, 18, 37, 56, 74]),
            (30, 12.0, [0, 7, 14, 22, 29]),
            (10, 30.0, [0, 3, 6, 9]),
            (3, 30.0, [0, 1, 2]),
            (
                300, 30_000.0 / 1001,
                [
                    0, 16, 31, 47, 63, 79, 94, 110, 126, 142, 157, 173, 189, 205, 220, 236, 252,
                    268, 283, 299,
                ]
            ),
        ])
    func frameIndices(sample: (Int, Double, [Int])) throws {
        let indices = try LFM3VLProcessor.frameIndices(
            total: sample.0, fps: sample.1, settings: .init(), sampling: .init())
        #expect(indices == sample.2)
    }

    @Test("A long clip keeps 768 frames, and the sampling options set the rate or the count")
    func frameIndexLimits() throws {
        let settings = LFM3VLProcessorConfiguration.VideoProcessor()
        let long = try LFM3VLProcessor.frameIndices(
            total: 30_000, fps: 30, settings: settings, sampling: .init())
        #expect(long.count == 768)
        #expect(Array(long.prefix(6)) == [0, 39, 78, 117, 156, 196])
        #expect(Array(long.suffix(3)) == [29_921, 29_960, 29_999])
        #expect(long.reduce(0, +) == 11_519_616)

        let faster = try LFM3VLProcessor.frameIndices(
            total: 75, fps: 30, settings: settings, sampling: .init(targetFramesPerSecond: 4))
        #expect(faster.count == 10)
        let fixed = try LFM3VLProcessor.frameIndices(
            total: 75, fps: 30, settings: settings, sampling: .init(targetFrames: 3))
        #expect(fixed == [0, 37, 74])
        #expect(throws: VLMError.self) {
            try LFM3VLProcessor.frameIndices(
                total: 75, fps: 0, settings: settings, sampling: .init())
        }
    }

    @Test(
        "A clip resizes to fit all its frames in the pixel budget, as the video smart_resize",
        arguments: [
            // (frames, height, width, maxPixels) -> expected, from the checkpoint's Python function
            (5, 360, 640, 25_165_824, 352, 640),
            (20, 1080, 1920, 25_165_824, 832, 1472),
            (768, 1080, 1920, 25_165_824, 128, 224),
            (4, 2160, 3840, 25_165_824, 1856, 3328),
            (4, 20, 1000, 25_165_824, 32, 1600),
            (2, 16, 16, 25_165_824, 64, 64),
            // 5 frames count as 4 (half to even), which fit; 6 do not.
            (5, 64, 64, 20_000, 64, 64),
            (6, 64, 64, 20_000, 32, 32),
        ])
    func videoTargetSize(sizes: (Int, Int, Int, Int, Int, Int)) throws {
        let size = try LFM3VLProcessor.videoTargetSize(
            frames: sizes.0, height: sizes.1, width: sizes.2, factor: 32, temporalFactor: 2,
            minPixels: 4096, maxPixels: sizes.3)
        #expect(size.height == sizes.4)
        #expect(size.width == sizes.5)
    }

    @Test("A clip needs two frames and an aspect ratio within 200")
    func videoTargetSizeLimits() {
        #expect(throws: VLMError.self) {
            try LFM3VLProcessor.videoTargetSize(
                frames: 1, height: 64, width: 64, factor: 32, temporalFactor: 2, minPixels: 4096,
                maxPixels: 25_165_824)
        }
        #expect(throws: VLMError.self) {
            try LFM3VLProcessor.videoTargetSize(
                frames: 4, height: 32, width: 6432, factor: 32, temporalFactor: 2,
                minPixels: 4096, maxPixels: 25_165_824)
        }
    }

    @Test("Each temporal group is labeled with the mean time of its frames")
    func videoPlaceholders() throws {
        let seconds = LFM3VLProcessor.groupSeconds(
            [0, 18, 37, 56, 74].map { Double($0) / 30 }, groupSize: 2)
        // The last frame fills the last group. The double nearest 1.55 lies above it: "1.6".
        #expect(seconds == [0.3, 1.55, 2.466666666666667])

        let tokenizer = PlaceholderTokenizer()
        let template = tokenizer.encode(
            text: "a<|vision_start|><|video|><|vision_end|>b", addSpecialTokens: false)
        let tokens = try LFM3VLProcessor.expandVideoPlaceholders(
            template, clips: [(seconds, 2)], tokenizer: tokenizer)
        let group = "<|vision_start|><|video|><|video|><|vision_end|>"
        #expect(
            tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
                == "a<|vision_start|><0.3 seconds>\(group)<1.6 seconds>\(group)<2.5 seconds>\(group)<|vision_end|>b"
        )
        #expect(throws: VLMError.self) {
            try LFM3VLProcessor.expandVideoPlaceholders(template, clips: [], tokenizer: tokenizer)
        }
    }

    @Test("A video file is sampled by frame index into labeled temporal groups")
    func processorSamplesVideoFile() async throws {
        let url = try await writeGrayClip(frames: 30, fps: 12, width: 96, height: 64)
        defer { try? FileManager.default.removeItem(at: url) }
        try await expectGrayClipInput(.url(url))
    }

    @Test("Decoded frames are sampled as a clip at the rate of their timestamps")
    func processorSamplesFrames() async throws {
        let frames = (0 ..< 30).map {
            UserInput.VideoFrame(
                image: .ciImage(grayImage(8 * $0, width: 96, height: 64)),
                timeStamp: CMTime(value: CMTimeValue($0), timescale: 12))
        }
        try await expectGrayClipInput(.frames(frames))
    }

    // MARK: - Processor

    @Test("The processor expands each image into one token per merge block")
    func processorExpandsPlaceholders() async throws {
        let config = try decode(
            LFM3VLProcessorConfiguration.self,
            #"{"image_processor": {"patch_size": 16, "size": {"shortest_edge": 1024, "longest_edge": 1048576}}}"#
        )
        let processor = LFM3VLProcessor(config, tokenizer: PlaceholderTokenizer())
        let image = CIImage(color: .gray).cropped(to: CGRect(x: 0, y: 0, width: 96, height: 64))
        let input = try await processor.prepare(
            input: UserInput(chat: [.user("hi", images: [.ciImage(image)])]))

        let frames = try #require(input.image?.frames)
        #expect(frames.map { [$0.t, $0.h, $0.w] } == [[1, 4, 6]])
        let tokens = input.text.tokens.asArray(Int.self)
        #expect(tokens.filter { $0 == 60 }.count == 4 * 6 / 4)
        #expect(tokens.first == 5 && tokens.last == 6)
        #expect(input.image?.pixels.shape == [24, 3 * 2 * 16 * 16])
    }

    @Test("Images come before the text, and reasoning is passed apart")
    func messageGenerator() {
        let generator = LFM3VLMessageGenerator()
        let user = generator.generate(message: .user("what?", images: [.ciImage(CIImage())]))
        let parts = user["content"] as? [[String: String]]
        #expect(parts?.map { $0["type"] } == ["image", "text"])

        let assistant = generator.generate(message: .assistant("<think>hmm</think>It is a cat."))
        #expect(assistant["reasoning_content"] as? String == "hmm")
        let content = assistant["content"] as? [[String: String]]
        #expect(content?.last?["text"] == "It is a cat.")
    }

    // MARK: - Model

    @Test("An image prompt prefills to next-token logits and text decodes after it")
    func imagePrefill() throws {
        let model = try tinyModel()
        // One 4 x 4 patch grid: 16 patches, 4 image tokens.
        let pixels = MLXRandom.normal([16, 3 * 2 * 2 * 2], key: MLXRandom.key(1))
        let ids = MLXArray([Int32(1), 60, 60, 60, 60, 2, 3])[.newAxis]
        let input = LMInput(
            text: .init(tokens: ids, mask: ones(like: ids).asType(.int8)),
            image: .init(pixels: pixels, frames: [THW(1, 4, 4)]))

        let cache = try model.newCache(parameters: nil)
        guard
            case .logits(let output) = try model.prepare(
                input, cache: cache, state: nil, prefill: .init(stepSize: 3))
        else {
            Issue.record("Expected logits from an image prefill")
            return
        }
        #expect(output.logits.shape == [1, 1, 64])

        // The same embeddings in one causal pass give the same last logits.
        let embeddings = try model.inputEmbeddings(input)
        let expected = model.languageModel(nil, cache: nil, inputEmbeddings: embeddings)
        expectClose(
            output.logits[0, -1], expected[0, -1], atol: 1e-4, reducedAtol: 2e-2)
        #expect(model(MLXArray([Int32(4)])[.newAxis], cache: cache).shape == [1, 1, 64])
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

    @Test("A video's temporal groups encode one at a time, in order, into its video tokens")
    func videoPrefill() throws {
        let model = try tinyModel()
        // Two temporal groups of a 4 x 4 patch grid: 32 patches, 8 video tokens.
        let pixels = MLXRandom.normal([32, 24], key: MLXRandom.key(3))
        let features = try model.features(pixels, grids: [THW(2, 4, 4)])
        let apart = try model.features(pixels, grids: [THW(1, 4, 4), THW(1, 4, 4)])
        expectClose(features, apart, atol: 1e-5, reducedAtol: 1e-3)

        let group = [Int32(62), 61, 61, 61, 61, 63]
        let ids = MLXArray([Int32(1)] + group + group + [2])[.newAxis]
        let input = LMInput(
            text: .init(tokens: ids, mask: ones(like: ids).asType(.int8)),
            video: .init(pixels: pixels, frames: [THW(2, 4, 4)]))
        let embeddings = try model.inputEmbeddings(input)
        let videoPositions = MLXArray([Int32(2), 3, 4, 5, 8, 9, 10, 11])
        expectClose(embeddings[0][videoPositions], features, atol: 1e-6, reducedAtol: 1e-6)

        let cache = try model.newCache(parameters: nil)
        guard
            case .logits(let output) = try model.prepare(
                input, cache: cache, state: nil, prefill: .init(stepSize: 5))
        else {
            Issue.record("Expected logits from a video prefill")
            return
        }
        let expected = model.languageModel(nil, cache: nil, inputEmbeddings: embeddings)
        expectClose(output.logits[0, -1], expected[0, -1], atol: 1e-4, reducedAtol: 2e-2)
    }

    @Test("A warm cache prefilled with the split suffix matches a cold prefill")
    func splitPreparedInput() throws {
        let model = try tinyModel()
        // Two images of 16 patches, one before and one after the cached prefix.
        let pixels = MLXRandom.normal([32, 24], key: MLXRandom.key(5))
        let ids: [Int32] = [1, 62, 60, 60, 60, 60, 63, 2, 3, 62, 60, 60, 60, 60, 63, 4]
        let tokens = MLXArray(ids)[.newAxis]
        let full = LMInput(
            text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)),
            image: .init(pixels: pixels, frames: [THW(1, 4, 4), THW(1, 4, 4)]))

        let cold = try model.newCache(parameters: nil)
        guard
            case .logits(let expected) = try model.prepare(
                full, cache: cold, state: nil, prefill: .init())
        else {
            Issue.record("Expected logits from an image prefill")
            return
        }

        let prefix = 9
        let firstTokens = tokens[0..., ..<prefix]
        let first = LMInput(
            text: .init(tokens: firstTokens, mask: ones(like: firstTokens).asType(.int8)),
            image: .init(pixels: pixels[..<16], frames: [THW(1, 4, 4)]))
        let warm = try model.newCache(parameters: nil)
        _ = try model.prepare(first, cache: warm, state: nil, prefill: .init())
        let suffix = try #require(model.splitPreparedInput(full, droppingFirst: prefix))
        #expect(suffix.text.tokens.dim(1) == ids.count - prefix)
        #expect(suffix.image?.frames?.count == 1)
        guard
            case .logits(let actual) = try model.prepare(
                suffix, cache: warm, state: nil, prefill: .init())
        else {
            Issue.record("Expected logits from the suffix prefill")
            return
        }
        expectClose(actual.logits[0, -1], expected.logits[0, -1], atol: 1e-4, reducedAtol: 2e-2)
        // A cut inside an image block cannot be split.
        #expect(model.splitPreparedInput(full, droppingFirst: 4) == nil)
    }

    @Test("Feature and placeholder counts must agree")
    func featureCountMismatch() throws {
        let model = try tinyModel()
        let pixels = MLXRandom.normal([16, 24], key: MLXRandom.key(2))
        let ids = MLXArray([Int32(1), 60, 60, 2])[.newAxis]
        let input = LMInput(
            text: .init(tokens: ids), image: .init(pixels: pixels, frames: [THW(1, 4, 4)]))
        #expect(throws: VLMError.self) { try model.inputEmbeddings(input) }
    }

    @Test("HF checkpoints lose the model wrapper, the tied head and torch conv layouts")
    func prepareCheckpoint() throws {
        let model = try tinyModel()
        let prepared = try model.prepareCheckpoint(
            .init(weights: [
                "model.language_model.layers.0.conv.conv.weight": MLXArray.zeros([32, 1, 7]),
                "model.vision_tower.layers.0.conv.spatial_conv.weight": MLXArray.zeros([
                    16, 1, 3, 3,
                ]),
                "model.multi_modal_projector.linear_1.bias": MLXArray.zeros([32]),
                "lm_head.weight": MLXArray.zeros([64, 32]),
            ]))
        #expect(
            Set(prepared.weights.keys) == [
                "language_model.model.layers.0.conv.conv.weight",
                "vision_tower.layers.0.conv.spatial_conv.weight",
                "multi_modal_projector.linear_1.bias",
            ])
        #expect(
            prepared.weights["language_model.model.layers.0.conv.conv.weight"]?.shape == [32, 7, 1])
        #expect(
            prepared.weights["vision_tower.layers.0.conv.spatial_conv.weight"]?.shape == [
                16, 3, 3, 1,
            ])
    }

    @Test("The model declares LFM3's chat conventions")
    func chatConventions() throws {
        let model = try tinyModel()
        #expect(model.toolCallFormat == .xmlFunction)
        #expect(model.reasoningConfig == .lfm3)
    }

    // MARK: - Local checkpoint

    /// Compares a local checkpoint with a fixture made by mlx-vlm's `lfm3_vl`: the vision
    /// features of the fixture's patch pixels, the last prefill logits and 48 greedy tokens.
    /// Set `MLX_LFM3_VL_MODEL` to the checkpoint and `MLX_LFM3_VL_FIXTURE` to the fixture path
    /// without its `.json` / `.safetensors` extension. With `MLX_LFM3_VL_IMAGE` set to the
    /// fixture's image, the Swift preprocessing is compared with the reference pixels too.
    @Test(
        "A local checkpoint matches the mlx-vlm reference",
        .enabled(if: ProcessInfo.processInfo.environment["MLX_LFM3_VL_FIXTURE"] != nil))
    func localCheckpointParity() async throws {
        let environment = ProcessInfo.processInfo.environment
        let reference = try await LocalReference(
            fixture: try #require(environment["MLX_LFM3_VL_FIXTURE"]))
        let ids = reference.ids
        try reference.expectGeneration(
            LMInput(
                text: .init(tokens: ids, mask: ones(like: ids).asType(.int8)),
                image: .init(pixels: reference.pixels, frames: reference.grids)))

        if let imagePath = environment["MLX_LFM3_VL_IMAGE"] {
            let processor = LFM3VLProcessor(
                try reference.processorConfiguration(), tokenizer: PlaceholderTokenizer())
            let image = try #require(CIImage(contentsOf: URL(fileURLWithPath: imagePath)))
            let (swiftPixels, grid) = try processor.preprocess(image: image, processing: nil)
            #expect([grid.t, grid.h, grid.w] == reference.fixture.grid[0])
            let pixelError = abs(swiftPixels - reference.pixels).mean().item(Float.self)
            #expect(pixelError < 0.02, "preprocessed pixels differ by \(pixelError) on average")
        }
    }

    /// ``localCheckpointParity()`` for a video fixture, set in `MLX_LFM3_VL_VIDEO_FIXTURE`. With
    /// `MLX_LFM3_VL_VIDEO` set to the fixture's video, the Swift processor must also sample the
    /// reference's frames and build its prompt ids, with pixels close to the reference's.
    @Test(
        "A local checkpoint matches the mlx-vlm reference on a video",
        .enabled(if: ProcessInfo.processInfo.environment["MLX_LFM3_VL_VIDEO_FIXTURE"] != nil))
    func localCheckpointVideoParity() async throws {
        let environment = ProcessInfo.processInfo.environment
        let reference = try await LocalReference(
            fixture: try #require(environment["MLX_LFM3_VL_VIDEO_FIXTURE"]))
        let ids = reference.ids
        try reference.expectGeneration(
            LMInput(
                text: .init(tokens: ids, mask: ones(like: ids).asType(.int8)),
                video: .init(pixels: reference.pixels, frames: reference.grids)))

        if let videoPath = environment["MLX_LFM3_VL_VIDEO"] {
            let fixture = reference.fixture
            let url = URL(fileURLWithPath: videoPath)
            let track = try #require(
                try await AVURLAsset(url: url).loadTracks(withMediaType: .video).first)
            let (total, fps) = try await LFM3VLProcessor.frameCountAndRate(of: track)
            #expect(total == fixture.totalFrames)
            #expect(abs(fps - (fixture.fps ?? 0)) < 1e-9, "the clip's frame rate is \(fps)")
            let config = try reference.processorConfiguration()
            let indices = try LFM3VLProcessor.frameIndices(
                total: total, fps: fps, settings: config.videoProcessor, sampling: .init())
            #expect(indices == fixture.frameIndices)

            let processor = LFM3VLProcessor(config, tokenizer: FixtureTokenizer(fixture))
            let input = try await processor.prepare(
                input: UserInput(chat: [
                    .user("Describe this video in one sentence.", videos: [.url(url)])
                ]))
            #expect(input.text.tokens.asArray(Int.self) == fixture.ids)
            #expect(input.video?.frames?.map { [$0.t, $0.h, $0.w] } == fixture.grid)
            let swiftPixels = try #require(input.video?.pixels)
            try #require(swiftPixels.shape == reference.pixels.shape)
            let pixelError = abs(swiftPixels - reference.pixels).mean().item(Float.self)
            // AVFoundation decodes the H.264 frames about 2 levels (0.016) from OpenCV
            // (measured): most of the 0.015 this clip shows.
            #expect(pixelError < 0.02, "preprocessed pixels differ by \(pixelError) on average")
        }
    }
}

/// A local checkpoint, `MLX_LFM3_VL_MODEL`, with a fixture made by mlx-vlm's `lfm3_vl`.
private struct LocalReference {
    let directory: URL
    let fixture: VisionFixture
    let model: LFM3VL
    /// The fixture's patch pixels and their grids.
    let pixels: MLXArray
    let grids: [THW]

    var exact: Bool { fixture.dtype == "float32" }
    var ids: MLXArray { MLXArray(fixture.ids.map(Int32.init))[.newAxis] }

    /// Loads the checkpoint in the fixture's dtype and compares the vision features of the
    /// fixture's pixels with the reference ones.
    init(fixture prefix: String) async throws {
        directory = URL(
            fileURLWithPath: try #require(ProcessInfo.processInfo.environment["MLX_LFM3_VL_MODEL"]))
        fixture = try JSONDecoder().decode(
            VisionFixture.self, from: Data(contentsOf: URL(fileURLWithPath: prefix + ".json")))
        let tensors = try loadArrays(url: URL(fileURLWithPath: prefix + ".safetensors"))
        pixels = try #require(tensors["pixel_values"])
        let referenceFeatures = try #require(tensors["features"])
        grids = fixture.grid.map { THW($0[0], $0[1], $0[2]) }
        let exact = fixture.dtype == "float32"

        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        model = LFM3VL(try JSONDecoder().decode(LFM3VLConfiguration.self, from: data))
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        if exact {
            try model.update(
                parameters: model.parameters().mapValues {
                    $0.dtype == .bfloat16 ? $0.asType(.float32) : $0
                }, verify: .none)
        }

        let features = try model.features(pixels, grids: grids).asType(.float32)
        let featureError =
            (norm(features - referenceFeatures) / norm(referenceFeatures)).item(Float.self)
        // bfloat16 runs of either port sit 2% from float32 (measured), so they differ by 2.5%.
        #expect(featureError < (exact ? 1e-4 : 0.05), "features differ by \(featureError)")
    }

    func processorConfiguration() throws -> LFM3VLProcessorConfiguration {
        try JSONDecoder().decode(
            LFM3VLProcessorConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("processor_config.json")))
    }

    /// Prefills `input`, the fixture's prompt with its media, and compares the last logits and
    /// the greedy tokens with the reference ones.
    func expectGeneration(_ input: LMInput) throws {
        let cache = try model.newCache(parameters: nil)
        guard
            case .logits(let output) = try model.prepare(
                input, cache: cache, state: nil, prefill: .init())
        else {
            Issue.record("Expected logits from a media prefill")
            return
        }
        let last = output.logits[0, -1].asType(.float32)
        let topIds = argSort(-last)[..<20].asArray(Int32.self).map(Int.init)
        #expect(topIds.first == fixture.lastTopIds.first)
        let reference = MLXArray(fixture.lastTopLogits)
        let relative =
            (abs(last[MLXArray(fixture.lastTopIds.map(Int32.init))] - reference).max()
            / abs(reference).max()).item(Float.self)
        #expect(relative < (exact ? 1e-4 : 0.03), "last logits differ by \(relative)")

        var logits = output.logits
        var generated = [Int]()
        for _ in fixture.greedyTokens {
            let token = argMax(logits[0, -1], axis: -1).item(Int.self)
            generated.append(token)
            logits = model(MLXArray([Int32(token)])[.newAxis], cache: cache)
        }
        let agreeing = zip(generated, fixture.greedyTokens).prefix { $0 == $1 }.count
        // In bfloat16 the description can change wording after its first words.
        #expect(
            agreeing >= (exact ? generated.count : 2),
            "greedy tokens agree for \(agreeing) of \(generated.count)")
    }
}

private struct VisionFixture: Decodable {
    let dtype: String?
    let ids: [Int]
    let grid: [[Int]]
    let lastTopIds: [Int]
    let lastTopLogits: [Float]
    let greedyTokens: [Int]
    // Video fixtures only: what `FixtureTokenizer` answers, and the frames the reference sampled.
    let templateIds: [Int]?
    let textIds: [String: [Int]]?
    let totalFrames: Int?
    let fps: Double?
    let frameIndices: [Int]?

    enum CodingKeys: String, CodingKey {
        case dtype, ids, grid, fps
        case lastTopIds = "last_top_ids"
        case lastTopLogits = "last_top_logits"
        case greedyTokens = "greedy_tokens"
        case templateIds = "template_ids"
        case textIds = "text_ids"
        case totalFrames = "total_frames"
        case frameIndices = "frame_indices"
    }
}

/// The checkpoint tokenizer's answers for one video fixture: the ids of its chat template and
/// of each text the processor encodes or looks up. Any other text encodes to -1.
private struct FixtureTokenizer: Tokenizer {
    let fixture: VisionFixture

    init(_ fixture: VisionFixture) {
        self.fixture = fixture
    }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        fixture.textIds?[text] ?? [-1]
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }

    func convertTokenToId(_ token: String) -> Int? {
        fixture.textIds?[token].flatMap { $0.count == 1 ? $0[0] : nil }
    }

    func convertIdToToken(_ id: Int) -> String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        fixture.templateIds ?? []
    }
}

// MARK: - Video clips

/// An opaque gray of `level` out of 255, as decoded 8-bit sRGB.
private func grayImage(_ level: Int, width: Int, height: Int) -> CIImage {
    let pixel = [UInt8(level), UInt8(level), UInt8(level), 255]
    return CIImage(
        bitmapData: Data(repeatElement(pixel, count: width * height).joined()),
        bytesPerRow: width * 4, size: CGSize(width: width, height: height), format: .RGBA8,
        colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
}

/// Writes an H.264 clip of `frames` frames at `fps` whose frame i is gray 8i.
private func writeGrayClip(frames: Int, fps: Int32, width: Int, height: Int) async throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("lfm3-vl-\(UUID().uuidString).mp4")
    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(
        mediaType: .video,
        outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width,
            AVVideoHeightKey: height,
            // Near lossless, so each frame keeps its gray.
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 20_000_000],
        ])
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
        ])
    writer.add(input)
    guard writer.startWriting(), let pool = adaptor.pixelBufferPool else {
        throw writer.error ?? CocoaError(.fileWriteUnknown)
    }
    writer.startSession(atSourceTime: .zero)
    for index in 0 ..< frames {
        while !input.isReadyForMoreMediaData {
            try await Task.sleep(for: .milliseconds(1))
        }
        var created: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &created)
        let buffer = try #require(created)
        CVPixelBufferLockBaseAddress(buffer, [])
        let bytes = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        for row in 0 ..< height {
            for column in 0 ..< width {
                let pixel = bytes + row * CVPixelBufferGetBytesPerRow(buffer) + column * 4
                (pixel[0], pixel[1], pixel[2], pixel[3]) = (
                    UInt8(8 * index), UInt8(8 * index), UInt8(8 * index), 255
                )
            }
        }
        CVPixelBufferUnlockBaseAddress(buffer, [])
        adaptor.append(
            buffer, withPresentationTime: CMTime(value: CMTimeValue(index), timescale: fps))
    }
    input.markAsFinished()
    // The last frame lasts as long as the others.
    writer.endSession(atSourceTime: CMTime(value: CMTimeValue(frames), timescale: fps))
    await writer.finishWriting()
    guard writer.status == .completed else { throw writer.error ?? CocoaError(.fileWriteUnknown) }
    return url
}

/// Checks the processor's input for a 2.5 s clip at 12 fps whose frame i is gray 8i: frames 0,
/// 7, 14, 22 and 29 (repeated to fill the last group) at 64 × 96, in three groups of 6 tokens
/// labeled with their mean times.
private func expectGrayClipInput(_ video: UserInput.Video) async throws {
    let processor = LFM3VLProcessor(
        try decode(LFM3VLProcessorConfiguration.self, "{}"), tokenizer: PlaceholderTokenizer())
    let input = try await processor.prepare(input: UserInput(chat: [.user("hi", videos: [video])]))
    #expect(input.video?.frames?.map { [$0.t, $0.h, $0.w] } == [[3, 4, 6]])
    let group = "<|vision_start|>" + String(repeating: "<|video|>", count: 6) + "<|vision_end|>"
    #expect(
        PlaceholderTokenizer().decode(
            tokenIds: input.text.tokens.asArray(Int.self), skipSpecialTokens: false)
            == "<|vision_start|><0.3 seconds>\(group)<1.5 seconds>\(group)<2.4 seconds>\(group)<|vision_end|>"
    )
    // Patch rows are [group, patch] and their values [channel, temporal slot, pixel].
    let pixels = try #require(input.video?.pixels).reshaped(3, 24, 3, 2, 256)
    let frames = MLX.round((pixels.mean(axes: [1, 2, 4]) + 1) * 255 / 16).asType(.int32)
    #expect(frames.asArray(Int32.self) == [0, 7, 14, 22, 29, 29])
}
