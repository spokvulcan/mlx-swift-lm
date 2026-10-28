// Copyright © 2026 Apple Inc.

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

/// Expands the image placeholder of a two-turn prompt into `count` image tokens.
private struct PlaceholderTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokens = [Int]()
        var rest = Substring(text)
        while !rest.isEmpty {
            if rest.hasPrefix("<|vision_start|>") {
                tokens.append(62)
                rest = rest.dropFirst("<|vision_start|>".count)
            } else if rest.hasPrefix("<|vision_end|>") {
                tokens.append(63)
                rest = rest.dropFirst("<|vision_end|>".count)
            } else if rest.hasPrefix("<|image|>") {
                tokens.append(60)
                rest = rest.dropFirst("<|image|>".count)
            } else {
                tokens.append(1)
                rest = rest.dropFirst()
            }
        }
        return tokens
    }

    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [5] + encode(text: "<|vision_start|><|image|><|vision_end|>", addSpecialTokens: false) + [6]
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
        let directory = URL(fileURLWithPath: try #require(environment["MLX_LFM3_VL_MODEL"]))
        let prefix = try #require(environment["MLX_LFM3_VL_FIXTURE"])
        let fixture = try JSONDecoder().decode(
            VisionFixture.self, from: Data(contentsOf: URL(fileURLWithPath: prefix + ".json")))
        let tensors = try loadArrays(url: URL(fileURLWithPath: prefix + ".safetensors"))
        let pixels = try #require(tensors["pixel_values"])
        let referenceFeatures = try #require(tensors["features"])
        let exact = fixture.dtype == "float32"

        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = LFM3VL(try JSONDecoder().decode(LFM3VLConfiguration.self, from: data))
        try await loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        if exact {
            try model.update(
                parameters: model.parameters().mapValues {
                    $0.dtype == .bfloat16 ? $0.asType(.float32) : $0
                }, verify: .none)
        }

        let grids = fixture.grid.map { THW($0[0], $0[1], $0[2]) }
        let features = try model.features(pixels, grids: grids).asType(.float32)
        let featureError =
            (norm(features - referenceFeatures) / norm(referenceFeatures)).item(Float.self)
        // bfloat16 runs of either port sit 2% from float32 (measured), so they differ by 2.5%.
        #expect(featureError < (exact ? 1e-4 : 0.05), "features differ by \(featureError)")

        let ids = MLXArray(fixture.ids.map(Int32.init))[.newAxis]
        let input = LMInput(
            text: .init(tokens: ids, mask: ones(like: ids).asType(.int8)),
            image: .init(pixels: pixels, frames: grids))
        let cache = try model.newCache(parameters: nil)
        guard
            case .logits(let output) = try model.prepare(
                input, cache: cache, state: nil, prefill: .init())
        else {
            Issue.record("Expected logits from an image prefill")
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

        if let imagePath = environment["MLX_LFM3_VL_IMAGE"] {
            let configData = try Data(
                contentsOf: directory.appendingPathComponent("processor_config.json"))
            let processor = LFM3VLProcessor(
                try JSONDecoder().decode(LFM3VLProcessorConfiguration.self, from: configData),
                tokenizer: PlaceholderTokenizer())
            let image = try #require(CIImage(contentsOf: URL(fileURLWithPath: imagePath)))
            let (swiftPixels, grid) = try processor.preprocess(image: image, processing: nil)
            #expect([grid.t, grid.h, grid.w] == fixture.grid[0])
            let pixelError = abs(swiftPixels - pixels).mean().item(Float.self)
            #expect(pixelError < 0.02, "preprocessed pixels differ by \(pixelError) on average")
        }
    }
}

private struct VisionFixture: Decodable {
    let dtype: String?
    let ids: [Int]
    let grid: [[Int]]
    let lastTopIds: [Int]
    let lastTopLogits: [Float]
    let greedyTokens: [Int]

    enum CodingKeys: String, CodingKey {
        case dtype, ids, grid
        case lastTopIds = "last_top_ids"
        case lastTopLogits = "last_top_logits"
        case greedyTokens = "greedy_tokens"
    }
}
