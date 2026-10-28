// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

// Offline checks of the LFM3 port on a tiny random model, plus an opt-in parity check of a
// local checkpoint against a fixture from mlx-vlm (scripts in the test's comment).

private func tinyLFM3JSON(_ extras: String = "") -> String {
    """
    {
      "model_type": "lfm3", "vocab_size": 64, "hidden_size": 32, "intermediate_size": 48,
      "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
      "head_dim": 8, "norm_eps": 1e-5, "conv_L_cache": 7, "sliding_window": 4,
      "layer_types": ["conv", "sliding_attention", "full_attention", "conv"],
      "rope_parameters": {
        "full_attention": {
          "rope_type": "default", "rope_theta": 1000000.0, "partial_rotary_factor": 0.5
        },
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000}
      }\(extras)
    }
    """
}

private func configuration(_ json: String = tinyLFM3JSON()) throws -> LFM3Configuration {
    try JSONDecoder().decode(LFM3Configuration.self, from: Data(json.utf8))
}

private func tinyModel() throws -> LFM3Model {
    let model = LFM3Model(try configuration())
    eval(model)
    return model
}

private func tokens(_ count: Int) -> MLXArray {
    MLXArray((0 ..< count).map { Int32(($0 * 7 + 3) % 64) }).reshaped(1, count)
}

struct LFM3Tests {

    // MARK: - Configuration

    @Test("The released LFM3-1.7B configuration decodes")
    func releasedConfiguration() throws {
        let json = """
            {
              "model_type": "lfm3", "vocab_size": 128000, "hidden_size": 2048,
              "intermediate_size": 7680, "num_hidden_layers": 24, "num_attention_heads": 16,
              "num_key_value_heads": 4, "head_dim": 128, "norm_eps": 1e-05, "conv_L_cache": 7,
              "conv_bias": false, "sliding_window": 256, "use_sandwich_norm": true,
              "tie_word_embeddings": true,
              "rope_parameters": {
                "full_attention": {
                  "partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"
                },
                "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}
              },
              "layer_types": ["conv", "conv", "full_attention", "conv", "sliding_attention",
                "conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention",
                "conv", "sliding_attention", "conv", "full_attention", "conv",
                "sliding_attention", "conv", "full_attention", "conv", "sliding_attention",
                "conv", "full_attention", "conv"],
              "vision_config": {}
            }
            """
        let config = try configuration(json)
        try config.validateModelConfiguration()
        #expect(config.headDim == 128)
        #expect(config.layerTypes.filter { $0 == .conv }.count == 13)
        #expect(config.layerTypes.filter { $0 == .slidingAttention }.count == 5)
        #expect(config.layerTypes.filter { $0 == .fullAttention }.count == 6)
        #expect(config.ropeParameters["sliding_attention"]?["rope_theta"]?.asFloat() == 10_000)
    }

    @Test("Missing fields take the reference defaults")
    func defaults() throws {
        let config = try configuration(#"{"num_hidden_layers": 8}"#)
        #expect(
            config.layerTypes == [
                .conv, .conv, .fullAttention, .conv, .slidingAttention, .conv, .fullAttention,
                .conv,
            ])
        #expect(config.headDim == 128)
        #expect(config.convLCache == 7)
        #expect(config.useSandwichNorm)
        #expect(config.tieWordEmbeddings)
        #expect(
            config.ropeParameters["full_attention"]?["partial_rotary_factor"]?.asFloat() == 0.5)
        try config.validateModelConfiguration()

        // A flat rope dictionary applies to both attention types.
        let flat = try configuration(
            #"{"num_hidden_layers": 8, "rope_parameters": {"rope_theta": 5000.0}}"#)
        #expect(flat.ropeParameters["sliding_attention"]?["rope_theta"]?.asFloat() == 5000)
        #expect(flat.ropeParameters["full_attention"]?["rope_theta"]?.asFloat() == 5000)

        // As in `Lfm3Config`, `tie_embedding` wins over `tie_word_embeddings`.
        let untied = try configuration(
            #"{"tie_embedding": false, "tie_word_embeddings": true}"#)
        #expect(!untied.tieWordEmbeddings)
    }

    @Test("Inconsistent configurations are refused")
    func validation() throws {
        let wrongCount = try configuration(#"{"num_hidden_layers": 2, "layer_types": ["conv"]}"#)
        #expect(throws: ModelFactoryError.self) { try wrongCount.validateModelConfiguration() }

        var noWindow = try configuration()
        noWindow.slidingWindow = nil
        #expect(throws: ModelFactoryError.self) { try noWindow.validateModelConfiguration() }

        var scaled = try configuration()
        scaled.ropeParameters["full_attention"]?["rope_type"] = .string("yarn")
        #expect(throws: ModelFactoryError.self) { try scaled.validateModelConfiguration() }

        var missing = try configuration()
        missing.ropeParameters["sliding_attention"] = nil
        #expect(throws: ModelFactoryError.self) { try missing.validateModelConfiguration() }

        #expect(throws: DecodingError.self) {
            try configuration(#"{"num_hidden_layers": 1, "layer_types": ["mamba"]}"#)
        }
    }

    // MARK: - Rotary embedding

    @Test("Partial RoPE rotates the leading frequencies of each head half")
    func partialRoPE() throws {
        // The reference zero-tail layout: the first `factor * dim / 2` frequencies follow the
        // full-head progression and the rest are zero; rotate_half pairs i with i + dim / 2.
        let (dim, length, offset, base): (Int, Int, Int, Float) = (16, 5, 3, 10_000)
        let rope = makeLFM3RoPE(
            headDim: dim,
            parameters: ["rope_theta": .float(base), "partial_rotary_factor": .float(0.5)])
        let x = MLXRandom.normal([1, 2, length, dim], key: MLXRandom.key(0))

        let angles = dim / 4
        var inverse = (0 ..< angles).map { pow(base, -2 * Float($0) / Float(dim)) }
        inverse += Array(repeating: 0, count: dim / 2 - angles)
        let positions = MLXArray((offset ..< offset + length).map(Float.init))[0..., .newAxis]
        let frequencies = positions * MLXArray(inverse)[.newAxis]
        let emb = concatenated([frequencies, frequencies], axis: -1)
        let halves = x.split(parts: 2, axis: -1)
        let rotated = concatenated([-halves[1], halves[0]], axis: -1)
        let expected = x * cos(emb) + rotated * sin(emb)

        expectClose(rope(x, offset: offset), expected, atol: 1e-5, reducedAtol: 1e-5)
        expectClose(
            rope(x, offset: MLXArray([Int32(offset)])), expected, atol: 1e-5, reducedAtol: 1e-5)
    }

    // MARK: - Forward and cache

    @Test("Caches follow the layer types")
    func cacheTopology() throws {
        let cache = try tinyModel().newCache(parameters: nil)
        #expect(cache.count == 4)
        #expect(cache[0] is MambaCache)
        #expect((cache[1] as? RotatingKVCache)?.maxSize == 4)
        #expect(cache[2] is KVCacheSimple)
        #expect(cache[3] is MambaCache)
        #expect(try tinyModel().kvHeads == [0, 2, 2, 0])
    }

    @Test("Cached decoding past the sliding window matches a full forward")
    func cachedDecodeMatchesFullForward() throws {
        let model = try tinyModel()
        let input = tokens(13)
        let expected = model(input, cache: nil)
        #expect(expected.shape == [1, 13, 64])

        let cache = try model.newCache(parameters: nil)
        let prefill = model(input[0..., ..<5], cache: cache)
        expectClose(prefill, expected[0..., ..<5], atol: 1e-4, reducedAtol: 2e-2)
        for position in 5 ..< 13 {
            let step = model(input[0..., position ..< (position + 1)], cache: cache)
            expectClose(
                step, expected[0..., position ..< (position + 1)], atol: 1e-4, reducedAtol: 2e-2)
        }
        #expect(cache[1].offset == 13)
        #expect(cache[2].offset == 13)
    }

    @Test("Chunked prefill leaves the same cache as one forward")
    func chunkedPrefill() throws {
        let model = try tinyModel()
        let input = LMInput(tokens: tokens(11)[0])
        let next = tokens(12)[0..., 11...]

        let oneShot = try model.newCache(parameters: nil)
        _ = model(tokens(11), cache: oneShot)
        let expected = model(next, cache: oneShot)

        let chunked = try model.newCache(parameters: nil)
        guard
            case .tokens(let rest) = try model.prepare(
                input, cache: chunked, state: nil, prefill: .init(stepSize: 3))
        else {
            Issue.record("Expected tokens for the iterator")
            return
        }
        _ = model(rest.tokens[.newAxis], cache: chunked)
        expectClose(model(next, cache: chunked), expected, atol: 1e-4, reducedAtol: 2e-2)
    }

    // MARK: - Checkpoint

    @Test("Checkpoints lose the language_model wrapper, the tied head and torch conv layout")
    func prepareCheckpoint() throws {
        let model = try tinyModel()
        let checkpoint = ModelCheckpoint(
            weights: [
                "language_model.model.layers.0.conv.conv.weight": MLXArray.zeros([32, 1, 7]),
                "model.layers.3.conv.conv.weight": MLXArray.zeros([32, 7, 1]),
                "language_model.model.embed_tokens.weight": MLXArray.zeros([64, 32]),
                "language_model.lm_head.weight": MLXArray.zeros([64, 32]),
            ],
            perLayerQuantization: .init(perLayerQuantization: [
                "language_model.model.embed_tokens": .quantize(.init(groupSize: 32, bits: 8))
            ]))
        let prepared = try model.prepareCheckpoint(checkpoint)
        #expect(
            Set(prepared.weights.keys) == [
                "model.layers.0.conv.conv.weight", "model.layers.3.conv.conv.weight",
                "model.embed_tokens.weight",
            ])
        #expect(prepared.weights["model.layers.0.conv.conv.weight"]?.shape == [32, 7, 1])
        #expect(prepared.weights["model.layers.3.conv.conv.weight"]?.shape == [32, 7, 1])
        #expect(prepared.perLayerQuantization?.quantization(layer: "model.embed_tokens")?.bits == 8)
    }

    @Test("The model declares LFM3's chat conventions")
    func chatConventions() throws {
        let model = try tinyModel()
        #expect(model.toolCallFormat == .xmlFunction)
        #expect(model.reasoningConfig == .lfm3)
    }

    // MARK: - Local checkpoint

    /// Compares a local checkpoint with a fixture made by mlx-vlm's `lfm3` model: the prompt
    /// ids, the last logits of a one-shot forward and 64 greedy tokens. Set
    /// `MLX_LFM3_MODEL` to the checkpoint directory and `MLX_LFM3_FIXTURE` to the JSON. A
    /// fixture with `"dtype": "float32"` runs both models in float32 and must match closely.
    @Test(
        "A local checkpoint matches the mlx-vlm reference",
        .enabled(if: ProcessInfo.processInfo.environment["MLX_LFM3_FIXTURE"] != nil))
    func localCheckpointParity() throws {
        let environment = ProcessInfo.processInfo.environment
        let directory = URL(fileURLWithPath: try #require(environment["MLX_LFM3_MODEL"]))
        let fixture = try JSONDecoder().decode(
            TextFixture.self,
            from: Data(
                contentsOf: URL(fileURLWithPath: try #require(environment["MLX_LFM3_FIXTURE"]))))

        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let model = LFM3Model(try JSONDecoder().decode(LFM3Configuration.self, from: data))
        try loadWeights(
            modelDirectory: directory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        if fixture.dtype == "float32" {
            // A float32 fixture separates porting errors from bf16 rounding.
            try model.update(
                parameters: model.parameters().mapValues {
                    $0.dtype == .bfloat16 ? $0.asType(.float32) : $0
                }, verify: .none)
        }

        let ids = MLXArray(fixture.ids.map(Int32.init))
        let last = model(ids[.newAxis], cache: nil)[0, -1].asType(.float32)
        let topIds = argSort(-last)[..<20].asArray(Int32.self).map(Int.init)
        #expect(topIds.first == fixture.lastTopIds.first)
        #expect(Set(topIds).intersection(fixture.lastTopIds).count >= 18)
        let reference = MLXArray(fixture.lastTopLogits)
        let actual = last[MLXArray(fixture.lastTopIds.map(Int32.init))]
        let relative = (abs(actual - reference).max() / abs(reference).max()).item(Float.self)
        // Measured: 2.4e-6 in float32; 1.6e-2 in bfloat16, from rounding in other kernels.
        let exact = fixture.dtype == "float32"
        #expect(
            relative < (exact ? 1e-4 : 0.03),
            "last logits differ by \(relative) of their largest magnitude")

        // The iterator's path: chunked prefill into the caches, then greedy steps.
        let cache = try model.newCache(parameters: nil)
        guard
            case .tokens(let rest) = try model.prepare(
                LMInput(tokens: ids), cache: cache, state: nil, prefill: .init())
        else {
            Issue.record("Expected tokens for the iterator")
            return
        }
        var logits = model(rest.tokens[.newAxis], cache: cache)
        var generated = [Int]()
        for _ in fixture.greedyTokens {
            let token = argMax(logits[0, -1], axis: -1).item(Int.self)
            generated.append(token)
            logits = model(MLXArray([Int32(token)])[.newAxis], cache: cache)
        }
        let agreeing = zip(generated, fixture.greedyTokens).prefix { $0 == $1 }.count
        #expect(
            agreeing >= (exact ? generated.count : 32),
            "greedy tokens agree for \(agreeing) of \(generated.count)")
    }
}

private struct TextFixture: Decodable {
    let dtype: String?
    let ids: [Int]
    let lastTopIds: [Int]
    let lastTopLogits: [Float]
    let greedyTokens: [Int]

    enum CodingKeys: String, CodingKey {
        case dtype
        case ids
        case lastTopIds = "last_top_ids"
        case lastTopLogits = "last_top_logits"
        case greedyTokens = "greedy_tokens"
    }
}
