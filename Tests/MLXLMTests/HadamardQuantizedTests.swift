// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import MLXVLM
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

final class HadamardQuantizedTests: XCTestCase {

    // MARK: - Fixtures

    /// A deterministic ±1 vector.
    private static func signs(_ n: Int) -> MLXArray {
        TurboQuantRotation.whtSigns(dim: n, seed: 7)
    }

    /// A deterministic float array with values in about `[-1, 1]`.
    private static func values(_ shape: [Int], phase: Float = 0) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray((0 ..< count).map { sin(Float($0) * 0.37 + phase) }, shape)
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: MLXArray, rtol: Double = 1e-4, atol: Double = 1e-4,
        _ message: String = "", file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, message, file: file, line: line)
        XCTAssertTrue(
            actual.allClose(expected, rtol: rtol, atol: atol).item(Bool.self),
            "\(message) actual=\(actual) expected=\(expected)", file: file, line: line)
    }

    private func assertThrows(
        _ expected: HadamardQuantizedCheckpointError, file: StaticString = #filePath,
        line: UInt = #line, _ body: () throws -> Void
    ) {
        XCTAssertThrowsError(try body(), file: file, line: line) {
            XCTAssertEqual(
                $0 as? HadamardQuantizedCheckpointError, expected, file: file, line: line)
        }
    }

    /// The validation the `PrismHadamardQwen35` classes run.
    private static func validateForQwen35(_ manifest: HadamardQuantizedManifest) throws {
        try manifest.validate(baseModelType: "qwen3_5", tensorNamespace: "mlx-vlm-qwen3_5")
    }

    // MARK: - Rotation

    func testForwardMatchesExplicitSignedHadamard() {
        let block = 8
        let rotation = SignedBlockHadamard(blockSize: block)
        let x = Self.values([3, 16])
        let signs = Self.signs(16)

        let h = TurboQuantRotation.hadamardMatrix(dim: block) / sqrt(Float(block))
        let expected = matmul((x * signs).reshaped(-1, block), h).reshaped(3, 16)

        assertClose(rotation.forward(x, signs: signs), expected)
    }

    func testInverseUndoesForwardAndKeepsDType() {
        let rotation = SignedBlockHadamard(blockSize: 32)
        let x = Self.values([2, 5, 64]).asType(.float16)
        let signs = Self.signs(64)

        let rotated = rotation.forward(x, signs: signs)
        XCTAssertEqual(rotated.dtype, .float16)
        assertClose(rotation.inverse(rotated, signs: signs), x, rtol: 1e-2, atol: 1e-2)
    }

    // MARK: - Layers

    func testLinearRotatesTheInputBeforeThePackedMatmul() {
        let (inputs, outputs, groupSize, bits) = (64, 8, 32, 2)
        let rotation = SignedBlockHadamard(blockSize: 32)
        let (weight, scales, biases) = quantized(
            Self.values([outputs, inputs]), groupSize: groupSize, bits: bits)
        let signs = Self.signs(inputs)
        let layer = HadamardQuantizedLinear(
            weight: weight, scales: scales, biases: biases, signs: signs,
            groupSize: groupSize, bits: bits, rotation: rotation)
        let x = Self.values([2, inputs], phase: 1)

        let rotated = rotation.forward(x, signs: signs)
        let expected = quantizedMM(
            rotated, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits)
        assertClose(layer(x), expected)

        let dequantizedWeight = dequantized(
            weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
        assertClose(layer(x), matmul(rotated, dequantizedWeight.T), rtol: 1e-3, atol: 1e-3)

        let unrotated = quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits)
        XCTAssertFalse(layer(x).allClose(unrotated, rtol: 1e-2, atol: 1e-2).item(Bool.self))
    }

    func testEmbeddingUnrotatesRowsAndRotatesForTheHead() {
        let (count, dimensions, groupSize, bits) = (16, 64, 32, 2)
        let rotation = SignedBlockHadamard(blockSize: 32)
        let (weight, scales, biases) = quantized(
            Self.values([count, dimensions]), groupSize: groupSize, bits: bits)
        let signs = Self.signs(dimensions)
        let layer = HadamardQuantizedEmbedding(
            replacing: Embedding(weight: Self.values([count, dimensions])), groupSize: groupSize,
            bits: bits, rotation: rotation)
        layer.update(parameters: ModuleParameters.unflattened(["signs": signs]))

        XCTAssertEqual(layer.shape.0, count)
        XCTAssertEqual(layer.shape.1, dimensions)

        let ids = MLXArray([3, 5, 1, 3], [2, 2])
        let rows = dequantized(
            weight[ids.flattened()], scales: scales[ids.flattened()],
            biases: biases.map { $0[ids.flattened()] }, groupSize: groupSize, bits: bits)
        let expected = rotation.inverse(rows, signs: signs).reshaped(2, 2, dimensions)
        assertClose(layer(ids), expected)

        let x = Self.values([3, dimensions], phase: 2)
        let expectedLogits = quantizedMM(
            rotation.forward(x, signs: signs), weight, scales: scales, biases: biases,
            transpose: true, groupSize: groupSize, bits: bits)
        assertClose(layer.asLinear(x), expectedLogits)
    }

    // MARK: - Substitution

    private final class TinyMLP: Module {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        override init() {
            _gate.wrappedValue = Linear(64, 32, bias: false)
            _up.wrappedValue = Linear(64, 32, bias: false)
        }
    }

    private final class TinyInner: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        let layers: [TinyMLP]

        override init() {
            _embedTokens.wrappedValue = Embedding(embeddingCount: 16, dimensions: 64)
            layers = [TinyMLP()]
        }
    }

    private final class TinyModel: Module {
        @ModuleInfo(key: "model") var model: TinyInner
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        override init() {
            _model.wrappedValue = TinyInner()
            _lmHead.wrappedValue = Linear(64, 16, bias: false)
        }
    }

    private final class TinyWrapper: Module {
        @ModuleInfo(key: "language_model") var languageModel: TinyModel

        override init() {
            _languageModel.wrappedValue = TinyModel()
        }
    }

    private static let tinyManifest = HadamardQuantizedManifest(
        baseModelType: "tiny", tensorNamespace: "tiny",
        modules: [
            .init(path: "model.embed_tokens", block: 32, embedding: true),
            .init(path: "model.layers.0.gate_proj", block: 32),
            .init(path: "lm_head", block: 32),
        ],
        quantization: .init(groupSize: 32, bits: 2))

    /// Packed arrays for one rotated module, keyed the way a checkpoint stores them.
    private static func packedWeights(
        path: String, rows: Int, columns: Int, phase: Float
    ) -> [String: MLXArray] {
        let (weight, scales, biases) = quantized(
            values([rows, columns], phase: phase).asType(.float16), groupSize: 32, bits: 2)
        return [
            "\(path).weight": weight, "\(path).scales": scales, "\(path).biases": biases!,
            "\(path).signs": signs(columns),
        ]
    }

    func testSubstitutionReplacesManifestModulesOnly() throws {
        let wrapper = TinyWrapper()
        try substituteHadamardQuantizedModules(
            in: wrapper, manifest: Self.tinyManifest, pathPrefix: "language_model.")

        let inner = wrapper.languageModel.model
        XCTAssertTrue(inner.embedTokens is HadamardQuantizedEmbedding)
        XCTAssertTrue(inner.layers[0].gate is HadamardQuantizedLinear)
        XCTAssertTrue(wrapper.languageModel.lmHead is HadamardQuantizedLinear)
        XCTAssertTrue(type(of: inner.layers[0].up) == Linear.self)

        let keys = Set(wrapper.parameters().flattened().map(\.0))
        XCTAssertTrue(keys.contains("language_model.model.layers.0.gate_proj.signs"))
        XCTAssertTrue(keys.contains("language_model.model.embed_tokens.scales"))
        XCTAssertFalse(keys.contains("language_model.model.layers.0.up_proj.signs"))

        let gate = try XCTUnwrap(inner.layers[0].gate as? HadamardQuantizedLinear)
        XCTAssertEqual(gate.weight.shape, [32, 4])
        XCTAssertEqual(gate.scales.shape, [32, 2])
        XCTAssertEqual(gate.signs.shape, [64])
        XCTAssertEqual(gate.rotation, SignedBlockHadamard(blockSize: 32))
    }

    func testSubstitutedModulesSurviveTheLoaderQuantizePassAndLoadWeights() throws {
        let wrapper = TinyWrapper()
        try substituteHadamardQuantizedModules(
            in: wrapper, manifest: Self.tinyManifest, pathPrefix: "language_model.")
        let gateBefore = wrapper.languageModel.model.layers[0].gate

        var weights = [
            Self.packedWeights(
                path: "language_model.model.embed_tokens", rows: 16, columns: 64, phase: 0),
            Self.packedWeights(
                path: "language_model.model.layers.0.gate_proj", rows: 32, columns: 64, phase: 1),
            Self.packedWeights(path: "language_model.lm_head", rows: 16, columns: 64, phase: 2),
        ].reduce(into: [String: MLXArray]()) { $0.merge($1) { $1 } }
        // An ordinary quantized module beside the rotated ones, as the vision tower or
        // an unrotated projection would be.
        let (upWeight, upScales, upBiases) = quantized(
            Self.values([32, 64], phase: 3).asType(.float16), groupSize: 32, bits: 2)
        weights["language_model.model.layers.0.up_proj.weight"] = upWeight
        weights["language_model.model.layers.0.up_proj.scales"] = upScales
        weights["language_model.model.layers.0.up_proj.biases"] = upBiases

        // The loader's pass: quantize whatever ships scales, skipping what is already quantized.
        quantize(model: wrapper) { path, _ in
            weights["\(path).scales"] != nil ? (groupSize: 32, bits: 2, mode: .affine) : nil
        }
        XCTAssertTrue(wrapper.languageModel.model.layers[0].gate === gateBefore)
        XCTAssertTrue(type(of: wrapper.languageModel.model.layers[0].up) == QuantizedLinear.self)

        try wrapper.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])

        let gate = try XCTUnwrap(
            wrapper.languageModel.model.layers[0].gate as? HadamardQuantizedLinear)
        assertClose(gate.signs, Self.signs(64))
        let x = Self.values([1, 64], phase: 4).asType(.float16)
        let expected = quantizedMM(
            gate.rotation.forward(x, signs: gate.signs), gate.weight, scales: gate.scales,
            biases: gate.biases, transpose: true, groupSize: 32, bits: 2)
        assertClose(gate(x), expected, rtol: 1e-2, atol: 1e-2)

        let ids = MLXArray([1, 2, 3], [1, 3])
        XCTAssertEqual(wrapper.languageModel.model.embedTokens(ids).shape, [1, 3, 64])
    }

    func testSubstitutionRejectsUnknownAndUnsubstitutablePaths() {
        let missing = HadamardQuantizedManifest(
            modules: [.init(path: "model.layers.9.gate_proj", block: 32)],
            quantization: .init(groupSize: 32, bits: 2))
        assertThrows(.moduleNotFound("language_model.model.layers.9.gate_proj")) {
            try substituteHadamardQuantizedModules(
                in: TinyWrapper(), manifest: missing, pathPrefix: "language_model.")
        }

        let wrongKind = HadamardQuantizedManifest(
            modules: [.init(path: "model.embed_tokens", block: 32, embedding: false)],
            quantization: .init(groupSize: 32, bits: 2))
        assertThrows(
            .moduleNotSubstitutable("language_model.model.embed_tokens", found: "Embedding")
        ) {
            try substituteHadamardQuantizedModules(
                in: TinyWrapper(), manifest: wrongKind, pathPrefix: "language_model.")
        }
    }

    // MARK: - Manifest

    private static let packConfig = """
        {
            "model_type": "prism_hadamard_qwen35",
            "base_model_type": "qwen3_5",
            "schema_version": 2,
            "tensor_namespace": "mlx-vlm-qwen3_5",
            "gdn_activation_layout": "grouped",
            "quantization": {"bits": 2, "group_size": 32, "mode": "affine"},
            "modules": [
                {"path": "model.embed_tokens", "block": 32, "embedding": true, "dtype": "float16"},
                {"path": "lm_head", "block": 32, "embedding": false, "dtype": "float16"},
                {"path": "model.layers.0.mlp.gate_proj", "block": 32, "embedding": false},
                {"path": "model.layers.0.mlp.up_proj", "block": 32, "embedding": false},
                {"path": "model.layers.0.mlp.down_proj", "block": 32, "embedding": false},
                {"path": "model.layers.0.linear_attn.in_proj_qkv", "block": 32},
                {"path": "model.layers.0.linear_attn.in_proj_z", "block": 32},
                {"path": "model.layers.0.linear_attn.out_proj", "block": 32},
                {"path": "model.layers.1.self_attn.q_proj", "block": 32},
                {"path": "model.layers.1.self_attn.k_proj", "block": 32},
                {"path": "model.layers.1.self_attn.v_proj", "block": 32},
                {"path": "model.layers.1.self_attn.o_proj", "block": 32}
            ],
            "image_token_id": 500,
            "video_token_id": 501,
            "vision_start_token_id": 502,
            "vision_end_token_id": 503,
            "vocab_size": 512,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 64,
                "num_hidden_layers": 4,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "vocab_size": 512,
                "full_attention_interval": 2,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4,
                "max_position_embeddings": 4096,
                "tie_word_embeddings": false,
                "rope_parameters": {
                    "type": "default",
                    "mrope_section": [8, 4, 4],
                    "rope_theta": 100000.0,
                    "partial_rotary_factor": 1.0
                }
            },
            "vision_config": {
                "model_type": "qwen3_vl",
                "depth": 2,
                "hidden_size": 32,
                "intermediate_size": 64,
                "out_hidden_size": 64,
                "num_heads": 2,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "num_position_embeddings": 64
            }
        }
        """

    func testManifestDecodesThePackConfig() throws {
        let manifest = try JSONDecoder().decode(
            HadamardQuantizedManifest.self, from: Data(Self.packConfig.utf8))
        XCTAssertEqual(try manifest.activationDType(), .float16)
        XCTAssertEqual(manifest.modules[0].dtype, "float16")
        XCTAssertNil(manifest.modules[2].dtype)
        XCTAssertEqual(manifest.schemaVersion, 2)
        XCTAssertEqual(manifest.baseModelType, "qwen3_5")
        XCTAssertEqual(manifest.tensorNamespace, "mlx-vlm-qwen3_5")
        XCTAssertEqual(manifest.gdnActivationLayout, "grouped")
        XCTAssertEqual(manifest.quantization.bits, 2)
        XCTAssertEqual(manifest.quantization.groupSize, 32)
        XCTAssertEqual(manifest.quantization.mode, .affine)
        XCTAssertEqual(manifest.modules.count, 12)
        XCTAssertEqual(
            manifest.modules[0],
            .init(path: "model.embed_tokens", block: 32, embedding: true, dtype: "float16"))
        XCTAssertEqual(manifest.modules[5].embedding, false)
        XCTAssertNoThrow(try Self.validateForQwen35(manifest))
    }

    func testManifestValidationRejectsWhatTheLoaderCannotHonor() throws {
        let base = try JSONDecoder().decode(
            HadamardQuantizedManifest.self, from: Data(Self.packConfig.utf8))

        var schema = base
        schema.schemaVersion = 1
        assertThrows(.unsupportedSchemaVersion(1)) { try Self.validateForQwen35(schema) }

        assertThrows(.unsupportedBaseModelType("qwen3_5", expected: "llama")) {
            try base.validate(baseModelType: "llama", tensorNamespace: "mlx-vlm-qwen3_5")
        }
        assertThrows(.unsupportedTensorNamespace("mlx-vlm-qwen3_5", expected: "mlx-lm-qwen3_5")) {
            try base.validate(baseModelType: "qwen3_5", tensorNamespace: "mlx-lm-qwen3_5")
        }

        var layout = base
        layout.gdnActivationLayout = "tiled"
        assertThrows(.unsupportedActivationLayout("tiled")) { try Self.validateForQwen35(layout) }

        var empty = base
        empty.modules = []
        assertThrows(.emptyManifest) { try Self.validateForQwen35(empty) }

        var mixed = base
        mixed.modules[2].dtype = "bfloat16"
        assertThrows(.mixedActivationDTypes(["bfloat16", "float16"])) {
            try Self.validateForQwen35(mixed)
        }

        var wide = base
        for i in wide.modules.indices { wide.modules[i].dtype = "float32" }
        assertThrows(.unsupportedActivationDType("float32")) { try Self.validateForQwen35(wide) }
    }

    // MARK: - Factories

    private func rotatedModulePaths(in model: Module) -> Set<String> {
        Set(
            model.leafModules().flattened().compactMap { path, module in
                module is HadamardQuantizedLinear || module is HadamardQuantizedEmbedding
                    ? path : nil
            })
    }

    private static let expectedRotatedPaths: Set<String> = [
        "language_model.model.embed_tokens",
        "language_model.lm_head",
        "language_model.model.layers.0.mlp.gate_proj",
        "language_model.model.layers.0.mlp.up_proj",
        "language_model.model.layers.0.mlp.down_proj",
        "language_model.model.layers.0.linear_attn.in_proj_qkv",
        "language_model.model.layers.0.linear_attn.in_proj_z",
        "language_model.model.layers.0.linear_attn.out_proj",
        "language_model.model.layers.1.self_attn.q_proj",
        "language_model.model.layers.1.self_attn.k_proj",
        "language_model.model.layers.1.self_attn.v_proj",
        "language_model.model.layers.1.self_attn.o_proj",
    ]

    /// Both classes substitute the same manifest modules and cast the same unpacked
    /// tensors; `visionTower` is false for the text class, whose sanitize drops the tower.
    private func assertRotatedQwen35(_ model: Module & LanguageModel, visionTower: Bool) {
        XCTAssertEqual(rotatedModulePaths(in: model), Self.expectedRotatedPaths)
        let paths = Set(model.leafModules().flattened().map(\.0))
        // Layer 0 is gated-delta and layer 1 full attention; both keep their unrotated parts.
        XCTAssertTrue(paths.contains("language_model.model.layers.0.linear_attn.in_proj_a"))
        XCTAssertTrue(paths.contains("language_model.model.layers.1.self_attn.q_norm"))
        XCTAssertEqual(paths.contains { $0.hasPrefix("vision_tower.") }, visionTower)
        // The pack stores its norms in float32; sanitize brings them to the activation dtype.
        let sanitized = model.sanitize(
            weights: Self.unpackedWeightsFixture(), metadata: ["format": "mlx"])
        Self.assertUnpackedWeightsCast(sanitized, visionTower: visionTower)
    }

    func testLLMRegistryBuildsTheQwen35TextClassWithRotatedModules() async throws {
        let model = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(Self.packConfig.utf8), modelType: "prism_hadamard_qwen35")
        XCTAssertTrue(model is Qwen35Model)
        XCTAssertTrue(model is PrismHadamardQwen35Model)
        assertRotatedQwen35(model, visionTower: false)
    }

    func testVLMRegistryBuildsTheQwen35VisionClassWithRotatedModules() async throws {
        let model = try await VLMTypeRegistry.shared.createModel(
            configuration: Data(Self.packConfig.utf8), modelType: "prism_hadamard_qwen35")
        XCTAssertTrue(model is Qwen35)
        XCTAssertTrue(model is PrismHadamardQwen35)
        assertRotatedQwen35(model, visionTower: true)
    }

    /// What a pack stores beside its packed modules: float32 norms, taps and the small
    /// gated-delta projections, float32 `A_log` and `signs`, float16 scales, a packed
    /// uint32 weight, and a float16 vision tower.
    private static func unpackedWeightsFixture() -> [String: MLXArray] {
        [
            "language_model.model.norm.weight": MLXArray.ones([64], dtype: .float32),
            "language_model.model.layers.0.input_layernorm.weight": MLXArray.ones(
                [64], dtype: .float32),
            "language_model.model.layers.0.linear_attn.conv1d.weight": MLXArray.ones(
                [96, 4, 1], dtype: .float32),
            "language_model.model.layers.0.linear_attn.in_proj_a.weight": MLXArray.ones(
                [2, 64], dtype: .float32),
            "language_model.model.layers.0.linear_attn.A_log": MLXArray.ones([2], dtype: .float32),
            "language_model.model.layers.0.linear_attn.dt_bias": MLXArray.ones(
                [2], dtype: .float32),
            "language_model.model.layers.0.mlp.gate_proj.signs": MLXArray.ones(
                [64], dtype: .float32),
            "language_model.model.layers.0.mlp.gate_proj.scales": MLXArray.ones(
                [32, 2], dtype: .float16),
            "language_model.model.layers.0.mlp.gate_proj.weight": MLXArray.zeros(
                [32, 4], dtype: .uint32),
            "vision_tower.blocks.0.norm1.weight": MLXArray.ones([8], dtype: .float16),
        ]
    }

    private static func assertUnpackedWeightsCast(
        _ weights: [String: MLXArray], visionTower: Bool
    ) {
        func dtype(_ key: String) -> DType? { weights[key]?.dtype }
        XCTAssertEqual(dtype("language_model.model.norm.weight"), .float16)
        XCTAssertEqual(dtype("language_model.model.layers.0.input_layernorm.weight"), .float16)
        XCTAssertEqual(dtype("language_model.model.layers.0.linear_attn.conv1d.weight"), .float16)
        XCTAssertEqual(
            dtype("language_model.model.layers.0.linear_attn.in_proj_a.weight"), .float16)
        XCTAssertEqual(dtype("language_model.model.layers.0.linear_attn.dt_bias"), .float16)
        XCTAssertEqual(dtype("language_model.model.layers.0.linear_attn.A_log"), .float32)
        XCTAssertEqual(dtype("language_model.model.layers.0.mlp.gate_proj.signs"), .float32)
        XCTAssertEqual(dtype("language_model.model.layers.0.mlp.gate_proj.scales"), .float16)
        XCTAssertEqual(dtype("language_model.model.layers.0.mlp.gate_proj.weight"), .uint32)
        XCTAssertEqual(dtype("vision_tower.blocks.0.norm1.weight"), visionTower ? .float16 : nil)
        XCTAssertEqual(weights.count, unpackedWeightsFixture().count - (visionTower ? 0 : 1))
    }

    func testRegistriesRejectAnUnsupportedSchema() async throws {
        let config = Self.packConfig.replacingOccurrences(
            of: "\"schema_version\": 2", with: "\"schema_version\": 1")
        for registry in [LLMTypeRegistry.shared, VLMTypeRegistry.shared] {
            do {
                _ = try await registry.createModel(
                    configuration: Data(config.utf8), modelType: "prism_hadamard_qwen35")
                XCTFail("schema 1 should not load")
            } catch {
                XCTAssertEqual(
                    error as? HadamardQuantizedCheckpointError, .unsupportedSchemaVersion(1))
            }
        }
    }

    // MARK: - Same-input stacking

    private func assertBitIdentical(
        _ actual: MLXArray, _ expected: MLXArray, _ message: String = "",
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.dtype, expected.dtype, message, file: file, line: line)
        XCTAssertEqual(actual.shape, expected.shape, message, file: file, line: line)
        let mismatches = zip(
            actual.asType(.float32).asArray(Float.self),
            expected.asType(.float32).asArray(Float.self)
        ).lazy.filter { $0.bitPattern != $1.bitPattern }.count
        XCTAssertEqual(
            mismatches, 0, "\(message): \(mismatches) values differ", file: file, line: line)
    }

    /// One module of a rotated fixture: its path, output rows, block and signs.
    private typealias RotatedModule = (path: String, rows: Int, block: Int, signs: MLXArray)

    /// Replaces `modules` of `module` with rotated placeholders and loads packed
    /// weights over `inputs` columns into them: the loader's sequence.
    private static func loadRotated(
        _ module: Module, modules: [RotatedModule], inputs: Int
    ) throws {
        let manifest = HadamardQuantizedManifest(
            modules: modules.map { .init(path: $0.path, block: $0.block) },
            quantization: .init(groupSize: 32, bits: 2))
        try substituteHadamardQuantizedModules(in: module, manifest: manifest)
        var weights = [String: MLXArray]()
        for (index, entry) in modules.enumerated() {
            weights.merge(
                packedWeights(
                    path: entry.path, rows: entry.rows, columns: inputs, phase: Float(index))
            ) { $1 }
            weights["\(entry.path).signs"] = entry.signs
        }
        try module.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: [.noUnusedKeys, .shapeMismatch])
    }

    func testStackingFoldsRotatedSiblingsThatShareTheirSigns() throws {
        let mlp = Qwen3NextMLP(dimensions: 64, hiddenDimensions: 96)
        let signs = Self.signs(64)
        try Self.loadRotated(
            mlp, modules: [("gate_proj", 96, 32, signs), ("up_proj", 96, 32, signs)], inputs: 64)
        let x = Self.values([1, 3, 64], phase: 4).asType(.float16)
        let before = mlp(x)
        eval(before)

        XCTAssertEqual(stackSameInputProjections(in: mlp), 1)

        let stacked = try XCTUnwrap(mlp.gateUp as? HadamardQuantizedLinear)
        XCTAssertEqual(stacked.rotation, SignedBlockHadamard(blockSize: 32))
        XCTAssertTrue(arrayEqual(stacked.signs, signs).item(Bool.self))
        XCTAssertEqual(stacked.shape.0, 192)
        XCTAssertEqual(stacked.shape.1, 64)
        XCTAssertEqual(stacked.bits, 2)
        XCTAssertEqual(mlp.gateDimensions, 96)
        XCTAssertTrue(type(of: mlp.gateProj) == Linear.self, "the originals are released")
        let after = mlp(x)
        eval(after)
        assertBitIdentical(after, before, "stacked MLP")
        XCTAssertEqual(stackSameInputProjections(in: mlp), 0, "stacks once")
    }

    func testStackingLeavesRotatedSiblingsWithDifferentSignsOrBlocksApart() throws {
        let signs = Self.signs(64)
        var flipped = signs.asArray(Float.self)
        flipped[5] = -flipped[5]

        let differentSigns = Qwen3NextMLP(dimensions: 64, hiddenDimensions: 96)
        try Self.loadRotated(
            differentSigns,
            modules: [("gate_proj", 96, 32, signs), ("up_proj", 96, 32, MLXArray(flipped))],
            inputs: 64)
        let differentBlocks = Qwen3NextMLP(dimensions: 64, hiddenDimensions: 96)
        try Self.loadRotated(
            differentBlocks, modules: [("gate_proj", 96, 32, signs), ("up_proj", 96, 64, signs)],
            inputs: 64)

        for (mlp, label) in [(differentSigns, "signs"), (differentBlocks, "blocks")] {
            let x = Self.values([2, 64], phase: 4).asType(.float16)
            let before = mlp(x)
            eval(before)
            XCTAssertEqual(stackSameInputProjections(in: mlp), 0, label)
            XCTAssertNil(mlp.gateUp, label)
            XCTAssertTrue(mlp.gateProj is HadamardQuantizedLinear, label)
            XCTAssertTrue(mlp.upProj is HadamardQuantizedLinear, label)
            let after = mlp(x)
            eval(after)
            assertBitIdentical(after, before, label)
        }

        // A rotated layer beside a plain quantized one shares no rotation to fold.
        let mixed = Qwen3NextMLP(dimensions: 64, hiddenDimensions: 96)
        try Self.loadRotated(mixed, modules: [("gate_proj", 96, 32, signs)], inputs: 64)
        quantize(model: mixed, groupSize: 32, bits: 2)
        XCTAssertTrue(type(of: mixed.upProj) == QuantizedLinear.self)
        XCTAssertEqual(stackSameInputProjections(in: mixed), 0)
        XCTAssertNil(mixed.gateUp)
    }

    /// A rotated layer with random packed arrays at a checkpoint's shape.
    private static func randomRotatedLinear(
        inputs: Int, outputs: Int, signs: MLXArray, rotation: SignedBlockHadamard
    ) -> HadamardQuantizedLinear {
        let (groupSize, bits) = (128, 2)
        return HadamardQuantizedLinear(
            weight: MLXRandom.randInt(0 ..< Int32.max, [outputs, inputs * bits / 32])
                .asType(.uint32),
            scales: MLXRandom.uniform(low: 0.01, high: 0.1, [outputs, inputs / groupSize])
                .asType(.float16),
            biases: MLXRandom.uniform(low: -0.1, high: 0.1, [outputs, inputs / groupSize])
                .asType(.float16),
            signs: signs, groupSize: groupSize, bits: bits, rotation: rotation)
    }

    /// The stacked layer is bitwise the concatenation of its parts at Bonsai 2
    /// 27B's projection shapes (q|k|v, GDN qkv|z, MLP gate|up over a 5120-wide
    /// residual), for a decode row and a prefill block, which the kernels route
    /// differently.
    func testStackedRotatedProjectionMatchesItsPartsAtTheCheckpointShapes() throws {
        let inputs = 5120
        let rotation = SignedBlockHadamard(blockSize: 1024)
        let signs = TurboQuantRotation.whtSigns(dim: inputs, seed: 11)
        MLXRandom.seed(3)
        for outputs in [[12288, 1024, 1024], [10240, 6144], [17408, 17408]] {
            let parts = outputs.map {
                Self.randomRotatedLinear(
                    inputs: inputs, outputs: $0, signs: signs, rotation: rotation)
            }
            let stacked = try XCTUnwrap(stackedSameInputProjection(parts))
            XCTAssertTrue(type(of: stacked) == HadamardQuantizedLinear.self)
            XCTAssertEqual(stacked.shape.0, outputs.reduce(0, +))
            for rows in [1, 13] {
                let x = MLXRandom.normal([1, rows, inputs]).asType(.float16)
                let expected = concatenated(parts.map { $0(x) }, axis: -1)
                let actual = stacked(x)
                eval(expected, actual)
                assertBitIdentical(actual, expected, "\(outputs) rows \(rows)")
            }
        }
    }

    private static let gdnConfiguration = """
        {
            "model_type": "qwen3_5_text",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 32,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "vocab_size": 32,
            "full_attention_interval": 2,
            "num_experts": 0,
            "num_experts_per_tok": 0
        }
        """

    private static func gatedDeltaNet() throws -> Qwen35GatedDeltaNet {
        Qwen35GatedDeltaNet(
            try JSONDecoder().decode(
                Qwen35TextConfiguration.self, from: Data(gdnConfiguration.utf8)))
    }

    /// A rotated pack stores `in_proj_b` / `in_proj_a` unquantized, so the
    /// four-projection fusion cannot apply; `qkv|z` still stack.
    func testGatedDeltaNetStacksRotatedQKVAndZWhenTheFusedRowCannotApply() throws {
        let layer = try Self.gatedDeltaNet()
        let signs = Self.signs(64)
        try Self.loadRotated(
            layer, modules: [("in_proj_qkv", 256, 32, signs), ("in_proj_z", 128, 32, signs)],
            inputs: 64)
        XCTAssertFalse(try layer.prepareFusedInputProjection())
        let x = MLXRandom.normal([1, 5, 64]).asType(.float16)
        let before = layer(x)
        eval(before)

        XCTAssertEqual(stackSameInputProjections(in: layer), 1)

        let stacked = try XCTUnwrap(layer.qkvzStacked as? HadamardQuantizedLinear)
        XCTAssertEqual(stacked.shape.0, 384)
        XCTAssertTrue(type(of: layer.inProjQKV) == Linear.self)
        XCTAssertTrue(type(of: layer.inProjB) == Linear.self, "the gates stay standalone")
        let after = layer(x)
        eval(after)
        assertBitIdentical(after, before, "GDN")
        XCTAssertEqual(stackSameInputProjections(in: layer), 0)
    }

    func testGatedDeltaNetKeepsTheFusedRowInsteadOfStacking() throws {
        let layer = try Self.gatedDeltaNet()
        quantize(model: layer, groupSize: 32, bits: 4)
        XCTAssertTrue(try layer.prepareFusedInputProjection())
        XCTAssertEqual(stackSameInputProjections(in: layer), 0)
        XCTAssertNil(layer.qkvzStacked)
        XCTAssertTrue(layer.hasFusedInputProjection)
    }
}
