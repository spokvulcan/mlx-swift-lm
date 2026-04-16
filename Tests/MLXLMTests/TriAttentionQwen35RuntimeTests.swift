import MLX
import MLXLLM
import XCTest
@testable import MLXLMCommon

final class TriAttentionQwen35RuntimeTests: XCTestCase {

    func testQwen35NewCacheSelectsSparseCachesOnlyForCompatibleTriAttentionRuntime() throws {
        let model = Qwen35TextModel(try makeConfiguration())
        let fullAttentionIndices = [1, 3]
        let linearIndices = [0, 2]

        let denseCache = model.newCache(parameters: nil)
        for index in fullAttentionIndices {
            XCTAssertTrue(denseCache[index] is KVCacheSimple)
        }
        for index in linearIndices {
            XCTAssertTrue(denseCache[index] is MambaCache)
        }

        let compatibleArtifact = makeArtifact(headDim: 8)
        let sparseParameters = GenerateParameters(
            triAttention: TriAttentionConfiguration(
                enabled: true,
                budgetTokens: 12_000,
                calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(
                    rawValue: "triattention-compatible"
                ),
                implementationVersion: .v1
            ),
            triAttentionCalibrationArtifact: compatibleArtifact
        )
        let sparseCache = model.newCache(parameters: sparseParameters)
        for index in fullAttentionIndices {
            XCTAssertTrue(sparseCache[index] is TriAttentionSparseKVCache)
        }
        for index in linearIndices {
            XCTAssertTrue(sparseCache[index] is MambaCache)
        }

        let incompatibleParameters = GenerateParameters(
            triAttention: TriAttentionConfiguration(
                enabled: true,
                budgetTokens: 12_000,
                calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(
                    rawValue: "triattention-incompatible"
                ),
                implementationVersion: .v1
            ),
            triAttentionCalibrationArtifact: makeArtifact(headDim: 6)
        )
        let incompatibleCache = model.newCache(parameters: incompatibleParameters)
        for index in fullAttentionIndices {
            XCTAssertTrue(incompatibleCache[index] is KVCacheSimple)
        }

        let missingLayerParameters = GenerateParameters(
            triAttention: TriAttentionConfiguration(
                enabled: true,
                budgetTokens: 12_000,
                calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(
                    rawValue: "triattention-missing-layer"
                ),
                implementationVersion: .v1
            ),
            triAttentionCalibrationArtifact: makeArtifact(
                headDim: 8,
                sampledHeads: [
                    TriAttentionCalibrationHeadKey(layerIndex: 1, headIndex: 0),
                    TriAttentionCalibrationHeadKey(layerIndex: 1, headIndex: 1),
                    TriAttentionCalibrationHeadKey(layerIndex: 1, headIndex: 2),
                    TriAttentionCalibrationHeadKey(layerIndex: 1, headIndex: 3),
                ]
            )
        )
        let missingLayerCache = model.newCache(parameters: missingLayerParameters)
        for index in fullAttentionIndices {
            XCTAssertTrue(missingLayerCache[index] is KVCacheSimple)
        }
    }

    func testDenseAndTriAttentionOutputsMatchBeforeFirstPrune() throws {
        let model = Qwen35TextModel(try makeConfiguration(fullAttentionInterval: 1))
        let tokens = MLXArray(Array(0 ..< 12).map(Int32.init)).reshaped(1, 12)

        var triConfiguration = TriAttentionConfiguration(
            enabled: true,
            budgetTokens: 256,
            calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(rawValue: "artifact"),
            implementationVersion: .v1
        )
        triConfiguration = triConfiguration.withCalibrationArtifactIdentity(
            TriAttentionCalibrationArtifactIdentity(rawValue: "artifact")
        )

        let denseOutput = model(tokens, cache: model.newCache(parameters: nil))
        let triOutput = model(
            tokens,
            cache: model.newCache(parameters: GenerateParameters(
                triAttention: triConfiguration,
                triAttentionCalibrationArtifact: makeArtifact(
                    headDim: 8,
                    sampledHeads: sampledHeads(forLayers: [0, 1, 2, 3])
                )
            ))
        )

        eval(denseOutput)
        eval(triOutput)
        XCTAssertEqual(denseOutput.shape, triOutput.shape)
        XCTAssertTrue(allClose(denseOutput, triOutput).item(Bool.self))
    }

    func testTriAttentionPrunesAtBudgetPlusDivideLengthAndSharesKeepPlanAcrossLayers() throws {
        let model = Qwen35TextModel(try makeConfiguration(fullAttentionInterval: 1))
        let parameters = GenerateParameters(
            triAttention: TriAttentionConfiguration(
                enabled: true,
                budgetTokens: 8,
                calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(rawValue: "artifact"),
                implementationVersion: .v1
            ),
            triAttentionCalibrationArtifact: makeArtifact(
                headDim: 8,
                sampledHeads: sampledHeads(forLayers: [0, 1, 2, 3])
            )
        )
        let cache = model.newCache(parameters: parameters)

        let prompt = MLXArray(Array(0 ..< 135).map(Int32.init)).reshaped(1, 135)
        _ = model(prompt, cache: cache)

        let firstAttentionCache = try unwrapSparseCache(cache[0])
        let secondAttentionCache = try unwrapSparseCache(cache[1])
        eval(firstAttentionCache)
        eval(secondAttentionCache)
        XCTAssertEqual(firstAttentionCache.retainedPositions?.dim(2), 135)
        XCTAssertEqual(secondAttentionCache.retainedPositions?.dim(2), 135)

        let nextToken = MLXArray([Int32(135)]).reshaped(1, 1)
        _ = model(nextToken, cache: cache)

        eval(firstAttentionCache)
        eval(secondAttentionCache)
        XCTAssertEqual(firstAttentionCache.retainedPositions?.dim(2), 8)
        XCTAssertEqual(secondAttentionCache.retainedPositions?.dim(2), 8)
        XCTAssertAllClose(firstAttentionCache.retainedPositions, secondAttentionCache.retainedPositions)
        XCTAssertEqual(firstAttentionCache.retainedKeys?.shape, secondAttentionCache.retainedKeys?.shape)
        XCTAssertEqual(firstAttentionCache.retainedValues?.shape, secondAttentionCache.retainedValues?.shape)

        let head0Positions = firstAttentionCache.retainedPositions?[0, 0].asArray(Int32.self)
        let head1Positions = firstAttentionCache.retainedPositions?[0, 1].asArray(Int32.self)
        XCTAssertTrue(Set(head0Positions ?? []).isSuperset(of: [129, 130, 131, 132, 133, 134, 135]))
        XCTAssertTrue(Set(head1Positions ?? []).isSuperset(of: [129, 130, 131, 132, 133, 134, 135]))

        guard let copied = firstAttentionCache.copy() as? TriAttentionSparseKVCache else {
            return XCTFail("Expected sparse cache copy after pruning")
        }
        XCTAssertEqual(copied.offset, firstAttentionCache.offset)
        XCTAssertEqual(copied.configuration, firstAttentionCache.configuration)
        XCTAssertAllClose(copied.retainedPositions, firstAttentionCache.retainedPositions)
        XCTAssertAllClose(copied.retainedKeys, firstAttentionCache.retainedKeys)
        XCTAssertAllClose(copied.retainedValues, firstAttentionCache.retainedValues)
    }

    func testTriAttentionPrunesSharedInterleavedFullAttentionLayerGroup() throws {
        let model = Qwen35TextModel(try makeConfiguration(fullAttentionInterval: 2))
        let parameters = GenerateParameters(
            triAttention: TriAttentionConfiguration(
                enabled: true,
                budgetTokens: 8,
                calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(rawValue: "artifact"),
                implementationVersion: .v1
            ),
            triAttentionCalibrationArtifact: makeArtifact(
                headDim: 8,
                sampledHeads: sampledHeads(forLayers: [1, 3])
            )
        )
        let cache = model.newCache(parameters: parameters)

        XCTAssertTrue(cache[0] is MambaCache)
        XCTAssertTrue(cache[2] is MambaCache)
        XCTAssertTrue(cache[1] is TriAttentionSparseKVCache)
        XCTAssertTrue(cache[3] is TriAttentionSparseKVCache)

        let firstAttentionCache = try unwrapSparseCache(cache[1])
        let secondAttentionCache = try unwrapSparseCache(cache[3])
        let keys = MLXArray(Array(repeating: Float(1), count: 1 * 2 * 136 * 8)).reshaped(1, 2, 136, 8)
        let values = MLXArray(Array(repeating: Float(2), count: 1 * 2 * 136 * 8)).reshaped(1, 2, 136, 8)
        _ = firstAttentionCache.update(keys: keys, values: values)
        _ = secondAttentionCache.update(keys: keys, values: values)

        TriAttentionQwen35Runtime.pruneIfNeeded(
            caches: [firstAttentionCache, secondAttentionCache],
            layerIndices: [1, 3]
        )

        eval(firstAttentionCache)
        eval(secondAttentionCache)
        XCTAssertEqual(firstAttentionCache.retainedPositions?.dim(2), 8)
        XCTAssertEqual(secondAttentionCache.retainedPositions?.dim(2), 8)
        XCTAssertAllClose(firstAttentionCache.retainedPositions, secondAttentionCache.retainedPositions)
    }

    func testTriAttentionProtectsFull128TokenWindowWhenBudgetExceedsWindowClamp() throws {
        let model = Qwen35TextModel(try makeConfiguration(fullAttentionInterval: 1))
        let parameters = GenerateParameters(
            triAttention: TriAttentionConfiguration(
                enabled: true,
                budgetTokens: 256,
                calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(rawValue: "artifact"),
                implementationVersion: .v1
            ),
            triAttentionCalibrationArtifact: makeArtifact(
                headDim: 8,
                sampledHeads: sampledHeads(forLayers: [0, 1, 2, 3])
            )
        )
        let cache = model.newCache(parameters: parameters)

        let prompt = MLXArray(Array(0 ..< 383).map(Int32.init)).reshaped(1, 383)
        _ = model(prompt, cache: cache)

        let nextToken = MLXArray([Int32(383)]).reshaped(1, 1)
        _ = model(nextToken, cache: cache)

        let attentionCache = try unwrapSparseCache(cache[0])
        eval(attentionCache)
        XCTAssertEqual(attentionCache.retainedPositions?.dim(2), 256)

        let protectedWindow = Set(Int32(256)...Int32(383))
        let head0Positions = Set(attentionCache.retainedPositions?[0, 0].asArray(Int32.self) ?? [])
        let head1Positions = Set(attentionCache.retainedPositions?[0, 1].asArray(Int32.self) ?? [])
        XCTAssertTrue(head0Positions.isSuperset(of: protectedWindow))
        XCTAssertTrue(head1Positions.isSuperset(of: protectedWindow))
    }

    func testTriAttentionAggregatesGroupedHeadsPerKVHeadUsingMaxReduction() {
        let configuration = TriAttentionConfiguration(
            enabled: true,
            budgetTokens: 8,
            calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(rawValue: "artifact"),
            implementationVersion: .v1
        )
        let runtimeState = tryMakeRuntimeState(
            fullAttentionLayerIndices: [1],
            configuration: configuration
        )
        let sampledHeads = sampledHeads(forLayers: [1])
        let normalizedScores = MLXArray([
            10 as Float, 0, 0,
            0, 9, 0,
            0, 0, 8,
            0, 0, 7,
        ]).reshaped(4, 3)

        let perKVHeadScores = TriAttentionQwen35Runtime.aggregatePerKVHeadScores(
            normalizedScores: normalizedScores,
            sampledHeads: sampledHeads,
            runtimeState: runtimeState
        )
        eval(perKVHeadScores)

        XCTAssertEqual(perKVHeadScores.shape, [2, 3])
        XCTAssertEqual(perKVHeadScores[0].asArray(Float.self), [10, 9, 0])
        XCTAssertEqual(perKVHeadScores[1].asArray(Float.self), [0, 0, 8])
    }

    private func makeConfiguration(fullAttentionInterval: Int = 2) throws -> Qwen35TextConfiguration {
        let json = """
        {
          "model_type": "qwen35",
          "hidden_size": 32,
          "num_hidden_layers": 4,
          "intermediate_size": 64,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "linear_num_value_heads": 4,
          "linear_num_key_heads": 2,
          "linear_key_head_dim": 4,
          "linear_value_head_dim": 4,
          "linear_conv_kernel_dim": 4,
          "rms_norm_eps": 1e-6,
          "vocab_size": 64,
          "rope_theta": 100000.0,
          "partial_rotary_factor": 1.0,
          "max_position_embeddings": 2048,
          "tie_word_embeddings": false,
          "attention_bias": false,
          "head_dim": 8,
          "rope_scaling": {
            "type": "default",
            "rope_theta": 100000.0,
            "partial_rotary_factor": 1.0
          },
          "full_attention_interval": \(fullAttentionInterval),
          "num_experts": 0,
          "num_experts_per_tok": 0,
          "decoder_sparse_step": 1,
          "shared_expert_intermediate_size": 0,
          "moe_intermediate_size": 0,
          "norm_topk_prob": true
        }
        """
        return try JSONDecoder().decode(Qwen35TextConfiguration.self, from: Data(json.utf8))
    }

    private func makeArtifact(
        headDim: Int,
        sampledHeads: [TriAttentionCalibrationHeadKey]? = nil
    ) -> TriAttentionCalibrationArtifact {
        let freqCount = headDim / 2
        let resolvedSampledHeads = sampledHeads ?? self.sampledHeads(forLayers: [1, 3])
        let stats = Dictionary(uniqueKeysWithValues: resolvedSampledHeads.map { key in
            let base = Float(key.layerIndex * 10 + key.headIndex + 1)
            let values = (0 ..< freqCount).map { offset in base + Float(offset) * 0.25 }
            return (
                key,
                TriAttentionCalibrationHeadStats(
                    qMeanReal: values,
                    qMeanImag: values.map { $0 * 0.5 },
                    qAbsMean: values.map { $0 * 1.5 }
                )
            )
        })
        return TriAttentionCalibrationArtifact(
            metadata: TriAttentionCalibrationMetadata(
                sampledHeads: resolvedSampledHeads,
                headDim: headDim,
                ropeStyle: "half",
                modelName: "tests/qwen35-triattention"
            ),
            statsByHead: stats
        )
    }

    private func tryMakeRuntimeState(
        fullAttentionLayerIndices: [Int],
        configuration: TriAttentionConfiguration
    ) -> TriAttentionQwen35RuntimeState {
        guard let runtimeState = TriAttentionQwen35Runtime.makeState(
            configuration: configuration,
            artifact: makeArtifact(
                headDim: 8,
                sampledHeads: sampledHeads(forLayers: fullAttentionLayerIndices)
            ),
            fullAttentionLayerIndices: fullAttentionLayerIndices,
            attentionHeads: 4,
            kvHeads: 2,
            headDim: 8,
            partialRotaryFactor: 1.0,
            ropeTheta: 100_000,
            ropeType: "default"
        ) else {
            XCTFail("Expected compatible TriAttention runtime state")
            fatalError("Missing TriAttention runtime state for test")
        }
        return runtimeState
    }

    private func unwrapSparseCache(_ cache: KVCache) throws -> TriAttentionSparseKVCache {
        guard let sparseCache = cache as? TriAttentionSparseKVCache else {
            XCTFail("Expected TriAttentionSparseKVCache")
            throw NSError(domain: "TriAttentionQwen35RuntimeTests", code: 1)
        }
        return sparseCache
    }

    private func sampledHeads(forLayers layers: [Int]) -> [TriAttentionCalibrationHeadKey] {
        layers.flatMap { layerIndex in
            (0 ..< 4).map { headIndex in
                TriAttentionCalibrationHeadKey(layerIndex: layerIndex, headIndex: headIndex)
            }
        }
    }

    private func XCTAssertAllClose(
        _ lhs: MLXArray?,
        _ rhs: MLXArray?,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        guard let lhs, let rhs else {
            return XCTFail("Expected both arrays to be non-nil", file: file, line: line)
        }
        eval(lhs)
        eval(rhs)
        XCTAssertEqual(lhs.shape, rhs.shape, file: file, line: line)
        XCTAssertTrue(allClose(lhs, rhs).item(Bool.self), file: file, line: line)
    }
}
