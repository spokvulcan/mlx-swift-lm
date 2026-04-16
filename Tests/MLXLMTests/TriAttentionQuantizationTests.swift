import MLX
import MLXLMCommon
import MLXLLM
import XCTest

final class TriAttentionQuantizationTests: XCTestCase {
    private let headDim = 32

    func testMaybeQuantizeKVCacheConvertsTriAttentionSparseCacheAfterThreshold() {
        let configuration = TriAttentionConfiguration(
            enabled: true,
            budgetTokens: 16,
            calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(rawValue: "artifact"),
            implementationVersion: .v1
        )
        let runtimeState = tryMakeRuntimeState(fullAttentionLayerIndices: [1], configuration: configuration)
        let sparseCache = TriAttentionSparseKVCache(
            configuration: configuration,
            runtimeState: runtimeState
        )
        let keys = MLXArray(Array(repeating: Float(1), count: 1 * 2 * 2 * headDim)).reshaped(1, 2, 2, headDim)
        let values = MLXArray(Array(repeating: Float(2), count: 1 * 2 * 2 * headDim)).reshaped(1, 2, 2, headDim)
        let (_, _) = sparseCache.update(keys: keys, values: values)

        var cache: [KVCache] = [sparseCache]
        maybeQuantizeKVCache(cache: &cache, kvBits: 4, kvGroupSize: 32, quantizedKVStart: 1)

        guard let quantized = cache[0] as? QuantizedTriAttentionSparseKVCache else {
            return XCTFail("Expected TriAttention cache to switch to QuantizedTriAttentionSparseKVCache")
        }

        XCTAssertEqual(quantized.offset, 2)
        XCTAssertEqual(quantized.configuration, configuration)
        XCTAssertEqual(quantized.groupSize, 32)
        XCTAssertEqual(quantized.bits, 4)
        XCTAssertFalse(quantized.isTrimmable)
        XCTAssertEqual(quantized.retainedPositions?.asArray(Int32.self), [0, 1, 0, 1])
        XCTAssertNotNil(quantized.getQuantizedState())
    }

    func testMaybeQuantizeKVCacheLeavesTriAttentionSparseCacheDenseWithoutBits() {
        let sparseCache = TriAttentionSparseKVCache(configuration: .v1Disabled)
        let keys = MLXArray(Array(repeating: Float(1), count: 1 * 2 * 2 * headDim)).reshaped(1, 2, 2, headDim)
        let values = MLXArray(Array(repeating: Float(2), count: 1 * 2 * 2 * headDim)).reshaped(1, 2, 2, headDim)
        let (_, _) = sparseCache.update(keys: keys, values: values)

        var cache: [KVCache] = [sparseCache]
        maybeQuantizeKVCache(cache: &cache, kvBits: nil, kvGroupSize: 32, quantizedKVStart: 0)

        XCTAssertTrue(cache[0] is TriAttentionSparseKVCache)
        XCTAssertFalse(cache[0] is QuantizedTriAttentionSparseKVCache)
    }

    func testMaybeQuantizeKVCacheWaitsUntilOffsetExceedsThreshold() {
        let sparseCache = TriAttentionSparseKVCache(configuration: .v1Disabled)
        let keys = MLXArray(Array(repeating: Float(1), count: 1 * 2 * 2 * headDim)).reshaped(1, 2, 2, headDim)
        let values = MLXArray(Array(repeating: Float(2), count: 1 * 2 * 2 * headDim)).reshaped(1, 2, 2, headDim)
        let (_, _) = sparseCache.update(keys: keys, values: values)

        var cache: [KVCache] = [sparseCache]
        maybeQuantizeKVCache(cache: &cache, kvBits: 4, kvGroupSize: 32, quantizedKVStart: 2)

        XCTAssertTrue(cache[0] is TriAttentionSparseKVCache)
        XCTAssertFalse(cache[0] is QuantizedTriAttentionSparseKVCache)
    }

    func testQuantizedTriAttentionSparseCachePrunesAndCopiesWithoutLosingState() {
        let configuration = TriAttentionConfiguration(
            enabled: true,
            budgetTokens: 2,
            calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(rawValue: "artifact"),
            implementationVersion: .v1
        )
        let runtimeState = tryMakeRuntimeState(fullAttentionLayerIndices: [1], configuration: configuration)
        let cache = QuantizedTriAttentionSparseKVCache(
            configuration: configuration,
            groupSize: 32,
            bits: 4,
            runtimeState: runtimeState
        )
        let tokenCount = 131
        let keys = MLXArray(Array(repeating: Float(1), count: 1 * 2 * tokenCount * headDim)).reshaped(
            1,
            2,
            tokenCount,
            headDim
        )
        let values = MLXArray(Array(repeating: Float(2), count: 1 * 2 * tokenCount * headDim)).reshaped(
            1,
            2,
            tokenCount,
            headDim
        )
        _ = cache.updateQuantized(keys: keys, values: values)

        TriAttentionQwen35Runtime.pruneIfNeeded(caches: [cache], layerIndices: [1])

        XCTAssertEqual(cache.offset, tokenCount)
        XCTAssertEqual(cache.retainedPositions?.dim(2), 2)
        XCTAssertFalse(cache.isTrimmable)

        switch cache.makeMask(n: 2, windowSize: nil, returnArray: false) {
        case .array(let mask):
            XCTAssertEqual(mask.shape, [1, 2, 2, 4])
        default:
            XCTFail("Expected pruned quantized TriAttention cache to use an explicit sparse mask")
        }

        guard let copied = cache.copy() as? QuantizedTriAttentionSparseKVCache else {
            return XCTFail("Expected quantized TriAttention cache copy")
        }
        XCTAssertEqual(copied.offset, cache.offset)
        XCTAssertEqual(copied.configuration, cache.configuration)
        XCTAssertEqual(copied.groupSize, cache.groupSize)
        XCTAssertEqual(copied.bits, cache.bits)
        XCTAssertEqual(copied.retainedPositions?.asArray(Int32.self), cache.retainedPositions?.asArray(Int32.self))
    }

    func testQwen35TextModelRunsWithQuantizedTriAttentionSparseCaches() throws {
        let model = Qwen35TextModel(try makeConfiguration(fullAttentionInterval: 1))
        let triConfiguration = TriAttentionConfiguration(
            enabled: true,
            budgetTokens: 256,
            calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(rawValue: "artifact"),
            implementationVersion: .v1
        )
        let parameters = GenerateParameters(
            kvBits: 4,
            kvGroupSize: 32,
            quantizedKVStart: 3,
            triAttention: triConfiguration,
            triAttentionCalibrationArtifact: makeArtifact(
                headDim: headDim,
                sampledHeads: sampledHeads(forLayers: [0, 1, 2, 3])
            )
        )

        var triCache = model.newCache(parameters: parameters)
        let prompt = MLXArray(Array(0 ..< 4).map(Int32.init)).reshaped(1, 4)
        _ = model(prompt, cache: triCache)
        maybeQuantizeKVCache(
            cache: &triCache,
            kvBits: parameters.kvBits,
            kvGroupSize: parameters.kvGroupSize,
            quantizedKVStart: parameters.quantizedKVStart
        )

        XCTAssertTrue(triCache.allSatisfy { $0 is QuantizedTriAttentionSparseKVCache })

        let nextToken = MLXArray([Int32(4)]).reshaped(1, 1)
        let triOutput = model(nextToken, cache: triCache)
        eval(triOutput)
        XCTAssertEqual(triOutput.shape, [1, 1, 64])
        XCTAssertTrue(triCache.allSatisfy { $0 is QuantizedTriAttentionSparseKVCache })

        var denseCache = model.newCache(parameters: nil)
        _ = model(prompt, cache: denseCache)
        maybeQuantizeKVCache(cache: &denseCache, kvBits: 4, kvGroupSize: 32, quantizedKVStart: 3)
        XCTAssertTrue(denseCache.allSatisfy { $0 is QuantizedKVCache })
    }

    private func makeConfiguration(fullAttentionInterval: Int) throws -> Qwen35TextConfiguration {
        let json = """
        {
          "model_type": "qwen35",
          "hidden_size": 128,
          "num_hidden_layers": 4,
          "intermediate_size": 256,
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
          "head_dim": 32,
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
        sampledHeads: [TriAttentionCalibrationHeadKey]
    ) -> TriAttentionCalibrationArtifact {
        let freqCount = headDim / 2
        let stats = Dictionary(uniqueKeysWithValues: sampledHeads.map { key in
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
                sampledHeads: sampledHeads,
                headDim: headDim,
                ropeStyle: "half",
                modelName: "tests/qwen35-triattention"
            ),
            statsByHead: stats
        )
    }

    private func sampledHeads(forLayers layers: [Int]) -> [TriAttentionCalibrationHeadKey] {
        layers.flatMap { layerIndex in
            (0 ..< 4).map { headIndex in
                TriAttentionCalibrationHeadKey(layerIndex: layerIndex, headIndex: headIndex)
            }
        }
    }

    private func tryMakeRuntimeState(
        fullAttentionLayerIndices: [Int],
        configuration: TriAttentionConfiguration
    ) -> TriAttentionQwen35RuntimeState {
        guard let runtimeState = TriAttentionQwen35Runtime.makeState(
            configuration: configuration,
            artifact: makeArtifact(
                headDim: headDim,
                sampledHeads: sampledHeads(forLayers: fullAttentionLayerIndices)
            ),
            fullAttentionLayerIndices: fullAttentionLayerIndices,
            attentionHeads: 4,
            kvHeads: 2,
            headDim: headDim,
            partialRotaryFactor: 1.0,
            ropeTheta: 100_000,
            ropeType: "default"
        ) else {
            XCTFail("Expected compatible TriAttention runtime state")
            fatalError("Missing TriAttention runtime state for test")
        }
        return runtimeState
    }
}
