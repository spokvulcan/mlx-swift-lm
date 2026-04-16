import MLXLMCommon
import MLXVLM
import XCTest

final class TriAttentionQwen35VLMDeferredTests: XCTestCase {

    func testVLMQwen35NewCacheIgnoresTriAttentionParametersAndStaysDense() throws {
        let model = Qwen35(try makeConfiguration())
        let cache = model.newCache(parameters: GenerateParameters(
            triAttention: TriAttentionConfiguration(
                enabled: true,
                budgetTokens: 12_000,
                calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(
                    rawValue: "triattention-vlm-deferred"
                ),
                implementationVersion: .v1
            ),
            triAttentionCalibrationArtifact: makeArtifact()
        ))

        XCTAssertEqual(cache.count, 4)
        XCTAssertTrue(cache[0] is MambaCache)
        XCTAssertTrue(cache[1] is KVCacheSimple)
        XCTAssertTrue(cache[2] is MambaCache)
        XCTAssertTrue(cache[3] is KVCacheSimple)
        XCTAssertFalse(cache.contains { $0 is TriAttentionSparseKVCache })
        XCTAssertFalse(cache.contains { $0 is QuantizedTriAttentionSparseKVCache })
    }

    func testVLMQwen35NewCacheStillUsesRotatingDenseCachesWithMaxKVSize() throws {
        let model = Qwen35(try makeConfiguration())
        let cache = model.newCache(parameters: GenerateParameters(
            maxKVSize: 32,
            triAttention: TriAttentionConfiguration(
                enabled: true,
                budgetTokens: 12_000,
                calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(
                    rawValue: "triattention-vlm-deferred"
                ),
                implementationVersion: .v1
            ),
            triAttentionCalibrationArtifact: makeArtifact()
        ))

        XCTAssertEqual(cache.count, 4)
        XCTAssertTrue(cache[0] is MambaCache)
        XCTAssertTrue(cache[1] is RotatingKVCache)
        XCTAssertTrue(cache[2] is MambaCache)
        XCTAssertTrue(cache[3] is RotatingKVCache)
        XCTAssertFalse(cache.contains { $0 is TriAttentionSparseKVCache })
        XCTAssertFalse(cache.contains { $0 is QuantizedTriAttentionSparseKVCache })
    }

    private func makeConfiguration() throws -> Qwen35Configuration {
        let json = """
        {
          "model_type": "qwen35_vl",
          "text_config": {
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
            "full_attention_interval": 2,
            "num_experts": 0,
            "num_experts_per_tok": 0,
            "decoder_sparse_step": 1,
            "shared_expert_intermediate_size": 0,
            "moe_intermediate_size": 0,
            "norm_topk_prob": true
          },
          "vision_config": {
            "model_type": "qwen3_vl",
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "out_hidden_size": 32,
            "num_heads": 1,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "num_position_embeddings": 16
          },
          "vocab_size": 64
        }
        """
        return try JSONDecoder().decode(Qwen35Configuration.self, from: Data(json.utf8))
    }

    private func makeArtifact() -> TriAttentionCalibrationArtifact {
        let sampledHeads = [
            TriAttentionCalibrationHeadKey(layerIndex: 1, headIndex: 0),
            TriAttentionCalibrationHeadKey(layerIndex: 1, headIndex: 1),
            TriAttentionCalibrationHeadKey(layerIndex: 3, headIndex: 0),
            TriAttentionCalibrationHeadKey(layerIndex: 3, headIndex: 1),
        ]
        let stats = Dictionary(uniqueKeysWithValues: sampledHeads.map { key in
            (
                key,
                TriAttentionCalibrationHeadStats(
                    qMeanReal: [1, 2, 3, 4],
                    qMeanImag: [0.5, 1, 1.5, 2],
                    qAbsMean: [2, 3, 4, 5]
                )
            )
        })
        return TriAttentionCalibrationArtifact(
            metadata: TriAttentionCalibrationMetadata(
                sampledHeads: sampledHeads,
                headDim: 8,
                ropeStyle: "half",
                modelName: "tests/qwen35-vlm-deferred"
            ),
            statsByHead: stats
        )
    }
}
