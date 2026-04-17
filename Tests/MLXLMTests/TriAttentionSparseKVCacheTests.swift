import MLX
import MLXLMCommon
import XCTest

final class TriAttentionSparseKVCacheTests: XCTestCase {

    func testEmptyCacheStartsDisabledAndNonTrimmable() {
        let configuration = TriAttentionConfiguration(
            enabled: false,
            budgetTokens: TriAttentionConfiguration.v1BudgetTokens,
            calibrationArtifactIdentity: nil,
            implementationVersion: .v1
        )
        let cache = TriAttentionSparseKVCache(configuration: configuration)

        XCTAssertEqual(cache.configuration, configuration)
        XCTAssertEqual(cache.offset, 0)
        XCTAssertTrue(cache.state.isEmpty)
        XCTAssertFalse(cache.isTrimmable)
        XCTAssertEqual(
            cache.metaState,
            ["v1", "false", "12000", "", "protectStablePrefixOnly"]
        )
        XCTAssertNil(cache.retainedPositions)
        XCTAssertNil(cache.retainedKeys)
        XCTAssertNil(cache.retainedValues)
    }

    func testUpdateAppendsRetainedStateAndPositionsUseAbsoluteOffsets() {
        let cache = TriAttentionSparseKVCache(configuration: .v1Disabled)

        let keys1 = MLXArray([1.0 as Float, 2.0, 3.0, 4.0]).reshaped(1, 1, 2, 2)
        let values1 = MLXArray([10.0 as Float, 11.0, 12.0, 13.0, 14.0, 15.0]).reshaped(1, 1, 2, 3)
        let (_, _) = cache.update(keys: keys1, values: values1)

        let keys2 = MLXArray([5.0 as Float, 6.0, 7.0, 8.0]).reshaped(1, 1, 2, 2)
        let values2 = MLXArray([16.0 as Float, 17.0, 18.0, 19.0, 20.0, 21.0]).reshaped(1, 1, 2, 3)
        let (returnedKeys, returnedValues) = cache.update(keys: keys2, values: values2)

        let expectedKeys = concatenated([keys1, keys2], axis: 2)
        let expectedValues = concatenated([values1, values2], axis: 2)

        XCTAssertEqual(cache.offset, 4)
        XCTAssertEqual(cache.retainedPositions?.shape, [1, 1, 4])
        XCTAssertEqual(cache.retainedPositions?.asArray(Int32.self), [0, 1, 2, 3])
        XCTAssertAllClose(cache.retainedKeys, expectedKeys)
        XCTAssertAllClose(cache.retainedValues, expectedValues)
        XCTAssertAllClose(returnedKeys, expectedKeys)
        XCTAssertAllClose(returnedValues, expectedValues)
    }

    func testStateSetterPreservesLogicalOffsetAndMetaStateRoundTripsConfiguration() {
        let cache = TriAttentionSparseKVCache(configuration: .v1Disabled)

        let initialKeys = MLXArray([1.0 as Float, 2.0, 3.0, 4.0]).reshaped(1, 1, 2, 2)
        let initialValues = MLXArray([5.0 as Float, 6.0, 7.0, 8.0]).reshaped(1, 1, 2, 2)
        let (_, _) = cache.update(keys: initialKeys, values: initialValues)
        XCTAssertEqual(cache.offset, 2)

        let retainedPositions = MLXArray([Int32(10), 11, 42]).reshaped(1, 1, 3)
        let retainedKeys = MLXArray([20.0 as Float, 21.0, 22.0, 23.0, 24.0, 25.0]).reshaped(1, 1, 3, 2)
        let retainedValues = MLXArray([30.0 as Float, 31.0, 32.0, 33.0, 34.0, 35.0]).reshaped(1, 1, 3, 2)
        cache.state = [retainedPositions, retainedKeys, retainedValues]

        XCTAssertEqual(cache.offset, 2)
        XCTAssertEqual(cache.retainedPositions?.asArray(Int32.self), [10, 11, 42])
        XCTAssertAllClose(cache.retainedKeys, retainedKeys)
        XCTAssertAllClose(cache.retainedValues, retainedValues)

        cache.metaState = ["v1", "true", "12000", "artifact-sha256"]

        XCTAssertEqual(
            cache.configuration,
            TriAttentionConfiguration(
                enabled: true,
                budgetTokens: 12_000,
                calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(
                    rawValue: "artifact-sha256"),
                implementationVersion: .v1,
                prefixProtectionMode: .protectNone
            )
        )
        XCTAssertEqual(cache.offset, 2)

        cache.state = []

        XCTAssertEqual(cache.offset, 2)
        XCTAssertTrue(cache.state.isEmpty)
        XCTAssertNil(cache.retainedPositions)
        XCTAssertNil(cache.retainedKeys)
        XCTAssertNil(cache.retainedValues)
    }

    func testStateSetterRoundsTripModernPrefixProtectionModeFromMetaState() {
        let cache = TriAttentionSparseKVCache(configuration: .v1Disabled)

        cache.metaState = ["v1", "true", "12000", "artifact-sha256", "protectAllPrefill"]

        XCTAssertEqual(
            cache.configuration,
            TriAttentionConfiguration(
                enabled: true,
                budgetTokens: 12_000,
                calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(
                    rawValue: "artifact-sha256"
                ),
                implementationVersion: .v1,
                prefixProtectionMode: .protectAllPrefill
            )
        )
    }

    func testCopyPreservesConfigurationAndProducesIndependentState() {
        let original = TriAttentionSparseKVCache(
            configuration: TriAttentionConfiguration(
                enabled: true,
                budgetTokens: 12_000,
                calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(
                    rawValue: "artifact-copy"),
                implementationVersion: .v1
            )
        )

        let initialKeys = MLXArray([1.0 as Float, 2.0, 3.0, 4.0]).reshaped(1, 1, 2, 2)
        let initialValues = MLXArray([5.0 as Float, 6.0, 7.0, 8.0]).reshaped(1, 1, 2, 2)
        let (_, _) = original.update(keys: initialKeys, values: initialValues)

        guard let copied = original.copy() as? TriAttentionSparseKVCache else {
            return XCTFail("Expected TriAttentionSparseKVCache copy")
        }

        XCTAssertEqual(copied.configuration, original.configuration)
        XCTAssertEqual(copied.metaState, original.metaState)
        XCTAssertEqual(copied.offset, original.offset)
        XCTAssertAllClose(copied.retainedPositions, original.retainedPositions)
        XCTAssertAllClose(copied.retainedKeys, original.retainedKeys)
        XCTAssertAllClose(copied.retainedValues, original.retainedValues)

        let moreKeys = MLXArray([9.0 as Float, 10.0, 11.0, 12.0]).reshaped(1, 1, 2, 2)
        let moreValues = MLXArray([13.0 as Float, 14.0, 15.0, 16.0]).reshaped(1, 1, 2, 2)
        let (_, _) = copied.update(keys: moreKeys, values: moreValues)

        XCTAssertEqual(original.offset, 2)
        XCTAssertEqual(original.retainedPositions?.asArray(Int32.self), [0, 1])
        XCTAssertEqual(copied.offset, 4)
        XCTAssertEqual(copied.retainedPositions?.asArray(Int32.self), [0, 1, 2, 3])
    }

    func testTrimIsBlockedAndPromptCacheHelpersRespectNonTrimmableCache() {
        let cache = TriAttentionSparseKVCache(configuration: .v1Disabled)
        let keys = MLXArray([1.0 as Float, 2.0, 3.0, 4.0]).reshaped(1, 1, 2, 2)
        let values = MLXArray([5.0 as Float, 6.0, 7.0, 8.0]).reshaped(1, 1, 2, 2)
        let (_, _) = cache.update(keys: keys, values: values)

        let retainedStateBeforeTrim = cache.state

        XCTAssertEqual(cache.trim(10), 0)
        XCTAssertEqual(cache.offset, 2)
        XCTAssertEqual(retainedStateBeforeTrim.count, cache.state.count)
        XCTAssertAllClose(retainedStateBeforeTrim[0], cache.state[0])
        XCTAssertAllClose(retainedStateBeforeTrim[1], cache.state[1])
        XCTAssertAllClose(retainedStateBeforeTrim[2], cache.state[2])

        XCTAssertFalse(canTrimPromptCache([KVCacheSimple(), cache]))
        XCTAssertEqual(trimPromptCache([cache], numTokens: 5), 0)
        XCTAssertEqual(cache.offset, 2)
        XCTAssertEqual(cache.retainedPositions?.asArray(Int32.self), [0, 1])
    }

    func testTrimPromptCacheBlocksMixedDensePlusTriAttention() {
        let dense = KVCacheSimple()
        let denseKeys = MLXArray([1.0 as Float, 2.0, 3.0, 4.0, 5.0, 6.0]).reshaped(1, 1, 3, 2)
        let denseValues = MLXArray([7.0 as Float, 8.0, 9.0, 10.0, 11.0, 12.0]).reshaped(1, 1, 3, 2)
        let (_, _) = dense.update(keys: denseKeys, values: denseValues)
        XCTAssertEqual(dense.offset, 3)

        let sparse = TriAttentionSparseKVCache(configuration: .v1Disabled)
        let sparseKeys = MLXArray([1.0 as Float, 2.0, 3.0, 4.0]).reshaped(1, 1, 2, 2)
        let sparseValues = MLXArray([5.0 as Float, 6.0, 7.0, 8.0]).reshaped(1, 1, 2, 2)
        let (_, _) = sparse.update(keys: sparseKeys, values: sparseValues)

        let mixed: [KVCache] = [dense, sparse]
        XCTAssertFalse(canTrimPromptCache(mixed))
        XCTAssertEqual(trimPromptCache(mixed, numTokens: 2), 0)

        XCTAssertEqual(dense.offset, 3,
                       "Dense layer must keep its offset when trim is blocked by a sibling TriAttention layer")
        XCTAssertEqual(sparse.offset, 2)
        XCTAssertEqual(sparse.retainedPositions?.asArray(Int32.self), [0, 1])
    }

    func testContainsTriAttentionStateIdentifiesSparseAndQuantizedLayers() {
        let denseOnly: [KVCache] = [KVCacheSimple(), KVCacheSimple()]
        XCTAssertFalse(containsTriAttentionState(denseOnly))

        let sparse = TriAttentionSparseKVCache(configuration: .v1Disabled)
        XCTAssertTrue(containsTriAttentionState([sparse]))
        XCTAssertTrue(containsTriAttentionState([KVCacheSimple(), sparse]))

        let quantizedSparse = QuantizedTriAttentionSparseKVCache(
            configuration: .v1Disabled,
            groupSize: 32,
            bits: 4
        )
        XCTAssertTrue(containsTriAttentionState([quantizedSparse]))
        XCTAssertTrue(containsTriAttentionState([KVCacheSimple(), quantizedSparse]))

        XCTAssertFalse(containsTriAttentionState([]))
    }

    func testTrimPromptCacheStillTrimsPureDenseCaches() {
        let dense1 = KVCacheSimple()
        let dense2 = KVCacheSimple()
        let keys = MLXArray([1.0 as Float, 2.0, 3.0, 4.0, 5.0, 6.0]).reshaped(1, 1, 3, 2)
        let values = MLXArray([7.0 as Float, 8.0, 9.0, 10.0, 11.0, 12.0]).reshaped(1, 1, 3, 2)
        let (_, _) = dense1.update(keys: keys, values: values)
        let (_, _) = dense2.update(keys: keys, values: values)

        let cache: [KVCache] = [dense1, dense2]
        XCTAssertTrue(canTrimPromptCache(cache))
        XCTAssertFalse(containsTriAttentionState(cache))

        XCTAssertEqual(trimPromptCache(cache, numTokens: 1), 1)
        XCTAssertEqual(dense1.offset, 2)
        XCTAssertEqual(dense2.offset, 2)
    }

    func testTrimPromptCacheBlocksMixedDensePlusQuantizedTriAttention() {
        let dense = KVCacheSimple()
        let denseKeys = MLXArray([1.0 as Float, 2.0, 3.0, 4.0]).reshaped(1, 1, 2, 2)
        let denseValues = MLXArray([5.0 as Float, 6.0, 7.0, 8.0]).reshaped(1, 1, 2, 2)
        let (_, _) = dense.update(keys: denseKeys, values: denseValues)

        let quantizedSparse = QuantizedTriAttentionSparseKVCache(
            configuration: .v1Disabled,
            groupSize: 32,
            bits: 4
        )
        let quantKeys = MLXArray(Array(repeating: Float(1), count: 1 * 1 * 2 * 32)).reshaped(1, 1, 2, 32)
        let quantValues = MLXArray(Array(repeating: Float(2), count: 1 * 1 * 2 * 32)).reshaped(1, 1, 2, 32)
        _ = quantizedSparse.updateQuantized(keys: quantKeys, values: quantValues)

        let mixed: [KVCache] = [dense, quantizedSparse]
        XCTAssertFalse(canTrimPromptCache(mixed))
        XCTAssertTrue(containsTriAttentionState(mixed))

        XCTAssertEqual(trimPromptCache(mixed, numTokens: 1), 0)
        XCTAssertEqual(dense.offset, 2)
        XCTAssertEqual(quantizedSparse.offset, 2)
    }

    func testMaskStaysSymbolicUntilRetainedStateBecomesSparse() {
        let cache = TriAttentionSparseKVCache(configuration: .v1Disabled)
        let keys = MLXArray([1.0 as Float, 2.0, 3.0, 4.0]).reshaped(1, 1, 2, 2)
        let values = MLXArray([5.0 as Float, 6.0, 7.0, 8.0]).reshaped(1, 1, 2, 2)
        let (_, _) = cache.update(keys: keys, values: values)

        switch cache.makeMask(n: 2, windowSize: nil, returnArray: false) {
        case .causal:
            break
        default:
            XCTFail("Expected contiguous retained state to keep the symbolic causal mask")
        }

        cache.state = [
            MLXArray([Int32(0), 2]).reshaped(1, 1, 2),
            keys,
            values,
        ]

        switch cache.makeMask(n: 2, windowSize: nil, returnArray: false) {
        case .array(let mask):
            XCTAssertEqual(mask.shape, [1, 1, 2, 4])
        default:
            XCTFail("Expected sparse retained state to use an explicit array mask")
        }
    }

    func testSparseMaskExpandsAlongHeadDimForGroupedQueryAttention() throws {
        // Simulate a GQA configuration (4 query heads, 2 kv heads, n_repeats=2)
        // with sparse retained positions. The produced mask must have
        // `attentionHeads` entries along dim 1 so it broadcasts against MLX's
        // rank-4 attention-score shape `(B, n_q_heads, Lq, Lkv)` — a rank-4
        // mask with `kvH` in dim 1 would fail GQA broadcast.
        let headDim = 8
        let freqCount = headDim / 2
        let sampledHeads = (0 ..< 4).map {
            TriAttentionCalibrationHeadKey(layerIndex: 0, headIndex: $0)
        }
        let stats = Dictionary(uniqueKeysWithValues: sampledHeads.map { key in
            (
                key,
                TriAttentionCalibrationHeadStats(
                    qMeanReal: Array(repeating: Float(0), count: freqCount),
                    qMeanImag: Array(repeating: Float(0), count: freqCount),
                    qAbsMean: Array(repeating: Float(0), count: freqCount)
                )
            )
        })
        let artifact = TriAttentionCalibrationArtifact(
            metadata: TriAttentionCalibrationMetadata(
                sampledHeads: sampledHeads,
                headDim: headDim,
                ropeStyle: "half",
                modelName: "tests/gqa-sparse-mask"
            ),
            statsByHead: stats
        )
        let configuration = TriAttentionConfiguration(
            enabled: true,
            budgetTokens: 4,
            calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(rawValue: "artifact"),
            implementationVersion: .v1
        )
        let runtimeState = try XCTUnwrap(
            TriAttentionQwen35Runtime.makeState(
                configuration: configuration,
                artifact: artifact,
                fullAttentionLayerIndices: [0],
                attentionHeads: 4,
                kvHeads: 2,
                headDim: headDim,
                partialRotaryFactor: 1.0,
                ropeTheta: 100_000,
                ropeType: "default"
            )
        )
        let cache = TriAttentionSparseKVCache(
            configuration: configuration,
            runtimeState: runtimeState
        )

        // Install a sparse retained state with kvH=2 along dim 1.
        cache.state = [
            MLXArray([Int32(0), 2, 1, 3]).reshaped(1, 2, 2),
            MLXArray(Array(repeating: Float(1), count: 1 * 2 * 2 * headDim)).reshaped(1, 2, 2, headDim),
            MLXArray(Array(repeating: Float(1), count: 1 * 2 * 2 * headDim)).reshaped(1, 2, 2, headDim),
        ]

        switch cache.makeMask(n: 2, windowSize: nil, returnArray: false) {
        case .array(let mask):
            // Raw sparse mask is `(B=1, kvH=2, Lq=2, Lkv=4)`; after GQA
            // expansion it becomes `(B=1, n_q_heads=4, Lq=2, Lkv=4)` so each
            // kvH's mask is tiled for its `nRepeats=2` query heads.
            XCTAssertEqual(mask.shape, [1, 4, 2, 4])
        default:
            XCTFail("Expected sparse retained state to use an explicit array mask")
        }
    }

    func testRestoredSparseCachePrunesAgainWhenRestoreContextIsProvided() throws {
        let (configuration, restoreContext) = try makeRestoreContext(
            budgetTokens: 2,
            headDim: 8,
            attentionHeads: 4,
            kvHeads: 2
        )
        let original = TriAttentionSparseKVCache(
            configuration: configuration,
            runtimeState: restoreContext.runtimeState
        )
        let initialTokenCount = configuration.budgetTokens + 128
        let initialKeys = MLXArray(
            Array(repeating: Float(1), count: 1 * 2 * initialTokenCount * 8)
        ).reshaped(1, 2, initialTokenCount, 8)
        let initialValues = MLXArray(
            Array(repeating: Float(2), count: 1 * 2 * initialTokenCount * 8)
        ).reshaped(1, 2, initialTokenCount, 8)
        _ = original.update(keys: initialKeys, values: initialValues)

        TriAttentionQwen35Runtime.pruneIfNeeded(caches: [original], layerIndices: [0])
        XCTAssertEqual(original.retainedPositions?.dim(2), configuration.budgetTokens)

        let snapshot = try XCTUnwrap(HybridCacheSnapshot.capture(
            cache: [original],
            offset: original.offset,
            type: .leaf
        ))
        let restored = snapshot.restore(triAttentionRestoreContext: restoreContext)
        let restoredCache = try XCTUnwrap(restored[0] as? TriAttentionSparseKVCache)
        XCTAssertEqual(restoredCache.retainedPositions?.dim(2), configuration.budgetTokens)

        let appendedTokenCount = 128
        let appendedKeys = MLXArray(
            Array(repeating: Float(3), count: 1 * 2 * appendedTokenCount * 8)
        ).reshaped(1, 2, appendedTokenCount, 8)
        let appendedValues = MLXArray(
            Array(repeating: Float(4), count: 1 * 2 * appendedTokenCount * 8)
        ).reshaped(1, 2, appendedTokenCount, 8)
        _ = restoredCache.update(keys: appendedKeys, values: appendedValues)

        TriAttentionQwen35Runtime.pruneIfNeeded(caches: [restoredCache], layerIndices: [0])
        XCTAssertEqual(restoredCache.retainedPositions?.dim(2), configuration.budgetTokens)
    }

    func testRestoredSparseCacheExpandsGroupedQueryMaskWhenRestoreContextIsProvided() throws {
        let (configuration, restoreContext) = try makeRestoreContext(
            budgetTokens: 4,
            headDim: 8,
            attentionHeads: 4,
            kvHeads: 2
        )
        let original = TriAttentionSparseKVCache(
            configuration: configuration,
            runtimeState: restoreContext.runtimeState
        )
        original.offset = 4
        original.state = [
            MLXArray([Int32(0), 2, 1, 3]).reshaped(1, 2, 2),
            MLXArray(Array(repeating: Float(1), count: 1 * 2 * 2 * 8)).reshaped(1, 2, 2, 8),
            MLXArray(Array(repeating: Float(2), count: 1 * 2 * 2 * 8)).reshaped(1, 2, 2, 8),
        ]

        let snapshot = try XCTUnwrap(HybridCacheSnapshot.capture(
            cache: [original],
            offset: original.offset,
            type: .system
        ))
        let restored = snapshot.restore(triAttentionRestoreContext: restoreContext)
        let restoredCache = try XCTUnwrap(restored[0] as? TriAttentionSparseKVCache)

        switch restoredCache.makeMask(n: 2, windowSize: nil, returnArray: false) {
        case .array(let mask):
            XCTAssertEqual(mask.shape, [1, 4, 2, 4])
        default:
            XCTFail("Expected restored sparse retained state to use an explicit array mask")
        }
    }

    func testProtectNoneKeepsCurrentBudgetedPruneBehavior() throws {
        let stablePrefixOffset = 4
        let (configuration, restoreContext) = try makeRestoreContext(
            budgetTokens: 2,
            headDim: 8,
            attentionHeads: 4,
            kvHeads: 2,
            prefixProtectionMode: .protectNone
        )
        let cache = TriAttentionSparseKVCache(
            configuration: configuration,
            runtimeState: restoreContext.runtimeState
        )
        var parameters = GenerateParameters(triAttention: configuration)
        parameters.triAttentionStablePrefixOffset = stablePrefixOffset
        let tokenCount = configuration.budgetTokens + 129
        parameters.configureTriAttentionCachesForPrefill([cache], inputTokenCount: tokenCount)

        let keys = MLXArray(
            Array(repeating: Float(1), count: 1 * 2 * tokenCount * 8)
        ).reshaped(1, 2, tokenCount, 8)
        let values = MLXArray(
            Array(repeating: Float(2), count: 1 * 2 * tokenCount * 8)
        ).reshaped(1, 2, tokenCount, 8)
        _ = cache.update(keys: keys, values: values)

        TriAttentionQwen35Runtime.pruneIfNeeded(caches: [cache], layerIndices: [0])

        let retainedPositions = try XCTUnwrap(cache.retainedPositions?[0, 0].asArray(Int32.self))
        XCTAssertEqual(retainedPositions.count, configuration.budgetTokens)
        XCTAssertFalse(Set(retainedPositions).isSuperset(of: Set(Int32(0) ..< Int32(stablePrefixOffset))))
    }

    func testProtectStablePrefixOnlyRetainsStablePrefixOutsideRecentWindow() throws {
        let stablePrefixOffset = 4
        let (configuration, restoreContext) = try makeRestoreContext(
            budgetTokens: 2,
            headDim: 8,
            attentionHeads: 4,
            kvHeads: 2,
            prefixProtectionMode: .protectStablePrefixOnly
        )
        let cache = TriAttentionSparseKVCache(
            configuration: configuration,
            runtimeState: restoreContext.runtimeState
        )
        var parameters = GenerateParameters(triAttention: configuration)
        parameters.triAttentionStablePrefixOffset = stablePrefixOffset
        let tokenCount = configuration.budgetTokens + 129
        parameters.configureTriAttentionCachesForPrefill([cache], inputTokenCount: tokenCount)

        let keys = MLXArray(
            Array(repeating: Float(1), count: 1 * 2 * tokenCount * 8)
        ).reshaped(1, 2, tokenCount, 8)
        let values = MLXArray(
            Array(repeating: Float(2), count: 1 * 2 * tokenCount * 8)
        ).reshaped(1, 2, tokenCount, 8)
        _ = cache.update(keys: keys, values: values)

        TriAttentionQwen35Runtime.pruneIfNeeded(caches: [cache], layerIndices: [0])

        let retainedPositions = try XCTUnwrap(cache.retainedPositions?[0, 0].asArray(Int32.self))
        XCTAssertEqual(retainedPositions.count, configuration.budgetTokens + stablePrefixOffset)
        XCTAssertTrue(Set(retainedPositions).isSuperset(of: Set(Int32(0) ..< Int32(stablePrefixOffset))))
        XCTAssertTrue(Set(retainedPositions).contains(Int32(tokenCount - 1)))
    }

    func testProtectAllPrefillRetainsWholePrefillThenPrunesDecodeTail() throws {
        let prefillTokenCount = 129
        let decodeTokenCount = 3
        let (configuration, restoreContext) = try makeRestoreContext(
            budgetTokens: 2,
            headDim: 8,
            attentionHeads: 4,
            kvHeads: 2,
            prefixProtectionMode: .protectAllPrefill
        )
        let cache = TriAttentionSparseKVCache(
            configuration: configuration,
            runtimeState: restoreContext.runtimeState
        )
        let parameters = GenerateParameters(triAttention: configuration)
        parameters.configureTriAttentionCachesForPrefill([cache], inputTokenCount: prefillTokenCount)

        let prefillKeys = MLXArray(
            Array(repeating: Float(1), count: 1 * 2 * prefillTokenCount * 8)
        ).reshaped(1, 2, prefillTokenCount, 8)
        let prefillValues = MLXArray(
            Array(repeating: Float(2), count: 1 * 2 * prefillTokenCount * 8)
        ).reshaped(1, 2, prefillTokenCount, 8)
        _ = cache.update(keys: prefillKeys, values: prefillValues)

        let decodeKeys = MLXArray(
            Array(repeating: Float(3), count: 1 * 2 * decodeTokenCount * 8)
        ).reshaped(1, 2, decodeTokenCount, 8)
        let decodeValues = MLXArray(
            Array(repeating: Float(4), count: 1 * 2 * decodeTokenCount * 8)
        ).reshaped(1, 2, decodeTokenCount, 8)
        _ = cache.update(keys: decodeKeys, values: decodeValues)

        TriAttentionQwen35Runtime.pruneIfNeeded(caches: [cache], layerIndices: [0])

        let retainedPositions = try XCTUnwrap(cache.retainedPositions?[0, 0].asArray(Int32.self))
        XCTAssertEqual(retainedPositions.count, prefillTokenCount + configuration.budgetTokens)
        XCTAssertTrue(Set(retainedPositions).isSuperset(of: Set(Int32(0) ..< Int32(prefillTokenCount))))
        XCTAssertLessThan(retainedPositions.count, prefillTokenCount + decodeTokenCount)
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

    private func makeRestoreContext(
        budgetTokens: Int,
        headDim: Int,
        attentionHeads: Int,
        kvHeads: Int,
        prefixProtectionMode: TriAttentionPrefixProtectionMode = .protectStablePrefixOnly
    ) throws -> (TriAttentionConfiguration, TriAttentionSnapshotRestoreContext) {
        let configuration = TriAttentionConfiguration(
            enabled: true,
            budgetTokens: budgetTokens,
            calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity(rawValue: "artifact"),
            implementationVersion: .v1,
            prefixProtectionMode: prefixProtectionMode
        )
        let runtimeState = try XCTUnwrap(
            TriAttentionQwen35Runtime.makeState(
                configuration: configuration,
                artifact: makeArtifact(headDim: headDim, attentionHeads: attentionHeads),
                fullAttentionLayerIndices: [0],
                attentionHeads: attentionHeads,
                kvHeads: kvHeads,
                headDim: headDim,
                partialRotaryFactor: 1.0,
                ropeTheta: 100_000,
                ropeType: "default"
            )
        )
        return (
            configuration,
            TriAttentionSnapshotRestoreContext(
                expectedConfiguration: configuration,
                runtimeState: runtimeState
            )
        )
    }

    private func makeArtifact(
        headDim: Int,
        attentionHeads: Int
    ) -> TriAttentionCalibrationArtifact {
        let sampledHeads = (0 ..< attentionHeads).map {
            TriAttentionCalibrationHeadKey(layerIndex: 0, headIndex: $0)
        }
        let freqCount = headDim / 2
        let stats = Dictionary(uniqueKeysWithValues: sampledHeads.map { key in
            (
                key,
                TriAttentionCalibrationHeadStats(
                    qMeanReal: Array(repeating: Float(1), count: freqCount),
                    qMeanImag: Array(repeating: Float(0.5), count: freqCount),
                    qAbsMean: Array(repeating: Float(1.5), count: freqCount)
                )
            )
        })
        return TriAttentionCalibrationArtifact(
            metadata: TriAttentionCalibrationMetadata(
                sampledHeads: sampledHeads,
                headDim: headDim,
                ropeStyle: "half",
                modelName: "tests/triattention-restore"
            ),
            statsByHead: stats
        )
    }
}
