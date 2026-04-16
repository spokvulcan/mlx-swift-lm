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
        XCTAssertEqual(cache.metaState, ["v1", "false", "12000", ""])
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
                implementationVersion: .v1
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
