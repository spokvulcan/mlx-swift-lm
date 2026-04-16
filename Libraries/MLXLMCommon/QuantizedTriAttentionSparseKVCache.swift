import Foundation
import MLX

/// Quantized TriAttention sparse cache used after the decode-time KV
/// quantization threshold is crossed.
public final class QuantizedTriAttentionSparseKVCache:
    BaseKVCache,
    QuantizedKVCacheProtocol,
    TriAttentionRuntimeCache
{
    public private(set) var configuration: TriAttentionConfiguration
    public private(set) var retainedPositions: MLXArray?
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    internal let runtimeState: TriAttentionQwen35RuntimeState?

    private var quantizedKeyValueCache: QuantizedKVCache
    private var usesSparseMask = false
    private var keyHeadDimension: Int?
    private var valueHeadDimension: Int?

    public init(
        configuration: TriAttentionConfiguration,
        groupSize: Int = 64,
        bits: Int = 8,
        mode: QuantizationMode = .affine,
        runtimeState: TriAttentionQwen35RuntimeState? = nil
    ) {
        self.configuration = configuration
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self.runtimeState = runtimeState
        self.quantizedKeyValueCache = QuantizedKVCache(groupSize: groupSize, bits: bits, mode: mode)
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "`update` was called on `QuantizedTriAttentionSparseKVCache`. Use `updateQuantized` instead."
        )
    }

    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        validateUpdateInputs(keys: keys, values: values)

        let sequenceLength = keys.dim(2)
        let positions = triAttentionMakeAbsolutePositions(
            offset: offset,
            batchSize: keys.dim(0),
            kvHeads: keys.dim(1),
            sequenceLength: sequenceLength
        )

        if let retainedPositions {
            self.retainedPositions = concatenated([retainedPositions, positions], axis: 2)
        } else {
            self.retainedPositions = positions
        }

        let quantizedState = quantizedKeyValueCache.updateQuantized(keys: keys, values: values)
        keyHeadDimension = keys.dim(3)
        valueHeadDimension = values.dim(3)
        offset += sequenceLength
        return quantizedState
    }

    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        quantizedKeyValueCache.getQuantizedState()
    }

    public override func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        triAttentionMaskMode(
            retainedPositions: retainedPositions,
            offset: offset,
            usesSparseMask: usesSparseMask,
            n: n,
            windowSize: windowSize,
            returnArray: returnArray
        )
    }

    public override var state: [MLXArray] {
        get {
            switch (retainedPositions, quantizedKeyValueCache.state.isEmpty) {
            case let (positions?, false):
                return [positions] + quantizedKeyValueCache.state
            case (nil, true):
                return []
            default:
                fatalError(
                    "QuantizedTriAttentionSparseKVCache internal state must contain retainedPositions and quantized key/value arrays together"
                )
            }
        }
        set {
            if newValue.isEmpty {
                retainedPositions = nil
                quantizedKeyValueCache = QuantizedKVCache(
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                )
                quantizedKeyValueCache.offset = 0
                keyHeadDimension = nil
                valueHeadDimension = nil
                usesSparseMask = false
                return
            }

            guard newValue.count == 5 || newValue.count == 7 else {
                fatalError(
                    "QuantizedTriAttentionSparseKVCache state must have retainedPositions plus 4 or 6 quantized key/value arrays"
                )
            }

            let positions = newValue[0]
            let quantizedState = Array(newValue.dropFirst())
            validateRetainedState(positions: positions, quantizedState: quantizedState)

            retainedPositions = positions
            quantizedKeyValueCache.state = quantizedState
            quantizedKeyValueCache.offset = positions.dim(2)

            let (keyDim, valueDim) = deriveHeadDimensions(from: quantizedState)
            keyHeadDimension = keyDim
            valueHeadDimension = valueDim
            usesSparseMask = !triAttentionRetainedStateMatchesDensePrefix(
                positions: positions,
                offset: offset
            )
        }
    }

    public override var metaState: [String] {
        get {
            [
                configuration.implementationVersion.rawValue,
                String(configuration.enabled),
                String(configuration.budgetTokens),
                configuration.calibrationArtifactIdentity?.rawValue ?? "",
            ]
        }
        set {
            guard newValue.count == 4 else {
                fatalError("QuantizedTriAttentionSparseKVCache metaState must have exactly 4 values")
            }
            guard let implementationVersion = TriAttentionImplementationVersion(rawValue: newValue[0]) else {
                fatalError(
                    "Unsupported TriAttention implementation version '\(newValue[0])' in metaState"
                )
            }

            let enabled: Bool
            switch newValue[1] {
            case "true":
                enabled = true
            case "false":
                enabled = false
            default:
                fatalError("Failed to parse TriAttention enabled flag from metaState: \(newValue[1])")
            }

            guard let budgetTokens = Int(newValue[2]) else {
                fatalError("Failed to parse TriAttention budget from metaState: \(newValue[2])")
            }

            let calibrationArtifactIdentity: TriAttentionCalibrationArtifactIdentity? =
                if newValue[3].isEmpty {
                    nil
                } else {
                    TriAttentionCalibrationArtifactIdentity(rawValue: newValue[3])
                }

            configuration = TriAttentionConfiguration(
                enabled: enabled,
                budgetTokens: budgetTokens,
                calibrationArtifactIdentity: calibrationArtifactIdentity,
                implementationVersion: implementationVersion
            )
        }
    }

    public override var isTrimmable: Bool { false }

    @discardableResult
    public override func trim(_ n: Int) -> Int { 0 }

    public override func copy() -> any KVCache {
        let new = QuantizedTriAttentionSparseKVCache(
            configuration: configuration,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            runtimeState: runtimeState
        )
        new.offset = offset
        new.usesSparseMask = usesSparseMask
        new.keyHeadDimension = keyHeadDimension
        new.valueHeadDimension = valueHeadDimension

        if let retainedPositions {
            new.retainedPositions = retainedPositions[.ellipsis]
            new.quantizedKeyValueCache.state = quantizedKeyValueCache.state.map { $0[.ellipsis] }
            new.quantizedKeyValueCache.offset = quantizedKeyValueCache.offset
        }

        return new
    }

    internal var retainedTokenCount: Int {
        retainedPositions?.dim(2) ?? 0
    }

    internal func dequantizedRetainedKeysForRuntime() -> MLXArray? {
        guard let quantizedState = quantizedKeyValueCache.getQuantizedState() else {
            return nil
        }
        let quantizedKeys = quantizedState.0
        return dequantized(
            quantizedKeys.0,
            scales: quantizedKeys.1,
            biases: quantizedKeys.2,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }

    internal func applyKeepIndices(_ keepIndices: MLXArray) {
        guard
            let retainedPositions,
            let quantizedState = quantizedKeyValueCache.getQuantizedState()
        else {
            return
        }

        let batchSize = retainedPositions.dim(0)
        let kvHeads = retainedPositions.dim(1)
        let keepCount = keepIndices.dim(1)
        let batchIndices = broadcast(
            expandedDimensions(keepIndices, axis: 0),
            to: [batchSize, kvHeads, keepCount]
        )

        self.retainedPositions = takeAlong(retainedPositions, batchIndices, axis: 2)

        let prunedKeys = takeAlongQuantizedTuple(quantizedState.0, batchIndices: batchIndices)
        let prunedValues = takeAlongQuantizedTuple(quantizedState.1, batchIndices: batchIndices)
        quantizedKeyValueCache.state = [
            prunedKeys.0, prunedKeys.1, prunedKeys.2,
            prunedValues.0, prunedValues.1, prunedValues.2,
        ].compactMap { $0 }
        quantizedKeyValueCache.offset = keepCount
        usesSparseMask = true
    }

    private func validateUpdateInputs(keys: MLXArray, values: MLXArray) {
        guard keys.ndim == 4 else {
            fatalError("TriAttentionSparseKVCache keys must be 4D [batch, kvHeads, seqLen, keyDim]")
        }
        guard values.ndim == 4 else {
            fatalError(
                "TriAttentionSparseKVCache values must be 4D [batch, kvHeads, seqLen, valueDim]"
            )
        }
        guard keys.dim(0) == values.dim(0),
            keys.dim(1) == values.dim(1),
            keys.dim(2) == values.dim(2)
        else {
            fatalError(
                "TriAttentionSparseKVCache keys and values must agree on batch, kvHeads, and seqLen"
            )
        }

        if let retainedPositions {
            guard retainedPositions.dim(0) == keys.dim(0),
                retainedPositions.dim(1) == keys.dim(1)
            else {
                fatalError(
                    "TriAttentionSparseKVCache update shape must match existing retained batch and kvHeads"
                )
            }
        }
        if let keyHeadDimension, let valueHeadDimension {
            guard keyHeadDimension == keys.dim(3),
                valueHeadDimension == values.dim(3)
            else {
                fatalError(
                    "TriAttentionSparseKVCache update head dimensions must match existing retained state"
                )
            }
        }
    }

    private func validateRetainedState(positions: MLXArray, quantizedState: [MLXArray]) {
        triAttentionValidateRetainedState(
            positions: positions,
            keyValueStates: quantizedState,
            keyPrefix: "quantized keys",
            valuePrefix: "quantized values"
        )
    }

    private func deriveHeadDimensions(from quantizedState: [MLXArray]) -> (Int, Int) {
        let quantizedKeys: (MLXArray, MLXArray, MLXArray?)
        let quantizedValues: (MLXArray, MLXArray, MLXArray?)

        switch quantizedState.count {
        case 4:
            quantizedKeys = (quantizedState[0], quantizedState[1], nil)
            quantizedValues = (quantizedState[2], quantizedState[3], nil)
        case 6:
            quantizedKeys = (quantizedState[0], quantizedState[1], quantizedState[2])
            quantizedValues = (quantizedState[3], quantizedState[4], quantizedState[5])
        default:
            fatalError(
                "QuantizedTriAttentionSparseKVCache state must have 4 or 6 quantized key/value arrays"
            )
        }

        let dequantizedKeys = dequantized(
            quantizedKeys.0,
            scales: quantizedKeys.1,
            biases: quantizedKeys.2,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        let dequantizedValues = dequantized(
            quantizedValues.0,
            scales: quantizedValues.1,
            biases: quantizedValues.2,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        return (dequantizedKeys.dim(3), dequantizedValues.dim(3))
    }

    private func takeAlongQuantizedTuple(
        _ tuple: (MLXArray, MLXArray, MLXArray?),
        batchIndices: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray?) {
        let keepCount = batchIndices.dim(2)
        let weightIndices = broadcast(
            expandedDimensions(batchIndices, axis: -1),
            to: [batchIndices.dim(0), batchIndices.dim(1), keepCount, tuple.0.dim(3)]
        )
        let scaleIndices = broadcast(
            expandedDimensions(batchIndices, axis: -1),
            to: [batchIndices.dim(0), batchIndices.dim(1), keepCount, tuple.1.dim(3)]
        )

        let prunedWeights = takeAlong(tuple.0, weightIndices, axis: 2)
        let prunedScales = takeAlong(tuple.1, scaleIndices, axis: 2)
        let prunedBiases: MLXArray?
        if let biases = tuple.2 {
            let biasIndices = broadcast(
                expandedDimensions(batchIndices, axis: -1),
                to: [batchIndices.dim(0), batchIndices.dim(1), keepCount, biases.dim(3)]
            )
            prunedBiases = takeAlong(biases, biasIndices, axis: 2)
        } else {
            prunedBiases = nil
        }

        return (prunedWeights, prunedScales, prunedBiases)
    }
}
