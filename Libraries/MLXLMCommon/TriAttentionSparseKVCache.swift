import Foundation
import MLX

/// Sparse KV cache placeholder for TriAttention layers.
///
/// This cache tracks retained positions plus retained keys/values separately
/// from the logical processed-token offset so later pruning can preserve sparse
/// state without redefining the cache contract.
public final class TriAttentionSparseKVCache: BaseKVCache {
    public private(set) var configuration: TriAttentionConfiguration
    public private(set) var retainedPositions: MLXArray?
    public private(set) var retainedKeys: MLXArray?
    public private(set) var retainedValues: MLXArray?

    public init(configuration: TriAttentionConfiguration) {
        self.configuration = configuration
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        validateUpdateInputs(keys: keys, values: values)

        let sequenceLength = keys.dim(2)
        let positions = makeAbsolutePositions(
            batchSize: keys.dim(0),
            kvHeads: keys.dim(1),
            sequenceLength: sequenceLength
        )

        if let retainedPositions,
            let retainedKeys,
            let retainedValues
        {
            self.retainedPositions = concatenated([retainedPositions, positions], axis: 2)
            self.retainedKeys = concatenated([retainedKeys, keys], axis: 2)
            self.retainedValues = concatenated([retainedValues, values], axis: 2)
        } else {
            self.retainedPositions = positions
            self.retainedKeys = keys
            self.retainedValues = values
        }

        offset += sequenceLength
        return (self.retainedKeys!, self.retainedValues!)
    }

    public override var state: [MLXArray] {
        get {
            switch (retainedPositions, retainedKeys, retainedValues) {
            case let (positions?, keys?, values?):
                return [positions, keys, values]
            case (nil, nil, nil):
                return []
            default:
                fatalError(
                    "TriAttentionSparseKVCache internal state must contain either all retained arrays or none"
                )
            }
        }
        set {
            if newValue.isEmpty {
                retainedPositions = nil
                retainedKeys = nil
                retainedValues = nil
                return
            }

            guard newValue.count == 3 else {
                fatalError(
                    "TriAttentionSparseKVCache state must have exactly 3 arrays (retainedPositions, retainedKeys, retainedValues)"
                )
            }

            let positions = newValue[0]
            let keys = newValue[1]
            let values = newValue[2]
            validateRetainedState(positions: positions, keys: keys, values: values)

            retainedPositions = positions
            retainedKeys = keys
            retainedValues = values
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
                fatalError("TriAttentionSparseKVCache metaState must have exactly 4 values")
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
        let new = TriAttentionSparseKVCache(configuration: configuration)
        new.offset = offset

        if let retainedPositions,
            let retainedKeys,
            let retainedValues
        {
            new.retainedPositions = retainedPositions[.ellipsis]
            new.retainedKeys = retainedKeys[.ellipsis]
            new.retainedValues = retainedValues[.ellipsis]
        }

        return new
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

        if let retainedPositions,
            let retainedKeys,
            let retainedValues
        {
            guard retainedPositions.dim(0) == keys.dim(0),
                retainedPositions.dim(1) == keys.dim(1),
                retainedKeys.dim(0) == keys.dim(0),
                retainedKeys.dim(1) == keys.dim(1),
                retainedValues.dim(0) == values.dim(0),
                retainedValues.dim(1) == values.dim(1)
            else {
                fatalError(
                    "TriAttentionSparseKVCache update shape must match existing retained batch and kvHeads"
                )
            }
            guard retainedKeys.dim(3) == keys.dim(3),
                retainedValues.dim(3) == values.dim(3)
            else {
                fatalError(
                    "TriAttentionSparseKVCache update head dimensions must match existing retained state"
                )
            }
        }
    }

    private func validateRetainedState(positions: MLXArray, keys: MLXArray, values: MLXArray) {
        guard positions.ndim == 3 else {
            fatalError(
                "TriAttentionSparseKVCache retainedPositions must be 3D [batch, kvHeads, retainedCount]"
            )
        }
        guard keys.ndim == 4 else {
            fatalError(
                "TriAttentionSparseKVCache retainedKeys must be 4D [batch, kvHeads, retainedCount, keyDim]"
            )
        }
        guard values.ndim == 4 else {
            fatalError(
                "TriAttentionSparseKVCache retainedValues must be 4D [batch, kvHeads, retainedCount, valueDim]"
            )
        }
        guard positions.dtype == .int32 else {
            fatalError("TriAttentionSparseKVCache retainedPositions must use Int32 dtype")
        }
        guard positions.dim(0) == keys.dim(0),
            positions.dim(0) == values.dim(0),
            positions.dim(1) == keys.dim(1),
            positions.dim(1) == values.dim(1),
            positions.dim(2) == keys.dim(2),
            positions.dim(2) == values.dim(2)
        else {
            fatalError(
                "TriAttentionSparseKVCache retained state must agree on batch, kvHeads, and retainedCount"
            )
        }
    }

    private func makeAbsolutePositions(
        batchSize: Int,
        kvHeads: Int,
        sequenceLength: Int
    ) -> MLXArray {
        let endOffset = offset + sequenceLength
        guard let start = Int32(exactly: offset),
            let end = Int32(exactly: endOffset)
        else {
            fatalError("TriAttentionSparseKVCache offset exceeded Int32 position range")
        }

        let basePositions = MLXArray(start ..< end).expandedDimensions(axes: [0, 1])
        return broadcast(basePositions, to: [batchSize, kvHeads, sequenceLength])
    }
}
