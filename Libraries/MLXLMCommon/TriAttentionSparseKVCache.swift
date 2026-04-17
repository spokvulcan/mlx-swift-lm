import Foundation
import MLX

internal protocol TriAttentionRuntimeCache: KVCache, AnyObject {
    var configuration: TriAttentionConfiguration { get }
    var runtimeState: TriAttentionQwen35RuntimeState? { get }
    var retainedPositions: MLXArray? { get }
    var retainedTokenCount: Int { get }
    var protectedPrefixOffset: Int? { get }

    func dequantizedRetainedKeysForRuntime() -> MLXArray?
    func applyKeepIndices(_ keepIndices: MLXArray)
    func attachRuntimeState(_ restoreContext: TriAttentionSnapshotRestoreContext?)
    func configureProtectedPrefixOffset(_ protectedPrefixOffset: Int?)
}

internal func triAttentionConfigurationMetaState(_ configuration: TriAttentionConfiguration) -> [String] {
    [
        configuration.implementationVersion.rawValue,
        String(configuration.enabled),
        String(configuration.budgetTokens),
        configuration.calibrationArtifactIdentity?.rawValue ?? "",
        configuration.prefixProtectionMode.rawValue,
    ]
}

internal func triAttentionConfigurationFromMetaState(
    _ newValue: [String],
    ownerName: String
) -> TriAttentionConfiguration {
    guard newValue.count == 4 || newValue.count == 5 else {
        fatalError("\(ownerName) metaState must have exactly 4 or 5 values")
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

    let prefixProtectionMode: TriAttentionPrefixProtectionMode
    if newValue.count == 5 {
        guard let mode = TriAttentionPrefixProtectionMode(rawValue: newValue[4]) else {
            fatalError(
                "Unsupported TriAttention prefix protection mode '\(newValue[4])' in metaState"
            )
        }
        prefixProtectionMode = mode
    } else {
        prefixProtectionMode = .protectNone
    }

    return TriAttentionConfiguration(
        enabled: enabled,
        budgetTokens: budgetTokens,
        calibrationArtifactIdentity: calibrationArtifactIdentity,
        implementationVersion: implementationVersion,
        prefixProtectionMode: prefixProtectionMode
    )
}

internal func triAttentionMaskMode(
    retainedPositions: MLXArray?,
    offset: Int,
    usesSparseMask: Bool,
    n: Int,
    windowSize: Int?,
    returnArray: Bool,
    attentionHeads: Int? = nil,
    kvHeads: Int? = nil
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    guard let retainedPositions else {
        if n == 1 {
            return .none
        }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }
    guard usesSparseMask else {
        if n == 1 {
            return .none
        }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }
    guard n > 1 else {
        return .none
    }

    let queryPositions = triAttentionMakeAbsolutePositions(
        offset: offset,
        batchSize: retainedPositions.dim(0),
        kvHeads: retainedPositions.dim(1),
        sequenceLength: n
    )
    let keyPositions = concatenated([retainedPositions, queryPositions], axis: 2)
    let queryGrid = expandedDimensions(queryPositions, axis: -1)
    let keyGrid = expandedDimensions(keyPositions, axis: -2)
    var mask = queryGrid .>= keyGrid

    if let windowSize {
        mask = mask & (queryGrid .< (keyGrid + MLXArray(Int32(windowSize))))
    }

    if let attentionHeads, let kvHeads {
        mask = expandMaskForGroupedQueryHeads(
            mask, attentionHeads: attentionHeads, kvHeads: kvHeads
        )
    }

    return .array(mask)
}

internal func triAttentionMakeAbsolutePositions(
    offset: Int,
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

internal func triAttentionRetainedStateMatchesDensePrefix(
    positions: MLXArray,
    offset: Int
) -> Bool {
    guard positions.dim(2) == offset else {
        return false
    }
    guard offset > 0 else {
        return positions.dim(2) == 0
    }

    let expected = triAttentionDensePrefixPositions(
        batchSize: positions.dim(0),
        kvHeads: positions.dim(1),
        count: positions.dim(2)
    )
    return positions.asArray(Int32.self) == expected.asArray(Int32.self)
}

internal func triAttentionDensePrefixPositions(
    batchSize: Int,
    kvHeads: Int,
    count: Int
) -> MLXArray {
    let basePositions = MLXArray(Int32(0) ..< Int32(count)).expandedDimensions(axes: [0, 1])
    return broadcast(basePositions, to: [batchSize, kvHeads, count])
}

internal func triAttentionValidateRetainedState(
    positions: MLXArray,
    keyValueStates: [MLXArray],
    keyPrefix: String,
    valuePrefix: String
) {
    guard positions.ndim == 3 else {
        fatalError(
            "TriAttentionSparseKVCache retainedPositions must be 3D [batch, kvHeads, retainedCount]"
        )
    }
    guard positions.dtype == .int32 else {
        fatalError("TriAttentionSparseKVCache retainedPositions must use Int32 dtype")
    }

    for array in keyValueStates {
        guard array.ndim == 4 else {
            fatalError(
                "TriAttentionSparseKVCache retained \(keyPrefix) and \(valuePrefix) state must be 4D"
            )
        }
        guard positions.dim(0) == array.dim(0),
            positions.dim(1) == array.dim(1),
            positions.dim(2) == array.dim(2)
        else {
            fatalError(
                "TriAttentionSparseKVCache retained state must agree on batch, kvHeads, and retainedCount"
            )
        }
    }
}

/// Sparse KV cache placeholder for TriAttention layers.
///
/// This cache tracks retained positions plus retained keys/values separately
/// from the logical processed-token offset so later pruning can preserve sparse
/// state without redefining the cache contract.
public final class TriAttentionSparseKVCache:
    BaseKVCache,
    DecodeTimeQuantizableKVCache,
    TriAttentionRuntimeCache
{
    public private(set) var configuration: TriAttentionConfiguration
    public private(set) var retainedPositions: MLXArray?
    public private(set) var retainedKeys: MLXArray?
    public private(set) var retainedValues: MLXArray?
    internal private(set) var runtimeState: TriAttentionQwen35RuntimeState?
    internal private(set) var protectedPrefixOffset: Int?
    private var usesSparseMask = false

    public init(configuration: TriAttentionConfiguration) {
        self.configuration = configuration
        self.runtimeState = nil
        self.protectedPrefixOffset = nil
        super.init()
    }

    public init(
        configuration: TriAttentionConfiguration,
        runtimeState: TriAttentionQwen35RuntimeState?
    ) {
        self.runtimeState = runtimeState
        self.configuration = configuration
        self.protectedPrefixOffset = nil
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        validateUpdateInputs(keys: keys, values: values)

        let sequenceLength = keys.dim(2)
        let positions = triAttentionMakeAbsolutePositions(
            offset: offset,
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
            returnArray: returnArray,
            attentionHeads: runtimeState?.attentionHeads,
            kvHeads: runtimeState?.kvHeads
        )
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
                protectedPrefixOffset = nil
                usesSparseMask = false
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
            usesSparseMask = !triAttentionRetainedStateMatchesDensePrefix(
                positions: positions,
                offset: offset
            )
        }
    }

    public override var metaState: [String] {
        get {
            triAttentionConfigurationMetaState(configuration)
        }
        set {
            configuration = triAttentionConfigurationFromMetaState(
                newValue,
                ownerName: "TriAttentionSparseKVCache"
            )
        }
    }

    public override var isTrimmable: Bool { false }

    @discardableResult
    public override func trim(_ n: Int) -> Int { 0 }

    public override func copy() -> any KVCache {
        let new = TriAttentionSparseKVCache(
            configuration: configuration,
            runtimeState: runtimeState
        )
        new.offset = offset
        new.usesSparseMask = usesSparseMask
        new.protectedPrefixOffset = protectedPrefixOffset

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

    internal var retainedTokenCount: Int {
        retainedPositions?.dim(2) ?? 0
    }

    internal func dequantizedRetainedKeysForRuntime() -> MLXArray? {
        retainedKeys
    }

    internal func applyKeepIndices(_ keepIndices: MLXArray) {
        guard
            let retainedPositions,
            let retainedKeys,
            let retainedValues
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
        let keyIndices = broadcast(
            expandedDimensions(batchIndices, axis: -1),
            to: [batchSize, kvHeads, keepCount, retainedKeys.dim(3)]
        )
        let valueIndices = broadcast(
            expandedDimensions(batchIndices, axis: -1),
            to: [batchSize, kvHeads, keepCount, retainedValues.dim(3)]
        )

        self.retainedPositions = takeAlong(retainedPositions, batchIndices, axis: 2)
        self.retainedKeys = takeAlong(retainedKeys, keyIndices, axis: 2)
        self.retainedValues = takeAlong(retainedValues, valueIndices, axis: 2)
        usesSparseMask = true
    }

    internal func attachRuntimeState(_ restoreContext: TriAttentionSnapshotRestoreContext?) {
        guard
            let restoreContext,
            configuration == restoreContext.expectedConfiguration
        else {
            return
        }
        runtimeState = restoreContext.runtimeState
    }

    internal func configureProtectedPrefixOffset(_ protectedPrefixOffset: Int?) {
        self.protectedPrefixOffset = protectedPrefixOffset
    }

    func toDecodeTimeQuantized(groupSize: Int, bits: Int) -> any KVCache {
        let quantizedCache = QuantizedTriAttentionSparseKVCache(
            configuration: configuration,
            groupSize: groupSize,
            bits: bits,
            runtimeState: runtimeState
        )
        quantizedCache.offset = offset
        quantizedCache.configureProtectedPrefixOffset(protectedPrefixOffset)

        if let retainedPositions,
            let retainedKeys,
            let retainedValues
        {
            quantizedCache.state = [
                retainedPositions[.ellipsis],
            ] + QuantizedKVCache(groupSize: groupSize, bits: bits).consumeQuantizedState(
                keys: retainedKeys,
                values: retainedValues
            )
        }

        return quantizedCache
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
        triAttentionValidateRetainedState(
            positions: positions,
            keyValueStates: [keys, values],
            keyPrefix: "keys",
            valuePrefix: "values"
        )
    }
}

private extension QuantizedKVCache {
    func consumeQuantizedState(keys: MLXArray, values: MLXArray) -> [MLXArray] {
        _ = updateQuantized(keys: keys, values: values)
        return state.map { $0[.ellipsis] }
    }
}
