import Foundation
import MLX

internal struct TriAttentionPreparedHeadStats {
    let qMeanReal: MLXArray
    let qMeanImag: MLXArray
    let qAbsMean: MLXArray
}

public final class TriAttentionQwen35RuntimeState {
    static let divideLength = 128
    static let windowSize = 128
    static let offsetMaxLength = 65_536
    static let sparseNormalizeScores = true
    static let includePrefillInBudget = true
    static let protectPrefill = false
    static let disableMlr = false
    static let disableTrig = false

    let artifact: TriAttentionCalibrationArtifact
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let partialRotaryFactor: Float
    let ropeTheta: Float
    let preparedStats: [TriAttentionCalibrationHeadKey: TriAttentionPreparedHeadStats]
    let sampledHeadsByLayer: [Int: [TriAttentionCalibrationHeadKey]]
    let omega: MLXArray
    let freqScaleSq: MLXArray
    let offsets: MLXArray

    init(
        artifact: TriAttentionCalibrationArtifact,
        attentionHeads: Int,
        kvHeads: Int,
        headDim: Int,
        partialRotaryFactor: Float,
        ropeTheta: Float,
        preparedStats: [TriAttentionCalibrationHeadKey: TriAttentionPreparedHeadStats],
        sampledHeadsByLayer: [Int: [TriAttentionCalibrationHeadKey]],
        omega: MLXArray,
        freqScaleSq: MLXArray,
        offsets: MLXArray
    ) {
        self.artifact = artifact
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeTheta = ropeTheta
        self.preparedStats = preparedStats
        self.sampledHeadsByLayer = sampledHeadsByLayer
        self.omega = omega
        self.freqScaleSq = freqScaleSq
        self.offsets = offsets
    }

    var kvHeadGroupSize: Int {
        max(1, attentionHeads / max(1, kvHeads))
    }

}

public enum TriAttentionQwen35Runtime {
    public static func makeState(
        configuration: TriAttentionConfiguration,
        artifact: TriAttentionCalibrationArtifact?,
        fullAttentionLayerIndices: [Int],
        attentionHeads: Int,
        kvHeads: Int,
        headDim: Int,
        partialRotaryFactor: Float,
        ropeTheta: Float,
        ropeType: String?
    ) -> TriAttentionQwen35RuntimeState? {
        guard
            configuration.enabled,
            let artifact,
            configuration.calibrationArtifactIdentity != nil
        else {
            return nil
        }

        guard artifact.metadata.headDim == nil || artifact.metadata.headDim == headDim else {
            return nil
        }
        if let ropeStyle = artifact.metadata.ropeStyle, ropeStyle != "half" {
            return nil
        }

        if let ropeType, ropeType != "default" {
            return nil
        }

        let rotaryPairs = max(1, Int(Float(headDim) * partialRotaryFactor) / 2)
        let freqCount = headDim / 2
        guard rotaryPairs <= freqCount else {
            return nil
        }

        let validLayerIndices = Set(fullAttentionLayerIndices)
        guard !validLayerIndices.isEmpty else {
            return nil
        }

        var sampledHeadsByLayer: [Int: [TriAttentionCalibrationHeadKey]] = [:]
        var preparedStats: [TriAttentionCalibrationHeadKey: TriAttentionPreparedHeadStats] = [:]
        var observedLayerIndices = Set<Int>()

        for headKey in artifact.metadata.sampledHeads {
            guard validLayerIndices.contains(headKey.layerIndex) else {
                return nil
            }
            guard (0 ..< attentionHeads).contains(headKey.headIndex) else {
                return nil
            }
            guard let rawStats = artifact.statsByHead[headKey] else {
                return nil
            }
            guard
                rawStats.qMeanReal.count == freqCount,
                rawStats.qMeanImag.count == freqCount,
                rawStats.qAbsMean.count == freqCount
            else {
                return nil
            }
            sampledHeadsByLayer[headKey.layerIndex, default: []].append(headKey)
            observedLayerIndices.insert(headKey.layerIndex)
            preparedStats[headKey] = TriAttentionPreparedHeadStats(
                qMeanReal: MLXArray(rawStats.qMeanReal),
                qMeanImag: MLXArray(rawStats.qMeanImag),
                qAbsMean: MLXArray(rawStats.qAbsMean)
            )
        }

        guard !preparedStats.isEmpty else {
            return nil
        }
        guard observedLayerIndices == validLayerIndices else {
            return nil
        }

        let omega = makeOmega(
            headDim: headDim,
            rotaryPairs: rotaryPairs,
            ropeTheta: ropeTheta
        )
        let freqScaleSq = MLXArray.ones([freqCount], dtype: .float32)
        let offsets = makeGeometricOffsets(maxLength: TriAttentionQwen35RuntimeState.offsetMaxLength)

        return TriAttentionQwen35RuntimeState(
            artifact: artifact,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            headDim: headDim,
            partialRotaryFactor: partialRotaryFactor,
            ropeTheta: ropeTheta,
            preparedStats: preparedStats,
            sampledHeadsByLayer: sampledHeadsByLayer,
            omega: omega,
            freqScaleSq: freqScaleSq,
            offsets: offsets
        )
    }

    public static func shouldUseSparseCaches(
        configuration: TriAttentionConfiguration,
        artifact: TriAttentionCalibrationArtifact?,
        fullAttentionLayerIndices: [Int],
        attentionHeads: Int,
        kvHeads: Int,
        headDim: Int,
        partialRotaryFactor: Float,
        ropeTheta: Float,
        ropeType: String?
    ) -> Bool {
        makeState(
            configuration: configuration,
            artifact: artifact,
            fullAttentionLayerIndices: fullAttentionLayerIndices,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            headDim: headDim,
            partialRotaryFactor: partialRotaryFactor,
            ropeTheta: ropeTheta,
            ropeType: ropeType
        ) != nil
    }

    public static func pruneIfNeeded(
        caches: [KVCache],
        layerIndices: [Int]
    ) {
        guard
            !caches.isEmpty,
            caches.count == layerIndices.count
        else {
            return
        }
        let runtimeCaches = caches.compactMap { $0 as? TriAttentionRuntimeCache }
        guard
            runtimeCaches.count == caches.count,
            let runtimeState = runtimeCaches.first?.runtimeState
        else {
            return
        }

        let threshold =
            runtimeCaches[0].configuration.budgetTokens + TriAttentionQwen35RuntimeState.divideLength
        guard runtimeCaches[0].retainedTokenCount >= threshold else {
            return
        }

        let roundStart = runtimeCaches[0].offset
        let keepIndices = sharedPerHeadKeepIndices(
            caches: runtimeCaches,
            layerIndices: layerIndices,
            runtimeState: runtimeState,
            budget: runtimeCaches[0].configuration.budgetTokens,
            roundStart: roundStart
        )
        guard let keepIndices else {
            return
        }

        for cache in runtimeCaches {
            cache.applyKeepIndices(keepIndices)
        }
    }

    private static func sharedPerHeadKeepIndices(
        caches: [TriAttentionRuntimeCache],
        layerIndices: [Int],
        runtimeState: TriAttentionQwen35RuntimeState,
        budget: Int,
        roundStart: Int
    ) -> MLXArray? {
        var perLayerScores: [MLXArray] = []

        for (cache, layerIndex) in zip(caches, layerIndices) {
            guard let layerScores = layerScores(
                cache: cache,
                layerIndex: layerIndex,
                runtimeState: runtimeState,
                roundStart: roundStart
            ) else {
                continue
            }
            perLayerScores.append(layerScores)
        }

        guard let firstLayerScores = perLayerScores.first else {
            return nil
        }

        var aggregated = firstLayerScores
        if perLayerScores.count > 1 {
            for layerScores in perLayerScores.dropFirst() {
                aggregated = aggregated + layerScores
            }
            aggregated = aggregated / Float(perLayerScores.count)
        }

        let tokenCount = aggregated.dim(1)
        let keepCount = min(budget, tokenCount)
        guard keepCount > 0 else {
            return MLXArray.zeros([runtimeState.kvHeads, 0], dtype: .int32)
        }

        let topUnsorted = argPartition(-aggregated, kth: keepCount - 1, axis: -1)[
            .ellipsis, ..<keepCount
        ]
        let sortOrder = argSort(topUnsorted, axis: -1)
        return takeAlong(topUnsorted, sortOrder, axis: -1)
    }

    private static func layerScores(
        cache: TriAttentionRuntimeCache,
        layerIndex: Int,
        runtimeState: TriAttentionQwen35RuntimeState,
        roundStart: Int
    ) -> MLXArray? {
        guard
            let sampledHeads = runtimeState.sampledHeadsByLayer[layerIndex],
            !sampledHeads.isEmpty,
            let retainedKeys = cache.dequantizedRetainedKeysForRuntime(),
            let retainedPositions = cache.retainedPositions
        else {
            return nil
        }

        precondition(retainedKeys.dim(0) == 1, "TriAttention runtime only supports batch size 1")

        var rawScoresByHead: [MLXArray] = []
        var guardMasksByHead: [MLXArray] = []

        for headKey in sampledHeads {
            guard let stats = runtimeState.preparedStats[headKey] else { continue }
            let kvHead = min(runtimeState.kvHeads - 1, headKey.headIndex / runtimeState.kvHeadGroupSize)

            let positions = retainedPositions[0, kvHead]
            let rotatedKeys = retainedKeys[0, kvHead]
            let keyValues = invertHalfRoPE(
                rotatedKeys: rotatedKeys,
                positions: positions,
                headDim: runtimeState.headDim,
                omega: runtimeState.omega
            )
            let scores = scoreHead(
                stats: stats,
                keyValues: keyValues,
                keyPositions: positions,
                roundStart: roundStart,
                runtimeState: runtimeState
            )
            rawScoresByHead.append(scores)

            let windowStart = roundStart - min(
                TriAttentionQwen35RuntimeState.windowSize,
                max(0, cache.configuration.budgetTokens - 1)
            )
            let guardMask = positions.asType(.int32) .>= MLXArray(Int32(max(0, windowStart)))
            guardMasksByHead.append(guardMask)
        }

        guard !rawScoresByHead.isEmpty else {
            return nil
        }

        var normalizedScores = stacked(rawScoresByHead, axis: 0)
        if TriAttentionQwen35RuntimeState.sparseNormalizeScores {
            let means = mean(normalizedScores, axis: -1, keepDims: true)
            let rawDeviations = std(normalizedScores, axis: -1, keepDims: true)
            let deviations = MLX.where(
                rawDeviations .< MLXArray(1e-6 as Float),
                MLXArray(1e-6 as Float),
                rawDeviations
            )
            normalizedScores = (normalizedScores - means) / deviations
        }

        let guardMasks = stacked(guardMasksByHead, axis: 0)
        normalizedScores = MLX.where(guardMasks, MLXArray(Float.infinity), normalizedScores)

        var perKVHeadScores: [MLXArray] = []
        let fallback = mean(normalizedScores, axis: 0)

        for kvHead in 0 ..< runtimeState.kvHeads {
            let headRows = sampledHeads.enumerated().compactMap { index, headKey -> MLXArray? in
                let mappedKVHead = min(runtimeState.kvHeads - 1, headKey.headIndex / runtimeState.kvHeadGroupSize)
                return mappedKVHead == kvHead ? normalizedScores[index] : nil
            }
            if headRows.isEmpty {
                perKVHeadScores.append(fallback)
            } else if headRows.count == 1 {
                perKVHeadScores.append(headRows[0])
            } else {
                perKVHeadScores.append(stacked(headRows, axis: 0).max(axis: 0))
            }
        }

        return stacked(perKVHeadScores, axis: 0)
    }

    private static func scoreHead(
        stats: TriAttentionPreparedHeadStats,
        keyValues: MLXArray,
        keyPositions: MLXArray,
        roundStart: Int,
        runtimeState: TriAttentionQwen35RuntimeState
    ) -> MLXArray {
        let freqCount = runtimeState.headDim / 2
        let keyReal = keyValues[.ellipsis, ..<freqCount]
        let keyImag = keyValues[.ellipsis, freqCount...]

        let qMeanAbs = sqrt(stats.qMeanReal * stats.qMeanReal + stats.qMeanImag * stats.qMeanImag)
        let keyAbs = sqrt(keyReal * keyReal + keyImag * keyImag)
        let relativeReal = keyReal * stats.qMeanReal + keyImag * stats.qMeanImag
        let relativeImag = keyReal * stats.qMeanImag - keyImag * stats.qMeanReal
        let extra = (stats.qAbsMean - qMeanAbs) * keyAbs

        let baseDelta = MLXArray(Float(roundStart), dtype: .float32) - keyPositions.asType(.float32)
        let deltaGrid = expandedDimensions(baseDelta, axis: -1) + expandedDimensions(runtimeState.offsets, axis: 0)
        let phase = expandedDimensions(deltaGrid, axis: -1) * runtimeState.omega

        let trigTerm =
            (
                expandedDimensions(relativeReal, axis: 1) * cos(phase)
                - expandedDimensions(relativeImag, axis: 1) * sin(phase)
            ) * runtimeState.freqScaleSq
        let baseScores = trigTerm.sum(axis: -1)
        let additive = (extra * runtimeState.freqScaleSq).sum(axis: -1, keepDims: true)
        return mean(baseScores + additive, axis: -1)
    }

    private static func invertHalfRoPE(
        rotatedKeys: MLXArray,
        positions: MLXArray,
        headDim: Int,
        omega: MLXArray
    ) -> MLXArray {
        let freqCount = headDim / 2
        let phase = expandedDimensions(positions.asType(.float32), axis: -1) * omega
        let cosPhase = cos(phase)
        let sinPhase = sin(phase)

        let rotatedReal = rotatedKeys[.ellipsis, ..<freqCount]
        let rotatedImag = rotatedKeys[.ellipsis, freqCount...]

        let unrotatedReal = rotatedReal * cosPhase + rotatedImag * sinPhase
        let unrotatedImag = rotatedImag * cosPhase - rotatedReal * sinPhase
        return concatenated([unrotatedReal, unrotatedImag], axis: -1)
    }

    private static func makeGeometricOffsets(maxLength: Int) -> MLXArray {
        var offsets: [Float] = []
        var value = 1
        while value <= maxLength {
            offsets.append(Float(value))
            value *= 2
        }
        return MLXArray(offsets)
    }

    private static func makeOmega(
        headDim: Int,
        rotaryPairs: Int,
        ropeTheta: Float
    ) -> MLXArray {
        let freqCount = headDim / 2
        let prefix = (0 ..< rotaryPairs).map { pairIndex in
            pow(ropeTheta, -Float(pairIndex) / Float(max(1, rotaryPairs)))
        }
        let padded = prefix + Array(repeating: Float.zero, count: max(0, freqCount - rotaryPairs))
        return MLXArray(padded)
    }
}
