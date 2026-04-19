import Foundation
import MLX
import os

private let triLog = Logger(subsystem: "app.tesseract.agent", category: "triattention-runtime")

internal struct TriAttentionPreparedHeadStats {
    let qMeanReal: MLXArray
    let qMeanImag: MLXArray
    let qAbsDelta: MLXArray
}

internal struct TriAttentionPreparedKVHeadScoreContext {
    let keyRealRotary: MLXArray
    let keyImagRotary: MLXArray
    let keyRealStatic: MLXArray
    let keyImagStatic: MLXArray
    let keyAbsRotary: MLXArray
    let keyAbsStatic: MLXArray
    let cosPhase: MLXArray
    let sinPhase: MLXArray
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
    let freqCount: Int
    let rotaryPairs: Int
    let preparedStats: [TriAttentionCalibrationHeadKey: TriAttentionPreparedHeadStats]
    let sampledHeadsByLayer: [Int: [TriAttentionCalibrationHeadKey]]
    let rotaryOmega: MLXArray
    let omega: MLXArray
    let freqScaleSq: MLXArray
    let rotaryFreqScaleSq: MLXArray
    let staticFreqScaleSq: MLXArray
    let offsets: MLXArray

    init(
        artifact: TriAttentionCalibrationArtifact,
        attentionHeads: Int,
        kvHeads: Int,
        headDim: Int,
        partialRotaryFactor: Float,
        ropeTheta: Float,
        freqCount: Int,
        rotaryPairs: Int,
        preparedStats: [TriAttentionCalibrationHeadKey: TriAttentionPreparedHeadStats],
        sampledHeadsByLayer: [Int: [TriAttentionCalibrationHeadKey]],
        rotaryOmega: MLXArray,
        omega: MLXArray,
        freqScaleSq: MLXArray,
        rotaryFreqScaleSq: MLXArray,
        staticFreqScaleSq: MLXArray,
        offsets: MLXArray
    ) {
        self.artifact = artifact
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.partialRotaryFactor = partialRotaryFactor
        self.ropeTheta = ropeTheta
        self.freqCount = freqCount
        self.rotaryPairs = rotaryPairs
        self.preparedStats = preparedStats
        self.sampledHeadsByLayer = sampledHeadsByLayer
        self.rotaryOmega = rotaryOmega
        self.omega = omega
        self.freqScaleSq = freqScaleSq
        self.rotaryFreqScaleSq = rotaryFreqScaleSq
        self.staticFreqScaleSq = staticFreqScaleSq
        self.offsets = offsets
    }

    var kvHeadGroupSize: Int {
        max(1, attentionHeads / max(1, kvHeads))
    }

    var staticPairs: Int {
        max(0, freqCount - rotaryPairs)
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
            let qMeanReal = MLXArray(rawStats.qMeanReal)
            let qMeanImag = MLXArray(rawStats.qMeanImag)
            let qAbsMean = MLXArray(rawStats.qAbsMean)
            let qMeanAbs = sqrt(qMeanReal * qMeanReal + qMeanImag * qMeanImag)
            sampledHeadsByLayer[headKey.layerIndex, default: []].append(headKey)
            observedLayerIndices.insert(headKey.layerIndex)
            preparedStats[headKey] = TriAttentionPreparedHeadStats(
                qMeanReal: qMeanReal,
                qMeanImag: qMeanImag,
                qAbsDelta: qAbsMean - qMeanAbs
            )
        }

        guard !preparedStats.isEmpty else {
            return nil
        }
        guard observedLayerIndices == validLayerIndices else {
            return nil
        }

        let rotaryOmega = makeRotaryOmega(
            rotaryPairs: rotaryPairs,
            ropeTheta: ropeTheta
        )
        let omega = makeOmega(
            headDim: headDim,
            rotaryPairs: rotaryPairs,
            ropeTheta: ropeTheta
        )
        let freqScaleSq = MLXArray.ones([freqCount], dtype: .float32)
        let rotaryFreqScaleSq = freqScaleSq[..<rotaryPairs]
        let staticFreqScaleSq =
            if rotaryPairs < freqCount {
                freqScaleSq[rotaryPairs...]
            } else {
                MLXArray.zeros([0], dtype: .float32)
            }
        let offsets = makeGeometricOffsets(maxLength: TriAttentionQwen35RuntimeState.offsetMaxLength)

        let headsPerLayer = sampledHeadsByLayer
            .keys.sorted()
            .map { "\($0):\(sampledHeadsByLayer[$0]!.count)" }
            .joined(separator: ",")
        triLog.info("runtime-state-created attnHeads=\(attentionHeads, privacy: .public) kvHeads=\(kvHeads, privacy: .public) headDim=\(headDim, privacy: .public) freqCount=\(freqCount, privacy: .public) rotaryPairs=\(rotaryPairs, privacy: .public) ropeTheta=\(ropeTheta, privacy: .public) fullAttnLayers=\(fullAttentionLayerIndices.count, privacy: .public) sampledHeadTotal=\(preparedStats.count, privacy: .public) headsPerLayer=\(headsPerLayer, privacy: .public) budget=\(configuration.budgetTokens, privacy: .public) prefixProtection=\(configuration.prefixProtectionMode.rawValue, privacy: .public)")

        return TriAttentionQwen35RuntimeState(
            artifact: artifact,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            headDim: headDim,
            partialRotaryFactor: partialRotaryFactor,
            ropeTheta: ropeTheta,
            freqCount: freqCount,
            rotaryPairs: rotaryPairs,
            preparedStats: preparedStats,
            sampledHeadsByLayer: sampledHeadsByLayer,
            rotaryOmega: rotaryOmega,
            omega: omega,
            freqScaleSq: freqScaleSq,
            rotaryFreqScaleSq: rotaryFreqScaleSq,
            staticFreqScaleSq: staticFreqScaleSq,
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
            triLog.info("prune skip=empty-or-mismatch caches=\(caches.count, privacy: .public) layerIndices=\(layerIndices.count, privacy: .public)")
            return
        }
        let runtimeCaches = caches.compactMap { $0 as? TriAttentionRuntimeCache }
        guard
            runtimeCaches.count == caches.count,
            let runtimeState = runtimeCaches.first?.runtimeState
        else {
            let nonRuntime = caches.count - runtimeCaches.count
            let hasRuntimeState = runtimeCaches.first?.runtimeState != nil
            triLog.info("prune skip=no-runtime-cache nonRuntime=\(nonRuntime, privacy: .public) hasRuntimeState=\(hasRuntimeState, privacy: .public)")
            return
        }

        let threshold =
            runtimeCaches[0].configuration.budgetTokens + TriAttentionQwen35RuntimeState.divideLength
        let retained = runtimeCaches[0].retainedTokenCount
        let offset = runtimeCaches[0].offset
        let budget = runtimeCaches[0].configuration.budgetTokens
        let protectedOffset = runtimeCaches[0].protectedPrefixOffset ?? -1
        guard retained >= threshold else {
            triLog.debug("prune skip=below-threshold retained=\(retained, privacy: .public) threshold=\(threshold, privacy: .public) offset=\(offset, privacy: .public)")
            return
        }

        triLog.info("prune enter retained=\(retained, privacy: .public) threshold=\(threshold, privacy: .public) budget=\(budget, privacy: .public) offset=\(offset, privacy: .public) protectedOffset=\(protectedOffset, privacy: .public) fullAttnLayers=\(runtimeCaches.count, privacy: .public)")

        let t0 = CFAbsoluteTimeGetCurrent()
        let roundStart = runtimeCaches[0].offset
        let keepIndices = sharedPerHeadKeepIndices(
            caches: runtimeCaches,
            layerIndices: layerIndices,
            runtimeState: runtimeState,
            budget: budget,
            roundStart: roundStart
        )
        let scoringMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

        guard let keepIndices else {
            triLog.info("prune done result=no-indices scoringMs=\(String(format: "%.2f", scoringMs), privacy: .public)")
            return
        }

        let t1 = CFAbsoluteTimeGetCurrent()
        let keepCount = keepIndices.dim(-1)
        for cache in runtimeCaches {
            cache.applyKeepIndices(keepIndices)
        }
        let applyMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0
        let totalMs = scoringMs + applyMs
        let retainedAfter = runtimeCaches[0].retainedTokenCount
        let dropped = retained - retainedAfter
        let isNoOp = dropped == 0
        triLog.info("prune done keepCount=\(keepCount, privacy: .public) retainedBefore=\(retained, privacy: .public) retainedAfter=\(retainedAfter, privacy: .public) dropped=\(dropped, privacy: .public) noop=\(isNoOp, privacy: .public) scoringMs=\(String(format: "%.2f", scoringMs), privacy: .public) applyMs=\(String(format: "%.2f", applyMs), privacy: .public) totalMs=\(String(format: "%.2f", totalMs), privacy: .public)")
    }

    private static func sharedPerHeadKeepIndices(
        caches: [TriAttentionRuntimeCache],
        layerIndices: [Int],
        runtimeState: TriAttentionQwen35RuntimeState,
        budget: Int,
        roundStart: Int
    ) -> MLXArray? {
        var perLayerScores: [MLXArray] = []
        let scoreStart = CFAbsoluteTimeGetCurrent()

        for (cache, layerIndex) in zip(caches, layerIndices) {
            let layerStart = CFAbsoluteTimeGetCurrent()
            guard let layerScores = layerScores(
                cache: cache,
                layerIndex: layerIndex,
                runtimeState: runtimeState,
                roundStart: roundStart
            ) else {
                triLog.debug("layerScores skip layer=\(layerIndex, privacy: .public) retained=\(cache.retainedTokenCount, privacy: .public)")
                continue
            }
            perLayerScores.append(layerScores)
            let layerMs = (CFAbsoluteTimeGetCurrent() - layerStart) * 1000.0
            let headCount = runtimeState.sampledHeadsByLayer[layerIndex]?.count ?? 0
            triLog.debug("layerScores done layer=\(layerIndex, privacy: .public) retained=\(cache.retainedTokenCount, privacy: .public) heads=\(headCount, privacy: .public) tookMs=\(String(format: "%.2f", layerMs), privacy: .public)")
        }

        let scoringMs = (CFAbsoluteTimeGetCurrent() - scoreStart) * 1000.0

        guard let firstLayerScores = perLayerScores.first else {
            triLog.info("keep-compute no-scores perLayerScoresEmpty=true scoringMs=\(String(format: "%.2f", scoringMs), privacy: .public)")
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
        let protectedCount = protectedPrefixRetainedCount(cache: caches[0])
        let unclampedKeep = budget + protectedCount
        let keepCount = min(tokenCount, unclampedKeep)
        let isNoOp = keepCount >= tokenCount
        triLog.info("keep-compute tokenCount=\(tokenCount, privacy: .public) protected=\(protectedCount, privacy: .public) budget=\(budget, privacy: .public) unclampedKeep=\(unclampedKeep, privacy: .public) keepCount=\(keepCount, privacy: .public) noop=\(isNoOp, privacy: .public) scoredLayers=\(perLayerScores.count, privacy: .public) scoringMs=\(String(format: "%.2f", scoringMs), privacy: .public)")

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
        var forcedKeepMasksByHead: [MLXArray] = []
        var preparedKVHeads: [Int: TriAttentionPreparedKVHeadScoreContext] = [:]
        var forcedKeepMasksByKVHead: [Int: MLXArray] = [:]
        let windowStart = roundStart - min(
            TriAttentionQwen35RuntimeState.windowSize,
            max(0, cache.configuration.budgetTokens - 1)
        )

        for headKey in sampledHeads {
            guard let stats = runtimeState.preparedStats[headKey] else { continue }
            let kvHead = min(runtimeState.kvHeads - 1, headKey.headIndex / runtimeState.kvHeadGroupSize)
            let preparedKVHead: TriAttentionPreparedKVHeadScoreContext
            if let cachedPreparedKVHead = preparedKVHeads[kvHead] {
                preparedKVHead = cachedPreparedKVHead
            } else {
                let computedPreparedKVHead = prepareKVHeadScoreContext(
                    rotatedKeys: retainedKeys[0, kvHead],
                    keyPositions: retainedPositions[0, kvHead],
                    roundStart: roundStart,
                    runtimeState: runtimeState
                )
                preparedKVHeads[kvHead] = computedPreparedKVHead
                preparedKVHead = computedPreparedKVHead
            }
            let scores = scoreHead(
                stats: stats,
                preparedKVHead: preparedKVHead,
                runtimeState: runtimeState
            )
            rawScoresByHead.append(scores)

            let forcedKeepMask: MLXArray
            if let cachedForcedKeepMask = forcedKeepMasksByKVHead[kvHead] {
                forcedKeepMask = cachedForcedKeepMask
            } else {
                let positions = retainedPositions[0, kvHead]
                var computedForcedKeepMask =
                    positions.asType(.int32) .>= MLXArray(Int32(max(0, windowStart)))
                if let protectedPrefixOffset = cache.protectedPrefixOffset {
                    computedForcedKeepMask = computedForcedKeepMask | (
                        positions.asType(.int32)
                            .< MLXArray(Int32(clamping: protectedPrefixOffset))
                    )
                }
                forcedKeepMasksByKVHead[kvHead] = computedForcedKeepMask
                forcedKeepMask = computedForcedKeepMask
            }
            forcedKeepMasksByHead.append(forcedKeepMask)
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

        let forcedKeepMasks = stacked(forcedKeepMasksByHead, axis: 0)
        normalizedScores = MLX.where(
            forcedKeepMasks,
            MLXArray(Float.infinity),
            normalizedScores
        )

        return aggregatePerKVHeadScores(
            normalizedScores: normalizedScores,
            sampledHeads: sampledHeads,
            runtimeState: runtimeState
        )
    }

    static func aggregatePerKVHeadScores(
        normalizedScores: MLXArray,
        sampledHeads: [TriAttentionCalibrationHeadKey],
        runtimeState: TriAttentionQwen35RuntimeState
    ) -> MLXArray {
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

    static func scoreHeadForTesting(
        stats: TriAttentionPreparedHeadStats,
        rotatedKeys: MLXArray,
        keyPositions: MLXArray,
        roundStart: Int,
        runtimeState: TriAttentionQwen35RuntimeState
    ) -> MLXArray {
        scoreHead(
            stats: stats,
            preparedKVHead: prepareKVHeadScoreContext(
                rotatedKeys: rotatedKeys,
                keyPositions: keyPositions,
                roundStart: roundStart,
                runtimeState: runtimeState
            ),
            runtimeState: runtimeState
        )
    }

    private static func scoreHead(
        stats: TriAttentionPreparedHeadStats,
        preparedKVHead: TriAttentionPreparedKVHeadScoreContext,
        runtimeState: TriAttentionQwen35RuntimeState
    ) -> MLXArray {
        let rotaryPairs = runtimeState.rotaryPairs
        let qMeanRealRotary = stats.qMeanReal[ ..<rotaryPairs]
        let qMeanImagRotary = stats.qMeanImag[ ..<rotaryPairs]
        let qAbsDeltaRotary = stats.qAbsDelta[ ..<rotaryPairs]

        let rotaryRelativeReal =
            preparedKVHead.keyRealRotary * qMeanRealRotary
            + preparedKVHead.keyImagRotary * qMeanImagRotary
        let rotaryRelativeImag =
            preparedKVHead.keyRealRotary * qMeanImagRotary
            - preparedKVHead.keyImagRotary * qMeanRealRotary

        let rotaryTrigTerm =
            (
                expandedDimensions(rotaryRelativeReal, axis: 1) * preparedKVHead.cosPhase
                - expandedDimensions(rotaryRelativeImag, axis: 1) * preparedKVHead.sinPhase
            ) * runtimeState.rotaryFreqScaleSq
        var scores = mean(rotaryTrigTerm.sum(axis: -1), axis: -1)
        var additive = (
            preparedKVHead.keyAbsRotary * qAbsDeltaRotary * runtimeState.rotaryFreqScaleSq
        ).sum(axis: -1)

        if runtimeState.staticPairs > 0 {
            let qMeanRealStatic = stats.qMeanReal[rotaryPairs...]
            let qMeanImagStatic = stats.qMeanImag[rotaryPairs...]
            let qAbsDeltaStatic = stats.qAbsDelta[rotaryPairs...]
            let staticRelativeReal =
                preparedKVHead.keyRealStatic * qMeanRealStatic
                + preparedKVHead.keyImagStatic * qMeanImagStatic
            scores = scores + (staticRelativeReal * runtimeState.staticFreqScaleSq).sum(axis: -1)
            additive = additive + (
                preparedKVHead.keyAbsStatic * qAbsDeltaStatic * runtimeState.staticFreqScaleSq
            ).sum(axis: -1)
        }

        return scores + additive
    }

    private static func protectedPrefixRetainedCount(cache: TriAttentionRuntimeCache) -> Int {
        guard
            let protectedPrefixOffset = cache.protectedPrefixOffset,
            let retainedPositions = cache.retainedPositions
        else {
            return 0
        }

        let protectedCounts = sum(
            (retainedPositions[0].asType(.int32) .< MLXArray(Int32(clamping: protectedPrefixOffset)))
                .asType(.int32),
            axis: -1
        )
        return Int(protectedCounts.max().item(Int32.self))
    }

    private static func prepareKVHeadScoreContext(
        rotatedKeys: MLXArray,
        keyPositions: MLXArray,
        roundStart: Int,
        runtimeState: TriAttentionQwen35RuntimeState
    ) -> TriAttentionPreparedKVHeadScoreContext {
        let rotatedReal = rotatedKeys[.ellipsis, ..<runtimeState.freqCount]
        let rotatedImag = rotatedKeys[.ellipsis, runtimeState.freqCount...]

        let rotaryReal = rotatedReal[.ellipsis, ..<runtimeState.rotaryPairs]
        let rotaryImag = rotatedImag[.ellipsis, ..<runtimeState.rotaryPairs]
        let ropePhase = expandedDimensions(keyPositions.asType(.float32), axis: -1)
            * runtimeState.rotaryOmega
        let cosRopePhase = cos(ropePhase)
        let sinRopePhase = sin(ropePhase)
        let keyRealRotary = rotaryReal * cosRopePhase + rotaryImag * sinRopePhase
        let keyImagRotary = rotaryImag * cosRopePhase - rotaryReal * sinRopePhase
        let keyAbsRotary = sqrt(
            keyRealRotary * keyRealRotary + keyImagRotary * keyImagRotary
        )

        let keyRealStatic: MLXArray
        let keyImagStatic: MLXArray
        let keyAbsStatic: MLXArray
        if runtimeState.staticPairs > 0 {
            keyRealStatic = rotatedReal[.ellipsis, runtimeState.rotaryPairs...]
            keyImagStatic = rotatedImag[.ellipsis, runtimeState.rotaryPairs...]
            keyAbsStatic = sqrt(
                keyRealStatic * keyRealStatic + keyImagStatic * keyImagStatic
            )
        } else {
            let tokenCount = keyPositions.dim(0)
            keyRealStatic = MLXArray.zeros([tokenCount, 0], dtype: rotatedKeys.dtype)
            keyImagStatic = MLXArray.zeros([tokenCount, 0], dtype: rotatedKeys.dtype)
            keyAbsStatic = MLXArray.zeros([tokenCount, 0], dtype: rotatedKeys.dtype)
        }

        let baseDelta = MLXArray(Float(roundStart), dtype: .float32) - keyPositions.asType(.float32)
        let deltaGrid = expandedDimensions(baseDelta, axis: -1)
            + expandedDimensions(runtimeState.offsets, axis: 0)
        let phase = expandedDimensions(deltaGrid, axis: -1) * runtimeState.rotaryOmega

        return TriAttentionPreparedKVHeadScoreContext(
            keyRealRotary: keyRealRotary,
            keyImagRotary: keyImagRotary,
            keyRealStatic: keyRealStatic,
            keyImagStatic: keyImagStatic,
            keyAbsRotary: keyAbsRotary,
            keyAbsStatic: keyAbsStatic,
            cosPhase: cos(phase),
            sinPhase: sin(phase)
        )
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

    private static func makeRotaryOmega(
        rotaryPairs: Int,
        ropeTheta: Float
    ) -> MLXArray {
        MLXArray((0 ..< rotaryPairs).map { pairIndex in
            pow(ropeTheta, -Float(pairIndex) / Float(max(1, rotaryPairs)))
        })
    }

    private static func makeOmega(
        headDim: Int,
        rotaryPairs: Int,
        ropeTheta: Float
    ) -> MLXArray {
        let freqCount = headDim / 2
        let rotaryOmega = makeRotaryOmega(
            rotaryPairs: rotaryPairs,
            ropeTheta: ropeTheta
        ).asArray(Float.self)
        let padded = rotaryOmega + Array(repeating: Float.zero, count: max(0, freqCount - rotaryPairs))
        return MLXArray(padded)
    }
}
