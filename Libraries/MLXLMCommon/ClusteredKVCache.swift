// Copyright © 2026 Tesseract Agent

import Foundation
import MLX
import MLXNN

// MARK: - Protocol

/// Protocol for KV caches that perform clustered approximate attention.
///
/// During generation, centroid K/V summaries replace the full middle context:
/// sink tokens + centroid K/V + recent window. This reduces memory bandwidth
/// by ~340x at 262K context.
///
/// Integration follows the same pattern as `QuantizedKVCacheProtocol`:
/// `attentionWithCacheUpdate()` detects this protocol via `as?` and
/// routes to `clusteredAttention()` instead of full attention.
public protocol ClusteredKVCacheProtocol: KVCache {
    /// Total number of clusters (K).
    var numClusters: Int { get }

    /// Number of recent tokens always included in exact attention
    var recentWindow: Int { get }

    /// Perform attention using clustered retrieval.
    ///
    /// Below `clusterThreshold`, this falls back to standard full attention.
    /// Above threshold, concatenates sink + centroid K/V + recent/chunk
    /// and runs a single fused SDPA call.
    ///
    /// - Parameters:
    ///   - queries: Query tensor [B, nQHeads, 1, D] (generation) or [B, nQHeads, S, D] (prefill)
    ///   - scale: Attention scale factor (1/sqrt(headDim))
    ///   - mask: Attention mask from the caller
    /// - Returns: Attention output [B, nQHeads, S, D]
    func clusteredAttention(
        queries: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray
}

// MARK: - Implementation

/// Clustered KV cache that reduces bandwidth by attending to centroid
/// summaries instead of reading the full KV sequence.
///
/// Below `clusterThreshold` tokens, behaves identically to `KVCacheSimple`.
/// Above threshold, clusters KV entries and builds centroid key/value means.
/// Attention concatenates:
/// - Sink tokens (first S=4 tokens, always exact)
/// - Centroid K/V (256 cluster means)
/// - Recent window (last W tokens) or chunk (last S tokens for prefill)
///
/// Single fused `MLXFast.scaledDotProductAttention` call — no manual matmul.
public class ClusteredKVCache: BaseKVCache {

    // MARK: - Configuration

    /// Number of K-means clusters (fixed)
    public let numClusters: Int

    /// Recent tokens always included in exact attention
    public let recentWindow: Int

    /// Number of sink tokens (first N) always included
    public let sinkTokens: Int

    /// Minimum offset before clustering activates
    public let clusterThreshold: Int

    /// Re-cluster every N new tokens
    public let reclusterInterval: Int

    // MARK: - Full KV Storage (same as KVCacheSimple)

    internal var keys: MLXArray?
    internal var values: MLXArray?
    private var step = 256

    // MARK: - Cluster Metadata

    /// Cluster centroids (key means): [nKVHeads, K, headDim] (float32)
    private var centroids: MLXArray?

    /// Centroid values (value means): [nKVHeads, K, headDim] (float32)
    private var centroidValues: MLXArray?

    /// Cluster assignment per token: [nKVHeads, L] (int32)
    private var assignments: MLXArray?

    /// Number of tokens in each cluster: [nKVHeads, K] (int32)
    private var clusterCounts: MLXArray?

    /// Running sum of keys per cluster: [nKVHeads, K, headDim] (float32)
    private var clusterSums: MLXArray?

    /// Running sum of values per cluster: [nKVHeads, K, headDim] (float32)
    private var clusterValueSums: MLXArray?

    /// Whether initial clustering has been performed
    private var clustersInitialized = false

    /// Tokens processed since last full re-cluster
    private var tokensSinceRecluster = 0

    // MARK: - Init

    public init(
        numClusters: Int = 256,
        recentWindow: Int = 2048,
        sinkTokens: Int = 4,
        clusterThreshold: Int = 4096,
        reclusterInterval: Int = 1024
    ) {
        self.numClusters = numClusters
        self.recentWindow = recentWindow
        self.sinkTokens = sinkTokens
        self.clusterThreshold = clusterThreshold
        self.reclusterInterval = reclusterInterval
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    // MARK: - KVCache Protocol

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset

        // Grow storage (same logic as KVCacheSimple)
        let reset =
            if let currentKeys = self.keys, (previous + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += keys.dim(2)

        self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.values?[.ellipsis, previous ..< self.offset, 0...] = values

        // Update cluster metadata for new tokens
        let newTokenCount = keys.dim(2)
        if self.offset >= clusterThreshold {
            if !clustersInitialized {
                initializeClusters()
            } else {
                assignNewTokens(from: previous, count: newTokenCount)
                tokensSinceRecluster += newTokenCount
                if tokensSinceRecluster >= reclusterInterval {
                    recluster()
                    tokensSinceRecluster = 0
                }
            }
        }

        // Return full K/V (protocol compliance)
        let returnedKeys = self.keys![.ellipsis, ..<self.offset, 0...]
        let returnedValues = self.values![.ellipsis, ..<self.offset, 0...]
        return (returnedKeys, returnedValues)
    }

    // MARK: - Serialization

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            let k: MLXArray
            let v: MLXArray
            if offset == keys.dim(2) {
                k = keys
                v = values
            } else {
                k = keys[.ellipsis, ..<offset, 0...]
                v = values[.ellipsis, ..<offset, 0...]
            }

            // If clusters initialized, include all cluster metadata (7 arrays)
            if let centroids = centroids, let assignments = assignments,
                let clusterCounts = clusterCounts,
                let clusterValueSums = clusterValueSums,
                let centroidValues = centroidValues
            {
                return [
                    k, v, centroids, assignments, clusterCounts, clusterValueSums, centroidValues,
                ]
            }
            return [k, v]
        }
        set {
            if newValue.count >= 2 {
                self.keys = newValue[0]
                self.values = newValue[1]
                self.offset = self.keys!.dim(2)
            }
            if newValue.count >= 7 {
                // New format: 7 arrays including value centroids
                self.centroids = newValue[2]
                self.assignments = newValue[3]
                self.clusterCounts = newValue[4]
                self.clusterValueSums = newValue[5]
                self.centroidValues = newValue[6]
                self.clustersInitialized = true
                // Rebuild key sums from centroids * counts
                let countsF = self.clusterCounts!.asType(.float32)[.ellipsis, .newAxis]
                self.clusterSums = self.centroids! * countsF
            } else if newValue.count >= 5 {
                // Old format (no value sums): mark uninitialized to trigger recomputation
                self.centroids = newValue[2]
                self.assignments = newValue[3]
                self.clusterCounts = newValue[4]
                self.clustersInitialized = false
                let countsF = self.clusterCounts!.asType(.float32)[.ellipsis, .newAxis]
                self.clusterSums = self.centroids! * countsF
            }
        }
    }

    public override var metaState: [String] {
        get {
            [
                "\(offset)",  // 0
                "\(numClusters)",  // 1
                "\(recentWindow)",  // 2
                "\(sinkTokens)",  // 3
                "\(clusterThreshold)",  // 4
                "\(reclusterInterval)",  // 5
                clustersInitialized ? "1" : "0",  // 6
                "\(tokensSinceRecluster)",  // 7
            ]
        }
        set {
            guard newValue.count >= 8 else { return }
            offset = Int(newValue[0]) ?? 0
            if newValue.count >= 9 {
                // Old format (9+ elements): clustersInitialized at [7], tokensSinceRecluster at [8]
                clustersInitialized = newValue[7] == "1"
                tokensSinceRecluster = Int(newValue[8]) ?? 0
            } else {
                // New format (8 elements)
                clustersInitialized = newValue[6] == "1"
                tokensSinceRecluster = Int(newValue[7]) ?? 0
            }
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed

        // Trim cluster assignments if present
        if let assignments = self.assignments, offset < assignments.dim(1) {
            self.assignments = assignments[.ellipsis, ..<offset]
        }

        return trimmed
    }

    // MARK: - Vectorized Clustering Helpers

    /// Build one-hot encoding: [nKVHeads, L, K] from assignments [nKVHeads, L]
    private func buildOneHot(_ assignments: MLXArray, K: Int) -> MLXArray {
        let clusterRange = MLXArray(0 ..< Int32(K)).reshaped(1, 1, K)
        let assignExpanded = assignments.expandedDimensions(axis: -1)
        return (assignExpanded .== clusterRange).asType(.float32)
    }

    /// Compute counts, key sums, value sums, and centroids from one-hot assignments.
    private func computeClusterStats(
        oneHot: MLXArray,
        keysPerHead: MLXArray,
        valuesPerHead: MLXArray
    ) {
        // oneHot: [nKVHeads, L, K], keys/values: [nKVHeads, L, headDim]
        let oneHotT = oneHot.transposed(0, 2, 1)  // [nKVHeads, K, L]

        self.clusterCounts = oneHot.asType(.int32).sum(axis: 1)  // [nKVHeads, K]
        self.clusterSums = oneHotT.matmul(keysPerHead)  // [nKVHeads, K, headDim]
        self.clusterValueSums = oneHotT.matmul(valuesPerHead)  // [nKVHeads, K, headDim]

        let countsF = self.clusterCounts!.asType(.float32).expandedDimensions(axis: -1)
        let safeCounts = MLX.maximum(countsF, MLXArray(Float(1.0)))
        self.centroids = self.clusterSums! / safeCounts
        self.centroidValues = self.clusterValueSums! / safeCounts
    }

    // MARK: - Clustering

    /// Initialize clusters using K-means++ on current key data.
    private func initializeClusters() {
        guard let keys = self.keys, let values = self.values else { return }

        let allKeys = keys[.ellipsis, ..<offset, 0...].asType(.float32)
        let allValues = values[.ellipsis, ..<offset, 0...].asType(.float32)
        let nKVHeads = allKeys.dim(1)
        let L = allKeys.dim(2)

        // Work on batch 0 (B=1 for generation)
        let keysPerHead = allKeys[0]  // [nKVHeads, L, headDim]
        let valuesPerHead = allValues[0]  // [nKVHeads, L, headDim]

        let K = min(numClusters, L)

        // K-means++ initialization: pick first centroid randomly, then distance-weighted
        var centroidsList = [MLXArray]()

        for h in 0 ..< nKVHeads {
            let headKeys = keysPerHead[h]  // [L, headDim]

            // First centroid: token 0 (deterministic for reproducibility)
            var cents = headKeys[0 ..< 1]  // [1, headDim]

            for _ in 1 ..< K {
                let dists = headKeys.matmul(cents.transposed())  // [L, numCents]
                let maxSim = dists.max(axis: -1)  // [L]
                let weights = -maxSim
                let shiftedWeights = weights - weights.max()
                let probs = MLX.exp(shiftedWeights)
                let nextIdx = probs.argMax()
                let nextCent = headKeys[nextIdx]  // [headDim]
                cents = concatenated([cents, expandedDimensions(nextCent, axis: 0)], axis: 0)
            }
            centroidsList.append(expandedDimensions(cents, axis: 0))  // [1, K, headDim]
        }

        self.centroids = concatenated(centroidsList, axis: 0)  // [nKVHeads, K, headDim]

        // Assign all tokens to nearest centroid
        let similarity = keysPerHead.matmul(self.centroids!.transposed(0, 2, 1))
        self.assignments = similarity.argMax(axis: -1).asType(.int32)  // [nKVHeads, L]

        // Vectorized cluster stats
        let oneHot = buildOneHot(self.assignments!, K: K)
        computeClusterStats(oneHot: oneHot, keysPerHead: keysPerHead, valuesPerHead: valuesPerHead)

        clustersInitialized = true
        tokensSinceRecluster = 0
    }

    /// Assign newly added tokens to nearest centroid and update running means.
    private func assignNewTokens(from startOffset: Int, count: Int) {
        guard let keys = self.keys, let values = self.values,
            let centroids = self.centroids
        else { return }

        let newKeys = keys[0, .ellipsis, startOffset ..< (startOffset + count), 0...]
            .asType(.float32)  // [nKVHeads, count, headDim]
        let newValues = values[0, .ellipsis, startOffset ..< (startOffset + count), 0...]
            .asType(.float32)  // [nKVHeads, count, headDim]

        // Compute similarity to centroids: [nKVHeads, count, K]
        let similarity = newKeys.matmul(centroids.transposed(0, 2, 1))
        let newAssignments = similarity.argMax(axis: -1).asType(.int32)  // [nKVHeads, count]

        // Append to assignments
        if let existingAssignments = self.assignments {
            self.assignments = concatenated([existingAssignments, newAssignments], axis: 1)
        } else {
            self.assignments = newAssignments
        }

        // Vectorized incremental update
        let K = centroids.dim(1)
        let oneHot = buildOneHot(newAssignments, K: K)
        let oneHotT = oneHot.transposed(0, 2, 1)

        let newCounts = oneHot.asType(.int32).sum(axis: 1)
        self.clusterCounts = self.clusterCounts! + newCounts

        let newKeySums = oneHotT.matmul(newKeys)
        self.clusterSums = self.clusterSums! + newKeySums

        let newValueSums = oneHotT.matmul(newValues)
        self.clusterValueSums = self.clusterValueSums! + newValueSums

        // Refresh centroids from sums/counts
        let countsF = self.clusterCounts!.asType(.float32).expandedDimensions(axis: -1)
        let safeCounts = MLX.maximum(countsF, MLXArray(Float(1.0)))
        self.centroids = self.clusterSums! / safeCounts
        self.centroidValues = self.clusterValueSums! / safeCounts
    }

    /// Full K-means re-clustering (3 iterations) to correct centroid drift.
    private func recluster() {
        guard let keys = self.keys, let values = self.values else { return }

        let allKeys = keys[0, .ellipsis, ..<offset, 0...].asType(.float32)
        let allValues = values[0, .ellipsis, ..<offset, 0...].asType(.float32)
        let K = numClusters

        var currentCentroids = self.centroids!

        for _ in 0 ..< 3 {
            let similarity = allKeys.matmul(currentCentroids.transposed(0, 2, 1))
            let newAssignments = similarity.argMax(axis: -1).asType(.int32)

            let oneHot = buildOneHot(newAssignments, K: K)
            let oneHotT = oneHot.transposed(0, 2, 1)

            let newCounts = oneHot.asType(.int32).sum(axis: 1)
            let newKeySums = oneHotT.matmul(allKeys)
            let newValueSums = oneHotT.matmul(allValues)

            let countsF = newCounts.asType(.float32).expandedDimensions(axis: -1)
            let safeCounts = MLX.maximum(countsF, MLXArray(Float(1.0)))
            currentCentroids = newKeySums / safeCounts

            self.assignments = newAssignments
            self.clusterCounts = newCounts
            self.clusterSums = newKeySums
            self.clusterValueSums = newValueSums
        }

        self.centroids = currentCentroids
        let countsF = self.clusterCounts!.asType(.float32).expandedDimensions(axis: -1)
        let safeCounts = MLX.maximum(countsF, MLXArray(Float(1.0)))
        self.centroidValues = self.clusterValueSums! / safeCounts
    }

    // MARK: - Mask

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        // For clustered attention, mask handling is done inside clusteredAttention()
        // For short contexts (below threshold), use standard mask
        if n == 1 {
            return .none
        }
        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }
}

// MARK: - ClusteredKVCacheProtocol Conformance

extension ClusteredKVCache: ClusteredKVCacheProtocol {

    public func clusteredAttention(
        queries: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        guard let keys = self.keys, let values = self.values else {
            fatalError("ClusteredKVCache.clusteredAttention called before any update()")
        }

        let allKeys = keys[.ellipsis, ..<offset, 0...]
        let allValues = values[.ellipsis, ..<offset, 0...]
        let L = offset

        // Below threshold: standard full attention (identical to KVCacheSimple path)
        if L <= clusterThreshold || !clustersInitialized {
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: allKeys,
                values: allValues,
                scale: scale,
                mask: mask
            )
        }

        // --- Centroid attention path ---
        let S = queries.dim(2)
        let sinkEnd = min(sinkTokens, L)
        let cacheDtype = allKeys.dtype

        // Centroid K/V: [1, nKVHeads, K, headDim]
        let centK = expandedDimensions(centroids!, axis: 0).asType(cacheDtype)
        let centV = expandedDimensions(centroidValues!, axis: 0).asType(cacheDtype)

        // Sink K/V
        let sinkK = allKeys[.ellipsis, ..<sinkEnd, 0...]
        let sinkV = allValues[.ellipsis, ..<sinkEnd, 0...]

        let K = centroids!.dim(1)

        if S == 1 {
            // Generation: sink + centroids + recent
            let recentStart = max(sinkEnd, L - recentWindow)

            // If recent window covers everything, full attention is fine
            if recentStart <= sinkEnd {
                return MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: allKeys,
                    values: allValues,
                    scale: scale,
                    mask: .none
                )
            }

            let recentK = allKeys[.ellipsis, recentStart..., 0...]
            let recentV = allValues[.ellipsis, recentStart..., 0...]
            let combinedK = concatenated([sinkK, centK, recentK], axis: 2)
            let combinedV = concatenated([sinkV, centV, recentV], axis: 2)

            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: combinedK,
                values: combinedV,
                scale: scale,
                mask: .none
            )
        } else {
            // Prefill: sink + centroids + chunk (last S tokens)
            let chunkK = allKeys[.ellipsis, (L - S) ..< L, 0...]
            let chunkV = allValues[.ellipsis, (L - S) ..< L, 0...]
            let combinedK = concatenated([sinkK, centK, chunkK], axis: 2)
            let combinedV = concatenated([sinkV, centV, chunkV], axis: 2)

            // Mask: [1, 1, S, sinkEnd+K+S]
            // First sinkEnd+K cols: all-true (all queries attend to sink + centroids)
            // Last S cols: lower-triangular causal (within the chunk)
            let prefixOnes = MLXArray.ones([1, 1, S, sinkEnd + K], dtype: .bool)
            let causal = MLXArray.tri(S, m: S, k: 0, dtype: .bool).reshaped(1, 1, S, S)
            let fullMask = concatenated([prefixOnes, causal], axis: -1)

            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: combinedK,
                values: combinedV,
                scale: scale,
                mask: .array(fullMask)
            )
        }
    }
}
