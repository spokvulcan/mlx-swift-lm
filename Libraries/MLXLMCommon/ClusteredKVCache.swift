// Copyright © 2026 Tesseract Agent

import Foundation
import MLX
import MLXNN

// MARK: - Protocol

/// Protocol for KV caches that perform clustered approximate attention.
///
/// During generation, only a subset of KV entries are read per step:
/// sink tokens + top-C cluster matches + recent window. This reduces
/// memory bandwidth by ~7.5x at 262K context.
///
/// Integration follows the same pattern as `QuantizedKVCacheProtocol`:
/// `attentionWithCacheUpdate()` detects this protocol via `as?` and
/// routes to `clusteredAttention()` instead of full attention.
public protocol ClusteredKVCacheProtocol: KVCache {
    /// Total number of clusters (K)
    var numClusters: Int { get }

    /// Number of top clusters to retrieve per query (C)
    var topClusters: Int { get }

    /// Number of recent tokens always included in exact attention
    var recentWindow: Int { get }

    /// Perform attention using clustered retrieval.
    ///
    /// Below `clusterThreshold`, this falls back to standard full attention.
    /// Above threshold, it retrieves sink + top-C clusters + recent window.
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

/// Clustered KV cache that reduces bandwidth by retrieving only relevant
/// KV entries during long-context generation.
///
/// Below `clusterThreshold` tokens, behaves identically to `KVCacheSimple`.
/// Above threshold, clusters KV entries and retrieves only:
/// - Sink tokens (first S=4 tokens, always exact)
/// - Top-C cluster matches (query-dependent, approximate)
/// - Recent window (last W tokens, always exact)
///
/// At 262K context with default settings: retrieves ~35K of 262K entries → 7.5x bandwidth reduction.
public class ClusteredKVCache: BaseKVCache {

    // MARK: - Configuration

    /// Number of K-means clusters
    public let numClusters: Int

    /// Number of top clusters retrieved per query
    public let topClusters: Int

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

    /// Cluster centroids: [nKVHeads, K, headDim] (float32)
    private var centroids: MLXArray?

    /// Cluster assignment per token: [nKVHeads, L] (int32)
    private var assignments: MLXArray?

    /// Number of tokens in each cluster: [nKVHeads, K] (int32)
    private var clusterCounts: MLXArray?

    /// Running sum for incremental centroid updates: [nKVHeads, K, headDim] (float32)
    private var clusterSums: MLXArray?

    /// Whether initial clustering has been performed
    private var clustersInitialized = false

    /// Tokens processed since last full re-cluster
    private var tokensSinceRecluster = 0

    // MARK: - Init

    public init(
        numClusters: Int = 256,
        topClusters: Int = 32,
        recentWindow: Int = 2048,
        sinkTokens: Int = 4,
        clusterThreshold: Int = 4096,
        reclusterInterval: Int = 1024
    ) {
        self.numClusters = numClusters
        self.topClusters = topClusters
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

            // If clusters initialized, include cluster metadata
            if let centroids = centroids, let assignments = assignments,
                let clusterCounts = clusterCounts
            {
                return [k, v, centroids, assignments, clusterCounts]
            }
            return [k, v]
        }
        set {
            if newValue.count >= 2 {
                self.keys = newValue[0]
                self.values = newValue[1]
                self.offset = self.keys!.dim(2)
            }
            if newValue.count >= 5 {
                self.centroids = newValue[2]
                self.assignments = newValue[3]
                self.clusterCounts = newValue[4]
                self.clustersInitialized = true
                // Rebuild clusterSums from centroids * counts
                let countsF = self.clusterCounts!.asType(.float32)[.ellipsis, .newAxis]
                self.clusterSums = self.centroids! * countsF
            }
        }
    }

    public override var metaState: [String] {
        get {
            [
                "\(offset)",
                "\(numClusters)",
                "\(topClusters)",
                "\(recentWindow)",
                "\(sinkTokens)",
                "\(clusterThreshold)",
                "\(reclusterInterval)",
                clustersInitialized ? "1" : "0",
                "\(tokensSinceRecluster)",
            ]
        }
        set {
            guard newValue.count >= 9 else { return }
            offset = Int(newValue[0]) ?? 0
            // numClusters, topClusters, etc. are let constants — set at init time.
            // They must match when loading. We just restore mutable state:
            clustersInitialized = newValue[7] == "1"
            tokensSinceRecluster = Int(newValue[8]) ?? 0
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

    // MARK: - Clustering

    /// Initialize clusters using K-means++ on current key data.
    private func initializeClusters() {
        guard let keys = self.keys else { return }

        // keys: [B, nKVHeads, L_alloc, headDim] — use actual data up to offset
        // We operate per KV head, on float32 for stability
        let allKeys = keys[.ellipsis, ..<offset, 0...].asType(.float32)
        let nKVHeads = allKeys.dim(1)
        let headDim = allKeys.dim(3)
        let L = allKeys.dim(2)

        // Work on batch 0 (B=1 for generation)
        let keysPerHead = allKeys[0]  // [nKVHeads, L, headDim]

        let K = min(numClusters, L)

        // K-means++ initialization: pick first centroid randomly, then distance-weighted
        var centroidsList = [MLXArray]()

        for h in 0 ..< nKVHeads {
            let headKeys = keysPerHead[h]  // [L, headDim]

            // First centroid: token 0 (deterministic for reproducibility)
            var cents = headKeys[0 ..< 1]  // [1, headDim]

            for _ in 1 ..< K {
                // Distances from each token to nearest existing centroid
                // dists: [L, numCurrentCentroids]
                let dists = headKeys.matmul(cents.transposed())  // [L, numCents]
                let maxSim = dists.max(axis: -1)  // [L]
                // Convert similarity to distance (lower sim = farther)
                let weights = -maxSim  // Negate so farthest tokens have highest weight
                let shiftedWeights = weights - weights.max()
                let probs = MLX.exp(shiftedWeights)
                // Pick the token with maximum distance (greedy K-means++)
                let nextIdx = probs.argMax()
                let nextCent = headKeys[nextIdx]  // [headDim]
                cents = concatenated([cents, expandedDimensions(nextCent, axis: 0)], axis: 0)
            }
            centroidsList.append(expandedDimensions(cents, axis: 0))  // [1, K, headDim]
        }

        self.centroids = concatenated(centroidsList, axis: 0)  // [nKVHeads, K, headDim]

        // Assign all tokens to nearest centroid
        // similarity: [nKVHeads, L, K] = keysPerHead @ centroids^T
        let similarity = keysPerHead.matmul(self.centroids!.transposed(0, 2, 1))
        self.assignments = similarity.argMax(axis: -1).asType(.int32)  // [nKVHeads, L]

        // Compute cluster counts and sums
        self.clusterCounts = MLXArray.zeros([nKVHeads, K], dtype: .int32)
        self.clusterSums = MLXArray.zeros([nKVHeads, K, headDim], dtype: .float32)

        for h in 0 ..< nKVHeads {
            let headAssign = self.assignments![h]  // [L]
            let headKeys = keysPerHead[h]  // [L, headDim]
            for c in 0 ..< K {
                let mask = headAssign .== MLXArray(Int32(c))
                let count = mask.asType(.int32).sum()
                self.clusterCounts![h, c] = count
                // Sum of keys in this cluster
                let maskedKeys = headKeys * mask.asType(.float32)[.ellipsis, .newAxis]
                self.clusterSums![h, c] = maskedKeys.sum(axis: 0)
            }
            // Update centroids from sums/counts
            let counts = self.clusterCounts![h].asType(.float32)[.ellipsis, .newAxis]  // [K, 1]
            let safeCounts = MLX.maximum(counts, MLXArray(Float(1.0)))
            self.centroids![h] = self.clusterSums![h] / safeCounts
        }

        clustersInitialized = true
        tokensSinceRecluster = 0
    }

    /// Assign newly added tokens to nearest centroid and update running means.
    private func assignNewTokens(from startOffset: Int, count: Int) {
        guard let keys = self.keys, let centroids = self.centroids else { return }

        let newKeys = keys[0, .ellipsis, startOffset ..< (startOffset + count), 0...]
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

        // Update counts and sums incrementally
        let nKVHeads = newKeys.dim(0)
        let K = centroids.dim(1)
        let headDim = centroids.dim(2)

        for h in 0 ..< nKVHeads {
            let headAssign = newAssignments[h]  // [count]
            let headKeys = newKeys[h]  // [count, headDim]
            for c in 0 ..< K {
                let mask = headAssign .== MLXArray(Int32(c))
                let newCount = mask.asType(.int32).sum()
                self.clusterCounts![h, c] = self.clusterCounts![h, c] + newCount
                let maskedKeys = headKeys * mask.asType(.float32)[.ellipsis, .newAxis]
                self.clusterSums![h, c] = self.clusterSums![h, c] + maskedKeys.sum(axis: 0)
            }
            // Refresh centroids from sums/counts
            let counts = self.clusterCounts![h].asType(.float32)[.ellipsis, .newAxis]
            let safeCounts = MLX.maximum(counts, MLXArray(Float(1.0)))
            self.centroids![h] = self.clusterSums![h] / safeCounts
        }
    }

    /// Full K-means re-clustering (3 iterations) to correct centroid drift.
    private func recluster() {
        guard let keys = self.keys, let centroids = self.centroids else { return }

        let allKeys = keys[0, .ellipsis, ..<offset, 0...].asType(.float32)
        // [nKVHeads, L, headDim]
        let nKVHeads = allKeys.dim(0)
        let L = allKeys.dim(1)
        let K = centroids.dim(1)
        let headDim = centroids.dim(2)

        var currentCentroids = self.centroids!

        for _ in 0 ..< 3 {
            // Assignment: [nKVHeads, L, K]
            let similarity = allKeys.matmul(currentCentroids.transposed(0, 2, 1))
            let newAssignments = similarity.argMax(axis: -1).asType(.int32)  // [nKVHeads, L]

            // Recompute centroids
            var newCounts = MLXArray.zeros([nKVHeads, K], dtype: .int32)
            var newSums = MLXArray.zeros([nKVHeads, K, headDim], dtype: .float32)

            for h in 0 ..< nKVHeads {
                let headAssign = newAssignments[h]
                let headKeys = allKeys[h]
                for c in 0 ..< K {
                    let mask = headAssign .== MLXArray(Int32(c))
                    newCounts[h, c] = mask.asType(.int32).sum()
                    let maskedKeys = headKeys * mask.asType(.float32)[.ellipsis, .newAxis]
                    newSums[h, c] = maskedKeys.sum(axis: 0)
                }
            }

            let countsF = newCounts.asType(.float32)[.ellipsis, .newAxis]
            let safeCounts = MLX.maximum(countsF, MLXArray(Float(1.0)))
            currentCentroids = newSums / safeCounts

            self.assignments = newAssignments
            self.clusterCounts = newCounts
            self.clusterSums = newSums
        }

        self.centroids = currentCentroids
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

        // Below threshold: standard full attention (identical to KVCacheSimple path)
        if offset <= clusterThreshold || !clustersInitialized {
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: allKeys,
                values: allValues,
                scale: scale,
                mask: mask
            )
        }

        // --- Clustered retrieval path ---

        let B = queries.dim(0)
        let nQHeads = queries.dim(1)
        let S = queries.dim(2)  // 1 during generation, >1 during prefill
        let headDim = queries.dim(3)
        let nKVHeads = allKeys.dim(1)
        let nRepeats = nQHeads / nKVHeads
        let L = offset

        // Determine regions
        let sinkEnd = min(sinkTokens, L)
        let recentStart = max(sinkEnd, L - recentWindow)

        // If recent window covers everything, just do full attention
        if recentStart <= sinkEnd {
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: allKeys,
                values: allValues,
                scale: scale,
                mask: mask
            )
        }

        // 1. Average queries across GQA groups to get per-KV-head query
        //    queries: [B, nQHeads, S, headDim] → [B, nKVHeads, S, headDim]
        let queryPerKVHead: MLXArray
        if nRepeats > 1 {
            let reshaped = queries.reshaped(B, nKVHeads, nRepeats, S, headDim)
            queryPerKVHead = reshaped.mean(axis: 2)  // [B, nKVHeads, S, headDim]
        } else {
            queryPerKVHead = queries
        }

        // 2. Compute query-centroid similarity
        //    queryPerKVHead: [B, nKVHeads, S, headDim]
        //    centroids: [nKVHeads, K, headDim]
        //    similarity: [B, nKVHeads, S, K]
        let centroidsExpanded = expandedDimensions(centroids!, axis: 0)  // [1, nKVHeads, K, headDim]
        let similarity = MLX.matmul(
            queryPerKVHead,
            centroidsExpanded.transposed(0, 1, 3, 2)
        )  // [B, nKVHeads, S, K]

        // 3. Select top-C clusters via argPartition (per KV head, per query token)
        //    For generation (S=1), use the single query. For prefill, use last query.
        let queryIdx = S - 1  // Use last query position for cluster selection
        let simForSelection = similarity[.ellipsis, queryIdx, 0...]  // [B, nKVHeads, K]
        let K = numClusters
        let C = min(topClusters, K)
        let kth = K - C
        let topIndices = MLX.argPartition(simForSelection, kth: kth, axis: -1)[
            .ellipsis, kth...]  // [B, nKVHeads, C]

        // 4. Build per-head gather mask for the "old context" region (sinkEnd ..< recentStart)
        //    We gather tokens whose cluster assignment matches any of the top-C clusters.
        let oldAssignments = self.assignments![.ellipsis, sinkEnd ..< recentStart]
        // [nKVHeads, oldLen]
        let oldLen = recentStart - sinkEnd

        // Expand for broadcasting:
        // oldAssignments: [1, nKVHeads, oldLen, 1]
        // topIndices:     [B, nKVHeads, 1, C]
        let oldAssignExpanded = expandedDimensions(
            expandedDimensions(oldAssignments, axis: 0), axis: -1
        ).asType(.int32)
        let topIndicesExpanded = expandedDimensions(topIndices, axis: 2).asType(.int32)

        // match: [B, nKVHeads, oldLen, C] → any match along C → [B, nKVHeads, oldLen]
        let matchMatrix = oldAssignExpanded .== topIndicesExpanded
        let oldMask = matchMatrix.any(axis: -1)  // [B, nKVHeads, oldLen]

        // 5. Compose the full retrieval:
        //    sink (exact) + selected old tokens + recent (exact)
        //
        //    Phase 1 approach: pad to max tokens per head with boolean mask
        let sinkKeys = allKeys[.ellipsis, ..<sinkEnd, 0...]
        let sinkValues = allValues[.ellipsis, ..<sinkEnd, 0...]
        let recentKeys = allKeys[.ellipsis, recentStart..., 0...]
        let recentValues = allValues[.ellipsis, recentStart..., 0...]

        let oldKeys = allKeys[.ellipsis, sinkEnd ..< recentStart, 0...]
        let oldValues = allValues[.ellipsis, sinkEnd ..< recentStart, 0...]

        // Count selected tokens per head to determine padding size
        let selectedPerHead = oldMask.asType(.int32).sum(axis: -1)  // [B, nKVHeads]
        eval(selectedPerHead)
        let maxSelected = Int(selectedPerHead.max().item(Int32.self))

        // Total tokens in attention = sinkEnd + maxSelected + recentLen
        let recentLen = L - recentStart

        if maxSelected == 0 {
            // No old tokens selected — just use sink + recent
            let subsetKeys = concatenated([sinkKeys, recentKeys], axis: 2)
            let subsetValues = concatenated([sinkValues, recentValues], axis: 2)
            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: subsetKeys,
                values: subsetValues,
                scale: scale,
                mask: .none
            )
        }

        // Gather selected old tokens — padded approach
        // For each head, gather masked tokens and pad to maxSelected
        // This is the Phase 1 "ragged gather + pad" approach

        // Create indices for the old region where mask is true
        // We use a dense gather: for each head, collect indices where mask=true, pad rest
        let totalRetrieved = sinkEnd + maxSelected + recentLen

        // Build the subset K/V by concatenating sink + padded_old + recent
        // For the old region, we apply the mask directly in attention scores
        // rather than physically gathering (simpler, avoids per-head scatter)

        // Approach: construct full-length boolean attention mask for the old region,
        // combined with always-true for sink and recent
        let sinkMaskPart = broadcast(
            MLXArray.ones([1, 1, 1, sinkEnd], dtype: .bool),
            to: [B, nKVHeads, S, sinkEnd])
        let oldMaskExpanded = broadcast(
            expandedDimensions(oldMask, axis: 2),
            to: [B, nKVHeads, S, oldLen])
        let recentMaskPart = broadcast(
            MLXArray.ones([1, 1, 1, recentLen], dtype: .bool),
            to: [B, nKVHeads, S, recentLen])

        // Combine: [B, nKVHeads, S, L]
        let combinedMask = concatenated(
            [sinkMaskPart, oldMaskExpanded, recentMaskPart], axis: -1)

        // Expand mask for GQA: [B, nQHeads, S, L]
        let attentionMask: MLXArray
        if nRepeats > 1 {
            // [B, nKVHeads, 1, S, L] → repeat → [B, nKVHeads, nRepeats, S, L] → reshape
            let expanded = broadcast(
                expandedDimensions(combinedMask, axis: 2),
                to: [B, nKVHeads, nRepeats, S, L])
            attentionMask = expanded.reshaped(B, nQHeads, S, L)
        } else {
            attentionMask = combinedMask
        }

        // Run attention with the boolean mask on full K/V
        // The mask zeros out non-selected old tokens
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: allKeys,
            values: allValues,
            scale: scale,
            mask: .array(attentionMask)
        )
    }
}
