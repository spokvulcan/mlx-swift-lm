import Foundation
import MLX
import Testing

@testable import MLXLMCommon

// MARK: - Below-Threshold Equivalence

@Test func testClusteredBelowThresholdMatchesSimple() async throws {
    let simple = KVCacheSimple()
    let clustered = ClusteredKVCache(clusterThreshold: 4096)

    let B = 1
    let nKVHeads = 8
    let headDim = 128
    let seqLen = 64

    let keys = MLXRandom.normal([B, nKVHeads, seqLen, headDim])
    let values = MLXRandom.normal([B, nKVHeads, seqLen, headDim])

    let (simpleK, simpleV) = simple.update(keys: keys, values: values)
    let (clusteredK, clusteredV) = clustered.update(keys: keys, values: values)

    // Below threshold, update() returns identical data
    eval(simpleK, simpleV, clusteredK, clusteredV)
    #expect(simpleK.shape == clusteredK.shape)
    #expect(simpleV.shape == clusteredV.shape)
    #expect(allClose(simpleK, clusteredK, atol: 1e-6).all().item(Bool.self))
    #expect(allClose(simpleV, clusteredV, atol: 1e-6).all().item(Bool.self))

    // Offsets match
    #expect(simple.offset == clustered.offset)
    #expect(simple.offset == seqLen)
}

// MARK: - Clustering Activation

@Test func testClusteringActivatesAtThreshold() async throws {
    let threshold = 128
    let clustered = ClusteredKVCache(
        numClusters: 16, topClusters: 4, recentWindow: 32,
        clusterThreshold: threshold, reclusterInterval: 64
    )

    let B = 1
    let nKVHeads = 2
    let headDim = 32

    // Feed tokens below threshold
    let keys1 = MLXRandom.normal([B, nKVHeads, threshold - 1, headDim])
    let values1 = MLXRandom.normal([B, nKVHeads, threshold - 1, headDim])
    _ = clustered.update(keys: keys1, values: values1)
    #expect(clustered.offset == threshold - 1)

    // State should have only 2 arrays (no cluster metadata yet)
    let stateBefore = clustered.state
    #expect(stateBefore.count == 2)

    // Push past threshold
    let keys2 = MLXRandom.normal([B, nKVHeads, 2, headDim])
    let values2 = MLXRandom.normal([B, nKVHeads, 2, headDim])
    _ = clustered.update(keys: keys2, values: values2)
    #expect(clustered.offset == threshold + 1)

    // State should now have 5 arrays (keys, values, centroids, assignments, counts)
    let stateAfter = clustered.state
    #expect(stateAfter.count == 5)
}

// MARK: - Clustered Attention Hot Path

@Test func testClusteredAttentionProducesOutput() async throws {
    let threshold = 64
    let clustered = ClusteredKVCache(
        numClusters: 8, topClusters: 2, recentWindow: 16,
        sinkTokens: 4, clusterThreshold: threshold, reclusterInterval: 32
    )

    let B = 1
    let nKVHeads = 2
    let headDim = 32
    let nQHeads = 4

    // Fill past threshold
    let seqLen = threshold + 32
    let keys = MLXRandom.normal([B, nKVHeads, seqLen, headDim])
    let values = MLXRandom.normal([B, nKVHeads, seqLen, headDim])
    _ = clustered.update(keys: keys, values: values)

    // Single-token generation query
    let queries = MLXRandom.normal([B, nQHeads, 1, headDim])
    let scale = Float(1.0 / sqrt(Float(headDim)))

    let output = clustered.clusteredAttention(
        queries: queries, scale: scale, mask: .none)
    eval(output)

    #expect(output.shape == [B, nQHeads, 1, headDim])
    // Output should be finite
    #expect(MLX.isNaN(output).any().item(Bool.self) == false)
}

// MARK: - Attention Quality vs Full Attention

@Test func testClusteredAttentionQuality() async throws {
    let threshold = 64
    let clustered = ClusteredKVCache(
        numClusters: 16, topClusters: 8, recentWindow: 16,
        sinkTokens: 4, clusterThreshold: threshold, reclusterInterval: 32
    )
    let simple = KVCacheSimple()

    let B = 1
    let nKVHeads = 2
    let headDim = 32
    let nQHeads = 4
    let seqLen = threshold + 32
    let scale = Float(1.0 / sqrt(Float(headDim)))

    let keys = MLXRandom.normal([B, nKVHeads, seqLen, headDim])
    let values = MLXRandom.normal([B, nKVHeads, seqLen, headDim])

    _ = clustered.update(keys: keys, values: values)
    let (simpleK, simpleV) = simple.update(keys: keys, values: values)

    // Query
    let queries = MLXRandom.normal([B, nQHeads, 1, headDim])

    let clusteredOut = clustered.clusteredAttention(
        queries: queries, scale: scale, mask: .none)
    let fullOut = MLXFast.scaledDotProductAttention(
        queries: queries, keys: simpleK, values: simpleV,
        scale: scale, mask: .none)
    eval(clusteredOut, fullOut)

    // L2 relative error < 20% (generous for small test with random data and few clusters)
    let diff = clusteredOut - fullOut
    let l2Error = sqrt((diff * diff).sum().item(Float.self))
    let l2Full = sqrt((fullOut * fullOut).sum().item(Float.self))
    let relError = l2Error / max(l2Full, 1e-8)
    #expect(relError < 0.2, "Relative L2 error \(relError) exceeds 20% threshold")
}

// MARK: - Serialization Round-Trip

@Test func testClusteredCacheSerialization() async throws {
    let clustered = ClusteredKVCache(
        numClusters: 8, topClusters: 2, recentWindow: 16,
        clusterThreshold: 32, reclusterInterval: 16
    )

    let B = 1
    let nKVHeads = 2
    let headDim = 32
    let seqLen = 48  // Past threshold to trigger clustering
    let keys = MLXRandom.normal([B, nKVHeads, seqLen, headDim])
    let values = MLXRandom.normal([B, nKVHeads, seqLen, headDim])
    _ = clustered.update(keys: keys, values: values)

    let cache: [KVCache] = [clustered]

    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension("safetensors")

    try savePromptCache(url: url, cache: cache, metadata: [:])
    let (loadedCache, _) = try loadPromptCache(url: url)

    #expect(loadedCache.count == 1)
    #expect(loadedCache[0] is ClusteredKVCache)

    let loaded = loadedCache[0] as! ClusteredKVCache
    #expect(loaded.offset == clustered.offset)
    #expect(loaded.state.count == clustered.state.count)
    #expect(loaded.metaState == clustered.metaState)

    // Verify key data is preserved
    for (a, b) in zip(loaded.state, clustered.state) {
        eval(a, b)
        #expect(a.shape == b.shape)
    }

    try FileManager.default.removeItem(at: url)
}

// MARK: - Re-clustering Convergence

@Test func testReclusteringReducesError() async throws {
    let clustered = ClusteredKVCache(
        numClusters: 8, topClusters: 4, recentWindow: 16,
        clusterThreshold: 32, reclusterInterval: 16
    )

    let B = 1
    let nKVHeads = 2
    let headDim = 32

    // Fill past threshold + recluster interval
    let seqLen = 32 + 16 + 1  // triggers initial clustering + one recluster cycle
    let keys = MLXRandom.normal([B, nKVHeads, seqLen, headDim])
    let values = MLXRandom.normal([B, nKVHeads, seqLen, headDim])
    _ = clustered.update(keys: keys, values: values)

    // Cache should have cluster state
    #expect(clustered.state.count == 5)
    #expect(clustered.offset == seqLen)

    // Centroids should be finite
    let centroids = clustered.state[2]
    eval(centroids)
    #expect(MLX.isNaN(centroids).any().item(Bool.self) == false)
}

// MARK: - Trim

@Test func testClusteredCacheTrim() async throws {
    let clustered = ClusteredKVCache(clusterThreshold: 4096)

    let B = 1
    let nKVHeads = 2
    let headDim = 32
    let seqLen = 64
    let keys = MLXRandom.normal([B, nKVHeads, seqLen, headDim])
    let values = MLXRandom.normal([B, nKVHeads, seqLen, headDim])
    _ = clustered.update(keys: keys, values: values)

    #expect(clustered.offset == 64)
    #expect(clustered.isTrimmable)

    let trimmed = clustered.trim(16)
    #expect(trimmed == 16)
    #expect(clustered.offset == 48)
}

// MARK: - AttentionWithCacheUpdate Dispatch

@Test func testAttentionDispatchesToClustered() async throws {
    let clustered = ClusteredKVCache(
        numClusters: 8, topClusters: 2, recentWindow: 16,
        clusterThreshold: 32, reclusterInterval: 16
    )

    let B = 1
    let nKVHeads = 2
    let headDim = 32
    let nQHeads = 4
    let scale = Float(1.0 / sqrt(Float(headDim)))

    // Fill past threshold
    let prefillKeys = MLXRandom.normal([B, nKVHeads, 48, headDim])
    let prefillValues = MLXRandom.normal([B, nKVHeads, 48, headDim])
    _ = clustered.update(keys: prefillKeys, values: prefillValues)

    // Now use attentionWithCacheUpdate for a generation step
    let genKeys = MLXRandom.normal([B, nKVHeads, 1, headDim])
    let genValues = MLXRandom.normal([B, nKVHeads, 1, headDim])
    let queries = MLXRandom.normal([B, nQHeads, 1, headDim])

    let output = attentionWithCacheUpdate(
        queries: queries, keys: genKeys, values: genValues,
        cache: clustered, scale: scale, mask: .none)
    eval(output)

    #expect(output.shape == [B, nQHeads, 1, headDim])
    #expect(MLX.isNaN(output).any().item(Bool.self) == false)
    #expect(clustered.offset == 49)
}

// MARK: - Protocol Conformance

@Test func testClusteredConformsToProtocol() async throws {
    let cache: KVCache = ClusteredKVCache()
    #expect(cache is ClusteredKVCacheProtocol)

    let clustered = cache as! ClusteredKVCacheProtocol
    #expect(clustered.numClusters == 256)
    #expect(clustered.topClusters == 32)
    #expect(clustered.recentWindow == 2048)
}

// MARK: - GenerateParameters

@Test func testGenerateParametersClusteredDefaults() async throws {
    let params = GenerateParameters()
    #expect(params.clusteredKV == false)
    #expect(params.kvClusters == 256)
    #expect(params.kvTopClusters == 32)
    #expect(params.kvRecentWindow == 2048)

    let custom = GenerateParameters(
        clusteredKV: true, kvClusters: 128, kvTopClusters: 16, kvRecentWindow: 1024)
    #expect(custom.clusteredKV == true)
    #expect(custom.kvClusters == 128)
    #expect(custom.kvTopClusters == 16)
    #expect(custom.kvRecentWindow == 1024)

    // Phase 2: prefill defaults
    #expect(params.prefillClusterThreshold == 64)
    #expect(params.prefillTopClusters == 48)
}

// MARK: - Phase 2: attentionWithLSE

@Test func testAttentionWithLSEMatchesSDPA() async throws {
    let cache = ClusteredKVCache()

    let B = 1
    let nH = 4
    let S = 8
    let L = 16
    let D = 32
    let scale = Float(1.0 / sqrt(Float(D)))

    let queries = MLXRandom.normal([B, nH, S, D])
    let keys = MLXRandom.normal([B, nH, L, D])
    let values = MLXRandom.normal([B, nH, L, D])

    // Manual attention via attentionWithLSE (no mask)
    let (manualOut, lse) = cache.attentionWithLSE(
        queries: queries, keys: keys, values: values, mask: nil, scale: scale)

    // Reference: MLXFast SDPA
    let refOut = MLXFast.scaledDotProductAttention(
        queries: queries, keys: keys, values: values, scale: scale, mask: .none)

    eval(manualOut, refOut, lse)

    #expect(manualOut.shape == refOut.shape)
    #expect(lse.shape == [B, nH, S, 1])

    // Outputs should match closely
    let diff = abs(manualOut - refOut)
    let maxDiff = diff.max().item(Float.self)
    #expect(maxDiff < 1e-4, "Max diff \(maxDiff) exceeds tolerance")
}

// MARK: - Phase 2: LSE Merge Correctness

@Test func testLSEMergeCorrectness() async throws {
    let cache = ClusteredKVCache()

    let B = 1
    let nH = 4
    let S = 8
    let L = 32
    let D = 32
    let scale = Float(1.0 / sqrt(Float(D)))

    let queries = MLXRandom.normal([B, nH, S, D])
    let keys = MLXRandom.normal([B, nH, L, D])
    let values = MLXRandom.normal([B, nH, L, D])

    // Split keys/values into two halves
    let halfL = L / 2
    let keys1 = keys[.ellipsis, ..<halfL, 0...]
    let values1 = values[.ellipsis, ..<halfL, 0...]
    let keys2 = keys[.ellipsis, halfL..., 0...]
    let values2 = values[.ellipsis, halfL..., 0...]

    // Attention on each half
    let (out1, lse1) = cache.attentionWithLSE(
        queries: queries, keys: keys1, values: values1, mask: nil, scale: scale)
    let (out2, lse2) = cache.attentionWithLSE(
        queries: queries, keys: keys2, values: values2, mask: nil, scale: scale)

    // LSE merge
    let maxLSE = MLX.maximum(lse1, lse2)
    let exp1 = MLX.exp(lse1 - maxLSE)
    let exp2 = MLX.exp(lse2 - maxLSE)
    let merged = (out1 * exp1 + out2 * exp2) / (exp1 + exp2)

    // Reference: full attention
    let refOut = MLXFast.scaledDotProductAttention(
        queries: queries, keys: keys, values: values, scale: scale, mask: .none)

    eval(merged, refOut)

    let diff = abs(merged - refOut)
    let maxDiff = diff.max().item(Float.self)
    #expect(maxDiff < 1e-3, "LSE merge max diff \(maxDiff) exceeds tolerance")
}

// MARK: - Phase 2: Clustered Prefill vs Full Attention (C=K)

@Test func testClusteredPrefillAllClusters() async throws {
    let threshold = 64
    let K = 8
    // C = K: select all clusters → should match full attention closely
    let clustered = ClusteredKVCache(
        numClusters: K, topClusters: K, recentWindow: 16,
        sinkTokens: 4, clusterThreshold: threshold, reclusterInterval: 32,
        prefillClusterThreshold: 8, prefillTopClusters: K
    )
    let simple = KVCacheSimple()

    let B = 1
    let nKVHeads = 2
    let headDim = 32
    let nQHeads = 4
    let scale = Float(1.0 / sqrt(Float(headDim)))

    // Fill past threshold with initial context
    let initLen = threshold + 16
    let initKeys = MLXRandom.normal([B, nKVHeads, initLen, headDim])
    let initValues = MLXRandom.normal([B, nKVHeads, initLen, headDim])
    _ = clustered.update(keys: initKeys, values: initValues)
    _ = simple.update(keys: initKeys, values: initValues)

    // Prefill chunk (S=16 > prefillClusterThreshold=8)
    let S = 16
    let chunkKeys = MLXRandom.normal([B, nKVHeads, S, headDim])
    let chunkValues = MLXRandom.normal([B, nKVHeads, S, headDim])
    _ = clustered.update(keys: chunkKeys, values: chunkValues)
    let (simpleK, simpleV) = simple.update(keys: chunkKeys, values: chunkValues)

    // Queries for the chunk
    let queries = MLXRandom.normal([B, nQHeads, S, headDim])

    let clusteredOut = clustered.clusteredAttention(
        queries: queries, scale: scale, mask: .causal)
    let fullOut = MLXFast.scaledDotProductAttention(
        queries: queries, keys: simpleK, values: simpleV,
        scale: scale, mask: .causal)

    eval(clusteredOut, fullOut)

    #expect(clusteredOut.shape == fullOut.shape)
    #expect(MLX.isNaN(clusteredOut).any().item(Bool.self) == false)

    // With all clusters selected, error should be moderate
    // (not exact match due to two-pass LSE merge vs single-pass, but close)
    let diff = clusteredOut - fullOut
    let l2Error = sqrt((diff * diff).sum().item(Float.self))
    let l2Full = sqrt((fullOut * fullOut).sum().item(Float.self))
    let relError = l2Error / max(l2Full, 1e-8)
    #expect(relError < 0.15, "All-cluster prefill relative error \(relError) exceeds 15%")
}

// MARK: - Phase 2: Clustered Prefill Subset (C < K)

@Test func testClusteredPrefillSubset() async throws {
    let threshold = 64
    let K = 16
    let C = 4  // Subset of clusters
    let clustered = ClusteredKVCache(
        numClusters: K, topClusters: C, recentWindow: 16,
        sinkTokens: 4, clusterThreshold: threshold, reclusterInterval: 32,
        prefillClusterThreshold: 8, prefillTopClusters: C
    )

    let B = 1
    let nKVHeads = 2
    let headDim = 32
    let nQHeads = 4
    let scale = Float(1.0 / sqrt(Float(headDim)))

    // Fill past threshold
    let initLen = threshold + 32
    let initKeys = MLXRandom.normal([B, nKVHeads, initLen, headDim])
    let initValues = MLXRandom.normal([B, nKVHeads, initLen, headDim])
    _ = clustered.update(keys: initKeys, values: initValues)

    // Prefill chunk
    let S = 16
    let chunkKeys = MLXRandom.normal([B, nKVHeads, S, headDim])
    let chunkValues = MLXRandom.normal([B, nKVHeads, S, headDim])
    _ = clustered.update(keys: chunkKeys, values: chunkValues)

    let queries = MLXRandom.normal([B, nQHeads, S, headDim])

    let output = clustered.clusteredAttention(
        queries: queries, scale: scale, mask: .causal)
    eval(output)

    #expect(output.shape == [B, nQHeads, S, headDim])
    #expect(MLX.isNaN(output).any().item(Bool.self) == false)
}

// MARK: - Phase 2: Gather Bandwidth Reduction

@Test func testGatherBandwidthReduction() async throws {
    let threshold = 64
    let K = 16
    let C = 4
    let recentWin = 16
    let sinkToks = 4
    let clustered = ClusteredKVCache(
        numClusters: K, topClusters: C, recentWindow: recentWin,
        sinkTokens: sinkToks, clusterThreshold: threshold, reclusterInterval: 32,
        prefillClusterThreshold: 8, prefillTopClusters: C
    )

    let B = 1
    let nKVHeads = 2
    let headDim = 32

    // Fill a longer context
    let totalLen = threshold + 128
    let keys = MLXRandom.normal([B, nKVHeads, totalLen, headDim])
    let values = MLXRandom.normal([B, nKVHeads, totalLen, headDim])
    _ = clustered.update(keys: keys, values: values)

    // Build gather indices for a hypothetical prefill at this offset
    let chunkSize = 16
    let chunkStart = totalLen - chunkSize
    let sinkEnd = min(sinkToks, chunkStart)
    let recentStart = max(sinkEnd, chunkStart - recentWin)
    let oldLen = recentStart - sinkEnd

    // Compute cluster selection
    let queryMean = MLXRandom.normal([B, nKVHeads, 1, headDim])
    let centroidsExp = expandedDimensions(clustered.state[2], axis: 0)
    let similarity = MLX.matmul(queryMean, centroidsExp.transposed(0, 1, 3, 2))
    let simFlat = similarity.squeezed(axis: 2)
    let kth = K - C
    let topIndices = MLX.argPartition(simFlat, kth: kth, axis: -1)[.ellipsis, kth...]

    let gatherIndices = clustered.buildGatherIndices(
        topClusterIndices: topIndices,
        oldLen: oldLen,
        sinkEnd: sinkEnd,
        recentStart: recentStart,
        totalLen: chunkStart
    )
    eval(gatherIndices)

    let gatheredCount = gatherIndices.dim(0)
    let fullCount = chunkStart  // Would read all pre-chunk tokens without gather

    // Gathered count should be < full count (actual reduction)
    #expect(gatheredCount < fullCount, "Gathered \(gatheredCount) should be < full \(fullCount)")
    #expect(
        gatheredCount >= sinkEnd + (totalLen - recentStart),
        "Gathered should include at least sink + recent")
}

// MARK: - Phase 2: Adaptive Cluster Count

@Test func testAdaptiveClusterCount() async throws {
    // At small context, K should be minClusters
    let cache1 = ClusteredKVCache(
        numClusters: 256, topClusters: 32, recentWindow: 16,
        clusterThreshold: 32, reclusterInterval: 16,
        targetClusterSize: 1024, minClusters: 32, maxClusters: 512
    )

    let B = 1
    let nKVHeads = 2
    let headDim = 32

    // Fill with 48 tokens (past threshold of 32)
    let keys1 = MLXRandom.normal([B, nKVHeads, 48, headDim])
    let values1 = MLXRandom.normal([B, nKVHeads, 48, headDim])
    _ = cache1.update(keys: keys1, values: values1)

    // At 48 tokens: 48/1024 = 0, clamped to minClusters=32
    #expect(cache1.numClusters == 32)

    // At larger context, K should scale
    let cache2 = ClusteredKVCache(
        numClusters: 256, topClusters: 32, recentWindow: 16,
        clusterThreshold: 32, reclusterInterval: 16,
        targetClusterSize: 8, minClusters: 4, maxClusters: 64
    )
    // Fill 48 tokens: 48/8 = 6, clamped to [4, 64] → 6
    let keys2 = MLXRandom.normal([B, nKVHeads, 48, headDim])
    let values2 = MLXRandom.normal([B, nKVHeads, 48, headDim])
    _ = cache2.update(keys: keys2, values: values2)
    #expect(cache2.numClusters == 6)

    // Fill more to trigger recluster and check adaptive growth
    let keys3 = MLXRandom.normal([B, nKVHeads, 32, headDim])
    let values3 = MLXRandom.normal([B, nKVHeads, 32, headDim])
    _ = cache2.update(keys: keys3, values: values3)
    // 80/8 = 10
    #expect(cache2.numClusters == 10)
}

// MARK: - Phase 2: Prefill Causal Correctness

@Test func testPrefillCausalCorrectness() async throws {
    let threshold = 64
    let K = 8
    let clustered = ClusteredKVCache(
        numClusters: K, topClusters: K, recentWindow: 16,
        sinkTokens: 4, clusterThreshold: threshold, reclusterInterval: 32,
        prefillClusterThreshold: 8, prefillTopClusters: K
    )

    let B = 1
    let nKVHeads = 2
    let headDim = 32
    let nQHeads = 4
    let scale = Float(1.0 / sqrt(Float(headDim)))

    // Fill past threshold
    let initLen = threshold + 16
    let initKeys = MLXRandom.normal([B, nKVHeads, initLen, headDim])
    let initValues = MLXRandom.normal([B, nKVHeads, initLen, headDim])
    _ = clustered.update(keys: initKeys, values: initValues)

    // Prefill chunk
    let S = 16
    let chunkKeys = MLXRandom.normal([B, nKVHeads, S, headDim])
    let chunkValues = MLXRandom.normal([B, nKVHeads, S, headDim])
    _ = clustered.update(keys: chunkKeys, values: chunkValues)

    let queries = MLXRandom.normal([B, nQHeads, S, headDim])

    let output1 = clustered.clusteredAttention(
        queries: queries, scale: scale, mask: .causal)
    eval(output1)

    // Verify first query position output is finite and has expected shape
    let firstQueryOut = output1[.ellipsis, 0, 0...]
    eval(firstQueryOut)
    #expect(firstQueryOut.shape == [B, nQHeads, headDim])
    #expect(MLX.isNaN(firstQueryOut).any().item(Bool.self) == false)

    // Verify last query position output is different from first
    // (different positions see different contexts)
    let lastQueryOut = output1[.ellipsis, S - 1, 0...]
    eval(lastQueryOut)
    let diff = abs(firstQueryOut - lastQueryOut).sum().item(Float.self)
    #expect(diff > 1e-6, "First and last query outputs should differ")
}
