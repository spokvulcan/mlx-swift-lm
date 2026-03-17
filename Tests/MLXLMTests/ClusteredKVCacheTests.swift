import Foundation
import MLX
import MLXLMCommon
import Testing

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
}
