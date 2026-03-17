# ClusteredKVCache: Final Research Conclusions & Next Steps

## Research Summary

### Generation — KEEP (proven valuable)

| Context | Baseline (tok/s) | Clustered (tok/s) | Speedup | Amdahl Limit | Efficiency |
|---------|------------------|-------------------|---------|--------------|------------|
| 4096    | 95.9             | 64.0              | 0.67x   | 1.01x        | regression |
| 8192    | 92.4             | 89.2              | 0.97x   | 1.04x        | 93%        |
| 16384   | 85.5             | 90.9              | 1.06x   | 1.12x        | 53%        |
| 32768   | 46.8             | 85.9              | 1.84x   | 1.94x        | **95%**    |
| 262K    | ~10 (projected)  | ~86               | ~9x     | 9.0x         | projected  |
| 1M      | ~2.8 (projected) | ~86               | ~32x    | 32.0x        | projected  |

At 32K, the implementation captures 95% of the theoretical Amdahl's Law maximum. The cluster overhead is only 0.61 ms/tok. The ceiling is physics (only 8/32 layers use full attention).

**Profiling bottleneck breakdown (per layer, median):**

| Operation | Time | Notes |
|-----------|------|-------|
| cluster_assign | 0.38ms | Dominant per-token cost; tiny GPU ops can't saturate hardware |
| attn_concat | 0.35ms | Concatenating sink+centroid+recent KV |
| attn_sdpa | 0.33ms | The actual SDPA — fast, confirms design works |
| init_clusters | 64ms | One-time K-means++ initialization at threshold crossing |
| recluster | 20ms (at 32K) | Every 1024 tokens; scales linearly with context |

### Prefill — REMOVE (proven dead end)

| Context | SDPA % of total | Max clustering speedup | Verdict |
|---------|----------------|------------------------|---------|
| 16K     | 5.7%           | 1.05x                  | Hopeless — SDPA is noise |
| 262K    | 48.5%          | 1.93x                  | Marginal — still 160s minimum |
| 1M      | 78.4%          | 4.60x                  | Still 80+ seconds |

The non-SDPA compute floor (projections + FFN + Mamba = 18.28 GFLOPs/token) is irreducible. At 262K context, even with ZERO attention cost, prefill takes 160 seconds. No SDPA optimization can fix this.

The hierarchical clustering "256x reduction" from earlier research is real for attention FLOPs only. After Amdahl's Law: 2.86x overall at 262K. Still unusable for agents.

**For agent use cases (multi-turn with tool calls):**
- Without KV cache: re-prefill each call = **133 minutes** at 128K context
- With KV cache persistence: first call 80s, then **305ms per tool call**
- Caching provides **73-1000x improvement**. Prefill optimization provides 2x. Not close.

### Conclusion

1. **Keep clustering for generation** — real, measured, near-optimal speedup
2. **Remove clustering for prefill** — proven dead end by Amdahl's Law
3. **Invest in KV cache persistence** — the actual path to usable agent experience

---

## Next Steps (Ordered by Impact)

### Step 1: Remove Clustered Prefill Path

**Priority:** P0
**Impact:** Eliminates unnecessary complexity; no performance loss
**Effort:** Small

**File:** `Libraries/MLXLMCommon/ClusteredKVCache.swift`

In `clusteredAttention()`, replace the `else` branch (S > 1, prefill) with full SDPA fallback:

```swift
} else {
    // Prefill: use full attention (clustering provides no measurable
    // speedup during prefill — SDPA is <6% of prefill compute at 16K)
    return MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: allKeys,
        values: allValues,
        scale: scale,
        mask: mask
    )
}
```

Removes ~40 lines of prefill-specific clustering code (concatenation, custom mask construction). Generation path (S==1) is untouched.

### Step 2: Raise clusterThreshold from 4096 to 8192

**Priority:** P0
**Impact:** Eliminates the 0.67x regression at ctx=4096
**Effort:** Trivial (change one default value)

**File:** `Libraries/MLXLMCommon/ClusteredKVCache.swift`

```swift
// Change:
public let clusterThreshold: Int  // default 4096
// To:
public let clusterThreshold: Int  // default 8192
```

**Math:** At ctx=4096, even perfect clustering gives 1.01x (SDPA is 1.7% of total). The current 4096 threshold guarantees a regression because any cluster overhead > 0 exceeds the savings. At 8192, SDPA is 5.3% — still marginal but at least not negative.

Also update `Evaluate.swift` if the default is exposed there, and any benchmark configurations.

### Step 3: Replace K-means++ with Strided Initialization

**Priority:** P1
**Impact:** Eliminates 512ms one-time stall at threshold crossing
**Effort:** Small

**File:** `Libraries/MLXLMCommon/ClusteredKVCache.swift`, method `initializeClusters()`

Replace the sequential K-means++ centroid selection loop with:

```swift
// Strided subsample: pick every L/K-th token as initial centroid
let stride = max(1, L / K)
let indices = MLXArray(Array(stride_at: 0, count: K, stride: stride))
self.centroids = keysPerHead[indices]  // [nKVHeads, K, headDim]
```

K-means++ is inherently sequential (each centroid depends on distances from all previous), requiring 256 serial matmuls. Strided init is a single gather. Quality loss is negligible because `recluster()` corrects drift within 1024 tokens.

**Expected improvement:**
- initClusters: 64ms → <1ms per layer
- At ctx=4096 (100 tokens): overhead drops from 5.1ms/tok to ~0.04ms/tok amortized

### Step 4: Lazy Centroid Refresh

**Priority:** P2
**Impact:** Reduces cluster_assign overhead by ~40%
**Effort:** Small

**File:** `Libraries/MLXLMCommon/ClusteredKVCache.swift`, method `assignNewTokens()`

Currently, after every single token, centroids are recomputed (division of sums by counts). Instead, only recompute every N tokens:

```swift
// Always update running sums and counts (cheap)
self.clusterCounts = self.clusterCounts! + newCounts
self.clusterSums = self.clusterSums! + newKeySums
self.clusterValueSums = self.clusterValueSums! + newValueSums

// Only recompute centroids every 32 tokens
if (self.offset - clusterThreshold) % 32 == 0 || tokensSinceRecluster >= reclusterInterval {
    let countsF = self.clusterCounts!.asType(.float32).expandedDimensions(axis: -1)
    let safeCounts = MLX.maximum(countsF, MLXArray(Float(1.0)))
    self.centroids = self.clusterSums! / safeCounts
    self.centroidValues = self.clusterValueSums! / safeCounts
}
```

The running sums/counts update via one-hot matmul is the expensive part. The division is relatively cheap. But deferring the division avoids forcing MLX to evaluate the sums immediately.

### Step 5: Invest in KV Cache Persistence

**Priority:** P2 (for agent use cases)
**Impact:** 100-1000x improvement for multi-turn conversations
**Effort:** Medium-Large

This is the highest-ROI work for agent/tool-calling scenarios:

1. **Cache serialization to disk** — save/load KV cache state between sessions
   - `ClusteredKVCache` already has `state`/`metaState` serialization
   - Need: efficient disk format, lazy loading, memory mapping

2. **Prefix caching** — detect shared prompt prefix, reuse cached KV
   - Hash-based prefix matching
   - Invalidation when system prompt changes

3. **Incremental update** — only prefill new tokens, extend existing cache
   - Already works via `update()` — the infrastructure exists
   - Need: higher-level API for "continue conversation with new tokens"

### Step 6: Adaptive Recluster Interval

**Priority:** P3
**Impact:** Eliminates 266ms recluster spikes at 32K
**Effort:** Small

**File:** `Libraries/MLXLMCommon/ClusteredKVCache.swift`

```swift
// Current: fixed interval
public let reclusterInterval: Int  // default 1024

// Proposed: adaptive
let effectiveInterval = max(reclusterInterval, offset / 32)
```

At longer contexts, centroids drift less (more tokens averaged per cluster). Scaling the interval with context length reduces the number of expensive full-K-means passes without sacrificing quality.

---

## Research Files

| File | Contents |
|------|----------|
| `research-clustered-kv-cache.md` | Original design research |
| `research-clustered-kv-profiling.md` | Profiling results + bottleneck analysis + Amdahl's Law for generation |
| `research-prefill-speedup-analysis.md` | Prefill analysis: every approach considered, all proven insufficient |
| `PLAN-clustering-next-steps.md` | This file — conclusions + ordered next steps |
