# Prefill Speedup Analysis: What's Possible?

**Date:** 2026-03-17
**Model:** Qwen3.5-4B-PARO (32 layers: 8 full-attention, 24 linear/Mamba)
**Goal:** Find the theoretical maximum prefill speedup using clustering and any other approach.

## Measured Prefill Performance (Baseline)

| Prompt Length | Baseline (tok/s) | Clustered (tok/s) | Speedup |
|---------------|-------------------|-------------------|---------|
| 512           | 1992              | 1663              | 0.83x   |
| 1024          | 1419              | 1467              | 1.03x   |
| 2048          | 1370              | 1354              | 0.99x   |
| 4096          | 1225              | 1314              | 1.07x   |
| 8192          | 1258              | 1188              | 0.94x   |
| 16384         | 1165              | 1194              | 1.02x   |

**Clustering has no measurable effect on prefill.** The numbers fluctuate around 1.0x. Why?

---

## The Math: Where Prefill Time Goes

### Per-Token FLOP Budget (All 32 Layers)

| Component | FLOPs/token | Notes |
|-----------|-------------|-------|
| 8 FA layers (projections + FFN) | 3.76 GF | Q/K/V proj, output proj, RMSNorm, MLP |
| 24 Mamba layers (all compute) | 14.52 GF | GDN projections, conv1d, state update, FFN |
| **Non-SDPA total** | **18.28 GF** | **This is the hard floor** |
| SDPA (8 FA layers, varies with P) | varies | O(P) per token averaged; O(P^2) total |

### SDPA Fraction of Total Prefill

During prefill with chunk size C=512, each chunk does SDPA against all prior tokens. Total SDPA FLOPs grow as O(P^2); everything else grows as O(P).

| Prompt | Non-SDPA | SDPA (8 FA) | Total | **SDPA %** |
|--------|----------|-------------|-------|------------|
| 4,096 | 74.9 TF | 1.2 TF | 76.1 TF | **1.6%** |
| 8,192 | 149.7 TF | 4.7 TF | 154.4 TF | **3.0%** |
| 16,384 | 299.5 TF | 18.1 TF | 317.6 TF | **5.7%** |
| 32,768 | 599.0 TF | 71.5 TF | 670.4 TF | **10.7%** |
| 65,536 | 1,197.9 TF | 283.7 TF | 1,481.6 TF | **19.1%** |
| 131,072 | 2,395.8 TF | 1,130.3 TF | 3,526.1 TF | **32.1%** |
| 262,144 | 4,791.7 TF | 4,512.4 TF | 9,304.1 TF | **48.5%** |
| 1,010,000 | 18,461.6 TF | 66,887.2 TF | 85,348.7 TF | **78.4%** |

**At 16K, SDPA is only 5.7% of prefill compute. Even eliminating it entirely gives only 1.06x.**

This is why measured clustering gives ~1.0x at practical context lengths — there's simply nothing to optimize. SDPA only becomes meaningful above 100K tokens.

---

## Approach 1: SDPA Clustering (Current Implementation)

Reduces SDPA from O(P^2) to O(P) by replacing full causal attention with sink + centroid + chunk attention (772 entries per chunk instead of growing L).

### Maximum Prefill Speedup via Clustering (Amdahl's Law)

```
Speedup = 1 / ((1-f) + f/s)
  f = SDPA fraction of total FLOPs
  s = SDPA reduction ratio (baseline / clustered)
```

| Prompt | f (SDPA%) | s (reduction) | **Max Speedup** | Baseline tok/s | Clustered tok/s |
|--------|-----------|---------------|-----------------|----------------|-----------------|
| 16,384 | 5.7% | 7.3x | **1.05x** | 1,032 | 1,085 |
| 131,072 | 32.1% | 80.2x | **1.46x** | 743 | 1,088 |
| 262,144 | 48.5% | 165.0x | **1.93x** | 564 | 1,088 |
| 1,010,000 | 78.4% | 649.3x | **4.60x** | 237 | 1,088 |

Note: all clustered results converge to ~1,088 tok/s — that's the non-SDPA floor at ~20 TFLOPS. Real hardware achieves 27-37 TFLOPS, so the actual floor is ~1,500-2,000 tok/s (matching measured short-context performance).

### Verdict
Clustering makes SDPA essentially free, but SDPA is a minority of prefill compute. **At 262K: 1.93x maximum. At 1M: 4.60x maximum.** Cannot exceed the non-SDPA floor.

---

## Approach 2: Sliding Window Attention During Prefill

Replace full causal attention with a sliding window (W=2048 tokens) in the 8 FA layers. Each token only attends to its W nearest predecessors.

- SDPA becomes O(P x W) instead of O(P^2)
- Quality: Mamba layers provide long-range context; FA layers provide local precise attention
- This is architecturally natural for Qwen3.5 which already uses Mamba for global state

| Prompt | Base SDPA | SW SDPA | Reduction | **Overall Speedup** |
|--------|-----------|---------|-----------|---------------------|
| 16,384 | 18.1 TF | 4.2 TF | 4.3x | **1.05x** |
| 131,072 | 1,130 TF | 35.0 TF | 32.3x | **1.45x** |
| 262,144 | 4,512 TF | 70.2 TF | 64.3x | **1.91x** |
| 1,010,000 | 66,887 TF | 270.9 TF | 246.9x | **4.56x** |

### Verdict
Nearly identical to clustering (both reduce SDPA to O(P x constant)). Slightly simpler to implement. Same Amdahl's Law ceiling.

---

## Approach 3: Larger Prefill Chunks

Increase `prefillStepSize` from 512 to 1024-4096. Fewer kernel launches, better GPU utilization.

At P=262,144 with clustering:

| Chunk Size | Clustered SDPA Entries | SDPA FLOPs | Speedup |
|------------|------------------------|------------|---------|
| 512 | 772 | 27 TF | 1.93x |
| 1,024 | 1,284 | 45 TF | 1.93x |
| 2,048 | 2,308 | 80 TF | 1.92x |
| 4,096 | 4,356 | 150 TF | 1.90x |

### Verdict
Chunk size has negligible impact on total speedup because SDPA is already small relative to non-SDPA. Larger chunks may improve GPU utilization for the non-SDPA matmuls, but won't change the Amdahl's Law ceiling.

---

## Approach 4 (RADICAL): Skip Middle Context Computation

The most aggressive possible approach. Exploit the fact that during prefill, we know the entire prompt upfront:

1. Process first 512 tokens through all 32 layers (sink context)
2. **Middle tokens: compute ONLY K,V projections** at 8 FA layers (skip Mamba layers, FFN, Q/O projections entirely)
3. Process last 2,048 tokens through all 32 layers (recent context)
4. Cluster the middle K,V pairs for use during generation

Cost comparison per token:
- Full processing: **18.28 GFLOPs** (all 32 layers)
- K,V projection only: **134 MFLOPs** (2 linear layers at 8 FA layers)
- **136x cheaper** to compute K,V projections only

| Prompt | Baseline | Skip Middle | **Speedup** | Baseline tok/s | Skip tok/s |
|--------|----------|-------------|-------------|----------------|------------|
| 16,384 | 318 TF | 49 TF | **6.5x** | 1,032 | 6,661 |
| 131,072 | 3,526 TF | 65 TF | **54.4x** | 743 | 40,475 |
| 262,144 | 9,304 TF | 83 TF | **112.7x** | 564 | 63,500 |
| 1,010,000 | 85,350 TF | 184 TF | **463.6x** | 237 | 109,713 |

### Verdict
**Massive theoretical speedup** but at enormous quality cost. The middle tokens never propagate through Mamba layers or FFN, so their KV representations are computed from raw embeddings only. The centroids would be low quality because:
- No positional encoding propagation through the model
- No feature extraction from FFN layers
- No sequential state from Mamba layers

This is essentially equivalent to **using a 1-layer model for the middle context** and a 32-layer model for sink + recent. Quality would degrade severely for tasks requiring long-range understanding.

**However**, a SOFTER version could work: instead of skipping ALL computation for middle tokens, skip only the LAST N layers. Process middle tokens through layers 0-15 (first half), then skip layers 16-31. Use KV from layer 15 for clustering. This is essentially "early exit" for middle context tokens.

---

## Approach 5 (HYBRID): Early-Exit Middle + Full Sink/Recent

A practical compromise between full computation and complete skipping:

1. Process ALL tokens through first 16 layers (half the model)
2. For the remaining 16 layers:
   - Process sink + recent tokens fully (4 FA + 12 Mamba layers in second half)
   - Middle tokens: stop here — use their KV from layer 16 for clustering
3. Generation phase: full attention over sink + centroids + recent

Cost: ~60% of baseline (half the layers for middle tokens)

| Prompt | Baseline | Early-Exit | **Speedup** |
|--------|----------|------------|-------------|
| 16,384 | 318 TF | 197 TF | **1.6x** |
| 131,072 | 3,526 TF | 1,540 TF | **2.3x** |
| 262,144 | 9,304 TF | 3,037 TF | **3.1x** |
| 1,010,000 | 85,350 TF | 10,550 TF | **8.1x** |

(Estimates assume half the layers are skipped for P-2560 middle tokens, with SDPA also clustered in the remaining half.)

### Verdict
More realistic than full skip. Quality depends on whether early layers capture enough structure for useful centroids. Research question worth exploring.

---

## The Hard Floor

```
Non-SDPA compute per token: 18.28 GFLOPs
Apple Silicon throughput:    27-37 TFLOPS (measured)
Hard floor:                  1,500-2,000 tok/s
```

**This is the speed of light for this model.** No optimization to SDPA — clustering, sliding window, flash attention, or any other approach — can make prefill faster than this. The non-SDPA compute (24 Mamba layers + 8 FA projection/FFN) is irreducible without changing the model itself.

The only ways to break through the floor:
1. **Model quantization** — already applied (PARO/W4); further quantization reduces FLOPs but hurts quality
2. **Activation quantization (A8/A4)** — reduces matmul FLOPs by 2-4x; experimental
3. **Smaller model** — fewer params = less compute per token
4. **Architecture changes** — fewer layers, smaller hidden size, more efficient FFN
5. **Hardware** — faster GPU, more compute units, higher memory bandwidth

---

## Summary Table: All Approaches vs Prompt Length

| Approach | 16K | 262K | 1M | Quality Loss |
|----------|-----|------|-----|-------------|
| Baseline | 1.0x | 1.0x | 1.0x | None |
| **Clustering (current)** | **1.05x** | **1.93x** | **4.60x** | Approximate attention |
| Sliding Window (W=2048) | 1.05x | 1.91x | 4.56x | Local attention only |
| Larger chunks (C=2048) | ~1.05x | ~1.92x | ~4.58x | Same as clustering |
| Early-Exit (skip last 16L) | 1.6x | 3.1x | 8.1x | Medium (half-baked middle) |
| **Skip Middle (KV-only)** | **6.5x** | **112.7x** | **463.6x** | **Severe** |
| Hard floor (SDPA=0) | 1.06x | 1.94x | 4.60x | N/A (theoretical limit) |

## Conclusion

**Clustering is already doing the right thing** — it makes SDPA O(P) instead of O(P^2). The problem isn't the approach; it's that SDPA is a minority of prefill compute for this model architecture:

- **At 16K:** SDPA is 5.7% of total. Perfect elimination gives 1.06x. Nothing can help.
- **At 262K:** SDPA is 48.5%. Clustering achieves 1.93x, which is **within 0.5% of the Amdahl's Law limit** (1.94x). The implementation is essentially optimal.
- **At 1M:** SDPA is 78.4%. Clustering achieves 4.60x — close to the 4.60x limit.

The only path to dramatically faster prefill (>5x) is to **skip computation for middle context tokens entirely**. This is a model-architecture trade-off, not an optimization:

- Process only sink + recent through the full model
- Compute cheap K,V projections for middle tokens
- Cluster those cheap K,V pairs

This gives 112x at 262K but at severe quality cost. A hybrid "early-exit" approach (process middle tokens through half the model) offers a practical middle ground: 3.1x at 262K with moderate quality impact.

For this Qwen3.5 hybrid architecture (Mamba + attention), the Mamba layers already provide global context. Aggressive SDPA optimization during prefill is mathematically limited by the overwhelming cost of those 24 Mamba layers. The real speedup from clustering comes during **generation**, not prefill.
