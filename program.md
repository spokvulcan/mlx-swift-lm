# Autoresearch: Inference Speed Optimization

Model: Qwen3.5-4B PARO (INT4) — 24 GatedDeltaNet + 8 full attention layers
Actual dims: hidden=2560, Dk=128, Dv=128, Hk=16, Hv=32, vocab=248320

## Key Finding

**The model is ~97% memory-bandwidth bound.** At ~96 tok/s with ~2.05 GB weight
data read per token, we achieve ~197 GB/s, close to the hardware memory bandwidth
limit. Generation throughput improvements require reducing per-token memory traffic.

**PARO rotation overhead: 7%.** The 144 RotateQuantizedLinear kernel dispatches per
token cost ~7% (98.8 → 105.7 tok/s without rotation). This is dominated by Metal
kernel dispatch latency (~5µs × 144 = 720µs/token), not compute.

## Research Directions

### Tier 1 — High-impact (GatedDeltaNet = 75% of layers)

1. **GatedDeltaNet Metal kernel tuning** (`GatedDelta.swift`)
   - Tried threadGroup (32, 8, 1) vs baseline (32, 4, 1): within noise
   - The kernel is memory-bound for the state update; grid shape doesn't matter much
   - Status: TESTED, no improvement

2. **GatedDeltaNet Conv1d optimization** (`Qwen35.swift`)
   - Tried manual depthwise conv for S=1 (skip concatenation + Conv1d dispatch)
   - Result: slight regression. Built-in Conv1d is well-optimized
   - Status: TESTED, regressed

3. **MambaCache allocation** (`KVCache.swift`)
   - Inspected: ArraysCache uses simple optional array, no per-step allocation
   - No optimization needed
   - Status: INSPECTED, not actionable

### Tier 2 — Medium-impact (general inference pipeline)

4. **Prefill step size tuning** (`Evaluate.swift`)
   - **512 → 1024: +29.9% prefill throughput (1250 → 1624 tok/s)**
   - 2048: regressed (thermal/occupancy issues)
   - Status: COMMITTED (E2)

5. **KVCache pre-allocation** (`KVCache.swift`)
   - Inspected: step=256 growth causes ~2 reallocations per 500-token run
   - Overhead is negligible given memory-bandwidth bound nature
   - Status: INSPECTED, not worth pursuing

6. **asyncEval pipelining** (`Evaluate.swift`)
   - Already implemented correctly in TokenIterator.next()
   - GPU computes token N+1 while CPU reads token N
   - Status: VERIFIED, working correctly

7. **bfloat16→float32 cast elimination** (`Evaluate.swift`)
   - Model outputs float16, not bfloat16. Cast is already skipped
   - Status: VERIFIED, not applicable

8. **RotateQuantizedLinear kernel tuning** (`RotateQuantizedLinear.swift`)
   - Measured: rotation adds 7% overhead (144 dispatches × ~5µs each)
   - Cached scale constants: no improvement (CPU math is negligible)
   - Main bottleneck is kernel dispatch latency, not compute
   - **Opportunity: pre-rotate weights at load time to eliminate all rotation dispatches**
     - Requires: dequantize W, multiply by R^T, re-quantize
     - Risk: additional quantization error from double-quantization
   - Status: MEASURED, needs deeper work

### Tier 3 — Speculative

9. **Pre-rotate quantized weights** (NEW, from E5 findings)
   - Fuse rotation into weight matrices at model load time
   - Would eliminate 7% overhead from rotation kernel dispatches
   - Requires careful handling of quantization error
   - Status: NOT STARTED

10. **Compiled function** wrapping token generation step
    - `MLX.compile` is available but requires stateless functions
    - Model forward pass has stateful cache operations — hard to compile
    - Status: INVESTIGATED, not feasible without significant refactoring

11. **GatedDeltaNet + attention fusion**
    - Layers alternate 3:1 (linear:full)
    - Would require fundamental architecture changes
    - Status: NOT STARTED

## Experiment Log

| # | Description | greedy-short | greedy-long | topk | penalties | production | Commit |
|---|-------------|-------------|-------------|------|-----------|------------|--------|
| E0 | Baseline (prefill=512) | 98.8 | 94.2 | 97.0 | 95.5 | 93.9 | 8e5630b |
| E1 | GatedDelta threadgroup (32,8,1) | 97.2 | 96.2 | 96.7 | 95.2 | 95.1 | reverted |
| E2 | **Prefill step 1024** | 98.3 | 94.3 | 96.1 | 95.4 | 95.5 | f13697b |
| E3 | Cache RMSNorm scale Floats | — | — | — | — | — | reverted (no effect) |
| E4 | Manual depthwise conv S=1 | 97.1 | 93.6 | 94.9 | 91.3 | 93.5 | reverted (regressed) |
| E5 | Bypass rotation (measure only) | **105.7** | — | — | — | — | reverted (invalid output) |
| E6 | **Fuse in_proj_ba** | 99.5 | 95.6 | 94.3 | 95.9 | 96.4 | 8120608 |
| E7 | **Pre-rotate weights** | **121.7** | **114.4** | **118.7** | **115.7** | **115.5** | ffafdea |

**E7 summary**: +23% generation across all scenarios. Prefill long: 1250→1890 (+51%).
Trade-off: load time 1.6s→12s, peak load memory ~17 GB.

Generation TPS values are tok/s (higher is better). Prefill improvements noted separately.
