# Autoresearch: Inference Speed Optimization

Model: Qwen3.5-4B PARO (INT4) — 24 GatedDeltaNet + 8 full attention layers

## Research Directions

Ordered by expected impact. Each experiment is a separate commit.

### Tier 1 — High-impact (GatedDeltaNet = 75% of layers)

1. **GatedDeltaNet Metal kernel tuning** (`GatedDelta.swift`)
   - Current grid: `(32, Dv, B*Hv)` with threadGroup `(32, 4, 1)`
   - Model dims: Dk=128, Dv=128, Hv=32 (value heads), n_per_t=4
   - Try: larger threadgroups, different grid shapes, occupancy tuning
   - Status: NOT STARTED

2. **GatedDeltaNet Conv1d optimization** (`Qwen35.swift`)
   - Conv1d in each linear attention layer: kernel_size=4, depthwise
   - During single-token generation, conv state is small
   - Optimize conv cache update path
   - Status: NOT STARTED

3. **MambaCache allocation** (`KVCache.swift`)
   - Uses `ArraysCache(size: 2)` — conv state + SSM state
   - Check if pre-allocation avoids per-step allocation overhead
   - Status: NOT STARTED

### Tier 2 — Medium-impact (general inference pipeline)

4. **Prefill step size tuning** (`Evaluate.swift`)
   - Default 512. Try 256, 1024, 2048
   - Affects both full attention and GatedDeltaNet prefill
   - Status: NOT STARTED

5. **KVCache pre-allocation** (`KVCache.swift`)
   - For 8 full attention layers
   - Current step=256 growth. Try doubling strategy
   - Status: NOT STARTED

6. **asyncEval pipelining** (`Evaluate.swift`)
   - Verify GPU is working ahead during `TokenIterator.next()`
   - Check for synchronization points that create bubbles
   - Status: NOT STARTED

7. **bfloat16→float32 cast elimination** (`Evaluate.swift`)
   - TopPSampler casts per-token: `if logits.dtype == .bfloat16`
   - Check if model output is already float16/float32
   - Status: NOT STARTED

8. **RotateQuantizedLinear kernel tuning** (`RotateQuantizedLinear.swift`)
   - PARO rotation kernel processes krot=8 rounds
   - Check if rounds can be parallelized or fused
   - Status: NOT STARTED

### Tier 3 — Speculative

9. **Compiled function** wrapping token generation step
   - `MLX.compile` the model forward + sample
   - Status: NOT STARTED

10. **GatedDeltaNet + attention fusion**
    - Layers alternate 3:1 (linear:full)
    - Explore batching the linear attention layers
    - Status: NOT STARTED

## Experiment Log

| # | Description | greedy-short | greedy-long | topk | penalties | production | Commit |
|---|-------------|-------------|-------------|------|-----------|------------|--------|
| E0 | Baseline | 98.8 | 94.2 | 97.0 | 95.5 | 93.9 | TBD |
