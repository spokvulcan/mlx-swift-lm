import Foundation
import MLX

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
public func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask
        )
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
    } else {
        let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: cachedKeys,
            values: cachedValues,
            scale: scale,
            mask: mask
        )
    }
}

/// Expand a per-kv-head mask `(B, kvHeads, Lq, Lkv)` to per-query-head
/// `(B, attentionHeads, Lq, Lkv)` for Grouped Query Attention.
///
/// MLX's fast SDPA internally unflattens queries to rank-5
/// `(B, kvHeads, nRepeats, Lq, D)` and broadcasts the mask against rank-5
/// scores `(B, kvHeads, nRepeats, Lq, Lkv)`. A rank-4 mask with `kvHeads`
/// in dim 1 fails that broadcast when `attentionHeads != kvHeads`. Tiling
/// each kv-head's mask `nRepeats` times along dim 1 matches the
/// `query_head = kv_head × nRepeats + repeat` layout.
///
/// Idempotent: already-expanded masks (`dim(1) == attentionHeads`) and
/// non-GQA shapes pass through unchanged.
public func expandMaskForGroupedQueryHeads(
    _ mask: MLXArray,
    attentionHeads: Int,
    kvHeads: Int
) -> MLXArray {
    guard kvHeads > 0,
        attentionHeads > kvHeads,
        attentionHeads % kvHeads == 0,
        mask.ndim == 4,
        mask.dim(1) == kvHeads
    else {
        return mask
    }
    let nRepeats = attentionHeads / kvHeads
    return repeated(mask, count: nRepeats, axis: 1)
}
