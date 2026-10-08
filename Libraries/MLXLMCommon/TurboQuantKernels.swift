// Copyright © 2026 Apple Inc.
//
// Single Metal kernel computes Q×K attention scores directly from packed
// codebook indices, skipping full dequantization. Pre-rotated queries
// eliminate the per-token inverse rotation.
//
// Key formula: score[t] = norm[t] * Σ_j q_rot[j] * codebook[idx[t,j]]
// where q_rot = Π · q (pre-rotated once) and idx are b-bit packed indices.

import Foundation
import MLX

// MARK: - Metal Kernel Source

enum TurboQuantMetalKernels {

    /// Scoring kernel: computes attention scores from packed codebook indices.
    ///
    /// Each SIMD group (32 threads) handles one (query, key_token) pair.
    /// Codebook (8-16 entries) is loaded into thread-local registers.
    /// Bit unpacking + codebook lookup + dot product all happen in-register.
    ///
    /// Grid: (32, totalQueries, tokenCount)
    /// Threadgroup: (32, 1, 1)
    ///
    /// Template params: Bits, Dim, PackedWidth, token_count, repeat_count
    static let scoreKernelSource = """
        // token_count/repeat_count/L via params buffer (vary per call; template
        // args would bake them into the kernel name = one compiled Metal
        // library cached per token).
        const uint token_count = params[0];
        const uint repeat_count = params[1];
        const uint chunk_len = params[2];
        // Template constants injected by MLXFast JIT
        constexpr uint MASK = (1u << Bits) - 1u;
        constexpr uint LEVELS = 1u << Bits;

        uint lane = thread_position_in_grid.x;      // SIMD lane (0-31)
        uint q_idx = thread_position_in_grid.y;     // query index (B*nQHeads*L)
        uint k_idx = thread_position_in_grid.z;     // key token index

        // Map query head to KV head (GQA)
        uint kv_idx = (q_idx / chunk_len) / repeat_count;

        // Pointers
        const device float* q_ptr = q_rot + q_idx * Dim;
        const device uint32_t* packed_ptr = packed + kv_idx * token_count * PackedWidth + k_idx * PackedWidth;
        float norm_val = norms[kv_idx * token_count + k_idx];

        // Load codebook into registers (small: 4-16 entries)
        float cb[LEVELS];
        for (uint i = 0; i < LEVELS; i++) {
            cb[i] = codebook[i];
        }

        // Parallel dot product: each lane handles dims [lane, lane+32, lane+64, ...]
        float acc = 0.0f;
        for (uint d = lane; d < Dim; d += 32) {
            // Unpack b-bit index for dimension d
            uint bit_offset = d * Bits;
            uint word_idx = bit_offset / 32;
            uint shift = bit_offset % 32;
            uint value = (packed_ptr[word_idx] >> shift);

            // Handle bits that spill across uint32 word boundary
            int spill = (int)shift + (int)Bits - 32;
            if (spill > 0) {
                value |= (packed_ptr[word_idx + 1] << ((uint)Bits - (uint)spill));
            }
            value &= MASK;

            // Codebook lookup + accumulate dot product
            acc += q_ptr[d] * cb[value];
        }

        // SIMD reduction across 32 lanes
        acc = simd_sum(acc);

        // Lane 0 writes final score (scaled by stored norm)
        if (thread_index_in_simdgroup == 0) {
            scores[q_idx * token_count + k_idx] = acc * norm_val;
        }
        """

    /// Fused encode kernel: norm + rotate + quantize + pack + norm correction in ONE dispatch.
    ///
    /// For each input vector [D]:
    ///   1. Compute L2 norm (SIMD reduction)
    ///   2. Normalize to unit vector
    ///   3. Rotate: y = Π · x_unit (shared memory matmul)
    ///   4. Quantize: find codebook index via boundary comparison
    ///   5. Pack bits into uint32 words (atomic OR)
    ///   6. Norm correction: compute reconstruction norm, store original_norm / recon_norm
    ///
    /// Norm correction compensates for quantization error so that
    /// centroid[idx] * corrected_norm more accurately reconstructs the original vector.
    /// This is why TurboQuant beats q8_0 on perplexity.
    ///
    /// Grid: (Dim, numRows, 1), one threadgroup per vector
    /// Threadgroup: (Dim, 1, 1), all D threads cooperate
    ///
    /// Template params: Bits, Dim, PackedWidth, NumBoundaries (= 2^Bits - 1)
    static let fusedEncodeSource = """
        constexpr uint LEVELS = 1u << Bits;

        uint d = thread_position_in_threadgroup.x;   // dimension index (0..Dim-1)
        uint row = thread_position_in_grid.y;         // vector index (B*H*T)

        // --- Step 1: Load input value ---
        float val = input[row * Dim + d];

        // --- Step 2: Compute L2 norm (SIMD reduction) ---
        float sq = val * val;
        float norm_sq = simd_sum(sq);
        // For Dim > 32, need threadgroup reduction
        threadgroup float shared_norm[32];  // up to 32 SIMD groups (dim <= 1024)
        uint sg_id = d / 32;
        if (d % 32 == 0) {
            shared_norm[sg_id] = norm_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_norm_sq = 0;
        uint num_groups = (Dim + 31) / 32;
        for (uint i = 0; i < num_groups; i++) {
            total_norm_sq += shared_norm[i];
        }
        float norm_val = sqrt(total_norm_sq);
        float inv_norm = (norm_val > 1e-8f) ? (1.0f / norm_val) : 0.0f;

        // --- Step 3: Normalize ---
        float unit_val = val * inv_norm;

        // --- Step 4: Rotate (y = Π · x_unit) via shared memory matmul ---
        // Each thread d computes: y[d] = Σ_j rotation[d * Dim + j] * x_unit[j]
        threadgroup float shared_unit[1024];  // max Dim = 1024
        shared_unit[d] = unit_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float rotated = 0.0f;
        for (uint j = 0; j < Dim; j++) {
            rotated += rotation[d * Dim + j] * shared_unit[j];
        }

        // --- Step 5: Quantize via branchless boundary comparison ---
        // V2.1 optimization: use arithmetic sum of comparisons instead of branching.
        // Metal compiles (rotated > boundaries[b]) to a predicated 0/1, summing these
        // is branchless and avoids SIMD lane divergence.
        uint idx = 0;
        for (uint b = 0; b < LEVELS - 1; b++) {
            idx += (uint)(rotated > boundaries[b]);
        }

        // --- Step 6: Pack bits into uint32 word (atomic OR) ---
        uint bit_offset = d * Bits;
        uint word_idx = bit_offset / 32;
        uint shift = bit_offset % 32;
        uint masked = idx & ((1u << Bits) - 1u);

        // Pack bits, use threadgroup shared memory to avoid atomic contention
        // Each thread writes its index bits to shared, then thread 0 per word writes output
        threadgroup uint shared_packed[64];  // max PackedWidth = 64 words
        if (d < PackedWidth) shared_packed[d] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each dimension contributes its bits via atomic OR on threadgroup memory
        uint primary_val = masked << shift;
        atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx], primary_val, memory_order_relaxed);

        int spill = (int)shift + (int)Bits - 32;
        if (spill > 0) {
            uint spill_val = masked >> ((uint)Bits - (uint)spill);
            atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx + 1], spill_val, memory_order_relaxed);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Write packed words to output (one thread per word)
        if (d < PackedWidth) {
            packed_out[row * PackedWidth + d] = shared_packed[d];
        }

        // --- Step 7: Norm correction ---
        // Compute reconstruction norm: ||codebook[idx]||₂ for the quantized unit vector.
        // Store corrected_norm = original_norm / recon_norm so that
        // decode(centroid[idx] * corrected_norm) better approximates the original vector.
        float centroid_val = codebook[idx];
        float recon_sq = centroid_val * centroid_val;
        float recon_norm_sq = simd_sum(recon_sq);
        // Threadgroup reduction for Dim > 32
        if (d % 32 == 0) {
            shared_norm[sg_id] = recon_norm_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_recon_sq = 0;
        for (uint i = 0; i < num_groups; i++) {
            total_recon_sq += shared_norm[i];
        }
        float recon_norm = sqrt(total_recon_sq);
        float corrected_norm = (recon_norm > 1e-8f) ? (norm_val / recon_norm) : norm_val;

        if (d == 0) {
            norms_out[row] = corrected_norm;
        }
        """

    /// Fused WHT encode kernel: norm + WHT rotation + quantize + pack (NO norm correction).
    ///
    /// Same as fusedEncodeSource but replaces dense O(d²) matmul with Fast Walsh-Hadamard
    /// Transform O(d log d) butterfly + random sign flip. 18× fewer ops for dim=128.
    ///
    /// Stores RAW norms (no reconstruction-norm correction), a measured choice,
    /// not an omission: for values consumed by attention's weighted sums, raw
    /// norms beat both matched-norm (attention cos 0.976 vs ≥0.980) and
    /// projection-optimal (0.9798) rescaling at 4-bit; per-vector rescales bias
    /// the decoded vectors toward zero while raw-norm errors stay unbiased and
    /// average out across tokens. The dense fallback path keeps matched-norm
    /// for canonical-formulation parity.
    ///
    /// WHT forward rotation: y = WHT(signs * x_unit) / sqrt(Dim)
    /// The butterfly pattern: for each stage s in 0..<log2(Dim), pairs at distance 2^s
    /// are combined: (a, b) → (a+b, a-b).
    ///
    /// Template params: Bits, Dim, PackedWidth, LogDim (= log2(Dim))
    static func whtEncodeSource(
        prologue: String = "", input: String = "input[row * Dim + d]",
        packedOut: String = "packed_out[row * PackedWidth + d]", normsOut: String = "norms_out[row]"
    ) -> String {
        """
        constexpr uint LEVELS = 1u << Bits;

        uint d = thread_position_in_threadgroup.x;   // dimension index (0..Dim-1)
        uint row = thread_position_in_grid.y;         // vector index (B*H*T)
        \(prologue)
        // --- Step 1: Load input value ---
        float val = \(input);

        // --- Step 2: Compute L2 norm (SIMD reduction) ---
        float sq = val * val;
        float norm_sq = simd_sum(sq);
        threadgroup float shared_norm[32];
        uint sg_id = d / 32;
        if (d % 32 == 0) {
            shared_norm[sg_id] = norm_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_norm_sq = 0;
        uint num_groups = (Dim + 31) / 32;
        for (uint i = 0; i < num_groups; i++) {
            total_norm_sq += shared_norm[i];
        }
        float norm_val = sqrt(total_norm_sq);
        float inv_norm = (norm_val > 1e-8f) ? (1.0f / norm_val) : 0.0f;

        // --- Step 3: Normalize + sign flip (fused) ---
        // V2.1 optimization: pre-compute inv_norm * sign to eliminate one multiply per element.
        // Instead of: unit_val = val * inv_norm; wht_val = sign * unit_val (2 muls)
        // We do:      wht_val = val * (inv_norm * sign) (1 mul + 1 FMA-friendly product)
        float inv_norm_sign = inv_norm * wht_signs[d];
        float wht_val = val * inv_norm_sign;

        // --- Step 4: WHT rotation via cooperative SIMD shuffle ---
        // V2.1 optimization: use simd_shuffle_xor for intra-SIMD butterfly stages
        // (register-to-register, no shared memory or barriers needed for first 5 stages)

        // Phase 1: Intra-SIMD butterfly via simd_shuffle_xor (stages 0..min(LogDim,5)-1)
        // Each stage s XORs lane indices at distance 2^s, effectively free on Apple GPU
        uint log_dim_u = uint(LogDim);
        uint simd_stages = min(log_dim_u, 5u);  // 5 stages covers 32 lanes (2^5 = 32)
        uint lane_in_simd = d % 32;
        for (uint s = 0; s < simd_stages; s++) {
            uint step = 1u << s;
            float other = simd_shuffle_xor(wht_val, step);
            wht_val = (lane_in_simd & step) ? (other - wht_val) : (other + wht_val);
        }

        // Phase 2: Cross-SIMD-group butterfly via shared memory (stages 5..LogDim-1)
        // Only needed when Dim > 32, these stages cross SIMD group boundaries
        threadgroup float shared_buf[1024];  // max Dim = 1024
        if (log_dim_u > 5u) {
            shared_buf[d] = wht_val;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint s = simd_stages; s < log_dim_u; s++) {
                uint half_block = 1u << s;
                uint block_size = half_block << 1;
                uint block_id = d / block_size;
                uint pos_in_block = d % block_size;

                float a, b;
                if (pos_in_block < half_block) {
                    a = shared_buf[block_id * block_size + pos_in_block];
                    b = shared_buf[block_id * block_size + pos_in_block + half_block];
                } else {
                    a = shared_buf[block_id * block_size + pos_in_block - half_block];
                    b = shared_buf[block_id * block_size + pos_in_block];
                }
                float next = (pos_in_block < half_block) ? (a + b) : (a - b);
                // All groups must finish reading this stage before any overwrite.
                threadgroup_barrier(mem_flags::mem_threadgroup);
                shared_buf[d] = next;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            wht_val = shared_buf[d];
        }

        // Normalize: WHT has scale factor sqrt(Dim)
        float inv_sqrt_dim = 1.0f / sqrt((float)Dim);
        float rotated = wht_val * inv_sqrt_dim;

        // --- Step 5: Quantize via branchless boundary comparison ---
        // V2.1 optimization: arithmetic sum avoids SIMD lane divergence
        uint idx = 0;
        for (uint b = 0; b < LEVELS - 1; b++) {
            idx += (uint)(rotated > boundaries[b]);
        }

        // --- Step 6: Pack bits into uint32 word (atomic OR) ---
        uint bit_offset = d * Bits;
        uint word_idx = bit_offset / 32;
        uint shift = bit_offset % 32;
        uint masked = idx & ((1u << Bits) - 1u);

        threadgroup uint shared_packed[64];
        if (d < PackedWidth) shared_packed[d] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint primary_val = masked << shift;
        atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx], primary_val, memory_order_relaxed);

        int spill = (int)shift + (int)Bits - 32;
        if (spill > 0) {
            uint spill_val = masked >> ((uint)Bits - (uint)spill);
            atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx + 1], spill_val, memory_order_relaxed);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d < PackedWidth) {
            \(packedOut) = shared_packed[d];
        }

        // --- Step 7: Store raw norm (WHT is orthogonal, no norm correction needed) ---
        // WHT preserves norms: ||WHT(x)||₂ = ||x||₂. Reconstruction norm ≈ original norm,
        // so the correction ratio ≈ 1.0. Skipping saves codebook lookup + norm + division.
        if (d == 0) {
            \(normsOut) = norm_val;
        }
        """
    }

    static let fusedEncodeWHTSource = whtEncodeSource()

    /// Writes `S` new rows per KV head at `position` straight into a cache's
    /// buffers, which the `inplace_*` outputs update in place: the WHT value
    /// encode above, then the key, raw or 8-bit affine. One threadgroup of
    /// `Dim` threads per row; `keys`/`values` are `[B, H, S, Dim]`.
    /// Each buffer is addressed at its own capacity (`*_shape[2]`): raw keys
    /// can keep their prefill allocation while values hold another.
    private static let rowWritePrologue = """
        const uint new_rows = keys_shape[2];
        const uint bh = row / new_rows;
        const uint pos = uint(position[0]) + (row - bh * new_rows);
        """

    private static func rowWriteValues() -> String {
        whtEncodeSource(
            prologue: rowWritePrologue, input: "values[row * Dim + d]",
            packedOut: "inplace_val_packed[(bh * val_packed_shape[2] + pos) * PackedWidth + d]",
            normsOut: "inplace_val_norms[bh * val_norms_shape[2] + pos]")
    }

    /// Template params: Bits, Dim, PackedWidth, LogDim, RawT.
    static let rowWriteRawKSource =
        rowWriteValues() + """

            inplace_k_raw[(bh * k_raw_shape[2] + pos) * Dim + d] =
                static_cast<RawT>(float(keys[row * Dim + d]));
            """

    /// mlx's `affine_quantize` at 8 bits, one group of `KGroup` per scale:
    /// the group's minimum and its maximum taken from 0, then
    /// `round((w - bias) / scale)` per element, four bytes to a word.
    ///
    /// Template params: Bits, Dim, PackedWidth, LogDim, KGroup, ScaleT.
    static let rowWriteAffineKSource =
        rowWriteValues() + """

            constexpr uint K_SIMDS = Dim / 32;
            constexpr uint SIMDS_PER_GROUP = KGroup / 32;
            const float kv = keys[row * Dim + d];
            const float lane_min = simd_min(kv);
            const float lane_max = simd_max(max(kv, 0.0f));
            threadgroup float tg_kmin[K_SIMDS];
            threadgroup float tg_kmax[K_SIMDS];
            if (d % 32 == 0) {
                tg_kmin[d / 32] = lane_min;
                tg_kmax[d / 32] = lane_max;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint g = d / KGroup;
            float w_min = tg_kmin[g * SIMDS_PER_GROUP];
            float w_max = tg_kmax[g * SIMDS_PER_GROUP];
            for (uint j = 1; j < SIMDS_PER_GROUP; j++) {
                w_min = min(w_min, tg_kmin[g * SIMDS_PER_GROUP + j]);
                w_max = max(w_max, tg_kmax[g * SIMDS_PER_GROUP + j]);
            }
            constexpr float eps = 1e-7;
            constexpr float n_bins = 255.0f;
            float k_scale = max((w_max - w_min) / n_bins, eps);
            const bool side = abs(w_min) > abs(w_max);
            k_scale = side ? k_scale : -k_scale;
            const float edge = side ? w_min : w_max;
            const float q0 = round(edge / k_scale);
            const bool at_zero = q0 == 0.0f;
            k_scale = at_zero ? k_scale : edge / q0;
            const float k_bias = at_zero ? 0 : edge;
            if (d % KGroup == 0) {
                const uint s_row = bh * k_scales_shape[2] + pos;
                const uint b_row = bh * k_biases_shape[2] + pos;
                inplace_k_scales[s_row * (Dim / KGroup) + g] = static_cast<ScaleT>(k_scale);
                inplace_k_biases[b_row * (Dim / KGroup) + g] = static_cast<ScaleT>(k_bias);
            }
            const uint8_t q_byte = min(round((kv - k_bias) / k_scale), n_bins);
            const uint q = q_byte;
            const uint q1 = simd_shuffle_down(q, 1);
            const uint q2 = simd_shuffle_down(q, 2);
            const uint q3 = simd_shuffle_down(q, 3);
            if (d % 4 == 0) {
                const uint w_row = bh * k_weights_shape[2] + pos;
                inplace_k_weights[w_row * (Dim / 4) + d / 4] = q | (q1 << 8) | (q2 << 16) | (q3 << 24);
            }
            """

    /// Quantize, pack, and norm-correct pre-rotated calibrated vectors.
    ///
    /// The rotation stays in MLX ops (MSECodec.rotatedUnit) so kernel and
    /// codec quantize bit-identical values; this kernel only does the parts
    /// that are slow as MLX ops: the boundary compare (which broadcasts a
    /// [rows, dim, levels] intermediate) and low-bit packing. The stored
    /// norm is norms_in / ||codebook[idx] * scale|| per row, matching
    /// MSECodec.encode(_:scale:).
    ///
    /// Template params: Bits, Dim, PackedWidth.
    static let fusedQuantizePackScaledSource = """
        constexpr uint LEVELS = 1u << Bits;

        uint d = thread_position_in_threadgroup.x;   // dimension index (0..Dim-1)
        uint row = thread_position_in_grid.y;         // vector index (B*H*T)

        float calibrated = rotated[row * Dim + d] / scale[d];
        uint idx = 0;
        for (uint b = 0; b < LEVELS - 1; b++) {
            idx += (uint)(calibrated > boundaries[b]);
        }

        uint bit_offset = d * Bits;
        uint word_idx = bit_offset / 32;
        uint shift = bit_offset % 32;
        uint masked = idx & ((1u << Bits) - 1u);

        threadgroup uint shared_packed[64];
        if (d < PackedWidth) shared_packed[d] = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint primary_val = masked << shift;
        atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx], primary_val, memory_order_relaxed);

        int spill = (int)shift + (int)Bits - 32;
        if (spill > 0) {
            uint spill_val = masked >> ((uint)Bits - (uint)spill);
            atomic_fetch_or_explicit((threadgroup atomic_uint*)&shared_packed[word_idx + 1], spill_val, memory_order_relaxed);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (d < PackedWidth) {
            packed_out[row * PackedWidth + d] = shared_packed[d];
        }

        // Norm correction against the calibrated reconstruction.
        float centroid_val = codebook[idx] * scale[d];
        float recon_sq = centroid_val * centroid_val;
        float recon_norm_sq = simd_sum(recon_sq);
        threadgroup float shared_norm[32];
        uint sg_id = d / 32;
        if (d % 32 == 0) {
            shared_norm[sg_id] = recon_norm_sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total_recon_sq = 0;
        uint num_groups = (Dim + 31) / 32;
        for (uint i = 0; i < num_groups; i++) {
            total_recon_sq += shared_norm[i];
        }
        float recon_norm = sqrt(total_recon_sq);

        if (d == 0) {
            float raw_norm = norms_in[row];
            norms_out[row] = (recon_norm > 1e-8f) ? (raw_norm / recon_norm) : raw_norm;
        }
        """

    /// TurboFlashAttention Pass 1: Per-block partial attention with online softmax.
    ///
    /// Parallelizes across both query heads AND token blocks. Each SIMD group (32 lanes)
    /// handles one (query, block) pair, producing partial online softmax state (m, l, o[D]).
    /// Pass 2 merges partials across blocks.
    ///
    /// Grid: (32, totalQueries, numBlocks)
    /// Threadgroup: (32, 1, 1)
    ///
    /// Template params: KeyBits, ValueBits, Dim, KeyPackedWidth, ValuePackedWidth,
    ///                  BlockSize, token_count, repeat_count, num_blocks
    static let turboFlashPass1Source = """
        // Per-call values arrive via the `params` input buffer, NOT as template
        // args: template values are baked into the kernel name, so a varying
        // token_count would JIT-compile and permanently cache one Metal
        // library per generated token (unbounded host-memory growth + a
        // shader compile every step).
        const uint token_count = params[0];
        const uint repeat_count = params[1];
        const uint num_blocks = params[2];
        constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
        constexpr uint KEY_LEVELS = 1u << KeyBits;
        constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
        constexpr uint VAL_LEVELS = 1u << ValueBits;
        constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

        uint lane = thread_position_in_grid.x;      // SIMD lane (0-31)
        uint q_idx = thread_position_in_grid.y;     // query index (B*nQHeads*L)
        uint block_idx = thread_position_in_grid.z; // token block index
        uint kv_idx = q_idx / repeat_count;         // map to KV head (GQA; L=1 only)

        // Token range for this block
        uint t_start = block_idx * BlockSize;
        uint t_end = t_start + BlockSize;
        if (t_end > (uint)token_count) t_end = (uint)token_count;

        // Load key codebook into registers
        float key_cb[KEY_LEVELS];
        for (uint i = 0; i < KEY_LEVELS; i++) {
            key_cb[i] = key_codebook[i];
        }

        // Load value codebook into registers
        float val_cb[VAL_LEVELS];
        for (uint i = 0; i < VAL_LEVELS; i++) {
            val_cb[i] = val_codebook[i];
        }

        // Load query values for this lane's dimensions
        float q_vals[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            q_vals[i] = (d < Dim) ? q_rot[q_idx * Dim + d] : 0.0f;
        }

        // Online softmax state for this block
        float m = -INFINITY;
        float l = 0.0f;
        float o[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

        // Process tokens in this block
        for (uint t = t_start; t < t_end; t++) {
            // --- Score: Q×K dot product ---
            const device uint32_t* k_packed_ptr = key_packed + kv_idx * token_count * KeyPackedWidth + t * KeyPackedWidth;
            float k_norm = key_norms[kv_idx * token_count + t];

            float dot_partial = 0.0f;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;

                uint k_bit_offset = d * KeyBits;
                uint k_word_idx = k_bit_offset / 32;
                uint k_shift = k_bit_offset % 32;
                uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
                int k_spill = (int)k_shift + (int)KeyBits - 32;
                if (k_spill > 0) {
                    k_value |= (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
                }
                k_value &= KEY_MASK;

                dot_partial += q_vals[i] * key_cb[k_value];
            }

            float score = simd_sum(dot_partial) * k_norm;

            // --- Online softmax update + V accumulation ---
            float new_m = max(m, score);
            float exp_diff = exp(m - new_m);
            float exp_score = exp(score - new_m);

            const device uint32_t* v_packed_ptr = val_packed + kv_idx * token_count * ValuePackedWidth + t * ValuePackedWidth;
            float v_norm = val_norms[kv_idx * token_count + t];

            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;

                uint v_bit_offset = d * ValueBits;
                uint v_word_idx = v_bit_offset / 32;
                uint v_shift = v_bit_offset % 32;
                uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
                int v_spill = (int)v_shift + (int)ValueBits - 32;
                if (v_spill > 0) {
                    v_value |= (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
                }
                v_value &= VAL_MASK;

                o[i] = o[i] * exp_diff + exp_score * (val_cb[v_value] * v_norm);
            }

            l = l * exp_diff + exp_score;
            m = new_m;
        }

        // Write partial results: o[D], m, l
        uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) {
                o_partials[partial_base + d] = o[i];
            }
        }
        if (lane == 0) {
            uint ml_idx = q_idx * num_blocks + block_idx;
            m_partials[ml_idx] = m;
            l_partials[ml_idx] = l;
        }
        """

    /// TurboFlashAttention bounded decode path: score, online softmax, value
    /// accumulation, normalization, and optional inverse value rotation in one
    /// dispatch. One SIMD group owns one query row and walks the cache in token
    /// order. This avoids allocating pass-1 partials and launching pass 2 when
    /// the context is short enough that cross-block parallelism is unnecessary.
    ///
    /// Grid: (32, totalQueries, 1)
    /// Threadgroup: (32, 1, 1)
    ///
    /// Template params: KeyBits, ValueBits, Dim, KeyPackedWidth,
    ///                  ValuePackedWidth, ApplyRotation
    static let turboFlashSinglePassSource = """
        const uint token_count = params[0];
        const uint repeat_count = params[1];
        constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
        constexpr uint KEY_LEVELS = 1u << KeyBits;
        constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
        constexpr uint VAL_LEVELS = 1u << ValueBits;
        constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

        uint lane = thread_position_in_grid.x;
        uint q_idx = thread_position_in_grid.y;
        uint kv_idx = q_idx / repeat_count;

        float key_cb[KEY_LEVELS];
        for (uint i = 0; i < KEY_LEVELS; i++) {
            key_cb[i] = key_codebook[i];
        }
        float val_cb[VAL_LEVELS];
        for (uint i = 0; i < VAL_LEVELS; i++) {
            val_cb[i] = val_codebook[i];
        }

        float q_vals[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            q_vals[i] = (d < Dim) ? q_rot[q_idx * Dim + d] : 0.0f;
        }

        float m = -INFINITY;
        float l = 0.0f;
        float o[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

        for (uint t = 0; t < token_count; t++) {
            const device uint32_t* k_packed_ptr =
                key_packed + kv_idx * token_count * KeyPackedWidth + t * KeyPackedWidth;
            float k_norm = key_norms[kv_idx * token_count + t];

            float dot_partial = 0.0f;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;
                uint bit_offset = d * KeyBits;
                uint word_idx = bit_offset / 32;
                uint shift = bit_offset % 32;
                uint value = k_packed_ptr[word_idx] >> shift;
                int spill = (int)shift + (int)KeyBits - 32;
                if (spill > 0) {
                    value |= k_packed_ptr[word_idx + 1] << ((uint)KeyBits - (uint)spill);
                }
                value &= KEY_MASK;
                dot_partial += q_vals[i] * key_cb[value];
            }
            float score = simd_sum(dot_partial) * k_norm;

            float new_m = max(m, score);
            float exp_diff = exp(m - new_m);
            float exp_score = exp(score - new_m);

            const device uint32_t* v_packed_ptr =
                val_packed + kv_idx * token_count * ValuePackedWidth + t * ValuePackedWidth;
            float v_norm = val_norms[kv_idx * token_count + t];
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;
                uint bit_offset = d * ValueBits;
                uint word_idx = bit_offset / 32;
                uint shift = bit_offset % 32;
                uint value = v_packed_ptr[word_idx] >> shift;
                int spill = (int)shift + (int)ValueBits - 32;
                if (spill > 0) {
                    value |= v_packed_ptr[word_idx + 1] << ((uint)ValueBits - (uint)spill);
                }
                value &= VAL_MASK;
                o[i] = o[i] * exp_diff + exp_score * (val_cb[value] * v_norm);
            }
            l = l * exp_diff + exp_score;
            m = new_m;
        }

        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        if (ApplyRotation != 0) {
            threadgroup float shared_out[Dim];
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d < Dim) shared_out[d] = o[i] * inv_l;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d < Dim) {
                    float acc = 0.0f;
                    for (uint j = 0; j < Dim; j++) {
                        acc += shared_out[j] * val_rotation[j * Dim + d];
                    }
                    output[q_idx * Dim + d] = acc;
                }
            }
        } else {
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d < Dim) output[q_idx * Dim + d] = o[i] * inv_l;
            }
        }
        """

    /// TurboFlashAttention Pass 1 (Causal): Per-block partial attention with causal masking.
    ///
    /// Same as turboFlashPass1Source but supports L>1 query chunks with causal masking.
    /// Each query position q_within_L only attends to tokens where t <= q_offset + q_within_L.
    /// Blocks that are entirely future-masked exit early.
    ///
    /// Grid: (32, totalQueries, numBlocks) where totalQueries = B * nQHeads * L
    /// Threadgroup: (32, 1, 1)
    ///
    /// Additional template params: L (query chunk length), q_offset (absolute offset of first query)
    /// Pass-1 variant: raw K scoring (KT template, matches the stored K
    /// cache dtype), turbo-V accumulation. Same SIMD-parallel online-softmax
    /// structure as the packed-K flash kernel, so the asymmetric family
    /// gets the same decode throughput.
    static let turboFlashPass1RawKSource = """
        // Per-call values arrive via the `params` input buffer, NOT as template
        // args: template values are baked into the kernel name, so a varying
        // token_count would JIT-compile and permanently cache one Metal
        // library per generated token (unbounded host-memory growth + a
        // shader compile every step).
        const uint token_count = params[0];
        const uint repeat_count = params[1];
        const uint num_blocks = params[2];
        constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
        constexpr uint VAL_LEVELS = 1u << ValueBits;
        constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

        uint lane = thread_position_in_grid.x;      // SIMD lane (0-31)
        uint q_idx = thread_position_in_grid.y;     // query index (B*nQHeads*L)
        uint block_idx = thread_position_in_grid.z; // token block index
        uint kv_idx = q_idx / repeat_count;         // map to KV head (GQA; L=1 only)

        // Token range for this block
        uint t_start = block_idx * BlockSize;
        uint t_end = t_start + BlockSize;
        if (t_end > (uint)token_count) t_end = (uint)token_count;

        // Load value codebook into registers
        float val_cb[VAL_LEVELS];
        for (uint i = 0; i < VAL_LEVELS; i++) {
            val_cb[i] = val_codebook[i];
        }

        // Load query values for this lane's dimensions
        float q_vals[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            q_vals[i] = (d < Dim) ? q_rot[q_idx * Dim + d] : 0.0f;
        }

        // Online softmax state for this block
        float m = -INFINITY;
        float l = 0.0f;
        float o[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

        // Process tokens in this block
        for (uint t = t_start; t < t_end; t++) {
            // --- Score: Q×K, K read raw f16 (no unpack, no rotation) ---
            const device KT* k_raw_ptr = (const device KT*)k_raw + kv_idx * token_count * Dim + t * Dim;
            float dot_partial = 0.0f;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;
                dot_partial += q_vals[i] * float(k_raw_ptr[d]);
            }
            float score = simd_sum(dot_partial);

            // --- Online softmax update + V accumulation ---
            float new_m = max(m, score);
            float exp_diff = exp(m - new_m);
            float exp_score = exp(score - new_m);

            const device uint32_t* v_packed_ptr = val_packed + kv_idx * token_count * ValuePackedWidth + t * ValuePackedWidth;
            float v_norm = val_norms[kv_idx * token_count + t];

            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;

                uint v_bit_offset = d * ValueBits;
                uint v_word_idx = v_bit_offset / 32;
                uint v_shift = v_bit_offset % 32;
                uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
                int v_spill = (int)v_shift + (int)ValueBits - 32;
                if (v_spill > 0) {
                    v_value |= (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
                }
                v_value &= VAL_MASK;

                o[i] = o[i] * exp_diff + exp_score * (val_cb[v_value] * v_norm);
            }

            l = l * exp_diff + exp_score;
            m = new_m;
        }

        // Write partial results: o[D], m, l
        uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) {
                o_partials[partial_base + d] = o[i];
            }
        }
        if (lane == 0) {
            uint ml_idx = q_idx * num_blocks + block_idx;
            m_partials[ml_idx] = m;
            l_partials[ml_idx] = l;
        }
        """

    /// Pass-1 variant: 8-bit affine K scoring (inline dequant, scale/bias in
    /// their native dtype via the KScaleT/KBiasT templates), turbo-V.
    static let turboFlashPass1AffineKSource = """
        // Per-call values arrive via the `params` input buffer, NOT as template
        // args: template values are baked into the kernel name, so a varying
        // token_count would JIT-compile and permanently cache one Metal
        // library per generated token (unbounded host-memory growth + a
        // shader compile every step).
        const uint token_count = params[0];
        const uint repeat_count = params[1];
        const uint num_blocks = params[2];
        constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
        constexpr uint VAL_LEVELS = 1u << ValueBits;
        constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

        uint lane = thread_position_in_grid.x;      // SIMD lane (0-31)
        uint q_idx = thread_position_in_grid.y;     // query index (B*nQHeads*L)
        uint block_idx = thread_position_in_grid.z; // token block index
        uint kv_idx = q_idx / repeat_count;         // map to KV head (GQA; L=1 only)

        // Token range for this block
        uint t_start = block_idx * BlockSize;
        uint t_end = t_start + BlockSize;
        if (t_end > (uint)token_count) t_end = (uint)token_count;

        // Load value codebook into registers
        float val_cb[VAL_LEVELS];
        for (uint i = 0; i < VAL_LEVELS; i++) {
            val_cb[i] = val_codebook[i];
        }

        // Load query values for this lane's dimensions
        float q_vals[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            q_vals[i] = (d < Dim) ? q_rot[q_idx * Dim + d] : 0.0f;
        }

        // Online softmax state for this block
        float m = -INFINITY;
        float l = 0.0f;
        float o[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

        // Process tokens in this block
        for (uint t = t_start; t < t_end; t++) {
            // --- Score: Q×K, K dequantized inline from 8-bit affine ---
            const device uint* k_w_ptr = k_weights + (kv_idx * token_count + t) * (Dim / 4);
            const device KScaleT* k_s_ptr =
                (const device KScaleT*)k_scales + (kv_idx * token_count + t) * (Dim / KGroup);
            const device KBiasT* k_b_ptr =
                (const device KBiasT*)k_biases + (kv_idx * token_count + t) * (Dim / KGroup);
            float dot_partial = 0.0f;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;
                uint g = d / KGroup;
                uint w8 = (k_w_ptr[d / 4] >> ((d % 4) * 8)) & 0xFFu;
                dot_partial += q_vals[i] * (float(k_s_ptr[g]) * float(w8) + float(k_b_ptr[g]));
            }
            float score = simd_sum(dot_partial);

            // --- Online softmax update + V accumulation ---
            float new_m = max(m, score);
            float exp_diff = exp(m - new_m);
            float exp_score = exp(score - new_m);

            const device uint32_t* v_packed_ptr = val_packed + kv_idx * token_count * ValuePackedWidth + t * ValuePackedWidth;
            float v_norm = val_norms[kv_idx * token_count + t];

            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;

                uint v_bit_offset = d * ValueBits;
                uint v_word_idx = v_bit_offset / 32;
                uint v_shift = v_bit_offset % 32;
                uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
                int v_spill = (int)v_shift + (int)ValueBits - 32;
                if (v_spill > 0) {
                    v_value |= (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
                }
                v_value &= VAL_MASK;

                o[i] = o[i] * exp_diff + exp_score * (val_cb[v_value] * v_norm);
            }

            l = l * exp_diff + exp_score;
            m = new_m;
        }

        // Write partial results: o[D], m, l
        uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) {
                o_partials[partial_base + d] = o[i];
            }
        }
        if (lane == 0) {
            uint ml_idx = q_idx * num_blocks + block_idx;
            m_partials[ml_idx] = m;
            l_partials[ml_idx] = l;
        }
        """

    static let turboFlashPass1CausalSource = """
        // Per-call values arrive via the `params` input buffer, NOT as template
        // args: template values are baked into the kernel name, so a varying
        // token_count would JIT-compile and permanently cache one Metal
        // library per generated token (unbounded host-memory growth + a
        // shader compile every step).
        const uint token_count = params[0];
        const uint repeat_count = params[1];
        const uint num_blocks = params[2];
        const uint L = params[3];
        const uint q_offset = params[4];
        constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
        constexpr uint KEY_LEVELS = 1u << KeyBits;
        constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
        constexpr uint VAL_LEVELS = 1u << ValueBits;
        constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

        uint lane = thread_position_in_grid.x;      // SIMD lane (0-31)
        uint q_idx = thread_position_in_grid.y;     // query index (B*nQHeads*L)
        uint block_idx = thread_position_in_grid.z; // token block index

        // For L>1, queries are laid out as [B * nQHeads * L, D] from reshape of [B, nQHeads, L, D].
        // q_idx = b * (nQHeads * L) + h * L + l
        // We need: l (position within chunk) and kv_head (for GQA mapping).
        uint q_within_L = q_idx % L;
        uint q_head_idx = q_idx / L;               // index into [B * nQHeads]
        uint kv_idx = q_head_idx / repeat_count;   // map to KV head (GQA)

        // Causal boundary: this query can attend to tokens 0..q_abs (inclusive)
        uint q_abs = q_offset + q_within_L;

        // Token range for this block
        uint t_start = block_idx * BlockSize;
        uint t_end = t_start + BlockSize;
        if (t_end > (uint)token_count) t_end = (uint)token_count;

        // Early exit: entire block is future-masked
        if (t_start > q_abs) {
            uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d < Dim) o_partials[partial_base + d] = 0.0f;
            }
            if (lane == 0) {
                uint ml_idx = q_idx * num_blocks + block_idx;
                m_partials[ml_idx] = -INFINITY;
                l_partials[ml_idx] = 0.0f;
            }
            return;
        }

        // Clamp t_end to causal boundary
        if (t_end > q_abs + 1) t_end = q_abs + 1;

        // Load key codebook into registers
        float key_cb[KEY_LEVELS];
        for (uint i = 0; i < KEY_LEVELS; i++) {
            key_cb[i] = key_codebook[i];
        }

        // Load value codebook into registers
        float val_cb[VAL_LEVELS];
        for (uint i = 0; i < VAL_LEVELS; i++) {
            val_cb[i] = val_codebook[i];
        }

        // Load query values for this lane's dimensions
        float q_vals[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            q_vals[i] = (d < Dim) ? q_rot[q_idx * Dim + d] : 0.0f;
        }

        // Online softmax state for this block
        float m = -INFINITY;
        float l = 0.0f;
        float o[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

        // Process tokens in this block (up to causal boundary)
        for (uint t = t_start; t < t_end; t++) {
            // --- Score: Q×K dot product ---
            const device uint32_t* k_packed_ptr = key_packed + kv_idx * token_count * KeyPackedWidth + t * KeyPackedWidth;
            float k_norm = key_norms[kv_idx * token_count + t];

            float dot_partial = 0.0f;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;

                uint k_bit_offset = d * KeyBits;
                uint k_word_idx = k_bit_offset / 32;
                uint k_shift = k_bit_offset % 32;
                uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
                int k_spill = (int)k_shift + (int)KeyBits - 32;
                if (k_spill > 0) {
                    k_value |= (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
                }
                k_value &= KEY_MASK;

                dot_partial += q_vals[i] * key_cb[k_value];
            }

            float score = simd_sum(dot_partial) * k_norm;

            // --- Online softmax update + V accumulation ---
            float new_m = max(m, score);
            float exp_diff = exp(m - new_m);
            float exp_score = exp(score - new_m);

            const device uint32_t* v_packed_ptr = val_packed + kv_idx * token_count * ValuePackedWidth + t * ValuePackedWidth;
            float v_norm = val_norms[kv_idx * token_count + t];

            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) break;

                uint v_bit_offset = d * ValueBits;
                uint v_word_idx = v_bit_offset / 32;
                uint v_shift = v_bit_offset % 32;
                uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
                int v_spill = (int)v_shift + (int)ValueBits - 32;
                if (v_spill > 0) {
                    v_value |= (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
                }
                v_value &= VAL_MASK;

                o[i] = o[i] * exp_diff + exp_score * (val_cb[v_value] * v_norm);
            }

            l = l * exp_diff + exp_score;
            m = new_m;
        }

        // Write partial results: o[D], m, l
        uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) {
                o_partials[partial_base + d] = o[i];
            }
        }
        if (lane == 0) {
            uint ml_idx = q_idx * num_blocks + block_idx;
            m_partials[ml_idx] = m;
            l_partials[ml_idx] = l;
        }
        """

    /// TurboFlashAttention Pass 1 NR0=2: Multi-row amortized KV dequant.
    ///
    /// Ported from llama.cpp V2.1 fused decode kernel concept. Each SIMD group processes
    /// NR0=2 queries against one KV block simultaneously. The key win: packed index unpacking
    /// + codebook lookup for K and V is done ONCE and reused across both queries.
    ///
    /// Register budget per thread (NR0=2, DIMS_PER_LANE=4 for dim=128):
    ///   - 2 × DIMS_PER_LANE q_vals = 8 floats (query data)
    ///   - 2 × 1 m/l = 4 floats (online softmax state)
    ///   - 2 × DIMS_PER_LANE o = 8 floats (value accumulators)
    ///   - codebook regs shared = KEY_LEVELS + VAL_LEVELS floats
    ///   Total: ~24 extra floats vs NR0=1. Well within Apple GPU register file.
    ///
    /// Zero threadgroup memory: all score computation + softmax + V accumulation happen
    /// in SIMD registers. No shared memory needed in pass 1 (same as NR0=1 baseline).
    /// Note: pass 2 still needs threadgroup memory for dim>32 (cross-SIMD gather for rotation).
    /// See turboFlashPass2FusedRotSource comments for details.
    ///
    /// Grid: (32, totalQueries/NR0, numBlocks)
    /// Threadgroup: (32, 1, 1)
    ///
    /// Template params: KeyBits, ValueBits, Dim, KeyPackedWidth, ValuePackedWidth,
    ///                  BlockSize, token_count, repeat_count, num_blocks, NR0
    static let turboFlashPass1NR0Source = """
        // Per-call values arrive via the `params` input buffer, NOT as template
        // args: template values are baked into the kernel name, so a varying
        // token_count would JIT-compile and permanently cache one Metal
        // library per generated token (unbounded host-memory growth + a
        // shader compile every step).
        const uint token_count = params[0];
        const uint repeat_count = params[1];
        const uint num_blocks = params[2];
        constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
        constexpr uint KEY_LEVELS = 1u << KeyBits;
        constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
        constexpr uint VAL_LEVELS = 1u << ValueBits;
        constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

        uint lane = thread_position_in_grid.x;          // SIMD lane (0-31)
        uint query_group = thread_position_in_grid.y;   // which group of NR0 queries
        uint block_idx = thread_position_in_grid.z;      // which KV block

        // Token range for this block
        uint t_start = block_idx * BlockSize;
        uint t_end = t_start + BlockSize;
        if (t_end > (uint)token_count) t_end = (uint)token_count;

        // Load key codebook into registers (shared across all NR0 queries)
        float key_cb[KEY_LEVELS];
        for (uint i = 0; i < KEY_LEVELS; i++) {
            key_cb[i] = key_codebook[i];
        }

        // Load value codebook into registers (shared across all NR0 queries)
        float val_cb[VAL_LEVELS];
        for (uint i = 0; i < VAL_LEVELS; i++) {
            val_cb[i] = val_codebook[i];
        }

        // Load query values for ALL NR0 rows, each row's dims interleaved in registers
        float q_vals[NR0 * DIMS_PER_LANE];
        for (uint r = 0; r < NR0; r++) {
            uint q_idx = query_group * NR0 + r;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                q_vals[r * DIMS_PER_LANE + i] = (d < Dim) ? q_rot[q_idx * Dim + d] : 0.0f;
            }
        }

        // Per-query KV head mapping (for GQA, each query may map to different KV head)
        uint kv_indices[NR0];
        for (uint r = 0; r < NR0; r++) {
            kv_indices[r] = (query_group * NR0 + r) / repeat_count;
        }

        // Online softmax state, NR0 independent streams, all in registers
        float m_state[NR0];
        float l_state[NR0];
        float o_state[NR0 * DIMS_PER_LANE];
        for (uint r = 0; r < NR0; r++) {
            m_state[r] = -INFINITY;
            l_state[r] = 0.0f;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                o_state[r * DIMS_PER_LANE + i] = 0.0f;
            }
        }

        // Process tokens in this block, KV dequant done ONCE, reused across NR0 queries
        for (uint t = t_start; t < t_end; t++) {
            // --- Dequant K for this token ONCE (amortized across NR0 queries) ---
            // Each lane unpacks its dims' codebook values into registers
            float k_decoded[DIMS_PER_LANE];
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) { k_decoded[i] = 0.0f; continue; }

                uint k_bit_offset = d * KeyBits;
                uint k_word_idx = k_bit_offset / 32;
                uint k_shift = k_bit_offset % 32;

                // All NR0 rows share one KV head by construction: the Swift
                // dispatcher only selects this kernel when repeat_count is a
                // multiple of NR0, so an aligned group can never span a
                // KV-head boundary and kv_indices[0] is exact.
                const device uint32_t* k_packed_ptr = key_packed + kv_indices[0] * token_count * KeyPackedWidth + t * KeyPackedWidth;

                uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
                int k_spill = (int)k_shift + (int)KeyBits - 32;
                if (k_spill > 0) {
                    k_value |= (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
                }
                k_value &= KEY_MASK;
                k_decoded[i] = key_cb[k_value];
            }
            float k_norm = key_norms[kv_indices[0] * token_count + t];

            // --- Dequant V for this token ONCE ---
            float v_decoded[DIMS_PER_LANE];
            const device uint32_t* v_packed_ptr = val_packed + kv_indices[0] * token_count * ValuePackedWidth + t * ValuePackedWidth;
            float v_norm = val_norms[kv_indices[0] * token_count + t];
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) { v_decoded[i] = 0.0f; continue; }

                uint v_bit_offset = d * ValueBits;
                uint v_word_idx = v_bit_offset / 32;
                uint v_shift = v_bit_offset % 32;
                uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
                int v_spill = (int)v_shift + (int)ValueBits - 32;
                if (v_spill > 0) {
                    v_value |= (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
                }
                v_value &= VAL_MASK;
                v_decoded[i] = val_cb[v_value] * v_norm;
            }

            // --- Score + softmax + V accumulate for each of NR0 queries ---
            // K/V dequant above is the expensive part, this loop is cheap ALU
            for (uint r = 0; r < NR0; r++) {
                // Dot product: q[r] · k (both already in registers)
                float dot_partial = 0.0f;
                for (uint i = 0; i < DIMS_PER_LANE; i++) {
                    dot_partial += q_vals[r * DIMS_PER_LANE + i] * k_decoded[i];
                }
                float score = simd_sum(dot_partial) * k_norm;

                // Online softmax update
                float new_m = max(m_state[r], score);
                float exp_diff = exp(m_state[r] - new_m);
                float exp_score = exp(score - new_m);

                // V accumulation (reusing pre-decoded values)
                for (uint i = 0; i < DIMS_PER_LANE; i++) {
                    o_state[r * DIMS_PER_LANE + i] = o_state[r * DIMS_PER_LANE + i] * exp_diff + exp_score * v_decoded[i];
                }

                l_state[r] = l_state[r] * exp_diff + exp_score;
                m_state[r] = new_m;
            }
        }

        // Write partial results for all NR0 queries
        for (uint r = 0; r < NR0; r++) {
            uint q_idx = query_group * NR0 + r;
            uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d < Dim) {
                    o_partials[partial_base + d] = o_state[r * DIMS_PER_LANE + i];
                }
            }
            if (lane == 0) {
                uint ml_idx = q_idx * num_blocks + block_idx;
                m_partials[ml_idx] = m_state[r];
                l_partials[ml_idx] = l_state[r];
            }
        }
        """

    /// TurboFlashAttention Pass 1 NR0 Causal: Multi-row amortized KV dequant with causal masking.
    ///
    /// Same as turboFlashPass1NR0Source but each query within the NR0 group has its own
    /// causal boundary. For L>1 prefill, q_within_L differs per row so each row may attend
    /// to a different number of tokens. We compute the conservative (minimum) causal boundary
    /// across the NR0 group for the shared K/V dequant, then mask per-row in the score loop.
    ///
    /// Grid: (32, totalQueries/NR0, numBlocks)
    /// Threadgroup: (32, 1, 1)
    ///
    /// Template params: KeyBits, ValueBits, Dim, KeyPackedWidth, ValuePackedWidth,
    ///                  BlockSize, token_count, repeat_count, num_blocks, NR0, L, q_offset
    static let turboFlashPass1NR0CausalSource = """
        // Per-call values arrive via the `params` input buffer, NOT as template
        // args: template values are baked into the kernel name, so a varying
        // token_count would JIT-compile and permanently cache one Metal
        // library per generated token (unbounded host-memory growth + a
        // shader compile every step).
        const uint token_count = params[0];
        const uint repeat_count = params[1];
        const uint num_blocks = params[2];
        const uint L = params[3];
        const uint q_offset = params[4];
        constexpr uint KEY_MASK = (1u << KeyBits) - 1u;
        constexpr uint KEY_LEVELS = 1u << KeyBits;
        constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
        constexpr uint VAL_LEVELS = 1u << ValueBits;
        constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

        uint lane = thread_position_in_grid.x;
        uint query_group = thread_position_in_grid.y;
        uint block_idx = thread_position_in_grid.z;

        // Token range for this block
        uint t_start = block_idx * BlockSize;
        uint t_end = t_start + BlockSize;
        if (t_end > (uint)token_count) t_end = (uint)token_count;

        // Compute per-row causal boundaries and find the maximum (most permissive)
        // for the shared token loop. Per-row masking happens inside the score loop.
        uint q_abs[NR0];
        uint max_q_abs = 0;
        for (uint r = 0; r < NR0; r++) {
            uint q_idx = query_group * NR0 + r;
            uint q_within_L = q_idx % L;
            q_abs[r] = q_offset + q_within_L;
            if (q_abs[r] > max_q_abs) max_q_abs = q_abs[r];
        }

        // Early exit: entire block is future-masked for ALL NR0 queries
        if (t_start > max_q_abs) {
            for (uint r = 0; r < NR0; r++) {
                uint q_idx = query_group * NR0 + r;
                uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
                for (uint i = 0; i < DIMS_PER_LANE; i++) {
                    uint d = lane + i * 32;
                    if (d < Dim) o_partials[partial_base + d] = 0.0f;
                }
                if (lane == 0) {
                    uint ml_idx = q_idx * num_blocks + block_idx;
                    m_partials[ml_idx] = -INFINITY;
                    l_partials[ml_idx] = 0.0f;
                }
            }
            return;
        }

        // Clamp t_end to the most permissive causal boundary
        if (t_end > max_q_abs + 1) t_end = max_q_abs + 1;

        // Load codebooks (shared across all NR0 queries)
        float key_cb[KEY_LEVELS];
        for (uint i = 0; i < KEY_LEVELS; i++) key_cb[i] = key_codebook[i];
        float val_cb[VAL_LEVELS];
        for (uint i = 0; i < VAL_LEVELS; i++) val_cb[i] = val_codebook[i];

        // Load query values for all NR0 rows
        float q_vals[NR0 * DIMS_PER_LANE];
        for (uint r = 0; r < NR0; r++) {
            uint q_idx = query_group * NR0 + r;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                q_vals[r * DIMS_PER_LANE + i] = (d < Dim) ? q_rot[q_idx * Dim + d] : 0.0f;
            }
        }

        // KV head mapping (use first query's head, same assumption as non-causal NR0)
        uint q_head_idx_0 = (query_group * NR0) / L;
        uint kv_idx = q_head_idx_0 / repeat_count;

        // Online softmax state, NR0 independent streams
        float m_state[NR0];
        float l_state[NR0];
        float o_state[NR0 * DIMS_PER_LANE];
        for (uint r = 0; r < NR0; r++) {
            m_state[r] = -INFINITY;
            l_state[r] = 0.0f;
            for (uint i = 0; i < DIMS_PER_LANE; i++) o_state[r * DIMS_PER_LANE + i] = 0.0f;
        }

        // Process tokens, KV dequant once, score per-row with causal mask
        for (uint t = t_start; t < t_end; t++) {
            // Dequant K once
            float k_decoded[DIMS_PER_LANE];
            const device uint32_t* k_packed_ptr = key_packed + kv_idx * token_count * KeyPackedWidth + t * KeyPackedWidth;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) { k_decoded[i] = 0.0f; continue; }
                uint k_bit_offset = d * KeyBits;
                uint k_word_idx = k_bit_offset / 32;
                uint k_shift = k_bit_offset % 32;
                uint k_value = (k_packed_ptr[k_word_idx] >> k_shift);
                int k_spill = (int)k_shift + (int)KeyBits - 32;
                if (k_spill > 0) {
                    k_value |= (k_packed_ptr[k_word_idx + 1] << ((uint)KeyBits - (uint)k_spill));
                }
                k_value &= KEY_MASK;
                k_decoded[i] = key_cb[k_value];
            }
            float k_norm = key_norms[kv_idx * token_count + t];

            // Dequant V once
            float v_decoded[DIMS_PER_LANE];
            const device uint32_t* v_packed_ptr = val_packed + kv_idx * token_count * ValuePackedWidth + t * ValuePackedWidth;
            float v_norm = val_norms[kv_idx * token_count + t];
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d >= Dim) { v_decoded[i] = 0.0f; continue; }
                uint v_bit_offset = d * ValueBits;
                uint v_word_idx = v_bit_offset / 32;
                uint v_shift = v_bit_offset % 32;
                uint v_value = (v_packed_ptr[v_word_idx] >> v_shift);
                int v_spill = (int)v_shift + (int)ValueBits - 32;
                if (v_spill > 0) {
                    v_value |= (v_packed_ptr[v_word_idx + 1] << ((uint)ValueBits - (uint)v_spill));
                }
                v_value &= VAL_MASK;
                v_decoded[i] = val_cb[v_value] * v_norm;
            }

            // Score + softmax + V for each query row (with per-row causal mask)
            for (uint r = 0; r < NR0; r++) {
                // Per-row causal: skip if this token is future for this specific query
                if (t > q_abs[r]) continue;

                float dot_partial = 0.0f;
                for (uint i = 0; i < DIMS_PER_LANE; i++) {
                    dot_partial += q_vals[r * DIMS_PER_LANE + i] * k_decoded[i];
                }
                float score = simd_sum(dot_partial) * k_norm;

                float new_m = max(m_state[r], score);
                float exp_diff = exp(m_state[r] - new_m);
                float exp_score = exp(score - new_m);

                for (uint i = 0; i < DIMS_PER_LANE; i++) {
                    o_state[r * DIMS_PER_LANE + i] = o_state[r * DIMS_PER_LANE + i] * exp_diff + exp_score * v_decoded[i];
                }
                l_state[r] = l_state[r] * exp_diff + exp_score;
                m_state[r] = new_m;
            }
        }

        // Write partial results for all NR0 queries
        for (uint r = 0; r < NR0; r++) {
            uint q_idx = query_group * NR0 + r;
            uint partial_base = (q_idx * num_blocks + block_idx) * Dim;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d < Dim) o_partials[partial_base + d] = o_state[r * DIMS_PER_LANE + i];
            }
            if (lane == 0) {
                uint ml_idx = q_idx * num_blocks + block_idx;
                m_partials[ml_idx] = m_state[r];
                l_partials[ml_idx] = l_state[r];
            }
        }
        """

    /// TurboFlashAttention Pass 2: Cross-block reduction.
    ///
    /// Merges partial online softmax states from pass 1 across token blocks.
    /// Each SIMD group handles one query, iterating over all blocks to produce
    /// the final normalized output.
    ///
    /// Grid: (32, totalQueries, 1)
    /// Threadgroup: (32, 1, 1)
    ///
    /// Template params: Dim, num_blocks
    static let turboFlashPass2Source = """
        // num_blocks via params buffer (varies per call; see pass-1 note).
        const uint num_blocks = params[0];
        constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

        uint lane = thread_position_in_grid.x;
        uint q_idx = thread_position_in_grid.y;

        float m = -INFINITY;
        float l = 0.0f;
        float o[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

        for (uint b = 0; b < (uint)num_blocks; b++) {
            uint ml_idx = q_idx * num_blocks + b;

            // All lanes read the same m/l (broadcast read from device memory)
            float block_m = m_partials[ml_idx];
            float block_l = l_partials[ml_idx];

            // Skip empty blocks
            if (block_l == 0.0f) continue;

            float new_m = max(m, block_m);
            float exp_old = exp(m - new_m);
            float exp_block = exp(block_m - new_m);

            uint partial_base = (q_idx * num_blocks + b) * Dim;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d < Dim) {
                    o[i] = o[i] * exp_old + o_partials[partial_base + d] * exp_block;
                }
            }

            l = l * exp_old + block_l * exp_block;
            m = new_m;
        }

        // Write normalized output
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) {
                output[q_idx * Dim + d] = o[i] * inv_l;
            }
        }
        """

    /// TurboFlashAttention Pass 2 with fused output rotation.
    ///
    /// Same as turboFlashPass2Source but applies inverse value rotation (Π_val) in-kernel
    /// after merging partials, eliminating a separate MLX matmul dispatch.
    /// Uses threadgroup shared memory to gather the full output vector across SIMD lanes,
    /// then each lane computes rotated output as dot product with rotation matrix rows.
    ///
    /// Grid: (32, totalQueries, 1)
    /// Threadgroup: (32, 1, 1)
    ///
    /// Template params: Dim, num_blocks
    static let turboFlashPass2FusedRotSource = """
        // num_blocks via params buffer (varies per call; see pass-1 note).
        const uint num_blocks = params[0];
        constexpr uint DIMS_PER_LANE = (Dim + 31) / 32;

        uint lane = thread_position_in_grid.x;
        uint q_idx = thread_position_in_grid.y;

        float m = -INFINITY;
        float l = 0.0f;
        float o[DIMS_PER_LANE];
        for (uint i = 0; i < DIMS_PER_LANE; i++) o[i] = 0.0f;

        for (uint b = 0; b < (uint)num_blocks; b++) {
            uint ml_idx = q_idx * num_blocks + b;

            float block_m = m_partials[ml_idx];
            float block_l = l_partials[ml_idx];

            if (block_l == 0.0f) continue;

            float new_m = max(m, block_m);
            float exp_old = exp(m - new_m);
            float exp_block = exp(block_m - new_m);

            uint partial_base = (q_idx * num_blocks + b) * Dim;
            for (uint i = 0; i < DIMS_PER_LANE; i++) {
                uint d = lane + i * 32;
                if (d < Dim) {
                    o[i] = o[i] * exp_old + o_partials[partial_base + d] * exp_block;
                }
            }

            l = l * exp_old + block_l * exp_block;
            m = new_m;
        }

        // Normalize
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;

        // Gather normalized output into threadgroup shared memory for rotation
        threadgroup float shared_out[Dim];
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) {
                shared_out[d] = o[i] * inv_l;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Apply inverse value rotation: output[d] = Σ_j shared_out[j] * Π_val[j][d]
        // matmul(x, Π_val) reads column d of Π_val for output dimension d.
        // Π_val is stored row-major [Dim, Dim], so column d = val_rotation[j * Dim + d]
        for (uint i = 0; i < DIMS_PER_LANE; i++) {
            uint d = lane + i * 32;
            if (d < Dim) {
                float acc = 0.0f;
                for (uint j = 0; j < Dim; j++) {
                    acc += shared_out[j] * val_rotation[j * Dim + d];
                }
                output[q_idx * Dim + d] = acc;
            }
        }
        """

    // MARK: - GQA flash decode (raw-K and affine-K with turbo V)

    /// GQA flash decode, pass 1. One threadgroup per (KV head, group of `HPS`
    /// query heads, token block); `NSG` SIMD groups split the block's tokens.
    /// Each token's K and V are decoded once and scored against every head in
    /// the group. Each lane owns `Dim / 32` contiguous dimensions (at most 32
    /// value bits). `k_stride` and `v_stride` are the rows allocated per KV
    /// head, so the caches are read in place.
    private static let gqaPass1Prologue = """
        const uint token_count = params[0];
        const uint tokens_per_block = params[1];
        const uint num_blocks = params[2];
        const uint k_stride = params[3];
        const uint v_stride = params[4];
        const float q_scale = as_type<float>(params[5]);
        constexpr uint DPL = Dim / 32;
        constexpr uint VAL_LEVELS = 1u << ValueBits;
        constexpr uint VAL_MASK = VAL_LEVELS - 1u;
        constexpr uint LANE_BITS = DPL * ValueBits;

        constexpr uint HEAD_GROUPS = Rep / HPS;

        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint tid = thread_index_in_threadgroup;
        const uint kv = threadgroup_position_in_grid.y / HEAD_GROUPS;
        const uint q_base = kv * Rep + (threadgroup_position_in_grid.y % HEAD_GROUPS) * HPS;
        const uint block = threadgroup_position_in_grid.z;

        threadgroup float cb[VAL_LEVELS];
        if (tid < VAL_LEVELS) cb[tid] = val_codebook[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float qv[HPS][DPL];
        float qsum[HPS];
        for (uint r = 0; r < HPS; r++) {
            auto qp = q_in + (q_base + r) * Dim + lane * DPL;
            qsum[r] = 0.0f;
            for (uint i = 0; i < DPL; i++) {
                qv[r][i] = float(qp[i]) * q_scale;
                qsum[r] += qv[r][i];
            }
        }
        float m[HPS];
        float l[HPS];
        float o[HPS][DPL];
        for (uint r = 0; r < HPS; r++) {
            m[r] = -INFINITY;
            l[r] = 0.0f;
            for (uint i = 0; i < DPL; i++) o[r][i] = 0.0f;
        }

        const uint t_begin = block * tokens_per_block;
        const uint t_end = min(t_begin + tokens_per_block, token_count);
        const uint lane_bit = lane * LANE_BITS;
        const uint lane_shift = lane_bit & 31u;
        const device uint32_t* v_head =
            val_packed + (size_t)kv * v_stride * ValuePackedWidth + (lane_bit >> 5);
        const device float* n_head = val_norms + (size_t)kv * v_stride;

        for (uint t = t_begin + sg; t < t_end; t += NSG) {
            float s[HPS];

        """

    private static let gqaPass1RawKScore = """
            auto kp = k_raw + ((size_t)kv * k_stride + t) * Dim + lane * DPL;
            float kf[DPL];
            for (uint i = 0; i < DPL; i++) kf[i] = float(kp[i]);
            for (uint r = 0; r < HPS; r++) {
                float acc = 0.0f;
                for (uint i = 0; i < DPL; i++) acc += qv[r][i] * kf[i];
                s[r] = simd_sum(acc);
            }

        """

    /// 8-bit affine keys: one group per lane (`KGroup % DPL == 0`), so the
    /// lane's partial dot is `scale · Σ q·w + bias · Σ q`.
    private static let gqaPass1AffineKScore = """
            const size_t k_row = (size_t)kv * k_stride + t;
            const device uint32_t* kw = k_weights + k_row * (Dim / 4) + lane * (DPL / 4);
            const uint k_group = (lane * DPL) / KGroup;
            const float k_scale = float(k_scales[k_row * (Dim / KGroup) + k_group]);
            const float k_bias = float(k_biases[k_row * (Dim / KGroup) + k_group]);
            float kf[DPL];
            for (uint wi = 0; wi < DPL / 4; wi++) {
                const uint w = kw[wi];
                for (uint j = 0; j < 4; j++) kf[wi * 4 + j] = float((w >> (8 * j)) & 0xFFu);
            }
            for (uint r = 0; r < HPS; r++) {
                float acc = 0.0f;
                for (uint i = 0; i < DPL; i++) acc += qv[r][i] * kf[i];
                s[r] = simd_sum(k_scale * acc + k_bias * qsum[r]);
            }

        """

    private static let gqaPass1Epilogue = """
            const device uint32_t* vp = v_head + (size_t)t * ValuePackedWidth;
            ulong v_bits = vp[0];
            if (lane_shift + LANE_BITS > 32) v_bits |= ((ulong)vp[1]) << 32;
            v_bits >>= lane_shift;
            const float v_norm = n_head[t];
            float vv[DPL];
            for (uint i = 0; i < DPL; i++) {
                vv[i] = cb[(uint)(v_bits >> (i * ValueBits)) & VAL_MASK] * v_norm;
            }
            // The maximum rarely moves: the common step pays one exp and no rescale.
            for (uint r = 0; r < HPS; r++) {
                if (s[r] > m[r]) {
                    const float f = exp(m[r] - s[r]);
                    m[r] = s[r];
                    l[r] = l[r] * f + 1.0f;
                    for (uint i = 0; i < DPL; i++) o[r][i] = o[r][i] * f + vv[i];
                } else {
                    const float p = fast::exp(s[r] - m[r]);
                    l[r] += p;
                    for (uint i = 0; i < DPL; i++) o[r][i] += p * vv[i];
                }
            }
        }

        // Merge the SIMD groups' states, then write one partial per query head.
        threadgroup float tg_m[NSG * HPS];
        threadgroup float tg_l[NSG * HPS];
        threadgroup float tg_o[HPS * Dim];
        if (lane == 0) {
            for (uint r = 0; r < HPS; r++) {
                tg_m[sg * HPS + r] = m[r];
                tg_l[sg * HPS + r] = l[r];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float merged_m[HPS];
        for (uint r = 0; r < HPS; r++) {
            merged_m[r] = -INFINITY;
            for (uint j = 0; j < NSG; j++) merged_m[r] = max(merged_m[r], tg_m[j * HPS + r]);
        }
        for (uint j = 0; j < NSG; j++) {
            if (sg == j) {
                for (uint r = 0; r < HPS; r++) {
                    const float scale = (m[r] == -INFINITY) ? 0.0f : exp(m[r] - merged_m[r]);
                    for (uint i = 0; i < DPL; i++) {
                        const uint idx = r * Dim + lane * DPL + i;
                        const float val = o[r][i] * scale;
                        tg_o[idx] = (j == 0) ? val : tg_o[idx] + val;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint idx = tid; idx < HPS * Dim; idx += NSG * 32) {
            const uint r = idx / Dim;
            const uint d = idx - r * Dim;
            o_partials[((size_t)(q_base + r) * num_blocks + block) * Dim + d] = tg_o[idx];
        }
        if (tid < HPS) {
            const float mx = merged_m[tid];
            float ls = 0.0f;
            for (uint j = 0; j < NSG; j++) {
                const float mj = tg_m[j * HPS + tid];
                if (mj != -INFINITY) ls += tg_l[j * HPS + tid] * exp(mj - mx);
            }
            m_partials[(q_base + tid) * num_blocks + block] = mx;
            l_partials[(q_base + tid) * num_blocks + block] = ls;
        }
        """

    static let turboFlashGQAPass1RawKSource =
        gqaPass1Prologue + gqaPass1RawKScore + gqaPass1Epilogue

    static let turboFlashGQAPass1AffineKSource =
        gqaPass1Prologue + gqaPass1AffineKScore + gqaPass1Epilogue

    /// GQA flash decode, pass 2. One threadgroup of `Dim` threads per query
    /// head: thread b weighs block b (`num_blocks <= Dim`), thread d merges
    /// output dimension d, then the threadgroup applies the inverse value
    /// rotation (`output = merged · Π_val`).
    static let turboFlashGQAPass2Source = """
        const uint num_blocks = params[0];
        constexpr uint NSIMD = Dim / 32;
        const uint tid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint q = threadgroup_position_in_grid.y;

        threadgroup float weights[Dim];
        threadgroup float red_max[NSIMD];
        threadgroup float red_sum[NSIMD];
        threadgroup float merged[Dim];

        const float mb = (tid < num_blocks) ? m_partials[q * num_blocks + tid] : -INFINITY;
        const float sg_max = simd_max(mb);
        if (lane == 0) red_max[sg] = sg_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float mx = red_max[0];
        for (uint j = 1; j < NSIMD; j++) mx = max(mx, red_max[j]);

        const float w = (mb == -INFINITY) ? 0.0f : exp(mb - mx);
        weights[tid] = w;
        const float lb = (tid < num_blocks) ? l_partials[q * num_blocks + tid] * w : 0.0f;
        const float sg_sum = simd_sum(lb);
        if (lane == 0) red_sum[sg] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (uint j = 0; j < NSIMD; j++) total += red_sum[j];

        float acc = 0.0f;
        for (uint b = 0; b < num_blocks; b++) {
            acc += o_partials[((size_t)q * num_blocks + b) * Dim + tid] * weights[b];
        }
        merged[tid] = (total > 0.0f) ? acc / total : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float out = 0.0f;
        for (uint j = 0; j < Dim; j++) out += merged[j] * val_rotation[j * Dim + tid];
        output[q * Dim + tid] = static_cast<OutT>(out);
        """

    // MARK: - Multi-query verify (raw-K and affine-K with turbo V)

    /// Multi-query attention, pass 1, for a block of query rows at a device-side
    /// position: speculative verify, and prefill chunks over a compressed cache.
    /// Shaped like mlx's GQA-packed MMA verify kernel. One threadgroup per
    /// (KV head, 8-row query chunk, key partition) carries every query head of
    /// the KV head as an 8-row stripe (`thread_position_in_threadgroup.z`), split
    /// into two halves of the head dimension (`.y`). Each 32-key block is
    /// dequantized once into threadgroup memory, keys and then values (in the
    /// rotated value basis), and every dot product runs on simdgroup MMA.
    /// Row `r` of chunk `c` sits at `position + 8c + r` and sees keys up to it.
    private static let verifyPass1Prologue = """
        constexpr int BK = 32;
        constexpr int QL = 8;
        constexpr int H2 = Dim / 2;
        constexpr int NT = H2 / 8;
        constexpr uint VAL_LEVELS = 1u << ValueBits;
        constexpr uint VAL_MASK = VAL_LEVELS - 1u;
        constexpr float NEG = -3.402823466e+38f;

        const uint N = params[0];
        const uint s_pad = params[1];
        const uint parts = params[2];
        const uint k_stride = params[3];
        const uint v_stride = params[4];
        const float scale = fparams[0];
        const int pos0 = position[0];

        const uint lane = thread_index_in_simdgroup;
        const uint dhalf = thread_position_in_threadgroup.y;
        const uint stripe = thread_position_in_threadgroup.z;
        const uint kvb = threadgroup_position_in_grid.x;
        const uint chunk = threadgroup_position_in_grid.y;
        const uint part = threadgroup_position_in_grid.z;
        const uint qbh = kvb * Rep + stripe;
        const uint tix = (stripe * 2 + dhalf) * 32 + lane;
        constexpr uint NTHREADS = Rep * 64;

        threadgroup float sS[2 * Rep * QL * BK];
        threadgroup T sP[Rep * QL * BK];
        threadgroup float sFactor[Rep * QL];
        threadgroup uint4 sKV4[BK * Dim * 2 / 16];
        threadgroup T* sKV = (threadgroup T*)sKV4;
        threadgroup float cb[VAL_LEVELS];
        if (tix < VAL_LEVELS) cb[tix] = val_codebook[tix];

        const uint span = ((N + parts * BK - 1) / (parts * BK)) * BK;
        const uint p0 = part * span;
        const uint pEnd = min(p0 + span, N);
        // A partition wholly past the chunk's last row contributes nothing.
        const int chunkLast = pos0 + int(chunk * QL + QL - 1);
        const uint loopEnd = (int(p0) > chunkLast) ? p0 : pEnd;

        simdgroup_matrix<T, 8, 8> Qf[NT];
        const device T* qBase = q_in + (size_t)(qbh * s_pad + chunk * QL) * Dim;
        for (int t = 0; t < NT; ++t) {
            simdgroup_load(Qf[t], qBase, Dim, ulong2(dhalf * H2 + t * 8, 0));
        }

        // The C-fragment row of this lane's thread_elements, and the row and
        // column quarter it owns in the scalar softmax.
        const int fragRow = int(((lane >> 2) & 4) + ((lane >> 1) & 3));
        const int smRow = int(lane >> 2);
        const int smCol = int(lane & 3) * (BK / 4);
        const int rowPos = pos0 + int(chunk * QL) + smRow;

        simdgroup_matrix<float, 8, 8> O[NT];
        for (int t = 0; t < NT; ++t) O[t] = simdgroup_matrix<float, 8, 8>(0);
        float mRun = NEG;
        float lRun = 0.0f;

        threadgroup float* sSMine = sS + (dhalf * Rep + stripe) * (QL * BK);
        threadgroup float* sSOther = sS + ((1 - dhalf) * Rep + stripe) * (QL * BK);
        threadgroup T* sPMine = sP + stripe * (QL * BK);

        for (uint n0 = p0; n0 < loopEnd; n0 += BK) {

        """

    private static let verifyStageRawK = """
            constexpr uint V4R = Dim * 2 / 16;
            for (uint i = tix; i < BK * V4R; i += NTHREADS) {
                const uint n = i / V4R;
                uint4 val = uint4(0);
                if (n0 + n < N) {
                    val = ((const device uint4*)(k_raw + ((size_t)kvb * k_stride + n0 + n) * Dim))[i % V4R];
                }
                sKV4[i] = val;
            }

        """

    /// Sixteen keys per thread step: one `uint4` of weights, one scale and
    /// bias (a group holds at least 32 keys), four 4-wide stores.
    private static let verifyStageAffineK = """
            constexpr uint WPR = Dim / 4;
            constexpr uint QPR = Dim / 16;
            constexpr uint GPR = Dim / KGroup;
            for (uint i = tix; i < BK * QPR; i += NTHREADS) {
                const uint n = i / QPR;
                const uint q = i % QPR;
                threadgroup vec<T, 4>* dst = (threadgroup vec<T, 4>*)(sKV + n * Dim + q * 16);
                if (n0 + n < N) {
                    const size_t row = (size_t)kvb * k_stride + n0 + n;
                    const uint4 words = ((const device uint4*)(k_weights + row * WPR))[q];
                    const uint g = (q * 16) / KGroup;
                    const float ks = float(k_scales[row * GPR + g]);
                    const float kb = float(k_biases[row * GPR + g]);
                    for (uint j = 0; j < 4; j++) {
                        const uint word = words[j];
                        dst[j] = vec<T, 4>(
                            T(float(word & 0xFFu) * ks + kb), T(float((word >> 8) & 0xFFu) * ks + kb),
                            T(float((word >> 16) & 0xFFu) * ks + kb), T(float(word >> 24) * ks + kb));
                    }
                } else {
                    for (uint j = 0; j < 4; j++) dst[j] = vec<T, 4>(T(0));
                }
            }

        """

    private static let verifyPass1Body = """
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Partial scores over this half's dims: S_half[8, BK].
            simdgroup_matrix<float, 8, 8> Sc[BK / 8];
            for (int c = 0; c < BK / 8; ++c) Sc[c] = simdgroup_matrix<float, 8, 8>(0);
            for (int c = 0; c < BK / 8; ++c) {
                for (int t = 0; t < NT; ++t) {
                    simdgroup_matrix<T, 8, 8> Kf;
                    simdgroup_load(Kf, sKV, (ulong)Dim, ulong2(dhalf * H2 + t * 8, c * 8), true);
                    simdgroup_multiply_accumulate(Sc[c], Qf[t], Kf, Sc[c]);
                }
            }
            for (int c = 0; c < BK / 8; ++c) {
                simdgroup_store(Sc[c], sSMine, (ulong)BK, ulong2(c * 8, 0));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Online softmax on row smRow, both halves redundantly.
            float sv[BK / 4];
            float rowMax = NEG;
            for (int j = 0; j < BK / 4; ++j) {
                const int c = smCol + j;
                const uint kpos = n0 + uint(c);
                const float s = (sSMine[smRow * BK + c] + sSOther[smRow * BK + c]) * scale;
                const bool masked = kpos >= pEnd || int(kpos) > rowPos;
                sv[j] = masked ? NEG : s;
                rowMax = max(rowMax, sv[j]);
            }
            rowMax = max(rowMax, simd_shuffle_xor(rowMax, 1));
            rowMax = max(rowMax, simd_shuffle_xor(rowMax, 2));
            const float mNew = max(mRun, rowMax);
            const float factor = mRun == NEG ? 1.0f : fast::exp(mRun - mNew);
            float rowSum = 0.0f;
            for (int j = 0; j < BK / 4; ++j) {
                const float p = sv[j] == NEG ? 0.0f : fast::exp(sv[j] - mNew);
                sv[j] = p;
                rowSum += p;
            }
            rowSum += simd_shuffle_xor(rowSum, 1);
            rowSum += simd_shuffle_xor(rowSum, 2);
            lRun = lRun * factor + rowSum;
            mRun = mNew;
            if (dhalf == 0) {
                if ((lane & 3) == 0) sFactor[stripe * QL + smRow] = factor;
                for (int j = 0; j < BK / 4; ++j) sPMine[smRow * BK + smCol + j] = T(sv[j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Values over the consumed key staging: codebook entry times norm,
            // in the rotated basis.
            constexpr uint G8 = Dim / 8;
            for (uint i = tix; i < BK * G8; i += NTHREADS) {
                const uint n = i / G8;
                const uint g = i % G8;
                threadgroup T* dst = sKV + n * Dim + g * 8;
                if (n0 + n < N) {
                    const size_t row = (size_t)kvb * v_stride + n0 + n;
                    const device uint32_t* vp = val_packed + row * ValuePackedWidth;
                    const uint bit = g * 8 * ValueBits;
                    const uint word = bit >> 5;
                    const uint shift = bit & 31u;
                    ulong bits = vp[word];
                    if (shift + 8 * ValueBits > 32) bits |= ((ulong)vp[word + 1]) << 32;
                    bits >>= shift;
                    const float nrm = val_norms[row];
                    for (uint j = 0; j < 8; j++) {
                        dst[j] = T(cb[(uint)(bits >> (j * ValueBits)) & VAL_MASK] * nrm);
                    }
                } else {
                    for (uint j = 0; j < 8; j++) dst[j] = T(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const float oFactor = sFactor[stripe * QL + fragRow];
            simdgroup_matrix<T, 8, 8> Pf[BK / 8];
            for (int c = 0; c < BK / 8; ++c) {
                simdgroup_load(Pf[c], sPMine, (ulong)BK, ulong2(c * 8, 0));
            }
            for (int t = 0; t < NT; ++t) {
                O[t].thread_elements()[0] *= oFactor;
                O[t].thread_elements()[1] *= oFactor;
                for (int c = 0; c < BK / 8; ++c) {
                    simdgroup_matrix<T, 8, 8> Vf;
                    simdgroup_load(Vf, sKV, (ulong)Dim, ulong2(dhalf * H2 + t * 8, c * 8));
                    simdgroup_multiply_accumulate(O[t], Pf[c], Vf, O[t]);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Unnormalized partials with each row's max and exp-sum, the contract
        // the GQA decode's pass 2 merges.
        const size_t row0 = (size_t)qbh * s_pad + chunk * QL;
        device float* oBase = o_partials + (row0 * parts + part) * Dim;
        for (int t = 0; t < NT; ++t) {
            simdgroup_store(O[t], oBase, (ulong)(parts * Dim), ulong2(dhalf * H2 + t * 8, 0));
        }
        if (dhalf == 0 && (lane & 3) == 0) {
            const size_t r = row0 + smRow;
            m_partials[r * parts + part] = mRun == NEG ? -INFINITY : mRun;
            l_partials[r * parts + part] = lRun;
        }
        """

    static let turboVerifyPass1RawKSource =
        verifyPass1Prologue + verifyStageRawK + verifyPass1Body

    static let turboVerifyPass1AffineKSource =
        verifyPass1Prologue + verifyStageAffineK + verifyPass1Body

    /// A tree block's mask: row `r` sees the committed rows and the block
    /// slots `j` whose bit is set in `tree_anc[r]`.
    private static let verifyPass1TreeBody = verifyPass1Body.replacingOccurrences(
        of: "const bool masked = kpos >= pEnd || int(kpos) > rowPos;",
        with: """
            const int slot = int(kpos) - pos0;
                            const bool masked = kpos >= pEnd || (slot >= 0
                                && (slot >= QL
                                    || ((tree_anc[chunk * QL + smRow] >> uint(slot)) & 1u) == 0u));
            """)

    static let turboVerifyPass1RawKTreeSource =
        verifyPass1Prologue + verifyStageRawK + verifyPass1TreeBody

    static let turboVerifyPass1AffineKTreeSource =
        verifyPass1Prologue + verifyStageAffineK + verifyPass1TreeBody

    // MARK: - Row dequantization (long query blocks)

    /// Expands the first rows of an affine-K / turbo-V cache to the activation
    /// dtype for a long query block, which then runs mlx's own attention:
    /// keys in their stored basis, values as codebook entry times norm in the
    /// rotated basis (the caller rotates the attention output once). One
    /// thread writes 8 consecutive dimensions of one row.
    private static let dequantRowsPrologue = """
        constexpr uint G8 = Dim / 8;
        constexpr uint VAL_MASK = (1u << ValueBits) - 1u;
        const uint rows = params[0];
        const uint k_stride = params[1];
        const uint v_stride = params[2];
        const uint g = thread_position_in_grid.x;
        const uint n = thread_position_in_grid.y;
        const uint kvb = thread_position_in_grid.z;
        if (g >= G8 || n >= rows) return;
        const size_t out_row = ((size_t)kvb * rows + n) * Dim + g * 8;

        """

    private static let dequantRowsAffineK = """
        {
            constexpr uint WPR = Dim / 4;
            constexpr uint GPR = Dim / KGroup;
            const size_t row = (size_t)kvb * k_stride + n;
            const uint grp = (g * 8) / KGroup;
            const float ks = float(k_scales[row * GPR + grp]);
            const float kb = float(k_biases[row * GPR + grp]);
            for (uint w = 0; w < 2; w++) {
                const uint word = k_weights[row * WPR + g * 2 + w];
                for (uint j = 0; j < 4; j++) {
                    k_out[out_row + w * 4 + j] = T(float((word >> (8 * j)) & 0xFFu) * ks + kb);
                }
            }
        }

        """

    private static let dequantRowsValues = """
        {
            const size_t row = (size_t)kvb * v_stride + n;
            const device uint32_t* vp = val_packed + row * ValuePackedWidth;
            const uint bit = g * 8 * ValueBits;
            const uint word = bit >> 5;
            const uint shift = bit & 31u;
            ulong bits = vp[word];
            if (shift + 8 * ValueBits > 32) bits |= ((ulong)vp[word + 1]) << 32;
            bits >>= shift;
            const float nrm = val_norms[row];
            for (uint j = 0; j < 8; j++) {
                v_out[out_row + j] = T(val_codebook[(uint)(bits >> (j * ValueBits)) & VAL_MASK] * nrm);
            }
        }
        """

    static let turboDequantRowsAffineKSource =
        dequantRowsPrologue + dequantRowsAffineK + dequantRowsValues

    static let turboDequantRowsValuesSource = dequantRowsPrologue + dequantRowsValues

    /// Value aggregation kernel: weighted sum of codebook-quantized values.
    ///
    /// output[d] = Σ_t weights[t] * norm[t] * codebook[val_idx[t,d]]
    /// Result is in rotated space, caller applies inverse rotation.
    ///
    /// Grid: (32, totalHeads, ceil(Dim/32))
    /// Threadgroup: (32, 1, 1)
    static let valueKernelSource = """
        // token_count/repeat_count/L via params buffer (vary per call; template
        // args would bake them into the kernel name = one compiled Metal
        // library cached per token).
        const uint token_count = params[0];
        const uint repeat_count = params[1];
        const uint chunk_len = params[2];
        constexpr uint MASK = (1u << Bits) - 1u;
        constexpr uint LEVELS = 1u << Bits;

        uint lane = thread_position_in_grid.x;
        uint head_idx = thread_position_in_grid.y;
        uint dim_block = thread_position_in_grid.z;

        uint d = dim_block * 32 + lane;
        if (d >= Dim) return;

        // Rows are flattened [B, heads, L]; divide out the chunk length
        // before the GQA repeat mapping.
        uint kv_head = (head_idx / chunk_len) / repeat_count;

        // Load codebook
        float cb[LEVELS];
        for (uint i = 0; i < LEVELS; i++) {
            cb[i] = codebook[i];
        }

        float acc = 0.0f;
        for (uint t = 0; t < (uint)token_count; t++) {
            float w = weights[head_idx * token_count + t];
            if (w < 1e-6f) continue;  // Sparse V: skip negligible attention weights

            float norm_val = norms[kv_head * token_count + t];
            const device uint32_t* packed_ptr = packed + kv_head * token_count * PackedWidth + t * PackedWidth;

            uint bit_offset = d * Bits;
            uint word_idx = bit_offset / 32;
            uint shift = bit_offset % 32;
            uint value = (packed_ptr[word_idx] >> shift);

            int spill = (int)shift + (int)Bits - 32;
            if (spill > 0) {
                value |= (packed_ptr[word_idx + 1] << ((uint)Bits - (uint)spill));
            }
            value &= MASK;

            acc += w * norm_val * cb[value];
        }

        output[head_idx * Dim + d] = acc;
        """
}

// MARK: - Kernel Dispatch Wrappers

enum TurboQuantKernelOps {

    // Kernel caches
    nonisolated(unsafe) private static var encodeKernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var scoreKernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var valueKernels: [String: MLXFast.MLXFastKernel] = [:]
    private static let lock = NSLock()

    /// Fused encode: norm + rotate + quantize + pack + norm correction in single GPU dispatch.
    ///
    /// - Parameters:
    ///   - input: Raw vectors [numRows, D] float32
    ///   - rotation: Rotation matrix Π [D, D] float32
    ///   - boundaries: Codebook boundaries [2^bits - 1] float32
    ///   - codebook: Centroids [2^bits] float32 (needed for norm correction)
    ///   - bits: Quantization bit-width
    ///   - dim: Vector dimension
    /// - Returns: (packed: [numRows, PackedWidth] uint32, norms: [numRows] float32)
    ///            norms are norm-corrected: original_norm / reconstruction_norm
    /// Kernel sources declare float buffers; normalize any half/bfloat input
    /// so a f16/bf16 activation stream is not silently reinterpreted.
    @inline(__always)
    private static func f32(_ x: MLXArray) -> MLXArray {
        x.dtype == .float32 ? x : x.asType(.float32)
    }

    static func fusedEncode(
        input: MLXArray,
        rotation: MLXArray,
        boundaries: MLXArray,
        codebook: MLXArray,
        bits: Int,
        dim: Int
    ) -> (packed: MLXArray, norms: MLXArray) {
        let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let key = "encode_nc_\(bits)_\(dim)"  // nc = norm-corrected

        let kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = encodeKernels[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_fused_encode_\(bits)_\(dim)",
                inputNames: ["input", "rotation", "boundaries", "codebook"],
                outputNames: ["packed_out", "norms_out"],
                source: TurboQuantMetalKernels.fusedEncodeSource,
                ensureRowContiguous: true
            )
            lock.lock()
            encodeKernels[key] = k
            lock.unlock()
            kernel = k
        }

        let numRows = input.dim(0)

        let results = kernel(
            [f32(input), f32(rotation), f32(boundaries), f32(codebook)],
            template: [
                ("Bits", bits), ("Dim", dim), ("PackedWidth", pw),
            ],
            grid: (dim, numRows, 1),
            threadGroup: (dim, 1, 1),
            outputShapes: [[numRows, pw], [numRows]],
            outputDTypes: [.uint32, .float32]
        )

        return (packed: results[0], norms: results[1])
    }

    /// Fused WHT encode: norm + WHT rotation + quantize + pack (raw norms, no correction).
    ///
    /// Same as fusedEncode but uses O(d log d) Walsh-Hadamard butterfly instead of
    /// O(d²) dense matmul. Only works for power-of-2 dimensions.
    ///
    /// WHT is orthogonal so norms are preserved, no norm correction needed.
    /// Codebook is NOT passed to the kernel (saves one buffer bind + GPU transfer).
    ///
    /// - Parameters:
    ///   - input: Raw vectors [numRows, D], float32, bfloat16 or float16
    ///   - whtSigns: Random ±1 signs [D] float32
    ///   - boundaries: Codebook boundaries [2^bits - 1] float32
    ///   - codebook: Centroids [2^bits] float32 (unused by kernel, kept in API for caller convenience)
    ///   - bits: Quantization bit-width
    ///   - dim: Vector dimension (must be power of 2)
    /// - Returns: (packed: [numRows, PackedWidth] uint32, norms: [numRows] float32, raw norms)
    static func fusedEncodeWHT(
        input: MLXArray,
        whtSigns: MLXArray,
        boundaries: MLXArray,
        codebook: MLXArray,
        bits: Int,
        dim: Int
    ) -> (packed: MLXArray, norms: MLXArray) {
        let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let logDim = Int(log2(Double(dim)))
        let key = "encode_wht_\(bits)_\(dim)_\(input.dtype)"

        let kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = encodeKernels[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_fused_encode_wht_\(bits)_\(dim)_\(input.dtype)",
                inputNames: ["input", "wht_signs", "boundaries"],
                outputNames: ["packed_out", "norms_out"],
                source: TurboQuantMetalKernels.fusedEncodeWHTSource,
                ensureRowContiguous: true
            )
            lock.lock()
            encodeKernels[key] = k
            lock.unlock()
            kernel = k
        }

        let numRows = input.dim(0)

        // NOTE: codebook no longer passed, WHT kernel stores raw norms (no norm correction)
        // The kernel loads `input` as float, so a bf16 or f16 row needs no cast.
        let results = kernel(
            [input, f32(whtSigns), f32(boundaries)],
            template: [
                ("Bits", bits), ("Dim", dim), ("PackedWidth", pw), ("LogDim", logDim),
            ],
            grid: (dim, numRows, 1),
            threadGroup: (dim, 1, 1),
            outputShapes: [[numRows, pw], [numRows]],
            outputDTypes: [.uint32, .float32]
        )

        return (packed: results[0], norms: results[1])
    }

    /// A cache's key buffers, `[B, H, capacity, *]`.
    enum RowKeyBuffers {
        case raw(MLXArray)
        case affine(weights: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int)
    }

    /// `TURBO_FUSED_WRITE=0` writes rows through separate encode, quantize and
    /// slice-update ops instead, as does an mlx without in-place outputs.
    static let rowWriteEnabled =
        ProcessInfo.processInfo.environment["TURBO_FUSED_WRITE"] != "0"
        && inplaceOutputsSupported

    /// Whether mlx updates a custom kernel's `inplace_<input>` output in that
    /// input's buffer (tesseract's mlx fork). Elsewhere the output is a fresh
    /// buffer, and a row write would lose every row it does not write.
    static let inplaceOutputsSupported: Bool = {
        let probe = MLXFast.metalKernel(
            name: "turbo_inplace_probe", inputNames: ["buf"], outputNames: ["inplace_buf"],
            source: "if (thread_position_in_grid.x == 0) { inplace_buf[0] = 7u; }")
        let pattern: [UInt32] = [0xA5A5_0001, 0xA5A5_0002, 0xA5A5_0003, 0xA5A5_0004]
        let written = probe(
            [MLXArray(pattern)], grid: (1, 1, 1), threadGroup: (1, 1, 1),
            outputShapes: [[pattern.count]], outputDTypes: [.uint32])[0]
        return written.asArray(UInt32.self) == [7] + pattern.dropFirst()
    }()

    /// Whether ``writeRows(keys:values:position:keyBuffers:valPacked:valNorms:whtSigns:boundaries:valueBits:dim:)``
    /// serves this shape: a WHT value codec and whole 32-lane key groups.
    static func rowWriteSupports(dim: Int, keyGroupSize: Int?) -> Bool {
        guard rowWriteEnabled, dim >= 32, dim <= 1024, dim & (dim - 1) == 0 else { return false }
        if let keyGroupSize { return keyGroupSize % 32 == 0 && dim % keyGroupSize == 0 }
        return true
    }

    /// Encodes `keys`/`values` (`[B, H, S, dim]`) and writes them at row
    /// `position` (an int32 `[1]`, possibly lazy) of every buffer in one
    /// dispatch. The returned buffers share the given ones' storage under
    /// `MLX_DYNSLICE_INPLACE`'s rule; mlx copies a buffer it cannot alias.
    static func writeRows(
        keys: MLXArray, values: MLXArray, position: MLXArray, keyBuffers: RowKeyBuffers,
        valPacked: MLXArray, valNorms: MLXArray, whtSigns: MLXArray, boundaries: MLXArray,
        valueBits: Int, dim: Int
    ) -> (keys: RowKeyBuffers, valPacked: MLXArray, valNorms: MLXArray) {
        let rows = keys.dim(0) * keys.dim(1) * keys.dim(2)
        var template: [(String, any KernelTemplateArg)] = [
            ("Bits", valueBits), ("Dim", dim),
            ("PackedWidth", TurboQuantPacking.packedWidth(count: dim, bits: valueBits)),
            ("LogDim", Int(log2(Double(dim)))),
        ]
        var inputs = [keys, values, f32(whtSigns), f32(boundaries), position]
        let kernel: MLXFast.MLXFastKernel
        let keyOutputs: Int
        switch keyBuffers {
        case .raw(let raw):
            kernel = gqaKernel(
                "turbo_write_rows_rawk_\(dim)_\(valueBits)_\(keys.dtype)_\(values.dtype)_\(raw.dtype)",
                inputNames: [
                    "keys", "values", "wht_signs", "boundaries", "position", "k_raw",
                    "val_packed", "val_norms",
                ],
                outputNames: ["inplace_k_raw", "inplace_val_packed", "inplace_val_norms"],
                source: TurboQuantMetalKernels.rowWriteRawKSource)
            template.append(("RawT", raw.dtype))
            inputs.append(raw)
            keyOutputs = 1
        case .affine(let weights, let scales, let biases, let groupSize):
            kernel = gqaKernel(
                "turbo_write_rows_affk_\(dim)_\(valueBits)_\(groupSize)_\(keys.dtype)_\(values.dtype)_\(scales.dtype)",
                inputNames: [
                    "keys", "values", "wht_signs", "boundaries", "position", "k_weights",
                    "k_scales", "k_biases", "val_packed", "val_norms",
                ],
                outputNames: [
                    "inplace_k_weights", "inplace_k_scales", "inplace_k_biases",
                    "inplace_val_packed", "inplace_val_norms",
                ],
                source: TurboQuantMetalKernels.rowWriteAffineKSource)
            template += [("KGroup", groupSize), ("ScaleT", scales.dtype)]
            inputs += [weights, scales, biases]
            keyOutputs = 3
        }
        inputs += [valPacked, valNorms]
        let buffers = Array(inputs[5...])
        let out = kernel(
            inputs, template: template,
            grid: (dim, rows, 1), threadGroup: (dim, 1, 1),
            outputShapes: buffers.map(\.shape), outputDTypes: buffers.map(\.dtype))
        let written: RowKeyBuffers
        switch keyBuffers {
        case .raw: written = .raw(out[0])
        case .affine(_, _, _, let groupSize):
            written = .affine(weights: out[0], scales: out[1], biases: out[2], groupSize: groupSize)
        }
        return (written, out[keyOutputs], out[keyOutputs + 1])
    }

    // Flash attention kernel caches
    nonisolated(unsafe) private static var flashPass1Kernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var flashPass1NR0Kernels: [String: MLXFast.MLXFastKernel] =
        [:]
    nonisolated(unsafe) private static var flashPass2Kernels: [String: MLXFast.MLXFastKernel] = [:]
    nonisolated(unsafe) private static var flashSinglePassKernels: [String: MLXFast.MLXFastKernel] =
        [:]

    /// Explicit NR0 override for profiling and device-specific tuning.
    /// Values must be powers of two so query groups remain naturally aligned.
    private static let flashNR0Override: Int? = {
        if let envValue = ProcessInfo.processInfo.environment["TURBO_FLASH_NR0"],
            let parsed = Int(envValue), parsed > 0, (parsed & (parsed - 1)) == 0
        {
            return parsed
        }
        return nil
    }()

    /// Choose the number of query rows processed by each SIMD group during decode.
    ///
    /// NR0=2 amortizes packed K/V decoding across two query rows and remains the
    /// long-context default. Immediately above the single-dispatch ceiling,
    /// however, that reuse does not repay the lost parallelism for small query
    /// batches. Keep one row per group in the measured crossover region.
    static func automaticFlashDecodeNR0(tokenCount: Int, totalQueries: Int, dim: Int) -> Int {
        if tokenCount > flashSinglePassMaxTokens && tokenCount <= 256
            && (16 ... 32).contains(totalQueries) && (dim == 64 || dim == 128)
        {
            return 1
        }
        return 2
    }

    private static func flashDecodeNR0(tokenCount: Int, totalQueries: Int, dim: Int) -> Int {
        flashNR0Override
            ?? automaticFlashDecodeNR0(
                tokenCount: tokenCount, totalQueries: totalQueries, dim: dim)
    }

    /// Default block size for TurboFlashAttention two-pass approach.
    /// Each SIMD group processes this many tokens per block.
    /// Tuned for M1 Max via sweep: B=64 wins or ties at all token counts (512-8192+).
    /// Smaller blocks = more parallelism but more pass-2 merge work.
    static let flashBlockSize = 64

    /// Short-context ceiling for the single-dispatch decode path. Longer
    /// contexts retain the established block-parallel two-pass implementation.
    static let flashSinglePassMaxTokens = 128

    private static func dispatchFlashSinglePass(
        rotatedQueries: MLXArray,
        keyPacked: MLXArray, keyNorms: MLXArray, keyCodebook: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int,
        keyBits: Int, valueBits: Int, dim: Int,
        valRotation: MLXArray?
    ) -> MLXArray {
        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let appliesRotation = valRotation != nil
        let key = "flash_single_\(keyBits)_\(valueBits)_\(dim)_\(appliesRotation)"
        let kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = flashSinglePassKernels[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_flash_single_\(keyBits)_\(valueBits)_\(dim)_\(appliesRotation)",
                inputNames: [
                    "q_rot", "key_packed", "key_norms", "key_codebook",
                    "val_packed", "val_norms", "val_codebook", "val_rotation", "params",
                ],
                outputNames: ["output"],
                source: TurboQuantMetalKernels.turboFlashSinglePassSource,
                ensureRowContiguous: true
            )
            lock.lock()
            flashSinglePassKernels[key] = k
            lock.unlock()
            kernel = k
        }

        let totalQ = rotatedQueries.dim(0)
        let params = MLXArray([UInt32(tokenCount), UInt32(repeatCount)])
        let rotation = valRotation.map(f32) ?? MLXArray.zeros([1])
        return kernel(
            [
                f32(rotatedQueries), keyPacked, f32(keyNorms), f32(keyCodebook),
                valPacked, f32(valNorms), f32(valCodebook), rotation, params,
            ],
            template: [
                ("KeyBits", keyBits), ("ValueBits", valueBits),
                ("Dim", dim), ("KeyPackedWidth", kpw), ("ValuePackedWidth", vpw),
                ("ApplyRotation", appliesRotation ? 1 : 0),
            ],
            grid: (32, totalQ, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ, dim]],
            outputDTypes: [.float32]
        )[0]
    }

    /// Shared pass 1 dispatch, used by both causal and non-causal variants.
    private static func dispatchFlashPass1(
        source: String, cachePrefix: String,
        rotatedQueries: MLXArray,
        keyPacked: MLXArray, keyNorms: MLXArray, keyCodebook: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int,
        keyBits: Int, valueBits: Int, dim: Int,
        blockSize: Int, queryChunkLength: Int = 1, queryOffset: Int = 0
    ) -> (oPartials: MLXArray, mPartials: MLXArray, lPartials: MLXArray) {
        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let numBlocks = (tokenCount + blockSize - 1) / blockSize
        let totalQ = rotatedQueries.dim(0)

        let pass1Key = "\(cachePrefix)_\(keyBits)_\(valueBits)_\(dim)"
        let pass1Kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = flashPass1Kernels[pass1Key] {
            pass1Kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_\(cachePrefix)_\(keyBits)_\(valueBits)_\(dim)",
                inputNames: [
                    "q_rot", "key_packed", "key_norms", "key_codebook",
                    "val_packed", "val_norms", "val_codebook", "params",
                ],
                outputNames: ["o_partials", "m_partials", "l_partials"],
                source: source,
                ensureRowContiguous: true
            )
            lock.lock()
            flashPass1Kernels[pass1Key] = k
            lock.unlock()
            pass1Kernel = k
        }

        let template: [(String, Int)] = [
            ("KeyBits", keyBits), ("ValueBits", valueBits),
            ("Dim", dim), ("KeyPackedWidth", kpw), ("ValuePackedWidth", vpw),
            ("BlockSize", blockSize),
        ]
        // Varying values travel in a buffer, not templates, a template value
        // is baked into the kernel name and would JIT + cache one Metal
        // library per distinct token_count (i.e., per generated token).
        let params = MLXArray(
            [
                UInt32(tokenCount), UInt32(repeatCount), UInt32(numBlocks),
                UInt32(queryChunkLength), UInt32(queryOffset),
            ])

        let partials = pass1Kernel(
            [
                f32(rotatedQueries), keyPacked, f32(keyNorms), f32(keyCodebook),
                valPacked, f32(valNorms), f32(valCodebook), params,
            ],
            template: template,
            grid: (32, totalQ, numBlocks),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ * numBlocks, dim], [totalQ, numBlocks], [totalQ, numBlocks]],
            outputDTypes: [.float32, .float32, .float32]
        )

        return (oPartials: partials[0], mPartials: partials[1], lPartials: partials[2])
    }

    /// NR0 multi-row pass 1 dispatch, processes NR0 queries per SIMD group.
    ///
    /// Each SIMD group loads K/V packed data once and computes scores for NR0 queries.
    /// The grid Y dimension is totalQueries/NR0 instead of totalQueries.
    /// Output shapes are the same as NR0=1 (partials indexed by original q_idx).
    ///
    /// Precondition: totalQueries must be divisible by NR0.
    private static func dispatchFlashPass1NR0(
        rotatedQueries: MLXArray,
        keyPacked: MLXArray, keyNorms: MLXArray, keyCodebook: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int,
        keyBits: Int, valueBits: Int, dim: Int,
        blockSize: Int, nr0: Int
    ) -> (oPartials: MLXArray, mPartials: MLXArray, lPartials: MLXArray) {
        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let numBlocks = (tokenCount + blockSize - 1) / blockSize
        let totalQ = rotatedQueries.dim(0)
        let queryGroups = totalQ / nr0

        let pass1Key = "flash_p1_nr0_\(keyBits)_\(valueBits)_\(dim)_\(nr0)"
        let pass1Kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = flashPass1NR0Kernels[pass1Key] {
            pass1Kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_flash_p1_nr0_\(keyBits)_\(valueBits)_\(dim)_\(nr0)",
                inputNames: [
                    "q_rot", "key_packed", "key_norms", "key_codebook",
                    "val_packed", "val_norms", "val_codebook", "params",
                ],
                outputNames: ["o_partials", "m_partials", "l_partials"],
                source: TurboQuantMetalKernels.turboFlashPass1NR0Source,
                ensureRowContiguous: true
            )
            lock.lock()
            flashPass1NR0Kernels[pass1Key] = k
            lock.unlock()
            pass1Kernel = k
        }

        let template: [(String, Int)] = [
            ("KeyBits", keyBits), ("ValueBits", valueBits),
            ("Dim", dim), ("KeyPackedWidth", kpw), ("ValuePackedWidth", vpw),
            ("BlockSize", blockSize), ("NR0", nr0),
        ]
        let params = MLXArray(
            [UInt32(tokenCount), UInt32(repeatCount), UInt32(numBlocks), 1, 0] as [UInt32])

        // Grid Y = queryGroups (totalQ / NR0), not totalQ
        let partials = pass1Kernel(
            [
                f32(rotatedQueries), keyPacked, f32(keyNorms), f32(keyCodebook),
                valPacked, f32(valNorms), f32(valCodebook), params,
            ],
            template: template,
            grid: (32, queryGroups, numBlocks),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ * numBlocks, dim], [totalQ, numBlocks], [totalQ, numBlocks]],
            outputDTypes: [.float32, .float32, .float32]
        )

        return (oPartials: partials[0], mPartials: partials[1], lPartials: partials[2])
    }

    /// NR0 multi-row causal pass 1 dispatch, processes NR0 queries with per-row causal masking.
    ///
    /// Same as dispatchFlashPass1NR0 but supports causal masking for L>1 prefill.
    /// Each query in the NR0 group has its own causal boundary.
    ///
    /// Precondition: totalQueries must be divisible by NR0.
    private static func dispatchFlashPass1NR0Causal(
        rotatedQueries: MLXArray,
        keyPacked: MLXArray, keyNorms: MLXArray, keyCodebook: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int,
        keyBits: Int, valueBits: Int, dim: Int,
        blockSize: Int, nr0: Int,
        queryChunkLength: Int, queryOffset: Int
    ) -> (oPartials: MLXArray, mPartials: MLXArray, lPartials: MLXArray) {
        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let numBlocks = (tokenCount + blockSize - 1) / blockSize
        let totalQ = rotatedQueries.dim(0)
        let queryGroups = totalQ / nr0

        let pass1Key = "flash_p1_nr0_causal_\(keyBits)_\(valueBits)_\(dim)_\(nr0)"
        let pass1Kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = flashPass1NR0Kernels[pass1Key] {
            pass1Kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_flash_p1_nr0_causal_\(keyBits)_\(valueBits)_\(dim)_\(nr0)",
                inputNames: [
                    "q_rot", "key_packed", "key_norms", "key_codebook",
                    "val_packed", "val_norms", "val_codebook", "params",
                ],
                outputNames: ["o_partials", "m_partials", "l_partials"],
                source: TurboQuantMetalKernels.turboFlashPass1NR0CausalSource,
                ensureRowContiguous: true
            )
            lock.lock()
            flashPass1NR0Kernels[pass1Key] = k
            lock.unlock()
            pass1Kernel = k
        }

        let template: [(String, Int)] = [
            ("KeyBits", keyBits), ("ValueBits", valueBits),
            ("Dim", dim), ("KeyPackedWidth", kpw), ("ValuePackedWidth", vpw),
            ("BlockSize", blockSize), ("NR0", nr0),
        ]
        let params = MLXArray(
            [
                UInt32(tokenCount), UInt32(repeatCount), UInt32(numBlocks),
                UInt32(queryChunkLength), UInt32(queryOffset),
            ])

        let partials = pass1Kernel(
            [
                f32(rotatedQueries), keyPacked, f32(keyNorms), f32(keyCodebook),
                valPacked, f32(valNorms), f32(valCodebook), params,
            ],
            template: template,
            grid: (32, queryGroups, numBlocks),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ * numBlocks, dim], [totalQ, numBlocks], [totalQ, numBlocks]],
            outputDTypes: [.float32, .float32, .float32]
        )

        return (oPartials: partials[0], mPartials: partials[1], lPartials: partials[2])
    }

    /// Shared pass 2 dispatch, with optional fused output rotation.
    ///
    /// When `valRotation` is provided, the inverse value rotation (Π_val) is applied
    /// in-kernel using threadgroup shared memory, eliminating a separate MLX matmul dispatch.
    /// Output is in original (non-rotated) space.
    ///
    /// When `valRotation` is nil, output is in rotated V space (caller must apply inverse rotation).
    private static func dispatchFlashPass2(
        oPartials: MLXArray, mPartials: MLXArray, lPartials: MLXArray,
        dim: Int, numBlocks: Int, totalQ: Int,
        valRotation: MLXArray? = nil
    ) -> MLXArray {
        let fused = valRotation != nil
        let pass2Key = fused ? "flash_p2_fused_\(dim)" : "flash_p2_\(dim)"
        let pass2Kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = flashPass2Kernels[pass2Key] {
            pass2Kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k: MLXFast.MLXFastKernel
            if fused {
                k = MLXFast.metalKernel(
                    name: "turbo_flash_p2_fused_\(dim)",
                    inputNames: [
                        "o_partials", "m_partials", "l_partials", "val_rotation", "params",
                    ],
                    outputNames: ["output"],
                    source: TurboQuantMetalKernels.turboFlashPass2FusedRotSource,
                    ensureRowContiguous: true
                )
            } else {
                k = MLXFast.metalKernel(
                    name: "turbo_flash_p2_\(dim)",
                    inputNames: ["o_partials", "m_partials", "l_partials", "params"],
                    outputNames: ["output"],
                    source: TurboQuantMetalKernels.turboFlashPass2Source,
                    ensureRowContiguous: true
                )
            }
            lock.lock()
            flashPass2Kernels[pass2Key] = k
            lock.unlock()
            pass2Kernel = k
        }

        let params = MLXArray([UInt32(numBlocks)])
        let inputs: [MLXArray] =
            fused
            ? [oPartials, mPartials, lPartials, valRotation!, params]
            : [oPartials, mPartials, lPartials, params]

        return pass2Kernel(
            inputs,
            template: [
                ("Dim", dim)
            ],
            grid: (32, totalQ, 1),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ, dim]],
            outputDTypes: [.float32]
        )[0]
    }

    nonisolated(unsafe) private static var quantizePackKernels: [String: MLXFast.MLXFastKernel] =
        [:]

    /// Quantize + pack + norm-correct pre-rotated calibrated vectors.
    /// `rotated` is [rows, dim] from MSECodec.rotatedUnit, `rawNorms` [rows].
    static func fusedQuantizePackScaled(
        rotated: MLXArray, rawNorms: MLXArray, scale: MLXArray,
        boundaries: MLXArray, codebook: MLXArray, bits: Int, dim: Int
    ) -> (packed: MLXArray, norms: MLXArray) {
        let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let key = "qpack_scaled_\(bits)_\(dim)"
        let kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = quantizePackKernels[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_qpack_scaled_\(bits)_\(dim)",
                inputNames: ["rotated", "norms_in", "scale", "boundaries", "codebook"],
                outputNames: ["packed_out", "norms_out"],
                source: TurboQuantMetalKernels.fusedQuantizePackScaledSource,
                ensureRowContiguous: true
            )
            lock.lock()
            quantizePackKernels[key] = k
            lock.unlock()
            kernel = k
        }
        let numRows = rotated.dim(0)
        let results = kernel(
            [f32(rotated), f32(rawNorms), f32(scale), f32(boundaries), f32(codebook)],
            template: [("Bits", bits), ("Dim", dim), ("PackedWidth", pw)],
            grid: (dim, numRows, 1),
            threadGroup: (dim, 1, 1),
            outputShapes: [[numRows, pw], [numRows]],
            outputDTypes: [.uint32, .float32]
        )
        return (packed: results[0], norms: results[1])
    }

    /// TurboFlashAttention: two-pass fused Score + Online Softmax + Value.
    ///
    /// Pass 1: Parallelizes across (query × token_block) pairs. Each SIMD group processes
    ///         BlockSize tokens, producing partial online softmax state (m, l, o[D]).
    /// Pass 2: Merges partial states across blocks to produce final normalized output.
    ///
    /// Eliminates intermediate score and attention weight arrays entirely.
    ///
    /// - Parameter valRotation: Optional [D, D] inverse value rotation matrix. When provided,
    ///   rotation is fused into pass 2, eliminating a separate MLX matmul dispatch.
    ///   Output is in original space. When nil, output is in rotated V space.
    /// - Parameter blockSize: Tokens per block (default: flashBlockSize). Smaller = more parallelism
    ///   but more pass-2 merge work. Must be > 0.
    /// - Returns: Output [totalQ, D] float32
    /// Pass-1 dispatch, raw-fp16 K + turbo-V. Reuses the packed-K flash
    /// structure and pass 2.
    private static func dispatchFlashPass1RawK(
        rotatedQueries: MLXArray, rawKeys: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int, valueBits: Int, dim: Int, blockSize: Int
    ) -> (oPartials: MLXArray, mPartials: MLXArray, lPartials: MLXArray) {
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let numBlocks = (tokenCount + blockSize - 1) / blockSize
        let totalQ = rotatedQueries.dim(0)
        let key = "flash_p1_rawk_\(valueBits)_\(dim)_\(rawKeys.dtype)"
        let kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = flashPass1Kernels[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_flash_p1_rawk_\(valueBits)_\(dim)_\(rawKeys.dtype)",
                inputNames: [
                    "q_rot", "k_raw", "val_packed", "val_norms", "val_codebook", "params",
                ],
                outputNames: ["o_partials", "m_partials", "l_partials"],
                source: TurboQuantMetalKernels.turboFlashPass1RawKSource,
                ensureRowContiguous: true
            )
            lock.lock()
            flashPass1Kernels[key] = k
            lock.unlock()
            kernel = k
        }
        let params = MLXArray(
            [UInt32(tokenCount), UInt32(repeatCount), UInt32(numBlocks), 1, 0] as [UInt32])
        // rawKeys stays in its native dtype (KT template): a per-step cast
        // would copy the whole growing cache every token and push bfloat16
        // values above the float16 range to infinity.
        let partials = kernel(
            [
                f32(rotatedQueries), rawKeys,
                valPacked, f32(valNorms), f32(valCodebook), params,
            ],
            template: [
                ("ValueBits", valueBits), ("Dim", dim), ("ValuePackedWidth", vpw),
                ("BlockSize", blockSize), ("KT", rawKeys.dtype),
            ],
            grid: (32, totalQ, numBlocks),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ * numBlocks, dim], [totalQ, numBlocks], [totalQ, numBlocks]],
            outputDTypes: [.float32, .float32, .float32]
        )
        return (partials[0], partials[1], partials[2])
    }

    /// Pass-1 dispatch, 8-bit affine K + turbo-V.
    private static func dispatchFlashPass1AffineK(
        rotatedQueries: MLXArray,
        kWeights: MLXArray, kScales: MLXArray, kBiases: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int, valueBits: Int, dim: Int, kGroup: Int,
        blockSize: Int
    ) -> (oPartials: MLXArray, mPartials: MLXArray, lPartials: MLXArray) {
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let numBlocks = (tokenCount + blockSize - 1) / blockSize
        let totalQ = rotatedQueries.dim(0)
        let key = "flash_p1_affk_\(valueBits)_\(dim)_\(kGroup)_\(kScales.dtype)_\(kBiases.dtype)"
        let kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = flashPass1Kernels[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name:
                    "turbo_flash_p1_affk_\(valueBits)_\(dim)_\(kGroup)_\(kScales.dtype)_\(kBiases.dtype)",
                inputNames: [
                    "q_rot", "k_weights", "k_scales", "k_biases",
                    "val_packed", "val_norms", "val_codebook", "params",
                ],
                outputNames: ["o_partials", "m_partials", "l_partials"],
                source: TurboQuantMetalKernels.turboFlashPass1AffineKSource,
                ensureRowContiguous: true
            )
            lock.lock()
            flashPass1Kernels[key] = k
            lock.unlock()
            kernel = k
        }
        let params = MLXArray(
            [UInt32(tokenCount), UInt32(repeatCount), UInt32(numBlocks), 1, 0] as [UInt32])
        // kScales/kBiases stay in their native dtype (KScaleT/KBiasT
        // templates): they are cache-resident and grow with token_count, so
        // a per-step cast would copy them every token.
        let partials = kernel(
            [
                f32(rotatedQueries), kWeights, kScales, kBiases,
                valPacked, f32(valNorms), f32(valCodebook), params,
            ],
            template: [
                ("ValueBits", valueBits), ("Dim", dim), ("ValuePackedWidth", vpw),
                ("BlockSize", blockSize), ("KGroup", kGroup),
                ("KScaleT", kScales.dtype), ("KBiasT", kBiases.dtype),
            ],
            grid: (32, totalQ, numBlocks),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ * numBlocks, dim], [totalQ, numBlocks], [totalQ, numBlocks]],
            outputDTypes: [.float32, .float32, .float32]
        )
        return (partials[0], partials[1], partials[2])
    }

    /// Flash decode, raw-fp16 K + turbo-V (single decode step, L=1).
    static func turboFlashRawK(
        rotatedQueries: MLXArray, rawKeys: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int, valueBits: Int, dim: Int,
        valRotation: MLXArray? = nil, blockSize: Int? = nil
    ) -> MLXArray {
        let bs = blockSize ?? flashBlockSize
        let numBlocks = (tokenCount + bs - 1) / bs
        let totalQ = rotatedQueries.dim(0)
        let (o, m, l) = dispatchFlashPass1RawK(
            rotatedQueries: rotatedQueries, rawKeys: rawKeys,
            valPacked: valPacked, valNorms: valNorms, valCodebook: valCodebook,
            tokenCount: tokenCount, repeatCount: repeatCount, valueBits: valueBits,
            dim: dim, blockSize: bs)
        return dispatchFlashPass2(
            oPartials: o, mPartials: m, lPartials: l, dim: dim, numBlocks: numBlocks,
            totalQ: totalQ, valRotation: valRotation)
    }

    /// Flash decode, 8-bit affine K + turbo-V (single decode step, L=1).
    static func turboFlashAffineK(
        rotatedQueries: MLXArray,
        kWeights: MLXArray, kScales: MLXArray, kBiases: MLXArray,
        valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        tokenCount: Int, repeatCount: Int, valueBits: Int, dim: Int, kGroup: Int,
        valRotation: MLXArray? = nil, blockSize: Int? = nil
    ) -> MLXArray {
        let bs = blockSize ?? flashBlockSize
        let numBlocks = (tokenCount + bs - 1) / bs
        let totalQ = rotatedQueries.dim(0)
        let (o, m, l) = dispatchFlashPass1AffineK(
            rotatedQueries: rotatedQueries,
            kWeights: kWeights, kScales: kScales, kBiases: kBiases,
            valPacked: valPacked, valNorms: valNorms, valCodebook: valCodebook,
            tokenCount: tokenCount, repeatCount: repeatCount, valueBits: valueBits,
            dim: dim, kGroup: kGroup, blockSize: bs)
        return dispatchFlashPass2(
            oPartials: o, mPartials: m, lPartials: l, dim: dim, numBlocks: numBlocks,
            totalQ: totalQ, valRotation: valRotation)
    }

    // MARK: - GQA flash decode

    nonisolated(unsafe) private static var gqaFlashKernels: [String: MLXFast.MLXFastKernel] = [:]

    /// `TURBO_FLASH_GQA=0` keeps the per-query-head kernels, for A/B runs.
    private static let gqaFlashEnabled =
        ProcessInfo.processInfo.environment["TURBO_FLASH_GQA"] != "0"

    // Defaults measured on an M3 Max at head dim 256, 24 over 4 heads,
    // 8K–64K tokens (TurboQuantDecodeMicrobench).

    /// SIMD groups per pass-1 threadgroup.
    static let gqaFlashSimdGroups = 2

    /// Fewest tokens a pass-1 threadgroup scores.
    static let gqaFlashMinTokensPerBlock = 64

    /// Most pass-1 blocks per query head (pass 2 reads them all).
    static let gqaFlashMaxBlocks = 64

    /// Query heads one pass-1 threadgroup scores: the largest divisor of the
    /// GQA repeat up to 3. Each group decodes the block again, but six heads'
    /// queries and accumulators in one lane cost more in registers (about
    /// 1.9x the time of three at 64K) than the second decode does.
    static func gqaFlashHeadsPerGroup(repeatCount: Int) -> Int {
        (1 ... min(3, repeatCount)).last { repeatCount % $0 == 0 } ?? 1
    }

    /// Key storage the GQA decode reads in place: whole buffers shaped
    /// `[B * nKVHeads, allocatedRows, ...]`.
    enum GQAKeys {
        case raw(MLXArray)
        case affine(weights: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int)
    }

    nonisolated(unsafe) private static var blockCountArrays: [Int: MLXArray] = [:]

    /// Pass 2's block count, one array per count for the process: a decode
    /// step would otherwise build it from host memory on every call.
    private static func blockCountArray(_ count: Int) -> MLXArray {
        lock.lock()
        defer { lock.unlock() }
        if let cached = blockCountArrays[count] { return cached }
        let array = MLXArray([UInt32(count)])
        blockCountArrays[count] = array
        return array
    }

    /// Whether the GQA decode serves this shape. Each lane holds `dim / 32`
    /// dimensions and their packed values in one 32-bit span; an affine key
    /// group must not split a lane.
    static func gqaFlashSupports(
        dim: Int, repeatCount: Int, valueBits: Int, keyGroupSize: Int?
    ) -> Bool {
        guard gqaFlashEnabled, dim % 32 == 0, dim >= 128, (1 ... 8).contains(repeatCount),
            (2 ... 4).contains(valueBits)
        else { return false }
        let dimsPerLane = dim / 32
        guard dimsPerLane * valueBits <= 32 else { return false }
        if let keyGroupSize {
            guard dimsPerLane % 4 == 0, keyGroupSize % dimsPerLane == 0, dim % keyGroupSize == 0
            else { return false }
        }
        return true
    }

    private static func gqaKernel(
        _ key: String, inputNames: [String], outputNames: [String], source: String
    ) -> MLXFast.MLXFastKernel {
        lock.lock()
        if let cached = gqaFlashKernels[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        let kernel = MLXFast.metalKernel(
            name: key, inputNames: inputNames, outputNames: outputNames, source: source,
            ensureRowContiguous: true)
        lock.lock()
        gqaFlashKernels[key] = kernel
        lock.unlock()
        return kernel
    }

    /// Flash decode (L=1) for raw-K and affine-K caches with turbo values,
    /// scoring every query head of a KV head from one decode of its block.
    ///
    /// - Parameters:
    ///   - queries: `[B * nQHeads, dim]` in the activation dtype; the kernel
    ///     applies `scale` as it loads them.
    ///   - valPacked: whole value buffer, `[B * nKVHeads, rows, packedWidth]`.
    ///   - valNorms: whole norm buffer, `[B * nKVHeads, rows]`.
    ///   - valRotation: `[dim, dim]` inverse value rotation.
    ///   - simdGroups, maxBlocks, headsPerGroup: overrides for tests and
    ///     benchmarks; `maxBlocks` is capped at `dim`, and a `headsPerGroup`
    ///     that does not divide `repeatCount` becomes `repeatCount`.
    /// - Returns: `[B * nQHeads, dim]` in the queries' dtype, in the original
    ///   value space.
    static func turboFlashGQA(
        queries: MLXArray, scale: Float = 1, keys: GQAKeys,
        valPacked: MLXArray, valNorms: MLXArray,
        valCodebook: MLXArray, valRotation: MLXArray,
        tokenCount: Int, repeatCount: Int, valueBits: Int, dim: Int,
        simdGroups: Int? = nil, maxBlocks: Int? = nil, headsPerGroup: Int? = nil
    ) -> MLXArray {
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let totalQ = queries.dim(0)
        let kvHeads = totalQ / repeatCount
        let nsg = simdGroups ?? gqaFlashSimdGroups
        var hps = headsPerGroup ?? gqaFlashHeadsPerGroup(repeatCount: repeatCount)
        if hps < 1 || repeatCount % hps != 0 { hps = repeatCount }
        // Pass 2 weighs one block per thread, so blocks stay <= dim.
        let targetBlocks = min(maxBlocks ?? gqaFlashMaxBlocks, dim)
        let tokensPerBlock = max(
            gqaFlashMinTokensPerBlock, (tokenCount + targetBlocks - 1) / targetBlocks)
        let numBlocks = max(1, (tokenCount + tokensPerBlock - 1) / tokensPerBlock)

        let kernel: MLXFast.MLXFastKernel
        var inputs: [MLXArray] = [queries]
        var template: [(String, any KernelTemplateArg)] = [
            ("Dim", dim), ("Rep", repeatCount), ("HPS", hps), ("ValueBits", valueBits),
            ("NSG", nsg), ("ValuePackedWidth", vpw),
        ]
        let keyRows: Int
        switch keys {
        case .raw(let rawKeys):
            kernel = gqaKernel(
                "turbo_flash_gqa_p1_rawk_\(dim)_\(repeatCount)_\(hps)_\(valueBits)_\(nsg)_\(rawKeys.dtype)_\(queries.dtype)",
                inputNames: [
                    "q_in", "k_raw", "val_packed", "val_norms", "val_codebook", "params",
                ],
                outputNames: ["o_partials", "m_partials", "l_partials"],
                source: TurboQuantMetalKernels.turboFlashGQAPass1RawKSource)
            inputs.append(rawKeys)
            keyRows = rawKeys.dim(1)
        case .affine(let weights, let scales, let biases, let groupSize):
            kernel = gqaKernel(
                "turbo_flash_gqa_p1_affk_\(dim)_\(repeatCount)_\(hps)_\(valueBits)_\(nsg)_\(groupSize)_\(scales.dtype)_\(biases.dtype)_\(queries.dtype)",
                inputNames: [
                    "q_in", "k_weights", "k_scales", "k_biases",
                    "val_packed", "val_norms", "val_codebook", "params",
                ],
                outputNames: ["o_partials", "m_partials", "l_partials"],
                source: TurboQuantMetalKernels.turboFlashGQAPass1AffineKSource)
            inputs += [weights, scales, biases]
            template.append(("KGroup", groupSize))
            keyRows = weights.dim(1)
        }
        let params = MLXArray(
            [
                UInt32(tokenCount), UInt32(tokensPerBlock), UInt32(numBlocks),
                UInt32(keyRows), UInt32(valPacked.dim(1)), scale.bitPattern,
            ] as [UInt32])
        inputs += [valPacked, f32(valNorms), f32(valCodebook), params]
        let partials = kernel(
            inputs,
            template: template,
            grid: (32 * nsg, kvHeads * (repeatCount / hps), numBlocks),
            threadGroup: (32 * nsg, 1, 1),
            outputShapes: [[totalQ * numBlocks, dim], [totalQ, numBlocks], [totalQ, numBlocks]],
            outputDTypes: [.float32, .float32, .float32]
        )

        let merge = gqaKernel(
            "turbo_flash_gqa_p2_\(dim)",
            inputNames: ["o_partials", "m_partials", "l_partials", "val_rotation", "params"],
            outputNames: ["output"],
            source: TurboQuantMetalKernels.turboFlashGQAPass2Source)
        return merge(
            [
                partials[0], partials[1], partials[2], f32(valRotation),
                blockCountArray(numBlocks),
            ],
            template: [("Dim", dim), ("OutT", queries.dtype)],
            grid: (dim, totalQ, 1),
            threadGroup: (dim, 1, 1),
            outputShapes: [[totalQ, dim]],
            outputDTypes: [queries.dtype]
        )[0]
    }

    /// Whether the multi-query verify kernel serves this shape: every query
    /// head of a KV head fits one threadgroup's 32 KB, and the staged K/V use
    /// the 2-byte activation dtype.
    static func verifyAttentionSupports(
        dim: Int, repeatCount: Int, valueBits: Int, keyGroupSize: Int?,
        queryDType: DType, rawKeyDType: DType?
    ) -> Bool {
        guard verifyKernelEnabled, dim == 128 || dim == 256, repeatCount >= 1,
            (2 ... 4).contains(valueBits), queryDType == .bfloat16 || queryDType == .float16
        else { return false }
        if let rawKeyDType, rawKeyDType != queryDType { return false }
        if let keyGroupSize, keyGroupSize % 16 != 0 || dim % keyGroupSize != 0 { return false }
        // Per query head: two score halves, the probabilities and the factors.
        let bytes = repeatCount * (2 * 8 * 32 * 4 + 8 * 32 * 2 + 8 * 4) + 32 * dim * 2 + 64
        return bytes <= 32 * 1024
    }

    static let verifyKernelEnabled = ProcessInfo.processInfo.environment["TURBO_VERIFY_MMA"] != "0"

    /// Key partitions for the verify kernel. Measured on the Qwen3.8-27B
    /// verify shape (8 rows, 24 query heads over 4 KV heads, dim 256), 16
    /// partitions beat 32 and 64 from 2K to 32K rows and 32 beat the rest at
    /// 64K (`TurboQuantDecodeMicrobench/testVerifyKernelPartitions`).
    static func verifyPartitions(visibleLength: Int, dim: Int) -> Int {
        min(dim, visibleLength <= 32_768 ? 16 : 32)
    }

    /// Multi-query attention over a raw-K or affine-K cache with turbo values.
    /// Query row `s` sits at `position + s` and sees the cache rows up to it,
    /// so rows written past the committed offset (a verify pass in flight)
    /// are visible only to the queries that follow them.
    ///
    /// - Parameters:
    ///   - queries: `[B, nQHeads, S, dim]` in the activation dtype, unscaled.
    ///   - keys, valPacked, valNorms: whole buffers, `[B * nKVHeads, rows, ...]`.
    ///   - position: `[1]` int32, possibly lazy: the position of query row 0.
    ///   - visibleLength: rows the pass may read; at least `position + S`.
    ///   - treeAncestry: a tree block's `[S]` uint32 slot masks (bit `j` of
    ///     row `i`: row `i` sees the row at `position + j`), `S <= 8`; nil
    ///     for a chain.
    /// - Returns: `[B, nQHeads, S, dim]` float32 in the original value space.
    static func turboVerifyAttention(
        queries: MLXArray, keys: GQAKeys,
        valPacked: MLXArray, valNorms: MLXArray,
        valCodebook: MLXArray, valRotation: MLXArray,
        position: MLXArray, visibleLength: Int, scale: Float,
        repeatCount: Int, valueBits: Int, dim: Int, partitions: Int? = nil,
        treeAncestry: MLXArray? = nil
    ) -> MLXArray {
        let B = queries.dim(0)
        let nQHeads = queries.dim(1)
        let S = queries.dim(2)
        let kvRows = B * nQHeads / repeatCount
        let chunks = (S + 7) / 8
        let sPad = chunks * 8
        var q = queries
        if sPad != S {
            q = padded(q, widths: [0, 0, IntOrPair((0, sPad - S)), 0])
        }
        let totalRows = B * nQHeads * sPad
        let parts = min(partitions ?? verifyPartitions(visibleLength: visibleLength, dim: dim), dim)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)

        let kernel: MLXFast.MLXFastKernel
        var inputs: [MLXArray] = [q.reshaped([totalRows, dim])]
        var template: [(String, any KernelTemplateArg)] = [
            ("T", queries.dtype), ("Dim", dim), ("Rep", repeatCount),
            ("ValueBits", valueBits), ("ValuePackedWidth", vpw),
        ]
        precondition(treeAncestry == nil || chunks == 1, "a tree block has at most 8 rows")
        let tree = treeAncestry != nil
        let treeName = tree ? "_tree" : ""
        let treeInput = tree ? ["tree_anc"] : []
        let keyRows: Int
        switch keys {
        case .raw(let rawKeys):
            kernel = gqaKernel(
                "turbo_verify_p1_rawk\(treeName)_\(dim)_\(repeatCount)_\(valueBits)_\(queries.dtype)",
                inputNames: [
                    "q_in", "k_raw", "val_packed", "val_norms", "val_codebook", "position",
                    "params", "fparams",
                ] + treeInput,
                outputNames: ["o_partials", "m_partials", "l_partials"],
                source: tree
                    ? TurboQuantMetalKernels.turboVerifyPass1RawKTreeSource
                    : TurboQuantMetalKernels.turboVerifyPass1RawKSource)
            inputs.append(rawKeys)
            keyRows = rawKeys.dim(1)
        case .affine(let weights, let scales, let biases, let groupSize):
            kernel = gqaKernel(
                "turbo_verify_p1_affk\(treeName)_\(dim)_\(repeatCount)_\(valueBits)_\(groupSize)_\(queries.dtype)_\(scales.dtype)_\(biases.dtype)",
                inputNames: [
                    "q_in", "k_weights", "k_scales", "k_biases", "val_packed", "val_norms",
                    "val_codebook", "position", "params", "fparams",
                ] + treeInput,
                outputNames: ["o_partials", "m_partials", "l_partials"],
                source: tree
                    ? TurboQuantMetalKernels.turboVerifyPass1AffineKTreeSource
                    : TurboQuantMetalKernels.turboVerifyPass1AffineKSource)
            inputs += [weights, scales, biases]
            template.append(("KGroup", groupSize))
            keyRows = weights.dim(1)
        }
        let params = MLXArray(
            [
                UInt32(visibleLength), UInt32(sPad), UInt32(parts), UInt32(keyRows),
                UInt32(valPacked.dim(1)),
            ] as [UInt32])
        inputs += [
            valPacked, f32(valNorms), f32(valCodebook), position.asType(.int32).reshaped([1]),
            params, MLXArray([scale]),
        ]
        if let treeAncestry {
            // Padded rows see the committed rows only.
            let bits = treeAncestry.asType(.uint32).reshaped([-1])
            inputs.append(
                S == sPad ? bits : concatenated([bits, MLXArray.zeros([sPad - S], dtype: .uint32)]))
        }
        let partials = kernel(
            inputs,
            template: template,
            grid: (32 * kvRows, 2 * chunks, repeatCount * parts),
            threadGroup: (32, 2, repeatCount),
            outputShapes: [[totalRows * parts, dim], [totalRows, parts], [totalRows, parts]],
            outputDTypes: [.float32, .float32, .float32]
        )

        let merge = gqaKernel(
            "turbo_flash_gqa_p2_\(dim)",
            inputNames: ["o_partials", "m_partials", "l_partials", "val_rotation", "params"],
            outputNames: ["output"],
            source: TurboQuantMetalKernels.turboFlashGQAPass2Source)
        let merged = merge(
            [
                partials[0], partials[1], partials[2], f32(valRotation),
                blockCountArray(parts),
            ],
            template: [("Dim", dim), ("OutT", DType.float32)],
            grid: (dim, totalRows, 1),
            threadGroup: (dim, 1, 1),
            outputShapes: [[totalRows, dim]],
            outputDTypes: [.float32]
        )[0].reshaped([B, nQHeads, sPad, dim])
        return sPad == S ? merged : merged[0..., 0..., ..<S, 0...]
    }

    /// The first `rows` rows of a raw-K or affine-K cache with turbo values,
    /// in `dtype`: keys in their stored basis, values in the rotated basis.
    ///
    /// - Parameters:
    ///   - keys, valPacked, valNorms: whole buffers, `[B * nKVHeads, rows, ...]`.
    /// - Returns: `[B * nKVHeads, rows, dim]` keys and rotated values.
    static func turboDequantizeRows(
        keys: GQAKeys, valPacked: MLXArray, valNorms: MLXArray, valCodebook: MLXArray,
        rows: Int, valueBits: Int, dim: Int, dtype: DType
    ) -> (keys: MLXArray, rotatedValues: MLXArray) {
        let kvRows = valPacked.dim(0)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        var template: [(String, any KernelTemplateArg)] = [
            ("T", dtype), ("Dim", dim), ("ValueBits", valueBits), ("ValuePackedWidth", vpw),
        ]
        let grid = (dim / 8, rows, kvRows)
        let threadGroup = (min(dim / 8, 32), 8, 1)
        switch keys {
        case .raw(let rawKeys):
            let kernel = gqaKernel(
                "turbo_dequant_rows_v_\(dim)_\(valueBits)_\(dtype)",
                inputNames: ["val_packed", "val_norms", "val_codebook", "params"],
                outputNames: ["v_out"],
                source: TurboQuantMetalKernels.turboDequantRowsValuesSource)
            let values = kernel(
                [
                    valPacked, f32(valNorms), f32(valCodebook),
                    MLXArray([UInt32(rows), UInt32(rawKeys.dim(1)), UInt32(valPacked.dim(1))]),
                ],
                template: template, grid: grid, threadGroup: threadGroup,
                outputShapes: [[kvRows, rows, dim]], outputDTypes: [dtype])[0]
            return (rawKeys[0..., ..<rows, 0...], values)
        case .affine(let weights, let scales, let biases, let groupSize):
            template.append(("KGroup", groupSize))
            let kernel = gqaKernel(
                "turbo_dequant_rows_kv_\(dim)_\(valueBits)_\(groupSize)_\(dtype)_\(scales.dtype)",
                inputNames: [
                    "k_weights", "k_scales", "k_biases", "val_packed", "val_norms",
                    "val_codebook", "params",
                ],
                outputNames: ["k_out", "v_out"],
                source: TurboQuantMetalKernels.turboDequantRowsAffineKSource)
            let out = kernel(
                [
                    weights, scales, biases, valPacked, f32(valNorms), f32(valCodebook),
                    MLXArray([UInt32(rows), UInt32(weights.dim(1)), UInt32(valPacked.dim(1))]),
                ],
                template: template, grid: grid, threadGroup: threadGroup,
                outputShapes: [[kvRows, rows, dim], [kvRows, rows, dim]],
                outputDTypes: [dtype, dtype])
            return (out[0], out[1])
        }
    }

    static func turboFlashAttention(
        rotatedQueries: MLXArray,
        keyPacked: MLXArray,
        keyNorms: MLXArray,
        keyCodebook: MLXArray,
        valPacked: MLXArray,
        valNorms: MLXArray,
        valCodebook: MLXArray,
        tokenCount: Int,
        repeatCount: Int,
        keyBits: Int,
        valueBits: Int,
        dim: Int,
        valRotation: MLXArray? = nil,
        blockSize: Int? = nil,
        singleDispatch: Bool? = nil,
        nr0: Int? = nil
    ) -> MLXArray {
        // An explicit block size is used by block-sweep callers and therefore
        // always selects the two-pass implementation. The optional override is
        // internal test/benchmark plumbing; normal callers use the conservative
        // context ceiling above.
        // `nr0` is also internal test/benchmark plumbing for exact A/B comparisons.
        let kernelSupportsSingleDispatch =
            tokenCount > 0
            && (2 ... 4).contains(keyBits) && (2 ... 4).contains(valueBits)
            && dim > 0 && dim <= 256 && blockSize == nil
        let defaultSingleDispatch =
            kernelSupportsSingleDispatch && tokenCount <= flashSinglePassMaxTokens
        let useSingleDispatch =
            singleDispatch.map { $0 && kernelSupportsSingleDispatch } ?? defaultSingleDispatch
        if useSingleDispatch {
            return dispatchFlashSinglePass(
                rotatedQueries: rotatedQueries,
                keyPacked: keyPacked, keyNorms: keyNorms, keyCodebook: keyCodebook,
                valPacked: valPacked, valNorms: valNorms, valCodebook: valCodebook,
                tokenCount: tokenCount, repeatCount: repeatCount,
                keyBits: keyBits, valueBits: valueBits, dim: dim,
                valRotation: valRotation)
        }

        let blockSize = blockSize ?? flashBlockSize
        let numBlocks = (tokenCount + blockSize - 1) / blockSize
        let totalQ = rotatedQueries.dim(0)
        let nr0 =
            nr0
            ?? flashDecodeNR0(tokenCount: tokenCount, totalQueries: totalQ, dim: dim)

        // Use NR0 multi-row kernel when totalQ is evenly divisible by NR0 and NR0 > 1.
        // Falls back to NR0=1 (original kernel) for remainder queries or when NR0=1.
        // NR0 groups read one kv head (kv_indices[0]) for all rows; only valid
        // when the GQA repeat factor is a multiple of nr0 so aligned groups can
        // never span a KV-head boundary. MHA and odd repeat factors take the
        // per-row kernel.
        let useNR0 =
            nr0 > 1 && totalQ % nr0 == 0 && totalQ >= nr0 && repeatCount % nr0 == 0

        let oPartials: MLXArray
        let mPartials: MLXArray
        let lPartials: MLXArray

        if useNR0 {
            (oPartials, mPartials, lPartials) = dispatchFlashPass1NR0(
                rotatedQueries: rotatedQueries,
                keyPacked: keyPacked, keyNorms: keyNorms, keyCodebook: keyCodebook,
                valPacked: valPacked, valNorms: valNorms, valCodebook: valCodebook,
                tokenCount: tokenCount, repeatCount: repeatCount,
                keyBits: keyBits, valueBits: valueBits, dim: dim,
                blockSize: blockSize, nr0: nr0
            )
        } else {
            (oPartials, mPartials, lPartials) = dispatchFlashPass1(
                source: TurboQuantMetalKernels.turboFlashPass1Source,
                cachePrefix: "flash_p1",
                rotatedQueries: rotatedQueries,
                keyPacked: keyPacked, keyNorms: keyNorms, keyCodebook: keyCodebook,
                valPacked: valPacked, valNorms: valNorms, valCodebook: valCodebook,
                tokenCount: tokenCount, repeatCount: repeatCount,
                keyBits: keyBits, valueBits: valueBits, dim: dim,
                blockSize: blockSize
            )
        }

        return dispatchFlashPass2(
            oPartials: oPartials, mPartials: mPartials, lPartials: lPartials,
            dim: dim, numBlocks: numBlocks, totalQ: totalQ,
            valRotation: valRotation
        )
    }

    /// TurboFlashAttention with causal masking for L>1 prefill chunks.
    ///
    /// Same as turboFlashAttention but each query position only attends to tokens
    /// where t <= queryOffset + q_within_L. Eliminates the need to materialize
    /// the full [nQHeads, L, T] score matrix for causal masking.
    ///
    /// - Parameter queryChunkLength: Number of query positions in the chunk (L)
    /// - Parameter queryOffset: Absolute position of the first query in the chunk
    /// - Returns: Output [totalQ, D] float32
    static func turboFlashAttentionCausal(
        rotatedQueries: MLXArray,
        keyPacked: MLXArray,
        keyNorms: MLXArray,
        keyCodebook: MLXArray,
        valPacked: MLXArray,
        valNorms: MLXArray,
        valCodebook: MLXArray,
        tokenCount: Int,
        repeatCount: Int,
        keyBits: Int,
        valueBits: Int,
        dim: Int,
        queryChunkLength: Int,
        queryOffset: Int,
        valRotation: MLXArray? = nil,
        blockSize: Int? = nil
    ) -> MLXArray {
        let blockSize = blockSize ?? flashBlockSize
        let numBlocks = (tokenCount + blockSize - 1) / blockSize
        let totalQ = rotatedQueries.dim(0)
        let nr0 = flashNR0Override ?? 2

        // NR0 groups read one kv head (kv_indices[0]) for all rows; only valid
        // when the GQA repeat factor is a multiple of nr0 so aligned groups can
        // never span a KV-head boundary. MHA and odd repeat factors take the
        // per-row kernel.
        let useNR0 =
            nr0 > 1 && totalQ % nr0 == 0 && totalQ >= nr0 && repeatCount % nr0 == 0

        let oPartials: MLXArray
        let mPartials: MLXArray
        let lPartials: MLXArray

        if useNR0 {
            (oPartials, mPartials, lPartials) = dispatchFlashPass1NR0Causal(
                rotatedQueries: rotatedQueries,
                keyPacked: keyPacked, keyNorms: keyNorms, keyCodebook: keyCodebook,
                valPacked: valPacked, valNorms: valNorms, valCodebook: valCodebook,
                tokenCount: tokenCount, repeatCount: repeatCount,
                keyBits: keyBits, valueBits: valueBits, dim: dim,
                blockSize: blockSize, nr0: nr0,
                queryChunkLength: queryChunkLength, queryOffset: queryOffset
            )
        } else {
            (oPartials, mPartials, lPartials) = dispatchFlashPass1(
                source: TurboQuantMetalKernels.turboFlashPass1CausalSource,
                cachePrefix: "flash_p1_causal",
                rotatedQueries: rotatedQueries,
                keyPacked: keyPacked, keyNorms: keyNorms, keyCodebook: keyCodebook,
                valPacked: valPacked, valNorms: valNorms, valCodebook: valCodebook,
                tokenCount: tokenCount, repeatCount: repeatCount,
                keyBits: keyBits, valueBits: valueBits, dim: dim,
                blockSize: blockSize,
                queryChunkLength: queryChunkLength, queryOffset: queryOffset
            )
        }

        return dispatchFlashPass2(
            oPartials: oPartials, mPartials: mPartials, lPartials: lPartials,
            dim: dim, numBlocks: numBlocks, totalQ: totalQ,
            valRotation: valRotation
        )
    }

    /// Compute Q×K attention scores from packed codebook indices.
    ///
    /// - Parameters:
    ///   - rotatedQueries: Pre-rotated queries [totalQ, D] (already scaled)
    ///   - packed: Packed key indices [totalKVHeads, T, PackedWidth] uint32
    ///   - norms: Key norms [totalKVHeads, T] float32
    ///   - codebook: Centroids [2^bits] float32
    ///   - tokenCount: Number of cached tokens
    ///   - repeatCount: GQA repeat factor (nQHeads / nKVHeads)
    ///   - bits: MSE bit-width
    ///   - dim: Vector dimension
    /// - Returns: Scores [totalQ, T] float32
    static func mseScore(
        rotatedQueries: MLXArray,
        packed: MLXArray,
        norms: MLXArray,
        codebook: MLXArray,
        tokenCount: Int,
        repeatCount: Int,
        bits: Int,
        dim: Int, queryChunkLength: Int = 1
    ) -> MLXArray {
        let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let key = "\(bits)_\(dim)"

        let kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = scoreKernels[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_score_\(bits)_\(dim)",
                inputNames: ["q_rot", "packed", "norms", "codebook", "params"],
                outputNames: ["scores"],
                source: TurboQuantMetalKernels.scoreKernelSource,
                ensureRowContiguous: true
            )
            lock.lock()
            scoreKernels[key] = k
            lock.unlock()
            kernel = k
        }

        let totalQ = rotatedQueries.dim(0)

        let params = MLXArray([UInt32(tokenCount), UInt32(repeatCount), UInt32(queryChunkLength)])
        return kernel(
            [f32(rotatedQueries), packed, f32(norms), f32(codebook), params],
            template: [
                ("Bits", bits), ("Dim", dim), ("PackedWidth", pw),
            ],
            grid: (32, totalQ, tokenCount),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalQ, tokenCount]],
            outputDTypes: [.float32]
        )[0]
    }

    /// Compute weighted sum of values from packed codebook indices.
    ///
    /// Result is in ROTATED space, caller must apply inverse rotation.
    ///
    /// - Returns: [totalHeads, D] float32 (rotated space)
    static func mseWeightedSum(
        weights: MLXArray,
        packed: MLXArray,
        norms: MLXArray,
        codebook: MLXArray,
        tokenCount: Int,
        repeatCount: Int,
        bits: Int,
        dim: Int, queryChunkLength: Int = 1
    ) -> MLXArray {
        let pw = TurboQuantPacking.packedWidth(count: dim, bits: bits)
        let key = "\(bits)_\(dim)"

        let kernel: MLXFast.MLXFastKernel
        lock.lock()
        if let cached = valueKernels[key] {
            kernel = cached
            lock.unlock()
        } else {
            lock.unlock()
            let k = MLXFast.metalKernel(
                name: "turbo_value_\(bits)_\(dim)",
                inputNames: ["weights", "packed", "norms", "codebook", "params"],
                outputNames: ["output"],
                source: TurboQuantMetalKernels.valueKernelSource,
                ensureRowContiguous: true
            )
            lock.lock()
            valueKernels[key] = k
            lock.unlock()
            kernel = k
        }

        let totalHeads = weights.dim(0)
        let dimBlocks = (dim + 31) / 32

        let params = MLXArray([UInt32(tokenCount), UInt32(repeatCount), UInt32(queryChunkLength)])
        return kernel(
            [f32(weights), packed, f32(norms), f32(codebook), params],
            template: [
                ("Bits", bits), ("Dim", dim), ("PackedWidth", pw),
            ],
            grid: (32, totalHeads, dimBlocks),
            threadGroup: (32, 1, 1),
            outputShapes: [[totalHeads, dim]],
            outputDTypes: [.float32]
        )[0]
    }
}
