// Copyright © 2026 Apple Inc.

import Foundation
import MLX

// MARK: - Tree verify attention

/// MLX's one-pass vector SDPA (`sdpa_vector`, the kernel a verify pass runs
/// below 1024 keys) for a tree block, with each row's own key read where a
/// chain block would hold it. Row `i` sees the rows before `position` and
/// block rows `position ..< position + depth[i]`, its ancestors, which the
/// tree caches at their depths; at index `position + depth[i]` it reads its
/// own key and value from block slot `slot[i]`. Each key keeps the index a
/// chain block gives it, so it lands in the same simdgroup in the same order
/// and the row reduces exactly as it does when its token is verified in a
/// chain block. The rest of the arithmetic is MLX's: the `fastmath_` name
/// compiles it the way the AOT metallib is compiled.
private func makeTreeAttentionKernel() -> MLXFast.MLXFastKernel? {
    let source = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = D / BD;
        typedef float U;

        thread U q[qk_per_thread];
        thread U k[qk_per_thread];
        thread U o[v_per_thread];

        threadgroup U outputs[BN * BD];
        threadgroup U max_scores[BN];
        threadgroup U sum_exp_scores[BN];

        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        const int q_batch_head_idx = threadgroup_position_in_grid.x;
        const int q_seq_idx = threadgroup_position_in_grid.y;
        const int kv_head_idx = q_batch_head_idx / GQA;
        const int o_offset = q_batch_head_idx * threadgroups_per_grid.y + q_seq_idx;
        const device T* qp = queries + o_offset * D + simd_lid * qk_per_thread;
        const int N = params[0];
        const size_t head_stride = (size_t)params[1] * D;
        const device T* kh = keys + kv_head_idx * head_stride + simd_lid * qk_per_thread;
        const device T* vh = values + kv_head_idx * head_stride + simd_lid * v_per_thread;
        device T* op = out + o_offset * D + simd_gid * v_per_thread;

        const int pos0 = position[0];
        const int self_index = pos0 + depth[q_seq_idx];
        const int self_row = pos0 + slot[q_seq_idx];
        const U sc = scale[0];

        for (int i = 0; i < qk_per_thread; i++) {
          q[i] = static_cast<U>(sc) * qp[i];
        }
        for (int i = 0; i < v_per_thread; i++) {
          o[i] = 0;
        }

        U max_score = -metal::numeric_limits<U>::max();
        U sum_exp_score = 0;

        for (int i = simd_gid; i < N; i += BN) {
          if (i <= self_index) {
            const size_t r = (size_t)(i == self_index ? self_row : i) * D;
            for (int j = 0; j < qk_per_thread; j++) {
              k[j] = kh[r + j];
            }
            U score = 0;
            for (int j = 0; j < qk_per_thread; j++) {
              score += q[j] * k[j];
            }
            score = simd_sum(score);

            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);

            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;

            for (int j = 0; j < v_per_thread; j++) {
              o[j] = o[j] * factor + exp_score * vh[r + j];
            }
          }
        }

        if (simd_lid == 0) {
          max_scores[simd_gid] = max_score;
          sum_exp_scores[simd_gid] = sum_exp_score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_score = max_scores[simd_lid];
        U new_max = simd_max(max_score);
        U factor = fast::exp(max_score - new_max);
        sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * factor);

        for (int i = 0; i < v_per_thread; i++) {
          outputs[simd_lid * BD + simd_gid] = o[i];
          threadgroup_barrier(mem_flags::mem_threadgroup);
          o[i] = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
          o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
          for (int i = 0; i < v_per_thread; i++) {
            op[i] = static_cast<T>(o[i]);
          }
        }
        """
    return MLXFast.metalKernel(
        name: "fastmath_dflash2_tree_sdpa_vector",
        inputNames: ["queries", "keys", "values", "position", "depth", "slot", "scale", "params"],
        outputNames: ["out"],
        source: source
    )
}

/// MLX's two-pass MMA SDPA, pass 1 (`sdpa_vector_2pass_1_mma`, the kernel a
/// verify pass runs from 1024 keys on: q [B, HQ, 8, 256], gqa <= 6), for a
/// tree block. Every key block runs as MLX's does; in the block that holds a
/// leaf row's depth index, the row's score and value for that column come
/// from its own key, through the same MMA sequence a chain block runs with
/// the key at that column: the K (then V) tile's row is patched, the
/// column group's MMAs rerun, and only the leaf row's results are kept.
/// Other rows add exact zeros in the extra MMAs.
private func makeTreeAttentionMMAKernel() -> MLXFast.MLXFastKernel? {
    let source = """
        constexpr int BK = 32;
        constexpr int QL = 8;
        constexpr int H2 = D / 2;
        constexpr int NT = H2 / 8;
        constexpr int V4R = D * 2 / 16;
        const float NEG = -metal::numeric_limits<float>::max();

        const uint3 tptg = threads_per_threadgroup;
        const uint3 tidtg = thread_position_in_threadgroup;
        const uint3 tid = threadgroup_position_in_grid;
        const uint3 tpg = threadgroups_per_grid;
        const uint simd_lid = thread_index_in_simdgroup;

        const int gqa = (int)tptg.z;
        const int stripe = (int)tidtg.z;
        const int dhalf = (int)tidtg.y;
        const int lane = (int)simd_lid;

        const int kv_head_idx = (int)tid.x;
        const int batch_idx = (int)tid.y;
        const int part_idx = params[2] + (int)tid.z;

        threadgroup float sS[2 * 6 * QL * BK];
        threadgroup T sP[6 * QL * BK];
        threadgroup float sFactor[6 * QL];
        threadgroup uint4 sKV4[BK * D * 2 / 16];
        threadgroup T* sKV = (threadgroup T*)sKV4;

        const int num_kv_heads = (int)tpg.x;
        const int num_q_heads = num_kv_heads * gqa;
        const int q_head_idx = gqa * kv_head_idx + stripe;
        const int q_batch_head_idx = batch_idx * num_q_heads + q_head_idx;
        const int q_seq_len = QL;

        const int N = params[0];
        const size_t k_head_stride = (size_t)params[1] * D;
        const size_t k_seq_stride = D;
        const float scale = scale_in[0];

        const int span_blocks = (N + BLOCKS * BK - 1) / (BLOCKS * BK);
        const int span = span_blocks * BK;
        const int p0 = part_idx * span;
        const int pEnd = min(p0 + span, N);

        const device T* kHead = keys + (size_t)(batch_idx * num_kv_heads + kv_head_idx) * k_head_stride;
        const device T* vHead = values + (size_t)(batch_idx * num_kv_heads + kv_head_idx) * k_head_stride;

        const int q_ld = D;
        const device T* qBase = queries + (size_t)q_batch_head_idx * 8 * D;
        simdgroup_matrix<T, 8, 8> Qf[NT];
        #pragma unroll
        for (int t = 0; t < NT; ++t) {
          simdgroup_load(Qf[t], qBase, q_ld, ulong2((ulong)(dhalf * H2 + t * 8), 0));
        }

        const int fragRow = (int)(((lane >> 2) & 4) + ((lane >> 1) & 3));
        const int smRow = lane >> 2;
        const int smCol = (lane & 3) * (BK / 4);

        simdgroup_matrix<float, 8, 8> O[NT];
        #pragma unroll
        for (int t = 0; t < NT; ++t) {
          O[t] = simdgroup_matrix<float, 8, 8>(0);
        }
        float mRun = NEG;
        float lRun = 0;

        // Tree: row smRow sees keys up to its own index, its depth. The
        // leaves' own indices bound the blocks that need a patch.
        const int pos0 = position[0];
        const int selfIndex = pos0 + depth[smRow];
        int leafLo = N;
        int leafHi = -1;
        for (int r = 0; r < QL; ++r) {
          if (TREE && slot[r] != depth[r]) {
            leafLo = min(leafLo, pos0 + depth[r]);
            leafHi = max(leafHi, pos0 + depth[r]);
          }
        }

        threadgroup float* sSMine = sS + (dhalf * gqa + stripe) * (QL * BK);
        threadgroup float* sSOther = sS + ((1 - dhalf) * gqa + stripe) * (QL * BK);
        threadgroup T* sPMine = sP + stripe * (QL * BK);

        const int tix = (int)(tidtg.z * 64 + tidtg.y * 32 + lane);
        const int nthreads = (int)(tptg.y * tptg.z) * 32;

        for (int n0 = p0; n0 < pEnd; n0 += BK) {
          for (int i = tix; i < BK * V4R; i += nthreads) {
            const int n = i / V4R;
            const int c4 = i % V4R;
            uint4 val = uint4(0);
            if (n0 + n < N) {
              val = *((const device uint4*)(kHead + (size_t)(n0 + n) * k_seq_stride) + c4);
            }
            sKV4[i] = val;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          simdgroup_matrix<float, 8, 8> Sc[BK / 8];
          #pragma unroll
          for (int c = 0; c < BK / 8; ++c) {
            Sc[c] = simdgroup_matrix<float, 8, 8>(0);
          }
          #pragma unroll
          for (int c = 0; c < BK / 8; ++c) {
            for (int t = 0; t < NT; ++t) {
              simdgroup_matrix<T, 8, 8> Kf;
              simdgroup_load(
                  Kf, sKV, (ulong)D,
                  ulong2((ulong)(dhalf * H2 + t * 8), (ulong)(c * 8)), true);
              simdgroup_multiply_accumulate(Sc[c], Qf[t], Kf, Sc[c]);
            }
          }
          #pragma unroll
          for (int c = 0; c < BK / 8; ++c) {
            simdgroup_store(Sc[c], sSMine, (ulong)BK, ulong2((ulong)(c * 8), 0));
          }

          // Tree: a leaf row whose depth index falls in this block scores
          // its own key there. Patch the K tile's row, rerun that column
          // group's MMAs, keep the leaf row's entry, restore the row.
          const bool anyLeaf = TREE && leafHi >= n0 && leafLo < n0 + BK;
          if (anyLeaf) {
            threadgroup float* scratch = (threadgroup float*)sP + (dhalf * 6 + stripe) * 64;
            for (int r = 0; r < QL; ++r) {
              const int idx = pos0 + depth[r];
              if (slot[r] == depth[r] || idx < n0 || idx >= n0 + BK) {
                continue;
              }
              const int c = idx - n0;
              const int g = c / 8;
              threadgroup_barrier(mem_flags::mem_threadgroup);
              for (int i = tix; i < V4R; i += nthreads) {
                sKV4[c * V4R + i] =
                    *((const device uint4*)(kHead + (size_t)(pos0 + slot[r]) * k_seq_stride) + i);
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              simdgroup_matrix<float, 8, 8> Sg = simdgroup_matrix<float, 8, 8>(0);
              for (int t = 0; t < NT; ++t) {
                simdgroup_matrix<T, 8, 8> Kf;
                simdgroup_load(
                    Kf, sKV, (ulong)D,
                    ulong2((ulong)(dhalf * H2 + t * 8), (ulong)(g * 8)), true);
                simdgroup_multiply_accumulate(Sg, Qf[t], Kf, Sg);
              }
              simdgroup_store(Sg, scratch, (ulong)8, ulong2(0, 0));
              simdgroup_barrier(mem_flags::mem_threadgroup);
              if (lane == 0) {
                sSMine[r * BK + c] = scratch[r * 8 + (c - g * 8)];
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              for (int i = tix; i < V4R; i += nthreads) {
                uint4 val = uint4(0);
                if (n0 + c < N) {
                  val = *((const device uint4*)(kHead + (size_t)(n0 + c) * k_seq_stride) + i);
                }
                sKV4[c * V4R + i] = val;
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          float sv[BK / 4];
          float rowMax = NEG;
          #pragma unroll
          for (int j = 0; j < BK / 4; ++j) {
            const int c = smCol + j;
            const int kpos = n0 + c;
            float s = (sSMine[smRow * BK + c] + sSOther[smRow * BK + c]) * scale;
            const bool masked = kpos >= pEnd || kpos >= N || kpos > selfIndex;
            sv[j] = masked ? NEG : s;
            rowMax = max(rowMax, sv[j]);
          }
          rowMax = max(rowMax, simd_shuffle_xor(rowMax, 1));
          rowMax = max(rowMax, simd_shuffle_xor(rowMax, 2));
          const float mNew = max(mRun, rowMax);
          const float factor = mRun == NEG ? 1.0f : fast::exp(mRun - mNew);
          float rowSum = 0;
          #pragma unroll
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
            if ((lane & 3) == 0) {
              sFactor[stripe * QL + smRow] = factor;
            }
            #pragma unroll
            for (int j = 0; j < BK / 4; ++j) {
              sPMine[smRow * BK + smCol + j] = (T)sv[j];
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          for (int i = tix; i < BK * V4R; i += nthreads) {
            const int n = i / V4R;
            const int c4 = i % V4R;
            uint4 val = uint4(0);
            if (n0 + n < N) {
              val = *((const device uint4*)(vHead + (size_t)(n0 + n) * k_seq_stride) + c4);
            }
            sKV4[i] = val;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          // Tree: the main P.V runs with the leaf rows' probabilities zeroed
          // (they add exact zeros); each leaf then adds its own row against
          // the V tile patched with its value, in the same column order.
          threadgroup T* sPMain = sPMine;
          if (anyLeaf) {
            sPMain = (threadgroup T*)sS + stripe * (QL * BK);
            if (dhalf == 0) {
              for (int i = lane; i < QL * BK; i += 32) {
                const int r = i / BK;
                const int idx = pos0 + depth[r];
                const bool leaf = slot[r] != depth[r] && idx >= n0 && idx < n0 + BK;
                sPMain[i] = leaf ? (T)0 : sPMine[i];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
          }

          const float oFactor = sFactor[stripe * QL + fragRow];
          simdgroup_matrix<T, 8, 8> Pf[BK / 8];
          #pragma unroll
          for (int c = 0; c < BK / 8; ++c) {
            simdgroup_load(Pf[c], sPMain, (ulong)BK, ulong2((ulong)(c * 8), 0));
          }
          #pragma unroll
          for (int t = 0; t < NT; ++t) {
            O[t].thread_elements()[0] *= oFactor;
            O[t].thread_elements()[1] *= oFactor;
            for (int c = 0; c < BK / 8; ++c) {
              simdgroup_matrix<T, 8, 8> Vf;
              simdgroup_load(
                  Vf, sKV, (ulong)D,
                  ulong2((ulong)(dhalf * H2 + t * 8), (ulong)(c * 8)));
              simdgroup_multiply_accumulate(O[t], Pf[c], Vf, O[t]);
            }
          }

          if (anyLeaf) {
            threadgroup T* sPLeaf = (threadgroup T*)sS + (6 + stripe) * (QL * BK);
            for (int r = 0; r < QL; ++r) {
              const int idx = pos0 + depth[r];
              if (slot[r] == depth[r] || idx < n0 || idx >= n0 + BK) {
                continue;
              }
              const int c = idx - n0;
              threadgroup_barrier(mem_flags::mem_threadgroup);
              for (int i = tix; i < V4R; i += nthreads) {
                sKV4[c * V4R + i] =
                    *((const device uint4*)(vHead + (size_t)(pos0 + slot[r]) * k_seq_stride) + i);
              }
              if (dhalf == 0) {
                for (int i = lane; i < QL * BK; i += 32) {
                  sPLeaf[i] = (i / BK == r) ? sPMine[i] : (T)0;
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              simdgroup_matrix<T, 8, 8> Pl[BK / 8];
              #pragma unroll
              for (int cc = 0; cc < BK / 8; ++cc) {
                simdgroup_load(Pl[cc], sPLeaf, (ulong)BK, ulong2((ulong)(cc * 8), 0));
              }
              #pragma unroll
              for (int t = 0; t < NT; ++t) {
                for (int cc = 0; cc < BK / 8; ++cc) {
                  simdgroup_matrix<T, 8, 8> Vf;
                  simdgroup_load(
                      Vf, sKV, (ulong)D,
                      ulong2((ulong)(dhalf * H2 + t * 8), (ulong)(cc * 8)));
                  simdgroup_multiply_accumulate(O[t], Pl[cc], Vf, O[t]);
                }
              }
              threadgroup_barrier(mem_flags::mem_threadgroup);
              for (int i = tix; i < V4R; i += nthreads) {
                uint4 val = uint4(0);
                if (n0 + c < N) {
                  val = *((const device uint4*)(vHead + (size_t)(n0 + c) * k_seq_stride) + i);
                }
                sKV4[c * V4R + i] = val;
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const int row0 = q_batch_head_idx * q_seq_len;
        device T* pOut = out + ((size_t)row0 * BLOCKS + part_idx) * D;
        threadgroup T* oStage = (threadgroup T*)sSMine;
        #pragma unroll
        for (int t = 0; t < NT; ++t) {
          simdgroup_matrix<T, 8, 8> Ot;
          Ot.thread_elements()[0] = (T)O[t].thread_elements()[0];
          Ot.thread_elements()[1] = (T)O[t].thread_elements()[1];
          simdgroup_store(Ot, oStage, (ulong)8, ulong2(0, 0));
          simdgroup_barrier(mem_flags::mem_threadgroup);
          {
            const int r = (int)lane >> 2;
            const int c = ((int)lane & 3) * 2;
            if (r < q_seq_len) {
              pOut[(size_t)r * BLOCKS * D + dhalf * H2 + t * 8 + c] = oStage[r * 8 + c];
              pOut[(size_t)r * BLOCKS * D + dhalf * H2 + t * 8 + c + 1] = oStage[r * 8 + c + 1];
            }
          }
          simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (dhalf == 0 && (lane & 3) == 0 && smRow < q_seq_len) {
          sums[(row0 + smRow) * BLOCKS + part_idx] = lRun;
          maxs[(row0 + smRow) * BLOCKS + part_idx] = mRun;
        }
        """
    return MLXFast.metalKernel(
        name: "fastmath_dflash2_tree_sdpa_mma_pass1",
        inputNames: [
            "queries", "keys", "values", "position", "depth", "slot", "scale_in", "params",
        ],
        outputNames: ["out", "sums", "maxs"],
        source: source
    )
}

/// MLX's two-pass merge (`sdpa_vector_2pass_2`), reading partitions below
/// `split[0]` from the first pass-1 launch and the rest from the second; the
/// arithmetic is MLX's.
private func makeTreeAttentionMergeKernel() -> MLXFast.MLXFastKernel? {
    let source = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = D / BD;
        typedef float U;

        const uint3 tid = threadgroup_position_in_grid;
        const uint3 tpg = threadgroups_per_grid;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;

        thread U o[elem_per_thread] = {0};
        threadgroup U outputs[BN * BD];

        const int head_idx = tid.x;
        const int q_seq_idx = tid.y;
        const int q_offset = head_idx * tpg.y + q_seq_idx;
        const int cut = split[0];
        const device T* ppA = partials_a + q_offset * BLOCKS * D + simd_lid * elem_per_thread;
        const device T* ppB = partials_b + q_offset * BLOCKS * D + simd_lid * elem_per_thread;
        const device float* spA = sums_a + q_offset * BLOCKS;
        const device float* spB = sums_b + q_offset * BLOCKS;
        const device float* mpA = maxs_a + q_offset * BLOCKS;
        const device float* mpB = maxs_b + q_offset * BLOCKS;
        device T* op = out + q_offset * D + simd_gid * elem_per_thread;

        U sum_exp_score = 0.0;
        U max_score = -metal::numeric_limits<U>::max();

        for (int b = 0; b < BLOCKS / BN; ++b) {
          const int j = simd_lid + BN * b;
          max_score = max(max_score, j < cut ? mpA[j] : mpB[j]);
        }
        max_score = simd_max(max_score);

        for (int b = 0; b < BLOCKS / BN; ++b) {
          const int j = simd_lid + BN * b;
          U factor = fast::exp((j < cut ? mpA[j] : mpB[j]) - max_score);
          sum_exp_score += factor * (j < cut ? spA[j] : spB[j]);
        }
        sum_exp_score = simd_sum(sum_exp_score);

        for (int b = 0; b < BLOCKS / BN; ++b) {
          const int j = simd_gid + BN * b;
          U factor = fast::exp((j < cut ? mpA[j] : mpB[j]) - max_score);
          const device T* pp = (j < cut ? ppA : ppB) + (size_t)j * D;
          for (int i = 0; i < elem_per_thread; i++) {
            o[i] += factor * static_cast<U>(pp[i]);
          }
        }

        for (int i = 0; i < elem_per_thread; i++) {
          outputs[simd_lid * BD + simd_gid] = o[i];
          threadgroup_barrier(mem_flags::mem_threadgroup);
          o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
          o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
          for (int i = 0; i < elem_per_thread; i++) {
            op[i] = static_cast<T>(o[i]);
          }
        }
        """
    return MLXFast.metalKernel(
        name: "fastmath_dflash2_tree_sdpa_merge",
        inputNames: ["partials_a", "sums_a", "maxs_a", "partials_b", "sums_b", "maxs_b", "split"],
        outputNames: ["out"],
        source: source
    )
}

private final class TreeAttentionKernelManager: Sendable {
    static let shared = TreeAttentionKernelManager()
    let kernel: MLXFast.MLXFastKernel?
    let mmaPass1: MLXFast.MLXFastKernel?
    let merge: MLXFast.MLXFastKernel?
    private init() {
        kernel = makeTreeAttentionKernel()
        mmaPass1 = makeTreeAttentionMMAKernel()
        merge = makeTreeAttentionMergeKernel()
    }
}

/// A tree block's attention over a whole bf16/f16 cache buffer, each row's
/// own key at its depth (see ``makeTreeAttentionKernel()``).
///
/// - Parameters:
///   - queries: `[B, HQ, S, D]`, row-contiguous.
///   - keys, values: the cache's whole buffers, `[B, HK, capacity, D]`,
///     holding the block's rows at `position` in slot order.
///   - position: `[1]` int32, possibly lazy: the block's first row.
///   - depths, slots: `[S]` int32 per row.
///   - visibleLength: rows the pass may read.
/// - Returns: `[B, HQ, S, D]`, or nil where MLX would run another kernel
///   than these two (the two-pass vector kernels) or the shape is not served.
public func dflash2TreeAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray, position: MLXArray,
    depths: MLXArray, slots: MLXArray, visibleLength: Int, scale: Float
) -> MLXArray? {
    let headDim = queries.dim(-1)
    guard queries.ndim == 4, keys.ndim == 4, values.shape == keys.shape,
        queries.dim(0) == keys.dim(0), queries.dim(1) % keys.dim(1) == 0,
        keys.dim(-1) == headDim, [64, 96, 128, 256].contains(headDim),
        queries.dtype == keys.dtype, values.dtype == keys.dtype,
        queries.dtype == .bfloat16 || queries.dtype == .float16,
        visibleLength <= keys.dim(2)
    else { return nil }
    let (B, HQ, S) = (queries.dim(0), queries.dim(1), queries.dim(2))
    let gqa = HQ / keys.dim(1)
    let manager = TreeAttentionKernelManager.shared
    if visibleLength >= 1024 {
        // MLX's pick from 1024 keys (Apple GPU families 'd' and 's'): the
        // MMA kernel for an 8-row block at head dim 256 and gqa <= 6.
        guard S == 8, headDim == 256, gqa <= 6, let pass1 = manager.mmaPass1,
            let merge = manager.merge
        else { return nil }
        let blocks = visibleLength < 2048 ? 32 : 64
        let span = 32 * ((visibleLength + blocks * 32 - 1) / (blocks * 32))
        // The block's rows start in the 15 keys before the visible end
        // (position is at most the upper bound, visibleLength - 8, and at
        // most 7 below it), so only partitions from that key on can hold a
        // leaf's own index. They run as their own launch with the leaves'
        // keys; the rest run without the tree code, which costs registers.
        let split = Swift.max(0, visibleLength - 15) / span
        let inputs = [
            queries, keys, values, position.asType(.int32).reshaped([1]),
            depths.asType(.int32).reshaped([S]), slots.asType(.int32).reshaped([S]),
            MLXArray([scale]),
        ]
        func launch(_ tree: Bool, partitions: Range<Int>) -> [MLXArray] {
            pass1(
                inputs + [
                    MLXArray([
                        Int32(visibleLength), Int32(keys.dim(2)), Int32(partitions.lowerBound),
                    ])
                ],
                template: [
                    ("T", queries.dtype), ("D", headDim), ("BLOCKS", blocks),
                    ("TREE", tree ? 1 : 0),
                ],
                grid: (keys.dim(1) * 32, B * 2, partitions.count * gqa),
                threadGroup: (32, 2, gqa),
                outputShapes: [
                    [B, HQ, S, blocks, headDim], [B, HQ, S, blocks], [B, HQ, S, blocks],
                ],
                outputDTypes: [queries.dtype, .float32, .float32])
        }
        // The leaf launch is encoded first: its few threadgroups then start
        // in the first wave instead of trailing the plain launch's.
        let leafSide = launch(true, partitions: split ..< blocks)
        let firstSide = split > 0 ? launch(false, partitions: 0 ..< split) : leafSide
        return merge(
            firstSide + leafSide + [MLXArray([Int32(split)])],
            template: [("T", queries.dtype), ("D", headDim), ("BLOCKS", blocks)],
            grid: (B * HQ * 1024, S, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [queries.shape],
            outputDTypes: [queries.dtype]
        )[0]
    }
    guard let kernel = manager.kernel else { return nil }
    return kernel(
        [
            queries, keys, values, position.asType(.int32).reshaped([1]),
            depths.asType(.int32).reshaped([S]), slots.asType(.int32).reshaped([S]),
            MLXArray([scale]), MLXArray([Int32(visibleLength), Int32(keys.dim(2))]),
        ],
        template: [("T", queries.dtype), ("D", headDim), ("GQA", HQ / keys.dim(1))],
        grid: (B * HQ * 1024, S, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [queries.shape],
        outputDTypes: [queries.dtype]
    )[0]
}
