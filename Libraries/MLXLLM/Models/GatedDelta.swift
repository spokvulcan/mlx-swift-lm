//
//  GatedDelta.swift
//  mlx-swift-lm
//
//  Port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py
//

import Foundation
import MLX
import MLXNN

// MARK: - Compute G

func computeGatedDeltaG(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
    let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    return decay.asType(a.dtype)
}

// MARK: - Fused Metal Kernel (computes g/beta inline — eliminates ~9 dispatch ops per call)

private func makeFusedGatedDeltaKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
    let maskSource = hasMask ? "mask[b_idx * T + t]" : "true"

    // Fused kernel: computes decay (g) and sigmoid (beta) inline from raw a, aLog, dtBias, b
    // instead of reading pre-computed g/beta arrays. Eliminates separate kernel dispatches for
    // computeGatedDeltaG (exp, softplus, negate, etc.) and sigmoid(b).
    let source = """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // a_raw, b_raw: [B, T, Hv] — raw projections
            auto a_raw_ = a_raw + b_idx * T * Hv;
            auto b_raw_ = b_raw + b_idx * T * Hv;
            // a_log: [Hv], dt_bias: [Hv] — per-head constants
            float a_log_f = static_cast<float>(a_log[hv_idx]);
            float exp_a_log = metal::exp(a_log_f);
            float dt_bias_f = static_cast<float>(dt_bias[hv_idx]);

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              if (\(maskSource)) {
                // Compute g (decay) inline: exp(-exp(aLog) * softplus(a + dtBias))
                float a_val = static_cast<float>(a_raw_[hv_idx]) + dt_bias_f;
                float sp = (a_val > 20.0f) ? a_val : metal::log(1.0f + metal::exp(a_val));
                float g_val = metal::exp(-exp_a_log * sp);

                // Compute beta (sigmoid) inline: 1 / (1 + exp(-b))
                float beta_val = 1.0f / (1.0f + metal::exp(-static_cast<float>(b_raw_[hv_idx])));

                float kv_mem = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] * g_val;
                  kv_mem += state[i] * k_[s_idx];
                }
                kv_mem = simd_sum(kv_mem);

                auto delta = (v_[dv_idx] - kv_mem) * beta_val;

                float out = 0.0f;
                for (int i = 0; i < n_per_t; ++i) {
                  auto s_idx = n_per_t * dk_idx + i;
                  state[i] = state[i] + k_[s_idx] * delta;
                  out += state[i] * q_[s_idx];
                }
                out = simd_sum(out);
                if (thread_index_in_simdgroup == 0) {
                  y[dv_idx] = static_cast<InT>(out);
                }
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              a_raw_ += Hv;
              b_raw_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<InT>(state[i]);
            }
        """

    var inputNames = ["q", "k", "v", "a_raw", "b_raw", "a_log", "dt_bias", "state_in", "T"]
    if hasMask {
        inputNames.append("mask")
    }

    let suffix = hasMask ? "_mask" : ""

    return MLXFast.metalKernel(
        name: "gated_delta_fused\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "state_out"],
        source: source
    )
}

private final class GatedDeltaKernelManager: Sendable {
    static let shared = GatedDeltaKernelManager()

    let fusedKernel: MLXFast.MLXFastKernel?
    let fusedKernelMasked: MLXFast.MLXFastKernel?

    private init() {
        fusedKernel = makeFusedGatedDeltaKernel(hasMask: false)
        fusedKernelMasked = makeFusedGatedDeltaKernel(hasMask: true)
    }
}

// MARK: - Fused Kernel Dispatch

private func gatedDeltaKernelFused(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    aRaw: MLXArray,
    bRaw: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = k.dim(0)
    let T = k.dim(1)
    let Hk = k.dim(2)
    let Dk = k.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)
    let inputType = q.dtype

    let selectedKernel: MLXFast.MLXFastKernel?
    var inputs: [MLXArray] = [q, k, v, aRaw, bRaw, aLog, dtBias, state, MLXArray(T)]
    if let mask {
        selectedKernel = GatedDeltaKernelManager.shared.fusedKernelMasked
        inputs.append(mask)
    } else {
        selectedKernel = GatedDeltaKernelManager.shared.fusedKernel
    }

    guard let kernel = selectedKernel else {
        fatalError("Fused gated delta kernel not available")
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", inputType),
            ("Dk", Dk),
            ("Dv", Dv),
            ("Hk", Hk),
            ("Hv", Hv),
        ],
        grid: (32, Dv, B * Hv),
        threadGroup: (32, 8, 1),
        outputShapes: [[B, T, Hv, Dv], state.shape],
        outputDTypes: [inputType, inputType]
    )

    return (outputs[0], outputs[1])
}

// MARK: - Ops Fallback

private func gatedDeltaStepOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let oldState = state
    let decay: MLXArray
    if g.ndim == 2 {
        decay = expandedDimensions(g, axes: [2, 3])
    } else if g.ndim == 3 {
        decay = expandedDimensions(g, axis: -2)
    } else {
        fatalError("Unsupported gating shape \(g.shape)")
    }

    var state = state * decay
    let kvMem = (state * expandedDimensions(k, axis: -2)).sum(axis: -1)
    let delta = (v - kvMem) * expandedDimensions(beta, axis: -1)
    state = state + expandedDimensions(k, axis: -2) * expandedDimensions(delta, axis: -1)
    let y = (state * expandedDimensions(q, axis: -2)).sum(axis: -1)

    if let mask {
        let expandedMask: MLXArray
        if mask.ndim == 1 {
            expandedMask = expandedDimensions(mask, axes: [1, 2, 3])
        } else if mask.ndim == 2 {
            expandedMask = expandedDimensions(mask, axes: [2, 3])
        } else if mask.ndim == 3 {
            expandedMask = expandedDimensions(mask, axis: -1)
        } else {
            fatalError("Unsupported mask shape \(mask.shape)")
        }
        state = MLX.where(expandedMask, state, oldState)
    }

    return (y, state)
}

func gatedDeltaOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let T = q.dim(1)
    let Hk = q.dim(2)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    var q = q
    var k = k

    let repeatFactor = Hv / Hk
    if repeatFactor > 1 {
        q = repeated(q, count: repeatFactor, axis: -2)
        k = repeated(k, count: repeatFactor, axis: -2)
    }

    var state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)

    var ys = [MLXArray]()
    ys.reserveCapacity(T)

    for t in 0 ..< T {
        let qT = q[0..., t]
        let kT = k[0..., t]
        let vT = v[0..., t]
        let gT = g[0..., t]
        let betaT = beta[0..., t]
        let maskT = mask == nil ? nil : mask![0..., t]

        let (y, newState) = gatedDeltaStepOps(
            q: qT,
            k: kT,
            v: vT,
            g: gT,
            beta: betaT,
            state: state,
            mask: maskT
        )
        ys.append(y)
        state = newState
    }

    let y = MLX.stacked(ys, axis: 1)
    return (y, state)
}

// MARK: - Public API

func gatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0)
    let Dk = q.dim(3)
    let Hv = v.dim(2)
    let Dv = v.dim(3)

    let state = state ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: q.dtype)

    // Fused kernel: computes g/beta inline, avoiding ~9 separate kernel dispatches
    if GatedDeltaKernelManager.shared.fusedKernel != nil {
        return gatedDeltaKernelFused(
            q: q, k: k, v: v, aRaw: a, bRaw: b,
            aLog: aLog, dtBias: dtBias, state: state, mask: mask)
    }

    // Fallback: compute g and beta separately, then use ops
    let beta = sigmoid(b)
    let g = computeGatedDeltaG(aLog, a, dtBias)
    return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
}
