// Copyright © 2026 Apple Inc.
//
// DFlash2Support.swift
// mlx-swift-lm
//
// Shared machinery for DFlash2 block-parallel speculative decoding
// (https://inco.ai/blog/dflash2/). The drafter model itself lives in the
// model modules (e.g. MLXLLM `DFlash2.swift`); this file carries everything
// the *iterator* needs to talk to a hybrid target model without naming it:
//
//  - ``DFlash2DrafterModel``: the protocol a block-parallel drafter exposes.
//  - ``DFlash2ContextCache``: the drafter's sliding-window context KV cache.
//  - ``GDNCaptureContext`` + ``rollbackSpeculativeHybridCaches``: exact
//    rollback of a hybrid attention/recurrent target after a verify pass that
//    committed only a prefix of its tokens. Attention rows trim; recurrent
//    (gated-delta) state is recomputed by replaying the captured layer inputs
//    over the accepted prefix from the pre-verify state — the same scheme as
//    the reference implementation's `_GDNStateCapture.rollback`.

import Foundation
import MLX

// MARK: - Cross-model LMOutput keys

/// Request: capture the residual-stream OUTPUTS of these target layer indices
/// during this forward pass (in addition to logits over the whole input).
/// Response is published under ``dflash2CapturedHiddenStatesKey`` in request
/// order.
public let dflash2CaptureLayerIdsKey =
    LMOutput.Key<[Int]>("dflash2.captureLayerIds")

/// Response: one `[B, S, hidden]` array per requested layer id, in the order
/// ``dflash2CaptureLayerIdsKey`` listed them.
public let dflash2CapturedHiddenStatesKey =
    LMOutput.Key<[MLXArray]>("dflash2.capturedHiddenStates")

/// Request: every gated-delta (linear-attention) layer records the inputs its
/// state update consumed, so a later rollback can replay the accepted prefix.
/// The same context object collects captures in layer order.
public let dflash2GDNCaptureContextKey =
    LMOutput.Key<GDNCaptureContext>("dflash2.gdnCaptureContext")

/// Request (stage 2): build the verify pass as an accept-invariant graph —
/// all accept-dependent scalars arrive as lazy arrays so the pass can be
/// constructed while the PREVIOUS round's verify is still in flight on the
/// GPU. See ``DFlash2PipelinedVerifyPlan``.
public let dflash2PipelinedVerifyPlanKey =
    LMOutput.Key<DFlash2PipelinedVerifyPlan>("dflash2.pipelinedVerifyPlan")

/// Accept-invariant verify construction (stage 2 of the pipelined round).
///
/// The pass's S rows land in each full-attention KV buffer at a LAZY start
/// offset (`anchor + validCount` of the unresolved previous round, computed
/// on-GPU), via a dynamic slice update; RoPE takes the same lazy scalar.
/// The SDPA over the grown cache is bounded host-side by `worstLen`
/// (`committed + 2S`, every outcome fits) and governed by `attnMask`, one
/// lazy comparison `col < start + row + 1` that encodes in-block causality,
/// history visibility, and the exclusion of stale rows past the true write
/// position in a single bool `[S, worstLen]` array.
public struct DFlash2PipelinedVerifyPlan {
    /// Position of the pass's first row — `[1]` int32, lazy.
    public let kvStart: MLXArray
    /// Host upper bound on visible KV rows (`committed + 2S`).
    public let worstLen: Int
    /// Bool `[S, worstLen]` visibility mask (see type doc), lazy.
    public let attnMask: MLXArray

    public init(kvStart: MLXArray, worstLen: Int, attnMask: MLXArray) {
        self.kvStart = kvStart
        self.worstLen = worstLen
        self.attnMask = attnMask
    }
}

extension KVCacheSimple {
    /// Stage-2 verify write: `keys`/`values` rows land at the lazy `start`
    /// offset (dynamic slice update — the write position resolves on the GPU
    /// with the previous round's accept). The host `offset` does NOT advance;
    /// the owner commits it after the accept resolves
    /// (``commitPipelined(rows:anchor:)``). Rejected rows need no cleanup:
    /// the next round's write starts at `anchor + accepted + 1` and overwrites
    /// them. Returns the K/V slices up to `worstLen` for the pass's SDPA.
    public func updatePipelined(
        keys newKeys: MLXArray, values newValues: MLXArray,
        start: MLXArray, worstLen: Int
    ) -> (MLXArray, MLXArray) {
        if self.keys == nil || worstLen > self.keys!.dim(2) {
            let B = newKeys.dim(0)
            let kvHeads = newKeys.dim(1)
            let kHeadDim = newKeys.dim(3)
            let vHeadDim = newValues.dim(3)
            let nSteps = (worstLen + step - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            if let currentKeys = self.keys, let currentValues = self.values {
                // Preserve the WHOLE buffer: staged rows beyond `offset` are
                // part of the in-flight lazy chain.
                let padK = MLXArray.zeros(
                    [B, kvHeads, nSteps * step - currentKeys.dim(2), kHeadDim],
                    dtype: newKeys.dtype)
                let padV = MLXArray.zeros(
                    [B, kvHeads, nSteps * step - currentValues.dim(2), vHeadDim],
                    dtype: newValues.dtype)
                self.keys = concatenated([currentKeys, padK], axis: 2)
                self.values = concatenated([currentValues, padV], axis: 2)
            } else {
                self.keys = MLXArray.zeros(kShape, dtype: newKeys.dtype)
                self.values = MLXArray.zeros(vShape, dtype: newValues.dtype)
            }
        }
        self.keys = dynamicSliceUpdated(
            self.keys!, update: newKeys, start: start, axes: [2])
        self.values = dynamicSliceUpdated(
            self.values!, update: newValues, start: start, axes: [2])
        return (
            self.keys![.ellipsis, ..<worstLen, 0...],
            self.values![.ellipsis, ..<worstLen, 0...]
        )
    }

    /// Resolve a pipelined round's bookkeeping once the accept is known:
    /// the committed row count becomes `anchor + rows`. Content is already
    /// correct (the buffer was written at the true offset on the GPU).
    public func commitPipelined(rows: Int, anchor: Int) {
        self.offset = anchor + rows
    }
}

// MARK: - GDN capture

/// Everything needed to restore one gated-delta layer's state to "as if only
/// the first `accepted + 1` positions of the captured pass had run".
public struct GDNCapture {
    /// Pre-convolution input: `concat([convState(K-1 rows), qkv(S rows)])`,
    /// after the SSM mask (if any) — `[B, K-1+S, convDim]`.
    public var convInput: MLXArray
    /// q/k (`[B, S, Hk, Dk]`) and v (`[B, S, Hv, Dv]`). q/k are post-norm,
    /// post-scale — unless ``qkNorm`` is set, in which case they are raw and
    /// the scan kernel norms them on load.
    public var q: MLXArray
    public var k: MLXArray
    public var v: MLXArray
    /// Raw gate projections (`[B, S, Hv]`).
    public var a: MLXArray
    public var b: MLXArray
    public var aLog: MLXArray
    public var dtBias: MLXArray
    /// Recurrent state BEFORE the captured pass.
    public var initialState: MLXArray?
    public var mask: MLXArray?
    /// Conv kernel size K (the conv state holds K-1 rows).
    public var convKernelSize: Int
    /// When true, `q`/`k` are RAW (pre-norm) and the replay must run the
    /// scan kernel with its fused q/k RMS norm, matching the capture pass.
    public var qkNorm: Bool

    public init(
        convInput: MLXArray, q: MLXArray, k: MLXArray, v: MLXArray,
        a: MLXArray, b: MLXArray, aLog: MLXArray, dtBias: MLXArray,
        initialState: MLXArray?, mask: MLXArray?, convKernelSize: Int,
        qkNorm: Bool = false
    ) {
        self.convInput = convInput
        self.q = q
        self.k = k
        self.v = v
        self.a = a
        self.b = b
        self.aLog = aLog
        self.dtBias = dtBias
        self.initialState = initialState
        self.mask = mask
        self.convKernelSize = convKernelSize
        self.qkNorm = qkNorm
    }
}

/// Collector for per-layer residual-stream captures, keyed by layer index.
/// Reference type so a model can fill it during the forward pass it is handed
/// to; read back in ``dflash2CaptureLayerIdsKey`` request order.
public final class DFlash2HiddenCaptureBox {
    public private(set) var values: [Int: MLXArray] = [:]

    public init() {}

    public func store(layer: Int, hidden: MLXArray) {
        values[layer] = hidden
    }

    public func value(for layer: Int) -> MLXArray? {
        values[layer]
    }
}

/// Ordered collector for per-layer ``GDNCapture`` values. Reference type so it
/// can cross the `LMOutput.State` boundary and be filled by the model during
/// the forward it is passed to. One context per verify pass; call ``clear()``
/// before each capture run.
public final class GDNCaptureContext {
    public private(set) var captures: [GDNCapture] = []

    public init() {}

    public func record(_ capture: GDNCapture) {
        captures.append(capture)
    }

    public func clear() {
        captures.removeAll(keepingCapacity: true)
    }
}

// MARK: - Hybrid rollback

/// Reconcile a hybrid attention/recurrent cache after a speculative verify
/// pass committed only a prefix of its input.
///
/// The verify pass processed `[anchor, draft_1, ..., draft_{S-1}]` and
/// `accepted + 1` of those positions stay committed (the anchor plus the
/// accepted draft prefix; the correction token is sampled from the logits but
/// never enters the cache). Trimmable attention caches drop their trailing
/// `rejected` rows. Each recurrent `MambaCache` — whose state cannot be
/// truncated — is restored by replaying its captured inputs over the accepted
/// prefix from the state captured before the pass, matching the reference
/// implementation's `_GDNStateCapture.rollback` exactly.
///
/// - Parameters:
///   - caches: the target model's cache array (attention + Mamba mixed).
///   - context: captures recorded during the verify pass, in gated-delta
///     layer order (one entry per non-trimmable cache).
///   - accepted: number of draft tokens accepted (0...S-1).
///   - rejected: number of trailing verify positions to drop
///     (`S - accepted - 1`).
public func rollbackSpeculativeHybridCaches(
    _ caches: [KVCache],
    context: GDNCaptureContext,
    accepted: Int,
    rejected: Int
) {
    precondition(rejected >= 0, "rollbackSpeculativeHybridCaches: negative rejection count")
    let committedPositions = accepted + 1
    var captureIndex = 0
    for cache in caches {
        if cache.isTrimmable {
            if rejected > 0 {
                let trimmed = cache.trim(rejected)
                precondition(
                    trimmed == rejected,
                    "attention cache trimmed \(trimmed), expected \(rejected)")
            }
            continue
        }
        guard let mambaCache = cache as? MambaCache else {
            preconditionFailure(
                "rollbackSpeculativeHybridCaches: non-trimmable, non-Mamba cache \(type(of: cache))"
            )
        }
        precondition(
            captureIndex < context.captures.count,
            "missing GDN capture \(captureIndex) (\(context.captures.count) recorded)")
        let capture = context.captures[captureIndex]
        captureIndex += 1

        let n = committedPositions
        let (_, state) = gatedDeltaUpdate(
            q: capture.q[0..., ..<n, 0..., 0...],
            k: capture.k[0..., ..<n, 0..., 0...],
            v: capture.v[0..., ..<n, 0..., 0...],
            a: capture.a[0..., ..<n, 0...],
            b: capture.b[0..., ..<n, 0...],
            aLog: capture.aLog,
            dtBias: capture.dtBias,
            state: capture.initialState,
            mask: capture.mask.map { $0[0..., ..<n] },
            qkNorm: capture.qkNorm)
        mambaCache[1] = state

        // Conv state after committing n positions = the last K-1 rows of the
        // conv input as of position n: rows [n, n + K - 1).
        let kSize = capture.convKernelSize
        mambaCache[0] = contiguous(
            capture.convInput[0..., n ..< (n + kSize - 1), 0...])
    }
    precondition(
        captureIndex == context.captures.count,
        "GDN capture count \(context.captures.count) exceeds non-trimmable cache count")
}

/// Compiled accept-invariant replay for one GDN capture (mask-free path).
/// One trace is shared by every GDN layer (uniform shapes), so the eager
/// elementwise pre/post chains around the fused scan kernel — sigmoid(b),
/// the decay-gate chain, the posMask compare, the conv gather indices —
/// fuse into a handful of launches instead of ~9 per layer. The replay runs
/// every pipelined round, so those launches are round-critical-path cost.
/// Inputs: q, k, v, a, b, aLog, dtBias, state, convInput, validCount.
/// Outputs: [newState, conv].
private func makeCompiledReplayCapture(qkNorm: Bool) -> @Sendable ([MLXArray]) -> [MLXArray] {
    compile { inputs in
        let (q, k, v) = (inputs[0], inputs[1], inputs[2])
        let (a, b, aLog, dtBias) = (inputs[3], inputs[4], inputs[5], inputs[6])
        let (state, convInput, validCount) = (inputs[7], inputs[8], inputs[9])
        let s = q.dim(1)
        let posMask = (MLXArray(Int32(0) ..< Int32(s)) .< validCount.asType(.int32))
            .expandedDimensions(axis: 0)  // [1, S]
        let (_, newState) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
            state: state, mask: posMask, qkNorm: qkNorm)
        let kSize = convInput.dim(1) - s + 1
        let indices = (validCount.asType(.int32) + MLXArray(Int32(0) ..< Int32(kSize - 1)))
            .reshaped([1, kSize - 1, 1])
        let conv = contiguous(takeAlong(convInput, indices, axis: 1))
        return [newState, conv]
    }
}

private let compiledReplayCapture = makeCompiledReplayCapture(qkNorm: false)
/// Twin trace with the fused q/k norm baked in — the flag selects a kernel,
/// so it cannot be a trace input.
private let compiledReplayCaptureNormed = makeCompiledReplayCapture(qkNorm: true)

/// Accept-invariant rollback build: replay each capture over ALL of its
/// positions with the steps beyond `validCount` masked out. The fused scan's
/// masked steps leave the state registers untouched (exact identity), so the
/// result equals the accepted-prefix replay for every accept outcome — which
/// makes the graph buildable while the verify pass (and the accept count) is
/// still in flight on the GPU. Returns lazy (recurrent state, conv state)
/// pairs in capture order; hand them to ``applySpeculativeRollback`` once the
/// accept resolves.
private let compiledReplayEnabled: Bool =
    ProcessInfo.processInfo.environment["DFLASH2_COMPILED_REPLAY"] != "0"

public func prebuildSpeculativeRollback(
    context: GDNCaptureContext,
    validCount: MLXArray
) -> [(state: MLXArray, conv: MLXArray)] {
    context.captures.map { capture in
        if compiledReplayEnabled, capture.mask == nil, let initialState = capture.initialState {
            let replay = capture.qkNorm ? compiledReplayCaptureNormed : compiledReplayCapture
            let out = replay([
                capture.q, capture.k, capture.v,
                capture.a, capture.b, capture.aLog, capture.dtBias,
                initialState, capture.convInput, validCount,
            ])
            return (out[0], out[1])
        }
        let s = capture.q.dim(1)
        let posMask = (MLXArray(Int32(0) ..< Int32(s)) .< validCount.asType(.int32))
            .expandedDimensions(axis: 0)  // [1, S]
        let mask = capture.mask.map { $0 & posMask } ?? posMask
        let (_, state) = gatedDeltaUpdate(
            q: capture.q,
            k: capture.k,
            v: capture.v,
            a: capture.a,
            b: capture.b,
            aLog: capture.aLog,
            dtBias: capture.dtBias,
            state: capture.initialState,
            mask: mask,
            qkNorm: capture.qkNorm)
        let kSize = capture.convKernelSize
        let indices = (validCount.asType(.int32) + MLXArray(Int32(0) ..< Int32(kSize - 1)))
            .reshaped([1, kSize - 1, 1])
        let conv = contiguous(takeAlong(capture.convInput, indices, axis: 1))
        return (state, conv)
    }
}

/// Assign a prebuilt rollback's GDN states WITHOUT touching the attention
/// caches (stage 2: pipelined rounds commit those by offset bookkeeping, and
/// the window assigns these same arrays before building the next verify).
public func applyGDNRollback(
    _ caches: [KVCache],
    prebuilt: [(state: MLXArray, conv: MLXArray)]
) {
    var index = 0
    for cache in caches where !cache.isTrimmable {
        guard let mambaCache = cache as? MambaCache else {
            preconditionFailure(
                "applyGDNRollback: non-trimmable, non-Mamba cache \(type(of: cache))")
        }
        precondition(
            index < prebuilt.count,
            "missing prebuilt rollback \(index) (\(prebuilt.count) built)")
        mambaCache[1] = prebuilt[index].state
        mambaCache[0] = prebuilt[index].conv
        index += 1
    }
    precondition(
        index == prebuilt.count,
        "prebuilt rollback count \(prebuilt.count) exceeds non-trimmable cache count")
}

/// Apply a prebuilt rollback: trimmable attention caches drop their trailing
/// `rejected` rows (host metadata), Mamba caches adopt the lazily-replayed
/// states. Call only when `rejected > 0` — an all-accepted round keeps the
/// verify pass's own committed states, exactly like the synchronous path.
public func applySpeculativeRollback(
    _ caches: [KVCache],
    prebuilt: [(state: MLXArray, conv: MLXArray)],
    rejected: Int
) {
    var index = 0
    for cache in caches {
        if cache.isTrimmable {
            if rejected > 0 {
                let trimmed = cache.trim(rejected)
                precondition(
                    trimmed == rejected,
                    "attention cache trimmed \(trimmed), expected \(rejected)")
            }
            continue
        }
        guard let mambaCache = cache as? MambaCache else {
            preconditionFailure(
                "applySpeculativeRollback: non-trimmable, non-Mamba cache \(type(of: cache))")
        }
        precondition(
            index < prebuilt.count,
            "missing prebuilt rollback \(index) (\(prebuilt.count) built)")
        mambaCache[1] = prebuilt[index].state
        mambaCache[0] = prebuilt[index].conv
        index += 1
    }
    precondition(
        index == prebuilt.count,
        "prebuilt rollback count \(prebuilt.count) exceeds non-trimmable cache count")
}

// MARK: - Draft context cache

/// Per-layer sliding-window cache for a DFlash2 drafter's CONTEXT keys/values
/// (projections of the target hidden states — the block's own K/V are
/// recomputed per round and never cached).
///
/// Padded-buffer variant of the reference port's
/// `RotatingKVCache(max_size: window - 1)`: the stored content is the
/// `storedCount`-row prefix of a slice-updated buffer, and the front-trim to
/// the window is LAZY (one compaction copy per ~256 appended rows instead of
/// a full-cache concat every round — per-cycle concat churns the Metal heap;
/// oMLX measured a progressive ~5x wall at ~7000 cycles). Attention-visible
/// behavior is identical: `DFlash2Attention.makeMask` windows by distance
/// (query - key < sliding_window), so rows kept past `maxSize` are masked
/// out exactly as if trimmed.
///  - ``append(keys:values:)`` slice-updates in place and returns the
///    logical prefix views.
///  - ``trimNewest(_:)`` rewinds the logical count (defensive reconcile
///    after a round) — no copies.
///
/// `offset` is the absolute position the next appended row will occupy; RoPE
/// is applied to keys BEFORE they enter the cache, so stored entries keep the
/// phase they were written with. Compaction does not touch `offset`.
public final class DFlash2ContextCache {
    /// Padded storage; the logical content is the `storedCount`-row prefix.
    private var keyStore: MLXArray?
    private var valueStore: MLXArray?
    private var storedCount = 0
    public var offset: Int = 0
    public let maxSize: Int
    /// Overflow tolerated past `maxSize` before the front-trim compaction
    /// fires (amortized: one copy per ~slack appended rows).
    private let compactSlack = 256

    /// Pipelined-propose bookkeeping (parallel to the stored rows): the RoPE
    /// position each row was written with, and whether the row is committed
    /// context or a masked-out placeholder (the pipelined path appends the
    /// full verify block before the accept count is known — see
    /// ``appendPipelined(keys:values:positions:)``).
    public private(set) var rowPositions: [Int32] = []
    public private(set) var rowValid: [Bool] = []

    public init(maxSize: Int) {
        self.maxSize = maxSize
    }

    /// Logical content views (prefix of the padded buffers).
    public var keys: MLXArray? { keyStore?[.ellipsis, ..<storedCount, 0...] }
    public var values: MLXArray? { valueStore?[.ellipsis, ..<storedCount, 0...] }

    public var count: Int { storedCount }

    @discardableResult
    public func append(keys newKeys: MLXArray, values newValues: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let n = newKeys.dim(2)
        if keyStore == nil {
            let cap = Swift.max(n, maxSize + compactSlack)
            keyStore = MLXArray.zeros(
                [newKeys.dim(0), newKeys.dim(1), cap, newKeys.dim(3)],
                dtype: newKeys.dtype)
            valueStore = MLXArray.zeros(
                [newValues.dim(0), newValues.dim(1), cap, newValues.dim(3)],
                dtype: newValues.dtype)
        }
        appendRows(keys: newKeys, values: newValues)
        rowPositions.append(contentsOf: (0 ..< n).map { Int32(offset + $0) })
        rowValid.append(contentsOf: Array(repeating: true, count: n))
        offset += n
        return (keys!, values!)
    }

    /// Pipelined-propose append: the rows carry explicit RoPE positions and
    /// enter as placeholders (`rowValid == false`) until the accept count
    /// resolves them via ``resolveNewestValidity(newestCount:validCount:)``.
    /// The write cursor `offset` is NOT advanced — pipelined rounds track
    /// positions explicitly, so `offset` stops being meaningful once this
    /// path engages.
    @discardableResult
    public func appendPipelined(
        keys newKeys: MLXArray, values newValues: MLXArray, positions: [Int32]
    ) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        if keyStore == nil {
            let cap = Swift.max(n, maxSize + compactSlack)
            keyStore = MLXArray.zeros(
                [newKeys.dim(0), newKeys.dim(1), cap, newKeys.dim(3)],
                dtype: newKeys.dtype)
            valueStore = MLXArray.zeros(
                [newValues.dim(0), newValues.dim(1), cap, newValues.dim(3)],
                dtype: newValues.dtype)
        }
        appendRows(keys: newKeys, values: newValues)
        rowPositions.append(contentsOf: positions)
        rowValid.append(contentsOf: Array(repeating: false, count: n))
        return (keys!, values!)
    }

    /// Shared buffer write for both append flavors: compaction (valid rows
    /// only survive — placeholders are masked anyway, so dropping them keeps
    /// the effective window at `maxSize` committed rows), then the in-place
    /// slice update.
    private func appendRows(keys newKeys: MLXArray, values newValues: MLXArray) {
        let n = newKeys.dim(2)
        var storeK = keyStore!
        var storeV = valueStore!
        if storedCount + n > storeK.dim(2) || storedCount > maxSize + compactSlack {
            let validIdx = (0 ..< storedCount).filter { rowValid[$0] }
            let keep = Swift.min(validIdx.count, Swift.max(maxSize, 0))
            let keptIdx = Array(validIdx.suffix(keep))
            let gather = MLXArray(keptIdx.map { Int32($0) })
            let keptK = MLX.take(storeK, gather, axis: 2)
            let keptV = MLX.take(storeV, gather, axis: 2)
            rowPositions = keptIdx.map { rowPositions[$0] }
            rowValid = Array(repeating: true, count: keep)
            let cap = Swift.max(maxSize + compactSlack, keep + n)
            let padShapeK = [storeK.dim(0), storeK.dim(1), cap - keep, storeK.dim(3)]
            let padShapeV = [storeV.dim(0), storeV.dim(1), cap - keep, storeV.dim(3)]
            storeK = concatenated(
                [keptK, MLXArray.zeros(padShapeK, dtype: storeK.dtype)], axis: 2)
            storeV = concatenated(
                [keptV, MLXArray.zeros(padShapeV, dtype: storeV.dtype)], axis: 2)
            storedCount = keep
            keyStore = storeK
            valueStore = storeV
        }
        storeK[.ellipsis, storedCount ..< (storedCount + n), 0...] = newKeys
        storeV[.ellipsis, storedCount ..< (storedCount + n), 0...] = newValues
        storedCount += n
    }

    /// Resolve the newest pipelined block's placeholders once the accept
    /// count is known: its first `validCount` rows become committed context.
    public func resolveNewestValidity(newestCount: Int, validCount: Int) {
        guard newestCount > 0, storedCount >= newestCount else { return }
        let base = storedCount - newestCount
        for i in 0 ..< newestCount {
            rowValid[base + i] = i < validCount
        }
    }

    /// A view-copy for speculative graph construction: appends mutate the
    /// clone's array objects only (subscript assignment rebinds the clone's
    /// handles, never the originals). Adopt with ``adopt(_:)`` when the
    /// prebuilt round is used; discard otherwise.
    public func stagingClone() -> DFlash2ContextCache {
        let clone = DFlash2ContextCache(maxSize: maxSize)
        clone.keyStore = keyStore.map { $0[0...] }
        clone.valueStore = valueStore.map { $0[0...] }
        clone.storedCount = storedCount
        clone.offset = offset
        clone.rowPositions = rowPositions
        clone.rowValid = rowValid
        return clone
    }

    public func adopt(_ clone: DFlash2ContextCache) {
        keyStore = clone.keyStore
        valueStore = clone.valueStore
        storedCount = clone.storedCount
        offset = clone.offset
        rowPositions = clone.rowPositions
        rowValid = clone.rowValid
    }

    public func trimNewest(_ n: Int) {
        guard n > 0, keyStore != nil else { return }
        let trimmed = Swift.min(n, storedCount)
        storedCount -= trimmed
        offset = Swift.max(0, offset - trimmed)
        rowPositions.removeLast(trimmed)
        rowValid.removeLast(trimmed)
    }
}

// MARK: - Pipelined proposal

/// A next-round greedy proposal built while the current round's verify is
/// still on the GPU: every accept-dependent value (anchor id, committed-row
/// count, block RoPE offset) rides as a lazy array, so the ONE prebuilt
/// graph is correct for any accept outcome. The staged caches carry the
/// appended verify-block rows; adopt them when the proposal is consumed.
public struct DFlash2PipelinedProposal {
    /// Selected draft ids [1, width − 1].
    public let tokens: MLXArray
    /// Selector top-K candidate ids [1, width − 1, K].
    public let candidates: MLXArray
    /// Per-layer staging clones holding the appended context rows.
    public let stagedCaches: [DFlash2ContextCache]
    /// Rows appended to each staged cache (the current round's verify rows).
    public let appendedRows: Int
    /// Block width the proposal was built for.
    public let width: Int

    public init(
        tokens: MLXArray, candidates: MLXArray,
        stagedCaches: [DFlash2ContextCache], appendedRows: Int, width: Int
    ) {
        self.tokens = tokens
        self.candidates = candidates
        self.stagedCaches = stagedCaches
        self.appendedRows = appendedRows
        self.width = width
    }
}

/// Drafters that can build an accept-invariant proposal from lazy inputs
/// (see ``DFlash2PipelinedProposal``). Optional capability — the iterator
/// falls back to the synchronous propose when absent or when the drafter
/// returns nil (escape hatches).
public protocol DFlash2PipelinedDrafter {
    func dflashProposePipelined(
        blockIds: MLXArray,
        targetHidden: MLXArray,
        validCount: MLXArray,
        contextPositionBase: Int,
        blockPositionOffset: MLXArray,
        caches: [DFlash2ContextCache]
    ) -> DFlash2PipelinedProposal?
}

// MARK: - Drafter protocol

/// A DFlash/DFlash2 block-parallel draft model as seen by
/// ``DFlash2SpeculativeTokenIterator``. Implemented by `DFlash2DraftModel` in
/// MLXLLM; the protocol keeps MLXLMCommon free of concrete-model imports.
public protocol DFlash2DrafterModel: BaseLanguageModel {
    /// Total tokens per speculation round (anchor + drafts); the checkpoint's
    /// training block size. The iterator may run smaller blocks.
    var dflashBlockSize: Int { get }
    /// Token id filling the block's non-anchor positions.
    var dflashMaskTokenId: Int { get }
    /// Target layer indices whose residual-stream outputs the drafter
    /// consumes (concatenated along the feature axis, in this order).
    var dflashTargetLayerIds: [Int] { get }
    /// Number of hidden rows the drafter's context window retains, when the
    /// drafter is sliding-only (`sliding_window - 1`); nil when the drafter
    /// attends over full context.
    var dflashContextKeepCount: Int? { get }
    /// Number of layers in the target (sanity-checked at bind time).
    var dflashNumTargetLayers: Int { get }

    /// Borrow the target's embedding table and LM head. Called once per
    /// stream before the first proposal.
    func bindDFlashTarget(_ target: any LanguageModel)

    /// Fresh per-stream context caches (one per drafter layer).
    func makeDFlashContextCaches() -> [DFlash2ContextCache]

    /// One-pass block proposal over `[anchor, MASK...]`:
    /// returns the selector's token path, the per-position top-K candidate
    /// ids, and (when `temperature > 0`) the per-position selection
    /// probabilities over those candidates for rejection sampling.
    func dflashPropose(
        _ inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [DFlash2ContextCache],
        temperature: Float,
        logitsStart: Int
    ) -> (tokens: MLXArray, candidates: MLXArray, probabilities: MLXArray?)

    /// Greedy proposal variant with an external path advisor: at each block
    /// position the advisor sees the tokens chosen so far in this block plus
    /// the position's candidate ids and may force one of them (a
    /// stream-history signal the drafter cannot see); nil keeps the
    /// selector's own choice. Advised proposals only change WHICH tokens are
    /// drafted — verification still gates every emitted token, so output
    /// equality with non-speculative decoding is unaffected. Implementations
    /// may ignore the advisor (the default forwards to `dflashPropose`).
    func dflashProposeAdvised(
        _ inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [DFlash2ContextCache],
        temperature: Float,
        logitsStart: Int,
        pathAdvisor: DFlash2PathAdvisor?
    ) -> (tokens: MLXArray, candidates: MLXArray, probabilities: MLXArray?)
}

/// Path advisor for `DFlash2DrafterModel.dflashProposeAdvised`: given the
/// tokens chosen so far in this block (empty at the first position — the
/// caller prepends its own committed history, anchor included) and a
/// position's candidate ids, return the candidate to force, or nil to keep
/// the selector's choice.
public typealias DFlash2PathAdvisor = (_ chosen: [Int], _ candidates: [Int]) -> Int?

extension DFlash2DrafterModel {
    public func dflashProposeAdvised(
        _ inputs: MLXArray,
        targetHidden: MLXArray,
        cache: [DFlash2ContextCache],
        temperature: Float,
        logitsStart: Int,
        pathAdvisor: DFlash2PathAdvisor?
    ) -> (tokens: MLXArray, candidates: MLXArray, probabilities: MLXArray?) {
        dflashPropose(
            inputs, targetHidden: targetHidden, cache: cache,
            temperature: temperature, logitsStart: logitsStart)
    }
}

// MARK: - Stream n-gram index

/// Longest-suffix n-gram continuation index over the decoded stream (prompt
/// plus committed tokens). The DFlash2 selector consults it while tracing a
/// path through the drafted candidate lattice: agent-typical content (code
/// edits, structured output, summaries quoting their source) is self-similar
/// enough that the stream's own continuations resolve most of the near-ties
/// the learned selector gets wrong. Purely a drafting signal — never touches
/// verification.
///
/// Orders 2...5 (context lengths 1...4). Contexts pack into a UInt64 key
/// (18 bits per token — vocab < 262144 — plus a length tag), one counter
/// dictionary per order.
public final class DFlash2NGramIndex {
    public static let maxOrder = 5
    private static let tokenBits: UInt64 = 18
    private static let tokenMask: UInt64 = (1 << tokenBits) - 1

    /// tables[o] covers order o+2 (context length o+1).
    private var tables: [[UInt64: [Int32: Int32]]] = Array(
        repeating: [:], count: maxOrder - 1)
    /// Trailing context of the stream (last maxOrder-1 tokens).
    private var tail: [Int] = []

    public init() {}

    private static func key(_ context: ArraySlice<Int>) -> UInt64 {
        var k: UInt64 = 1  // length tag / non-zero sentinel
        for token in context {
            k = (k << tokenBits) | (UInt64(token) & tokenMask)
        }
        return k
    }

    /// Append committed tokens, updating every order's counters.
    public func extend(_ tokens: [Int]) {
        for token in tokens {
            let n = tail.count
            for order in 2 ... Self.maxOrder where n >= order - 1 {
                let context = tail[(n - (order - 1)) ..< n]
                tables[order - 2][Self.key(context), default: [:]][
                    Int32(token), default: 0] += 1
            }
            tail.append(token)
            if tail.count > Self.maxOrder - 1 {
                tail.removeFirst(tail.count - (Self.maxOrder - 1))
            }
        }
    }

    /// Longest-suffix continuation of `context`: the most frequent next
    /// token at the longest matching order, with that order and the token's
    /// share of the context's continuations.
    public func predict(context: [Int]) -> (token: Int, order: Int, share: Double)? {
        for order in stride(from: Self.maxOrder, through: 2, by: -1) {
            guard context.count >= order - 1 else { continue }
            let ctx = context[(context.count - (order - 1))...]
            guard let counts = tables[order - 2][Self.key(ctx)], !counts.isEmpty
            else { continue }
            var bestToken: Int32 = 0
            var bestCount: Int32 = 0
            var total: Int32 = 0
            for (token, count) in counts {
                total += count
                if count > bestCount {
                    bestCount = count
                    bestToken = token
                }
            }
            return (Int(bestToken), order, Double(bestCount) / Double(total))
        }
        return nil
    }
}
