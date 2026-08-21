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

// MARK: - GDN capture

/// Everything needed to restore one gated-delta layer's state to "as if only
/// the first `accepted + 1` positions of the captured pass had run".
public struct GDNCapture {
    /// Pre-convolution input: `concat([convState(K-1 rows), qkv(S rows)])`,
    /// after the SSM mask (if any) — `[B, K-1+S, convDim]`.
    public var convInput: MLXArray
    /// Post-norm, post-scale q/k (`[B, S, Hk, Dk]`) and v (`[B, S, Hv, Dv]`).
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

    public init(
        convInput: MLXArray, q: MLXArray, k: MLXArray, v: MLXArray,
        a: MLXArray, b: MLXArray, aLog: MLXArray, dtBias: MLXArray,
        initialState: MLXArray?, mask: MLXArray?, convKernelSize: Int
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
                "rollbackSpeculativeHybridCaches: non-trimmable, non-Mamba cache \(type(of: cache))")
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
            mask: capture.mask.map { $0[0..., ..<n] })
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
        var storeK = keyStore!
        var storeV = valueStore!
        if storedCount + n > storeK.dim(2) || storedCount > maxSize + compactSlack {
            // Compaction (or first-round growth): keep the newest `maxSize`
            // rows in a fresh buffer — never an overlapping in-place shift.
            let keep = Swift.min(storedCount, Swift.max(maxSize, 0))
            let keptK = storeK[.ellipsis, (storedCount - keep)..<storedCount, 0...]
            let keptV = storeV[.ellipsis, (storedCount - keep)..<storedCount, 0...]
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
        storeK[.ellipsis, storedCount..<(storedCount + n), 0...] = newKeys
        storeV[.ellipsis, storedCount..<(storedCount + n), 0...] = newValues
        storedCount += n
        offset += n
        return (keys!, values!)
    }

    public func trimNewest(_ n: Int) {
        guard n > 0, keyStore != nil else { return }
        let trimmed = Swift.min(n, storedCount)
        storedCount -= trimmed
        offset = Swift.max(0, offset - trimmed)
    }
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
}
