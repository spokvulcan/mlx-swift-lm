// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// DFlash2 (https://inco.ai/blog/dflash2/, reference implementation
// https://github.com/z-lab/dflash) drafts a whole block of tokens in one
// parallel pass from the target model's hidden states, then the target
// verifies the block in one forward pass. These protocols are what the
// ``DFlash2SpeculativeTokenIterator`` needs from the two models; the concrete
// drafter (`DFlash2DraftModel`) and target (`Qwen35TextModel`) live in MLXLLM.

// MARK: - Drafter

/// One block proposal: the selector's token path and the per-position
/// candidates it chose from.
public struct DFlash2Proposal {
    /// Drafted token ids, `[1, blockSize - 1]`.
    public var tokens: MLXArray
    /// Top-K candidate ids per drafted position, `[1, blockSize - 1, K]`.
    public var candidates: MLXArray
    /// Selection probability over `candidates` per position, present only
    /// when the proposal was sampled (`temperature > 0`).
    public var probabilities: MLXArray?

    public init(tokens: MLXArray, candidates: MLXArray, probabilities: MLXArray? = nil) {
        self.tokens = tokens
        self.candidates = candidates
        self.probabilities = probabilities
    }
}

/// A drafter's whole candidate lattice for one block: what the selector
/// chooses its path from, for analysis and tree search.
public struct DFlash2Lattice {
    /// Top-K candidate ids per drafted position, `[1, L, K]`.
    public var candidates: MLXArray
    /// Each candidate's draft logit, `[1, L, K]`.
    public var unary: MLXArray
    /// Score of following candidate `i` at position `t` with candidate `j`
    /// at `t + 1`, `[1, L - 1, K, K]`.
    public var edges: MLXArray
    /// Score of following the anchor with each candidate at position 0, `[1, K]`.
    public var anchorEdges: MLXArray
    /// The greedy path, as ``DFlash2Proposal/tokens``, `[1, L]`.
    public var tokens: MLXArray

    public init(
        candidates: MLXArray, unary: MLXArray, edges: MLXArray, anchorEdges: MLXArray,
        tokens: MLXArray
    ) {
        self.candidates = candidates
        self.unary = unary
        self.edges = edges
        self.anchorEdges = anchorEdges
        self.tokens = tokens
    }
}

/// A block drafted as a tree: the selector's greedy chain plus leaf rows,
/// alternatives that branch off it where the chain is least sure. Every
/// array may be lazy and has the block's static width `S`.
public struct DFlash2TreeProposal {
    /// `[1, S]` token ids in topological order; row 0 is the anchor.
    public var tokens: MLXArray
    /// What the verify pass needs to run the block as a tree.
    public var layout: DFlash2TreeLayout
    /// `[S]` int32: each row's parent row (0 for the anchor itself).
    public var parents: MLXArray
    /// `[]` int32: the chain's length, anchor excluded.
    public var chainLength: MLXArray

    public init(
        tokens: MLXArray, layout: DFlash2TreeLayout, parents: MLXArray, chainLength: MLXArray
    ) {
        self.tokens = tokens
        self.layout = layout
        self.parents = parents
        self.chainLength = chainLength
    }
}

/// Per-stream drafter state, owned by the iterator and passed to the drafter
/// on every proposal. Drafter instances hold no per-stream state, so one
/// drafter serves many iterators.
public struct DFlash2DrafterState {
    /// One sliding-window context cache per drafter layer.
    public var contextCaches: [DFlash2ContextCache]

    public init(contextCaches: [DFlash2ContextCache]) {
        self.contextCaches = contextCaches
    }
}

/// A DFlash2 block-parallel drafter.
///
/// The drafter attends over projections of the target's hidden states (its
/// context) and predicts every position of a `[anchor, MASK, ...]` block at
/// once. It borrows the target's embedding table and LM head per call and is
/// stateless with respect to the target, like ``MTPDrafterModel``.
public protocol DFlash2DrafterModel: BaseLanguageModel {
    /// Tokens per verify pass the checkpoint was trained for (anchor included).
    var blockSize: Int { get }
    /// Token id filling the block's non-anchor positions.
    var maskTokenId: Int { get }
    /// Target layers whose outputs the drafter reads, in feature-concat order.
    var targetLayerIds: [Int] { get }
    /// Depth of the target the drafter was distilled for.
    var targetLayerCount: Int { get }
    /// Target hidden rows the drafter's context window retains.
    var contextWindow: Int { get }

    /// Fresh per-stream state.
    func makeState() -> DFlash2DrafterState

    /// Propose one block.
    ///
    /// Every accept-dependent input may be a lazy array, so the proposal can
    /// be built while the previous verify pass is still running on the GPU.
    ///
    /// - Parameters:
    ///   - block: `[1, blockSize]` token ids; position 0 is the anchor.
    ///   - targetHidden: `[1, S, targetLayerIds.count * hidden]` target
    ///     outputs for the `S` positions verified since the last proposal.
    ///   - contextPosition: absolute position of `targetHidden` row 0.
    ///   - validRows: `[]` int32 count of leading `targetHidden` rows that
    ///     are committed context; the rest are masked out.
    ///   - temperature: 0 selects greedily; above 0 samples the path and
    ///     fills ``DFlash2Proposal/probabilities``.
    ///   - target: the model whose embedding and head the drafter borrows.
    ///   - state: the stream's context caches; `targetHidden` rows are
    ///     appended as placeholders until `resolve(newest:valid:)`.
    func propose(
        block: MLXArray,
        targetHidden: MLXArray,
        contextPosition: Int,
        validRows: MLXArray,
        temperature: Float,
        target: any DFlash2TargetModel,
        state: inout DFlash2DrafterState
    ) -> DFlash2Proposal

    /// Tokens the iterator just committed (accepted drafts and the bonus),
    /// so a drafter that predicts over a vocabulary prefix can widen when
    /// the target leaves it. Default: ignored.
    func observeCommitted(_ tokens: [Int])

    /// Propose one greedy block as a tree of `block.dim(1)` rows; nil when
    /// the drafter only drafts chains. Same inputs and state handling as
    /// ``propose(block:targetHidden:contextPosition:validRows:temperature:target:state:)``.
    func proposeTree(
        block: MLXArray,
        targetHidden: MLXArray,
        contextPosition: Int,
        validRows: MLXArray,
        target: any DFlash2TargetModel,
        state: inout DFlash2DrafterState
    ) -> DFlash2TreeProposal?
}

extension DFlash2DrafterModel {
    public func observeCommitted(_ tokens: [Int]) {}

    public func proposeTree(
        block: MLXArray, targetHidden: MLXArray, contextPosition: Int, validRows: MLXArray,
        target: any DFlash2TargetModel, state: inout DFlash2DrafterState
    ) -> DFlash2TreeProposal? { nil }
}

// MARK: - Target

/// One verify pass over `[anchor, draft_1, ..., draft_{S-1}]`.
/// A verify block whose rows form a tree: a chain from the anchor plus leaf
/// rows that branch off it, in topological order (every row after its
/// parent). Each row attends to its ancestors, rotates at its depth, and
/// convolves and scans along its own ancestry. Every row computes exactly
/// what a chain block of its own path computes: a leaf's attention reads
/// its own key at its depth (``dflash2TreeAttention(queries:keys:values:position:depths:slots:visibleLength:scale:)``).
public struct DFlash2TreeLayout {
    /// `[S]` int32 depth of each row (0 for the anchor): its position past
    /// the block's.
    public var depths: MLXArray
    /// `[S, S]` bool: row `i` sees block slot `j` (it holds an ancestor, or
    /// the row itself). The block's keys and values are cached in slot
    /// order, see ``slotRows``.
    public var ancestry: MLXArray
    /// `[1, S]` int32: 1 where the row continues the recurrent state (the
    /// anchor and the chain), 0 for a leaf.
    public var commits: MLXArray
    /// `[S]` int32: the row holding the chain at each depth (`chainRows[0]`
    /// is the anchor's row, 0); depths past the chain repeat its last row.
    /// A row at depth `d` descends from `chainRows[0 ..< d]`.
    public var chainRows: MLXArray
    /// `[S]` int32: the row whose keys and values go in each cache slot of
    /// the block: the anchor and the chain by depth first, exactly where a
    /// chain block puts them, then the leaves. A chain row's attention then
    /// reads the same keys in the same slots as in a chain block, so it
    /// reduces in the same order; a leaf's reads its own key at its depth.
    public var slotRows: MLXArray
    /// `[S]` int32: each row's slot (the inverse of ``slotRows``).
    public var slots: MLXArray

    public init(
        depths: MLXArray, ancestry: MLXArray, commits: MLXArray, chainRows: MLXArray,
        slotRows: MLXArray, slots: MLXArray
    ) {
        self.depths = depths
        self.ancestry = ancestry
        self.commits = commits
        self.chainRows = chainRows
        self.slotRows = slotRows
        self.slots = slots
    }

    /// `[1, S, K]` int32 conv windows for a depthwise conv of width `K` over
    /// the virtual input `K - 1` state rows then the block's rows (see
    /// ``gatedDeltaConvNormQKV(convState:rows:rowOffset:weight:numKHeads:numVHeads:headDim:scales:eps:windows:)``):
    /// each row's `K - 1` ancestors along the chain, or state rows above the
    /// anchor, then the row itself. A chain gets `s ..< s + K`.
    public func convWindows(kernel K: Int) -> MLXArray {
        let S = depths.dim(0)
        let depth = depths.asType(.int32).reshaped([S, 1])
        // Path positions depth - (K - 1) ..< depth: a state row above the
        // anchor (negative), else the chain row at that depth.
        let pathPosition = depth + MLXArray(Int32(-(K - 1)) ..< Int32(0)).reshaped([1, K - 1])
        let chainRow = take(chainRows.asType(.int32), maximum(pathPosition, 0).reshaped([-1]))
            .reshaped([S, K - 1])
        let ancestors = MLX.where(
            pathPosition .< 0, pathPosition + Int32(K - 1), chainRow + Int32(K - 1))
        let own = MLXArray(Int32(K - 1) ..< Int32(K - 1 + S)).reshaped([S, 1])
        return concatenated([ancestors, own], axis: 1).reshaped([1, S, K])
    }
}

public struct DFlash2VerifyRequest {
    /// `[1, S]` token ids.
    public var tokens: MLXArray
    /// `[1]` int32 absolute position of `tokens[0]`; may be lazy.
    public var position: MLXArray
    /// Largest value `position` can resolve to. Bounds the attention span.
    public var positionUpperBound: Int
    /// Layers whose outputs the drafter needs, in ``DFlash2VerifyResult/hidden`` order.
    public var captureLayers: [Int]
    /// Rotary offset of text past the prompt's images: cache row `p` rotates
    /// at `p + positionDelta`. Zero for a text-only prompt.
    public var positionDelta: Int
    /// The block's tree shape; nil for a chain (row `i` at depth `i`).
    public var tree: DFlash2TreeLayout?

    public init(
        tokens: MLXArray, position: MLXArray, positionUpperBound: Int, captureLayers: [Int],
        positionDelta: Int = 0, tree: DFlash2TreeLayout? = nil
    ) {
        self.tokens = tokens
        self.position = position
        self.positionUpperBound = positionUpperBound
        self.captureLayers = captureLayers
        self.positionDelta = positionDelta
        self.tree = tree
    }
}

public struct DFlash2VerifyResult {
    /// `[1, S, vocab]`; row `i` predicts input position `i + 1`.
    public var logits: MLXArray
    /// `[1, S, hidden]` per requested capture layer.
    public var hidden: [MLXArray]
    /// One capture per gated-delta layer, in cache order, so the iterator can
    /// rewind recurrent state to any accepted prefix.
    public var recurrentCaptures: [GatedDeltaCapture]

    public init(logits: MLXArray, hidden: [MLXArray], recurrentCaptures: [GatedDeltaCapture]) {
        self.logits = logits
        self.hidden = hidden
        self.recurrentCaptures = recurrentCaptures
    }
}

/// A hybrid attention/recurrent target that a DFlash2 drafter can speculate for.
///
/// A verify pass computes; it never commits. Attention rows are written into
/// the cache buffers at the request's (possibly lazy) position without moving
/// any offset, and recurrent state is returned as captures. The iterator
/// commits once the accept count is known.
public protocol DFlash2TargetModel: LanguageModel {
    /// Decoder layer count, checked against the drafter's target geometry.
    var dflash2LayerCount: Int { get }
    /// Embedding table the drafter borrows for the block.
    var dflash2Embedding: Embedding { get }
    /// LM head the drafter borrows; nil when tied to the embedding.
    var dflash2Head: Linear? { get }

    /// Whether the verify pass can drive this cache: `DFlash2AttentionCache`
    /// attention entries and `MambaCache` recurrent entries.
    func dflash2SupportsCache(_ cache: [KVCache]) -> Bool

    /// Ordinary prefill over `tokens` that also returns the outputs of
    /// `captureLayers` (`[1, S, hidden]` each, in request order). Cache row
    /// `p` rotates at `p + positionDelta` (see
    /// ``DFlash2VerifyRequest/positionDelta``).
    func dflash2Prefill(
        _ tokens: MLXArray, cache: [KVCache], captureLayers: [Int], positionDelta: Int
    ) -> (logits: MLXArray, hidden: [MLXArray])

    /// The verify pass. See ``DFlash2VerifyRequest`` and ``DFlash2VerifyResult``.
    func dflash2Verify(_ request: DFlash2VerifyRequest, cache: [KVCache]) -> DFlash2VerifyResult

    /// Whether `dflash2Verify` can run a ``DFlash2TreeLayout`` block over
    /// `cache`. Default: false.
    func dflash2SupportsTree(_ cache: [KVCache]) -> Bool
}

extension DFlash2TargetModel {
    public func dflash2SupportsTree(_ cache: [KVCache]) -> Bool { false }
}

/// A DFlash2 target that takes prompts with images: it prefills a prompt
/// through its last image itself, and the iterator speculates over the text
/// after it.
public protocol DFlash2MediaTargetModel: DFlash2TargetModel {
    /// Prefill `input` into the empty `cache` through its last image or video
    /// row. Returns the prompt tokens that covers and the rope delta of the
    /// text after them (``DFlash2VerifyRequest/positionDelta``), or nil,
    /// before touching the cache, when the prompt has no such row or no text
    /// after it.
    func dflash2PrefillMedia(
        _ input: LMInput, cache: [KVCache], prefill: PrefillParameters
    ) throws -> (prefilledTokens: Int, positionDelta: Int)?
}

// MARK: - Recurrent capture

/// What one gated-delta layer consumed during a verify pass, enough to
/// recompute its state as if only a prefix of the pass had run.
public struct GatedDeltaCapture {
    /// `concat([convState, qkv])`: the conv input, `[1, K - 1 + S, convDim]`.
    public var convInput: MLXArray
    /// Post-norm, post-scale `k` (`[1, S, Hk, Dk]`) and `v` (`[1, S, Hv, Dv]`).
    public var k: MLXArray
    public var v: MLXArray
    /// The scan's gates: precomputed `[1, S, Hv]` f32 `g`/`beta`, or the
    /// pre-activation source the kernel reads them from (see ``GatedDeltaGates``).
    public var gates: GatedDeltaGates
    /// Recurrent state before the pass.
    public var initialState: MLXArray

    public init(
        convInput: MLXArray, k: MLXArray, v: MLXArray, gates: GatedDeltaGates,
        initialState: MLXArray
    ) {
        self.convInput = convInput
        self.k = k
        self.v = v
        self.gates = gates
        self.initialState = initialState
    }

    public init(
        convInput: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray,
        initialState: MLXArray
    ) {
        self.init(
            convInput: convInput, k: k, v: v, gates: .precomputed(g: g, beta: beta),
            initialState: initialState)
    }

    /// The capture's per-pass arrays, in ``init(arrays:initialState:)``
    /// order, so a compiled verify body can return them as outputs: the
    /// conv input, `k`, `v`, then `g`/`beta` or the two gate source arrays.
    package var arrays: [MLXArray] {
        switch gates {
        case .precomputed(let g, let beta): [convInput, k, v, g, beta]
        case .source(let source): [convInput, k, v, source.aSource, source.bSource]
        }
    }
    package static let arrayCount = 5

    /// The layer-side half of a gate source: what `init(arrays:initialState:layout:)`
    /// adds back to the two source arrays a trace returned.
    public struct GateLayout {
        public var aOffset: Int
        public var bOffset: Int
        public var aLog: MLXArray
        public var dtBias: MLXArray

        public init(aOffset: Int, bOffset: Int, aLog: MLXArray, dtBias: MLXArray) {
            self.aOffset = aOffset
            self.bOffset = bOffset
            self.aLog = aLog
            self.dtBias = dtBias
        }
    }

    package init(arrays: [MLXArray], initialState: MLXArray, layout: GateLayout? = nil) {
        precondition(arrays.count == Self.arrayCount, "convInput, k, v, gates")
        let gates: GatedDeltaGates =
            if let layout {
                .source(
                    GatedDeltaGateSource(
                        aSource: arrays[3], aOffset: layout.aOffset, bSource: arrays[4],
                        bOffset: layout.bOffset, aLog: layout.aLog, dtBias: layout.dtBias))
            } else {
                .precomputed(g: arrays[3], beta: arrays[4])
            }
        self.init(
            convInput: arrays[0], k: arrays[1], v: arrays[2], gates: gates,
            initialState: initialState)
    }

    /// The capture with its block rows taken in `rows` order (`[S]` int32,
    /// possibly lazy): a tree block's accepted path as a chain, which
    /// ``replay(validCount:)`` then replays like any other.
    public func gathering(rows: MLXArray) -> GatedDeltaCapture {
        let stateRows = convInput.dim(1) - k.dim(1)
        let gatedGates: GatedDeltaGates
        switch gates {
        case .precomputed(let g, let beta):
            gatedGates = .precomputed(g: take(g, rows, axis: 1), beta: take(beta, rows, axis: 1))
        case .source(var source):
            source.aSource = take(source.aSource, rows, axis: 1)
            source.bSource = take(source.bSource, rows, axis: 1)
            gatedGates = .source(source)
        }
        return GatedDeltaCapture(
            convInput: concatenated(
                [
                    convInput[0..., ..<stateRows],
                    take(convInput[0..., stateRows...], rows, axis: 1),
                ], axis: 1),
            k: take(k, rows, axis: 1), v: take(v, rows, axis: 1), gates: gatedGates,
            initialState: initialState)
    }

    /// The layer's state after the first `validCount` positions of the pass.
    ///
    /// Replays every position with the steps past `validCount` skipped.
    /// A skipped step leaves the scan state untouched, so the result equals
    /// the accepted-prefix replay for every count, and `validCount` may be a
    /// lazy `[]` int32 array. The conv state is the `K - 1` rows of the conv
    /// input from `validCount` on, copied by the same launch. Returns the
    /// recurrent and conv states.
    public func replay(validCount: MLXArray) -> (recurrent: MLXArray, conv: MLXArray) {
        let replayed = gatedDeltaStateAfter(
            validCount: validCount, k: k, v: v, gates: gates, state: initialState,
            convInput: convInput)
        return (replayed.state, replayed.conv)
    }
}

// MARK: - Attention cache rows

/// An attention cache the verify pass drives. A pass writes its `S` rows at
/// `position` (a `[1]` int32, possibly lazy) without moving `offset` and
/// attends over the first `visibleLength` rows, row `i` seeing columns up to
/// `position + i`. `mask` is that position mask (`[S, visibleLength]` bool);
/// a conformer may rebuild it from `position` instead. A tree block (`tree`,
/// nil for a chain) arrives in slot order; its mask lets each row see the
/// rows before `position`, its ancestors and itself. The iterator then
/// commits the accepted prefix with ``commitRows(count:)``.
package protocol DFlash2AttentionCache: KVCache {
    func dflash2Attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, position: MLXArray,
        visibleLength: Int, mask: MLXArray, tree: DFlash2TreeLayout?, scale: Float
    ) -> MLXArray

    func commitRows(count: Int)

    /// Reorder a tree block's rows before the commit: block row `k` (cache
    /// row `position + k`) takes the content of block row `rows[k]` for
    /// every `k < width`, so the accepted path lies contiguous from
    /// `position`. `rows` (`[width]` int32) may be lazy.
    func gatherRows(position: Int, rows: MLXArray, width: Int)
}

extension DFlash2AttentionCache {
    /// A chain block's attention.
    package func dflash2Attention(
        queries: MLXArray, keys: MLXArray, values: MLXArray, position: MLXArray,
        visibleLength: Int, mask: MLXArray, scale: Float
    ) -> MLXArray {
        dflash2Attention(
            queries: queries, keys: keys, values: values, position: position,
            visibleLength: visibleLength, mask: mask, tree: nil, scale: scale)
    }
}

extension KVCacheSimple: DFlash2AttentionCache {
    package func dflash2Attention(
        queries: MLXArray, keys newKeys: MLXArray, values newValues: MLXArray,
        position: MLXArray, visibleLength: Int, mask: MLXArray, tree: DFlash2TreeLayout?,
        scale: Float
    ) -> MLXArray {
        let (keys, values) = writeRows(
            keys: newKeys, values: newValues, position: position, visibleLength: visibleLength)
        // A tree row reads its own key where a chain block holds its token,
        // so it reduces exactly as that chain block would.
        if let tree, let bufferKeys = self.keys, let bufferValues = self.values,
            let attention = dflash2TreeAttention(
                queries: queries, keys: bufferKeys, values: bufferValues, position: position,
                depths: tree.depths, slots: tree.slots, visibleLength: visibleLength,
                scale: scale)
        {
            return attention
        }
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: .array(mask))
    }
}

extension TurboQuantKVCache: DFlash2AttentionCache {
    /// The compressed verify attends by position (and a tree's slot masks),
    /// not by `mask`.
    package func dflash2Attention(
        queries: MLXArray, keys newKeys: MLXArray, values newValues: MLXArray,
        position: MLXArray, visibleLength: Int, mask: MLXArray, tree: DFlash2TreeLayout?,
        scale: Float
    ) -> MLXArray {
        // Each tree row's visible block slots as bits.
        let treeAncestry = tree.map { tree in
            let S = tree.ancestry.dim(1)
            return
                (tree.ancestry.asType(.uint32)
                * (MLXArray(UInt32(1)) << MLXArray(Int32(0) ..< Int32(S)).asType(.uint32)))
                .sum(axis: 1)
        }
        return verifyAttention(
            queries: queries, keys: newKeys, values: newValues, position: position,
            visibleLength: visibleLength, scale: scale, treeAncestry: treeAncestry)
    }
}

extension KVCacheSimple {
    /// Write `S` rows at `position` (a `[1]` int32 array, possibly lazy)
    /// without moving `offset`, and return the first `visibleLength` rows for
    /// the pass's attention. Rows past the committed offset are scratch: a
    /// later write at a smaller position simply overwrites them.
    package func writeRows(
        keys newKeys: MLXArray, values newValues: MLXArray,
        position: MLXArray, visibleLength: Int
    ) -> (MLXArray, MLXArray) {
        if keys == nil || visibleLength > keys!.dim(2) {
            let capacity = (visibleLength + step - 1) / step * step
            let kShape = [newKeys.dim(0), newKeys.dim(1), capacity, newKeys.dim(3)]
            let vShape = [newValues.dim(0), newValues.dim(1), capacity, newValues.dim(3)]
            let grownKeys = MLXArray.zeros(kShape, dtype: newKeys.dtype)
            let grownValues = MLXArray.zeros(vShape, dtype: newValues.dtype)
            if let keys, let values {
                // Keep the whole buffer: scratch rows may belong to a pass in flight.
                self.keys = concatenated(
                    [keys, grownKeys[.ellipsis, keys.dim(2)..., 0...]], axis: 2)
                self.values = concatenated(
                    [values, grownValues[.ellipsis, values.dim(2)..., 0...]], axis: 2)
            } else {
                keys = grownKeys
                values = grownValues
            }
        }
        // A dynamic slice update, not a scatter: the rows are contiguous, and
        // the mlx fork can write them in place (MLX_DYNSLICE_INPLACE=1)
        // instead of copying the whole store per pass.
        let start = position.asType(.int32).reshaped([1])
        keys = dynamicSliceUpdated(keys!, update: newKeys, start: start, axes: [2])
        values = dynamicSliceUpdated(values!, update: newValues, start: start, axes: [2])
        return (
            keys![.ellipsis, ..<visibleLength, 0...],
            values![.ellipsis, ..<visibleLength, 0...]
        )
    }

    /// Commit rows written by ``writeRows(keys:values:position:visibleLength:)``:
    /// the cache now holds `count` positions.
    package func commitRows(count: Int) {
        offset = count
    }

    package func gatherRows(position: Int, rows: MLXArray, width: Int) {
        guard let keys, let values else { return }
        let start = MLXArray([Int32(position)])
        let block = position ..< (position + width)
        self.keys = dynamicSliceUpdated(
            keys, update: take(keys[.ellipsis, block, 0...], rows, axis: 2), start: start,
            axes: [2])
        self.values = dynamicSliceUpdated(
            values, update: take(values[.ellipsis, block, 0...], rows, axis: 2), start: start,
            axes: [2])
    }
}

// MARK: - Drafter context cache

/// Sliding-window cache of one drafter layer's context keys and values.
///
/// Rows enter as placeholders with explicit positions, because a pipelined
/// round appends the whole verify block before the accept count is known;
/// `resolve(newest:valid:)` commits the accepted prefix once it is. The
/// window is enforced lazily: rows are stored in a padded buffer and
/// compacted to the newest valid `window` rows once per few hundred appends,
/// while the attention mask windows by distance and hides placeholders, so
/// what attention sees is exactly a trimmed cache.
public final class DFlash2ContextCache {
    /// Rows the window retains.
    public let window: Int

    private var keyStore: MLXArray?
    private var valueStore: MLXArray?
    private var storedCount = 0
    private let compactionSlack = 256

    /// Absolute position each stored row was written with.
    package private(set) var rowPositions: [Int32] = []
    /// Whether each stored row is committed context or a placeholder.
    package private(set) var rowValid: [Bool] = []

    public init(window: Int) {
        precondition(window > 0, "DFlash2ContextCache needs a positive window")
        self.window = window
    }

    /// Stored rows, placeholders included.
    public var count: Int { storedCount }

    package var keys: MLXArray? { keyStore?[.ellipsis, ..<storedCount, 0...] }
    package var values: MLXArray? { valueStore?[.ellipsis, ..<storedCount, 0...] }

    /// Append rows (`[1, heads, n, dim]`) as placeholders at `positions` and
    /// return the stored keys and values.
    @discardableResult
    package func append(
        keys newKeys: MLXArray, values newValues: MLXArray, positions: [Int32]
    ) -> (MLXArray, MLXArray) {
        let n = newKeys.dim(2)
        precondition(positions.count == n, "one position per appended row")
        if keyStore == nil {
            let capacity = Swift.max(n, window + compactionSlack)
            keyStore = MLXArray.zeros(
                [newKeys.dim(0), newKeys.dim(1), capacity, newKeys.dim(3)], dtype: newKeys.dtype)
            valueStore = MLXArray.zeros(
                [newValues.dim(0), newValues.dim(1), capacity, newValues.dim(3)],
                dtype: newValues.dtype)
        }
        if storedCount + n > keyStore!.dim(2) || storedCount > window + compactionSlack {
            compact(reserving: n)
        }
        keyStore![.ellipsis, storedCount ..< (storedCount + n), 0...] = newKeys
        valueStore![.ellipsis, storedCount ..< (storedCount + n), 0...] = newValues
        storedCount += n
        rowPositions.append(contentsOf: positions)
        rowValid.append(contentsOf: Array(repeating: false, count: n))
        return (keys!, values!)
    }

    /// The stored rows followed by the block's own `n` rows, written into
    /// the store's slack past the stored count (a dynamic slice update, in
    /// place under the fork's `MLX_DYNSLICE_INPLACE`) and returned as views,
    /// so a pass never concatenates the context with the block. The scratch
    /// rows are overwritten by the next `append`. Falls back to a concat when
    /// the store has no room.
    package func withBlock(
        keys blockKeys: MLXArray, values blockValues: MLXArray
    ) -> (MLXArray, MLXArray) {
        let n = blockKeys.dim(2)
        guard let keyStore, let valueStore, storedCount + n <= keyStore.dim(2) else {
            return (
                concatenated([keys ?? blockKeys[.ellipsis, ..<0, 0...], blockKeys], axis: 2),
                concatenated([values ?? blockValues[.ellipsis, ..<0, 0...], blockValues], axis: 2)
            )
        }
        let start = MLXArray([Int32(storedCount)])
        let k = dynamicSliceUpdated(keyStore, update: blockKeys, start: start, axes: [2])
        let v = dynamicSliceUpdated(valueStore, update: blockValues, start: start, axes: [2])
        let visible = storedCount + n
        return (k[.ellipsis, ..<visible, 0...], v[.ellipsis, ..<visible, 0...])
    }

    /// Commit the first `valid` of the newest `newest` rows; the rest stay
    /// placeholders and drop at the next compaction.
    public func resolve(newest: Int, valid: Int) {
        precondition(newest <= storedCount, "resolving more rows than stored")
        let base = storedCount - newest
        for i in 0 ..< newest {
            rowValid[base + i] = i < valid
        }
    }

    /// Keep the newest `window` valid rows in a fresh padded buffer.
    private func compact(reserving n: Int) {
        let kept = (0 ..< storedCount).filter { rowValid[$0] }.suffix(window)
        let gather = MLXArray(kept.map { Int32($0) })
        let keptKeys = MLX.take(keyStore!, gather, axis: 2)
        let keptValues = MLX.take(valueStore!, gather, axis: 2)
        let capacity = Swift.max(window + compactionSlack, kept.count + n)
        let padK = [keptKeys.dim(0), keptKeys.dim(1), capacity - kept.count, keptKeys.dim(3)]
        let padV = [
            keptValues.dim(0), keptValues.dim(1), capacity - kept.count, keptValues.dim(3),
        ]
        keyStore = concatenated([keptKeys, MLXArray.zeros(padK, dtype: keptKeys.dtype)], axis: 2)
        valueStore = concatenated(
            [keptValues, MLXArray.zeros(padV, dtype: keptValues.dtype)], axis: 2)
        rowPositions = kept.map { rowPositions[$0] }
        rowValid = Array(repeating: true, count: kept.count)
        storedCount = kept.count
    }
}
