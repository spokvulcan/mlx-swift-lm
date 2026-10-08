// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - Tree blocks

/// A greedy block drafted as a chain with leaf siblings.
///
/// The chain is the selector's greedy path. Every position also offers its
/// `siblingRanks` next-best candidates as leaves hanging off the chain node
/// before it. An item scores the summed log-probability of its path under
/// the selector's local distributions (softmax of unary plus the edge from
/// the chain's previous choice, over `temperature`); the block keeps the
/// `S - 1` best items. No item
/// outscores its parent (a sibling's log-probability is at most the chain
/// choice's, and every log-probability is at most zero), and ties go to the
/// earlier item, so the kept items always form a tree: a chain prefix plus
/// leaves on it. Rows are in topological order with each position's leaves
/// right after their parent, which is what a scan that keeps a leaf's state
/// off the chain needs. Static shapes throughout; every output may be lazy.
public func dflash2TreeProposal(
    lattice: DFlash2Lattice, anchor: MLXArray, siblingRanks: Int = 3, temperature: Float = 1
) -> DFlash2TreeProposal {
    let candidates = lattice.candidates[0].asType(.int32)
    let L = candidates.dim(0)
    let K = candidates.dim(1)
    let S = L + 1
    let ranks = Swift.min(siblingRanks, K - 1)
    let chain = lattice.tokens[0].asType(.int32)
    if ranks <= 0 {
        return dflash2ChainAsTree(anchor: anchor, chain: chain)
    }

    // The chain's candidate index at each position, and every position's
    // scores given the chain's previous choice (as the greedy walk sees them).
    let chainIndex = argMax(
        (candidates .== chain.reshaped([L, 1])).asType(.int32), axis: -1
    ).asType(.int32)
    let unary = lattice.unary[0]
    var scores = unary[0 ..< 1] + lattice.anchorEdges
    if L > 1 {
        let fromChain = takeAlong(
            lattice.edges[0], chainIndex[0 ..< (L - 1)].reshaped([L - 1, 1, 1]), axis: 1
        )[0..., 0, 0...]
        scores = concatenated([scores, unary[1...] + fromChain], axis: 0)
    }
    let logProbs = logSoftmax(scores.asType(.float32) / temperature, axis: -1)
    let chainLogProbs = takeAlong(logProbs, chainIndex.reshaped([L, 1]), axis: -1)[0..., 0]
    let pathLogProb = concatenated([MLXArray.zeros([1], dtype: .float32), cumsum(chainLogProbs)])

    // Siblings: the best other candidates at each position.
    let others = MLX.where(
        MLXArray(Int32(0) ..< Int32(K)).reshaped([1, K]) .== chainIndex.reshaped([L, 1]),
        MLXArray(-Float.infinity), logProbs)
    let siblingIndex = argSort(-others, axis: -1)[0..., 0 ..< ranks].asType(.int32)
    let siblingLogProbs = takeAlong(others, siblingIndex, axis: -1)

    // Items: chain node t (score: its path), then sibling (t, r). The small
    // index term breaks ties toward the earlier item.
    let items = L + L * ranks
    let itemScores =
        concatenated([
            pathLogProb[1...],
            (pathLogProb[0 ..< L].reshaped([L, 1]) + siblingLogProbs).reshaped([-1]),
        ])
        - MLXArray(Int32(0) ..< Int32(items)).asType(.float32) * 1e-6
    let kept = argSort(-itemScores)[0 ..< (S - 1)].asType(.int32)
    let keptIsChain = kept .< Int32(L)
    let siblingItem = maximum(kept - Int32(L), Int32(0))
    let keptPosition = MLX.where(keptIsChain, kept, floorDivide(siblingItem, Int32(ranks)))
    let keptRank = remainder(siblingItem, Int32(ranks))

    // Topological order: position t's leaves, then its chain node.
    let sortKey = MLX.where(
        keptIsChain, (2 * keptPosition + 1) * Int32(K), 2 * keptPosition * Int32(K) + keptRank)
    let order = argSort(sortKey)
    let rowIsChain = take(keptIsChain, order)
    let rowPosition = take(keptPosition, order)
    let rowCandidate = MLX.where(
        rowIsChain, take(chainIndex, rowPosition),
        take(
            siblingIndex.reshaped([-1]), rowPosition * Int32(ranks) + take(keptRank, order)))
    let rowToken = take(candidates.reshaped([-1]), rowPosition * Int32(K) + rowCandidate)

    let tokens = concatenated([anchor.asType(.int32).reshaped([1]), rowToken]).reshaped([1, S])
    let depths = concatenated([MLXArray([Int32(0)]), rowPosition + 1])
    let commits = concatenated([MLXArray([Int32(1)]), rowIsChain.asType(.int32)])
    let chainLength = rowIsChain.asType(.int32).sum()

    // The chain's row at each depth (the anchor's, 0, at depth 0); past the
    // chain, its last row.
    let rowNumbers = MLXArray(Int32(1) ..< Int32(S)).reshaped([1, S - 1])
    let depthGrid = MLXArray(Int32(0) ..< Int32(S)).reshaped([S, 1])
    let chainAt =
        rowIsChain.reshaped([1, S - 1]) .&& ((rowPosition + 1).reshaped([1, S - 1]) .== depthGrid)
    var chainRows = (chainAt.asType(.int32) * rowNumbers).sum(axis: 1)
    chainRows = MLX.where(
        MLXArray(Int32(0) ..< Int32(S)) .> chainLength,
        take(chainRows, chainLength.reshaped([1])), chainRows)

    // A row's parent is the chain one depth up; its ancestors are the chain
    // rows (and the anchor) shallower than it.
    let parents = take(chainRows, maximum(depths - 1, Int32(0)))
    let rowIndex = MLXArray(Int32(0) ..< Int32(S))
    let ancestorRows =
        (rowIndex.reshaped([S, 1]) .== rowIndex.reshaped([1, S]))
        .|| ((commits.reshaped([1, S]) .== 1)
            .&& (depths.reshaped([1, S]) .< depths.reshaped([S, 1])))

    // Cache slots: the anchor and the chain at their depths, then the leaves
    // in row order.
    let isLeaf = 1 - commits
    let slots = MLX.where(
        commits .== 1, depths, chainLength + cumsum(isLeaf) - isLeaf + 1)
    let slotRows = argSort(slots).asType(.int32)
    let ancestry = take(ancestorRows, slotRows, axis: 1)

    return DFlash2TreeProposal(
        tokens: tokens,
        layout: DFlash2TreeLayout(
            depths: depths, ancestry: ancestry, commits: commits.reshaped([1, S]),
            chainRows: chainRows, slotRows: slotRows, slots: slots),
        parents: parents, chainLength: chainLength)
}

/// A chain laid out as a tree (no leaves): what a tree verify of it must
/// reproduce bit for bit.
public func dflash2ChainAsTree(anchor: MLXArray, chain: MLXArray) -> DFlash2TreeProposal {
    let L = chain.dim(0)
    let S = L + 1
    let rows = MLXArray(Int32(0) ..< Int32(S))
    return DFlash2TreeProposal(
        tokens: concatenated([anchor.asType(.int32).reshaped([1]), chain.asType(.int32)])
            .reshaped([1, S]),
        layout: DFlash2TreeLayout(
            depths: rows, ancestry: rows.reshaped([1, S]) .<= rows.reshaped([S, 1]),
            commits: MLXArray.ones([1, S], dtype: .int32), chainRows: rows, slotRows: rows,
            slots: rows),
        parents: maximum(rows - 1, Int32(0)), chainLength: MLXArray(Int32(L)))
}

/// Greedy acceptance of a tree block, as lazy arrays.
public struct DFlash2TreeAcceptance {
    /// `[]` int32 drafts accepted (the path's length past the anchor).
    public var accepted: MLXArray
    /// `[1]` int32: the target's argmax after the path.
    public var bonus: MLXArray
    /// `[S]` int32: the path's rows from the anchor, padded with its last row.
    public var pathRows: MLXArray
    /// `[S]` int32: the cache slots of ``pathRows``.
    public var pathSlots: MLXArray
    /// `[path tokens past the anchor (S - 1), accepted, bonus]`, the chain's
    /// packed layout, for the round's host sync.
    public var packed: MLXArray
}

/// The chain is accepted for as long as each node is the target's argmax at
/// its parent; at the first miss, a leaf at that depth that is the target's
/// argmax there extends the path by one. The bonus is the target's argmax at
/// the path's last row. Every accepted token is the target's own greedy
/// choice. One compiled launch group per round.
public func dflash2TreeAcceptance(
    _ proposal: DFlash2TreeProposal, logits: MLXArray
) -> DFlash2TreeAcceptance {
    let layout = proposal.layout
    let outputs = compiledTreeAcceptance([
        proposal.tokens, proposal.parents, proposal.chainLength, layout.depths,
        layout.chainRows, layout.commits, layout.slots, logits,
    ])
    return DFlash2TreeAcceptance(
        accepted: outputs[0], bonus: outputs[1], pathRows: outputs[2], pathSlots: outputs[3],
        packed: outputs[4])
}

private let compiledTreeAcceptance: @Sendable ([MLXArray]) -> [MLXArray] = compile {
    (arguments: [MLXArray]) -> [MLXArray] in
    let S = arguments[0].dim(1)
    let tokens = arguments[0][0].asType(.int32)
    let parents = arguments[1]
    let chainLength = arguments[2]
    let depths = arguments[3].asType(.int32)
    let chainRows = arguments[4].asType(.int32)
    let commits = arguments[5].reshaped([-1]).asType(.int32)
    let slots = arguments[6].asType(.int32)
    let targets = argMax(arguments[7], axis: -1).asType(.int32)
    let matched = tokens .== take(targets, parents)

    let chainMatched =
        take(matched, chainRows[1...]) .&& (MLXArray(Int32(1) ..< Int32(S)) .<= chainLength)
    let chainAccepted = cumprod(chainMatched.asType(.int32)).sum()
    let leafHit = (commits .== 0) .&& (depths .== chainAccepted + 1) .&& matched
    let hit = leafHit.any()
    let leafRow = argMax(leafHit.asType(.int32)).asType(.int32)
    let accepted = chainAccepted + hit.asType(.int32)
    let lastRow = MLX.where(hit, leafRow, take(chainRows, chainAccepted.reshaped([1]))[0])
    let bonus = take(targets, lastRow.reshaped([1]))

    let step = MLXArray(Int32(0) ..< Int32(S))
    let pathRows = MLX.where(
        step .<= chainAccepted, take(chainRows, minimum(step, chainAccepted)), lastRow)
    return [
        accepted, bonus, pathRows, take(slots, pathRows),
        concatenated([take(tokens, pathRows[1...]), accepted.reshaped([1]), bonus]),
    ]
}
