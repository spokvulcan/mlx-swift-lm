// Copyright © 2026 Apple Inc.

// Tree blocks for DFlash2: the proposal's layout, greedy acceptance of a
// path, and a Qwen 3.5 verify pass that computes each row as a chain block
// of its own path would.

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXLMCommon

/// One fixed tree, rows in topological order:
///
///     row  0  1  2  3  4  5  6  7
///     kind a  c  l  c  l  c  c  c     (anchor, chain, leaf)
///     depth 0 1  2  2  3  3  4  5
///
/// The leaf at depth 2 is the chain node at depth 2's sibling, the leaf at
/// depth 3 the chain node at depth 3's.
private struct FixedTree {
    let tokens: [Int32]
    let depths: [Int32] = [0, 1, 2, 2, 3, 3, 4, 5]
    let commits: [Int32] = [1, 1, 0, 1, 0, 1, 1, 1]
    let parents: [Int32] = [0, 0, 1, 1, 3, 3, 5, 6]
    let chainRows: [Int32] = [0, 1, 3, 5, 6, 7, 7, 7]
    let slots: [Int32] = [0, 1, 6, 2, 7, 3, 4, 5]

    init(tokens: [Int32]) { self.tokens = tokens }

    var proposal: DFlash2TreeProposal {
        let S = tokens.count
        let slotRows = (0 ..< S).map { slot in Int32(slots.firstIndex(of: Int32(slot))!) }
        var ancestry: [Bool] = []
        for row in 0 ..< S {
            for slot in 0 ..< S {
                let other = Int(slotRows[slot])
                ancestry.append(
                    other == row || (commits[other] == 1 && depths[other] < depths[row]))
            }
        }
        return DFlash2TreeProposal(
            tokens: MLXArray(tokens).reshaped([1, S]),
            layout: DFlash2TreeLayout(
                depths: MLXArray(depths), ancestry: MLXArray(ancestry).reshaped([S, S]),
                commits: MLXArray(commits).reshaped([1, S]), chainRows: MLXArray(chainRows),
                slotRows: MLXArray(slotRows), slots: MLXArray(slots)),
            parents: MLXArray(parents), chainLength: MLXArray(Int32(5)))
    }

    /// The tokens from the anchor down to `row`.
    func path(to row: Int) -> [Int32] {
        var rows = [row]
        while rows[0] != 0 { rows.insert(Int(parents[rows[0]]), at: 0) }
        return rows.map { tokens[$0] }
    }
}

/// Logits whose argmax at each row is `targets[row]`.
private func logits(argmax targets: [Int], vocabulary: Int = 128) -> MLXArray {
    var values = [Float](repeating: 0, count: targets.count * vocabulary)
    for (row, target) in targets.enumerated() { values[row * vocabulary + target] = 1 }
    return MLXArray(values).reshaped([targets.count, vocabulary])
}

@Test
func testDFlash2TreeAcceptanceTakesTheChainThenOneLeaf() {
    let tree = FixedTree(tokens: [100, 11, 22, 12, 23, 13, 14, 15])

    // The chain holds through depth 2, misses at depth 3, and the leaf there
    // is the target's choice: the path ends on the leaf's row.
    var path = dflash2TreeAcceptance(
        tree.proposal, logits: logits(argmax: [11, 12, 0, 23, 44, 0, 0, 0]))
    #expect(path.accepted.item(Int.self) == 3)
    #expect(path.bonus.asArray(Int32.self) == [44])
    #expect(path.pathRows.asArray(Int32.self) == [0, 1, 3, 4, 4, 4, 4, 4])
    #expect(path.pathSlots.asArray(Int32.self) == [0, 1, 2, 7, 7, 7, 7, 7])
    #expect(path.packed.asArray(Int32.self) == [11, 12, 23, 23, 23, 23, 23, 3, 44])

    // No leaf matches: the chain's own prefix and the bonus at its end.
    path = dflash2TreeAcceptance(
        tree.proposal, logits: logits(argmax: [11, 12, 0, 99, 0, 0, 0, 0]))
    #expect(path.accepted.item(Int.self) == 2)
    #expect(path.bonus.asArray(Int32.self) == [99])
    #expect(path.pathRows.asArray(Int32.self) == [0, 1, 3, 3, 3, 3, 3, 3])

    // The whole chain.
    path = dflash2TreeAcceptance(
        tree.proposal, logits: logits(argmax: [11, 12, 0, 13, 0, 14, 15, 77]))
    #expect(path.accepted.item(Int.self) == 5)
    #expect(path.bonus.asArray(Int32.self) == [77])
    #expect(path.pathRows.asArray(Int32.self) == [0, 1, 3, 5, 6, 7, 7, 7])
}

@Test
func testDFlash2TreeProposalKeepsTheChainAndItsBestSiblings() {
    // Four candidates per position. Candidate 0 leads everywhere (the
    // chain); candidate 1 nearly ties it at position 1 and trails it by 0.5
    // from position 3 on, so the chain's tail scores below two siblings.
    let L = 7
    let K = 4
    let candidates = MLXArray((0 ..< (L * K)).map { Int32(10 + $0) }).reshaped([1, L, K])
    var unary = [Float](repeating: -8, count: L * K)
    for t in 0 ..< L { unary[t * K] = 0 }
    unary[1 * K + 1] = -0.1
    for t in 3 ..< L { unary[t * K + 1] = -0.5 }
    let chain = MLXArray((0 ..< L).map { Int32(10 + $0 * K) }).reshaped([1, L])
    let lattice = DFlash2Lattice(
        candidates: candidates, unary: MLXArray(unary).reshaped([1, L, K]),
        edges: MLXArray.zeros([1, L - 1, K, K]), anchorEdges: MLXArray.zeros([1, K]),
        tokens: chain)
    let anchor = MLXArray([Int32(5)])

    let tree = dflash2TreeProposal(lattice: lattice, anchor: anchor, siblingRanks: 3)
    let tokens = tree.tokens.asArray(Int32.self)
    let depths = tree.layout.depths.asArray(Int32.self)
    let commits = tree.layout.commits.asArray(Int32.self)
    let slots = tree.layout.slots.asArray(Int32.self)
    let slotRows = tree.layout.slotRows.asArray(Int32.self)
    let chainRows = tree.layout.chainRows.asArray(Int32.self)
    let parents = tree.parents.asArray(Int32.self)
    let chainLength = tree.chainLength.item(Int.self)

    // Kept: the chain through depth 5 and the siblings at depths 2 and 4.
    #expect(tokens.count == L + 1)
    #expect(tokens[0] == 5)
    #expect(chainLength == 5)
    for depth in 0 ... chainLength {
        let row = Int(chainRows[depth])
        #expect(commits[row] == 1 && depths[row] == Int32(depth))
        #expect(slots[row] == Int32(depth))
        if depth > 0 { #expect(tokens[row] == Int32(10 + (depth - 1) * K)) }
    }
    let leaves = (0 ..< tokens.count).filter { commits[$0] == 0 }
    #expect(leaves.map { tokens[$0] } == [15, 23])
    #expect(leaves.map { depths[$0] } == [2, 4])
    for (index, leaf) in leaves.enumerated() {
        let parent = Int(parents[leaf])
        #expect(parent == Int(chainRows[Int(depths[leaf]) - 1]))
        #expect(leaf > parent)
        #expect(slots[leaf] == Int32(chainLength + 1 + index))
    }
    for slot in 0 ..< tokens.count { #expect(slots[Int(slotRows[slot])] == Int32(slot)) }
    // Each row sees exactly itself and its chain ancestors.
    let ancestry = tree.layout.ancestry.asArray(Bool.self)
    for row in 0 ..< tokens.count {
        for slot in 0 ..< tokens.count {
            let other = Int(slotRows[slot])
            let expected = other == row || (commits[other] == 1 && depths[other] < depths[row])
            #expect(ancestry[row * tokens.count + slot] == expected, "row \(row) slot \(slot)")
        }
    }

    // Without siblings the tree is the chain.
    let plain = dflash2TreeProposal(lattice: lattice, anchor: anchor, siblingRanks: 0)
    #expect(plain.tokens.asArray(Int32.self) == [5] + chain.asArray(Int32.self))
    #expect(plain.layout.commits.asArray(Int32.self) == Array(repeating: 1, count: L + 1))
    #expect(plain.layout.slots.asArray(Int32.self) == (0 ... L).map(Int32.init))
}

/// A Qwen 3.5 small enough for a test whose verify serves trees: 128-wide
/// linear heads (the fused conv and scan kernels) and stacked 4-bit
/// projections (the fused q/k norm and rope).
private func treeTestModel() throws -> Qwen35TextModel {
    // 128-wide linear heads, which the fused conv and scan kernels need.
    let configuration = try JSONDecoder().decode(
        Qwen35TextConfiguration.self,
        from: Data(
            """
            {
                "model_type": "qwen3_5",
                "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 128,
                "linear_num_value_heads": 2, "linear_num_key_heads": 1,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "vocab_size": 128,
                "full_attention_interval": 2,
                "num_experts": 0, "num_experts_per_tok": 0,
                "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16
            }
            """.utf8))
    let model = withRandomState(MLXRandom.RandomState(seed: 31)) {
        Qwen35TextModel(configuration)
    }
    try model.update(
        parameters: model.parameters().mapValues { $0.asType(.bfloat16) }, verify: [])
    quantize(model: model, groupSize: 64, bits: 4)
    #expect(stackSameInputProjections(in: model) > 0)
    return model
}

@Test
func testQwen35TreeVerifyComputesEachRowAsItsPathsChain() throws {
    let model = try treeTestModel()
    let cache = try model.newCache(parameters: nil)
    let promptLength = 40
    eval(
        model(
            MLXArray((0 ..< promptLength).map { Int32($0 * 5 % 97) }).reshaped(1, promptLength),
            cache: cache))
    try #require(model.dflash2SupportsTree(cache))

    let tree = FixedTree(tokens: [7, 11, 22, 12, 23, 13, 14, 15])
    let position = MLXArray([Int32(promptLength)])
    let treeLogits = model.dflash2Verify(
        DFlash2VerifyRequest(
            tokens: tree.proposal.tokens, position: position,
            positionUpperBound: promptLength, captureLayers: [1, 2],
            tree: tree.proposal.layout),
        cache: cache.map { $0.copy() }
    ).logits

    for row in 0 ..< tree.tokens.count {
        // The row's path as a chain block of the same width.
        let path = tree.path(to: row)
        let padded = path + Array(repeating: Int32(0), count: tree.tokens.count - path.count)
        let chainLogits = model.dflash2Verify(
            DFlash2VerifyRequest(
                tokens: MLXArray(padded).reshaped(1, padded.count), position: position,
                positionUpperBound: promptLength, captureLayers: [1, 2]),
            cache: cache.map { $0.copy() }
        ).logits
        let actual = treeLogits[0, row].asType(.float32)
        let expected = chainLogits[0, path.count - 1].asType(.float32)
        // Chain rows read their keys in the same slots, and a leaf reads
        // its own key at its depth: bit for bit either way.
        #expect(abs(actual - expected).max().item(Float.self) == 0, "row \(row)")
    }
}

/// Over TurboQuant attention caches a tree verify tracks the bf16 cache's,
/// and the commit's row gather moves the compressed rows unchanged.
@Test(.serialized, arguments: [0, 8])
func testQwen35TreeVerifyOverTurboQuantTracksThePlainCache(keyBits: Int) throws {
    let model = try treeTestModel()
    var cache = try model.newCache(parameters: nil)
    let promptLength = 300
    eval(
        model(
            MLXArray((0 ..< promptLength).map { Int32($0 % 97) }).reshaped(1, promptLength),
            cache: cache))
    let plain = cache.map { $0.copy() }
    _ = try applyKVCacheConfiguration(
        cache: &cache,
        configuration: KVCacheConfiguration(
            strategy: .turboQuant(
                try TurboQuantKVCacheConfiguration(
                    keyPrecision: keyBits == 8 ? .affineEightBit : .fp16,
                    valuePrecision: .fourBit)),
            compatibility: .requireAllLayers))
    try #require(model.dflash2SupportsTree(cache))
    let step = MLXArray([Int32(5)]).reshaped(1, 1)
    eval(model(step, cache: cache), model(step, cache: plain))

    let tree = FixedTree(tokens: [7, 11, 22, 12, 23, 13, 14, 15])
    let position = promptLength + 1
    let request = DFlash2VerifyRequest(
        tokens: tree.proposal.tokens, position: MLXArray([Int32(position)]),
        positionUpperBound: position, captureLayers: [1, 2], tree: tree.proposal.layout)
    let turbo = model.dflash2Verify(request, cache: cache)
    let reference = model.dflash2Verify(request, cache: plain)
    for row in 0 ..< tree.tokens.count {
        let a = turbo.logits[0, row].asType(.float32)
        let r = reference.logits[0, row].asType(.float32)
        let cos = ((a * r).sum() / (sqrt((a * a).sum()) * sqrt((r * r).sum()) + 1e-9))
            .item(Float.self)
        #expect(cos > 0.95, "keyBits \(keyBits) row \(row): cos \(cos)")
    }

    // The path anchor, chain, chain, leaf at depth 3, padded.
    let slots: [Int32] = [0, 1, 2, 7, 7, 7, 7, 7]
    for entry in cache {
        guard let turboCache = entry as? TurboQuantKVCache else { continue }
        let before = turboCache.dequantizedRows(position + 8)
        turboCache.gatherRows(position: position, rows: MLXArray(slots), width: 8)
        let after = turboCache.dequantizedRows(position + 8)
        let moved = MLXArray(slots.map { Int32(position) + $0 })
        let expectedKeys = take(before.keys, moved, axis: 2)
        let expectedValues = take(before.values, moved, axis: 2)
        let block = position ..< (position + 8)
        #expect(abs(after.keys[0..., 0..., block] - expectedKeys).max().item(Float.self) == 0)
        #expect(
            abs(after.values[0..., 0..., block] - expectedValues).max().item(Float.self) == 0)
        #expect(
            abs(after.keys[0..., 0..., ..<position] - before.keys[0..., 0..., ..<position]).max()
                .item(Float.self) == 0)
    }
}

/// The tree attention kernel is MLX's one-pass vector SDPA bit for bit: on a
/// chain block it reproduces `scaledDotProductAttention`, and a leaf row
/// reproduces it as computed with its own key at its depth, where a chain
/// block puts the token.
@Test(arguments: [
    (position: 300, upper: 307), (position: 800, upper: 803), (position: 1100, upper: 1104),
    (position: 3001, upper: 3008), (position: 6140, upper: 6141),
])
func testDFlash2TreeAttentionMatchesTheChainKernel(spot: (position: Int, upper: Int)) throws {
    let (B, HQ, HK, S, D) = (1, 24, 4, 8, 256)
    let capacity = 6400
    let visible = spot.upper + S
    let (q, chainKeys, chainValues, leafKey, leafValue) = withRandomState(
        MLXRandom.RandomState(seed: 23)
    ) {
        (
            (MLXRandom.normal([B, HQ, S, D]) * 2).asType(.bfloat16),
            (MLXRandom.normal([B, HK, capacity, D]) * 2).asType(.bfloat16),
            MLXRandom.normal([B, HK, capacity, D]).asType(.bfloat16),
            (MLXRandom.normal([B, HK, 1, D]) * 2).asType(.bfloat16),
            MLXRandom.normal([B, HK, 1, D]).asType(.bfloat16)
        )
    }
    let scale: Float = 1 / Float(D).squareRoot()
    let position = MLXArray([Int32(spot.position)])
    let columns = MLXArray(Int32(0) ..< Int32(visible)).reshaped([1, visible])
    let rows = (MLXArray(Int32(spot.position)) + MLXArray(Int32(0) ..< Int32(S))).reshaped([S, 1])
    let chainMask = columns .<= rows

    // A chain block: the kernel equals MLX's SDPA.
    let reference = MLXFast.scaledDotProductAttention(
        queries: q, keys: chainKeys[.ellipsis, ..<visible, 0...],
        values: chainValues[.ellipsis, ..<visible, 0...], scale: scale, mask: .array(chainMask))
    let chainRows = MLXArray(Int32(0) ..< Int32(S))
    let chain = try #require(
        dflash2TreeAttention(
            queries: q, keys: chainKeys, values: chainValues, position: position,
            depths: chainRows, slots: chainRows, visibleLength: visible, scale: scale))
    #expect(abs(chain.asType(.float32) - reference.asType(.float32)).max().item(Float.self) == 0)

    // A leaf at depth 2 cached in slot 6: as a chain block holding it at
    // slot 2 computes it.
    let leafSlot = spot.position + 6
    let treeKeys = dynamicSliceUpdated(
        chainKeys, update: leafKey, start: MLXArray([Int32(leafSlot)]), axes: [2])
    let treeValues = dynamicSliceUpdated(
        chainValues, update: leafValue, start: MLXArray([Int32(leafSlot)]), axes: [2])
    let pathKeys = dynamicSliceUpdated(
        chainKeys, update: leafKey, start: MLXArray([Int32(spot.position + 2)]), axes: [2])
    let pathValues = dynamicSliceUpdated(
        chainValues, update: leafValue, start: MLXArray([Int32(spot.position + 2)]), axes: [2])
    let pathReference = MLXFast.scaledDotProductAttention(
        queries: q, keys: pathKeys[.ellipsis, ..<visible, 0...],
        values: pathValues[.ellipsis, ..<visible, 0...], scale: scale, mask: .array(chainMask))
    let tree = try #require(
        dflash2TreeAttention(
            queries: q, keys: treeKeys, values: treeValues, position: position,
            depths: MLXArray([Int32(0), 1, 2, 3, 4, 5, 6, 7]),
            slots: MLXArray([Int32(0), 1, 6, 3, 4, 5, 2, 7]), visibleLength: visible,
            scale: scale))
    let leafRow = tree[0..., 0..., 2, 0...].asType(.float32)
    #expect(
        abs(leafRow - pathReference[0..., 0..., 2, 0...].asType(.float32)).max().item(Float.self)
            == 0)

}
