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

/// Over TurboQuant caches a Qwen 3.5 tree verify gives each row the logits a
/// chain block of its path gives, bit for bit: with the whole block in one
/// key block (position 301), and with the leaves' depths and slots in
/// different key blocks and partitions (position 316). At this model's size
/// a reordered sum rarely moves a logit;
/// `testTurboVerifyTreeRowsMatchTheirPathsChain` checks the attention itself.
@Test(.serialized, arguments: [0, 8], [300, 315])
func testQwen35TreeVerifyOverTurboQuantComputesEachRowAsItsPathsChain(
    keyBits: Int, promptLength: Int
) throws {
    let model = try treeTestModel()
    var cache = try model.newCache(parameters: nil)
    eval(
        model(
            MLXArray((0 ..< promptLength).map { Int32($0 % 97) }).reshaped(1, promptLength),
            cache: cache))
    _ = try applyKVCacheConfiguration(
        cache: &cache,
        configuration: KVCacheConfiguration(
            strategy: .turboQuant(
                try TurboQuantKVCacheConfiguration(
                    keyPrecision: keyBits == 8 ? .affineEightBit : .fp16,
                    valuePrecision: .fourBit)),
            compatibility: .requireAllLayers))
    try #require(model.dflash2SupportsTree(cache))
    eval(model(MLXArray([Int32(5)]).reshaped(1, 1), cache: cache))

    let tree = FixedTree(tokens: [7, 11, 22, 12, 23, 13, 14, 15])
    let position = promptLength + 1
    func verify(_ tokens: MLXArray, tree layout: DFlash2TreeLayout?) -> MLXArray {
        model.dflash2Verify(
            DFlash2VerifyRequest(
                tokens: tokens, position: MLXArray([Int32(position)]),
                positionUpperBound: position, captureLayers: [1, 2], tree: layout),
            cache: cache.map { $0.copy() }
        ).logits
    }
    let treeLogits = verify(tree.proposal.tokens, tree: tree.proposal.layout)
    for row in 0 ..< tree.tokens.count {
        let path = tree.path(to: row)
        let padded = path + Array(repeating: Int32(0), count: tree.tokens.count - path.count)
        let chainLogits = verify(MLXArray(padded).reshaped(1, padded.count), tree: nil)
        let actual = treeLogits[0, row].asType(.float32)
        let expected = chainLogits[0, path.count - 1].asType(.float32)
        #expect(abs(actual - expected).max().item(Float.self) == 0, "row \(row)")
    }
}

/// A TurboQuant tree of other than 8 rows takes the dequantizing path: a
/// chain-shaped tree there tracks the chain (through the verify kernel at 4
/// rows, the same path at 16).
@Test(arguments: [0, 8], [4, 16])
func testTurboQuantTreeOfOtherWidthsTakesTheMaskedPath(keyBits: Int, width S: Int) throws {
    let (HQ, HK, D, rows) = (4, 2, 128, 300)
    let cache = TurboQuantKVCache(bits: 4, keyBits: keyBits, valueBits: 4)
    let (k, v, q, newK, newV) = withRandomState(MLXRandom.RandomState(seed: 41)) {
        (
            MLXRandom.normal([1, HK, rows, D]).asType(.bfloat16),
            MLXRandom.normal([1, HK, rows, D]).asType(.bfloat16),
            MLXRandom.normal([1, HQ, S, D]).asType(.bfloat16),
            MLXRandom.normal([1, HK, S, D]).asType(.bfloat16),
            MLXRandom.normal([1, HK, S, D]).asType(.bfloat16)
        )
    }
    _ = cache.update(keys: k, values: v)
    let chainRows = MLXArray(Int32(0) ..< Int32(S))
    let tree = DFlash2TreeLayout(
        depths: chainRows,
        ancestry: chainRows.reshaped([S, 1]) .>= chainRows.reshaped([1, S]),
        commits: MLXArray.ones([1, S], dtype: .int32), chainRows: chainRows,
        slotRows: chainRows, slots: chainRows)
    func attend(_ tree: DFlash2TreeLayout?) -> MLXArray {
        cache.verifyAttention(
            queries: q, keys: newK, values: newV, position: MLXArray([Int32(rows)]),
            visibleLength: rows + S, scale: 1 / Float(D).squareRoot(), tree: tree
        ).asType(.float32)
    }
    let chain = attend(nil)
    let error = (abs(attend(tree) - chain).max() / abs(chain).max()).item(Float.self)
    // At 16 rows the chain takes the same path, so the masks agree exactly.
    #expect(S == 16 ? error == 0 : error < 2e-2, "error \(error)")
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

/// The TurboQuant verify kernel computes each tree row as it computes a chain
/// block of the row's path, bit for bit in its float32 output: chain rows
/// read their keys in their own slots, and a leaf reads its key and value at
/// its depth, from its slot. At the Qwen3.8-27B shape, from one 32-key block
/// to 6K keys; at 316 the leaves' depths and slots fall in different key
/// blocks and partitions.
@Test(arguments: [0, 8], [301, 316, 3001, 6140])
func testTurboVerifyTreeRowsMatchTheirPathsChain(keyBits: Int, position: Int) throws {
    let (HQ, HK, S, D) = (24, 4, 8, 256)
    let rows = position + 64
    let visible = position + S + 3
    let scale: Float = 1 / Float(D).squareRoot()
    let codec = MSECodec(dim: D, bits: 4, seed: 43)
    let (q, keys, values) = withRandomState(MLXRandom.RandomState(seed: 29)) {
        (
            MLXRandom.normal([1, HQ, S, D]).asType(.bfloat16),
            MLXRandom.normal([HK, rows, D]).asType(.bfloat16),
            MLXRandom.normal([HK, rows, D]).asType(.bfloat16)
        )
    }
    let (packed, norms) = TurboQuantKernelOps.fusedEncodeWHT(
        input: values.reshaped([-1, D]).asType(.float32), whtSigns: codec.whtSigns!,
        boundaries: codec.boundaries, codebook: codec.codebook, bits: 4, dim: D)
    let quantizedKeys = quantized(keys, groupSize: 64, bits: 8)
    let buffers = [
        keys, quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases!,
        packed.reshaped([HK, rows, -1]), norms.reshaped([HK, rows]),
    ]
    // Rows in topological order, cached in slot order: the anchor and the
    // chain at their depths, then leaves at depths 2 and 3.
    let depths: [Int32] = [0, 1, 2, 2, 3, 3, 4, 5]
    let slots: [Int32] = [0, 1, 6, 2, 7, 3, 4, 5]

    func attention(_ queries: MLXArray, cacheRows: [Int32]?, tree: Bool) -> MLXArray {
        let b =
            cacheRows.map { rows in buffers.map { take($0, MLXArray(rows), axis: 1) } }
            ?? buffers
        return TurboQuantKernelOps.turboVerifyAttention(
            queries: queries,
            keys: keyBits == 8
                ? .affine(weights: b[1], scales: b[2], biases: b[3], groupSize: 64) : .raw(b[0]),
            valPacked: b[4], valNorms: b[5], valCodebook: codec.codebook,
            valRotation: codec.rotation, position: MLXArray([Int32(position)]),
            visibleLength: visible, scale: scale, repeatCount: HQ / HK, valueBits: 4, dim: D,
            treeRows: tree ? (MLXArray(depths), MLXArray(slots)) : nil)
    }
    let tree = attention(q, cacheRows: nil, tree: true)
    for row in 0 ..< S {
        // The row's path as a chain block: its query, key and value at its
        // depth, its ancestors (the chain) before it.
        let depth = Int(depths[row])
        var cacheRows = (0 ..< rows).map(Int32.init)
        cacheRows[position + depth] = Int32(position) + slots[row]
        var queryRows = (0 ..< S).map(Int32.init)
        queryRows[depth] = Int32(row)
        let chain = attention(
            take(q, MLXArray(queryRows), axis: 2), cacheRows: cacheRows, tree: false)
        #expect(
            abs(tree[0..., 0..., row] - chain[0..., 0..., depth]).max().item(Float.self) == 0,
            "row \(row)")
    }
}

/// Opt-in timing (`TEST_RUNNER_TREE_ATTENTION_BENCH=1`): one verify pass's
/// attention over 16 layers at the Qwen3.8-27B shape, one layer after
/// another: MLX's SDPA on a chain block against the tree kernels on a chain
/// block and on a tree with two leaves, then the same for turbo8v4's verify
/// kernel.
/// `TEST_RUNNER_TREE_ATTENTION_BENCH_CONTEXTS=600,6000` picks N.
@Test(
    .enabled(if: ProcessInfo.processInfo.environment["TREE_ATTENTION_BENCH"] == "1"),
    .serialized)
func benchDFlash2TreeAttention() throws {
    let (B, HQ, HK, S, D, layers) = (1, 24, 4, 8, 256, 16)
    let contexts =
        (ProcessInfo.processInfo.environment["TREE_ATTENTION_BENCH_CONTEXTS"] ?? "600,6000,16000")
        .split(separator: ",").compactMap { Int($0) }
    for context in contexts {
        let capacity = context + 256
        let visible = context + S
        let (q, keys, values) = withRandomState(MLXRandom.RandomState(seed: 5)) {
            (
                MLXRandom.normal([B, HQ, S, D]).asType(.bfloat16),
                (0 ..< layers).map { _ in MLXRandom.normal([B, HK, capacity, D]).asType(.bfloat16)
                },
                (0 ..< layers).map { _ in MLXRandom.normal([B, HK, capacity, D]).asType(.bfloat16) }
            )
        }
        eval([q] + keys + values)
        let scale: Float = 1 / Float(D).squareRoot()
        let position = MLXArray([Int32(context)])
        let columns = MLXArray(Int32(0) ..< Int32(visible)).reshaped([1, visible])
        let rows = (MLXArray(Int32(context)) + MLXArray(Int32(0) ..< Int32(S))).reshaped([S, 1])
        let mask = columns .<= rows
        let chainRows = MLXArray(Int32(0) ..< Int32(S))
        let treeDepths = MLXArray([Int32(0), 1, 2, 2, 3, 3, 4, 5])
        let treeSlots = MLXArray([Int32(0), 1, 6, 2, 7, 3, 4, 5])
        // The same keys and values as a turbo8v4 cache holds them.
        let codec = MSECodec(dim: D, bits: 4, seed: 43)
        let turbo = (0 ..< layers).map { l in
            let (packed, norms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: values[l].reshaped([-1, D]).asType(.float32), whtSigns: codec.whtSigns!,
                boundaries: codec.boundaries, codebook: codec.codebook, bits: 4, dim: D)
            let k = quantized(keys[l].reshaped([HK, capacity, D]), groupSize: 64, bits: 8)
            return [
                k.wq, k.scales, k.biases!, packed.reshaped([HK, capacity, -1]),
                norms.reshaped([HK, capacity]),
            ]
        }
        eval(turbo.flatMap { $0 })
        func turboLayer(
            _ x: MLXArray, _ l: Int, _ treeRows: (depths: MLXArray, slots: MLXArray)?
        ) -> MLXArray {
            let t = turbo[l]
            return TurboQuantKernelOps.turboVerifyAttention(
                queries: x,
                keys: .affine(weights: t[0], scales: t[1], biases: t[2], groupSize: 64),
                valPacked: t[3], valNorms: t[4], valCodebook: codec.codebook,
                valRotation: codec.rotation, position: position, visibleLength: visible,
                scale: scale, repeatCount: HQ / HK, valueBits: 4, dim: D, treeRows: treeRows)
        }
        // One verify pass: each layer's queries wait for the previous layer's
        // output (a zero-weighted term), so the layers run one after another.
        func measure(_ layer: (MLXArray, Int) -> MLXArray) -> Double {
            func pass() -> MLXArray {
                var x = q
                for l in 0 ..< layers { x = q + (layer(x, l) * 0).asType(q.dtype) }
                return x
            }
            let start = Date()
            for _ in 0 ..< 20 { eval(pass()) }
            return Date().timeIntervalSince(start) * 1000 / 20
        }
        let variants: [(String, (MLXArray, Int) -> MLXArray)] = [
            (
                "MLX SDPA, chain",
                { x, l in
                    MLXFast.scaledDotProductAttention(
                        queries: x, keys: keys[l][.ellipsis, ..<visible, 0...],
                        values: values[l][.ellipsis, ..<visible, 0...], scale: scale,
                        mask: .array(mask))
                }
            ),
            (
                "tree kernel, chain",
                { x, l in
                    dflash2TreeAttention(
                        queries: x, keys: keys[l], values: values[l], position: position,
                        depths: chainRows, slots: chainRows, visibleLength: visible,
                        scale: scale)!
                }
            ),
            (
                "tree kernel, 2 leaves",
                { x, l in
                    dflash2TreeAttention(
                        queries: x, keys: keys[l], values: values[l], position: position,
                        depths: treeDepths, slots: treeSlots, visibleLength: visible,
                        scale: scale)!
                }
            ),
            ("turbo8v4 kernel, chain", { x, l in turboLayer(x, l, nil) }),
            ("turbo8v4 tree kernel, chain", { x, l in turboLayer(x, l, (chainRows, chainRows)) }),
            (
                "turbo8v4 tree kernel, 2 leaves",
                { x, l in turboLayer(x, l, (treeDepths, treeSlots)) }
            ),
        ]
        for (_, layer) in variants { for _ in 0 ..< 3 { _ = measure(layer) } }
        var samples = Array(repeating: [Double](), count: variants.count)
        for _ in 0 ..< 7 {
            for (i, variant) in variants.enumerated() { samples[i].append(measure(variant.1)) }
        }
        for (i, variant) in variants.enumerated() {
            let median = samples[i].sorted()[samples[i].count / 2]
            print(
                "[TREE-ATTN-BENCH] N \(visible) \(variant.0): \(String(format: "%.3f", median)) ms / 16 layers (median of 7)"
            )
        }
    }
}
