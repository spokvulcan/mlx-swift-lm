// Copyright © 2026 Apple Inc.
//
// DFlash2ParityTests.swift
// mlx-swift-lm
//
// Real-weights parity of the DFlash2 draft model against the Python
// reference (z-lab/dflash `model_mlx.py`). The fixture
// (`research/fixtures/draft_only.safetensors` in the tesseract repo) feeds a
// fixed block + synthetic target hidden states through the real bf16 draft
// checkpoint and records every intermediate. This test replays it through
// the Swift port and compares.
//
// Gated on the fixture + checkpoint existing locally (skipped otherwise) —
// regenerate the fixture with `research/dump_draft_only.py`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM

private func fixtureDirectory() -> URL? {
    let fixtures = URL(
        fileURLWithPath: "/Users/owl/projects/tesseract/research/fixtures")
    let file = fixtures.appendingPathComponent("draft_only.safetensors")
    return FileManager.default.fileExists(atPath: file.path) ? fixtures : nil
}

private func draftSnapshotDirectory() -> URL? {
    let hub = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots")
    guard let entries = try? FileManager.default.contentsOfDirectory(
        at: hub, includingPropertiesForKeys: nil)
    else { return nil }
    return entries.first(where: {
        FileManager.default.fileExists(
            atPath: $0.appendingPathComponent("model.safetensors").path)
    })
}

@Test(.serialized)
func testDFlash2DraftModelParityWithPythonReference() throws {
    guard let fixtures = fixtureDirectory(), let draftDir = draftSnapshotDirectory()
    else {
        Issue.record("DFlash2 parity fixture or draft checkpoint not present; skipping")
        return
    }

    let draft = try loadDFlash2Draft(from: draftDir)
    let fixture = try loadArrays(url: fixtures.appendingPathComponent("draft_only.safetensors"))
    let meta = try JSONSerialization.jsonObject(
        with: Data(contentsOf: fixtures.appendingPathComponent("draft_only.json"))
    ) as! [String: Any]
    let expectedTokens = meta["tokens"] as! [Int]
    let cacheOffset = meta["cache_offset"] as! Int

    // Bind the fixture's synthetic embedding + head (stand-ins for the target's).
    let vocab = draft.config.vocabularySize
    let hidden = draft.config.hiddenSize
    let embed = Embedding(embeddingCount: vocab, dimensions: hidden)
    let head = Linear(hidden, vocab, bias: false)
    try embed.update(
        parameters: ModuleParameters.unflattened([("weight", fixture["embed_weight"]!)]),
        verify: [])
    try head.update(
        parameters: ModuleParameters.unflattened([("weight", fixture["head_weight"]!)]),
        verify: [])
    draft.bindDFlashTarget(embedding: embed, head: head)

    let block = fixture["block"]!
    let targetHidden = fixture["target_hidden"]!

    func freshCaches() -> [DFlash2ContextCache] {
        let caches = draft.makeDFlashContextCaches()
        for cache in caches { cache.offset = cacheOffset }
        return caches
    }

    let caches = freshCaches()
    let finalHidden = draft.hiddenStates(
        block, targetHidden: targetHidden, cache: caches, logitsStart: 1)
    let logits = draft.computeLogits(finalHidden)
    let (tokens, candidates, _) = draft.dflashPropose(
        block, targetHidden: targetHidden, cache: freshCaches(),
        temperature: 0, logitsStart: 1)
    eval(finalHidden, logits, tokens, candidates)

    // Numeric parity (bf16 accumulation-order slack).
    func assertClose(_ got: MLXArray, _ expected: MLXArray, _ name: String, tol: Float = 0.06) {
        let diff = (got.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self)
        let scale = expected.asType(.float32).abs().max().item(Float.self)
        #expect(
            diff < tol * max(scale, 1),
            "\(name): max |diff| \(diff) vs scale \(scale)")
    }

    assertClose(finalHidden, fixture["draft_final_hidden"]!, "final_hidden")
    // Logits: loose tolerance (bf16 head over 5120-dim), tokens must match.
    assertClose(logits, fixture["draft_logits"]!, "logits", tol: 0.10)
    #expect(tokens.asArray(Int32.self) == expectedTokens.map { Int32($0) })

    // Context cache advanced by the ctx rows fed (propose ran on fresh caches,
    // so offset == cacheOffset is not asserted; the hiddenStates call above
    // appended `targetHidden.dim(1)` rows).
    #expect(caches[0].offset == cacheOffset + targetHidden.dim(1))
}

// MARK: - End-to-end parity (real target + draft)

/// Minimal tokenizer — the parity trace carries raw prompt/output ids, so no
/// tokenization is ever performed.
private struct ParityStubTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private struct ParityStubTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        ParityStubTokenizer()
    }
}

private func cleanTargetDirectory() -> URL? {
    let dir = URL(
        fileURLWithPath: "/Users/owl/projects/tesseract/research/models/qwen3.8-27b-4bit-clean")
    return FileManager.default.fileExists(
        atPath: dir.appendingPathComponent("config.json").path) ? dir : nil
}

/// Full greedy DFlash2 generation through the real 4-bit Qwen3.8-27B target +
/// bf16 draft, compared against the Python reference trace
/// (`research/fixtures/trace_draftbf16.json`, produced by
/// `research/dump_reference.py` against the same clean checkpoint).
/// Exact token identity holds for a long prefix; a draft-side bf16 near-tie
/// legitimately flips late in the trace, so the assertion is prefix length +
/// acceptance volume, not identity (round-0 tensors are checked separately by
/// `testDFlash2Round0TensorParity`).
@Test(.serialized, .timeLimit(.minutes(15)))
func testDFlash2EndToEndParityWithPythonTrace() async throws {
    try await runDFlash2TraceParity(
        traceName: "trace_draftbf16.json", draftQuantization: nil)
}

/// Same trace parity with the production 4-bit draft (reference dump:
/// `dump_reference.py --draft-bits 4`).
@Test(.serialized, .timeLimit(.minutes(15)))
func testDFlash2EndToEndParityWithPythonTrace4BitDraft() async throws {
    try await runDFlash2TraceParity(
        traceName: "trace_draft4bit.json", draftQuantization: (groupSize: 64, bits: 4))
}

private func runDFlash2TraceParity(
    traceName: String,
    draftQuantization: (groupSize: Int, bits: Int)?
) async throws {
    // A 27B load per test: never let MLX's buffer cache accumulate across
    // tests in this process (51 GB footprints swapped a 48 GB machine).
    Memory.clearCache()
    defer { Memory.clearCache() }
    guard let fixtures = fixtureDirectory(), let draftDir = draftSnapshotDirectory(),
        let targetDir = cleanTargetDirectory()
    else {
        Issue.record("DFlash2 end-to-end parity inputs not present; skipping")
        return
    }
    let traceURL = fixtures.appendingPathComponent(traceName)
    guard FileManager.default.fileExists(atPath: traceURL.path) else {
        Issue.record("\(traceName) not present; skipping")
        return
    }

    let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: traceURL)) as! [String: Any]
    let promptIDs = (meta["prompt_ids"] as! [NSNumber]).map { $0.intValue }
    let blockSize = (meta["block_size"] as! NSNumber).intValue
    let trace = meta["trace"] as! [[String: Any]]
    let maxTokens = (trace.compactMap { $0["tokens"] as? [NSNumber] }.reduce(0) { $0 + $1.count })

    var expected = [Int]()
    for round in trace {
        expected.append(contentsOf: (round["tokens"] as! [NSNumber]).map { $0.intValue })
    }

    let context = try await LLMModelFactory.shared.load(
        from: targetDir, using: ParityStubTokenizerLoader())
    let draft = try loadDFlash2Draft(from: draftDir, quantization: draftQuantization)

    var parameters = GenerateParameters(maxTokens: maxTokens)
    parameters.temperature = 0
    var iterator = try DFlash2SpeculativeTokenIterator(
        input: LMInput(tokens: MLXArray(promptIDs.map { Int32($0) })),
        mainModel: context.model, drafter: draft,
        parameters: parameters, blockSize: blockSize)

    var produced = [Int]()
    while let token = iterator.next() {
        produced.append(token)
    }

    // Token-for-token identity is NOT attainable across the whole trace: the
    // reference and this port diverge on draft-side near-ties (the drafts
    // differ; every emitted token is still target-verified). With the round-2
    // kernel set (mma8 verify-width QMM) the first divergence moved from
    // token 45 to token 10 — measured 2026-08-21 as a DEAD TIE at the seam
    // (ids 279 vs 1092, logits 20.7500 == 20.7500; which one argmax picks is
    // sub-ULP evaluation-order happenstance, not a verify defect). So the
    // prefix gate is two-part: a sanity floor against gross pipeline
    // breakage, plus the principled gate — the seam's two candidates must be
    // near-tied in the TARGET's own logits (this is the property a broken
    // port would violate).
    let commonPrefix =
        zip(produced, expected).enumerated().first(where: { $0.element.0 != $0.element.1 })?.offset
        ?? min(produced.count, expected.count)
    #expect(
        produced.count == expected.count,
        "count: produced \(produced.count) vs expected \(expected.count)")
    #expect(
        commonPrefix >= 8,
        """
        common prefix \(commonPrefix) < 8 (\(traceName)) — gross pipeline breakage, \
        not a near-tie: produced \(produced.dropFirst(Swift.max(0, commonPrefix - 2)).prefix(5)) vs \
        expected \(expected.dropFirst(Swift.max(0, commonPrefix - 2)).prefix(5)) at the seam; \
        accepted \(iterator.acceptedCount)/\(iterator.proposedCount)
        """)
    if commonPrefix < 40, commonPrefix < produced.count {
        // Seam gate: forward the shared prefix through the target and compare
        // the two candidates' logits. Near-tie = within two bf16 ULPs at
        // logit scale ~20 (0.25); a real divergence shows a clear gap.
        let seamIDs = promptIDs + expected[0 ..< commonPrefix]
        var seamCache = try context.model.newCache(parameters: GenerateParameters())
        var start = 0
        var seamRow: MLXArray?
        while start < seamIDs.count {
            let end = Swift.min(seamIDs.count, start + 2048)
            let r = context.model(
                LMInput.Text(tokens: MLXArray(seamIDs[start ..< end].map { Int32($0) }))[
                    text: .newAxis],
                cache: seamCache, state: nil)
            seamRow = r.logits[0, -1, 0...].asType(.float32)
            start = end
        }
        if let seamRow {
            let expectedLogit = seamRow[expected[commonPrefix]].item(Float.self)
            let producedLogit = seamRow[produced[commonPrefix]].item(Float.self)
            #expect(
                abs(expectedLogit - producedLogit) < 0.25,
                """
                seam divergence at \(commonPrefix) is NOT a near-tie (\(traceName)): \
                expected \(expected[commonPrefix]) logit \(expectedLogit) vs \
                produced \(produced[commonPrefix]) logit \(producedLogit)
                """)
        }
    }
    #expect(
        iterator.acceptedCount >= 25,
        "accepted \(iterator.acceptedCount)/\(iterator.proposedCount) below floor")
}

/// Round-0 tensor-for-tensor parity: prefill hidden window, draft proposal,
/// and target verify pass against `research/fixtures/round0_draftbf16.safetensors`
/// (captured from the Python reference on the same checkpoints). This is the
/// load-bearing numeric check — the full trace above can legitimately diverge
/// late on a draft-side near-tie, but round 0 must match closely.
@Test(.serialized, .timeLimit(.minutes(15)))
func testDFlash2Round0TensorParity() async throws {
    Memory.clearCache()
    defer { Memory.clearCache() }
    guard let fixtures = fixtureDirectory(), let draftDir = draftSnapshotDirectory(),
        let targetDir = cleanTargetDirectory()
    else {
        Issue.record("DFlash2 round-0 parity inputs not present; skipping")
        return
    }
    let round0URL = fixtures.appendingPathComponent("round0_draftbf16.safetensors")
    let traceURL = fixtures.appendingPathComponent("trace_draftbf16.json")
    guard FileManager.default.fileExists(atPath: round0URL.path),
        FileManager.default.fileExists(atPath: traceURL.path)
    else {
        Issue.record("round0 fixture not present; skipping")
        return
    }
    let ref = try loadArrays(url: round0URL)
    let meta = try JSONSerialization.jsonObject(with: Data(contentsOf: traceURL)) as! [String: Any]
    let promptIDs = (meta["prompt_ids"] as! [NSNumber]).map { Int32($0.intValue) }
    let expectedFirst = meta["first_token"] as! Int
    let expectedAccepted = meta["accepted_round0"] as! Int

    let context = try await LLMModelFactory.shared.load(
        from: targetDir, using: ParityStubTokenizerLoader())
    let model = context.model
    let draft = try loadDFlash2Draft(from: draftDir)
    draft.bindDFlashTarget(model)

    func assertClose(_ got: MLXArray, _ expected: MLXArray, _ name: String, tol: Float = 0.06) {
        let diff = (got.asType(.float32) - expected.asType(.float32)).abs().max().item(Float.self)
        let scale = expected.asType(.float32).abs().max().item(Float.self)
        #expect(
            diff < tol * max(scale, 1),
            "\(name): max |diff| \(diff) vs scale \(scale) (rel \(diff / max(scale, 1e-6)))")
    }

    /// bf16 residual streams accumulate large absolute error on outlier
    /// channels by layer 61 (max |diff| ~38 on a 255-scale tensor, all of it
    /// noise the draft's norm+fc absorbs). The robust criterion is relative
    /// Frobenius distance; round-0 measured values: window 0.128, draft final
    /// hidden 0.040, draft logits 0.033.
    func assertRelL2(_ got: MLXArray, _ expected: MLXArray, _ name: String, tol: Float) {
        let r = expected.asType(.float32)
        let d = got.asType(.float32) - r
        let rel = (d.square().sum().sqrt() / r.square().sum().sqrt()).item(Float.self)
        #expect(rel < tol, "\(name): relL2 \(rel) vs tol \(tol)")
    }

    // ---- prefill (mirrors the iterator's chunked capture prefill) ----
    let debugDump = ProcessInfo.processInfo.environment["DFLASH2_DEBUG_DUMP"] != nil
    let promptTokens = MLXArray(promptIDs)
    let promptLength = promptIDs.count
    var cache = try model.newCache(parameters: GenerateParameters())
    var hiddenWindow: MLXArray? = nil
    var lastLogits: MLXArray? = nil
    var start = 0
    while start < promptLength {
        let remaining = promptLength - start
        let end = start + (remaining == 1 ? 1 : Swift.min(2048, remaining - 1))
        var chunkState = LMOutput.State()
        chunkState[dflash2CaptureLayerIdsKey] = draft.dflashTargetLayerIds
        let chunk = LMInput.Text(tokens: promptTokens[start ..< end])
        let result = model(chunk[text: .newAxis], cache: cache, state: chunkState)
        let captured = result.state?[dflash2CapturedHiddenStatesKey]!
        let chunkHidden = captured!.count == 1 ? captured![0] : concatenated(captured!, axis: -1)
        hiddenWindow =
            hiddenWindow.map { concatenated([$0, chunkHidden], axis: 1) } ?? chunkHidden
        lastLogits = result.logits
        start = end
    }
    let hidden = hiddenWindow!
    eval(hidden, lastLogits!)
    assertRelL2(hidden, ref["target_hidden_window"]!, "target_hidden_window", tol: 0.16)

    let firstToken = argMax(lastLogits![0..., -1, 0...], axis: -1).item(Int.self)
    #expect(firstToken == expectedFirst)

    // ---- draft proposal ----
    if debugDump { FileManager.default.createFile(atPath: "/tmp/dflash2_round0.marker", contents: nil) }
    print("[DFLASH2-DBG] prefill done; draft proposal next")
    let block = ref["block_ids"]!.asType(.int32)
    let finalHidden = draft.hiddenStates(
        block, targetHidden: hidden, cache: draft.makeDFlashContextCaches(), logitsStart: 1)
    let draftLogits = draft.computeLogits(finalHidden)
    let (draftTokens, _, _) = draft.dflashPropose(
        block, targetHidden: hidden, cache: draft.makeDFlashContextCaches(),
        temperature: 0, logitsStart: 1)
    eval(finalHidden, draftLogits, draftTokens)
    print("[DFLASH2-DBG] draft proposal done")
    assertRelL2(finalHidden, ref["draft_final_hidden"]!, "draft_final_hidden", tol: 0.06)
    assertRelL2(draftLogits, ref["draft_logits"]!, "draft_logits", tol: 0.06)
    #expect(draftTokens.asArray(Int32.self) == ref["draft_tokens"]!.asType(.int32).asArray(Int32.self))

    if debugDump {
        let url = URL(fileURLWithPath: "/tmp/swift_round0.safetensors")
        try? save(
            arrays: [
                "target_hidden_window": hidden,
                "draft_final_hidden": finalHidden,
                "draft_logits": draftLogits,
                "draft_tokens": draftTokens.asType(.int32),
            ], url: url)
    }

    // ---- target verify ----
    print("[DFLASH2-DBG] verify next")
    var verifyState = LMOutput.State()
    verifyState[dflash2CaptureLayerIdsKey] = draft.dflashTargetLayerIds
    let verifyInput = LMInput.Text(tokens: ref["verify_input"]!.asType(.int32).flattened())
    let verifyResult = model(verifyInput[text: .newAxis], cache: cache, state: verifyState)
    let verifyCaptured = verifyResult.state?[dflash2CapturedHiddenStatesKey]!
    let verifyHidden =
        verifyCaptured!.count == 1 ? verifyCaptured![0] : concatenated(verifyCaptured!, axis: -1)
    eval(verifyResult.logits, verifyHidden)
    assertClose(verifyHidden, ref["verify_hidden"]!, "verify_hidden")
    assertClose(verifyResult.logits, ref["verify_logits"]!, "verify_logits", tol: 0.10)

    let targetArgmax = argMax(verifyResult.logits, axis: -1).asType(.int32).asArray(Int32.self)
    #expect(targetArgmax == ref["verify_target_argmax"]!.asType(.int32).asArray(Int32.self))

    let draftList = draftTokens.asArray(Int32.self)
    var accepted = draftList.count
    for (i, d) in draftList.enumerated() where targetArgmax[i] != d {
        accepted = i
        break
    }
    #expect(accepted == expectedAccepted)
}
