// Copyright © 2026 Apple Inc.
//
// The Qwen3.5 vision wrapper runs the text model's own engine. Text rotates
// exactly as the text model does, a pass with image rows rotates with
// interleaved M-RoPE, text after an image rotates by the image's rope delta,
// and DFlash2 speculates over prompts with images. Tiny random-weight model,
// no downloads.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXLLM
@testable import MLXVLM

private let tinyVisionConfiguration = """
    {
        "model_type": "qwen3_5_vl",
        "image_token_id": 500,
        "video_token_id": 501,
        "vision_start_token_id": 502,
        "vision_end_token_id": 503,
        "vocab_size": 512,
        "text_config": {
            "model_type": "qwen3_5",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 512,
            "full_attention_interval": 2,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 4096,
            "rope_parameters": {
                "type": "default",
                "mrope_section": [8, 4, 4],
                "rope_theta": 100000.0,
                "partial_rotary_factor": 1.0
            }
        },
        "vision_config": {
            "model_type": "qwen3_vl",
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 64,
            "out_hidden_size": 64,
            "num_heads": 2,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 64
        }
    }
    """

/// The state slot the vision wrapper carries its rope delta in.
private let ropeDeltasKey = LMOutput.Key<MLXArray>("qwen35.ropeDeltas")

private func makeTinyVisionModel(seed: UInt64 = 1) throws -> MLXVLM.Qwen35 {
    let configuration = try JSONDecoder().decode(
        MLXVLM.Qwen35Configuration.self, from: Data(tinyVisionConfiguration.utf8))
    return withRandomState(MLXRandom.RandomState(seed: seed)) { MLXVLM.Qwen35(configuration) }
}

/// Plain-text tokens away from the special ids (500...503), `[1, count]`.
private func textTokens(_ count: Int, seed: Int = 0) -> MLXArray {
    MLXArray((0 ..< count).map { Int32(($0 * 13 + 7 + seed) % 480) }).expandedDimensions(axis: 0)
}

/// One image: grid (1, 4, 4), merge 2, so four placeholder rows after the
/// vision-start marker.
private func imagePrompt(before: Int, after: Int) -> (
    tokens: MLXArray, image: LMInput.ProcessedImage
) {
    let run = MLXArray([Int32(502), 500, 500, 500, 500]).expandedDimensions(axis: 0)
    let tokens = concatenated(
        [textTokens(before), run, textTokens(after, seed: 5)], axis: 1)
    let image = LMInput.ProcessedImage(
        pixels: MLXRandom.normal([16, 3 * 2 * 16 * 16]), frames: [THW(1, 4, 4)])
    return (tokens, image)
}

private func greedy(maxTokens: Int) -> GenerateParameters {
    var parameters = GenerateParameters(maxTokens: maxTokens)
    parameters.temperature = 0
    return parameters
}

private func drain(_ iterator: inout some TokenIteratorProtocol) -> [Int] {
    var tokens: [Int] = []
    while let token = iterator.next() { tokens.append(token) }
    return tokens
}

/// Proposes the target's own greedy continuation, read off `reference`,
/// with every `missEvery`-th draft wrong, so rounds both accept and reject.
private final class OracleDrafter: Module, DFlash2DrafterModel {
    let reference: [Int]
    let promptLength: Int
    let missEvery: Int
    private var drafted = 0

    init(reference: [Int], promptLength: Int, missEvery: Int) {
        self.reference = reference
        self.promptLength = promptLength
        self.missEvery = missEvery
        super.init()
    }

    var blockSize: Int { 4 }
    var maskTokenId: Int { 511 }
    var targetLayerIds: [Int] { [1, 2] }
    var targetLayerCount: Int { 4 }
    var contextWindow: Int { 16 }

    func makeState() -> DFlash2DrafterState { DFlash2DrafterState(contextCaches: []) }

    func propose(
        block: MLXArray, targetHidden: MLXArray, contextPosition: Int, validRows: MLXArray,
        temperature: Float, target: any DFlash2TargetModel, state: inout DFlash2DrafterState
    ) -> DFlash2Proposal {
        // The anchor sits right after the committed context rows; generated
        // token k sits at row `promptLength + k`.
        let anchor = contextPosition + validRows.item(Int.self)
        let width = block.dim(1) - 1
        let tokens: [Int32] = (1 ... width).map { offset in
            let index = anchor + offset - promptLength
            let truth = index < reference.count ? reference[index] : 0
            drafted += 1
            let miss = missEvery > 0 && drafted % missEvery == 0
            return Int32(miss ? (truth + 1) % 480 : truth)
        }
        let drafts = MLXArray(tokens).expandedDimensions(axis: 0)
        return DFlash2Proposal(tokens: drafts, candidates: drafts.expandedDimensions(axis: -1))
    }
}

@Suite(.serialized)
struct Qwen35VisionEngineTests {

    /// A text prompt through the wrapper is the text model's own forward:
    /// prefill and decode logits match the hosted text model bitwise.
    @Test
    func textRunsTheTextModelBitwise() throws {
        let model = try makeTinyVisionModel()
        let prompt = textTokens(24)

        let wrapperCache = try model.newCache(parameters: nil)
        guard
            case .logits(let prefilled) = try model.prepare(
                LMInput(text: .init(tokens: prompt)), cache: wrapperCache, state: nil,
                prefill: .init())
        else {
            Issue.record("expected logits")
            return
        }
        let textCache = try model.languageModel.newCache(parameters: nil)
        let direct = model.languageModel(LMInput.Text(tokens: prompt), cache: textCache, state: nil)
        #expect(arrayEqual(prefilled.logits, direct.logits).item(Bool.self))

        var state = prefilled.state
        for step in 0 ..< 6 {
            let token = MLXArray([Int32(40 + step)]).expandedDimensions(axis: 0)
            let viaWrapper = model(LMInput.Text(tokens: token), cache: wrapperCache, state: state)
            state = viaWrapper.state
            let viaText = model.languageModel(
                LMInput.Text(tokens: token), cache: textCache, state: nil)
            #expect(arrayEqual(viaWrapper.logits, viaText.logits).item(Bool.self), "step \(step)")
        }
    }

    /// Text after an image rotates at its cache row plus the image's (negative)
    /// delta: the shifted plain rope puts the rows where explicit, axis-equal
    /// positions do.
    @Test
    func shiftedPositionsMatchExplicitTextPositions() throws {
        let model = try makeTinyVisionModel(seed: 3)
        let inner = model.languageModel.model
        let prefix = textTokens(20)
        let continuation = textTokens(12, seed: 4)

        func continued(_ positions: Qwen35RotaryPositions) throws -> MLXArray {
            let cache = try model.newCache(parameters: nil)
            _ = inner.forwardLayers(prefix, cache: cache, captureLayers: [])
            return inner.forwardLayers(
                continuation, cache: cache, positions: positions, captureLayers: []
            ).hidden
        }
        let rows = MLXArray(Int32(17) ..< Int32(29)).reshaped(1, 1, 12)
        let shifted = try continued(.shifted(-3))
        let explicit = try continued(.multimodal(broadcast(rows, to: [3, 1, 12])))
        let unshifted = try continued(.unshifted)
        #expect(allClose(shifted, explicit, rtol: 1e-4, atol: 1e-4).item(Bool.self))
        #expect(!allClose(shifted, unshifted, rtol: 1e-4, atol: 1e-4).item(Bool.self))
    }

    /// The engine's M-RoPE rotates as the vision model's own rotary did.
    @Test
    func multimodalRopeMatchesTheVisionRotary() throws {
        try withRandomState(MLXRandom.RandomState(seed: 11)) {
            let sections = [8, 4, 4]
            let reference = Qwen35Language.RotaryEmbedding(
                dim: 32, base: 100_000, mropeSection: sections)
            let rope = Qwen35MultimodalRoPE(
                dimensions: 32, base: 100_000, scale: 1, sections: sections)
            let q = MLXRandom.normal([1, 4, 10, 32])
            let k = MLXRandom.normal([1, 2, 10, 32])
            let positions = MLXRandom.randInt(Int32(0) ..< Int32(50), [3, 1, 10])

            let (cosines, sines) = reference(x: q, positionIds: positions)
            let (expectedQ, expectedK) = Qwen35Language.applyMultimodalRotaryPosEmb(
                q: q, k: k, cos: cosines, sin: sines)
            #expect(allClose(rope(q, positions: positions), expectedQ, atol: 1e-5).item(Bool.self))
            #expect(allClose(rope(k, positions: positions), expectedK, atol: 1e-5).item(Bool.self))
        }
    }

    @Test
    func pairsAsADFlash2Target() throws {
        let model = try makeTinyVisionModel()
        let target: any LanguageModel = model
        #expect(target is any DFlash2MediaTargetModel)
        #expect(model.dflash2LayerCount == 4)
        #expect(model.kvHeads.count == 4)
        #expect(model.dflash2SupportsCache(try model.newCache(parameters: nil)))
    }

    /// DFlash2 over a prompt with an image decodes exactly what the target
    /// decodes greedily, both when the iterator hands the image to the target
    /// and when the caller prefilled it and passes the rope delta.
    @Test
    func dflash2OverAnImagePromptMatchesGreedyDecoding() throws {
        try withRandomState(MLXRandom.RandomState(seed: 23)) {
            let model = try makeTinyVisionModel(seed: 5)
            let (tokens, image) = imagePrompt(before: 10, after: 8)
            let promptLength = tokens.dim(1)
            let parameters = greedy(maxTokens: 24)

            var plain = try TokenIterator(
                input: LMInput(text: .init(tokens: tokens), image: image), model: model,
                parameters: parameters)
            let reference = drain(&plain)
            #expect(reference.count == 24)

            // The target prefills through the image itself.
            var handed = try DFlash2SpeculativeTokenIterator(
                input: LMInput(text: .init(tokens: tokens), image: image), mainModel: model,
                drafter: OracleDrafter(
                    reference: reference, promptLength: promptLength, missEvery: 3),
                parameters: parameters)
            #expect(drain(&handed) == reference)
            #expect(handed.acceptedCount > 0)
            #expect(handed.positionDelta != 0)

            // The caller prefilled the image and hands over text and delta.
            let imageEnd = 10 + 5
            let cache = try model.newCache(parameters: nil)
            guard
                case .logits(let prefix) = try model.prepare(
                    LMInput(text: .init(tokens: tokens[0..., ..<imageEnd]), image: image),
                    cache: cache, state: nil, prefill: .init())
            else {
                Issue.record("expected logits")
                return
            }
            eval(cache)
            let delta = try #require(prefix.state?[ropeDeltasKey]).item(Int.self)
            #expect(delta == handed.positionDelta)
            var continued = try DFlash2SpeculativeTokenIterator(
                input: LMInput(text: .init(tokens: tokens)), mainModel: model,
                drafter: OracleDrafter(
                    reference: reference, promptLength: promptLength, missEvery: 2),
                mainCache: cache, prefilledPrefixTokens: imageEnd, positionDelta: delta,
                parameters: parameters)
            #expect(drain(&continued) == reference)
        }
    }

    /// No text after the last image leaves nothing to speculate over: the
    /// iterator refuses before touching the cache.
    @Test
    func imagePromptWithNoTextAfterItIsRefused() throws {
        try withRandomState(MLXRandom.RandomState(seed: 29)) {
            let model = try makeTinyVisionModel()
            let (tokens, image) = imagePrompt(before: 6, after: 0)
            let cache = try model.newCache(parameters: nil)
            #expect(throws: DFlash2SpeculationError.promptTooShort) {
                _ = try DFlash2SpeculativeTokenIterator(
                    input: LMInput(text: .init(tokens: tokens), image: image), mainModel: model,
                    drafter: OracleDrafter(reference: [], promptLength: 0, missEvery: 0),
                    mainCache: cache, parameters: greedy(maxTokens: 4))
            }
            #expect(cache.allSatisfy { $0.offset == 0 })
        }
    }
}
