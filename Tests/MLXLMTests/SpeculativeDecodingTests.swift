// Copyright © 2025 Apple Inc.

import Foundation
import MLX
import MLXLLM
import MLXNN
@testable import MLXLMCommon
import Testing

struct SpeculativeDecodingTests {

    let processor: any UserInputProcessor
    let mainContext: ModelContext
    let draftContext: ModelContext

    init() {
        let processor = TestInputProcessor()
        let modelConfig = Gemma3TextConfiguration(
            modelType: "text",
            hiddenSize: 64, hiddenLayers: 8, intermediateSize: 64,
            attentionHeads: 4, headDim: 64,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 4,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000,
            ropeTraditional: false, queryPreAttnScalar: 256,
            slidingWindow: 512, slidingWindowPattern: 6,
            maxPositionEmbeddings: 32768
        )

        let mainModel = Gemma3TextModel(modelConfig)
        let mainContext = ModelContext(
            configuration: processor.configuration,
            model: mainModel,
            processor: processor,
            tokenizer: processor.tokenizer
        )

        let draftModel = Gemma3TextModel(modelConfig)
        let draftContext = ModelContext(
            configuration: processor.configuration,
            model: draftModel,
            processor: processor,
            tokenizer: processor.tokenizer
        )

        eval(mainModel, draftModel)
        self.processor = processor
        self.mainContext = mainContext
        self.draftContext = draftContext
    }

    @Test(arguments: [2, 8, 48], [false, true])
    func `Speculative decoding matches default token generation`(
        numDraftTokens: Int,
        withLogitProcessor: Bool
    ) async throws {
        let input = UserInput(prompt: "Input text")
        let modelInput = try await processor.prepare(input: input)
        let parameters = GenerateParameters(
            maxTokens: 32,
            temperature: 0.0,  // Use greedy decoding for deterministic output
            repetitionPenalty: withLogitProcessor ? 1.5 : nil,
            presencePenalty: withLogitProcessor ? 0.5 : nil,
            frequencyPenalty: withLogitProcessor ? 0.2 : nil,
        )

        var normalTokens: [Int] = []
        for await generation in try generateTokens(
            input: modelInput, parameters: parameters, context: mainContext
        ) {
            if let token = generation.token { normalTokens.append(token) }
        }

        var speculativeTokens: [Int] = []
        for await generation in try generateTokens(
            input: modelInput, parameters: parameters, context: mainContext,
            draftModel: draftContext.model, numDraftTokens: numDraftTokens
        ) {
            if let token = generation.token { speculativeTokens.append(token) }
        }

        #expect(!normalTokens.isEmpty)
        #expect(!speculativeTokens.isEmpty)
        #expect(normalTokens == speculativeTokens)
    }

    @Test
    func speculativePrefillConfiguresTriAttentionProtectionOnMainAndDraftCaches() throws {
        let configuration = TriAttentionConfiguration(
            enabled: true,
            budgetTokens: 4,
            calibrationArtifactIdentity: nil,
            implementationVersion: .v1,
            prefixProtectionMode: .protectAllPrefill
        )
        let parameters = GenerateParameters(
            checkpointBaseOffset: 11,
            triAttention: configuration
        )
        let mainCache = TrimmableTriAttentionTestCache(configuration: configuration)
        let draftCache = TrimmableTriAttentionTestCache(configuration: configuration)
        let input = LMInput(tokens: MLXArray([Int32(1), 2, 3, 4, 5, 6, 7]))

        _ = try SpeculativeTokenIterator(
            input: input,
            mainModel: FakeSpeculativeModel(),
            draftModel: FakeSpeculativeModel(),
            mainCache: [mainCache],
            draftCache: [draftCache],
            parameters: parameters,
            numDraftTokens: 2
        )

        #expect(mainCache.protectedPrefixOffset == 18)
        #expect(draftCache.protectedPrefixOffset == 18)
    }
}

private final class FakeSpeculativeModel: Module, LanguageModel {
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        LMOutput(logits: MLXArray.zeros([1, 1, 1], dtype: .float32))
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        MLXArray.zeros([1, 1, 1], dtype: .float32)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }
}

private final class TrimmableTriAttentionTestCache: BaseKVCache, TriAttentionRuntimeCache {
    var configuration: TriAttentionConfiguration
    var runtimeState: TriAttentionQwen35RuntimeState?
    var retainedPositions: MLXArray?
    var retainedTokenCount: Int { offset }
    var protectedPrefixOffset: Int?

    init(configuration: TriAttentionConfiguration) {
        self.configuration = configuration
        self.runtimeState = nil
        self.retainedPositions = nil
        self.protectedPrefixOffset = nil
        super.init()
    }

    override var isTrimmable: Bool { true }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        offset += keys.dim(-2)
        return (keys, values)
    }

    func dequantizedRetainedKeysForRuntime() -> MLXArray? { nil }

    func applyKeepIndices(_ keepIndices: MLXArray) {}

    func attachRuntimeState(_ restoreContext: TriAttentionSnapshotRestoreContext?) {
        guard
            let restoreContext,
            configuration == restoreContext.expectedConfiguration
        else {
            return
        }
        runtimeState = restoreContext.runtimeState
    }

    func configureProtectedPrefixOffset(_ protectedPrefixOffset: Int?) {
        self.protectedPrefixOffset = protectedPrefixOffset
    }

    override func copy() -> any KVCache {
        let copy = TrimmableTriAttentionTestCache(configuration: configuration)
        copy.offset = offset
        copy.runtimeState = runtimeState
        copy.retainedPositions = retainedPositions
        copy.protectedPrefixOffset = protectedPrefixOffset
        return copy
    }
}
