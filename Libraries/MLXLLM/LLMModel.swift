// Copyright © 2024 Apple Inc.

import MLX
import MLXLMCommon

/// Marker protocol for LLMModels
public protocol LLMModel: LanguageModel, LoRAModel {

    /// Models can implement this is they need a custom `MessageGenerator`.
    ///
    /// The default implementation returns `DefaultMessageGenerator`.
    func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator
}

extension LLMModel {

    /// Default prepare step for ``LLMModel``.
    ///
    /// This will evaluate the prompt in chunks until there is a small number of
    /// tokens left to feed into the `TokenIterator`.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512
        var y = input.text

        // Prepare the prompt in chunks if larger than the prefill size
        while y.tokens.size > prefillStepSize {
            let input = y[.newAxis, ..<prefillStepSize]
            _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
            eval(cache)
            Memory.clearCache()
            y = y[prefillStepSize...]
        }

        return .tokens(y)
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }

    public func prepareWithCheckpoints(
        _ input: LMInput, cache: [KVCache], windowSize: Int?,
        checkpointAtOffsets: Set<Int>, checkpointBaseOffset: Int
    ) throws -> (PrepareResult, [HybridCacheSnapshot]) {
        guard !checkpointAtOffsets.isEmpty else {
            let result = try prepare(input, cache: cache, windowSize: windowSize)
            return (result, [])
        }

        let prefillStepSize = windowSize ?? 512
        var y = input.text

        let (_, snapshots) = try HybridCacheSnapshot.chunkedPrefill(
            totalTokens: y.tokens.size,
            prefillStepSize: prefillStepSize,
            checkpointAtOffsets: checkpointAtOffsets,
            checkpointBaseOffset: checkpointBaseOffset,
            cache: cache
        ) { chunkSize in
            let chunk = y[.newAxis, ..<chunkSize]
            _ = self(chunk, cache: cache.isEmpty ? nil : cache, state: nil)
            eval(cache)
            y = y[chunkSize...]
        }

        return (.tokens(y), snapshots)
    }
}
