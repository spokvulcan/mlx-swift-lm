import Foundation
import MLX
import MLXLMCommon
import Tokenizers

struct DemoGenerate {
    let container: ModelContainer

    func run(prompts: [String], maxTokens: Int, temperature: Float) async throws {
        for (i, prompt) in prompts.enumerated() {
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("Prompt \(i + 1): \"\(prompt)\"")
            print("────────────────────────────────────────────────────────────────────────────")

            let (output, tokensGenerated, prefillTime, genTime) = try await generate(
                prompt: prompt, maxTokens: maxTokens, temperature: temperature)

            print(output)
            print("────────────────────────────────────────────────────────────────────────────")
            let genTPS = genTime > 0 ? Double(tokensGenerated) / genTime : 0
            let prefillTPS =
                prefillTime > 0 ? Double(prompt.split(separator: " ").count) / prefillTime : 0
            print(
                "  \(tokensGenerated) tokens | \(String(format: "%.1f", genTPS)) tok/s | prefill \(String(format: "%.2f", prefillTime))s | gen \(String(format: "%.2f", genTime))s"
            )
            print()
        }
    }

    private func generate(
        prompt: String, maxTokens: Int, temperature: Float
    ) async throws -> (String, Int, TimeInterval, TimeInterval) {
        try await container.perform { context in
            let tokenizer = context.tokenizer
            let model = context.model

            let promptTokens = tokenizer.encode(text: prompt)
            let input = LMInput(tokens: MLXArray(promptTokens))
            let params = GenerateParameters(
                maxTokens: maxTokens,
                temperature: temperature,
                topP: temperature > 0 ? 0.9 : 1.0,
                topK: temperature > 0 ? 20 : 0,
                minP: temperature > 0 ? 0.02 : 0.0
            )
            let cache = model.newCache(parameters: params)

            let prefillStart = Date.timeIntervalSinceReferenceDate
            var iterator = try TokenIterator(
                input: input, model: model, cache: cache, parameters: params)
            let prefillEnd = Date.timeIntervalSinceReferenceDate

            let genStart = Date.timeIntervalSinceReferenceDate
            var tokens = [Int]()
            // Build up text incrementally for stop detection
            var eosTokenIds: Set<Int> = []
            if let eos = tokenizer.eosTokenId { eosTokenIds.insert(eos) }

            while let token = iterator.next() {
                tokens.append(token)
                if eosTokenIds.contains(token) { break }
            }
            Stream().synchronize()
            let genEnd = Date.timeIntervalSinceReferenceDate

            let text = tokenizer.decode(tokens: tokens)
            return (text, tokens.count, prefillEnd - prefillStart, genEnd - genStart)
        }
    }
}
