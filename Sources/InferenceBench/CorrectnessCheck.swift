import Foundation
import MLX
import MLXLMCommon
import Tokenizers

struct CorrectnessCheck {
    let container: ModelContainer

    func run() async throws {
        print("\n  Correctness check:")

        // Generate with greedy decode (deterministic)
        let output = try await generate(prompt: "The capital of France is", maxTokens: 30)
        print("    Prompt: \"The capital of France is\"")
        print("    Output: \"\(output)\"")

        let coherent = output.lowercased().contains("paris")
        print("    Contains 'Paris': \(coherent ? "YES" : "NO — WARNING: model may be degraded")")

        // Check determinism — two greedy runs should produce identical output
        let output2 = try await generate(prompt: "The capital of France is", maxTokens: 30)
        let deterministic = output == output2
        print("    Deterministic: \(deterministic ? "YES" : "NO — WARNING: non-deterministic")")

        if !coherent {
            print(
                "    ⚠ Pre-rotation may have introduced too much quantization error")
        }
    }

    private func generate(prompt: String, maxTokens: Int) async throws -> String {
        try await container.perform { context in
            let tokenizer = context.tokenizer
            let model = context.model

            let promptTokens = tokenizer.encode(text: prompt)
            let input = LMInput(tokens: MLXArray(promptTokens))
            let params = GenerateParameters(maxTokens: maxTokens, temperature: 0)
            let cache = model.newCache(parameters: params)

            var iterator = try TokenIterator(
                input: input, model: model, cache: cache, parameters: params)
            var tokens = [Int]()
            while let token = iterator.next() {
                tokens.append(token)
            }
            Stream().synchronize()
            return tokenizer.decode(tokens: tokens)
        }
    }
}
