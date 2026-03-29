import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Testing

private let modelPath =
    ProcessInfo.processInfo.environment["BENCHMARK_MODEL_PATH"]
    ?? "/Users/owl/projects/mlx-swift-lm/models/mlx-community_Qwen3.5-4B-MLX-8bit"

private let contextSizes = [4096, 16384]
private let generateTokenCount = 50

@Suite(.serialized)
struct DecodeBenchmarks {

    @Test(.timeLimit(.minutes(5)))
    func decodeBenchmark() async throws {
        let modelURL = URL(fileURLWithPath: modelPath)
        let config = ModelConfiguration(directory: modelURL)

        // Load model
        let modelContext = try await LLMModelFactory.shared.load(
            configuration: config
        ) { progress in
            if progress.fractionCompleted < 1.0 {
                print(
                    "Loading: \(String(format: "%.0f", progress.fractionCompleted * 100))%")
            }
        }

        print("Model loaded successfully")

        var results: [(context: Int, tokPerSec: Double)] = []

        for contextSize in contextSizes {
            let tokPerSec = try runDecodeAtContextSize(
                contextSize: contextSize,
                model: modelContext.model,
                generateCount: generateTokenCount
            )
            results.append((contextSize, tokPerSec))
            print(
                "\(contextSize / 1024)K context: \(String(format: "%.1f", tokPerSec)) tok/s"
            )
            Memory.clearCache()
        }

        let average = results.map(\.tokPerSec).reduce(0, +) / Double(results.count)
        print("METRIC: \(String(format: "%.2f", average))")
    }

    private func runDecodeAtContextSize(
        contextSize: Int,
        model: any LanguageModel,
        generateCount: Int
    ) throws -> Double {
        // Create a prompt of the desired context size using repeating token IDs.
        // Token ID 220 (" ") is a safe, common token for most tokenizers.
        let promptTokens = [Int](repeating: 220, count: contextSize)
        let input = LMInput(tokens: MLXArray(promptTokens).reshaped(1, contextSize))

        var parameters = GenerateParameters()
        parameters.temperature = 0.0
        parameters.maxTokens = generateCount

        let cache = model.newCache(parameters: parameters)

        // Prefill: process the prompt to fill the KV cache
        let prefillStart = CFAbsoluteTimeGetCurrent()
        var output = model(input.text.tokens, cache: cache)
        eval(output)
        let prefillTime = CFAbsoluteTimeGetCurrent() - prefillStart
        print(
            "  Prefill \(contextSize) tokens: \(String(format: "%.1f", prefillTime))s (\(String(format: "%.0f", Double(contextSize) / prefillTime)) tok/s)"
        )

        // Decode: generate tokens one at a time, measuring only decode phase
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var generatedCount = 0

        for _ in 0 ..< generateCount {
            let lastToken = argMax(output[0..., (-1)..., 0...], axis: -1)
            eval(lastToken)
            output = model(lastToken.reshaped(1, 1), cache: cache)
            eval(output)
            generatedCount += 1
        }

        let decodeTime = CFAbsoluteTimeGetCurrent() - decodeStart
        let tokPerSec = Double(generatedCount) / decodeTime

        return tokPerSec
    }
}
