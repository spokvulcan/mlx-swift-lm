import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct BenchmarkResult {
    let generationTPSValues: [Double]
    let prefillTPSValues: [Double]
    let peakMemoryMB: Double
    let tokensGenerated: Int
    let promptTokens: Int

    var meanGenerationTPS: Double {
        generationTPSValues.reduce(0, +) / Double(generationTPSValues.count)
    }

    var meanPrefillTPS: Double {
        prefillTPSValues.reduce(0, +) / Double(prefillTPSValues.count)
    }

    var medianGenerationTPS: Double {
        let sorted = generationTPSValues.sorted()
        let n = sorted.count
        if n % 2 == 0 {
            return (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }
        return sorted[n / 2]
    }

    var stddevGenerationTPS: Double {
        let mean = meanGenerationTPS
        let variance =
            generationTPSValues.map { ($0 - mean) * ($0 - mean) }.reduce(0, +)
            / Double(generationTPSValues.count)
        return variance.squareRoot()
    }

    var minGenerationTPS: Double { generationTPSValues.min() ?? 0 }
    var maxGenerationTPS: Double { generationTPSValues.max() ?? 0 }
}

struct BenchmarkRunner {
    let container: ModelContainer
    let warmupRuns: Int
    let measuredRuns: Int

    func run(scenario: BenchmarkScenario) async throws -> BenchmarkResult {
        print("  [\(scenario.name)] ", terminator: "")

        var generationTPS = [Double]()
        var prefillTPS = [Double]()
        var peakMemoryMB: Double = 0
        var lastTokensGenerated = 0
        var lastPromptTokens = 0

        let totalRuns = warmupRuns + measuredRuns

        for i in 0 ..< totalRuns {
            let isWarmup = i < warmupRuns

            // Clear GPU cache between runs
            Memory.clearCache()

            let (genTPS, preTPS, tokensGen, promptTok) = try await runSingleIteration(
                scenario: scenario
            )

            if isWarmup {
                print("W", terminator: "")
            } else {
                generationTPS.append(genTPS)
                prefillTPS.append(preTPS)
                lastTokensGenerated = tokensGen
                lastPromptTokens = promptTok
                print(".", terminator: "")
            }

            // Track peak memory after each run
            let peakMB = Double(Memory.peakMemory) / (1024 * 1024)
            peakMemoryMB = max(peakMemoryMB, peakMB)

            // Sync GPU between iterations
            Stream().synchronize()
            fflush(stdout)
        }

        print(" done")

        return BenchmarkResult(
            generationTPSValues: generationTPS,
            prefillTPSValues: prefillTPS,
            peakMemoryMB: peakMemoryMB,
            tokensGenerated: lastTokensGenerated,
            promptTokens: lastPromptTokens
        )
    }

    private func runSingleIteration(
        scenario: BenchmarkScenario
    ) async throws -> (genTPS: Double, prefillTPS: Double, tokensGen: Int, promptTok: Int) {
        try await container.perform { context in
            let tokenizer = context.tokenizer
            let model = context.model

            // Tokenize prompt
            let promptTokens = tokenizer.encode(text: scenario.prompt)
            let input = LMInput(tokens: MLXArray(promptTokens))

            // Create fresh cache for each run
            let cache = model.newCache(parameters: scenario.parameters)

            // Create iterator (this also does prefill)
            let prefillStart = Date.timeIntervalSinceReferenceDate
            var iterator = try TokenIterator(
                input: input,
                model: model,
                cache: cache,
                parameters: scenario.parameters
            )
            let prefillEnd = Date.timeIntervalSinceReferenceDate
            let prefillTime = prefillEnd - prefillStart

            // Generate tokens
            let genStart = Date.timeIntervalSinceReferenceDate
            var generatedTokens = [Int]()

            while let token = iterator.next() {
                generatedTokens.append(token)
            }

            Stream().synchronize()
            let genEnd = Date.timeIntervalSinceReferenceDate
            let genTime = genEnd - genStart

            let genTPS = genTime > 0 ? Double(generatedTokens.count) / genTime : 0
            let prefillTPS = prefillTime > 0 ? Double(promptTokens.count) / prefillTime : 0

            return (genTPS, prefillTPS, generatedTokens.count, promptTokens.count)
        }
    }
}
