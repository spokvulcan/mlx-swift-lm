import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Tokenizers

// MARK: - CLI Argument Parsing

struct CLIArgs {
    var modelPath: String = "models/z-lab_Qwen3.5-4B-PARO"
    var tokens: Int = 500
    var warmup: Int = 1
    var iterations: Int = 3
    var scenario: String = "all"
    var output: String = "experiments/results.jsonl"
}

func parseArgs() -> CLIArgs {
    var args = CLIArgs()
    var i = 1
    let argv = CommandLine.arguments
    while i < argv.count {
        switch argv[i] {
        case "--model-path":
            i += 1
            args.modelPath = argv[i]
        case "--tokens":
            i += 1
            args.tokens = Int(argv[i])!
        case "--warmup":
            i += 1
            args.warmup = Int(argv[i])!
        case "--iterations":
            i += 1
            args.iterations = Int(argv[i])!
        case "--scenario":
            i += 1
            args.scenario = argv[i]
        case "--output":
            i += 1
            args.output = argv[i]
        default:
            print("Unknown argument: \(argv[i])")
        }
        i += 1
    }
    return args
}

// MARK: - Main

let args = parseArgs()
let modelDirectory = URL(fileURLWithPath: args.modelPath)

print("╔══════════════════════════════════════════════════════╗")
print("║           InferenceBench — mlx-swift-lm             ║")
print("╚══════════════════════════════════════════════════════╝")
print()
print("Model:      \(modelDirectory.lastPathComponent)")
print("Tokens:     \(args.tokens)")
print("Warmup:     \(args.warmup)")
print("Iterations: \(args.iterations)")
print("Scenario:   \(args.scenario)")
print("Output:     \(args.output)")
print()

// Load model
print("Loading model...")
let loadStart = Date()

let isParo = isParoQuantModel(directory: modelDirectory)
let container: ModelContainer

if isParo {
    print("  Detected ParoQuant model — using custom loader")
    container = try await loadParoQuantModel(
        from: modelDirectory,
        typeRegistry: LLMTypeRegistry.shared
    )
} else {
    print("  Using standard LLM loader")
    container = try await LLMModelFactory.shared.loadContainer(
        configuration: ModelConfiguration(directory: modelDirectory)
    )
}

let loadTime = Date().timeIntervalSince(loadStart)
print("  Model loaded in \(String(format: "%.1f", loadTime))s")
print()

// Select scenarios
let allScenarios = BenchmarkScenarios.all(maxTokens: args.tokens)
let scenarios: [BenchmarkScenario]
if args.scenario == "all" {
    scenarios = allScenarios
} else {
    scenarios = allScenarios.filter { $0.name == args.scenario }
    if scenarios.isEmpty {
        print("Unknown scenario: \(args.scenario)")
        print("Available: \(allScenarios.map(\.name).joined(separator: ", "))")
        exit(1)
    }
}

// Run benchmarks
let runner = BenchmarkRunner(
    container: container,
    warmupRuns: args.warmup,
    measuredRuns: args.iterations
)

let resultLogger = ResultLogger(outputPath: args.output)

print("Running \(scenarios.count) scenario(s)...")
print(String(repeating: "─", count: 72))

for scenario in scenarios {
    let result = try await runner.run(scenario: scenario)

    resultLogger.log(result: result, scenario: scenario)

    let genTPS = String(format: "%.1f", result.meanGenerationTPS)
    let prefillTPS = String(format: "%.1f", result.meanPrefillTPS)
    let stddev = String(format: "%.1f", result.stddevGenerationTPS)
    let peakMB = String(format: "%.0f", result.peakMemoryMB)

    print("  \(scenario.name.padding(toLength: 24, withPad: " ", startingAt: 0))", terminator: "")
    print("gen: \(genTPS) tok/s (±\(stddev))", terminator: "  ")
    print("prefill: \(prefillTPS) tok/s", terminator: "  ")
    print("peak: \(peakMB) MB")
}

print(String(repeating: "─", count: 72))
print("Done. Results written to \(args.output)")
