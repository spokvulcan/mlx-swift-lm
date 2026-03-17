// InferenceBench — ClusteredKVCache benchmark CLI
// Measures generation tok/s and prefill tok/s comparing baseline vs clustered KV cache.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import Tokenizers

// MARK: - Configuration

let defaultModelPath = "models/z-lab_Qwen3.5-4B-PARO"
let defaultGenerateTokens = 100
let defaultIterations = 3
let generationContextLengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
let prefillContextLengths = [512, 1024, 2048, 4096, 8192, 16384]
let warmupContextLength = 64

// MARK: - CLI Argument Parsing

struct CLIArgs {
    var model: String = defaultModelPath
    var tokens: Int = defaultGenerateTokens
    var iterations: Int = defaultIterations
    var profile: Bool = false

    init(_ arguments: [String]) {
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--model":
                i += 1
                if i < arguments.count { model = arguments[i] }
            case "--tokens":
                i += 1
                if i < arguments.count { tokens = Int(arguments[i]) ?? defaultGenerateTokens }
            case "--iterations":
                i += 1
                if i < arguments.count { iterations = Int(arguments[i]) ?? defaultIterations }
            case "--profile":
                profile = true
            case "--help", "-h":
                printUsage()
                exit(0)
            default:
                break
            }
            i += 1
        }
    }

    func printUsage() {
        print(
            """
            Usage: InferenceBench [OPTIONS]

            Options:
              --model <path|id>     Model path or Hub ID (default: \(defaultModelPath))
              --tokens <n>          Tokens to generate per run (default: \(defaultGenerateTokens))
              --iterations <n>      Iterations per benchmark (default: \(defaultIterations))
              --profile             Enable per-operation profiling for ClusteredKVCache
              -h, --help            Show this help
            """)
    }
}

// MARK: - Statistics

struct BenchmarkStats {
    let mean: Double
    let median: Double
    let stdDev: Double
    let min: Double
    let max: Double

    init(values: [Double]) {
        precondition(!values.isEmpty)
        let sorted = values.sorted()
        self.min = sorted.first!
        self.max = sorted.last!
        let mean = values.reduce(0, +) / Double(values.count)
        self.mean = mean
        self.median = sorted[sorted.count / 2]
        let squaredDiffs = values.map { ($0 - mean) * ($0 - mean) }
        self.stdDev = sqrt(squaredDiffs.reduce(0, +) / Double(values.count))
    }
}

// MARK: - JSONL Result

struct BenchmarkResult: Codable {
    let commit: String
    let timestamp: String
    let model: String
    let scenario: String
    let context_length: Int
    let cache_type: String
    let iterations: Int
    let median_tps: Double
    let mean_tps: Double
    let min_tps: Double
    let max_tps: Double
    let stddev_tps: Double
    let tokens_generated: Int
    let peak_memory_mb: Int
    let kv_clusters: Int
    let kv_recent_window: Int
    let description: String
}

// MARK: - Helpers

func gitCommitShort() -> String {
    let process = Process()
    let pipe = Pipe()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    process.arguments = ["rev-parse", "--short", "HEAD"]
    process.standardOutput = pipe
    process.standardError = FileHandle.nullDevice
    try? process.run()
    process.waitUntilExit()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        ?? "unknown"
}

func isoTimestamp() -> String {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime]
    return f.string(from: Date())
}

func loadModel(_ modelSpec: String) async throws -> ModelContainer {
    let url = URL(fileURLWithPath: modelSpec)
    let isDirectory =
        FileManager.default.fileExists(atPath: modelSpec)
        && {
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: modelSpec, isDirectory: &isDir)
            return isDir.boolValue
        }()

    // Check for ParoQuant model (local directory only)
    if isDirectory, isParoQuantModel(directory: url) {
        print("Loading ParoQuant model: \(modelSpec) ...")
        let container = try await loadParoQuantModel(
            from: url,
            typeRegistry: LLMModelFactory.shared.typeRegistry
        )
        print("Model loaded (ParoQuant).")
        return container
    }

    let config: ModelConfiguration
    if isDirectory {
        config = ModelConfiguration(directory: url)
    } else {
        config = ModelConfiguration(id: modelSpec)
    }

    print("Loading model: \(config.name) ...")
    let container = try await LLMModelFactory.shared.loadContainer(
        configuration: config
    ) { progress in
        if progress.fractionCompleted < 1.0 {
            print(
                "  Downloading: \(String(format: "%.0f%%", progress.fractionCompleted * 100))",
                terminator: "\r")
            fflush(stdout)
        }
    }
    print("Model loaded.")
    return container
}

/// Create a synthetic LMInput of exact token count using repeating token IDs.
func makeSyntheticInput(tokenCount: Int, tokenizer: any Tokenizer) -> LMInput {
    let seedTokens = tokenizer.encode(text: " hello")
    var tokens = [Int]()
    while tokens.count < tokenCount {
        tokens.append(contentsOf: seedTokens)
    }
    tokens = Array(tokens.prefix(tokenCount))
    return LMInput(tokens: MLXArray(tokens))
}

/// Check if the cache array contains any ClusteredKVCache instances.
func cacheIsClustered(_ cache: [KVCache]) -> Bool {
    cache.contains(where: { $0 is ClusteredKVCacheProtocol })
}

// MARK: - JSONL Writer

struct JSONLWriter {
    let path: String

    init() {
        let dir = "experiments"
        try? FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)
        self.path = "\(dir)/results.jsonl"
    }

    func append(_ result: BenchmarkResult) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(result),
            let line = String(data: data, encoding: .utf8)
        else { return }
        let lineData = (line + "\n").data(using: .utf8)!

        if FileManager.default.fileExists(atPath: path) {
            guard let handle = FileHandle(forWritingAtPath: path) else { return }
            handle.seekToEndOfFile()
            handle.write(lineData)
            handle.closeFile()
        } else {
            FileManager.default.createFile(atPath: path, contents: lineData)
        }
    }
}

// MARK: - Table Printer

struct TablePrinter {
    struct Row {
        let context: Int
        let baseline: Double?
        let clustered: Double?
    }

    static func print(title: String, contextLabel: String, rows: [Row], iterations: Int) {
        Swift.print()
        Swift.print("\(title) (median of \(iterations) runs)")

        let w0 = 10
        let w1 = 10
        let w2 = 10
        let w3 = 9
        let line =
            "┌\(String(repeating: "─", count: w0))┬\(String(repeating: "─", count: w1))┬\(String(repeating: "─", count: w2))┬\(String(repeating: "─", count: w3))┐"
        let mid =
            "├\(String(repeating: "─", count: w0))┼\(String(repeating: "─", count: w1))┼\(String(repeating: "─", count: w2))┼\(String(repeating: "─", count: w3))┤"
        let bot =
            "└\(String(repeating: "─", count: w0))┴\(String(repeating: "─", count: w1))┴\(String(repeating: "─", count: w2))┴\(String(repeating: "─", count: w3))┘"

        Swift.print(line)
        Swift.print(
            "│\(pad(contextLabel, w0))│\(pad("Baseline", w1))│\(pad("Clustered", w2))│\(pad("Speedup", w3))│"
        )
        Swift.print(mid)

        for row in rows {
            let ctx = pad("\(row.context)", w0)
            let bl = row.baseline.map { pad(String(format: "%.1f", $0), w1) } ?? pad("ERR", w1)
            let cl = row.clustered.map { pad(String(format: "%.1f", $0), w2) } ?? pad("ERR", w2)
            let sp: String
            if let b = row.baseline, let c = row.clustered, b > 0 {
                sp = pad(String(format: "%.2fx", c / b), w3)
            } else {
                sp = pad("—", w3)
            }
            Swift.print("│\(ctx)│\(bl)│\(cl)│\(sp)│")
        }
        Swift.print(bot)
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        let padding = max(0, width - s.count - 1)
        return String(repeating: " ", count: padding) + s + " "
    }
}

// MARK: - Benchmark Runner

struct BenchmarkRunner {
    let container: ModelContainer
    let args: CLIArgs
    let writer = JSONLWriter()
    let commit = gitCommitShort()

    func run() async throws {
        let modelName = await container.configuration.name

        // Warm-up run
        print("\nWarm-up run...")
        let _ = try await runSingleGeneration(
            contextLength: warmupContextLength, clustered: false, maxTokens: 10)
        Memory.clearCache()
        print("Warm-up complete.\n")

        // Check if model supports clustered KV
        let supportsClustered = await checkClusteredSupport()
        if !supportsClustered {
            print(
                "⚠ Model does not create ClusteredKVCache — clustered results will use KVCacheSimple"
            )
        }

        // Generation benchmarks
        print("═══ Generation Throughput Benchmark ═══")
        var genRows = [TablePrinter.Row]()

        for ctx in generationContextLengths {
            var baselineTPS: Double? = nil
            var clusteredTPS: Double? = nil

            for clustered in [false, true] {
                let label = clustered ? "clustered" : "baseline"
                print("  ctx=\(ctx) \(label): ", terminator: "")
                fflush(stdout)

                var tpsValues = [Double]()
                var peakMem = 0
                var succeeded = true

                for iter in 1 ... args.iterations {
                    Memory.clearCache()
                    do {
                        let (tps, mem, text) = try await runSingleGeneration(
                            contextLength: ctx, clustered: clustered, maxTokens: args.tokens)
                        tpsValues.append(tps)
                        peakMem = max(peakMem, mem)
                        Swift.print("\(iter):\(String(format: "%.1f", tps)) ", terminator: "")
                        // Show generated text on first iteration for quality verification
                        if iter == 1 {
                            let preview = String(text.prefix(120)).replacingOccurrences(
                                of: "\n", with: "\\n")
                            Swift.print("\n    → \"\(preview)\"")
                        }
                        fflush(stdout)
                    } catch {
                        Swift.print("OOM/ERR at iter \(iter) ", terminator: "")
                        succeeded = false
                        break
                    }
                }
                Swift.print()

                if succeeded, !tpsValues.isEmpty {
                    let stats = BenchmarkStats(values: tpsValues)
                    if clustered { clusteredTPS = stats.median } else { baselineTPS = stats.median }

                    let result = BenchmarkResult(
                        commit: commit,
                        timestamp: isoTimestamp(),
                        model: modelName,
                        scenario: "generation",
                        context_length: ctx,
                        cache_type: label,
                        iterations: args.iterations,
                        median_tps: stats.median,
                        mean_tps: stats.mean,
                        min_tps: stats.min,
                        max_tps: stats.max,
                        stddev_tps: stats.stdDev,
                        tokens_generated: args.tokens,
                        peak_memory_mb: peakMem,
                        kv_clusters: clustered ? 256 : 0,
                        kv_recent_window: clustered ? 2048 : 0,
                        description: "Generation at \(ctx) ctx, \(label)"
                    )
                    writer.append(result)
                }
            }

            genRows.append(
                TablePrinter.Row(context: ctx, baseline: baselineTPS, clustered: clusteredTPS))
        }

        // Prefill benchmarks
        print("\n═══ Prefill Speed Benchmark ═══")
        var prefillRows = [TablePrinter.Row]()

        for ctx in prefillContextLengths {
            var baselineTPS: Double? = nil
            var clusteredTPS: Double? = nil

            for clustered in [false, true] {
                let label = clustered ? "clustered" : "baseline"
                print("  prompt=\(ctx) \(label): ", terminator: "")
                fflush(stdout)

                var tpsValues = [Double]()
                var peakMem = 0
                var succeeded = true

                for iter in 1 ... args.iterations {
                    Memory.clearCache()
                    do {
                        let (tps, mem) = try await runSinglePrefill(
                            promptLength: ctx, clustered: clustered)
                        tpsValues.append(tps)
                        peakMem = max(peakMem, mem)
                        Swift.print("\(iter):\(String(format: "%.1f", tps)) ", terminator: "")
                        fflush(stdout)
                    } catch {
                        Swift.print("OOM/ERR at iter \(iter) ", terminator: "")
                        succeeded = false
                        break
                    }
                }
                Swift.print()

                if succeeded, !tpsValues.isEmpty {
                    let stats = BenchmarkStats(values: tpsValues)
                    if clustered { clusteredTPS = stats.median } else { baselineTPS = stats.median }

                    let result = BenchmarkResult(
                        commit: commit,
                        timestamp: isoTimestamp(),
                        model: modelName,
                        scenario: "prefill",
                        context_length: ctx,
                        cache_type: label,
                        iterations: args.iterations,
                        median_tps: stats.median,
                        mean_tps: stats.mean,
                        min_tps: stats.min,
                        max_tps: stats.max,
                        stddev_tps: stats.stdDev,
                        tokens_generated: 1,
                        peak_memory_mb: peakMem,
                        kv_clusters: clustered ? 256 : 0,
                        kv_recent_window: clustered ? 2048 : 0,
                        description: "Prefill at \(ctx) prompt, \(label)"
                    )
                    writer.append(result)
                }
            }

            prefillRows.append(
                TablePrinter.Row(context: ctx, baseline: baselineTPS, clustered: clusteredTPS))
        }

        // Print summary tables
        TablePrinter.print(
            title: "Generation Throughput (tok/s)",
            contextLabel: "Context",
            rows: genRows,
            iterations: args.iterations
        )
        TablePrinter.print(
            title: "Prefill Speed (tok/s)",
            contextLabel: "Prompt",
            rows: prefillRows,
            iterations: args.iterations
        )

        print("\nResults appended to \(writer.path)")
    }

    // MARK: - Single-run measurement

    func runSingleGeneration(contextLength: Int, clustered: Bool, maxTokens: Int) async throws
        -> (tps: Double, peakMemMB: Int, generatedText: String)
    {
        try await container.perform { context in
            let tokenizer = context.tokenizer
            let model = context.model

            let input = makeSyntheticInput(tokenCount: contextLength, tokenizer: tokenizer)

            let params = GenerateParameters(
                maxTokens: maxTokens,
                clusteredKV: clustered,
                temperature: 0
            )
            let cache = model.newCache(parameters: params)

            Memory.peakMemory = 0

            // Prefill (inside TokenIterator init)
            var iterator = try TokenIterator(
                input: input, model: model, cache: cache, parameters: params)

            // Generation loop — collect token IDs for quality verification
            let genStart = Date.timeIntervalSinceReferenceDate
            var tokensGenerated = 0
            var tokenIds = [Int]()
            while let token = iterator.next() {
                tokenIds.append(token)
                tokensGenerated += 1
            }
            Stream().synchronize()
            let genEnd = Date.timeIntervalSinceReferenceDate

            let genTime = genEnd - genStart
            let tps = genTime > 0 ? Double(tokensGenerated) / genTime : 0
            let peakMB = Int(Memory.peakMemory / (1024 * 1024))
            let text = tokenizer.decode(tokens: tokenIds)
            return (tps, peakMB, text)
        }
    }

    func runSinglePrefill(promptLength: Int, clustered: Bool) async throws
        -> (tps: Double, peakMemMB: Int)
    {
        try await container.perform { context in
            let tokenizer = context.tokenizer
            let model = context.model

            let input = makeSyntheticInput(tokenCount: promptLength, tokenizer: tokenizer)

            let params = GenerateParameters(
                maxTokens: 1,
                clusteredKV: clustered,
                temperature: 0
            )
            let cache = model.newCache(parameters: params)

            Memory.peakMemory = 0

            let iterator = try TokenIterator(
                input: input, model: model, cache: cache, parameters: params)
            Stream().synchronize()

            let prefillTime = iterator.promptPrefillTime
            let tps = prefillTime > 0 ? Double(promptLength) / prefillTime : 0
            let peakMB = Int(Memory.peakMemory / (1024 * 1024))
            return (tps, peakMB)
        }
    }

    func checkClusteredSupport() async -> Bool {
        await container.perform { context in
            let model = context.model
            let params = GenerateParameters(
                maxTokens: 1, clusteredKV: true, temperature: 0)
            let cache = model.newCache(parameters: params)
            let result = cacheIsClustered(cache)
            return result
        }
    }

    // MARK: - Profiled Run

    func runProfiled() async throws {
        print("\n═══ ClusteredKVCache Per-Operation Profiling ═══")
        print(
            "NOTE: eval-fence timing breaks kernel fusion — times are inflated, use for relative comparison only.\n"
        )

        for ctx in generationContextLengths {
            print("Profiling ctx=\(ctx), \(args.tokens) tokens...")

            let report = try await runProfiledGeneration(
                contextLength: ctx, maxTokens: args.tokens)

            if let report {
                print(
                    "\nClusteredKVCache Per-Operation Breakdown (\(args.tokens) steps at ctx=\(ctx))"
                )
                print(report)
            } else {
                print("  (below cluster threshold or not clustered)")
            }
            print()
            Memory.clearCache()
        }
    }

    func runProfiledGeneration(contextLength: Int, maxTokens: Int) async throws -> String? {
        try await container.perform { context in
            let tokenizer = context.tokenizer
            let model = context.model

            let input = makeSyntheticInput(tokenCount: contextLength, tokenizer: tokenizer)

            let params = GenerateParameters(
                maxTokens: maxTokens,
                clusteredKV: true,
                profileClustered: true,
                temperature: 0
            )
            let cache = model.newCache(parameters: params)

            // Prefill
            var iterator = try TokenIterator(
                input: input, model: model, cache: cache, parameters: params)

            // Generation loop
            var tokensGenerated = 0
            while iterator.next() != nil {
                tokensGenerated += 1
            }
            Stream().synchronize()

            // Extract profiler reports from clustered cache layers
            let reports = cache.compactMap { ($0 as? ClusteredKVCache)?.profiler?.report() }
            guard let firstReport = reports.first else { return nil as String? }

            // Return report from first clustered layer (representative)
            return firstReport
        }
    }
}

// MARK: - Entry Point

let args = CLIArgs(CommandLine.arguments)

do {
    let container = try await loadModel(args.model)
    let runner = BenchmarkRunner(container: container, args: args)

    if args.profile {
        try await runner.runProfiled()
    } else {
        try await runner.run()
    }
} catch {
    print("Error: \(error)")
    exit(1)
}
