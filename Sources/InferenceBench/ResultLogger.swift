import Foundation

struct ResultLogger {
    let outputPath: String

    func log(result: BenchmarkResult, scenario: BenchmarkScenario) {
        let commitHash = gitCommitHash()
        let timestamp = ISO8601DateFormatter().string(from: Date())

        let entry: [String: Any] = [
            "timestamp": timestamp,
            "commit": commitHash,
            "scenario": scenario.name,
            "description": scenario.description,
            "generation_tps": round(result.meanGenerationTPS * 10) / 10,
            "prefill_tps": round(result.meanPrefillTPS * 10) / 10,
            "peak_memory_mb": round(result.peakMemoryMB),
            "tokens_generated": result.tokensGenerated,
            "prompt_tokens": result.promptTokens,
            "iterations": result.generationTPSValues.count,
            "stddev_tps": round(result.stddevGenerationTPS * 10) / 10,
            "min_tps": round(result.minGenerationTPS * 10) / 10,
            "max_tps": round(result.maxGenerationTPS * 10) / 10,
            "median_tps": round(result.medianGenerationTPS * 10) / 10,
        ]

        guard
            let jsonData = try? JSONSerialization.data(
                withJSONObject: entry, options: [.sortedKeys])
        else {
            print("  Warning: failed to serialize result for \(scenario.name)")
            return
        }

        let jsonLine = String(data: jsonData, encoding: .utf8)! + "\n"

        // Ensure output directory exists
        let outputURL = URL(fileURLWithPath: outputPath)
        let dir = outputURL.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        // Append to file
        if FileManager.default.fileExists(atPath: outputPath) {
            if let handle = FileHandle(forWritingAtPath: outputPath) {
                handle.seekToEndOfFile()
                handle.write(jsonLine.data(using: .utf8)!)
                handle.closeFile()
            }
        } else {
            try? jsonLine.write(toFile: outputPath, atomically: true, encoding: .utf8)
        }
    }

    private func gitCommitHash() -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["rev-parse", "--short", "HEAD"]
        process.currentDirectoryURL = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice

        do {
            try process.run()
            process.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8)?.trimmingCharacters(
                in: .whitespacesAndNewlines) ?? "unknown"
        } catch {
            return "unknown"
        }
    }
}
