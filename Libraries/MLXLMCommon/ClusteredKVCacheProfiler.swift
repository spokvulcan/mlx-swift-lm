// Copyright © 2026 Tesseract Agent

import Foundation
import MLX

/// Per-operation profiler for ClusteredKVCache.
///
/// When attached to a `ClusteredKVCache`, inserts `eval()` + `Stream().synchronize()`
/// barriers around key operations to measure their GPU time individually.
///
/// **Important:** Eval-fence timing breaks MLX kernel fusion, so measured times are
/// inflated. Use for *relative* comparison between operations only.
public final class ClusteredKVCacheProfiler {

    public enum Operation: String, CaseIterable {
        case kvStore = "kv_store"
        case clusterAssign = "cluster_assign"
        case recluster = "recluster"
        case initClusters = "init_clusters"
        case attentionConcat = "attn_concat"
        case attentionSdpa = "attn_sdpa"
    }

    /// Accumulated wall-clock durations per operation across all steps.
    private var timings: [Operation: [TimeInterval]] = {
        var d = [Operation: [TimeInterval]]()
        for op in Operation.allCases { d[op] = [] }
        return d
    }()

    public init() {}

    /// Measure a single operation by forcing evaluation of its outputs.
    ///
    /// `body()` returns the arrays produced by the operation. After the body,
    /// `eval(arrays)` + `Stream().synchronize()` flushes the GPU pipeline so
    /// wall-clock time reflects actual GPU work.
    public func measure(_ op: Operation, _ body: () -> [MLXArray]) {
        let start = Date.timeIntervalSinceReferenceDate
        let arrays = body()
        eval(arrays)
        Stream().synchronize()
        let elapsed = Date.timeIntervalSinceReferenceDate - start
        timings[op, default: []].append(elapsed)
    }

    /// Formatted breakdown table suitable for terminal output.
    public func report() -> String {
        let w0 = 18
        let w1 = 10
        let w2 = 10
        let w3 = 10
        let w4 = 7
        let top =
            "\u{250C}\(bar(w0))\u{252C}\(bar(w1))\u{252C}\(bar(w2))\u{252C}\(bar(w3))\u{252C}\(bar(w4))\u{2510}"
        let mid =
            "\u{251C}\(bar(w0))\u{253C}\(bar(w1))\u{253C}\(bar(w2))\u{253C}\(bar(w3))\u{253C}\(bar(w4))\u{2524}"
        let bot =
            "\u{2514}\(bar(w0))\u{2534}\(bar(w1))\u{2534}\(bar(w2))\u{2534}\(bar(w3))\u{2534}\(bar(w4))\u{2518}"

        var lines = [String]()
        lines.append(top)
        lines.append(
            "\u{2502}\(pad("Operation", w0))\u{2502}\(pad("Mean(ms)", w1))\u{2502}\(pad("Med(ms)", w2))\u{2502}\(pad("Max(ms)", w3))\u{2502}\(pad("Count", w4))\u{2502}"
        )
        lines.append(mid)

        for op in Operation.allCases {
            let samples = timings[op] ?? []
            let count = samples.count
            let name = op.rawValue

            if count == 0 {
                lines.append(
                    "\u{2502}\(pad(name, w0))\u{2502}\(pad("---", w1))\u{2502}\(pad("---", w2))\u{2502}\(pad("---", w3))\u{2502}\(pad("\(count)", w4))\u{2502}"
                )
            } else {
                let sorted = samples.sorted()
                let mean = samples.reduce(0, +) / Double(count) * 1000
                let median = sorted[count / 2] * 1000
                let maxVal = sorted.last! * 1000

                lines.append(
                    "\u{2502}\(pad(name, w0))\u{2502}\(pad(f(mean), w1))\u{2502}\(pad(f(median), w2))\u{2502}\(pad(f(maxVal), w3))\u{2502}\(pad("\(count)", w4))\u{2502}"
                )
            }
        }

        lines.append(bot)
        return lines.joined(separator: "\n")
    }

    /// Reset all accumulated timings.
    public func reset() {
        for op in Operation.allCases {
            timings[op] = []
        }
    }

    // MARK: - Formatting Helpers

    private func bar(_ width: Int) -> String {
        String(repeating: "\u{2500}", count: width)
    }

    private func pad(_ s: String, _ width: Int) -> String {
        let padding = max(0, width - s.count - 1)
        return String(repeating: " ", count: padding) + s + " "
    }

    private func f(_ ms: Double) -> String {
        String(format: "%.2f", ms)
    }
}
