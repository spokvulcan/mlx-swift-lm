import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

/// Diagnostic probes for MLX multi-output ("sibling") array graphs: which drop
/// paths break the sibling reference cycle and release the inputs the graph
/// holds, and which leave the inputs resident.
///
/// Every measurement waits for the GPU and lets Metal's completion handlers
/// release their temporaries first, so a delta is persistent retention, not
/// command-buffer lag.
final class SiblingCycleTests: XCTestCase {

    private let rows = 2048
    private let cols = 2048

    /// 8 MiB of bf16, evaluated.
    private func bigEvaluated() -> MLXArray {
        let w = MLXArray.ones([rows, cols], dtype: .bfloat16)
        eval(w)
        return w
    }

    private func settled() -> Int {
        Stream.gpu.synchronize()
        usleep(150_000)
        Memory.clearCache()
        return Memory.activeMemory
    }

    private func assertReleased(
        _ base: Int, _ message: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        let after = settled()
        XCTAssertEqual(
            after - base, 0, "\(message): \(after - base) bytes retained", file: file, line: line)
    }

    // MARK: - Eager graphs

    func testEagerSplitDroppedUnevaluatedReleasesInputs() {
        let base = settled()
        autoreleasepool {
            let w = bigEvaluated()
            let parts = (w * 2).split(parts: 2)
            XCTAssertEqual(parts.count, 2)
        }
        assertReleased(base, "eager split")
    }

    func testEagerTwoOutputKernelReleasesInputs() {
        let x = bigEvaluated()
        let base = settled()
        autoreleasepool {
            let constant = bigEvaluated()
            let weight = MLXArray.ones([cols], dtype: .bfloat16)
            let (h, out) = rmsNormResidual(x, constant, weight: weight, eps: 1e-6)
            eval(h, out)
        }
        assertReleased(base, "eager two-output kernel")
        _ = x
    }

    /// In-place `update` (mlx_array_set) on every sibling of an unevaluated
    /// multi-output graph: the old descriptors are released by assignment, so the
    /// destructor's cycle check never runs.
    func testUpdateInPlaceOnSiblingsReleasesInputs() {
        XCTExpectFailure(
            "mlx: assigning over a multi-output sibling skips the cycle break in ~array; "
                + "fixed by ml-explore/mlx#4453 (2026-09-11), not yet in the pinned mlx"
        )
        let base = settled()
        autoreleasepool {
            let w = bigEvaluated()
            let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4)
            let holders: [MLXArray] = [wq, scales] + (biases.map { [$0] } ?? [])
            for h in holders {
                h._updateInternal(MLXArray(Float(1)))
            }
            eval(holders)
        }
        assertReleased(base, "in-place update on siblings")
    }

    // MARK: - Compiled graphs with a captured constant

    func testCompiledSplitOfConstantReleasedOnErase() {
        XCTExpectFailure(
            "mlx: erasing a compiled function whose tape splits a captured constant keeps the "
                + "constant alive (ml-explore/mlx#3932, open)"
        )
        let x = MLXArray.ones([1, cols], dtype: .bfloat16)
        eval(x)
        let base = settled()
        autoreleasepool {
            let constant = bigEvaluated()
            var f: (([MLXArray]) -> [MLXArray])? = MLX.compile { (inputs: [MLXArray]) in
                let parts = (inputs[0] * constant).split(parts: 2, axis: 1)
                return [parts[0] + parts[1]]
            }
            let out = f!([x])
            eval(out)
            f = nil
        }
        assertReleased(base, "compiled split of a captured constant")
        _ = x
    }

    func testCompiledSingleOutputOfConstantReleasedOnErase() {
        let x = MLXArray.ones([1, cols], dtype: .bfloat16)
        eval(x)
        let base = settled()
        autoreleasepool {
            let constant = bigEvaluated()
            var f: (([MLXArray]) -> [MLXArray])? = MLX.compile { (inputs: [MLXArray]) in
                [(inputs[0] * constant).sum(axis: 1)]
            }
            let out = f!([x])
            eval(out)
            f = nil
        }
        assertReleased(base, "compiled single-output use of a captured constant")
        _ = x
    }

    func testCompiledTwoOutputKernelOnConstantReleasedOnErase() {
        let x = bigEvaluated()
        let base = settled()
        autoreleasepool {
            let constant = bigEvaluated()
            let weight = MLXArray.ones([cols], dtype: .bfloat16)
            eval(weight)
            var f: (([MLXArray]) -> [MLXArray])? = MLX.compile { (inputs: [MLXArray]) in
                let (h, out) = rmsNormResidual(inputs[0], constant, weight: weight, eps: 1e-6)
                return [h, out]
            }
            let out = f!([x])
            eval(out)
            f = nil
        }
        assertReleased(base, "compiled two-output kernel on a captured constant")
        _ = x
    }

    func testCompiledTwoOutputKernelOneOutputUnusedReleasedOnErase() {
        let x = bigEvaluated()
        let base = settled()
        autoreleasepool {
            let constant = bigEvaluated()
            let weight = MLXArray.ones([cols], dtype: .bfloat16)
            eval(weight)
            var f: (([MLXArray]) -> [MLXArray])? = MLX.compile { (inputs: [MLXArray]) in
                let (h, _) = rmsNormResidual(inputs[0], constant, weight: weight, eps: 1e-6)
                return [h]
            }
            let out = f!([x])
            eval(out)
            f = nil
        }
        assertReleased(base, "compiled two-output kernel with one output unused")
        _ = x
    }

    func testCompiledRealGraphDroppedUnevaluatedReleasesInputs() {
        let x = bigEvaluated()
        let base = settled()
        autoreleasepool {
            let constant = bigEvaluated()
            let weight = MLXArray.ones([cols], dtype: .bfloat16)
            eval(weight)
            var f: (([MLXArray]) -> [MLXArray])? = MLX.compile { (inputs: [MLXArray]) in
                let (h, out) = rmsNormResidual(inputs[0], constant, weight: weight, eps: 1e-6)
                return [h, out]
            }
            let warm = f!([x])
            eval(warm)
            let dropped = f!([x])
            XCTAssertEqual(dropped.count, 2)
            f = nil
        }
        assertReleased(base, "unevaluated compiled outputs")
        _ = x
    }

    // MARK: - CompiledTrace

    private final class Plain {
        let constant: MLXArray
        init(constant: MLXArray) { self.constant = constant }
    }

    private final class StateModule: Module {
        let constant: MLXArray
        init(constant: MLXArray) {
            self.constant = constant
            super.init()
        }
    }

    private final class Owner: Module {
        let weight: MLXArray
        let plain: Plain
        init(weight: MLXArray, constant: MLXArray) {
            self.weight = weight
            self.plain = Plain(constant: constant)
            super.init()
        }
    }

    /// The decode-segment shape: module weights as compile inputs, an array a
    /// plain (non-Module) holder owns read as a constant, a two-output kernel.
    func testCompiledTraceWithPlainHeldConstantReleasedOnInvalidate() {
        let x = bigEvaluated()
        let base = settled()
        autoreleasepool {
            let owner = Owner(
                weight: MLXArray.ones([cols], dtype: .bfloat16), constant: bigEvaluated())
            eval(owner.weight)
            XCTAssertEqual(
                owner.innerState().count, 1, "the plain holder must stay invisible to reflection")
            let trace = CompiledTrace<Owner> { owner, inputs in
                let (h, out) = rmsNormResidual(
                    inputs[0], owner.plain.constant, weight: owner.weight, eps: 1e-6)
                return [h, out]
            }
            let out = trace(owner, [x])
            eval(out)
            trace.invalidate()
        }
        assertReleased(base, "CompiledTrace reading a plain-held constant")
        _ = x
    }

    /// The mitigation: the plain holder's array declared as compile state
    /// through a wrapper module, so the tape holds a tracer instead of the array.
    func testCompiledTraceWithPlainHeldArrayAsStateReleasedOnInvalidate() {
        let x = bigEvaluated()
        let base = settled()
        autoreleasepool {
            let owner = Owner(
                weight: MLXArray.ones([cols], dtype: .bfloat16), constant: bigEvaluated())
            eval(owner.weight)
            let trace = CompiledTrace<Owner>(state: {
                [$0, StateModule(constant: $0.plain.constant)]
            }) {
                owner, inputs in
                let (h, out) = rmsNormResidual(
                    inputs[0], owner.plain.constant, weight: owner.weight, eps: 1e-6)
                return [h, out]
            }
            let out = trace(owner, [x])
            eval(out)
            trace.invalidate()
        }
        assertReleased(base, "CompiledTrace with the plain-held array as state")
        _ = x
    }
}
