// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// A module that can fold projections sharing one input (`gate`/`up`,
/// `q`/`k`/`v`) into a single stacked matmul.
///
/// Stacking concatenates weights along the output axis, so every output row
/// keeps its own accumulation order and quantization groups: results are
/// bitwise identical, only the launch count drops. Speculative decoding
/// feels the difference most, where a round is a few hundred small launches.
protocol SameInputProjectionStacking: Module {
    /// Stack once. Returns false, changing nothing, when the projections are
    /// not quantized layers one stacked layer can reproduce.
    func stackSameInputProjections() -> Bool
}

/// Stack every foldable projection group in `model`; returns the count.
public func stackSameInputProjections(in model: Module) -> Int {
    var stacked = 0
    for module in model.modules() {
        if let stacking = module as? SameInputProjectionStacking,
            stacking.stackSameInputProjections()
        {
            stacked += 1
        }
    }
    if stacked > 0 {
        model.invalidateCompiledTraces()
    }
    return stacked
}

/// `module` when it is exactly a `QuantizedLinear`. Subclasses transform
/// their input before the matmul (a ParoQuant rotation, say), which folding
/// their weights into a plain layer would drop.
func plainQuantizedLinear(_ module: Module) -> QuantizedLinear? {
    type(of: module) == QuantizedLinear.self ? module as? QuantizedLinear : nil
}

/// `module` when it is exactly a `HadamardQuantizedLinear`.
private func rotatedQuantizedLinear(_ module: Module) -> HadamardQuantizedLinear? {
    type(of: module) == HadamardQuantizedLinear.self ? module as? HadamardQuantizedLinear : nil
}

/// One layer computing `layers` on a shared input, or nil when no single
/// stacked layer reproduces them.
///
/// Plain `QuantizedLinear`s stack when their quantization matches. Rotated
/// `HadamardQuantizedLinear`s stack when they also share the rotation and
/// the sign vector (compared on the arrays, here, once): the stacked layer
/// rotates the shared input once and runs one packed matmul, where the
/// originals each rotated it again. Any other subclass is left alone.
func stackedSameInputProjection(_ layers: [Linear]) -> QuantizedLinear? {
    let plain = layers.compactMap(plainQuantizedLinear)
    if plain.count == layers.count {
        return stackedQuantizedLinear(plain)
    }
    let rotated = layers.compactMap(rotatedQuantizedLinear)
    guard rotated.count == layers.count, let first = rotated.first,
        rotated.allSatisfy({
            $0.rotation == first.rotation && arrayEqual($0.signs, first.signs).item(Bool.self)
        }),
        let packed = stackedQuantization(rotated)
    else { return nil }
    return HadamardQuantizedLinear(
        weight: packed.weight, bias: nil, scales: packed.scales, biases: packed.biases,
        signs: first.signs, groupSize: first.groupSize, bits: first.bits, mode: first.mode,
        rotation: first.rotation)
}

/// One layer computing `layers` on a shared input, or nil when they differ in
/// quantization or carry a bias.
func stackedQuantizedLinear(_ layers: [QuantizedLinear]) -> QuantizedLinear? {
    guard let first = layers.first, let packed = stackedQuantization(layers) else { return nil }
    return QuantizedLinear(
        weight: packed.weight, bias: nil, scales: packed.scales, biases: packed.biases,
        groupSize: first.groupSize, bits: first.bits, mode: first.mode)
}

/// The packed arrays of `layers` concatenated along the output axis, or nil
/// when the layers differ in quantization or carry a bias.
private func stackedQuantization(_ layers: [QuantizedLinear]) -> (
    weight: MLXArray, scales: MLXArray, biases: MLXArray?
)? {
    guard let first = layers.first,
        layers.allSatisfy({
            $0.bias == nil && $0.groupSize == first.groupSize && $0.bits == first.bits
                && $0.mode == first.mode
        })
    else { return nil }
    let quantBiases = layers.compactMap(\.biases)
    let biases: MLXArray?
    switch quantBiases.count {
    case layers.count: biases = concatenated(quantBiases, axis: 0)
    case 0: biases = nil
    default: return nil
    }
    let weight = concatenated(layers.map(\.weight), axis: 0)
    let scales = concatenated(layers.map(\.scales), axis: 0)
    eval(weight, scales, biases ?? weight)
    return (weight, scales, biases)
}

extension Module {
    /// Replace the folded originals with placeholders so their weights free.
    /// Registered `@ModuleInfo` properties only change through `update`.
    func releaseStackedProjections(_ keys: [String]) {
        update(
            modules: ModuleChildren.unflattened(
                keys.map { ($0, Linear(weight: MLXArray.zeros([1, 1]))) }))
    }
}
