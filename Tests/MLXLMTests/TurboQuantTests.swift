// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import Testing
import XCTest

@testable import MLXLLM
@testable import MLXLMCommon

// MARK: - Codebook Tests

@Suite("TurboQuant Codebook")
struct TurboQuantCodebookTests {

    @Test func codebookGeneration_3bit() {
        let cb = TurboQuantCodebook.codebook(dim: 128, bits: 3)
        #expect(cb.count == 8)
        let vals = cb.asArray(Float.self)
        for i in 0 ..< vals.count - 1 {
            #expect(vals[i] <= vals[i + 1], "Codebook not sorted")
        }
    }

    @Test func codebookGeneration_4bit() {
        let cb = TurboQuantCodebook.codebook(dim: 128, bits: 4)
        #expect(cb.count == 16)
        let vals = cb.asArray(Float.self)
        for v in vals {
            #expect(v >= -1.0 && v <= 1.0, "Centroid \(v) out of range")
        }
    }

    @Test func boundaryCount() {
        let b = TurboQuantCodebook.boundaries(dim: 128, bits: 3)
        #expect(b.count == 7)  // 2^3 - 1 boundaries for 8 centroids
    }

    @Test func codebookDeterminism() {
        let cb1 = TurboQuantCodebook.codebook(dim: 64, bits: 3)
        let cb2 = TurboQuantCodebook.codebook(dim: 64, bits: 3)
        let v1 = cb1.asArray(Float.self)
        let v2 = cb2.asArray(Float.self)
        #expect(v1 == v2, "Codebook should be deterministic")
    }
}

// MARK: - Rotation Tests

@Suite("TurboQuant Rotation")
struct TurboQuantRotationTests {

    @Test func rotationOrthogonality() {
        let dim = 64
        let R = TurboQuantRotation.rotationMatrix(dim: dim, seed: 42)
        #expect(R.shape == [dim, dim])
        let RRt = matmul(R, R.transposed())
        let identity = MLXArray.eye(dim)
        let diff = MLX.abs(RRt - identity)
        let maxDiff = diff.max().item(Float.self)
        // MLX QR is f32 internally; Householder orthogonality error at d=64
        // is ~1e-3. The dense path's matched-norm scales absorb this.
        #expect(maxDiff < 2e-3, "R @ R^T differs from I by \(maxDiff)")
    }

    @Test func rotationDeterminism() {
        let R1 = TurboQuantRotation.rotationMatrix(dim: 32, seed: 123)
        let R2 = TurboQuantRotation.rotationMatrix(dim: 32, seed: 123)
        let diff = MLX.abs(R1 - R2).max().item(Float.self)
        #expect(diff < 1e-6, "Same seed should produce same rotation")
    }
}

// MARK: - Bit Packing Tests

@Suite("TurboQuant Bit Packing")
struct TurboQuantPackingTests {

    @Test func packedWidth() {
        #expect(TurboQuantPacking.packedWidth(count: 128, bits: 4) == 16)
        #expect(TurboQuantPacking.packedWidth(count: 128, bits: 3) == 12)
        #expect(TurboQuantPacking.packedWidth(count: 128, bits: 1) == 4)
    }

    @Test func packUnpack_3bit() {
        let indices = MLXArray([0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4] as [UInt32]).reshaped([1, 12])
        let packed = TurboQuantPacking.packLowBit(indices, bits: 3)
        let unpacked = TurboQuantPacking.unpackLowBit(packed, bits: 3, count: 12)
        let orig = indices.asArray(UInt32.self)
        let result = unpacked.asArray(UInt32.self)
        #expect(orig == result, "3-bit pack/unpack round-trip failed")
    }

    @Test func packUnpack_1bit() {
        let indices = MLXArray([0, 1, 0, 1, 1, 0, 1, 0] as [UInt32]).reshaped([1, 8])
        let packed = TurboQuantPacking.packLowBit(indices, bits: 1)
        let unpacked = TurboQuantPacking.unpackLowBit(packed, bits: 1, count: 8)
        let orig = indices.asArray(UInt32.self)
        let result = unpacked.asArray(UInt32.self)
        #expect(orig == result, "1-bit pack/unpack round-trip failed")
    }

    @Test func packUnpack_4bit() {
        let indices = MLXArray([0, 5, 10, 15, 3, 7, 11, 14] as [UInt32]).reshaped([1, 8])
        let packed = TurboQuantPacking.packLowBit(indices, bits: 4)
        let unpacked = TurboQuantPacking.unpackLowBit(packed, bits: 4, count: 8)
        let orig = indices.asArray(UInt32.self)
        let result = unpacked.asArray(UInt32.self)
        #expect(orig == result, "4-bit pack/unpack round-trip failed")
    }
}

// MARK: - MSE Codec Tests

@Suite("TurboQuant MSE Codec")
struct TurboQuantMSECodecTests {

    @Test func encodeDecodeRoundTrip_4bit() {
        let codec = MSECodec(dim: 32, bits: 4, seed: 42)
        let vectors = MLXRandom.normal([1, 2, 4, 32])
        eval(vectors)

        let state = codec.encode(vectors)
        let decoded = codec.decode(state)

        #expect(state.tokenCount == 4)
        #expect(state.dim == 32)
        #expect(state.bits == 4)

        let mse = ((vectors - decoded) * (vectors - decoded)).mean().item(Float.self)
        #expect(mse < 0.5, "4-bit MSE too high: \(mse)")
    }

    @Test func encodeDecodeRoundTrip_3bit() {
        let codec = MSECodec(dim: 32, bits: 3, seed: 42)
        let vectors = MLXRandom.normal([1, 2, 4, 32])
        eval(vectors)

        let state = codec.encode(vectors)
        let decoded = codec.decode(state)

        let mse = ((vectors - decoded) * (vectors - decoded)).mean().item(Float.self)
        #expect(mse < 1.0, "3-bit MSE too high: \(mse)")
    }

    @Test func cosineSimilarity_4bit() {
        let codec = MSECodec(dim: 64, bits: 4, seed: 42)
        let vectors = MLXRandom.normal([1, 1, 8, 64])
        eval(vectors)

        let decoded = codec.decode(codec.encode(vectors))
        let dot = (vectors * decoded).sum(axis: -1)
        let normOrig = sqrt((vectors * vectors).sum(axis: -1))
        let normDec = sqrt((decoded * decoded).sum(axis: -1))
        let cosSim = dot / (normOrig * normDec + 1e-8)
        let avgCosSim = cosSim.mean().item(Float.self)

        #expect(avgCosSim > 0.95, "4-bit cosine similarity too low: \(avgCosSim)")
    }

    @Test func boundaryQuantizeMatchesArgmin() {
        // Verify boundary-based quantization gives same result as argmin
        let codec = MSECodec(dim: 16, bits: 3, seed: 42)
        let vectors = MLXRandom.normal([1, 1, 4, 16])
        eval(vectors)

        let norms = sqrt((vectors * vectors).sum(axis: -1))
        let unit = vectors / expandedDimensions(maximum(norms, MLXArray(Float(1e-8))), axis: -1)
        let rotated = matmul(unit, codec.rotationT)

        // Boundary quantize
        let boundaryIdx = codec.boundaryQuantize(rotated)

        // Argmin quantize (naive but correct)
        let expanded = expandedDimensions(rotated, axis: -1)
        let cb = codec.codebook.reshaped([1, 1, 1, 1, -1])
        let distances = MLX.abs(expanded - cb)
        let argminIdx = argMin(distances, axis: -1)

        let b = boundaryIdx.asArray(UInt32.self)
        let a = argminIdx.asArray(UInt32.self)
        #expect(b == a, "Boundary quantize should match argmin")
    }

    @Test func fwhtRoundTrip() {
        // FWHT forward → inverse should recover original vectors
        let signs = TurboQuantRotation.whtSigns(dim: 128, seed: 42)

        let vectors = MLXRandom.normal([2, 4, 8, 128])
        eval(vectors)

        let rotated = TurboQuantRotation.fwhtForward(vectors, signs: signs)
        let recovered = TurboQuantRotation.fwhtInverse(rotated, signs: signs)
        eval(recovered)

        let diff = MLX.abs(vectors - recovered).max().item(Float.self)
        #expect(diff < 1e-4, "FWHT round-trip max diff: \(diff)")
    }

    @Test func whtRotationOrthogonality() {
        // WHT rotation H*D/sqrt(d) should be orthogonal: Π·Π^T ≈ I
        let codec = MSECodec(dim: 128, bits: 3, seed: 42)
        #expect(codec.useWHT, "dim=128 should use WHT")

        let product = matmul(codec.rotation, codec.rotationT, stream: .cpu)
        let identity = MLXArray.identity(128)
        let diff = MLX.abs(product - identity).max().item(Float.self)
        // TF32 matmuls (neural accelerators) leave ~2e-4 here; see MatmulPrecision.
        let tolerance: Float = MatmulPrecision.tolerance(float32: 1e-4, reduced: 1e-3)
        #expect(diff < tolerance, "WHT rotation should be orthogonal, max diff: \(diff)")
    }

    @Test func whtEncodeDecodeRoundTrip() {
        // Verify encode/decode with WHT rotation produces reasonable reconstruction
        let codec = MSECodec(dim: 128, bits: 4, seed: 42)
        #expect(codec.useWHT, "dim=128 should use WHT")

        let vectors = MLXRandom.normal([1, 1, 8, 128])
        eval(vectors)

        let state = codec.encode(vectors)
        let decoded = codec.decode(state)

        let cosDot = (vectors * decoded).sum(axis: -1)
        let normOrig = sqrt((vectors * vectors).sum(axis: -1))
        let normDec = sqrt((decoded * decoded).sum(axis: -1))
        let cosSim = cosDot / (normOrig * normDec + 1e-8)
        let avgCosSim = cosSim.mean().item(Float.self)

        #expect(avgCosSim > 0.90, "WHT 4-bit cosine similarity too low: \(avgCosSim)")
    }

    @Test func normCorrectionImprovesNormAccuracy() {
        // Norm correction ensures the reconstructed vector's L2 norm matches the original.
        // This improves attention score accuracy (dot products), which is what matters
        // for perplexity. Element-wise MSE may not improve, but norm accuracy should.
        // dim 96 is not a power of two, so the codec takes the dense-rotation
        // path, the only path that applies norm correction (WHT preserves
        // norms exactly and skips it).
        let codec = MSECodec(dim: 96, bits: 3, seed: 42)
        let vectors = MLXRandom.normal([1, 1, 64, 96])
        eval(vectors)

        // Original norms
        let origNorms = sqrt((vectors * vectors).sum(axis: -1))

        // Encode with norm correction (current implementation)
        let state = codec.encode(vectors)
        let decodedCorrected = codec.decode(state)
        let correctedNorms = sqrt((decodedCorrected * decodedCorrected).sum(axis: -1))

        // Norm error with correction
        let normErrorCorrected = MLX.abs(origNorms - correctedNorms).mean().item(Float.self)

        // Manually encode WITHOUT norm correction for comparison
        let norms = sqrt((vectors * vectors).sum(axis: -1))
        let safeNorms = maximum(norms, MLXArray(Float(1e-8)))
        let unit = vectors / expandedDimensions(safeNorms, axis: -1)
        let rotated = matmul(unit, codec.rotationT)
        let indices = codec.boundaryQuantize(rotated)
        let packed = TurboQuantPacking.packLowBit(indices, bits: 3)
        let uncorrectedState = MSECodecState(
            norms: norms, packedIndices: packed, tokenCount: 64, dim: 96, bits: 3)
        let decodedUncorrected = codec.decode(uncorrectedState)
        let uncorrectedNorms = sqrt((decodedUncorrected * decodedUncorrected).sum(axis: -1))

        // Norm error without correction
        let normErrorUncorrected = MLX.abs(origNorms - uncorrectedNorms).mean().item(Float.self)

        #expect(
            normErrorCorrected < normErrorUncorrected,
            "Norm-corrected norm error (\(normErrorCorrected)) should be lower than uncorrected (\(normErrorUncorrected))"
        )

        // Also verify dot product accuracy improves (key for attention scoring)
        // Dot product of original with itself = ||v||², so compare dot products
        let dotCorrected = (vectors * decodedCorrected).sum(axis: -1)
        let dotUncorrected = (vectors * decodedUncorrected).sum(axis: -1)
        let dotOriginal = (vectors * vectors).sum(axis: -1)
        let dotErrorCorrected = MLX.abs(dotOriginal - dotCorrected).mean().item(Float.self)
        let dotErrorUncorrected = MLX.abs(dotOriginal - dotUncorrected).mean().item(Float.self)

        #expect(
            dotErrorCorrected < dotErrorUncorrected,
            "Norm-corrected dot product error (\(dotErrorCorrected)) should be lower than uncorrected (\(dotErrorUncorrected))"
        )
    }

    /// Key calibration: on synthetic data with a 10x per-dimension variance
    /// spread in the post-rotation space (the exact anisotropy calibration
    /// targets), quantize(rotate(k) / s) with per-vector norm recomputed after
    /// the scale divide should round-trip at least as well as the uncalibrated
    /// quantize(rotate(k)).
    @Test func keyCalibrationImprovesAnisotropicRoundTrip() {
        let dim = 64
        let codec = MSECodec(dim: dim, bits: 4, seed: 42)
        #expect(codec.useWHT, "dim=64 should use WHT")

        // Build vectors whose post-rotation coordinates have a 10x std spread:
        // first half of dims sqrt(10)x, second half 1x. Map the anisotropic
        // pattern back through the codec's own rotation so encode's internal
        // rotate(unit(vectors)) recovers (up to per-row norm) this shape.
        var stdVec = [Float](repeating: 1.0, count: dim)
        for i in 0 ..< dim / 2 { stdVec[i] = Float(10.0).squareRoot() }
        let stdArray = MLXArray(stdVec)
        let rotatedSynthetic = MLXRandom.normal([1, 1, 512, dim], key: MLXRandom.key(7)) * stdArray
        let vectors = matmul(rotatedSynthetic, codec.rotation)
        eval(vectors)

        // Calibration scale, mirrors TurboQuantKVCache.computeKeyCalibrationScale:
        // std per rotated dimension, normalized to mean 1, clamped to [0.25, 4.0].
        let norms = sqrt((vectors * vectors).sum(axis: -1))
        let unit = vectors / expandedDimensions(maximum(norms, MLXArray(Float(1e-8))), axis: -1)
        let rotated = matmul(unit, codec.rotationT).reshaped([-1, dim])
        let perDimStd = std(rotated, axis: 0)
        let meanStd = maximum(perDimStd.mean(), MLXArray(Float(1e-8)))
        let scale = clip(perDimStd / meanStd, min: Float(0.25), max: Float(4.0))
        eval(scale)

        let uncalibratedDecoded = codec.decode(codec.encode(vectors))
        let uncalibratedMSE =
            ((vectors - uncalibratedDecoded) * (vectors - uncalibratedDecoded)).mean().item(
                Float.self)

        let calibratedDecoded = codec.decode(codec.encode(vectors, scale: scale), scale: scale)
        let calibratedMSE =
            ((vectors - calibratedDecoded) * (vectors - calibratedDecoded)).mean().item(
                Float.self)

        #expect(
            calibratedMSE <= uncalibratedMSE,
            "calibrated MSE \(calibratedMSE) should not exceed uncalibrated \(uncalibratedMSE)")
    }
}

// MARK: - KV Cache Tests

@Suite("TurboQuantKVCache")
struct TurboQuantKVCacheTests {

    @Test func cacheUpdate() {
        let cache = TurboQuantKVCache(bits: 4)
        let keys = MLXRandom.normal([1, 4, 8, 64])
        let values = MLXRandom.normal([1, 4, 8, 64])
        eval(keys, values)

        let (outKeys, outValues) = cache.update(keys: keys, values: values)
        #expect(cache.offset == 8)
        #expect(outKeys.shape == [1, 4, 8, 64])
        #expect(outValues.shape == [1, 4, 8, 64])
    }

    @Test func cacheIncrementalUpdate() {
        let cache = TurboQuantKVCache(bits: 4)

        let k1 = MLXRandom.normal([1, 2, 4, 32])
        let v1 = MLXRandom.normal([1, 2, 4, 32])
        eval(k1, v1)
        let (_, _) = cache.update(keys: k1, values: v1)
        #expect(cache.offset == 4)

        let k2 = MLXRandom.normal([1, 2, 1, 32])
        let v2 = MLXRandom.normal([1, 2, 1, 32])
        eval(k2, v2)
        let (outK, _) = cache.update(keys: k2, values: v2)
        #expect(cache.offset == 5)
        #expect(outK.shape == [1, 2, 5, 32])
    }

    @Test func cacheTrim() {
        let cache = TurboQuantKVCache(bits: 4)
        let keys = MLXRandom.normal([1, 2, 8, 32])
        let values = MLXRandom.normal([1, 2, 8, 32])
        eval(keys, values)
        _ = cache.update(keys: keys, values: values)
        #expect(cache.offset == 8)

        let trimmed = cache.trim(3)
        #expect(trimmed == 3)
        #expect(cache.offset == 5)
    }

    @Test func cacheState() {
        let cache = TurboQuantKVCache(bits: 4)
        let keys = MLXRandom.normal([1, 2, 4, 32])
        let values = MLXRandom.normal([1, 2, 4, 32])
        eval(keys, values)
        _ = cache.update(keys: keys, values: values)

        let state = cache.state
        // In raw phase (prefill): 2 arrays (rawKeys, rawValues)
        // In compressed phase: 4 arrays (keyPacked, keyNorms, valPacked, valNorms)
        #expect(
            state.count == 2 || state.count == 4,
            "State should have 2 or 4 arrays, got \(state.count)")
    }

    @Test func cacheIsTrimmable() {
        let cache = TurboQuantKVCache(bits: 4)
        #expect(cache.isTrimmable == true)
    }

    /// Regression test: asymmetric bits (e.g. turbo3v2 = 3-bit K, 2-bit V) must encode
    /// values with valueBits, not the legacy `bits` field. A mismatch causes packed width
    /// errors during compressRawCache → reshape.
    @Test func cacheAsymmetricCompression() {
        // turbo3v2 config: keyBits=3, valueBits=2
        let cache = TurboQuantKVCache(bits: 3, keyBits: 3, valueBits: 2)
        let B = 1
        let H = 4  // KV heads
        let T = 32  // prefill tokens
        let D = 128

        // Phase 1: Prefill (raw FP16 storage)
        let keys = MLXRandom.normal([B, H, T, D])
        let values = MLXRandom.normal([B, H, T, D])
        eval(keys, values)
        let (_, _) = cache.update(keys: keys, values: values)
        #expect(cache.offset == T)

        // Phase 2: First decode triggers compressRawCache()
        let newKey = MLXRandom.normal([B, H, 1, D])
        let newVal = MLXRandom.normal([B, H, 1, D])
        let queries = MLXRandom.normal([B, H * 2, 1, D])  // nQHeads = 2 * nKVHeads (GQA)
        eval(newKey, newVal, queries)

        let output = cache.compressedAttention(
            queries: queries, keys: newKey, values: newVal,
            scale: 1.0 / sqrt(Float(D))
        )
        eval(output)
        #expect(output.shape == [B, H * 2, 1, D])
        #expect(cache.isCompressed)

        // Phase 3: Several more decode steps to confirm incremental encode also works
        for _ in 0 ..< 5 {
            let dk = MLXRandom.normal([B, H, 1, D])
            let dv = MLXRandom.normal([B, H, 1, D])
            let dq = MLXRandom.normal([B, H * 2, 1, D])
            eval(dk, dv, dq)
            let out = cache.compressedAttention(
                queries: dq, keys: dk, values: dv,
                scale: 1.0 / sqrt(Float(D))
            )
            eval(out)
            #expect(out.shape == [B, H * 2, 1, D])
        }
        #expect(cache.offset == T + 6)
    }

    @Test func cacheAllBitWidths() {
        for bits in 2 ... 4 {
            let cache = TurboQuantKVCache(bits: bits)
            let keys = MLXRandom.normal([1, 2, 4, 32])
            let values = MLXRandom.normal([1, 2, 4, 32])
            eval(keys, values)
            let (outKeys, outValues) = cache.update(keys: keys, values: values)
            #expect(cache.offset == 4, "\(bits)-bit cache offset wrong")
            #expect(outKeys.shape == [1, 2, 4, 32], "\(bits)-bit key shape wrong")
            #expect(outValues.shape == [1, 2, 4, 32], "\(bits)-bit value shape wrong")
        }
    }
}

// MARK: - TurboFlashAttention Tests

@Suite("TurboFlashAttention")
struct TurboFlashAttentionTests {

    /// Validate that fused TurboFlashAttention produces the same output as
    /// separated Score → Softmax → Value kernels.
    @Test func flashMatchesSeparated() {
        let dim = 128
        let keyBits = 4
        let valueBits = 4
        let nQHeads = 8
        let nKVHeads = 4
        let tokenCount = 64
        let repeatCount = nQHeads / nKVHeads

        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 42)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 43)

        // Generate random KV cache: encode random vectors
        let rawKeys = MLXRandom.normal([nKVHeads, tokenCount, dim])
        let rawValues = MLXRandom.normal([nKVHeads, tokenCount, dim])
        eval(rawKeys, rawValues)

        // Encode K and V
        let flatKeys = rawKeys.reshaped([nKVHeads * tokenCount, dim])
        let flatVals = rawValues.reshaped([nKVHeads * tokenCount, dim])

        let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatKeys, whtSigns: keyCodec.whtSigns!,
            boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
            bits: keyBits, dim: dim)
        let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatVals, whtSigns: valCodec.whtSigns!,
            boundaries: valCodec.boundaries, codebook: valCodec.codebook,
            bits: valueBits, dim: dim)

        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let flatKeyPacked = keyPacked.reshaped([nKVHeads, tokenCount, kpw])
        let flatKeyNorms = keyNorms.reshaped([nKVHeads, tokenCount])
        let flatValPacked = valPacked.reshaped([nKVHeads, tokenCount, vpw])
        let flatValNorms = valNorms.reshaped([nKVHeads, tokenCount])

        // Generate random queries (pre-rotated and scaled)
        let scale: Float = 1.0 / sqrt(Float(dim))
        let queries = MLXRandom.normal([nQHeads, dim]) * MLXArray(scale)
        eval(queries)

        // === Separated path: Score → Softmax → Value ===
        let scores = TurboQuantKernelOps.mseScore(
            rotatedQueries: queries, packed: flatKeyPacked, norms: flatKeyNorms,
            codebook: keyCodec.codebook, tokenCount: tokenCount,
            repeatCount: repeatCount, bits: keyBits, dim: dim)
        let attnWeights = softmax(scores, axis: -1)
        let separatedOutput = TurboQuantKernelOps.mseWeightedSum(
            weights: attnWeights, packed: flatValPacked, norms: flatValNorms,
            codebook: valCodec.codebook, tokenCount: tokenCount,
            repeatCount: repeatCount, bits: valueBits, dim: dim)
        eval(separatedOutput)

        // === Fused path: TurboFlashAttention ===
        let fusedOutput = TurboQuantKernelOps.turboFlashAttention(
            rotatedQueries: queries,
            keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
            keyCodebook: keyCodec.codebook,
            valPacked: flatValPacked, valNorms: flatValNorms,
            valCodebook: valCodec.codebook,
            tokenCount: tokenCount, repeatCount: repeatCount,
            keyBits: keyBits, valueBits: valueBits, dim: dim)
        eval(fusedOutput)

        // Compare outputs, should match within floating point tolerance
        // Online softmax may have slightly different numerical behavior than
        // materialized softmax, so allow a small tolerance
        let diff = abs(separatedOutput - fusedOutput)
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = mean(diff).item(Float.self)

        print("[TurboFlash] Max diff: \(maxDiff), Mean diff: \(meanDiff)")
        print(
            "[TurboFlash] Separated output range: [\(separatedOutput.min().item(Float.self)), \(separatedOutput.max().item(Float.self))]"
        )
        print(
            "[TurboFlash] Fused output range: [\(fusedOutput.min().item(Float.self)), \(fusedOutput.max().item(Float.self))]"
        )

        #expect(maxDiff < 1e-3, "Max diff \(maxDiff) exceeds tolerance 1e-3")
        #expect(meanDiff < 1e-4, "Mean diff \(meanDiff) exceeds tolerance 1e-4")
    }

    /// Test with asymmetric K/V bits (4-bit K, 2-bit V)
    @Test func flashAsymmetricBits() {
        let dim = 128
        let keyBits = 4
        let valueBits = 2
        let nQHeads = 4
        let nKVHeads = 2
        let tokenCount = 32
        let repeatCount = nQHeads / nKVHeads

        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 42)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 43)

        let rawKeys = MLXRandom.normal([nKVHeads, tokenCount, dim])
        let rawValues = MLXRandom.normal([nKVHeads, tokenCount, dim])
        eval(rawKeys, rawValues)

        let flatKeys = rawKeys.reshaped([nKVHeads * tokenCount, dim])
        let flatVals = rawValues.reshaped([nKVHeads * tokenCount, dim])

        let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatKeys, whtSigns: keyCodec.whtSigns!,
            boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
            bits: keyBits, dim: dim)
        let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatVals, whtSigns: valCodec.whtSigns!,
            boundaries: valCodec.boundaries, codebook: valCodec.codebook,
            bits: valueBits, dim: dim)

        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let flatKeyPacked = keyPacked.reshaped([nKVHeads, tokenCount, kpw])
        let flatKeyNorms = keyNorms.reshaped([nKVHeads, tokenCount])
        let flatValPacked = valPacked.reshaped([nKVHeads, tokenCount, vpw])
        let flatValNorms = valNorms.reshaped([nKVHeads, tokenCount])

        let scale: Float = 1.0 / sqrt(Float(dim))
        let queries = MLXRandom.normal([nQHeads, dim]) * MLXArray(scale)
        eval(queries)

        // Separated
        let scores = TurboQuantKernelOps.mseScore(
            rotatedQueries: queries, packed: flatKeyPacked, norms: flatKeyNorms,
            codebook: keyCodec.codebook, tokenCount: tokenCount,
            repeatCount: repeatCount, bits: keyBits, dim: dim)
        let attnWeights = softmax(scores, axis: -1)
        let separatedOutput = TurboQuantKernelOps.mseWeightedSum(
            weights: attnWeights, packed: flatValPacked, norms: flatValNorms,
            codebook: valCodec.codebook, tokenCount: tokenCount,
            repeatCount: repeatCount, bits: valueBits, dim: dim)

        // Fused
        let fusedOutput = TurboQuantKernelOps.turboFlashAttention(
            rotatedQueries: queries,
            keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
            keyCodebook: keyCodec.codebook,
            valPacked: flatValPacked, valNorms: flatValNorms,
            valCodebook: valCodec.codebook,
            tokenCount: tokenCount, repeatCount: repeatCount,
            keyBits: keyBits, valueBits: valueBits, dim: dim)

        eval(separatedOutput, fusedOutput)

        let maxDiff = abs(separatedOutput - fusedOutput).max().item(Float.self)
        print("[TurboFlash Asymmetric 4K/2V] Max diff: \(maxDiff)")
        #expect(maxDiff < 1e-3, "Max diff \(maxDiff) exceeds tolerance for asymmetric bits")
    }

    /// The bounded path intentionally changes only scheduling: it must agree
    /// with the established block-parallel implementation, including when the
    /// inverse value rotation is fused into the same dispatch.
    @Test func singleDispatchMatchesTwoPass() {
        let dim = 128
        let keyBits = 4
        let valueBits = 2
        let nQHeads = 8
        let nKVHeads = 2
        let tokenCount = 193  // spans multiple 64-token blocks in the reference path
        let repeatCount = nQHeads / nKVHeads
        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 91)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 92)

        let rawKeys = MLXRandom.normal([nKVHeads * tokenCount, dim])
        let rawValues = MLXRandom.normal([nKVHeads * tokenCount, dim])
        let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: rawKeys, whtSigns: keyCodec.whtSigns!,
            boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
            bits: keyBits, dim: dim)
        let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: rawValues, whtSigns: valCodec.whtSigns!,
            boundaries: valCodec.boundaries, codebook: valCodec.codebook,
            bits: valueBits, dim: dim)
        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let keys = keyPacked.reshaped(nKVHeads, tokenCount, kpw)
        let keyNormsByHead = keyNorms.reshaped(nKVHeads, tokenCount)
        let values = valPacked.reshaped(nKVHeads, tokenCount, vpw)
        let valNormsByHead = valNorms.reshaped(nKVHeads, tokenCount)
        let queries = MLXRandom.normal([nQHeads, dim]) / sqrt(Float(dim))

        func attention(singleDispatch: Bool, rotation: MLXArray? = nil) -> MLXArray {
            TurboQuantKernelOps.turboFlashAttention(
                rotatedQueries: queries,
                keyPacked: keys, keyNorms: keyNormsByHead,
                keyCodebook: keyCodec.codebook,
                valPacked: values, valNorms: valNormsByHead,
                valCodebook: valCodec.codebook,
                tokenCount: tokenCount, repeatCount: repeatCount,
                keyBits: keyBits, valueBits: valueBits, dim: dim,
                valRotation: rotation, singleDispatch: singleDispatch)
        }

        let single = attention(singleDispatch: true)
        let twoPass = attention(singleDispatch: false)
        let identity = MLXArray.eye(dim)
        let singleRotated = attention(singleDispatch: true, rotation: identity)
        let twoPassRotated = attention(singleDispatch: false, rotation: identity)
        eval(single, twoPass, singleRotated, twoPassRotated)

        let maxDiff = abs(single - twoPass).max().item(Float.self)
        let rotatedMaxDiff = abs(singleRotated - twoPassRotated).max().item(Float.self)
        #expect(maxDiff < 1e-4, "single/two-pass max diff \(maxDiff) exceeds 1e-4")
        #expect(
            rotatedMaxDiff < 1e-4,
            "single/two-pass rotated max diff \(rotatedMaxDiff) exceeds 1e-4")
    }

    @Test func microbenchSingleDispatchVsTwoPass() {
        let dim = 128
        let keyBits = 4
        let valueBits = 2
        let nQHeads = 24
        let nKVHeads = 4
        let repeatCount = nQHeads / nKVHeads
        let iterations = 300
        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 93)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 94)

        for tokenCount in [1, 8, 16, 32, 64, 128, 256] {
            let rawKeys = MLXRandom.normal([nKVHeads * tokenCount, dim])
            let rawValues = MLXRandom.normal([nKVHeads * tokenCount, dim])
            let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: rawKeys, whtSigns: keyCodec.whtSigns!,
                boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
                bits: keyBits, dim: dim)
            let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: rawValues, whtSigns: valCodec.whtSigns!,
                boundaries: valCodec.boundaries, codebook: valCodec.codebook,
                bits: valueBits, dim: dim)
            let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
            let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
            let keys = keyPacked.reshaped(nKVHeads, tokenCount, kpw)
            let values = valPacked.reshaped(nKVHeads, tokenCount, vpw)
            let keyNormsByHead = keyNorms.reshaped(nKVHeads, tokenCount)
            let valNormsByHead = valNorms.reshaped(nKVHeads, tokenCount)
            let queries = MLXRandom.normal([nQHeads, dim]) / sqrt(Float(dim))

            func run(singleDispatch: Bool) -> MLXArray {
                TurboQuantKernelOps.turboFlashAttention(
                    rotatedQueries: queries,
                    keyPacked: keys,
                    keyNorms: keyNormsByHead,
                    keyCodebook: keyCodec.codebook,
                    valPacked: values,
                    valNorms: valNormsByHead,
                    valCodebook: valCodec.codebook,
                    tokenCount: tokenCount, repeatCount: repeatCount,
                    keyBits: keyBits, valueBits: valueBits, dim: dim,
                    singleDispatch: singleDispatch)
            }

            for _ in 0 ..< 20 {
                eval(run(singleDispatch: true), run(singleDispatch: false))
            }
            let singleStart = Date()
            for _ in 0 ..< iterations { eval(run(singleDispatch: true)) }
            let singleMs = Date().timeIntervalSince(singleStart) * 1000 / Double(iterations)
            let twoPassStart = Date()
            for _ in 0 ..< iterations { eval(run(singleDispatch: false)) }
            let twoPassMs = Date().timeIntervalSince(twoPassStart) * 1000 / Double(iterations)
            let singleText = String(format: "%.3f", singleMs)
            let twoPassText = String(format: "%.3f", twoPassMs)
            let speedupText = String(format: "%.2f", twoPassMs / singleMs)
            print(
                "[MICROBENCH single] T=\(tokenCount): single=\(singleText)ms, "
                    + "two-pass=\(twoPassText)ms, speedup=\(speedupText)x")
        }
    }

    @Test func shortDecodeNR0PolicyIsBitExact() {
        struct Shape {
            let dim: Int
            let keyBits: Int
            let valueBits: Int
            let queryHeads: Int
            let keyValueHeads: Int
            let tokenCount: Int
        }

        #expect(
            TurboQuantKernelOps.automaticFlashDecodeNR0(
                tokenCount: 129, totalQueries: 16, dim: 64) == 1)
        #expect(
            TurboQuantKernelOps.automaticFlashDecodeNR0(
                tokenCount: 256, totalQueries: 32, dim: 128) == 1)
        #expect(
            TurboQuantKernelOps.automaticFlashDecodeNR0(
                tokenCount: 128, totalQueries: 32, dim: 128) == 2)
        #expect(
            TurboQuantKernelOps.automaticFlashDecodeNR0(
                tokenCount: 257, totalQueries: 32, dim: 128) == 2)
        #expect(
            TurboQuantKernelOps.automaticFlashDecodeNR0(
                tokenCount: 256, totalQueries: 33, dim: 128) == 2)
        #expect(
            TurboQuantKernelOps.automaticFlashDecodeNR0(
                tokenCount: 256, totalQueries: 15, dim: 128) == 2)
        #expect(
            TurboQuantKernelOps.automaticFlashDecodeNR0(
                tokenCount: 256, totalQueries: 32, dim: 96) == 2)

        let shapes = [
            Shape(
                dim: 64, keyBits: 2, valueBits: 2,
                queryHeads: 16, keyValueHeads: 4, tokenCount: 129),
            Shape(
                dim: 64, keyBits: 4, valueBits: 4,
                queryHeads: 32, keyValueHeads: 8, tokenCount: 256),
            Shape(
                dim: 128, keyBits: 3, valueBits: 3,
                queryHeads: 24, keyValueHeads: 4, tokenCount: 256),
            Shape(
                dim: 128, keyBits: 4, valueBits: 2,
                queryHeads: 32, keyValueHeads: 8, tokenCount: 129),
        ]

        for (shapeIndex, shape) in shapes.enumerated() {
            let repeatCount = shape.queryHeads / shape.keyValueHeads
            let keyCodec = MSECodec(
                dim: shape.dim, bits: shape.keyBits, seed: UInt64(200 + shapeIndex))
            let valueCodec = MSECodec(
                dim: shape.dim, bits: shape.valueBits, seed: UInt64(300 + shapeIndex))
            let rawKeys = MLXRandom.normal([
                shape.keyValueHeads * shape.tokenCount, shape.dim,
            ])
            let rawValues = MLXRandom.normal([
                shape.keyValueHeads * shape.tokenCount, shape.dim,
            ])
            let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: rawKeys, whtSigns: keyCodec.whtSigns!,
                boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
                bits: shape.keyBits, dim: shape.dim)
            let (valuePacked, valueNorms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: rawValues, whtSigns: valueCodec.whtSigns!,
                boundaries: valueCodec.boundaries, codebook: valueCodec.codebook,
                bits: shape.valueBits, dim: shape.dim)
            let keyWidth = TurboQuantPacking.packedWidth(
                count: shape.dim, bits: shape.keyBits)
            let valueWidth = TurboQuantPacking.packedWidth(
                count: shape.dim, bits: shape.valueBits)
            let keys = keyPacked.reshaped(
                shape.keyValueHeads, shape.tokenCount, keyWidth)
            let values = valuePacked.reshaped(
                shape.keyValueHeads, shape.tokenCount, valueWidth)
            let keyNormsByHead = keyNorms.reshaped(
                shape.keyValueHeads, shape.tokenCount)
            let valueNormsByHead = valueNorms.reshaped(
                shape.keyValueHeads, shape.tokenCount)
            let queries =
                MLXRandom.normal([shape.queryHeads, shape.dim])
                / sqrt(Float(shape.dim))

            func run(_ nr0: Int) -> MLXArray {
                TurboQuantKernelOps.turboFlashAttention(
                    rotatedQueries: queries,
                    keyPacked: keys, keyNorms: keyNormsByHead,
                    keyCodebook: keyCodec.codebook,
                    valPacked: values, valNorms: valueNormsByHead,
                    valCodebook: valueCodec.codebook,
                    tokenCount: shape.tokenCount, repeatCount: repeatCount,
                    keyBits: shape.keyBits, valueBits: shape.valueBits, dim: shape.dim,
                    valRotation: valueCodec.rotation,
                    singleDispatch: false, nr0: nr0)
            }

            let oneRow = run(1)
            let twoRows = run(2)
            eval(oneRow, twoRows)
            #expect(
                oneRow.asArray(Float.self) == twoRows.asArray(Float.self),
                "NR0 scheduling changed output for shape \(shape)")
        }
    }

    /// Microbenchmark: fused vs separated at various token counts
    @Test func microbenchFlashVsSeparated() {
        let dim = 128
        let keyBits = 4
        let valueBits = 2
        let nQHeads = 24  // Qwen3.5-2B query heads
        let nKVHeads = 4  // Qwen3.5-2B KV heads
        let repeatCount = nQHeads / nKVHeads
        let iterations = 200
        let warmup = 50

        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 42)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 43)

        for tokenCount in [128, 512, 1024, 2048, 4096, 8192] {
            let rawKeys = MLXRandom.normal([nKVHeads, tokenCount, dim])
            let rawValues = MLXRandom.normal([nKVHeads, tokenCount, dim])
            eval(rawKeys, rawValues)

            let flatKeys = rawKeys.reshaped([nKVHeads * tokenCount, dim])
            let flatVals = rawValues.reshaped([nKVHeads * tokenCount, dim])

            let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: flatKeys, whtSigns: keyCodec.whtSigns!,
                boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
                bits: keyBits, dim: dim)
            let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: flatVals, whtSigns: valCodec.whtSigns!,
                boundaries: valCodec.boundaries, codebook: valCodec.codebook,
                bits: valueBits, dim: dim)

            let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
            let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
            let flatKeyPacked = keyPacked.reshaped([nKVHeads, tokenCount, kpw])
            let flatKeyNorms = keyNorms.reshaped([nKVHeads, tokenCount])
            let flatValPacked = valPacked.reshaped([nKVHeads, tokenCount, vpw])
            let flatValNorms = valNorms.reshaped([nKVHeads, tokenCount])

            let scale: Float = 1.0 / sqrt(Float(dim))
            let queries = MLXRandom.normal([nQHeads, dim]) * MLXArray(scale)
            eval(queries)

            // Warmup both paths
            for _ in 0 ..< warmup {
                let s = TurboQuantKernelOps.mseScore(
                    rotatedQueries: queries, packed: flatKeyPacked, norms: flatKeyNorms,
                    codebook: keyCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: keyBits, dim: dim)
                let w = softmax(s, axis: -1)
                let v = TurboQuantKernelOps.mseWeightedSum(
                    weights: w, packed: flatValPacked, norms: flatValNorms,
                    codebook: valCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: valueBits, dim: dim)
                eval(v)

                let f = TurboQuantKernelOps.turboFlashAttention(
                    rotatedQueries: queries,
                    keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
                    keyCodebook: keyCodec.codebook,
                    valPacked: flatValPacked, valNorms: flatValNorms,
                    valCodebook: valCodec.codebook,
                    tokenCount: tokenCount, repeatCount: repeatCount,
                    keyBits: keyBits, valueBits: valueBits, dim: dim)
                eval(f)
            }

            // Benchmark separated
            let startSep = Date()
            for _ in 0 ..< iterations {
                let s = TurboQuantKernelOps.mseScore(
                    rotatedQueries: queries, packed: flatKeyPacked, norms: flatKeyNorms,
                    codebook: keyCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: keyBits, dim: dim)
                let w = softmax(s, axis: -1)
                let v = TurboQuantKernelOps.mseWeightedSum(
                    weights: w, packed: flatValPacked, norms: flatValNorms,
                    codebook: valCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: valueBits, dim: dim)
                eval(v)
            }
            let sepMs = Date().timeIntervalSince(startSep) * 1000 / Double(iterations)

            // Benchmark fused (default block size)
            let startFused = Date()
            for _ in 0 ..< iterations {
                let f = TurboQuantKernelOps.turboFlashAttention(
                    rotatedQueries: queries,
                    keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
                    keyCodebook: keyCodec.codebook,
                    valPacked: flatValPacked, valNorms: flatValNorms,
                    valCodebook: valCodec.codebook,
                    tokenCount: tokenCount, repeatCount: repeatCount,
                    keyBits: keyBits, valueBits: valueBits, dim: dim)
                eval(f)
            }
            let fusedMs = Date().timeIntervalSince(startFused) * 1000 / Double(iterations)

            let speedup = sepMs / fusedMs
            print(
                "[MICROBENCH] T=\(tokenCount): separated=\(String(format: "%.3f", sepMs))ms, fused(B=\(TurboQuantKernelOps.flashBlockSize))=\(String(format: "%.3f", fusedMs))ms, speedup=\(String(format: "%.2f", speedup))x"
            )
        }
    }

    /// Block size sweep: find optimal block size for each token count
    @Test func microbenchBlockSizeSweep() {
        let dim = 128
        let keyBits = 4
        let valueBits = 2
        let nQHeads = 24
        let nKVHeads = 4
        let repeatCount = nQHeads / nKVHeads
        let iterations = 200
        let warmup = 30

        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 42)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 43)

        let blockSizes = [32, 64, 128, 256, 512, 1024]
        let tokenCounts = [512, 1024, 2048, 4096, 8192]

        for tokenCount in tokenCounts {
            let rawKeys = MLXRandom.normal([nKVHeads, tokenCount, dim])
            let rawValues = MLXRandom.normal([nKVHeads, tokenCount, dim])
            eval(rawKeys, rawValues)

            let flatKeys = rawKeys.reshaped([nKVHeads * tokenCount, dim])
            let flatVals = rawValues.reshaped([nKVHeads * tokenCount, dim])

            let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: flatKeys, whtSigns: keyCodec.whtSigns!,
                boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
                bits: keyBits, dim: dim)
            let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: flatVals, whtSigns: valCodec.whtSigns!,
                boundaries: valCodec.boundaries, codebook: valCodec.codebook,
                bits: valueBits, dim: dim)

            let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
            let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
            let flatKeyPacked = keyPacked.reshaped([nKVHeads, tokenCount, kpw])
            let flatKeyNorms = keyNorms.reshaped([nKVHeads, tokenCount])
            let flatValPacked = valPacked.reshaped([nKVHeads, tokenCount, vpw])
            let flatValNorms = valNorms.reshaped([nKVHeads, tokenCount])

            let scale: Float = 1.0 / sqrt(Float(dim))
            let queries = MLXRandom.normal([nQHeads, dim]) * MLXArray(scale)
            eval(queries)

            // Separated baseline
            for _ in 0 ..< warmup {
                let s = TurboQuantKernelOps.mseScore(
                    rotatedQueries: queries, packed: flatKeyPacked, norms: flatKeyNorms,
                    codebook: keyCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: keyBits, dim: dim)
                let w = softmax(s, axis: -1)
                let v = TurboQuantKernelOps.mseWeightedSum(
                    weights: w, packed: flatValPacked, norms: flatValNorms,
                    codebook: valCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: valueBits, dim: dim)
                eval(v)
            }
            let startSep = Date()
            for _ in 0 ..< iterations {
                let s = TurboQuantKernelOps.mseScore(
                    rotatedQueries: queries, packed: flatKeyPacked, norms: flatKeyNorms,
                    codebook: keyCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: keyBits, dim: dim)
                let w = softmax(s, axis: -1)
                let v = TurboQuantKernelOps.mseWeightedSum(
                    weights: w, packed: flatValPacked, norms: flatValNorms,
                    codebook: valCodec.codebook, tokenCount: tokenCount,
                    repeatCount: repeatCount, bits: valueBits, dim: dim)
                eval(v)
            }
            let sepMs = Date().timeIntervalSince(startSep) * 1000 / Double(iterations)

            var results: [(Int, Double)] = []
            for bs in blockSizes where bs <= tokenCount {
                // Warmup
                for _ in 0 ..< warmup {
                    let f = TurboQuantKernelOps.turboFlashAttention(
                        rotatedQueries: queries,
                        keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
                        keyCodebook: keyCodec.codebook,
                        valPacked: flatValPacked, valNorms: flatValNorms,
                        valCodebook: valCodec.codebook,
                        tokenCount: tokenCount, repeatCount: repeatCount,
                        keyBits: keyBits, valueBits: valueBits, dim: dim,
                        blockSize: bs)
                    eval(f)
                }

                let start = Date()
                for _ in 0 ..< iterations {
                    let f = TurboQuantKernelOps.turboFlashAttention(
                        rotatedQueries: queries,
                        keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
                        keyCodebook: keyCodec.codebook,
                        valPacked: flatValPacked, valNorms: flatValNorms,
                        valCodebook: valCodec.codebook,
                        tokenCount: tokenCount, repeatCount: repeatCount,
                        keyBits: keyBits, valueBits: valueBits, dim: dim,
                        blockSize: bs)
                    eval(f)
                }
                let ms = Date().timeIntervalSince(start) * 1000 / Double(iterations)
                results.append((bs, ms))
            }

            let best = results.min(by: { $0.1 < $1.1 })!
            let resultStr = results.map { "B=\($0.0):\(String(format: "%.2f", $0.1))ms" }.joined(
                separator: "  ")
            print(
                "[SWEEP] T=\(tokenCount): sep=\(String(format: "%.2f", sepMs))ms  \(resultStr)  BEST=B\(best.0)(\(String(format: "%.1f", sepMs/best.1))x)"
            )
        }
    }

    /// Validate that causal TurboFlashAttention matches per-position reference.
    /// Computes reference by running non-causal flash on truncated KV for each query position.
    @Test func flashCausalMatchesSeparated() {
        let dim = 128
        let keyBits = 4
        let valueBits = 2
        let nQHeads = 8
        let nKVHeads = 4
        let L = 4  // query chunk length
        let tokenCount = 32  // total KV cache length
        let repeatCount = nQHeads / nKVHeads
        let queryOffset = tokenCount - L  // queries cover positions 28..31

        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 42)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 43)

        let rawKeys = MLXRandom.normal([nKVHeads, tokenCount, dim])
        let rawValues = MLXRandom.normal([nKVHeads, tokenCount, dim])
        eval(rawKeys, rawValues)

        let flatKeys = rawKeys.reshaped([nKVHeads * tokenCount, dim])
        let flatVals = rawValues.reshaped([nKVHeads * tokenCount, dim])

        let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatKeys, whtSigns: keyCodec.whtSigns!,
            boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
            bits: keyBits, dim: dim)
        let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatVals, whtSigns: valCodec.whtSigns!,
            boundaries: valCodec.boundaries, codebook: valCodec.codebook,
            bits: valueBits, dim: dim)

        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let flatKeyPacked = keyPacked.reshaped([nKVHeads, tokenCount, kpw])
        let flatKeyNorms = keyNorms.reshaped([nKVHeads, tokenCount])
        let flatValPacked = valPacked.reshaped([nKVHeads, tokenCount, vpw])
        let flatValNorms = valNorms.reshaped([nKVHeads, tokenCount])

        let scale: Float = 1.0 / sqrt(Float(dim))
        // Queries in [nQHeads, L, dim] layout, flattened to [nQHeads * L, dim]
        let queries = MLXRandom.normal([nQHeads, L, dim]) * MLXArray(scale)
        let flatQueries = queries.reshaped([nQHeads * L, dim])
        eval(flatQueries)

        // === Reference: compute per-position using non-causal flash on truncated KV ===
        // For each query position l, attend only to tokens 0...(queryOffset + l)
        var refOutputs: [MLXArray] = []
        for l in 0 ..< L {
            let visibleTokens = queryOffset + l + 1  // causal: can see up to and including position
            let truncKeyPacked = flatKeyPacked[0..., ..<visibleTokens, 0...]
            let truncKeyNorms = flatKeyNorms[0..., ..<visibleTokens]
            let truncValPacked = flatValPacked[0..., ..<visibleTokens, 0...]
            let truncValNorms = flatValNorms[0..., ..<visibleTokens]

            // Extract queries for position l across all heads: queries[:, l, :]
            let posQueries = queries[0..., l, 0...].reshaped([nQHeads, dim])

            let posOutput = TurboQuantKernelOps.turboFlashAttention(
                rotatedQueries: posQueries,
                keyPacked: truncKeyPacked, keyNorms: truncKeyNorms,
                keyCodebook: keyCodec.codebook,
                valPacked: truncValPacked, valNorms: truncValNorms,
                valCodebook: valCodec.codebook,
                tokenCount: visibleTokens, repeatCount: repeatCount,
                keyBits: keyBits, valueBits: valueBits, dim: dim)
            refOutputs.append(posOutput)  // [nQHeads, dim]
        }
        // Stack and interleave to match [nQHeads * L, dim] layout from [nQHeads, L, dim]
        let refStacked = stacked(refOutputs, axis: 1)  // [nQHeads, L, dim]
        let refOutput = refStacked.reshaped([nQHeads * L, dim])
        eval(refOutput)

        // === Causal TurboFlashAttention ===
        let causalOutput = TurboQuantKernelOps.turboFlashAttentionCausal(
            rotatedQueries: flatQueries,
            keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
            keyCodebook: keyCodec.codebook,
            valPacked: flatValPacked, valNorms: flatValNorms,
            valCodebook: valCodec.codebook,
            tokenCount: tokenCount, repeatCount: repeatCount,
            keyBits: keyBits, valueBits: valueBits, dim: dim,
            queryChunkLength: L, queryOffset: queryOffset)
        eval(causalOutput)

        let diff = abs(refOutput - causalOutput)
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = mean(diff).item(Float.self)

        print("[TurboFlash Causal] Max diff: \(maxDiff), Mean diff: \(meanDiff)")
        #expect(maxDiff < 1e-3, "Max diff \(maxDiff) exceeds tolerance 1e-3")
        #expect(meanDiff < 1e-4, "Mean diff \(meanDiff) exceeds tolerance 1e-4")
    }

    /// Validate that fused rotation in pass 2 matches separate matmul rotation.
    @Test func flashFusedRotationMatchesSeparate() {
        let dim = 128
        let keyBits = 4
        let valueBits = 2
        let nQHeads = 8
        let nKVHeads = 4
        let tokenCount = 64
        let repeatCount = nQHeads / nKVHeads

        let keyCodec = MSECodec(dim: dim, bits: keyBits, seed: 42)
        let valCodec = MSECodec(dim: dim, bits: valueBits, seed: 43)

        let rawKeys = MLXRandom.normal([nKVHeads, tokenCount, dim])
        let rawValues = MLXRandom.normal([nKVHeads, tokenCount, dim])
        eval(rawKeys, rawValues)

        let flatKeys = rawKeys.reshaped([nKVHeads * tokenCount, dim])
        let flatVals = rawValues.reshaped([nKVHeads * tokenCount, dim])

        let (keyPacked, keyNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatKeys, whtSigns: keyCodec.whtSigns!,
            boundaries: keyCodec.boundaries, codebook: keyCodec.codebook,
            bits: keyBits, dim: dim)
        let (valPacked, valNorms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: flatVals, whtSigns: valCodec.whtSigns!,
            boundaries: valCodec.boundaries, codebook: valCodec.codebook,
            bits: valueBits, dim: dim)

        let kpw = TurboQuantPacking.packedWidth(count: dim, bits: keyBits)
        let vpw = TurboQuantPacking.packedWidth(count: dim, bits: valueBits)
        let flatKeyPacked = keyPacked.reshaped([nKVHeads, tokenCount, kpw])
        let flatKeyNorms = keyNorms.reshaped([nKVHeads, tokenCount])
        let flatValPacked = valPacked.reshaped([nKVHeads, tokenCount, vpw])
        let flatValNorms = valNorms.reshaped([nKVHeads, tokenCount])

        let scale: Float = 1.0 / sqrt(Float(dim))
        let queries = MLXRandom.normal([nQHeads, dim]) * MLXArray(scale)
        eval(queries)

        // Without rotation fusion: get rotated output, then matmul
        let rotatedOutput = TurboQuantKernelOps.turboFlashAttention(
            rotatedQueries: queries,
            keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
            keyCodebook: keyCodec.codebook,
            valPacked: flatValPacked, valNorms: flatValNorms,
            valCodebook: valCodec.codebook,
            tokenCount: tokenCount, repeatCount: repeatCount,
            keyBits: keyBits, valueBits: valueBits, dim: dim)
        let separateRotOutput = matmul(rotatedOutput, valCodec.rotation)
        eval(separateRotOutput)

        // With rotation fusion: rotation applied in pass 2 kernel
        let fusedRotOutput = TurboQuantKernelOps.turboFlashAttention(
            rotatedQueries: queries,
            keyPacked: flatKeyPacked, keyNorms: flatKeyNorms,
            keyCodebook: keyCodec.codebook,
            valPacked: flatValPacked, valNorms: flatValNorms,
            valCodebook: valCodec.codebook,
            tokenCount: tokenCount, repeatCount: repeatCount,
            keyBits: keyBits, valueBits: valueBits, dim: dim,
            valRotation: valCodec.rotation)
        eval(fusedRotOutput)

        let diff = abs(separateRotOutput - fusedRotOutput)
        let maxDiff = diff.max().item(Float.self)
        let meanDiff = mean(diff).item(Float.self)

        print("[TurboFlash Fused Rotation] Max diff: \(maxDiff), Mean diff: \(meanDiff)")
        #expect(maxDiff < 1e-3, "Max diff \(maxDiff) exceeds tolerance 1e-3")
    }
}

// MARK: - Encode Kernel Microbenchmark

@Suite("TurboQuant Encode Microbench")
struct TurboQuantEncodeMicrobenchTests {

    /// Microbenchmark: Dense rotation fused encode kernel.
    /// Simulates the hot path: 1 token × nKVHeads encode calls per decode step.
    /// Runs many iterations to average out GPU scheduling noise.
    @Test func microbenchDenseEncode() {
        let dim = 128
        let bits = 4
        let nKVHeads = 4  // typical GQA KV head count for 27B
        let iterations = 500
        let warmup = 50

        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        // Dense rotation, force non-WHT by using the QR rotation directly
        let rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: 99)

        // Simulate single-token encode: [nKVHeads, dim] per call
        let input = MLXRandom.normal([nKVHeads, dim])
        eval(input)

        // Warmup
        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        // Timed iterations
        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print(
            "[MICROBENCH] Dense encode: \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call"
        )
        // Just a smoke test, the print output is what we care about
        #expect(perCall > 0)
    }

    /// Microbenchmark: WHT rotation fused encode kernel.
    @Test func microbenchWHTEncode() {
        let dim = 128
        let bits = 4
        let nKVHeads = 4
        let iterations = 500
        let warmup = 50

        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        guard codec.useWHT, let signs = codec.whtSigns else {
            Issue.record("dim=128 should use WHT")
            return
        }

        let input = MLXRandom.normal([nKVHeads, dim])
        eval(input)

        // Warmup
        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        // Timed iterations
        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print(
            "[MICROBENCH] WHT encode:   \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call"
        )
        #expect(perCall > 0)
    }

    /// Microbenchmark: Batch encode (prefill transition), 512 tokens at once.
    @Test func microbenchBatchEncodeDense() {
        let dim = 128
        let bits = 4
        let batchSize = 512 * 4  // 512 tokens × 4 KV heads
        let iterations = 100
        let warmup = 10

        let rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: 99)
        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        let input = MLXRandom.normal([batchSize, dim])
        eval(input)

        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print(
            "[MICROBENCH] Dense batch encode (512×4): \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call"
        )
        #expect(perCall > 0)
    }

    /// Microbenchmark: Batch encode WHT, 512 tokens at once.
    @Test func microbenchBatchEncodeWHT() {
        let dim = 128
        let bits = 4
        let batchSize = 512 * 4
        let iterations = 100
        let warmup = 10

        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        guard codec.useWHT, let signs = codec.whtSigns else {
            Issue.record("dim=128 should use WHT")
            return
        }
        let input = MLXRandom.normal([batchSize, dim])
        eval(input)

        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print(
            "[MICROBENCH] WHT batch encode dim=128 (512×4):   \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call"
        )
        #expect(perCall > 0)
    }

    // --- dim=256 variants (Qwen3.5-27B full attention head_dim) ---

    @Test func microbenchDenseEncode256() {
        let dim = 256
        let bits = 4
        let nKVHeads = 4
        let iterations = 500
        let warmup = 50

        let rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: 99)
        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        let input = MLXRandom.normal([nKVHeads, dim])
        eval(input)

        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print(
            "[MICROBENCH] Dense encode dim=256: \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call"
        )
        #expect(perCall > 0)
    }

    @Test func microbenchWHTEncode256() {
        let dim = 256
        let bits = 4
        let nKVHeads = 4
        let iterations = 500
        let warmup = 50

        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        guard codec.useWHT, let signs = codec.whtSigns else {
            Issue.record("dim=256 should use WHT")
            return
        }
        let input = MLXRandom.normal([nKVHeads, dim])
        eval(input)

        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print(
            "[MICROBENCH] WHT encode dim=256:   \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call"
        )
        #expect(perCall > 0)
    }

    @Test func microbenchBatchEncodeDense256() {
        let dim = 256
        let bits = 4
        let batchSize = 512 * 4
        let iterations = 100
        let warmup = 10

        let rotation = TurboQuantRotation.rotationMatrix(dim: dim, seed: 99)
        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        let input = MLXRandom.normal([batchSize, dim])
        eval(input)

        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncode(
                input: input, rotation: rotation,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print(
            "[MICROBENCH] Dense batch encode dim=256 (512×4): \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call"
        )
        #expect(perCall > 0)
    }

    @Test func microbenchBatchEncodeWHT256() {
        let dim = 256
        let bits = 4
        let batchSize = 512 * 4
        let iterations = 100
        let warmup = 10

        let codec = MSECodec(dim: dim, bits: bits, seed: 99)
        guard codec.useWHT, let signs = codec.whtSigns else {
            Issue.record("dim=256 should use WHT")
            return
        }
        let input = MLXRandom.normal([batchSize, dim])
        eval(input)

        for _ in 0 ..< warmup {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }

        let start = Date()
        for _ in 0 ..< iterations {
            let (p, n) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: signs,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(p, n)
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        let perCall = elapsed / Double(iterations)

        print(
            "[MICROBENCH] WHT batch encode dim=256 (512×4):   \(iterations) iters, \(String(format: "%.1f", elapsed))ms total, \(String(format: "%.3f", perCall))ms/call"
        )
        #expect(perCall > 0)
    }
}

// MARK: - kvScheme routing + integration (XCTest)

final class TurboQuantIntegrationTests: XCTestCase {

    func testResolveTurboScheme() {
        XCTAssertNil(resolveTurboScheme(nil))
        XCTAssertNil(resolveTurboScheme("affine4"))
        XCTAssertNil(resolveTurboScheme("nonsense"))

        // Raw-key schemes are the validated surface; keyBits 0 = FP16 keys.
        XCTAssertEqual(resolveTurboScheme("turbo0v4")?.keyBits, 0)
        XCTAssertEqual(resolveTurboScheme("turbo0v4")?.valueBits, 4)
        XCTAssertEqual(resolveTurboScheme("turbo0v2")?.valueBits, 2)
        // Key-quantizing schemes are exposed on the canonical codec.
        XCTAssertEqual(resolveTurboScheme("turbo4")?.keyBits, 4)
        XCTAssertEqual(resolveTurboScheme("turbo4v2")?.valueBits, 2)
        // Affine-K asymmetric ladder: keyBits 8 = 8-bit affine keys.
        XCTAssertEqual(resolveTurboScheme("turbo8v4")?.keyBits, 8)
        XCTAssertEqual(resolveTurboScheme("turbo8v3")?.valueBits, 3)
        XCTAssertEqual(resolveTurboScheme("turbo8v2")?.valueBits, 2)
        XCTAssertEqual(resolveTurboScheme("turbo0v3")?.valueBits, 3)
        XCTAssertNil(resolveTurboScheme("turbo8"))
    }

    func testMaybeTurboQuantizeConvertsEligibleLayers() {
        let (b, h, d, t) = (1, 2, 64, 8)
        let simple = KVCacheSimple()
        let keys = MLXRandom.normal([b, h, t, d], key: MLXRandom.key(9))
        let values = MLXRandom.normal([b, h, t, d], key: MLXRandom.key(10))
        _ = simple.update(keys: keys, values: values)

        var caches: [KVCache] = [simple]
        maybeTurboQuantizeKVCache(cache: &caches, keyBits: 4, valueBits: 4, quantizedKVStart: 0)

        XCTAssertTrue(caches[0] is TurboQuantKVCache, "eligible layer not converted")
        XCTAssertEqual(caches[0].offset, t, "offset lost in conversion")

        // Below the start threshold: untouched.
        let early = KVCacheSimple()
        _ = early.update(
            keys: MLXRandom.normal([b, h, 2, d], key: MLXRandom.key(11)),
            values: MLXRandom.normal([b, h, 2, d], key: MLXRandom.key(12)))
        var earlyCaches: [KVCache] = [early]
        maybeTurboQuantizeKVCache(
            cache: &earlyCaches, keyBits: 4, valueBits: 4, quantizedKVStart: 100)
        XCTAssertTrue(earlyCaches[0] is KVCacheSimple, "layer below threshold was converted")
    }

    // MARK: - Mixed rotating/standard cache lists (Gemma-style sliding window)

    func testMixedRotatingAndStandardCacheLeavesRotatingLayersUntouched() {
        // Gemma-family layouts interleave RotatingKVCache (sliding window)
        // with KVCacheSimple (global) layers. TurboQuant should only ever
        // touch the KVCacheSimple layers: RotatingKVCache's eviction/rotation
        // storage layout is not compatible with the sequential-append
        // compression path, and its window is already memory-bounded.
        let (b, h, d, t) = (1, 2, 64, 8)

        func filledStandard() -> KVCacheSimple {
            let c = KVCacheSimple()
            _ = c.update(
                keys: MLXRandom.normal([b, h, t, d], key: MLXRandom.key(61)),
                values: MLXRandom.normal([b, h, t, d], key: MLXRandom.key(62)))
            return c
        }

        func filledRotating() -> RotatingKVCache {
            let c = RotatingKVCache(maxSize: 16, keep: 0)
            _ = c.update(
                keys: MLXRandom.normal([b, h, t, d], key: MLXRandom.key(63)),
                values: MLXRandom.normal([b, h, t, d], key: MLXRandom.key(64)))
            return c
        }

        // turbo8v4 is a near-lossless (non-fragile) scheme, so boundary-layer
        // protection does not engage and every eligible layer converts --
        // isolating the rotating-vs-standard split this test targets.
        var caches: [KVCache] = [
            filledRotating(), filledStandard(), filledRotating(), filledStandard(),
        ]
        maybeTurboQuantizeKVCache(cache: &caches, keyBits: 8, valueBits: 4, quantizedKVStart: 0)

        XCTAssertTrue(caches[0] is RotatingKVCache, "rotating layer 0 was converted")
        XCTAssertTrue(caches[1] is TurboQuantKVCache, "standard layer 1 was not converted")
        XCTAssertTrue(caches[2] is RotatingKVCache, "rotating layer 2 was converted")
        XCTAssertTrue(caches[3] is TurboQuantKVCache, "standard layer 3 was not converted")
        XCTAssertEqual(caches[0].offset, t, "rotating cache offset disturbed by conversion pass")
        XCTAssertEqual(caches[2].offset, t, "rotating cache offset disturbed by conversion pass")
    }

    func testAllRotatingCacheIsNoOp() {
        // Edge case: every layer is a RotatingKVCache (no global layers at
        // all). There is nothing eligible to convert; the scheme should
        // no-op cleanly rather than crash or silently drop state.
        let (b, h, d, t) = (1, 2, 64, 8)

        func filledRotating(_ seed: UInt64) -> RotatingKVCache {
            let c = RotatingKVCache(maxSize: 16, keep: 0)
            _ = c.update(
                keys: MLXRandom.normal([b, h, t, d], key: MLXRandom.key(seed)),
                values: MLXRandom.normal([b, h, t, d], key: MLXRandom.key(seed + 1)))
            return c
        }

        var caches: [KVCache] = [filledRotating(71), filledRotating(73)]
        maybeTurboQuantizeKVCache(cache: &caches, keyBits: 4, valueBits: 4, quantizedKVStart: 0)

        XCTAssertTrue(
            caches.allSatisfy { $0 is RotatingKVCache }, "all-rotating cache list should no-op")
        XCTAssertEqual(caches[0].offset, t)
        XCTAssertEqual(caches[1].offset, t)
    }

    func testEndToEndGenerationWithTurboScheme() throws {
        // Tiny random-weight Llama: prefill, then a short greedy decode loop
        // with the turbo4 scheme applied after each step, the same call
        // pattern as TokenIterator.step(). Catches integration breaks
        // (cache conversion, attention routing, NaN logits) without a
        // checkpoint.
        let config = LlamaConfiguration(
            hiddenSize: 128, hiddenLayers: 6, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 1e-5, vocabularySize: 100, kvHeads: 4)
        let model = LlamaModel(config)
        quantize(model: model, groupSize: 64, bits: 4)
        eval(model)

        func decode(scheme: String?) throws -> [Int] {
            var cache = try model.newCache(parameters: nil)
            let prompt = MLXArray([1, 7, 3, 12, 5, 9, 2, 8])[.newAxis, .ellipsis]
            var logits = model(prompt, cache: cache)
            var tokens: [Int] = []
            for _ in 0 ..< 8 {
                maybeQuantizeKVCache(
                    cache: &cache, kvBits: nil, quantizedKVStart: 0, kvScheme: scheme)
                let next = argMax(logits[0..., -1, 0...], axis: -1)
                let token = next.item(Int.self)
                tokens.append(token)
                logits = model(next[.newAxis, .ellipsis], cache: cache)
                let bad = MLX.isNaN(logits).any().item(Bool.self)
                XCTAssertFalse(bad, "NaN logits during \(scheme ?? "fp16") decode")
            }
            if scheme != nil {
                // turbo0v4 is a non-fragile scheme: no boundary protection,
                // every layer converts.
                XCTAssertTrue(
                    cache.allSatisfy { $0 is TurboQuantKVCache },
                    "caches not converted to TurboQuantKVCache")
            }
            return tokens
        }

        let turbo = try decode(scheme: "turbo0v4")
        let reference = try decode(scheme: nil)
        XCTAssertEqual(turbo.count, 8)
        // The first decode token is computed from the still-raw cache, so it
        // must match the FP16 run exactly; later tokens may diverge within
        // 4-bit quantization error.
        XCTAssertEqual(turbo[0], reference[0], "first decode token diverged before compression")
    }

    func testEndToEndGemmaLikeMixedCacheDecode() throws {
        // Tiny random-weight Gemma3Text: alternating sliding-window
        // (RotatingKVCache) and global (KVCacheSimple) layers via
        // slidingWindowPattern: 2. Prefill, then a short greedy decode loop
        // applying the turbo8v4 scheme after each step, the same call
        // pattern as TokenIterator.step(). Only the global layers should
        // ever convert; the sliding-window layers must keep decoding
        // correctly as plain fp16 RotatingKVCache throughout.
        //
        // headDim 64 (not a smaller tiny value) is required here: turbo8v4's
        // 8-bit affine key path uses a fixed groupSize of 64 internally, so
        // headDim must be a multiple of it.
        let config = Gemma3TextConfiguration(
            modelType: "gemma3_text", hiddenSize: 64, hiddenLayers: 4, intermediateSize: 128,
            attentionHeads: 4, headDim: 64, rmsNormEps: 1e-5, vocabularySize: 50, kvHeads: 2,
            ropeTheta: 1_000_000, ropeLocalBaseFreq: 10_000, ropeTraditional: false,
            queryPreAttnScalar: 64, slidingWindow: 8, slidingWindowPattern: 2,
            maxPositionEmbeddings: 128)
        let model = Gemma3TextModel(config)
        eval(model)

        var cache = try model.newCache(parameters: nil)
        // isGlobalLayer = (i % slidingWindowPattern == slidingWindowPattern - 1),
        // so with pattern 2: layers 0, 2 are sliding-window; 1, 3 are global.
        XCTAssertTrue(cache[0] is RotatingKVCache, "layer 0 should be sliding-window")
        XCTAssertTrue(cache[1] is KVCacheSimple, "layer 1 should be global")
        XCTAssertTrue(cache[2] is RotatingKVCache, "layer 2 should be sliding-window")
        XCTAssertTrue(cache[3] is KVCacheSimple, "layer 3 should be global")

        let prompt = MLXArray([1, 7, 3, 12, 5])[.newAxis, .ellipsis]
        var logits = model(prompt, cache: cache)
        eval(logits)
        XCTAssertFalse(MLX.isNaN(logits).any().item(Bool.self), "NaN logits during prefill")

        for _ in 0 ..< 6 {
            maybeQuantizeKVCache(
                cache: &cache, kvBits: nil, quantizedKVStart: 0, kvScheme: "turbo8v4")
            let next = argMax(logits[0..., -1, 0...], axis: -1)
            logits = model(next[.newAxis, .ellipsis], cache: cache)
            eval(logits)
            XCTAssertFalse(MLX.isNaN(logits).any().item(Bool.self), "NaN logits during decode")
        }

        XCTAssertTrue(cache[0] is RotatingKVCache, "sliding-window layer 0 was converted")
        XCTAssertTrue(cache[1] is TurboQuantKVCache, "global layer 1 was not converted")
        XCTAssertTrue(cache[2] is RotatingKVCache, "sliding-window layer 2 was converted")
        XCTAssertTrue(cache[3] is TurboQuantKVCache, "global layer 3 was not converted")
    }

    func testPromptCacheSaveRestoreRoundTrip() throws {
        // Regression test: TurboQuant caches used to serialize into a
        // prompt cache and restore as plain KVCacheSimple, silently
        // dropping the compressed (packed indices + scales) state.
        let (b, hq, hkv, d) = (1, 4, 2, 64)
        let prefill = 16

        let cache = TurboQuantKVCache(bits: 4, keyBits: 4, valueBits: 2, seed: 123)
        let keys = MLXRandom.normal([b, hkv, prefill, d], key: MLXRandom.key(31)) * 0.3
        let values = MLXRandom.normal([b, hkv, prefill, d], key: MLXRandom.key(32)) * 0.3
        _ = cache.update(keys: keys, values: values)

        let newK = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(33)) * 0.3
        let newV = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(34)) * 0.3
        let q = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(35)) * 0.3
        let scale = 1.0 / Float(d).squareRoot()

        // Force compression so the saved state exercises the compressed
        // (packed + scales) branch, not the raw-prefill branch.
        _ = cache.compressedAttention(queries: q, keys: newK, values: newV, scale: scale)
        XCTAssertTrue(cache.isCompressed)
        XCTAssertEqual(cache.offset, prefill + 1)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")

        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        XCTAssertEqual(loaded.count, 1)
        guard let restored = loaded[0] as? TurboQuantKVCache else {
            XCTFail("expected TurboQuantKVCache, got \(type(of: loaded[0]))")
            return
        }

        XCTAssertEqual(restored.offset, cache.offset)
        XCTAssertEqual(restored.metaState, cache.metaState)
        XCTAssertEqual(restored.keyBits, cache.keyBits)
        XCTAssertEqual(restored.valueBits, cache.valueBits)

        let originalState = cache.state
        let restoredState = restored.state
        XCTAssertEqual(originalState.count, restoredState.count)
        for (i, (orig, rest)) in zip(originalState, restoredState).enumerated() {
            XCTAssertEqual(orig.shape, rest.shape, "state[\(i)] shape mismatch")
            XCTAssertTrue(allClose(orig, rest).item(Bool.self), "state[\(i)] values diverged")
        }

        // Restored cache must still be functional for further decode steps.
        let nextK = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(36)) * 0.3
        let nextV = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(37)) * 0.3
        let nextQ = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(38)) * 0.3
        let out = restored.compressedAttention(
            queries: nextQ, keys: nextK, values: nextV, scale: scale)

        XCTAssertEqual(out.shape, [b, hq, 1, d])
        XCTAssertEqual(restored.offset, prefill + 2)
        XCTAssertFalse(MLX.isNaN(out).any().item(Bool.self), "post-restore decode produced NaN")
    }

    /// Key calibration scale must round-trip through save/restore: build
    /// anisotropic keys (same 10x post-rotation spread as the codec-level
    /// test) so compressRawCache actually engages calibration, then verify
    /// the extra 5th state array (keyCalibScale) survives a save/load cycle.
    func testKeyCalibrationPromptCacheRoundTrip() throws {
        let (b, hkv, hq, d) = (1, 2, 4, 64)
        let prefill = 64
        let seed: UInt64 = 42

        // Same seed the cache will use internally for its key codec, so the
        // rotation matrix here matches the one compressRawCache rotates by.
        let refCodec = MSECodec(dim: d, bits: 4, seed: seed)
        var stdVec = [Float](repeating: 1.0, count: d)
        for i in 0 ..< d / 2 { stdVec[i] = Float(10.0).squareRoot() }
        let stdArray = MLXArray(stdVec)
        let rotatedSynthetic =
            MLXRandom.normal([b, hkv, prefill, d], key: MLXRandom.key(71)) * stdArray
        let keys = matmul(rotatedSynthetic, refCodec.rotation)
        let values = MLXRandom.normal([b, hkv, prefill, d], key: MLXRandom.key(72)) * 0.3
        eval(keys, values)

        let cache = TurboQuantKVCache(bits: 4, keyBits: 4, valueBits: 4, seed: seed)
        _ = cache.update(keys: keys, values: values)

        let newK = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(73)) * 0.3
        let newV = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(74)) * 0.3
        let q = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(75)) * 0.3
        let scale = 1.0 / Float(d).squareRoot()
        _ = cache.compressedAttention(queries: q, keys: newK, values: newV, scale: scale)
        XCTAssertTrue(cache.isCompressed)

        let originalState = cache.state
        XCTAssertEqual(
            originalState.count, 5,
            "anisotropic key data should engage key calibration (5-array state)")

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)
        let restored = try XCTUnwrap(loaded[0] as? TurboQuantKVCache)

        let restoredState = restored.state
        XCTAssertEqual(originalState.count, restoredState.count)
        for (i, (orig, rest)) in zip(originalState, restoredState).enumerated() {
            XCTAssertEqual(orig.shape, rest.shape, "state[\(i)] shape mismatch")
            XCTAssertTrue(allClose(orig, rest).item(Bool.self), "state[\(i)] values diverged")
        }

        // Restored cache must still decode correctly using the restored scale.
        let nextK = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(76)) * 0.3
        let nextV = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(77)) * 0.3
        let nextQ = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(78)) * 0.3
        let out = restored.compressedAttention(
            queries: nextQ, keys: nextK, values: nextV, scale: scale)
        XCTAssertEqual(out.shape, [b, hq, 1, d])
        XCTAssertFalse(MLX.isNaN(out).any().item(Bool.self), "post-restore decode produced NaN")
    }

    func testBoundaryProtectionGating() {
        // Fragile schemes (turbo-quantized keys) protect the first/last 2
        // attention layers; near-lossless schemes convert everything.
        func layers(_ n: Int) -> [KVCache] {
            (0 ..< n).map { _ in
                let c = KVCacheSimple()
                _ = c.update(
                    keys: MLXArray.ones([1, 2, 8, 64], dtype: .float16),
                    values: MLXArray.ones([1, 2, 8, 64], dtype: .float16))
                return c
            }
        }

        var fragile = layers(6)
        maybeTurboQuantizeKVCache(
            cache: &fragile, keyBits: 4, valueBits: 4, quantizedKVStart: 0)
        // Boundary layers get near-lossless q8 affine, interior layers turbo.
        XCTAssertTrue(fragile[0] is QuantizedKVCache && fragile[5] is QuantizedKVCache)
        XCTAssertTrue(fragile[2] is TurboQuantKVCache && fragile[3] is TurboQuantKVCache)

        var safe = layers(6)
        maybeTurboQuantizeKVCache(
            cache: &safe, keyBits: 8, valueBits: 4, quantizedKVStart: 0)
        XCTAssertTrue(safe.allSatisfy { $0 is TurboQuantKVCache })

        var aggressiveV = layers(6)
        maybeTurboQuantizeKVCache(
            cache: &aggressiveV, keyBits: 0, valueBits: 2, quantizedKVStart: 0)
        XCTAssertTrue(aggressiveV[0] is QuantizedKVCache, "2-bit V should re-engage protection")
        XCTAssertTrue(aggressiveV[2] is TurboQuantKVCache)
    }

    /// Head dimension 80 divides none of the supported affine-quantization
    /// group sizes (32, 64, 128). Regular-path affine-K conversion
    /// (keyBits == 8, non-fragile so boundary protection doesn't engage)
    /// must skip such layers, leaving them fp16, rather than crashing deep
    /// inside `quantized()`.
    func testAffineKeyModeSkipsIncompatibleHeadDim() {
        let (b, h, d, t) = (1, 2, 80, 8)
        let simple = KVCacheSimple()
        _ = simple.update(
            keys: MLXRandom.normal([b, h, t, d], key: MLXRandom.key(81)),
            values: MLXRandom.normal([b, h, t, d], key: MLXRandom.key(82)))

        var caches: [KVCache] = [simple]
        // turbo8v4 is non-fragile (keyBits == 8, valueBits > 2), so this
        // exercises the regular affine-K path, not boundary protection.
        maybeTurboQuantizeKVCache(cache: &caches, keyBits: 8, valueBits: 4, quantizedKVStart: 0)

        XCTAssertTrue(
            caches[0] is KVCacheSimple,
            "head dim 80 is incompatible with every affine group size, should stay fp16")
        XCTAssertEqual(caches[0].offset, t, "offset lost while skipping conversion")
    }

    /// Same head-dim-80 incompatibility, but on a fragile scheme so the
    /// first/last layers route through boundary protection (always 8-bit
    /// affine for both K and V, regardless of the requested scheme).
    func testBoundaryProtectionSkipsIncompatibleHeadDim() {
        let (b, h, d, t) = (1, 2, 80, 8)
        func filled(_ seed: UInt64) -> KVCacheSimple {
            let c = KVCacheSimple()
            _ = c.update(
                keys: MLXRandom.normal([b, h, t, d], key: MLXRandom.key(seed)),
                values: MLXRandom.normal([b, h, t, d], key: MLXRandom.key(seed + 1)))
            return c
        }

        var caches: [KVCache] = [filled(83), filled(85), filled(87), filled(89)]
        // turbo2 is fragile (valueBits <= 2): every layer in this 4-layer
        // list lands in the protected boundary set.
        maybeTurboQuantizeKVCache(cache: &caches, keyBits: 2, valueBits: 2, quantizedKVStart: 0)

        XCTAssertTrue(
            caches.allSatisfy { $0 is KVCacheSimple },
            "head dim 80 is incompatible with every affine group size, boundary layers should stay fp16"
        )
    }

    // MARK: - Affine-K asymmetric mode (turbo8vN)

    func testAffineKeyModeMatchesReference() throws {
        // q8-affine keys + turbo4 values vs exact attention over raw K/V.
        let (b, hq, hkv, t, d) = (1, 4, 2, 256, 128)
        let scale = 1.0 / Float(d).squareRoot()
        let keys = MLXRandom.normal([b, hkv, t, d], key: MLXRandom.key(51)) * 0.5
        let values = MLXRandom.normal([b, hkv, t, d], key: MLXRandom.key(52)) * 0.5
        let q = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(53)) * 0.5
        let newK = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(54)) * 0.5
        let newV = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(55)) * 0.5

        let cache = TurboQuantKVCache(bits: 4, keyBits: 8, valueBits: 4, seed: 42)
        XCTAssertTrue(cache.affineKeyMode)
        _ = cache.update(keys: keys, values: values)
        let out = cache.compressedAttention(
            queries: q, keys: newK, values: newV, scale: scale)

        let nRep = hq / hkv
        let fullK = concatenated([keys, newK], axis: 2)
        let fullV = concatenated([values, newV], axis: 2)
        let ek = MLX.tiled(expandedDimensions(fullK, axis: 2), repetitions: [1, 1, nRep, 1, 1])
            .reshaped([b, hq, t + 1, d])
        let ev = MLX.tiled(expandedDimensions(fullV, axis: 2), repetitions: [1, 1, nRep, 1, 1])
            .reshaped([b, hq, t + 1, d])
        let ref = matmul(
            softmax(matmul(q, ek.transposed(0, 1, 3, 2)) * scale, axis: -1), ev)

        let a = out.asType(.float32).reshaped([-1])
        let r = ref.asType(.float32).reshaped([-1])
        let cos = ((a * r).sum() / (sqrt((a * a).sum()) * sqrt((r * r).sum()) + 1e-9))
            .item(Float.self)
        // 4-bit V + q8 K at T=256 lands at cos ~0.976-0.980; the exact value
        // shifts a few 1e-4 across mlx-swift patch versions. 0.97 catches real
        // breakage (bugs push this below 0.9) without pinning numerics.
        XCTAssertGreaterThan(cos, 0.97, "affine-K attention diverged from reference (cos \(cos))")
        XCTAssertFalse(MLX.isNaN(out).any().item(Bool.self))
    }

    /// rawKeyMode (keyBits: 0) + turbo4 values, bfloat16 K cache: the raw-K
    /// flash kernel reads K in its native dtype (KT template) instead of
    /// casting the whole cache to float16 every decode step. One cached key
    /// row is pushed above float16 max (65504) to prove that no longer
    /// turns finite bfloat16 activations into inf.
    func testRawKeyModeBFloat16MatchesReference() throws {
        let (b, hq, hkv, t, d) = (1, 4, 2, 256, 128)
        let scale = 1.0 / Float(d).squareRoot()
        let keys = (MLXRandom.normal([b, hkv, t, d], key: MLXRandom.key(71)) * 0.5)
            .asType(.bfloat16)
        let values = (MLXRandom.normal([b, hkv, t, d], key: MLXRandom.key(72)) * 0.5)
            .asType(.bfloat16)
        let q = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(73)) * 0.5
        let newK = (MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(74)) * 0.5)
            .asType(.bfloat16)
        let newV = (MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(75)) * 0.5)
            .asType(.bfloat16)

        // Beyond float16's max finite value: forcing this cache through
        // asType(.float16) would turn it into inf and NaN the softmax.
        keys[0..., 0..., 0, 0] = MLXArray(Float(70000)).asType(.bfloat16)
        eval(keys)

        let cache = TurboQuantKVCache(bits: 4, keyBits: 0, valueBits: 4, seed: 42)
        XCTAssertTrue(cache.rawKeyMode)
        _ = cache.update(keys: keys, values: values)
        let out = cache.compressedAttention(
            queries: q, keys: newK, values: newV, scale: scale)

        XCTAssertFalse(MLX.isNaN(out).any().item(Bool.self))
        XCTAssertFalse(MLX.isInf(out).any().item(Bool.self))

        let nRep = hq / hkv
        let fullK = concatenated([keys, newK], axis: 2).asType(.float32)
        let fullV = concatenated([values, newV], axis: 2).asType(.float32)
        let ek = MLX.tiled(expandedDimensions(fullK, axis: 2), repetitions: [1, 1, nRep, 1, 1])
            .reshaped([b, hq, t + 1, d])
        let ev = MLX.tiled(expandedDimensions(fullV, axis: 2), repetitions: [1, 1, nRep, 1, 1])
            .reshaped([b, hq, t + 1, d])
        let ref = matmul(
            softmax(matmul(q.asType(.float32), ek.transposed(0, 1, 3, 2)) * scale, axis: -1), ev)

        let a = out.asType(.float32).reshaped([-1])
        let r = ref.asType(.float32).reshaped([-1])
        let cos = ((a * r).sum() / (sqrt((a * a).sum()) * sqrt((r * r).sum()) + 1e-9))
            .item(Float.self)
        // Same cos-similarity style/threshold as testAffineKeyModeMatchesReference:
        // 4-bit V at T=256 plus bfloat16 K rounding lands well above 0.97;
        // a real regression (e.g. inf from an f16 cast) collapses this.
        XCTAssertGreaterThan(
            cos, 0.97, "raw-K bfloat16 attention diverged from reference (cos \(cos))")
    }

    func testAffineKeyModePromptCacheRoundTrip() throws {
        let (b, hkv, hq, d) = (1, 2, 4, 64)
        let cache = TurboQuantKVCache(bits: 4, keyBits: 8, valueBits: 4, seed: 7)
        let keys = MLXRandom.normal([b, hkv, 16, d], key: MLXRandom.key(61))
        let values = MLXRandom.normal([b, hkv, 16, d], key: MLXRandom.key(62))
        _ = cache.update(keys: keys, values: values)
        let q = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(63))
        let nk = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(64))
        let nv = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(65))
        _ = cache.compressedAttention(
            queries: q, keys: nk, values: nv, scale: 0.125)
        XCTAssertTrue(cache.isCompressed)
        XCTAssertEqual(cache.state.count, 5)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)
        let restored = try XCTUnwrap(loaded[0] as? TurboQuantKVCache)
        XCTAssertTrue(restored.affineKeyMode)
        XCTAssertEqual(restored.offset, cache.offset)

        let out = restored.compressedAttention(
            queries: q, keys: nk, values: nv, scale: 0.125)
        XCTAssertEqual(restored.offset, 18)
        XCTAssertFalse(MLX.isNaN(out).any().item(Bool.self))
    }

    // MARK: - Persistence for the remaining state layouts

    func testRawKeyModePromptCacheRoundTrip() throws {
        // turbo0vN (the recommended family) serializes as [rawKeys, valPacked,
        // valNorms], 3 arrays.
        let (b, hq, hkv, d) = (1, 4, 2, 64)
        let cache = TurboQuantKVCache(bits: 4, keyBits: 0, valueBits: 4, seed: 9)
        _ = cache.update(
            keys: MLXRandom.normal([b, hkv, 16, d], key: MLXRandom.key(71)),
            values: MLXRandom.normal([b, hkv, 16, d], key: MLXRandom.key(72)))
        let q = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(73))
        let nk = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(74))
        let nv = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(75))
        _ = cache.compressedAttention(queries: q, keys: nk, values: nv, scale: 0.125)
        XCTAssertTrue(cache.isCompressed)
        XCTAssertEqual(cache.state.count, 3)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathExtension("safetensors")
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)
        let restored = try XCTUnwrap(loaded[0] as? TurboQuantKVCache)
        XCTAssertTrue(restored.rawKeyMode)
        XCTAssertEqual(restored.offset, cache.offset)
        let out = restored.compressedAttention(queries: q, keys: nk, values: nv, scale: 0.125)
        XCTAssertEqual(restored.offset, 18)
        XCTAssertFalse(MLX.isNaN(out).any().item(Bool.self))
    }

    func testRawPhasePromptCacheRoundTrip() throws {
        // Saving straight after prefill (before any decode) is the primary
        // prompt-cache use case: state is the raw 2-array layout.
        let (b, hq, hkv, d) = (1, 4, 2, 64)
        let cache = TurboQuantKVCache(bits: 4, keyBits: 0, valueBits: 4, seed: 9)
        _ = cache.update(
            keys: MLXRandom.normal([b, hkv, 16, d], key: MLXRandom.key(81)),
            values: MLXRandom.normal([b, hkv, 16, d], key: MLXRandom.key(82)))
        XCTAssertFalse(cache.isCompressed)
        XCTAssertEqual(cache.state.count, 2)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathExtension("safetensors")
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)
        let restored = try XCTUnwrap(loaded[0] as? TurboQuantKVCache)
        XCTAssertFalse(restored.isCompressed)
        XCTAssertEqual(restored.offset, 16)

        // Decode through the compression transition on the restored cache.
        let q = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(83))
        let nk = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(84))
        let nv = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(85))
        let out = restored.compressedAttention(queries: q, keys: nk, values: nv, scale: 0.125)
        XCTAssertTrue(restored.isCompressed)
        XCTAssertEqual(restored.offset, 17)
        XCTAssertFalse(MLX.isNaN(out).any().item(Bool.self))
    }

    // MARK: - GQA repeat factors the NR0 kernel must not mis-map

    func testStandardAttentionRepeatFactors() throws {
        // rep=1 (MHA) and rep=3 (odd) previously aliased KV heads through the
        // NR0 fast path; the dispatcher now excludes them. Compare against
        // exact attention over the dequantized cache contents at loose 4-bit
        // tolerance.
        for (hq, hkv) in [(4, 4), (6, 2), (4, 2)] {
            let (b, t, d) = (1, 192, 64)
            let scale = 1.0 / Float(d).squareRoot()
            let k = MLXRandom.normal([b, hkv, t, d], key: MLXRandom.key(91)) * 0.5
            let v = MLXRandom.normal([b, hkv, t, d], key: MLXRandom.key(92)) * 0.5
            let q = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(93)) * 0.5
            let nk = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(94)) * 0.5
            let nv = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(95)) * 0.5

            let cache = TurboQuantKVCache(bits: 4, keyBits: 4, valueBits: 4, seed: 5)
            _ = cache.update(keys: k, values: v)
            let out = cache.compressedAttention(queries: q, keys: nk, values: nv, scale: scale)

            let nRep = hq / hkv
            let fullK = concatenated([k, nk], axis: 2)
            let fullV = concatenated([v, nv], axis: 2)
            let ek = MLX.tiled(
                expandedDimensions(fullK, axis: 2), repetitions: [1, 1, nRep, 1, 1]
            ).reshaped([b, hq, t + 1, d])
            let ev = MLX.tiled(
                expandedDimensions(fullV, axis: 2), repetitions: [1, 1, nRep, 1, 1]
            ).reshaped([b, hq, t + 1, d])
            let ref = matmul(
                softmax(matmul(q, ek.transposed(0, 1, 3, 2)) * scale, axis: -1), ev)

            let a = out.asType(.float32).reshaped([-1])
            let r = ref.asType(.float32).reshaped([-1])
            let cos = ((a * r).sum() / (sqrt((a * a).sum()) * sqrt((r * r).sum()) + 1e-9))
                .item(Float.self)
            XCTAssertGreaterThan(cos, 0.95, "rep=\(nRep): cos \(cos)")
        }
    }

    // MARK: - Trim on a compressed cache, then keep decoding

    func testCompressedTrimThenDecode() throws {
        for (kb, vb) in [(0, 4), (8, 4), (4, 4)] {
            let (b, hq, hkv, d) = (1, 4, 2, 64)
            let cache = TurboQuantKVCache(bits: 4, keyBits: kb, valueBits: vb, seed: 3)
            _ = cache.update(
                keys: MLXRandom.normal([b, hkv, 32, d], key: MLXRandom.key(101)),
                values: MLXRandom.normal([b, hkv, 32, d], key: MLXRandom.key(102)))
            let q = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(103))
            let nk = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(104))
            let nv = MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(105))
            _ = cache.compressedAttention(queries: q, keys: nk, values: nv, scale: 0.125)
            XCTAssertTrue(cache.isCompressed)
            XCTAssertEqual(cache.offset, 33)

            let trimmed = cache.trim(5)
            XCTAssertEqual(trimmed, 5)
            XCTAssertEqual(cache.offset, 28)

            // Two more decode steps after the partial trim must stay finite
            // and advance the offset from the trimmed position.
            for i in 0 ..< 2 {
                let out = cache.compressedAttention(
                    queries: q,
                    keys: MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(UInt64(110 + i))),
                    values: MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(UInt64(120 + i))),
                    scale: 0.125)
                XCTAssertFalse(
                    MLX.isNaN(out).any().item(Bool.self), "kb=\(kb) vb=\(vb) step \(i)")
            }
            XCTAssertEqual(cache.offset, 30, "kb=\(kb) vb=\(vb)")
        }
    }

    // MARK: - Speculative decoding × turbo caches

    func testSpeculativeDecodeWithTurboScheme() throws {
        // Speculative rounds trim the cache every round and verify with L>1
        // chunks, the composition that previously hit the memory-unsafe raw
        // update() path on compressed caches.
        let config = LlamaConfiguration(
            hiddenSize: 128, hiddenLayers: 6, intermediateSize: 128, attentionHeads: 8,
            rmsNormEps: 1e-5, vocabularySize: 100, kvHeads: 4)
        let main = LlamaModel(config)
        let draft = LlamaModel(config)
        quantize(model: main, groupSize: 64, bits: 4)
        quantize(model: draft, groupSize: 64, bits: 4)
        eval(main, draft)

        let prompt = MLXArray([1, 7, 3, 12, 5, 9, 2, 8, 4, 11])
        var iterator = try SpeculativeTokenIterator(
            input: .init(text: .init(tokens: prompt)),
            mainModel: main, draftModel: draft,
            parameters: GenerateParameters(maxTokens: 12, kvScheme: "turbo0v4", temperature: 0),
            numDraftTokens: 2)

        var produced = 0
        while iterator.next() != nil {
            produced += 1
            // Mirror the generate() wrapper's per-round conversion.
            var cache = iterator.mainCache
            maybeQuantizeKVCache(
                cache: &cache, kvBits: nil, quantizedKVStart: 0, kvScheme: "turbo0v4")
        }
        XCTAssertGreaterThan(produced, 4)
        XCTAssertTrue(
            iterator.mainCache.contains { $0 is TurboQuantKVCache },
            "main cache never converted")
    }

    // MARK: - Fused WHT encoder determinism

    /// The fused WHT encoder gives the same codes on every run: its
    /// cross-SIMD-group butterfly stages read both slots before writing.
    func testFusedEncodeWHTIsDeterministic() {
        let dim = 256
        let codec = MSECodec(dim: dim, bits: 4, seed: 43)
        let input = MLXRandom.normal([16_384, dim], key: MLXRandom.key(7))
        func encode() -> (MLXArray, MLXArray) {
            let (packed, norms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: input, whtSigns: codec.whtSigns!, boundaries: codec.boundaries,
                codebook: codec.codebook, bits: 4, dim: dim)
            eval(packed, norms)
            return (packed, norms)
        }
        let (firstPacked, firstNorms) = encode()
        for run in 1 ..< 20 {
            let (packed, norms) = encode()
            XCTAssertTrue(
                MLX.arrayEqual(packed, firstPacked).item(Bool.self),
                "packed codes differ on run \(run)")
            XCTAssertTrue(
                MLX.arrayEqual(norms, firstNorms).item(Bool.self), "norms differ on run \(run)")
        }
    }

    // MARK: - Dense (non-pow2) encode kernel parity with the reference codec

    func testDenseEncodeKernelMatchesCodecDim80() throws {
        let d = 80
        let codec = MSECodec(dim: d, bits: 4, seed: 42)
        let vectors = MLXRandom.normal([1, 1, 48, d], key: MLXRandom.key(131))
        let ref = codec.encode(vectors.asType(.float32))
        let (packed, norms) = TurboQuantKernelOps.fusedEncode(
            input: vectors.reshaped([48, d]).asType(.float32),
            rotation: codec.rotation, boundaries: codec.boundaries,
            codebook: codec.codebook, bits: 4, dim: d)
        // Exact bit parity is not achievable: the kernel's per-thread dot
        // product and MLX's matmul reduce in different orders, flipping
        // borderline quantization ties. Assert equivalence where it matters:
        // reconstructions agree and only a small fraction of indices differ.
        let unpackedK = TurboQuantPacking.unpackLowBit(packed, bits: 4, count: d)
        let unpackedR = TurboQuantPacking.unpackLowBit(
            ref.packedIndices.reshaped([48, -1]), bits: 4, count: d)
        let mismatch = MLX.notEqual(unpackedK, unpackedR).asType(.float32).mean()
            .item(Float.self)
        XCTAssertLessThan(mismatch, 0.02, "index mismatch fraction \(mismatch)")

        let recK = matmul(
            codec.codebook[unpackedK.asType(.int32)] * expandedDimensions(norms, axis: -1),
            codec.rotation)
        let recR = codec.decode(ref).reshaped([48, d])
        let a = recK.reshaped([-1])
        let r = recR.asType(.float32).reshaped([-1])
        let cos = ((a * r).sum() / (sqrt((a * a).sum()) * sqrt((r * r).sum()) + 1e-9))
            .item(Float.self)
        XCTAssertGreaterThan(cos, 0.995, "dense kernel reconstruction diverges (cos \(cos))")
    }

    // MARK: - Scaled (key-calibrated) encode kernel parity with the reference codec
    //
    // F-85 follow-up: per-dimension key calibration forced calibrated keys
    // through MSECodec.encode(_:scale:)'s MLX-ops path, dropping decode
    // throughput 124 -> 27.5 tps on Qwen3-1.7B turbo4. These verify the
    // fused Metal kernel variants (fusedEncodeWHTScaled / fusedEncodeScaled)
    // reproduce that math so calibrated keys can go back through the kernel.

    /// The calibrated encode path rotates via MSECodec.rotatedUnit (MLX ops)
    /// and quantizes in the fused kernel; indices must match
    /// MSECodec.encode(_:scale:) exactly since both quantize bit-identical
    /// rotated values with the same boundary predicate. Norms carry a
    /// reduce-order tolerance only.
    func testScaledQuantizePackKernelMatchesCodecExactly() throws {
        for dim in [64, 80, 128] {
            let bits = 4
            let n = 128
            let codec = MSECodec(dim: dim, bits: bits, seed: 42)
            var scaleVals = [Float](repeating: 1.0, count: dim)
            for i in 0 ..< dim {
                scaleVals[i] = 0.25 + 3.75 * Float(i) / Float(dim - 1)
            }
            let scale = MLXArray(scaleVals)
            let vectors = MLXRandom.normal([1, 1, n, dim], key: MLXRandom.key(211))
                .asType(.float32)
            let ref = codec.encode(vectors, scale: scale)

            let flat = vectors.reshaped([n, dim])
            let (rotated, norms) = codec.rotatedUnit(flat)
            let (packed, kernelNorms) = TurboQuantKernelOps.fusedQuantizePackScaled(
                rotated: rotated, rawNorms: norms, scale: scale,
                boundaries: codec.boundaries, codebook: codec.codebook,
                bits: bits, dim: dim)
            eval(packed, kernelNorms)

            let refIdx = TurboQuantPacking.unpackLowBit(
                ref.packedIndices.reshaped([n, -1]), bits: bits, count: dim)
            let kernelIdx = TurboQuantPacking.unpackLowBit(packed, bits: bits, count: dim)
            let mismatches = (refIdx .!= kernelIdx).asType(.float32).sum().item(Float.self)
            XCTAssertEqual(mismatches, 0, "dim=\(dim): indices must match exactly")

            let normRatio = kernelNorms / ref.norms.reshaped([n])
            let maxDev = (normRatio - MLXArray(Float(1.0))).abs().max().item(Float.self)
            XCTAssertLessThan(maxDev, 1e-4, "dim=\(dim): norm deviation \(maxDev)")
        }
    }

    // MARK: - Buffer growth across the step allocation boundary

    func testCacheGrowthAcrossStepBoundary() throws {
        // step = 256: prefill 250 then decode 12 steps so compressed storage
        // re-allocates mid-decode for every layout.
        for (kb, vb) in [(0, 4), (8, 4), (4, 4)] {
            let (b, hq, hkv, d) = (1, 2, 1, 64)
            let cache = TurboQuantKVCache(bits: 4, keyBits: kb, valueBits: vb, seed: 11)
            _ = cache.update(
                keys: MLXRandom.normal([b, hkv, 250, d], key: MLXRandom.key(141)),
                values: MLXRandom.normal([b, hkv, 250, d], key: MLXRandom.key(142)))
            let q = MLXRandom.normal([b, hq, 1, d], key: MLXRandom.key(143))
            for i in 0 ..< 12 {
                let out = cache.compressedAttention(
                    queries: q,
                    keys: MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(UInt64(150 + i))),
                    values: MLXRandom.normal([b, hkv, 1, d], key: MLXRandom.key(UInt64(170 + i))),
                    scale: 0.125)
                XCTAssertFalse(
                    MLX.isNaN(out).any().item(Bool.self),
                    "kb=\(kb) vb=\(vb) offset \(cache.offset)")
            }
            XCTAssertEqual(cache.offset, 262, "kb=\(kb) vb=\(vb)")
        }
    }

}

// MARK: - GQA flash decode (XCTest)

final class TurboQuantGQAFlashTests: XCTestCase {

    // Qwen3.8-27B attention: head dim 256, 24 query heads over 4 KV heads.
    private let (queryHeads, kvHeads, dim) = (24, 4, 256)

    private struct Inputs {
        let queries: MLXArray
        let keys: MLXArray
        let valPacked: MLXArray
        let valNorms: MLXArray
        let codec: MSECodec
    }

    /// Whole buffers of `rows` random rows per KV head, so a kernel that reads
    /// past the token count disagrees with the sliced per-head reference.
    private func inputs(rows: Int, keyDType: DType, seed: UInt64) -> Inputs {
        let codec = MSECodec(dim: dim, bits: 4, seed: 43)
        let values = MLXRandom.normal([kvHeads * rows, dim], key: MLXRandom.key(seed)) * 0.5
        let (packed, norms) = TurboQuantKernelOps.fusedEncodeWHT(
            input: values, whtSigns: codec.whtSigns!, boundaries: codec.boundaries,
            codebook: codec.codebook, bits: 4, dim: dim)
        let keys = (MLXRandom.normal([kvHeads, rows, dim], key: MLXRandom.key(seed + 1)) * 0.5)
            .asType(keyDType)
        let queries =
            MLXRandom.normal([queryHeads, dim], key: MLXRandom.key(seed + 2))
            / MLXArray(Float(dim).squareRoot())
        return Inputs(
            queries: queries, keys: keys,
            valPacked: packed.reshaped([kvHeads, rows, -1]),
            valNorms: norms.reshaped([kvHeads, rows]), codec: codec)
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: MLXArray, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let bound = expected.abs().max().item(Float.self)
        let diff = (actual - expected).abs().max().item(Float.self)
        XCTAssertFalse(MLX.isNaN(actual).any().item(Bool.self), label, file: file, line: line)
        XCTAssertLessThanOrEqual(
            diff, 1e-4 * bound + 1e-6, "\(label): max diff \(diff) over max \(bound)",
            file: file, line: line)
    }

    private let tokenCounts = [1, 5, 63, 64, 65, 200, 1000, 4100, 20_000]

    /// The GQA kernel computes what the per-query-head kernel computes, from
    /// whole step-padded buffers instead of `..<T` slices.
    func testRawKeyGQAMatchesPerHeadKernel() {
        for (index, tokens) in tokenCounts.enumerated() {
            let rows = ((tokens + 255) / 256) * 256
            let input = inputs(rows: rows, keyDType: .bfloat16, seed: UInt64(300 + 10 * index))
            let perHead = TurboQuantKernelOps.turboFlashRawK(
                rotatedQueries: input.queries,
                rawKeys: input.keys[0..., ..<tokens, 0...],
                valPacked: input.valPacked[0..., ..<tokens, 0...],
                valNorms: input.valNorms[0..., ..<tokens],
                valCodebook: input.codec.codebook,
                tokenCount: tokens, repeatCount: queryHeads / kvHeads, valueBits: 4, dim: dim,
                valRotation: input.codec.rotation)
            for headsPerGroup in [1, 2, 3, 6] {
                let gqa = TurboQuantKernelOps.turboFlashGQA(
                    queries: input.queries, keys: .raw(input.keys),
                    valPacked: input.valPacked, valNorms: input.valNorms,
                    valCodebook: input.codec.codebook, valRotation: input.codec.rotation,
                    tokenCount: tokens, repeatCount: queryHeads / kvHeads, valueBits: 4, dim: dim,
                    headsPerGroup: headsPerGroup)
                assertClose(
                    gqa, perHead, "raw K, \(tokens) tokens, \(headsPerGroup) heads per group")
            }
        }
    }

    func testAffineKeyGQAMatchesPerHeadKernel() {
        for (index, tokens) in tokenCounts.enumerated() {
            let rows = ((tokens + 255) / 256) * 256
            let input = inputs(rows: rows, keyDType: .bfloat16, seed: UInt64(500 + 10 * index))
            let quantizedKeys = quantized(input.keys, groupSize: 64, bits: 8)
            let (weights, scales) = (quantizedKeys.wq, quantizedKeys.scales)
            let biases = quantizedKeys.biases!
            let perHead = TurboQuantKernelOps.turboFlashAffineK(
                rotatedQueries: input.queries,
                kWeights: weights[0..., ..<tokens, 0...],
                kScales: scales[0..., ..<tokens, 0...],
                kBiases: biases[0..., ..<tokens, 0...],
                valPacked: input.valPacked[0..., ..<tokens, 0...],
                valNorms: input.valNorms[0..., ..<tokens],
                valCodebook: input.codec.codebook,
                tokenCount: tokens, repeatCount: queryHeads / kvHeads, valueBits: 4, dim: dim,
                kGroup: 64, valRotation: input.codec.rotation)
            for headsPerGroup in [1, 2, 3, 6] {
                let gqa = TurboQuantKernelOps.turboFlashGQA(
                    queries: input.queries,
                    keys: .affine(weights: weights, scales: scales, biases: biases, groupSize: 64),
                    valPacked: input.valPacked, valNorms: input.valNorms,
                    valCodebook: input.codec.codebook, valRotation: input.codec.rotation,
                    tokenCount: tokens, repeatCount: queryHeads / kvHeads, valueBits: 4, dim: dim,
                    headsPerGroup: headsPerGroup)
                assertClose(
                    gqa, perHead, "affine K, \(tokens) tokens, \(headsPerGroup) heads per group")
            }
        }
    }

    /// Through the cache, across a 256-row growth of the compressed buffers:
    /// every decode step stays close to exact attention over the raw K/V.
    func testCacheDecodeMatchesReferenceAtQwenShape() {
        let scale = 1.0 / Float(dim).squareRoot()
        for keyBits in [0, 8] {
            let prefill = 250
            let keys = (MLXRandom.normal([1, kvHeads, prefill, dim], key: MLXRandom.key(81)) * 0.5)
                .asType(.bfloat16)
            let values =
                (MLXRandom.normal([1, kvHeads, prefill, dim], key: MLXRandom.key(82)) * 0.5)
                .asType(.bfloat16)
            let cache = TurboQuantKVCache(bits: 4, keyBits: keyBits, valueBits: 4, seed: 42)
            _ = cache.update(keys: keys, values: values)
            var allKeys = keys
            var allValues = values
            for step in 0 ..< 12 {
                let q = MLXRandom.normal(
                    [1, queryHeads, 1, dim], key: MLXRandom.key(UInt64(90 + step))
                ).asType(.bfloat16)
                let newK =
                    (MLXRandom.normal(
                        [1, kvHeads, 1, dim], key: MLXRandom.key(UInt64(110 + step))) * 0.5)
                    .asType(.bfloat16)
                let newV =
                    (MLXRandom.normal(
                        [1, kvHeads, 1, dim], key: MLXRandom.key(UInt64(130 + step))) * 0.5)
                    .asType(.bfloat16)
                let out = cache.compressedAttention(
                    queries: q, keys: newK, values: newV, scale: scale)
                allKeys = concatenated([allKeys, newK], axis: 2)
                allValues = concatenated([allValues, newV], axis: 2)

                let reference = MLXFast.scaledDotProductAttention(
                    queries: q.asType(.float32), keys: allKeys.asType(.float32),
                    values: allValues.asType(.float32), scale: scale, mask: .none)
                let a = out.asType(.float32).reshaped([-1])
                let r = reference.reshaped([-1])
                let cos = ((a * r).sum() / (sqrt((a * a).sum()) * sqrt((r * r).sum()) + 1e-9))
                    .item(Float.self)
                XCTAssertGreaterThan(
                    cos, 0.97,
                    "keyBits \(keyBits) step \(step) (offset \(cache.offset)): cos \(cos)")
            }
            XCTAssertEqual(cache.offset, prefill + 12)
        }
    }

    /// TurboQuant caches take Qwen 3.5's compiled decode segments (attention
    /// runs untraced between them through `compressedAttention`) and decode
    /// close to the same prompt continued on the unquantized cache.
    func testQwen35TurboQuantDecodeUsesCompiledSegments() throws {
        let json = """
            {
                "model_type": "qwen3_5",
                "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 128,
                "linear_num_value_heads": 2, "linear_num_key_heads": 1,
                "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4, "vocab_size": 32,
                "full_attention_interval": 2,
                "num_experts": 0, "num_experts_per_tok": 0,
                "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16
            }
            """
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: 29)) {
            Qwen35TextModel(configuration)
        }
        var cache = try model.newCache(parameters: nil)
        let prompt = MLXArray((0 ..< 300).map { Int32($0 % 32) }).reshaped(1, 300)
        eval(model(prompt, cache: cache))
        let unquantized = cache.map { $0.copy() }
        let applied = try applyKVCacheConfiguration(
            cache: &cache,
            configuration: KVCacheConfiguration(
                strategy: .turboQuant(
                    try TurboQuantKVCacheConfiguration(
                        keyPrecision: .affineEightBit, valuePrecision: .fourBit)),
                compatibility: .requireAllLayers))
        XCTAssertEqual(applied.convertedLayerCount, 2)
        XCTAssertEqual(model.model.compiledDecodeSegmentCount, 0)

        var turboLogits: [MLXArray] = []
        for token in Int32(1) ... 6 {
            let logits = model(MLXArray([token]).reshaped(1, 1), cache: cache)
            eval(logits)
            turboLogits.append(logits)
        }
        XCTAssertGreaterThan(
            model.model.compiledDecodeSegmentCount, 0,
            "TurboQuant decode left the compiled segments")
        XCTAssertTrue(cache.compactMap { $0 as? TurboQuantKVCache }.allSatisfy(\.isCompressed))

        for (step, token) in (Int32(1) ... 6).enumerated() {
            let reference = model(MLXArray([token]).reshaped(1, 1), cache: unquantized)
            let a = turboLogits[step].reshaped([-1])
            let r = reference.reshaped([-1])
            XCTAssertFalse(MLX.isNaN(a).any().item(Bool.self))
            let cos = ((a * r).sum() / (sqrt((a * a).sum()) * sqrt((r * r).sum()) + 1e-9))
                .item(Float.self)
            XCTAssertGreaterThan(cos, 0.95, "step \(step): cos \(cos)")
        }
    }
}

// MARK: - Positioned rows and multi-query verify

/// Rows written at a device-side position past the committed offset, and the
/// multi-query MMA kernel that attends over them, against dequantize + SDPA.
final class TurboQuantVerifyTests: XCTestCase {
    private let kvHeads = 4
    private let queryHeads = 24
    private let dim = 256
    private let scale = Float(1) / Float(256).squareRoot()

    private func makeCache(keyBits: Int, rows: Int, seed: UInt64) -> TurboQuantKVCache {
        let cache = TurboQuantKVCache(bits: 4, keyBits: keyBits, valueBits: 4)
        if rows > 0 {
            let (k, v) = withRandomState(MLXRandom.RandomState(seed: seed)) {
                (
                    MLXRandom.normal([1, kvHeads, rows, dim]).asType(.bfloat16),
                    MLXRandom.normal([1, kvHeads, rows, dim]).asType(.bfloat16)
                )
            }
            _ = cache.update(keys: k, values: v)
        }
        return cache
    }

    private func block(_ rows: Int, seed: UInt64) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        withRandomState(MLXRandom.RandomState(seed: seed)) {
            (
                MLXRandom.normal([1, queryHeads, rows, dim]).asType(.bfloat16),
                MLXRandom.normal([1, kvHeads, rows, dim]).asType(.bfloat16),
                MLXRandom.normal([1, kvHeads, rows, dim]).asType(.bfloat16)
            )
        }
    }

    /// Max abs error relative to the reference's max magnitude.
    private func relativeError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let a = a.asType(.float32)
        let b = b.asType(.float32)
        return (abs(a - b).max() / maximum(abs(b).max(), MLXArray(Float(1e-6)))).item(Float.self)
    }

    func testVerifyKernelMatchesDequantizedReference() throws {
        for keyBits in [0, 8] {
            for (rows, s) in [(1, 8), (31, 8), (32, 8), (33, 5), (200, 8), (1000, 13), (4100, 8)] {
                let cache = makeCache(keyBits: keyBits, rows: rows, seed: UInt64(rows))
                let (q, k, v) = block(s, seed: UInt64(rows + 7))
                // A lazy position, and three rows of slack past the block.
                let position = MLXArray(Int32(rows - 1)) + MLXArray(Int32(1))
                let visible = rows + s + 3
                let out = cache.verifyAttention(
                    queries: q, keys: k, values: v, position: position, visibleLength: visible,
                    scale: scale)
                let reference = cache.referenceAttention(
                    queries: q.asType(.float32), position: position, visibleLength: visible,
                    scale: scale)
                XCTAssertEqual(out.shape, [1, queryHeads, s, dim])
                XCTAssertEqual(out.dtype, .bfloat16)
                XCTAssertEqual(cache.offset, rows, "a positioned write leaves the offset")
                let error = relativeError(out, reference)
                XCTAssertLessThan(error, 2e-2, "keyBits \(keyBits) rows \(rows) S \(s)")
            }
        }
    }

    /// A pass sees the rows an earlier, uncommitted pass wrote before its own
    /// position, and never the ones after its rows.
    func testScratchRowsAreVisibleOnlyBelowEachRow() throws {
        for keyBits in [0, 8] {
            let rows = 300
            let a = makeCache(keyBits: keyBits, rows: rows, seed: 3)
            let b = makeCache(keyBits: keyBits, rows: rows, seed: 3)
            let first = block(8, seed: 4)
            let second = block(8, seed: 5)
            // a: two rounds in flight; the first commits three rows.
            _ = a.verifyAttention(
                queries: first.q, keys: first.k, values: first.v,
                position: MLXArray(Int32(rows)), visibleLength: rows + 8, scale: scale)
            a.commitRows(count: rows + 3)
            let outA = a.verifyAttention(
                queries: second.q, keys: second.k, values: second.v,
                position: MLXArray(Int32(rows + 3)), visibleLength: rows + 11, scale: scale)
            // b: only the three accepted rows, then the same second round.
            _ = b.verifyAttention(
                queries: first.q[0..., 0..., ..<3, 0...], keys: first.k[0..., 0..., ..<3, 0...],
                values: first.v[0..., 0..., ..<3, 0...],
                position: MLXArray(Int32(rows)), visibleLength: rows + 3, scale: scale)
            b.commitRows(count: rows + 3)
            let outB = b.verifyAttention(
                queries: second.q, keys: second.k, values: second.v,
                position: MLXArray(Int32(rows + 3)), visibleLength: rows + 11, scale: scale)
            XCTAssertEqual(
                (outA.asType(.float32) - outB.asType(.float32)).abs().max().item(Float.self), 0,
                "keyBits \(keyBits)")
        }
    }

    /// Growth past the allocation keeps rows written past the offset.
    func testGrowthKeepsRowsPastTheOffset() throws {
        for keyBits in [0, 8] {
            let cache = makeCache(keyBits: keyBits, rows: 250, seed: 9)
            let first = block(4, seed: 10)
            _ = cache.verifyAttention(
                queries: first.q, keys: first.k, values: first.v,
                position: MLXArray(Int32(250)), visibleLength: 254, scale: scale)
            // In flight: rows 250 ..< 254 are written, nothing is committed.
            let second = block(8, seed: 11)
            _ = cache.verifyAttention(
                queries: second.q, keys: second.k, values: second.v,
                position: MLXArray(Int32(254)), visibleLength: 262, scale: scale)
            cache.commitRows(count: 262)
            let (keys, _) = cache.dequantizedRows(262)
            let written = keys[0..., 0..., 250 ..< 254, 0...]
            let error = relativeError(written, first.k)
            XCTAssertLessThan(error, 2e-2, "keyBits \(keyBits)")
        }
    }

    /// A causal prefill chunk over a compressed cache takes the kernel.
    func testCausalChunkOverCompressedCacheMatchesReference() throws {
        for keyBits in [0, 8] {
            let cache = makeCache(keyBits: keyBits, rows: 700, seed: 21)
            let one = block(1, seed: 22)
            _ = cache.compressedAttention(
                queries: one.q, keys: one.k, values: one.v, scale: scale, mask: .none)
            XCTAssertTrue(cache.isCompressed)
            let chunk = block(37, seed: 23)
            let out = cache.compressedAttention(
                queries: chunk.q, keys: chunk.k, values: chunk.v, scale: scale, mask: .causal)
            XCTAssertEqual(cache.offset, 738)
            let reference = cache.referenceAttention(
                queries: chunk.q.asType(.float32), position: MLXArray(Int32(701)),
                visibleLength: 738, scale: scale)
            XCTAssertLessThan(relativeError(out, reference), 2e-2, "keyBits \(keyBits)")
        }
    }

    func testStateSlicesOnTheTokenAxisAndRoundTrips() throws {
        for keyBits in [0, 8] {
            let cache = makeCache(keyBits: keyBits, rows: 100, seed: 31)
            let one = block(1, seed: 32)
            _ = cache.compressedAttention(
                queries: one.q, keys: one.k, values: one.v, scale: scale, mask: .none)
            let state = cache.state
            XCTAssertEqual(state.count, keyBits == 8 ? 5 : 3)
            for array in state {
                XCTAssertEqual(array.ndim, 4)
                XCTAssertEqual(array.dim(-2), 101)
            }
            XCTAssertEqual(cache.metaState.count, 6)

            let restored = TurboQuantKVCache(bits: 4, keyBits: keyBits, valueBits: 4)
            restored.state = state
            restored.metaState = cache.metaState
            XCTAssertEqual(restored.offset, 101)
            XCTAssertEqual(restored.keyGroupSize, cache.keyGroupSize)
            // Older prompt caches stored norms without the trailing axis.
            let legacy = TurboQuantKVCache(bits: 4, keyBits: keyBits, valueBits: 4)
            legacy.state = state.dropLast() + [state.last!.squeezed(axis: -1)]
            let next = block(8, seed: 33)
            var outputs: [MLXArray] = []
            for c in [cache, restored, legacy] {
                outputs.append(
                    c.verifyAttention(
                        queries: next.q, keys: next.k, values: next.v,
                        position: MLXArray(Int32(101)), visibleLength: 109, scale: scale))
            }
            XCTAssertEqual(abs(outputs[0] - outputs[1]).max().item(Float.self), 0)
            XCTAssertEqual(abs(outputs[0] - outputs[2]).max().item(Float.self), 0)
        }
    }

    func testCopyIsIndependent() throws {
        let cache = makeCache(keyBits: 8, rows: 64, seed: 41)
        let one = block(1, seed: 42)
        _ = cache.compressedAttention(
            queries: one.q, keys: one.k, values: one.v, scale: scale, mask: .none)
        let before = cache.dequantizedRows(65).keys
        let copy = cache.copy() as! TurboQuantKVCache
        XCTAssertFalse(cache.innerState().isEmpty)
        let more = block(8, seed: 43)
        _ = copy.verifyAttention(
            queries: more.q, keys: more.k, values: more.v, position: MLXArray(Int32(60)),
            visibleLength: 68, scale: scale)
        copy.commitRows(count: 68)
        XCTAssertEqual(cache.offset, 65)
        XCTAssertEqual(
            abs(cache.dequantizedRows(65).keys - before).max().item(Float.self), 0)
        XCTAssertGreaterThan(
            abs(copy.dequantizedRows(65).keys - before).max().item(Float.self), 0)
    }

    /// Qwen 3.5's verify pass writes and reads TurboQuant layers through the
    /// kernel and tracks the same pass over the unquantized cache.
    func testQwen35DFlash2VerifyOverTurboQuantTracksThePlainCache() throws {
        let configuration = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(Self.tinyQwen35.utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: 29)) {
            Qwen35TextModel(configuration)
        }
        try model.update(
            parameters: model.parameters().mapValues { $0.asType(.bfloat16) }, verify: [])
        for keyBits in [0, 8] {
            var cache = try model.newCache(parameters: nil)
            let prompt = MLXArray((0 ..< 300).map { Int32($0 % 97) }).reshaped(1, 300)
            eval(model(prompt, cache: cache))
            let plain = cache.map { $0.copy() }
            _ = try applyKVCacheConfiguration(
                cache: &cache,
                configuration: KVCacheConfiguration(
                    strategy: .turboQuant(
                        try TurboQuantKVCacheConfiguration(
                            keyPrecision: keyBits == 8 ? .affineEightBit : .fp16,
                            valuePrecision: .fourBit)),
                    compatibility: .requireAllLayers))
            XCTAssertTrue(model.dflash2SupportsCache(cache))
            let step = MLXArray([Int32(5)]).reshaped(1, 1)
            eval(model(step, cache: cache), model(step, cache: plain))

            let request = DFlash2VerifyRequest(
                tokens: MLXArray((0 ..< 8).map { Int32($0 * 7 % 97) }).reshaped(1, 8),
                position: MLXArray([Int32(301)]), positionUpperBound: 303, captureLayers: [1, 2])
            let turbo = model.dflash2Verify(request, cache: cache)
            let reference = model.dflash2Verify(request, cache: plain)
            eval(turbo.logits, reference.logits)
            XCTAssertEqual(cache.map(\.offset), plain.map(\.offset))
            for row in 0 ..< 8 {
                let a = turbo.logits[0, row].asType(.float32)
                let r = reference.logits[0, row].asType(.float32)
                let cos = ((a * r).sum() / (sqrt((a * a).sum()) * sqrt((r * r).sum()) + 1e-9))
                    .item(Float.self)
                XCTAssertGreaterThan(cos, 0.95, "keyBits \(keyBits) row \(row): cos \(cos)")
            }
        }
        let standard = TurboQuantKVCache(bits: 4)
        var cache = try model.newCache(parameters: nil)
        cache[1] = standard
        XCTAssertFalse(model.dflash2SupportsCache(cache), "4-bit keys have no positioned rows")
    }

    static let tinyQwen35 = """
        {
            "model_type": "qwen3_5",
            "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 128,
            "linear_num_value_heads": 2, "linear_num_key_heads": 1,
            "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4, "vocab_size": 128,
            "full_attention_interval": 2,
            "num_experts": 0, "num_experts_per_tok": 0,
            "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16
        }
        """

    /// `compress()` stores the raw rows the way the first decode step would.
    func testCompressMatchesTheFirstDecodeStep() throws {
        for keyBits in [0, 8] {
            let a = makeCache(keyBits: keyBits, rows: 300, seed: 61)
            let b = makeCache(keyBits: keyBits, rows: 300, seed: 61)
            a.compress()
            XCTAssertTrue(a.isCompressed)
            XCTAssertEqual(a.offset, 300)
            let one = block(1, seed: 62)
            let outA = a.compressedAttention(
                queries: one.q, keys: one.k, values: one.v, scale: scale, mask: .none)
            let outB = b.compressedAttention(
                queries: one.q, keys: one.k, values: one.v, scale: scale, mask: .none)
            XCTAssertEqual(abs(outA - outB).max().item(Float.self), 0, "keyBits \(keyBits)")
            let empty = makeCache(keyBits: keyBits, rows: 0, seed: 0)
            empty.compress()
            XCTAssertFalse(empty.isCompressed)
        }
    }

    /// The first call on an empty cache compresses as it encodes.
    func testFirstEncodeOnAnEmptyCacheMarksItCompressed() throws {
        let cache = makeCache(keyBits: 0, rows: 0, seed: 0)
        let one = block(1, seed: 51)
        _ = cache.compressedAttention(
            queries: one.q, keys: one.k, values: one.v, scale: scale, mask: .none)
        XCTAssertTrue(cache.isCompressed)
        XCTAssertEqual(cache.state.count, 3)
    }
}

// MARK: - Decode microbenchmark (opt-in)

/// One attention layer's decode step at the Qwen3.8-27B shape, TurboQuant
/// against bf16 SDPA. Opt-in: `TEST_RUNNER_TURBOQUANT_DECODE_BENCH=1`;
/// `TEST_RUNNER_TURBOQUANT_DECODE_BENCH_CONTEXTS=8192,32768` picks the
/// contexts. Each timed iteration runs one call per attention layer and one eval.
final class TurboQuantDecodeMicrobench: XCTestCase {

    private let (queryHeads, kvHeads, dim) = (24, 4, 256)
    private let layers = 16

    override func setUpWithError() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["TURBOQUANT_DECODE_BENCH"] == "1",
            "set TEST_RUNNER_TURBOQUANT_DECODE_BENCH=1")
    }

    private func time(_ label: String, iterations: Int = 20, _ body: () -> [MLXArray]) -> Double {
        for _ in 0 ..< 3 { eval(body()) }
        let start = Date()
        for _ in 0 ..< iterations { eval(body()) }
        let ms = Date().timeIntervalSince(start) * 1000 / Double(iterations)
        print("[DECODE-BENCH] \(label): \(String(format: "%.3f", ms)) ms per \(layers) layers")
        return ms
    }

    private var contexts: [Int] {
        (ProcessInfo.processInfo.environment["TURBOQUANT_DECODE_BENCH_CONTEXTS"]
            ?? "8192,32768,65536")
            .split(separator: ",").compactMap { Int($0) }
    }

    func testAttentionKernels() {
        let codec = MSECodec(dim: dim, bits: 4, seed: 43)
        for tokens in contexts {
            let rows = ((tokens + 255) / 256) * 256
            let q = (MLXRandom.normal([1, queryHeads, 1, dim]) * 0.1).asType(.bfloat16)
            let flatQ = (q.asType(.float32) / MLXArray(Float(dim).squareRoot()))
                .reshaped([queryHeads, dim])
            let keys = (MLXRandom.normal([1, kvHeads, rows, dim]) * 0.5).asType(.bfloat16)
            let values = (MLXRandom.normal([1, kvHeads, rows, dim]) * 0.5).asType(.bfloat16)
            let (packed, norms) = TurboQuantKernelOps.fusedEncodeWHT(
                input: values.reshaped([-1, dim]).asType(.float32), whtSigns: codec.whtSigns!,
                boundaries: codec.boundaries, codebook: codec.codebook, bits: 4, dim: dim)
            let valPacked = packed.reshaped([kvHeads, rows, -1])
            let valNorms = norms.reshaped([kvHeads, rows])
            let quantizedKeys = quantized(
                keys.reshaped([kvHeads, rows, dim]), groupSize: 64, bits: 8)
            let rawKeys = keys.reshaped([kvHeads, rows, dim])
            eval(
                q, flatQ, keys, values, valPacked, valNorms, quantizedKeys.wq,
                quantizedKeys.scales, quantizedKeys.biases!)

            let sdpa = time("T=\(tokens) bf16 SDPA") {
                (0 ..< layers).map { _ in
                    MLXFast.scaledDotProductAttention(
                        queries: q, keys: keys[.ellipsis, ..<tokens, 0...],
                        values: values[.ellipsis, ..<tokens, 0...],
                        scale: 1 / Float(dim).squareRoot(), mask: .none)
                }
            }
            func gqa(
                _ keys: TurboQuantKernelOps.GQAKeys, nsg: Int? = nil, maxBlocks: Int? = nil,
                hps: Int? = nil
            ) -> [MLXArray] {
                (0 ..< layers).map { _ in
                    TurboQuantKernelOps.turboFlashGQA(
                        queries: flatQ, keys: keys, valPacked: valPacked, valNorms: valNorms,
                        valCodebook: codec.codebook, valRotation: codec.rotation,
                        tokenCount: tokens, repeatCount: queryHeads / kvHeads, valueBits: 4,
                        dim: dim, simdGroups: nsg, maxBlocks: maxBlocks, headsPerGroup: hps)
                }
            }
            let affine = TurboQuantKernelOps.GQAKeys.affine(
                weights: quantizedKeys.wq, scales: quantizedKeys.scales,
                biases: quantizedKeys.biases!, groupSize: 64)
            let gqa8 = time("T=\(tokens) turbo8v4 GQA") { gqa(affine) }
            let gqa0 = time("T=\(tokens) turbo0v4 GQA") { gqa(.raw(rawKeys)) }
            let legacy8 = time("T=\(tokens) turbo8v4 per-head (sliced)") {
                (0 ..< layers).map { _ in
                    TurboQuantKernelOps.turboFlashAffineK(
                        rotatedQueries: flatQ,
                        kWeights: quantizedKeys.wq[0..., ..<tokens, 0...],
                        kScales: quantizedKeys.scales[0..., ..<tokens, 0...],
                        kBiases: quantizedKeys.biases![0..., ..<tokens, 0...],
                        valPacked: valPacked[0..., ..<tokens, 0...],
                        valNorms: valNorms[0..., ..<tokens], valCodebook: codec.codebook,
                        tokenCount: tokens, repeatCount: queryHeads / kvHeads, valueBits: 4,
                        dim: dim, kGroup: 64, valRotation: codec.rotation)
                }
            }
            print(
                "[DECODE-BENCH] T=\(tokens) ratio vs SDPA: turbo8v4 GQA \(String(format: "%.2f", gqa8 / sdpa))x, turbo0v4 GQA \(String(format: "%.2f", gqa0 / sdpa))x, turbo8v4 per-head \(String(format: "%.2f", legacy8 / sdpa))x"
            )
            for hps in [1, 2, 3, 6] {
                for nsg in [1, 2, 4] {
                    for maxBlocks in [64, 128] {
                        _ = time(
                            "T=\(tokens) turbo8v4 GQA hps=\(hps) nsg=\(nsg) maxBlocks=\(maxBlocks)",
                            iterations: 10
                        ) {
                            gqa(affine, nsg: nsg, maxBlocks: maxBlocks, hps: hps)
                        }
                        _ = time(
                            "T=\(tokens) turbo0v4 GQA hps=\(hps) nsg=\(nsg) maxBlocks=\(maxBlocks)",
                            iterations: 10
                        ) {
                            gqa(.raw(rawKeys), nsg: nsg, maxBlocks: maxBlocks, hps: hps)
                        }
                    }
                }
            }
        }
    }

    /// One DFlash2 verify pass's attention (an 8-row block at the end of the
    /// context, under the position mask): bf16 SDPA over the plain cache
    /// against the TurboQuant kernel and its dequantize + SDPA reference.
    func testVerifyAttention() {
        let s = 8
        let scale = 1 / Float(dim).squareRoot()
        for tokens in contexts {
            let rows = ((tokens + s + 255) / 256) * 256
            let q = (MLXRandom.normal([1, queryHeads, s, dim]) * 0.1).asType(.bfloat16)
            let keys = (MLXRandom.normal([1, kvHeads, rows, dim]) * 0.5).asType(.bfloat16)
            let values = (MLXRandom.normal([1, kvHeads, rows, dim]) * 0.5).asType(.bfloat16)
            let visible = tokens + s
            let columns = MLXArray(Int32(0) ..< Int32(visible)).expandedDimensions(axis: 0)
            let positions = (MLXArray(Int32(tokens)) + MLXArray(Int32(0) ..< Int32(s)))
                .expandedDimensions(axis: 1)
            let mask = columns .< (positions + 1)
            eval(q, keys, values, mask)
            let sdpa = time("verify T=\(tokens) bf16 SDPA", iterations: 10) {
                (0 ..< layers).map { _ in
                    MLXFast.scaledDotProductAttention(
                        queries: q, keys: keys[.ellipsis, ..<visible, 0...],
                        values: values[.ellipsis, ..<visible, 0...], scale: scale,
                        mask: .array(mask))
                }
            }
            for keyBits in [8, 0] {
                let caches = (0 ..< layers).map { _ -> TurboQuantKVCache in
                    let c = TurboQuantKVCache(bits: 4, keyBits: keyBits, valueBits: 4)
                    _ = c.update(
                        keys: keys[.ellipsis, ..<tokens, 0...],
                        values: values[.ellipsis, ..<tokens, 0...])
                    return c
                }
                let newK = keys[.ellipsis, tokens ..< visible, 0...]
                let newV = values[.ellipsis, tokens ..< visible, 0...]
                let position = MLXArray(Int32(tokens))
                eval(
                    caches.map {
                        $0.verifyAttention(
                            queries: q, keys: newK, values: newV, position: position,
                            visibleLength: visible, scale: scale)
                    })
                let kernel = time("verify T=\(tokens) turbo\(keyBits)v4 kernel", iterations: 10) {
                    caches.map {
                        $0.verifyAttention(
                            queries: q, keys: newK, values: newV, position: position,
                            visibleLength: visible, scale: scale)
                    }
                }
                let reference = time(
                    "verify T=\(tokens) turbo\(keyBits)v4 dequantize + SDPA", iterations: 3
                ) {
                    caches.map {
                        $0.referenceAttention(
                            queries: q, position: position, visibleLength: visible, scale: scale)
                    }
                }
                print(
                    "[DECODE-BENCH] verify T=\(tokens) turbo\(keyBits)v4: kernel \(String(format: "%.2f", kernel / sdpa))x SDPA, reference \(String(format: "%.2f", reference / sdpa))x"
                )
            }
        }
    }

    /// The one-time switch-over: 16 layers' first decode step on freshly
    /// converted caches, which compresses each layer's whole prefill.
    func testSwitchOver() {
        let scale = 1 / Float(dim).squareRoot()
        for tokens in contexts {
            let keys = (MLXRandom.normal([1, kvHeads, tokens, dim]) * 0.5).asType(.bfloat16)
            let values = (MLXRandom.normal([1, kvHeads, tokens, dim]) * 0.5).asType(.bfloat16)
            let q = (MLXRandom.normal([1, queryHeads, 1, dim]) * 0.1).asType(.bfloat16)
            let newK = (MLXRandom.normal([1, kvHeads, 1, dim]) * 0.5).asType(.bfloat16)
            let newV = (MLXRandom.normal([1, kvHeads, 1, dim]) * 0.5).asType(.bfloat16)
            eval(keys, values, q, newK, newV)
            for keyBits in [8, 0] {
                var samples: [Double] = []
                for _ in 0 ..< 3 {
                    let caches = (0 ..< layers).map { _ -> TurboQuantKVCache in
                        let c = TurboQuantKVCache(bits: 4, keyBits: keyBits, valueBits: 4, seed: 42)
                        _ = c.update(keys: keys, values: values)
                        return c
                    }
                    eval(caches.flatMap(\.state))
                    let start = Date()
                    eval(
                        caches.map {
                            $0.compressedAttention(
                                queries: q, keys: newK, values: newV, scale: scale)
                        })
                    samples.append(Date().timeIntervalSince(start) * 1000)
                }
                print(
                    "[DECODE-BENCH] switch-over T=\(tokens) turbo\(keyBits)v4: "
                        + samples.map { String(format: "%.1f", $0) }.joined(separator: " / ")
                        + " ms per \(layers) layers")
            }
        }
    }

    /// The whole per-layer decode step through each cache: append one token,
    /// then attend. TurboQuant adds its value encode and key quantization.
    func testCacheStep() {
        for tokens in contexts { cacheStep(tokens: tokens) }
    }

    private func cacheStep(tokens: Int) {
        let scale = 1 / Float(dim).squareRoot()
        let keys = (MLXRandom.normal([1, kvHeads, tokens, dim]) * 0.5).asType(.bfloat16)
        let values = (MLXRandom.normal([1, kvHeads, tokens, dim]) * 0.5).asType(.bfloat16)
        let q = (MLXRandom.normal([1, queryHeads, 1, dim]) * 0.1).asType(.bfloat16)
        let newK = (MLXRandom.normal([1, kvHeads, 1, dim]) * 0.5).asType(.bfloat16)
        let newV = (MLXRandom.normal([1, kvHeads, 1, dim]) * 0.5).asType(.bfloat16)
        eval(keys, values, q, newK, newV)

        let simple = (0 ..< layers).map { _ -> KVCacheSimple in
            let c = KVCacheSimple()
            _ = c.update(keys: keys, values: values)
            return c
        }
        _ = time("step T=\(tokens) bf16 KVCacheSimple + SDPA", iterations: 30) {
            simple.map { cache in
                let (k, v) = cache.update(keys: newK, values: newV)
                return MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: .none)
            }
        }
        for keyBits in [8, 0] {
            let turbo = (0 ..< layers).map { _ -> TurboQuantKVCache in
                let c = TurboQuantKVCache(bits: 4, keyBits: keyBits, valueBits: 4, seed: 42)
                _ = c.update(keys: keys, values: values)
                return c
            }
            _ = time("step T=\(tokens) turbo\(keyBits)v4", iterations: 30) {
                turbo.map { cache in
                    cache.compressedAttention(queries: q, keys: newK, values: newV, scale: scale)
                }
            }
        }
    }
}
