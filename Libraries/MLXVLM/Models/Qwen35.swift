//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/25.
//
//  Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen3_5
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

private enum Qwen35VLError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
}

/// Per batch row, how far text rotates past its cache row: the images
/// before it span fewer positions than tokens.
private let ropeDeltasKey = LMOutput.Key<MLXArray>(
    "qwen35.ropeDeltas")

// MARK: - Configuration

public struct Qwen35Configuration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public var modelType: String = ""
        public var hiddenSize: Int = 4096
        public var hiddenLayers: Int = 32
        public var intermediateSize: Int = 14_336
        public var attentionHeads: Int = 32
        public var kvHeads: Int = 8
        public var linearNumValueHeads: Int = 64
        public var linearNumKeyHeads: Int = 16
        public var linearKeyHeadDim: Int = 192
        public var linearValueHeadDim: Int = 128
        public var linearConvKernelDim: Int = 4
        public var rmsNormEps: Float = 1e-6
        public var vocabularySize: Int = 248_320
        public var ropeTheta: Float = 100_000.0
        public var partialRotaryFactor: Float = 0.25
        public var maxPositionEmbeddings: Int = 131_072
        public var tieWordEmbeddings: Bool = false
        public var attentionBias: Bool = false
        public var headDim: Int?
        public var ropeParameters: [String: StringOrNumber]?
        public var fullAttentionInterval: Int = 4
        public var mtpNumHiddenLayers: Int = 0
        public var mtpUseDedicatedEmbeddings: Bool = false

        // MoE fields
        public var numExperts: Int = 0
        public var numExpertsPerTok: Int = 0
        public var decoderSparseStep: Int = 1
        public var sharedExpertIntermediateSize: Int = 0
        public var moeIntermediateSize: Int = 0
        public var normTopkProb: Bool = true

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case linearNumValueHeads = "linear_num_value_heads"
            case linearNumKeyHeads = "linear_num_key_heads"
            case linearKeyHeadDim = "linear_key_head_dim"
            case linearValueHeadDim = "linear_value_head_dim"
            case linearConvKernelDim = "linear_conv_kernel_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case maxPositionEmbeddings = "max_position_embeddings"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case headDim = "head_dim"
            case ropeParameters = "rope_parameters"
            case fullAttentionInterval = "full_attention_interval"
            case mtpNumHiddenLayers = "mtp_num_hidden_layers"
            case mtpUseDedicatedEmbeddings = "mtp_use_dedicated_embeddings"
            case numExperts = "num_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case decoderSparseStep = "decoder_sparse_step"
            case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case normTopkProb = "norm_topk_prob"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)

            self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
            self.intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14_336
            self.attentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            self.linearNumValueHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
            self.linearNumKeyHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
            self.linearKeyHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
            self.linearValueHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
            self.linearConvKernelDim =
                try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
            self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            self.vocabularySize =
                try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248_320
            self.maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
            self.tieWordEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            self.attentionBias =
                try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            self.fullAttentionInterval =
                try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
            self.mtpNumHiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0
            self.mtpUseDedicatedEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .mtpUseDedicatedEmbeddings)
                ?? false

            self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
            self.numExpertsPerTok =
                try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
            self.decoderSparseStep =
                try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
            self.sharedExpertIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
            self.moeIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
            self.normTopkProb =
                try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true

            let defaultRopeParameters: [String: StringOrNumber] = [
                "type": .string("default"),
                "mrope_section": .ints([11, 11, 10]),
                "rope_theta": .float(100_000.0),
                "partial_rotary_factor": .float(0.25),
            ]

            var decodedRope = try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)

            if decodedRope == nil {
                let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                let partial = try container.decodeIfPresent(
                    Float.self, forKey: .partialRotaryFactor)
                if ropeTheta != nil || partial != nil {
                    decodedRope = defaultRopeParameters
                    if let ropeTheta {
                        decodedRope?["rope_theta"] = .float(ropeTheta)
                    }
                    if let partial {
                        decodedRope?["partial_rotary_factor"] = .float(partial)
                    }
                }
            }

            if var decodedRope {
                if decodedRope["type"] == nil, let ropeType = decodedRope["rope_type"] {
                    decodedRope["type"] = ropeType
                }
                self.ropeParameters = decodedRope
                self.ropeTheta = decodedRope["rope_theta"]?.asFloat() ?? 100_000.0
                self.partialRotaryFactor = decodedRope["partial_rotary_factor"]?.asFloat() ?? 0.25
            } else {
                self.ropeParameters = defaultRopeParameters
                self.ropeTheta = 100_000.0
                self.partialRotaryFactor = 0.25
            }

            if self.headDim == nil {
                self.headDim = self.hiddenSize / self.attentionHeads
            }
        }
    }

    public typealias VisionConfiguration = Qwen3VLConfiguration.VisionConfiguration

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _ignoreIndex: Int?
    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? 248_056 }
    private let _videoTokenId: Int?
    public var videoTokenId: Int { _videoTokenId ?? 248_057 }
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    private let _videoTokenIndex: Int?
    public var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenId }
    private let _visionStartTokenId: Int?
    public var visionStartTokenId: Int { _visionStartTokenId ?? 248_045 }
    private let _visionEndTokenId: Int?
    public var visionEndTokenId: Int { _visionEndTokenId ?? 248_046 }
    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? textConfiguration.vocabularySize }
    private let _eosTokenId: IntOrIntArray?
    public var eosTokenId: [Int]? { _eosTokenId?.values }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenId = "vision_start_token_id"
        case _visionEndTokenId = "vision_end_token_id"
        case _vocabSize = "vocab_size"
        case _eosTokenId = "eos_token_id"
    }
}

// MARK: - Language

public enum Qwen35Language {

    final class RotaryEmbedding {
        private let invFreq: MLXArray
        private let mropeIndices: MLXArray

        init(dim: Int, base: Float, mropeSection: [Int]) {
            let safeDim = max(1, dim)
            var freq = MLXArray(stride(from: 0, to: safeDim, by: 2)).asType(.float32)
            freq = freq / Float(safeDim)
            self.invFreq = 1.0 / pow(MLXArray(base), freq)

            let sections = mropeSection.count >= 3 ? mropeSection : [11, 11, 10]
            var indices = [Int32](repeating: 0, count: freq.dim(0))
            for (dimension, offset) in [(1, 1), (2, 2)] {
                let end = min(sections[dimension] * 3, indices.count)
                for index in stride(from: offset, to: end, by: 3) {
                    indices[index] = Int32(dimension)
                }
            }
            self.mropeIndices = MLXArray(indices).reshaped(1, 1, 1, -1)
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            takeAlong(freqs, mropeIndices, axis: 0).squeezed(axis: 0)
        }

        func callAsFunction(x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
            var positionIds = positionIds
            if positionIds.ndim == 2 {
                positionIds = broadcast(
                    positionIds[.newAxis, 0..., 0...],
                    to: [3, positionIds.dim(0), positionIds.dim(1)])
            }

            let pos = positionIds.asType(.float32)
            var inv = invFreq.asType(.float32)
            inv = inv[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * inv
            freqs = applyInterleavedMRope(freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            return (cos(emb).asType(x.dtype), sin(emb).asType(x.dtype))
        }
    }

    static func applyMultimodalRotaryPosEmb(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let cos = expandedDimensions(cos, axis: 1)
        let sin = expandedDimensions(sin, axis: 1)

        let rotaryDim = cos.dim(-1)
        let qDim = q.dim(-1)
        let kDim = k.dim(-1)

        let qRot = q[.ellipsis, ..<rotaryDim]
        let kRot = k[.ellipsis, ..<rotaryDim]

        let qEmbedded = (qRot * cos) + (QwenVL.rotateHalf(qRot) * sin)
        let kEmbedded = (kRot * cos) + (QwenVL.rotateHalf(kRot) * sin)

        let qOut: MLXArray
        if rotaryDim < qDim {
            qOut = concatenated([qEmbedded, q[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            qOut = qEmbedded
        }

        let kOut: MLXArray
        if rotaryDim < kDim {
            kOut = concatenated([kEmbedded, k[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            kOut = kEmbedded
        }

        return (qOut, kOut)
    }

    final class RMSNormGated: Module {
        @ParameterInfo(key: "weight") var weight: MLXArray
        let eps: Float

        init(dimensions: Int, eps: Float = 1e-6) {
            self.eps = eps
            _weight.wrappedValue = MLXArray.ones([dimensions])
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
            var x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
            if let gate {
                x = x * silu(gate)
            }
            return x
        }
    }

    final class Attention: Module {
        let numKeyValueHeads: Int
        let numAttentionHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.numKeyValueHeads = args.kvHeads
            self.numAttentionHeads = args.attentionHeads
            self.headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
            self.scale = pow(Float(headDim), -0.5)

            _qProj.wrappedValue = Linear(
                args.hiddenSize, numAttentionHeads * headDim * 2, bias: args.attentionBias)
            _kProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _vProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _oProj.wrappedValue = Linear(
                numAttentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

            let mrope = args.ropeParameters?["mrope_section"]?.asInts() ?? [11, 11, 10]
            let rotaryDim = Int(Float(headDim) * args.partialRotaryFactor)
            self.rotaryEmbedding = RotaryEmbedding(
                dim: rotaryDim, base: args.ropeTheta, mropeSection: mrope)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)

            let qProjOutput = qProj(x)
            let qSplit = qProjOutput.reshaped(B, L, numAttentionHeads, -1).split(parts: 2, axis: -1)
            var queries = qSplit[0]
            let gate = qSplit[1].reshaped(B, L, -1)

            var keys = kProj(x)
            var values = vProj(x)

            queries = qNorm(queries).transposed(0, 2, 1, 3)
            keys = kNorm(keys.reshaped(B, L, numKeyValueHeads, -1)).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

            var kvSeqLen = keys.dim(-2)
            var positionIds = positionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                kvSeqLen += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + L, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [B, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else if let cache {
                kvSeqLen += cache.offset + 1
            }

            let (cosValues, sinValues) = rotaryEmbedding(x: values, positionIds: positionIds!)
            (queries, keys) = applyMultimodalRotaryPosEmb(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                attentionMask = .array(mask[.ellipsis, 0 ..< kvSeqLen])
            } else {
                attentionMask = .none
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return oProj(output * sigmoid(gate))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    open class GatedDeltaNet: Module {
        let hiddenSize: Int
        let numVHeads: Int
        let numKHeads: Int
        let headKDim: Int
        let headVDim: Int
        let keyDim: Int
        let valueDim: Int
        let convKernelSize: Int
        let convDim: Int

        @ModuleInfo(key: "conv1d") var conv1d: Conv1d
        @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
        @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
        @ModuleInfo(key: "in_proj_b") var inProjB: Linear
        @ModuleInfo(key: "in_proj_a") var inProjA: Linear

        // Inference-only physical projection. The four registered modules
        // remain as views so checkpoint and adapter paths stay unchanged.
        private let fusedInputProjection = FusedQuantizedLinearProjectionCache()
        var fusedInputProjectionEnabled = qwen35FourGDNEnabled

        @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
        @ParameterInfo(key: "A_log") var aLog: MLXArray

        @ModuleInfo(key: "norm") var norm: RMSNormGated
        @ModuleInfo(key: "out_proj") var outProj: Linear

        public init(_ args: Qwen35Configuration.TextConfiguration) {
            self.hiddenSize = args.hiddenSize
            self.numVHeads = args.linearNumValueHeads
            self.numKHeads = args.linearNumKeyHeads
            self.headKDim = args.linearKeyHeadDim
            self.headVDim = args.linearValueHeadDim
            self.keyDim = headKDim * numKHeads
            self.valueDim = headVDim * numVHeads
            self.convKernelSize = args.linearConvKernelDim
            self.convDim = keyDim * 2 + valueDim

            precondition(
                numVHeads % numKHeads == 0,
                "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
            )

            _conv1d.wrappedValue = Conv1d(
                inputChannels: convDim,
                outputChannels: convDim,
                kernelSize: convKernelSize,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: convDim,
                bias: false
            )

            _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
            _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
            _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
            _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

            _dtBias.wrappedValue = MLXArray.ones([numVHeads])
            let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
            _aLog.wrappedValue = log(a)

            _norm.wrappedValue = RMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
            _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)
            super.init()
        }

        @discardableResult
        open override func update(
            parameters: ModuleParameters, verify: VerifyUpdate,
            path: [String] = [], modulePath: [String] = []
        ) throws -> Self {
            let inputProjectionPrefixes = [
                "in_proj_qkv.", "in_proj_z.", "in_proj_b.", "in_proj_a.",
            ]
            let replacesInputProjection = parameters.flattened().contains { key, _ in
                inputProjectionPrefixes.contains(where: key.hasPrefix)
            }
            defer {
                if replacesInputProjection {
                    fusedInputProjection.invalidate()
                }
            }
            return try super.update(
                parameters: parameters, verify: verify, path: path, modulePath: modulePath)
        }

        open override func updateModule(key: String, _ value: Any) throws {
            let replacesInputProjection =
                key == "in_proj_qkv" || key == "in_proj_z"
                || key == "in_proj_b" || key == "in_proj_a"
            defer {
                if replacesInputProjection {
                    fusedInputProjection.invalidate()
                }
            }
            try super.updateModule(key: key, value)
        }

        var hasFusedInputProjection: Bool { fusedInputProjection.isPrepared }

        @discardableResult
        func prepareFusedInputProjection() throws -> Bool {
            try fusedInputProjection.prepare(
                enabled: fusedInputProjectionEnabled,
                linears: [
                    inProjQKV, inProjZ, inProjB, inProjA,
                ]
            ) { sourceViews in
                try update(
                    modules: ModuleChildren(values: [
                        "in_proj_qkv": .value(sourceViews[0]),
                        "in_proj_z": .value(sourceViews[1]),
                        "in_proj_b": .value(sourceViews[2]),
                        "in_proj_a": .value(sourceViews[3]),
                    ]), verify: [])
            }
        }

        func projectInputs(_ inputs: MLXArray, batch: Int, sequence: Int) -> (
            qkv: MLXArray, z: MLXArray, b: MLXArray, a: MLXArray
        ) {
            guard fusedInputProjectionEnabled, let fusedInProj = fusedInputProjection.fused else {
                return (
                    inProjQKV(inputs),
                    inProjZ(inputs).reshaped(batch, sequence, numVHeads, headVDim),
                    inProjB(inputs),
                    inProjA(inputs)
                )
            }

            let projected = fusedInProj(inputs)
            let qkvEnd = keyDim * 2 + valueDim
            let zEnd = qkvEnd + valueDim
            let bEnd = zEnd + numVHeads
            let aEnd = bEnd + numVHeads
            return (
                projected[0..., 0..., ..<qkvEnd],
                projected[0..., 0..., qkvEnd ..< zEnd].reshaped(
                    batch, sequence, numVHeads, headVDim),
                projected[0..., 0..., zEnd ..< bEnd],
                projected[0..., 0..., bEnd ..< aEnd]
            )
        }

        open func callAsFunction(
            _ inputs: MLXArray,
            mask: MLXArray? = nil,
            cache: MambaCache? = nil,
            checkpointAfter: Int? = nil
        ) -> MLXArray {
            let B = inputs.dim(0)
            let S = inputs.dim(1)

            var (mixedQKV, z, b, a) = projectInputs(
                inputs, batch: B, sequence: S)

            let convState: MLXArray
            if let cacheState = cache?[0] {
                convState = cacheState
            } else {
                convState = MLXArray.zeros(
                    [B, max(0, convKernelSize - 1), convDim], dtype: inputs.dtype)
            }

            if let mask {
                mixedQKV = MLX.where(mask[.ellipsis, .newAxis], mixedQKV, 0)
            }

            let convInput = concatenated([convState, mixedQKV], axis: 1)
            if let cache, convKernelSize > 1 {
                cache[0] = contiguous(convInput[0..., (-(convKernelSize - 1))..., 0...])
            }

            let convOut = silu(conv1d(convInput))
            let split = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
            let q = split[0].reshaped(B, S, numKHeads, headKDim)
            let k = split[1].reshaped(B, S, numKHeads, headKDim)
            let v = split[2].reshaped(B, S, numVHeads, headVDim)

            var state = cache?[1]
            let dtype = q.dtype
            let invScale = pow(Float(headKDim), -0.5)
            let qNormed =
                MLXArray(pow(invScale, 2)).asType(dtype)
                * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
            let kNormed =
                MLXArray(invScale).asType(dtype)
                * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

            let out: MLXArray
            if let split = checkpointAfter, split > 0, split < S {
                let prefixMask = mask.map { $0[0..., ..<split] }
                let suffixMask = mask.map { $0[0..., split...] }
                let (prefixOut, prefixState) = gatedDeltaUpdate(
                    q: qNormed[0..., ..<split, 0..., 0...],
                    k: kNormed[0..., ..<split, 0..., 0...],
                    v: v[0..., ..<split, 0..., 0...],
                    a: a[0..., ..<split, 0...],
                    b: b[0..., ..<split, 0...],
                    aLog: aLog,
                    dtBias: dtBias,
                    state: state,
                    mask: prefixMask,
                    useKernel: !training)
                let (suffixOut, suffixState) = gatedDeltaUpdate(
                    q: qNormed[0..., split..., 0..., 0...],
                    k: kNormed[0..., split..., 0..., 0...],
                    v: v[0..., split..., 0..., 0...],
                    a: a[0..., split..., 0...],
                    b: b[0..., split..., 0...],
                    aLog: aLog,
                    dtBias: dtBias,
                    state: prefixState,
                    mask: suffixMask,
                    useKernel: !training)
                out = concatenated([prefixOut, suffixOut], axis: 1)
                state = suffixState

                if let cache {
                    let checkpointConv: MLXArray
                    if convKernelSize > 1 {
                        checkpointConv = contiguous(
                            convInput[
                                0..., split ..< (split + convKernelSize - 1), 0...])
                    } else {
                        checkpointConv = MLXArray.zeros(
                            [B, 0, convDim], dtype: mixedQKV.dtype)
                    }
                    cache.saveSpeculativeCheckpoint(
                        convState: checkpointConv,
                        recurrentState: prefixState,
                        advancedBy: split)
                }
            } else {
                (out, state) = gatedDeltaUpdate(
                    q: qNormed,
                    k: kNormed,
                    v: v,
                    a: a,
                    b: b,
                    aLog: aLog,
                    dtBias: dtBias,
                    state: state,
                    mask: mask,
                    useKernel: !training)
            }

            if let cache {
                cache[1] = state
                cache.advance(S)
            }

            let gated = norm(out, gate: z)
            return outProj(gated.reshaped(B, S, -1))
        }
    }

    open class SparseMoeBlock: Module, UnaryLayer {
        let normTopkProb: Bool
        let numExperts: Int
        let topK: Int

        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

        @ModuleInfo(key: "shared_expert") var sharedExpert: MLP
        @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

        public init(_ args: Qwen35Configuration.TextConfiguration) {
            self.normTopkProb = args.normTopkProb
            self.numExperts = args.numExperts
            self.topK = args.numExpertsPerTok

            _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.moeIntermediateSize,
                numExperts: args.numExperts
            )

            _sharedExpert.wrappedValue = MLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.sharedExpertIntermediateSize
            )
            _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
            super.init()
        }

        open func callAsFunction(_ x: MLXArray) -> MLXArray {
            var gates = gate(x)
            gates = MLX.softmax(gates, axis: -1, precise: true)

            let (inds, scores) = moeRouterTopK(
                gates, k: topK, normalize: normTopkProb)

            let y = switchMLP(x, inds)
            let combined = weightedExpertSum(y, scores)

            var sharedY = sharedExpert(x)
            sharedY = sigmoid(sharedExpertGate(x)) * sharedY

            return combined + sharedY
        }
    }

    open class DecoderLayer: Module {
        let isLinear: Bool

        @ModuleInfo(key: "self_attn") var selfAttn: Attention?
        @ModuleInfo(key: "linear_attn") var linearAttn: GatedDeltaNet?

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        @ModuleInfo(key: "mlp") var mlp: Module

        public init(
            _ args: Qwen35Configuration.TextConfiguration, layerIdx: Int,
            forceFullAttention: Bool = false
        ) {
            self.isLinear =
                forceFullAttention
                ? false : (layerIdx + 1) % args.fullAttentionInterval != 0

            if isLinear {
                _linearAttn.wrappedValue = GatedDeltaNet(args)
            } else {
                _selfAttn.wrappedValue = Attention(args)
            }

            if args.numExperts > 0 {
                _mlp.wrappedValue = SparseMoeBlock(args)
            } else {
                _mlp.wrappedValue = MLP(
                    dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            }

            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)

            super.init()
        }

        open func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXArray?,
            ssmMask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?,
            checkpointAfter: Int? = nil
        ) -> MLXArray {
            let r: MLXArray
            if isLinear {
                r = linearAttn!(
                    inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache,
                    checkpointAfter: checkpointAfter)
            } else {
                r = selfAttn!(
                    inputLayerNorm(x), mask: attentionMask, cache: cache, positionIds: positionIds)
            }

            let h = x + r
            return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        }
    }
}

// MARK: - Model

extension Qwen35Configuration.TextConfiguration {
    /// This configuration as the text model reads it: the values resolved
    /// here, defaults included, decoded into the text model's own type.
    var languageModelConfiguration: Qwen35TextConfiguration {
        do {
            return try JSONDecoder().decode(
                Qwen35TextConfiguration.self, from: JSONEncoder().encode(self))
        } catch {
            preconditionFailure("Qwen3.5 text configuration does not round-trip: \(error)")
        }
    }
}

/// Qwen3.5 vision-language model: the vision tower in front of the text
/// model's own engine (`MLXLLM.Qwen35TextModel`), so compiled decode,
/// DFlash2 verify and MTP state serve image prompts as they serve text.
public class Qwen35: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") var languageModel: Qwen35TextModel

    public let config: Qwen35Configuration

    public init(_ config: Qwen35Configuration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen35TextModel(
            config.textConfiguration.languageModelConfiguration)
        super.init()
    }

    public var vocabularySize: Int { config.vocabSize }

    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.loraLayers
    }

    public func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        try languageModel.newCache(parameters: parameters)
    }

    public func prepare() throws {
        try languageModel.prepare()
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) throws -> (MLXArray, MLXArray) {
        let imageMask = (inputIds .== MLXArray(imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(videoTokenIndex))
        var specialMask = imageMask .|| videoMask

        let nImageTokens = specialMask.sum().item(Int.self)

        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)

        let nImageFeatures = imageFeatures.dim(0)
        let nImageMaskElements = maskExpanded.sum().item(Int.self)
        let imageFeatureSize = imageFeatures.size

        guard nImageMaskElements == imageFeatureSize else {
            throw Qwen35VLError.featureTokenMismatch(expected: nImageTokens, actual: nImageFeatures)
        }

        let originalShape = inputEmbeds.shape
        let flattenedEmbeds = inputEmbeds.flattened()
        let flattenedFeatures = imageFeatures.flattened()
        let flattenedMask = maskExpanded.flattened()

        let indices = nonZero(flattenedMask.asType(.bool))

        var result = flattenedEmbeds
        if !indices.isEmpty && indices.count == flattenedFeatures.size {
            let indexArray = MLXArray(indices.map { UInt32($0) })
            result[indexArray] = flattenedFeatures
        }

        result = result.reshaped(originalShape)
        let visualMask = specialMask.squeezed(axis: -1).asType(.bool)
        return (result, visualMask)
    }

    private func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }

    private func combinedFrames(imageFrames: [THW]?, videoFrames: [THW]?) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    /// Vision tower + image→token merge over the input, shared by `prepare`
    /// and `prepareContinuation` (mirrors `Qwen25VL.inputEmbeddings`).
    private func visionInputEmbeddings(
        _ input: LMInput
    ) throws -> (
        pixelValues: MLXArray?, imageFrames: [THW]?, videoFrames: [THW]?,
        inputEmbeddings: MLXArray?
    ) {
        var pixelValues: MLXArray?
        var imageFrames: [THW]?
        var videoFrames: [THW]?

        let visionDType = visionModel.patchEmbed.proj.weight.dtype
        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels.asType(visionDType))
            imageFrames = image.frames
        }
        if let video = input.video {
            pixelParts.append(video.pixels.asType(visionDType))
            videoFrames = video.frames
        }
        if !pixelParts.isEmpty {
            pixelValues = concatenated(pixelParts)
        }

        var inputEmbeddings: MLXArray?

        if let pixelValues,
            let frames = combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames)
                .nilIfEmpty
        {
            let inputIds = input.text.tokens
            let textEmbeds = languageModel.embedTokens(inputIds)
            let (visionHidden, _) = visionModel(pixelValues, gridTHW: frames)
            let visionFeatures = visionHidden.asType(textEmbeds.dtype)

            let (mergedEmbeds, _) = try mergeInputIdsWithImageFeatures(
                imageFeatures: visionFeatures,
                inputEmbeds: textEmbeds,
                inputIds: inputIds,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex
            )
            inputEmbeddings = mergedEmbeds
        }

        return (pixelValues, imageFrames, videoFrames, inputEmbeddings)
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let inputIds = input.text.tokens
        let inputIds2D = inputIds.ndim == 1 ? inputIds[.newAxis, 0...] : inputIds
        let cacheOffset = faCacheOffset(cache)
        // Resolved before the routing decision so a warm cache fails closed on either path.
        let positionOffset = try QwenVL.continuationAnchor(
            model: "Qwen35", key: ropeDeltasKey, cacheOffset: cacheOffset,
            batchSize: inputIds2D.dim(0), state: state)

        // Windowed (chunked) prefill — the remaining #344 deferred item for
        // Qwen3.5 — with the same default as the sibling chunked prefills
        // (Gemma3/LLMModel: `prefill.resolvedStepSize()`). The windowed forward also
        // owns every warm continuation (multi-turn chat, tool restart,
        // restored prompt cache): a cold cache is just a continuation
        // anchored at offset 0, and a warm one anchors M-RoPE positions at
        // the cache offset plus the rope delta carried in `state` — never
        // back at zero. The windowed forward is single-sequence; batched
        // inputs keep the single-shot path below.
        let window = prefill.resolvedStepSize()
        if inputIds2D.dim(0) == 1, inputIds2D.dim(-1) > 0,
            cacheOffset > 0 || inputIds2D.dim(-1) > window
        {
            return try prepareContinuation(
                input, inputIds: inputIds2D, cache: cache, cacheOffset: cacheOffset,
                positionOffset: positionOffset, prefill: prefill)
        }

        let (pixelValues, imageFrames, videoFrames, inputEmbeddings) =
            try visionInputEmbeddings(input)

        // A cold prompt: text rotates as the text model does; images get
        // their M-RoPE positions from zero.
        var state = state ?? .init()
        let positions: Qwen35RotaryPositions
        if pixelValues == nil {
            positions = .unshifted
            state[ropeDeltasKey] = MLXArray([Int32](repeating: 0, count: inputIds2D.dim(0)))
        } else {
            let (positionIds, deltas) = Qwen3VLLanguage.getRopeIndex(
                inputIds: inputIds2D,
                imageGridTHW: imageFrames,
                videoGridTHW: videoFrames,
                spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                imageTokenId: config.imageTokenId,
                videoTokenId: config.videoTokenId,
                visionStartTokenId: config.visionStartTokenId,
                attentionMask: input.text.mask)
            positions = .multimodal(positionIds)
            state[ropeDeltasKey] = deltas
        }

        let typedCache = castCache(cache)
        let output = withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            languageForward(
                inputIds2D, inputEmbeddings: inputEmbeddings, cache: typedCache,
                positions: positions, state: state)
        }

        let total = inputIds.dim(-1)
        prefill.progress?(total, total)
        return .logits(output)
    }

    /// Offset of the first full-attention layer's cache — the model's notion
    /// of "how many tokens are already cached".
    private func faCacheOffset(_ cache: [any KVCache]) -> Int {
        let faIdx = config.textConfiguration.fullAttentionInterval - 1
        return cache.indices.contains(faIdx) ? cache[faIdx].offset : 0
    }

    /// The text model's forward. The returned state keeps the rope delta
    /// `state` carries, and hands it to an MTP drafter that asked for state.
    private func languageForward(
        _ inputs: MLXArray, inputEmbeddings: MLXArray?, cache: [KVCache]?,
        positions: Qwen35RotaryPositions, state: LMOutput.State
    ) -> LMOutput {
        let output = languageModel(
            inputs, inputEmbeddings: inputEmbeddings, cache: cache, positions: positions,
            state: state)
        var outState = output.state ?? state
        if state[mtpEmitFlagKey] ?? false {
            outState[mtpPositionDeltasKey] = state[ropeDeltasKey]
        }
        return LMOutput(logits: output.logits, state: outState)
    }

    /// Warm, windowed continuation through an image-bearing remainder — the
    /// windowed forward that `prepare` also delegates to for long prompts.
    ///
    /// It runs the vision tower and the image→token merge **once** over the
    /// remainder, computes the new image's M-RoPE positions **once** from the
    /// seeded **Position Anchor** (so the image's diverging t/h/w indices
    /// start at the anchor, not zero), then drives the language-model forward
    /// in chunks from the prefill parameters. The full-attention scratch is bounded to
    /// `[heads, chunk, L]` instead of `[heads, L, L]`, so it cannot crash on
    /// a long prefix or a large image.
    ///
    /// `cache` must already be warmed to the restore offset `P` (a fresh cache
    /// means `P = 0`, a crash-safe cold prefill). `state` carries the anchor's
    /// rope delta in the `qwen35.ropeDeltas` slot (zero / absent for an
    /// image-free prefix). The returned state carries the rope delta the
    /// post-image text tail resumes with (`getRopeIndex` delta − `P`), so a
    /// caller threading state end-to-end continues that tail with the ordinary
    /// flat-continuation branch. Decode stays caller-owned.
    private func prepareContinuation(
        _ input: LMInput,
        inputIds: MLXArray,
        cache: [any KVCache],
        cacheOffset: Int,
        positionOffset: Int,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        let remainderLength = inputIds.dim(-1)
        precondition(remainderLength > 0, "prepareContinuation needs a non-empty remainder")

        // Vision tower + image→token merge — once, over the whole remainder;
        // chunks slice the merged embeddings below.
        let (_, imageFrames, videoFrames, inputEmbeddings) =
            try visionInputEmbeddings(input)

        // Where each chunk's rows rotate. Text rows sit at their cache row
        // plus a delta and take the text model's own rope; only a chunk that
        // holds image rows needs explicit M-RoPE positions. `resumeDelta` is
        // the delta the text after this remainder continues with.
        let chunkPositions: (Range<Int>) -> Qwen35RotaryPositions
        let resumeDelta: Int
        if inputEmbeddings == nil {
            resumeDelta = positionOffset - cacheOffset
            chunkPositions = { _ in .shifted(resumeDelta) }
        } else {
            // Offset-aware M-RoPE positions for the whole remainder — once. The
            // new image's t/h/w indices diverge from the anchor; the returned
            // delta is in the same offset frame.
            let (positionIds, ropeDeltas) = Qwen3VLLanguage.getRopeIndex(
                inputIds: inputIds,
                imageGridTHW: imageFrames,
                videoGridTHW: videoFrames,
                spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                imageTokenId: config.imageTokenId,
                videoTokenId: config.videoTokenId,
                visionStartTokenId: config.visionStartTokenId,
                attentionMask: input.text.mask,
                positionOffset: positionOffset
            )
            resumeDelta = ropeDeltas[0].item(Int.self) - cacheOffset
            let tokens = inputIds[0].asArray(Int32.self)
            let rowPositions = positionIds[0, 0].asArray(Int32.self)
            let media = Set([Int32(config.imageTokenIndex), Int32(config.videoTokenIndex)])
            chunkPositions = { range in
                if tokens[range].contains(where: media.contains) {
                    return .multimodal(positionIds[0..., 0..., range])
                }
                return .shifted(
                    Int(rowPositions[range.lowerBound]) - (cacheOffset + range.lowerBound))
            }
        }

        // Chunk the forward. Each window forwards `chunk` query tokens against
        // the growing cache, so the full-attention scratch stays `[heads,
        // chunk, L]`. Chunk logits are dropped un-evaluated (only the cache is
        // realized between windows) and the logits the iterator samples come
        // from the reserved final position, so `lm_head` materializes
        // `[1, 1, vocab]`, never `[1, L, vocab]`. `asyncEval` bounds the
        // un-evaluated graph while letting the GPU run window i as the CPU
        // builds window i+1 (same shape as the sibling chunked prefills).
        let typedCache = castCache(cache)

        /// One forward over `range`, slicing positions and embeddings in lockstep.
        func forward(_ range: Range<Int>) -> LMOutput {
            languageModel(
                inputIds[0..., range],
                inputEmbeddings: inputEmbeddings.map { $0[0..., range, 0...] },
                cache: typedCache,
                positions: chunkPositions(range),
                state: nil)
        }

        let processed = try prefill.forEachChunk(total: remainderLength) { range in
            _ = forward(range)
            if let typedCache {
                asyncEval(typedCache)
            }
        }
        if processed > 0, let typedCache {
            eval(typedCache)
        }

        let lastLogits = forward(processed ..< remainderLength).logits
        prefill.progress?(remainderLength, remainderLength)

        // Seed the post-image text tail's anchor: tail token j rotates at
        // `tailCacheOffset + delta + j`. Host-built, so decode reads it
        // without waiting on the GPU.
        var resumeState = LMOutput.State()
        resumeState[ropeDeltasKey] = MLXArray([Int32(resumeDelta)])
        return .logits(LMOutput(logits: lastLogits, state: resumeState))
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let cacheOffset = faCacheOffset(cache ?? [])
        precondition(
            cacheOffset == 0 || state?[ropeDeltasKey] != nil,
            "Qwen35 cannot continue a warm prompt cache without \(ropeDeltasKey.id)")
        let inputs =
            input.tokens.ndim == 1 ? input.tokens.expandedDimensions(axis: 0) : input.tokens
        var state = state ?? .init()
        let positions: Qwen35RotaryPositions
        if cacheOffset == 0 {
            positions = .unshifted
            state[ropeDeltasKey] = MLXArray([Int32](repeating: 0, count: inputs.dim(0)))
        } else {
            positions = continuingPositions(
                deltas: state[ropeDeltasKey]!, cacheOffset: cacheOffset, inputs: inputs)
        }
        return languageForward(
            inputs, inputEmbeddings: nil, cache: castCacheOptional(cache), positions: positions,
            state: state)
    }

    /// Positions for text continuing a warm cache: each row's cache row plus
    /// its batch row's delta. One delta for every row is the text model's
    /// own rope, shifted.
    private func continuingPositions(
        deltas: MLXArray, cacheOffset: Int, inputs: MLXArray
    ) -> Qwen35RotaryPositions {
        let batchSize = inputs.dim(0)
        // Read every decode step: an int32 delta is read as stored, with no
        // conversion op to wait on.
        var rowDeltas = (deltas.dtype == .int32 ? deltas : deltas.asType(.int32))
            .asArray(Int32.self)
        if rowDeltas.count < batchSize {
            rowDeltas += Array(repeating: rowDeltas.last ?? 0, count: batchSize - rowDeltas.count)
        }
        rowDeltas = Array(rowDeltas.prefix(batchSize))
        if Set(rowDeltas).count == 1 {
            return .shifted(Int(rowDeltas[0]))
        }
        let length = inputs.dim(1)
        let rows = MLXArray(Int32(cacheOffset) ..< Int32(cacheOffset + length))
        let perRow = rows[.newAxis, 0...] + MLXArray(rowDeltas)[0..., .newAxis]
        return .multimodal(broadcast(perRow[.newAxis], to: [3, batchSize, length]))
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        if metadata["format"]?.lowercased() == "mlx" {
            // Converted checkpoints can keep the MTP head; the drafter loads it, not this model.
            return weights.filter { !$0.key.contains("mtp.") }
        }
        return sanitize(weights: weights)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // Whether the checkpoint stores RMSNorm weights in the raw (un-shifted)
        // convention that needs the `+1` offset. Mirrors the MLXLLM Qwen35 gate
        // so the VLM and text paths agree: a pre-converted MLX checkpoint
        // (conv1d already sanitized) has already been shifted at
        // conversion time and must NOT be shifted again — doing so double-shifts
        // every layernorm and produces garbage tokens. Computed on the incoming
        // weights, before the `mtp.` filter below.
        let hasUnsanitizedConv1d = weights.contains { key, value in
            key.contains("conv1d.weight") && value.dim(-1) != 1
        }
        // MTP tensors are not proof of a raw checkpoint: a converted checkpoint can
        // keep them (the framework uses them for speculative decoding), and shifting
        // its already-shifted norms a second time produces garbage tokens. The conv1d
        // layout is the reliable signal on its own.
        let shouldShiftNormWeights = hasUnsanitizedConv1d

        var weights = weights.filter { !$0.key.contains("mtp.") }

        weights = filterLMHeadWeights(
            from: weights,
            tiedWordEmbeddings: config.textConfiguration.tieWordEmbeddings)

        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for (key, originalValue) in weights {
            var key = key
            var value = originalValue

            if key.contains("model") {
                if key.contains("model.language_model") {
                    key = key.replacingOccurrences(
                        of: "model.language_model", with: "language_model.model")
                } else if key.contains("model.visual") {
                    key = key.replacingOccurrences(of: "model.visual", with: "vision_tower")
                } else if key.hasPrefix("model.") {
                    // Unified Qwen 3.5 checkpoints (e.g. Qwen3.5-0.8B-MLX-4bit) ship
                    // language model tensors at bare `model.*` paths instead of
                    // `model.language_model.*`. Mirror the LLM-side fallback.
                    key = "language_model." + key
                }
            } else if key.contains("lm_head") {
                key = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }

            if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                value = value.movedAxis(source: 2, destination: 1)
            }
            if shouldShiftNormWeights
                && normKeys.contains(where: { key.hasSuffix($0) }) && value.ndim == 1
            {
                value = value + MLXArray(1, dtype: value.dtype)
            }

            sanitized[key] = value
        }

        return visionModel.sanitize(weights: sanitized)
    }
}

extension Qwen35: SpeculativeCacheRewindModel {
    public var maximumNativeTargetCacheRewind: Int { 1 }
}

/// The text model's DFlash2 target. A prompt's images are prefilled through
/// ``prepare(_:cache:state:prefill:)`` first; the speculative iterator then
/// takes the text after them, rotated by their rope delta.
extension Qwen35: DFlash2TargetModel {
    public var dflash2LayerCount: Int { languageModel.dflash2LayerCount }
    public var dflash2Embedding: Embedding { languageModel.dflash2Embedding }
    public var dflash2Head: Linear? { languageModel.dflash2Head }

    public func dflash2SupportsCache(_ cache: [KVCache]) -> Bool {
        languageModel.dflash2SupportsCache(cache)
    }

    public func dflash2Prefill(
        _ tokens: MLXArray, cache: [KVCache], captureLayers: [Int], positionDelta: Int
    ) -> (logits: MLXArray, hidden: [MLXArray]) {
        languageModel.dflash2Prefill(
            tokens, cache: cache, captureLayers: captureLayers, positionDelta: positionDelta)
    }

    public func dflash2Verify(
        _ request: DFlash2VerifyRequest, cache: [KVCache]
    ) -> DFlash2VerifyResult {
        languageModel.dflash2Verify(request, cache: cache)
    }
}

extension Qwen35: DFlash2MediaTargetModel {
    /// Runs ``prepare(_:cache:state:prefill:)`` over the prompt through its
    /// last image or video row. Nil when the prompt has no such row or no
    /// text after it.
    public func dflash2PrefillMedia(
        _ input: LMInput, cache: [KVCache], prefill: PrefillParameters
    ) throws -> (prefilledTokens: Int, positionDelta: Int)? {
        let tokens = input.text.tokens.reshaped(1, -1)
        let ids = tokens[0].asArray(Int32.self)
        let media = Set([Int32(config.imageTokenIndex), Int32(config.videoTokenIndex)])
        guard let last = ids.lastIndex(where: media.contains), last + 1 < ids.count else {
            return nil
        }
        let prefix = LMInput(
            text: LMInput.Text(tokens: tokens[0..., ..<(last + 1)]), image: input.image,
            video: input.video)
        guard
            case .logits(let output) = try prepare(
                prefix, cache: cache, state: nil, prefill: prefill)
        else { return nil }
        eval(cache)
        let delta = output.state?[ropeDeltasKey].map { $0.asType(.int32)[0].item(Int.self) } ?? 0
        return (last + 1, delta)
    }
}

extension Array where Element == THW {
    fileprivate var nilIfEmpty: [THW]? { isEmpty ? nil : self }
}

extension Qwen35 {
    fileprivate func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    fileprivate func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }
}

// MARK: - Chat conventions

// `Qwen35MoE` subclasses `Qwen35` and inherits both declarations.
extension Qwen35 {
    public var toolCallFormat: ToolCallFormat? { .qwen35 }
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }
}
