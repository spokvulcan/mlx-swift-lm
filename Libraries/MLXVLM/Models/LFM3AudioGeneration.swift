// Copyright © 2026 Apple Inc.

// Port of `Model.stream_slots` in
// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/lfm3_audio/lfm3_audio.py

import Foundation
import MLX
import MLXLMCommon

/// What LFM3-Audio writes.
public enum LFM3AudioOutput: Sendable, Equatable {
    /// Start in text and let the model open speech itself: `<|audio_start|>` for TTS,
    /// `<|mixed_start|>` for a spoken chat answer. A prompt ending in a marker starts there.
    case automatic
    /// Text only: the speech markers are suppressed.
    case text
    /// Speech until an end-of-speech frame. Appends `<|audio_start|>`.
    case speech
    /// Blocks of text tokens and speech frames, then speech after `<|text_end|>`. Appends
    /// `<|mixed_start|>`.
    case interleaved
}

/// Sampling of the speech codes. The defaults are the model card's.
public struct LFM3AudioSampling: Sendable {
    public var temperature: Float
    public var topK: Int
    public var repetitionPenalty: Float
    /// Earlier speech frames the repetition penalty reads.
    public var repetitionWindow: Int

    public init(
        temperature: Float = 1.0, topK: Int = 128, repetitionPenalty: Float = 1.1,
        repetitionWindow: Int = 4096
    ) {
        self.temperature = temperature
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.repetitionWindow = repetitionWindow
    }
}

/// One generated slot: a text token or an 80 ms speech frame.
public enum LFM3AudioSlot: Sendable, Equatable {
    case text(Int)
    /// The codes of one speech frame, one per codebook.
    case speech([Int32])
    /// The model ended its speech.
    case endOfSpeech
}

/// Generates text and speech from ``LFM3Audio`` one slot at a time, as the reference
/// `generate` state machine: a text token sampled from the backbone, or a speech frame whose
/// codebooks the depth decoder reveals by MaskGIT. Each slot feeds the next.
///
/// Decode the speech frames with ``LFM3AudioCodec``. The generator holds its own cache and
/// is not thread safe.
public struct LFM3AudioGenerator: Sequence, IteratorProtocol {

    private enum Mode {
        case text, speech, interleaved
    }

    private let model: LFM3Audio
    private let config: LFM3AudioConfiguration
    private let cache: [KVCache]
    private let sampler: LogitSampler
    private var processor: LogitProcessor?
    private let suppressor: SuppressTokensProcessor?
    private let audio: LFM3AudioSampling
    private let endOfTurnTokenIds: Set<Int>
    private let frameOrders: [[Int]]?

    private var hidden: MLXArray
    private var mode: Mode
    private var speaking = false
    private var textDone = false
    private var slotsLeft: Int
    private var remaining: Int
    private var finished = false
    private var frameIndex = 0
    private var history = [MLXArray]()

    /// - Parameters:
    ///   - input: a prompt from ``LFM3AudioProcessor``: text, and speech in `audio`.
    ///   - output: what to write; see ``LFM3AudioOutput``.
    ///   - parameters: text sampling. The model card decodes text greedily.
    ///   - audio: speech sampling.
    ///   - maxSlots: the most text tokens and speech frames to write.
    ///   - endOfTurnTokenIds: tokens that end the answer (`<|im_end|>`).
    ///   - frameOrders: the MaskGIT reveal order of each frame, random when `nil`.
    ///   - prefill: the chunking of the prompt.
    public init(
        model: LFM3Audio, input: LMInput, output: LFM3AudioOutput = .automatic,
        parameters: GenerateParameters = GenerateParameters(temperature: 0),
        audio: LFM3AudioSampling = .init(), maxSlots: Int = 2048,
        endOfTurnTokenIds: Set<Int> = [124_900], frameOrders: [[Int]]? = nil,
        prefill: PrefillParameters = .init()
    ) throws {
        let config = model.config
        self.model = model
        self.config = config
        self.audio = audio
        self.endOfTurnTokenIds = endOfTurnTokenIds
        self.frameOrders = frameOrders
        self.remaining = maxSlots
        self.slotsLeft = config.interleavedTextTokens
        self.sampler = parameters.sampler()

        var ids = input.text.tokens.reshaped(-1).asType(.int32).asArray(Int32.self).map(Int.init)
        let speechMarkers = [config.audioStartTokenId, config.audioOutTokenId]
        switch output {
        case .speech where !speechMarkers.contains(ids.last ?? -1):
            ids.append(config.audioStartTokenId)
        case .interleaved where ids.last != config.mixedStartTokenId:
            ids.append(config.mixedStartTokenId)
        default:
            break
        }
        let last = ids.last ?? -1
        if speechMarkers.contains(last) {
            mode = .speech
        } else if last == config.mixedStartTokenId {
            mode = .interleaved
        } else {
            mode = .text
        }

        var suppressed: Set<Int> = [config.audioTokenId, config.audioOutTokenId]
        if output == .text {
            suppressed.formUnion([config.audioStartTokenId, config.mixedStartTokenId])
        }
        self.suppressor = SuppressTokensProcessor(tokenIds: suppressed)

        let tokens = MLXArray(ids.map(Int32.init))
        var processor = parameters.processor()
        processor?.prompt(tokens)
        self.processor = processor

        self.cache = try model.newCache(parameters: parameters)
        let embeddings = try model.inputEmbeddings(tokens, audio: input.audio)
        self.hidden = try model.prefillHidden(embeddings, cache: cache, prefill: prefill)
    }

    /// Penalized, tempered, top-k sampled codes `[1, codebooks]` from `[1, codebooks, size]`
    /// logits; `history` `[1, codebooks, frames]` holds the codes to penalize.
    private func sampleCodes(_ logits: MLXArray, history: MLXArray?) -> MLXArray {
        var scores = logits.asType(.float32)
        if let history, audio.repetitionPenalty != 1 {
            let picked = takeAlong(scores, history, axis: -1)
            let penalized = MLX.where(
                picked .< 0, picked * audio.repetitionPenalty, picked / audio.repetitionPenalty)
            scores = putAlong(scores, history, values: penalized, axis: -1)
        }
        guard audio.temperature > 0 else {
            return argMax(scores, axis: -1)
        }
        scores = scores / audio.temperature
        scores = scores - logSumExp(scores, axis: -1, keepDims: true)
        let size = scores.dim(-1)
        if audio.topK > 0, audio.topK < size {
            let dropped = argPartition(-scores, kth: audio.topK - 1, axis: -1)[
                .ellipsis, audio.topK...]
            scores = putAlong(scores, dropped, values: MLXArray(-Float.infinity), axis: -1)
        }
        return MLXRandom.categorical(scores)
    }

    /// One frame of codes `[codebooks]`, before end-of-speech handling.
    private func sampleFrame() -> MLXArray {
        let count = config.codebooks
        let order =
            frameOrders.flatMap { $0.indices.contains(frameIndex) ? $0[frameIndex] : nil }
            ?? MLXRandom.permutation(count).asType(.int32).asArray(Int32.self).map(Int.init)
        let slots = model.depthDecoder.slots(hidden.reshaped(1, -1))
        let window = history.suffix(audio.repetitionWindow)
        let penalized = window.isEmpty ? nil : stacked(Array(window), axis: -1)[.newAxis]

        var codes = MLXArray.zeros([1, count], dtype: .int32)
        var revealed = [Bool](repeating: false, count: count)
        var start = 0
        for size in config.maskgitSchedule {
            let picked = Set(order[start ..< start + size])
            let logits = model.depthDecoder(
                slots, codes: codes, masked: MLXArray(revealed.map { !$0 })[.newAxis])
            let sampled = sampleCodes(logits, history: penalized)
            let pick = MLXArray((0 ..< count).map { picked.contains($0) })[.newAxis]
            codes = MLX.where(pick, sampled.asType(.int32), codes)
            for index in picked {
                revealed[index] = true
            }
            start += size
        }
        return codes[0]
    }

    public mutating func next() -> LFM3AudioSlot? {
        guard !finished, remaining > 0 else { return nil }
        remaining -= 1
        let interleaving = mode == .interleaved
        let slot: LFM3AudioSlot
        let embedding: MLXArray
        var endsTurn = false

        if mode == .speech || (interleaving && speaking) {
            var codes = sampleFrame()
            frameIndex += 1
            let values = codes.asArray(Int32.self)
            // A frame with any end-of-speech code is all end-of-speech.
            if values.contains(Int32(config.endOfSpeechCode)) {
                codes = MLXArray(
                    Array(repeating: Int32(config.endOfSpeechCode), count: config.codebooks))
                mode = .text
                slot = .endOfSpeech
            } else {
                history.append(codes)
                slot = .speech(values)
            }
            embedding = model.frameEmbeddings(codes[.newAxis])[.newAxis]
        } else {
            var logits = model.languageModel.logits(hidden)[0..., -1, 0...]
            logits = suppressor?.process(logits: logits) ?? logits
            logits = processor?.process(logits: logits) ?? logits
            let sampled = sampler.sample(logits: logits)
            processor?.didSample(token: sampled)
            let token = sampled.item(Int.self)
            if token == config.textEndTokenId {
                // Ends the text of an interleaved answer, else the turn.
                if interleaving {
                    textDone = true
                    speaking = true
                } else {
                    endsTurn = true
                }
            } else if endOfTurnTokenIds.contains(token) {
                endsTurn = true
            } else if interleaving {
                // Markers are plain text inside an interleaved answer.
            } else if token == config.audioStartTokenId {
                mode = .speech
            } else if token == config.mixedStartTokenId {
                mode = .interleaved
                slotsLeft = config.interleavedTextTokens
                speaking = false
                textDone = false
            }
            slot = .text(token)
            embedding = model.languageModel.model.embedTokens!(
                MLXArray([Int32(token)])[.newAxis])
        }

        if interleaving && !textDone {
            slotsLeft -= 1
            if slotsLeft == 0 {
                speaking.toggle()
                slotsLeft = speaking ? config.interleavedAudioFrames : config.interleavedTextTokens
            }
        }
        if endsTurn || remaining == 0 {
            finished = true
            return slot
        }
        hidden = model.languageModel.model(nil, cache: cache, inputEmbeddings: embedding)
        asyncEval(hidden)
        return slot
    }
}

extension LFM3AudioCodec {
    /// Codes `[codebooks, frames]` for ``decode(_:)`` from generated speech frames.
    public static func codes(frames: [[Int32]]) -> MLXArray {
        guard let first = frames.first else { return MLXArray.zeros([12, 0], dtype: .int32) }
        return MLXArray(frames.flatMap { $0 }, [frames.count, first.count]).transposed()
    }
}
