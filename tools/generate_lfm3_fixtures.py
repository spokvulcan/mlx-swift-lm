#!/usr/bin/env python3
"""
generate_lfm3_fixtures.py — reference fixtures for the opt-in LFM3 parity tests
(`LFM3Tests`, `LFM3VLTests`, `LFM3AudioTests`, `localCheckpointParity()`).

Pinned to mlx-vlm commit: 230fc308efb577d87446306bfb8a3e9f1a129558

Usage:
    python generate_lfm3_fixtures.py text  MODEL_DIR OUT_PREFIX [--float32]
    python generate_lfm3_fixtures.py vl    MODEL_DIR OUT_PREFIX --image IMAGE [--float32]
    python generate_lfm3_fixtures.py video MODEL_DIR OUT_PREFIX --video VIDEO [--float32]
    python generate_lfm3_fixtures.py audio MODEL_DIR OUT_PREFIX --wav WAV [--float32]

`text` writes OUT_PREFIX.json; `vl`, `video` and `audio` also write OUT_PREFIX.safetensors.
With --float32 both models run in float32 and the Swift test checks them to 1e-4; otherwise it
allows the measured bfloat16 rounding. The video is sampled by mlx-vlm's loader (OpenCV), as
`mlx_vlm.generate` samples it; a short clip keeps the fixture small, e.g. 2.5 s at 30 fps:

    ffmpeg -loop 1 -framerate 30 -i cats.jpg -frames:v 75 \\
        -vf "crop=560:315:'80*t/2.5':'165*t/2.5',scale=640:360:out_range=tv,format=yuv420p" \\
        -c:v libx264 -crf 18 -colorspace smpte170m -color_range tv cats_pan.mp4

The audio clip must be 16 kHz mono float32 WAV, e.g.:

    say -o speech.aiff "The quick brown fox jumps over the lazy dog."
    afconvert -f WAVE -d LEF32@16000 -c 1 speech.aiff speech.wav

Then run the Swift test with the fixture (paths are examples):

    TEST_RUNNER_MLX_LFM3_MODEL=/path/to/LFM3-1.7B-MLX-4bit \\
    TEST_RUNNER_MLX_LFM3_FIXTURE=/tmp/lfm3_text.json \\
    xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \\
        -skipPackagePluginValidation -only-testing:'MLXLMTests/LFM3Tests'

The VL test reads MLX_LFM3_VL_MODEL, MLX_LFM3_VL_FIXTURE (the prefix) and optionally
MLX_LFM3_VL_IMAGE; the video test MLX_LFM3_VL_MODEL, MLX_LFM3_VL_VIDEO_FIXTURE and optionally
MLX_LFM3_VL_VIDEO; the audio test MLX_LFM3_AUDIO_MODEL and MLX_LFM3_AUDIO_FIXTURE.
"""

import argparse
import json
import re
import struct

import mlx.core as mx
import numpy as np
from mlx_vlm import load


def text_fixture(model, processor, out_prefix, dtype):
    tokenizer = getattr(processor, "tokenizer", processor)
    paragraph = (
        "Liquid neural networks are a class of time-continuous recurrent models whose "
        "hidden state follows an input-dependent differential equation. "
    )
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {
            "role": "user",
            "content": paragraph * 30
            + "\nSummarize the text above in one sentence, then name the capital of Australia.",
        },
    ]
    # Long enough to cross the 256-token sliding window and the 512-token prefill chunks.
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    lm = model.language_model
    x = mx.array([ids])
    last = lm(x).logits.astype(mx.float32)[0, -1]
    order = mx.argsort(-last)[:20]

    cache = model.make_cache()
    out = lm(x, cache=cache).logits.astype(mx.float32)
    tokens = [int(mx.argmax(out[0, -1]).item())]
    for _ in range(63):
        out = lm(mx.array([[tokens[-1]]]), cache=cache).logits.astype(mx.float32)
        tokens.append(int(mx.argmax(out[0, -1]).item()))

    fixture = {
        "dtype": dtype,
        "prompt": prompt,
        "ids": ids,
        "last_top_ids": np.array(order).tolist(),
        "last_top_logits": np.array(last[order]).tolist(),
        "greedy_tokens": tokens,
        "greedy_text": tokenizer.decode(tokens),
    }
    json.dump(fixture, open(out_prefix + ".json", "w"))
    print(fixture["greedy_text"])


def vl_fixture(model, processor, out_prefix, dtype, image_path):
    from PIL import Image

    tokenizer = processor.tokenizer
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=[image], return_tensors="mlx")
    ids, pixels, grid = inputs["input_ids"], inputs["pixel_values"], inputs["image_grid_thw"]

    weight_dtype = model.vision_tower.embedding_norm.weight.dtype
    features = model.encode_images(pixels.astype(weight_dtype), image_grid_thw=grid)
    embeds = model.get_input_embeddings(ids, pixels, image_grid_thw=grid).inputs_embeds
    lm = model.language_model
    last = lm(ids, inputs_embeds=embeds).logits.astype(mx.float32)[0, -1]
    order = mx.argsort(-last)[:20]

    cache = model.make_cache()
    out = lm(ids, cache=cache, inputs_embeds=embeds).logits.astype(mx.float32)
    tokens = [int(mx.argmax(out[0, -1]).item())]
    for _ in range(47):
        out = lm(mx.array([[tokens[-1]]]), cache=cache).logits.astype(mx.float32)
        tokens.append(int(mx.argmax(out[0, -1]).item()))

    mx.save_safetensors(
        out_prefix + ".safetensors",
        {"pixel_values": pixels.astype(mx.float32), "features": features.astype(mx.float32)},
    )
    json.dump(
        {
            "dtype": dtype,
            "prompt": prompt,
            "ids": np.array(ids).reshape(-1).tolist(),
            "grid": np.array(grid).tolist(),
            "last_top_ids": np.array(order).tolist(),
            "last_top_logits": np.array(last[order]).tolist(),
            "greedy_tokens": tokens,
            "greedy_text": tokenizer.decode(tokens),
        },
        open(out_prefix + ".json", "w"),
    )
    print(tokenizer.decode(tokens))


def video_fixture(model, processor, out_prefix, dtype, video_path):
    from mlx_vlm.utils import load_video

    tokenizer = processor.tokenizer
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Describe this video in one sentence."},
            ],
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    frames, metadata = load_video(
        video_path, frame_sampler=processor.video_processor.sample_frames
    )
    inputs = processor(
        text=[prompt], videos=[frames], video_metadata=[metadata], return_tensors="mlx"
    )
    ids, pixels = inputs["input_ids"], inputs["pixel_values_videos"]
    grid = inputs["video_grid_thw"]

    weight_dtype = model.vision_tower.embedding_norm.weight.dtype
    features = model.encode_images(pixels.astype(weight_dtype), image_grid_thw=grid)
    embeds = model.get_input_embeddings(
        ids, pixel_values_videos=pixels, video_grid_thw=grid
    ).inputs_embeds
    lm = model.language_model
    last = lm(ids, inputs_embeds=embeds).logits.astype(mx.float32)[0, -1]
    order = mx.argsort(-last)[:20]

    cache = model.make_cache()
    out = lm(ids, cache=cache, inputs_embeds=embeds).logits.astype(mx.float32)
    tokens = [int(mx.argmax(out[0, -1]).item())]
    for _ in range(47):
        out = lm(mx.array([[tokens[-1]]]), cache=cache).logits.astype(mx.float32)
        tokens.append(int(mx.argmax(out[0, -1]).item()))

    # The Swift test's tokenizer answers with the template's ids and those of each text the
    # processor encodes: the time labels and the vision tokens.
    ids = np.array(ids).reshape(-1).tolist()
    texts = sorted(set(re.findall(r"<\d+\.\d seconds>", tokenizer.decode(ids))))
    texts += ["<|image|>", "<|video|>", "<|vision_start|>", "<|vision_end|>"]
    mx.save_safetensors(
        out_prefix + ".safetensors",
        {"pixel_values": pixels.astype(mx.float32), "features": features.astype(mx.float32)},
    )
    json.dump(
        {
            "dtype": dtype,
            "prompt": prompt,
            "template_ids": tokenizer.encode(prompt, add_special_tokens=False),
            "text_ids": {
                text: tokenizer.encode(text, add_special_tokens=False) for text in texts
            },
            "ids": ids,
            "grid": np.array(grid).tolist(),
            "total_frames": metadata.total_num_frames,
            "fps": metadata.fps,
            "frame_indices": list(metadata.frames_indices),
            "last_top_ids": np.array(order).tolist(),
            "last_top_logits": np.array(last[order]).tolist(),
            "greedy_tokens": tokens,
            "greedy_text": tokenizer.decode(tokens),
        },
        open(out_prefix + ".json", "w"),
    )
    print(texts, tokenizer.decode(tokens))


def read_float32_wav(path):
    data = open(path, "rb").read()
    start = data.index(b"data")
    size = struct.unpack("<I", data[start + 4 : start + 8])[0]
    return np.frombuffer(data[start + 8 : start + 8 + size], dtype="<f4").astype(np.float32)


def audio_fixture(model, processor, out_prefix, dtype, wav_path):
    tokenizer = processor.tokenizer
    config = model.config
    clip = read_float32_wav(wav_path)

    # ASR
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Perform ASR in English."},
            {"role": "user", "content": [{"type": "audio"}]},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=[prompt], audio=[clip], return_tensors="mlx")
    ids, features = inputs["input_ids"], inputs["input_features"]
    mask = inputs["feature_attention_mask"]
    lengths = mx.array(mask).sum(-1)
    speech = model.audio_projector(model.audio_tower(mx.array(features), lengths))
    asr_tokens = [
        int(token)
        for token, _ in model.stream_slots(
            ids, max_tokens=64, generation_mode="text", temperature=0,
            input_features=features, feature_attention_mask=mask,
        )
    ]
    embeds = model.get_input_embeddings(
        ids, input_features=features, feature_attention_mask=mask
    ).inputs_embeds
    last = model.language_model(ids, inputs_embeds=embeds).logits.astype(mx.float32)[0, -1]
    order = mx.argsort(-last)[:20]

    # TTS: greedy codes in a fixed MaskGIT order
    tts_prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Perform TTS in American English using a female voice."},
            {"role": "user", "content": "Hello from the Swift port."},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    tts_ids = mx.array([tokenizer.encode(tts_prompt, add_special_tokens=False)])
    rng = np.random.default_rng(0)
    frame_order = [rng.permutation(12).tolist() for _ in range(48)]
    slots = [
        None if codes is None else np.array(codes).astype(int).tolist()
        for _, codes in model.stream_slots(
            tts_ids, max_tokens=48, generation_mode="audio", temperature=0,
            audio_temperature=0, audio_repetition_penalty=1.1, audio_repetition_window=4096,
            frame_order=frame_order,
        )
    ]
    frames = [c for c in slots if c is not None and c[0] != config.audio_eos_code]
    waveform = (
        model.codec().decode(mx.array(np.array(frames, dtype=np.int32).T))
        if frames
        else mx.zeros((0,))
    )

    mx.save_safetensors(
        out_prefix + ".safetensors",
        {
            "input_features": mx.array(features).astype(mx.float32),
            "speech_features": speech.astype(mx.float32),
            "waveform": waveform.astype(mx.float32),
            "clip": mx.array(clip),
        },
    )
    json.dump(
        {
            "dtype": dtype,
            "asr_ids": np.array(ids).reshape(-1).tolist(),
            "asr_tokens": asr_tokens,
            "asr_text": tokenizer.decode(asr_tokens),
            "asr_last_top_ids": np.array(order).tolist(),
            "asr_last_top_logits": np.array(last[order]).tolist(),
            "tts_ids": np.array(tts_ids).reshape(-1).tolist(),
            "tts_frame_order": frame_order,
            "tts_slots": slots,
        },
        open(out_prefix + ".json", "w"),
    )
    print("ASR:", tokenizer.decode(asr_tokens))
    print("TTS frames:", len(frames), "of", len(slots))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("kind", choices=["text", "vl", "video", "audio"])
    parser.add_argument("model_dir")
    parser.add_argument("out_prefix")
    parser.add_argument("--image")
    parser.add_argument("--video")
    parser.add_argument("--wav")
    parser.add_argument("--float32", action="store_true")
    args = parser.parse_args()

    model, processor = load(args.model_dir)
    dtype = "float32" if args.float32 else "bfloat16"
    if args.float32:
        model.set_dtype(mx.float32)
    if args.kind == "text":
        text_fixture(model, processor, args.out_prefix, dtype)
    elif args.kind == "vl":
        vl_fixture(model, processor, args.out_prefix, dtype, args.image)
    elif args.kind == "video":
        video_fixture(model, processor, args.out_prefix, dtype, args.video)
    else:
        audio_fixture(model, processor, args.out_prefix, dtype, args.wav)


if __name__ == "__main__":
    main()
