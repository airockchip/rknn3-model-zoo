"""Prepare GRQ/ONNX calibration tensors and original model references."""

import argparse
import json
import shutil
from pathlib import Path

if __package__:
    from .model_source import resolve_model_path
else:
    from model_source import resolve_model_path


def prepare_sample(aligner, audio_path, text, language, output_path):
    """Prepare one sample with an already loaded FP32 reference model."""
    import numpy as np
    import soundfile as sf
    import torch

    wave, rate = sf.read(audio_path, dtype="float32")
    if rate != 16000 or wave.ndim != 1 or not 400 <= len(wave) <= 480000:
        raise ValueError(
            "Audio must be 16 kHz mono and between 25 ms and 30 s long"
        )
    if not np.isfinite(wave).all() or "\n" in text or "\r" in text:
        raise ValueError(
            "Audio contains nonfinite values or text contains newlines"
        )
    units, prompt = aligner.aligner_processor.encode_timestamp(
        text,
        language,
    )
    inputs = aligner.processor(
        text=[prompt], audio=[wave], return_tensors="pt", padding=True
    )
    thinker = aligner.model.thinker
    n = inputs["input_ids"].shape[1]
    frames = int(inputs["feature_attention_mask"].sum())
    if n > 2048 or frames > 3000 or not units:
        raise ValueError(
            "Input exceeds token/frame limits or has no alignable text"
        )
    with torch.inference_mode():
        audio = thinker.get_audio_features(
            inputs["input_features"],
            inputs["feature_attention_mask"],
        )
        embeds = thinker.model.embed_tokens(inputs["input_ids"])
        audio_mask = inputs["input_ids"] == thinker.config.audio_token_id
        embeds = embeds.masked_scatter(
            audio_mask.unsqueeze(-1).expand_as(embeds),
            audio,
        )
        logits = thinker(**inputs, use_cache=False).logits
    audio_rows = np.flatnonzero(audio_mask[0].numpy())
    slots = np.flatnonzero(
        inputs["input_ids"][0].numpy() == aligner.timestamp_token_id,
    )
    if len(slots) != len(units) * 2 or len(audio_rows) != len(audio):
        raise ValueError("Text timestamp or audio token count mismatch")
    indices = logits[0, slots].argmax(-1).numpy()
    padded = np.zeros((1, 2048, 1024), dtype=np.float32)
    padded[:, :n] = embeds.numpy()
    mask = np.zeros((1, 2048), dtype=np.float32)
    mask[:, :n] = 1
    pos = np.arange(2048, dtype=np.int64).reshape(1, 1, -1)
    out = output_path
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / "input_embeds.npy", padded)
    np.save(out / "attention_mask.npy", mask)
    np.save(out / "position_ids.npy", pos)
    mel = inputs["input_features"][0, :, :frames].numpy()
    chunks = np.zeros((30, 1, 128, 100), dtype=np.float32)
    for i in range(30):
        width = min(100, max(0, frames - i * 100))
        chunks[i, 0, :, :width] = mel[:, i * 100 : i * 100 + width]
    np.save(out / "mel_chunks.npy", chunks)
    metadata = dict(
        text=text,
        units=units,
        seq_len=n,
        audio_rows=audio_rows.tolist(),
        slots=slots.tolist(),
        samples=len(wave),
        raw_indices=indices.tolist(),
        items_ms=aligner.aligner_processor.parse_timestamp(
            units, indices * 80
        ),
        language=language,
    )
    (out / "reference.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
    )
    shutil.copyfile(audio_path, out / "audio.wav")
    return metadata


def prepare_dataset(model_path, manifest_path, output_path):
    """Prepare all audio/text records, or reuse a prepared-directory list."""
    cases = json.loads(manifest_path.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("Calibration manifest must be a nonempty array")
    if all(isinstance(case, str) and case for case in cases):
        return manifest_path
    records = []
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"Invalid calibration record: {index}")
        audio = case.get("audio_path")
        text = case.get("text")
        language = case.get("language")
        if not isinstance(audio, str) or not audio:
            raise ValueError(f"Missing audio_path in record {index}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Missing text in record {index}")
        if language not in ("Chinese", "English"):
            raise ValueError(f"Unsupported language in record {index}")
        audio_path = (manifest_path.parent / audio).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        records.append((audio_path, text, language))

    import torch
    from qwen_asr import Qwen3ForcedAligner

    torch.set_num_threads(8)
    aligner = Qwen3ForcedAligner.from_pretrained(
        str(model_path),
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    )
    output_path.mkdir(parents=True, exist_ok=False)
    directories = []
    for index, (audio_path, text, language) in enumerate(records):
        name = f"sample_{index:04d}"
        prepare_sample(aligner, audio_path, text, language, output_path / name)
        directories.append(name)
        print(f"Prepared calibration sample {index + 1}/{len(records)}")
    manifest = output_path / "dataset.json"
    manifest.write_text(json.dumps(directories, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        required=True,
        help="Local model directory or Hugging Face / ModelScope model ID",
    )
    parser.add_argument(
        "--modelscope",
        action="store_true",
        help="Download model IDs from ModelScope instead of Hugging Face",
    )
    parser.add_argument(
        "--audio_path", type=Path, required=True, help="16 kHz mono WAV file"
    )
    parser.add_argument("--text", required=True, help="Known transcript")
    parser.add_argument(
        "--language", choices=["Chinese", "English"], default="Chinese"
    )
    parser.add_argument("--output_path", type=Path, required=True)
    args = parser.parse_args()
    if not args.audio_path.is_file():
        parser.error("Audio file does not exist")
    if args.output_path.exists():
        parser.error("Output path already exists")
    model_path = resolve_model_path(args.model_path, args.modelscope)
    import torch
    from qwen_asr import Qwen3ForcedAligner

    torch.set_num_threads(8)
    aligner = Qwen3ForcedAligner.from_pretrained(
        str(model_path),
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    )
    metadata = prepare_sample(
        aligner, args.audio_path, args.text, args.language, args.output_path
    )
    print(json.dumps(metadata["items_ms"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
