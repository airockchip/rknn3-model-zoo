"""Export tokenizer, embeddings, and text rules for board deployment."""

import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

if __package__:
    from .model_source import resolve_model_path
else:
    from model_source import resolve_model_path


RESOURCE_PREFIX = "Qwen3-ForcedAligner-0.6B_decoder"
TEXT_RESOURCE_FILES = (
    f"{RESOURCE_PREFIX}.embed.bin",
    f"{RESOURCE_PREFIX}.tokenizer.gguf",
    f"{RESOURCE_PREFIX}_config.txt",
    f"{RESOURCE_PREFIX}_rules.bin",
)
RESOURCE_FILES = (*TEXT_RESOURCE_FILES, "resources.json")


def write_resources(model_path, output_path):
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    config = json.loads((model_path / "config.json").read_text())
    text_config = config["thinker_config"]["text_config"]
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    special = [
        tokenizer.convert_tokens_to_ids(token)
        for token in [
            "<|audio_start|>",
            "<|audio_pad|>",
            "<|audio_end|>",
            "<timestamp>",
        ]
    ]
    if special[-1] != config["timestamp_token_id"] or any(
        x is None for x in special
    ):
        raise ValueError("Model special token configuration mismatch")
    embed = None
    for path in sorted(model_path.glob("*.safetensors")):
        with safe_open(path, framework="pt") as weights:
            key = "thinker.model.embed_tokens.weight"
            if key in weights.keys():
                embed = weights.get_tensor(key).to(torch.float16)
                break
    if embed is None or embed.shape[1] != 1024:
        raise ValueError("Missing text embeddings with shape [vocab, 1024]")
    embed.numpy().tofile(output_path / f"{RESOURCE_PREFIX}.embed.bin")
    (output_path / f"{RESOURCE_PREFIX}_config.txt").write_text(
        f"{embed.shape[0]} 1024 " + " ".join(map(str, special)) + "\n"
    )
    # Unicode data follows the same Python runtime as the reference processor.
    ranges = []
    start, previous = 0, 0
    for code in range(0x110000):
        char = chr(code)
        flag = int(char.isspace()) | (
            int(unicodedata.category(char)[0] in "LN") << 1
        )
        if flag != previous:
            if previous:
                ranges.append((start, code - 1, previous))
            start, previous = code, flag
    if previous:
        ranges.append((start, 0x10FFFF, previous))
    with (output_path / f"{RESOURCE_PREFIX}_rules.bin").open("wb") as file:
        file.write(struct.pack("<II", 0x31525551, len(ranges)))
        for item in ranges:
            file.write(struct.pack("<III", *item))
    converter = (
        Path(__file__).resolve().parents[4] / "tokenizer/thirdparty/"
        "llama_vocab/convert_hf_to_gguf.py"
    )
    # The vocabulary exporter understands Qwen2 text architecture, not the
    # outer ASR wrapper. Use text-only metadata and the unmodified tokenizer.
    with tempfile.TemporaryDirectory(prefix="aligner_vocab_") as directory:
        directory = Path(directory)
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
        ]:
            if (model_path / name).is_file():
                shutil.copyfile(model_path / name, directory / name)
        text_config = dict(text_config)
        text_config.update(
            architectures=["Qwen2ForCausalLM"], model_type="qwen2"
        )
        (directory / "config.json").write_text(json.dumps(text_config))
        subprocess.run(
            [
                sys.executable,
                str(converter),
                "--vocab-only",
                "--outtype",
                "f16",
                "--outfile",
                str(output_path / f"{RESOURCE_PREFIX}.tokenizer.gguf"),
                str(directory),
            ],
            check=True,
        )
    (output_path / "resources.json").write_text(
        json.dumps(
            {
                "unicode_version": unicodedata.unidata_version,
                "embedding_shape": list(embed.shape),
                "special_ids": special,
                "source": (
                    "Qwen3-ForcedAligner original tokenizer and embeddings"
                ),
            },
            indent=2,
        )
        + "\n"
    )


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
        "--output_path",
        required=True,
        type=Path,
        help="New directory for deployment text resources",
    )
    args = parser.parse_args()
    destination = args.output_path.resolve()
    if destination.exists():
        parser.error("Output path already exists; use a new directory")
    args.model_path = resolve_model_path(args.model_path, args.modelscope)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        staging = Path(directory) / "resources"
        staging.mkdir()
        write_resources(args.model_path.resolve(), staging)
        staging.rename(destination)


if __name__ == "__main__":
    main()
