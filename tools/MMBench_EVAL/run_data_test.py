#!/usr/bin/env python3
"""Run an lmms-eval visual task against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Union

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="mmbench_en_dev_static")
    parser.add_argument("--server-ip", required=True)
    parser.add_argument("--server-port", required=True, type=int)
    parser.add_argument("--image-width", required=True, type=int)
    parser.add_argument("--image-height", required=True, type=int)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key", default="no-key")
    parser.add_argument("--timeout", default=180, type=int)
    parser.add_argument("--max-retries", default=2, type=int)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def install_fixed_size_encoder(width: int, height: int) -> None:
    """Resize each image at every upstream OpenAI serialization entry."""
    if width < 1 or height < 1:
        raise ValueError("image width and height must both be positive")

    from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64
    from lmms_eval.models.simple.openai import OpenAICompatible
    from lmms_eval.protocol import ChatMessages

    target_size = (width, height)

    def encode_image(self, image: Union[Image.Image, str]) -> str:
        if isinstance(image, str):
            with Image.open(image) as loaded_image:
                working_image = loaded_image.convert("RGB")
        elif isinstance(image, Image.Image):
            working_image = image.convert("RGB")
        else:
            raise TypeError(f"unsupported image type: {type(image)!r}")

        if working_image.size != target_size:
            working_image = working_image.resize(target_size, Image.Resampling.LANCZOS)

        # The upstream OpenAI backend labels this payload as image/png.
        return encode_image_to_base64(
            working_image,
            image_format="PNG",
            convert_rgb=False,
            copy_if_pil=False,
            use_path_cache=False,
        )

    OpenAICompatible.encode_image = encode_image

    # Current lmms-eval registers the chat OpenAI adapter.  It serializes
    # images through ChatMessages directly, rather than the model method above.
    def encode_chat_image(self, image, cache=None, image_format="PNG", quality=None) -> str:
        if isinstance(image, str):
            with Image.open(image) as loaded_image:
                working_image = loaded_image.convert("RGB")
        elif isinstance(image, Image.Image):
            working_image = image.convert("RGB")
        else:
            raise TypeError(f"unsupported image type: {type(image)!r}")

        if working_image.size != target_size:
            working_image = working_image.resize(target_size, Image.Resampling.LANCZOS)

        normalized_format = image_format.upper()
        return encode_image_to_base64(
            working_image,
            image_format=normalized_format,
            convert_rgb=normalized_format in {"JPEG", "JPG", "WEBP"},
            quality=quality,
            copy_if_pil=False,
            cache=cache,
            use_path_cache=False,
        )

    ChatMessages.encode_image = encode_chat_image


def allow_anonymous_mmbench_download() -> None:
    """Use anonymous access for lmms-eval's public MMBench dataset."""
    import datasets

    original_load_dataset = datasets.load_dataset

    def load_dataset(*args, **kwargs):
        dataset_path = kwargs.get("path", args[0] if args else None)
        if dataset_path == "lmms-lab-encoder/MMBench" and kwargs.get("token") is True:
            kwargs = dict(kwargs)
            kwargs["token"] = False
        return original_load_dataset(*args, **kwargs)

    datasets.load_dataset = load_dataset


def suppress_throughput_table() -> None:
    """Keep upstream throughput data in JSON but omit its terminal table."""
    from lmms_eval import __main__ as lmms_main

    original_make_table = lmms_main.make_table

    def make_table_without_throughput(result_dict, *args, **kwargs):
        column = args[0] if args else kwargs.get("column", "results")
        if column == "results" and isinstance(result_dict, dict):
            result_dict = dict(result_dict)
            result_dict.pop("throughput", None)
        return original_make_table(result_dict, *args, **kwargs)

    lmms_main.make_table = make_table_without_throughput


def main() -> None:
    args = parse_args()
    allow_anonymous_mmbench_download()
    install_fixed_size_encoder(args.image_width, args.image_height)
    suppress_throughput_table()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_MODE", "disabled")

    base_url = f"http://{args.server_ip}:{args.server_port}/v1"
    model_args = ",".join(
        [
            f"model={args.model_name}",
            f"base_url={base_url}",
            f"api_key={args.api_key}",
            f"timeout={args.timeout}",
            f"max_retries={args.max_retries}",
            "retry_backoff_s=3",
            "num_concurrent=1",
        ]
    )

    cli_args = [
        "lmms-eval",
        "eval",
        "--model",
        "openai",
        "--model_args",
        model_args,
        "--tasks",
        args.task_name,
        "--batch_size",
        "1",
        "--gen_kwargs",
        "max_new_tokens=32,temperature=0",
        "--output_path",
        args.output_path,
        "--log_samples",
    ]
    print(f"[client] task: {args.task_name}")
    print(f"[client] server: {base_url}")
    print(f"[client] enforced image size: {args.image_width}x{args.image_height}")
    sys.argv = cli_args

    from lmms_eval.cli.dispatch import main as lmms_eval_main

    lmms_eval_main()


if __name__ == "__main__":
    main()
