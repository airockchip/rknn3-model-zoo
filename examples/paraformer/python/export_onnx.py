#!/usr/bin/env python3
# Copyright (c) 2026 Rockchip Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Export offline SeACo-Paraformer submodels to ONNX.

The encoder includes the alpha predictor head while dynamic CIF accumulation
remains on the CPU. The baseline decoder consumes an explicit token mask. The
optional hotword group exports a two-output decoder, SeaCo decoder, official
ASF scorer, CPU BiasEncoder weights, and tokenizer resources.
"""

import argparse
import os
from pathlib import Path

import torch

from paraformer_export import (
    append_alpha_head,
    build_export_models,
    build_hotword_export_models,
    copy_hotword_resources,
    copy_runtime_resources,
    export_bias_encoder_weights,
    export_graph,
    install_torchaudio_import_fallback,
    rescale_encoder_for_fp16,
    verify_graph,
)


DEFAULT_MODEL_PATH = (
    "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export offline SeACo-Paraformer encoder and masked decoder ONNX"
    )
    parser.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="ModelScope or Hugging Face model ID, or local checkpoint directory",
    )
    parser.add_argument(
        "--modelscope",
        action="store_true",
        help="download a non-local custom --model_path from ModelScope",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="directory used for ONNX and runtime resource outputs",
    )
    parser.add_argument(
        "--stage",
        choices=("encoder", "decoder", "hotword", "all"),
        default="all",
        help="submodel group to export",
    )
    return parser.parse_args()


def resolve_model_path(model_path, use_modelscope):
    """Resolve a local checkpoint or download the requested model snapshot."""
    local_path = Path(model_path).expanduser()
    if local_path.is_dir():
        return local_path.resolve()
    if local_path.exists():
        raise NotADirectoryError(f"Model path is not a directory: {local_path}")

    if use_modelscope or model_path == DEFAULT_MODEL_PATH:
        from modelscope import snapshot_download

        return Path(snapshot_download(model_path)).resolve()

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model_path)).resolve()


def main():
    args = parse_args()
    model_path = resolve_model_path(args.model_path, args.modelscope)
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    install_torchaudio_import_fallback()
    from funasr import AutoModel

    print(f"Loading model: {model_path}")
    auto_model = AutoModel(
        model=str(model_path),
        device="cpu",
        disable_update=True,
        disable_log=True,
    )
    predictor = auto_model.model.predictor

    encoder_model, decoder_model = build_export_models(auto_model.model)
    hotword_models = None
    if args.stage in ("hotword", "all"):
        hotword_models = build_hotword_export_models(auto_model.model)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    original_cwd = Path.cwd()
    os.chdir(output_path)
    try:
        if args.stage in ("encoder", "all"):
            encoder_path = output_path / "paraformer_encoder.onnx"
            encoder_base_path = output_path / ".paraformer_encoder_base.onnx"
            encoder_device = "cuda" if torch.cuda.is_available() else "cpu"
            # Keep the upstream onnx_fp16 topology: trace at the 512-frame
            # ceiling, use opset 13, and rescale the SANM blocks before export.
            # Every RKNN dynamic bucket is no larger than this trace length.
            encoder_model = encoder_model.to(encoder_device)
            encoder_dummy = (
                torch.randn(1, 512, 560, dtype=torch.float32, device=encoder_device),
                torch.ones(1, 512, 1, dtype=torch.float32, device=encoder_device),
            )
            rescale_encoder_for_fp16(encoder_model)
            export_graph(
                encoder_model,
                encoder_dummy,
                encoder_base_path,
                True,
                opset_version=13,
                device=encoder_device,
            )
            append_alpha_head(encoder_base_path, predictor, encoder_path)
            verify_graph(encoder_path, "encoder")

        if args.stage in ("decoder", "all"):
            decoder_path = output_path / "paraformer_decoder.onnx"
            decoder_dummy = (
                torch.randn(1, 334, 512, dtype=torch.float32),
                torch.randn(1, 64, 512, dtype=torch.float32),
                torch.ones(1, 64, 1, dtype=torch.float32),
                torch.ones(1, 334, 1, dtype=torch.float32),
            )
            export_graph(decoder_model, decoder_dummy, decoder_path, False)
            verify_graph(decoder_path, "decoder")

        if args.stage in ("hotword", "all"):
            decoder_hotword, seaco_decoder, asf_scorer = hotword_models
            enc = torch.randn(1, 334, 512, dtype=torch.float32)
            embeds = torch.randn(1, 64, 512, dtype=torch.float32)
            token_mask = torch.ones(1, 64, 1, dtype=torch.float32)
            enc_mask = torch.ones(1, 334, 1, dtype=torch.float32)
            contextual_info = torch.randn(1, 16, 512, dtype=torch.float32)
            hotword_mask = torch.ones(1, 16, dtype=torch.float32)

            decoder_hotword_path = output_path / "paraformer_decoder_hotword.onnx"
            export_graph(
                decoder_hotword,
                (enc, embeds, token_mask, enc_mask),
                decoder_hotword_path,
                False,
            )
            verify_graph(decoder_hotword_path, "decoder_hotword")

            seaco_decoder_path = output_path / "seaco_decoder.onnx"
            export_graph(
                seaco_decoder,
                (
                    contextual_info,
                    embeds,
                    embeds,
                    token_mask,
                    hotword_mask,
                ),
                seaco_decoder_path,
                False,
            )
            verify_graph(seaco_decoder_path, "seaco_decoder")

            asf_scorer_path = output_path / "asf_scorer.onnx"
            export_graph(
                asf_scorer,
                (contextual_info, embeds, token_mask, hotword_mask),
                asf_scorer_path,
                False,
            )
            verify_graph(asf_scorer_path, "asf_scorer")

            export_bias_encoder_weights(
                auto_model.model,
                output_path / "bias_encoder.bin",
            )
            copy_hotword_resources(model_path, output_path)
    finally:
        os.chdir(original_cwd)
        intermediate = output_path / "enc.pt"
        if intermediate.exists():
            intermediate.unlink()
        encoder_base_path = output_path / ".paraformer_encoder_base.onnx"
        if encoder_base_path.exists():
            encoder_base_path.unlink()

    copy_runtime_resources(model_path, output_path)
    print("ONNX export completed")


if __name__ == "__main__":
    main()
