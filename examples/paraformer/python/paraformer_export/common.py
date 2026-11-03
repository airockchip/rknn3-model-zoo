#!/usr/bin/env python3
# Copyright (c) 2026 Rockchip Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Shared graph and resource helpers for Paraformer export."""

import shutil
import sys
import types
from pathlib import Path

import onnx
import torch


MODEL_INPUTS = {
    "encoder": ["speech", "enc_mask"],
    "decoder": ["enc", "pre_acoustic_embeds", "token_mask", "enc_mask"],
    "decoder_hotword": [
        "enc",
        "pre_acoustic_embeds",
        "token_mask",
        "enc_mask",
    ],
    "seaco_decoder": [
        "contextual_info",
        "pre_acoustic_embeds",
        "decoder_hidden",
        "token_mask",
        "hotword_mask",
    ],
    "asf_scorer": [
        "contextual_info",
        "decoder_hidden",
        "token_mask",
        "hotword_mask",
    ],
}
MODEL_OUTPUTS = {
    "encoder": ["enc", "alphas"],
    "decoder": ["decoder_out"],
    "decoder_hotword": ["decoder_out", "decoder_hidden"],
    "seaco_decoder": ["seaco_logits"],
    "asf_scorer": ["hotword_scores"],
}


def install_torchaudio_import_fallback():
    """Allow model export when an incompatible torchaudio wheel is present."""
    try:
        import torchaudio  # noqa: F401
        return
    except (ImportError, OSError):
        pass

    torchaudio = types.ModuleType("torchaudio")
    compliance = types.ModuleType("torchaudio.compliance")
    kaldi = types.ModuleType("torchaudio.compliance.kaldi")
    torchaudio.__version__ = torch.__version__
    kaldi.warp_frequency = lambda value: value
    compliance.kaldi = kaldi
    torchaudio.compliance = compliance
    sys.modules["torchaudio"] = torchaudio
    sys.modules["torchaudio.compliance"] = compliance
    sys.modules["torchaudio.compliance.kaldi"] = kaldi


def export_graph(
    model,
    dummy_inputs,
    output_path,
    fold_constants,
    opset_version=14,
    device="cpu",
):
    """Atomically export one subgraph with the wrapper's public interface."""
    model = model.to(device).eval()
    dummy_inputs = tuple(value.to(device) for value in dummy_inputs)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    input_names = list(model.export_input_names())
    output_names = list(model.export_output_names())
    dynamic_axes = model.export_dynamic_axes()

    print(f"Exporting {output_path}")
    print(f"  inputs:  {input_names}")
    print(f"  outputs: {output_names}")
    print(f"  shapes:  {[tuple(value.shape) for value in dummy_inputs]}")

    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_inputs,
                str(temporary_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=fold_constants,
                dynamo=False,
            )
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def verify_graph(path, stage):
    """Run the ONNX checker and enforce the public subgraph interface."""
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    inputs = [value.name for value in model.graph.input]
    outputs = [value.name for value in model.graph.output]
    if inputs != MODEL_INPUTS[stage] or outputs != MODEL_OUTPUTS[stage]:
        raise RuntimeError(
            f"Unexpected {stage} interface: inputs={inputs}, outputs={outputs}"
        )
    print(f"Verified {stage}: inputs={inputs}, outputs={outputs}")


def copy_runtime_resources(model_path, output_path):
    """Copy vocabulary and sample inputs alongside the exported graphs."""
    resources = {
        model_path / "tokens.json": output_path / "tokens.json",
        model_path / "example" / "asr_example.wav": output_path / "asr_example.wav",
        model_path / "asr_example_hotword.wav": (
            output_path / "asr_example_hotword.wav"
        ),
        Path(__file__).resolve().parents[2] / "model" / "hotwords.txt": (
            output_path / "hotwords.txt"
        ),
    }
    for source, destination in resources.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required model resource not found: {source}")
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        print(f"Copied {destination}")
