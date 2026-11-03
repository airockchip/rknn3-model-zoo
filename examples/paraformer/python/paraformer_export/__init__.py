#!/usr/bin/env python3
# Copyright (c) 2026 Rockchip Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Paraformer ONNX export helpers."""

import copy

from .common import (
    copy_runtime_resources,
    export_graph,
    install_torchaudio_import_fallback,
    verify_graph,
)


def append_alpha_head(encoder_path, predictor, output_path):
    """Lazily append the validated alpha predictor to an encoder graph."""
    from .alpha import append_alpha_head as append

    append(encoder_path, predictor, output_path)


def build_export_models(model, encoder_name="SANMEncoder"):
    """Return encoder and masked-decoder export views of a FunASR model."""
    # Import after install_torchaudio_import_fallback() has run. Some CPU
    # environments have a torchaudio wheel whose optional shared libraries
    # cannot be loaded, while FunASR's export adapters do not require them.
    from .decoder import bind_decoder_export_interface, prepare_decoder_for_export
    from .encoder import bind_encoder_export_interface, prepare_encoder_for_export

    prepare_encoder_for_export(model, encoder_name)
    prepare_decoder_for_export(model)

    encoder_model = copy.copy(model)
    decoder_model = copy.copy(model)
    bind_encoder_export_interface(encoder_model)
    bind_decoder_export_interface(decoder_model)
    return encoder_model, decoder_model


def rescale_encoder_for_fp16(model):
    """Lazily apply the validated FunASR encoder FP16 transform."""
    from .encoder import rescale_encoder_for_fp16 as apply_rescale

    apply_rescale(model)


def build_hotword_export_models(model):
    """Lazily build the SeaCo decoder and ASF export adapters."""
    from .hotword import build_hotword_export_models as build

    return build(model)


def export_bias_encoder_weights(model, output_path):
    """Lazily export the CPU bias-encoder weights."""
    from .hotword import export_bias_encoder_weights as export_weights

    export_weights(model, output_path)


def copy_hotword_resources(model_path, output_path):
    """Lazily copy resources required by the text hotword path."""
    from .hotword import copy_hotword_resources as copy_resources

    copy_resources(model_path, output_path)


__all__ = [
    "build_export_models",
    "build_hotword_export_models",
    "append_alpha_head",
    "copy_hotword_resources",
    "copy_runtime_resources",
    "export_bias_encoder_weights",
    "export_graph",
    "install_torchaudio_import_fallback",
    "rescale_encoder_for_fp16",
    "verify_graph",
]
