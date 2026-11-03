#!/usr/bin/env python3
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR).
# SPDX-License-Identifier: MIT
"""SeaCo hotword and ASF export adapters for Paraformer."""

import shutil
from pathlib import Path

import numpy as np
import torch

from .decoder import ParaformerSANMDecoderDynamicMaskExport


BIAS_ENCODER_TENSORS = (
    ("embedding", (8404, 512)),
    ("weight_ih_l0", (2048, 512)),
    ("weight_hh_l0", (2048, 512)),
    ("bias_ih_l0", (2048,)),
    ("bias_hh_l0", (2048,)),
    ("weight_ih_l1", (2048, 512)),
    ("weight_hh_l1", (2048, 512)),
    ("bias_ih_l1", (2048,)),
    ("bias_hh_l1", (2048,)),
)


class DecoderHotwordExport(torch.nn.Module):
    """Expose the baseline decoder logits and hidden states together."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, enc, pre_acoustic_embeds, token_mask, enc_mask):
        hidden = self.decoder.forward_hidden(
            enc,
            pre_acoustic_embeds,
            token_mask,
            enc_mask,
        )
        logits = (
            self.decoder.output_layer(hidden)
            if self.decoder.output_layer is not None
            else hidden
        )
        return logits, hidden

    def export_input_names(self):
        return ["enc", "pre_acoustic_embeds", "token_mask", "enc_mask"]

    def export_output_names(self):
        return ["decoder_out", "decoder_hidden"]

    def export_dynamic_axes(self):
        return {
            "enc": {0: "batch_size", 1: "enc_length"},
            "pre_acoustic_embeds": {0: "batch_size", 1: "token_length"},
            "token_mask": {0: "batch_size", 1: "token_length"},
            "enc_mask": {0: "batch_size", 1: "enc_length"},
            "decoder_out": {0: "batch_size", 1: "token_length"},
            "decoder_hidden": {0: "batch_size", 1: "token_length"},
        }


class SeacoDecoderExport(torch.nn.Module):
    """Run the SeaCo bias decoder over CIF and baseline decoder features."""

    def __init__(self, decoder, output_layer):
        super().__init__()
        self.decoder = decoder
        self.output_layer = output_layer

    def forward(
        self,
        contextual_info,
        pre_acoustic_embeds,
        decoder_hidden,
        token_mask,
        hotword_mask,
    ):
        cif_attended = self.decoder.forward_hidden(
            contextual_info,
            pre_acoustic_embeds,
            token_mask,
            hotword_mask,
        )
        decoder_attended = self.decoder.forward_hidden(
            contextual_info,
            decoder_hidden,
            token_mask,
            hotword_mask,
        )
        return self.output_layer(cif_attended + decoder_attended)

    def export_input_names(self):
        return [
            "contextual_info",
            "pre_acoustic_embeds",
            "decoder_hidden",
            "token_mask",
            "hotword_mask",
        ]

    def export_output_names(self):
        return ["seaco_logits"]

    def export_dynamic_axes(self):
        return {
            "contextual_info": {0: "batch_size", 1: "num_hotwords"},
            "pre_acoustic_embeds": {0: "batch_size", 1: "token_length"},
            "decoder_hidden": {0: "batch_size", 1: "token_length"},
            "token_mask": {0: "batch_size", 1: "token_length"},
            "hotword_mask": {0: "batch_size", 1: "num_hotwords"},
            "seaco_logits": {0: "batch_size", 1: "token_length"},
        }


class AsfScorerExport(torch.nn.Module):
    """Export the official sixth-layer attention score used for top-k ASF."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        contextual_info,
        decoder_hidden,
        token_mask,
        hotword_mask,
    ):
        target_mask = token_mask.to(
            dtype=decoder_hidden.dtype,
            device=decoder_hidden.device,
        )
        if len(target_mask.shape) == 3 and target_mask.shape[2] == 1:
            target_mask = target_mask[:, :, 0]
        target_mask, _ = self.decoder.prepare_mask(target_mask)

        memory_mask = hotword_mask.to(
            dtype=decoder_hidden.dtype,
            device=decoder_hidden.device,
        )
        if len(memory_mask.shape) == 3 and memory_mask.shape[2] == 1:
            memory_mask = memory_mask[:, :, 0]
        _, memory_mask = self.decoder.prepare_mask(memory_mask)

        hidden = decoder_hidden
        memory = contextual_info
        for layer_index in range(5):
            hidden, target_mask, memory, memory_mask, _ = (
                self.decoder.model.decoders[layer_index](
                    hidden,
                    target_mask,
                    memory,
                    memory_mask,
                )
            )
        attention = self.decoder.model.decoders[5].get_attn_mat(
            hidden,
            target_mask,
            memory,
            memory_mask,
        )
        return attention[0].sum(0).sum(0)

    def export_input_names(self):
        return [
            "contextual_info",
            "decoder_hidden",
            "token_mask",
            "hotword_mask",
        ]

    def export_output_names(self):
        return ["hotword_scores"]

    def export_dynamic_axes(self):
        return {
            "contextual_info": {0: "batch_size", 1: "num_hotwords"},
            "decoder_hidden": {0: "batch_size", 1: "token_length"},
            "token_mask": {0: "batch_size", 1: "token_length"},
            "hotword_mask": {0: "batch_size", 1: "num_hotwords"},
            "hotword_scores": {0: "num_hotwords"},
        }


def build_hotword_export_models(model):
    """Build the three deployed hotword subgraphs from one prepared model."""
    if not isinstance(model.decoder, ParaformerSANMDecoderDynamicMaskExport):
        raise RuntimeError("Prepare the baseline decoder before hotword export")
    decoder_hotword = DecoderHotwordExport(model.decoder)
    model.seaco_decoder = ParaformerSANMDecoderDynamicMaskExport(
        model.seaco_decoder
    )
    seaco_decoder = SeacoDecoderExport(
        model.seaco_decoder,
        model.hotword_output_layer,
    )
    asf_scorer = AsfScorerExport(model.seaco_decoder)
    return decoder_hotword, seaco_decoder, asf_scorer


def export_bias_encoder_weights(model, output_path):
    """Atomically export all CPU BiasEncoder parameters as one FP32 file."""
    output_path = Path(output_path)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    decoder = model.decoder.model
    embedding = decoder.embed[0] if isinstance(
        decoder.embed,
        torch.nn.Sequential,
    ) else decoder.embed
    tensors = {"embedding": embedding.weight}
    tensors.update(dict(model.bias_encoder.named_parameters()))
    try:
        with temporary_path.open("wb") as output_file:
            for name, expected_shape in BIAS_ENCODER_TENSORS:
                if name not in tensors:
                    raise RuntimeError(f"Missing bias-encoder tensor: {name}")
                array = (
                    tensors[name]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                if array.shape != expected_shape:
                    raise RuntimeError(
                        f"Unexpected {name} shape: {array.shape}, "
                        f"expected {expected_shape}"
                    )
                array.tofile(output_file)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"Exported CPU BiasEncoder weights: {output_path}")


def copy_hotword_resources(model_path, output_path):
    """Copy the tokenizer segmentation dictionary used by hotword text."""
    source = Path(model_path) / "seg_dict"
    destination = Path(output_path) / "seg_dict"
    if not source.is_file():
        raise FileNotFoundError(f"Required hotword resource not found: {source}")
    shutil.copy2(source, destination)
    print(f"Copied {destination}")
