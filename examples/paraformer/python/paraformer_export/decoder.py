#!/usr/bin/env python3
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR).
# SPDX-License-Identifier: MIT
"""FunASR SANM decoder adapter with an explicit token mask."""

import types

import torch


class ParaformerSANMDecoderDynamicMaskExport(torch.nn.Module):
    """Export wrapper for the bucketed decoder with an explicit token mask."""

    def __init__(self, model):
        super().__init__()
        self.model = model

        from funasr.models.paraformer.decoder import DecoderLayerSANMExport
        from funasr.models.sanm.attention import MultiHeadedAttentionCrossAtt
        from funasr.models.sanm.attention import MultiHeadedAttentionCrossAttExport
        from funasr.models.sanm.attention import MultiHeadedAttentionSANMDecoder
        from funasr.models.sanm.attention import MultiHeadedAttentionSANMDecoderExport

        for index, decoder in enumerate(self.model.decoders):
            if isinstance(decoder.self_attn, MultiHeadedAttentionSANMDecoder):
                decoder.self_attn = MultiHeadedAttentionSANMDecoderExport(
                    decoder.self_attn
                )
            if isinstance(decoder.src_attn, MultiHeadedAttentionCrossAtt):
                decoder.src_attn = MultiHeadedAttentionCrossAttExport(
                    decoder.src_attn
                )
            self.model.decoders[index] = DecoderLayerSANMExport(decoder)

        if self.model.decoders2 is not None:
            for index, decoder in enumerate(self.model.decoders2):
                if isinstance(decoder.self_attn, MultiHeadedAttentionSANMDecoder):
                    decoder.self_attn = MultiHeadedAttentionSANMDecoderExport(
                        decoder.self_attn
                    )
                self.model.decoders2[index] = DecoderLayerSANMExport(decoder)

        for index, decoder in enumerate(self.model.decoders3):
            self.model.decoders3[index] = DecoderLayerSANMExport(decoder)

        self.output_layer = model.output_layer
        self.after_norm = model.after_norm

    def prepare_mask(self, mask):
        mask_3d_btd = mask[:, :, None]
        if len(mask.shape) == 2:
            mask_4d_bhlt = 1 - mask[:, None, None, :]
        elif len(mask.shape) == 3:
            mask_4d_bhlt = 1 - mask[:, None, :]
        else:
            raise RuntimeError(f"Unsupported decoder mask rank: {len(mask.shape)}")
        return mask_3d_btd, mask_4d_bhlt * -10000.0

    def forward_hidden(
        self,
        hs_pad,
        ys_in_pad,
        token_mask,
        memory_token_mask,
    ):
        """Return decoder hidden states using explicit target/memory masks."""
        target = ys_in_pad
        target_mask = token_mask.to(
            dtype=ys_in_pad.dtype,
            device=ys_in_pad.device,
        )
        if len(target_mask.shape) == 3 and target_mask.shape[2] == 1:
            target_mask = target_mask[:, :, 0]
        target_mask, _ = self.prepare_mask(target_mask)

        memory = hs_pad
        memory_mask_2d = memory_token_mask.to(
            dtype=ys_in_pad.dtype,
            device=ys_in_pad.device,
        )
        if len(memory_mask_2d.shape) == 3:
            memory_mask_2d = memory_mask_2d[:, :, 0]
        _, memory_mask = self.prepare_mask(memory_mask_2d)

        hidden = target
        hidden, target_mask, memory, memory_mask, _ = self.model.decoders(
            hidden,
            target_mask,
            memory,
            memory_mask,
        )
        if self.model.decoders2 is not None:
            hidden, target_mask, memory, memory_mask, _ = self.model.decoders2(
                hidden,
                target_mask,
                memory,
                memory_mask,
            )
        hidden, target_mask, memory, memory_mask, _ = self.model.decoders3(
            hidden,
            target_mask,
            memory,
            memory_mask,
        )
        hidden = self.after_norm(hidden)
        return hidden

    def forward(self, hs_pad, ys_in_pad, token_mask, enc_mask):
        hidden = self.forward_hidden(
            hs_pad,
            ys_in_pad,
            token_mask,
            enc_mask,
        )
        if self.output_layer is not None:
            hidden = self.output_layer(hidden)
        return hidden


def _decoder_forward(self, enc, pre_acoustic_embeds, token_mask, enc_mask):
    return self.decoder(enc, pre_acoustic_embeds, token_mask, enc_mask)


def _decoder_input_names(self):
    del self
    return ["enc", "pre_acoustic_embeds", "token_mask", "enc_mask"]


def _decoder_output_names(self):
    del self
    return ["decoder_out"]


def _decoder_dynamic_axes(self):
    del self
    return {
        "enc": {0: "batch_size", 1: "enc_length"},
        "pre_acoustic_embeds": {
            0: "batch_size",
            1: "pre_acoustic_embeds_length",
        },
        "token_mask": {0: "batch_size", 1: "pre_acoustic_embeds_length"},
        "enc_mask": {0: "batch_size", 1: "enc_length"},
        "decoder_out": {0: "batch_size", 1: "logits_length"},
    }


def prepare_decoder_for_export(model):
    """Replace the training decoder with the explicit-mask export adapter."""
    model.decoder = ParaformerSANMDecoderDynamicMaskExport(model.decoder)


def bind_decoder_export_interface(model):
    """Attach the stable decoder ONNX interface to a shallow model view."""
    model.forward = types.MethodType(_decoder_forward, model)
    model.export_input_names = types.MethodType(_decoder_input_names, model)
    model.export_output_names = types.MethodType(_decoder_output_names, model)
    model.export_dynamic_axes = types.MethodType(_decoder_dynamic_axes, model)
