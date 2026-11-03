#!/usr/bin/env python3
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR).
# SPDX-License-Identifier: MIT
"""FunASR SANM encoder adapter for RKNN-compatible ONNX export."""

import functools
import types

import torch
import torch.nn.functional as F

from funasr.register import tables


# FunASR's onnx_fp16 export rescales the SANM encoder to keep intermediate
# activations inside the FP16 range. This is the checkpoint's validated scale.
ENCODER_FP16_SCALE = 12


def _encoder_forward(self, speech, enc_mask):
    encoded, _ = self.encoder(
        speech=speech,
        speech_mask=enc_mask,
    )
    return encoded


def _prepare_sanm_mask(self, mask):
    """Build data and attention masks without scalar broadcast constants."""
    mask_3d_btd = mask[:, :, None]
    if len(mask.shape) == 2:
        mask_4d_bhlt = (
            torch.ones_like(mask[:, None, None, :])
            - mask[:, None, None, :]
        )
    elif len(mask.shape) == 3:
        mask_4d_bhlt = torch.ones_like(mask[:, None, :]) - mask[:, None, :]
    else:
        raise RuntimeError(f"Unsupported encoder mask rank: {len(mask.shape)}")
    return mask_3d_btd, mask_4d_bhlt * -10000.0


def _sanm_encoder_forward(
    self,
    speech,
    speech_lengths=None,
    online=False,
    speech_mask=None,
):
    """Run SANM with a mask whose time axis follows the traced input."""
    if not online:
        speech = speech * self._output_size**0.5

    actual_seq_len = speech.shape[1]
    if speech_mask is not None:
        mask = speech_mask.to(
            dtype=speech.dtype,
            device=speech.device,
        )
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        speech_lengths = mask.sum(dim=1).to(dtype=torch.long)
    else:
        if speech_lengths is None or speech_lengths[0].item() != actual_seq_len:
            speech_lengths = torch.tensor(
                [actual_seq_len],
                dtype=torch.long,
                device=speech.device,
            )
        mask = self.make_pad_mask(speech_lengths, max_seq_len=actual_seq_len)
    mask = self.prepare_mask(mask)
    xs_pad = speech if self.embed is None else self.embed(speech)

    encoder_outs = self.model.encoders0(xs_pad, mask)
    xs_pad = encoder_outs[0]
    encoder_outs = self.model.encoders(xs_pad, mask)
    xs_pad = self.model.after_norm(encoder_outs[0])

    if self.ctc_linear is not None:
        xs_pad = F.softmax(self.ctc_linear(xs_pad), dim=2)
    return xs_pad, speech_lengths


def _encoder_input_names(self):
    del self
    return ["speech", "enc_mask"]


def _encoder_output_names(self):
    del self
    return ["enc"]


def _encoder_dynamic_axes(self):
    del self
    return {
        "speech": {0: "batch_size", 1: "speech_length"},
        "enc_mask": {0: "batch_size", 1: "speech_length"},
        "enc": {0: "batch_size", 1: "enc_length"},
    }


def _rescale_input_hook(module, inputs, scale):
    del module
    return (inputs[0] / scale, *inputs[1:])


def _rescale_output_hook(module, inputs, output, scale):
    del module, inputs
    if isinstance(output, tuple):
        return (output[0] / scale, *output[1:])
    return output / scale


def prepare_encoder_for_export(model, encoder_name="SANMEncoder"):
    """Replace the training encoder with FunASR's ONNX export adapter."""
    encoder_class = tables.encoder_classes.get(encoder_name + "Export")
    if encoder_class is None:
        raise RuntimeError(f"FunASR encoder export class not found: {encoder_name}")
    model.encoder = encoder_class(model.encoder, onnx=True)
    model.encoder.prepare_mask = types.MethodType(
        _prepare_sanm_mask,
        model.encoder,
    )
    model.encoder.forward = types.MethodType(
        _sanm_encoder_forward,
        model.encoder,
    )


def bind_encoder_export_interface(model):
    """Attach the stable encoder ONNX interface to a shallow model view."""
    model.forward = types.MethodType(_encoder_forward, model)
    model.export_input_names = types.MethodType(_encoder_input_names, model)
    model.export_output_names = types.MethodType(_encoder_output_names, model)
    model.export_dynamic_axes = types.MethodType(_encoder_dynamic_axes, model)


def rescale_encoder_for_fp16(model, scale=ENCODER_FP16_SCALE):
    """Apply FunASR's FP16 overflow-avoidance transform to the encoder."""
    encoder = model.encoder.model
    encoder.encoders0.register_forward_pre_hook(
        functools.partial(_rescale_input_hook, scale=scale)
    )
    for name, module in encoder.named_modules():
        if name.endswith("self_attn"):
            module.register_forward_hook(
                functools.partial(_rescale_output_hook, scale=scale)
            )
        if name.endswith("feed_forward.w_2"):
            state = {key: value / scale for key, value in module.state_dict().items()}
            module.load_state_dict(state)
    print(f"Applied encoder FP16 rescale: {scale}")
