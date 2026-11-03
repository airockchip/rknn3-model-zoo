#!/usr/bin/env python3
# Copyright (c) 2026 Rockchip Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Append the Paraformer alpha predictor head to an encoder ONNX graph."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ENCODER_DIM = 512
CONV_KERNEL = 3


def _predictor_weights(predictor):
    """Return alpha-head weights in the layout required by ONNX operators."""
    conv_weight = predictor.cif_conv1d.weight.detach().cpu().numpy().astype(np.float32)
    conv_bias = predictor.cif_conv1d.bias.detach().cpu().numpy().astype(np.float32)
    linear_weight = (
        predictor.cif_output.weight.detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        .reshape(ENCODER_DIM, 1)
    )
    linear_bias = predictor.cif_output.bias.detach().cpu().numpy().astype(np.float32)
    expected_shapes = (
        (ENCODER_DIM, ENCODER_DIM, CONV_KERNEL),
        (ENCODER_DIM,),
        (ENCODER_DIM, 1),
        (1,),
    )
    tensors = (conv_weight, conv_bias, linear_weight, linear_bias)
    actual_shapes = tuple(tensor.shape for tensor in tensors)
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            f"Unexpected alpha-head weight shapes: {actual_shapes}, "
            f"expected {expected_shapes}"
        )
    return tensors


def append_alpha_head(encoder_path, predictor, output_path):
    """Create an encoder graph that also predicts frame-level CIF alphas."""
    encoder_path = Path(encoder_path)
    output_path = Path(output_path)
    model = onnx.load(str(encoder_path), load_external_data=False)
    graph = model.graph
    input_names = [value.name for value in graph.input]
    output_names = [value.name for value in graph.output]
    if input_names != ["speech", "enc_mask"] or output_names != ["enc"]:
        raise RuntimeError(
            f"Unexpected encoder interface: inputs={input_names}, "
            f"outputs={output_names}"
        )

    alpha_names = {
        "alphas",
        "predictor_alpha_conv_weight",
        "predictor_alpha_conv_bias",
        "predictor_alpha_linear_weight",
        "predictor_alpha_linear_bias",
        "predictor_alpha_masked_enc",
        "predictor_alpha_nct",
        "predictor_alpha_conv",
        "predictor_alpha_relu",
        "predictor_alpha_ntc",
        "predictor_alpha_mm",
        "predictor_alpha_logits",
        "predictor_alpha_raw",
    }
    used_names = {
        name
        for node in graph.node
        for name in (*node.input, *node.output)
        if name
    }
    used_names.update(value.name for value in graph.input)
    used_names.update(value.name for value in graph.output)
    used_names.update(value.name for value in graph.initializer)
    collision = sorted(used_names & alpha_names)
    if collision:
        raise RuntimeError(f"Alpha-head tensor name collision: {collision}")

    conv_weight, conv_bias, linear_weight, linear_bias = _predictor_weights(
        predictor
    )
    graph.initializer.extend(
        [
            numpy_helper.from_array(conv_weight, "predictor_alpha_conv_weight"),
            numpy_helper.from_array(conv_bias, "predictor_alpha_conv_bias"),
            numpy_helper.from_array(
                linear_weight, "predictor_alpha_linear_weight"
            ),
            numpy_helper.from_array(linear_bias, "predictor_alpha_linear_bias"),
        ]
    )
    graph.node.extend(
        [
            helper.make_node(
                "Mul",
                ["enc", "enc_mask"],
                ["predictor_alpha_masked_enc"],
                name="PredictorAlpha/MaskEncoderOutput",
            ),
            helper.make_node(
                "Transpose",
                ["predictor_alpha_masked_enc"],
                ["predictor_alpha_nct"],
                name="PredictorAlpha/ToNCT",
                perm=[0, 2, 1],
            ),
            helper.make_node(
                "Conv",
                [
                    "predictor_alpha_nct",
                    "predictor_alpha_conv_weight",
                    "predictor_alpha_conv_bias",
                ],
                ["predictor_alpha_conv"],
                name="PredictorAlpha/Conv1D",
                kernel_shape=[CONV_KERNEL],
                pads=[1, 1],
                strides=[1],
            ),
            helper.make_node(
                "Relu",
                ["predictor_alpha_conv"],
                ["predictor_alpha_relu"],
                name="PredictorAlpha/Relu",
            ),
            helper.make_node(
                "Transpose",
                ["predictor_alpha_relu"],
                ["predictor_alpha_ntc"],
                name="PredictorAlpha/ToNTC",
                perm=[0, 2, 1],
            ),
            helper.make_node(
                "MatMul",
                ["predictor_alpha_ntc", "predictor_alpha_linear_weight"],
                ["predictor_alpha_mm"],
                name="PredictorAlpha/Linear",
            ),
            helper.make_node(
                "Add",
                ["predictor_alpha_mm", "predictor_alpha_linear_bias"],
                ["predictor_alpha_logits"],
                name="PredictorAlpha/AddBias",
            ),
            helper.make_node(
                "Sigmoid",
                ["predictor_alpha_logits"],
                ["predictor_alpha_raw"],
                name="PredictorAlpha/Sigmoid",
            ),
            helper.make_node(
                "Mul",
                ["predictor_alpha_raw", "enc_mask"],
                ["alphas"],
                name="PredictorAlpha/MaskAlphas",
            ),
        ]
    )
    graph.output.append(
        helper.make_tensor_value_info(
            "alphas",
            TensorProto.FLOAT,
            ["batch_size", "enc_length", 1],
        )
    )

    model.producer_name = "Rockchip Paraformer encoder-alpha experiment"
    model.doc_string = (
        "SeACo Paraformer encoder with Conv1D/ReLU/Linear/Sigmoid alpha head. "
        "Dynamic CIF accumulation remains on the CPU."
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    onnx.save(model, str(temporary_path))
    temporary_path.replace(output_path)
    print(f"Appended alpha head: {output_path}")
