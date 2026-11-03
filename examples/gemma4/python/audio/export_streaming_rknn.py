#!/usr/bin/env python3
"""Convert a stateful Gemma4 Audio ONNX model to dynamic-bucket RKNN3."""

import argparse
import os

import onnx
from rknn.api import RKNN


MEL_BINS = 128
CACHE_TOKENS = 12
NUM_LAYERS = 12
NUM_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024

EXPECTED_INPUTS = [
    "input_features",
    "input_features_mask",
    "conv_state_0",
    "conv_state_1",
    "audio_history_mask",
    "audio_history_key",
    "audio_history_value",
    "past_conv",
]
EXPECTED_OUTPUTS = [
    "hidden_states",
    "output_mask",
    "next_conv_state_0",
    "next_conv_state_1",
    "audio_current_key",
    "audio_current_value",
    "next_conv",
]


def parse_buckets(value):
    buckets = sorted({int(item) for item in value.split(",") if item.strip()})
    if not buckets or any(bucket <= 0 or bucket % 48 for bucket in buckets):
        raise ValueError("buckets must be positive multiples of 48")
    return buckets


def shapes_for_bucket(mel_frames):
    return [
        [1, mel_frames, MEL_BINS],
        [1, mel_frames],
        [1, 1, 1, MEL_BINS],
        [1, 128, 1, 64],
        [1, CACHE_TOKENS],
        [1, NUM_LAYERS, NUM_HEADS, CACHE_TOKENS, HEAD_DIM],
        [1, NUM_LAYERS, NUM_HEADS, CACHE_TOKENS, HEAD_DIM],
        [1, NUM_LAYERS, HIDDEN_SIZE, 4],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_path", required=True)
    parser.add_argument("--rknn_path", required=True)
    parser.add_argument("--buckets", default="240,480,720")
    parser.add_argument("--platform", default="rk1820")
    parser.add_argument("--core_num", type=int, default=8)
    args = parser.parse_args()

    buckets = parse_buckets(args.buckets)
    dynamic_input = [shapes_for_bucket(bucket) for bucket in buckets]
    model = onnx.load(args.onnx_path, load_external_data=False)
    input_names = [value.name for value in model.graph.input]
    output_names = [value.name for value in model.graph.output]
    if input_names != EXPECTED_INPUTS or output_names != EXPECTED_OUTPUTS:
        raise RuntimeError(
            "unexpected streaming interface:\n"
            f"inputs={input_names}\noutputs={output_names}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.rknn_path)), exist_ok=True)
    input_attrs = {
        name: {"dtype": "float16", "layout": "UNDEFINED"} for name in input_names
    }
    output_attrs = {
        name: {"dtype": "float16", "layout": "UNDEFINED"} for name in output_names
    }

    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(
            target_platform=args.platform,
            core_num=args.core_num,
            profile_mode=False,
            input_attrs=input_attrs,
            output_attrs=output_attrs,
            dynamic_input=dynamic_input,
        )
        if ret != 0:
            raise RuntimeError(f"RKNN config failed: {ret}")
        ret = rknn.load_onnx(model=args.onnx_path)
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed: {ret}")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"RKNN build failed: {ret}")
        ret = rknn.export_rknn(args.rknn_path, save_ctx=False)
        if ret != 0:
            raise RuntimeError(f"RKNN export failed: {ret}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
