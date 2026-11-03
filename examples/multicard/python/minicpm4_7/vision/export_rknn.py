#!/usr/bin/env python3
"""Convert the MiniCPM-V 4.7 vision ONNX model to RKNN.

The MiniCPM-V 4.7 graph uses the packed NaViT inputs emitted by
``export_vision.py``:

* ``pixel_values``: ``[1, 3, 14, 14336]`` for a 448x448 image;
* ``target_sizes``: ``[1, 2]`` containing ``[[32, 32]]``.

Unlike the Qwen-VL exporter, this graph does not have ``pixel``/``grid_thw``
inputs and should not use image mean/std preprocessing inside RKNN.
"""

import argparse
import json
import os

import numpy as np

from rknn.api import RKNN


MODEL_BASENAME = "MiniCPM-V-4.7-35B-A3B_vision"
DEFAULT_ONNX = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../model/vision/minicpm4_7_35b_a3b_vision",
        MODEL_BASENAME + ".onnx",
    )
)
DEFAULT_RKNN = os.path.splitext(DEFAULT_ONNX)[0] + ".rknn"


def load_input_shapes(onnx_path):
    """Read the fixed MiniCPM input shapes and patch grid next to the ONNX model."""
    model_dir = os.path.dirname(os.path.abspath(onnx_path))
    config_path = os.path.join(model_dir, "vision_config.json")
    if not os.path.isfile(config_path):
        return (
            [[1, 3, 14, 14336], [1, 2]],
            np.array([[32, 32]], dtype=np.int32),
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    pixel_shape = config.get("pixel_values_shape", [1, 3, 14, 14336])
    target_sizes = np.asarray(config.get("target_sizes", [[32, 32]]), dtype=np.int32)
    if target_sizes.ndim != 2 or target_sizes.shape[1] != 2:
        raise ValueError(
            f"Invalid target_sizes in {config_path}: expected [N, 2], "
            f"got {target_sizes.shape}"
        )
    target_shape = list(target_sizes.shape)
    return [list(map(int, pixel_shape)), target_shape], target_sizes


def main():
    parser = argparse.ArgumentParser(
        description="Export MiniCPM-V 4.7 vision ONNX model to RKNN"
    )
    parser.add_argument("--onnx_path", type=str, default=DEFAULT_ONNX)
    parser.add_argument("--rknn_path", type=str, default=DEFAULT_RKNN)
    parser.add_argument("--platform", type=str, default="rk1820")
    parser.add_argument("--core_num", type=int, default=8)
    parser.add_argument(
        "--quantized_dtype",
        choices=["w4a16", "w8a16", "fp16"],
        default="w4a16",
        help="RKNN weight format; default is w4a16",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="optional calibration dataset for RKNN normal quantization",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite an existing RKNN output",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {args.onnx_path}")
    if os.path.exists(args.rknn_path) and not args.overwrite:
        raise FileExistsError(
            f"RKNN already exists: {args.rknn_path}; use --overwrite to rebuild"
        )

    input_size_list, target_sizes = load_input_shapes(args.onnx_path)
    input_names = ["pixel_values", "target_sizes"]
    print(f"ONNX: {os.path.abspath(args.onnx_path)}")
    print(f"RKNN: {os.path.abspath(args.rknn_path)}")
    print(f"input_size_list: {input_size_list}")
    print(f"static target_sizes: {target_sizes.tolist()} ({target_sizes.dtype})")
    print("RKNN runtime inputs after constant folding: ['pixel_values']")

    rknn = RKNN(verbose=True)
    try:
        print("--> config model")
        if args.quantized_dtype == "fp16":
            rknn.config(target_platform=args.platform, core_num=args.core_num)
            do_quantization = False
        else:
            rknn.config(
                target_platform=args.platform,
                core_num=args.core_num,
                quantized_dtype=args.quantized_dtype,
                quantized_algorithm="normal",
                quantized_method="group32",
            )
            do_quantization = True
        print("done")

        print("--> Loading ONNX model")
        ret = rknn.load_onnx(
            model=args.onnx_path,
            inputs=input_names,
            input_size_list=input_size_list,
            # target_sizes describes the fixed 448x448 patch grid.  Supplying
            # it here prevents RKNN's ONNX checker from using [[0, 0]], which
            # would create a zero-length position embedding.  gongga-toolkit
            # turns this input into an initializer, so the exported RKNN only
            # needs pixel_values at runtime.
            input_initial_val=[None, target_sizes],
        )
        if ret != 0:
            raise RuntimeError(f"load_onnx failed with code {ret}")
        print("done")

        print("--> Building RKNN model")
        ret = rknn.build(
            do_quantization=do_quantization,
            dataset=args.dataset_path if do_quantization else None,
        )
        if ret != 0:
            raise RuntimeError(f"build failed with code {ret}")
        print("done")

        output_dir = os.path.dirname(os.path.abspath(args.rknn_path))
        os.makedirs(output_dir, exist_ok=True)
        print("--> Exporting RKNN model")
        ret = rknn.export_rknn(args.rknn_path)
        if ret != 0:
            raise RuntimeError(f"export_rknn failed with code {ret}")
        print(f"done: {os.path.abspath(args.rknn_path)}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
