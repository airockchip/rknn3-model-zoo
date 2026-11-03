#!/usr/bin/env python3
"""Export Microsoft Mage-VL vision ONNX model to RKNN."""

import os

from rknn.api import RKNN


ONNX_MODEL = "../../model/vision/mage_vision_patch576_grq_w4a16.onnx"
RKNN_MODEL = "../../model/vision/mage_vision_patch576_grq_w4a16.rknn"
DATASET_PATH = None


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Mage-VL vision model to RKNN")
    parser.add_argument("--onnx_path", type=str, default=ONNX_MODEL, help="onnx model path")
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL, help="output rknn model path")
    parser.add_argument("--platform", type=str, default="rk1820", help="Target platform (e.g. rk1820)")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="model quantization dataset path")
    parser.add_argument("--core_num", type=int, default=8, help="core_num (1-8)")
    parser.add_argument("--save_ctx", action="store_true", help="save rknn context file")
    parser.add_argument("--no_quant", dest="quant", action="store_false", help="disable RKNN quantization")
    parser.set_defaults(quant=True)
    args = parser.parse_args()

    rknn = RKNN(verbose=True)

    print("--> config model")
    if args.quant:
        rknn.config(
            target_platform=args.platform,
            core_num=args.core_num,
            quantized_dtype="w4a16",
            quantized_algorithm="normal",
            quantized_method="group32",
        )
    else:
        rknn.config(target_platform=args.platform, core_num=args.core_num)
    print("done")

    print("--> Loading model")
    ret = rknn.load_onnx(model=args.onnx_path)
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    print("--> Building model")
    ret = rknn.build(do_quantization=args.quant, dataset=args.dataset_path)
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    print("--> Export RKNN model")
    export_dir = os.path.dirname(args.rknn_path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    ret = rknn.export_rknn(args.rknn_path, save_ctx=args.save_ctx)
    if ret != 0:
        print("Export rknn model failed! {}".format(args.rknn_path))
        exit(ret)
    print("done {}".format(args.rknn_path))

    rknn.release()
