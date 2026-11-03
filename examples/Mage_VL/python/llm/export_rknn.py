#!/usr/bin/env python3
"""Export Mage-VL Qwen3-4B LLM ONNX model to RKNN."""

import copy
import os

from rknn.api import DEFAULT_RKNN_LLM_CONFIG, RKNN


ONNX_MODEL = "../../model/llm/model_llm.onnx"
LLM_CONFIG = os.path.splitext(ONNX_MODEL)[0] + ".config.pkl"
RKNN_MODEL = "../../model/llm/model_llm_w4a16.rknn"
DATASET_PATH = None


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentTypeError

    def parse_seq_lens(value):
        try:
            seq_lens = [int(item.strip()) for item in value.split(",") if item.strip()]
        except ValueError as error:
            raise ArgumentTypeError(
                "seq_lens must be comma-separated positive integers"
            ) from error
        if not seq_lens or any(item <= 0 for item in seq_lens):
            raise ArgumentTypeError(
                "seq_lens must be comma-separated positive integers"
            )
        return seq_lens

    parser = ArgumentParser(description="Export Mage-VL Qwen3-4B llm to RKNN model")
    parser.add_argument("--onnx_path", type=str, default=ONNX_MODEL, help="onnx model path")
    parser.add_argument("--config", type=str, default=LLM_CONFIG, help="config file path")
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL, help="output rknn model path")
    parser.add_argument("--platform", type=str, default="rk1820", help="Target platform (e.g. rk1820)")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="model quantization dataset path")
    parser.add_argument("--save_ctx", action="store_true", help="save rknn context file")
    parser.add_argument("--context_len", type=int, default=4096,
                        help="KV cache capacity and maximum position")
    parser.add_argument("--seq_lens", type=parse_seq_lens, default=[1, 128],
                        help="compiled decode/prefill sequence lengths, e.g. 1,128")
    args = parser.parse_args()
    if args.context_len <= 0:
        parser.error("--context_len must be positive")

    llm_config = copy.deepcopy(DEFAULT_RKNN_LLM_CONFIG)
    llm_config["attention_config"][0]["kvcache_buffer_len"] = args.context_len
    llm_config["attention_config"][0]["max_position_embeddings"] = args.context_len
    llm_config["attention_config"][0]["kvcache_store_method"] = "GroupQuant"
    llm_config["attention_config"][0]["kvcache_dtype"] = "Int4_to_F16"
    llm_config["attention_config"][0]["kvcache_group_size"] = 16
    llm_config["attention_config"][0]["kvcache_residual_depth"] = 64

    rknn = RKNN(verbose=True)

    print("--> config model")
    ret = rknn.config(
        target_platform=args.platform,
        quantized_dtype="w4a16",
        quantized_algorithm="normal",
        quantized_method="group32",
        llm_config=llm_config,
    )
    if ret != 0:
        print("Config model failed!")
        exit(ret)
    print("done")

    print("--> Loading model")
    print("context_len: {}".format(args.context_len))
    print("seq_lens: {}".format(args.seq_lens))
    ret = rknn.load_llm(model=args.onnx_path, config=args.config,
                        seq_lens=args.seq_lens)
    if ret != 0:
        print("Load model failed!")
        exit(ret)
    print("done")

    print("--> Building model")
    ret = rknn.build(do_quantization=True, dataset=args.dataset_path)
    if ret != 0:
        print("Build model failed!")
        exit(ret)
    print("done")

    print("--> Export RKNN model")
    export_rknn_dirname = os.path.dirname(args.rknn_path)
    if export_rknn_dirname and not os.path.exists(export_rknn_dirname):
        os.makedirs(export_rknn_dirname, exist_ok=True)
    ret = rknn.export_rknn(args.rknn_path, save_ctx=args.save_ctx)
    if ret != 0:
        print("Export rknn model failed!")
        exit(ret)
    print("done")

    rknn.release()
