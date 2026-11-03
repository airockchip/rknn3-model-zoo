import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import onnx
from rknn.api import DEFAULT_RKNN_LLM_CONFIG, RKNN


def compat_shim():
    if not hasattr(onnx, "_mapping"):
        return
    if not hasattr(onnx, "mapping"):
        onnx.mapping = onnx._mapping
    if not hasattr(onnx._mapping, "TENSOR_TYPE_TO_NP_TYPE") and hasattr(onnx._mapping, "TENSOR_TYPE_MAP"):
        onnx._mapping.TENSOR_TYPE_TO_NP_TYPE = {k: v.np_dtype for k, v in onnx._mapping.TENSOR_TYPE_MAP.items()}
    if not hasattr(onnx._mapping, "NP_TYPE_TO_TENSOR_TYPE") and hasattr(onnx._mapping, "TENSOR_TYPE_TO_NP_TYPE"):
        onnx._mapping.NP_TYPE_TO_TENSOR_TYPE = {np.dtype(v): k for k, v in onnx._mapping.TENSOR_TYPE_TO_NP_TYPE.items()}


def check_file(path, desc):
    path = Path(path)
    if not path.exists():
        print(f"{desc} not found: {path}")
        return False
    return True


def check_ret(ret, msg):
    if ret != 0:
        print(f"{msg} failed: {ret}")
        sys.exit(ret)


def stage_paths(args, stage):
    onnx_dir = Path(args.onnx_dir)
    onnx_path = onnx_dir / f"whisper_{stage}_{args.suffix}.onnx"
    rknn_path = onnx_dir / f"whisper_{stage}_{args.suffix}.rknn"
    config_path = onnx_dir / f"whisper_{stage}_{args.suffix}.config.pkl"
    return onnx_path, rknn_path, config_path


def validate_decode1_config(cfg):
    stage = cfg.get("stage")
    if stage and "decode1" not in stage:
        raise RuntimeError(f"Unsupported decode1 config stage for load_llm: {stage}")
    required_keys = ("num_layers", "num_heads", "head_dim", "hidden_size", "encoder_seq_len")
    missing_keys = [key for key in required_keys if key not in cfg]
    if missing_keys:
        raise RuntimeError(f"Invalid decode1 config for load_llm, missing keys: {missing_keys}")


def load_decode1_config(config_path):
    if not check_file(config_path, "decode1 config"):
        return None
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    validate_decode1_config(cfg)
    return cfg


def make_decode1_shapes(cfg, seq_len):
    num_layers = cfg["num_layers"]
    num_heads = cfg["num_heads"]
    head_dim = cfg["head_dim"]
    encoder_seq_len = cfg["encoder_seq_len"]

    default_cross_shape = [1, num_heads, encoder_seq_len, head_dim]
    cross_shape = cfg.get("cross_kv_shape", default_cross_shape)
    return [[1, seq_len], [1, 1, seq_len, seq_len], [1, seq_len], [1]] + [cross_shape] * (num_layers * 2)


def make_llm_config():
    llm_config = copy.deepcopy(DEFAULT_RKNN_LLM_CONFIG)
    llm_config["attention_config"][0]["position_name"] = None
    llm_config["keep_one_logit"] = []
    # Keep the projection in FP16. The toolkit default may otherwise select a
    # weight-only LLM-head quantization policy.
    llm_config["llm_head"] = []
    return llm_config


def export_encoder(args):
    onnx_path, rknn_path, _ = stage_paths(args, "encoder")
    if not check_file(onnx_path, "encoder ONNX"):
        return

    rknn = RKNN()
    try:
        print("--> Config encoder")
        config_kwargs = {
            "target_platform": args.target_platform,
            "core_num": 1 if args.target_platform == "rk3572" else 8,
        }

        rknn.config(**config_kwargs)
        print("done")

        print(f"--> Loading encoder ONNX: {onnx_path}")
        check_ret(rknn.load_onnx(model=str(onnx_path)), "load_onnx")
        print("done")

        print("--> Building encoder")
        check_ret(rknn.build(do_quantization=False), "build")
        print("done")

        print(f"--> Export encoder RKNN: {rknn_path}")
        check_ret(rknn.export_rknn(str(rknn_path)), "export_rknn")
        print(f"saved {rknn_path}")
    finally:
        rknn.release()


def export_decode0(args):
    onnx_path, rknn_path, _ = stage_paths(args, "decode0")
    if not check_file(onnx_path, "decode0 ONNX"):
        return

    rknn = RKNN()
    try:
        print("--> Config decode0")
        config_kwargs = {
            "target_platform": args.target_platform,
            "core_num": 1 if args.target_platform == "rk3572" else 8,
        }

        rknn.config(**config_kwargs)
        print("done")

        print(f"--> Loading decode0 ONNX: {onnx_path}")
        check_ret(rknn.load_onnx(model=str(onnx_path)), "load_onnx")
        print("done")

        print("--> Building decode0")
        check_ret(rknn.build(do_quantization=False), "build")
        print("done")

        print(f"--> Export decode0 RKNN: {rknn_path}")
        check_ret(rknn.export_rknn(str(rknn_path)), "export_rknn")
        print(f"saved {rknn_path}")
    finally:
        rknn.release()


def export_decode1(args):
    onnx_path, rknn_path, config_path = stage_paths(args, "decode1")
    if not check_file(onnx_path, "decode1 ONNX"):
        return

    cfg = load_decode1_config(config_path)
    if cfg is None:
        return

    # rk1820 exAttention 算子 seq_len 硬件上限为 352，默认取 256（32 对齐且留余量）
    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else 256
    dynamic_input = [make_decode1_shapes(cfg, 1), make_decode1_shapes(cfg, max_seq_len)]
    print(f"decode1 dynamic_input={dynamic_input}")

    rknn = RKNN()
    try:
        print("--> Config decode1 load_llm")
        config_kwargs = {
            "target_platform": args.target_platform,
            "core_num": 1 if args.target_platform == "rk3572" else 8,
            "llm_config": make_llm_config(),
            "dynamic_input": dynamic_input,
        }

        rknn.config(**config_kwargs)
        print("done")

        print(f"--> Loading decode1 LLM ONNX: {onnx_path}")
        check_ret(rknn.load_llm(model=str(onnx_path), config=str(config_path)), "load_llm")
        print("done")

        print("--> Building decode1")
        check_ret(rknn.build(do_quantization=False), "build")
        print("done")

        print(f"--> Export decode1 RKNN: {rknn_path}")
        check_ret(rknn.export_rknn(str(rknn_path)), "export_rknn")
        print(f"saved {rknn_path}")
    finally:
        rknn.release()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Whisper models to RKNN")
    parser.add_argument("--onnx_dir", type=str, default="whisper-base-model", help="directory containing Whisper ONNX/config files")
    parser.add_argument("--suffix", type=str, default="base", help="model suffix used by ONNX filenames, e.g. base/large-v3")
    parser.add_argument("--model", type=str, default="all", choices=["encoder", "decode0", "decode1", "all"], help="which part to export")
    parser.add_argument("--target_platform", type=str, default="rk1820", help="RKNN target platform")
    parser.add_argument("--max_seq_len", type=int, default=0, help="decode1 max sequence length; 0 means default 256")
    args = parser.parse_args()

    compat_shim()

    if args.model in ("encoder", "all"):
        export_encoder(args)
    if args.model in ("decode0", "all"):
        export_decode0(args)
    if args.model in ("decode1", "all"):
        export_decode1(args)
