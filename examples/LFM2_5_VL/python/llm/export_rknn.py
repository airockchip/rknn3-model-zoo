#!/usr/bin/env python3
"""Convert LFM2.5-VL LLM Conv Streaming ONNX to RKNN.

Supports both 1.6B (10 conv layers) and 3B (22 conv layers) variants.

技术路线：
  - 标准 4 输入 convstream.onnx（Conv pads 已改为 [2,0]）
  - 自动收集 ShortConv 输出
  - 传给 rknn.config(cvt_conv_streaming=...)
  - W4A16 量化 (quantized_dtype="w4a16", quantized_algorithm="normal",
    quantized_method="group32")
  - Runtime 内部维护 Conv Cache
"""
import sys
import json
import copy
import argparse
import traceback
from pathlib import Path

import numpy as np
import onnx

from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG

_SCRIPT_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _SCRIPT_DIR.parent.parent / "model" / "llm"

# Model variant configurations
MODEL_CONFIGS = {
    "1.6B": {
        "output_basename": "LFM2.5-VL-1.6B",
        "expected_conv_count": 10,
    },
    "3B": {
        "output_basename": "LFM2.5-VL-3B",
        "expected_conv_count": 22,
    },
}

EXPECTED_INPUT_NAMES = {"input_ids", "attention_mask", "position_ids", "num_logits_to_keep"}


def _onnx_io(model_proto):
    inputs = [inp.name for inp in model_proto.graph.input]
    outputs = [out.name for out in model_proto.graph.output]
    return inputs, outputs


_ONNX_ETYPE_TO_NP = {
    1: np.float32,   # FLOAT
    6: np.int32,     # INT32
    7: np.int64,     # INT64
    9: np.bool_,     # BOOL
    10: np.float16,  # FLOAT16
}


def _onnx_elem_type_to_np(elem_type):
    """ONNX TensorProto elem_type -> numpy dtype（缺省按 float32 处理）。"""
    return _ONNX_ETYPE_TO_NP.get(elem_type, np.float32)


def _conv_kernel(model_proto, node, attrs):
    """返回 1D Conv 的 kernel size [k]。

    Dynamo exporter（torch>=2.4 / onnxscript 后端）可能省略 kernel_shape 属性，
    此时从权重张量（initializer 或 Constant）的最后一维推断。
    1D Conv 权重形状: [M, C/group, k]。
    """
    kernel = attrs.get("kernel_shape")
    if kernel:
        return list(kernel)
    if len(node.input) < 2:
        return None
    w_name = node.input[1]
    for init in model_proto.graph.initializer:
        if init.name == w_name and init.dims:
            return [int(list(init.dims)[-1])]
    for cn in model_proto.graph.node:
        if cn.op_type == "Constant" and cn.output and cn.output[0] == w_name:
            for a in cn.attribute:
                if a.name == "value" and a.t.dims:
                    return [int(list(a.t.dims)[-1])]
    return None


def collect_lfm2_streaming_conv_outputs(onnx_path, hidden_size=2048, expected_conv_count=10):
    """从 convstream.onnx 中收集 LFM2 ShortConv 的输出 tensor 名。"""
    import onnx

    print(f"--> Collecting LFM2 streaming Conv outputs from: {onnx_path}")
    try:
        model_proto = onnx.load(onnx_path, load_external_data=False)
    except Exception as e:
        print(f"[ERROR] Cannot load ONNX for Conv collection: {e}")
        sys.exit(1)

    # Validate I/O protocol
    input_names, output_names = _onnx_io(model_proto)
    if len(input_names) != 4:
        print(f"[ERROR] ONNX must have 4 inputs, got {len(input_names)}: {input_names}")
        sys.exit(1)
    if len(output_names) != 1:
        print(f"[ERROR] ONNX must have 1 output, got {len(output_names)}: {output_names}")
        sys.exit(1)
    missing = EXPECTED_INPUT_NAMES - set(input_names)
    if missing:
        print(f"[ERROR] Missing required inputs: {sorted(missing)}")
        sys.exit(1)

    conv_nodes = [n for n in model_proto.graph.node if n.op_type == "Conv"]
    print(f"  Total Conv nodes: {len(conv_nodes)}")

    target_outputs = []
    for node in conv_nodes:
        output_name = node.output[0] if node.output else ""
        attrs = {}
        for attr in node.attribute:
            if attr.name == "pads":
                attrs["pads"] = list(attr.ints)
            elif attr.name == "kernel_shape":
                attrs["kernel_shape"] = list(attr.ints)
            elif attr.name == "dilations":
                attrs["dilations"] = list(attr.ints)
            elif attr.name == "strides":
                attrs["strides"] = list(attr.ints)
            elif attr.name == "group":
                attrs["group"] = attr.i

        kernel = _conv_kernel(model_proto, node, attrs)
        pads = attrs.get("pads", [])
        dilations = attrs.get("dilations", [1])
        strides = attrs.get("strides", [1])
        group = attrs.get("group", 0)

        if (kernel == [3] and dilations == [1] and strides == [1]
                and group == hidden_size and pads == [2, 0]):
            target_outputs.append(output_name)
            print(f"  MATCH [{len(target_outputs)-1}]: {output_name}")

    if len(target_outputs) != expected_conv_count:
        print(f"\n[ERROR] Expected {expected_conv_count} ShortConv with pads=[2,0], "
              f"found {len(target_outputs)}")
        print("[ERROR] All Conv nodes for debugging:")
        for node in conv_nodes:
            attrs = {}
            for attr in node.attribute:
                if attr.name == "pads":
                    attrs["pads"] = list(attr.ints)
                elif attr.name == "kernel_shape":
                    attrs["kernel_shape"] = list(attr.ints)
                elif attr.name == "group":
                    attrs["group"] = attr.i
            print(f"  {node.output[0] if node.output else ''}: {attrs} "
                  f"kernel(inferred)={_conv_kernel(model_proto, node, attrs)}")
        sys.exit(1)

    if len(set(target_outputs)) != len(target_outputs):
        print(f"[ERROR] Duplicate Conv output names: {target_outputs}")
        sys.exit(1)

    print(f"\n  Collected {len(target_outputs)} Conv streaming outputs")
    return target_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Convert LFM2.5-VL LLM Conv Streaming ONNX to RKNN. "
                    "Supports 1.6B and 3B variants."
    )
    parser.add_argument("--model_size", type=str, default="1.6B", choices=list(MODEL_CONFIGS.keys()),
                        help="Model variant (default: 1.6B)")
    parser.add_argument("--onnx_path", type=str, default=None,
                        help="Conv streaming ONNX path")
    parser.add_argument("--config", type=str, default=None,
                        help="LLM config.pkl path")
    parser.add_argument("--rknn_path", type=str, default=None,
                        help="Output RKNN model path")
    parser.add_argument("--kvcache", type=int, default=4096,
                        help="KV cache buffer length (kvcache_buffer_len, default 4096)")
    parser.add_argument("--context", type=int, default=8192,
                        help="Max context length (max_position_embeddings, default 8192)")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Prefill seq_len for load_llm (default: 128)")
    parser.add_argument("--target_platform", type=str, default="rk1820",
                        choices=["rk1820"],
                        help="Target platform (locked to rk1820)")
    parser.add_argument("--hidden_size", type=int, default=2048,
                        help="Hidden size of LLM (for conv matching)")
    parser.add_argument("--no_conv_streaming", action="store_true",
                        help="Disable cvt_conv_streaming (debug only)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="model quantization dataset path")
    parser.add_argument("--no_quant", action="store_true",
                        help="Export FP16 model (do_quantization=False, no w4a16)")
    parser.add_argument("--quant_type", type=str, default="w4a16", choices=["w4a16", "w8a16"],
                        help="Weight quantization dtype (default: w4a16). w8a16 = int8 weights / fp16 activations.")
    args = parser.parse_args()

    # Resolve model config
    cfg = MODEL_CONFIGS[args.model_size]
    OUTPUT_BASENAME = cfg["output_basename"]
    EXPECTED_CONV_COUNT = cfg["expected_conv_count"]

    DEFAULT_ONNX = _MODEL_DIR / f"{OUTPUT_BASENAME}-convstream.onnx"
    DEFAULT_CONFIG = _MODEL_DIR / f"{OUTPUT_BASENAME}.config.pkl"
    DEFAULT_RKNN = _MODEL_DIR / f"{OUTPUT_BASENAME}-convstream.rknn"

    if args.onnx_path is None:
        args.onnx_path = str(DEFAULT_ONNX)
    if args.config is None:
        args.config = str(DEFAULT_CONFIG)
    if args.rknn_path is None:
        args.rknn_path = str(DEFAULT_RKNN)

    print(f"--> Model variant: {args.model_size} (basename={OUTPUT_BASENAME}, conv_count={EXPECTED_CONV_COUNT})")

    if args.no_quant and args.rknn_path == str(DEFAULT_RKNN):
        # avoid overwriting the quantized model when exporting FP16
        args.rknn_path = str(_MODEL_DIR / f"{OUTPUT_BASENAME}-convstream-fp16.rknn")
    if not args.no_quant and args.quant_type == "w8a16" and args.rknn_path == str(DEFAULT_RKNN):
        # avoid overwriting the w4a16 model when exporting w8a16
        args.rknn_path = str(_MODEL_DIR / f"{OUTPUT_BASENAME}-convstream-w8a16.rknn")
    if args.no_quant:
        quant_desc = "NO (FP16)"
    else:
        quant_desc = f"YES ({args.quant_type})"
    print(f"[INFO] quantized={quant_desc}")

    if args.seq_len <= 1:
        print(f"[ERROR] --seq_len must be > 1, got {args.seq_len}")
        sys.exit(1)

    onnx_p = Path(args.onnx_path)
    config_p = Path(args.config)
    rknn_p = Path(args.rknn_path)
    weight_p = rknn_p.with_suffix(".weight")

    # Input file checks
    if not onnx_p.exists():
        print(f"[ERROR] ONNX model not found: {onnx_p}")
        sys.exit(1)
    if not config_p.exists():
        print(f"[ERROR] Config file not found: {config_p}")
        sys.exit(1)

    # Collect Conv streaming outputs
    if args.no_conv_streaming:
        print("[DEBUG] cvt_conv_streaming DISABLED (isolation run)")
        cvt_conv_streaming = []
    else:
        cvt_conv_streaming = collect_lfm2_streaming_conv_outputs(args.onnx_path, args.hidden_size, EXPECTED_CONV_COUNT)

    # ---- 与官方 LFM2.5-1.2B-Instruct 保持一致：不使用 dynamic_input / input_initial_value ----
    # 简化 wrapper 后（去除 index_select+clamp），图结构与 1.2B 一致，
    # 不再需要 dynamic_input/input_initial_value 来规避 register 段错误。
    dynamic_shapes = None
    input_initial_value = None

    # Clean old artifacts
    for p in (rknn_p, weight_p):
        if p.exists():
            p.unlink()

    print(f"\n--> Conversion parameters:")
    print(f"    Body quantization       : {args.quant_type} / normal / {'channel' if args.quant_type == 'w8a16' else 'group32'}")
    print(f"    KV cache buffer length  : {args.kvcache}")
    print(f"    Max position embeddings : {args.context}")
    print(f"    KV cache dtype          : Float16")
    print(f"    Decode seq_len          : 1")
    print(f"    Prefill seq_len         : {args.seq_len}")
    print(f"    Conv streaming nodes    : {EXPECTED_CONV_COUNT}")

    # RKNN conversion
    rknn = RKNN(verbose=True)
    try:
        # Config
        print("--> Configuring model")
        llm_config = copy.deepcopy(DEFAULT_RKNN_LLM_CONFIG)
        attn_cfg = llm_config["attention_config"][0]
        attn_cfg["kvcache_buffer_len"] = args.kvcache
        attn_cfg["max_position_embeddings"] = args.context
        attn_cfg["kvcache_dtype"] = "Float16"

        config_kwargs = dict(
            target_platform=args.target_platform,
            core_num=8,
            llm_config=llm_config,
            cvt_conv_streaming=cvt_conv_streaming,
        )
        if not args.no_quant:
            if args.quant_type == "w8a16":
                config_kwargs.update(
                    quantized_dtype="w8a16",
                    quantized_algorithm="normal",
                    quantized_method="channel",
                )
            else:
                config_kwargs.update(
                    quantized_dtype="w4a16",
                    quantized_algorithm="normal",
                    quantized_method="group32",
                )
        ret = rknn.config(**config_kwargs)
        if ret != 0:
            print(f"[FAILED] rknn.config() ret={ret}")
            sys.exit(1)
        print("done")

        # Load LLM
        print("--> Loading LLM model")
        ret = rknn.load_llm(
            model=args.onnx_path,
            config=args.config,
            seq_lens=[1, args.seq_len],
        )
        if ret != 0:
            print(f"[FAILED] rknn.load_llm() ret={ret}")
            sys.exit(1)
        print("done")

        # Build
        print(f"--> Building model ({'FP16' if args.no_quant else args.quant_type + ' quantization'})")
        ret = rknn.build(do_quantization=not args.no_quant, dataset=None if args.no_quant else args.dataset_path)
        if ret != 0:
            print(f"[FAILED] rknn.build() ret={ret}")
            sys.exit(1)
        print("done")

        # Export
        print("--> Exporting RKNN model")
        rknn_out_dir = rknn_p.parent
        if rknn_out_dir and not rknn_out_dir.exists():
            rknn_out_dir.mkdir(parents=True, exist_ok=True)

        ret = rknn.export_rknn(args.rknn_path, save_ctx=True)
        if ret != 0:
            print(f"[FAILED] rknn.export_rknn() ret={ret}")
            sys.exit(1)
        print("done")

    except SystemExit:
        raise
    except Exception as e:
        print(f"\n[ERROR] Unexpected exception: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        rknn.release()

    print("\n" + "=" * 60)
    print("RKNN Conv Streaming export complete!")
    print("=" * 60)
    print(f"  cvt_conv_streaming: {len(cvt_conv_streaming)} conv nodes")
    print(f"  RKNN model : {rknn_p}")
    print(f"  Weight     : {weight_p}")
    print(f"  RKNN size  : {rknn_p.stat().st_size:,} bytes "
          f"({rknn_p.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  Weight size: {weight_p.stat().st_size:,} bytes "
          f"({weight_p.stat().st_size / 1024 / 1024:.2f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
