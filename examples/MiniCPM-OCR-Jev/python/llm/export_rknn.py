"""把 value head 的 LLM ONNX 转成 RKNN3 可部署的 ``.rknn`` + ``.weight``。

用法::

    python export_rknn.py --onnx_path ../../model/llm/value_llm.onnx
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx

CHUNK_LENGTH = 128
QUANTIZED_METHOD = "group32"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the value-head LLM ONNX to RKNN3 (.rknn + .weight)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--onnx_path", required=True, help="输入 LLM ONNX（与 .config.pkl 同名同目录）")
    parser.add_argument(
        "--output_path",
        default=None,
        help="输出 .rknn 路径；默认与 ONNX 同目录同名",
    )
    parser.add_argument(
        "--target_platform",
        choices=("rk1820", "rk1828"),
        default="rk1820",
        help="RKNN 目标平台",
    )
    parser.add_argument("--core_num", type=int, default=8, help="NPU 核数，1-8")
    parser.add_argument(
        "--quantized_dtype",
        choices=("w4a16", "w6a16", "w8a16"),
        default="w4a16",
        help="权重量化类型",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=4096,
        help=f"最大上下文长度，至少 {CHUNK_LENGTH}",
    )
    parser.add_argument(
        "--keep_all_logits",
        action="store_true",
        help=(
            "不裁剪 logits 输出。默认（推荐）保留 keep_one_logit 配置，"
            "让 RKNN3 复用 num_logits_to_keep 只算最后 1 个 token 的分类结果"
        ),
    )
    parser.add_argument(
        "--report_source",
        default=None,
        help="RKNN3 生成的 tmp/model_report.html 位置；默认 ./tmp/model_report.html",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.core_num <= 8:
        parser.error("--core_num 必须在 [1, 8] 区间")
    if args.context_length < CHUNK_LENGTH:
        parser.error(f"--context_length 不能小于 {CHUNK_LENGTH}")
    return args


# --------------------------------------------------------------------------- #
# ONNX 预处理
# --------------------------------------------------------------------------- #


def shape_of(value_info) -> list[int]:
    return [dim.dim_value if dim.dim_value else 1 for dim in value_info.type.tensor_type.shape.dim]


def prepare_onnx(onnx_path: Path, chunk_length: int = CHUNK_LENGTH):
    """实例化动态轴、删除 conv_state_out 输出、收集 Conv 节点名。

    注意 ``load_external_data=False``：value head 的 ONNX 权重超过 2GB，
    torch.onnx.export 已经把它们切成 external data sidecar 文件了。
    我们不修改权重，所以不需要把几个 GB 读进内存。
    """
    model = onnx.load(str(onnx_path), load_external_data=False)
    base_shapes = [shape_of(value) for value in model.graph.input]

    dynamic_shapes = []
    for sequence_length in (chunk_length,):
        shapes = [list(shape) for shape in base_shapes]
        # 前 3 个输入是 input_ids / position_ids / attention_mask，都带 sequence 轴
        for index in range(min(3, len(shapes))):
            shapes[index] = [1, sequence_length]
        dynamic_shapes.append(shapes)

    conv_outputs = [output for output in model.graph.output if output.name == "conv_state_out"]
    if conv_outputs:
        remaining = [output for output in model.graph.output if output.name != "conv_state_out"]
        del model.graph.output[:]
        model.graph.output.extend(remaining)
        converted_path = onnx_path.with_name(onnx_path.stem + "_rknn_input.onnx")
        onnx.save_model(model, str(converted_path))
        print(f"    已剥离 conv_state_out 输出 -> {converted_path.name}")
    else:
        converted_path = onnx_path

    conv_streaming = [node.output[0] for node in model.graph.node if node.op_type == "Conv"]
    print(f"    检测到 {len(conv_streaming)} 个 Conv 节点需要流式改写")

    initial_values = []
    for bucket in dynamic_shapes:
        # 最后一个输入是 conv_state，它的初始值必须是全 0（没有任何历史）
        initial_values.append([None] * (len(bucket) - 1) + [np.zeros(bucket[-1], dtype=np.float32)])

    return converted_path, dynamic_shapes, initial_values, conv_streaming


def build_llm_config(context_length: int, keep_one_logit: bool = True):
    from rknn.api import DEFAULT_RKNN_LLM_CONFIG

    # 必须深拷贝：DEFAULT_RKNN_LLM_CONFIG 里含嵌套 list/dict，
    # 浅拷贝会让本次修改污染后续调用（同一进程内多次转换时会踩坑）。
    import copy

    llm_config = copy.deepcopy(DEFAULT_RKNN_LLM_CONFIG)
    attention_config = llm_config["attention_config"][0]
    attention_config["kvcache_buffer_len"] = context_length
    attention_config["max_position_embeddings"] = context_length
    attention_config["mrope_type"] = "Qwen3.5"
    attention_config["mrope_section"] = [11, 11, 10]
    attention_config["mrope_new_id_name"] = "mrope_id_input"
    attention_config["attention_type"] = "FullAttention"

    if not keep_one_logit:
        llm_config["keep_one_logit"] = []
    return llm_config


def copy_report(report_source: Path, output_path: Path) -> Path | None:
    if not report_source.is_file():
        print(f"    [!] 未找到量化报告 {report_source}，跳过")
        return None
    report_target = output_path.parent / f"{output_path.stem}_model_report.html"
    shutil.copy2(report_source, report_target)
    return report_target


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    onnx_path = Path(args.onnx_path).expanduser().resolve()
    config_path = onnx_path.with_suffix(".config.pkl")
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else onnx_path.with_suffix(".rknn")
    )

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX 不存在: {onnx_path}")
    if not config_path.is_file():
        raise FileNotFoundError(
            f"配套配置不存在: {config_path}\n"
            f"请先运行 python/llm/export_onnx.py 生成 .config.pkl"
        )
    if output_path.suffix.lower() != ".rknn":
        raise ValueError("--output_path 必须以 .rknn 结尾")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_source = (
        Path(args.report_source).expanduser().resolve()
        if args.report_source
        else Path.cwd() / "tmp" / "model_report.html"
    )
    report_source.unlink(missing_ok=True)

    print("=" * 78)
    print("RKNN3 转换：value head LLM")
    print("=" * 78)
    print(f"  onnx     : {onnx_path}")
    print(f"  config   : {config_path}")
    print(f"  output   : {output_path}")
    print(f"  platform : {args.target_platform}  cores={args.core_num}")
    print(f"  quant    : {args.quantized_dtype} / {QUANTIZED_METHOD} / normal")
    print(f"  ctx_len  : {args.context_length}")
    print()
    print("  [!] rknn.build 会写出 ./tmp/ —— 多个 shape 的转换请**串行**执行，")
    print("      并发会互相覆盖中间产物。")
    print()

    print("[1/4] 预处理 ONNX")
    converted_path, dynamic_shapes, initial_values, conv_streaming = prepare_onnx(onnx_path)
    print(f"    dynamic_input = {dynamic_shapes}")

    print("[2/4] 构造 llm_config")
    llm_config = build_llm_config(args.context_length, keep_one_logit=not args.keep_all_logits)
    print(f"    mrope_type={llm_config['attention_config'][0]['mrope_type']} "
          f"section={llm_config['attention_config'][0]['mrope_section']}")
    print(f"    keep_one_logit={llm_config['keep_one_logit']}")

    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    try:
        print("[3/4] rknn.config + load_llm")
        ret = rknn.config(
            target_platform=args.target_platform,
            core_num=args.core_num,
            dynamic_input=dynamic_shapes,
            quantized_dtype=args.quantized_dtype,
            quantized_algorithm="normal",
            quantized_method=QUANTIZED_METHOD,
            llm_config=llm_config,
            input_initial_value=initial_values,
            cvt_conv_streaming=conv_streaming,
            linear_attn_out_project_dtype=None,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config 失败: {ret}")

        ret = rknn.load_llm(
            model=str(converted_path),
            config=str(config_path),
            seq=[1, CHUNK_LENGTH],
        )
        if ret != 0:
            raise RuntimeError(f"rknn.load_llm 失败: {ret}")

        print("[4/4] build（PTQ 量化）+ export_rknn")
        ret = rknn.build(do_quantization=True)
        if ret != 0:
            raise RuntimeError(f"rknn.build 失败: {ret}")

        ret = rknn.export_rknn(str(output_path), save_ctx=False)
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn 失败: {ret}")
    finally:
        rknn.release()

    weight_path = output_path.with_suffix(".weight")
    report_path = copy_report(report_source, output_path)

    print()
    print("=" * 78)
    print("完成")
    print("=" * 78)
    print(f"  .rknn   : {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")
    if weight_path.is_file():
        print(f"  .weight : {weight_path}  ({weight_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  [!] .weight 未生成 —— 检查上面的日志")
    if report_path:
        print(f"  量化报告 : {report_path}")

    if not weight_path.is_file():
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
