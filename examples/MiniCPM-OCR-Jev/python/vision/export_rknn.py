"""把视觉塔 ONNX 转成 RKNN3 的 ``.rknn`` + ``.weight``

用法::

    python export_rknn.py --onnx_path ../../model/vision/320x832/vision.onnx
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import onnx

QUANTIZED_METHOD = "group32"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert the value-head vision ONNX to RKNN3 (.rknn + .weight)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx_path", required=True, help="输入视觉 ONNX")
    p.add_argument("--output_path", default=None, help="输出 .rknn；默认与 ONNX 同名")
    p.add_argument(
        "--target_platform",
        choices=("rk1820", "rk1828"),
        default="rk1820",
    )
    p.add_argument("--core_num", type=int, default=8, help="NPU 核数，1-8")
    p.add_argument(
        "--quantized_dtype",
        choices=("w4a16", "w6a16", "w8a16"),
        default="w4a16",
    )
    p.add_argument(
        "--fp_input",
        action="store_true",
        help=(
            "声明 pixel 输入为 float32 NCHW（归一化在 host 侧做）。"
            "默认走 uint8 NHWC + mean/std，与 WeMM-Embedding 一致"
        ),
    )
    p.add_argument("--report_source", default=None)
    p.add_argument(
        "--calib",
        default=None,
        help=(
            "量化校准用的 dataset.txt（每行 '<图片路径> [重复次数]'）。"
            "注意：w4a16/w6a16/w8a16 只量化权重、激活保持 16-bit，"
            "toolkit 会明确忽略该参数（产物与随机噪声校准逐字节相同）；"
            "保留本选项是为了将来切换到 w8a8/i8 这类量化激活的方案"
        ),
    )
    args = p.parse_args(argv)
    if not 1 <= args.core_num <= 8:
        p.error("--core_num 必须在 [1, 8] 区间")
    if args.calib is not None and not Path(args.calib).expanduser().is_file():
        p.error(f"--calib 文件不存在: {args.calib}")
    return args


def load_vision_input(onnx_path: Path):
    """读取并校验视觉 ONNX 的 ``pixel`` 输入，返回 (shape, mode)。"""
    model = onnx.load(str(onnx_path), load_external_data=False)
    pixel_inputs = [value for value in model.graph.input if value.name == "pixel"]
    if len(pixel_inputs) != 1:
        raise ValueError("视觉 ONNX 必须有且只有一个名为 pixel 的输入")

    shape: list[int] = []
    for dimension in pixel_inputs[0].type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError("视觉 ONNX 的 pixel 输入必须是静态 shape")
        shape.append(dimension.dim_value)

    if len(shape) != 4 or shape[1] != 3:
        raise ValueError(f"pixel 必须是 [frames, 3, H, W]，实际 {shape}")
    if shape[0] not in (1, 2):
        raise ValueError(f"pixel 的 frames 必须是 1 或 2，实际 {shape[0]}")

    if shape[2] % 32 or shape[3] % 32:
        raise ValueError(f"H/W 必须是 32 的倍数，实际 {shape[2]}x{shape[3]}")

    mode = "image" if shape[0] == 1 else "video"
    return shape, mode


def copy_report(report_source: Path, output_path: Path) -> Path | None:
    if not report_source.is_file():
        print(f"    [!] 未找到量化报告 {report_source}，跳过")
        return None
    target = output_path.with_name(f"{output_path.stem}_report.html")
    shutil.copy2(report_source, target)
    return target


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    onnx_path = Path(args.onnx_path).expanduser().resolve()
    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else onnx_path.with_suffix(".rknn")
    )
    if not onnx_path.is_file():
        raise FileNotFoundError(f"视觉 ONNX 不存在: {onnx_path}")
    if output_path.suffix.lower() != ".rknn":
        raise ValueError("--output_path 必须以 .rknn 结尾")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_shape, mode = load_vision_input(onnx_path)

    report_source = (
        Path(args.report_source).expanduser().resolve()
        if args.report_source
        else Path.cwd() / "tmp" / "model_report.html"
    )
    report_source.unlink(missing_ok=True)

    print("=" * 78)
    print("RKNN3 转换：value head 视觉塔")
    print("=" * 78)
    print(f"  onnx     : {onnx_path}")
    print(f"  output   : {output_path}")
    print(f"  pixel    : {input_shape}  ({mode})")
    print(f"  platform : {args.target_platform}  cores={args.core_num}")
    print(f"  quant    : {args.quantized_dtype} / {QUANTIZED_METHOD} / normal")
    print(f"  预处理   : {'host 侧 fp32' if args.fp_input else 'NPU 内 uint8->fp32 + mean/std 127.5'}")
    print(f"  校准集   : {args.calib if args.calib else '未提供 -> 随机噪声'}")
    print()
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    try:
        config_kwargs = dict(
            target_platform=args.target_platform,
            core_num=args.core_num,
            quantized_dtype=args.quantized_dtype,
            quantized_algorithm="normal",
            quantized_method=QUANTIZED_METHOD,
        )
        if args.fp_input:
            config_kwargs["input_attrs"] = {"pixel": {"dtype": "float32", "layout": "NCHW"}}
        else:
            # (x/255 - 0.5)/0.5  <=>  (x - 127.5)/127.5
            config_kwargs["mean_values"] = [[127.5, 127.5, 127.5]]
            config_kwargs["std_values"] = [[127.5, 127.5, 127.5]]
            config_kwargs["input_attrs"] = {"pixel": {"dtype": "uint8", "layout": "NHWC"}}

        print("[1/3] rknn.config")
        ret = rknn.config(**config_kwargs)
        if ret != 0:
            raise RuntimeError(f"rknn.config 失败: {ret}")

        print("[2/3] rknn.load_onnx")
        ret = rknn.load_onnx(
            model=str(onnx_path),
            inputs=["pixel"],
            input_size_list=[input_shape],
        )
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx 失败: {ret}")

        print("[3/3] build + export_rknn")
        ret = rknn.build(do_quantization=True, dataset=args.calib)
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
        print("  [!] .weight 未生成")
    if report_path:
        print(f"  报告    : {report_path}")

    return 0 if weight_path.is_file() else 1


if __name__ == "__main__":
    sys.exit(main())
