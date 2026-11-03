import argparse
import shutil
from pathlib import Path

import onnx
from rknn.api import RKNN


QUANTIZED_METHOD = "group32"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert the WeMM-Embedding-2B vision ONNX to RKNN3")
    parser.add_argument("--onnx_path", required=True, help="Input vision ONNX file")
    parser.add_argument("--output_path", required=True, help="Output RKNN file")
    parser.add_argument(
        "--target_platform",
        choices=("rk1820", "rk1828"),
        default="rk1820",
        help="RKNN target platform (default: rk1820)",
    )
    parser.add_argument(
        "--core_num",
        type=int,
        default=8,
        help="NPU core number in the range 1-8 (default: 8)",
    )
    parser.add_argument(
        "--quantized_dtype",
        choices=("w4a16", "w6a16", "w8a16"),
        default="w4a16",
        help="RKNN weight quantization dtype (default: w4a16)",
    )
    args = parser.parse_args()
    if not 1 <= args.core_num <= 8:
        parser.error("--core_num must be in the range [1, 8]")
    return args


def load_vision_input(onnx_path):
    model = onnx.load(str(onnx_path), load_external_data=False)
    pixel_inputs = [value for value in model.graph.input if value.name == "pixel"]
    if len(pixel_inputs) != 1:
        raise ValueError("Vision ONNX must contain exactly one input named pixel")

    shape = []
    for dimension in pixel_inputs[0].type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError("Vision ONNX input pixel must have a static shape")
        shape.append(dimension.dim_value)
    if len(shape) != 4 or shape[1] != 3:
        raise ValueError(f"Vision ONNX input pixel must be [frames, 3, height, width], got {shape}")
    input_frames = shape[0]
    if input_frames not in (1, 2):
        raise ValueError(f"Vision ONNX input frames must be 1 or 2, got {input_frames}")
    mode = "image" if input_frames == 1 else "video"
    return shape, mode


def copy_report(report_source, output_path):
    if not report_source.is_file():
        raise FileNotFoundError(f"RKNN3 conversion report not found: {report_source}")
    report_target = output_path.with_name(f"{output_path.stem}_report.html")
    shutil.copy2(report_source, report_target)
    return report_target


def main():
    args = parse_args()
    onnx_path = Path(args.onnx_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Vision ONNX file not found: {onnx_path}")
    if output_path.suffix.lower() != ".rknn":
        raise ValueError("--output_path must end with .rknn")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_shape, mode = load_vision_input(onnx_path)
    report_source = Path.cwd() / "tmp" / "model_report.html"
    report_source.unlink(missing_ok=True)

    rknn = RKNN(verbose=True)
    try:
        rknn.config(
            target_platform=args.target_platform,
            core_num=args.core_num,
            quantized_dtype=args.quantized_dtype,
            quantized_algorithm="normal",
            quantized_method=QUANTIZED_METHOD,
            mean_values=[[127.5, 127.5, 127.5]],
            std_values=[[127.5, 127.5, 127.5]],
            input_attrs={"pixel": {"dtype": "uint8", "layout": "NHWC"}},
        )
        ret = rknn.load_onnx(
            model=str(onnx_path),
            inputs=["pixel"],
            input_size_list=[input_shape],
        )
        if ret != 0:
            raise RuntimeError(f"Failed to load the vision ONNX model: {ret}")
        ret = rknn.build(do_quantization=True)
        if ret != 0:
            raise RuntimeError(f"Failed to build the RKNN model: {ret}")
        ret = rknn.export_rknn(str(output_path), save_ctx=False)
        if ret != 0:
            raise RuntimeError(f"Failed to export the RKNN model: {ret}")
        report_path = copy_report(report_source, output_path)
    finally:
        rknn.release()

    weight_path = output_path.with_suffix(".weight")
    if not weight_path.is_file():
        raise FileNotFoundError(f"RKNN weight file not found: {weight_path}")
    print(f"RKNN export completed: {output_path}")
    print(f"Vision mode: {mode}")
    print(f"RKNN quantization: {args.quantized_dtype}, normal, {QUANTIZED_METHOD}")
    print(f"RKNN weights: {weight_path}")
    print(f"Model report: {report_path}")


if __name__ == "__main__":
    main()
