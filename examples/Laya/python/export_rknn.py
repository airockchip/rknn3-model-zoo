"""Convert the Laya main graph to a multi-bucket RKNN model."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from rknn.api import RKNN

from laya_config import (
    INPUT_NAMES, input_shapes, load_config, runtime_config, validate_onnx,
)

RESOURCE_FILES = ("embedding.bin", "act.bin", "tokenizer.json",
                  "tokenizer_config.json", "laya.json")


def check(code, step):
    if code != 0:
        raise RuntimeError(f"{step} failed: {code}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Laya for RK1820 with W8A16 and 8 cores")
    parser.add_argument(
        "--onnx_path", required=True, help="Input ONNX file path")
    parser.add_argument(
        "--output_path", required=True,
        help="Output .rknn file path; CPU resources are copied alongside it")
    return parser.parse_args()


def convert_model(source, output):
    """Build all sequence buckets and package the CPU resources."""
    # Deployment metadata is smaller than the ONNX-side configuration.
    if source.parent.resolve() == output.parent.resolve():
        raise ValueError(
            "ONNX source and RKNN output must use different directories")
    resources = [source.parent / name for name in RESOURCE_FILES]
    for resource in resources:
        if not resource.is_file() or resource.stat().st_size == 0:
            raise FileNotFoundError(f"Missing model resource: {resource}")
    config = load_config(source.parent / "laya.json")
    validate_onnx(source, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = Path("tmp/model_report.html")
    if report.exists():
        report.unlink()
    rknn = RKNN(verbose=True)
    try:
        names = INPUT_NAMES
        check(rknn.config(
            target_platform="rk1820", core_num=8, profile_mode=False,
            quantized_dtype="w8a16", quantized_method="channel",
            quantized_algorithm="normal",
            dynamic_input=[
                input_shapes(config, length)
                for length in config["sequence_buckets"]
            ],
            input_attrs={
                name: {
                    "dtype": "float16" if name == "inputs_embeds" else "int32",
                    "layout": "NCHW",
                } for name in names}), "config")
        check(rknn.load_onnx(model=str(source)), "load")
        # W8A16 normal quantizes weights without activation calibration.
        check(rknn.build(do_quantization=True), "build")
        if not report.is_file() or report.stat().st_size == 0:
            raise RuntimeError("Missing current model report")
        with tempfile.TemporaryDirectory(
                prefix="laya-rknn-", dir=output.parent) as temp:
            model = Path(temp) / output.name
            check(rknn.export_rknn(str(model), save_ctx=False), "export")
            weight = model.with_suffix(".weight")
            if not model.is_file() or not weight.is_file():
                raise RuntimeError("Incomplete RKNN/weight pair")
            # Stage all resources before publishing the deployment files.
            for resource in resources:
                if resource.name == "laya.json":
                    # Bucket/dimension metadata is only needed by the compiler.
                    (Path(temp) / resource.name).write_text(
                        json.dumps(runtime_config(config), indent=2),
                        encoding="utf-8")
                else:
                    shutil.copy2(resource, Path(temp) / resource.name)
            model.replace(output)
            weight.replace(output.with_suffix(".weight"))
            for name in RESOURCE_FILES:
                (Path(temp) / name).replace(output.parent / name)
    finally:
        rknn.release()


def main():
    args = parse_args()
    source = Path(args.onnx_path).expanduser().resolve()
    output = Path(args.output_path).expanduser().resolve()
    if not source.is_file() or output.suffix != ".rknn":
        raise ValueError(
            "Expected an existing ONNX file and an output path "
            "ending in .rknn")
    convert_model(source, output)
    print(f"Conversion completed: {output}")


if __name__ == "__main__":
    main()
