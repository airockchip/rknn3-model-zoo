"""Convert multi-bucket AuT FP16 and decoder W4A16 for RK1828 / 8 cores."""

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from qwen3_forcealign.resources import TEXT_RESOURCE_FILES

AUDIO_BUCKETS = (4, 8, 16, 20, 30)
SEQUENCE_BUCKETS = (256, 512, 1024, 2048)


def require_ok(status, stage):
    if status != 0:
        raise RuntimeError(f"{stage} failed: {status}")


def build_config(stage):
    """Return the validated tensor contract and compilation settings."""
    if stage not in ("encoder", "decoder"):
        raise ValueError(f"Unknown model stage: {stage}")
    config = dict(target_platform="rk1828", core_num=8)
    if stage == "encoder":
        config.update(
            quantized_dtype="w16a16",
            input_attrs={
                "x": {"dtype": "float32", "layout": "NCHW"},
                "audio_mask": {"dtype": "float32", "layout": "NCHW"},
            },
            dynamic_input=[
                [[n, 1, 128, 100], [1, 1, 1, n * 13]] for n in AUDIO_BUCKETS
            ],
        )
    else:
        config.update(
            quantized_dtype="w4a16",
            quantized_algorithm="normal",
            quantized_method="group32",
            dynamic_input=[
                [[1, n, 1024], [1, n], [1, 1, n]] for n in SEQUENCE_BUCKETS
            ],
            input_attrs={
                "input_embeds": {"dtype": "float16", "layout": "NCHW"},
                "attention_mask": {"dtype": "float16", "layout": "NCHW"},
                "position_ids": {"dtype": "int64", "layout": "NCHW"},
            },
        )
    return config


def convert(stage, source, destination):
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)
    try:
        config = build_config(stage)
        require_ok(rknn.config(**config), "config")
        require_ok(rknn.load_onnx(model=str(source)), "load_onnx")
        require_ok(rknn.build(do_quantization=stage == "decoder"), "build")
        require_ok(
            rknn.export_rknn(str(destination), save_ctx=False), "export"
        )
        for path in [destination, destination.with_suffix(".weight")]:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing conversion artifact: {path}")
        if not Path("tmp/model_report.html").is_file():
            raise RuntimeError("Compilation report was not generated")
        shutil.copyfile(
            "tmp/model_report.html",
            destination.with_name(f"{stage}_report.html"),
        )
    finally:
        rknn.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx_path",
        type=Path,
        required=True,
        help="Directory containing encoder.onnx and decoder.onnx",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="RKNN/weight output directory; no overwriting",
    )
    parser.add_argument(
        "--stage", choices=["encoder", "decoder", "all"], default="all"
    )
    args = parser.parse_args()
    source, out = args.onnx_path.resolve(), args.output_path.resolve()
    stages = ["encoder", "decoder"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if not (source / f"{stage}.onnx").is_file():
            parser.error(f"Missing {stage}.onnx")
        if (out / f"{stage}_report.html").exists():
            parser.error(f"Report for {stage} already exists")
        if any(
            (out / f"{stage}.{suffix}").exists()
            for suffix in ["rknn", "weight"]
        ):
            parser.error(f"Output for {stage} already exists")
    resources = TEXT_RESOURCE_FILES if "decoder" in stages else ()
    for name in resources:
        path = source / name
        if not path.is_file() or path.stat().st_size == 0:
            parser.error(f"Missing text resource: {path}; export ONNX first")
        if (out / name).exists():
            parser.error(f"Resource already exists: {out / name}")
    out.mkdir(parents=True, exist_ok=True)
    original = Path.cwd()
    for stage in stages:
        # A new working directory prevents stale reports from passing checks.
        with tempfile.TemporaryDirectory(prefix=f"{stage}_", dir=out) as work:
            try:
                os.chdir(work)
                model = Path(work) / f"{stage}.rknn"
                convert(stage, source / f"{stage}.onnx", model)
                if stage == "decoder":
                    for name in resources:
                        shutil.copyfile(source / name, Path(work) / name)
                model.rename(out / model.name)
                weight = model.with_suffix(".weight")
                weight.rename(out / weight.name)
                report = model.with_name(f"{stage}_report.html")
                report.rename(out / report.name)
                if stage == "decoder":
                    for name in resources:
                        (Path(work) / name).rename(out / name)
            finally:
                os.chdir(original)


if __name__ == "__main__":
    main()
