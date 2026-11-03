#!/usr/bin/env python3
# Copyright (c) 2026 Rockchip Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Convert Paraformer ONNX submodels and assemble RKNN3 runtime resources."""

import argparse
import shutil
import tempfile
from pathlib import Path

from rknn.api import RKNN


ENCODER_BUCKETS = [
    [[1, 1, 560], [1, 1, 1]],
    [[1, 128, 560], [1, 128, 1]],
    [[1, 256, 560], [1, 256, 1]],
    [[1, 334, 560], [1, 334, 1]],
]
DECODER_BUCKETS = [
    [[1, 1, 512], [1, 1, 512], [1, 1, 1], [1, 1, 1]],
    [[1, 128, 512], [1, 21, 512], [1, 21, 1], [1, 128, 1]],
    [[1, 256, 512], [1, 43, 512], [1, 43, 1], [1, 256, 1]],
    [[1, 334, 512], [1, 64, 512], [1, 64, 1], [1, 334, 1]],
    [[1, 334, 512], [1, 128, 512], [1, 128, 1], [1, 334, 1]],
]
SEACO_DECODER_BUCKETS = [
    [[1, 1, 512], [1, 1, 512], [1, 1, 512], [1, 1, 1], [1, 1]],
    [[1, 8, 512], [1, 16, 512], [1, 16, 512], [1, 16, 1], [1, 8]],
    [[1, 32, 512], [1, 32, 512], [1, 32, 512], [1, 32, 1], [1, 32]],
    [[1, 51, 512], [1, 64, 512], [1, 64, 512], [1, 64, 1], [1, 51]],
    [[1, 128, 512], [1, 128, 512], [1, 128, 512], [1, 128, 1], [1, 128]],
]
ASF_SCORER_BUCKETS = [
    [[1, 128, 512], [1, 128, 512], [1, 128, 1], [1, 128]],
]

MODEL_BUCKETS = {
    "encoder": ENCODER_BUCKETS,
    "decoder": DECODER_BUCKETS,
    "decoder_hotword": DECODER_BUCKETS,
    "seaco_decoder": SEACO_DECODER_BUCKETS,
    "asf_scorer": ASF_SCORER_BUCKETS,
}

MODEL_FILE_STEMS = {
    "encoder": "paraformer_encoder",
    "decoder": "paraformer_decoder",
    "decoder_hotword": "paraformer_decoder_hotword",
    "seaco_decoder": "seaco_decoder",
    "asf_scorer": "asf_scorer",
}

MODEL_INPUT_ATTRS = {
    "encoder": {
        "speech": {"dtype": "float16", "layout": "UNDEFINED"},
        "enc_mask": {"dtype": "float16", "layout": "UNDEFINED"},
    },
    "decoder": {
        "enc": {"dtype": "float16", "layout": "UNDEFINED"},
        "pre_acoustic_embeds": {
            "dtype": "float16",
            "layout": "UNDEFINED",
        },
        "token_mask": {"dtype": "float16", "layout": "UNDEFINED"},
        "enc_mask": {"dtype": "float16", "layout": "UNDEFINED"},
    },
    "decoder_hotword": {
        "enc": {"dtype": "float16", "layout": "UNDEFINED"},
        "pre_acoustic_embeds": {
            "dtype": "float16",
            "layout": "UNDEFINED",
        },
        "token_mask": {"dtype": "float16", "layout": "UNDEFINED"},
        "enc_mask": {"dtype": "float16", "layout": "UNDEFINED"},
    },
    "seaco_decoder": {
        "contextual_info": {"dtype": "float16", "layout": "UNDEFINED"},
        "pre_acoustic_embeds": {
            "dtype": "float16",
            "layout": "UNDEFINED",
        },
        "decoder_hidden": {"dtype": "float16", "layout": "UNDEFINED"},
        "token_mask": {"dtype": "float16", "layout": "UNDEFINED"},
        "hotword_mask": {"dtype": "float16", "layout": "UNDEFINED"},
    },
    "asf_scorer": {
        "contextual_info": {"dtype": "float16", "layout": "UNDEFINED"},
        "decoder_hidden": {"dtype": "float16", "layout": "UNDEFINED"},
        "token_mask": {"dtype": "float16", "layout": "UNDEFINED"},
        "hotword_mask": {"dtype": "float16", "layout": "UNDEFINED"},
    },
}

BIAS_ENCODER_FILE = "bias_encoder.bin"

STAGE_GROUPS = {
    "encoder": ("encoder",),
    "decoder": ("decoder",),
    "decoder_hotword": ("decoder_hotword",),
    "seaco_decoder": ("seaco_decoder",),
    "asf_scorer": ("asf_scorer",),
    "hotword": ("decoder_hotword", "seaco_decoder", "asf_scorer"),
    "all": (
        "encoder",
        "decoder",
        "decoder_hotword",
        "seaco_decoder",
        "asf_scorer",
    ),
}


def _check(ret, stage):
    if ret != 0:
        raise RuntimeError(f"RKNN {stage} failed with code {ret}")


def _convert(stage, onnx_path, output_path, platform):
    buckets = MODEL_BUCKETS[stage]
    input_attrs = MODEL_INPUT_ATTRS[stage]
    model_stem = MODEL_FILE_STEMS[stage]

    model_path = onnx_path / f"{model_stem}.onnx"
    rknn_path = output_path / f"{model_stem}.rknn"
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    report_path = Path.cwd() / "tmp" / "model_report.html"
    if report_path.exists():
        report_path.unlink()

    print(f"Converting {model_path}")
    print(f"  target={platform}, precision=fp16, core_num=1")
    print(f"  dynamic_input={buckets}")
    rknn = RKNN(verbose=False)
    try:
        _check(
            rknn.config(
                target_platform=platform,
                core_num=1,
                input_attrs=input_attrs,
                dynamic_input=buckets,
            ),
            "config",
        )
        _check(
            rknn.load_onnx(
                model=str(model_path),
                input_size_list=buckets[0],
            ),
            "load_onnx",
        )
        _check(rknn.build(do_quantization=False), "build")
        if not report_path.is_file():
            raise RuntimeError(f"Current model report not found: {report_path}")
        _check(rknn.export_rknn(str(rknn_path)), "export_rknn")
        validation_report_path = (
            Path.cwd()
            / "tmp"
            / "paraformer_reports"
            / f"{model_stem}_model_report.html"
        )
        validation_report_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, validation_report_path)
    finally:
        rknn.release()

    weight_path = rknn_path.with_suffix(".weight")
    if not rknn_path.is_file() or not weight_path.is_file():
        raise RuntimeError(f"Incomplete RKNN output for {stage}")
    print(f"Saved {rknn_path}")
    print(f"Saved {weight_path}")
    print(f"Saved validation report: {validation_report_path}")


def _copy_file(source, destination, required):
    if not source.is_file():
        if required:
            raise FileNotFoundError(f"Required runtime resource not found: {source}")
        return
    if source.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"Copied runtime resource: {destination}")


def _copy_runtime_resources(onnx_path, output_path, stage):
    hotword_stage = stage in ("hotword", "all")
    _copy_file(
        onnx_path / "tokens.json",
        output_path / "tokens.json",
        required=hotword_stage,
    )
    _copy_file(
        onnx_path / "asr_example.wav",
        output_path / "asr_example.wav",
        required=stage == "all",
    )
    if hotword_stage:
        _copy_file(
            onnx_path / "asr_example_hotword.wav",
            output_path / "asr_example_hotword.wav",
            required=True,
        )
        _copy_file(
            onnx_path / "hotwords.txt",
            output_path / "hotwords.txt",
            required=True,
        )
        _copy_file(
            onnx_path / "seg_dict",
            output_path / "seg_dict",
            required=True,
        )
        _copy_file(
            onnx_path / BIAS_ENCODER_FILE,
            output_path / BIAS_ENCODER_FILE,
            required=True,
        )


def _validate_inputs(onnx_path, stages, stage_group):
    """Validate every selected model and required resource before conversion."""
    required_paths = [
        onnx_path / f"{MODEL_FILE_STEMS[stage]}.onnx" for stage in stages
    ]
    if stage_group in ("hotword", "all"):
        required_paths.extend(
            [
                onnx_path / "tokens.json",
                onnx_path / "asr_example_hotword.wav",
                onnx_path / "hotwords.txt",
                onnx_path / "seg_dict",
                onnx_path / BIAS_ENCODER_FILE,
            ]
        )
    if stage_group == "all":
        required_paths.append(onnx_path / "asr_example.wav")

    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"RKNN export inputs are incomplete:\n{formatted}"
        )


def _remove_path(path):
    """Remove a file or directory created by a failed publish attempt."""
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _publish_outputs(staging_path, output_path, stages):
    """Publish a complete staged result and restore old files on failure."""
    output_path.mkdir(parents=True, exist_ok=True)
    staged_items = sorted(staging_path.iterdir(), key=lambda path: path.name)
    obsolete_reports = [
        output_path / f"{model_stem}_model_report.html"
        for model_stem in MODEL_FILE_STEMS.values()
    ]
    backup_path = Path(
        tempfile.mkdtemp(
            prefix=f".{output_path.name}.backup-",
            dir=output_path.parent,
        )
    )
    published = []
    backed_up = []
    try:
        for staged_item in staged_items:
            destination = output_path / staged_item.name
            if destination.exists():
                destination.replace(backup_path / destination.name)
                backed_up.append(destination.name)
        for report in obsolete_reports:
            if report.exists() and report.name not in backed_up:
                report.replace(backup_path / report.name)
                backed_up.append(report.name)

        for staged_item in staged_items:
            destination = output_path / staged_item.name
            staged_item.replace(destination)
            published.append(destination)
    except Exception:
        for destination in reversed(published):
            _remove_path(destination)
        for name in reversed(backed_up):
            (backup_path / name).replace(output_path / name)
        raise
    finally:
        shutil.rmtree(backup_path, ignore_errors=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Paraformer FP16 single-core RKNN models"
    )
    parser.add_argument(
        "--onnx_path",
        required=True,
        help="directory containing the ONNX files selected by --stage",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="directory used for RKNN and weight outputs",
    )
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_GROUPS),
        default="all",
        help="submodel or submodel group to convert",
    )
    parser.add_argument(
        "--platform",
        default="rk1820",
        help="RKNN target platform (default: rk1820)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    if not onnx_path.is_dir():
        raise FileNotFoundError(f"ONNX directory not found: {onnx_path}")
    stages = STAGE_GROUPS[args.stage]
    _validate_inputs(onnx_path, stages, args.stage)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.name}.staging-",
        dir=output_path.parent,
    ) as temporary_directory:
        staging_path = Path(temporary_directory)
        for stage in stages:
            _convert(stage, onnx_path, staging_path, args.platform)
        _copy_runtime_resources(onnx_path, staging_path, args.stage)
        _publish_outputs(staging_path, output_path, stages)
    print("RKNN export completed")


if __name__ == "__main__":
    main()
