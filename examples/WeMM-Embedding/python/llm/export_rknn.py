import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import onnx
from rknn.api import DEFAULT_RKNN_LLM_CONFIG, RKNN


CHUNK_LENGTH = 128
QUANTIZED_METHOD = "group32"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert the WeMM-Embedding-2B LLM ONNX to RKNN3")
    parser.add_argument("--onnx_path", required=True, help="Input LLM ONNX file")
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
    parser.add_argument(
        "--context_length",
        type=int,
        default=4096,
        help="Maximum LLM context length, at least 128 (default: 4096)",
    )
    args = parser.parse_args()
    if not 1 <= args.core_num <= 8:
        parser.error("--core_num must be in the range [1, 8]")
    if args.context_length < CHUNK_LENGTH:
        parser.error(f"--context_length must be at least {CHUNK_LENGTH}")
    return args


def shape_of(value_info):
    return [dim.dim_value if dim.dim_value else 1 for dim in value_info.type.tensor_type.shape.dim]


def prepare_onnx(onnx_path):
    model = onnx.load(str(onnx_path), load_external_data=False)
    base_shapes = [shape_of(value) for value in model.graph.input]
    dynamic_shapes = []
    for sequence_length in (CHUNK_LENGTH,):
        shapes = [list(shape) for shape in base_shapes]
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
    else:
        converted_path = onnx_path

    conv_streaming = [node.output[0] for node in model.graph.node if node.op_type == "Conv"]
    initial_values = []
    for bucket in dynamic_shapes:
        initial_values.append(
            [None] * (len(bucket) - 1) + [np.zeros(bucket[-1], dtype=np.float32)]
        )
    return converted_path, dynamic_shapes, initial_values, conv_streaming


def copy_report(report_source, output_path):
    if not report_source.is_file():
        raise FileNotFoundError(f"RKNN3 conversion report not found: {report_source}")
    report_target = output_path.parent / "model_report.html"
    shutil.copy2(report_source, report_target)
    return report_target


def main():
    args = parse_args()
    onnx_path = Path(args.onnx_path).expanduser().resolve()
    config_path = onnx_path.with_suffix(".config.pkl")
    output_path = Path(args.output_path).expanduser().resolve()
    if not onnx_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"Missing ONNX file or matching configuration file: {onnx_path}, {config_path}"
        )
    if output_path.suffix.lower() != ".rknn":
        raise ValueError("--output_path must end with .rknn")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_source = Path.cwd() / "tmp" / "model_report.html"
    report_source.unlink(missing_ok=True)

    converted_path, dynamic_shapes, initial_values, conv_streaming = prepare_onnx(onnx_path)
    llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    attention_config = llm_config["attention_config"][0]
    attention_config["kvcache_buffer_len"] = args.context_length
    attention_config["max_position_embeddings"] = args.context_length
    attention_config["mrope_type"] = "Qwen3.5"
    attention_config["mrope_section"] = [11, 11, 10]
    attention_config["mrope_new_id_name"] = "mrope_id_input"

    rknn = RKNN(verbose=True)
    try:
        rknn.config(
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
        ret = rknn.load_llm(
            model=str(converted_path),
            config=str(config_path),
            seq=[1, CHUNK_LENGTH],
        )
        if ret != 0:
            raise RuntimeError(f"Failed to load the LLM ONNX model: {ret}")
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
    print(f"RKNN weights: {weight_path}")
    print(f"Model report: {report_path}")


if __name__ == "__main__":
    main()
