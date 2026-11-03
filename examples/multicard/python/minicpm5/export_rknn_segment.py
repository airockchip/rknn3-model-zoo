import copy
import glob
import os
import re
import shutil
import sys

import onnx
from rknn.api import DEFAULT_RKNN_LLM_CONFIG, RKNN


ONNX_MODEL = "../../model/llm_minicpm5/MiniCPM5-15B-A2.5B.onnx"
LLM_CONFIG = "../../model/llm_minicpm5/MiniCPM5-15B-A2.5B.config.pkl"
RKNN_MODEL = "./MiniCPM5-15B-A2.5B.rknn"

MAX_CONTEXT_LEN = 4 * 1024
DEFAULT_MAX_SEQ_LEN = 128
NUM_DYNAMIC_SEQ_INPUTS = 3


def _extract_seg_idx(filepath):
    match = re.search(r"seg(\d+)", filepath)
    if match is None:
        raise ValueError(f"Cannot parse segment index from path: {filepath}")
    return int(match.group(1))


def _build_llm_config(
    is_multi_segment=False,
    is_last_segment=True,
    max_context_len=MAX_CONTEXT_LEN,
):
    """Build the full-attention RKNN LLM configuration for MiniCPM5."""
    llm_config = copy.deepcopy(DEFAULT_RKNN_LLM_CONFIG)
    attention_config = llm_config["attention_config"][0]
    attention_config["kvcache_buffer_len"] = max_context_len
    attention_config["max_position_embeddings"] = max_context_len
    attention_config["kvcache_dtype"] = "Int4_to_F16"
    attention_config['position_embeddings_host_storage'] = True
    llm_config["vocab"] = []

    if is_multi_segment and not is_last_segment:
        # Intermediate stages only return hidden states.  There is no lm_head
        # or vocabulary projection in those ONNX graphs.
        llm_config["llm_head"] = []

    return llm_config


def _get_dynamic_input_shapes(onnx_path):
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    static_shapes = []
    for graph_input in onnx_model.graph.input:
        shape = []
        for dim in graph_input.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value else 1)
        static_shapes.append(shape)

    min_shapes = []
    max_shapes = []
    for input_index, shape in enumerate(static_shapes):
        if input_index < NUM_DYNAMIC_SEQ_INPUTS:
            if len(shape) < 2:
                raise ValueError(
                    f"Expected a sequence input at index {input_index}, got shape {shape}"
                )
            min_shapes.append([1] + shape[1:])
            max_shapes.append([1, DEFAULT_MAX_SEQ_LEN] + shape[2:])
        else:
            min_shapes.append(list(shape))
            max_shapes.append(list(shape))

    dynamic_shapes = [min_shapes, max_shapes]
    print(f"onnx model dynamic input shapes: {dynamic_shapes}")
    return dynamic_shapes


def _export_single_segment(
    rknn,
    onnx_path,
    rknn_path,
    llm_config,
    config_path,
    args,
):
    dynamic_input = _get_dynamic_input_shapes(onnx_path)

    print("--> config model")
    rknn.config(
        target_platform=args.platform,
        dynamic_input=dynamic_input,
        quantized_dtype="w4a16",
        quantized_algorithm="normal",
        quantized_method="group32",
        llm_config=llm_config,
        profile_mode=getattr(args, "profile_mode", True),
    )
    print("done")

    if args.rebuild:
        print("--> Rebuild rknn model")
        ret = rknn.rebuild("./tmp")
        if ret != 0:
            print("Rebuild rknn model failed!")
            sys.exit(ret)
        print("done")
    else:
        print("--> Loading model")
        ret = rknn.load_llm(
            model=onnx_path,
            config=config_path,
            seq=[1, DEFAULT_MAX_SEQ_LEN],
        )
        if ret != 0:
            print("Load model failed!")
            sys.exit(ret)
        print("done")

        print("--> Building model")
        ret = rknn.build(do_quantization=True)
        if ret != 0:
            print("Build model failed!")
            sys.exit(ret)
        print("done")

    export_dirname = os.path.dirname(rknn_path)
    if export_dirname:
        os.makedirs(export_dirname, exist_ok=True)

    print("--> Export RKNN model")
    ret = rknn.export_rknn(rknn_path, save_ctx=getattr(args, "save_ctx", True))
    if ret != 0:
        print("Export rknn model failed!")
        sys.exit(ret)
    print("done")


def _find_segment_count(onnx_path):
    base_dir = os.path.dirname(onnx_path)
    base_name = os.path.splitext(os.path.basename(onnx_path))[0]
    pattern = os.path.join(base_dir, "seg*", f"{base_name}_seg*.onnx")
    segment_files = glob.glob(pattern)
    if not segment_files:
        raise FileNotFoundError(f"No segment files found matching {pattern}")
    segment_files.sort(key=_extract_seg_idx)
    return len(segment_files)


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser(
        description="Export MiniCPM5 ONNX segments to RKNN for multicard deployment"
    )
    parser.add_argument("--onnx_path", type=str, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL)
    parser.add_argument("--platform", type=str, default="rk1820")
    parser.add_argument(
        "--max_context_len",
        "--max-context-len",
        type=int,
        default=MAX_CONTEXT_LEN,
        help="maximum KV cache/context length (default: %(default)s)",
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument(
        "--profile_mode",
        action=BooleanOptionalAction,
        default=False,
        help="enable per-layer dump support for runtime accuracy analysis (default: disabled)",
    )
    parser.add_argument(
        "--save_ctx",
        action=BooleanOptionalAction,
        default=False,
        help="save the RKNN context required by load_rknn(load_ctx=True) (default: disabled)",
    )
    parser.add_argument("--multi_segment", action="store_true")
    parser.add_argument("--num_segments", type=int, default=0)
    parser.add_argument(
        "--stop_after_segment",
        type=int,
        default=None,
        help=(
            "stop after this zero-based segment; useful for block-level validation"
        ),
    )
    parser.add_argument("--rebuild", action="store_true", default=False)
    args = parser.parse_args()

    if args.multi_segment:
        segment_count = args.num_segments
        base_dir = os.path.dirname(args.onnx_path)
        base_name = os.path.splitext(os.path.basename(args.onnx_path))[0]
        rknn_output_dir = os.path.dirname(args.rknn_path) or "."

        if segment_count <= 0:
            try:
                segment_count = _find_segment_count(args.onnx_path)
            except FileNotFoundError as exc:
                print(exc)
                sys.exit(1)
            print(f"Found {segment_count} segments")

        if args.stop_after_segment is not None and not 0 <= args.stop_after_segment < segment_count:
            raise ValueError(
                "--stop_after_segment must be within the segment range: "
                f"0 <= {args.stop_after_segment} < {segment_count}"
            )
        segment_limit = (
            segment_count if args.stop_after_segment is None else args.stop_after_segment + 1
        )
        for seg_idx in range(segment_limit):
            seg_onnx = os.path.join(
                base_dir,
                f"seg{seg_idx}",
                f"{base_name}_seg{seg_idx}.onnx",
            )
            seg_rknn = os.path.join(
                rknn_output_dir,
                f"{base_name}_seg{seg_idx}.rknn",
            )
            if not os.path.isfile(seg_onnx) and not args.rebuild:
                print(f"Missing segment ONNX model: {seg_onnx}")
                sys.exit(1)

            is_last = seg_idx == segment_count - 1
            tmp_dir = os.path.join(os.getcwd(), "tmp")
            tmp_seg_dir = os.path.join(os.getcwd(), f"tmp_seg{seg_idx}")
            if args.rebuild and os.path.isdir(tmp_seg_dir):
                if os.path.isdir(tmp_dir):
                    shutil.rmtree(tmp_dir)
                shutil.move(tmp_seg_dir, tmp_dir)
                print(f"--> Restored tmp_seg{seg_idx} -> tmp for rebuild")

            rknn = RKNN(verbose=False)
            try:
                print(
                    f"\n--> Exporting segment {seg_idx + 1}/{segment_count} "
                    f"(last={is_last}): {seg_onnx}"
                )
                llm_config = _build_llm_config(
                    is_multi_segment=True,
                    is_last_segment=is_last,
                    max_context_len=args.max_context_len,
                )
                _export_single_segment(
                    rknn,
                    seg_onnx,
                    seg_rknn,
                    llm_config,
                    args.config,
                    args,
                )
            finally:
                rknn.release()

            if os.path.isdir(tmp_dir):
                if os.path.isdir(tmp_seg_dir):
                    shutil.rmtree(tmp_seg_dir)
                shutil.move(tmp_dir, tmp_seg_dir)
                print(f"--> Renamed tmp -> tmp_seg{seg_idx}")
            if args.stop_after_segment is not None and seg_idx >= args.stop_after_segment:
                print(
                    f"Stopping after segment {seg_idx} as requested by "
                    "--stop_after_segment"
                )
    else:
        rknn = RKNN(verbose=False)
        try:
            llm_config = _build_llm_config(
                max_context_len=args.max_context_len,
            )
            _export_single_segment(
                rknn,
                args.onnx_path,
                args.rknn_path,
                llm_config,
                args.config,
                args,
            )
        finally:
            rknn.release()
