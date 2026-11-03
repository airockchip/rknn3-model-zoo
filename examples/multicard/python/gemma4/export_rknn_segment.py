import glob
import os
import numpy as np
from rknn.api import RKNN
import onnx
import shutil
import re
import sys

MODEL_CONFIGS = {
    '12b': {
        'onnx_path':    '../../model/llm_gemma/gemma-4-12b-it.onnx',
        'config_path':  '../../model/llm_gemma/gemma-4-12b-it.config.pkl',
        'rknn_path':    './gemma-4-12b-it.rknn',
    },
}

DATASET_PATH = None
MAX_CONTEXT_LEN = 4*1024
SLIDING_WINDOW = 1024

NUM_DYNAMIC_SEQ_INPUTS = 5
DEFAULT_MAX_SEQ_LEN = 128


def _extract_seg_idx(filepath):
    m = re.search(r'_seg(\d+)', filepath)
    if m is None:
        raise ValueError(f"Cannot parse segment index from path: {filepath}")
    return int(m.group(1))


def _segment_onnx_patterns(base_dir, base_name):
    filename_pattern = f"{base_name}_seg*.onnx"
    return (
        os.path.join(base_dir, "seg_*", filename_pattern),
        os.path.join(base_dir, filename_pattern),
    )


def _find_segment_onnx_files(base_dir, base_name):
    patterns = _segment_onnx_patterns(base_dir, base_name)
    for pattern in patterns:
        seg_files = glob.glob(pattern)
        if seg_files:
            return sorted(seg_files, key=_extract_seg_idx), patterns
    return [], patterns


def _resolve_segment_onnx(base_dir, base_name, seg_idx):
    filename = f"{base_name}_seg{seg_idx}.onnx"
    candidates = (
        os.path.join(base_dir, f"seg_{seg_idx}", filename),
        os.path.join(base_dir, filename),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"Segment {seg_idx} ONNX file not found; tried: "
        + ", ".join(candidates)
    )


def _build_rknn_config(args, is_multi_segment=False, is_last_segment=True):
    from rknn.api import DEFAULT_RKNN_LLM_CONFIG

    max_context_len = getattr(args, "max_context_len", MAX_CONTEXT_LEN)
    kvcache_buffer_len = max_context_len
    max_position_embeddings = max_context_len
    my_config = DEFAULT_RKNN_LLM_CONFIG.copy()

    attn_config = {
        'mask_name': 'attention_mask', 'position_name': 'position_ids',
        'kvcache_buffer_len': kvcache_buffer_len, 'max_position_embeddings': max_position_embeddings,
        'kvcache_dtype': 'Int4_to_F16', 'kvcache_store_method': 'GroupQuant',
        'kvcache_group_size': 16, 'kvcache_residual_depth': 64,
        'sliding_window_size': -1, 'attention_type': 'FullAttention',
        'position_embeddings_host_storage': True,
    }

    attn_sld_config = {
        'mask_name': 'attention_mask_1', 'position_name': 'position_ids_1',
        'kvcache_buffer_len': SLIDING_WINDOW, 'max_position_embeddings': max_position_embeddings,
        'kvcache_dtype': 'Float16', 'kvcache_store_method': 'Normal',
        'kvcache_group_size': 32, 'kvcache_residual_depth': 32,
        'sliding_window_size': SLIDING_WINDOW, 'attention_type': 'SlidingAttention',
        'position_embeddings_host_storage': True,
    }

    my_config['attention_config'] = [attn_sld_config, attn_config]

    if is_multi_segment and not is_last_segment:
        my_config['llm_head'] = []
    my_config['vocab'] = []

    print('LLM config is:', my_config)
    return my_config


def _export_single_segment(
    rknn,
    onnx_path,
    rknn_path,
    llm_config,
    dataset_path,
    config_path=None,
    rebuild=False,
    profile_mode=False,
    save_ctx=False,
):
    print('--> config model')
    onnx_model = onnx.load(onnx_path, load_external_data=False)
    dynamic_shapes = []
    for _input in onnx_model.graph.input:
        shape = []
        for dim in _input.type.tensor_type.shape.dim:
            if not dim.dim_value:
                shape.append(1)
            else:
                shape.append(dim.dim_value)
        dynamic_shapes.append(shape)
    min_shapes = []
    max_shapes = []
    for i, shape in enumerate(dynamic_shapes):
        if i < NUM_DYNAMIC_SEQ_INPUTS:
            min_shapes.append([1] + shape[1:])
            max_shapes.append([1, DEFAULT_MAX_SEQ_LEN] + shape[2:])
        else:
            min_shapes.append(list(shape))
            max_shapes.append(list(shape))
    dynamic_input = [min_shapes, max_shapes]
    print(f"onnx model dynamic input shapes: {dynamic_input}")

    rknn.config(
        target_platform='rk1820', dynamic_input=dynamic_input,
        profile_mode=profile_mode, quantized_dtype='w4a16',
        quantized_algorithm='normal', quantized_method='group32',
        llm_config=llm_config,
    )
    print('done')

    if rebuild:
        print('--> Rebuild rknn model')
        ret = rknn.rebuild("./tmp")
        if ret != 0:
            print('Rebuild rknn model failed!')
            sys.exit(ret)
        print('done')
    else:
        print('--> Loading model')
        ret = rknn.load_llm(model=onnx_path, config=config_path)
        if ret != 0:
            print('Load model failed!')
            sys.exit(ret)
        print('done')
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset=dataset_path)
        if ret != 0:
            print('Build model failed!')
            sys.exit(ret)
        print('done')

    print('--> Export RKNN model')
    ret = rknn.export_rknn(rknn_path, save_ctx=save_ctx)
    if ret != 0:
        print('Export rknn failed!')
        sys.exit(ret)
    print('done')


if __name__ == '__main__':
    from argparse import ArgumentParser, BooleanOptionalAction
    parser = ArgumentParser(description="Export Gemma4Unified LLM to RKNN model")
    parser.add_argument("--onnx_path", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--rknn_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH)
    parser.add_argument("--model_type", type=str, choices=['12b'], default='12b')
    parser.add_argument(
        "--max_context_len",
        "--max-context-len",
        type=int,
        default=MAX_CONTEXT_LEN,
        help="maximum KV cache/context length (default: %(default)s)",
    )
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
    parser.add_argument("--multi_segment", action='store_true')
    parser.add_argument("--num_segments", type=int, default=0)
    parser.add_argument(
        "--stop_after_segment",
        type=int,
        default=None,
        help="stop after this zero-based segment while preserving the full layout",
    )
    parser.add_argument('--rebuild', action='store_true', default=False)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_type]
    args.onnx_path = args.onnx_path or cfg['onnx_path']
    args.config = args.config or cfg['config_path']
    args.rknn_path = args.rknn_path or cfg['rknn_path']

    if args.multi_segment:
        segment_count = args.num_segments

        base_dir = os.path.dirname(args.onnx_path)
        base_name = os.path.splitext(os.path.basename(args.onnx_path))[0]
        rknn_output_dir = os.path.dirname(args.rknn_path) or '.'

        if segment_count <= 0:
            seg_files, patterns = _find_segment_onnx_files(base_dir, base_name)
            if not seg_files:
                print(
                    "No segment files found matching: " + ", ".join(patterns),
                    file=sys.stderr,
                )
                sys.exit(1)
            segment_count = len(seg_files)
            print(f"Found {segment_count} segments")

        if args.stop_after_segment is not None and not 0 <= args.stop_after_segment < segment_count:
            raise ValueError(
                "--stop_after_segment must be within the segment range: "
                f"0 <= {args.stop_after_segment} < {segment_count}"
            )
        segment_limit = (
            segment_count
            if args.stop_after_segment is None
            else args.stop_after_segment + 1
        )

        for seg_idx in range(segment_limit):
            seg_onnx = _resolve_segment_onnx(base_dir, base_name, seg_idx)
            seg_rknn = os.path.join(rknn_output_dir, f"{base_name}_seg{seg_idx}.rknn")
            is_last = (seg_idx == segment_count - 1)
            tmp_dir = os.path.join(os.getcwd(), "tmp")
            tmp_seg_dir = os.path.join(os.getcwd(), f"tmp_seg{seg_idx}")
            if args.rebuild and os.path.isdir(tmp_seg_dir):
                if os.path.isdir(tmp_dir):
                    shutil.rmtree(tmp_dir)
                shutil.move(tmp_seg_dir, tmp_dir)
                print(f'--> Restored tmp_seg{seg_idx} -> tmp for rebuild')
            rknn = RKNN(verbose=False)
            try:
                print(f'\n--> Exporting segment {seg_idx + 1}/{segment_count} (last={is_last}): {seg_onnx}')
                seg_config = _build_rknn_config(args, is_multi_segment=True, is_last_segment=is_last)
                _export_single_segment(
                    rknn,
                    seg_onnx,
                    seg_rknn,
                    seg_config,
                    args.dataset_path,
                    args.config,
                    rebuild=args.rebuild,
                    profile_mode=args.profile_mode,
                    save_ctx=args.save_ctx,
                )
            finally:
                rknn.release()
            if os.path.isdir(tmp_dir):
                if os.path.isdir(tmp_seg_dir):
                    shutil.rmtree(tmp_seg_dir)
                shutil.move(tmp_dir, tmp_seg_dir)
                print(f'--> Renamed tmp -> tmp_seg{seg_idx}')
            if args.stop_after_segment is not None and seg_idx >= args.stop_after_segment:
                print(
                    f"Stopping after segment {seg_idx} as requested by "
                    "--stop_after_segment"
                )
    else:
        rknn = RKNN(verbose=False)
        try:
            config = _build_rknn_config(args, is_multi_segment=False)
            _export_single_segment(
                rknn,
                args.onnx_path,
                args.rknn_path,
                config,
                args.dataset_path,
                args.config,
                rebuild=args.rebuild,
                profile_mode=args.profile_mode,
                save_ctx=args.save_ctx,
            )
        finally:
            rknn.release()
