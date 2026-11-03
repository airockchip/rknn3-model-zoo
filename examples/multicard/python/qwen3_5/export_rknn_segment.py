import numpy as np
import os
from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG
import onnx
import shutil
import re
import sys

ONNX_MODEL = '../../model/llm_qwen/Qwen3.5-9B-llm.onnx'
LLM_CONFIG = '../../model/llm_qwen/Qwen3.5-9B-llm.config.pkl'
RKNN_MODEL = './Qwen3.5-9B-llm.rknn'
MAX_CONTEXT_LEN = 4 * 1024

NUM_DYNAMIC_SEQ_INPUTS = 3
DEFAULT_MAX_SEQ_LEN = 128


def _extract_seg_idx(filepath):
    m = re.search(r'seg(\d+)', filepath)
    if m is None:
        raise ValueError(f"Cannot parse segment index from path: {filepath}")
    return int(m.group(1))


def _build_llm_config(max_context_len=MAX_CONTEXT_LEN):
    llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    llm_config['attention_config'][0]['kvcache_buffer_len'] = max_context_len
    llm_config['attention_config'][0]['max_position_embeddings'] = max_context_len
    llm_config['attention_config'][0]['kvcache_dtype'] = 'Int4_to_F16'
    llm_config['attention_config'][0]['position_embeddings_host_storage'] = True
    llm_config['vocab'] = []
    return llm_config


def _export_single_segment(rknn, onnx_path, rknn_path, llm_config, config_path, args):
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
    dynamic_shapes = [min_shapes, max_shapes]
    print(f"onnx model dynamic input shapes: {dynamic_shapes}")

    input_initial_value = []
    for i, dyn_s in enumerate(dynamic_shapes):
        _tmp = [None] * (len(dyn_s) - 1) + [np.zeros(dyn_s[-1], dtype=np.int64)]
        input_initial_value.append(_tmp)

    cvt_conv_streaming = []
    _model = onnx.load(onnx_path, load_external_data=False)
    remain_output = [o for o in _model.graph.output if o.name != 'conv_state_out']
    if len(remain_output) != len(_model.graph.output):
        del _model.graph.output[:]
        _model.graph.output.extend(remain_output)
        _new_path = onnx_path[:-5] + '_rm_conv_statue_out.onnx'
        onnx.save(_model, _new_path)
        onnx_path = _new_path
    for node in _model.graph.node:
        if node.op_type == 'Conv':
            cvt_conv_streaming.append(node.output[0])

    print('--> config model')
    rknn.config(
        target_platform=args.platform, dynamic_input=dynamic_shapes,
        quantized_dtype='w4a16', quantized_algorithm='normal',
        quantized_method='group32', llm_config=llm_config,
        linear_attn_out_project_dtype=None,
        cvt_conv_streaming=cvt_conv_streaming,
        profile_mode=getattr(args, "profile_mode", False)
    )
    print('done')

    if args.rebuild:
        print('--> Rebuild rknn model')
        ret = rknn.rebuild("./tmp")
        if ret != 0:
            print('Rebuild rknn model failed!')
            sys.exit(ret)
        print('done')
    else:
        print('--> Loading model')
        ret = rknn.load_llm(model=onnx_path, config=config_path, seq=[1, DEFAULT_MAX_SEQ_LEN])
        if ret != 0:
            print('Load model failed!')
            sys.exit(ret)
        print('done')
        print('--> Building model')
        ret = rknn.build(do_quantization=True)
        if ret != 0:
            print('Build model failed!')
            sys.exit(ret)
        print('done')

    print('--> Export rknn model')
    export_rknn_dirname = os.path.dirname(rknn_path)
    if export_rknn_dirname and not os.path.exists(export_rknn_dirname):
        os.makedirs(export_rknn_dirname, exist_ok=True)
    ret = rknn.export_rknn(
        rknn_path,
        save_ctx=getattr(args, "save_ctx", False),
    )
    if ret != 0:
        print('Export rknn model failed!')
        sys.exit(ret)
    print('done')


if __name__ == '__main__':
    from argparse import ArgumentParser, BooleanOptionalAction
    parser = ArgumentParser(description="Export Qwen3_5 segment RKNN model for multicard deployment")
    parser.add_argument("--onnx_path", type=str, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, default=RKNN_MODEL)
    parser.add_argument('--platform', type=str, default="rk1820")
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
    parser.add_argument('--rebuild', action='store_true', default=False)
    parser.add_argument("--multi_segment", action='store_true')
    parser.add_argument("--num_segments", type=int, default=0)
    parser.add_argument(
        "--stop_after_segment",
        type=int,
        default=None,
        help="stop after this zero-based segment while preserving the full layout",
    )
    args = parser.parse_args()

    if args.multi_segment:
        segment_count = args.num_segments

        base_dir = os.path.dirname(args.onnx_path)
        base_name = os.path.splitext(os.path.basename(args.onnx_path))[0]
        rknn_output_dir = os.path.dirname(args.rknn_path) or '.'

        if segment_count <= 0:
            import glob
            pattern = os.path.join(base_dir, f"seg*", f"{base_name}_seg*.onnx")
            seg_files = glob.glob(pattern)
            if not seg_files:
                print(f"No segment files found matching {pattern}")
                sys.exit(1)
            seg_files.sort(key=_extract_seg_idx)
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
            seg_onnx = os.path.join(base_dir, f"seg{seg_idx}", f"{base_name}_seg{seg_idx}.onnx")
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
                llm_config = _build_llm_config(args.max_context_len)
                if not is_last:
                    llm_config['llm_head'] = []
                _export_single_segment(rknn, seg_onnx, seg_rknn, llm_config, args.config, args)
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
            llm_config = _build_llm_config(args.max_context_len)
            _export_single_segment(rknn, args.onnx_path, args.rknn_path, llm_config, args.config, args)
        finally:
            rknn.release()