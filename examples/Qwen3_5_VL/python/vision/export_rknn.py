import numpy as np
import onnx
import os,sys
from rknn.api import RKNN
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lora_node_map import build_lora_node_map

ONNX_MODEL = '../../model/vision/Qwen3.5-4B-vision.onnx'
RKNN_MODEL = '../../model/vision/Qwen3.5-4B-vision.rknn'
LORA_MODEL = None
LORA_CONFIG = None
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'vision_config.json')

def load_config(config_path: str):
    import json
    import os
    if not os.path.exists(config_path):
        return 384, 384, np.array([[1,24,24]], dtype=np.int64)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    required_keys = ['img_h', 'img_w']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing key '{key}' in config file")
    grid_thw = config.get('grid_thw', [1, 24, 24])
    return config["img_h"], config["img_w"], np.array([grid_thw], dtype=np.int64)


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Qwen/Qwen3.5-4B vision convert rknn")
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--lora_path", type=str, help="LoRA adapter safetensors path; omit to export Base only", default=LORA_MODEL)
    parser.add_argument("--lora_config_path", type=str, help="LoRA adapter_config.json path", default=LORA_CONFIG)
    parser.add_argument("--no_prune_mode", dest="prune_mode", action="store_false", help="close prune mode")
    parser.add_argument('--core_num', type=int, default=8, help='core_num (1-8)')
    parser.add_argument('--platform', type=str, default="rk1820", choices=["rk1820", "rk1828", "rk3572"],
                        help='Target platform (rk1820/rk1828/rk3572)')
    parser.add_argument("--hf_onnx_map", type=str, default=None,
                        help="HF weight to ONNX op mapping; defaults to <onnx_path>.hf_onnx_map.json")
    parser.add_argument("--lora_node_map_output", type=str, default=None,
                        help="optional exact RKNN LoRA node map JSON output path")
    parser.add_argument("--lora_node_map_report", type=str, default=None,
                        help="optional LoRA node map report output path")
    parser.add_argument("--lora_filtered_path", type=str, default=None,
                        help="optional filtered LoRA safetensors output path")
    parser.set_defaults(prune_mode=True)
    args = parser.parse_args()

    has_lora_args = args.lora_path is not None or args.lora_config_path is not None
    if has_lora_args and (args.lora_path is None or args.lora_config_path is None):
        raise ValueError('--lora_path and --lora_config_path must be provided together')
    if has_lora_args:
        if not os.path.isfile(args.lora_path):
            raise FileNotFoundError(f'LoRA safetensors file not found: {args.lora_path}')
        if not os.path.isfile(args.lora_config_path):
            raise FileNotFoundError(f'LoRA config file not found: {args.lora_config_path}')

    img_h, img_w, grid_thw = load_config(CONFIG_PATH)
    print(f"Using img_h={img_h}, img_w={img_w}, grid_thw={grid_thw}")
    
    # Create RKNN object
    rknn = RKNN(verbose=True)

    print('--> config model')
    if args.prune_mode == True:
        rknn.config(target_platform=args.platform, core_num=args.core_num,
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32')
    else:
        rknn.config(target_platform=args.platform, core_num=args.core_num,
                    quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                    # Qwen3.5's exported visual graph is NCHW float input.
                    # Keep this layout so RKNN does not reinterpret the
                    # graph's [1, 3, H, W] shape as NHWC during shape
                    # inference. C++ performs the [-1, 1] normalization.
                    input_attrs={'pixel': {'dtype': 'float16', 'layout': 'NCHW'}}
                    )
    print('done')

    # Load model
    print('--> Loading model')
    if args.prune_mode == True:
        ret = rknn.load_onnx(model=args.onnx_path,
                    inputs=['/Reshape_1_output_0'],
                    input_size_list=[[int(grid_thw[0][1] * grid_thw[0][2]), 1536]]
                    )
    else:
        # export_qwen2_vl_vision may constant-fold grid_thw for a fixed
        # resolution. Pass only the inputs that are still present in the
        # exported graph, otherwise RKNN rejects the stale input name.
        graph_input_names = {
            value.name for value in onnx.load(args.onnx_path, load_external_data=False).graph.input
        }
        full_inputs = ['pixel']
        full_input_sizes = [[1, 3, img_h, img_w]]
        full_initial_values = [None]
        if 'grid_thw' in graph_input_names:
            full_inputs.append('grid_thw')
            full_input_sizes.append([1, 3])
            full_initial_values.append(grid_thw)
        load_kwargs = {
            'model': args.onnx_path,
            'inputs': full_inputs,
            'input_size_list': full_input_sizes,
        }
        if len(full_inputs) > 1:
            load_kwargs['input_initial_val'] = full_initial_values
        ret = rknn.load_onnx(**load_kwargs)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    if has_lora_args:
        onnx_stem = os.path.splitext(args.onnx_path)[0]
        hf_onnx_map_path = args.hf_onnx_map or (onnx_stem + '.hf_onnx_map.json')
        rknn_stem = os.path.splitext(args.rknn_path)[0]
        node_map_output = args.lora_node_map_output or (rknn_stem + '.lora_node_map.json')
        node_map_report = args.lora_node_map_report or (rknn_stem + '.lora_node_map.report.json')
        filtered_lora_path = args.lora_filtered_path or (rknn_stem + '.vision.safetensors')
        print('--> Building exact vision LoRA node map')
        lora_node_map, lora_load_path, _ = build_lora_node_map(
            lora_path=args.lora_path,
            hf_onnx_map_path=hf_onnx_map_path,
            node_map_output=node_map_output,
            report_output=node_map_report,
            filtered_lora_output=filtered_lora_path,
            scope="vision",
        )
        ret = rknn.load_lora(
            lora_paths=[[lora_load_path]],
            lora_config_paths=[[args.lora_config_path]],
            lora_patterns=[["lora0_pattern0"]],
            lora_quantized_dtype="int4",
            lora_quantized_method="group32",
            lora_distribute_strategy="less_mem",
            lora_node_map=lora_node_map,
        )
        if ret != 0:
            print('Load LoRA failed!')
            exit(ret)

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    export_rknn_dirname = os.path.dirname(args.rknn_path)
    if export_rknn_dirname and not os.path.exists(export_rknn_dirname):
        print(f"create export_rknn_dirname: {export_rknn_dirname}")
        os.makedirs(export_rknn_dirname, exist_ok=True)

    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print(f"Export rknn model failed! {args.rknn_path}")
        exit(ret)
    print(f"done {args.rknn_path}")

    rknn.release()

