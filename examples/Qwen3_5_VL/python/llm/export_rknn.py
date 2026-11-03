import numpy as np
import os
import sys
from rknn.api import RKNN,DEFAULT_RKNN_LLM_CONFIG
import onnx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lora_node_map import build_lora_node_map


ONNX_MODEL = '../../model/llm/Qwen3.5-4B-llm.onnx'
LORA_MODEL = None
LORA_CONFIG = None
LLM_CONFIG = '../../model/llm/Qwen3.5-4B-llm.config.pkl'
RKNN_MODEL = '../../model/llm/Qwen3.5-4B-llm.rknn'

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export Qwen/Qwen3_5 llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--lora_path", type=str, help="LoRA adapter safetensors path; omit to export Base only", default=LORA_MODEL)
    parser.add_argument("--lora_config_path", type=str, help="LoRA adapter_config.json path", default=LORA_CONFIG)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument('--platform', type=str, default="rk1820", help='Target platform (e.g. rk1820)')
    parser.add_argument("--hf_onnx_map", type=str, default=None,
                        help="HF weight to ONNX op mapping; defaults to <onnx_path>.hf_onnx_map.json")
    parser.add_argument("--lora_node_map_output", type=str, default=None,
                        help="optional exact RKNN LoRA node map JSON output path")
    parser.add_argument("--lora_node_map_report", type=str, default=None,
                        help="optional LoRA node map report output path")
    parser.add_argument("--lora_filtered_path", type=str, default=None,
                        help="optional filtered LoRA safetensors output path")
    parser.add_argument('--rebuild', action='store_true', required=False, default=False, help='Whether to rebuild the model')
    args = parser.parse_args()

    has_lora_args = args.lora_path is not None or args.lora_config_path is not None
    if args.rebuild and has_lora_args:
        raise ValueError('--rebuild cannot load a new LoRA; run a full export when LoRA is enabled')
    if has_lora_args and (args.lora_path is None or args.lora_config_path is None):
        raise ValueError('--lora_path and --lora_config_path must be provided together')
    if has_lora_args:
        if not os.path.isfile(args.lora_path):
            raise FileNotFoundError(f'LoRA safetensors file not found: {args.lora_path}')
        if not os.path.isfile(args.lora_config_path):
            raise FileNotFoundError(f'LoRA config file not found: {args.lora_config_path}')

    # The converter may create a temporary graph with conv_state_out removed;
    # the HF/ONNX map still points to the original exported graph.
    source_onnx_path = args.onnx_path

    rknn = RKNN(verbose=True)

    llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    llm_config['attention_config'][0]['kvcache_buffer_len'] = 8*1024
    llm_config['attention_config'][0]['max_position_embeddings'] = 8*1024
    llm_config['attention_config'][0]['kvcache_dtype'] = 'Int4_to_F16' # 若有需要可开启kvcache量化
    llm_config['attention_config'][0]['kvcache_store_method'] = 'GroupQuant'
    llm_config['attention_config'][0]['kvcache_group_size'] = 16
    llm_config['attention_config'][0]['kvcache_residual_depth'] = 64
    llm_config['attention_config'][0]['mrope_type'] = 'Qwen3.5'
    llm_config['attention_config'][0]['mrope_section'] = [11,11,10] # Please refer to config.json to configure this parameters. eg, https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/config.json
    llm_config['attention_config'][0]['mrope_new_id_name'] = 'mrope_id_input'

    
    onnx_model = onnx.load(args.onnx_path, load_external_data=False)
    dynamic_shapes = []
    for _input in onnx_model.graph.input:
        shape = []
        for dim in _input.type.tensor_type.shape.dim:
            if not dim.dim_value:
                shape.append(1)
            else:
                shape.append(dim.dim_value)
        dynamic_shapes.append(shape)
    dynamic_shapes = [[[1, 128,], [1, 128], [1, 128]] + dynamic_shapes[3:], dynamic_shapes]
    # dynamic_shapes = [dynamic_shapes]
    print(f"onnx model dynamic input shapes: {dynamic_shapes}")

    # 初始化 conv state 输入为全零矩阵
    input_initial_value = []
    for i, dyn_s in enumerate(dynamic_shapes):
        _tmp = [None]* (len(dyn_s) - 1) + [np.zeros(dyn_s[-1], dtype=np.float32)]
        input_initial_value.append(_tmp)

    # remove onnx conv_state_out 输出
    cvt_conv_streaming = []
    _model = onnx.load(args.onnx_path, load_external_data=False)
    remain_output = [o for o in _model.graph.output if o.name != 'conv_state_out']
    if len(remain_output) != len(_model.graph.output):
        del _model.graph.output[:]
        _model.graph.output.extend(remain_output)
        _new_path = args.onnx_path[:-5] + '_rm_conv_statue_out.onnx'
        onnx.save(_model, _new_path)
        args.onnx_path = _new_path
    for node in _model.graph.node:
        if node.op_type == 'Conv':
            cvt_conv_streaming.append(node.output[0])

    # pre-process config
    print('--> config model')
    
    rknn.config(
        target_platform = args.platform, dynamic_input = dynamic_shapes,
        quantized_dtype='w4a16', quantized_algorithm='normal',
        quantized_method='group32',llm_config=llm_config,
        input_initial_value=input_initial_value,
        cvt_conv_streaming=cvt_conv_streaming,
        linear_attn_out_project_dtype=None
    )
    print('done')

    if args.rebuild:
        print('--> Rebuild rknn model')
        ret = rknn.rebuild("./tmp")
        if ret != 0:
            print('Rebuild rknn model failed!')
            exit(ret)
        print('done')
    else:
        # Load model
        print('--> Loading model')
        ret = rknn.load_llm(model=args.onnx_path, config=args.config, seq=[1,128])

        if ret != 0:
            print('Load model failed!')
            exit(ret)

        if has_lora_args:
            hf_onnx_map_path = args.hf_onnx_map or (os.path.splitext(source_onnx_path)[0] + '.hf_onnx_map.json')
            rknn_stem = os.path.splitext(args.rknn_path)[0]
            node_map_output = args.lora_node_map_output or (rknn_stem + '.lora_node_map.json')
            node_map_report = args.lora_node_map_report or (rknn_stem + '.lora_node_map.report.json')
            filtered_lora_path = args.lora_filtered_path or (rknn_stem + '.llm.safetensors')
            print('--> Building exact LoRA node map')
            lora_node_map, lora_load_path, _ = build_lora_node_map(
                lora_path=args.lora_path,
                hf_onnx_map_path=hf_onnx_map_path,
                node_map_output=node_map_output,
                report_output=node_map_report,
                filtered_lora_output=filtered_lora_path,
                scope="LLM",
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
        print('done')

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
        os.makedirs(export_rknn_dirname, exist_ok=True)
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    rknn.release()
