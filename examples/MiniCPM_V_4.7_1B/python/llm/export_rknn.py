import numpy as np
import os
from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG
import onnx


ONNX_MODEL = '../../model/llm/MiniCPM-V-4.7-1B-llm.onnx'
LLM_CONFIG = '../../model/llm/MiniCPM-V-4.7-1B-llm.config.pkl'
RKNN_MODEL = '../../model/llm/MiniCPM-V-4.7-1B-llm.rknn'

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export openbmb/MiniCPM-V-4.7-1B llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument('--platform', type=str, default="rk1820", help='Target platform (e.g. rk1820)')
    parser.add_argument('--rebuild', action='store_true', required=False, default=False, help='Whether to rebuild the model')
    args = parser.parse_args()

    rknn = RKNN(verbose=True)

    llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    llm_config['attention_config'][0]['kvcache_buffer_len'] = 4*1024
    llm_config['attention_config'][0]['max_position_embeddings'] = 4*1024
    # llm_config['attention_config'][0]['kvcache_dtype'] = 'Int4_to_F16' # 若有需要可开启kvcache量化
    # llm_config['attention_config'][0]['kvcache_store_method'] = 'GroupQuant'
    # llm_config['attention_config'][0]['kvcache_group_size'] = 16
    # llm_config['attention_config'][0]['kvcache_residual_depth'] = 64
    
    llm_config['attention_config'][0]['mrope_type'] = 'Qwen3.5'
    llm_config['attention_config'][0]['mrope_section'] = [11, 11, 10]
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
        quantized_method='group32',llm_config=llm_config, linear_attn_out_project_dtype=None,
        # op_quantized_dtype='./layer_bit.json', ## 如果在export_llm.py脚本里面开启了混合量化，则需要指定op量化配置json文件。
        input_initial_value=input_initial_value,
        cvt_conv_streaming=cvt_conv_streaming,
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
