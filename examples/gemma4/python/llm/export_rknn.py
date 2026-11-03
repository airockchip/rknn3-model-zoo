import numpy as np
from rknn.api import RKNN

# e2b / e4b 模型路径配置
MODEL_CONFIGS = {
    'e2b': {
        'onnx_path':    '../../model/llm/gemma-4-e2b-it.onnx',
        'config_path':  '../../model/llm/gemma-4-e2b-it.config.pkl',
        'rknn_path':    '../../model/llm/gemma-4-e2b-it.rknn',
    },
    'e4b': {
        'onnx_path':    '../../model/llm/gemma-4-e4b-it.onnx',
        'config_path':  '../../model/llm/gemma-4-e4b-it.config.pkl',
        'rknn_path':    '../../model/llm/gemma-4-e4b-it.rknn',
    },
}

DATASET_PATH = None

MAX_CONTEXT_LEN = 4096

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export gemma llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, default=None, help="onnx model path")
    parser.add_argument("--config", type=str, default=None, help="config file path")
    parser.add_argument("--rknn_path", type=str, default=None, help="output rknn model path")
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument("--model_type", type=str, choices=['e2b', 'e4b'], default='e2b',
                        help="选择要导出的 Gemma-4 模型版本: e2b 或 e4b")
    parser.add_argument('--platform', type=str, default= "rk1820", help='Target platform (e.g. rk1820)')
    args = parser.parse_args()

    # 根据 model_type 选择对应配置
    cfg = MODEL_CONFIGS[args.model_type]
    args.onnx_path = args.onnx_path or cfg['onnx_path']
    args.config    = args.config or cfg['config_path']
    args.rknn_path = args.rknn_path or cfg['rknn_path']

    import os
    # os.chdir("../../model/llm/")

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    
    from rknn.api import DEFAULT_RKNN_LLM_CONFIG
    kvcache_buffer_len = MAX_CONTEXT_LEN
    max_position_embeddings = MAX_CONTEXT_LEN
    my_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    sliding_window = 512

    attn_config = {                                   # accept multi internal kvcache as list(dict)
        'mask_name'              : 'attention_mask',         # required, this is used for recognize which attn belong to this config
        'position_name'          : 'position_ids',           # allow None if no position_ids input
        'kvcache_buffer_len'     : kvcache_buffer_len,                     # feed int or str "real_time"
        'max_position_embeddings': max_position_embeddings,
        'kvcache_dtype'          : 'Int4_to_F16',                # Float16, Int8_to_F16, Int4_to_F16
        'kvcache_store_method'   : 'GroupQuant',                 # GroupQuant, Normal
        'kvcache_group_size'     : 16,
        'kvcache_residual_depth' : 64,
        'sliding_window_size'    : -1,
        'attention_type'         : 'FullAttention',          # FullAttention, SlidingAttention
        'position_embeddings_host_storage'     : True,
    }
        
    attn_sld_config = {                                   # accept multi internal kvcache as list(dict)
        'mask_name'              : 'attention_mask_1',         # required, this is used for recognize which attn belong to this config
        'position_name'          : 'position_ids_1',           # allow None if no position_ids input
        'kvcache_buffer_len'     : sliding_window,                     # feed int or str "real_time"
        'max_position_embeddings': max_position_embeddings,
        'kvcache_dtype'          : 'Float16',                # Float16, Int8_to_F16, Int4_to_F16
        'kvcache_store_method'   : 'Normal',                 # GroupQuant, Normal
        'kvcache_group_size'     : 32,
        'kvcache_residual_depth' : 32,
        'sliding_window_size'    : sliding_window,
        'attention_type'         : 'SlidingAttention',          # FullAttention, SlidingAttention
        'position_embeddings_host_storage'     : True,
    }
    attention_config = []
    attention_config.append(attn_sld_config)
    attention_config.append(attn_config)

    my_config['attention_config'] = attention_config

    print('LLM config is:', my_config)

    if "2b" in args.onnx_path.lower():
        dynamic_input = [[[1, 1], [1, 1, 35, 256], [1, 1], [1, 1], [1, 1], [1, 1], [1]], 
                            [[1, 128], [1, 128, 35, 256], [1, 128], [1, 128], [1, 128], [1, 128], [1]]]
    elif "4b" in args.onnx_path.lower():
        dynamic_input = [[[1, 1], [1, 1, 42, 256], [1, 1], [1, 1], [1, 1], [1, 1], [1]], 
                            [[1, 128], [1, 128, 42, 256], [1, 128], [1, 128], [1, 128], [1, 128], [1]]]

    rknn.config(target_platform=args.platform, dynamic_input = dynamic_input, profile_mode=False,
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                llm_config=my_config,
                input_attrs={'per_layer_inputs': {'dtype': 'float16', 'layout': 'NCHW'}})
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_llm(model=args.onnx_path, config=args.config)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=args.dataset_path)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    #Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('done')

    rknn.release()

