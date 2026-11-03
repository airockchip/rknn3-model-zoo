import os
from rknn.api import RKNN

MODEL_SIZE = os.environ.get("SPARK_MODEL_SIZE", "1.7B")
ONNX_MODEL = f'../model/llm/Spark-X2.5-{MODEL_SIZE}.onnx'
LLM_CONFIG = f'../model/llm/Spark-X2.5-{MODEL_SIZE}.config.pkl'
RKNN_MODEL = f'../model/llm/Spark-X2.5-{MODEL_SIZE}.rknn'
DATASET_PATH = '../../../datasets/Spark_X2_5/dataset.txt'

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export Spark2_5 llm to RKNN model")
    parser.add_argument("--model_size", type=str, help="model size, used to derive onnx/config/rknn paths (1.7B/4B)",
                        required=False, default=MODEL_SIZE, choices=["1.7B", "4B"])
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=None)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=None)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=None)
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument("--platform", type=str, help="target platform, e.g. rk1820/rk1828/rk3572", required=False, default='rk1820')
    parser.add_argument("--max_context_len", type=int, help="max context length for full attention kvcache", required=False, default=4096)
    parser.add_argument("--seq_len", type=int, help="prefill seq length for build", required=False, default=128)
    parser.add_argument("--core_num", type=int, help="NPU core number", required=False, default=8)
    args = parser.parse_args()

    if args.onnx_path is None:
        args.onnx_path = f'../model/llm/Spark-X2.5-{args.model_size}.onnx'
    if args.config is None:
        args.config = f'../model/llm/Spark-X2.5-{args.model_size}.config.pkl'
    if args.rknn_path is None:
        args.rknn_path = f'../model/llm/Spark-X2.5-{args.model_size}.rknn'

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')

    from rknn.api import DEFAULT_RKNN_LLM_CONFIG
    my_config = DEFAULT_RKNN_LLM_CONFIG.copy()
    MAX_CONTEXT_LEN = args.max_context_len
    kvcache_buffer_len = MAX_CONTEXT_LEN
    max_position_embeddings = MAX_CONTEXT_LEN
    sliding_window = 512  # Spark-X2.5 config.json sliding_window

    attn_config = {                                   # accept multi internal kvcache as list(dict)
        'mask_name'              : 'attention_mask',         # required, this is used for recognize which attn belong to this config
        'position_name'          : 'position_ids',           # allow None if no position_ids input
        'kvcache_buffer_len'     : kvcache_buffer_len,                     # feed int or str "real_time"
        'max_position_embeddings': max_position_embeddings,
        'kvcache_dtype'          : 'Float16',                # Float16, Int8_to_F16, Int4_to_F16
        'kvcache_store_method'   : 'Normal',                 # GroupQuant, Normal
        'kvcache_group_size'     : 32,
        'kvcache_residual_depth' : 32,
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

    dynamic_input = [[[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1]],
                        [[1, args.seq_len], [1, args.seq_len], [1, args.seq_len], [1, args.seq_len], [1, args.seq_len], [1]]]

    rknn.config(target_platform=args.platform, dynamic_input=dynamic_input,
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                llm_config=my_config,
                core_num=args.core_num,
                # profile_mode=True,
                )
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_llm(model=args.onnx_path, config=args.config, seq=[1, args.seq_len])
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
    ret = rknn.export_rknn(args.rknn_path, save_ctx=True)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('done')

    rknn.release()
