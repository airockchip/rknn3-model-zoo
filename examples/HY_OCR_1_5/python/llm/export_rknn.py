import numpy as np
from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG

ONNX_MODEL = '../../model/llm/HunyuanOCR-1.5-llm.onnx'
LLM_CONFIG = '../../model/llm/HunyuanOCR-1.5-llm.config.pkl'
RKNN_MODEL = '../../model/llm/HunyuanOCR-1.5-llm.rknn'
DATASET_PATH = None

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export Tencent-Hunyuan/HY-OCR1.5 llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument('--platform', type=str, default= "rk1820", help='Target platform (e.g. rk1820)')
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument('--core_num', type=int, default=8, help='core_num (1-8)')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()

    # HunyuanOCR-1.5 使用 xdrope (XD-RoPE) 位置编码
    # 根据 config.json：rope_scaling.xdrope_section = [16, 16, 16, 16] 
    # head_dim=128, qk_rope_head_dim=64, 分为 4 个部分，每部分 16
    llm_config['attention_config'][0]['mrope_type'] = 'HunyuanVL'
    llm_config['attention_config'][0]['mrope_section'] = [16, 16, 16, 16]
    llm_config['attention_config'][0]['mrope_new_id_name'] = 'xd_rope_id_input'

    llm_config['attention_config'][0]['kvcache_buffer_len'] = 1024
    llm_config['attention_config'][0]['max_position_embeddings'] = 1024

    # config.json has tie_word_embeddings=true, but ONNX uses separate embedding/lm_head weights,
    # so tie_embeddings must be set to False
    
    llm_config['vocab'][0]['tie_embeddings'] = False

    # pre-process config
    print('--> config model')
    rknn.config(target_platform=args.platform,
                core_num=args.core_num,
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                llm_config=llm_config)
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