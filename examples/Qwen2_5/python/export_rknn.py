import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '../model/llm/Qwen2.5-3B-Instruct.onnx'
LLM_CONFIG = '../model/llm/Qwen2.5-3B-Instruct.config.pkl'
RKNN_MODEL = '../model/llm/Qwen2.5-3B-Instruct.rknn'
DATASET_PATH = None

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export Qwen/Qwen2.5 llm to RKNN model") 
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--config", type=str, help="config file path", required=False, default=LLM_CONFIG)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument('--platform', type=str, required=True, help='Target platform (e.g. rk1820)')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(target_platform=args.platform, 
                quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                )
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

