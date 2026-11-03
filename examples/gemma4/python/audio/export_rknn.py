import numpy as np
from rknn.api import RKNN
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# e2b / e4b 模型路径配置
MODEL_CONFIGS = {
    'e2b': {
        'onnx_path': '../../model/audio/gemma-4-e2b-it-audio.onnx',
        'rknn_path': '../../model/audio/gemma-4-e2b-it-audio.rknn',
    },
    'e4b': {
        'onnx_path': '../../model/audio/gemma-4-e4b-it-audio.onnx',
        'rknn_path': '../../model/audio/gemma-4-e4b-it-audio.rknn',
    },
}

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="google/gemma-4 audio convert rknn")
    parser.add_argument("--onnx_path", type=str, default=None, help="onnx model path")
    parser.add_argument("--rknn_path", type=str, default=None, help="output rknn model path")
    parser.add_argument("--model_type", type=str, choices=['e2b', 'e4b'], default='e2b',
                        help="选择要导出的 Gemma-4 模型版本: e2b 或 e4b")
    parser.add_argument('--platform', type=str, default= "rk1820", help='Target platform (e.g. rk1820)')
    parser.add_argument('--core_num', type=int, default=4, help='core_num')
    args = parser.parse_args()

    # 根据 model_type 选择对应配置
    cfg = MODEL_CONFIGS[args.model_type]
    args.onnx_path = args.onnx_path or cfg['onnx_path']
    args.rknn_path = args.rknn_path or cfg['rknn_path']

    # Create RKNN object
    rknn = RKNN(verbose=True)

    print('--> config model')
    rknn.config(target_platform=args.platform, core_num=args.core_num,
                quantized_dtype='w8a16', quantized_algorithm='normal', quantized_method='group32',
                profile_mode=False)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=args.onnx_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset="dataset.txt")
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(args.rknn_path, save_ctx=False)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    
    rknn.release()
