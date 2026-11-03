import sys
import argparse
import os
from rknn.api import RKNN

# 默认配置

DEFAULT_QUANT = False
DEFAULT_DATASET = '../../../datasets/imagenet/RAFT-Stereo_img_samples/image_pairs.txt'

def parse_args():
    parser = argparse.ArgumentParser(description='Convert ONNX to RKNN.')
    parser.add_argument(
        '--onnx_model_path', 
        type=str, 
        default='onnx_out/raft_stereo.onnx',
        help='Path to ONNX model.')
    parser.add_argument(
        '--platform',
        type=str,
        default='rk1820',
        choices=['rk1820', 'rk1828', 'rk3572'],
        help='Target platform.'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='fp',
        choices=['i8', 'u8', 'fp'],
        help='Model data type. i8/u8 will trigger quantization unless --no_quant is set.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./test.rknn',
        help='Output RKNN file path.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=DEFAULT_DATASET,
        help='Calibration dataset file for quantization.'
    )
    parser.add_argument(
        '--no_quant',
        action='store_true',
        help='Force disable quantization even if dtype is i8/u8.'
    )
    parser.add_argument(
        '--use_sparsity',
        default=False,
        action='store_true',
        help='Enable sparsity training')
    parser.add_argument(
        '--profile_mode',
        action='store_true',
        help='Enable profile mode for board-connected accuracy analysis (rknn.inference accuracy_analysis=True)')
    args = parser.parse_args()

    # 确定是否量化
    do_quant = DEFAULT_QUANT
    if args.dtype in ['i8', 'u8'] and not args.no_quant:
        do_quant = True
    elif args.dtype == 'fp' or args.no_quant:
        do_quant = False

    return args, do_quant


if __name__ == '__main__':
    args, do_quant = parse_args()
    abs_path = os.path.abspath(args.output)
    directory = os.path.dirname(abs_path)
    os.makedirs(directory, exist_ok=True)

    rknn = RKNN(verbose=False)

    print('--> Config model')
    
    # 基础配置
    config_params = {
        'mean_values': [[0, 0, 0], [0, 0, 0]],
        'std_values': [[1, 1, 1], [1, 1, 1]],
        'target_platform': args.platform,
        'profile_mode': args.profile_mode,
        'input_attrs': {'image_left': {'dtype': 'uint8', 'layout': 'NHWC'},
                        'image_right': {'dtype': 'uint8', 'layout': 'NHWC'}}
    }
    
    if args.use_sparsity:
        config_params.update({
            'model_pruning': False,
            'sparse_infer': True,
        })
    else:
        config_params.update({
            'model_pruning': False,
            'sparse_infer': False,
        })
    
    # 如果需要量化，添加量化参数
    if do_quant:
        config_params.update({
            'quantized_dtype': 'w8a16',
            'quantized_algorithm': 'normal',
            'quantized_method': 'channel',
        })
        print(f"do_quant enabled: dtype:{config_params['quantized_dtype']}, "
              f"algorithm: {config_params['quantized_algorithm']}, "
              f"method: {config_params['quantized_method']}")
    
    rknn.config(**config_params)
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=args.onnx_model_path)
    if ret != 0:
        print('Load model failed!')
        sys.exit(ret)
    print('done')

    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=args.dataset)
    if ret != 0:
        print('Build model failed!')
        sys.exit(ret)
    print('done')

    print('--> Export rknn model')
    # save_ctx=True：生成 .rknn.ctx 文件，load_rknn(load_ctx=True) 精度分析时需要
    ret = rknn.export_rknn(args.output, save_ctx=args.profile_mode)
    if ret != 0:
        print('Export rknn model failed!')
        sys.exit(ret)
    print('done')
