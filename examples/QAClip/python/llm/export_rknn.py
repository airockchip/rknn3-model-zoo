import numpy as np
from rknn.api import RKNN

ONNX_MODEL = '../../models/chinese_clip_text.onnx'
RKNN_MODEL = '../../rknn/chinese_clip_text.rknn'

# 多桶动态shape
TEXT_BUCKETS = [
    [[1, 128, 768], [1, 128], [1, 128]],
    [[1, 256, 768], [1, 256], [1, 256]],
    [[1, 352, 768], [1, 352], [1, 352]],
]

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Chinese clip text convert rknn")
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument('--platform', type=str, default="rk1820", help='Target platform (e.g. rk1820)')
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    print('--> config model')
    rknn.config(target_platform=args.platform, core_num=1,
                quantized_dtype='w6a16', quantized_algorithm='normal', quantized_method='group32',
                dynamic_input=TEXT_BUCKETS,
                input_attrs={'input_embeds': {'dtype': 'float16'},
                             'attention_mask': {'dtype': 'float16'},
                             'position_ids': {'dtype': 'int32'}},
                output_attrs={'output': {'dtype': 'float16'}})
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
    ret = rknn.build(do_quantization=True)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    rknn.release()
