import numpy as np
from rknn.api import RKNN
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

ONNX_MODEL = '../../model/vision/dots-ocr-vision.onnx'
RKNN_MODEL = '../../model/vision/dots-ocr-vision.rknn'
DATASET_PATH = '../../../../datasets/MMBench/vision/datasets.txt'

IMAGE_MEAN = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
IMAGE_STD = [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]
PATCH_SIZE = 14
FLATTEN_DIM = 588  # 3 * temporal_patch_size(1) * 14 * 14


def load_config(config_path: str):
    import json
    if not os.path.exists(config_path):
        return 448, 448, np.array([[1, 32, 32]], dtype=np.int64)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    required_keys = ['img_h', 'img_w', 'grid_thw']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing key '{key}' in config file")
    return config["img_h"], config["img_w"], np.array([config["grid_thw"]], dtype=np.int64)


def gen_dots_ocr_vision_prune_quantize_dataset(dataset_path, dataset_out_path, img_h, img_w, mean, std, patch_size):
    import cv2
    base_dir = os.path.dirname(os.path.abspath(dataset_path))
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            img_path = line.strip()
            abs_img_path = os.path.join(base_dir, img_path) if not os.path.isabs(img_path) else img_path
            if abs_img_path.lower().endswith('.npy'):
                print("The prune model does not support npy input.")
                return -1
            dataset.append(abs_img_path)

    out_dir = os.path.dirname(dataset_out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    os.makedirs(os.path.join(out_dir, "npy"), exist_ok=True)

    output_paths = []
    for img_path in dataset:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_w, img_h)).transpose(2, 0, 1).reshape(1, 3, img_h, img_w)
        img = np.float32(img)
        img[0, 2, ...] = (img[0, 2, ...] - mean[2]) / std[2]
        img[0, 1, ...] = (img[0, 1, ...] - mean[1]) / std[1]
        img[0, 0, ...] = (img[0, 0, ...] - mean[0]) / std[0]
        h = img.shape[2]
        w = img.shape[3]
        patches = img.reshape(1, 1, 3, h // 2 // patch_size, 2, patch_size, w // 2 // patch_size, 2, patch_size)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        feature = patches.reshape(1 * h // patch_size * w // patch_size, 3 * 1 * patch_size * patch_size)

        name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(out_dir, f"npy/{name_without_ext}.npy")
        np.save(str(output_path), feature)
        # 写相对路径（相对 dataset 文件所在目录），保证 datasets_npy.txt 可跨机器使用
        output_paths.append(os.path.relpath(os.path.abspath(output_path), start=os.path.abspath(out_dir)))

    with open(dataset_out_path, 'w', encoding='utf-8') as outfile:
        for output_path in output_paths:
            outfile.write(output_path + '\n')

    return 0


img_h, img_w, grid_thw = load_config('vision_config.json')
print(f"Using img_h={img_h}, img_w={img_w}, grid_thw={grid_thw}")

if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="dots.ocr vision convert rknn")
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument('--platform', type=str, default="rk1820", help='Target platform (e.g. rk1820)')
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument("--no_prune_mode", dest="prune_mode", action="store_false", help="close prune mode")
    parser.add_argument('--core_num', type=int, default=8, help='core_num (1-8)')
    parser.set_defaults(prune_mode=True)
    args = parser.parse_args()

    # Create RKNN object
    rknn = RKNN(verbose=True)

    print('--> config model')
    if args.prune_mode == True:
        rknn.config(target_platform=args.platform, core_num=args.core_num,
                    quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32')
    else:
        rknn.config(target_platform=args.platform, core_num=args.core_num,
                    quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                    mean_values=[IMAGE_MEAN],
                    std_values=[IMAGE_STD],
                    input_attrs={'pixel': {'dtype': 'uint8', 'layout': 'NHWC'}})
    print('done')

    # Load model
    print('--> Loading model')
    if args.prune_mode == True:
        ret = rknn.load_onnx(model=args.onnx_path,
                             inputs=['/Reshape_1_output_0', 'grid_thw'],
                             input_size_list=[[int(grid_thw[0][1] * grid_thw[0][2]), FLATTEN_DIM], [1, 3]],
                             input_initial_val=[None, grid_thw])
    else:
        ret = rknn.load_onnx(model=args.onnx_path,
                             inputs=['pixel', 'grid_thw'],
                             input_size_list=[[1, 3, img_h, img_w], [1, 3]],
                             input_initial_val=[None, grid_thw])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    if args.prune_mode == True:
        ret = gen_dots_ocr_vision_prune_quantize_dataset(
            args.dataset_path, "../../data/vision/datasets_npy.txt",
            img_h, img_w, IMAGE_MEAN, IMAGE_STD, PATCH_SIZE)
        if ret == 0:
            args.dataset_path = "../../data/vision/datasets_npy.txt"

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=args.dataset_path)
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
