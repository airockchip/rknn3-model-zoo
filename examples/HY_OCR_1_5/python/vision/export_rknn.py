import numpy as np
from rknn.api import RKNN
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

ONNX_MODEL = '../../model/vision/HunyuanOCR-1.5-vision.onnx'
RKNN_MODEL = '../../model/vision/HunyuanOCR-1.5-vision.rknn'
DATASET_PATH = None

# HunyuanOCR-1.5 vision tower 参数
PATCH_SIZE = 16          # vision_config.patch_size
MERGE_SIZE = 2           # vision_config.spatial_merge_size
TEMPORAL_PATCH_SIZE = 1  # vision_config.temporal_patch_size
CHANNEL = 3              # vision_config.num_channels
# 裁剪版输入维度 = channel * temporal_patch_size * patch_size * patch_size
PATCH_DIM = CHANNEL * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE  # 768
# CLIP/SigLIP 归一化参数 (preprocessor_config.json)
MEAN = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255]
STD = [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255]


def load_config(config_path: str):
    import json
    import os
    # 默认 512x512，与 export_vision.py 的默认值保持一致
    if not os.path.exists(config_path):
        return 512, 512, np.array([[1, 32, 32]], dtype=np.int64)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    required_keys = ['img_h', 'img_w', 'grid_thw']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing key '{key}' in config file")
    return config["img_h"], config["img_w"], np.array([config["grid_thw"]], dtype=np.int64)


def gen_hunyuan_vision_prune_quantize_dataset(dataset_path, dataset_out_path, img_h, img_w, mean, std):
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

    patch_size = PATCH_SIZE
    merge_size = MERGE_SIZE
    temporal_patch_size = TEMPORAL_PATCH_SIZE
    channel = CHANNEL

    output_paths = []
    for img_path in dataset:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_LANCZOS4).transpose(2, 0, 1).reshape(1, channel, img_h, img_w)
        img = np.float32(img)
        # 归一化 (RGB 顺序)
        img[0, 2, ...] = (img[0, 2, ...] - mean[2]) / std[2]
        img[0, 1, ...] = (img[0, 1, ...] - mean[1]) / std[1]
        img[0, 0, ...] = (img[0, 0, ...] - mean[0]) / std[0]

        h = img.shape[2]
        w = img.shape[3]
        # 与 export_vision.py forward 一致 (temporal_patch_size=1, 不需 repeat)
        patches = img.reshape(
            1, temporal_patch_size, channel,
            h // merge_size // patch_size, merge_size, patch_size,
            w // merge_size // patch_size, merge_size, patch_size
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        feature = patches.reshape(
            h // patch_size * w // patch_size,
            channel * temporal_patch_size * patch_size * patch_size
        )

        name_without_ext = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(out_dir, f"npy/{name_without_ext}.npy")
        np.save(str(output_path), feature)
        output_paths.append(os.path.abspath(output_path))

    with open(dataset_out_path, 'w', encoding='utf-8') as outfile:
        for output_path in output_paths:
            outfile.write(output_path + '\n')

    return 0


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser(description="Export HunyuanOCR-1.5 vision to RKNN model")
    parser.add_argument("--onnx_path", type=str, help="onnx model path", required=False, default=ONNX_MODEL)
    parser.add_argument("--rknn_path", type=str, help="output rknn model path", required=False, default=RKNN_MODEL)
    parser.add_argument('--platform', type=str, default="rk1820", help='Target platform (e.g. rk1820)')
    parser.add_argument("--dataset_path", type=str, help="model quantization dataset path", required=False, default=DATASET_PATH)
    parser.add_argument("--no_prune", dest="no_prune", action="store_true", help="export full model (default: prune mode)")
    parser.add_argument("--prune_input", type=str, default="/Reshape_1_output_0", help="ONNX node name of flattened patches for prune mode")
    parser.add_argument('--core_num', type=int, default=8, help='core_num (1-8)')
    parser.set_defaults(no_prune=False)
    args = parser.parse_args()

    img_h, img_w, grid_thw = load_config("vision_config.json")
    print(f"Using img_h={img_h}, img_w={img_w}, grid_thw={grid_thw}")

    # Create RKNN object
    rknn = RKNN(verbose=True)

    print('--> config model')
    if args.no_prune:
        # 完整版：输入原始像素 (uint8, NHWC)，由 RKNN 完成 mean/std 归一化
        rknn.config(target_platform=args.platform, core_num=args.core_num,
                    quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
                    mean_values=[MEAN],
                    std_values=[STD],
                    input_attrs={'pixel': {'dtype': 'uint8', 'layout': 'NHWC'}}
                    )
    else:
        # 裁剪版：patch 展平预处理在 CPU 完成，直接输入 flatten_patches
        rknn.config(target_platform=args.platform, core_num=args.core_num,
                    quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32')
    print('done')

    # Load model
    print('--> Loading model')
    if args.no_prune:
        # 完整版：输入像素 [1, 3, img_h, img_w]
        ret = rknn.load_onnx(model=args.onnx_path,
                             inputs=['pixel'],
                             input_size_list=[[1, CHANNEL, img_h, img_w]]
                             )
    else:
        # 裁剪版：2 个输入 flatten_patches + pos_embed (grid_h/grid_w 已固化为常量)
        num_patches = int(grid_thw[0][1] * grid_thw[0][2])
        ret = rknn.load_onnx(model=args.onnx_path,
                             inputs=['flatten_patches', 'pos_embed'],
                             input_size_list=[[num_patches, PATCH_DIM],
                                              [num_patches, 1152]]
                             )
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # prune 模式下，若提供了图像列表数据集，则生成展平后的 npy 校准数据
    if not args.no_prune and args.dataset_path is not None:
        ret = gen_hunyuan_vision_prune_quantize_dataset(
            args.dataset_path, "../../data/vision/datasets_npy.txt",
            img_h, img_w, MEAN, STD)
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
    export_rknn_dirname = os.path.dirname(args.rknn_path)
    if export_rknn_dirname and not os.path.exists(export_rknn_dirname):
        print(f"create export_rknn_dirname: {export_rknn_dirname}")
        os.makedirs(export_rknn_dirname, exist_ok=True)

    ret = rknn.export_rknn(args.rknn_path)
    if ret != 0:
        print(f'Export rknn model failed! {args.rknn_path}')
        exit(ret)
    print(f'done {args.rknn_path}')

    rknn.release()
