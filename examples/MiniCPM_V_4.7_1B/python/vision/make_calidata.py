import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.cali_data import capture_module_input
from PIL import Image
import json
import torch
import pickle
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoImageProcessor

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='openbmb/MiniCPM-V-4.7-1B', help='model path', required=False)
parser.add_argument('--datapath', type=str, default='../../../../datasets/MMBench/llm/dataset.json', help='model path', required=False)
parser.add_argument('--export_datapath', type=str, default='./quant_data/model_inputs.json', help='model path', required=False)
args = parser.parse_args()

## 加载模型
kwargs = {
    'trust_remote_code': True,
    'torch_dtype': torch.float32,
}
config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
kwargs['config'] = config
model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)

image_processor = AutoImageProcessor.from_pretrained(args.model_path, trust_remote_code=True)

## 生成量化校准数据
info = []
with open(args.datapath, "r") as f:
    datasets = json.load(f)
for idx, data in enumerate(tqdm(datasets)):
    path_ = "/".join(args.datapath.split("/")[:-1])
    imgp = os.path.join(path_, data["image_path"], data["image"])

    # 固定 448x448 单切片，与 export_vision.py 的静态输入一致
    image = Image.open(imgp).convert('RGB')
    image = image.resize((448, 448))

    with torch.inference_mode():
        inputs = image_processor(images=image, return_tensors='pt', max_slice_nums=1)
        pixel_values = inputs['pixel_values']
        target_sizes = inputs['target_sizes']
        if tuple(pixel_values.shape) != (1, 3, 14, 14336):
            print(f"warning: sample {idx} pixel_values shape {tuple(pixel_values.shape)} != (1,3,14,14336), skip")
            continue
        temp = capture_module_input(
            model.model.vision_tower,
            lambda: model.get_image_features(pixel_values=pixel_values, target_sizes=target_sizes)
        )
    sample_name = "sample_{}".format(idx)
    path_ = "/".join(args.export_datapath.split("/")[:-1])
    path_dir = os.path.join(path_, "model_inputs")
    os.makedirs(path_dir, exist_ok=True)
    pickle_path = os.path.join(path_dir, sample_name)
    info.append({"sample":"model_inputs/{}".format(sample_name)})
    with open(pickle_path, 'wb') as f:
        pickle.dump(temp, f)

with open(args.export_datapath, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
