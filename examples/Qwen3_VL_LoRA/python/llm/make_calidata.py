import os
from PIL import Image
import json
import torch
import pickle
from tqdm import tqdm
from transformers import AutoProcessor
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.cali_data import capture_module_input
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling_qwen3_vl import Qwen3VLForConditionalGeneration



parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-VL-2B-Instruct', help='model path', required=False)
parser.add_argument('--datapath', type=str, default='../../../../datasets/MMBench/llm/dataset.json', help='model path', required=False)
parser.add_argument('--export_datapath', type=str, default='./quant_data/model_inputs.json', help='model path', required=False)
args = parser.parse_args()

## 加载模型
model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype=torch.float32, device_map="cpu",
    trust_remote_code=True).eval()

processor = AutoProcessor.from_pretrained(args.model_path)

## 生成量化校准数据
info = []
with open(args.datapath, "r") as f:
    datasets = json.load(f)
for idx, data in enumerate(tqdm(datasets)):
    image_name = data["image"].split(".")[0]
    path_ = "/".join(args.datapath.split("/")[:-1])
    imgp = os.path.join(path_, data["image_path"], data["image"])

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": imgp
                },
                {"type": "text", "text": data["input"]},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        temp = capture_module_input(
            model.model.language_model,
            lambda: model.generate(**inputs, max_new_tokens=128)
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
