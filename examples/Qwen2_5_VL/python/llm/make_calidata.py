import os
from PIL import Image
import json
import torch
import pickle
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.cali_data import capture_module_input
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration



parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct', help='model path', required=False)
parser.add_argument('--datapath', type=str, default='../../../../datasets/MMBench/llm/dataset.json', help='model path', required=False)
parser.add_argument('--export_datapath', type=str, default='./quant_data/model_inputs.json', help='model path', required=False)
args = parser.parse_args()

## 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path, device_map=device, torch_dtype=torch.bfloat16,
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
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        temp = capture_module_input(
            model.model,
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
