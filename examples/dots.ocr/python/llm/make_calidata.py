import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import json
import pickle
import sys
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.cali_data import capture_module_input

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='rednote-hilab/dots.ocr', help='model path')
parser.add_argument('--datapath', type=str, default='../../../../datasets/MMBench/llm/dataset.json', help='dataset json path')
parser.add_argument('--export_datapath', type=str, default='./quant_data/model_inputs.json', help='output calibration data path')
args = parser.parse_args()

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    args.model_path, device_map="cpu", torch_dtype=torch.bfloat16,
    trust_remote_code=True).eval()

processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

# 生成量化校准数据
info = []
with open(args.datapath, "r") as f:
    datasets = json.load(f)
for idx, data in enumerate(tqdm(datasets)):
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

    from qwen_vl_utils import process_vision_info
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
    info.append({"sample": "model_inputs/{}".format(sample_name)})
    with open(pickle_path, 'wb') as f:
        pickle.dump(temp, f)

with open(args.export_datapath, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print(f"Generated {len(info)} calibration samples to {args.export_datapath}")
