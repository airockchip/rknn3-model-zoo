import os
import sys
import json
import pickle
from tqdm import tqdm
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.cali_data import capture_module_input

from transformers import AutoProcessor, AutoModelForImageTextToText


MODEL_CONFIGS = {
    "1.6B": "LiquidAI/LFM2.5-VL-1.6B",
    "3B": "LiquidAI/LFM2.5-VL-3B",
}

parser = argparse.ArgumentParser()
parser.add_argument('--model_size', type=str, default='1.6B', choices=list(MODEL_CONFIGS.keys()),
                    help='Model variant (default: 1.6B)')
parser.add_argument('--model_path', type=str, default=None, help='model path (overrides --model_size default)')
parser.add_argument('--datapath', type=str, default='../../../../datasets/MMBench/llm/dataset.json', help='dataset json path')
parser.add_argument('--export_datapath', type=str, default='./quant_data/model_inputs.json', help='output calibration data index path')
args = parser.parse_args()

if args.model_path is None:
    args.model_path = MODEL_CONFIGS[args.model_size]
print(f"--> Model variant: {args.model_size} (model={args.model_path})")

# Load model
model = AutoModelForImageTextToText.from_pretrained(
    args.model_path, torch_dtype=torch.float32, device_map="cpu",
    trust_remote_code=True).eval()

processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

# Find language model for hooking
lang_model = model.model.language_model if hasattr(model.model, 'language_model') else model.model.text_model
if lang_model is None:
    lang_model = model.model

if lang_model is None:
    print("[ERROR] Cannot find language model!")
    exit(1)

# Generate calibration data
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
                {"type": "image", "image": imgp},
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
            lang_model,
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
print(f"Calibration data saved to: {args.export_datapath}")
