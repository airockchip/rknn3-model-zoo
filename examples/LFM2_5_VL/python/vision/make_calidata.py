import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.cali_data import capture_module_input
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import argparse
import json
import pickle
from tqdm import tqdm

MODEL_CONFIGS = {
    "1.6B": "LiquidAI/LFM2.5-VL-1.6B",
    "3B":   "LiquidAI/LFM2.5-VL-3B",
}

parser = argparse.ArgumentParser(description="Generate GRQ calibration data for LFM2.5-VL vision model")
parser.add_argument('--model_size', type=str, default='1.6B', choices=list(MODEL_CONFIGS.keys()))
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--datapath', type=str,
                    default='../../../../datasets/MMBench/llm/dataset.json',
                    help='MMBench dataset json for calibration images')
parser.add_argument('--export_datapath', type=str, default='./quant_data/model_inputs.json',
                    help='output json index path (pickle samples go to model_inputs/)')
parser.add_argument('--modelscope', action='store_true')
args = parser.parse_args()

if args.model_path is None:
    args.model_path = MODEL_CONFIGS[args.model_size]
if args.modelscope:
    from modelscope import snapshot_download
    args.model_path = snapshot_download(args.model_path)

print(f"--> Model: {args.model_size} ({args.model_path})")

model = AutoModelForImageTextToText.from_pretrained(
    args.model_path, torch_dtype=torch.float32, device_map="cpu",
    trust_remote_code=True).eval()
processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

vision_model = model.model.vision_tower

## 生成量化校准数据
info = []
with open(args.datapath, "r") as f:
    datasets = json.load(f)
for idx, data in enumerate(tqdm(datasets)):
    path_ = "/".join(args.datapath.split("/")[:-1])
    imgp = os.path.join(path_, data["image_path"], data["image"])
    if not os.path.exists(imgp):
        print(f"  skip {idx}: image not found {imgp}")
        continue

    img = Image.open(imgp).convert('RGB')
    inputs = processor(images=img, text='<image>', return_tensors='pt', do_image_splitting=False)

    # Hook vision_tower and call it directly; the pre-hook captures
    # (pixel_values, pixel_attention_mask, spatial_shapes) and short-circuits.
    with torch.inference_mode():
        temp = capture_module_input(
            vision_model,
            lambda: vision_model(
                pixel_values=inputs['pixel_values'].float(),
                pixel_attention_mask=inputs['pixel_attention_mask'].float(),
                spatial_shapes=inputs['spatial_shapes'],
            )
        )

    sample_name = "sample_{}".format(idx)
    path_dir = os.path.join("/".join(args.export_datapath.split("/")[:-1]), "model_inputs")
    os.makedirs(path_dir, exist_ok=True)
    pickle_path = os.path.join(path_dir, sample_name)
    info.append({"sample": "model_inputs/{}".format(sample_name)})
    with open(pickle_path, 'wb') as f:
        pickle.dump(temp, f)

with open(args.export_datapath, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)

print(f"\nCalibration data saved: {len(info)} samples")
print(f"  index: {args.export_datapath}")
