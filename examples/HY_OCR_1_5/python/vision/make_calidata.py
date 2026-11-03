import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import json
import torch
import pickle
from tqdm import tqdm
from transformers import AutoProcessor, AutoConfig, HunYuanVLForConditionalGeneration # transformers>=5.13.0
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.cali_data import capture_module_input
from py_utils.export_llm_helper import update_config


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='tencent/HunyuanOCR', help='model path', required=False)
parser.add_argument('--datapath', type=str, default='../../../../datasets/OmniDocBench_ROI/llm/dataset.json', help='calibration dataset path', required=False)
parser.add_argument('--export_datapath', type=str, default='./quant_data/model_inputs.json', help='export calibration data path', required=False)
parser.add_argument("--hf_download", action='store_true', help="Whether download model from HuggingFace")
parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
args = parser.parse_args()

if args.hf_download and args.modelscope:
    parser.error("--hf_download and --modelscope are mutually exclusive")

if args.modelscope:
    from modelscope import snapshot_download
    args.model_path = snapshot_download(args.model_path)
elif args.hf_download:
    from huggingface_hub import snapshot_download
    rawmodel_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model/rawmodel'))
    args.model_path = snapshot_download(repo_id=args.model_path, local_dir=rawmodel_dir)

## 加载模型
# 与 export_vision.py 保持一致：float32 + eager attention
kwargs = {
    'trust_remote_code': True,
    'attn_implementation': 'eager',
    'torch_dtype': torch.float32,
}
config = AutoConfig.from_pretrained(args.model_path, **kwargs)
update_config(config, ['_attn_implementation_autoset'], False)
kwargs['config'] = config
model = HunYuanVLForConditionalGeneration.from_pretrained(args.model_path, device_map="cpu", **kwargs).eval()
model = model.float()

processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)

## 生成量化校准数据
# hook 目标为 model.model.vision_tower，与 export_vision.py 中 QuantTool.load_model(model=model.model.vision_tower) 一致
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
            model.model.vision_tower,
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
