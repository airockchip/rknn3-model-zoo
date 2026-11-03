import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.cali_data import capture_module_input
from PIL import Image
import json
import torch
import pickle
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

# MiniCPM-V-4.7 的远程 video_processing_minicpmv4_7.py 引用了当前 transformers 已移除的
# BASE_VIDEO_PROCESSOR_DOCSTRING（仅用作 @add_start_docstrings 的文档字符串常量），
# 这里补一个空字符串别名，避免 AutoProcessor 加载远程代码时 ImportError。
try:
    import transformers.video_processing_utils as _vpu
    if not hasattr(_vpu, "BASE_VIDEO_PROCESSOR_DOCSTRING"):
        _vpu.BASE_VIDEO_PROCESSOR_DOCSTRING = ""
except Exception:
    pass

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='openbmb/MiniCPM-V-4.7-1B', help='model path', required=False)
parser.add_argument('--datapath', type=str, default='../../../../datasets/MMBench/llm/dataset.json', help='model path', required=False)
parser.add_argument('--export_datapath', type=str, default='./quant_data/model_inputs.json', help='model path', required=False)
parser.add_argument('--modelscope', action='store_true', help='Whether download model from www.modelscope.cn')
args = parser.parse_args()

if args.modelscope:
    from modelscope import snapshot_download
    args.model_path = snapshot_download(args.model_path)

## 加载模型
kwargs = {
    'trust_remote_code': True,
}
config = AutoConfig.from_pretrained(args.model_path, **kwargs)
kwargs['config'] = config
model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, _attn_implementation="eager", trust_remote_code=True)

processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

## 生成量化校准数据
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
                {"type": "text", "text": data['input']},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    inputs = processor(text=[prompt], images=[imgp], return_tensors="pt")
    inputs.pop("mm_token_type_ids", None)  # MiniCPM-V-4.7 模型不走 mm_token_type_ids 的 mrope 路径
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
