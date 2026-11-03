import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

# dots.mocr 的 modeling_dots_vision.py 第 7 行硬编码 `from flash_attn import flash_attn_varlen_func`
# （无 try/except fallback）。CPU 环境无 flash_attn 时，注入纯 PyTorch 等价实现：
# 对 cu_seqlens 划分的每个变长序列做 scaled_dot_product_attention（数学等价）。
import types
import importlib.machinery
import sys as _sys
import torch
import torch.nn.functional as F


def _flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=False, return_attn_probs=False):
    # q/k/v: [total_seq, nheads, dim]
    outputs = []
    nheads = q.shape[1]
    for i in range(len(cu_seqlens_q) - 1):
        s, e = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        qi, ki, vi = q[s:e], k[s:e], v[s:e]  # [seq, nheads, dim]
        qi = qi.transpose(0, 1)
        ki = ki.transpose(0, 1)
        vi = vi.transpose(0, 1)
        out = F.scaled_dot_product_attention(qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0), is_causal=causal)
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


_stub = types.ModuleType("flash_attn")
_stub.flash_attn_varlen_func = _flash_attn_varlen_func
_stub.__version__ = "2.7.4"
_stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
_sys.modules["flash_attn"] = _stub

# transformers 兼容性 patch：configuration_dots.py 顶层调用 AutoProcessor.register
# 的字符串 key 与新版 transformers 不兼容
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING
_orig_register = PROCESSOR_MAPPING.register


def _safe_register(key, processor_class=None, exist_ok=False):
    if isinstance(key, str):
        return None
    return _orig_register(key, processor_class, exist_ok=exist_ok)


PROCESSOR_MAPPING.register = _safe_register

# transformers 4.51 兼容 patch：mocr 的 DotsVLProcessor.__init__ 按新版签名
# (image_processor, tokenizer, video_processor, chat_template) 位置传参，而
# 4.51 的 Qwen2_5_VLProcessor.__init__ 是 (image_processor, tokenizer, chat_template)，
# video_processor 会错位到 chat_template 上。把位置参数按关键字转发修正。
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

if "video_processor" not in Qwen2_5_VLProcessor.__init__.__code__.co_varnames:
    _orig_qwen25vl_init = Qwen2_5_VLProcessor.__init__

    def _compat_init(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        _orig_qwen25vl_init(self, image_processor, tokenizer, chat_template=chat_template, **kwargs)

    Qwen2_5_VLProcessor.__init__ = _compat_init

import json
import pickle
import sys
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.cali_data import capture_module_input

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='rednote-hilab/dots.mocr', help='model path')
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
