#!/usr/bin/env python3
"""Export Mage-VL Qwen3-4B LLM configuration and ONNX model for RKNN."""

import json
import os
import pickle
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, masking_utils


if hasattr(masking_utils, "sdpa_mask_older_torch"):
    masking_utils.sdpa_mask = masking_utils.sdpa_mask_older_torch

MODEL_ZOO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.append(MODEL_ZOO)

from py_utils.export_llm_helper import causal_llm_to_onnx, export_embed_weight, export_tokenizer, split_chat_template_prompt


MODEL_PATH = "microsoft/Mage-VL"
EXPORT_LLM_PATH = "../../model/llm/model_llm.onnx"

prompt = "RKLLM"
chat_context = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ],
    "add_generation_prompt": True,
}


def update_config(config, attr_names, value, visited=None):
    from transformers import PretrainedConfig

    if visited is None:
        visited = set()
    if id(config) in visited:
        return
    visited.add(id(config))

    for attr in dir(config):
        if attr in attr_names:
            setattr(config, attr, value)
            continue

        try:
            attr_value = getattr(config, attr)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue

        if isinstance(attr_value, PretrainedConfig):
            update_config(attr_value, attr_names, value, visited)


class LanguageModelWithLMHead(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        if hasattr(model.model, "language_model"):
            self.language_model = model.model.language_model
            self.config = model.config.text_config
        else:
            self.language_model = model.model
            self.config = model.config
        weight_t = model.lm_head.weight.detach().t().contiguous()
        self.lm_head_weight_t = torch.nn.Parameter(weight_t, requires_grad=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, logits_to_keep=None):
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = outputs[0]
        logits_to_keep = logits_to_keep.to(device=hidden_states.device, dtype=torch.long)
        selected_hidden_states = torch.index_select(hidden_states, dim=1, index=logits_to_keep)
        return torch.matmul(selected_hidden_states, self.lm_head_weight_t)


def get_auxiliary_paths(model_path):
    return model_path, model_path


def export_mage_vl_llm_config(model_path, tokenizer_path, vl_config_path, output_path):
    local_files_only = os.path.isdir(model_path)
    config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only, trust_remote_code=True)
    text_config = config.text_config if hasattr(config, "text_config") else config
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=local_files_only, trust_remote_code=True)

    system_prompt = ""
    prompt_prefix = ""
    prompt_postfix = ""
    chat_template = tokenizer.chat_template or ""
    template_path = os.path.join(vl_config_path, "chat_template.jinja")
    if not chat_template and os.path.isfile(template_path):
        with open(template_path, "r", encoding="utf-8") as file:
            chat_template = file.read()

    if chat_template:
        try:
            system_prompt, prompt_prefix, prompt_postfix = split_chat_template_prompt(chat_template, chat_context, prompt)
        except Exception:
            system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            prompt_prefix = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n"

    hf_config = text_config.to_dict()
    vl_config = config.to_dict()
    for name in ["image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"]:
        if name in vl_config:
            hf_config[name] = vl_config[name]

    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": chat_template,
        "vocab_size": int(text_config.vocab_size),
        "hidden_size": int(text_config.hidden_size),
        "hf_config_json": json.dumps(hf_config),
    }
    with open(output_path, "wb") as file:
        pickle.dump(llm_config, file)

    print("system_prompt:", repr(system_prompt))
    print("prompt_prefix:", repr(prompt_prefix))
    print("prompt_postfix:", repr(prompt_postfix))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Mage-VL Qwen3-4B llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="model path or name")
    parser.add_argument("--export_llm_path", type=str, default=EXPORT_LLM_PATH, help="export llm onnx model path")
    parser.add_argument("--quant", action="store_true", help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default="./quant_data/model_inputs.json", help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action="store_true", help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download

        args.model_path = snapshot_download(args.model_path)

    if os.path.isdir(args.model_path):
        args.model_path = os.path.abspath(args.model_path)
    args.export_llm_path = os.path.abspath(args.export_llm_path)
    tokenizer_path, vl_config_path = get_auxiliary_paths(args.model_path)

    kwargs = {
        "config": AutoConfig.from_pretrained(args.model_path, local_files_only=os.path.isdir(args.model_path), trust_remote_code=True),
        "local_files_only": os.path.isdir(args.model_path),
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "eager",
        "low_cpu_mem_usage": True,
    }
    update_config(kwargs["config"], ["use_cache"], False)
    update_config(kwargs["config"], ["_attn_implementation_autoset"], False)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs).eval()
    print("Loaded Mage-VL Qwen3-4B model weights successfully.")

    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        QuantTool = RKQuantizer(verbose=True)
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        ret = QuantTool.load_model(model=language_model, tokenizer=None, device="cuda")
        if ret != 0:
            print("Load model failed!")
            exit(ret)

        dataset = args.cali_dataset
        language_model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)
        if hasattr(model.model, "language_model"):
            model.model.language_model = language_model
        else:
            model.model = language_model
        model = model.cpu()
    elif args.quant:
        print("Warning: CUDA is unavailable, export the non-GRQ ONNX model.")
    wrapped_model = LanguageModelWithLMHead(model)
    wrapped_model.eval()

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if export_llm_dirname and not os.path.exists(export_llm_dirname):
        os.makedirs(export_llm_dirname, exist_ok=True)

    text_config = kwargs["config"].text_config if hasattr(kwargs["config"], "text_config") else kwargs["config"]
    args.hidden_size = text_config.hidden_size
    args.dynamo_export = True
    causal_llm_to_onnx(wrapped_model, args)

    export_path = os.path.splitext(args.export_llm_path)[0]
    export_mage_vl_llm_config(args.model_path, tokenizer_path, vl_config_path, export_path + ".config.pkl")
    export_tokenizer(tokenizer_path, export_path + ".tokenizer.gguf")
    if not os.path.isfile(export_path + ".tokenizer.gguf") or os.path.getsize(export_path + ".tokenizer.gguf") == 0:
        raise RuntimeError("Tokenizer export failed. Please use the standard Qwen3 tokenizer directory.")
    language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
    export_embed_weight(language_model.embed_tokens.weight, export_path + ".embed.bin")
    print("done")
