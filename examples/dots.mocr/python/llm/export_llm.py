import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

try:
    import flash_attn  # noqa: F401
except ImportError:
    import importlib.machinery, sys as _sys, types as _types
    _stub = _types.ModuleType("flash_attn")
    _stub.flash_attn_varlen_func = None
    _stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    _stub.__version__ = "2.7.4"
    _sys.modules["flash_attn"] = _stub

import torch
import sys
os.environ["PATH"] = os.path.dirname(sys.executable) + ":" + os.environ.get("PATH", "")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from transformers import AutoConfig, AutoModelForCausalLM

prompt = "RKLLM"
chat_context = {
    "messages": [
        {
            "role": "user",
            "content": prompt,
        }
    ],
    "add_generation_prompt": True,
}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export dots.mocr llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="rednote-hilab/dots.mocr")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/dots-mocr-llm.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    args = parser.parse_args()

    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True,
        'torch_dtype': torch.bfloat16,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    update_config(config, ['_attn_implementation_autoset'], False)
    if hasattr(config, 'vision_config'):
        try:
            config.vision_config.attn_implementation = 'eager'
        except Exception:
            pass
    kwargs['config'] = config
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)

    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)

        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.model, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)

        ## 执行量化算法
        dataset = args.cali_dataset
        model.model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)

        model = model.cpu()

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_llm_dirname):
        os.makedirs(export_llm_dirname)

    # Export llm to onnx
    # 替换 forward 为标准 Qwen2ForCausalLM 实现，绕过 vision 部分
    from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    model.forward = Qwen2ForCausalLM.forward.__get__(model)
    causal_llm_to_onnx(model, args)

    # Export LLM configuration
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')
