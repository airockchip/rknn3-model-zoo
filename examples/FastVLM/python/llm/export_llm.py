import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from transformers import AutoConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llava.model import *  # 增加logits_to_keep输入

def replace_image_pad(tokenizer_config_path, tokenizer_config_new_path):
    with open(tokenizer_config_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # 检查是否包含 <|image_pad|>
    if "<|image_pad|>" not in content:
        print("<|image_pad|> not found, replacing tokenizer_config.json context")
        with open(tokenizer_config_new_path, 'r', encoding='utf-8') as file:
            new_content = file.read()
        with open(tokenizer_config_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print("tokenizer_config.json has been replaced.")


prompt = "RKLLM"
chat_context = {
    "messages":[
        {"role": "user", "content": prompt}
    ],
    "add_generation_prompt": True,
}


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export FastVLM llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default='../../llava-fastvithd_1.5b_stage3') # 模型下载链接：https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/FastVLM-llm.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    args = parser.parse_args()

    
    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    kwargs['config'] = config
    model = LlavaQwen2ForCausalLM.from_pretrained(args.model_path, **kwargs)

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
    causal_llm_to_onnx(model, args)

    replace_image_pad(os.path.join(args.model_path, '')+'tokenizer_config.json', "./tokenizer_config.json")

    # Export LLM configuration 
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf') # LlavaQwen2ForCausalLM Tokenizer=Qwen2ForCausalLM Tokenizer

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')