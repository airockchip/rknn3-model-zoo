import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import torch
from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_qwen2_5_omni_quantize_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Qwen2_5OmniForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


prompt = "RKLLM"
chat_context = {
    "messages":[
        {
            "role": "user",
            "content": [
                {"type": "image",},
                {"type": "text", "text": prompt},
            ],
        }
    ],
    "add_generation_prompt": True,
}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen2.5-Omni llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/Qwen2.5-Omni-3B-llm.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True,
        'torch_dtype': torch.bfloat16 
    }
        
    new_llm_path = "Qwen2.5-Omni-3B-language"
    omni_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        **kwargs
    ).thinker.eval()


    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.save_pretrained(new_llm_path)

    # del omni_model.visual
    del omni_model.audio_tower
    
    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=omni_model.model, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        omni_model.model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)

        omni_model = omni_model.cpu()
        
    omni_model.save_pretrained(new_llm_path)
    os.system("cp config.json {}/".format(new_llm_path))


    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True,
        'torch_dtype': torch.bfloat16
    }
    config = AutoConfig.from_pretrained(new_llm_path, **kwargs)
    update_config(config, ['use_cache'], False)
    kwargs['config'] = config
    model = AutoModelForCausalLM.from_pretrained(new_llm_path, **kwargs)

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
        os.makedirs(export_dirname)

    # Export llm to onnx
    causal_llm_to_onnx(model, args)

    # Export LLM configuration 
    export_llm_config(new_llm_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')