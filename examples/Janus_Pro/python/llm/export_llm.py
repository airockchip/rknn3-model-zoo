import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_janus_pro_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, make_dataset_for_janus_pro
from transformers import AutoModelForCausalLM, AutoConfig
from janus.models import MultiModalityCausalLM, VLChatProcessor # please execute install_janus.sh

prompt = "RKLLM"
conversation = [
    {
        "role": "<|User|>", 
        "content": f"<image_placeholder>\n{prompt}",
        "images":["../../data/vision/img.jpg"],
    },

    {
        "role": "<|Assistant|>", 
        "content": "",
    },
]

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export deepseek-ai/Janus-Pro llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="deepseek-ai/Janus-Pro-1B")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/Janus-Pro-1B-llm.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    ori_model_name = args.model_path

    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    update_config(config, ['_attn_implementation'], 'eager')
    kwargs['config'] = config
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)

        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.language_model, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        model.language_model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)

        model = model.cpu()
        
    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_llm_dirname):
        os.makedirs(export_llm_dirname)

    # Export llm to onnx
    causal_llm_to_onnx(model.language_model, args)

    # Export LLM configuration 
    export_janus_pro_llm_config(ori_model_name, args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', conversation, prompt)

    # Export tokenizer
    export_tokenizer(ori_model_name, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')  # LlamaTokenizer

    # Export embedding weight
    export_embed_weight(model.language_model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')