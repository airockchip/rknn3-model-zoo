import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from transformers import AutoConfig, HunYuanVLForConditionalGeneration

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

    parser = ArgumentParser(description="Export HunyuanOCR-1.5 llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="tencent/HunyuanOCR")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/HunyuanOCR-1.5-llm.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
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

    kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.bfloat16,
        'attn_implementation': 'eager',
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    update_config(config, ['_attn_implementation_autoset'], False)
    kwargs['config'] = config
    
    model = HunYuanVLForConditionalGeneration.from_pretrained(args.model_path, **kwargs)

    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        QuantTool = RKQuantizer(verbose=True)

        ret = QuantTool.load_model(model=model.model, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)

        dataset = args.cali_dataset
        model.model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)

        model = model.cpu()

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_llm_dirname):
        os.makedirs(export_llm_dirname)

    # Export llm to onnx
    causal_llm_to_onnx(model, args)

    # Export LLM configuration
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.language_model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')
