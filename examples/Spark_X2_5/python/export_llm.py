#!/usr/bin/env python3
"""Export Spark-X2.5 (1.7B / 4B, model_type=spark2_5) to ONNX for RKNN3 load_llm.

Loads the model with the local python/spark2_5 modeling package, then calls
spark2_5_to_onnx which turns the two mask streams and two RoPE streams into
explicit ONNX inputs.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch

from py_utils.export_llm_helper import update_config, export_tokenizer, export_llm_config, export_embed_weight
from export_onnx import spark2_5_to_onnx

from spark2_5.modeling_spark import Spark2_5ForCausalLM

PROMPT = "RKLLM"
chat_context = {
    "messages": [
        {"role": "user", "content": PROMPT}
    ],
    "add_generation_prompt": True,
}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Spark-X2.5 (1.7B/4B) llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False,
                        default="XHToken/Spark-X2.5-1.7B")
    parser.add_argument("--model_size", type=str, help="model size, used to derive export_llm_path default (1.7B/4B)",
                        required=False, default="1.7B", choices=["1.7B", "4B"])
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "llm",
                                             "Spark-X2.5-" + os.environ.get("SPARK_MODEL_SIZE", "1.7B") + ".onnx"))
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='../../../datasets/llm_quant.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        if args.model_path == "XHToken/Spark-X2.5-1.7B" and args.model_size == "4B":
            args.model_path = "XHToken/Spark-X2.5-4B"
        args.model_path = snapshot_download(args.model_path)

    # Derive export path from model_path/model_size when not explicitly given
    if "--export_llm_path" not in sys.argv:
        model_name = os.path.basename(args.model_path.rstrip('/'))
        if "Spark-X2.5" in model_name:
            args.export_llm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "llm",
                                                model_name + ".onnx")
        else:
            args.export_llm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "llm",
                                                "Spark-X2.5-" + args.model_size + ".onnx")

    from transformers import AutoConfig, AutoTokenizer

    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True,
        'torch_dtype': torch.bfloat16,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    kwargs['config'] = config
    model = Spark2_5ForCausalLM.from_pretrained(args.model_path, **kwargs)

    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)

        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model, tokenizer=tokenizer, device='cuda',
                                   system_prompt=None, tools=None, system_role='system', user_role='user')
        if ret != 0:
            print('Load model failed!')
            exit(ret)

        ## 执行量化算法
        dataset = args.cali_dataset
        model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm='grq', dataset=dataset)

        model = model.cpu()

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
        os.makedirs(export_dirname)

    # Export llm to onnx
    ## 导出onnx时，要求attention实现方式必须为eager
    model.config._attn_implementation = "eager"
    model = model.float()  # bf16 -> fp32
    spark2_5_to_onnx(model, args)

    # Export LLM configuration
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, PROMPT)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.embedding.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')
