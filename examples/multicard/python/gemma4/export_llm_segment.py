import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segment_planner import add_plan_arguments, write_plan_json

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_quantize_dataset
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from export_onnx_segment import causal_gemma4_unified_to_onnx

prompt = "RKLLM"
chat_context = {
    "messages":[
        {"role": "user", "content": prompt}
    ],
    "add_generation_prompt": True,
}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export google/gemma-4 llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="google/gemma-4-12B-it")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm_gemma/gemma-4-12b-it.onnx")
    parser.add_argument("--multi_segment", action='store_true', help="Export model as multiple ONNX segments split by transformer blocks")
    parser.add_argument("--num_segments", type=int, default=0, help="Number of segments to split into (0=auto, default splits ~8 layers per segment)")
    parser.add_argument("--segment_layer_counts", "--segment-layer-counts", type=int, nargs="+", default=None, help="manual block count per segment; counts must sum to the total layer count and override --num_segments")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='../../../../datasets/llm_quant.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--stop_after_segment", type=int, default=None, help="stop after this zero-based segment; useful for block-level validation")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    add_plan_arguments(parser)
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    fakeq_state = None
    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True
            )

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        QuantTool = RKQuantizer(verbose=True)
        
        ret = QuantTool.load_model(model=model.model.language_model, tokenizer=tokenizer, device='cuda',
                                   system_prompt=None, tools=None, system_role='system', user_role='user')
        if ret != 0:
            print('Load model failed!')
            sys.exit(ret)
        
        dataset = args.cali_dataset
        model.model.language_model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm='grq', dataset=dataset)
        
        model = model.cpu()
        fakeq_state = model.state_dict()
        del model
        torch.cuda.empty_cache()

    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    kwargs['config'] = config
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    if fakeq_state is not None:
        model.load_state_dict(
                fakeq_state,
                strict=False
            )

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
            os.makedirs(export_dirname)

    # Export llm to onnx
    causal_gemma4_unified_to_onnx(model, args)

    # Export LLM configuration 
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.language_model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')
    if args.multi_segment:
        write_plan_json(args)
