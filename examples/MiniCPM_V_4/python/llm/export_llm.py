import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_minicpm_3o_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, make_dataset_for_minicpm_v
from transformers import AutoConfig, AutoModel
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MiniCPM_V_4.modeling_minicpmv import MiniCPMV

prompt = "RKLLM"
img_path = "../../data/demo.jpg"
image = Image.open(img_path).convert('RGB')
message = [{'role': 'user', 'content': [image, prompt]}]

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export openbmb/MiniCPM-V-4 llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default='openbmb/MiniCPM-V-4')
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/MiniCPM-V-4-llm.onnx")
    parser.add_argument("--max_position_embeddings", type=int, help="max position embeddings", required=False, default=8192)
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    ori_model_path = args.model_path

    
    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    config.max_position_embeddings = args.max_position_embeddings
    kwargs['config'] = config
    model = MiniCPMV.from_pretrained(args.model_path, **kwargs)
        
    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.llm, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        model.llm = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)

        model = model.cpu()
        
    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_llm_dirname):
        os.makedirs(export_llm_dirname)

    # Export llm to onnx
    causal_llm_to_onnx(model.llm, args)

    # Export LLM configuration 
    export_minicpm_3o_llm_config(ori_model_path, args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', message, prompt)

    # Export tokenizer
    os.system("cp ../MiniCPM_V_4/tokenizer_config.json {}".format(ori_model_path))
    export_tokenizer(ori_model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.llm.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')