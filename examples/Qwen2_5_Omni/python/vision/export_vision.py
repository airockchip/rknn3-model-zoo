import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import torch
from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import export_qwen2_omni_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen2.5-Omni vision configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/Qwen2.5-Omni-3B-vision.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--img_size", type=int, help="Input image size (e.g., 224, 384, 448). Must be a multiple of 28.", required=False, default=392)
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True,
        'torch_dtype': torch.float32,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    kwargs['config'] = config
    kwargs['low_cpu_mem_usage'] = True
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model_path, **kwargs).thinker.eval()

    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.visual, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        model.visual = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)

        model = model.cpu()
        
    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if not os.path.exists(export_vision_dirname):
        os.makedirs(export_vision_dirname)

    # export vision model
    export_qwen2_omni_vision(model.visual, args)