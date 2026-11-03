import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import export_qwen2_vl_vision
from transformers import AutoConfig
from transformers.utils.versions import require_version

require_version(
    "transformers<4.52.0",
    "This code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3"
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modeling_gme_qwen2vl import GmeQwen2VL

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export GmeQwen2VL vision configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="iic/gme-Qwen2-VL-2B-Instruct")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/GmeQwen2VL-vision.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--img_h", type=int, help="Input image size (e.g., 224, 392, 448). Must be a multiple of 28.", required=False, default=448)
    parser.add_argument("--img_w", type=int, help="Input image size (e.g., 224, 392, 448). Must be a multiple of 28.", required=False, default=448)
    
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    kwargs['config'] = config
    model = GmeQwen2VL.from_pretrained(args.model_path, **kwargs)
        
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
    export_qwen2_vl_vision(model.visual, args) # 添加grid_thw对输入图片的patch