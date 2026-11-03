import os, torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import export_internvl_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig, AutoModel


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export OpenGVLab/InternVL3.5-2B vision configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default='OpenGVLab/InternVL3.5-2B') # 模型下载链接：https://huggingface.co/OpenGVLab/InternVL3_5-2B/tree/main
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="./vision/InternVL3.5-2B-vision.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use AWQ and GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    kwargs = {   
        'device_map': "cpu",
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    update_config(config, ['_attn_implementation'], 'eager')
    # 关闭 InternViT 的 flash_attn, 量化阶段改用 naive attention, 避免 dtype 冲突。
    update_config(config, ['use_flash_attn'], False)
    model = AutoModel.from_pretrained(args.model_path, **kwargs).eval()
    # Also disable custom InternViT attention flags before export.
    for _name, _mod in model.vision_model.named_modules():
        if hasattr(_mod, 'use_flash_attn'):
            _mod.use_flash_attn = False

    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.vision_model, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        model.vision_model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)

        model = model.cpu()

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if not os.path.exists(export_vision_dirname):
        os.makedirs(export_vision_dirname)
    
    # export vision model
    export_internvl_vision(model, args)
