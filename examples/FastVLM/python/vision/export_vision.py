import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import export_fastvlm_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llava.model import *  # 增加logits_to_keep输入


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export FastVLM vision configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default='../../llava-fastvithd_1.5b_stage3') # 模型下载链接：https://ml-site.cdn-apple.com/datasets/fastvlm/llava-fastvithd_1.5b_stage3.zip
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/FastVLM-vision.onnx")
    parser.add_argument("--img_size", type=int, help="vision model input image size", required=False, default=512)
    args = parser.parse_args()

    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    kwargs['config'] = config
    model = LlavaQwen2ForCausalLM.from_pretrained(args.model_path, **kwargs)

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if not os.path.exists(export_vision_dirname):
        os.makedirs(export_vision_dirname)
    
    # export vision model
    export_fastvlm_vision(model, args)