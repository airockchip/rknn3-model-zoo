import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

import torch
from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import export_qwen2_omni_audio
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig, Qwen2_5OmniForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen2.5-Omni audio configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen2.5-Omni-3B")
    parser.add_argument("--export_audio_path", type=str, help="export audio onnx model path", required=False, default="../../model/audio/Qwen2.5-Omni-3B-audio.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    
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
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.model_path, **kwargs).thinker.eval()

    export_audio_dirname = os.path.dirname(args.export_audio_path)
    if not os.path.exists(export_audio_dirname):
        os.makedirs(export_audio_dirname)

    # export audio model
    export_qwen2_omni_audio(model.audio_tower, args)