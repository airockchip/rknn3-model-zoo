import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.tools import clear_llm_external_weight_in_dir
from py_utils.export_vision_helper import (
    export_gemma4_audio,
    export_gemma4_audio_streaming,
)
from transformers import AutoModelForCausalLM, AutoConfig

# Get the custom module path
custom_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'gemma4'))
print(f"Custom gemma4 module path: {custom_path}")

# Load our custom modeling_gemma4 module using importlib
import importlib.util
modeling_spec = importlib.util.spec_from_file_location(
    'transformers.models.gemma4.modeling_gemma4',
    os.path.join(custom_path, 'modeling_gemma4.py')
)
modeling_module = importlib.util.module_from_spec(modeling_spec)
sys.modules['transformers.models.gemma4.modeling_gemma4'] = modeling_module
modeling_spec.loader.exec_module(modeling_module)
print(f"Loaded custom modeling_gemma4 from: {modeling_module.__file__}")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export google/gemma-4 audio configuration and onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="google/gemma-4-E2B-it")
    parser.add_argument("--export_audio_path", type=str, help="export audio onnx model path", required=False, default="../../model/audio/gemma-4-e2b-it-audio.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--streaming", action='store_true', help="Export the stateful Audio streaming graph")
    
    args = parser.parse_args()

    if "E4B" in args.model_path:
        args.export_audio_path = "../../model/audio/gemma-4-e4b-it-audio.onnx"

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True,
        'torch_dtype': torch.float32,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    if args.load_weight:
        kwargs['config'] = config
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs).eval()
    else:
        kwargs.pop('trust_remote_code', True)
        model = AutoModelForCausalLM._from_config(config, **kwargs).eval()

    export_audio_dirname = os.path.dirname(args.export_audio_path)
    if not os.path.exists(export_audio_dirname):
        os.makedirs(export_audio_dirname)

    # export audio model
    if args.streaming:
        export_gemma4_audio_streaming(model, args)
    else:
        export_gemma4_audio(model, args)

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_audio_dirname)