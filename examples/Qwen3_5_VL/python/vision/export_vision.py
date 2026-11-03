import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from py_utils.export_vision_helper import export_qwen2_vl_vision
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration
import torch
import torch.nn as nn
from lora_node_map import build_hf_onnx_mapping



class Qwen3_5VisionExportWrapper(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        outputs = self.visual(
            pixel_values,
            grid_thw=grid_thw,
            return_dict=True,
        )
        return outputs.pooler_output

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen3.5-4B vision configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/Qwen3.5-4B-vision.onnx")
    parser.add_argument(
        "--hf_onnx_map",
        type=str,
        default=None,
        help="HF weight to ONNX initializer/op mapping; defaults to <export_vision_path>.hf_onnx_map.json",
    )
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--img_h", type=int, help="Input image size (e.g., 384, 416, 448). Must be a multiple of 32.", required=False, default=384)
    parser.add_argument("--img_w", type=int, help="Input image size (e.g., 384, 416, 448). Must be a multiple of 32.", required=False, default=384)
    
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    kwargs['config'] = config
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=torch.float32,
        # Use float32 for this RKNN export path. If restricting the loading dtype,
        # set "use_flash_attn" to false in config.json.
        low_cpu_mem_usage=True, _attn_implementation="eager",
        trust_remote_code=True)
    
    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        # Initialize the quantizer.
        QuantTool = RKQuantizer(verbose=True)
        
        # Load the vision model into the quantizer.
        ret = QuantTool.load_model(model=model.model.visual, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        # Quantize the model with the calibration dataset.
        dataset = args.cali_dataset
        model.model.visual = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)
        

        model = model.cpu()

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if export_vision_dirname and not os.path.exists(export_vision_dirname):
        print(f"create export_vision_dirname: {export_vision_dirname}")
        os.makedirs(export_vision_dirname)

    # Default to patch size 16 unless the model configuration specifies a value.
    patch_size = 16
    if getattr(config, "vision_config", None) is not None and config.vision_config.patch_size is not None:
        patch_size = config.vision_config.patch_size
    else:
        print(f"vision_config.patch_size is None, use default value {patch_size}")

    vision_wrapper = Qwen3_5VisionExportWrapper(model.model.visual).eval()

    # export vision model
    export_qwen2_vl_vision(vision_wrapper, args, patch_size) # Include grid_thw for the input image patches.

    # The vision ONNX contains only model.visual.  Prefix the wrapper-local
    # state-dict keys so they match the full HF model key space used by the
    # LoRA adapter (thinker.visual is resolved by lora_node_map.py).
    vision_state_dict = {
        f"model.visual.{name}": value.detach().cpu()
        for name, value in model.model.visual.state_dict().items()
    }
    hf_onnx_map_path = args.hf_onnx_map or (os.path.splitext(args.export_vision_path)[0] + '.hf_onnx_map.json')
    build_hf_onnx_mapping(args.export_vision_path, vision_state_dict, hf_onnx_map_path)
