import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

try:
    import flash_attn  # noqa: F401
except ImportError:
    import importlib.machinery, sys as _sys, types as _types
    _stub = _types.ModuleType("flash_attn")
    _stub.flash_attn_varlen_func = None
    _stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    _stub.__version__ = "2.7.4"
    _sys.modules["flash_attn"] = _stub

import sys, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import save_config, check_torch_version
from transformers import AutoConfig, AutoModelForCausalLM


def export_dots_mocr_vision(model, args, patch_size=14):
    model.float()

    class dots_mocr_vision(torch.nn.Module):
        def __init__(self, vl_m, in_h, in_w, patch_size):
            super(dots_mocr_vision, self).__init__()
            self.vl_m = vl_m

            self.grid_t = 1
            self.merge_size = 2
            self.channel = 3
            self.temporal_patch_size = 1
            self.patch_size = patch_size

            align_size = self.merge_size * self.patch_size  # 28
            align_h = (in_h + align_size - 1) // align_size * align_size
            align_w = (in_w + align_size - 1) // align_size * align_size
            self.in_h = align_h
            self.in_w = align_w

        def forward(self, pixel, grid_thw):
            h = pixel.shape[2]
            w = pixel.shape[3]
            patches = pixel.reshape(
                self.grid_t,
                self.temporal_patch_size,
                self.channel,
                h // self.merge_size // self.patch_size,
                self.merge_size,
                self.patch_size,
                w // self.merge_size // self.patch_size,
                self.merge_size,
                self.patch_size,
            )
            patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flatten_patches = patches.reshape(
                self.grid_t * h // self.patch_size * w // self.patch_size,
                self.channel * self.temporal_patch_size * self.patch_size * self.patch_size,
            )
            return self.vl_m(flatten_patches, grid_thw, bf16=False)

    in_h = args.img_h
    in_w = args.img_w
    model = dots_mocr_vision(model, in_h, in_w, patch_size)
    save_config("vision_config.json", args.img_h, args.img_w, model.patch_size)
    in_h = model.in_h
    in_w = model.in_w

    fake_input = torch.randn(1, 3, in_h, in_w)
    grid_thw = torch.tensor([[1, in_h // model.patch_size, in_w // model.patch_size]], dtype=torch.int64)

    model(fake_input, grid_thw)

    if check_torch_version():
        torch.onnx.export(
            model,
            (fake_input, grid_thw),
            args.export_vision_path,
            input_names=['pixel', 'grid_thw'],
            dynamic_axes={'pixel': {2: 'height', 3: 'width'}},
            opset_version=17,
            dynamo=False,
        )
    else:
        torch.onnx.export(
            model,
            (fake_input, grid_thw),
            args.export_vision_path,
            input_names=['pixel', 'grid_thw'],
            dynamic_axes={'pixel': {2: 'height', 3: 'width'}},
            opset_version=17,
        )
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export dots.mocr vision configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="rednote-hilab/dots.mocr")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/dots-mocr-vision.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--img_h", type=int, help="Input image size (e.g., 224, 392, 448). Must be a multiple of 28.", required=False, default=448)
    parser.add_argument("--img_w", type=int, help="Input image size (e.g., 224, 392, 448). Must be a multiple of 28.", required=False, default=448)

    args = parser.parse_args()

    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    if hasattr(config, 'vision_config'):
        try:
            config.vision_config.attn_implementation = 'eager'
        except Exception:
            pass
    kwargs['config'] = config
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)

    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)

        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.vision_tower, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)

        ## 执行量化算法
        dataset = args.cali_dataset
        model.vision_tower = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)

        model = model.cpu()

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if not os.path.exists(export_vision_dirname):
        os.makedirs(export_vision_dirname)

    # export vision model
    export_dots_mocr_vision(model.vision_tower, args)
