import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

import sys
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import save_config
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig

from model.modeling_locateanything import LocateAnythingForConditionalGeneration
from model.modeling_vit import patch_merger


DEFAULT_IMG_H = 448
DEFAULT_IMG_W = 448


class LocateAnythingVisionWrapper(torch.nn.Module):
    def __init__(self, vision_model, mlp1, grid_hws):
        super().__init__()
        self.vision_model = vision_model
        self.mlp1 = mlp1
        # Store as plain attribute (NOT registered buffer) so that torch.export
        # treats grid_hws as a constant tensor and bakes its concrete values
        # into the graph.  When registered as a buffer, torch.export treats it
        # as a symbolic input, causing grid_hws.tolist() to return symbolic
        # ints (SymInt) which trigger GuardOnDataDependentSymNode errors in
        # asserts and comparisons downstream.
        self.grid_hws = grid_hws

    def forward(self, image):
        # The HF processor originally patchifies the image on CPU and feeds
        # [num_patches, 3, patch_h, patch_w] into patch_embed.proj.  For RKNN
        # deployment we keep the same Conv2D in the exported graph and feed the
        # image directly as [1, 3, H, W].  Standalone ONNX expects normalized
        # float input; export_rknn.py configures uint8 RGB input with mean/std.
        hidden_states = self.vision_model.patch_embed.proj(image)
        hidden_states = hidden_states.flatten(2).transpose(1, 2).reshape(-1, hidden_states.size(1))
        hidden_states = self.vision_model.patch_embed.pos_emb(hidden_states, self.grid_hws)
        hidden_states = self.vision_model.encoder(hidden_states, self.grid_hws)
        vit_embeds = patch_merger(
            hidden_states,
            self.grid_hws,
            merge_kernel_size=self.vision_model.merge_kernel_size,
        )
        if isinstance(vit_embeds, list):
            vit_embeds = torch.cat(vit_embeds, dim=0)
        return self.mlp1(vit_embeds)


def _prepare_export_config(config):
    if not hasattr(config, 'rope_parameters'):
        config.rope_parameters = None
    if not hasattr(config.vision_config, 'rope_parameters'):
        config.vision_config.rope_parameters = None
    update_config(config, ['use_cache'], False)
    update_config(config, ['_attn_implementation_autoset'], False)
    config._attn_implementation = 'eager'
    config.text_config._attn_implementation = 'eager'
    config.vision_config._attn_implementation = 'eager'
    return config


def export_locateanything_vision(model, args, patch_size):
    grid_h = args.img_h // patch_size
    grid_w = args.img_w // patch_size
    image = torch.randn(1, 3, args.img_h, args.img_w, dtype=torch.float32)
    grid_hws = torch.tensor([[grid_h, grid_w]], dtype=torch.int64)

    wrapper = LocateAnythingVisionWrapper(model.vision_model, model.mlp1, grid_hws).eval().float()
    save_config("vision_config.json", args.img_h, args.img_w, patch_size)

    print(f"image: {tuple(image.shape)}, fixed grid_hws: {grid_hws.tolist()}")
    with torch.no_grad():
        vision_output = wrapper(image)
    print(f"vision_output: {tuple(vision_output.shape)}")

    with open("vision_config.json", 'r', encoding='utf-8') as f:
        vision_cfg = json.load(f)
    vision_cfg["input_type"] = "image"
    vision_cfg["input_shape"] = [1, 3, args.img_h, args.img_w]
    vision_cfg["merge_kernel_size"] = list(model.config.vision_config.merge_kernel_size)
    vision_cfg["num_vision_tokens"] = int(vision_output.shape[0])
    with open("vision_config.json", 'w', encoding='utf-8') as f:
        json.dump(vision_cfg, f, indent=4)

    torch.onnx.export(
        wrapper,
        (image,),
        args.export_vision_path,
        input_names=['image'],
        output_names=['vision_output'],
        opset_version=19,
        # grid_hws-derived sequence boundaries are concrete for this fixed-shape
        # export, but the dynamo exporter treats them as data-dependent SymInts.
        dynamo=False,
    )
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export LocateAnything-3B vision to ONNX for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument(
        "--model_path",
        type=str,
        help="model path or name",
        required=False,
        default="../../models/LocateAnything-3B/hf/",
    )
    parser.add_argument(
        "--export_vision_path",
        type=str,
        help="export vision onnx model path",
        required=False,
        default="../../model/vision/locateanything-3b-vision_pixel.onnx",
    )
    parser.add_argument("--img_h", type=int, help="reference image height", required=False, default=DEFAULT_IMG_H)
    parser.add_argument("--img_w", type=int, help="reference image width", required=False, default=DEFAULT_IMG_W)
    parser.add_argument("--quant", action='store_true', help="Whether use AWQ and GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--quantized_dtype", type=str, required=False, default="w8a16", help="quantized dtype, e.g. w4a16/w8a16")
    parser.add_argument("--quantized_method", type=str, required=False, default="group32", help="quantized method, e.g. group32")
    parser.add_argument("--quantized_algorithm", type=str, required=False, default="grq", help="quantized algorithm, e.g. grq/normal")
    args = parser.parse_args()

    kwargs = {'trust_remote_code': True}
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    _prepare_export_config(config)

    if args.load_weight:
        kwargs['config'] = config
        model = LocateAnythingForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
            **kwargs,
        ).eval()
    else:
        model = LocateAnythingForConditionalGeneration(config)

    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        print("--> quantizing LocateAnything vision model with RKQuantizer")
        QuantTool = RKQuantizer(verbose=True)

        ret = QuantTool.load_model(model=model.vision_model, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)

        dataset = args.cali_dataset
        model.vision_model = QuantTool.quantize(
            quantized_dtype=args.quantized_dtype,
            quantized_method=args.quantized_method,
            quantized_algorithm=args.quantized_algorithm,
            dataset=dataset,
        )

        model = model.cpu()
    elif args.quant and not torch.cuda.is_available():
        print("[warn] --quant specified but CUDA is not available, skip quantization")

    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if export_vision_dirname and not os.path.exists(export_vision_dirname):
        os.makedirs(export_vision_dirname)

    patch_size = config.vision_config.patch_size
    if args.img_h % patch_size != 0 or args.img_w % patch_size != 0:
        raise ValueError(f"img_h/img_w must be divisible by patch_size={patch_size}, got {args.img_h}x{args.img_w}")
    export_locateanything_vision(model, args, patch_size)

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_vision_dirname)
