import argparse
import json
import os
import sys

import torch
from transformers import AutoConfig, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.export_llm_helper import update_config

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from make_calidata import UnlimitedOCRVisionStack


DEFAULT_MODEL_PATH = "baidu/Unlimited-OCR"
DEFAULT_EXPORT_PATH = "../../model/vision/Unlimited-OCR-vision.onnx"
DEFAULT_EXPORT_SAM_PATH = "../../model/vision/Unlimited-OCR-sam.onnx"


class UnlimitedOCRSam(torch.nn.Module):
    def __init__(self, vision_stack):
        super().__init__()
        self.sam_model = vision_stack.sam_model

    def forward(self, pixel_values):
        return self.sam_model(pixel_values)


class UnlimitedOCRVisionWithoutSam(torch.nn.Module):
    def __init__(self, vision_stack):
        super().__init__()
        self.vision_model = vision_stack.vision_model
        self.projector = vision_stack.projector

    def _encode_one(self, sam_features):
        pixel_values = sam_features[:, :3, :1, :1]
        clip_features = self.vision_model(pixel_values, sam_features)
        features = torch.cat(
            (clip_features[:, 1:], sam_features.flatten(2).permute(0, 2, 1)),
            dim=-1,
        )
        return self.projector(features)


class UnlimitedOCRBaseVisionWithoutSam(UnlimitedOCRVisionWithoutSam):
    def __init__(self, vision_stack):
        super().__init__(vision_stack)
        self.image_newline = vision_stack.image_newline
        self.view_seperator = vision_stack.view_seperator

    def forward(self, image_ori_sam_features):
        global_features = self._encode_one(image_ori_sam_features)
        _, hw, n_dim = global_features.shape
        h = w = int(hw**0.5)
        global_features = global_features.view(h, w, n_dim)
        global_features = torch.cat(
            [
                global_features,
                self.image_newline[None, None, :].expand(h, 1, n_dim),
            ],
            dim=1,
        ).view(-1, n_dim)
        return torch.cat(
            [global_features, self.view_seperator[None, :]],
            dim=0,
        )


class UnlimitedOCRGundamVisionWithoutSam(UnlimitedOCRVisionWithoutSam):
    def __init__(self, vision_stack, crop_width_num, crop_height_num):
        super().__init__(vision_stack)
        self.image_newline = vision_stack.image_newline
        self.view_seperator = vision_stack.view_seperator
        self.crop_width_num = int(crop_width_num)
        self.crop_height_num = int(crop_height_num)

    def forward(
        self,
        patches_sam_features,
        image_ori_sam_features,
    ):
        local_features = self._encode_one(patches_sam_features)
        global_features = self._encode_one(image_ori_sam_features)

        _, hw, n_dim = global_features.shape
        h = w = int(hw**0.5)
        global_features = global_features.view(h, w, n_dim)
        global_features = torch.cat(
            [
                global_features,
                self.image_newline[None, None, :].expand(h, 1, n_dim),
            ],
            dim=1,
        ).view(-1, n_dim)

        _, hw2, n_dim2 = local_features.shape
        h2 = w2 = int(hw2**0.5)
        local_features = (
            local_features.view(
                self.crop_height_num,
                self.crop_width_num,
                h2,
                w2,
                n_dim2,
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(self.crop_height_num * h2, self.crop_width_num * w2, n_dim2)
        )
        local_features = torch.cat(
            [
                local_features,
                self.image_newline[None, None, :].expand(
                    self.crop_height_num * h2, 1, n_dim2
                ),
            ],
            dim=1,
        ).view(-1, n_dim2)

        return torch.cat(
            [local_features, global_features, self.view_seperator[None, :]],
            dim=0,
        )


def default_image_size(mode):
    return 1024 if mode == "base" else 640


def save_vision_config(args, output_path):
    config = {
        "mode": args.mode,
        "base_size": args.base_size,
        "image_size": args.image_size,
        "crop_mode": args.mode == "gundam",
        "crop_width_num": args.crop_width_num if args.mode == "gundam" else 1,
        "crop_height_num": args.crop_height_num if args.mode == "gundam" else 1,
        "patch_size": 16,
        "downsample_ratio": 4,
        "hidden_size": 1280,
        "sam_hidden_size": 1024,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"Config saved to {os.path.abspath(output_path)}")


def build_sam_export_inputs(vision_stack, args):
    sam_model = UnlimitedOCRSam(vision_stack).eval()
    if args.mode == "base":
        fake_image_ori = torch.randn(1, 3, args.image_size, args.image_size)
        with torch.inference_mode():
            fake_image_ori_sam_features = sam_model(fake_image_ori)
        sam_inputs = (fake_image_ori,)
        sam_features = (fake_image_ori_sam_features,)
    else:
        crop_num = args.crop_width_num * args.crop_height_num
        fake_patches = torch.randn(crop_num, 3, args.image_size, args.image_size)
        fake_image_ori = torch.randn(1, 3, args.base_size, args.base_size)
        with torch.inference_mode():
            fake_patches_sam_features = sam_model(fake_patches)
            fake_image_ori_sam_features = sam_model(fake_image_ori)
        sam_inputs = (fake_patches,)
        sam_features = (fake_patches_sam_features, fake_image_ori_sam_features)
    return sam_model, sam_inputs, sam_features


def build_vision_export_model(vision_stack, sam_features, args):
    if args.mode == "base":
        fake_image_ori_sam_features = sam_features[0]
        export_model = UnlimitedOCRBaseVisionWithoutSam(vision_stack).eval()
        inputs = (fake_image_ori_sam_features,)
        input_names = ["image_ori_sam_features"]
    else:
        fake_patches_sam_features, fake_image_ori_sam_features = sam_features
        export_model = UnlimitedOCRGundamVisionWithoutSam(
            vision_stack,
            crop_width_num=args.crop_width_num,
            crop_height_num=args.crop_height_num,
        ).eval()
        inputs = (
            fake_patches_sam_features,
            fake_image_ori_sam_features,
        )
        input_names = [
            "patches_sam_features",
            "image_ori_sam_features",
        ]
    return export_model, inputs, input_names


def maybe_quantize(vision_stack, args):
    if not args.quant:
        return vision_stack
    if not torch.cuda.is_available():
        print("CUDA is unavailable, skip GRQ quantization.")
        return vision_stack

    from rknn.quantization.api import RKQuantizer

    quant_tool = RKQuantizer(verbose=True)
    ret = quant_tool.load_model(model=vision_stack, tokenizer=None, device="cuda")
    if ret != 0:
        raise RuntimeError(f"Load model failed, ret={ret}")

    quantized = quant_tool.quantize(
        quantized_dtype="w4a16",
        quantized_method="group32",
        quantized_algorithm="grq",
        dataset=args.cali_dataset,
    )
    return quantized.cpu().eval()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Unlimited-OCR vision stack ONNX model for RKNN"
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--export_vision_path", type=str, default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--export_sam_path", type=str, default=DEFAULT_EXPORT_SAM_PATH)
    parser.add_argument("--vision_config_path", type=str, default="vision_config.json")
    parser.add_argument("--mode", choices=["base", "gundam"], default="base")
    parser.add_argument("--base_size", type=int, default=1024)
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="defaults to 1024 in base mode and 640 in gundam mode",
    )
    parser.add_argument(
        "--crop_width_num",
        type=int,
        default=1,
        help="static local-crop grid width for gundam mode",
    )
    parser.add_argument(
        "--crop_height_num",
        type=int,
        default=1,
        help="static local-crop grid height for gundam mode",
    )
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument(
        "--quant",
        action="store_true",
        help="Use GRQ w4a16 quantization for the CLIP/vision branch only. SAM ONNX stays unquantized.",
    )
    parser.add_argument(
        "--cali_dataset",
        default="../../data/vision/model_inputs.json",
        help="calibration dataset generated by make_calidata.py",
    )
    args = parser.parse_args()
    if args.image_size is None:
        args.image_size = default_image_size(args.mode)
    if args.mode == "gundam" and args.crop_width_num * args.crop_height_num <= 1:
        print("Warning: gundam mode is using a 1x1 crop grid; set --crop_width_num/--crop_height_num for large-image export.")
    return args


def main():
    args = parse_args()
    export_paths = [args.export_vision_path, args.export_sam_path]
    for export_path in export_paths:
        export_dir = os.path.dirname(export_path)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)

    kwargs = {"trust_remote_code": True}
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ["use_cache"], False)
    update_config(config, ["_attn_implementation"], "eager")

    model = AutoModel.from_pretrained(
        args.model_path,
        config=config,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    ).eval()

    vision_stack = UnlimitedOCRVisionStack(model.model).eval()
    sam_model, sam_inputs, sam_features = build_sam_export_inputs(vision_stack, args)
    with torch.inference_mode():
        sam_output = sam_model(*sam_inputs)
    print(f"sam output ({args.export_sam_path}):", tuple(sam_output.shape))
    sam_dynamic_axes = None
    if args.mode == "gundam":
        sam_dynamic_axes = {
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "sam_features": {0: "batch", 2: "feature_height", 3: "feature_width"},
        }
    torch.onnx.export(
        sam_model,
        sam_inputs,
        args.export_sam_path,
        input_names=["pixel_values"],
        output_names=["sam_features"],
        dynamic_axes=sam_dynamic_axes,
        opset_version=args.opset,
    )
    print(f"Exported SAM to {os.path.abspath(args.export_sam_path)}")

    vision_stack = maybe_quantize(vision_stack, args)
    export_model, inputs, input_names = build_vision_export_model(
        vision_stack, sam_features, args
    )
    with torch.inference_mode():
        output = export_model(*inputs)
    print("vision output:", tuple(output.shape))

    save_vision_config(args, args.vision_config_path)

    torch.onnx.export(
        export_model,
        inputs,
        args.export_vision_path,
        input_names=input_names,
        output_names=["image_embeds"],
        opset_version=args.opset,
    )
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")


if __name__ == "__main__":
    main()
