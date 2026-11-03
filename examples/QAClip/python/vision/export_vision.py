import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_vision_helper import save_config
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import ChineseCLIPModel


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


class ChineseCLIPVisionWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.FloatTensor):
        vision_outputs = self.model.vision_model(pixel_values=pixel_values, return_dict=True)
        pooled = vision_outputs[1]

        image_feats = self.model.visual_projection(pooled)
        image_feats = image_feats / _get_vector_norm(image_feats)
        return image_feats


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Chinese-CLIP vision model to ONNX for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=False)
    parser.add_argument("--model_path", type=str, help="model path or name", required=False,
                       default="../../models/QA-CLIP-ViT-L-14")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False,
                       default="./chinese_clip_vision.onnx")
    parser.add_argument("--img_h", type=int, required=False, default=224)
    parser.add_argument("--img_w", type=int, required=False, default=224)
    args = parser.parse_args()

    kwargs = {
        'trust_remote_code': True,
    }

    config = AutoConfig.from_pretrained(args.model_path, **kwargs)

    model = ChineseCLIPModel.from_pretrained(args.model_path, torch_dtype=torch.float32, trust_remote_code=True, attn_implementation="eager")

    export_dir = os.path.dirname(args.export_vision_path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir)

    wrapper = ChineseCLIPVisionWrapper(model)
    wrapper.eval()

    # Save simple vision config for runtime (grid sizes)
    save_config(os.path.join(export_dir, 'vision_config.json'), args.img_h, args.img_w, patch_size=getattr(model.vision_model.config, 'patch_size', 16))

    # fake input
    fake_input = torch.randn(1, 3, args.img_h, args.img_w)
    # import numpy as np
    # np.save(os.path.join(export_dir, 'pixel_values.npy'), fake_input.numpy())

    print("Exporting Chinese-CLIP vision model to ONNX...")
    torch.onnx.export(
        wrapper,
        (fake_input,),
        args.export_vision_path,
        input_names=['pixel'],
        output_names=['image_features'],
        # dynamic_axes={'pixel': {2: 'height', 3: 'width'}},
        opset_version=17,
    )

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_dir)

    print(f"Chinese-CLIP vision model exported successfully to {args.export_vision_path}")
