import argparse
import os
import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from transformers import AutoConfig, Qwen3_5ForConditionalGeneration


class VisionExportWrapper(nn.Module):
    """Export the vision tower with an explicit image or video input contract.

    Image mode accepts one frame and duplicates it inside the ONNX graph to
    satisfy the temporal patch size. Video mode accepts two adjacent frames.
    """

    def __init__(self, visual, patch_size, mode, image_size=384):
        super().__init__()
        self.visual = visual
        self.patch_size = patch_size
        self.mode = mode
        self.image_size = image_size
        self.temporal_patch_size = 2
        self.merge_size = 2

    def forward(self, pixel_values):
        h = self.image_size
        w = self.image_size
        if self.mode == "image":
            pixel_values = torch.cat((pixel_values, pixel_values), dim=0)
        patches = pixel_values.reshape(
            1,
            self.temporal_patch_size,
            3,
            h // self.merge_size // self.patch_size,
            self.merge_size,
            self.patch_size,
            w // self.merge_size // self.patch_size,
            self.merge_size,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            h // self.patch_size * w // self.patch_size,
            3 * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        grid_thw = torch.tensor(
            [[1, h // self.patch_size, w // self.patch_size]],
            dtype=torch.int64,
            device=pixel_values.device,
        )
        outputs = self.visual(flatten_patches, grid_thw=grid_thw, return_dict=True)
        return outputs.pooler_output


def parse_args():
    parser = argparse.ArgumentParser(description="Export the WeMM-Embedding-2B vision tower to ONNX")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Local model directory or Hugging Face/ModelScope model ID",
    )
    parser.add_argument(
        "--mode",
        default="image",
        choices=("image", "video", "all"),
        help="Model to export: image (default), video, or all",
    )
    parser.add_argument("--output_path", required=True, help="Output ONNX file")
    parser.add_argument(
        "--modelscope",
        action="store_true",
        help="Download non-local model IDs from ModelScope instead of Hugging Face",
    )
    return parser.parse_args()


def resolve_model_path(model_path, use_modelscope):
    if os.path.isdir(model_path):
        return model_path
    if use_modelscope:
        from modelscope import snapshot_download

        return snapshot_download(model_path)
    return model_path


def get_output_paths(output_path, mode):
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("--output_path must end with .onnx")
    if mode == "all":
        return {
            "image": output_path.with_name(f"{output_path.stem}-image.onnx"),
            "video": output_path.with_name(f"{output_path.stem}-video.onnx"),
        }
    return {mode: output_path}


def apply_grq(visual, work_dir):
    if not torch.cuda.is_available():
        raise RuntimeError("GRQ requires CUDA")

    from rknn.quantization.api import RKQuantizer
    from prepare_grq_dataset import prepare_dataset

    dataset_dir = (
        Path(__file__).resolve().parents[4]
        / "datasets"
        / "imagenet"
        / "ILSVRC2012_img_val_samples"
    )
    source_paths = sorted(dataset_dir.glob("*.JPEG"))
    if len(source_paths) < 2:
        raise FileNotFoundError(
            f"Expected ImageNet calibration images in {dataset_dir}"
        )

    with tempfile.TemporaryDirectory(prefix="wemm-grq-", dir=work_dir) as temp_dir:
        dataset_path = prepare_dataset(source_paths, temp_dir, samples=20)
        quantizer = RKQuantizer(verbose=True)
        ret = quantizer.load_model(model=visual, tokenizer=None, device="cuda")
        if ret != 0:
            raise RuntimeError(f"RKQuantizer.load_model failed: {ret}")
        return quantizer.quantize(
            quantized_dtype="w4a16",
            quantized_method="group32",
            quantized_algorithm="grq",
            dataset=str(dataset_path),
        ).cpu().eval()


def export_mode(visual, patch_size, mode, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = VisionExportWrapper(visual, patch_size, mode).eval()
    input_frames = 1 if mode == "image" else 2

    with tempfile.TemporaryDirectory(prefix="wemm-vision-", dir=output_path.parent) as temp_dir:
        temp_path = Path(temp_dir) / output_path.name
        sample_frames = torch.zeros(input_frames, 3, 384, 384)
        torch.onnx.export(
            wrapper,
            (sample_frames,),
            str(temp_path),
            input_names=["pixel"],
            output_names=["vision_embedding"],
            opset_version=17,
        )
        onnx.checker.check_model(str(temp_path), full_check=False)
        for generated_file in Path(temp_dir).iterdir():
            os.replace(generated_file, output_path.parent / generated_file.name)
    print(f"ONNX export completed ({mode}): {output_path}")


def main():
    args = parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_paths = get_output_paths(output_path, args.mode)
    model_path = resolve_model_path(args.model_path, args.modelscope)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval()
    visual = apply_grq(model.model.visual, output_path.parent)
    for mode, mode_output_path in output_paths.items():
        export_mode(
            visual,
            config.vision_config.patch_size,
            mode,
            mode_output_path,
        )


if __name__ == "__main__":
    main()
