#!/usr/bin/env python3
"""Export Microsoft Mage-VL vision model to ONNX."""

import argparse
import gc
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM


MODEL_PATH = "microsoft/Mage-VL"
ONNX_MODEL = "../../model/vision/mage_vision_patch576_grq_w4a16.onnx"
CALI_DATASET = "./quant_data/model_inputs.json"


class MageImageEncoder(torch.nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.embeddings = visual.embeddings
        self.layernorm_pre = visual.layernorm_pre
        self.layers = visual.encoder.layers
        self.layernorm_post = visual.layernorm_post
        self.merger = visual.merger

        for layer in self.layers:
            layer.self_attn.config._attn_implementation = "eager"

    def forward(self, pixel_values, rotary_pos_emb):
        hidden_states = self.embeddings(pixel_values).unsqueeze(0)
        hidden_states = self.layernorm_pre(hidden_states)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                output_attentions=False,
                cu_seqlens=None,
                max_seqlen=None,
            )[0]

        if self.layernorm_post is not None:
            hidden_states = self.layernorm_post(hidden_states)
        return self.merger(hidden_states, patch_positions=None)


def export_mage_vl_vision(model, args):
    model = MageImageEncoder(model).eval()
    grid_size = int(args.patch_num ** 0.5)
    if grid_size * grid_size != args.patch_num:
        raise ValueError("patch_num must be a square number")
    if grid_size % 2 != 0:
        raise ValueError("the square root of patch_num must be divisible by 2")

    pixel_values = torch.randn(args.patch_num, 3 * 16 * 16, dtype=torch.float32)
    row = np.repeat(np.arange(grid_size), grid_size)
    column = np.tile(np.arange(grid_size), grid_size)
    positions = np.stack([np.zeros_like(row), row, column], axis=1).astype(np.float32)
    rotary_dim = 64
    section_sizes = (8, 12, 12)
    inverse_frequencies = [
        1.0 / (10000.0 ** (np.arange(size, dtype=np.float32) / size))
        for size in section_sizes
    ]
    frequencies = np.concatenate(
        [positions[:, index:index + 1] * inverse_frequencies[index][None, :] for index in range(3)],
        axis=1,
    )
    rotary_pos_emb = torch.from_numpy(
        np.concatenate([frequencies, frequencies], axis=1)[None]
    ).float()
    if rotary_pos_emb.shape[-1] != rotary_dim:
        raise RuntimeError("unexpected rotary embedding dimension")

    with torch.inference_mode():
        reference = model(pixel_values, rotary_pos_emb).detach().float().cpu()

    export_dir = os.path.dirname(args.export_vision_path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    torch.onnx.export(
        model,
        (pixel_values, rotary_pos_emb),
        args.export_vision_path,
        input_names=["pixel_values", "rotary_pos_emb"],
        output_names=["image_embeddings"],
        opset_version=19,
        do_constant_folding=True,
        dynamo=False,
    )

    output_path = os.path.splitext(args.export_vision_path)[0]
    np.save(output_path + ".reference.npy", reference.numpy())
    np.savez(
        output_path + ".inputs.npz",
        pixel_values=pixel_values.numpy(),
        rotary_pos_emb=rotary_pos_emb.numpy(),
    )

    print("Exported to {}".format(os.path.abspath(args.export_vision_path)))
    print("pixel_values: {}".format(tuple(pixel_values.shape)))
    print("rotary_pos_emb: {}".format(tuple(rotary_pos_emb.shape)))
    print("image_embeddings: {}".format(tuple(reference.shape)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Microsoft Mage-VL vision model to ONNX"
    )
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    parser.add_argument("--export_vision_path", type=str, default=ONNX_MODEL)
    parser.add_argument("--quant", action="store_true", help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", type=str, default=CALI_DATASET, help="some samples for grq quantized_algorithm")
    parser.add_argument("--patch_num", type=int, default=576, help="number of input image patches")
    parser.add_argument("--modelscope", action="store_true")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download

        args.model_path = snapshot_download(args.model_path)

    if args.patch_num % 4 != 0:
        raise ValueError("patch_num must be divisible by 4")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=os.path.isdir(args.model_path),
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    model = model.model.visual.float().eval()
    gc.collect()

    if args.quant and not torch.cuda.is_available():
        raise RuntimeError("GRQ quantization requires an available CUDA device")

    if args.quant:
        from rknn.quantization.api import RKQuantizer

        if not os.path.isfile(args.cali_dataset):
            raise FileNotFoundError(
                "GRQ calibration dataset not found: {}".format(args.cali_dataset)
            )

        QuantTool = RKQuantizer(verbose=True)
        ret = QuantTool.load_model(model=model, tokenizer=None, device="cuda")
        if ret != 0:
            print("Load model failed!")
            exit(ret)

        model = QuantTool.quantize(
            quantized_dtype="w4a16",
            quantized_method="group32",
            quantized_algorithm="grq",
            dataset=args.cali_dataset,
        )
        model = model.cpu()

    export_mage_vl_vision(model, args)
