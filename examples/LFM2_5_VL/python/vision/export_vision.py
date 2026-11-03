#!/usr/bin/env python
"""Export LFM2.5-VL vision model to ONNX with EXTERNAL pos_emb input (方案 B).

Key design (verified mathematically equivalent to HF reference):
- pixel_values:  (1, 1024, 768)   fixed (NaFlex max_num_patches=1024, patch_flat=768)
- pos_emb:       (1, 1024, 1152)  external input, computed on device (C++ side):
                  16x16 base (position_embedding_size) -> antialias bilinear
                  interpolate to (gh, gw) -> pad to 1024 (first-row fill)
- pixel_attention_mask: (1, 1024)
- Graph is fully static: NO dynamic crop / reshape.
  hidden (1,1024,1152) -> reshape(1,32,32,1152) [constant] -> projector
  (pixel_unshuffle -> (1,16,16,4C) -> Linear) -> flatten (1,256,2048) [constant]
- Output: ALL 256 rows. The device extracts the valid region
  (rows r*16+c with r<gh/2, c<gw/2) outside the model.
This works because gh/gw are always even for MMBench grids, so the 2x2
pixel_unshuffle neighborhoods never straddle the valid/invalid boundary.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from transformers.models.siglip2.modeling_siglip2 import create_bidirectional_mask

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))
from py_utils.tools import clear_llm_external_weight_in_dir


class VisionModelWrapper(torch.nn.Module):
    """Vision encoder + projector with external pos_emb, fully static graph."""

    def __init__(self, vision_model, projector=None):
        super().__init__()
        self.vm = vision_model      # Siglip2VisionModel
        self.projector = projector  # Lfm2VlMultiModalProjector
        self.grid_size = 32         # sqrt(max_num_patches) - static reshape target

    def forward(self, pixel_values, pixel_attention_mask, pos_emb):
        # patch embedding + external position embedding (instead of internal resize)
        emb = self.vm.embeddings.patch_embedding(pixel_values)
        hidden = emb + pos_emb

        # bidirectional attention mask (identical to Siglip2VisionModel.forward)
        mask = create_bidirectional_mask(self.vm.config, hidden, pixel_attention_mask)
        enc = self.vm.encoder(hidden, attention_mask=mask)
        hidden = self.vm.post_layernorm(enc.last_hidden_state)  # (1, 1024, 1152)

        # static reshape to (1, 32, 32, D) - no crop, no dynamic shape
        B, N, D = hidden.shape
        hidden = hidden.reshape(B, self.grid_size, self.grid_size, D)

        if self.projector is not None:
            embeds = self.projector(hidden)  # unshuffle + LN + Linear1 + act + Linear2
        else:
            embeds = hidden
        # flatten: (B, 256, D')
        embeds = embeds.reshape(B, -1, embeds.shape[-1])
        return embeds


def main():
    from argparse import ArgumentParser

    MODEL_CONFIGS = {
        "1.6B": {"model_name": "LiquidAI/LFM2.5-VL-1.6B", "output_basename": "LFM2.5-VL-1.6B"},
        "3B":   {"model_name": "LiquidAI/LFM2.5-VL-3B",   "output_basename": "LFM2.5-VL-3B"},
    }

    parser = ArgumentParser(description="Export LFM2.5-VL vision model with external pos_emb input")
    parser.add_argument("--model_size", type=str, default="1.6B", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--export_vision_path", type=str, default=None)
    parser.add_argument("--modelscope", action='store_true')
    parser.add_argument("--quant", action='store_true',
                        help="GRQ quantization (requires CUDA, default off)")
    parser.add_argument("--quant_dtype", type=str, default="w4a16", choices=["w4a16", "w8a16"],
                        help="GRQ quantization precision (default w4a16)")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json',
                        help="calibration data for GRQ (run make_calidata.py first)")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model_size]
    if args.model_path is None:
        args.model_path = cfg["model_name"]
    if args.export_vision_path is None:
        args.export_vision_path = f"../../model/vision/{cfg['output_basename']}-vision.onnx"

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.use_cache = False
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        config=config,
        attn_implementation="eager",
    )
    model.eval()

    export_dir = os.path.dirname(args.export_vision_path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir)
    clear_llm_external_weight_in_dir(export_dir)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    vision_config = getattr(config, 'vision_config', None)
    print(f"Vision config: {vision_config}")

    vision_model = model.model.vision_tower
    projector = getattr(model.model, 'multi_modal_projector', None) or getattr(model.model, 'projector', None)
    print(f"Vision model: {type(vision_model).__name__}, projector: {type(projector).__name__}")

    # processor output gives canonical shapes
    dummy = Image.new('RGB', (512, 512), color=(128, 128, 128))
    inputs = processor(images=dummy, text="<image>", return_tensors="pt", do_image_splitting=False)
    pixel_values = inputs['pixel_values'].float()
    pixel_attention_mask = inputs['pixel_attention_mask'].float()
    print(f"pixel_values: {pixel_values.shape}, mask: {pixel_attention_mask.shape}, spatial: {inputs['spatial_shapes'].tolist()}")

    D = vision_model.embeddings.position_embedding.weight.shape[1]
    pos_emb = torch.zeros(1, pixel_values.shape[1], D, dtype=torch.float32)

    # Save the base position-embedding table (fp32, grid_base x grid_base x D)
    # BEFORE optional GRQ quantization replaces vision_model.
    pos_emb_base = vision_model.embeddings.position_embedding.weight.data.detach().cpu().clone()
    pos_base_size = vision_model.embeddings.position_embedding_size

    if args.quant:
        if not torch.cuda.is_available():
            print("[ERROR] --quant requires CUDA GPU"); sys.exit(1)
        print(f"--> GRQ quantization ({args.quant_dtype}, dataset: {args.cali_dataset})")
        from rknn.quantization.api import RKQuantizer
        q = RKQuantizer(verbose=True)
        assert q.load_model(model=vision_model, tokenizer=None, device='cuda') == 0
        vision_model = q.quantize(quantized_dtype=args.quant_dtype, quantized_method="group32",
                                  quantized_algorithm="grq", dataset=args.cali_dataset).cpu()

    wrapped = VisionModelWrapper(vision_model, projector)
    wrapped.eval()

    # torch reference check vs HF original path (same image)
    with torch.no_grad():
        # build the CORRECT pos_emb for the dummy image's grid via HF resize
        gh, gw = int(inputs['spatial_shapes'][0][0]), int(inputs['spatial_shapes'][0][1])
        pos_w = vision_model.embeddings.position_embedding.weight.data
        ps = vision_model.embeddings.position_embedding_size
        pos2d = pos_w.reshape(ps, ps, D).permute(2, 0, 1).unsqueeze(0).float()
        resized = F.interpolate(pos2d, size=(gh, gw), mode='bilinear',
                                align_corners=False, antialias=True)
        resized = resized.reshape(D, gh * gw).transpose(0, 1)
        if gh * gw < pixel_values.shape[1]:
            resized = torch.cat([resized, resized[0:1].expand(pixel_values.shape[1] - gh * gw, -1)], dim=0)
        pos_emb_real = resized.unsqueeze(0)  # (1, 1024, D)

        out_new = wrapped(pixel_values, pixel_attention_mask, pos_emb_real)  # (1,256,2048)
        # HF original: crop + dynamic reshape + projector
        h_ref = model.model.vision_tower(pixel_values=pixel_values,
                                         pixel_attention_mask=pixel_attention_mask,
                                         spatial_shapes=inputs['spatial_shapes']).last_hidden_state
        N = int(pixel_attention_mask.sum())
        hc = h_ref[:, :N].reshape(1, gh, gw, -1)
        ref = projector(hc).reshape(-1, out_new.shape[-1])
        # extract valid rows from new output
        idxs = torch.tensor([r * 16 + c for r in range(gh // 2) for c in range(gw // 2)])
        out_c = out_new[0][idxs]
        diff = (ref - out_c).abs().max().item()
        print(f"[CHECK] grid {gh}x{gw}: HF ref {tuple(ref.shape)} vs new {tuple(out_c.shape)}  maxdiff={diff:.3e}")
        assert diff < 1e-4, "wrapper mismatch with HF reference!"
        print("[CHECK] PASS - mathematically equivalent to HF")

    # export ONNX: 3 inputs, static graph, output (1, 256, 2048)
    print(f"--> Exporting to {args.export_vision_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (pixel_values, pixel_attention_mask, pos_emb),
            args.export_vision_path,
            input_names=['pixel_values', 'pixel_attention_mask', 'pos_emb'],
            output_names=['image_embeds'],
            opset_version=19,
            do_constant_folding=True,
            dynamo=False,
        )
    print("ONNX exported.")

    # Dump the base position-embedding table for the C++ side
    # (aa_bilinear_posemb interpolates this grid_base x grid_base table
    #  to the runtime image grid, then pads to pos_emb_rows).
    pos_emb_base_path = os.path.join(export_dir, 'pos_emb_base.bin')
    pos_emb_base.numpy().astype(np.float32).tofile(pos_emb_base_path)
    print(f"pos_emb_base.bin saved: {pos_emb_base_path} "
          f"({int(pos_base_size)}x{int(pos_base_size)}x{D} fp32, "
          f"{os.path.getsize(pos_emb_base_path)} bytes)")

    # Pre-process params read by the C++ side (rknn_lfm2_vl_vision.cc),
    # taken from the HF image processor with LFM2.5-VL deployed defaults
    # as fallbacks so the json is usable out of the box.
    vcfg = {}
    if isinstance(vision_config, dict):
        for k in ['patch_size', 'downsample_factor', 'hidden_size', 'num_patches']:
            if k in vision_config: vcfg[k] = vision_config[k]
    else:
        for k in ['patch_size', 'downsample_factor', 'hidden_size', 'num_patches']:
            vcfg[k] = getattr(vision_config, k, None)
    ip = getattr(processor, 'image_processor', None)
    image_mean = list(getattr(ip, 'image_mean', None) or [0.5, 0.5, 0.5])
    image_std = list(getattr(ip, 'image_std', None) or [0.5, 0.5, 0.5])
    rescale_factor = float(getattr(ip, 'rescale_factor', None) or (1.0 / 255.0))
    do_splitting = bool(getattr(ip, 'do_image_splitting',
                                getattr(processor, 'do_image_splitting', False)))
    ps_cfg = int(vcfg.get('patch_size') or 16)
    ds_cfg = int(vcfg.get('downsample_factor') or 2)
    total_factor = ps_cfg * ds_cfg
    min_tokens = 64
    max_tokens = int(pixel_values.shape[1]) // (ds_cfg * ds_cfg)
    min_px = getattr(ip, 'min_pixels', None)
    max_px = getattr(ip, 'max_pixels', None)
    if min_px: min_tokens = int(min_px) // (total_factor * total_factor)
    if max_px: max_tokens = int(max_px) // (total_factor * total_factor)

    # save vision config for C++ side
    out_cfg = {
        'max_num_patches': int(pixel_values.shape[1]),
        'num_patches': int(pixel_values.shape[1]),
        'patch_flat': int(pixel_values.shape[2]),
        'pos_emb_rows': int(pixel_values.shape[1]),
        'pos_emb_dim': D,
        'grid_base': int(pos_base_size),
        'out_rows': 256,
        'out_dim': int(out_new.shape[-1]),
        'encoder_patch_size': ps_cfg,
        'downsample_factor': ds_cfg,
        'min_image_tokens': int(min_tokens),
        'max_image_tokens': int(max_tokens),
        'do_image_splitting': do_splitting,
        'rescale_factor': rescale_factor,
        'image_mean': image_mean,
        'image_std': image_std,
        'vision_config': vcfg,
        'spatial_shapes_value': inputs['spatial_shapes'].tolist(),
    }
    with open(os.path.join(export_dir, 'vision_config.json'), 'w') as f:
        json.dump(out_cfg, f, indent=2)
    print(f"vision_config.json saved: {out_cfg}")


if __name__ == '__main__':
    from PIL import Image
    main()
