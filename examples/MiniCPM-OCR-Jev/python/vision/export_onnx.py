"""导出 value head 的视觉塔为 ONNX

用法::

    # 从 HuggingFace 下载（默认）
    python export_onnx.py --height 320 --width 832 --output_dir ../../model/vision

    # 从 ModelScope 下载
    python export_onnx.py --modelscope \
        --height 320 --width 832 --output_dir ../../model/vision
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "llm"))

from export_onnx import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    ZOO_ROOT,
    load_export_modeling_module,
    resolve_export_modeling,
    resolve_model_source,
    setup_env,
)


class VisionExportWrapper(nn.Module):
    """把 ``[1, 3, H, W]`` 的图片变成视觉 token 序列。

    图像模式：在**图内**把单帧复制两份以满足 ``temporal_patch_size=2``。
    这样运行时只需传一张图，不用在 host 侧拼帧。
    """

    def __init__(self, visual, patch_size: int, height: int, width: int):
        super().__init__()
        self.visual = visual
        self.patch_size = patch_size
        self.height = height
        self.width = width
        self.temporal_patch_size = 2
        self.merge_size = 2

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h, w = self.height, self.width
        p, m, t = self.patch_size, self.merge_size, self.temporal_patch_size

        # [1, 3, H, W] -> [2, 3, H, W]（伪造成两帧相同的视频）
        pixel_values = torch.cat((pixel_values, pixel_values), dim=0)

        # [2, 3, H, W] -> [1, t, 3, H/m/p, m, p, W/m/p, m, p]
        patches = pixel_values.reshape(
            1, t, 3,
            h // m // p, m, p,
            w // m // p, m, p,
        )
        # 重排成 [1, H/p, W/p, m, m, 3, t, p, p]
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
        # -> [H/p * W/p, 3*t*p*p]
        flatten_patches = patches.reshape(
            h // p * w // p,
            3 * t * p * p,
        )

        grid_thw = torch.tensor(
            [[1, h // p, w // p]], dtype=torch.int64, device=flatten_patches.device
        )
        outputs = self.visual(flatten_patches, grid_thw=grid_thw, return_dict=True)
        return outputs.pooler_output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export the value-head vision tower to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="模型仓库 id（如 openbmb/MiniCPM-OCR-Jev），会从 HuggingFace 或 ModelScope 下载",
    )
    p.add_argument(
        "--modelscope",
        action="store_true",
        help="改用 www.modelscope.cn 下载模型（默认走 HuggingFace）",
    )
    p.add_argument("--output_path", default=None, help="输出 ONNX；默认 <output_dir>/vision.onnx")
    p.add_argument("--output_dir", default="../../model/vision", help="仅给目录时用这个")
    p.add_argument("--name", default="vision", help="输出文件名（不含扩展名）")
    p.add_argument("--height", type=int, default=320, help="输入高，必须是 32 的倍数")
    p.add_argument("--width", type=int, default=832, help="输入宽，必须是 32 的倍数")
    p.add_argument("--zoo_root", default=str(ZOO_ROOT), help="rknn3-model-zoo 仓库根目录")
    p.add_argument("--export_modeling", default=None, help="导出 modeling 路径；默认取 ../llm/modeling_qwen3_5_export.py")
    p.add_argument("--hf_endpoint", default="https://hf-mirror.com/", help="HF_ENDPOINT；--modelscope 时忽略")
    args = p.parse_args(argv)

    for name, value in (("--height", args.height), ("--width", args.width)):
        if value % 32 != 0:
            p.error(f"{name} 必须是 32 的倍数（patch_size=16, merge_size=2），当前 {value}")
        if value < 64:
            p.error(f"{name} 太小，至少 64")
    return args


def load_visual_module(model_path: Path, config, modeling_module, dtype=torch.float32):
    """只构建视觉塔并把检查点里的 ``model.visual.*`` 权重装进去。

    相比 ``Qwen3_5Model.from_pretrained`` 的收益：
      * 不会把 4.4GB 文本主干读进内存（export 只需要 vision）
      * 不会因为缺 ``lm_head`` 触发任何 warning

    这里用 ``safe_open`` 按 key 逐个取张量，而不是 ``load_file`` ——
    后者会把整个 4.1GB shard 一次性读进内存（其中约 4.0GB 是我们不要的
    文本主干）。
    """
    from safetensors import safe_open

    visual = modeling_module.Qwen3_5VisionModel._from_config(config.vision_config)
    # `_from_config` 会 deepcopy config，这里再强制一次，避免默认 sdpa 把 GQA
    # 喂给 ONNX symbolic（见 main() 里的长注释）。
    visual.config._attn_implementation = "eager"
    visual = visual.to(dtype).eval()

    prefix = "model.visual."
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key[len(prefix):]] = handle.get_tensor(key).to(dtype)

    missing, unexpected = visual.load_state_dict(state, strict=False)
    # rotary 相关 buffer 在导出实现里是实时算出来的，不在检查点里
    real_missing = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
    print(f"    视觉塔权重: 加载 {len(state)} 个张量")
    if real_missing:
        print(f"    [!] 缺失 {len(real_missing)} 个: {real_missing[:5]}")
    if unexpected:
        print(f"    [!] 多余 {len(unexpected)} 个: {unexpected[:5]}")
    return visual


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_env(args.hf_endpoint)

    # --model_path 是模型仓库 id：先下载到本地缓存，下游只跟本地目录打交道。
    model_path = resolve_model_source(args.model_path, args.modelscope)

    if args.output_path:
        output_path = Path(args.output_path).expanduser().resolve()
    else:
        out_dir = Path(args.output_dir).expanduser().resolve()
        # 目录名带分辨率，避免不同 shape 的 external data 互相覆盖
        if out_dir.name in ("vision", f"{args.name}"):
            out_dir = out_dir / f"{args.height}x{args.width}"
        output_path = out_dir / f"{args.name}.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.export_modeling:
        export_modeling_path = resolve_export_modeling(args.export_modeling)
    else:
        export_modeling_path = Path(__file__).resolve().parent.parent / "llm" / "modeling_qwen3_5_export.py"
        if not export_modeling_path.is_file():
            raise FileNotFoundError(f"找不到导出 modeling: {export_modeling_path}")
    modeling_module = load_export_modeling_module(export_modeling_path)

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)

    # 必须强制 eager attention。
    # transformers 5.x 默认 _attn_implementation="sdpa"，而 torch 的 ONNX
    # symbolic 里 scaled_dot_product_attention 带 GQA（enable_gqa=True）会直接
    # 断言失败：
    #   AssertionError: conversion of scaled_dot_product_attention not
    #   implemented if enable_gqa is True
    # 视觉塔的 num_heads=16 / num_key_value_heads 更少，正好走 GQA 分支。
    # 导出实现里的 eager_attention_forward 会把 num_key_value_groups 固定为 1，
    # 展开成普通 matmul。官方 vision/export_vision.py 也是这么做的。
    config._attn_implementation = "eager"

    vision_config = config.vision_config
    vision_config._attn_implementation = "eager"
    patch_size = vision_config.patch_size

    print("=" * 78)
    print("导出视觉塔")
    print("=" * 78)
    print(f"  模型来源   : {args.model_path}")
    print(f"  检查点     : {model_path}")
    print(f"  分辨率     : {args.height}x{args.width}  patch_size={patch_size}")
    n_patches = (args.height // patch_size) * (args.width // patch_size)
    merge = vision_config.spatial_merge_size
    print(f"  patch 数   : {n_patches}  -> 视觉 token 数 {n_patches // (merge * merge)}")

    visual = load_visual_module(model_path, config, modeling_module)
    wrapper = VisionExportWrapper(visual, patch_size, args.height, args.width).eval()

    # 契约自检：先跑一遍 torch，确认输出 shape 符合预期
    with torch.inference_mode():
        probe = wrapper(torch.zeros(1, 3, args.height, args.width))
    expected_tokens = n_patches // (merge * merge)
    print(f"  torch 输出 : {tuple(probe.shape)}  (期望 [{expected_tokens}, {vision_config.out_hidden_size}])")
    if probe.shape[0] != expected_tokens:
        print("    [!] token 数不符预期")
    del probe

    sample = torch.zeros(1, 3, args.height, args.width)
    print(f"  导出 ONNX  : {output_path}")

    # 先导到临时目录，再整体搬到 output_path 所在目录。
    # 直接写目标目录会让 torch.onnx 的临时文件与已有 sidecar 混在一起。
    with tempfile.TemporaryDirectory(prefix="value-vision-", dir=output_path.parent) as temp_dir:
        temp_path = Path(temp_dir) / output_path.name
        torch.onnx.export(
            wrapper,
            (sample,),
            str(temp_path),
            input_names=["pixel"],
            output_names=["vision_embedding"],
            opset_version=17,
        )
        for generated in Path(temp_dir).iterdir():
            os.replace(generated, output_path.parent / generated.name)

    import onnx

    model = onnx.load(str(output_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=False)
    print("    ONNX checker: OK")
    for vi in model.graph.input:
        dims = [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim]
        print(f"    in  {vi.name:20s} {dims}")
    for vi in model.graph.output:
        dims = [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim]
        print(f"    out {vi.name:20s} {dims}")

    print()
    print(f"完成: {output_path}  ({output_path.stat().st_size / 1e6:.1f} MB)")
    print("下一步: python export_rknn.py --onnx_path " + str(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
