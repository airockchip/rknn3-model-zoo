#!/usr/bin/env python3
"""验证 ONNX 模型与 PyTorch 模型输出是否一致"""

import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort
import os
import sys

def build_args(**kwargs):
    defaults = {
        "corr_implementation": "reg",
        "shared_backbone": False,
        "corr_levels": 4,
        "corr_radius": 4,
        "n_downsample": 2,
        "context_norm": "batch",
        "slow_fast_gru": False,
        "n_gru_layers": 3,
        "hidden_dims": [128] * 3,
        "mixed_precision": False,
    }
    defaults.update(kwargs)
    class A: pass
    a = A()
    for k, v in defaults.items():
        setattr(a, k, v)
    return a


def load_raft_weights(model, ckpt_path):
    """加载预训练权重：可学习参数 strict，buffer（running_mean/var）仅警告"""
    if ckpt_path is None:
        print("   WARNING: no --ckpt provided, using random weights (logic check only)")
        return model

    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj
    if isinstance(obj, dict):
        if "model_state" in obj:
            state = obj["model_state"]
        elif "state_dict" in obj:
            state = obj["state_dict"]

    new_state = {}
    for k, v in state.items():
        new_state[k[len("module."):] if k.startswith("module.") else k] = v

    model_state = model.state_dict()
    param_keys  = set(dict(model.named_parameters()).keys())
    buffer_keys = set(dict(model.named_buffers()).keys())

    # checkpoint 中多余的 key → 报错（model 没有这个层，架构不匹配）
    extra_keys = [k for k in new_state if k not in model_state]
    if extra_keys:
        raise RuntimeError(
            "Weight load failed: checkpoint has keys not in model (wrong arch):\n" +
            "\n".join(f"  {k}" for k in extra_keys[:10])
        )

    # 可学习参数缺失 → 报错
    missing_params = [k for k in param_keys if k not in new_state]
    if missing_params:
        raise RuntimeError(
            f"Weight load failed: {len(missing_params)} learnable params missing from checkpoint:\n" +
            "\n".join(f"  {k}" for k in missing_params[:10]) +
            (f"\n  ... total {len(missing_params)}" if len(missing_params) > 10 else "")
        )

    # buffer 缺失 → 仅警告（常见于 InstanceNorm checkpoint vs BatchNorm model）
    missing_buffers = [k for k in buffer_keys if k not in new_state]
    if missing_buffers:
        print(f"   WARNING: {len(missing_buffers)} buffers missing (using defaults).")
        print(f"     e.g. {missing_buffers[:3]}")
        print(f"     Likely cause: checkpoint fnet uses InstanceNorm (no running_mean/var)")

    model.load_state_dict(new_state, strict=False)
    print(f"   Learnable params: {len(param_keys)}/{len(param_keys)} matched OK")
    if missing_buffers:
        ok = len(buffer_keys) - len(missing_buffers)
        print(f"   Buffers: {ok}/{len(buffer_keys)} matched, {len(missing_buffers)} using defaults")
    else:
        print(f"   Buffers: {len(buffer_keys)}/{len(buffer_keys)} matched OK")
    return model


def load_images(left_path, right_path, height, width):
    """读取真实图像，resize，返回 [0,255] float32 NCHW tensor"""
    from PIL import Image
    def _load(path):
        img = Image.open(path).convert("RGB").resize((width, height), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)        # [H, W, 3]，值域 [0, 255]
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return _load(left_path), _load(right_path)


def make_random_inputs(height, width, seed):
    """生成随机输入，值域 [0, 255] 模拟真实图像"""
    torch.manual_seed(seed)
    # Fix #1: 值域 [0, 255]，而非 [0, 1]
    image1 = torch.rand(1, 3, height, width) * 255.0
    image2 = torch.rand(1, 3, height, width) * 255.0
    return image1, image2


def clean_onnx_inputs(onnx_path):
    """移除 ONNX graph input 中的 initializer 权重"""
    import tempfile, os
    m = onnx.load(onnx_path)
    init_names = {i.name for i in m.graph.initializer}
    real_inputs = [i for i in m.graph.input if i.name not in init_names]
    weight_inputs = len(m.graph.input) - len(real_inputs)
    if weight_inputs > 0:
        print(f"   ⚠️  ONNX 中有 {weight_inputs} 个权重被列为 graph input，已自动清除")
        m.graph.ClearField("input")
        m.graph.input.extend(real_inputs)
    else:
        print(f"   ✅ ONNX graph input 正常（{len(real_inputs)} 个运行时输入）")
    tmp_path = os.path.join(tempfile.gettempdir(), "verify_onnx_clean.onnx")
    onnx.save(m, tmp_path)
    return tmp_path, [i.name for i in real_inputs]


def save_vis(left_img, right_img, pt_disp, onnx_disp, out_path):
    """保存 PyTorch vs ONNX 视差图对比（5列：左图/右图/PyTorch/ONNX/差值）"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("   [vis] matplotlib not available, skipping visualization")
        return

    def to_rgb(img_tensor):
        arr = img_tensor[0].numpy().transpose(1, 2, 0).astype(np.uint8)
        return arr

    def colorize_disp(disp, vmin=None, vmax=None):
        d = disp[0, 0].astype(np.float32)
        vmin = d.min() if vmin is None else vmin
        vmax = d.max() if vmax is None else vmax
        norm = (d - vmin) / (vmax - vmin + 1e-6)
        return (cm.plasma(norm)[:, :, :3] * 255).astype(np.uint8)

    # 以 PyTorch 输出范围为基准，ONNX 用相同 vmin/vmax 便于对比
    vmin = pt_disp[0, 0].min()
    vmax = pt_disp[0, 0].max()

    diff = np.abs(pt_disp - onnx_disp)[0, 0]
    diff_norm = diff / (diff.max() + 1e-6)
    diff_color = (cm.hot(diff_norm)[:, :, :3] * 255).astype(np.uint8)

    cols = [
        (to_rgb(left_img.to(torch.uint8)),  "Left"),
        (to_rgb(right_img.to(torch.uint8)), "Right"),
        (colorize_disp(pt_disp, vmin, vmax), "PyTorch disp"),
        (colorize_disp(onnx_disp, vmin, vmax), "ONNX disp"),
        (diff_color, f"Diff (max={diff.max():.3f})"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (img, title) in zip(axes, cols):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   [vis] saved -> {out_path}")


def compare(pt_np, onnx_out, label):
    diff = np.abs(pt_np - onnx_out)
    cos_sim = np.dot(pt_np.flatten(), onnx_out.flatten()) / (
        np.linalg.norm(pt_np) * np.linalg.norm(onnx_out) + 1e-12)
    print(f"   [{label}] max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}"
          f"  MAE={diff.mean():.4f}  Cosine={cos_sim:.6f}")
    return cos_sim


def verify_onnx(args):
    print("=" * 60)

    # 1. 创建 PyTorch 模型
    print("1. 创建 PyTorch 模型...")
    n_gru = args.n_gru_layers
    raft_args = build_args(
        corr_levels=args.corr_levels, corr_radius=args.corr_radius,
        n_downsample=args.n_downsample, context_norm=args.context_norm,
        shared_backbone=args.shared_backbone, slow_fast_gru=args.slow_fast_gru,
        n_gru_layers=n_gru, hidden_dims=args.hidden_dims,
        mixed_precision=args.mixed_precision,
    )
    print(f"   架构: corr_levels={raft_args.corr_levels}, corr_radius={raft_args.corr_radius}, "
          f"n_gru_layers={raft_args.n_gru_layers}, n_downsample={raft_args.n_downsample}, "
          f"shared_backbone={raft_args.shared_backbone}")

    model = RAFTStereo(raft_args)
    model = load_raft_weights(model, args.ckpt)
    model.eval()

    # 2. 准备 ONNX session
    print("2. 加载 ONNX...")
    clean_path, input_names = clean_onnx_inputs(args.onnx)
    sess = ort.InferenceSession(clean_path, providers=["CPUExecutionProvider"])

    # 3. Fix #2: 多组输入测试
    test_cases = []

    # 3a. 真实图像（若提供）
    if args.left_img and args.right_img:
        print(f"3a. 真实图像: {args.left_img}, {args.right_img}")
        left, right = load_images(args.left_img, args.right_img, args.height, args.width)
        test_cases.append(("real_image", left, right))
    else:
        print("3a. 未提供真实图像，跳过")

    # 3b. Fix #1 + Fix #2: 多个随机种子，值域 [0, 255]
    for seed in [0, 1, 42]:
        left, right = make_random_inputs(args.height, args.width, seed)
        test_cases.append((f"random_seed{seed}", left, right))

    # 4. 逐组推理对比
    print(f"4. 推理对比（iters={args.iters}）...")
    results = []
    for label, image1, image2 in test_cases:
        print(f"   --- {label} (input range: [{image1.min():.0f}, {image1.max():.0f}]) ---")
        with torch.no_grad():
            _, pytorch_out = model(image1, image2, iters=args.iters, test_mode=True)
        pt_np = pytorch_out.numpy()

        feed = {input_names[0]: image1.numpy(), input_names[1]: image2.numpy()}
        onnx_out = sess.run(None, feed)[0]

        print(f"   PyTorch  range=[{pt_np.min():.4f}, {pt_np.max():.4f}] mean={pt_np.mean():.4f}")
        print(f"   ONNX     range=[{onnx_out.min():.4f}, {onnx_out.max():.4f}] mean={onnx_out.mean():.4f}")
        cos = compare(pt_np, onnx_out, label)
        results.append((label, cos))

        # 真实图像 → 保存可视化
        if label == "real_image" and args.left_img and args.vis_dir:
            model_tag = os.path.splitext(os.path.basename(args.onnx))[0]
            out_path = os.path.join(args.vis_dir, f"{model_tag}_onnx.png")
            save_vis(image1, image2, pt_np, onnx_out, out_path)

    # 5. 汇总结论
    print("=" * 60)
    all_pass = all(c > 0.999 for _, c in results)
    for label, cos in results:
        status = "✅" if cos > 0.999 else ("⚠️ " if cos > 0.99 else "❌")
        print(f"  {status} {label}: Cosine={cos:.6f}")
    print("-" * 60)
    if all_pass:
        print("✅ 全部通过！PyTorch 与 ONNX 输出一致")
    else:
        print("❌ 存在不一致，请检查架构参数或权重")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--raft_source",    type=str, default=None, help="RAFT-Stereo source path from github")
    p.add_argument("--onnx",           required=True,  help="ONNX 模型路径")
    p.add_argument("--ckpt",           default=None,   help="PyTorch 权重路径（.pth）")
    p.add_argument("--left_img",       default=None,   help="左图路径（真实图像验证）")
    p.add_argument("--right_img",      default=None,   help="右图路径（真实图像验证）")
    p.add_argument("--height",         type=int, default=160)
    p.add_argument("--width",          type=int, default=240)
    p.add_argument("--iters",          type=int, default=1)
    p.add_argument("--corr_levels",    type=int, default=4)
    p.add_argument("--corr_radius",    type=int, default=4)
    p.add_argument("--context_norm",   default="batch",
                   choices=["group", "batch", "instance", "none"])
    p.add_argument("--n_gru_layers",   type=int, default=3)
    p.add_argument("--hidden_dims",    type=int, nargs="+", default=[128, 128, 128])
    p.add_argument("--slow_fast_gru",  action="store_true")
    p.add_argument("--shared_backbone",action="store_true")
    p.add_argument("--n_downsample",   type=int, default=2)
    p.add_argument("--mixed_precision",action="store_true")
    p.add_argument("--vis_dir", default="vis",
                   help="可视化输出目录（仅真实图像有效，默认 vis/）")
    args = p.parse_args()

    if not os.path.exists(args.raft_source):
        raise FileNotFoundError(f"Error：dictionory '{args.raft_source}' isn't exist")
    if args.raft_source not in sys.path:
        sys.path.insert(0, args.raft_source)
    from core.raft_stereo import RAFTStereo
    verify_onnx(args)
