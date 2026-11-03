import argparse
import os
import torch
import torch.nn as nn
import sys

class ONNXWrapper(nn.Module):
    def __init__(self, model, iters):
        super().__init__()
        self.model = model
        self.iters = iters
    def forward(self, image1, image2):
        _, flow_up = self.model(image1, image2, iters=self.iters, test_mode=True)
        return flow_up

def build_args(
    corr_implementation="reg",
    shared_backbone=False,
    corr_levels=4,
    corr_radius=4,
    n_downsample=2,
    context_norm="batch",
    slow_fast_gru=False,
    n_gru_layers=3,
    hidden_dims=None,
    mixed_precision=False,
):
    if hidden_dims is None:
        hidden_dims = [128]*3
    class A:
        pass
    a = A()
    a.corr_implementation = corr_implementation
    a.shared_backbone = shared_backbone
    a.corr_levels = corr_levels
    a.corr_radius = corr_radius
    a.n_downsample = n_downsample
    a.context_norm = context_norm
    a.slow_fast_gru = slow_fast_gru
    a.n_gru_layers = n_gru_layers
    a.hidden_dims = hidden_dims
    a.mixed_precision = mixed_precision
    return a

def load_raft_weights(model, ckpt_path):
    if ckpt_path is None:
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
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v
    model_state = model.state_dict()
    filtered = {k: v for k, v in new_state.items() if (k in model_state and model_state[k].shape == v.shape)}
    model.load_state_dict(filtered, strict=False)
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raft_source", type=str, default=None, help="RAFT-Stereo source path from github")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="../model/onnx")
    p.add_argument("--save_name", type=str, default="raft_stereo")
    p.add_argument("--iters", type=int, default=1)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--opset", type=int, default=16)
    p.add_argument("--corr", type=str, choices=["reg","alt","reg_cuda","alt_cuda"], default="reg")
    p.add_argument("--shared_backbone", action="store_true")
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument('--corr_levels', type=int, default=1, help="number of levels in the correlation pyramid")
    p.add_argument('--corr_radius', type=int, default=2, help="width of the correlation pyramid")
    p.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    p.add_argument('--context_norm', type=str, default="none", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    p.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    p.add_argument('--n_gru_layers', type=int, default=1, help="number of hidden GRU levels")
    p.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    args = p.parse_args()

    if not os.path.exists(args.raft_source):
        raise FileNotFoundError(f"Error：dictionory '{args.raft_source}' isn't exist")
    if args.raft_source not in sys.path:
        sys.path.insert(0, args.raft_source)
    from core.raft_stereo import RAFTStereo

    raft_args = build_args(
        corr_implementation=args.corr,
        shared_backbone=args.shared_backbone,
        mixed_precision=args.mixed_precision,
        corr_levels=args.corr_levels,
        corr_radius=args.corr_radius,
        n_downsample=args.n_downsample,
        context_norm=args.context_norm,
        slow_fast_gru=args.slow_fast_gru,
        n_gru_layers=args.n_gru_layers,
        hidden_dims=args.hidden_dims,
    )
    model = RAFTStereo(raft_args)
    model = load_raft_weights(model, args.ckpt)
    model.eval()
    wrapper = ONNXWrapper(model, args.iters).to("cpu").eval()

    image_left = torch.rand(1, 3, args.height, args.width)
    image_right = torch.rand(1, 3, args.height, args.width)

    os.makedirs(args.out_dir, exist_ok=True)
    onnx_path = os.path.join(args.out_dir, args.save_name + ".onnx")

    torch.onnx.export(
        wrapper,
        (image_left, image_right),
        onnx_path,
        do_constant_folding=False,
        keep_initializers_as_inputs=True,
        opset_version=args.opset,
        input_names=["image_left", "image_right"],
        output_names=["disp"],
        kwargs={"iters": 32},
        # verbose=True,
    )
    print(f"Exported: {onnx_path}")

if __name__ == "__main__":
    main()
