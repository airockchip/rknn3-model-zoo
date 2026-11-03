import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import sys, torch, torch.nn as nn, types
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import update_config
from py_utils.export_vision_helper import save_config
from py_utils.tools import clear_llm_external_weight_in_dir
from transformers import AutoConfig, HunYuanVLForConditionalGeneration # transformers>=5.13.0
import numpy as np

def export_hunyuan_vl_vision(model, args, patch_size=16, merge_size=2, temporal_patch_size=1):
    # 裁剪版 wrapper: 输入 flatten_patches + pos_embed + grid_h + grid_w
    # - 9 维 transpose 在模型外部完成 (RKNN 不支持 9 维 Transpose)
    # - position encoding 插值在模型外部完成 (F.interpolate 生成 If 节点, RKNN 不支持)
    # - grid_h/grid_w 作为标量输入, 用于 patch merger 的 reshape

    # 自定义动态 Reshape: forward 用 tolist(), symbolic 映射为 ONNX Reshape (动态 shape)
    class _DynamicReshape(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, shape_tensor):
            return x.reshape(tuple(shape_tensor.tolist()))

        @staticmethod
        def symbolic(g, x, shape_tensor):
            return g.op("Reshape", x, shape_tensor)

    def dynamic_reshape(x, shape_tensor):
        return _DynamicReshape.apply(x, shape_tensor)
    class hunyuan_vl_vision(torch.nn.Module):
        def __init__(self, vl_m, patch_size, merge_size, temporal_patch_size):
            super().__init__()
            self.vl_m = vl_m  # HunYuanVLVisionTransformer
            self.patch_size = patch_size
            self.merge_size = merge_size
            self.temporal_patch_size = temporal_patch_size
            self.channel = 3
            self.patch_dim = self.channel * self.temporal_patch_size * self.patch_size * self.patch_size

        def forward(self, flatten_patches, pos_embed, grid_h, grid_w):
            # flatten_patches: [N, 768]  (外部完成 reshape+permute)
            # pos_embed: [N, 1152]       (外部完成 bilinear 插值)
            # grid_h: int64 标量          (H // patch_size) — 固化为常量, 每个尺寸需独立导出
            # grid_w: int64 标量          (W // patch_size) — 固化为常量, 每个尺寸需独立导出

            vl_m = self.vl_m

            # ---- 1. Patch Embedding ----
            embed = vl_m.embeddings
            num_patches = flatten_patches.shape[0]
            pixel_values = flatten_patches.reshape(-1, self.channel, self.patch_size, self.patch_size)
            patch_embeds = embed.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(1)  # [N, hidden] (避免 squeeze 生成 If 节点)
            hidden_states = (embeddings + pos_embed).unsqueeze(0)  # [1, N, hidden]

            # ---- 2. Transformer Layers (内联 attention，移除 split/tolist) ----
            for layer in vl_m.layers:
                residual = hidden_states
                hidden_states = layer.layer_norm1(hidden_states)

                attn = layer.self_attn
                query = attn.q_proj(hidden_states)
                key = attn.k_proj(hidden_states)
                value = attn.v_proj(hidden_states)
                bsz, seq_len, _ = query.shape
                q = query.view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
                k = key.view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
                v = value.view(bsz, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)

                k = k.repeat_interleave(attn.num_key_value_groups, dim=1)
                v = v.repeat_interleave(attn.num_key_value_groups, dim=1)

                attn_weights = torch.matmul(q, k.transpose(2, 3)) * attn.scaling
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
                attn_output = torch.matmul(attn_weights, v)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, seq_len, -1).contiguous()
                attn_output = attn.o_proj(attn_output)

                hidden_states = residual + attn_output

                residual = hidden_states
                hidden_states = layer.layer_norm2(hidden_states)
                hidden_states = layer.mlp(hidden_states)
                hidden_states = residual + hidden_states

            # ---- 3. Patch Merger (grid_h/grid_w 固化为常量, 不同尺寸需独立导出) ----
            merger = vl_m.patch_merger
            merged = merger.before_rms(hidden_states)
            dtype = merged.dtype
            merged = merged.permute(0, 2, 1)  # [1, C, N]
            merged = merged.reshape(1, -1, int(grid_h), int(grid_w))
            merged = merger.proj_conv(merged)
            merged = merger.proj_act(merged)
            merged = merger.proj_out(merged)
            bsz2, ch2, ht2, wt2 = merged.shape
            merged = torch.cat([
                merged,
                merger.image_newline.reshape(1, ch2, 1, 1).expand(bsz2, ch2, ht2, 1).to(dtype)
            ], dim=-1)
            merged = merged.reshape(bsz2, ch2, -1).permute(0, 2, 1)
            merged = merger.mlp(merged)
            begin = merger.image_begin.reshape(1, 1, -1).expand(bsz2, 1, merged.shape[-1]).to(dtype)
            end = merger.image_end.reshape(1, 1, -1).expand(bsz2, 1, merged.shape[-1]).to(dtype)
            merged = torch.cat([begin, merged, end], dim=1)
            merged = merger.after_rms(merged)

            return merged

    vision_tower = model.model.vision_tower
    wrapper = hunyuan_vl_vision(vision_tower, patch_size, merge_size, temporal_patch_size)

    # Align once and use the effective dimensions consistently for ONNX,
    # vision_config.json, and the pos_embed filename.
    align_size = merge_size * patch_size
    in_h = (args.img_h + align_size - 1) // align_size * align_size
    in_w = (args.img_w + align_size - 1) // align_size * align_size
    save_config("vision_config.json", in_h, in_w, patch_size)

    # 构造测试输入
    grid_h_val = in_h // patch_size
    grid_w_val = in_w // patch_size
    num_patches = grid_h_val * grid_w_val
    patch_dim = 3 * temporal_patch_size * patch_size * patch_size

    fake_flatten = torch.randn(num_patches, patch_dim)
    # 构造 position embedding (调用原始 interpolate_pos_encoding)
    embed = vision_tower.embeddings
    fake_embeddings = torch.randn(num_patches, embed.embed_dim)
    fake_pos = embed.interpolate_pos_encoding(fake_embeddings, grid_h_val, grid_w_val).squeeze(0)
    fake_grid_h = torch.tensor(grid_h_val, dtype=torch.int64)
    fake_grid_w = torch.tensor(grid_w_val, dtype=torch.int64)

    out = wrapper(fake_flatten, fake_pos, fake_grid_h, fake_grid_w)
    print(f"wrapper output shape: {out.shape}")

    torch.onnx.export(wrapper,
                    (fake_flatten, fake_pos, fake_grid_h, fake_grid_w),
                    args.export_vision_path,
                    input_names=['flatten_patches', 'pos_embed', 'grid_h', 'grid_w'],
                    dynamic_axes={
                        'flatten_patches': {0: 'num_patches'},
                        'pos_embed': {0: 'num_patches'},
                    },
                    opset_version=17
                    )
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")

    # grid_h/grid_w are intentionally converted to Python ints in forward and
    # must be folded into the graph. Fail fast if a future exporter keeps them
    # as graph inputs, because export_rknn.py accepts exactly two inputs.
    import onnx
    onnx_model = onnx.load(args.export_vision_path, load_external_data=False)
    graph_inputs = [value.name for value in onnx_model.graph.input]
    expected_inputs = ['flatten_patches', 'pos_embed']
    if graph_inputs != expected_inputs:
        raise RuntimeError(
            f"Unexpected ONNX inputs: {graph_inputs}; expected {expected_inputs}. "
            "grid_h/grid_w must be constant-folded."
        )
    print(f"Verified ONNX inputs: {graph_inputs} (grid_h/grid_w are constants)")

    # 保存预计算的 pos_embed 供 C++ 端加载
    pos_embed_out = fake_pos.detach().to(torch.float16).numpy()
    pos_embed_path = os.path.join(os.path.dirname(args.export_vision_path), f"pos_embed_{in_w}x{in_h}.bin")
    pos_embed_out.tofile(pos_embed_path)
    print(f"Saved pos_embed to {pos_embed_path} ({pos_embed_out.nbytes} bytes)")

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export HunyuanOCR-1.5 vision configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="tencent/HunyuanOCR")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/HunyuanOCR-1.5-vision.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--hf_download", action='store_true', help="Whether download model from HuggingFace")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--img_h", type=int, help="Input image height. Rounded up to a multiple of 32.", required=False, default=512)
    parser.add_argument("--img_w", type=int, help="Input image width. Rounded up to a multiple of 32.", required=False, default=512)

    args = parser.parse_args()

    if args.hf_download and args.modelscope:
        parser.error("--hf_download and --modelscope are mutually exclusive")
    if args.img_h <= 0 or args.img_w <= 0:
        parser.error("--img_h and --img_w must be positive")

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)
    elif args.hf_download:
        from huggingface_hub import snapshot_download
        rawmodel_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../model/rawmodel'))
        args.model_path = snapshot_download(repo_id=args.model_path, local_dir=rawmodel_dir)

    kwargs = {
        'trust_remote_code': True,
        'attn_implementation': 'eager',
        'torch_dtype': torch.float32,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ['use_cache'], False)
    update_config(config, ['_attn_implementation_autoset'], False)
    kwargs['config'] = config
    model = HunYuanVLForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    model = model.float()

    import types
    orig_forward = type(model.model.vision_tower).forward
    if hasattr(orig_forward, '__wrapped__'):
        model.model.vision_tower.forward = types.MethodType(orig_forward.__wrapped__, model.model.vision_tower)

    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.model.vision_tower, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        model.model.vision_tower = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)

        model = model.cpu()
        
    export_vision_dirname = os.path.dirname(args.export_vision_path)
    if not os.path.exists(export_vision_dirname):
        os.makedirs(export_vision_dirname)

    # export vision model
    export_hunyuan_vl_vision(model, args, patch_size=16, merge_size=2, temporal_patch_size=1)
