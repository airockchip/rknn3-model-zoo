import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from transformers import AutoConfig, AutoModelForImageTextToText


class minicpmV4_7_vision(torch.nn.Module):
    """MiniCPMV4.7 视觉塔导出 wrapper（固定单切片 448x448 图像输入）。

    输入为真实图像 pixel_values [1, 3, 448, 448]（fp32，板端由 rknn 预处理做
    uint8->fp32 归一化），wrapper 内部用 view/permute/reshape 完成 NaViT 打包
    （等价 image_processing_minicpmv4_7.reshape_by_patch），再走视觉塔。
    bucketize 位置编码、window_index 等数据相关计算在 __init__ 里用真实张量
    预计算；vit_merger 的窗口自注意力用批量 sdpa 等价实现。
    """

    def __init__(self, vlm):
        super().__init__()
        self.vit = vlm.model.vision_tower
        self.merger = vlm.model.merger

        vc = vlm.config.vision_config
        self.patch_size = vc.patch_size
        self.image_size = 448              # 单切片固定分辨率
        self.patch_num = self.image_size // self.patch_size   # 32
        h_p = w_p = self.patch_num
        npps = self.vit.embeddings.num_patches_per_side

        # 与 MiniCPMV4_7VisionEmbeddings.forward 相同的 bucketize 位置编码（预计算）
        boundaries = torch.arange(1 / npps, 1.0, 1 / npps)
        frac_h = torch.arange(0, 1 - 1e-6, 1 / h_p)
        frac_w = torch.arange(0, 1 - 1e-6, 1 / w_p)
        bc_h = torch.bucketize(frac_h, boundaries, right=True)
        bc_w = torch.bucketize(frac_w, boundaries, right=True)
        self.position_ids = (bc_h[:, None] * npps + bc_w).flatten().long()

        # vit_merger 前常量
        self.target_sizes = torch.tensor([[h_p, w_p]], dtype=torch.int32)
        self.cu_seqlens = torch.tensor([0, h_p * w_p], dtype=torch.int32)
        self.max_seqlen = h_p * w_p

        # vit_merger 后（get_downsampled_inputs 等价）常量
        h2 = w2 = h_p // 2
        self.target_sizes2 = torch.tensor([[h2, w2]], dtype=torch.int32)
        self.cu_seqlens2 = torch.tensor([0, h2 * w2], dtype=torch.int32)
        self.max_seqlen2 = self.max_seqlen // 4

        self.insert_layer_id = vc.insert_layer_id

        # vit_merger 窗口常量（等价 get_window_index，预计算）
        vm = self.vit.vit_merger
        wh, ww = vm.window_kernel_size
        self.num_windows = (h_p // wh) * (w_p // ww)
        window_index = torch.arange(h_p * w_p).reshape(h_p, w_p)
        window_index = window_index.reshape(h_p // wh, wh, w_p // ww, ww)
        window_index = window_index.permute(0, 2, 1, 3).reshape(self.num_windows, wh * ww)
        self.window_index = window_index.reshape(-1)
        self.window_sort_idx = torch.argsort(self.window_index)

        self.num_heads = vm.self_attn.num_heads
        self.head_dim = vm.self_attn.head_dim
        self.scaling = vm.self_attn.scaling

    def _vit_merger(self, hidden):
        """等价 MiniCPMV4_7ViTWindowAttentionMerger.forward（单切片），
        窗口自注意力按 batch 维度合并为一次 sdpa。"""
        vm = self.vit.vit_merger
        residual = hidden
        h = vm.layer_norm1(hidden)
        h = h[:, self.window_index, :]  # [1, 1024, C]，按窗口序

        B, L, C = h.shape
        q = vm.self_attn.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = vm.self_attn.k_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = vm.self_attn.v_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # [B, heads, n_windows*4, dim] -> [B, heads, n_windows, 4, dim] -> [B*n_windows, heads, 4, dim]
        q = q.view(B, self.num_heads, self.num_windows, 4, self.head_dim).transpose(1, 2).reshape(B * self.num_windows, self.num_heads, 4, self.head_dim)
        k = k.view(B, self.num_heads, self.num_windows, 4, self.head_dim).transpose(1, 2).reshape(B * self.num_windows, self.num_heads, 4, self.head_dim)
        v = v.view(B, self.num_heads, self.num_windows, 4, self.head_dim).transpose(1, 2).reshape(B * self.num_windows, self.num_heads, 4, self.head_dim)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=self.scaling)
        # 还原 [B, heads, L, dim]
        attn = attn.view(B, self.num_windows, self.num_heads, 4, self.head_dim).permute(0, 2, 1, 3, 4).reshape(B, self.num_heads, L, self.head_dim)
        attn = attn.transpose(1, 2).reshape(B, L, C)
        attn = vm.self_attn.out_proj(attn)
        h = residual + attn[:, self.window_sort_idx, :]

        # 2x2 窗口合并 + MLP（单切片，cu_seqlens [0, 1024]）
        patch = h[0, 0:1024, :]
        wh, ww = vm.window_kernel_size
        mh = mw = 32 // 2
        p5 = patch.view(mh, wh, mw, ww, C).permute(0, 2, 1, 3, 4)
        hidden_state = p5.reshape(mh * mw, wh * ww * C)
        res = p5.reshape(mh * mw, wh * ww, C).mean(dim=1)
        hidden_state = vm.pre_norm(hidden_state)
        hidden_state = vm.linear_1(hidden_state)
        hidden_state = vm.act(hidden_state)
        hidden_state = vm.linear_2(hidden_state)
        return (hidden_state + res).unsqueeze(0)  # [1, 256, C]

    def forward(self, pixel):
        # 1. NaViT 打包（等价 reshape_by_patch：unfold -> [C,14,14,P] -> [C,14,P*14]）。
        #    注意 PyTorch 的 permute->reshape 在非连续张量上按视图顺序物理重排，
        #    必须用两个 permute->reshape 对得到 unfold 布局。
        B, C, H, W = pixel.shape
        ps = self.patch_size
        P = (H // ps) * (W // ps)
        pixel = pixel.view(B, C, H // ps, ps, W // ps, ps) \
                     .permute(0, 1, 2, 4, 3, 5) \
                     .reshape(B, C, H // ps, W // ps, ps, ps) \
                     .permute(0, 1, 4, 5, 2, 3) \
                     .reshape(B, C, ps, ps, P) \
                     .permute(0, 1, 2, 4, 3) \
                     .reshape(B, C, ps, -1)

        # 2. patch embedding（打包输入 [1,3,14,14336] -> [1,1024,C]）+ 位置编码
        hidden = self.vit.embeddings.patch_embedding(pixel)
        hidden = hidden.flatten(2).transpose(1, 2)
        hidden = hidden + self.vit.embeddings.position_embedding(self.position_ids)

        # 3. encoder（insert_layer_id 处插入 vit_merger，随后降采样）
        cu_seqlens = self.cu_seqlens
        max_seqlen = self.max_seqlen
        layers = self.vit.encoder.layers
        for i in range(len(layers)):
            hidden = layers[i](hidden, None, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            if i == self.insert_layer_id:
                hidden = self._vit_merger(hidden)
                cu_seqlens = self.cu_seqlens2
                max_seqlen = self.max_seqlen2

        hidden = self.vit.post_layernorm(hidden)

        # 4. 降采样 MLP merger（等价 get_image_features 后半段）
        feats = self.merger(hidden, self.target_sizes2)
        return torch.cat(feats, dim=0).unsqueeze(0)  # [1, 64, 1024]


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export openbmb/MiniCPM-V-4.7-1B vision configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="./openbmb/MiniCPM-V-4.7-1B")
    parser.add_argument("--export_vision_path", type=str, help="export vision onnx model path", required=False, default="../../model/vision/MiniCPM-V-4.7-1B-vision.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True,
        'torch_dtype': torch.float32,
    }
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    kwargs['config'] = config
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **kwargs)

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

    vision_model = minicpmV4_7_vision(model)

    # 真实图像输入 [1, 3, 448, 448]（fp32），NaViT 打包在 ONNX 图内完成
    pixel = torch.randn(1, 3, vision_model.image_size, vision_model.image_size)

    torch.onnx.export(vision_model,
                      (pixel, ),
                      args.export_vision_path,
                      input_names=['pixel'],
                      opset_version=17)
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")
