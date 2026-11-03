import torch
import os

def save_config(config_path: str, img_h: int, img_w: int, patch_size: int):
    import json
    import os

    if img_h % patch_size != 0:
        raise ValueError(f"img_h ({img_h}) must be divisible by patch_size ({patch_size})")
    if img_w % patch_size != 0:
        raise ValueError(f"img_w ({img_w}) must be divisible by patch_size ({patch_size})")
    
    grid_t = 1
    grid_h = img_h // patch_size
    grid_w = img_w // patch_size
    grid_thw = [grid_t, grid_h, grid_w]

    config = {
        "img_h": img_h,
        "img_w": img_w,
        "patch_size": patch_size,
        "grid_thw": grid_thw
    }

    # 获取目录部分，如果为空则表示当前目录，无需创建
    config_dir = os.path.dirname(config_path)
    if config_dir:  # 只有非空时才创建目录
        os.makedirs(config_dir, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"Config saved to {config_path}")


def check_torch_version():
    version = torch.__version__
    
    version_parts = version.split('+')[0].split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1])
    
    if major > 2 or (major == 2 and minor >= 9):
        return True
    else:
        return False


def export_qwen2_vl_vision(model, args, patch_size=14):
    class qwen2vl(torch.nn.Module):
        def __init__(self, vl_m, in_h, in_w, patch_size):
            super(qwen2vl, self).__init__()
            self.vl_m = vl_m

            self.grid_t = 1
            self.merge_size = 2
            self.channel = 3
            self.temporal_patch_size = 2
            self.patch_size = patch_size

            align_size = self.merge_size* self.patch_size
            align_h = (in_h + align_size - 1) // align_size* align_size
            align_w = (in_w + align_size - 1) // align_size* align_size
            self.in_h = align_h
            self.in_w = align_w

        def forward(self, pixel, grid_thw):
            patches = pixel.repeat(self.temporal_patch_size, 1, 1, 1)
            h = pixel.shape[2]
            w = pixel.shape[3]
            patches = patches.reshape(self.grid_t, self.temporal_patch_size, self.channel, h//self.merge_size//self.patch_size, self.merge_size, self.patch_size, w//self.merge_size//self.patch_size, self.merge_size, self.patch_size)
            patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flatten_patches = patches.reshape(self.grid_t * h//self.patch_size * w//self.patch_size, self.channel * self.temporal_patch_size * self.patch_size * self.patch_size)

            return self.vl_m(flatten_patches, grid_thw)

    in_h = args.img_h
    in_w = args.img_w
    model = qwen2vl(model, in_h, in_w, patch_size)
    save_config("vision_config.json", args.img_h, args.img_w, model.patch_size)
    in_h = model.in_h
    in_w = model.in_w

    fake_input = torch.randn(1, 3, in_h, in_w)
    grid_thw = torch.tensor([[1, in_h//model.patch_size, in_w//model.patch_size]], dtype=torch.int64)

    out = model(fake_input, grid_thw)

    if check_torch_version():
        torch.onnx.export(model,
                        (fake_input, grid_thw),
                        args.export_vision_path,
                        input_names=['pixel', 'grid_thw'],
                        dynamic_axes={'pixel': {2: 'height', 3: 'width'}},
                        opset_version=17,
                        dynamo=False
                        )
    else:
        torch.onnx.export(model,
                        (fake_input, grid_thw),
                        args.export_vision_path,
                        input_names=['pixel', 'grid_thw'],
                        dynamic_axes={'pixel': {2: 'height', 3: 'width'}},
                        opset_version=17
                        )
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")


def export_minicpmv4_7_vision(vision_model, merger, args, config):
    """Export the packed NaViT vision path used by MiniCPM-V 4.7.

    MiniCPM-V 4.7 does not accept a normal ``[N, 3, H, W]`` image tensor.
    Its image processor patchifies one image into
    ``[1, 3, patch_size, num_patches * patch_size]`` and passes the original
    patch grid separately as ``target_sizes``.  Keep that layout here instead
    of routing the model through the Qwen-VL ``grid_thw`` adapter above.

    The exported graph is intentionally static for one image.  The two
    2x2 merge stages require both patch-grid dimensions to be divisible by 4,
    and a fixed graph is also what the RKNN compiler can consume reliably.
    """
    import json

    class MiniCPMV4_7VisionExportWrapper(torch.nn.Module):
        def __init__(self, vision_model, merger):
            super().__init__()
            self.vision_model = vision_model
            self.merger = merger

        def forward(self, pixel_values, target_sizes):
            vision_outputs = self.vision_model(
                pixel_values,
                target_sizes=target_sizes,
                use_vit_merger=True,
            )
            # The ViT merger halves the patch grid once before the final
            # MiniCPM merger.  The export contract is one image, so unwrap
            # the single-item list returned by MiniCPMV4_7Merger.
            merged_sizes = target_sizes // 2
            return self.merger(vision_outputs.last_hidden_state, merged_sizes)[0]

    vision_config = config.vision_config
    patch_size = int(vision_config.patch_size)
    img_h = int(args.img_h)
    img_w = int(args.img_w)
    align = patch_size * 4
    if img_h % align != 0 or img_w % align != 0:
        raise ValueError(
            f"MiniCPM-V 4.7 image size must be divisible by patch_size*4 ({align}), "
            f"got {img_h}x{img_w}"
        )

    grid_h = img_h // patch_size
    grid_w = img_w // patch_size
    target_sizes = torch.tensor([[grid_h, grid_w]], dtype=torch.int32)
    # reshape_by_patch() packs H*W patches along the last dimension.
    fake_pixel_values = torch.randn(
        1, 3, patch_size, grid_h * grid_w * patch_size, dtype=torch.float32
    )

    export_model = MiniCPMV4_7VisionExportWrapper(vision_model, merger).eval()
    with torch.no_grad():
        output = export_model(fake_pixel_values, target_sizes)
    print(
        f"MiniCPM-V 4.7 vision input: pixel_values={tuple(fake_pixel_values.shape)}, "
        f"target_sizes={target_sizes.tolist()}, output={tuple(output.shape)}"
    )

    export_dir = os.path.dirname(os.path.abspath(args.export_vision_path))
    os.makedirs(export_dir, exist_ok=True)
    torch.onnx.export(
        export_model,
        (fake_pixel_values, target_sizes),
        args.export_vision_path,
        input_names=["pixel_values", "target_sizes"],
        output_names=["image_embeds"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    _fix_onnx_slice_index_types(args.export_vision_path)
    _fix_minicpmv4_7_static_vit_merger_if(
        args.export_vision_path, patch_count=grid_h * grid_w
    )

    output_config = {
        "model_type": "minicpmv4_7",
        "img_h": img_h,
        "img_w": img_w,
        "patch_size": patch_size,
        "target_sizes": [[grid_h, grid_w]],
        "pixel_values_shape": list(fake_pixel_values.shape),
        "output_shape": list(output.shape),
        "downsample_mode": str(getattr(config, "downsample_mode", "16x")),
        "input_layout": "[batch, channels, patch_size, num_patches * patch_size]",
    }
    config_path = os.path.join(export_dir, "vision_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(output_config, f, indent=2)

    print(f"Exported to {os.path.abspath(args.export_vision_path)}")
    print(f"Config saved to {config_path}: {output_config}")


def _fix_onnx_slice_index_types(onnx_path):
    """Normalize exported integer control tensors to int64.

    The legacy PyTorch exporter can emit int32 ``starts``/``ends`` from
    MiniCPM's int32 cumulative sequence lengths while the other Slice inputs
    are int64.  It can also mix int32 and int64 inputs in the ``Concat`` nodes
    used to build cumulative sequence lengths.  ONNX Runtime rejects both
    cases.  Cast only these integer control inputs; model activations and the
    original int32 sequence-length tensors remain unchanged.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(onnx_path, load_external_data=False)
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
    except Exception:
        inferred = model

    tensor_types = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if value.type.HasField("tensor_type"):
            tensor_types[value.name] = value.type.tensor_type.elem_type
    for value in inferred.graph.initializer:
        tensor_types[value.name] = value.data_type

    new_nodes = []
    cast_count = 0

    def append_cast(source, node, input_index, suffix):
        nonlocal cast_count
        cast_output = f"{source}__{suffix}_{node_index}_{input_index}"
        cast_node = helper.make_node(
            "Cast",
            [source],
            [cast_output],
            name=f"{node.name or node.op_type}_cast_{input_index}_{suffix}",
            to=TensorProto.INT64,
        )
        new_nodes.append(cast_node)
        node.input[input_index] = cast_output
        tensor_types[cast_output] = TensorProto.INT64
        cast_count += 1

    for node_index, node in enumerate(model.graph.node):
        if node.op_type == "Slice":
            for input_index in range(1, min(len(node.input), 5)):
                if tensor_types.get(node.input[input_index]) != TensorProto.INT64:
                    append_cast(node.input[input_index], node, input_index, "slice_int64")
        elif node.op_type == "Concat":
            input_types = [tensor_types.get(name) for name in node.input]
            has_mixed_integer_types = (
                TensorProto.INT32 in input_types and TensorProto.INT64 in input_types
            )
            if has_mixed_integer_types:
                for input_index, input_type in enumerate(input_types):
                    if input_type in (TensorProto.INT32, TensorProto.INT64):
                        if input_type != TensorProto.INT64:
                            append_cast(node.input[input_index], node, input_index, "concat_int64")
        new_nodes.append(node)

    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    # External initializer locations are relative to the ONNX file, not to
    # the caller's current working directory.
    previous_cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(onnx_path)) or previous_cwd)
        onnx.checker.check_model(model)
        onnx.save(model, onnx_path)
    finally:
        os.chdir(previous_cwd)
    print(f"Normalized {cast_count} ONNX Slice index inputs to int64")


def _fix_minicpmv4_7_static_vit_merger_if(onnx_path, patch_count):
    """Remove the unsupported batch-squeeze ``If`` from the fixed image graph.

    ``MiniCPMV4_7ViTWindowAttentionMerger`` calls ``squeeze(0)`` on a slice
    whose first dimension is the number of patches.  For the exported single
    448x448 image this is 1024, so the squeeze is a no-op.  The legacy ONNX
    exporter represents that no-op as an ``If`` with a ``Squeeze`` then-branch
    and an ``Identity`` else-branch.  RKNN's constant-folding shape inference
    rejects the branch because the squeeze axis is not statically 1.

    The exporter is intentionally static and only uses this rewrite when the
    fixed patch count is not 1.  The matching else branch is therefore the
    exact computation for this graph and can be wired directly to the
    ``Slice`` output.
    """
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    replacements = {}
    removed = 0

    for node in model.graph.node:
        if node.op_type != "If" or node.name != "/vision_model/vit_merger/If":
            continue
        if patch_count == 1:
            raise ValueError(
                "The static vit_merger squeeze rewrite is only valid when "
                f"patch_count != 1, got {patch_count}"
            )

        branches = {attr.name: attr.g for attr in node.attribute}
        then_branch = branches.get("then_branch")
        else_branch = branches.get("else_branch")
        if then_branch is None or else_branch is None:
            raise ValueError("MiniCPM4.7 vit_merger If is missing a branch")

        squeeze_nodes = [n for n in then_branch.node if n.op_type == "Squeeze"]
        identity_nodes = [n for n in else_branch.node if n.op_type == "Identity"]
        if len(squeeze_nodes) != 1 or len(identity_nodes) != 1:
            raise ValueError(
                "Unexpected MiniCPM4.7 vit_merger If branches; refusing "
                "to apply the static rewrite"
            )

        squeeze_input = squeeze_nodes[0].input[0]
        identity_input = identity_nodes[0].input[0]
        if squeeze_input != identity_input:
            raise ValueError(
                "MiniCPM4.7 vit_merger If branches do not share the same "
                "Slice input"
            )

        for output in node.output:
            replacements[output] = identity_input
        removed += 1

    if not replacements:
        print("MiniCPM4.7 vit_merger static If rewrite: no matching node")
        return

    for node in model.graph.node:
        for index, input_name in enumerate(node.input):
            if input_name in replacements:
                node.input[index] = replacements[input_name]

    kept_nodes = [
        node
        for node in model.graph.node
        if not (node.op_type == "If" and node.name == "/vision_model/vit_merger/If")
    ]
    model.graph.ClearField("node")
    model.graph.node.extend(kept_nodes)

    previous_cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(os.path.abspath(onnx_path)) or previous_cwd)
        onnx.checker.check_model(model)
        onnx.save(model, onnx_path)
    finally:
        os.chdir(previous_cwd)
    print(f"Removed {removed} static MiniCPM4.7 vit_merger If node(s)")


def export_smolvl_vision(model, args):
    class smolvl(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vision_model = model.model.vision_model
            self.connector = model.model.connector
            self.config = model.model.config
            self.pixel_shape = [1, 3, 512, 512]

        def forward(
            self,
            pixel_values: torch.FloatTensor,
        ) -> torch.FloatTensor:
            """
            输入：
                pixel_values: [batch_size * num_images, 3, height, width]
                pixel_attention_mask: [batch_size * num_images, height, width] （可选）

            输出：
                image_hidden_states: [batch_size * num_images, image_seq_len, hidden_size]
            """

            # 如果没有提供 attention mask，则创建全 1 的 mask
            pixel_attention_mask = torch.ones(
                size=(self.pixel_shape[0], self.pixel_shape[2], self.pixel_shape[3]),
                dtype=torch.bool,
                device=pixel_values.device,
            )

            # 构建 patch-level 的 attention mask
            patch_size = self.config.vision_config.patch_size
            patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            # 视觉编码器前向传播
            vision_outputs = self.vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            )
            image_features = vision_outputs.last_hidden_state  # [N, L_vision, D_vision]
            import numpy as np
            np.save("image_features.npy", image_features.cpu().detach().numpy())
            # 通过 connector 投影到文本嵌入空间
            image_hidden_states = self.connector(image_features)  # [N, L_text, D_text]

            return image_hidden_states

    
    model_vision = smolvl(model)
    fake_input = torch.randn(1, 3, 512, 512)
    torch.onnx.export(model_vision,
                    (fake_input),
                    args.export_vision_path,
                    input_names=['pixel'],
                    dynamic_axes={'pixel': {2: 'height', 3: 'width'}},
                    opset_version=17
                    )
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")



def export_qwen2_omni_vision(model, args):
    import numpy as np
    import torch.nn.functional as F

    class Qwen2OmniVision(torch.nn.Module):
        def __init__(self, visual_model, in_h, in_w):
            super(Qwen2OmniVision, self).__init__()
            self.visual = visual_model

            self.grid_t = 1
            self.merge_size = 2
            self.channel = 3
            self.temporal_patch_size = 2
            self.patch_size = 14

            align_size = self.merge_size* self.patch_size
            align_h = (in_h + align_size - 1) // align_size* align_size
            align_w = (in_w + align_size - 1) // align_size* align_size
            self.in_h = align_h
            self.in_w = align_w

        def forward(self, pixel, grid_thw=None):
            patches = pixel.repeat(self.temporal_patch_size, 1, 1, 1)
            h = pixel.shape[2]
            w = pixel.shape[3]
            patches = patches.reshape(self.grid_t, self.temporal_patch_size, self.channel, h//self.merge_size//self.patch_size, self.merge_size, self.patch_size, w//self.merge_size//self.patch_size, self.merge_size, self.patch_size)
            patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flatten_patches = patches.reshape(self.grid_t * (h//self.patch_size) * (w//self.patch_size), self.channel * self.temporal_patch_size * self.patch_size * self.patch_size)

            hidden_states = self.visual.patch_embed(flatten_patches)
            seq_len, _ = hidden_states.size()

            if grid_thw is not None:
                # 动态计算模式（生成预计算文件）
                rotary_pos_emb = self.visual.rot_pos_emb(grid_thw)
                window_index, cu_window_seqlens = self.visual.get_window_index(grid_thw)
                cu_window_seqlens = torch.tensor(
                    cu_window_seqlens,
                    device=hidden_states.device,
                    dtype=torch.int32,
                )
                cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

                rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.visual.spatial_merge_unit, 
                                                      self.visual.spatial_merge_unit, -1)
                rotary_pos_emb = rotary_pos_emb[window_index, :, :]
                rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

                cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
                    dim=0, dtype=torch.int32
                )
                cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
                np.save("./cu_seqlens.npy", cu_seqlens.cpu().detach().numpy())
                np.save("./window_index.npy", window_index.cpu().detach().numpy())
                np.save("./cu_window_seqlens.npy", cu_window_seqlens.cpu().detach().numpy())
                np.save("./rotary_pos_emb.npy", rotary_pos_emb.cpu().detach().numpy())
            else:
                cu_seqlens = torch.from_numpy(np.load("./cu_seqlens.npy")).to(dtype=torch.int32, device=hidden_states.device)
                window_index = torch.from_numpy(np.load("./window_index.npy")).to(dtype=torch.long, device=hidden_states.device)
                cu_window_seqlens = torch.from_numpy(np.load("./cu_window_seqlens.npy")).to(dtype=torch.int32, device=hidden_states.device)
                rotary_pos_emb = torch.from_numpy(np.load("./rotary_pos_emb.npy")).to(dtype=hidden_states.dtype, device=hidden_states.device)

            hidden_states = hidden_states.reshape(seq_len // self.visual.spatial_merge_unit, 
                                                self.visual.spatial_merge_unit, -1)
            hidden_states = hidden_states[window_index, :, :]
            hidden_states = hidden_states.reshape(seq_len, -1)

            for layer_num, blk in enumerate(self.visual.blocks):
                cu_seqlens_now = cu_seqlens if layer_num in self.visual.fullatt_block_indexes else cu_window_seqlens
                hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rotary_pos_emb)

            hidden_states = self.visual.merger(hidden_states)
            reverse_indices = torch.argsort(window_index)
            return hidden_states[reverse_indices, :]

    model = Qwen2OmniVision(model, args.img_size, args.img_size)

    fake_input = torch.randn(1, 3, model.in_h, model.in_w)
    grid_thw = torch.tensor([[1, model.in_h//model.patch_size, model.in_w//model.patch_size]], dtype=torch.int64)
    _ = model(fake_input, grid_thw)

    torch.onnx.export(model,
                    (fake_input,),
                    args.export_vision_path,
                    input_names=['pixel'],
                    opset_version=17)
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")

def export_qwen2_omni_audio(model, args):
    import torch
    from torch import nn
    import math
    def encoder_forward_export(self):
        def enc_tmp_expt(padded_feature=None, padded_mask=None, attention_mask=None):
            padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
            padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

            padded_embed = padded_embed + self.positional_embedding.positional_embedding[
                : padded_embed.shape[1], :
            ].unsqueeze(0).to(padded_embed.dtype)
            hidden_states = padded_embed[0]
            for idx, encoder_layer in enumerate(self.layers):
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                )

                hidden_states = layer_outputs[0]
            
            hidden_states = self.avg_pooler(hidden_states.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose_(0, 1) # Reshape to avoid ONNX ShapeInferenceError
            hidden_states = self.ln_post(hidden_states)
            hidden_states = self.proj(hidden_states)

            return hidden_states
        return enc_tmp_expt

    def layer_forward_export(self):
        def layr_tmp_expt(hidden_states=None, attention_mask=None):
            residual = hidden_states
            hidden_states = self.self_attn_layer_norm(hidden_states)
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = self.fc2(hidden_states)
            hidden_states = residual + hidden_states

            if hidden_states.dtype == torch.float16:
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            outputs = (hidden_states,)

            return outputs
        return layr_tmp_expt

    def attention_forward_export(self):
        def attn_tmp_expt(hidden_states=None, attention_mask=None):
            seq_length, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
            key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
            value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

            query_states = query_states.transpose(0, 1)
            key_states = key_states.transpose(0, 1)
            value_states = value_states.transpose(0, 1)
            attn_weights = torch.matmul(query_states, key_states.transpose(1, 2)) / math.sqrt(self.head_dim)

            attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states).transpose(0, 1).reshape(seq_length, self.embed_dim)

            # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
            # partitioned across GPUs when using tensor-parallelism.

            attn_output = self.out_proj(attn_output)

            return attn_output
        return attn_tmp_expt

    for encoder_layer in model.layers:
        encoder_layer.forward = layer_forward_export(encoder_layer)
        encoder_layer.self_attn.forward = attention_forward_export(encoder_layer.self_attn)
    model.forward = encoder_forward_export(model)

    n_frame = 300
    attn_mask_size = 150
    padded_feature = torch.zeros(1, 128, n_frame, dtype=torch.float32)
    padded_mask = torch.ones(1, 1, n_frame, dtype=torch.float32)
    attention_mask = torch.zeros(1, attn_mask_size, attn_mask_size, dtype=torch.float32)

    input_names = ["padded_feature", "padded_mask", "attention_mask"]
    output_names = ["output"]
    dynamic_axes = {
        'padded_feature': {2: 'n_frame'},
        'padded_mask': {2: 'n_frame'},
        'attention_mask': {1: 'attn_mask_size', 2: 'attn_mask_size'},
    }

    torch.onnx.export(
        model,
        (padded_feature, padded_mask, attention_mask),
        args.export_audio_path,
        opset_version=19,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported to {os.path.abspath(args.export_audio_path)}")

def export_vila_1_5_vision(vision_model, proj_model, args):
    class vila1_5(torch.nn.Module):
        def __init__(self, vision, proj):
            super(vila1_5, self).__init__()
            self.vision = vision
            self.proj = proj


        def forward(self, pixel):
            last_hidden_state = self.vision(pixel)['last_hidden_state']
            vision_embeds = self.proj(last_hidden_state)

            return vision_embeds

    img_size = vision_model.config.image_size

    vila_vision = vila1_5(vision_model, proj_model)

    pixel = torch.randn(1, 3, img_size, img_size)

    out = vila_vision(pixel)

    torch.onnx.export(vila_vision,
                     (pixel),
                     args.export_vision_path,
                     input_names=['pixel'],
                     opset_version=17)
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")

def export_minicpm3o_vision(vlm, args):
    import math
    class minicpm_v_vision(torch.nn.Module):
        def __init__(self, vlm, batch_size, in_h, in_w):
            super(minicpm_v_vision, self).__init__()
            self.vpm = vlm.vpm
            self.resampler = vlm.resampler
            patch_size = vlm.config.patch_size
            num_patches_per_side = vlm.vpm.embeddings.num_patches_per_side
            tgt_sizes = torch.Tensor([[(in_h // patch_size), math.ceil(in_w / patch_size)]]).type(torch.int32)
            patch_attention_mask = torch.ones(
                size=(batch_size, in_h // patch_size, in_w // patch_size),
                dtype=torch.bool, device=vlm.device,
            )
            max_im_h, max_im_w = in_h, in_w
            max_nb_patches_h, max_nb_patches_w = max_im_h // patch_size, max_im_w // patch_size
            boundaries = torch.arange(1 / num_patches_per_side, 1.0, 1 / num_patches_per_side)
            position_ids = torch.full(
                size=(batch_size, max_nb_patches_h * max_nb_patches_w),
                fill_value=0,
            )
            for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
                if tgt_sizes is not None:
                    nb_patches_h = tgt_sizes[batch_idx][0]
                    nb_patches_w = tgt_sizes[batch_idx][1]
                else:
                    nb_patches_h = p_attn_mask[:, 0].sum()
                    nb_patches_w = p_attn_mask[0].sum()

                fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
                fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

                bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
                bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

                pos_ids = (bucket_coords_h[:, None] * num_patches_per_side + bucket_coords_w).flatten()
                position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

                position_ids = position_ids.to(vlm.device)
            self.position_ids = position_ids
            
            patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]
            max_patch_len = torch.max(patch_len)
            key_padding_mask = torch.zeros((batch_size, max_patch_len), dtype=torch.bool, device=vlm.device)
            pos_embed = []
            for i in range(batch_size):
                tgt_h, tgt_w = tgt_sizes[i]
                pos_embed.append(self.resampler.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(torch.float32))  # patches * D
                key_padding_mask[i, patch_len[i]:] = True

            self.pos_embed = torch.nn.utils.rnn.pad_sequence(
                pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D
        
        def forward(self, pixel_values):
            batch_size = pixel_values.size(0)
            # patch embedding
            patch_embeds = self.vpm.embeddings.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(2).transpose(1, 2)
            hidden_states = embeddings + self.vpm.embeddings.position_embedding(self.position_ids)
            # encoder
            encoder_outputs = self.vpm.encoder(inputs_embeds=hidden_states)
            last_hidden_state = encoder_outputs[0]
            last_hidden_state = self.vpm.post_layernorm(last_hidden_state)
            # resampler
            x = self.resampler.kv_proj(last_hidden_state)  # B * L * D
            x = self.resampler.ln_kv(x).permute(1, 0, 2)  # L * B * D

            q = self.resampler.ln_q(self.resampler.query)  # Q * D

            out = self.resampler.attn(
                self.resampler._repeat(q, batch_size),  # Q * B * D
                x + self.pos_embed,  # L * B * D +  L * B * D
                x)[0]
            #  out: Q * B * D
            x = out.permute(1, 0, 2)  # B * Q * D

            x = self.resampler.ln_post(x)
            x = x @ self.resampler.proj
            return x

    img_size = vlm.config.image_size

    minicpmv3_vision = minicpm_v_vision(vlm, 1, img_size, img_size)

    pixel = torch.randn(1, 3, img_size, img_size)

    # out = minicpmv3_vision(pixel)

    torch.onnx.export(minicpmv3_vision,
                     (pixel),
                     args.export_vision_path,
                     input_names=['pixel'],
                     opset_version=17)
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")

def export_fastvlm_vision(model, args):
    class fastvlm(torch.nn.Module):
        def __init__(self, model):
            super(fastvlm, self).__init__()
            self.vision_tower = model.get_vision_tower()
            self.mm_projector = model.get_model().mm_projector

        def forward(self, pixel):
            vision_feats = self.vision_tower(pixel)
            projected = self.mm_projector(vision_feats)
            return projected

    # img_size = model.get_vision_tower().image_processor.size["shortest_edge"] # 1024
    img_size = args.img_size

    fastvlm_vision = fastvlm(model)

    pixel = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(fastvlm_vision,
                     (pixel),
                     args.export_vision_path,
                     input_names=['pixel'],
                     opset_version=17)
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")

def export_janus_pro_vision(model, args):
    class janus(torch.nn.Module):
        def __init__(self, model):
            super(janus, self).__init__()
            self.vision_model = model.vision_model
            self.aligner = model.aligner

        def forward(self, pixel):
            img_embeds = self.aligner(self.vision_model(pixel))
            return img_embeds

    img_size = model.config.vision_config.params.image_size

    janus_vision = janus(model)

    pixel = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(janus_vision,
                     (pixel),
                     args.export_vision_path,
                     input_names=['pixel'],
                     opset_version=17)
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")

def export_internvl_vision(internvl_model, args):
    class InternVLVisionModel(torch.nn.Module):
        """
        从 InternVLChatModel 中分离出的视觉模型。
        包括 vision_model、下采样（如果 downsample_ratio < 1）和 mlp1 投影。
        """
        def __init__(self, vision_model, select_layer, downsample_ratio, mlp1, image_size=448, patch_size=14):
            super().__init__()
            self.vision_model = vision_model  # InternVisionModel
            self.select_layer = select_layer
            self.downsample_ratio = downsample_ratio
            self.mlp1 = mlp1  # nn.Sequential for projection
            self.image_size = image_size
            self.patch_size = patch_size
            self.num_patches = (image_size // patch_size) ** 2  # 用于动态调整
        def pixel_shuffle(self, x, scale_factor=0.5):
            n, w, h, c = x.size()
            # N, W, H, C --> N, W, H * scale, C // scale
            x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
            # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
            x = x.permute(0, 2, 1, 3).contiguous()
            # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
            x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                    int(c / (scale_factor * scale_factor)))
           
            x = x.permute(0, 2, 1, 3).contiguous()
            return x

        def forward(self, pixel_values):
            # 提取视觉特征（基于典型 InternVL extract_feature 实现）
            # Step 1: 从 vision_model 获取隐藏状态
            # vision_outputs = self.vision_model(pixel_values, output_hidden_states=True)
            # image_features = vision_outputs.hidden_states[self.select_layer]
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
            
            # Step 2: 移除 CLS token（假设第一个 token 是 CLS）
            vit_embeds = vit_embeds[:, 1:, :]
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
            vit_embeds = self.mlp1(vit_embeds)
            return vit_embeds
               
    vision_model = internvl_model.vision_model
    select_layer = internvl_model.select_layer
    downsample_ratio = internvl_model.downsample_ratio
    mlp1 = internvl_model.mlp1
    image_size = internvl_model.config.vision_config.image_size
    patch_size = internvl_model.config.vision_config.patch_size
    # 从原有模型提取视觉部分
    model = InternVLVisionModel(vision_model, select_layer, downsample_ratio, mlp1, image_size, patch_size)
    
    # 假输入（动态大小，类似 Qwen）
    in_h = 448  # 可调整
    in_w = 448
    fake_input = torch.randn(1, 3, in_h, in_w)  # [B, C, H, W]
    
    # 导出 ONNX
    torch.onnx.export(
        model,
        fake_input,
        args.export_vision_path,
        input_names=['pixel_values'],
        output_names=['image_features'],
        opset_version=17
    )
    print(f"Exported to {os.path.abspath(args.export_vision_path)}")

def export_gemma4_audio(model, args):
    import torch
    import os

    BATCH_SIZE = 1
    MAX_MEL_FRAMES = 756
    MEL_BINS = 128
    class Gemma4AudioWrapper(torch.nn.Module):
        def __init__(self, audio_tower, embed_audio):
            super().__init__()
            self.audio_tower = audio_tower
            self.embed_audio = embed_audio

        def forward(self, input_features, input_features_mask):
            outputs = self.audio_tower(
                input_features=input_features,
                attention_mask=input_features_mask,
                return_dict=True,
            )

            hidden_states = outputs.last_hidden_state   # [B, T, D]
            output_mask = outputs.attention_mask        # [B, T]

            hidden_states = self.embed_audio(hidden_states)

            return hidden_states

    audio_tower = model.model.audio_tower
    embed_audio = model.model.embed_audio

    audio_tower.eval()

    wrapper = Gemma4AudioWrapper(audio_tower, embed_audio).eval()

    input_features = torch.zeros(
        BATCH_SIZE, MAX_MEL_FRAMES, MEL_BINS, dtype=torch.float32
    )
    input_features_mask = torch.ones(
        BATCH_SIZE, MAX_MEL_FRAMES, dtype=torch.float32
    )

    # ====== 导出 ONNX ======
    torch.onnx.export(
        wrapper,
        (input_features, input_features_mask),
        args.export_audio_path,
        input_names=["input_features", "input_features_mask"],
        output_names=["hidden_states"],
        opset_version=19,
        do_constant_folding=True,
    )

    print(f"Exported to {os.path.abspath(args.export_audio_path)}")


def export_gemma4_audio_streaming(model, args):
    """Export a block-aligned stateful Audio Tower streaming graph."""
    import torch

    BATCH_SIZE = 1
    # The legacy ONNX exporter requires a concrete example input even though
    # the exported Mel time axis is dynamic. RKNN bucket sizes are configured
    # separately by export_streaming_rknn.py.
    MEL_FRAMES = 720
    MEL_BINS = 128
    CACHE_TOKENS = 12
    HIDDEN_SIZE = model.model.audio_tower.config.hidden_size
    NUM_LAYERS = model.model.audio_tower.config.num_hidden_layers
    NUM_HEADS = model.model.audio_tower.config.num_attention_heads
    HEAD_DIM = HIDDEN_SIZE // NUM_HEADS

    class Gemma4AudioStreamingWrapper(torch.nn.Module):
        def __init__(self, audio_tower, embed_audio):
            super().__init__()
            self.audio_tower = audio_tower
            self.embed_audio = embed_audio

        def forward(self, input_features, input_features_mask, conv_state_0, conv_state_1,
                    past_mask, past_key, past_value, past_conv):
            layer_states = []
            for i in range(NUM_LAYERS):
                layer_states.extend([past_key[:, i], past_value[:, i], past_conv[:, i]])
            hidden_states, output_mask, next_states = self.audio_tower.forward_stream(
                input_features,
                input_features_mask,
                conv_state_0,
                conv_state_1,
                past_mask,
                *layer_states,
            )
            hidden_states = self.embed_audio(hidden_states)
            next_key = torch.stack([next_states[2 + 3*i] for i in range(NUM_LAYERS)], dim=1)
            next_value = torch.stack([next_states[3 + 3*i] for i in range(NUM_LAYERS)], dim=1)
            next_conv = torch.stack([next_states[4 + 3*i] for i in range(NUM_LAYERS)], dim=1)
            return (hidden_states, output_mask, next_states[0], next_states[1], next_key, next_value, next_conv)

    audio_tower = model.model.audio_tower.eval()
    wrapper = Gemma4AudioStreamingWrapper(audio_tower, model.model.embed_audio).eval()

    inputs = [
        torch.zeros(BATCH_SIZE, MEL_FRAMES, MEL_BINS, dtype=torch.float32),
        torch.ones(BATCH_SIZE, MEL_FRAMES, dtype=torch.float32),
        torch.zeros(BATCH_SIZE, 1, 1, MEL_BINS, dtype=torch.float32),
        torch.zeros(BATCH_SIZE, 128, 1, 64, dtype=torch.float32),
        torch.ones(BATCH_SIZE, CACHE_TOKENS, dtype=torch.float32),
        torch.ones(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, CACHE_TOKENS, HEAD_DIM, dtype=torch.float32),
        torch.ones(BATCH_SIZE, NUM_LAYERS, NUM_HEADS, CACHE_TOKENS, HEAD_DIM, dtype=torch.float32),
        torch.ones(BATCH_SIZE, NUM_LAYERS, HIDDEN_SIZE, 4, dtype=torch.float32),
    ]
    input_names = [
        "input_features", "input_features_mask", "conv_state_0", "conv_state_1", "audio_history_mask",
        "audio_history_key", "audio_history_value", "past_conv"
    ]
    output_names = ["hidden_states", "output_mask", "next_conv_state_0", "next_conv_state_1", "audio_current_key", "audio_current_value", "next_conv"]

    torch.onnx.export(
        wrapper,
        tuple(inputs),
        args.export_audio_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=19,
        do_constant_folding=True,
        dynamic_axes={
            "input_features": {1: "mel_frames"},
            "input_features_mask": {1: "mel_frames"},
            "hidden_states": {1: "audio_tokens"},
            "output_mask": {1: "audio_tokens"},
        },
        dynamo=False,
    )
    print(f"Exported streaming Audio ONNX to {os.path.abspath(args.export_audio_path)}")
