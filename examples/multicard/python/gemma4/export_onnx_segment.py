import sys, os
import inspect
import torch
from collections import UserDict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from segment_planner import prepare_plan
from py_utils.export_llm_helper import (
    _estimate_module_weight_size, _resolve_segment_layer_counts,
    check_gptq, register_bitwise_right_shift,
)
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


def _fuse_embed_scale(embed_tokens):
    if hasattr(embed_tokens, 'fuse_scale'):
        embed_tokens.fuse_scale()
    elif hasattr(embed_tokens, 'embed_scale') and hasattr(embed_tokens, 'scalar_embed_scale'):
        scale = embed_tokens.scalar_embed_scale
        if scale != 1.0:
            with torch.no_grad():
                embed_tokens.weight.mul_(scale)
            print(f"  Fused embed_scale={scale} into embedding weights manually")


def _expand_kv_sharing(text_model):
    patched = 0
    for i, layer in enumerate(text_model.layers):
        attn = layer.self_attn
        if attn.v_proj is None and attn.k_proj is not None:
            v_proj = torch.nn.Linear(
                attn.k_proj.in_features,
                attn.k_proj.out_features,
                bias=attn.k_proj.bias is not None,
            )
            v_proj.weight.data.copy_(attn.k_proj.weight.data)
            if attn.k_proj.bias is not None:
                v_proj.bias.data.copy_(attn.k_proj.bias.data)
            attn.v_proj = v_proj
            attn.use_alternative_attention = False
            patched += 1
            print(f"  Layer {i}: expanded k_proj -> v_proj (sliding={attn.is_sliding})")
    print(f"  Expanded {patched} layers")


def causal_gemma4_unified_to_onnx(model, args):
    if getattr(args, 'multi_segment', False):
        return causal_gemma4_unified_to_onnx_multi_segment(model, args)
    return _causal_gemma4_unified_to_onnx_single(model, args)


def _causal_gemma4_unified_to_onnx_single(model, args):
    import torch

    text_model = model.model.language_model
    text_config = text_model.config

    model.eval().float()
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())

    _fuse_embed_scale(text_model.embed_tokens)
    print("Expanding KV sharing...")
    _expand_kv_sharing(text_model)

    class ModelWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = text_model.embed_tokens
            self.layers = text_model.layers
            self.rotary_emb = text_model.rotary_emb
            self.norm = text_model.norm
            self.lm_head = model.lm_head
            self.config = text_config
            self.unique_layer_types = set(text_config.layer_types)
            self.final_logit_softcapping = text_config.final_logit_softcapping

        def forward(self, input_ids, attention_mask_global, position_ids_global,
                    attention_mask_local, position_ids_local):
            hidden_states = self.embed_tokens(input_ids)

            pos_emb = {}
            for layer_type in self.unique_layer_types:
                if layer_type == "full_attention":
                    pos_emb[layer_type] = self.rotary_emb(hidden_states, position_ids_global, layer_type)
                elif layer_type == "sliding_attention":
                    pos_emb[layer_type] = self.rotary_emb(hidden_states, position_ids_local, layer_type)

            mask_kwargs_global = {
                "config": self.config, "inputs_embeds": hidden_states,
                "attention_mask": attention_mask_global, "past_key_values": None,
                "position_ids": position_ids_global,
            }
            mask_kwargs_local = {
                "config": self.config, "inputs_embeds": hidden_states,
                "attention_mask": attention_mask_local, "past_key_values": None,
                "position_ids": position_ids_local,
            }
            causal_mask_global = create_causal_mask(**mask_kwargs_global)
            causal_mask_local = create_sliding_window_causal_mask(**mask_kwargs_local)

            position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
            position_embeddings_global = pos_emb['full_attention']
            position_embeddings_local = pos_emb['sliding_attention']

            shared_kv_states = UserDict()
            for i, layer in enumerate(self.layers[:self.config.num_hidden_layers]):
                layer_type = self.config.layer_types[i]
                hidden_states = layer(
                    hidden_states, shared_kv_states=shared_kv_states,
                    position_embeddings=(position_embeddings_local if layer_type == "sliding_attention" else position_embeddings_global),
                    attention_mask=(causal_mask_local if layer_type == "sliding_attention" else causal_mask_global),
                    position_ids=position_ids, past_key_values=None,
                )

            hidden_states = self.norm(hidden_states)
            logits = self.lm_head(hidden_states)
            if self.final_logit_softcapping is not None:
                logits = logits / self.final_logit_softcapping
                logits = torch.tanh(logits)
                logits = logits * self.final_logit_softcapping
            return logits

    wrapped_model = ModelWrapper()
    wrapped_model.eval()
    wrapped_model.float()

    args.prompt_size = getattr(args, 'prompt_size', 64)
    args.dynamic_shape = True
    in_len = args.prompt_size

    dummy_input = torch.zeros((1, in_len), dtype=torch.long)
    attention_mask_global = torch.ones((1, in_len), dtype=torch.float)
    position_ids_global = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)
    attention_mask_local = torch.ones((1, in_len), dtype=torch.float)
    position_ids_local = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)

    inputs = (dummy_input, attention_mask_global, position_ids_global, attention_mask_local, position_ids_local)
    input_names = ["input_ids", "attention_mask", "position_ids", "attention_mask_1", "position_ids_1"]
    dynamic_axes = {
        'input_ids': {1: 'sequence'}, 'attention_mask': {1: 'sequence'},
        'position_ids': {1: 'sequence'}, 'attention_mask_1': {1: 'sequence'}, 'position_ids_1': {1: 'sequence'},
    }
    output_names = ["output"]

    logit_keep_keys = ['logits_to_keep', 'num_logits_to_keep']
    logit_keep_key = None
    _forward_func = model.forward
    while hasattr(_forward_func, '__wrapped__'):
        _forward_func = _forward_func.__wrapped__
    _forward_params = set(inspect.signature(_forward_func).parameters.keys())

    for key in logit_keep_keys:
        if key in _forward_params:
            logit_keep_key = key
            break
    if logit_keep_key:
        num_logits_to_keep = torch.tensor(-1, dtype=torch.int32).reshape(1)
        inputs = (*inputs, num_logits_to_keep)
        input_names.append('num_logits_to_keep')

    if hasattr(model.config, 'quantization_config'):
        q_config = model.config.quantization_config
        if not check_gptq(q_config.bits, q_config.group_size):
            print("GRQ model quantization not supported.")
            sys.exit(1)
        register_bitwise_right_shift()
    else:
        model.float()

    print(f"Exporting ONNX to {args.export_llm_path} ...")
    with torch.no_grad():
        torch.onnx.export(wrapped_model, inputs, args.export_llm_path,
                          export_params=True, opset_version=19, do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes, dynamo=True)
    print(f"Exported to {os.path.abspath(args.export_llm_path)}")


def causal_gemma4_unified_to_onnx_multi_segment(model, args):
    import torch

    text_model = model.model.language_model
    text_config = text_model.config

    model.eval().float()
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())

    _fuse_embed_scale(text_model.embed_tokens)
    print("Expanding KV sharing...")
    _expand_kv_sharing(text_model)

    args.prompt_size = getattr(args, 'prompt_size', 64)
    args.dynamic_shape = True
    in_len = args.prompt_size

    num_total_layers = text_config.num_hidden_layers
    num_segments = getattr(args, 'num_segments', None) or 0
    if num_segments <= 0:
        layers_per_segment = 8
        num_segments = max(1, (num_total_layers + layers_per_segment - 1) // layers_per_segment)

    # Match the shared RKNN packed-weight estimate: INT4 blocks and an INT6
    # lm_head, with the final norm kept as FP16.
    block_weight_bits = 4
    lm_head_weight_bits = 6

    layer_weight_sizes = [
        _estimate_module_weight_size(
            layer,
            block_weight_bits,
        )
        for layer in text_model.layers
    ]
    last_segment_extra_size = (
        _estimate_module_weight_size(
            text_model.norm,
            16,
            metadata_bytes_per_group=0,
        )
        + _estimate_module_weight_size(
            model.lm_head,
            lm_head_weight_bits,
        )
    )
    prepare_plan(
        "gemma4", text_config, layer_weight_sizes,
        last_segment_extra_size, args,
    )
    segment_layer_counts = _resolve_segment_layer_counts(
        layer_weight_sizes,
        num_segments,
        last_segment_extra_size,
        getattr(args, 'segment_layer_counts', None),
    )
    num_segments = len(segment_layer_counts)

    stop_after_segment = getattr(args, 'stop_after_segment', None)
    if stop_after_segment is not None and not 0 <= stop_after_segment < num_segments:
        raise ValueError(
            "--stop_after_segment must be within the exported segment range: "
            f"0 <= {stop_after_segment} < {num_segments}"
        )
    segment_weight_sizes = []
    layer_start = 0
    for segment_index, layer_count in enumerate(segment_layer_counts):
        layer_end = layer_start + layer_count
        segment_weight_size = sum(layer_weight_sizes[layer_start:layer_end])
        if segment_index == num_segments - 1:
            segment_weight_size += last_segment_extra_size
        segment_weight_sizes.append(segment_weight_size)
        layer_start = layer_end

    print(f"Total layers: {num_total_layers}, Segments: {num_segments}")
    print(f"Layer distribution: {segment_layer_counts}")
    print(
        "Theoretical quantized weight distribution (MiB, INT4 blocks / INT6 lm_head): "
        f"{[round(size / (1024 ** 2), 2) for size in segment_weight_sizes]}"
    )
    print(
        "Last segment fixed norm + INT6 lm_head weight (MiB): "
        f"{last_segment_extra_size / (1024 ** 2):.2f}"
    )

    dummy_input_ids = torch.zeros((1, in_len), dtype=torch.long)
    with torch.no_grad():
        inputs_embeds = text_model.embed_tokens(dummy_input_ids)

    attention_mask_global = torch.ones((1, in_len), dtype=torch.float)
    position_ids_global = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)
    attention_mask_local = torch.ones((1, in_len), dtype=torch.float)
    position_ids_local = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)

    base_path = args.export_llm_path
    base_dir = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]

    if hasattr(model.config, 'quantization_config'):
        q_config = model.config.quantization_config
        if not check_gptq(q_config.bits, q_config.group_size):
            print("GRQ model quantization not supported.")
            sys.exit(1)
        register_bitwise_right_shift()
    else:
        model.float()

    unique_layer_types = set(text_config.layer_types)
    final_logit_softcapping = text_config.final_logit_softcapping

    logit_keep_keys = ['logits_to_keep', 'num_logits_to_keep']
    logit_keep_key = None
    _forward_func = model.forward
    while hasattr(_forward_func, '__wrapped__'):
        _forward_func = _forward_func.__wrapped__
    _forward_params = set(inspect.signature(_forward_func).parameters.keys())

    for key in logit_keep_keys:
        if key in _forward_params:
            logit_keep_key = key
            break
    seg_inputs = (inputs_embeds, attention_mask_global, position_ids_global, attention_mask_local, position_ids_local)
    seg_input_names = ["input_embeds", "attention_mask", "position_ids", "attention_mask_1", "position_ids_1"]
    if logit_keep_key:
        num_logits_to_keep = torch.tensor(-1, dtype=torch.int32).reshape(1)
        seg_inputs = (*seg_inputs, num_logits_to_keep)
        seg_input_names.append('num_logits_to_keep')

    dynamic_axes = {
        'input_embeds': {1: 'sequence'}, 'attention_mask': {1: 'sequence'},
        'position_ids': {1: 'sequence'}, 'attention_mask_1': {1: 'sequence'}, 'position_ids_1': {1: 'sequence'},
    }
    output_names = ["output"]

    layer_start = 0
    for seg_idx, layer_count in enumerate(segment_layer_counts):
        layer_end = layer_start + layer_count
        is_last = (seg_idx == num_segments - 1)

        segment_layers = list(text_model.layers[layer_start:layer_end])
        seg_norm = text_model.norm if is_last else None
        seg_lm_head = model.lm_head if is_last else None

        class Gemma4SegmentWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(segment_layers)
                self.rotary_emb = text_model.rotary_emb
                self.norm = seg_norm
                self.lm_head = seg_lm_head
                self.is_last_segment = is_last
                self.config = text_config
                self.unique_layer_types = unique_layer_types
                self.final_logit_softcapping = final_logit_softcapping

            def forward(self, hidden_states, attention_mask_global, position_ids_global,
                        attention_mask_local, position_ids_local, num_logits_to_keep):
                pos_emb = {}
                for layer_type in self.unique_layer_types:
                    if layer_type == "full_attention":
                        pos_emb[layer_type] = self.rotary_emb(hidden_states, position_ids_global, layer_type)
                    elif layer_type == "sliding_attention":
                        pos_emb[layer_type] = self.rotary_emb(hidden_states, position_ids_local, layer_type)

                mask_kwargs_global = {
                    "config": self.config, "inputs_embeds": hidden_states,
                    "attention_mask": attention_mask_global, "past_key_values": None,
                    "position_ids": position_ids_global,
                }
                mask_kwargs_local = {
                    "config": self.config, "inputs_embeds": hidden_states,
                    "attention_mask": attention_mask_local, "past_key_values": None,
                    "position_ids": position_ids_local,
                }
                causal_mask_global = create_causal_mask(**mask_kwargs_global)
                causal_mask_local = create_sliding_window_causal_mask(**mask_kwargs_local)

                position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
                position_embeddings_global = pos_emb['full_attention']
                position_embeddings_local = pos_emb['sliding_attention']

                shared_kv_states = UserDict()
                for i, layer in enumerate(self.layers):
                    global_layer_idx = layer_start + i
                    layer_type = self.config.layer_types[global_layer_idx]
                    hidden_states = layer(
                        hidden_states, shared_kv_states=shared_kv_states,
                        position_embeddings=(position_embeddings_local if layer_type == "sliding_attention" else position_embeddings_global),
                        attention_mask=(causal_mask_local if layer_type == "sliding_attention" else causal_mask_global),
                        position_ids=position_ids, past_key_values=None,
                    )

                if self.is_last_segment:
                    if self.norm is not None:
                        hidden_states = self.norm(hidden_states)
                    if self.lm_head is not None:
                        hidden_states = self.lm_head(hidden_states.index_select(dim=1, index=num_logits_to_keep))
                    if self.final_logit_softcapping is not None:
                        hidden_states = hidden_states / self.final_logit_softcapping
                        hidden_states = torch.tanh(hidden_states)
                        hidden_states = hidden_states * self.final_logit_softcapping
                return hidden_states

        segment_model = Gemma4SegmentWrapper()
        segment_model.eval()
        segment_model.float()

        if num_segments == 1:
            seg_path = base_path
        else:
            seg_path = os.path.join(base_dir, f"seg_{seg_idx}", f"{base_name}_seg{seg_idx}.onnx")
            os.makedirs(os.path.dirname(seg_path), exist_ok=True)

        print(f"Exporting segment {seg_idx}: layers {layer_start}-{layer_end-1} ({layer_count} layers)"
              f"{' + norm + lm_head' if is_last else ''} -> {seg_path}")

        with torch.no_grad():
            torch.onnx.export(segment_model, seg_inputs, seg_path,
                              export_params=True, opset_version=19, do_constant_folding=True,
                              input_names=seg_input_names, output_names=output_names,
                              dynamic_axes=dynamic_axes, dynamo=True)

        print(f"  Segment {seg_idx} exported to {os.path.abspath(seg_path)}")
        layer_start = layer_end
        if stop_after_segment is not None and seg_idx >= stop_after_segment:
            print(
                f"Stopping after segment {seg_idx} as requested by "
                "--stop_after_segment"
            )
            break

    exported_segments = seg_idx + 1
    if stop_after_segment is None:
        print(f"All {num_segments} segments exported successfully.")
    else:
        print(
            f"Exported requested prefix: {exported_segments}/{num_segments} segments."
        )
