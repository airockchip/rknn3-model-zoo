import inspect
import os
import sys
import types

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segment_planner import add_plan_arguments, prepare_plan, write_plan_json

from py_utils.export_llm_helper import (
    _estimate_module_weight_size,
    _resolve_segment_layer_counts,
    causal_llm_to_onnx,
    check_gptq,
    export_embed_weight,
    export_llm_config,
    export_tokenizer,
    register_bitwise_right_shift,
    update_config,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


prompt = "RKLLM"
chat_context = {
    "messages": [
        {"role": "user", "content": prompt},
    ],
    "add_generation_prompt": True,
    "enable_thinking": True,
}

MOE_CUSTOM_DOMAIN = "rknn"
MOE_CUSTOM_OP = "exUnifiedMoE"


def _qwen3_text_config(config):
    """Return the text config for text-only and wrapped checkpoints."""
    text_config = getattr(config, "text_config", None)
    return config if text_config is None else text_config


def _is_qwen3_moe_config(config):
    """Recognize Qwen3-MoE without confusing it with dense Qwen3."""
    configs = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config is not config:
        configs.append(text_config)

    for candidate in configs:
        model_type = str(getattr(candidate, "model_type", ""))
        if model_type in {"qwen3_moe", "qwen3_moe_text"}:
            return True
        architectures = getattr(candidate, "architectures", None) or []
        if any("Qwen3Moe" in str(architecture) for architecture in architectures):
            return True
    return False


def _is_qwen3_moe_block(mlp):
    """Detect both packed and upstream-Transformers Qwen3 sparse layers.

    The RKNN fork of Qwen3 exposes ``experts.gate_up_proj`` and
    ``experts.down_proj`` as packed tensors, while recent Transformers releases
    expose ``experts`` as a ModuleList of MLPs with separate gate/up/down
    projections. Thinking-2507 uses the latter layout.
    """
    if mlp is None:
        return False
    if getattr(mlp, "_rknn_qwen3_moe_patched", False):
        return True
    experts = getattr(mlp, "experts", None)
    gate = getattr(mlp, "gate", None)
    if experts is None or gate is None or not hasattr(gate, "weight"):
        return False
    if hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj"):
        return True
    try:
        first_expert = experts[0]
    except (IndexError, KeyError, TypeError):
        return False
    return all(
        hasattr(first_expert, name)
        and hasattr(getattr(first_expert, name), "weight")
        for name in ("gate_proj", "up_proj", "down_proj")
    )


def _make_qwen3_moe_function(moe, config, layer_id):
    """Build an eager-compatible Function that emits exUnifiedMoE.

    Qwen3 stores all routed expert parameters in packed tensors:

      gate.weight:          [num_experts, hidden_size]
      gate_up_proj:         [num_experts, 2 * moe_intermediate_size, hidden_size]
      down_proj:            [num_experts, hidden_size, moe_intermediate_size]

    ``exUnifiedMoE`` uses mathematical ``[in, out]`` weights.  The symbolic
    function therefore transposes the three constant inputs while the eager
    implementation keeps the original PyTorch Linear layout.
    """
    experts = moe.experts
    router = moe.gate

    n_experts = int(
        getattr(experts, "num_experts", getattr(config, "num_experts", 0))
    )
    top_k = int(
        getattr(router, "top_k", getattr(config, "num_experts_per_tok", 0))
    )
    normalize_topk = bool(
        getattr(
            router,
            "norm_topk_prob",
            getattr(config, "norm_topk_prob", False),
        )
    )

    packed_experts = hasattr(experts, "gate_up_proj") and hasattr(
        experts, "down_proj"
    )
    if packed_experts:
        gate_up_weight = experts.gate_up_proj
        down_weight = experts.down_proj
    else:
        # Upstream Transformers stores one Qwen3MoeMLP per expert. Pack the
        # three Linear matrices into the layout expected by exUnifiedMoE while
        # retaining them as registered, non-trainable Parameters so the legacy
        # ONNX exporter emits proper initializers rather than graph constants.
        expert_list = list(experts)
        if not expert_list:
            raise ValueError("Qwen3-MoE expert ModuleList is empty")
        gate_up_weight = torch.nn.Parameter(
            torch.stack(
                [
                    torch.cat(
                        (expert.gate_proj.weight, expert.up_proj.weight), dim=0
                    )
                    for expert in expert_list
                ],
                dim=0,
            ).detach(),
            requires_grad=False,
        )
        down_weight = torch.nn.Parameter(
            torch.stack(
                [expert.down_proj.weight for expert in expert_list], dim=0
            ).detach(),
            requires_grad=False,
        )
        # Register the packed tensors on the layer. The original ModuleList
        # is no longer needed by the patched forward and can be released to
        # avoid holding a second full copy of all expert weights in memory.
        moe.register_parameter("_rknn_qwen3_gate_up_proj", gate_up_weight)
        moe.register_parameter("_rknn_qwen3_down_proj", down_weight)
        del moe.experts
        experts = moe

    gate_up_shape = tuple(int(dim) for dim in gate_up_weight.shape)
    down_shape = tuple(int(dim) for dim in down_weight.shape)
    router_shape = tuple(int(dim) for dim in router.weight.shape)
    if len(gate_up_shape) != 3 or len(down_shape) != 3 or len(router_shape) != 2:
        raise ValueError(
            "Qwen3-MoE expects packed expert weights with ranks [3, 3, 2], "
            f"got [{len(gate_up_shape)}, {len(down_shape)}, {len(router_shape)}]"
        )

    if gate_up_shape[0] != n_experts or down_shape[0] != n_experts:
        raise ValueError(
            "Qwen3-MoE expert count does not match config: "
            f"config={n_experts}, gate_up={gate_up_shape[0]}, down={down_shape[0]}"
        )
    if gate_up_shape[1] % 2 != 0:
        raise ValueError(
            f"Qwen3-MoE gate_up_proj middle dimension must be even, got {gate_up_shape}"
        )

    hidden_size = gate_up_shape[2]
    intermediate_size = gate_up_shape[1] // 2
    if down_shape[1:] != (hidden_size, intermediate_size):
        raise ValueError(
            "Qwen3-MoE packed expert shapes are inconsistent: "
            f"gate_up={gate_up_shape}, down={down_shape}"
        )
    if router_shape != (n_experts, hidden_size):
        raise ValueError(
            "Qwen3-MoE router weight shape is inconsistent with expert weights: "
            f"router={router_shape}, expected={(n_experts, hidden_size)}"
        )
    if not 1 <= top_k <= n_experts:
        raise ValueError(
            f"Qwen3-MoE num_experts_per_tok must be in [1, {n_experts}], got {top_k}"
        )

    activation = getattr(config, "hidden_act", "silu")
    activation = getattr(activation, "__name__", activation)
    activation = str(activation).lower()
    if activation not in {"silu", "swish"}:
        raise NotImplementedError(
            "Qwen3-MoE exUnifiedMoE export supports only silu/swish, "
            f"got {activation!r}"
        )

    # Passing complete packed parameters to apply() is important: tracing the
    # Python expert loop alone would retain only experts selected by the dummy
    # input and silently discard the other expert weights.
    parameters = (
        router.weight,
        gate_up_weight,
        down_weight,
    )

    def eager_routed(hidden_states, weights):
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_size)
        flat_states_fp32 = flat_states.to(torch.float32)
        router_weight, gate_up_weight, down_weight = weights

        # This is Qwen3MoeTopKRouter: softmax over all experts, then greedy
        # top-k.  Normalization is controlled by norm_topk_prob in the config.
        router_logits = F.linear(
            flat_states_fp32,
            router_weight.to(torch.float32),
        )
        router_scores = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(
            router_scores,
            k=top_k,
            dim=-1,
            sorted=True,
        )
        if normalize_topk:
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )

        routed = torch.zeros_like(flat_states_fp32)
        for expert_id in range(n_experts):
            token_weight = (topk_weights * (topk_indices == expert_id)).sum(dim=-1)
            token_indices = torch.nonzero(token_weight, as_tuple=False).flatten()
            if token_indices.numel() == 0:
                continue

            expert_input = flat_states_fp32.index_select(0, token_indices)
            gate_up = F.linear(
                expert_input,
                gate_up_weight[expert_id].to(torch.float32),
            )
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(
                F.silu(gate) * up,
                down_weight[expert_id].to(torch.float32),
            )
            expert_output = expert_output * token_weight.index_select(
                0, token_indices
            ).unsqueeze(-1)
            routed.index_add_(0, token_indices, expert_output)

        return routed.to(hidden_states.dtype).reshape(original_shape)

    class Qwen3MoEFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden_states, *weights):
            del ctx
            return eager_routed(hidden_states, weights)

        @staticmethod
        def symbolic(graph, hidden_states, *weights):
            # PyTorch Linear stores [out, in], while exUnifiedMoE consumes
            # mathematical [in, out]. Preserve the expert batch dimension.
            router_weight = graph.op("Transpose", weights[0], perm_i=[1, 0])
            gate_up_weight = graph.op(
                "Transpose", weights[1], perm_i=[0, 2, 1]
            )
            down_weight = graph.op("Transpose", weights[2], perm_i=[0, 2, 1])

            attributes = dict(
                layer_id_i=layer_id,
                n_routed_experts_i=n_experts,
                num_experts_per_token_i=top_k,
                topk_method_s="greedy",
                scoring_func_s="softmax",
                routed_scaling_factor_f=1.0,
                expert_weight_packed_s="batch",
                gate_up_packed_i=1,
                router_bias_i=0,
                gate_bias_i=0,
                up_bias_i=0,
                down_bias_i=0,
                activation_s=activation,
                routed_scores_before_ffn_i=0,
                outputs=1,
            )
            if normalize_topk:
                attributes["norm_selected_score_s"] = "reducesum"

            output = graph.op(
                f"{MOE_CUSTOM_DOMAIN}::{MOE_CUSTOM_OP}",
                hidden_states,
                router_weight,
                gate_up_weight,
                down_weight,
                **attributes,
            )
            output.setType(hidden_states.type())
            return output

    Qwen3MoEFunction.__name__ = f"Qwen3MoEFunctionLayer{layer_id}"
    return Qwen3MoEFunction, parameters, {
        "layer_id": layer_id,
        "n_experts": n_experts,
        "top_k": top_k,
        "normalize_topk": normalize_topk,
        "intermediate_size": intermediate_size,
    }


def patch_qwen3_moe_for_onnx(model, layer_ids=None):
    """Replace Qwen3 routed-MoE blocks with exUnifiedMoE nodes.

    ``layer_ids`` can limit patching to an exported prefix. This is useful for
    block-level validation because the model still contains all checkpoint
    weights, but only the requested segment needs packed expert tensors.
    """
    text_model = getattr(model, "model", None)
    if text_model is None or not hasattr(text_model, "layers"):
        raise TypeError("Qwen3-MoE export expects a causal LM with model.layers")

    manifests = []
    for layer_id, layer in enumerate(text_model.layers):
        if layer_ids is not None and layer_id not in layer_ids:
            continue
        moe = getattr(layer, "mlp", None)
        if not _is_qwen3_moe_block(moe):
            continue

        if getattr(moe, "_rknn_qwen3_moe_patched", False):
            manifests.append(moe._rknn_qwen3_moe_manifest)
            continue

        function, parameters, manifest = _make_qwen3_moe_function(
            moe,
            text_model.config,
            layer_id,
        )

        def patched_forward(
            self, hidden_states, *, _function=function, _parameters=parameters
        ):
            return _function.apply(hidden_states, *_parameters)

        moe.forward = types.MethodType(patched_forward, moe)
        moe._rknn_qwen3_moe_patched = True
        moe._rknn_qwen3_moe_manifest = manifest
        manifests.append(manifest)

    if not manifests:
        raise RuntimeError("No Qwen3-MoE layers found in the model")

    expert_counts = sorted({item["n_experts"] for item in manifests})
    topk_counts = sorted({item["top_k"] for item in manifests})
    if len(expert_counts) != 1 or len(topk_counts) != 1:
        raise ValueError(
            "Qwen3-MoE requires one expert/top-k configuration across layers, "
            f"got experts={expert_counts}, top_k={topk_counts}"
        )

    print(
        f"Patched {len(manifests)} Qwen3-MoE layers as "
        f"{MOE_CUSTOM_DOMAIN}::{MOE_CUSTOM_OP} "
        f"(batch-packed experts={expert_counts[0]}, top_k={topk_counts[0]})"
    )
    return manifests


def _set_qwen3_eager_attention(model):
    """Make attention export deterministic and ONNX-friendly."""
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "_attn_implementation"):
        config._attn_implementation = "eager"
    text_model = getattr(model, "model", None)
    if text_model is None:
        return
    for layer in getattr(text_model, "layers", []):
        attention = getattr(layer, "self_attn", None)
        attention_config = getattr(attention, "config", None)
        if attention_config is not None and hasattr(
            attention_config, "_attn_implementation"
        ):
            attention_config._attn_implementation = "eager"


def _output_logits(model_output):
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output.logits


class _Qwen3CausalLMExportWrapper(torch.nn.Module):
    """Keep the single-segment MoE graph's public inputs compact and stable."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, position_ids, num_logits_to_keep):
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=num_logits_to_keep,
        )
        return _output_logits(model_output)


def causal_qwen3_moe_llm_to_onnx(model, args):
    """Export a single Qwen3-MoE ONNX graph with a direct custom op."""
    args.prompt_size = getattr(args, "prompt_size", 64)
    model.eval()
    model.float()
    _set_qwen3_eager_attention(model)

    in_len = args.prompt_size
    inputs = (
        torch.zeros((1, in_len), dtype=torch.long),
        torch.ones((1, in_len), dtype=torch.float32),
        torch.arange(0, in_len, dtype=torch.long).unsqueeze(0),
        torch.tensor([in_len - 1], dtype=torch.long),
    )
    input_names = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "num_logits_to_keep",
    ]
    dynamic_axes = {
        "input_ids": {1: "sequence"},
        "attention_mask": {1: "sequence"},
        "position_ids": {1: "sequence"},
    }

    export_dir = os.path.dirname(args.export_llm_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            _Qwen3CausalLMExportWrapper(model),
            inputs,
            args.export_llm_path,
            export_params=True,
            external_data=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            custom_opsets={MOE_CUSTOM_DOMAIN: 1},
            dynamo=False,
        )

    print(f"Exported Qwen3-MoE to {os.path.abspath(args.export_llm_path)}")


class Qwen3SegmentWrapper(torch.nn.Module):
    """Wrap contiguous Qwen3 decoder layers for multicard pipeline export."""

    def __init__(
        self,
        text_model,
        layers,
        norm=None,
        lm_head=None,
        is_last_segment=False,
    ):
        super().__init__()
        if not layers:
            raise ValueError("Qwen3 segment must contain at least one decoder layer")
        self.layers = torch.nn.ModuleList(layers)
        self.rotary_emb = text_model.rotary_emb
        self.norm = norm
        self.lm_head = lm_head
        self.is_last_segment = is_last_segment
        self.config = text_model.config

        layer_forward = inspect.signature(layers[0].forward).parameters
        self._layer_forward_params = set(layer_forward)
        if "past_key_values" in layer_forward:
            self._past_key_arg = "past_key_values"
        elif "past_key_value" in layer_forward:
            self._past_key_arg = "past_key_value"
        else:
            self._past_key_arg = None

    @staticmethod
    def _make_causal_mask(hidden_states, attention_mask, sliding_window=None):
        """Build a dynamic additive mask for eager Qwen3 attention.

        Calling the Transformers mask factory during tracing can turn the
        dummy prompt length into a Python constant.  The segmented graph must
        accept every sequence length supported by RKNN, so derive the mask
        entirely from tensor-shaped values here.  Qwen3 segments do not carry
        a Python DynamicCache; the runtime owns the attention cache.
        """
        device = hidden_states.device
        dtype = hidden_states.dtype
        positions = torch.arange(
            hidden_states.shape[1],
            dtype=torch.long,
            device=device,
        )
        q_position = positions.unsqueeze(1)
        kv_position = positions.unsqueeze(0)
        allowed = kv_position <= q_position
        if sliding_window is not None:
            allowed = allowed & (
                (q_position - kv_position) < int(sliding_window)
            )
        allowed = allowed.unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            padding_allowed = attention_mask[:, None, None, :] > 0
            allowed = allowed & padding_allowed

        zero = torch.zeros((), dtype=dtype, device=device)
        min_value = torch.full(
            (),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=device,
        )
        return torch.where(allowed, zero, min_value)

    def forward(
        self,
        input_embeds,
        attention_mask,
        position_ids,
        num_logits_to_keep,
    ):
        hidden_states = input_embeds
        batch_size, seq_len = hidden_states.shape[:2]
        del batch_size

        cache_position = torch.arange(
            seq_len,
            dtype=torch.long,
            device=hidden_states.device,
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask_mapping = {
            "full_attention": self._make_causal_mask(
                hidden_states,
                attention_mask,
            ),
        }
        attention_types = {
            getattr(layer, "attention_type", "full_attention")
            for layer in self.layers
        }
        if "sliding_attention" in attention_types:
            causal_mask_mapping["sliding_attention"] = (
                self._make_causal_mask(
                    hidden_states,
                    attention_mask,
                    getattr(self.config, "sliding_window", None),
                )
            )

        for layer in self.layers:
            attention_type = getattr(layer, "attention_type", "full_attention")
            layer_kwargs = {
                "attention_mask": causal_mask_mapping.get(
                    attention_type,
                    causal_mask_mapping["full_attention"],
                ),
                "position_ids": position_ids,
            }
            if self._past_key_arg is not None:
                layer_kwargs[self._past_key_arg] = None
            if "use_cache" in self._layer_forward_params:
                layer_kwargs["use_cache"] = False
            if "cache_position" in self._layer_forward_params:
                layer_kwargs["cache_position"] = cache_position
            if "position_embeddings" in self._layer_forward_params:
                layer_kwargs["position_embeddings"] = position_embeddings

            layer_output = layer(hidden_states, **layer_kwargs)
            if isinstance(layer_output, (tuple, list)):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

        if self.is_last_segment:
            if self.norm is None or self.lm_head is None:
                raise RuntimeError(
                    "The last Qwen3 segment must contain norm and lm_head"
                )
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.index_select(
                dim=1,
                index=num_logits_to_keep,
            )
            hidden_states = self.lm_head(hidden_states)

        return hidden_states


def _verify_qwen3_moe_nodes(
    onnx_path,
    expected_layers,
    expected_experts,
    expected_top_k,
    normalize_topk,
):
    """Check that every exported sparse layer retained all packed experts."""
    import onnx

    graph = onnx.load(onnx_path, load_external_data=False).graph
    nodes = [
        node
        for node in graph.node
        if node.domain == MOE_CUSTOM_DOMAIN and node.op_type == MOE_CUSTOM_OP
    ]
    for node_index, node in enumerate(nodes):
        attributes = {attribute.name: attribute for attribute in node.attribute}
        expert_attr = attributes.get("n_routed_experts")
        if expert_attr is None or int(expert_attr.i) != expected_experts:
            actual = None if expert_attr is None else int(expert_attr.i)
            raise RuntimeError(
                f"MoE node {node_index} in {onnx_path} declares {actual} experts; "
                f"expected {expected_experts}"
            )
        top_k_attr = attributes.get("num_experts_per_token")
        if top_k_attr is None or int(top_k_attr.i) != expected_top_k:
            actual = None if top_k_attr is None else int(top_k_attr.i)
            raise RuntimeError(
                f"MoE node {node_index} in {onnx_path} declares top_k={actual}; "
                f"expected {expected_top_k}"
            )
        if len(node.input) != 4:
            raise RuntimeError(
                f"MoE node {node_index} in {onnx_path} has {len(node.input)} inputs; "
                "expected 4 for batch-packed Qwen3 experts"
            )
        packed = attributes.get("expert_weight_packed")
        if packed is None or packed.s.decode() != "batch":
            actual = None if packed is None else packed.s.decode()
            raise RuntimeError(
                f"MoE node {node_index} in {onnx_path} uses expert packing "
                f"{actual!r}; expected 'batch'"
            )
        gate_up_packed = attributes.get("gate_up_packed")
        if gate_up_packed is None or int(gate_up_packed.i) != 1:
            actual = None if gate_up_packed is None else int(gate_up_packed.i)
            raise RuntimeError(
                f"MoE node {node_index} in {onnx_path} has gate_up_packed={actual}; "
                "expected 1"
            )

    if normalize_topk:
        for node in nodes:
            attributes = {attribute.name: attribute for attribute in node.attribute}
            norm_attr = attributes.get("norm_selected_score")
            if norm_attr is None or norm_attr.s.decode() != "reducesum":
                raise RuntimeError(
                    f"MoE node in {onnx_path} is missing norm_selected_score=reducesum"
                )
    elif any(
        attribute.name == "norm_selected_score"
        for node in nodes
        for attribute in node.attribute
    ):
        raise RuntimeError(
            f"MoE graph {onnx_path} unexpectedly normalizes Qwen3 top-k scores"
        )

    if len(nodes) != expected_layers:
        raise RuntimeError(
            f"Expected {expected_layers} {MOE_CUSTOM_DOMAIN}::{MOE_CUSTOM_OP} "
            f"nodes in {onnx_path}, found {len(nodes)}"
        )
    print(
        f"Verified {len(nodes)} {MOE_CUSTOM_DOMAIN}::{MOE_CUSTOM_OP} nodes in "
        f"{onnx_path}; each carries all {expected_experts} routed experts"
    )


def causal_qwen3_llm_to_onnx_multi_segment(model, args, moe_manifests=None):
    """Export Qwen3 as contiguous ONNX segments for multicard inference."""
    text_model = getattr(model, "model", None)
    if text_model is None or not hasattr(text_model, "layers"):
        raise TypeError("Qwen3 multicard export expects a causal LM with model.layers")

    all_layers = list(text_model.layers)
    num_total_layers = len(all_layers)
    if num_total_layers == 0:
        raise ValueError("Qwen3 model has no decoder layers")

    args.prompt_size = getattr(args, "prompt_size", 64)
    in_len = args.prompt_size
    num_segments = int(getattr(args, "num_segments", 0) or 0)
    if num_segments <= 0:
        num_segments = max(1, (num_total_layers + 7) // 8)
    num_segments = min(max(num_segments, 1), num_total_layers)

    quantization_config = getattr(model.config, "quantization_config", None)
    is_quantized_model = quantization_config is not None
    if is_quantized_model:
        q_config = quantization_config
        if not check_gptq(q_config.bits, q_config.group_size):
            raise ValueError(
                "Only W4A16 GPTQ models with group_size in "
                "{32, 64, 128, -1} are supported"
            )
        register_bitwise_right_shift()
    else:
        model.float()
    model.eval()
    _set_qwen3_eager_attention(model)

    layer_weight_sizes = [
        _estimate_module_weight_size(layer, 4)
        for layer in all_layers
    ]
    model_norm = text_model.norm
    lm_head = getattr(model, "lm_head", None)
    last_segment_extra_size = (
        _estimate_module_weight_size(
            model_norm,
            16,
            metadata_bytes_per_group=0,
        )
        + _estimate_module_weight_size(lm_head, 6)
    )
    prepare_plan(
        "qwen3", text_model.config, layer_weight_sizes,
        last_segment_extra_size, args,
    )
    segment_layer_counts = _resolve_segment_layer_counts(
        layer_weight_sizes,
        num_segments,
        last_segment_extra_size,
        getattr(args, "segment_layer_counts", None),
    )
    num_segments = len(segment_layer_counts)

    stop_after_segment = getattr(args, "stop_after_segment", None)
    if stop_after_segment is not None and not 0 <= stop_after_segment < num_segments:
        raise ValueError(
            "--stop_after_segment must be within the exported segment range: "
            f"0 <= {stop_after_segment} < {num_segments}"
        )
    segment_weight_sizes = []
    layer_start = 0
    for segment_index, layer_count in enumerate(segment_layer_counts):
        layer_end = layer_start + layer_count
        segment_size = sum(layer_weight_sizes[layer_start:layer_end])
        if segment_index == num_segments - 1:
            segment_size += last_segment_extra_size
        segment_weight_sizes.append(segment_size)
        layer_start = layer_end

    print(f"Total layers: {num_total_layers}, Segments: {num_segments}")
    print(f"Layer distribution: {segment_layer_counts}")
    print(
        "Theoretical quantized weight distribution (MiB, INT4 blocks / INT6 lm_head): "
        f"{[round(size / (1024 ** 2), 2) for size in segment_weight_sizes]}"
    )

    dummy_input_ids = torch.zeros((1, in_len), dtype=torch.long)
    with torch.no_grad():
        inputs_embeds = text_model.embed_tokens(dummy_input_ids)
    attention_mask = torch.ones((1, in_len), dtype=torch.float32)
    position_ids = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)
    num_logits_to_keep = torch.tensor([in_len - 1], dtype=torch.long)

    base_path = args.export_llm_path
    base_dir = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)

    input_names = [
        "input_embeds",
        "attention_mask",
        "position_ids",
        "num_logits_to_keep",
    ]
    dynamic_axes = {
        "input_embeds": {1: "sequence"},
        "attention_mask": {1: "sequence"},
        "position_ids": {1: "sequence"},
    }

    expected_moe = None
    if moe_manifests:
        expected_moe = {
            "n_experts": moe_manifests[0]["n_experts"],
            "top_k": moe_manifests[0]["top_k"],
            "normalize_topk": moe_manifests[0]["normalize_topk"],
        }

    layer_start = 0
    for segment_index, layer_count in enumerate(segment_layer_counts):
        layer_end = layer_start + layer_count
        is_last = segment_index == num_segments - 1
        segment_layers = all_layers[layer_start:layer_end]
        segment_model = Qwen3SegmentWrapper(
            text_model=text_model,
            layers=segment_layers,
            norm=model_norm if is_last else None,
            lm_head=lm_head if is_last else None,
            is_last_segment=is_last,
        )
        segment_model.eval()
        if not is_quantized_model:
            segment_model.float()

        if num_segments == 1:
            segment_path = base_path
        else:
            segment_dir = os.path.join(base_dir, f"seg{segment_index}")
            os.makedirs(segment_dir, exist_ok=True)
            segment_path = os.path.join(
                segment_dir,
                f"{base_name}_seg{segment_index}.onnx",
            )

        print(
            f"Exporting segment {segment_index}: layers {layer_start}-{layer_end - 1} "
            f"({layer_count} layers)"
            f"{' + norm + lm_head' if is_last else ''} -> {segment_path}"
        )
        with torch.no_grad():
            torch.onnx.export(
                segment_model,
                (
                    inputs_embeds,
                    attention_mask,
                    position_ids,
                    num_logits_to_keep,
                ),
                segment_path,
                export_params=True,
                external_data=True,
                opset_version=19,
                do_constant_folding=True,
                input_names=input_names,
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                custom_opsets={MOE_CUSTOM_DOMAIN: 1},
                dynamo=False,
            )

        segment_moe_layers = sum(
            _is_qwen3_moe_block(getattr(layer, "mlp", None))
            for layer in segment_layers
        )
        if expected_moe is not None:
            _verify_qwen3_moe_nodes(
                segment_path,
                expected_layers=segment_moe_layers,
                expected_experts=expected_moe["n_experts"],
                expected_top_k=expected_moe["top_k"],
                normalize_topk=expected_moe["normalize_topk"],
            )
        elif segment_moe_layers:
            raise RuntimeError(
                "Qwen3-MoE layers were found during segment export but no MoE "
                "manifest was supplied"
            )

        print(f"  Segment {segment_index} exported to {os.path.abspath(segment_path)}")
        layer_start = layer_end
        if stop_after_segment is not None and segment_index >= stop_after_segment:
            print(
                f"Stopping after segment {segment_index} as requested by "
                "--stop_after_segment"
            )
            break

    exported_segments = segment_index + 1
    if stop_after_segment is None:
        print(f"All {num_segments} segments exported successfully.")
    else:
        print(
            f"Exported requested prefix: {exported_segments}/{num_segments} segments."
        )


def _load_model(args, config, tokenizer, is_moe_config):
    """Load and optionally GRQ-quantize the model, matching Qwen3's exporter."""
    if args.quant and is_moe_config:
        raise ValueError(
            "GRQ quantization is not supported together with Qwen3-MoE "
            "exUnifiedMoE export"
        )

    if args.quant and torch.cuda.is_available():
        attention_impl = None
        try:
            import flash_attn  # noqa: F401

            attention_impl = "flash_attention_2"
            print("Use FlashAttention2")
        except Exception as exc:
            print(
                f"FlashAttention2 is unavailable ({exc}), "
                "falling back to standard attention."
            )

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "config": config,
        }
        if attention_impl is not None:
            load_kwargs["attn_implementation"] = attention_impl
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)

        from rknn.quantization.api import RKQuantizer

        quant_tool = RKQuantizer(verbose=True)
        ret = quant_tool.load_model(
            model=model,
            tokenizer=tokenizer,
            device="cuda",
            system_prompt=None,
            tools=None,
            system_role="system",
            user_role="user",
        )
        if ret != 0:
            raise RuntimeError(f"Load model into RKQuantizer failed: {ret}")
        quant_model = quant_tool.quantize(
            quantized_dtype="w4a16",
            quantized_method="group32",
            quantized_algorithm="grq",
            dataset=args.cali_dataset,
        )
        return quant_model.cpu(), quant_tool

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    return model, None


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Export Qwen3/Qwen3-MoE LLM as multicard ONNX segments"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen3-30B-A3B",
        help="model path or Hugging Face model name",
    )
    parser.add_argument("--export_llm_path", type=str, default="../../model/llm_qwen3/Qwen3-30B-A3B.onnx")
    parser.add_argument("--quant", action="store_true", help="use GRQ quantization")
    parser.add_argument("--cali_dataset", default="../../../../datasets/llm_quant.json")
    parser.add_argument("--modelscope", action="store_true", help="download from ModelScope")
    parser.add_argument("--multi_segment", action="store_true", help="export multiple ONNX segments")
    parser.add_argument("--num_segments", type=int, default=0, help="segment count; 0 uses the default")
    parser.add_argument("--segment_layer_counts", "--segment-layer-counts", type=int, nargs="+", default=None, help="manual block count per segment; counts must sum to the total layer count and override --num_segments")
    parser.add_argument("--prompt_size", type=int, default=64, help="dummy sequence length used for tracing")
    parser.add_argument("--stop_after_segment", type=int, default=None, help="stop after this zero-based segment; useful for block-level validation")
    add_plan_arguments(parser)
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download

        args.model_path = snapshot_download(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32,
    }
    config = AutoConfig.from_pretrained(args.model_path, **load_kwargs)
    update_config(config, ["use_cache"], False)
    is_moe_config = _is_qwen3_moe_config(config)
    model, quant_tool = _load_model(args, config, tokenizer, is_moe_config)

    text_model = getattr(model, "model", None)
    if text_model is None or not hasattr(text_model, "layers"):
        raise TypeError("Qwen3 export expects a causal LM with model.layers")

    moe_layers = [
        layer
        for layer in text_model.layers
        if _is_qwen3_moe_block(getattr(layer, "mlp", None))
    ]
    is_moe = bool(moe_layers)
    if is_moe_config and not is_moe:
        raise RuntimeError(
            "The config identifies Qwen3-MoE, but no Qwen3-MoE sparse layer was found"
        )
    # For a stopped multi-segment export, only pack the layers that can appear
    # in the requested prefix.  The manual layer-count form is deterministic;
    # when it is not supplied, keep the conservative all-layer behavior.
    patch_layer_ids = None
    if (
        args.multi_segment
        and args.stop_after_segment is not None
        and args.segment_layer_counts is not None
    ):
        counts = [int(count) for count in args.segment_layer_counts]
        prefix_layers = sum(counts[: args.stop_after_segment + 1])
        if prefix_layers > 0:
            patch_layer_ids = set(range(prefix_layers))
    moe_manifests = (
        patch_qwen3_moe_for_onnx(model, layer_ids=patch_layer_ids)
        if is_moe
        else None
    )

    export_dir = os.path.dirname(args.export_llm_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    if args.multi_segment:
        causal_qwen3_llm_to_onnx_multi_segment(model, args, moe_manifests)
    elif is_moe:
        causal_qwen3_moe_llm_to_onnx(model, args)
    else:
        # Preserve the original Qwen3 dense exporter behavior exactly.
        causal_llm_to_onnx(model, args)

    if quant_tool is not None:
        quant_tool.export_op_quantized_dtype(
            args.export_llm_path,
            op_dtype_path="layer_bit.json",
        )

    output_base = os.path.splitext(args.export_llm_path)[0]
    export_llm_config(
        args.model_path,
        output_base + ".config.pkl",
        chat_context,
        prompt,
    )
    export_tokenizer(args.model_path, output_base + ".tokenizer.gguf")
    export_embed_weight(
        model.get_input_embeddings().weight,
        output_base + ".embed.bin",
    )
    if args.multi_segment:
        write_plan_json(args)
