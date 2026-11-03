import inspect
import os
import sys
import types
from contextlib import contextmanager

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segment_planner import add_plan_arguments, prepare_plan, write_plan_json

from py_utils.export_llm_helper import (
    _resolve_segment_layer_counts,
    _estimate_module_weight_size,
    causal_llm_to_onnx,
    check_gptq,
    export_embed_weight,
    export_llm_config,
    export_tokenizer,
    register_bitwise_right_shift,
    update_config,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_accelerate_available



prompt = "RKLLM"
chat_context = {
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "add_generation_prompt": True,
}


def _configure_model_load_kwargs(load_kwargs, use_low_cpu_mem_usage):
    """Enable Transformers' low-memory loader when its dependency is present."""
    if not use_low_cpu_mem_usage:
        return

    if not is_accelerate_available():
        print(
            "WARNING: low_cpu_mem_usage was requested, but accelerate is not "
            "installed; using the standard loader. Install it with: "
            f"{sys.executable} -m pip install accelerate"
        )
        return

    load_kwargs["low_cpu_mem_usage"] = True
    print("Enabled Transformers low_cpu_mem_usage model loading")


MOE_CUSTOM_DOMAIN = "rknn"
MOE_CUSTOM_OP = "exUnifiedMoE"


def _make_minicpm5_moe_function(moe, layer_id):
    """Create an eager-compatible Function with an exUnifiedMoE symbolic.

    MiniCPM5 dispatches experts with Python-side data-dependent branches. A
    normal ONNX trace therefore records only the experts selected by the dummy
    input. Passing every expert parameter to a custom symbolic node makes the
    complete routed-MoE parameter set part of the ONNX graph while keeping the
    eager forward useful for tracing and PyTorch-side checks.
    """
    config = moe.config
    router = moe.gate
    n_experts = int(config.n_routed_experts)
    top_k = int(config.num_experts_per_tok)
    n_group = int(getattr(config, "n_group", 1) or 1)
    topk_group = int(getattr(config, "topk_group", 1) or 1)
    routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0))

    # The current RKNN lowering uses max-score grouping. The released
    # MiniCPM5 checkpoint uses one group, where its top-2 group score is
    # equivalent to a normal global top-k. Reject other configurations instead
    # of silently changing their routing semantics.
    if n_group != 1 or topk_group != 1:
        raise ValueError(
            "MiniCPM5 exUnifiedMoE export currently requires n_group=1 and "
            f"topk_group=1, got n_group={n_group}, topk_group={topk_group}"
        )

    # Keep the original PyTorch [out, in] layout for eager execution and
    # transpose only in symbolic(). The score correction bias is zero in this
    # checkpoint, but exporting it preserves the router contract.
    parameters = [router.weight, router.e_score_correction_bias]
    parameter_names = ["gate.weight", "gate.e_score_correction_bias"]
    for expert_id, expert in enumerate(moe.experts):
        parameters.extend(
            (expert.gate_proj.weight, expert.up_proj.weight, expert.down_proj.weight)
        )
        parameter_names.extend(
            (
                f"experts.{expert_id}.gate_proj.weight",
                f"experts.{expert_id}.up_proj.weight",
                f"experts.{expert_id}.down_proj.weight",
            )
        )

    def eager_routed(hidden_states, weights):
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_weight, score_select_bias = weights[:2]

        router_logits = F.linear(
            flat_states.to(torch.float32), router_weight.to(torch.float32)
        )
        scores = router_logits.sigmoid()
        selection_scores = scores + score_select_bias.to(torch.float32)
        topk_indices = torch.topk(
            selection_scores, k=top_k, dim=-1, sorted=False
        )[1]
        topk_weights = scores.gather(1, topk_indices)
        if bool(config.norm_topk_prob):
            topk_weights = topk_weights / (
                topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        topk_weights = topk_weights * routed_scaling_factor

        routed = torch.zeros_like(flat_states, dtype=topk_weights.dtype)
        cursor = 2
        for expert_id in range(n_experts):
            gate_weight, up_weight, down_weight = weights[cursor : cursor + 3]
            cursor += 3
            token_weight = (topk_weights * (topk_indices == expert_id)).sum(dim=-1)
            token_indices = torch.nonzero(token_weight, as_tuple=False).flatten()
            if token_indices.numel() == 0:
                continue

            expert_input = flat_states.index_select(0, token_indices)
            expert_output = F.linear(
                F.silu(F.linear(expert_input, gate_weight))
                * F.linear(expert_input, up_weight),
                down_weight,
            )
            routed.index_add_(
                0,
                token_indices,
                expert_output
                * token_weight.index_select(0, token_indices).unsqueeze(-1),
            )

        return routed.to(hidden_states.dtype).reshape(original_shape)

    class MiniCPM5MoEFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden_states, *weights):
            del ctx
            return eager_routed(hidden_states, weights)

        @staticmethod
        def symbolic(graph, hidden_states, *weights):
            router_weight = graph.op(
                "Transpose", weights[0], perm_i=[1, 0]
            )
            # exUnifiedMoE inputs: hidden, router weight, optional selection
            # bias, then expert weights in gate/up/down order.
            node_inputs = [hidden_states, router_weight, weights[1]]
            node_inputs.extend(
                graph.op("Transpose", weight, perm_i=[1, 0])
                for weight in weights[2:]
            )
            output = graph.op(
                f"{MOE_CUSTOM_DOMAIN}::{MOE_CUSTOM_OP}",
                *node_inputs,
                layer_id_i=layer_id,
                n_routed_experts_i=n_experts,
                num_experts_per_token_i=top_k,
                n_group_i=n_group,
                topk_group_i=topk_group,
                topk_method_s="greedy",
                scoring_func_s="sigmoid",
                score_select_bias_i=1,
                norm_selected_score_s=(
                    "reducesum" if bool(config.norm_topk_prob) else "none"
                ),
                routed_scaling_factor_f=routed_scaling_factor,
                expert_weight_packed_s="weight",
                gate_up_packed_i=0,
                gate_bias_i=0,
                up_bias_i=0,
                down_bias_i=0,
                activation_s="silu",
                routed_scores_before_ffn_i=0,
                outputs=1,
            )
            output.setType(hidden_states.type())
            return output

    MiniCPM5MoEFunction.__name__ = f"MiniCPM5MoEFunctionLayer{layer_id}"
    return MiniCPM5MoEFunction, tuple(parameters), tuple(parameter_names)


def patch_minicpm5_moe_for_onnx(text_model):
    """Replace dynamic MiniCPM5 MoE dispatch with RKNN custom OPs."""
    manifests = []
    for layer_id, layer in enumerate(text_model.layers):
        moe = getattr(layer, "mlp", None)
        if type(moe).__name__ != "MiniCPM5MoEMoE":
            continue
        if getattr(moe, "_rknn_minicpm5_moe_patched", False):
            manifests.append(
                {"layer_id": layer_id, "n_experts": int(moe.config.n_routed_experts)}
            )
            continue

        function, parameters, parameter_names = _make_minicpm5_moe_function(
            moe, layer_id
        )

        def patched_forward(
            self, hidden_states, *, _function=function, _parameters=parameters
        ):
            routed_output = _function.apply(hidden_states, *_parameters)
            # Shared experts deliberately remain standard ONNX operators.
            return routed_output + self.shared_experts(hidden_states)

        moe.forward = types.MethodType(patched_forward, moe)
        moe._rknn_minicpm5_moe_patched = True
        manifests.append(
            {
                "layer_id": layer_id,
                "n_experts": int(moe.config.n_routed_experts),
                "parameter_names": parameter_names,
            }
        )

    if not manifests:
        raise RuntimeError("No MiniCPM5 MoE layers found in the text model")
    print(
        f"Patched {len(manifests)} MiniCPM5 MoE layers as "
        f"{MOE_CUSTOM_DOMAIN}::{MOE_CUSTOM_OP}"
    )
    return manifests


def _prepare_minicpm5_moe_export_buffers(text_model, dtype):
    """Cast MoE buffers before custom-op closures capture their tensor objects."""
    converted = 0
    for layer in text_model.layers:
        moe = getattr(layer, "mlp", None)
        if type(moe).__name__ != "MiniCPM5MoEMoE":
            continue
        bias = getattr(moe.gate, "e_score_correction_bias", None)
        if bias is None or bias.dtype == dtype:
            continue
        # This is a registered buffer, not an expert parameter.  Assignment
        # replaces the registered buffer before _make_minicpm5_moe_function
        # stores it in the custom-op parameter list.
        moe.gate.e_score_correction_bias = bias.to(dtype=dtype)
        converted += 1

    if converted:
        print(
            f"Converted {converted} MoE e_score_correction_bias buffers to "
            f"{dtype} before custom-op patching"
        )


def _verify_no_bfloat16_tensors(onnx_path):
    """Fail early if RKNN-incompatible BF16 tensors remain in the ONNX graph."""
    import onnx
    from onnx import TensorProto

    graph = onnx.load(onnx_path, load_external_data=False).graph
    locations = []
    for initializer in graph.initializer:
        if initializer.data_type == TensorProto.BFLOAT16:
            locations.append(f"initializer:{initializer.name}")
    for node in graph.node:
        for attribute in node.attribute:
            if (
                attribute.type == onnx.AttributeProto.TENSOR
                and attribute.t.data_type == TensorProto.BFLOAT16
            ):
                locations.append(f"node:{node.name or node.op_type}:{attribute.name}")

    if locations:
        raise RuntimeError(
            f"Exported ONNX contains BFLOAT16 tensors ({len(locations)}): "
            f"{locations[:8]}. Convert them to FP32 before RKNN build."
        )
    print("  Verified ONNX contains no BFLOAT16 initializers or constants")


def _verify_minicpm5_moe_nodes(onnx_path, expected_layers, expected_experts):
    """Verify that every exported MoE node carries all routed experts."""
    import onnx

    graph = onnx.load(onnx_path, load_external_data=False).graph
    nodes = [
        node
        for node in graph.node
        if node.domain == MOE_CUSTOM_DOMAIN and node.op_type == MOE_CUSTOM_OP
    ]
    if len(nodes) != expected_layers:
        raise RuntimeError(
            f"Expected {expected_layers} {MOE_CUSTOM_DOMAIN}::{MOE_CUSTOM_OP} "
            f"nodes, found {len(nodes)} in {onnx_path}"
        )

    expected_inputs = 3 + 3 * expected_experts
    for index, node in enumerate(nodes):
        if len(node.input) != expected_inputs:
            raise RuntimeError(
                f"MoE node {index} in {onnx_path} has {len(node.input)} inputs; "
                f"expected {expected_inputs} (router/bias + {expected_experts}*3 experts)"
            )
    print(
        f"  Verified {len(nodes)} {MOE_CUSTOM_DOMAIN}::{MOE_CUSTOM_OP} nodes; "
        f"each carries all {expected_experts} routed experts"
    )




def _verify_dynamic_causal_mask(onnx_path, prompt_size):
    """Reject the legacy-tracer causal-mask shape captured at prompt_size.

    ``dynamic_axes`` only marks graph inputs.  A traced mask implementation can
    still leave a ``Range -> Reshape`` with a constant ``[prompt_size, 1]``
    target.  RKNN builds the graph at its maximum sequence length, so that
    pattern fails as soon as the build length differs from the export length.
    """
    import onnx
    from onnx import TensorProto, numpy_helper

    graph = onnx.load(onnx_path, load_external_data=False).graph
    shape_values = {}

    for initializer in graph.initializer:
        if initializer.data_location == TensorProto.EXTERNAL:
            continue
        value = numpy_helper.to_array(initializer)
        shape_values[initializer.name] = value.reshape(-1).tolist()

    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                value = numpy_helper.to_array(attribute.t)
                shape_values[node.output[0]] = value.reshape(-1).tolist()
                break

    range_outputs = {
        output
        for node in graph.node
        if node.op_type == "Range"
        for output in node.output
    }
    fixed_shapes = []
    for node in graph.node:
        if (
            node.op_type == "Reshape"
            and len(node.input) > 1
            and node.input[0] in range_outputs
            and shape_values.get(node.input[1]) == [prompt_size, 1]
        ):
            fixed_shapes.append(node.name or node.output[0])

    if fixed_shapes:
        raise RuntimeError(
            "Exported causal mask still contains a fixed sequence reshape "
            f"[{prompt_size}, 1] in {onnx_path}: {fixed_shapes}. "
            "Keep eager attention computation separate from the dynamic mask "
            "implementation and re-export the model."
        )

    print("  Verified causal-mask sequence reshapes remain dynamic")


def _get_minicpm5_eager_attention(text_model):
    """Get MiniCPM5's model-defined eager attention implementation."""
    for layer in text_model.layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue
        forward = getattr(attention, "forward", None)
        forward = getattr(forward, "__func__", forward)
        eager_attention = getattr(forward, "__globals__", {}).get(
            "eager_attention_forward"
        )
        if eager_attention is not None:
            return eager_attention

        module = sys.modules.get(type(attention).__module__)
        eager_attention = getattr(module, "eager_attention_forward", None)
        if eager_attention is not None:
            return eager_attention

    raise RuntimeError(
        "Cannot find MiniCPM5 eager_attention_forward in the loaded model module"
    )


@contextmanager
def _minicpm5_eager_attention_with_dynamic_mask(text_model, model=None):
    """Use eager attention computation without selecting the eager mask factory.

    MiniCPM5's attention layer chooses its kernel from
    ``config._attn_implementation``.  Transformers' ``create_causal_mask``
    uses the same setting to choose a mask factory.  The Qwen3.5 exporter
    decouples these two choices; this context does the same for MiniCPM5 by
    temporarily routing the ``sdpa`` attention entry to the model's eager
    function while keeping the config on the dynamic ``sdpa`` mask path.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    eager_attention = _get_minicpm5_eager_attention(text_model)
    try:
        previous_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    except KeyError as exc:
        raise RuntimeError(
            "The installed Transformers version has no sdpa attention interface"
        ) from exc

    configurations = []
    seen = set()
    for candidate in (
        getattr(text_model, "config", None),
        getattr(model, "config", None),
    ):
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        configurations.append(
            (candidate, getattr(candidate, "_attn_implementation", None))
        )

    try:
        for config, _ in configurations:
            # ``sdpa`` selects the dynamic mask factory.  The attention
            # interface below is replaced with eager for this export only.
            config._attn_implementation = "sdpa"
        ALL_ATTENTION_FUNCTIONS["sdpa"] = eager_attention
        yield
    finally:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = previous_sdpa
        for config, previous in configurations:
            config._attn_implementation = previous


def _estimate_minicpm5_layer_weight_size(
    layer,
    weight_bits=4,
    group_size=32,
    metadata_bytes_per_group=3,
):
    """Estimate one layer using the complete exported MiniCPM5 MoE payload.

    ``Module.parameters()`` already includes ModuleList entries in ordinary
    PyTorch, but the explicit decomposition here makes the balancing contract
    visible and guards against accidentally estimating only the routed experts
    reached by the ONNX dummy trace. The router correction bias is a buffer,
    not a parameter, so it is added separately as FP32 bytes.
    """
    layer_size = _estimate_module_weight_size(
        layer,
        weight_bits,
        group_size=group_size,
        metadata_bytes_per_group=metadata_bytes_per_group,
    )
    moe = getattr(layer, "mlp", None)
    if type(moe).__name__ != "MiniCPM5MoEMoE":
        return layer_size

    configured_experts = int(getattr(moe.config, "n_routed_experts", 0) or 0)
    actual_experts = len(moe.experts)
    if configured_experts != actual_experts:
        raise RuntimeError(
            "MiniCPM5 MoE weight estimation found "
            f"{actual_experts} experts, config requires {configured_experts}"
        )

    moe_parameter_size = _estimate_module_weight_size(
        moe,
        weight_bits,
        group_size=group_size,
        metadata_bytes_per_group=metadata_bytes_per_group,
    )
    non_moe_size = layer_size - moe_parameter_size
    router_size = _estimate_module_weight_size(
        moe.gate,
        weight_bits,
        group_size=group_size,
        metadata_bytes_per_group=metadata_bytes_per_group,
    )
    routed_experts_size = sum(
        _estimate_module_weight_size(
            expert,
            weight_bits,
            group_size=group_size,
            metadata_bytes_per_group=metadata_bytes_per_group,
        )
        for expert in moe.experts
    )
    shared_expert_size = _estimate_module_weight_size(
        moe.shared_experts,
        weight_bits,
        group_size=group_size,
        metadata_bytes_per_group=metadata_bytes_per_group,
    )
    explicit_moe_parameter_size = router_size + routed_experts_size + shared_expert_size
    if explicit_moe_parameter_size != moe_parameter_size:
        raise RuntimeError(
            "MiniCPM5 MoE weight accounting mismatch: "
            f"module={moe_parameter_size}, explicit={explicit_moe_parameter_size}"
        )

    score_bias = getattr(moe.gate, "e_score_correction_bias", None)
    score_bias_size = 0 if score_bias is None else score_bias.numel() * score_bias.element_size()
    return non_moe_size + explicit_moe_parameter_size + score_bias_size

def _make_causal_mask(
    text_model,
    hidden_states,
    attention_mask,
    position_ids,
):
    """
    Build an ONNX-traceable dynamic 4D additive causal mask.

    Shape:
        hidden_states:  [B, S, H]
        attention_mask: [B, S]
        causal_mask:    [B, 1, S, S]

    No cache is used by the segmented exporter, therefore Q == KV == S.
    """
    del text_model, position_ids

    device = hidden_states.device
    dtype = hidden_states.dtype

    # IMPORTANT:
    # Keep S as a traced tensor-derived dimension. Do not convert it
    # through int(), .item(), bool(), Python list construction, etc.
    cache_position = torch.arange(
        hidden_states.shape[1],
        dtype=torch.long,
        device=device,
    )

    # [S, 1]
    q_position = cache_position.unsqueeze(1)

    # [1, S]
    kv_position = cache_position.unsqueeze(0)

    # [S, S]
    #
    # True means KV position can be attended by Q position.
    causal_allowed = kv_position <= q_position

    # [1, 1, S, S]
    causal_allowed = causal_allowed.unsqueeze(0).unsqueeze(0)

    if attention_mask is not None:
        # attention_mask:
        #     1 / True  -> valid KV token
        #     0 / False -> padding
        #
        # [B, S] -> [B, 1, 1, S]
        padding_allowed = (
            attention_mask[:, None, None, :] > 0
        )

        # broadcasting:
        # [1,1,S,S] & [B,1,1,S]
        #       -> [B,1,S,S]
        causal_allowed = causal_allowed & padding_allowed

    # MiniCPM5 eager attention adds attention_mask to attention logits,
    # so use an additive floating-point mask.
    #
    # allowed     -> 0
    # disallowed  -> dtype minimum
    zero = torch.zeros(
        (),
        dtype=dtype,
        device=device,
    )

    min_value = torch.full(
        (),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )

    causal_mask = torch.where(
        causal_allowed,
        zero,
        min_value,
    )

    return causal_mask, cache_position

class MiniCPM5SegmentWrapper(torch.nn.Module):
    """Export a contiguous group of standard Llama decoder layers.

    The multicard runtime supplies token embeddings from ``embed.bin``.  Each
    segment therefore consumes ``input_embeds`` and returns the hidden states
    needed by the next segment.  Only the final segment owns the final norm
    and lm_head and returns logits for the requested token.
    """

    def __init__(
        self,
        text_model,
        layers,
        norm=None,
        lm_head=None,
        is_last_segment=False,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.rotary_emb = text_model.rotary_emb
        self.norm = norm
        self.lm_head = lm_head
        self.is_last_segment = is_last_segment
        self.config = text_model.config

        # Keep the mask helper available without registering the complete
        # original model (which would unnecessarily retain all layers).
        object.__setattr__(self, "_mask_model", text_model)

        layer_forward = inspect.signature(layers[0].forward).parameters
        self._layer_forward_params = set(layer_forward)
        if "past_key_value" in layer_forward:
            self._past_key_arg = "past_key_value"
        elif "past_key_values" in layer_forward:
            self._past_key_arg = "past_key_values"
        else:
            self._past_key_arg = None

    def forward(
        self,
        input_embeds,
        attention_mask,
        position_ids,
        num_logits_to_keep,
    ):
        hidden_states = input_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask, cache_position = _make_causal_mask(
            self._mask_model,
            hidden_states,
            attention_mask,
            position_ids,
        )

        for layer in self.layers:
            layer_kwargs = {
                "attention_mask": causal_mask,
                "position_ids": position_ids,
            }
            if self._past_key_arg is not None:
                layer_kwargs[self._past_key_arg] = None
            if "output_attentions" in self._layer_forward_params:
                layer_kwargs["output_attentions"] = False
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
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
            if self.lm_head is None:
                raise RuntimeError("The last MiniCPM5 segment must contain lm_head")

            # The RKNN LLM runtime samples from the last token only.  Keeping
            # this input matches the single-card exporter and avoids materializing
            # the full vocabulary projection for every prefill token.
            hidden_states = hidden_states.index_select(
                dim=1,
                index=num_logits_to_keep,
            )
            hidden_states = self.lm_head(hidden_states)

        return hidden_states


def _get_llama_text_model(model):
    text_model = getattr(model, "model", None)
    if text_model is None or not hasattr(text_model, "layers"):
        raise TypeError(
            "MiniCPM5 multicard export expects a causal LM with model.layers"
        )
    required = ("embed_tokens", "norm", "rotary_emb")
    missing = [name for name in required if not hasattr(text_model, name)]
    if missing:
        raise TypeError(f"MiniCPM5 Llama model is missing: {', '.join(missing)}")
    if not hasattr(model, "lm_head"):
        raise TypeError("MiniCPM5 Llama model is missing lm_head")
    return text_model


def causal_minicpm5_llm_to_onnx_multi_segment(model, args):
    """Export MiniCPM5 as contiguous ONNX segments for multicard inference."""
    text_model = _get_llama_text_model(model)
    # Keep the causal-mask implementation dynamic.  Eager attention is
    # selected only around torch.onnx.export below, following the Qwen3.5
    # exporter pattern; setting config._attn_implementation to eager would
    # also select the eager mask factory and capture prompt_size in the graph.
    is_quantized_model = hasattr(model.config, "quantization_config")
    export_dtype = None if is_quantized_model else torch.float32
    all_layers = list(text_model.layers)
    if export_dtype is not None:
        # Must happen before patch_minicpm5_moe_for_onnx: its custom-op
        # closure captures the score-correction buffer object.
        _prepare_minicpm5_moe_export_buffers(text_model, export_dtype)
    patch_minicpm5_moe_for_onnx(text_model)
    num_total_layers = len(all_layers)
    if num_total_layers == 0:
        raise ValueError("MiniCPM5 model has no decoder layers")

    args.prompt_size = getattr(args, "prompt_size", 64)
    in_len = args.prompt_size
    num_segments = getattr(args, "num_segments", 0) or 0
    if num_segments <= 0:
        layers_per_segment = 8
        num_segments = max(1, (num_total_layers + layers_per_segment - 1) // layers_per_segment)


    layer_weight_sizes = [
        _estimate_minicpm5_layer_weight_size(layer, 4)
        for layer in all_layers
    ]
    moe_layers = [
        layer for layer in all_layers if type(layer.mlp).__name__ == "MiniCPM5MoEMoE"
    ]
    if moe_layers:
        configured_experts = int(text_model.config.n_routed_experts)
        actual_experts = {len(layer.mlp.experts) for layer in moe_layers}
        if actual_experts != {configured_experts}:
            raise RuntimeError(
                "MiniCPM5 MoE layers do not all contain the configured number "
                f"of experts: configured={configured_experts}, actual={sorted(actual_experts)}"
            )
        print(
            "Weight balance includes all MoE weights: "
            f"{len(moe_layers)} layers × {configured_experts} routed experts "
            "+ 1 shared expert per layer"
        )
    last_segment_extra_size = (
        _estimate_module_weight_size(text_model.norm, 16, metadata_bytes_per_group=0)
        + _estimate_module_weight_size(model.lm_head, 6)
    )
    prepare_plan(
        "minicpm5", text_model.config, layer_weight_sizes,
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

    model.eval()
    if is_quantized_model:
        q_config = model.config.quantization_config
        if not check_gptq(q_config.bits, q_config.group_size):
            print(
                "GRQ model quantization not supported. Only W4A16 group32 "
                "quantization is supported."
            )
            sys.exit(1)
        register_bitwise_right_shift()

    dummy_input_ids = torch.zeros((1, in_len), dtype=torch.long)
    with torch.no_grad():
        inputs_embeds = text_model.embed_tokens(dummy_input_ids)
        if export_dtype is not None:
            # Preserve the old public FP32 input dtype while keeping the
            # embedding weights in their checkpoint dtype during loading.
            inputs_embeds = inputs_embeds.to(dtype=export_dtype)
    attention_mask = torch.ones((1, in_len), dtype=torch.float32)
    position_ids = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)
    num_logits_to_keep = torch.tensor([in_len - 1], dtype=torch.long)

    base_path = args.export_llm_path
    base_dir = os.path.dirname(base_path)
    base_name = os.path.splitext(os.path.basename(base_path))[0]
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

    layer_start = 0
    for seg_idx, layer_count in enumerate(segment_layer_counts):
        layer_end = layer_start + layer_count
        is_last = seg_idx == num_segments - 1
        segment_model = MiniCPM5SegmentWrapper(
            text_model=text_model,
            layers=all_layers[layer_start:layer_end],
            norm=text_model.norm if is_last else None,
            lm_head=model.lm_head if is_last else None,
            is_last_segment=is_last,
        )
        segment_model.eval()
        if export_dtype is not None:
            # Convert only this segment, plus rotary_emb and final norm/head
            # when they belong to it.
            segment_model.to(dtype=export_dtype)

        if num_segments == 1:
            seg_path = base_path
        else:
            seg_dir = os.path.join(base_dir, f"seg{seg_idx}")
            os.makedirs(seg_dir, exist_ok=True)
            seg_path = os.path.join(seg_dir, f"{base_name}_seg{seg_idx}.onnx")

        print(
            f"Exporting segment {seg_idx}: layers {layer_start}-{layer_end - 1} "
            f"({layer_count} layers)"
            f"{' + norm + lm_head' if is_last else ''} -> {seg_path}"
        )
        with torch.no_grad():
            with _minicpm5_eager_attention_with_dynamic_mask(text_model, model):
                torch.onnx.export(
                    segment_model,
                    (inputs_embeds, attention_mask, position_ids, num_logits_to_keep),
                    seg_path,
                    export_params=True,
                    external_data=True,
                    opset_version=19,
                    do_constant_folding=True,
                    input_names=input_names,
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                    custom_opsets={MOE_CUSTOM_DOMAIN: 1},
                    # exUnifiedMoE is emitted by autograd.Function.symbolic.
                    dynamo=False,
                )

        segment_moe_layers = sum(
            type(layer.mlp).__name__ == "MiniCPM5MoEMoE"
            for layer in all_layers[layer_start:layer_end]
        )
        _verify_dynamic_causal_mask(seg_path, prompt_size=in_len)
        _verify_minicpm5_moe_nodes(
            seg_path,
            expected_layers=segment_moe_layers,
            expected_experts=int(text_model.config.n_routed_experts),
        )
        _verify_no_bfloat16_tensors(seg_path)
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
        print(f"Exported requested prefix: {exported_segments}/{num_segments} segments.")


if __name__ == "__main__":
    from argparse import ArgumentParser, BooleanOptionalAction

    parser = ArgumentParser(
        description="Export openbmb/MiniCPM5 LLM as multicard ONNX segments"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/path/to/MiniCPM5-15B-A2.5B",
        help="model path or Hugging Face model name",
    )
    parser.add_argument(
        "--export_llm_path",
        type=str,
        default="../../model/llm_minicpm5/MiniCPM5-15B-A2.5B.onnx",
        help="base ONNX output path",
    )
    parser.add_argument("--quant", action="store_true", help="use GRQ quantization")
    parser.add_argument(
        "--cali_dataset",
        default="../../../../datasets/llm_quant.json",
        help="calibration dataset for GRQ quantization",
    )
    parser.add_argument(
        "--modelscope",
        action="store_true",
        help="download the model from modelscope.cn",
    )
    parser.add_argument(
        "--multi_segment",
        action="store_true",
        help="export multiple ONNX segments for multicard deployment",
    )
    parser.add_argument(
        "--num_segments",
        type=int,
        default=0,
        help="number of segments; 0 uses the default layer-based split",
    )
    parser.add_argument("--segment_layer_counts", "--segment-layer-counts", type=int, nargs="+", default=None, help="manual block count per segment; counts must sum to the total layer count and override --num_segments")
    parser.add_argument(
        "--prompt_size",
        type=int,
        default=128,
        help="dummy export sequence length; the exported causal mask remains dynamic",
    )
    parser.add_argument(
        "--stop_after_segment",
        type=int,
        default=None,
        help=(
            "stop after exporting this zero-based segment (the remaining model "
            "segments are not exported); useful for block-level validation"
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "use Transformers low-memory checkpoint loading when accelerate is "
            "installed (default: enabled)"
        ),
    )
    add_plan_arguments(parser)
    args = parser.parse_args()

    if args.quant and args.multi_segment:
        parser.error(
            "--quant/GRQ is not supported with MiniCPM5 exUnifiedMoE export; "
            "quantize during RKNN build instead"
        )

    if args.modelscope:
        from modelscope import snapshot_download

        args.model_path = snapshot_download(args.model_path)

    kwargs = {
        'attn_implementation': 'eager',
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    update_config(config, ["use_cache"], False)
    kwargs["config"] = config
    _configure_model_load_kwargs(kwargs, args.low_cpu_mem_usage)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)

    if args.quant and torch.cuda.is_available():
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
            print("Load model failed!")
            sys.exit(ret)

        model = quant_tool.quantize(
            quantized_dtype="w4a16",
            quantized_method="group32",
            quantized_algorithm="grq",
            dataset=args.cali_dataset,
        ).cpu()

    export_dirname = os.path.dirname(args.export_llm_path)
    if export_dirname:
        os.makedirs(export_dirname, exist_ok=True)

    if args.multi_segment:
        causal_minicpm5_llm_to_onnx_multi_segment(model, args)
    else:
        # Keep the single-segment path compatible with the existing MiniCPM5
        # exporter while sharing model loading and artifact generation here.
        causal_llm_to_onnx(model, args)

    output_prefix = os.path.splitext(args.export_llm_path)[0]
    export_llm_config(
        args.model_path,
        output_prefix + ".config.pkl",
        chat_context,
        prompt,
    )
    export_tokenizer(
        args.model_path,
        output_prefix + ".tokenizer.gguf",
    )
    export_embed_weight(
        model.get_input_embeddings().weight,
        output_prefix + ".embed.bin",
    )
    if args.multi_segment:
        write_plan_json(args)
