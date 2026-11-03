import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
import types
import torch
import torch.nn.functional as F
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from segment_planner import add_plan_arguments, prepare_plan, write_plan_json

from py_utils.export_llm_helper import _estimate_module_weight_size, causal_qwen_3_5_llm_to_onnx, causal_qwen_3_5_llm_to_onnx_multi_segment, update_config, export_tokenizer, export_llm_config, export_embed_weight
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

try:
    import transformers.modeling_flash_attention_utils as _fa_utils
    _orig_is_packed_sequence = _fa_utils._is_packed_sequence
    def _is_packed_sequence(position_ids, batch_size):
        return False
    _fa_utils._is_packed_sequence = _is_packed_sequence
except:
    pass

prompt = "RKLLM"
chat_context = {"role": "user", "content": prompt}

MOE_CUSTOM_DOMAIN = "rknn"
MOE_CUSTOM_OP = "exUnifiedMoE"


def _qwen35_text_config(config):
    """Return the text config for both text-only and VLM checkpoints."""
    text_config = getattr(config, "text_config", None)
    return config if text_config is None else text_config


def _is_qwen35_moe_config(config):
    """Detect Qwen3.5-MoE without relying on optional config attributes.

    Dense Qwen3.5 configs also have the hybrid layer layout, so the model type
    (or the MoE-only intermediate-size fields) is the discriminator here.
    """
    model_type = str(getattr(config, "model_type", ""))
    text_config = _qwen35_text_config(config)
    text_model_type = str(getattr(text_config, "model_type", ""))
    if model_type in {"qwen3_5_moe", "qwen3_5_moe_text"}:
        return True
    if text_model_type == "qwen3_5_moe_text":
        return True

    # Keep this fallback narrow so a dense config with a stray 'num_experts'
    # field is not accidentally routed through the MoE implementation.
    return (
        hasattr(text_config, "num_experts")
        and hasattr(text_config, "num_experts_per_tok")
        and hasattr(text_config, "moe_intermediate_size")
        and not hasattr(text_config, "intermediate_size")
    )


def _is_qwen35_moe_block(moe):
    """Recognize the packed-expert block used by Qwen3.5-MoE."""
    experts = getattr(moe, "experts", None)
    return (
        experts is not None
        and hasattr(experts, "gate_up_proj")
        and hasattr(experts, "down_proj")
        and hasattr(moe, "gate")
        and hasattr(moe, "shared_expert")
        and hasattr(moe, "shared_expert_gate")
    )


def _make_qwen35_moe_function(moe, layer_id):
    """Create an eager-compatible Function with a packed exUnifiedMoE node.

    Qwen3.5-MoE stores all routed experts in two tensors:
      gate_up_proj: [num_experts, 2 * intermediate_size, hidden_size]
      down_proj:    [num_experts, hidden_size, intermediate_size]

    The RKNN custom op expects mathematical [in, out] weight layout, hence
    the batch-preserving transposes in symbolic().
    """
    config = moe.config
    router = moe.gate
    experts = moe.experts

    n_experts = int(getattr(config, "num_experts", experts.gate_up_proj.shape[0]))
    top_k = int(
        getattr(
            config,
            "num_experts_per_tok",
            getattr(config, "num_experts_per_token", 0),
        )
    )
    gate_up_shape = tuple(experts.gate_up_proj.shape)
    down_shape = tuple(experts.down_proj.shape)
    router_shape = tuple(router.weight.shape)

    if len(gate_up_shape) != 3 or len(down_shape) != 3 or len(router_shape) != 2:
        raise ValueError(
            "Qwen3.5-MoE expects packed expert weights with ranks "
            f"[3, 3, 2], got [{len(gate_up_shape)}, {len(down_shape)}, {len(router_shape)}]"
        )
    hidden_size = int(gate_up_shape[2])
    intermediate_size = int(gate_up_shape[1])
    if gate_up_shape[0] != n_experts or down_shape[0] != n_experts:
        raise ValueError(
            "Qwen3.5-MoE expert count does not match config: "
            f"config={n_experts}, gate_up={gate_up_shape[0]}, down={down_shape[0]}"
        )
    if intermediate_size % 2 != 0 or down_shape[1:] != (hidden_size, intermediate_size // 2):
        raise ValueError(
            "Qwen3.5-MoE packed expert shapes are inconsistent: "
            f"gate_up={gate_up_shape}, down={down_shape}"
        )
    if router_shape != (n_experts, hidden_size):
        raise ValueError(
            "Qwen3.5-MoE router weight shape is inconsistent with expert weights: "
            f"router={router_shape}, expected={(n_experts, hidden_size)}"
        )
    if not 1 <= top_k <= n_experts:
        raise ValueError(
            f"Qwen3.5-MoE num_experts_per_tok must be in [1, {n_experts}], got {top_k}"
        )

    # Keep the complete packed tensors as Function inputs. A normal trace of
    # the Python expert dispatch would retain only experts selected by the
    # dummy prompt and silently drop the other expert weights.
    parameters = (
        router.weight,
        experts.gate_up_proj,
        experts.down_proj,
    )

    def eager_routed(hidden_states, weights):
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_size)
        flat_states_fp32 = flat_states.to(torch.float32)
        router_weight, gate_up_weight, down_weight = weights

        # This follows Qwen3.5-MoE's router: softmax over all experts, top-k,
        # then normalization of the selected probabilities.
        router_logits = F.linear(
            flat_states_fp32,
            router_weight.to(torch.float32),
        )
        scores = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(
            scores,
            k=top_k,
            dim=-1,
            sorted=True,
        )
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

    class Qwen35MoEFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden_states, *weights):
            del ctx
            return eager_routed(hidden_states, weights)

        @staticmethod
        def symbolic(graph, hidden_states, *weights):
            # PyTorch Linear stores [out, in], while exUnifiedMoE consumes
            # mathematical [in, out] layout. Preserve the expert batch axis.
            router_weight = graph.op(
                "Transpose", weights[0], perm_i=[1, 0]
            )
            gate_up_weight = graph.op(
                "Transpose", weights[1], perm_i=[0, 2, 1]
            )
            down_weight = graph.op(
                "Transpose", weights[2], perm_i=[0, 2, 1]
            )
            output = graph.op(
                f"{MOE_CUSTOM_DOMAIN}::{MOE_CUSTOM_OP}",
                hidden_states,
                router_weight,
                gate_up_weight,
                down_weight,
                layer_id_i=layer_id,
                n_routed_experts_i=n_experts,
                num_experts_per_token_i=top_k,
                topk_method_s="greedy",
                scoring_func_s="softmax",
                norm_selected_score_s="reducesum",
                routed_scaling_factor_f=1.0,
                expert_weight_packed_s="batch",
                gate_up_packed_i=1,
                router_bias_i=0,
                gate_bias_i=0,
                up_bias_i=0,
                down_bias_i=0,
                activation_s="silu",
                routed_scores_before_ffn_i=0,
                outputs=1,
            )
            output.setType(hidden_states.type())
            return output

    Qwen35MoEFunction.__name__ = f"Qwen35MoEFunctionLayer{layer_id}"
    return Qwen35MoEFunction, parameters


def patch_qwen35_moe_for_onnx(text_model):
    """Replace Qwen3.5-MoE routed dispatch with RKNN custom OPs."""
    manifests = []
    for layer_id, layer in enumerate(text_model.layers):
        moe = getattr(layer, "mlp", None)
        if not _is_qwen35_moe_block(moe):
            continue

        n_experts = int(moe.experts.gate_up_proj.shape[0])
        if getattr(moe, "_rknn_qwen35_moe_patched", False):
            manifests.append({"layer_id": layer_id, "n_experts": n_experts})
            continue

        function, parameters = _make_qwen35_moe_function(moe, layer_id)

        def patched_forward(
            self, hidden_states, *, _function=function, _parameters=parameters
        ):
            routed_output = _function.apply(hidden_states, *_parameters)
            # Qwen3.5 shared expert remains a standard FFN. Its output is
            # weighted by sigmoid(shared_expert_gate(hidden_states)).
            shared_output = self.shared_expert(hidden_states)
            shared_output = shared_output * torch.sigmoid(
                self.shared_expert_gate(hidden_states)
            )
            return routed_output + shared_output

        moe.forward = types.MethodType(patched_forward, moe)
        moe._rknn_qwen35_moe_patched = True
        manifests.append({"layer_id": layer_id, "n_experts": n_experts})

    if not manifests:
        raise RuntimeError("No Qwen3.5-MoE layers found in the text model")

    expert_counts = sorted({item["n_experts"] for item in manifests})
    if len(expert_counts) != 1:
        raise ValueError(
            "Qwen3.5-MoE requires one packed expert count across all layers, "
            f"got {expert_counts}"
        )
    print(
        f"Patched {len(manifests)} Qwen3.5-MoE layers as "
        f"{MOE_CUSTOM_DOMAIN}::{MOE_CUSTOM_OP} "
        f"(batch-packed experts={expert_counts[0]})"
    )
    return manifests


def _build_qwen35_moe_model_classes(modeling_module):
    """Build MoE classes that reuse the local exLinearAttention exporter.

    The official Qwen3.5-MoE classes use the same hybrid attention structure as
    the dense exporter, but their decoder MLP and cache/config classes live in
    a different Transformers module. Reusing the local attention classes keeps
    the existing 'exLinearAttention' export intact while matching the MoE
    checkpoint key layout ('model.language_model.layers.*.mlp.experts.*').
    """
    nn = torch.nn

    class Qwen3_5MoeExperts(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.num_experts = int(config.num_experts)
            self.hidden_dim = int(config.hidden_size)
            self.intermediate_dim = int(config.moe_intermediate_size)
            self.gate_up_proj = nn.Parameter(
                torch.empty(
                    self.num_experts,
                    2 * self.intermediate_dim,
                    self.hidden_dim,
                )
            )
            self.down_proj = nn.Parameter(
                torch.empty(
                    self.num_experts,
                    self.hidden_dim,
                    self.intermediate_dim,
                )
            )
            self.act_fn = F.silu

        def forward(self, hidden_states, top_k_index, top_k_weights):
            # This fallback is equivalent to the HF eager implementation. In
            # export mode the containing block is patched before it is called.
            final_hidden_states = torch.zeros_like(hidden_states)
            expert_mask = F.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()
            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate, up = F.linear(
                    current_state,
                    self.gate_up_proj[expert_idx],
                ).chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = F.linear(
                    current_hidden_states,
                    self.down_proj[expert_idx],
                )
                current_hidden_states = current_hidden_states * top_k_weights[
                    token_idx, top_k_pos, None
                ]
                final_hidden_states.index_add_(
                    0,
                    token_idx,
                    current_hidden_states.to(final_hidden_states.dtype),
                )
            return final_hidden_states

    class Qwen3_5MoeTopKRouter(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.top_k = int(config.num_experts_per_tok)
            self.num_experts = int(config.num_experts)
            self.hidden_dim = int(config.hidden_size)
            self.weight = nn.Parameter(
                torch.zeros(self.num_experts, self.hidden_dim)
            )

        def forward(self, hidden_states):
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = F.linear(hidden_states, self.weight)
            router_logits = F.softmax(
                router_logits, dtype=torch.float, dim=-1
            )
            router_top_value, router_indices = torch.topk(
                router_logits, self.top_k, dim=-1
            )
            router_top_value = router_top_value / (
                router_top_value.sum(dim=-1, keepdim=True) + 1e-20
            )
            return router_logits, router_top_value, router_indices

    class Qwen3_5MoeSparseMoeBlock(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.gate = Qwen3_5MoeTopKRouter(config)
            self.experts = Qwen3_5MoeExperts(config)
            self.shared_expert = modeling_module.Qwen3_5MLP(
                config,
                int(config.shared_expert_intermediate_size),
            )
            self.shared_expert_gate = nn.Linear(
                int(config.hidden_size),
                1,
                bias=False,
            )

        def forward(self, hidden_states):
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
            shared_expert_output = self.shared_expert(hidden_states_reshaped)
            _, routing_weights, selected_experts = self.gate(
                hidden_states_reshaped
            )
            expert_output = self.experts(
                hidden_states_reshaped,
                selected_experts,
                routing_weights,
            )
            shared_expert_output = F.sigmoid(
                self.shared_expert_gate(hidden_states_reshaped)
            ) * shared_expert_output
            return (expert_output + shared_expert_output).reshape(
                batch_size, sequence_length, hidden_dim
            )

    class Qwen3_5MoeDecoderLayer(modeling_module.Qwen3_5DecoderLayer):
        def __init__(self, config, layer_idx):
            modeling_module.GradientCheckpointingLayer.__init__(self)
            self.hidden_size = int(config.hidden_size)
            self.layer_type = config.layer_types[layer_idx]
            if self.layer_type == "linear_attention":
                self.linear_attn = modeling_module.Qwen3_5GatedDeltaNet(
                    config,
                    layer_idx,
                )
            elif self.layer_type == "full_attention":
                self.self_attn = modeling_module.Qwen3_5Attention(
                    config,
                    layer_idx,
                )
            else:
                raise ValueError(
                    f"Unsupported Qwen3.5 layer type: {self.layer_type}"
                )
            self.mlp = Qwen3_5MoeSparseMoeBlock(config)
            self.input_layernorm = modeling_module.Qwen3_5RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = modeling_module.Qwen3_5RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )

    class Qwen3_5MoeTextModel(modeling_module.Qwen3_5TextModel):
        def __init__(self, config):
            modeling_module.Qwen3_5PreTrainedModel.__init__(self, config)
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
            )
            self.layers = nn.ModuleList(
                [
                    Qwen3_5MoeDecoderLayer(config, layer_idx)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
            self.norm = modeling_module.Qwen3_5RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.rotary_emb = modeling_module.Qwen3_5TextRotaryEmbedding(
                config=config
            )
            self.gradient_checkpointing = False
            self.post_init()

    class Qwen3_5MoeModel(modeling_module.Qwen3_5Model):
        def __init__(self, config):
            modeling_module.Qwen3_5PreTrainedModel.__init__(self, config)
            self.visual = modeling_module.Qwen3_5VisionModel._from_config(
                config.vision_config,
                attn_implementation="eager",
            )
            self.language_model = Qwen3_5MoeTextModel._from_config(
                config.text_config,
                attn_implementation="eager",
            )
            self.rope_deltas = None
            self.post_init()

    class Qwen3_5MoeForCausalLM(modeling_module.Qwen3_5ForCausalLM):
        def __init__(self, config):
            modeling_module.Qwen3_5PreTrainedModel.__init__(self, config)
            self.model = Qwen3_5MoeTextModel(config)
            self.vocab_size = config.vocab_size
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )
            self.post_init()

    class Qwen3_5MoeForConditionalGeneration(
        modeling_module.Qwen3_5ForConditionalGeneration
    ):
        def __init__(self, config):
            modeling_module.Qwen3_5PreTrainedModel.__init__(self, config)
            self.model = Qwen3_5MoeModel(config)
            self.lm_head = nn.Linear(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                bias=False,
            )
            self.post_init()

    return {
        "causal_lm": Qwen3_5MoeForCausalLM,
        "conditional_generation": Qwen3_5MoeForConditionalGeneration,
    }


class Qwen3_5MoeExportAdapter(torch.nn.Module):
    """Expose a VLM's text tower in the shape expected by export helpers."""

    def __init__(self, model):
        super().__init__()
        # Keep the full model alive for forward(), but do not register it as a
        # second child: the text tower and lm_head below are the only weights a
        # segment export should own.
        object.__setattr__(self, "_outer_model", model)
        self.model = model.model.language_model
        self.lm_head = model.lm_head
        self.config = self.model.config

    @property
    def device(self):
        return self.model.embed_tokens.weight.device

    def forward(self, *args, **kwargs):
        return self._outer_model(*args, **kwargs)


def _qwen35_moe_export_paths(
    export_llm_path,
    multi_segment,
    num_layers,
    requested_segments,
    segment_layer_counts=None,
    stop_after_segment=None,
):
    """Return the deterministic ONNX paths emitted by the Qwen helper."""
    if not multi_segment:
        return [export_llm_path]

    if segment_layer_counts is not None:
        if isinstance(segment_layer_counts, str):
            segment_layer_counts = segment_layer_counts.replace(",", " ").split()
        try:
            manual_counts = [int(count) for count in segment_layer_counts]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "--segment_layer_counts must be a list of positive integers"
            ) from exc
        if not manual_counts or any(count <= 0 for count in manual_counts):
            raise ValueError(
                "--segment_layer_counts must contain only positive integers"
            )
        if sum(manual_counts) != num_layers:
            raise ValueError(
                "--segment_layer_counts must sum to the total number of layers "
                f"({num_layers}), got {sum(manual_counts)}"
            )
        num_segments = len(manual_counts)
    else:
        num_segments = int(requested_segments or 0)
        if num_segments <= 0:
            num_segments = max(1, (num_layers + 7) // 8)
        num_segments = min(max(num_segments, 1), num_layers)

    if stop_after_segment is not None and not 0 <= stop_after_segment < num_segments:
        raise ValueError(
            "--stop_after_segment must be within the exported segment range: "
            f"0 <= {stop_after_segment} < {num_segments}"
        )

    if num_segments == 1:
        paths = [export_llm_path]
    else:
        base_dir = os.path.dirname(export_llm_path)
        base_name = os.path.splitext(os.path.basename(export_llm_path))[0]
        paths = [
            os.path.join(
                base_dir,
                f"seg{segment_index}",
                f"{base_name}_seg{segment_index}.onnx",
            )
            for segment_index in range(num_segments)
        ]

    if stop_after_segment is not None:
        paths = paths[:stop_after_segment + 1]
    return paths


def verify_qwen35_moe_custom_nodes(export_paths, expected_layers, expected_experts):
    """Validate that every segment carries a complete packed MoE node."""
    import onnx

    total_nodes = 0
    for onnx_path in export_paths:
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(
                f"Expected exported Qwen3.5-MoE ONNX file is missing: {onnx_path}"
            )
        graph = onnx.load(onnx_path, load_external_data=False).graph
        nodes = [
            node
            for node in graph.node
            if node.domain == MOE_CUSTOM_DOMAIN and node.op_type == MOE_CUSTOM_OP
        ]
        total_nodes += len(nodes)
        for node_index, node in enumerate(nodes):
            attributes = {attribute.name: attribute for attribute in node.attribute}
            routed_experts = attributes.get("n_routed_experts")
            if routed_experts is None or int(routed_experts.i) != expected_experts:
                actual = None if routed_experts is None else int(routed_experts.i)
                raise RuntimeError(
                    f"MoE node {node_index} in {onnx_path} declares "
                    f"{actual} routed experts; expected {expected_experts}"
                )
            if len(node.input) != 4:
                raise RuntimeError(
                    f"MoE node {node_index} in {onnx_path} has {len(node.input)} "
                    "inputs; expected 4 for batch-packed Qwen3.5 experts"
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
                    f"MoE node {node_index} in {onnx_path} has gate_up_packed="
                    f"{actual}; expected 1"
                )

    if expected_layers is not None and total_nodes != expected_layers:
        raise RuntimeError(
            f"Expected {expected_layers} {MOE_CUSTOM_DOMAIN}::{MOE_CUSTOM_OP} "
            f"nodes, found {total_nodes} across {len(export_paths)} files"
        )
    print(
        f"Verified {total_nodes} {MOE_CUSTOM_DOMAIN}::{MOE_CUSTOM_OP} nodes; "
        f"each carries all {expected_experts} routed experts"
    )

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen3_5 multi-segment ONNX model for multicard deployment")
    parser.add_argument("--model_path", type=str, required=False, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--cali_dataset", default='../../../../datasets/llm_quant.json')
    parser.add_argument("--export_llm_path", type=str, required=False, default="../../model/llm_qwen/Qwen3.5-9B-llm.onnx")
    parser.add_argument("--multi_segment", action='store_true')
    parser.add_argument("--num_segments", type=int, default=0)
    parser.add_argument("--segment_layer_counts", "--segment-layer-counts", type=int, nargs="+", default=None, help="manual block count per segment; counts must sum to the total layer count and override --num_segments")
    parser.add_argument("--stop_after_segment", type=int, default=None, help="stop after this zero-based segment; useful for block-level validation")
    parser.add_argument("--modelscope", action='store_true')
    add_plan_arguments(parser)
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    is_moe = _is_qwen35_moe_config(config)
    if is_moe and args.quant and torch.cuda.is_available():
        parser.error(
            "GRQ quantization is not supported together with "
            "Qwen3.5-MoE exUnifiedMoE export"
        )

    fakeq_state = None
    if args.quant and torch.cuda.is_available():
        ATTN_IMPL = None
        try:
            import flash_attn
            ATTN_IMPL = "flash_attention_2"
            print("Use FlashAttention2")
        except Exception as e:
            print(f"FlashAttention2 is unavailable ({e}), falling back to standard attention.")
        from rknn.quantization.api import RKQuantizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            _attn_implementation=ATTN_IMPL if ATTN_IMPL is not None else "eager",
            trust_remote_code=True)
        QuantTool = RKQuantizer(verbose=True)
        ret = QuantTool.load_model(model=model, tokenizer=tokenizer, device='cuda',
                                   system_prompt=None, tools=None, system_role='system', user_role='user')
        if ret != 0:
            print('Load model failed!')
            sys.exit(ret)
        dataset = args.cali_dataset
        quant_model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32",
                                         quantized_algorithm='grq', dataset=dataset,
                                         auto_hybrid_rate=0.)
        quant_model = quant_model.cpu()
        fakeq_state = quant_model.state_dict()
        del quant_model

    custom_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    modeling_spec = importlib.util.spec_from_file_location(
        'transformers.models.qwen3_5.modeling_qwen3_5',
        os.path.join(custom_path, 'modeling_qwen3_5_export.py'))
    modeling_module = importlib.util.module_from_spec(modeling_spec)
    sys.modules['transformers.models.qwen3_5.modeling_qwen3_5'] = modeling_module
    modeling_spec.loader.exec_module(modeling_module)

    def pack_dynamic_cache(conv_states, recurrent_states, key_cache, value_cache, model_config):
        past_key_values = modeling_module.Qwen3_5DynamicCache(model_config)
        assert conv_states.shape[0] == past_key_values.layer_types.count('linear_attention'), "conv_states mismatch"
        conv_states = list(torch.split(conv_states, dim=0, split_size_or_sections=1))
        for i, layer_types in enumerate(past_key_values.layer_types):
            if layer_types == 'linear_attention':
                past_key_values.conv_states[i] = conv_states.pop(0)
        return past_key_values

    class Qwen3_5_export_model_cst(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.config = model.config

        def forward(self, input_ids, position_ids, attention_mask, Tc, logits_to_keep, conv_status,
                    recurrent_status=None, key_cache=None, value_cache=None):
            past_key_values = pack_dynamic_cache(conv_status, recurrent_status, key_cache, value_cache, self.config)
            model_out = self.model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask,
                                   past_key_values=past_key_values, Tc=Tc, logits_to_keep=logits_to_keep)
            logits = model_out.logits
            conv_status = model_out.past_key_values.conv_states
            conv_status = [v for v in conv_status if v is not None]
            conv_status = torch.concat(conv_status, dim=0)
            return logits, conv_status, None, None, None


    moe_manifests = None
    if is_moe:
        moe_model_classes = _build_qwen35_moe_model_classes(modeling_module)
        if getattr(config, "text_config", None) is not None:
            model = moe_model_classes["conditional_generation"].from_pretrained(
                args.model_path,
                config=config,
                torch_dtype="float32",
                device_map="cpu",
                attn_implementation="eager",
            )
            text_model = model.model.language_model
            export_model = Qwen3_5MoeExportAdapter(model)
        else:
            text_config = _qwen35_text_config(config)
            model = moe_model_classes["causal_lm"].from_pretrained(
                args.model_path,
                config=text_config,
                torch_dtype="float32",
                device_map="cpu",
                attn_implementation="eager",
            )
            text_model = model.model
            export_model = model
        moe_manifests = patch_qwen35_moe_for_onnx(text_model)
    else:
        # Keep the original dense loader untouched. In particular, this
        # preserves the existing VLM -> text-only checkpoint loading behavior.
        model = modeling_module.Qwen3_5ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype="float32",
            device_map="cpu",
            attn_implementation="eager",
        )
        export_model = model

    if fakeq_state is not None:
        model.load_state_dict(fakeq_state, strict=False)

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
        os.makedirs(export_dirname)

    exported_model = Qwen3_5_export_model_cst(export_model)
    args.tokenizer = tokenizer
    args.chat_context = [chat_context]

    if args.multi_segment:
        planning_text_model = export_model.model
        # Qwen3.5-MoE RKNN reports show 5 B/group metadata on block weights;
        # dense blocks use the usual 3 B/group packing estimate.
        block_metadata_bytes = 5 if is_moe else 3
        layer_weight_sizes = [
            _estimate_module_weight_size(
                layer, 4, metadata_bytes_per_group=block_metadata_bytes,
            )
            for layer in planning_text_model.layers
        ]
        last_segment_extra_size = (
            _estimate_module_weight_size(
                planning_text_model.norm, 16, metadata_bytes_per_group=0,
            )
            + _estimate_module_weight_size(export_model.lm_head, 6)
        )
        prepare_plan(
            "qwen3_5", export_model.config, layer_weight_sizes,
            last_segment_extra_size, args,
        )
        causal_qwen_3_5_llm_to_onnx_multi_segment(exported_model, args)
    else:
        causal_qwen_3_5_llm_to_onnx(exported_model, args)

    if is_moe:
        export_paths = _qwen35_moe_export_paths(
            args.export_llm_path,
            args.multi_segment,
            len(text_model.layers),
            args.num_segments,
            args.segment_layer_counts,
            args.stop_after_segment,
        )
        verify_qwen35_moe_custom_nodes(
            export_paths,
            expected_layers=(
                len(moe_manifests)
                if args.stop_after_segment is None
                else None
            ),
            expected_experts=moe_manifests[0]["n_experts"],
        )

    if args.quant and torch.cuda.is_available():
        QuantTool.export_op_quantized_dtype(args.export_llm_path, op_dtype_path='layer_bit.json')

    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')
    export_embed_weight(model.get_input_embeddings().weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')
    if args.multi_segment:
        write_plan_json(args)
