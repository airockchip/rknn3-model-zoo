import os
import shutil
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from typing import Any, Optional, Sequence
import json
import types


import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from py_utils.export_llm_helper import (  # noqa: E402
    export_embed_weight,
    export_llm_config,
    update_config,
)


CUSTOM_DOMAIN = "rknn"
CUSTOM_OP = "DeepseekV2MoE"


def _moe_parameter_layout(moe: nn.Module) -> tuple[list[str], tuple[torch.Tensor, ...]]:
    """Return a stable custom-op input order, independent of Module registration order."""
    names: list[str] = ["gate.weight"]
    tensors: list[torch.Tensor] = [moe.gate.weight]
    if hasattr(moe.gate, "e_score_correction_bias"):
        names.append("gate.e_score_correction_bias")
        tensors.append(moe.gate.e_score_correction_bias)

    for index, expert in enumerate(moe.experts):
        if expert is None:
            raise ValueError("expert-parallel (ep_size > 1) MoE export is not supported")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            names.append(f"experts.{index}.{projection}.weight")
            tensors.append(getattr(expert, projection).weight)

    shared = getattr(moe, "shared_experts", None)
    n_shared = int(moe.config.n_shared_experts or 0)
    if shared is not None and n_shared > 0:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            names.append(f"shared_experts.{projection}.weight")
            tensors.append(getattr(shared, projection).weight)
    return names, tuple(tensors)


def _make_moe_function(
    moe: nn.Module,
    layer_id: int,
    parameter_names: Sequence[str],
):
    """Create a per-layer Function so symbolic attributes capture that layer's config."""
    config = moe.config
    layout = "|".join(parameter_names)
    has_correction_bias = hasattr(moe.gate, "e_score_correction_bias")
    n_experts = int(config.n_routed_experts)

    def mlp(x, gate_weight, up_weight, down_weight):
        gate = torch.nn.functional.linear(x, gate_weight)
        up = torch.nn.functional.linear(x, up_weight)
        return torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up, down_weight
        )

    def eager_forward(hidden_states, weights):
        """Equivalent eval-mode MoE using only explicit Function inputs."""
        cursor = 0
        gate_weight = weights[cursor]
        cursor += 1
        correction_bias = None
        if has_correction_bias:
            correction_bias = weights[cursor]
            cursor += 1

        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = torch.nn.functional.linear(
            flat_states.float(), gate_weight.float()
        )
        if config.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        elif config.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        else:
            raise NotImplementedError(f"unsupported MoE scoring: {config.scoring_func}")

        scores_for_choice = scores
        if config.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=config.num_experts_per_tok, dim=-1, sorted=False
            )
        else:
            if config.topk_method == "noaux_tc":
                scores_for_choice = scores + correction_bias.unsqueeze(0)
                group_scores = scores_for_choice.view(
                    flat_states.shape[0], config.n_group, -1
                ).topk(2, dim=-1)[0].sum(dim=-1)
            elif config.topk_method == "group_limited_greedy":
                group_scores = scores.view(
                    flat_states.shape[0], config.n_group, -1
                ).max(dim=-1).values
            else:
                raise NotImplementedError(
                    f"unsupported MoE top-k method: {config.topk_method}"
                )
            group_idx = torch.topk(
                group_scores, k=config.topk_group, dim=-1, sorted=False
            ).indices
            group_mask = torch.zeros_like(group_scores)
            group_mask.scatter_(1, group_idx, 1)
            score_mask = group_mask.unsqueeze(-1).expand(
                flat_states.shape[0],
                config.n_group,
                config.n_routed_experts // config.n_group,
            ).reshape(flat_states.shape[0], -1)
            choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
            _, topk_idx = torch.topk(
                choice, k=config.num_experts_per_tok, dim=-1, sorted=False
            )
            topk_weight = scores.gather(1, topk_idx)

        if config.num_experts_per_tok > 1 and config.norm_topk_prob:
            topk_weight = topk_weight / (
                topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            )
        topk_weight = topk_weight * config.routed_scaling_factor

        # Only selected experts execute during the example forward.  This keeps
        # tracing practical while symbolic still receives every expert weight.
        routed = torch.zeros(
            flat_states.shape, device=flat_states.device, dtype=torch.float32
        )
        for expert_id in range(n_experts):
            gate_proj, up_proj, down_proj = weights[cursor : cursor + 3]
            cursor += 3
            token_weight = (
                topk_weight * (topk_idx == expert_id).to(topk_weight.dtype)
            ).sum(dim=-1)
            token_indices = torch.nonzero(token_weight, as_tuple=False).flatten()
            if token_indices.numel() == 0:
                continue
            selected = flat_states.index_select(0, token_indices)
            expert_output = mlp(selected, gate_proj, up_proj, down_proj)
            weighted = expert_output.float() * token_weight.index_select(
                0, token_indices
            ).unsqueeze(-1)
            routed.index_add_(0, token_indices, weighted)

        routed = routed.to(hidden_states.dtype).view(original_shape)
        n_shared = int(config.n_shared_experts or 0)
        if n_shared > 0:
            shared_gate, shared_up, shared_down = weights[cursor : cursor + 3]
            cursor += 3
            routed = routed + mlp(
                hidden_states, shared_gate, shared_up, shared_down
            )
        return routed

    class DeepseekV2MoEFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden_states, *weights):  # noqa: ANN001
            # The symbolic below replaces this dispatch in the ONNX graph.
            del ctx
            return eager_forward(hidden_states, weights)

        @staticmethod
        def symbolic(graph, hidden_states, *weights):  # noqa: ANN001
            output = graph.op(
                f"{CUSTOM_DOMAIN}::{CUSTOM_OP}",
                hidden_states,
                *weights,
                layer_id_i=layer_id,
                n_routed_experts_i=int(config.n_routed_experts),
                num_experts_per_tok_i=int(config.num_experts_per_tok),
                n_shared_experts_i=1 if config.n_shared_experts else 0,                 # deepseekv4 把 n_shared_experts 拼成了一个
                moe_intermediate_size_i=int(config.moe_intermediate_size),
                scoring_func_s=str(config.scoring_func),
                topk_method_s=str(config.topk_method),
                norm_topk_prob_i=int(bool(config.norm_topk_prob)),
                routed_scaling_factor_f=float(config.routed_scaling_factor),
                activation_s=str(config.hidden_act),
                weight_layout_s=layout,
                outputs=1,
            )
            # Custom domains have no ONNX schema to infer this automatically.
            output.setType(hidden_states.type())
            return output

    DeepseekV2MoEFunction.__name__ = f"DeepseekV2MoEFunctionLayer{layer_id}"
    return DeepseekV2MoEFunction


def patch_moe_for_onnx(decoder: nn.Module) -> list[dict[str, Any]]:
    """Monkey-patch MoE forwards and return their custom-op manifests."""
    manifests: list[dict[str, Any]] = []
    for layer_id, layer in enumerate(decoder.layers):
        moe = layer.mlp
        if type(moe).__name__ != "DeepseekV2MoE":
            continue
        if type(moe.gate).__name__ != "MoEGate":
            raise TypeError(
                f"layer {layer_id} has unexpected router {type(moe.gate).__name__}"
            )

        parameter_names, parameter_tensors = _moe_parameter_layout(moe)
        function = _make_moe_function(moe, layer_id, parameter_names)

        def patched_forward(
            self, hidden_states, *, _function=function, _weights=parameter_tensors
        ):
            return _function.apply(hidden_states, *_weights)

        moe.forward = types.MethodType(patched_forward, moe)
        manifests.append(
            {
                "layer_id": layer_id,
                "domain": CUSTOM_DOMAIN,
                "op_type": CUSTOM_OP,
                "inputs": ["hidden_states", *parameter_names],
                "router": {
                    "scoring_func": moe.config.scoring_func,
                    "topk_method": moe.config.topk_method,
                    "norm_topk_prob": bool(moe.config.norm_topk_prob),
                    "routed_scaling_factor": float(
                        moe.config.routed_scaling_factor
                    ),
                },
            }
        )

    if not manifests:
        raise RuntimeError(
            "No Unlimited-OCR DeepseekV2MoE modules were found below decoder.layers"
        )
    print(f"Monkey-patched {len(manifests)} MoE layers as {CUSTOM_DOMAIN}::{CUSTOM_OP}")
    return manifests




class UnlimitedOCRTextCausalLM(nn.Module):
    def __init__(self, text_model, lm_head, config):
        super().__init__()
        self.model = text_model
        self.lm_head = lm_head
        self.config = config

    def forward(
        self,
        input_ids,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        num_logits_to_keep: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            images=None,
            images_seq_mask=None,
            images_spatial_crop=None,
            return_dict=True,
        )

        hidden_states = outputs[0]
        if num_logits_to_keep is not None:
            keep_index = num_logits_to_keep.to(device=hidden_states.device, dtype=torch.long).reshape(-1)
            hidden_states = hidden_states[:, keep_index, :]

        logits = self.lm_head(hidden_states).float()
        return logits



def causal_llm_to_onnx_with_custom_moe(model, args):
    args.prompt_size = 64
    args.dynamic_shape = True

    model.eval()
    in_len = args.prompt_size

    input_ids = torch.zeros((1, in_len), dtype=torch.long)
    attention_mask = torch.ones((1, in_len), dtype=torch.float)
    position_ids = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)
    num_logits_to_keep = torch.tensor(-1, dtype=torch.int64).reshape(1)

    inputs = (input_ids, attention_mask, position_ids, num_logits_to_keep)
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

    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            args.export_llm_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=input_names,
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            custom_opsets={CUSTOM_DOMAIN: 1},
            dynamo=False,
        )


def verify_custom_moe_nodes(onnx_path, expected):
    import onnx

    graph = onnx.load(onnx_path, load_external_data=False).graph
    custom_nodes = [
        node
        for node in graph.node
        if node.domain == CUSTOM_DOMAIN and node.op_type == CUSTOM_OP
    ]
    if len(custom_nodes) != expected:
        raise RuntimeError(
            f"Expected {expected} {CUSTOM_DOMAIN}::{CUSTOM_OP} nodes, "
            f"found {len(custom_nodes)} in {onnx_path}"
        )
    print(f"Verified {len(custom_nodes)} custom MoE nodes in {os.path.abspath(onnx_path)}")


def save_moe_manifest(moe_manifest, output_path):
    manifest_path = os.path.splitext(output_path)[0] + ".moe.json"
    manifest = {
        "custom_moe": {
            "domain": CUSTOM_DOMAIN,
            "op_type": CUSTOM_OP,
            "opset": 1,
            "layers": moe_manifest,
        }
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"MoE manifest saved to {os.path.abspath(manifest_path)}")


def _is_local_path(path):
    return path.startswith((".", "/", "~")) or os.path.isdir(path)


def _load_config_dict(model_path):
    local_path = os.path.expanduser(model_path)
    config_path = os.path.join(local_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return config.to_dict()


def _prepare_tokenizer_only_dir(model_path, output_dir):
    local_path = os.path.expanduser(model_path)
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ]

    if os.path.isdir(local_path):
        copied = False
        for filename in tokenizer_files:
            src = os.path.join(local_path, filename)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(output_dir, filename))
                copied = True
        if not copied:
            raise FileNotFoundError(f"No tokenizer files found in {local_path}")
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir)

    config = _load_config_dict(model_path)
    language_config = config.get("language_config", {})
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "rope_theta",
        "rms_norm_eps",
        "num_experts_per_tok",
    ):
        if key in language_config:
            config[key] = language_config[key]

    config.pop("auto_map", None)
    config["architectures"] = ["LlamaForCausalLM"]
    config["model_type"] = "llama"
    config.setdefault("hidden_size", 1280)
    config.setdefault("intermediate_size", 6848)
    config.setdefault("num_hidden_layers", 12)
    config.setdefault("num_attention_heads", 10)
    config.setdefault("num_key_value_heads", config["num_attention_heads"])
    config.setdefault("vocab_size", 129280)
    config.setdefault("max_position_embeddings", 32768)

    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def export_unlimited_ocr_tokenizer(model_path, tokenizer_path):
    """Export Unlimited-OCR tokenizer as GGUF through a tokenizer-only llama config."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    converter = os.path.abspath(
        os.path.join(
            current_dir,
            "../../../../tokenizer/thirdparty/llama_vocab/convert_hf_to_gguf.py",
        )
    )
    os.makedirs(os.path.dirname(os.path.abspath(tokenizer_path)), exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="unlimited_ocr_tokenizer_") as tmpdir:
        _prepare_tokenizer_only_dir(model_path, tmpdir)
        cmd = [
            sys.executable,
            converter,
            "--vocab-only",
            "--outtype",
            "f16",
            "--outfile",
            tokenizer_path,
            tmpdir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        raise RuntimeError("Tokenizer exported failed.")
    print(f"Tokenizer exported to {tokenizer_path}")



def load_model(model_path):
    kwargs = {
        "trust_remote_code": True,
    }
    config = AutoConfig.from_pretrained(model_path, **kwargs)
    update_config(config, ["use_cache"], False)
    update_config(config, ["output_attentions"], False)
    config._attn_implementation = "eager"
    config._attn_implementation_autoset = False

    model = AutoModel.from_pretrained(
        model_path,
        config=config,
        attn_implementation="eager",
        torch_dtype=torch.float32,
        device_map="cpu",
        use_safetensors=True,
        trust_remote_code=True,
    ).eval()
    model.config.sliding_window = None

    return UnlimitedOCRTextCausalLM(model.model, model.lm_head, config).eval()


def quantize_text_model(model, cali_dataset):
    if not torch.cuda.is_available():
        print("CUDA is not available, skip GRQ quantization.")
        return model

    from rknn.quantization.api import RKQuantizer

    quant_tool = RKQuantizer(verbose=True)
    ret = quant_tool.load_model(model=model.model, tokenizer=None, device="cuda")
    if ret != 0:
        raise RuntimeError(f"Load model failed, ret={ret}")

    model.model = quant_tool.quantize(
        quantized_dtype="w4a16",
        quantized_method="group32",
        quantized_algorithm="grq",
        dataset=cali_dataset,
    )
    return model.cpu().eval()


def main():
    parser = ArgumentParser(description="Export Unlimited-OCR LLM ONNX and RKLLM side files.")
    parser.add_argument("--model_path", type=str, default="baidu/Unlimited-OCR")
    parser.add_argument(
        "--export_llm_path",
        type=str,
        default="../../model/llm/Unlimited-OCR-llm.onnx",
        help="Path of exported LLM ONNX model.",
    )
    parser.add_argument("--quant", action="store_true", help="Enable GRQ w4a16 group32 quantization.")
    parser.add_argument(
        "--cali_dataset",
        type=str,
        default="../../data/llm/model_inputs.json",
        help="Calibration dataset generated by make_calidata.py.",
    )
    args = parser.parse_args()

    export_dir = os.path.dirname(os.path.abspath(args.export_llm_path))
    os.makedirs(export_dir, exist_ok=True)

    model = load_model(args.model_path)
    if args.quant:
        model = quantize_text_model(model, args.cali_dataset)

    moe_manifest = patch_moe_for_onnx(model.model)
    causal_llm_to_onnx_with_custom_moe(model, args)
    verify_custom_moe_nodes(args.export_llm_path, len(moe_manifest))
    save_moe_manifest(moe_manifest, args.export_llm_path)

    prompt = "RKLLM"
    chat_context = {
        "messages": [{"role": "user", "content": prompt}],
        "add_generation_prompt": True,
    }
    export_base = os.path.splitext(args.export_llm_path)[0]
    export_llm_config(
        args.model_path,
        export_base + ".config.pkl",
        chat_context,
        prompt,
        user_config={
            "image_token_id": 128815,
            "image_token": "<image>",
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
        },
    )
    export_unlimited_ocr_tokenizer(args.model_path, export_base + ".tokenizer.gguf")
    export_embed_weight(model.model.embed_tokens.weight, export_base + ".embed.bin")


if __name__ == "__main__":
    main()
