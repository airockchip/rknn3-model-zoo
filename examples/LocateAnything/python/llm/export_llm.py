import json
import os
import pickle
import shutil
import sys

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from py_utils.export_llm_helper import export_embed_weight, export_llm_config, export_tokenizer, update_config
from py_utils.tools import clear_llm_external_weight_in_dir
from model.modeling_locateanything import LocateAnythingForConditionalGeneration
import model.modeling_qwen2 as modeling_qwen2
from model.modeling_qwen2 import rotate_half


# RoPE cos/sin computed once per forward, driven ONLY by position_ids. Shared
# across all decoder layers so the exported graph contains a single Cos/Sin
# branch that RKNN's `replace_rope_branch` can isolate and offload to the NPU.
_EXPORT_ROPE_CACHE = {}


def _export_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Drop-in replacement for ``apply_rotary_pos_emb`` used during export.

    The stock implementation gathers from a cos/sin table that is sliced by a
    shape-derived ``seq_len`` (tainting the branch with ``input_embeds``), which
    prevents RKNN from detecting a position-only RoPE branch. Here cos/sin are
    precomputed from ``position_ids`` alone in the wrapper forward.
    """
    cos = _EXPORT_ROPE_CACHE["cos"].unsqueeze(unsqueeze_dim)
    sin = _EXPORT_ROPE_CACHE["sin"].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


PROMPT = "RKLLM"
CHAT_CONTEXT = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT},
            ],
        }
    ],
    "add_generation_prompt": True,
}


class LocateAnythingTextOnly(torch.nn.Module):
    """Export the Qwen2 decoder + lm_head as a plain causal (AR) LLM for RKNN.

    The model natively decodes with MTP / block-diffusion attention
    (``block_size`` > 1, ``causal_attn=False``), built via data-dependent Python
    control flow that cannot be traced to ONNX and is not supported by the
    standard RKNN LLM path. For RKNN 1820 deployment we therefore export the
    AR / "slow" mode: a standard causal Qwen2 decoder.

    This wrapper bypasses the block-mask builder and runs the decoder layers with
    a plain 4D causal mask. KV-cache is intentionally disabled here; on device the
    RKNN runtime manages the KV-cache via ``attention_config``.

    The exported RKNN LLM takes ``input_embeds`` directly. During deployment, look
    up text embeddings from ``*.embed.bin`` and replace image-token slots with the
    Vision RKNN output before calling the LLM.
    """

    def __init__(self, language_model):
        super().__init__()
        # Qwen2Model (decoder stack) and the lm_head, kept separately so we can
        # drive the layers ourselves without the MTP block-mask logic.
        self.model = language_model.model
        self.lm_head = language_model.lm_head
        self.config = language_model.config
        # Constant rotary inv_freq (taken from the model so it matches exactly).
        inv_freq = language_model.model.layers[0].self_attn.rotary_emb.inv_freq
        self.register_buffer("inv_freq", inv_freq.clone(), persistent=False)
        # Route every layer's RoPE through the position-only implementation.
        modeling_qwen2.apply_rotary_pos_emb = _export_apply_rotary_pos_emb

    def forward(self, input_embeds, attention_mask=None, position_ids=None):
        bsz, seq_len, _ = input_embeds.shape

        # Standard causal mask (folds in the 2D padding mask if provided).
        # past_key_values_length=0 because the exported graph is cache-free.
        attn_mask_4d = _prepare_4d_causal_attention_mask(
            attention_mask,
            (bsz, seq_len),
            input_embeds,
            0,
            sliding_window=self.config.sliding_window,
        )

        # Build cos/sin from position_ids only (single Cos/Sin branch for RKNN).
        inv_freq = self.inv_freq.to(torch.float32).view(1, 1, -1)
        freqs = position_ids[..., None].to(torch.float32) * inv_freq  # [1, seq, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [1, seq, dim]
        _EXPORT_ROPE_CACHE["cos"] = emb.cos().to(input_embeds.dtype)
        _EXPORT_ROPE_CACHE["sin"] = emb.sin().to(input_embeds.dtype)

        hidden_states = input_embeds
        for decoder_layer in self.model.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                past_key_value=None,
                use_cache=False,
            )[0]
        hidden_states = self.model.norm(hidden_states)

        logits = self.lm_head(hidden_states)
        return logits.float()


def prepare_export_config(config):
    if not hasattr(config, 'rope_parameters'):
        config.rope_parameters = None
    if not hasattr(config.vision_config, 'rope_parameters'):
        config.vision_config.rope_parameters = None
    update_config(config, ["use_cache"], False)
    update_config(config, ["_attn_implementation_autoset"], False)
    config._attn_implementation = 'eager'
    config.text_config._attn_implementation = 'eager'
    config.vision_config._attn_implementation = 'eager'
    return config


def load_tokenizer(model_path):
    try:
        return AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def export_locateanything_llm_to_onnx(model, args):
    model.eval().float()
    hidden_size = args.hidden_size
    seq_len = args.prompt_size

    input_embeds = torch.zeros((1, seq_len, hidden_size), dtype=torch.float32)
    attention_mask = torch.ones((1, seq_len), dtype=torch.float32)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    input_names = ["input_embeds", "attention_mask", "position_ids"]
    output_names = ["output"]
    dynamic_axes = {
        "input_embeds": {1: "sequence"},
        "attention_mask": {1: "sequence"},
        "position_ids": {1: "sequence"},
    }

    print(f"input_embeds: {tuple(input_embeds.shape)}")
    print(f"attention_mask: {tuple(attention_mask.shape)}")
    print(f"position_ids: {tuple(position_ids.shape)}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_embeds, attention_mask, position_ids),
            args.export_llm_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    print(f"Exported ONNX to {os.path.abspath(args.export_llm_path)}")


def gen_locateanything_llm_dataset(model_path, embed_layer, dataset_path, dataset_out_path, dataset_np_dir):
    if os.path.exists(dataset_np_dir):
        shutil.rmtree(dataset_np_dir)
    os.makedirs(dataset_np_dir, exist_ok=True)
    os.makedirs(os.path.dirname(dataset_out_path), exist_ok=True)

    tokenizer = load_tokenizer(model_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        datasets = json.load(f)

    embed_layer = embed_layer.cpu().eval()
    with open(dataset_out_path, "w", encoding="utf-8") as out:
        for i, data in enumerate(datasets):
            text = data.get("input", "") + data.get("target", "")
            input_ids = tokenizer.encode(text, return_tensors="pt")
            with torch.no_grad():
                input_embeds = embed_layer(input_ids).float().numpy()

            seq_len = input_embeds.shape[1]
            attention_mask = np.ones((1, seq_len), dtype=np.float32)
            position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)

            input_path = os.path.join(dataset_np_dir, f"input_embeds_{i}.npy")
            mask_path = os.path.join(dataset_np_dir, f"attention_mask_{i}.npy")
            pos_path = os.path.join(dataset_np_dir, f"position_ids_{i}.npy")
            np.save(input_path, input_embeds)
            np.save(mask_path, attention_mask)
            np.save(pos_path, position_ids)

            out.write(
                os.path.abspath(input_path)
                + " "
                + os.path.abspath(mask_path)
                + " "
                + os.path.abspath(pos_path)
                + "\n"
            )
    print(f"Quantization dataset exported to {dataset_out_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export LocateAnything-3B LLM ONNX/config/tokenizer/embed for RKNN")
    parser.add_argument("--load_weight", type=int, default=True, help="Whether to load model weights")
    parser.add_argument("--quan_dataset", type=int, default=True, help="Whether to generate quantization dataset")
    parser.add_argument("--model_path", type=str, default="../../models/LocateAnything-3B/hf/")
    parser.add_argument("--export_llm_path", type=str, default="../../model/llm/locateanything-3b-llm.onnx")
    parser.add_argument("--dataset", type=str, default="../../../../datasets/MMBench/llm/dataset.json")
    parser.add_argument("--dataset_out", type=str, default="../../data/llm/dataset.txt")
    parser.add_argument("--dataset_np_dir", type=str, default="../../data/llm/dataset_np")
    parser.add_argument("--prompt_size", type=int, default=64)
    args = parser.parse_args()

    kwargs = {"trust_remote_code": True}
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    prepare_export_config(config)

    if args.load_weight:
        model = LocateAnythingForConditionalGeneration.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
    else:
        model = LocateAnythingForConditionalGeneration(config).eval()

    export_dir = os.path.dirname(args.export_llm_path)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)

    wrapped_model = LocateAnythingTextOnly(model.language_model).eval()
    args.hidden_size = config.text_config.hidden_size
    export_locateanything_llm_to_onnx(wrapped_model, args)

    if args.load_weight and args.quan_dataset:
        gen_locateanything_llm_dataset(
            args.model_path,
            model.language_model.get_input_embeddings(),
            args.dataset,
            args.dataset_out,
            args.dataset_np_dir,
        )

    config_path = os.path.splitext(args.export_llm_path)[0] + ".config.pkl"
    user_config = {
        "llm_input_type": "input_embeds",
        "image_token_index": config.image_token_index,
        "block_size": config.text_config.block_size,
        "causal_attn": config.text_config.causal_attn,
    }
    export_llm_config(args.model_path, config_path, CHAT_CONTEXT, PROMPT, user_config)

    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + ".tokenizer.gguf")
    export_embed_weight(
        model.language_model.get_input_embeddings().weight,
        os.path.splitext(args.export_llm_path)[0] + ".embed.bin",
    )

    if not args.load_weight:
        clear_llm_external_weight_in_dir(export_dir)
