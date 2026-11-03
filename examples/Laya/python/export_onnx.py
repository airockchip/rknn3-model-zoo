"""Export the dynamic Laya decision graph and CPU resources."""

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn.functional as functional
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer

from laya_config import (
    ACT_STATS, DEFAULT_MAX_OPTIONS, INPUT_NAMES, OUTPUT_NAMES, QTYPES,
    SEQUENCE_BUCKET_STEP, positive_dimension, validate_config,
)


class DecisionModel(torch.nn.Module):
    """Checkpoint architecture; deployment uses the unmasked decision head."""

    def __init__(self, encoder_config, head_layers, act_hidden, act_outputs):
        super().__init__()
        self.encoder = AutoModel.from_config(encoder_config,
                                             attn_implementation="eager")
        width = encoder_config.hidden_size
        layer = torch.nn.TransformerEncoderLayer(
            width, width // 64, 4 * width, 0.1, batch_first=True,
            norm_first=True)
        self.head = torch.nn.TransformerEncoder(
            layer, head_layers, enable_nested_tensor=False)
        self.type_emb = torch.nn.Embedding(len(QTYPES), width)
        self.scorer = torch.nn.Sequential(
            torch.nn.LayerNorm(width), torch.nn.Linear(width, width),
            torch.nn.GELU(), torch.nn.Linear(width, 1))
        self.act_head = torch.nn.Sequential(
            torch.nn.Linear(width + ACT_STATS, act_hidden), torch.nn.GELU(),
            torch.nn.Linear(act_hidden, act_outputs))
        self.register_buffer("temperature", torch.ones(len(QTYPES)))


class LayaMainGraph(torch.nn.Module):
    """Export encoder/scorer outputs consumed by the CPU act head."""

    def __init__(self, model, max_options=DEFAULT_MAX_OPTIONS):
        super().__init__()
        self.model = model
        self.max_options = max_options

    def head_layer(self, layer, hidden_states):
        # Keep sequence-dependent reshapes dynamic in the legacy ONNX exporter.
        normalized = layer.norm1(hidden_states)
        attention = layer.self_attn
        projected = functional.linear(
            normalized, attention.in_proj_weight, attention.in_proj_bias)
        projected = projected.reshape(
            1, -1, 3, attention.num_heads,
            attention.embed_dim // attention.num_heads)
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(0)
        attended = functional.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(1, -1, attention.embed_dim)
        hidden_states = hidden_states + attention.out_proj(attended)
        normalized = layer.norm2(hidden_states)
        return hidden_states + layer.linear2(
            layer.activation(layer.linear1(normalized)))

    def forward(self, inputs_embeds, attention_mask, marker_pos,
                marker_mask, qtype):
        hidden_states = self.model.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.long()).last_hidden_state
        hidden_states = (
            hidden_states + self.model.type_emb(qtype.long())[:, None, :]
        )
        # Decision-head attention is unmasked; encoder padding remains masked.
        for layer in self.model.head.layers:
            hidden_states = self.head_layer(layer, hidden_states)
        indices = marker_pos.long().clamp(min=0, max=hidden_states.size(1) - 1)
        indices = indices[:, :, None].expand(-1, -1, hidden_states.size(-1))
        markers = torch.gather(hidden_states, 1, indices)
        logits = self.model.scorer(markers).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask.bool(), -1e4)
        probabilities = torch.softmax(logits.detach(), -1)
        option_count = (
            marker_mask.bool().sum(-1).float().clamp(
                min=2.0, max=float(self.max_options))
        )
        return (
            logits[0], probabilities[0], option_count,
            hidden_states[:, 0].float()[0],
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export the Laya main graph and CPU resources")
    parser.add_argument(
        "--model_path", required=True,
        help="Local model directory or Hugging Face/ModelScope ID")
    parser.add_argument(
        "--output_path", required=True,
        help="Output directory for ONNX and CPU resources")
    parser.add_argument(
        "--modelscope", action="store_true",
        help="Download all non-local model paths from ModelScope")
    return parser.parse_args()


def resolve_model_path(model_path, modelscope):
    """Resolve a local checkpoint or download the required remote files."""
    source = Path(model_path).expanduser()
    if not source.is_dir():
        if source.is_absolute() or model_path.startswith("."):
            raise FileNotFoundError(source)
        if modelscope:
            from modelscope import snapshot_download
        else:
            from huggingface_hub import snapshot_download
        source = Path(snapshot_download(model_path, allow_patterns=[
            "rl_agent_config.json", "model.safetensors", "encoder/*",
            "tokenizer/*"]))
    for name in ["rl_agent_config.json", "model.safetensors",
                 "encoder/config.json", "tokenizer/tokenizer.json",
                 "tokenizer/tokenizer_config.json"]:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    return source


def load_encoder_config(source):
    """Translate modern RoPE fields for the pinned Transformers 4.57.3."""
    raw = json.loads((source / "encoder/config.json").read_text())
    encoder_config = AutoConfig.from_pretrained(
        source / "encoder", local_files_only=True)
    if encoder_config.model_type != "modernbert":
        raise ValueError("Only Laya with a ModernBERT encoder is supported")
    # 4.57.3 chooses attention types by interval, ignoring layer_types.
    interval = positive_dimension(
        encoder_config.global_attn_every_n_layers, "global_attn_every_n_layers")
    expected = [
        "sliding_attention" if index % interval else "full_attention"
        for index in range(encoder_config.num_hidden_layers)
    ]
    if raw.get("layer_types", expected) != expected:
        raise ValueError("layer_types disagrees with global_attn_every_n_layers")
    parameters = raw.get("rope_parameters")
    if parameters is not None:
        fields = {"full_attention": "global_rope_theta",
                  "sliding_attention": "local_rope_theta"}
        if not isinstance(parameters, dict) or set(parameters) != set(fields):
            raise ValueError("RoPE requires full_attention and sliding_attention")
        if raw.get("rope_scaling") is not None:
            raise ValueError("Cannot combine rope_parameters and rope_scaling")
        for kind, field in fields.items():
            rope = parameters[kind]
            if (not isinstance(rope, dict)
                    or set(rope) != {"rope_type", "rope_theta"}
                    or rope["rope_type"] != "default"):
                raise ValueError(f"Only default RoPE is supported for {kind}")
            theta = rope["rope_theta"]
            if (type(theta) not in (int, float)
                    or not math.isfinite(theta) or theta <= 0):
                raise ValueError(f"Invalid RoPE theta for {kind}")
            if field in raw and raw[field] != theta:
                raise ValueError(f"{field} conflicts with rope_parameters")
            # Set legacy fields explicitly; preserve rope_parameters for 5.x.
            setattr(encoder_config, field, theta)
    return encoder_config


def sequence_buckets(config, encoder_config):
    """Keep the checkpoint context limit, including a non-aligned final bucket."""
    length = positive_dimension(config["max_len"], "max_len")
    capacity = positive_dimension(
        encoder_config.max_position_embeddings, "max_position_embeddings")
    if not DEFAULT_MAX_OPTIONS <= length <= capacity:
        raise ValueError(
            "max_len must fit the encoder capacity and supported option count")
    return [*range(SEQUENCE_BUCKET_STEP, length, SEQUENCE_BUCKET_STEP), length]


def load_model(source):
    config = json.loads((source / "rl_agent_config.json").read_text())
    encoder_config = load_encoder_config(source)
    sequence_buckets(config, encoder_config)
    if type(config["head_layers"]) is not int or config["head_layers"] < 1:
        raise ValueError("Laya requires at least one decision-head layer")
    weights = load_file(str(source / "model.safetensors"))
    act_hidden, act_inputs = weights["act_head.0.weight"].shape
    act_outputs = weights["act_head.2.weight"].shape[0]
    if (act_inputs != encoder_config.hidden_size + ACT_STATS
            or act_outputs != len(config["act_costs"]) + 1):
        raise ValueError("Act weights disagree with the encoder/configuration")
    torch.set_num_threads(4)
    model = DecisionModel(encoder_config, config["head_layers"],
                          act_hidden, act_outputs).eval()
    model.load_state_dict(weights, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(
        source / "tokenizer", local_files_only=True)
    return model, tokenizer, config


def export_cpu_resources(model, tokenizer, config, source, work):
    """Write CPU resource layouts shared by the exporter and C demo."""
    buckets = sequence_buckets(config, model.encoder.config)
    embedding = model.encoder.get_input_embeddings().weight.detach().numpy()
    embedding.astype("<f2").tofile(work / "embedding.bin")
    with (work / "act.bin").open("wb") as stream:
        for name in ["0.weight", "0.bias", "2.weight", "2.bias"]:
            model.act_head.state_dict()[name].numpy().astype("<f4").tofile(
                stream)
    tokenizer_config = json.loads(
        (source / "tokenizer/tokenizer.json").read_text())
    # Sequence assembly handles special tokens, padding and truncation.
    tokenizer_config.update(post_processor=None, padding=None, truncation=None)
    (work / "tokenizer.json").write_text(
        json.dumps(tokenizer_config, ensure_ascii=False), encoding="utf-8")
    # Keep the tokenizer's role names; the C API resolves their vocabulary IDs.
    shutil.copy2(source / "tokenizer/tokenizer_config.json",
                 work / "tokenizer_config.json")
    metadata = {
        'seq_len': buckets[-1],
        'sequence_buckets': buckets,
        'hidden_size': embedding.shape[1],
        'max_options': DEFAULT_MAX_OPTIONS,
        'vocab_size': len(embedding),
        'act_head': {
            'input_size': model.act_head[0].in_features,
            'hidden_size': model.act_head[0].out_features,
            'output_size': model.act_head[2].out_features,
        },
        'head_max_len': config['head_max_len'],
        'temperature': config['temperature'],
        'temperature_by_options': config.get('temperature_by_options', {}),
    }
    validate_config(metadata)
    (work / "laya.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def make_export_inputs(model, tokenizer, metadata):
    """Build a valid sample request for ONNX tracing."""
    length = metadata["sequence_buckets"][0]
    options = metadata["max_options"]
    ids = torch.full((1, length), tokenizer.pad_token_id, dtype=torch.long)
    ids[0, :4] = torch.tensor([tokenizer.cls_token_id, tokenizer.mask_token_id,
                              tokenizer.mask_token_id, tokenizer.sep_token_id])
    attn = torch.zeros(1, length, dtype=torch.int32)
    attn[0, :4] = 1
    markers = torch.zeros(1, options, dtype=torch.int32)
    markers[0, :2] = torch.tensor([1, 2])
    mask = torch.zeros(1, options, dtype=torch.int32)
    mask[0, :2] = 1
    inputs = (model.encoder.get_input_embeddings()(ids).detach(), attn,
             markers, mask, torch.zeros(1, dtype=torch.int32))
    return inputs


def export_model(source, destination):
    model, tokenizer, config = load_model(source)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            prefix="laya-export-", dir=destination) as temp:
        work = Path(temp)
        metadata = export_cpu_resources(model, tokenizer, config, source, work)
        inputs = make_export_inputs(model, tokenizer, metadata)
        torch.onnx.export(
            LayaMainGraph(model, metadata["max_options"]).eval(), inputs,
            str(work / "laya.onnx"),
            opset_version=17,
            dynamo=False, do_constant_folding=True,
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_axes={
                "inputs_embeds": {1: "sequence"},
                "attention_mask": {1: "sequence"},
            })
        onnx.checker.check_model(str(work / "laya.onnx"))
        for item in work.iterdir():
            item.replace(destination / item.name)


def main():
    args = parse_args()
    source = resolve_model_path(args.model_path, args.modelscope)
    destination = Path(args.output_path).expanduser().resolve()
    export_model(source, destination)
    print(f"Export completed: {destination}")


if __name__ == "__main__":
    main()
