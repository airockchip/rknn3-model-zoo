import argparse
import base64
import pickle
import tempfile
from pathlib import Path

import onnx
import torch
from transformers import AutoTokenizer, WhisperForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
from rknn.utils import onnx_edit

def model_suffix(model_id):
    return model_id.rsplit("/", 1)[-1].replace("whisper-", "")


def cache_names(prefix, num_layers):
    names = []
    for layer_idx in range(num_layers):
        names.append(f"{prefix}_key_{layer_idx}")
        names.append(f"{prefix}_value_{layer_idx}")
    return names


def write_config(path, payload):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def print_onnx_io_shapes(path, names, io_kind):
    model = onnx.load(str(path))
    values = model.graph.output if io_kind == "output" else model.graph.input
    value_by_name = {value.name: value for value in values}
    for name in names:
        value = value_by_name.get(name)
        if value is None:
            print(f"{path.name} {io_kind} {name} shape=<not found>")
            continue
        dims = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param)
        print(f"{path.name} {io_kind} {name} shape={dims}")


def build_empty_encoder_decoder_cache():
    return EncoderDecoderCache(DynamicCache(), DynamicCache())


def flatten_cross_cache(cache, num_layers):
    tensors = []
    for layer_idx in range(num_layers):
        tensors.append(cache.cross_attention_cache.key_cache[layer_idx])
        tensors.append(cache.cross_attention_cache.value_cache[layer_idx])
    return tuple(tensors)


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_features):
        encoder = self.encoder
        hidden_states = torch.nn.functional.gelu(encoder.conv1(input_features))
        hidden_states = torch.nn.functional.gelu(encoder.conv2(hidden_states)).permute(0, 2, 1)
        hidden_states = hidden_states + encoder.embed_positions.weight[:hidden_states.shape[1], :]
        for encoder_layer in encoder.layers:
            hidden_states = encoder_layer(hidden_states, None, None, output_attentions=False)[0]
        return encoder.layer_norm(hidden_states)


class EncoderWithMaskWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_features, encoder_attention_mask):
        encoder = self.encoder

        hidden_states = torch.nn.functional.gelu(encoder.conv1(input_features))
        hidden_states = torch.nn.functional.gelu(encoder.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)

        hidden_states = hidden_states + encoder.embed_positions.weight[:hidden_states.shape[1], :]
        hidden_states = torch.nn.functional.dropout(hidden_states, p=encoder.dropout, training=encoder.training)

        for encoder_layer in encoder.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                encoder_attention_mask,
                None,
                output_attentions=False,
            )
            hidden_states = layer_outputs[0]

        hidden_states = encoder.layer_norm(hidden_states)
        return hidden_states


class Decode0Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder
        self.num_layers = model.config.decoder_layers

    def forward(self, input_embeds, encoder_outputs):
        cache = build_empty_encoder_decoder_cache()
        outputs = self.decoder(inputs_embeds=input_embeds, encoder_hidden_states=encoder_outputs, past_key_values=cache, use_cache=True, return_dict=True)
        return flatten_cross_cache(outputs.past_key_values, self.num_layers)


class Decode1CrossKVWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.decoder.embed_tokens
        self.embed_positions = model.model.decoder.embed_positions
        self.layers = model.model.decoder.layers
        self.layer_norm = model.model.decoder.layer_norm
        self.proj_out = model.proj_out
        self.num_layers = model.config.decoder_layers
        self.num_heads = model.config.decoder_attention_heads
        self.head_dim = model.config.d_model // model.config.decoder_attention_heads
        self.hidden_size = model.config.d_model
        self.dropout = model.model.decoder.dropout

    def forward(self, input_ids, attention_mask, position_ids, num_logits_to_keep, *cross_tensors):
        expected_len = 2 * self.num_layers
        if len(cross_tensors) != expected_len:
            raise ValueError(f"Invalid cross_tensors length: expected={expected_len}, actual={len(cross_tensors)}")

        inputs_embeds = self.embed_tokens(input_ids)
        positions = self.embed_positions(inputs_embeds, position_ids=position_ids).to(inputs_embeds.device)
        hidden_states = inputs_embeds + positions
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        for layer_idx, layer in enumerate(self.layers):
            cross_key = cross_tensors[layer_idx * 2]
            cross_value = cross_tensors[layer_idx * 2 + 1]

            residual = hidden_states
            hidden_states = layer.self_attn_layer_norm(hidden_states)
            hidden_states, _ = layer.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=False)[:2]
            hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = layer.encoder_attn_layer_norm(hidden_states)
            bsz, tgt_len, _ = hidden_states.shape
            q = layer.encoder_attn.q_proj(hidden_states)
            q = (q * layer.encoder_attn.scaling).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
            attn_weights = torch.matmul(q, cross_key.transpose(-2, -1))
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, cross_value)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.hidden_size)
            attn_output = layer.encoder_attn.out_proj(attn_output)
            attn_output = torch.nn.functional.dropout(attn_output, p=self.dropout, training=self.training)
            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = layer.final_layer_norm(hidden_states)
            hidden_states = layer.activation_fn(layer.fc1(hidden_states))
            hidden_states = torch.nn.functional.dropout(hidden_states, p=layer.activation_dropout, training=self.training)
            hidden_states = layer.fc2(hidden_states)
            hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        logits = self.proj_out(hidden_states)
        return logits, num_logits_to_keep


def validate_args(args):
    if args.seq_len < 1:
        raise ValueError("--seq_len must be >= 1")
    if args.past_seq_len < 1:
        raise ValueError("--past_seq_len must be >= 1")


def prepare_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_model_path(args):
    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)
    return args.model_path


def load_whisper_model(model_path):
    model = WhisperForConditionalGeneration.from_pretrained(model_path, attn_implementation="eager")
    model.eval()
    model.requires_grad_(False)
    return model


def print_model_info(model, model_path, suffix):
    config = model.config
    print(f"model_path={model_path}")
    print(f"suffix={suffix}")
    print(f"decoder_layers={config.decoder_layers}")
    print(f"decoder_attention_heads={config.decoder_attention_heads}")
    print(f"d_model={config.d_model}")
    print(f"max_source_positions={config.max_source_positions}")
    print(f"max_target_positions={config.max_target_positions}")


def export_runtime_resources(model, model_path, output_path):
    embedding = model.model.decoder.embed_tokens.weight.detach().cpu().to(torch.float16).numpy()
    expected_shape = (model.config.vocab_size, model.config.d_model)
    if tuple(embedding.shape) != expected_shape:
        raise ValueError(f"Invalid token embedding shape: expected={expected_shape}, actual={tuple(embedding.shape)}")
    embedding.tofile(output_path / "token_embed.bin")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError(f"Tokenizer/model vocab mismatch: tokenizer={len(tokenizer)}, model={model.config.vocab_size}")
    # Byte-level BPE tokens can contain incomplete UTF-8 characters.
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            characters.append(256 + extra)
            byte_values.append(value)
            extra += 1
    byte_decoder = dict(zip(map(chr, characters), byte_values))
    special_ids = set(tokenizer.all_special_ids)
    with open(output_path / "vocab_en.txt", "w", encoding="utf-8") as en_file, \
         open(output_path / "vocab_zh.txt", "w", encoding="utf-8") as zh_file:
        for token_id in range(model.config.vocab_size):
            token_piece = tokenizer.convert_ids_to_tokens(token_id) or ""
            raw_piece = (
                b"" if token_id in special_ids
                else bytes(byte_decoder[char] for char in token_piece)
            )
            encoded_piece = base64.b64encode(raw_piece).decode("ascii")
            en_file.write(f"{token_id} {token_piece}\n")
            zh_file.write(f"{token_id} {encoded_piece}\n")


def audio_duration_to_encoder_seq_len(audio_duration):
    if audio_duration <= 0 or audio_duration > 30:
        raise ValueError("--audio_duration must be in [1, 30]")
    return audio_duration * 50


def export_encoder(model, output_dir, suffix, model_path, encoder_seq_len, encoder_mask=False):
    config = model.config
    encoder_input_features_len = encoder_seq_len * 2
    input_features = torch.randn(1, config.num_mel_bins, encoder_input_features_len, dtype=torch.float32)
    encoder_attention_mask = None

    if encoder_mask:
        encoder = EncoderWithMaskWrapper(model)
        encoder_attention_mask = torch.zeros(1, 1, 1, encoder_seq_len, dtype=torch.float32)
        encoder_inputs = (input_features, encoder_attention_mask)
        encoder_input_names = ["x", "encoder_attention_mask"]
    else:
        encoder = EncoderWrapper(model)
        encoder_inputs = (input_features,)
        encoder_input_names = ["x"]

    with torch.no_grad():
        encoder_outputs = encoder(*encoder_inputs)

    print(f"encoder input_features shape={tuple(input_features.shape)}")
    if encoder_attention_mask is not None:
        print(f"encoder attention_mask shape={tuple(encoder_attention_mask.shape)}")
    print(f"encoder outputs shape={tuple(encoder_outputs.shape)}")

    expected_encoder_shape = (1, encoder_seq_len, config.d_model)
    if tuple(encoder_outputs.shape) != expected_encoder_shape:
        raise ValueError(f"Invalid encoder output shape: expected={expected_encoder_shape}, actual={tuple(encoder_outputs.shape)}")

    encoder_path = output_dir / f"whisper_encoder_{suffix}.onnx"
    encoder_config_path = output_dir / f"whisper_encoder_{suffix}.config.pkl"
    encoder_output_names = ["out"]

    torch.onnx.export(encoder, encoder_inputs, str(encoder_path), input_names=encoder_input_names, output_names=encoder_output_names, opset_version=18, do_constant_folding=True)
    onnx.checker.check_model(str(encoder_path))
    encoder_config = {
        "model_id": model_path,
        "stage": "encoder",
        "inputs": encoder_input_names,
        "outputs": encoder_output_names,
        "num_mel_bins": config.num_mel_bins,
        "hidden_size": config.d_model,
        "audio_duration": encoder_input_features_len // 100,
        "input_features_shape": tuple(input_features.shape),
        "encoder_seq_len": encoder_outputs.shape[1],
        "encoder_mask": encoder_mask,
    }
    if encoder_attention_mask is not None:
        encoder_config["encoder_attention_mask_shape"] = tuple(encoder_attention_mask.shape)
        encoder_config["encoder_attention_mask_type"] = "additive_padding_mask"
        encoder_config["encoder_attention_mask_valid_value"] = 0.0
        encoder_config["encoder_attention_mask_pad_value"] = -65504.0
    write_config(encoder_config_path, encoder_config)
    print(f"saved {encoder_path}")
    print(f"saved {encoder_config_path}")


def build_decode0_inputs_and_cross_cache(model, seq_len, encoder_seq_len):
    config = model.config
    num_layers = config.decoder_layers
    num_heads = config.decoder_attention_heads
    head_dim = config.d_model // config.decoder_attention_heads

    input_ids = torch.full((1, seq_len), config.decoder_start_token_id, dtype=torch.long)
    inputs_embeds = model.model.decoder.embed_tokens(input_ids)
    encoder_outputs = torch.randn(1, encoder_seq_len, config.d_model, dtype=torch.float32)
    cache = build_empty_encoder_decoder_cache()

    with torch.no_grad():
        outputs = model.model.decoder(inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_outputs, past_key_values=cache, use_cache=True, return_dict=True)
        logits = model.proj_out(outputs.last_hidden_state)

    cross_cache = flatten_cross_cache(outputs.past_key_values, num_layers)
    print(f"decode0 logits shape={tuple(logits.shape)}")
    print(f"decode0 first cross cache shape={tuple(cross_cache[0].shape)}")

    expected_logits_shape = (1, seq_len)
    actual_logits_shape = tuple(logits.shape[:2])
    if actual_logits_shape != expected_logits_shape:
        raise ValueError(f"Invalid logits leading shape: expected={expected_logits_shape}, actual={actual_logits_shape}")

    expected_cross_cache_shape = (1, num_heads, encoder_seq_len, head_dim)
    actual_cross_cache_shape = tuple(cross_cache[0].shape)
    if actual_cross_cache_shape != expected_cross_cache_shape:
        raise ValueError(f"Invalid first cross cache shape: expected={expected_cross_cache_shape}, actual={actual_cross_cache_shape}")

    return inputs_embeds, encoder_outputs, cross_cache

def make_cross_kv_hdns_transform(num_layers):
    # 原始: [1, num_heads, encoder_seq_len, head_dim] = [N,H,S,D]
    # 外部: [num_heads, head_dim, 1, encoder_seq_len] = [H,D,N,S]
    # 公式: [N,H,S,D] -> [H,D,N,S]
    transform = "a,b,c,d->b,d,a,c"
    return {name: transform for name in cache_names("cross", num_layers)}

def export_decode0(model, output_dir, suffix, model_path, seq_len, encoder_seq_len):
    config = model.config
    num_layers = config.decoder_layers
    num_heads = config.decoder_attention_heads
    head_dim = config.d_model // config.decoder_attention_heads

    inputs_embeds, encoder_outputs, cross_cache = build_decode0_inputs_and_cross_cache(model, seq_len, encoder_seq_len)

    decode0_path = output_dir / f"whisper_decode0_{suffix}.onnx"
    decode0_config_path = output_dir / f"whisper_decode0_{suffix}.config.pkl"

    decode0 = Decode0Wrapper(model)
    decode0_output_names = cache_names("cross", num_layers)
    outputs_transform = make_cross_kv_hdns_transform(num_layers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_decode0_path = Path(tmp_dir) / "raw_decode0.onnx"
        torch.onnx.export(decode0, (inputs_embeds, encoder_outputs), str(raw_decode0_path), input_names=["input_embeds", "encoder_outputs"], output_names=decode0_output_names, opset_version=18, do_constant_folding=True)
        onnx.checker.check_model(str(raw_decode0_path))
        onnx_edit(model=str(raw_decode0_path), export_path=str(decode0_path), outputs_transform=outputs_transform)

    onnx.checker.check_model(str(decode0_path))
    print_onnx_io_shapes(decode0_path, ["cross_key_0", "cross_value_0"], "output")

    write_config(
        decode0_config_path,
        {
            "model_id": model_path,
            "stage": "decode0",
            "inputs": ["input_embeds", "encoder_outputs"],
            "outputs": decode0_output_names,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "hidden_size": config.d_model,
            "encoder_seq_len": encoder_seq_len,
            "seq_len": seq_len,
            "cross_kv_layout": "HDNS",
            "cross_kv_shape": [num_heads, head_dim, 1, encoder_seq_len],
            "outputs_transform": outputs_transform,
        },
    )
    print(f"saved {decode0_path}")
    print(f"saved {decode0_config_path}")

    return cross_cache


def export_decode1(model, output_dir, suffix, model_path, seq_len, past_seq_len, encoder_seq_len, cross_cache=None):
    config = model.config
    num_layers = config.decoder_layers
    num_heads = config.decoder_attention_heads
    head_dim = config.d_model // config.decoder_attention_heads

    if cross_cache is None:
        _, _, cross_cache = build_decode0_inputs_and_cross_cache(model, seq_len, encoder_seq_len)

    decode1_input_ids = torch.full((1, 1), config.decoder_start_token_id, dtype=torch.long)
    decode1_position_ids = torch.full((1, 1), past_seq_len, dtype=torch.long)
    decode1_num_logits_to_keep = torch.zeros((1,), dtype=torch.long)
    decode1_attention_mask = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
    decode1_inputs = (decode1_input_ids, decode1_attention_mask, decode1_position_ids, decode1_num_logits_to_keep, *cross_cache)

    decode1 = Decode1CrossKVWrapper(model)
    with torch.no_grad():
        decode1_logits, decode1_num_logits_out = decode1(*decode1_inputs)

    print(f"decode1 logits shape={tuple(decode1_logits.shape)}")
    print(f"decode1 num_logits_to_keep_out shape={tuple(decode1_num_logits_out.shape)}")

    if tuple(decode1_logits.shape[:2]) != (1, 1):
        raise ValueError(f"Invalid decode1 logits leading shape: expected={(1, 1)}, actual={tuple(decode1_logits.shape[:2])}")

    decode1_path = output_dir / f"whisper_decode1_{suffix}.onnx"
    decode1_config_path = output_dir / f"whisper_decode1_{suffix}.config.pkl"
    decode1_input_names = ["input_ids", "attention_mask", "position_ids", "num_logits_to_keep"] + cache_names("cross", num_layers)
    decode1_output_names = ["logits", "num_logits_to_keep_out"]
    inputs_transform = make_cross_kv_hdns_transform(num_layers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_decode1_path = Path(tmp_dir) / "raw_decode1.onnx"
        torch.onnx.export(decode1, decode1_inputs, str(raw_decode1_path), input_names=decode1_input_names, output_names=decode1_output_names, dynamic_axes={"input_ids": {1: "decode_seq_len"}, "attention_mask": {2: "decode_seq_len", 3: "decode_seq_len"}, "position_ids": {1: "decode_seq_len"}, "logits": {1: "decode_seq_len"}}, opset_version=18, do_constant_folding=True)
        onnx.checker.check_model(str(raw_decode1_path))
        onnx_edit(model=str(raw_decode1_path), export_path=str(decode1_path), inputs_transform=inputs_transform)

    onnx.checker.check_model(str(decode1_path))
    print_onnx_io_shapes(decode1_path, ["cross_key_0", "cross_value_0"], "input")

    write_config(
        decode1_config_path,
        {
            "model_id": model_path,
            "stage": "decode1_rknn_llm_cross_kv",
            "inputs": decode1_input_names,
            "outputs": decode1_output_names,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "hidden_size": config.d_model,
            "encoder_seq_len": encoder_seq_len,
            "past_seq_len": past_seq_len,
            "rknn_llm_cache": True,
            "max_position_embeddings": config.max_target_positions,
            "cross_kv_layout": "HDNS",
            "cross_kv_shape": [num_heads, head_dim, 1, encoder_seq_len],
            "inputs_transform": inputs_transform,
        },
    )
    print(f"saved {decode1_path}")
    print(f"saved {decode1_config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Transformers Whisper ONNX models for RKNN")
    parser.add_argument("--model_path", type=str, default="openai/whisper-base", help="HF model id or local model path")
    parser.add_argument("--output_dir", type=str, default="whisper-base-model", help="directory to save exported ONNX/config files")
    parser.add_argument("--modelscope", action="store_true", help="download model from modelscope before export")
    parser.add_argument("--model", type=str, default="all", choices=["encoder", "decode0", "decode1", "all"], help="which ONNX part to export")
    parser.add_argument("--seq_len", type=int, default=4, help="dummy decoder sequence length for decode0 export")
    parser.add_argument("--past_seq_len", type=int, default=4, help="dummy past length used by decode1 position_ids")
    parser.add_argument("--audio_duration", type=int, default=30, help="fixed input duration in seconds; use 15 for large-v3-turbo on RK3572")
    parser.add_argument("--encoder_mask", action="store_true", default=True, help="export encoder with additive padding mask input")

    args = parser.parse_args()
    validate_args(args)

    model_path = resolve_model_path(args)
    output_dir = prepare_output_dir(args.output_dir)

    model = load_whisper_model(model_path)
    suffix = model_suffix(model_path)
    print_model_info(model, model_path, suffix)
    encoder_seq_len = audio_duration_to_encoder_seq_len(args.audio_duration)
    print(f"audio_duration={args.audio_duration}, encoder_seq_len={encoder_seq_len}")

    cross_cache = None
    if args.model == "all":
        export_runtime_resources(model, model_path, output_dir)
    if args.model in ("encoder", "all"):
        export_encoder(model, output_dir, suffix, model_path, encoder_seq_len, args.encoder_mask)

    if args.model in ("decode0", "all"):
        cross_cache = export_decode0(model, output_dir, suffix, model_path, args.seq_len, encoder_seq_len)

    if args.model in ("decode1", "all"):
        export_decode1(model, output_dir, suffix, model_path, args.seq_len, args.past_seq_len, encoder_seq_len, cross_cache)
