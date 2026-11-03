"""Shared deployment metadata, tensor contract and upstream algorithm constants."""

import json
import math
from pathlib import Path

SEQUENCE_BUCKET_STEP = 128
DEFAULT_MAX_OPTIONS = 128
ACT_STATS = 4
HEAD_RESERVED_TOKENS = 16
QTYPES = {"choice": 0, "score": 1, "noul": 2}
INPUT_NAMES = ("inputs_embeds", "attention_mask", "marker_pos", "marker_mask",
               "qtype")
OUTPUT_NAMES = ("logits", "p", "k", "h0")


def positive_dimension(value, name):
    if type(value) is not int or not 0 < value <= 2**31 - 1:
        raise ValueError(f"Invalid {name}: expected a positive int32 dimension")
    return value


def act_dimensions(config):
    """Read the act dimensions recorded by the ONNX exporter."""
    width = positive_dimension(config["hidden_size"], "hidden_size")
    head = config.get("act_head")
    if not isinstance(head, dict):
        raise ValueError("act_head must contain linear-layer dimensions")
    dims = tuple(positive_dimension(head[key], f"act_head.{key}") for key in
                 ("input_size", "hidden_size", "output_size"))
    inputs, hidden, outputs = dims
    if inputs != width + ACT_STATS:
        raise ValueError("Act input size must equal hidden_size plus four stats")
    count = hidden * inputs + hidden + outputs * hidden + outputs
    if count > (2**31 - 1) // 4:
        raise ValueError("Act weights exceed the C resource reader capacity")
    return dims


def validate_config(config):
    """Validate ONNX-side metadata before conversion."""
    if not isinstance(config, dict):
        raise ValueError("Laya metadata must be an object")
    for name in ("seq_len", "hidden_size", "max_options", "vocab_size",
                 "head_max_len"):
        positive_dimension(config[name], name)
    if not 2 <= config["max_options"] <= config["seq_len"]:
        raise ValueError("max_options must be between 2 and seq_len")
    if config["head_max_len"] < HEAD_RESERVED_TOKENS:
        raise ValueError("head_max_len is smaller than the reserved head budget")
    buckets = config["sequence_buckets"]
    if not isinstance(buckets, list) or not buckets:
        raise ValueError("sequence_buckets must be a non-empty list")
    for length in buckets:
        positive_dimension(length, "sequence bucket")
    if buckets != sorted(set(buckets)) or buckets[-1] != config["seq_len"]:
        raise ValueError("Buckets must increase uniquely and end at seq_len")
    temperatures = config["temperature"]
    overrides = config["temperature_by_options"]
    if (not isinstance(temperatures, list) or len(temperatures) != len(QTYPES)
            or not isinstance(overrides, dict)):
        raise ValueError("Invalid temperature configuration")
    for value in [*temperatures, *overrides.values()]:
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Temperatures must be positive and finite")
    act_dimensions(config)
    return config


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def runtime_config(config):
    """Package only CPU parameters; RKNN/tokenizer provide the other properties."""
    _, hidden, outputs = act_dimensions(config)
    return {
        "head_max_len": config["head_max_len"],
        "act_head": {"hidden_size": hidden, "output_size": outputs},
        "temperature": config["temperature"],
        "temperature_by_options": config["temperature_by_options"],
    }


def input_shapes(config, length):
    width, options = config["hidden_size"], config["max_options"]
    return [[1, length, width], [1, length], [1, options], [1, options], [1]]


def validate_onnx(path, config):
    """Reject metadata/graph mismatches before starting the RKNN compiler."""
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    shapes = input_shapes(config, None)
    shapes += [[config["max_options"]], [config["max_options"]], [1],
               [config["hidden_size"]]]
    tensors = [*graph.input, *graph.output]
    if [item.name for item in tensors] != [*INPUT_NAMES, *OUTPUT_NAMES]:
        raise ValueError("ONNX input/output names do not match the Laya contract")
    for index, (tensor, expected) in enumerate(zip(tensors, shapes)):
        value = tensor.type.tensor_type
        dtype = (onnx.TensorProto.INT32 if 0 < index < len(INPUT_NAMES)
                 else onnx.TensorProto.FLOAT)
        if value.elem_type != dtype or len(value.shape.dim) != len(expected):
            raise ValueError(f"Invalid ONNX dtype or rank: {tensor.name}")
        for dim, size in zip(value.shape.dim, expected):
            if size is None:
                if dim.HasField("dim_value"):
                    if config["sequence_buckets"] != [dim.dim_value]:
                        raise ValueError(
                            "Static ONNX length does not match the buckets")
                elif not dim.dim_param:
                    raise ValueError("ONNX sequence dimension is missing")
            elif not dim.HasField("dim_value") or dim.dim_value != size:
                raise ValueError(f"ONNX shape disagrees with metadata: {tensor.name}")
