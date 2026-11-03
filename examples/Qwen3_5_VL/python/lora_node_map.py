"""Build HF-to-ONNX and exact RKNN LoRA node mappings.

The matching algorithm follows the torch/ONNX mapping demo supplied with the
Qwen3-VL LoRA task: shape filtering (including a two-dimensional transpose),
an absolute-sum fingerprint, and a greedy streaming L1 comparison.

``rknn.load_lora(lora_node_map=...)`` does not use prefix/postfix matching.
The exact map produced here therefore contains the real ONNX node name and
the exact A/B keys present in the safetensors file.

The helpers are shared by both export pipelines of this example:

* ``python/llm/export_llm.py`` + ``python/llm/export_rknn.py`` use them for the
  language model (``scope="LLM"``).
* ``python/vision/export_vision.py`` + ``python/vision/export_rknn.py`` use them
  for the vision tower (``scope="vision"``).

Each converter exports the ONNX of one sub-model and immediately writes an
``*.hf_onnx_map.json`` with :func:`build_hf_onnx_mapping`. ``export_rknn.py``
later resolves the LoRA safetensors through that map with
:func:`build_lora_node_map`, which keeps only the pairs that belong to the
exported sub-model and copies them into a filtered safetensors file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import onnx
import torch
from onnx import numpy_helper
from safetensors import safe_open
from safetensors.torch import save_file


LORA_A_POSTFIX = ".lora_A.weight"
LORA_B_POSTFIX = ".lora_B.weight"
FINGERPRINT_TOLERANCE = 1.0
L1_CLEAN_TOLERANCE = 1.0


def _write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _as_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu")


def _torch_fingerprint(value: torch.Tensor, n: int = 1024) -> float:
    value = _as_cpu_tensor(value).reshape(-1)
    if not value.is_floating_point():
        value = value.float()
    if value.numel() > n:
        value = value[:n]
    return value.abs().sum().item()


def _onnx_fingerprint(initializer: onnx.TensorProto, n: int = 1024) -> float:
    # Slice before copying: Vision ONNX contains very large matrices, but the
    # fingerprint only needs the first ``n`` values.
    array = numpy_helper.to_array(initializer).reshape(-1)
    if array.size > n:
        array = array[:n]
    value = torch.from_numpy(array.copy())
    if not value.is_floating_point():
        value = value.float()
    return value.abs().sum().item()


def _load_onnx_tensor(onnx_info: dict) -> torch.Tensor:
    """Materialize one ONNX initializer at most once during matching.

    The original implementation converted the same ``TensorProto`` for every
    candidate comparison.  A Qwen3-VL graph contains many large matrices, so
    that repeated deserialization makes mapping unnecessarily expensive.
    Keep the cache lazy to avoid materializing initializers that are rejected
    by the shape/fingerprint filters.
    """

    value = onnx_info.get("value")
    if value is None:
        array = numpy_helper.to_array(onnx_info["init"])
        if not array.flags.writeable:
            array = array.copy()
        value = torch.from_numpy(array)
        if not value.is_contiguous():
            value = value.contiguous()
        onnx_info["value"] = value
    return value


def _l1_streaming(a: torch.Tensor, b: torch.Tensor, best: float, block: int = 4096) -> float:
    a = _as_cpu_tensor(a).reshape(-1).float()
    b = _as_cpu_tensor(b).reshape(-1).float()
    total = 0.0
    for start in range(0, a.numel(), block):
        total += (a[start : start + block] - b[start : start + block]).abs().sum().item()
        if total > best:
            return total
    return total


def _build_onnx_dict(onnx_path: str | Path) -> Tuple[Dict[str, dict], Dict[str, list[str]], Dict[str, list[str]]]:
    """Return initializer information and the nodes consuming each initializer."""

    print(f"--> Loading ONNX model for HF mapping: {onnx_path}")
    # load_external_data=True is required when export produces an .onnx.data file.
    onnx_model = onnx.load(str(onnx_path), load_external_data=True)

    onnx_dict: Dict[str, dict] = {}
    for initializer in onnx_model.graph.initializer:
        onnx_dict[initializer.name] = {
            "init": initializer,
            "shape": tuple(initializer.dims),
            "fp": _onnx_fingerprint(initializer),
        }

    init_to_node: Dict[str, list[str]] = {}
    init_to_op_type: Dict[str, list[str]] = {}
    for node in onnx_model.graph.node:
        for input_name in node.input:
            if input_name not in onnx_dict:
                continue
            init_to_node.setdefault(input_name, []).append(node.name)
            init_to_op_type.setdefault(input_name, []).append(node.op_type)

    print(f"    {len(onnx_dict)} ONNX initializers in graph")
    return onnx_dict, init_to_node, init_to_op_type


def _match_torch_to_onnx(
    onnx_dict: Mapping[str, dict],
    torch_state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, dict]:
    """Greedily match every HF tensor to at most one ONNX initializer."""

    remaining = dict(onnx_dict)
    mapping: Dict[str, dict] = {}

    for hf_key in sorted(torch_state_dict):
        value = _as_cpu_tensor(torch_state_dict[hf_key])
        value_shape = tuple(value.shape)
        value_fp = _torch_fingerprint(value)
        value_transposed_fp = (
            _torch_fingerprint(value.transpose(0, 1))
            if len(value_shape) == 2
            else None
        )
        best_similarity = float("inf")
        best_onnx_name = None
        best_transpose = False

        for onnx_name, onnx_info in remaining.items():
            onnx_shape = tuple(onnx_info["shape"])
            transpose = False
            if onnx_shape == value_shape:
                pass
            elif len(onnx_shape) == 2 and onnx_shape[::-1] == value_shape:
                transpose = True
            else:
                continue

            fingerprint = abs(onnx_info["fp"] - value_fp)
            if len(value_shape) == 2:
                fingerprint_transposed = abs(onnx_info["fp"] - value_transposed_fp)
                if min(fingerprint, fingerprint_transposed) > FINGERPRINT_TOLERANCE:
                    continue
                if fingerprint_transposed < fingerprint and not transpose:
                    transpose = True
            elif fingerprint > FINGERPRINT_TOLERANCE:
                continue

            onnx_value = _load_onnx_tensor(onnx_info)
            if transpose:
                onnx_value = onnx_value.transpose(0, 1)
            similarity = _l1_streaming(onnx_value, value, best_similarity)
            if similarity < best_similarity:
                best_similarity = similarity
                best_onnx_name = onnx_name
                best_transpose = transpose
            if best_similarity < 0.1:
                break

        if best_onnx_name is None:
            mapping[hf_key] = {
                "onnx_tensor": None,
                "onnx_shape": None,
                "transpose": False,
                "l1": None,
                "clean": False,
            }
            continue

        remaining.pop(best_onnx_name)
        mapping[hf_key] = {
            "onnx_tensor": best_onnx_name,
            "onnx_shape": list(onnx_dict[best_onnx_name]["shape"]),
            "transpose": best_transpose,
            "l1": best_similarity,
            "clean": best_similarity <= L1_CLEAN_TOLERANCE,
        }

    return mapping


def build_hf_onnx_mapping(
    onnx_path: str | Path,
    hf_state_dict: Mapping[str, torch.Tensor],
    output_path: str | Path,
) -> dict:
    """Create the persisted HF-weight -> ONNX-op mapping used by export_rknn.

    ``hf_state_dict`` may either contain only the sub-model that was exported
    (for example the vision tower) or the full model state dict. Tensors that
    have no counterpart in the ONNX are kept in the mapping and marked as
    unmatched, which is expected for the other sub-model.
    """

    onnx_dict, init_to_node, init_to_op_type = _build_onnx_dict(onnx_path)
    matched = _match_torch_to_onnx(onnx_dict, hf_state_dict)

    mapping: Dict[str, dict] = {}
    for hf_key in sorted(hf_state_dict):
        item = matched[hf_key]
        onnx_tensor = item["onnx_tensor"]
        mapping[hf_key] = {
            **item,
            "hf_shape": list(_as_cpu_tensor(hf_state_dict[hf_key]).shape),
            "onnx_node": init_to_node.get(onnx_tensor, []) if onnx_tensor else [],
            "onnx_op_type": init_to_op_type.get(onnx_tensor, []) if onnx_tensor else [],
        }

    stats = {
        "hf_tensors": len(mapping),
        "clean_matches": sum(1 for item in mapping.values() if item["clean"] and item["onnx_tensor"]),
        "approx_matches": sum(1 for item in mapping.values() if item["onnx_tensor"] and not item["clean"]),
        "unmatched": sum(1 for item in mapping.values() if item["onnx_tensor"] is None),
    }
    result = {
        "onnx_path": str(Path(onnx_path).resolve()),
        "stats": stats,
        "mapping": mapping,
    }
    _write_json(output_path, result)
    print(f"--> HF/ONNX mapping written to {output_path}")
    print(f"    {json.dumps(stats, ensure_ascii=False)}")
    return result


def _load_lora_keys(lora_path: str | Path) -> Tuple[list[str], dict, Dict[str, dict]]:
    with safe_open(str(lora_path), framework="pt", device="cpu") as source:
        keys = sorted(source.keys())
        try:
            metadata = source.metadata() or {}
        except AttributeError:
            metadata = {}

    pairs: Dict[str, dict] = {}
    for key in keys:
        if key.endswith(LORA_A_POSTFIX):
            pairs.setdefault(key[: -len(LORA_A_POSTFIX)], {})["A"] = key
        elif key.endswith(LORA_B_POSTFIX):
            pairs.setdefault(key[: -len(LORA_B_POSTFIX)], {})["B"] = key
    return keys, metadata, pairs


def _hf_key_candidates(lora_stem: str) -> Iterable[str]:
    """Resolve common PEFT wrapper prefixes to the base HF weight key."""

    seen_stems = set()
    pending = [lora_stem]

    def emit(value: str):
        key = value + ".weight"
        if value not in seen_stems:
            seen_stems.add(value)
            yield key

    # PEFT wrappers seen in practice include both
    # ``base_model.model.model.<path>`` and ``base_model.model.<path>``.
    # Walk all removable prefixes so both forms can resolve to either a full
    # model key (``model.<path>.weight``) or a sub-model key.
    while pending:
        value = pending.pop(0)
        if value in seen_stems:
            continue
        yield from emit(value)
        for prefix in ("base_model.model.", "base_model.", "model."):
            if value.startswith(prefix):
                pending.append(value[len(prefix) :])
        # Qwen3.5-VL vision adapters trained through the ``thinker`` wrapper
        # use ``thinker.visual.*`` while the exported base model state dict
        # and ONNX mapping use ``model.visual.*``.
        if value.startswith("thinker."):
            pending.append("model." + value[len("thinker.") :])


def _resolve_hf_key(lora_stem: str, hf_mapping: Mapping[str, dict]) -> str | None:
    for candidate in _hf_key_candidates(lora_stem):
        if candidate in hf_mapping:
            return candidate
    return None


def _choose_onnx_node(mapping_item: Mapping[str, Any]) -> str | None:
    nodes = mapping_item.get("onnx_node", [])
    op_types = mapping_item.get("onnx_op_type", [])
    if isinstance(nodes, str):
        nodes = [nodes]
    if not isinstance(nodes, list):
        return None

    # Linear weights normally have one MatMul/Gemm consumer. Prefer one of
    # those if an initializer is reused by more than one operation.
    for node_name, op_type in zip(nodes, op_types if isinstance(op_types, list) else []):
        if node_name and op_type in {"MatMul", "Gemm"}:
            return node_name
    return next((node_name for node_name in nodes if node_name), None)


def build_lora_node_map(
    lora_path: str | Path,
    hf_onnx_map_path: str | Path,
    node_map_output: str | Path | None = None,
    report_output: str | Path | None = None,
    filtered_lora_output: str | Path | None = None,
    scope: str = "LLM",
) -> Tuple[dict, str, dict]:
    """Build an exact RKNN node map and filter LoRA tensors outside ``scope``.

    The Qwen3-VL adapter can contain both language and visual LoRA modules,
    while each export script loads only one sub-model ONNX (``scope`` is either
    ``"LLM"`` or ``"vision"``). The toolkit's strict ``lora_node_map``
    validation rejects unmapped keys, so tensors that belong to the other
    sub-model are copied to a derived safetensors file and omitted from the
    load.
    """

    if scope not in {"LLM", "vision"}:
        raise ValueError(f"scope must be 'LLM' or 'vision', got {scope!r}")

    with open(hf_onnx_map_path, "r", encoding="utf-8") as f:
        map_payload = json.load(f)
    hf_mapping = map_payload.get("mapping", map_payload)
    if not isinstance(hf_mapping, dict):
        raise ValueError(f"HF/ONNX mapping root must contain an object: {hf_onnx_map_path}")

    all_keys, metadata, pairs = _load_lora_keys(lora_path)
    node_map: Dict[str, list] = {}
    selected_keys = set()
    mapped_pairs = []
    unmatched_pairs = []

    for lora_stem in sorted(pairs):
        pair = pairs[lora_stem]
        if set(pair) != {"A", "B"}:
            raise ValueError(
                f"LoRA module '{lora_stem}' does not contain both "
                f"{LORA_A_POSTFIX} and {LORA_B_POSTFIX}"
            )

        hf_key = _resolve_hf_key(lora_stem, hf_mapping)
        mapping_item = hf_mapping.get(hf_key) if hf_key else None
        onnx_node = _choose_onnx_node(mapping_item or {})
        if hf_key is None or onnx_node is None:
            unmatched_pairs.append(
                {
                    "lora_stem": lora_stem,
                    "hf_key": hf_key,
                    "onnx_node": onnx_node,
                }
            )
            continue

        node_map.setdefault(onnx_node, [[]])[0].append([pair["A"], pair["B"], 0])
        selected_keys.update(pair.values())
        mapped_pairs.append(
            {
                "lora_stem": lora_stem,
                "hf_key": hf_key,
                "onnx_node": onnx_node,
                "clean": bool(mapping_item.get("clean", False)),
            }
        )

    if not node_map:
        raise RuntimeError(
            f"No LoRA pair can be mapped to the exported {scope} ONNX. "
            f"Check {hf_onnx_map_path} and the base model revision."
        )

    unmapped_keys = sorted(set(all_keys) - selected_keys)
    load_path = str(Path(lora_path).resolve())
    if unmapped_keys:
        if filtered_lora_output is None:
            raise RuntimeError(
                f"LoRA contains {len(unmapped_keys)} keys outside the {scope} ONNX; "
                "filtered_lora_output is required for strict lora_node_map loading."
            )
        filtered_path = Path(filtered_lora_output).resolve()
        source_path = Path(lora_path).resolve()
        if filtered_path == source_path:
            raise ValueError("filtered_lora_output must not overwrite the source LoRA file")
        filtered_path.parent.mkdir(parents=True, exist_ok=True)
        with safe_open(str(source_path), framework="pt", device="cpu") as source:
            tensors = {key: source.get_tensor(key).contiguous() for key in sorted(selected_keys)}
        save_file(tensors, str(filtered_path), metadata=metadata)
        load_path = str(filtered_path)
        print(
            f"--> Filtered {len(unmapped_keys)} non-{scope} LoRA tensors; "
            f"using {load_path}"
        )

    report = {
        "lora_source": str(Path(lora_path).resolve()),
        "lora_load_path": load_path,
        "hf_onnx_map": str(Path(hf_onnx_map_path).resolve()),
        "scope": scope,
        "source_tensor_count": len(all_keys),
        "source_pair_count": len(pairs),
        "mapped_pair_count": len(mapped_pairs),
        "unmatched_pair_count": len(unmatched_pairs),
        "filtered_tensor_count": len(unmapped_keys),
        "node_count": len(node_map),
        "unmatched_pairs": unmatched_pairs,
        "mapped_pairs": mapped_pairs,
    }
    if node_map_output is not None:
        _write_json(node_map_output, node_map)
        print(f"--> LoRA node map written to {node_map_output}")
    if report_output is not None:
        _write_json(report_output, report)
        print(f"--> LoRA node map report written to {report_output}")
    print(
        f"    LoRA pairs: source={len(pairs)}, mapped={len(mapped_pairs)}, "
        f"unmatched={len(unmatched_pairs)}, ONNX nodes={len(node_map)}"
    )
    return node_map, load_path, report
