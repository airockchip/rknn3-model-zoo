#!/usr/bin/env python3
"""Export LiquidAI/LFM2.5-1.2B-Instruct to ONNX + config.pkl + tokenizer.gguf + embed.bin.

技术路线：
  1. HuggingFace 模型导出标准四输入 LLM ONNX（input_ids, attention_mask,
     position_ids, num_logits_to_keep），单 logits 输出。
  2. 将 10 个 LFM2 ShortConv 的 padding 从 [2,2] 改为因果卷积 [2,0]。
  3. 生成 *-convstream.onnx，供 RKNN cvt_conv_streaming 使用。
  4. 导出 config.pkl / tokenizer.gguf / embed.bin。

所有默认路径基于脚本目录计算，不依赖当前工作目录。
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")

import sys
import argparse
import traceback
from pathlib import Path

import torch

# Make py_utils importable regardless of the caller's working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent  # python -> example -> examples -> repo root
sys.path.insert(0, str(_REPO_ROOT))

from py_utils.export_llm_helper import (
    causal_llm_to_onnx,
    update_config,
    export_tokenizer,
    export_llm_config,
    export_embed_weight,
)
from py_utils.tools import clear_llm_external_weight_in_dir

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


# ---------------------------------------------------------------------------
# Compatibility patches – PyTorch 2.13 / LFM2 config key differences
# ---------------------------------------------------------------------------

# Patch 1: PyTorch ≥ 2.13 defaults torch.onnx.export to dynamo=True, which
# rejects dynamic_axes and demands dynamic_shapes instead.  Force the legacy
# (TorchScript) exporter so the existing dynamic_axes usage keeps working.
_original_torch_onnx_export = torch.onnx.export


def _patched_onnx_export(*args, **kwargs):
    kwargs.setdefault("dynamo", False)
    return _original_torch_onnx_export(*args, **kwargs)


torch.onnx.export = _patched_onnx_export


# Patch 2: convert_hf_to_gguf.py looks up hparams["block_ff_dim"], but
# AutoConfig.to_dict() renames it to "intermediate_size" for LFM2.  Replace
# export_tokenizer with an in-process version that monkey-patches the
# converter's LFM2Model._add_feed_forward_length before running it.
def _export_tokenizer_lfm2(model_path, tokenizer_path):
    """Export tokenizer.gguf in-process, patching LFM2 config key mismatch."""
    import importlib.util

    converter_path = _REPO_ROOT / "tokenizer" / "thirdparty" / "llama_vocab" / "convert_hf_to_gguf.py"
    spec = importlib.util.spec_from_file_location("convert_hf_to_gguf", str(converter_path))
    conv = importlib.util.module_from_spec(spec)

    # Pre-patch before the module-level code executes (the class decorator
    # @ModelBase.register runs at import time, so we patch after import).
    spec.loader.exec_module(conv)

    original_add_ff = conv.LFM2Model._add_feed_forward_length

    def _patched_add_feed_forward_length(self):
        ff_dim = self.hparams.get("block_ff_dim", self.hparams.get("intermediate_size"))
        auto_adjust_ff_dim = self.hparams["block_auto_adjust_ff_dim"]
        ffn_dim_multiplier = self.hparams["block_ffn_dim_multiplier"]
        multiple_of = self.hparams["block_multiple_of"]

        if auto_adjust_ff_dim:
            ff_dim = int(2 * ff_dim / 3)
            if ffn_dim_multiplier is not None:
                ff_dim = int(ffn_dim_multiplier * ff_dim)
            ff_dim = multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)

        self.gguf_writer.add_feed_forward_length(ff_dim)

    conv.LFM2Model._add_feed_forward_length = _patched_add_feed_forward_length

    try:
        # Build argv and call main() in-process
        remote_flag = [] if model_path.startswith((".", "/", "~")) else ["--remote"]
        sys.argv = [
            "convert_hf_to_gguf.py",
            model_path,
            "--vocab-only",
            "--outtype", "f16",
            "--outfile", str(tokenizer_path),
            *remote_flag,
        ]
        conv.main()
        print(f"Tokenizer exported to {tokenizer_path}")
    finally:
        conv.LFM2Model._add_feed_forward_length = original_add_ff


export_tokenizer = _export_tokenizer_lfm2

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "LiquidAI/LFM2.5-1.2B-Instruct"
OUTPUT_BASENAME = "LFM2.5-1.2B-Instruct"
EXPECTED_CONV_COUNT = 10

# Standard 4-input LLM ONNX protocol (managed by RKNN runtime).
EXPECTED_INPUT_NAMES = {"input_ids", "attention_mask", "position_ids", "num_logits_to_keep"}

# Split token used by the RKNN LLM runtime to separate system / user / assistant.
PROMPT_TOKEN = "RKLLM"
CHAT_CONTEXT = {
    "messages": [{"role": "user", "content": PROMPT_TOKEN}],
    "add_generation_prompt": True,
}


# ---------------------------------------------------------------------------
# Helpers – structure inspection (only under --inspect)
# ---------------------------------------------------------------------------

def _safe_getattr(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def inspect_model_structure(model, config, tokenizer):
    """Print structural information for debugging (only under --inspect)."""
    print("=" * 70)
    print("LFM2.5 Model Structure Inspection (--inspect)")
    print("=" * 70)

    print(f"model class          : {type(model).__name__}")
    print(f"config class         : {type(config).__name__}")
    print(f"config.model_type    : {getattr(config, 'model_type', 'N/A')}")
    print(f"hidden_size          : {getattr(config, 'hidden_size', 'N/A')}")
    print(f"vocab_size           : {getattr(config, 'vocab_size', 'N/A')}")
    print(f"num_hidden_layers   : {getattr(config, 'num_hidden_layers', 'N/A')}")
    print(f"max_position_embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}")
    print(f"tie_word_embeddings  : {getattr(config, 'tie_word_embeddings', 'N/A')}")

    # per-layer block types
    print("-" * 40)
    layers = _safe_getattr(_safe_getattr(model, "model"), "layers")
    if layers is None:
        layers = _safe_getattr(model, "layers")
    attention_layers = []
    conv_layers = []
    if layers is not None:
        print(f"total layers         : {len(layers)}")
        for i, layer in enumerate(layers):
            layer_type = type(layer).__name__
            has_self_attn = hasattr(layer, "self_attn") or hasattr(layer, "attention")
            has_conv = any(
                "conv" in type(sub_mod).__name__.lower() or "liv" in type(sub_mod).__name__.lower()
                for _, sub_mod in layer.named_modules()
            )
            tag = ""
            if has_self_attn:
                attention_layers.append(i)
                tag += "[Attn]"
            if has_conv:
                conv_layers.append(i)
                tag += "[Conv/LIV]"
            print(f"  layer {i:3d}: {layer_type:40s} {tag}")
    else:
        print("layers: <not found>")

    print(f"Attention layer count: {len(attention_layers)}")
    print(f"Conv/LIV layer count : {len(conv_layers)}")
    if attention_layers:
        print(f"  Attention layer indices: {attention_layers}")
    if conv_layers:
        print(f"  Conv/LIV layer indices : {conv_layers}")

    # dummy forward to inspect output type / cache
    print("-" * 40)
    print("Running a dummy forward to inspect output type ...")
    device = next(model.parameters()).device
    dummy_ids = torch.zeros((1, 4), dtype=torch.long, device=device)
    dummy_mask = torch.ones((1, 4), dtype=torch.float, device=device)
    dummy_pos = torch.arange(0, 4, dtype=torch.long, device=device).unsqueeze(0)
    fwd = model.forward
    while hasattr(fwd, "__wrapped__"):
        fwd = fwd.__wrapped__
    fwd_args = list(getattr(fwd, "__code__", None).co_varnames[:getattr(fwd, "__code__", None).co_argcount]) if hasattr(fwd, "__code__") else []
    fwd_kwargs = dict(input_ids=dummy_ids, attention_mask=dummy_mask, position_ids=dummy_pos)
    logit_keep_key = None
    for key in ("num_logits_to_keep", "logits_to_keep"):
        if key in fwd_args:
            logit_keep_key = key
            break
    if logit_keep_key:
        fwd_kwargs[logit_keep_key] = torch.tensor([3], dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            out = model(**fwd_kwargs)
        print(f"model output type    : {type(out).__name__}")
        if hasattr(out, "logits"):
            print(f"output.logits shape  : {tuple(out.logits.shape)}")
        elif isinstance(out, torch.Tensor):
            print(f"output tensor shape  : {tuple(out.shape)}")
        pkv = _safe_getattr(out, "past_key_values")
        if pkv is not None:
            print(f"past_key_values type : {type(pkv).__name__}")
            for field in ("conv_states", "conv_state", "past_conv"):
                val = _safe_getattr(pkv, field)
                if val is not None:
                    if isinstance(val, (list, tuple)):
                        print(f"  past_key_values.{field}: list len={len(val)}")
                    else:
                        print(f"  past_key_values.{field}: {type(val).__name__}")
    except Exception:
        print("dummy forward failed (structure inspection continues):")
        traceback.print_exc()

    print("=" * 70)


# ---------------------------------------------------------------------------
# Helpers – post-export ONNX validation
# ---------------------------------------------------------------------------

def _tensor_type_to_np(elem_type):
    mapping = {
        1: "FLOAT", 2: "UINT8", 3: "INT8", 4: "UINT16", 5: "INT16",
        6: "INT32", 7: "INT64", 9: "BOOL", 10: "FLOAT16", 11: "DOUBLE",
        12: "UINT32", 13: "UINT64", 16: "BFLOAT16",
    }
    return mapping.get(elem_type, f"UNKNOWN({elem_type})")


def _onnx_io(model_proto):
    """Return (input_names, output_names) from an onnx ModelProto."""
    inputs = [inp.name for inp in model_proto.graph.input]
    outputs = [out.name for out in model_proto.graph.output]
    return inputs, outputs


def validate_external_data_files(model_proto, onnx_path: Path):
    """校验 ONNX External Data 文件存在、非空、范围合法。

    Convstream ONNX 复用源 ONNX 的 External Data 文件，因此基准目录
    必须是 onnx_path 所在目录。

    返回错误列表（空列表表示全部通过），不在函数内 sys.exit()。
    """
    import onnx

    base_dir = Path(onnx_path).resolve().parent
    errors = []
    external_tensor_count = 0
    external_files = set()

    for tensor in model_proto.graph.initializer:
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue

        external_tensor_count += 1

        info = {item.key: item.value for item in tensor.external_data}

        location = info.get("location", "")
        if not location:
            errors.append(
                f"Tensor {tensor.name} uses external data but has no location"
            )
            continue

        data_path = (base_dir / location).resolve()

        if not data_path.is_file():
            errors.append(
                f"External data for tensor {tensor.name} is not a regular file: {data_path}"
            )
            continue

        external_files.add(data_path)

        file_size = data_path.stat().st_size
        if file_size <= 0:
            errors.append(
                f"External data file is empty for tensor {tensor.name}: {data_path}"
            )
            continue

        try:
            offset = int(info.get("offset", "0"))
            length = int(info.get("length", "0"))
        except ValueError:
            errors.append(
                f"Invalid external-data offset/length for tensor {tensor.name}: {info}"
            )
            continue

        if offset < 0 or length < 0:
            errors.append(
                f"Negative external-data range for {tensor.name}: "
                f"offset={offset}, length={length}"
            )
            continue

        if length > 0 and offset + length > file_size:
            errors.append(
                f"External-data range exceeds file size for {tensor.name}: "
                f"offset={offset}, length={length}, file_size={file_size}"
            )

    print(f"  External tensors: {external_tensor_count}")
    print(f"  External files  : {len(external_files)}")

    return errors


def validate_convstream_onnx(onnx_path: Path, hidden_size: int):
    """对最终交付的 *-convstream.onnx 执行完整校验（Section 2）。

    任意检查失败抛出 RuntimeError，由调用方决定退出。
    """
    import onnx

    errors = []

    # 1. 文件存在且大小非零
    if not onnx_path.exists():
        raise RuntimeError(f"convstream ONNX not found: {onnx_path}")
    if onnx_path.stat().st_size == 0:
        raise RuntimeError(f"convstream ONNX is empty (0 bytes): {onnx_path}")

    # 2. 加载（不加载外部权重，仅用于图结构检查）
    try:
        model_proto = onnx.load(str(onnx_path), load_external_data=False)
    except Exception as e:
        raise RuntimeError(f"Cannot load convstream ONNX [{onnx_path}]: {e}")

    input_names, output_names = _onnx_io(model_proto)

    # 3. 输入数量 == 4
    if len(input_names) != 4:
        errors.append(f"Expected 4 inputs, got {len(input_names)}: {input_names}")

    # 4. 输出数量 == 1
    if len(output_names) != 1:
        errors.append(f"Expected 1 output, got {len(output_names)}: {output_names}")

    # 5. 必需输入存在
    missing = EXPECTED_INPUT_NAMES - set(input_names)
    if missing:
        errors.append(f"Missing required inputs: {sorted(missing)}")

    # 6. 不能有显式 Attention KV Cache 输入/输出
    for name in input_names + output_names:
        low = name.lower()
        if any(k in low for k in ("past_key", "past_value", "present_key", "present_value")):
            errors.append(f"Explicit KV Cache tensor found in graph: {name}")

    # 7. 不能有显式 Conv State 输入/输出
    for name in input_names + output_names:
        if "conv_state" in name.lower():
            errors.append(f"Explicit Conv State tensor found in graph: {name}")

    # 8. 输出应为 logits（名为 output 或 logits）
    if output_names and output_names[0] not in ("output", "logits"):
        errors.append(f"Output is not logits: name={output_names[0]} (expected 'output' or 'logits')")

    # 9. 恰好 10 个 ShortConv，且 pads == [2,0]
    conv_results = _find_lfm2_shortconv(model_proto, hidden_size)
    conv_count = len(conv_results)
    if conv_count != EXPECTED_CONV_COUNT:
        errors.append(f"Expected {EXPECTED_CONV_COUNT} ShortConv with pads=[2,0], found {conv_count}")
    else:
        for node_name, out_name, pads, _, _, _, _ in conv_results:
            if pads != [2, 0]:
                errors.append(f"Conv {out_name} pads={pads}, expected [2,0]")

    # 打印校验详情
    print("\n" + "=" * 70)
    print(f"Convstream ONNX Validation: {onnx_path}")
    print("=" * 70)
    print(f"  inputs ({len(input_names)}): {input_names}")
    for inp in model_proto.graph.input:
        shape = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
        dtype_str = _tensor_type_to_np(inp.type.tensor_type.elem_type)
        print(f"    {inp.name}: shape={shape} dtype={dtype_str}")
    print(f"  outputs ({len(output_names)}): {output_names}")
    for out in model_proto.graph.output:
        shape = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
        dtype_str = _tensor_type_to_np(out.type.tensor_type.elem_type)
        print(f"    {out.name}: shape={shape} dtype={dtype_str}")
    print(f"  ShortConv count: {conv_count} (expected {EXPECTED_CONV_COUNT})")
    for i, (node_name, out_name, pads, kernel, dilations, strides, group) in enumerate(conv_results):
        print(f"    [{i}] node={node_name}")
        print(f"        output={out_name} pads={pads} kernel={kernel} "
              f"dilation={dilations} strides={strides} group={group}")
    print(f"  ONNX file size: {onnx_path.stat().st_size:,} bytes "
          f"({onnx_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # 10. External Data 文件和范围检查
    errors.extend(validate_external_data_files(model_proto, onnx_path))

    # 11. Path-based ONNX Checker（传文件路径，不是 ModelProto）
    #    Checker 需要文件路径才能正确解析 External Data 的相对目录。
    try:
        try:
            onnx.checker.check_model(str(onnx_path), full_check=True)
        except TypeError:
            # 兼容较旧 ONNX 版本（不支持 full_check 参数）
            onnx.checker.check_model(str(onnx_path))
        print("  [OK] onnx.checker.check_model passed")
    except Exception as exc:
        errors.append(f"onnx.checker.check_model failed: {exc}")

    if errors:
        print("\n  [FAILED] Validation errors:")
        for e in errors:
            print(f"    - {e}")
        print("=" * 70)
        raise RuntimeError(f"convstream ONNX validation failed ({len(errors)} errors): {onnx_path}")

    print("\n  [OK] All convstream ONNX checks passed.")
    print("=" * 70)


def validate_embed_file(embed_path: Path, vocab_size: int, hidden_size: int):
    """校验 embedding .bin 文件大小 == vocab_size * hidden_size * 2。"""
    if not embed_path.exists():
        raise RuntimeError(f"Embedding file not found: {embed_path}")
    actual = embed_path.stat().st_size
    expected = vocab_size * hidden_size * 2  # float16 = 2 bytes
    print(f"\n--- Embedding file check ---")
    print(f"  path   : {embed_path}")
    print(f"  actual : {actual:,} bytes")
    print(f"  expected: {expected:,} bytes (vocab={vocab_size} x hidden={hidden_size} x 2)")
    if actual != expected:
        raise RuntimeError(
            f"Embedding file size mismatch: actual={actual} expected={expected} "
            f"(vocab_size={vocab_size} hidden_size={hidden_size})"
        )
    print("  [OK] Embedding file size matches.")


def validate_artifact(path: Path, label: str):
    """校验产物存在且非空。"""
    if not path.exists():
        raise RuntimeError(f"{label} not found: {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"{label} is empty (0 bytes): {path}")


# ---------------------------------------------------------------------------
# Conv Streaming: rewrite LFM2 Conv padding for cvt_conv_streaming
# ---------------------------------------------------------------------------

def _find_lfm2_shortconv(model_proto, hidden_size: int):
    """查找 LFM2 ShortConv 节点。

    目标 Conv 必须同时满足：
      - 属于 LFM2 ShortConv 节点（节点名/output 名含 layers. 或 /conv/conv/Conv）
      - kernel_shape == [3]
      - dilations == [1]
      - strides == [1]
      - group == hidden_size
      - 原始 pads 为 [2,2] 或已经是 [2,0]

    返回 [(node_name, output_name, pads, kernel, dilations, strides, group), ...]
    """
    results = []
    for node in model_proto.graph.node:
        if node.op_type != "Conv":
            continue
        output_name = node.output[0] if node.output else ""
        node_name = node.name or output_name

        is_lfm2 = ("layers." in output_name) or ("/conv/conv/Conv" in output_name) \
            or ("layers." in node_name) or ("/conv/conv/Conv" in node_name)
        if not is_lfm2:
            continue

        attrs = {}
        for attr in node.attribute:
            if attr.name == "pads":
                attrs["pads"] = list(attr.ints)
            elif attr.name == "kernel_shape":
                attrs["kernel_shape"] = list(attr.ints)
            elif attr.name == "dilations":
                attrs["dilations"] = list(attr.ints)
            elif attr.name == "strides":
                attrs["strides"] = list(attr.ints)
            elif attr.name == "group":
                attrs["group"] = attr.i

        kernel = attrs.get("kernel_shape", [])
        pads = attrs.get("pads", [])
        dilations = attrs.get("dilations", [])
        strides = attrs.get("strides", [])
        group = attrs.get("group", 0)

        if (kernel == [3] and dilations == [1] and strides == [1]
                and group == hidden_size
                and pads in ([2, 2], [2, 0])):
            results.append((node_name, output_name, pads, kernel, dilations, strides, group))
    return results


def rewrite_lfm2_conv_for_streaming(src_onnx: str, dst_onnx: str, hidden_size: int):
    """修改标准 ONNX 中 LFM2 ShortConv 的 padding 为 [2, 0]。

    RKNN3 cvt_conv_streaming 要求：left_pad = (kernel-1)*dilation = 2, right_pad = 0。
    原始 Conv pads=[2,2]（左右各 pad 2）→ 修改后 pads=[2,0]（只左 pad 2）。

    校验：恰好 10 个目标 Conv，节点名无重复，修改后全部 pads==[2,0]。
    不加载外部权重，只修改图结构并保存。

    Convstream ONNX 复用源 ONNX 的 External Data 文件，因此必须保存在
    与源 ONNX 相同的目录下。
    """
    import onnx

    src_path = Path(src_onnx).resolve()
    dst_path = Path(dst_onnx).resolve()

    if src_path.parent != dst_path.parent:
        raise RuntimeError(
            "Convstream ONNX must be saved in the same directory as the source "
            f"ONNX because it reuses the source external-data files: "
            f"src_dir={src_path.parent}, dst_dir={dst_path.parent}"
        )

    print(f"--> Loading ONNX for Conv rewrite: {src_path}")
    model_proto = onnx.load(str(src_path), load_external_data=False)

    input_names, output_names = _onnx_io(model_proto)
    print(f"  ONNX inputs: {input_names}")
    print(f"  ONNX outputs: {output_names}")

    # 确保没有显式 conv_state 输入/输出
    for name in input_names:
        if "conv_state" in name.lower():
            raise RuntimeError(f"ONNX has explicit conv_state input: {name}")
    for name in output_names:
        if "conv_state" in name.lower():
            raise RuntimeError(f"ONNX has explicit conv_state output: {name}")

    targets = _find_lfm2_shortconv(model_proto, hidden_size)
    print(f"  Found {len(targets)} LFM2 ShortConv nodes (expected {EXPECTED_CONV_COUNT})")

    if len(targets) != EXPECTED_CONV_COUNT:
        print("[ERROR] All Conv nodes for debugging:")
        for node in model_proto.graph.node:
            if node.op_type == "Conv":
                attrs = {}
                for attr in node.attribute:
                    if attr.name == "pads":
                        attrs["pads"] = list(attr.ints)
                    elif attr.name == "kernel_shape":
                        attrs["kernel_shape"] = list(attr.ints)
                    elif attr.name == "group":
                        attrs["group"] = attr.i
                print(f"  {node.output[0]}: {attrs}")
        raise RuntimeError(
            f"Expected {EXPECTED_CONV_COUNT} LFM2 ShortConv (group={hidden_size}), "
            f"got {len(targets)}"
        )

    # 节点名不能重复
    seen = set()
    for node_name, out_name, _, _, _, _, _ in targets:
        if out_name in seen:
            raise RuntimeError(f"Duplicate Conv output name: {out_name}")
        seen.add(out_name)

    # 修改 pads
    print(f"\n--> Modifying Conv pads to [2, 0]")
    modified = 0
    for node in model_proto.graph.node:
        if node.op_type != "Conv":
            continue
        output_name = node.output[0] if node.output else ""
        node_name = node.name or output_name
        is_lfm2 = ("layers." in output_name) or ("/conv/conv/Conv" in output_name) \
            or ("layers." in node_name) or ("/conv/conv/Conv" in node_name)
        if not is_lfm2:
            continue
        attrs = {}
        for attr in node.attribute:
            if attr.name == "kernel_shape":
                attrs["kernel_shape"] = list(attr.ints)
            elif attr.name == "dilations":
                attrs["dilations"] = list(attr.ints)
            elif attr.name == "strides":
                attrs["strides"] = list(attr.ints)
            elif attr.name == "group":
                attrs["group"] = attr.i
            elif attr.name == "pads":
                attrs["pads"] = list(attr.ints)
        if (attrs.get("kernel_shape") == [3] and attrs.get("dilations") == [1]
                and attrs.get("strides") == [1] and attrs.get("group") == hidden_size):
            for attr in node.attribute:
                if attr.name == "pads":
                    old = list(attr.ints)
                    attr.ints[:] = [2, 0]
                    print(f"  {output_name}: pads {old} -> [2, 0]")
                    modified += 1

    if modified != EXPECTED_CONV_COUNT:
        raise RuntimeError(f"Only modified {modified} Conv pads, expected {EXPECTED_CONV_COUNT}")

    # 保存（不保存外部数据，只保存图结构，复用源 ONNX 的 External Data 文件）
    onnx.save_model(model_proto, str(dst_path))
    print(f"  Saved convstream ONNX to: {dst_path}")

    # 打印最终 10 个 Conv 节点名和输出 tensor 名
    print(f"\n--> Conv streaming output names ({len(targets)}):")
    for i, (node_name, out_name, _, _, _, _, _) in enumerate(targets):
        print(f"  [{i}] node={node_name}")
        print(f"      output={out_name}")

    return [out_name for _, out_name, _, _, _, _, _ in targets]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Export LiquidAI/LFM2.5-1.2B-Instruct to ONNX + config + tokenizer + embed for RKNN."
    )
    parser.add_argument(
        "--model_path", type=str, default=MODEL_NAME,
        help="Hugging Face model name or local path (default: LiquidAI/LFM2.5-1.2B-Instruct)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(_SCRIPT_DIR.parent / "model"),
        help="Output directory for model artifacts (default: ../model)",
    )
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp16", "fp32"],
        help="Torch dtype for model loading (default: fp16)",
    )
    parser.add_argument(
        "--modelscope", action="store_true",
        help="Download model from www.modelscope.cn instead of Hugging Face.",
    )
    parser.add_argument(
        "--hf_download", action="store_true",
        help="Download model from Hugging Face (via huggingface_hub snapshot_download).",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Run model structure inspection (dummy forward, layer types, cache probe). Off by default.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 参数校验
    # ------------------------------------------------------------------
    if not args.model_path or not args.model_path.strip():
        print("[ERROR] --model_path must not be empty.")
        sys.exit(1)
    if args.modelscope and args.hf_download:
        print("[ERROR] --modelscope and --hf_download are mutually exclusive.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 产物路径
    onnx_path = output_dir / f"{OUTPUT_BASENAME}.onnx"
    convstream_onnx_path = output_dir / f"{OUTPUT_BASENAME}-convstream.onnx"
    config_path = output_dir / f"{OUTPUT_BASENAME}.config.pkl"
    tokenizer_path = output_dir / f"{OUTPUT_BASENAME}.tokenizer.gguf"
    embed_path = output_dir / f"{OUTPUT_BASENAME}.embed.bin"

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # 清理旧产物，避免失败后误用
    for p in (onnx_path, convstream_onnx_path, config_path, tokenizer_path, embed_path):
        if p.exists():
            p.unlink()
    # 清理 ONNX 外部权重（.onnx.data 等）
    clear_llm_external_weight_in_dir(str(output_dir))

    # ------------------------------------------------------------------
    # 0. Optionally download model
    # ------------------------------------------------------------------
    model_source = args.model_path
    if args.modelscope:
        print(f"--> Downloading model from ModelScope: {args.model_path}")
        from modelscope import snapshot_download
        model_source = snapshot_download(args.model_path)
        print(f"    downloaded to: {model_source}")
    elif args.hf_download:
        print(f"--> Downloading model from Hugging Face: {args.model_path}")
        from huggingface_hub import snapshot_download
        model_source = snapshot_download(repo_id=args.model_path)
        print(f"    downloaded to: {model_source}")

    # Resolve the actual model directory (ModelScope cache may put files
    # under snapshots/<rev>/ rather than the root).  The GGUF tokenizer
    # converter needs a directory that directly contains config.json.
    tokenizer_model_path = model_source
    _model_root = Path(model_source)
    if not (_model_root / "config.json").exists():
        for candidate in sorted(_model_root.rglob("config.json")):
            tokenizer_model_path = str(candidate.parent)
            break

    # ------------------------------------------------------------------
    # 1. Load config + model
    # ------------------------------------------------------------------
    print(f"--> Loading config from {model_source}")
    config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
    update_config(config, ["use_cache"], False)

    hidden_size = getattr(config, "hidden_size", None)
    vocab_size = getattr(config, "vocab_size", None)
    if hidden_size is None or vocab_size is None:
        print("[ERROR] Cannot determine hidden_size / vocab_size from config.")
        sys.exit(1)

    print(f"--> Loading model (dtype={args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    model.eval()

    # Force eager attention for ONNX export (flash/sdpa cannot be traced).
    model.config._attn_implementation = "eager"
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", [])
    for layer in layers:
        sa = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if sa is not None and hasattr(sa, "config"):
            sa.config._attn_implementation = "eager"

    # ------------------------------------------------------------------
    # 2. Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)

    # ------------------------------------------------------------------
    # 3. Optional: inspect model structure
    # ------------------------------------------------------------------
    if args.inspect:
        inspect_model_structure(model, config, tokenizer)
    else:
        print(f"\n--> Model: {type(model).__name__} "
              f"(hidden_size={hidden_size}, vocab_size={vocab_size}, "
              f"layers={getattr(config, 'num_hidden_layers', 'N/A')})")

    # ------------------------------------------------------------------
    # 4. Export standard ONNX (always 4 inputs, 1 output)
    # ------------------------------------------------------------------
    print(f"\n--> Exporting standard ONNX to {onnx_path}")
    try:
        args.export_llm_path = str(onnx_path)
        args.prompt_size = 64
        args.dynamic_shape = True
        with torch.no_grad():
            causal_llm_to_onnx(model, args)
    except Exception as e:
        print("\n" + "=" * 70)
        print("[FAILED] ONNX EXPORT")
        print("=" * 70)
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Rewrite Conv pads -> *-convstream.onnx 
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Conv Streaming: rewriting LFM2 Conv pads to [2, 0]")
    print("=" * 70)
    try:
        rewrite_lfm2_conv_for_streaming(str(onnx_path), str(convstream_onnx_path), hidden_size)
    except Exception as e:
        print(f"\n[FAILED] Conv rewrite: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 6. Export config.pkl / tokenizer.gguf / embed.bin
    # ------------------------------------------------------------------
    print(f"\n--> Exporting config to {config_path}")
    export_llm_config(model_source, str(config_path), CHAT_CONTEXT, PROMPT_TOKEN)

    print(f"--> Exporting tokenizer to {tokenizer_path}")
    export_tokenizer(tokenizer_model_path, str(tokenizer_path))

    print(f"--> Exporting embedding weight to {embed_path}")
    embed_weight_tensor = model.get_input_embeddings().weight
    export_embed_weight(embed_weight_tensor, str(embed_path))

    # 释放模型资源（显存等），后续仅做文件校验
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 7. 最终交付校验（以 convstream.onnx 为准）
    # ------------------------------------------------------------------
    try:
        validate_convstream_onnx(convstream_onnx_path, hidden_size)
        validate_artifact(config_path, "config.pkl")
        validate_artifact(tokenizer_path, "tokenizer.gguf")
        validate_embed_file(embed_path, vocab_size, hidden_size)
    except Exception as e:
        print(f"\n[FAILED] Final validation: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------
    # 8. 成功
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Export complete! All checks passed.")
    print("=" * 70)
    print(f"  ONNX (standard) : {onnx_path}")
    print(f"  ONNX (convstream): {convstream_onnx_path}")
    print(f"  config          : {config_path}")
    print(f"  tokenizer       : {tokenizer_path}")
    print(f"  embed           : {embed_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
