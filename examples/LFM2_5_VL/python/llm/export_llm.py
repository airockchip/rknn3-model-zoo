#!/usr/bin/env python3
"""Export LiquidAI/LFM2.5-VL LLM to ONNX + config.pkl + tokenizer.gguf + embed.bin.

Supports both 1.6B and 3B model variants.

技术路线：
  1. 加载 VL 模型，提取 language_model + lm_head
  2. 导出标准四输入 LLM ONNX（input_ids, attention_mask, position_ids, num_logits_to_keep）
  3. 将 ShortConv 的 padding 从 [2,2] 改为因果卷积 [2,0]
  4. 生成 *-convstream.onnx，供 RKNN cvt_conv_streaming 使用
  5. 导出 config.pkl / tokenizer.gguf / embed.bin
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")

import sys
import argparse
import traceback
from pathlib import Path

import onnx
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent.parent.parent  # python/llm -> python -> example -> examples -> repo root
sys.path.insert(0, str(_REPO_ROOT))

from py_utils.export_llm_helper import (
    causal_llm_to_onnx,
    export_llm_config,
    export_embed_weight,
)
from py_utils.tools import clear_llm_external_weight_in_dir

from transformers import AutoModelForImageTextToText, AutoConfig, PretrainedConfig


# ---------------------------------------------------------------------------
# Compatibility patches – PyTorch 2.13 / LFM2 config key differences
# (mirrors LFM2_5/python/export_onnx.py for consistency)
# ---------------------------------------------------------------------------

# Patch 1: PyTorch ≥ 2.13 defaults torch.onnx.export to dynamo=True, which
# rejects dynamic_axes and demands dynamic_shapes instead.  Force the legacy
# (TorchScript) exporter so causal_llm_to_onnx's dynamic_axes usage keeps working.
_original_torch_onnx_export = torch.onnx.export


def _patched_onnx_export(*args, **kwargs):
    kwargs.setdefault("dynamo", False)
    return _original_torch_onnx_export(*args, **kwargs)


torch.onnx.export = _patched_onnx_export


# Note: convert_hf_to_gguf.py already handles the LFM2.5-VL config key
# mismatch (LFM2Model._add_feed_forward_length falls back from "block_ff_dim"
# to "intermediate_size"), so the converter is used as-is without patching.
def export_tokenizer(model_path, tokenizer_path):
    """Export tokenizer.gguf in-process via the fixed converter script."""
    import importlib.util

    converter_path = _REPO_ROOT / "tokenizer" / "thirdparty" / "llama_vocab" / "convert_hf_to_gguf.py"
    spec = importlib.util.spec_from_file_location("convert_hf_to_gguf", str(converter_path))
    conv = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(conv)

    # Local directory (absolute or relative) -> load from disk; anything else
    # is treated as a remote HuggingFace repo id. startswith(".") is not
    # reliable (e.g. "models/LFM2.5-VL-1.6B" is a local relative path).
    remote_flag = [] if os.path.isdir(model_path) else ["--remote"]
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


# ---------------------------------------------------------------------------

# Model variant configurations
MODEL_CONFIGS = {
    "1.6B": {
        "model_name": "LiquidAI/LFM2.5-VL-1.6B",
        "output_basename": "LFM2.5-VL-1.6B",
        "expected_conv_count": 10,
    },
    "3B": {
        "model_name": "LiquidAI/LFM2.5-VL-3B",
        "output_basename": "LFM2.5-VL-3B",
        "expected_conv_count": 22,
    },
}

PROMPT_TOKEN = "RKLLM"
CHAT_CONTEXT = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_TOKEN},
            ],
        }
    ],
    "add_generation_prompt": True,
}


def _disable_use_cache(cfg):
    """递归关闭所有子配置的 use_cache。

    与 py_utils.update_config 不同，这里用 try/except 包住 getattr，
    避免新版 transformers 访问 PretrainedConfig.rope_scaling property
    时抛 AttributeError: 'Lfm2VlConfig' object has no attribute 'rope_parameters'。
    """
    try:
        if hasattr(cfg, "use_cache"):
            setattr(cfg, "use_cache", False)
    except Exception:
        pass
    for _name in dir(cfg):
        try:
            _sub = getattr(cfg, _name)
        except Exception:
            continue
        if isinstance(_sub, PretrainedConfig):
            _disable_use_cache(_sub)


class LanguageModelWithLMHead(torch.nn.Module):
    """Wrapper that extracts language_model + lm_head from VL model for ONNX export."""

    def __init__(self, model):
        super().__init__()
        # Get the language model from VL model
        self.language_model = model.model.language_model if hasattr(model.model, 'language_model') else model.model.text_model
        if self.language_model is None:
            self.language_model = model.model
        self.config = model.config

        # Get lm_head weight (transposed for matmul)
        weight_t = model.lm_head.weight.detach().t().contiguous()
        self.lm_head_weight_t = torch.nn.Parameter(weight_t, requires_grad=False)

        bias = getattr(model.lm_head, "bias", None)
        if bias is not None:
            self.lm_head_bias = torch.nn.Parameter(bias.detach().contiguous(), requires_grad=False)
        else:
            self.lm_head_bias = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        num_logits_to_keep=None,
    ):
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

        hidden_states = outputs[0]

        # runtime 分桶/填充后可传实际索引读取真实末 token 的 logits。此前
        # 静态焊死 [:, -1:, :] 使右填充桶的末槽=pad 槽，runtime 无法补偿，是板上
        # 首 token 垃圾与多模态图像瞎编的根因。语义逐字对齐 transformers
        # modeling_lfm2.py Lfm2ForCausalLM.forward:
        #   slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        slice_indices = (
            slice(-num_logits_to_keep, None)
            if isinstance(num_logits_to_keep, int)
            else num_logits_to_keep
        )
        hidden_states = hidden_states[:, slice_indices, :]
        logits = torch.matmul(hidden_states, self.lm_head_weight_t)

        return logits


def _onnx_io(model_proto):
    inputs = [inp.name for inp in model_proto.graph.input]
    outputs = [out.name for out in model_proto.graph.output]
    return inputs, outputs


def _conv_kernel(model_proto, node, attrs):
    """返回 1D Conv 的 kernel size [k]。

    Dynamo exporter（torch>=2.4 / onnxscript 后端）可能省略 kernel_shape 属性，
    此时从权重张量（initializer 或 Constant）的最后一维推断。
    1D Conv 权重形状: [M, C/group, k]。
    """
    kernel = attrs.get("kernel_shape")
    if kernel:
        return list(kernel)
    if len(node.input) < 2:
        return None
    w_name = node.input[1]
    for init in model_proto.graph.initializer:
        if init.name == w_name and init.dims:
            return [int(list(init.dims)[-1])]
    for cn in model_proto.graph.node:
        if cn.op_type == "Constant" and cn.output and cn.output[0] == w_name:
            for a in cn.attribute:
                if a.name == "value" and a.t.dims:
                    return [int(list(a.t.dims)[-1])]
    return None


def _find_lfm2_shortconv(model_proto, hidden_size):
    """查找 LFM2 ShortConv 节点。

    判定不依赖节点名（兼容 Dynamo 导出的 conv1d* 与 legacy 导出的
    layers.* / .../conv/conv/Conv 命名），目标 Conv 必须满足：
      - kernel == [3]（kernel_shape 属性缺失时从权重形状推断）
      - dilations == [1]
      - strides == [1]
      - group == hidden_size
      - pads 为 [2,2] 或已经是 [2,0]
    """
    results = []
    for node in model_proto.graph.node:
        if node.op_type != "Conv":
            continue
        output_name = node.output[0] if node.output else ""
        node_name = node.name or output_name

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

        kernel = _conv_kernel(model_proto, node, attrs)
        pads = attrs.get("pads", [])
        dilations = attrs.get("dilations", [1])
        strides = attrs.get("strides", [1])
        group = attrs.get("group", 0)

        if (kernel == [3] and dilations == [1] and strides == [1]
                and group == hidden_size
                and pads in ([2, 2], [2, 0])):
            results.append((node_name, output_name, pads, kernel, dilations, strides, group))
    return results


def rewrite_lfm2_conv_for_streaming(src_onnx, dst_onnx, hidden_size, expected_conv_count):
    """修改标准 ONNX 中 LFM2 ShortConv 的 padding 为 [2, 0]。

    RKNN3 cvt_conv_streaming 要求：left_pad = (kernel-1)*dilation = 2, right_pad = 0。
    """
    import onnx

    src_path = Path(src_onnx).resolve()
    dst_path = Path(dst_onnx).resolve()

    if src_path.parent != dst_path.parent:
        raise RuntimeError(
            "Convstream ONNX must be saved in the same directory as the source "
            f"ONNX: src_dir={src_path.parent}, dst_dir={dst_path.parent}"
        )

    print(f"--> Loading ONNX for Conv rewrite: {src_path}")
    model_proto = onnx.load(str(src_path), load_external_data=False)

    input_names, output_names = _onnx_io(model_proto)
    print(f"  ONNX inputs: {input_names}")
    print(f"  ONNX outputs: {output_names}")

    targets = _find_lfm2_shortconv(model_proto, hidden_size)
    print(f"  Found {len(targets)} LFM2 ShortConv nodes (expected {expected_conv_count})")

    if len(targets) != expected_conv_count:
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
                print(f"  {node.output[0]}: {attrs} kernel(inferred)={_conv_kernel(model_proto, node, attrs)}")
        raise RuntimeError(
            f"Expected {expected_conv_count} LFM2 ShortConv (group={hidden_size}), "
            f"got {len(targets)}"
        )

    # Check for duplicates
    seen = set()
    for node_name, out_name, _, _, _, _, _ in targets:
        if out_name in seen:
            raise RuntimeError(f"Duplicate Conv output name: {out_name}")
        seen.add(out_name)

    # Modify pads
    print(f"\n--> Modifying Conv pads to [2, 0]")
    modified = 0
    for node in model_proto.graph.node:
        if node.op_type != "Conv":
            continue
        output_name = node.output[0] if node.output else ""
        node_name = node.name or output_name
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
        kernel = _conv_kernel(model_proto, node, attrs)
        if (kernel == [3] and attrs.get("dilations", [1]) == [1]
                and attrs.get("strides", [1]) == [1]
                and attrs.get("group", 0) == hidden_size):
            for attr in node.attribute:
                if attr.name == "pads":
                    old = list(attr.ints)
                    attr.ints[:] = [2, 0]
                    print(f"  {output_name}: pads {old} -> [2, 0]")
                    modified += 1

    if modified != expected_conv_count:
        raise RuntimeError(f"Only modified {modified} Conv pads, expected {expected_conv_count}")

    onnx.save_model(model_proto, str(dst_path))
    print(f"  Saved convstream ONNX to: {dst_path}")

    return [out_name for _, out_name, _, _, _, _, _ in targets]


def validate_external_data_files(model_proto, onnx_path):
    """Check that every EXTERNAL data initializer has a real file next to the ONNX."""
    base_dir = Path(onnx_path).parent
    external_files = {}
    for initializer in model_proto.graph.initializer:
        if initializer.data_location == onnx.TensorProto.EXTERNAL:
            for entry in initializer.external_data:
                if entry.key == "location":
                    external_files[initializer.name] = entry.value
    if not external_files:
        print("      (no external data)")
        return
    for name, location in external_files.items():
        file_path = base_dir / location
        if not file_path.exists():
            raise ValueError(f"missing external data file: {file_path}")
    print(f"      checked {len(external_files)} external data files")


def validate_convstream_onnx(onnx_path, hidden_size, expected_conv_count):
    """Validate the *-convstream.onnx so bad artifacts fail early (mirrors the
    official LFM2.5 export_onnx.py): 4 inputs / 1 output, no KV-cache or
    conv-state nodes, exactly expected_conv_count LFM2 ShortConv ops with pads [2, 0],
    external data files present, and onnx.checker passes."""
    print(f"\n--> Validating {onnx_path}")

    model_proto = onnx.load(onnx_path, load_external_data=False)
    input_names, output_names = _onnx_io(model_proto)

    # 1) Inputs
    if len(input_names) != 4:
        raise ValueError(f"expected exactly 4 inputs, got {len(input_names)}: {input_names}")
    expected_inputs = ["input_ids", "attention_mask", "position_ids", "num_logits_to_keep"]
    for expected in expected_inputs:
        if expected not in input_names:
            raise ValueError(f"missing input '{expected}'. actual inputs: {input_names}")

    # 2) Output
    if len(output_names) != 1:
        raise ValueError(f"expected exactly 1 output, got {len(output_names)}: {output_names}")
    if output_names[0] not in ("output", "logits"):
        raise ValueError(
            f"unexpected output name: '{output_names[0]}' (expected 'output' or 'logits')"
        )

    # 3) No KV-cache / conv-state leftovers
    non_supported = ["past_key_values", "present", "conv_state", "kv_cache"]
    unsupported_found = []
    for node in model_proto.graph.node:
        node_name_lower = node.name.lower()
        output_names_lower = {o.lower() for o in node.output}
        for candidate in non_supported:
            if candidate in node_name_lower or candidate in output_names_lower:
                unsupported_found.append((node.name, candidate))
    if unsupported_found:
        raise ValueError(f"unexpected KV/state nodes found: {unsupported_found}")

    # 4) Exactly expected_conv_count LFM2 ShortConv ops
    shortconv = _find_lfm2_shortconv(model_proto, hidden_size)
    if len(shortconv) != expected_conv_count:
        raise ValueError(f"expected {expected_conv_count} LFM2 ShortConv ops, got {len(shortconv)}")

    # 5) All ShortConv pads == [2, 0]
    invalid = []
    for node_name, _output_name, pads, _kernel, _dilations, _strides, _group in shortconv:
        if pads != [2, 0]:
            invalid.append((node_name, pads))
    if invalid:
        raise ValueError(f"some ShortConv pads != [2, 0]: {invalid}")

    # 6) External data files
    validate_external_data_files(model_proto, onnx_path)

    # 7) Official checker
    #    传文件路径而非 ModelProto，否则 checker 按当前工作目录解析
    #    External Data，找不到同目录下的 *.onnx.data 会误报失败。
    try:
        try:
            onnx.checker.check_model(str(onnx_path), full_check=True)
        except TypeError:
            # 兼容较旧 ONNX 版本（不支持 full_check 参数）
            onnx.checker.check_model(str(onnx_path))
    except Exception as exc:
        raise RuntimeError(f"onnx.checker.check_model failed: {exc}")

    print("    validate_convstream_onnx: OK")


def validate_artifact(path, description):
    print(f"    - Checking {description} ...")
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"missing {description} at {file_path}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"empty {description} at {file_path}")
    print(f"      {file_path} ({file_path.stat().st_size} bytes)")


def validate_embed_file(embed_path, vocab_size, hidden_size):
    print("    - Checking embed.bin ...")
    file_path = Path(embed_path)
    if not file_path.exists():
        raise ValueError(f"missing embed.bin at {file_path}")
    expected = vocab_size * hidden_size * 2  # fp16
    actual = file_path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"embed.bin size mismatch: expected {expected} (vocab={vocab_size}, "
            f"hidden={hidden_size}, fp16), got {actual}"
        )
    print(f"      {file_path} ({actual} bytes)")


def main():
    sys.setrecursionlimit(1000000)
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser(
        description="Export LiquidAI/LFM2.5-VL LLM to ONNX + config + tokenizer + embed for RKNN. "
                    "Supports 1.6B and 3B variants."
    )
    parser.add_argument(
        "--model_size", type=str, default="1.6B", choices=list(MODEL_CONFIGS.keys()),
        help="Model variant to export (default: 1.6B)",
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Hugging Face model name or local path (overrides --model_size default)",
    )
    parser.add_argument(
        "--export_llm_path", type=str, default=None,
        help="Output ONNX model path (default: ../../model/llm/<basename>-llm.onnx)",
    )
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp16", "fp32"],
        help="Torch dtype for model loading (default: fp16)",
    )
    parser.add_argument(
        "--modelscope", action="store_true",
        help="Download model from www.modelscope.cn",
    )
    parser.add_argument(
        "--hf_download", action="store_true",
        help="Download model from Hugging Face (via huggingface_hub snapshot_download)",
    )
    parser.add_argument(
        "--quant", action="store_true",
        help="Use GRQ quantization (requires CUDA GPU)",
    )
    parser.add_argument(
        "--cali_dataset", type=str,
        default=str(_SCRIPT_DIR.parent.parent.parent.parent / "datasets" / "CMMLU" / "dataset.json"),
        help="Calibration dataset for GRQ (JSON format with 'input'/'target' fields)",
    )
    args = parser.parse_args()

    # Resolve model config based on --model_size
    cfg = MODEL_CONFIGS[args.model_size]
    MODEL_NAME = cfg["model_name"]
    OUTPUT_BASENAME = cfg["output_basename"]
    EXPECTED_CONV_COUNT = cfg["expected_conv_count"]

    if args.model_path is None:
        args.model_path = MODEL_NAME
    if args.export_llm_path is None:
        args.export_llm_path = str(_SCRIPT_DIR.parent.parent / "model" / "llm" / f"{OUTPUT_BASENAME}-llm.onnx")

    print(f"--> Model variant: {args.model_size} (model={MODEL_NAME}, conv_count={EXPECTED_CONV_COUNT})")

    if args.modelscope and args.hf_download:
        print("[ERROR] --modelscope and --hf_download are mutually exclusive.")
        sys.exit(1)

    output_dir = Path(args.export_llm_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = Path(args.export_llm_path)
    convstream_onnx_path = output_dir / f"{OUTPUT_BASENAME}-convstream.onnx"
    config_path = output_dir / f"{OUTPUT_BASENAME}.config.pkl"
    tokenizer_path = output_dir / f"{OUTPUT_BASENAME}.tokenizer.gguf"
    embed_path = output_dir / f"{OUTPUT_BASENAME}.embed.bin"

    # Clean old artifacts
    for p in (onnx_path, convstream_onnx_path, config_path, tokenizer_path, embed_path):
        if p.exists():
            p.unlink()
    clear_llm_external_weight_in_dir(str(output_dir))

    # Download model if needed
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

    # ModelScope cache may place the files under snapshots/<rev>/ instead of the
    # repo root; the tokenizer must be exported from the actual file directory.
    tokenizer_model_path = model_source
    _model_root = Path(model_source)
    if not (_model_root / "config.json").exists():
        for candidate in sorted(_model_root.rglob("config.json")):
            tokenizer_model_path = str(candidate.parent)
            break

    # Determine dtype and attention implementation
    if args.quant:
        if not torch.cuda.is_available():
            print("[ERROR] --quant requires CUDA GPU for GRQ quantization")
            sys.exit(1)
        torch_dtype = torch.bfloat16
        print("--> GRQ quantization enabled (bfloat16 + FlashAttention2 recommended)")
    else:
        torch_dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # Try FlashAttention2 for GRQ (saves GPU memory)
    attn_impl = "eager"
    if args.quant:
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            print("    Use FlashAttention2 (recommended for GRQ)")
        except ImportError:
            print("    FlashAttention2 not available, using eager (will use more GPU memory)")

    # Load config + model
    print(f"--> Loading config from {model_source}")
    config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
    _disable_use_cache(config)

    # Get hidden_size / vocab_size from text config (VL models nest the LLM
    # config under text_config)
    text_config = getattr(config, 'text_config', None)
    base = text_config if text_config is not None else config
    hidden_size = getattr(base, 'hidden_size', None)
    vocab_size = getattr(base, 'vocab_size', None)

    if hidden_size is None or vocab_size is None:
        print("[ERROR] Cannot determine hidden_size / vocab_size from config.")
        sys.exit(1)

    print(f"--> Loading model (dtype={'bf16' if args.quant else args.dtype}, attn={attn_impl})")
    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    model.eval()

    # Force eager attention for ONNX export (GRQ uses flash_attn during quantization,
    # but ONNX export requires eager)
    model.config._attn_implementation = "eager"
    inner = getattr(model, "model", model)
    layers = getattr(getattr(inner, "language_model", inner), "layers", []) or getattr(inner, "layers", [])
    for layer in layers:
        sa = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
        if sa is not None and hasattr(sa, "config"):
            sa.config._attn_implementation = "eager"

    # GRQ quantization (if --quant)
    # Quantize the raw language_model (not the wrapper).
    # The wrapper's lm_head_weight_t is a plain Parameter (not a Linear layer),
    # so it should not be passed to RKQuantizer.
    if args.quant:
        print(f"\n--> Running GRQ quantization (dataset: {args.cali_dataset})")
        from rknn.quantization.api import RKQuantizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)

        language_model = model.model.language_model if hasattr(model.model, 'language_model') else model.model.text_model

        QuantTool = RKQuantizer(verbose=True)
        ret = QuantTool.load_model(
            model=language_model, tokenizer=tokenizer, device='cuda',
            system_prompt=None, tools=None,
            system_role='system', user_role='user',
        )
        if ret != 0:
            print("[ERROR] RKQuantizer.load_model failed!")
            sys.exit(ret)

        language_model = QuantTool.quantize(
            quantized_dtype="w4a16",
            quantized_method="group32",
            quantized_algorithm='grq',
            dataset=args.cali_dataset,
        )
        language_model = language_model.cpu()
        # Write quantized weights back into the original model
        if hasattr(model.model, 'language_model'):
            model.model.language_model = language_model
        else:
            model.model.text_model = language_model
        print("    GRQ quantization done")

    # Create wrapper (extracts language_model + lm_head from VL model)
    wrapped_model = LanguageModelWithLMHead(model)
    wrapped_model.eval()

    # Export standard ONNX
    print(f"\n--> Exporting standard ONNX to {onnx_path}")
    try:
        args.export_llm_path = str(onnx_path)
        args.prompt_size = 64
        args.dynamic_shape = True
        with torch.no_grad():
            causal_llm_to_onnx(wrapped_model, args)
    except Exception as e:
        print(f"[FAILED] ONNX EXPORT: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Rewrite Conv pads -> *-convstream.onnx
    print("\n" + "=" * 70)
    print("Conv Streaming: rewriting LFM2 Conv pads to [2, 0]")
    print("=" * 70)
    try:
        rewrite_lfm2_conv_for_streaming(str(onnx_path), str(convstream_onnx_path), hidden_size, EXPECTED_CONV_COUNT)
    except Exception as e:
        print(f"[FAILED] Conv rewrite: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Export config.pkl / tokenizer.gguf / embed.bin
    print(f"\n--> Exporting config to {config_path}")
    export_llm_config(model_source, str(config_path), CHAT_CONTEXT, PROMPT_TOKEN)

    print(f"--> Exporting tokenizer to {tokenizer_path}")
    export_tokenizer(tokenizer_model_path, str(tokenizer_path))

    print(f"--> Exporting embedding weight to {embed_path}")
    embed_weight_tensor = model.get_input_embeddings().weight
    export_embed_weight(embed_weight_tensor, str(embed_path))

    # Release model resources
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Export complete!")
    print("=" * 70)
    print(f"  ONNX (standard) : {onnx_path}")
    print(f"  ONNX (convstream): {convstream_onnx_path}")
    print(f"  config          : {config_path}")
    print(f"  tokenizer       : {tokenizer_path}")
    print(f"  embed           : {embed_path}")
    print("=" * 70)

    # Final validation (mirrors the official LFM2.5 export_onnx.py)
    print("\n" + "=" * 70)
    print("Final validation")
    print("=" * 70)
    try:
        validate_convstream_onnx(str(convstream_onnx_path), hidden_size, EXPECTED_CONV_COUNT)
        validate_artifact(str(config_path), "config.pkl")
        validate_artifact(str(tokenizer_path), "tokenizer.gguf")
        validate_embed_file(str(embed_path), vocab_size, hidden_size)
    except Exception as e:
        print(f"\n[FAILED] Final validation: {e}")
        traceback.print_exc()
        sys.exit(1)
    print("\nAll artifacts validated. Export complete!")


if __name__ == "__main__":
    main()
