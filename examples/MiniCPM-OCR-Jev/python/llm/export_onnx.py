#!/usr/bin/env python3
"""把 Qwen3.5-2B 序列分类（value head）检查点导出成 RKNN3 需要的 ONNX 产物

用法::

    # 从 HuggingFace 下载（默认）
    python export_onnx.py --output_dir ../../model/llm --name value_llm

    # 从 ModelScope 下载
    python export_onnx.py --modelscope --output_dir ../../model/llm --name value_llm
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 路径解析
# --------------------------------------------------------------------------- #

#: 本示例在 zoo 里的层级：examples/<name>/python/llm/ -> parents[4] 是 zoo 根。
ZOO_ROOT = Path(__file__).resolve().parents[4]

#: 保证与线上验证过的产物使用同一份算子实现）。
DEFAULT_EXPORT_MODELING = Path(__file__).resolve().parent / "modeling_qwen3_5_export.py"

#: 默认模型来源：HuggingFace repo id。换模型时用 --model_path 覆盖。
DEFAULT_MODEL_PATH = "openbmb/MiniCPM-OCR-Jev"

#: 导出时用来做 shape 追踪的提示词片段（和官方示例保持一致）。
PROMPT = "RKLLM"
CHAT_CONTEXT = {"role": "user", "content": PROMPT}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export a Qwen3.5-2B sequence-classification (value head) checkpoint to RKNN3 ONNX artifacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="模型仓库 id（如 openbmb/MiniCPM-OCR-Jev），会从 HuggingFace 或 ModelScope 下载",
    )
    p.add_argument(
        "--modelscope",
        action="store_true",
        help="改用 www.modelscope.cn 下载模型（默认走 HuggingFace）",
    )
    p.add_argument("--output_dir", default="../../model/llm", help="产物输出目录")
    p.add_argument("--name", default="value_llm", help="产物文件名前缀")

    p.add_argument("--zoo_root", default=str(ZOO_ROOT), help="rknn3-model-zoo 仓库根目录（需要其中的 py_utils/）")
    p.add_argument(
        "--export_modeling",
        default=str(DEFAULT_EXPORT_MODELING),
        help="导出用 modeling_qwen3_5_export.py 的路径；默认取本示例 python/llm/ 下的同名文件",
    )

    p.add_argument("--head_key", default="score.weight", help="safetensors 中分类头权重的 key")
    p.add_argument("--num_labels", type=int, default=2, help="分类头输出维度（二分类 = 2）")
    p.add_argument(
        "--seq_len",
        type=int,
        default=64,
        help="ONNX 追踪时使用的序列长度（只是 dummy，动态轴由 dynamic_axes 决定）",
    )
    p.add_argument(
        "--skip_embed_bin",
        action="store_true",
        help="不导出 .embed.bin（仅用于调试；正常运行时应导出）",
    )
    p.add_argument(
        "--task_type",
        type=int,
        default=1,
        help="写进 config.pkl / hf_config_json 的 task_type；1 = 向量/分类类任务（对齐 WeMM-Embedding 样例）。传 -1 表示不写",
    )
    p.add_argument(
        "--hf_endpoint",
        default="https://hf-mirror.com/",
        help="HF_ENDPOINT（只在走 HuggingFace 下载时生效，--modelscope 时忽略）",
    )
    return p


# --------------------------------------------------------------------------- #
# 环境准备
# --------------------------------------------------------------------------- #


def resolve_model_source(model_path: str | Path, use_modelscope: bool = False, allow_patterns=None) -> Path:
    """把 ``--model_path``（模型仓库 id）下载到本地缓存，返回**本地目录**。

    * ``use_modelscope=False``（默认）：``huggingface_hub.snapshot_download``，
      遵循 ``HF_ENDPOINT``（默认 ``https://hf-mirror.com/``）；
    * ``use_modelscope=True``：``modelscope.snapshot_download``。

    两者都只会下载导出真正需要的文件类型，避免拉一堆 ``*.md`` / ``*.pth``。
    重复调用会直接复用本地缓存，不会重复下载。
    """
    repo_id = str(model_path).strip()
    patterns = allow_patterns or [
        "*.json",
        "*.safetensors",
        "*.jinja",
        "*.txt",
        "*.model",
    ]

    if use_modelscope:
        try:
            from modelscope import snapshot_download as ms_snapshot_download
        except ImportError as exc:  # pragma: no cover - 仅在缺依赖时触发
            raise ImportError(
                "使用 --modelscope 需要先安装 modelscope：pip install modelscope"
            ) from exc
        print(f"--> 从 ModelScope 下载/复用: {repo_id}")
        local_dir = ms_snapshot_download(
            repo_id,
            allow_patterns=patterns,
            ignore_patterns=["*.pth", "*.bin", "*.gguf", "*.h5", "*.msgpack"],
        )
    else:
        from huggingface_hub import snapshot_download

        print(
            f"--> 从 HuggingFace 下载/复用: {repo_id} "
            f"(endpoint={os.environ.get('HF_ENDPOINT', 'default')})"
        )
        local_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=patterns,
            # 只用本地缓存，不额外拉 .bin 权重
            ignore_patterns=["*.pth", "*.bin", "*.gguf", "*.h5", "*.msgpack"],
        )

    local = Path(local_dir).expanduser().resolve()
    if not local.is_dir():
        raise NotADirectoryError(f"下载后的模型目录不存在: {local}")
    print(f"    本地目录: {local}")
    return local


def resolve_export_modeling(explicit: str | None) -> Path:
    """返回导出用 modeling 的路径。

    用远端 repo id 时，优先用本示例自带的
    ``modeling_qwen3_5_export.py``（已经过本仓库验证），
    不要用 ``AutoModel`` 那种依赖 checkpoint 里 ``auto_map`` 的远程代码 ——
    远端 repo 通常只提供推理用的 modeling，无法 ONNX 导出。
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到导出 modeling: {path}")
        return path
    if not DEFAULT_EXPORT_MODELING.is_file():
        raise FileNotFoundError(f"找不到导出 modeling: {DEFAULT_EXPORT_MODELING}")
    return DEFAULT_EXPORT_MODELING


def setup_env(hf_endpoint: str) -> None:
    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

    # 避免 flash attention 的 packed-sequence 检测在导出时报错。
    # 官方 export_llm.py 里同样的 monkeypatch。
    try:
        import transformers.modeling_flash_attention_utils as fa_utils

        fa_utils._is_packed_sequence = lambda position_ids, batch_size: False
    except Exception:
        pass


def ensure_zoo_on_path(zoo_root: str | Path | None = None) -> Path:
    """把 zoo 根目录插到 ``sys.path``，使 ``import py_utils`` 可用。"""
    root = Path(zoo_root or ZOO_ROOT).expanduser().resolve()
    if not (root / "py_utils" / "export_llm_helper.py").is_file():
        raise FileNotFoundError(
            f"找不到 {root}/py_utils/export_llm_helper.py；请用 --zoo_root 指定 rknn3-model-zoo 根目录"
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_export_modeling_module(export_modeling_path: Path):
    """把导出版 modeling 注入为 ``transformers.models.qwen3_5.modeling_qwen3_5``。

    必须在任何 ``from_pretrained`` 之前完成，否则 transformers 会复用原生
    （带 fla triton kernel 的）实现，在 CPU 上无法导出 ONNX。
    """
    module_name = "transformers.models.qwen3_5.modeling_qwen3_5"
    spec = importlib.util.spec_from_file_location(module_name, export_modeling_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载导出 modeling: {export_modeling_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# 模型组装
# --------------------------------------------------------------------------- #


def read_head_weight(model_path: Path, head_key: str) -> torch.Tensor:
    """从 safetensors 里直接读出分类头权重。

    不能用 ``from_pretrained`` 一次性加载整个
    ``Qwen3_5ForSequenceClassification`` 再取 ``.score``：那会把 4.1 GB 的
    视觉塔也加载进来，而视觉塔在这里完全用不上（我们单独导出）。
    """
    from safetensors import safe_open

    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"{model_path} 下没有 .safetensors 文件")

    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if head_key in keys:
                return f.get_tensor(head_key)
    raise KeyError(f"在 {[s.name for s in shards]} 中找不到分类头权重 {head_key!r}")


def pack_dynamic_cache(conv_state: torch.Tensor, model_config, modeling_module):
    """把扁平化的 conv_state 张量还原成 ``Qwen3_5DynamicCache``。

    24 层里有 18 层是 ``linear_attention``（GatedDeltaNet），每层一个卷积状态；
    导出的 ONNX 把它们拼成 ``[18, ...]`` 一个张量，图里再拆回去。
    """
    cache = modeling_module.Qwen3_5DynamicCache(model_config)
    expected = cache.layer_types.count("linear_attention")
    if conv_state.shape[0] != expected:
        raise ValueError(f"conv_state 层数不匹配: {conv_state.shape[0]} != {expected}")

    states = list(torch.split(conv_state, split_size_or_sections=1, dim=0))
    for index, layer_type in enumerate(cache.layer_types):
        if layer_type == "linear_attention":
            cache.conv_states[index] = states.pop(0)
    return cache


class ValueHeadExportWrapper(torch.nn.Module):
    """文本主干 + score 头 + 池化，一起固化进 ONNX 图。

    forward 语义（与 HF ``Qwen3_5ForSequenceClassification`` 对齐）::

        hidden = text_model(input_ids, ...)          # [1, seq, 2048]
        last   = hidden[:, num_logits_to_keep, :]    # [1, 1, 2048]
        logits = last @ score.weight.T               # [1, num_labels]

    ``num_logits_to_keep`` 复用 RKNN3 已有的 ``keep_one_logit.idx_name`` 槽位，
    因此不需要给运行时新增输入张量。
    """

    def __init__(self, text_model, head_weight: torch.Tensor, modeling_module):
        super().__init__()
        self.model = text_model
        self.config = text_model.config
        self.modeling_module = modeling_module
        # 用 Parameter 包一下，导出时才会作为 initializer 落盘
        self.head_weight = torch.nn.Parameter(head_weight.clone(), requires_grad=False)

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        Tc,
        num_logits_to_keep,
        conv_state,
    ):
        cache = pack_dynamic_cache(conv_state, self.model.config, self.modeling_module)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            Tc=Tc,
        )

        hidden = outputs.last_hidden_state
        hidden = hidden.index_select(dim=1, index=num_logits_to_keep)
        logits = F.linear(hidden, self.head_weight)  # [1, 1, num_labels]

        states = [v for v in outputs.past_key_values.conv_states if v is not None]
        return logits, torch.cat(states, dim=0)


def build_trace_inputs(tokenizer, seq_len: int):
    """构造一组固定 shape 的示例输入，用于 torch.onnx.export 追踪。"""
    try:
        text = tokenizer.apply_chat_template(
            [CHAT_CONTEXT], tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            [CHAT_CONTEXT], tokenize=False, add_generation_prompt=False
        )

    enc = tokenizer([text], return_tensors="pt")
    input_ids = enc["input_ids"]
    # 追踪长度无所谓（动态轴会覆盖），但必须 <= seq_len 以便保持云淡风轻
    if input_ids.shape[1] > seq_len:
        input_ids = input_ids[:, :seq_len]
    real_len = input_ids.shape[1]

    attention_mask = torch.ones(1, real_len, dtype=torch.float32)
    position_ids = torch.arange(real_len, dtype=torch.long).unsqueeze(0)
    tc = torch.tensor([real_len], dtype=torch.int64)
    keep = torch.tensor([real_len - 1], dtype=torch.long)
    return text, input_ids, position_ids, attention_mask, tc, keep


# --------------------------------------------------------------------------- #
# 配置/分词器/embedding 导出
# --------------------------------------------------------------------------- #


def export_value_config(
    model_path: str,
    config_path: Path,
    chat_context,
    prompt: str,
    num_labels: int,
    task_type=None,
    zoo_root: str | Path | None = None,
) -> None:
    """写 ``.config.pkl``。

    与官方 ``export_llm_config`` 的差别：
      * ``hf_config_json`` 里把 ``architectures`` 改成
        ``Qwen3_5ForSequenceClassification`` 并补上 ``num_labels``，
        否则运行时可能按 CausalLM 去解释日志；
      * 额外写入 ``num_labels`` 顶层字段，方便下游直接读取。
    """
    from transformers import AutoConfig, AutoTokenizer

    # 必须用官方 helper 的切分逻辑，保证 system_prompt / prefix / postfix 与
    # C++ 侧期望的格式完全一致。
    ensure_zoo_on_path(zoo_root)
    from py_utils.export_llm_helper import split_chat_template_prompt

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    system_prompt = prompt_prefix = prompt_postfix = chat_template = ""
    if tokenizer.chat_template is not None and chat_context is not None and prompt is not None:
        try:
            system_prompt, prompt_prefix, prompt_postfix = split_chat_template_prompt(
                tokenizer.chat_template, chat_context, prompt
            )
            chat_template = tokenizer.chat_template
        except Exception as exc:  # noqa: BLE001 - 官方 helper 也是这么兜底的
            print(f"[warn] chat template 切分失败({exc})，回退到 Qwen 默认模板")
            system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            prompt_prefix = "<|im_start|>user\n"
            prompt_postfix = "<|im_end|>\n"
            chat_template = tokenizer.chat_template or ""

    text_config = config.get_text_config() if hasattr(config, "get_text_config") else config

    merged = config.to_dict()
    merged["architectures"] = ["Qwen3_5ForSequenceClassification"]
    merged["num_labels"] = num_labels
    if task_type is not None:
        merged["task_type"] = task_type

    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": chat_template,
        "vocab_size": text_config.vocab_size,
        "hidden_size": text_config.hidden_size,
        "num_labels": num_labels,
        "hf_config_json": json.dumps(merged, default=str),
    }
    if task_type is not None:
        llm_config["task_type"] = task_type

    with open(config_path, "wb") as fh:
        pickle.dump(llm_config, fh)

    print("system_prompt :", repr(system_prompt))
    print("prompt_prefix :", repr(prompt_prefix))
    print("prompt_postfix:", repr(prompt_postfix))
    print(f"Model configuration exported to {config_path}")


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def main() -> int:
    args = build_parser().parse_args()
    setup_env(args.hf_endpoint)

    # --model_path 是模型仓库 id：先下载到本地缓存，下游（transformers /
    # safetensors / convert_hf_to_gguf）只跟本地目录打交道，无需再分支。
    model_path = resolve_model_source(args.model_path, args.modelscope)

    zoo_root = ensure_zoo_on_path(args.zoo_root)

    export_modeling_path = resolve_export_modeling(args.export_modeling)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / f"{args.name}.onnx"
    prefix = onnx_path.with_suffix("")
    config_path = Path(str(prefix) + ".config.pkl")
    tokenizer_path = Path(str(prefix) + ".tokenizer.gguf")
    embed_path = Path(str(prefix) + ".embed.bin")

    modeling_module = load_export_modeling_module(export_modeling_path)

    from transformers import AutoConfig, AutoTokenizer

    print(f"--> 加载分词器: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    print("--> 加载文本主干 (Qwen3_5TextModel)")
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    # 强制使用 eager attention，避免 transformers 5.x 默认 SDPA 在 ONNX 导出时
    # 触发 GQA / scaled_dot_product_attention 的 symbolic 转换问题。
    config._attn_implementation = "eager"
    text_config = config.get_text_config()
    text_config._attn_implementation = "eager"
    text_model = modeling_module.Qwen3_5TextModel.from_pretrained(
        str(model_path),
        config=text_config,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    ).eval()

    print(f"--> 读取分类头 {args.head_key}")
    head_weight = read_head_weight(model_path, args.head_key).to(torch.float32)
    if tuple(head_weight.shape) != (args.num_labels, text_config.hidden_size):
        raise ValueError(
            f"分类头形状不符: {tuple(head_weight.shape)} != "
            f"({args.num_labels}, {text_config.hidden_size})"
        )

    wrapper = ValueHeadExportWrapper(text_model, head_weight, modeling_module).eval()

    # ---- 先跑一次，拿到 conv_state 的合法 shape ----
    text, input_ids, position_ids, attention_mask, tc, keep = build_trace_inputs(tokenizer, args.seq_len)
    print(f"--> 追踪提示词: {text!r} (len={input_ids.shape[1]})")

    with torch.inference_mode():
        warm = text_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            Tc=tc,
        )
    conv_states = [v for v in warm.past_key_values.conv_states if v is not None]
    conv_probe = torch.cat(conv_states, dim=0)

    # 追踪用的 conv_state 必须全 0，不能用上面 warm 出来的值！
    #
    # 原生的 prefill 路径是（modeling_qwen3_5_export.py 里的 conv_state 分支）：
    #     conv_state = F.pad(mixed_qkv, (conv_kernel_size - seq, 0))
    # 也就是用**输入自身的**前几个 token 当前导历史，等价于"开始前状态为 0"。
    # 而一旦传入非空 conv_state，模型会走 decode 分支
    # （``use_precomputed_states or conv_state is not None``），把这段历史
    # 当成序列之前已处理过的 token，结果完全不同。
    #
    # 实测：zeros -> 与原生 max|Δ|=3.9e-07（逐位吻合）；warm -> max|Δ|=2.2e+00。
    # RKNN3 侧也是靠 input_initial_value 传全 0 来对齐的，见 export_rknn.py。
    conv_state0 = torch.zeros_like(conv_probe)
    print(f"    conv_state: {tuple(conv_state0.shape)} ({len(conv_states)} 个线性注意力层, 全 0)")

    # ---- 导出 ONNX ----
    print(f"--> 导出 ONNX: {onnx_path}")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (input_ids, position_ids, attention_mask, tc, keep, conv_state0),
            str(onnx_path),
            export_params=True,
            do_constant_folding=True,
            opset_version=19,
            input_names=[
                "input_ids",
                "position_ids",
                "attention_mask",
                "Linear_Tc",
                "num_logits_to_keep",
                "conv_state",
            ],
            output_names=["output", "conv_state_out"],
            dynamic_axes={
                "input_ids": {1: "sequence"},
                "position_ids": {1: "sequence"},
                "attention_mask": {1: "sequence"},
            },
            custom_opsets={"rknn": 1},
        )

    # ---- 校验 + 打印 I/O 契约 ----
    import onnx

    # 注意：权重超过 2GB 时 torch.onnx.export 会自动切成 external data
    # （value_llm.onnx 本身只有几百 KB，权重在同目录的 onnx__* / *.weight 文件里）。
    # 所以：
    #   * 校验必须把**路径**交给 checker，让 C++ 侧按相对路径去找 sidecar，
    #     自己 load_external_data=True 会把几个 GB 全读进内存（没必要且很慢）；
    #   * 打印 shape 时用 load_external_data=False，只读图结构。
    onnx.checker.check_model(str(onnx_path))
    print("    ONNX checker: OK")

    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    for vi in onnx_model.graph.input:
        dims = [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim]
        print(f"    in  {vi.name:24s} {dims}")
    for vi in onnx_model.graph.output:
        dims = [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim]
        print(f"    out {vi.name:24s} {dims}")
    print(f"    节点数 {len(onnx_model.graph.node)}, initializer 数 {len(onnx_model.graph.initializer)}")

    # ---- 三个附属产物 ----
    print("--> 导出 config.pkl")
    export_value_config(
        str(model_path),
        config_path,
        {"messages": [CHAT_CONTEXT], "add_generation_prompt": False},
        PROMPT,
        args.num_labels,
        task_type=None if args.task_type < 0 else args.task_type,
        zoo_root=zoo_root,
    )

    print("--> 导出 tokenizer.gguf")
    from py_utils.export_llm_helper import export_embed_weight, export_tokenizer

    export_tokenizer(str(model_path), str(tokenizer_path))

    if args.skip_embed_bin:
        print("--> 跳过 .embed.bin")
    else:
        print("--> 导出 embed.bin")
        export_embed_weight(text_model.get_input_embeddings().weight, str(embed_path))

    print("\n=== 产物 ===")
    for path in (onnx_path, config_path, tokenizer_path, embed_path):
        if path.is_file():
            print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
