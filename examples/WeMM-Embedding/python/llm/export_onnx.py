import argparse
import importlib.util
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download as hf_snapshot_download
from transformers import AutoConfig, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from py_utils.export_llm_helper import (  # noqa: E402
    export_embed_weight,
    export_tokenizer,
    split_chat_template_prompt,
)


PROMPT = "RKNN embedding"
CHAT_CONTEXT = {"role": "user", "content": PROMPT}
EMBEDDING_TOKEN = "<embedding>"
EMBEDDING_TEMPLATE_SUFFIX = "{{- '<embedding>' }}"


def parse_args():
    parser = argparse.ArgumentParser(description="Export the WeMM-Embedding-2B LLM to ONNX")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Local model directory or Hugging Face/ModelScope model ID",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output ONNX file; config, tokenizer, and embedding files are written beside it",
    )
    parser.add_argument(
        "--modelscope",
        action="store_true",
        help="Download non-local model IDs from ModelScope instead of Hugging Face",
    )
    return parser.parse_args()


def resolve_model_path(model_path, use_modelscope):
    local_path = Path(model_path).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())
    if use_modelscope:
        from modelscope import snapshot_download

        return snapshot_download(model_path)
    return hf_snapshot_download(repo_id=model_path)


def load_export_module():
    module_path = Path(__file__).with_name("modeling_qwen3_5_export.py")
    module_name = "transformers.models.qwen3_5.modeling_qwen3_5"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load the export model definition: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def pack_dynamic_cache(conv_state, model_config, modeling_module):
    cache = modeling_module.Qwen3_5DynamicCache(model_config)
    expected = cache.layer_types.count("linear_attention")
    if conv_state.shape[0] != expected:
        raise ValueError(f"conv_state layer count mismatch: {conv_state.shape[0]} != {expected}")
    states = list(torch.split(conv_state, split_size_or_sections=1, dim=0))
    for index, layer_type in enumerate(cache.layer_types):
        if layer_type == "linear_attention":
            cache.conv_states[index] = states.pop(0)
    return cache


class WeMMEmbeddingExportWrapper(torch.nn.Module):
    def __init__(self, text_model, modeling_module):
        super().__init__()
        self.model = text_model
        self.config = text_model.config
        self.modeling_module = modeling_module

    def forward(self, input_ids, position_ids, attention_mask, tc, num_logits_to_keep, conv_state):
        cache = pack_dynamic_cache(conv_state, self.model.config, self.modeling_module)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            Tc=tc,
        )
        embedding = outputs.last_hidden_state.index_select(dim=1, index=num_logits_to_keep)
        embedding = F.normalize(embedding, p=2.0, dim=-1)
        states = [value for value in outputs.past_key_values.conv_states if value is not None]
        return embedding, torch.cat(states, dim=0)


def export_onnx(model, tokenizer, output_path, modeling_module):
    text = tokenizer.apply_chat_template(
        [CHAT_CONTEXT], tokenize=False, add_generation_prompt=False
    )
    model_inputs = tokenizer([text], return_tensors="pt")
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"].to(torch.float32)
    sequence_length = input_ids.shape[1]
    position_ids = torch.arange(sequence_length, dtype=torch.long).unsqueeze(0)
    tc = torch.tensor([sequence_length], dtype=torch.int64)
    num_logits_to_keep = torch.tensor([sequence_length - 1], dtype=torch.long)

    with torch.inference_mode():
        initial_output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
            Tc=tc,
        )
    conv_state = torch.cat(
        [value for value in initial_output.past_key_values.conv_states if value is not None],
        dim=0,
    )
    wrapper = WeMMEmbeddingExportWrapper(model, modeling_module).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wemm-onnx-", dir=output_path.parent) as temp_dir:
        temp_path = Path(temp_dir) / output_path.name
        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (input_ids, position_ids, attention_mask, tc, num_logits_to_keep, conv_state),
                str(temp_path),
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
                output_names=["embedding", "conv_state_out"],
                dynamic_axes={
                    "input_ids": {1: "sequence"},
                    "position_ids": {1: "sequence"},
                    "attention_mask": {1: "sequence"},
                },
                custom_opsets={"rknn": 1},
            )
        onnx.checker.check_model(str(temp_path), full_check=False)
        for generated_file in Path(temp_dir).iterdir():
            os.replace(generated_file, output_path.parent / generated_file.name)


def export_embedding_config(model_path, config_path):
    template_path = Path(model_path) / "embedding_chat_template.jinja"
    if not template_path.is_file():
        raise FileNotFoundError(f"Missing embedding chat template: {template_path}")
    chat_template = template_path.read_text(encoding="utf-8")
    system_prompt, prompt_prefix, prompt_postfix = split_chat_template_prompt(
        chat_template,
        {"messages": [CHAT_CONTEXT]},
        PROMPT,
    )
    # WeMM's tokenizer post-processor already appends <embedding>. Keeping it in
    # the runtime postfix would append the pooling token twice.
    if prompt_postfix.endswith(EMBEDDING_TOKEN):
        prompt_postfix = prompt_postfix[: -len(EMBEDDING_TOKEN)]
    runtime_chat_template = chat_template.rstrip()
    if runtime_chat_template.endswith(EMBEDDING_TEMPLATE_SUFFIX):
        runtime_chat_template = runtime_chat_template[: -len(EMBEDDING_TEMPLATE_SUFFIX)].rstrip()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    merged_config = {**config.to_dict(), "task_type": 1}
    llm_config = {
        "system_prompt": system_prompt,
        "prompt_prefix": prompt_prefix,
        "prompt_postfix": prompt_postfix,
        "chat_template": runtime_chat_template,
        "vocab_size": config.text_config.vocab_size,
        "hidden_size": config.text_config.hidden_size,
        "hf_config_json": json.dumps(merged_config, default=str),
    }
    with config_path.open("wb") as file:
        pickle.dump(llm_config, file)
    print(f"Embedding configuration exported to {config_path}")


def publish_artifacts(staging_path, output_path):
    artifacts = sorted(path for path in staging_path.iterdir() if path.is_file())
    if not artifacts:
        raise RuntimeError("No LLM export artifacts were generated")

    # Preserve the previous artifact set until every staged file has been
    # published. If a replacement fails, restore the original files.
    with tempfile.TemporaryDirectory(prefix="wemm-backup-", dir=output_path.parent) as backup_dir:
        backup_path = Path(backup_dir)
        backups = []
        published = []
        try:
            for artifact in artifacts:
                target = output_path.parent / artifact.name
                if target.exists():
                    if not target.is_file():
                        raise IsADirectoryError(f"Artifact target is not a file: {target}")
                    backup = backup_path / artifact.name
                    os.replace(target, backup)
                    backups.append((backup, target))
            for artifact in artifacts:
                target = output_path.parent / artifact.name
                os.replace(artifact, target)
                published.append(target)
        except Exception:
            for target in published:
                target.unlink(missing_ok=True)
            for backup, target in backups:
                os.replace(backup, target)
            raise


def main():
    args = parse_args()
    output_path = Path(args.output_path).expanduser().resolve()
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("--output_path must end with .onnx")

    model_path = resolve_model_path(args.model_path, args.modelscope)
    modeling_module = load_export_module()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = modeling_module.Qwen3_5TextModel.from_pretrained(
        model_path,
        config=config.text_config,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wemm-release-", dir=output_path.parent) as temp_dir:
        staging_path = Path(temp_dir)
        staged_onnx_path = staging_path / output_path.name
        prefix = staged_onnx_path.with_suffix("")
        config_path = Path(str(prefix) + ".config.pkl")
        tokenizer_path = Path(str(prefix) + ".tokenizer.gguf")
        embedding_path = Path(str(prefix) + ".embed.bin")

        export_onnx(model, tokenizer, staged_onnx_path, modeling_module)
        export_embedding_config(model_path, config_path)
        export_tokenizer(model_path, str(tokenizer_path))
        export_embed_weight(model.get_input_embeddings().weight, str(embedding_path))

        expected_paths = (staged_onnx_path, config_path, tokenizer_path, embedding_path)
        missing_paths = [path for path in expected_paths if not path.is_file()]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise RuntimeError(f"LLM export did not generate all required artifacts: {missing}")
        publish_artifacts(staging_path, output_path)
    print(f"ONNX export completed: {output_path}")


if __name__ == "__main__":
    main()
