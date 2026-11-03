"""Export Intern-Decision vision features and decision logits for RKNN."""
import argparse
from pathlib import Path
import sys
import tempfile

import onnx
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from py_utils.export_llm_helper import (  # noqa: E402
    causal_qwen_3_5_llm_to_onnx, export_embed_weight,
    export_llm_config, export_tokenizer,
)

from modeling_intern_decision_export import (  # noqa: E402
    Qwen3_5DynamicCache, Qwen3_5ForCausalLM,
)

SYMBOLS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'


def resolve_model(path, modelscope):
    if Path(path).is_dir():
        return str(Path(path).resolve())
    if modelscope:
        from modelscope import snapshot_download
    else:
        from huggingface_hub import snapshot_download
    return snapshot_download(path)


def export_llm(model, tokenizer, model_path, output, prefix):
    text = Qwen3_5ForCausalLM(model.config.text_config).eval()
    text.model.load_state_dict(model.model.language_model.state_dict(), strict=True)
    text.lm_head.weight = text.model.embed_tokens.weight
    ids = [tokenizer.encode(s, add_special_tokens=False) for s in SYMBOLS]
    if any(len(v) != 1 for v in ids):
        raise ValueError('Answer symbols must each encode to one token')

    class DecisionExport(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = text
            self.config = text.config
            self.register_buffer('symbol_ids', torch.tensor([v[0] for v in ids]))

        def forward(self, input_ids, position_ids, attention_mask, tc,
                    logits_to_keep, conv_state):
            cache = Qwen3_5DynamicCache(self.config)
            states = iter(conv_state.split(1, dim=0))
            for i, kind in enumerate(cache.layer_types):
                if kind == 'linear_attention':
                    cache.conv_states[i] = next(states)
            out = self.model(input_ids=input_ids, position_ids=position_ids,
                             attention_mask=attention_mask, past_key_values=cache,
                             Tc=tc, logits_to_keep=logits_to_keep)
            conv = torch.cat([v for v in out.past_key_values.conv_states
                              if v is not None], dim=0)
            return out.logits.index_select(-1, self.symbol_ids), conv

    args = argparse.Namespace(tokenizer=tokenizer,
                              chat_context=[{'role': 'user', 'content': 'RKLLM'}],
                              export_llm_path=str(output / f'{prefix}.onnx'))
    causal_qwen_3_5_llm_to_onnx(DecisionExport().eval(), args)
    export_llm_config(model_path, str(output / f'{prefix}.config.pkl'), None, None,
                      {'task_type': 2})
    export_tokenizer(model_path, str(output / f'{prefix}.tokenizer.gguf'))
    export_embed_weight(text.get_input_embeddings().weight,
                        str(output / f'{prefix}.embed.bin'))


def export_vision(model, output, prefix):
    class VisionExport(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = model.model.visual
            self.register_buffer('grid', torch.tensor([[1, 24, 24]]))

        def forward(self, pixel):
            patches = pixel.repeat(2, 1, 1, 1)
            patches = patches.reshape(1, 2, 3, 12, 2, 16, 12, 2, 16)
            patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            patches = patches.reshape(576, 1536)
            return self.visual(patches, grid_thw=self.grid,
                               return_dict=True).pooler_output

    with torch.no_grad():
        torch.onnx.export(VisionExport().eval(), torch.zeros(1, 3, 384, 384),
                          str(output / f'{prefix}.onnx'), input_names=['pixel'],
                          output_names=['image_embed'], opset_version=17,
                          dynamo=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', required=True, help='Local model directory or model ID')
    parser.add_argument('--output_path', required=True, help='ONNX output directory')
    parser.add_argument('--stage', choices=['vision', 'llm', 'all'], default='all')
    parser.add_argument('--modelscope', action='store_true')
    args = parser.parse_args()
    # Use the requested model name, not the downloaded snapshot's hash directory.
    source = Path(args.model_path)
    model_name = source.resolve().name if source.is_dir() else source.name
    model_path = resolve_model(args.model_path, args.modelscope)
    output = Path(args.output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation='eager').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    for stage in ('vision', 'llm') if args.stage == 'all' else (args.stage,):
        prefix = f'{model_name}-{stage}'
        # Publish only after export and format validation have succeeded.
        with tempfile.TemporaryDirectory(prefix=f'.{stage}-', dir=output) as temp:
            pending = Path(temp)
            if stage == 'vision':
                export_vision(model, pending, prefix)
            else:
                export_llm(model, tokenizer, model_path, pending, prefix)
            graph = pending / f'{prefix}.onnx'
            onnx.checker.check_model(str(graph))
            for resource in pending.iterdir():
                if resource != graph:
                    resource.replace(output / resource.name)
            graph.replace(output / graph.name)
        print(f'{stage}: export and ONNX checker passed', flush=True)


if __name__ == '__main__':
    main()
