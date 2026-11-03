"""Export Spark2_5 (hybrid sliding/full attention) to ONNX for RKNN3 load_llm.

The wrapper calls Spark2_5ForCausalLM.forward directly (no control-flow
reimplementation); the two mask streams and two RoPE streams are explicit ONNX
inputs:

- input_ids          [1,S] int64
- attention_mask     [1,S] float  -> additive mask for the full_attention layers
- position_ids       [1,S] int64  -> RoPE stream theta=5e6 partial 0.25 (full_attention)
- attention_mask_1   [1,S] float  -> additive mask for the sliding_attention layers
                                    (window 512 enforced by the mask content the
                                    session feeds at runtime; the graph only adds it)
- position_ids_1     [1,S] int64  -> RoPE stream theta=1e4 partial 1.0 (sliding_attention)
- num_logits_to_keep  [1] int64   -> zero-based non-negative token index gathered
                                    along the sequence dim (prefill=seq_len-1, decode=0)

Opset 19, dynamic sequence length, output "output".
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import torch
from torch import nn


class Spark2_5ModelWrapper(nn.Module):
    """Passes the explicit inputs straight through to Spark2_5ForCausalLM.forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model  # Spark2_5ForCausalLM

    def forward(self, input_ids, attention_mask_full, position_ids_full,
                attention_mask_sld, position_ids_sld, num_logits_to_keep):
        # no PLE: 直接传 input_ids 给 model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask_full=attention_mask_full,
            attention_mask_sld=attention_mask_sld,
            position_ids_full=position_ids_full,
            position_ids_sld=position_ids_sld,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=False,
            cache_position=None,
            logits_to_keep=num_logits_to_keep,
        )
        return outputs.logits


def spark2_5_to_onnx(model, args):
    wrapped_model = Spark2_5ModelWrapper(model)

    # Debug Parameter
    args.prompt_size = 64
    args.dynamic_shape = True

    model.eval()
    in_len = args.prompt_size

    dummy_input = torch.zeros((1, in_len), dtype=torch.long)

    attention_mask_full = torch.ones((1, in_len), dtype=torch.float)
    position_ids_full = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)

    attention_mask_sld = torch.ones((1, in_len), dtype=torch.float)
    position_ids_sld = torch.arange(0, in_len, dtype=torch.long).unsqueeze(0)

    inputs = (dummy_input,
              attention_mask_full,
              position_ids_full,
              attention_mask_sld,
              position_ids_sld
              )
    # 打印inputs的成员的shape
    for i, input_tensor in enumerate(inputs):
        print(f"Input {i} shape: {input_tensor.shape}")

    input_names = ["input_ids", "attention_mask", "position_ids", "attention_mask_1", "position_ids_1"]

    dynamic_axes = {}
    if args.dynamic_shape:
        dynamic_axes.update({
            'input_ids': {1: 'sequence'},
            'attention_mask': {1: 'sequence'},
            'position_ids': {1: 'sequence'},
            'attention_mask_1': {1: 'sequence'},
            'position_ids_1': {1: 'sequence'},
        })

    output_names = ["output"]

    # num_logits_to_keep：沿序列维 gather 的非负 token 下标。dummy 用 0；
    # 部署时 prefill 传 seq_len-1，decode 传 0。
    num_logits_to_keep = torch.tensor(0, dtype=torch.long).reshape(1)
    inputs = (*inputs, num_logits_to_keep)
    input_names.append('num_logits_to_keep')
    dynamic_axes['num_logits_to_keep'] = {0: 'num_logits_to_keep'}

    if getattr(args, 'output_hidden_states', False):
        inputs = (*inputs, True)

    with torch.no_grad():
        torch.onnx.export(
            wrapped_model,
            inputs,
            args.export_llm_path,
            export_params=True,
            opset_version=19,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=True,
        )

    print(f"Exported to {os.path.abspath(args.export_llm_path)}")
