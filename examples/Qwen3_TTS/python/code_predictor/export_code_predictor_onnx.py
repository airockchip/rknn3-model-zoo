import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")
import sys

import numpy as np
import onnx
import torch
import pickle
from onnx import numpy_helper


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.dirname(SCRIPT_DIR)

if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)


from Qwen3_TTS.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration


UNORDERED_MAP_HINT = """Export failed with 'RuntimeError: unordered_map::at'.

Please patch `transformers/masking_utils.py` before retrying:
  Replace:
    sdpa_mask = sdpa_mask_recent_torch if _is_torch_greater_or_equal_than_2_6 else sdpa_mask_older_torch
  With:
    sdpa_mask = sdpa_mask_older_torch

如果导出时遇到 `RuntimeError: unordered_map::at`，请先修改
`transformers/masking_utils.py` 后再重试：
  把：
    sdpa_mask = sdpa_mask_recent_torch if _is_torch_greater_or_equal_than_2_6 else sdpa_mask_older_torch
  改成：
    sdpa_mask = sdpa_mask_older_torch
"""


def resolve_path(path, base_dir=PYTHON_ROOT):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def patch_reshape_for_dynamic_shape_error(onnx_path):
    target_reshape_name = "/model/model/Reshape"
    new_shape = np.array([-1, 1], dtype=np.int64)

    print(f"[INFO] Loading ONNX: {onnx_path}")
    model = onnx.load(onnx_path, load_external_data=False)
    graph = model.graph
    modified = False

    for node in graph.node:
        if node.op_type == "Reshape" and node.name == target_reshape_name:
            print(f"[INFO] Found target Reshape: {node.name}")

            if len(node.input) != 2:
                raise RuntimeError("Unexpected Reshape input count")

            shape_input = node.input[1]

            for i, init in enumerate(graph.initializer):
                if init.name == shape_input:
                    graph.initializer[i].CopyFrom(numpy_helper.from_array(new_shape, shape_input))
                    print(f"[OK] Replaced initializer shape: {shape_input}")
                    modified = True
                    break

            if not modified:
                for const_node in graph.node:
                    if const_node.op_type == "Constant" and const_node.output[0] == shape_input:
                        for attr in const_node.attribute:
                            if attr.name == "value":
                                attr.t.CopyFrom(numpy_helper.from_array(new_shape))
                                print(f"[OK] Replaced Constant node shape: {shape_input}")
                                modified = True
                                break
                        if modified:
                            break

            if not modified:
                raise RuntimeError(f"Shape source not found for Reshape: {shape_input}")

            break

    if not modified:
        raise RuntimeError("Target Reshape node not modified")

    print(f"[INFO] Saving RKNN-compatible ONNX: {onnx_path}")
    onnx.save(model, onnx_path)
    print("[DONE] ONNX modification finished successfully")


class CodePredictorExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        num_logits_to_keep=None,
        generation_steps=0,
    ):
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            num_logits_to_keep=num_logits_to_keep,
            generation_steps=generation_steps,
        )
        return outputs.logits


def build_model(args):
    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)
    else:
        args.model_path = resolve_path(args.model_path)

    config = Qwen3TTSConfig.from_pretrained(args.model_path)
    config.use_cache = False
    config._attn_implementation = args.attn_implementation

    if args.load_weight:
        model = Qwen3TTSForConditionalGeneration.from_pretrained(
            args.model_path,
            config=config,
            torch_dtype=torch.float32,
            attn_implementation=args.attn_implementation,
        )
    else:
        model = Qwen3TTSForConditionalGeneration._from_config(config)

    return model.eval()


def export_code_predictor(model, args):
    args.export_code_predictor_path = resolve_path(args.export_code_predictor_path)
    export_dir = os.path.dirname(args.export_code_predictor_path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir)

    hidden_size = model.config.talker_config.hidden_size
    fake_inputs = (
        torch.randn((1, args.seq_len, hidden_size), dtype=torch.float32),
        torch.ones((1, args.seq_len), dtype=torch.float32),
        torch.arange(args.seq_len, dtype=torch.int32).unsqueeze(0),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
    )

    wrapper = CodePredictorExportWrapper(model.talker.code_predictor).eval()
    input_names = [
        "input_embeds",
        "attention_mask",
        "position_ids",
        "num_logits_to_keep",
        "generation_steps",
    ]
    output_names = ["logits"]
    dynamic_axes = {
        "input_embeds": {1: "sequence_length"},
        "attention_mask": {1: "sequence_length"},
        "position_ids": {1: "sequence_length"},
    }

    try:
        torch.onnx.export(
            wrapper,
            fake_inputs,
            args.export_code_predictor_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            dynamo=False,
        )
    except RuntimeError as exc:
        if "unordered_map::at" in str(exc):
            raise RuntimeError(UNORDERED_MAP_HINT) from exc
        raise

    # Fix the Reshape shape source to avoid dynamic-shape related reshape errors in RKNN.
    patch_reshape_for_dynamic_shape_error(args.export_code_predictor_path)


def export_code_predictor_config(model, args):
    code_predictor_config = model.config.talker_config.code_predictor_config
    args.export_code_predictor_path = resolve_path(args.export_code_predictor_path)
    export_config_path = args.export_code_predictor_path.rsplit(".", 1)[0] + ".config.pkl"
    export_dir = os.path.dirname(export_config_path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir)

    metadata = {
        "vocab_size": code_predictor_config.vocab_size,
        "hidden_size": model.config.talker_config.hidden_size,
    }

    with open(export_config_path, "wb") as file_obj:
        pickle.dump(metadata, file_obj, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"config_path exported to {export_config_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen3-TTS code_predictor onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument(
        "--model_path",
        type=str,
        help="model path or name",
        required=False,
        default="../../../../CKPT/Qwen3-TTS-12Hz-1.7B-Base",
    )
    parser.add_argument(
        "--export_code_predictor_path",
        type=str,
        help="export code_predictor onnx model path",
        required=False,
        default="../models/code_predictor/code_predictor.onnx",
    )
    parser.add_argument("--modelscope", action="store_true", help="Whether download model from www.modelscope.cn")
    parser.add_argument("--seq_len", type=int, help="Fake input sequence length for export", required=False, default=1)
    parser.add_argument("--opset", type=int, help="ONNX opset version", required=False, default=19)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices=["eager"],
        help="ONNX export requires eager attention",
        required=False,
        default="eager",
    )
    args = parser.parse_args()
    model = build_model(args)
        
    export_code_predictor(model, args)
    export_code_predictor_config(model, args)
