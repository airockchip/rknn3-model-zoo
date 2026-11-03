import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")
import sys

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.dirname(SCRIPT_DIR)

if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)


from Qwen3_TTS.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration


def resolve_path(path, base_dir=PYTHON_ROOT):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


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


def export_text_projector(model, args):
    args.export_text_projector_path = resolve_path(args.export_text_projector_path)
    export_dir = os.path.dirname(args.export_text_projector_path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir)

    text_projector = model.talker.text_projection.eval()
    fake_input = torch.randn((1, args.seq_len, model.config.talker_config.text_hidden_size), dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            text_projector,
            (fake_input,),
            args.export_text_projector_path,
            input_names=["text_embed"],
            output_names=["text_projection"],
            opset_version=args.opset,
            do_constant_folding=True,
            external_data=False,
        )

    print(f"text_projector onnx exported to {args.export_text_projector_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen3-TTS text projector onnx model for RKNN")
    parser.add_argument("--load_weight", type=int, help="Whether load model weight", required=False, default=True)
    parser.add_argument(
        "--model_path",
        type=str,
        help="model path or name",
        required=False,
        default="../../../../CKPT/Qwen3-TTS-12Hz-1.7B-Base",
    )
    parser.add_argument(
        "--export_text_projector_path",
        type=str,
        help="export text projector onnx model path",
        required=False,
        default="../models/text_projector/text_projection.onnx",
    )
    parser.add_argument("--modelscope", action="store_true", help="Whether download model from www.modelscope.cn")
    parser.add_argument("--seq_len", type=int, help="Fake input sequence length for export", required=False, default=512)
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
    export_text_projector(model, args)
