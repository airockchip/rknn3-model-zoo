import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com/")
import sys

import torch
import onnx
from onnx import external_data_helper


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_ROOT = os.path.dirname(SCRIPT_DIR)

if PYTHON_ROOT not in sys.path:
    sys.path.insert(0, PYTHON_ROOT)


from Qwen3_TTS.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer


def resolve_path(path, base_dir=PYTHON_ROOT):
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def build_tokenizer(args):
    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)
    else:
        args.model_path = resolve_path(args.model_path)

    return Qwen3TTSTokenizer.from_pretrained(
        args.model_path,
        dtype=torch.float32,
        attn_implementation=args.attn_implementation,
    )


def export_speech_decoder(tokenizer, args):
    args.export_speech_decoder_path = resolve_path(args.export_speech_decoder_path)
    export_dir = os.path.dirname(args.export_speech_decoder_path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir)

    decoder = tokenizer.model.decoder.eval()
    fake_codes = torch.randint(
        low=1,
        high=16,
        size=(1, 16, args.left_context_size + args.chunk_size),
        dtype=torch.int64,
        device=tokenizer.device,
    )

    with torch.no_grad():
        torch.onnx.export(
            decoder,
            fake_codes,
            args.export_speech_decoder_path,
            input_names=["codes"],
            output_names=["audio_values"],
            opset_version=args.opset,
            do_constant_folding=True,
            # dynamo=False,
        )

    if args.single_file:
        merge_external_data_to_single_file(args.export_speech_decoder_path)


def merge_external_data_to_single_file(onnx_path):
    model = onnx.load(onnx_path, load_external_data=True)
    external_data_helper.convert_model_from_external_data(model)

    single_file_path = f"{onnx_path}.single"
    onnx.save_model(
        model,
        single_file_path,
        save_as_external_data=False,
    )
    os.replace(single_file_path, onnx_path)

    for external_path in get_external_data_candidates(onnx_path):
        if os.path.exists(external_path):
            os.remove(external_path)


def get_external_data_candidates(onnx_path):
    stem, ext = os.path.splitext(onnx_path)
    candidates = [
        f"{onnx_path}.data",
        f"{stem}.data",
    ]
    if ext:
        candidates.append(f"{stem}{ext}.data")

    # Keep order stable while removing duplicates.
    return list(dict.fromkeys(candidates))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen3-TTS speech decoder onnx model for RKNN")
    parser.add_argument(
        "--model_path",
        type=str,
        help="speech tokenizer path or name",
        required=False,
        default="../../../../CKPT/Qwen3-TTS-12Hz-1.7B-Base/speech_tokenizer",
    )
    parser.add_argument(
        "--export_speech_decoder_path",
        type=str,
        help="export speech decoder onnx model path",
        required=False,
        default="../models/speech_decoder/speech_decoder.onnx",
    )
    parser.add_argument("--modelscope", action="store_true", help="Whether download model from www.modelscope.cn")
    parser.add_argument("--chunk_size", type=int, help="Chunk size used by the original decoder export flow", required=False, default=25)
    parser.add_argument("--left_context_size", type=int, help="Left context size used by the original decoder export flow", required=False, default=25)
    parser.add_argument("--opset", type=int, help="ONNX opset version", required=False, default=19)
    parser.add_argument("--single_file", type=int, help="Whether convert ONNX external data into a single file", required=False, default=True)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        choices=["eager"],
        help="ONNX export requires eager attention",
        required=False,
        default="eager",
    )

    args = parser.parse_args()
    tokenizer = build_tokenizer(args)
    export_speech_decoder(tokenizer, args)
