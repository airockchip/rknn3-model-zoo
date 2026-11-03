"""Export Qwen3-ForcedAligner ONNX, with optional decoder GRQ quantization."""

import argparse
import json
import pickle
import tempfile
from pathlib import Path

from qwen3_forcealign.model_source import resolve_model_path
from qwen3_forcealign.prepare_calibration import prepare_dataset
from qwen3_forcealign.resources import RESOURCE_FILES, write_resources


def add_timestamp_argmax(model_path):
    """Select timestamp classes on device to avoid transferring full logits."""
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(model_path))
    logits = model.graph.output[0].name
    model.graph.node.extend(
        [
            helper.make_node(
                "ArgMax",
                [logits],
                ["timestamp_indices_i64"],
                name="TimestampArgMax",
                axis=-1,
                keepdims=0,
                select_last_index=0,
            ),
            helper.make_node(
                "Cast",
                ["timestamp_indices_i64"],
                ["timestamp_indices"],
                name="TimestampIndicesInt32",
                to=TensorProto.INT32,
            ),
        ]
    )
    del model.graph.output[:]
    model.graph.output.append(
        helper.make_tensor_value_info(
            "timestamp_indices", TensorProto.INT32, [1, "sequence"]
        )
    )
    onnx.save(model, str(model_path))


def export_encoder(model_path, output_path):
    import torch
    from qwen3_forcealign.audio_encoder import (
        Qwen3ASRAudioEncoderConfig,
        qwen3_asr_audio,
    )
    from safetensors.torch import load_file

    config = json.loads((model_path / "config.json").read_text())
    config = Qwen3ASRAudioEncoderConfig(
        **config["thinker_config"]["audio_config"]
    )
    encoder = qwen3_asr_audio(config).eval()
    weights = {}
    prefix = "thinker.audio_tower."
    for path in sorted(model_path.glob("*.safetensors")):
        weights.update(
            {
                k[len(prefix) :]: v
                for k, v in load_file(path).items()
                if k.startswith(prefix)
            }
        )
    encoder.load_state_dict(weights, strict=True)
    torch.manual_seed(0)
    x = torch.randn(4, 1, 128, 100)
    audio_mask = torch.zeros(1, 1, 1, 52)
    # Match the eager reference and mask padded audio positions.
    torch.onnx.export(
        encoder,
        (x, audio_mask),
        str(output_path / "encoder.onnx"),
        opset_version=18,
        input_names=["x", "audio_mask"],
        dynamo=False,
        dynamic_axes={
            "x": {0: "chunks"},
            "audio_mask": {3: "audio_tokens"},
            "h": {0: "audio_tokens"},
        },
        output_names=["h"],
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
    )


def capture_decoder_inputs(decoder, embeds, mask, positions):
    """Capture GRQ inputs after the wrapper constructs its causal mask."""
    import torch

    class Captured(Exception):
        pass

    captured = {}

    def hook(module, args, kwargs):
        captured.update(args=args, kwargs=kwargs)
        raise Captured()

    handle = decoder.model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            decoder(embeds, mask, positions)
    except Captured:
        pass
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("Decoder input capture did not run")
    return captured


def load_calibration_paths(calibration_path):
    """Validate prepared samples before loading or quantizing the model."""
    cases = json.loads(calibration_path.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("Calibration manifest must be a nonempty array")
    paths = []
    for case in cases:
        if not isinstance(case, str) or not case:
            raise ValueError("Calibration entries must be directory paths")
        path = (calibration_path.parent / case).resolve()
        for name in (
            "reference.json",
            "input_embeds.npy",
            "attention_mask.npy",
            "position_ids.npy",
        ):
            if not (path / name).is_file():
                raise FileNotFoundError(path / name)
        paths.append(path)
    return paths


def export_decoder(model_path, calibration_path, output_path, quant=False):
    import numpy as np
    import torch
    from qwen3_forcealign.decoder import TextAllLogits
    from qwen_asr import Qwen3ForcedAligner

    paths = load_calibration_paths(calibration_path) if quant else []
    aligner = Qwen3ForcedAligner.from_pretrained(
        str(model_path),
        dtype=torch.float16 if quant else torch.float32,
        device_map="cpu",
        attn_implementation="eager",
    )
    decoder = TextAllLogits(aligner.model.thinker).eval()

    if quant:
        from rknn.quantization.api import RKQuantizer

        with tempfile.TemporaryDirectory(prefix="forced_aligner_grq_") as work:
            records = []
            for i, path in enumerate(paths):
                meta = json.loads((path / "reference.json").read_text())
                n = meta["seq_len"]
                if not isinstance(n, int) or not 0 < n <= 2048:
                    raise ValueError(
                        f"Invalid calibration sequence length: {n}"
                    )
                x = torch.from_numpy(
                    np.load(path / "input_embeds.npy")[:, :n]
                ).half()
                mask = torch.ones(1, n).half()
                pos = torch.arange(n, dtype=torch.int64).reshape(1, 1, -1)
                captured = capture_decoder_inputs(decoder, x, mask, pos)
                sample = Path(work) / f"{i}.pkl"
                with sample.open("wb") as file:
                    pickle.dump(captured, file)
                records.append({"sample": sample.name})
            dataset = Path(work) / "dataset.json"
            dataset.write_text(json.dumps(records))
            quantizer = RKQuantizer(verbose=True)
            if (
                quantizer.load_model(
                    model=decoder.model, tokenizer=None, device="cuda"
                )
                != 0
            ):
                raise RuntimeError("Failed to load the GRQ model")
            decoder.model = quantizer.quantize(
                quantized_dtype="w4a16",
                quantized_method="group32",
                quantized_algorithm="grq",
                dataset=str(dataset),
            ).cpu()
    decoder = decoder.float().eval()
    if quant:
        tensors = tuple(
            torch.from_numpy(np.load(paths[0] / f"{name}.npy"))
            for name in ["input_embeds", "attention_mask", "position_ids"]
        )
    else:
        # Shape tracing inputs only; normal weight quantization needs no data.
        tensors = (
            torch.zeros(1, 256, 1024, dtype=torch.float32),
            torch.ones(1, 256, dtype=torch.float32),
            torch.arange(256, dtype=torch.int64).reshape(1, 1, -1),
        )
    with torch.inference_mode():
        torch.onnx.export(
            decoder,
            tensors,
            str(output_path / "decoder.onnx"),
            input_names=["input_embeds", "attention_mask", "position_ids"],
            output_names=["output"],
            dynamic_axes={
                "input_embeds": {1: "sequence"},
                "attention_mask": {1: "sequence"},
                "position_ids": {2: "sequence"},
                "output": {1: "sequence"},
            },
            opset_version=18,
            dynamo=False,
            do_constant_folding=True,
        )

    add_timestamp_argmax(output_path / "decoder.onnx")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        required=True,
        help="Local model directory or Hugging Face / ModelScope model ID",
    )
    parser.add_argument(
        "--modelscope",
        action="store_true",
        help="Download model IDs from ModelScope instead of Hugging Face",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="ONNX output directory; destination files must not exist",
    )
    parser.add_argument(
        "--quant",
        action="store_true",
        help="Apply GRQ before ONNX export; default skips GRQ",
    )
    parser.add_argument(
        "--calibration_path",
        type=Path,
        help="Calibration JSON for --quant: audio/text or prepared inputs",
    )
    parser.add_argument(
        "--stage", choices=["encoder", "decoder", "all"], default="all"
    )
    args = parser.parse_args()
    stages = ["encoder", "decoder"] if args.stage == "all" else [args.stage]
    if args.quant and "decoder" not in stages:
        parser.error("--quant applies to decoder export only")
    if args.calibration_path is not None and not args.quant:
        parser.error("--calibration_path requires --quant")
    if args.quant and (
        args.calibration_path is None or not args.calibration_path.is_file()
    ):
        parser.error("GRQ export requires --calibration_path")
    args.output_path.mkdir(parents=True, exist_ok=True)
    if any((args.output_path / f"{stage}.onnx").exists() for stage in stages):
        parser.error("Output model already exists; use a new output directory")
    if "decoder" in stages:
        for name in RESOURCE_FILES:
            if (args.output_path / name).exists():
                parser.error(f"Resource already exists: {name}")
    # The shared resolver uses the selected provider's snapshot_download.
    args.model_path = resolve_model_path(args.model_path, args.modelscope)
    import onnx
    import torch

    if args.quant and not torch.cuda.is_available():
        parser.error("GRQ export requires CUDA")
    torch.set_num_threads(8)
    for stage in stages:
        with tempfile.TemporaryDirectory(dir=args.output_path) as work:
            work = Path(work)
            if stage == "encoder":
                export_encoder(args.model_path, work)
            elif args.quant:
                with tempfile.TemporaryDirectory(
                    prefix="calibration_", dir=work
                ) as calibration_work:
                    manifest = prepare_dataset(
                        args.model_path,
                        args.calibration_path.resolve(),
                        Path(calibration_work) / "samples",
                    )
                    export_decoder(args.model_path, manifest, work, quant=True)
            else:
                export_decoder(args.model_path, None, work)
            onnx.checker.check_model(str(work / f"{stage}.onnx"))
            if stage == "decoder":
                write_resources(args.model_path, work)
            # Check all external-data destinations before publishing files.
            artifacts = list(work.iterdir())
            for path in artifacts:
                if (args.output_path / path.name).exists():
                    raise FileExistsError(args.output_path / path.name)
            for path in artifacts:
                path.rename(args.output_path / path.name)


if __name__ == "__main__":
    main()
