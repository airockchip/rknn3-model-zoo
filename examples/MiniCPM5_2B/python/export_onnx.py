"""Export MiniCPM5-2B to ONNX with configuration, tokenizer, and embedding files."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', default='openbmb/MiniCPM5-2B',
                        help='Local model directory or Hugging Face/ModelScope model ID')
    parser.add_argument('--output_path', required=True, help='Output .onnx file path')
    parser.add_argument('--modelscope', action='store_true', help='Download remote models from ModelScope; use local directories directly')
    parser.add_argument('--no-quant', dest='quant', action='store_false', default=True,
                        help='Disable GRQ quantization (enabled by default; requires CUDA)')
    parser.add_argument('--cali_dataset', default='../../../datasets/llm_quant.json',
                        help='GRQ calibration JSON file (default: ../../../datasets/llm_quant.json)')
    args = parser.parse_args()
    dataset = None
    if args.quant:
        dataset = Path(args.cali_dataset).expanduser().resolve()
        if not dataset.is_file():
            parser.error('--cali_dataset must point to an existing JSON file')
        with dataset.open() as handle:
            samples = json.load(handle)
        if not isinstance(samples, list) or not samples or any(
                not isinstance(sample, dict) or not isinstance(sample.get('input'), str)
                or not isinstance(sample.get('target'), str) for sample in samples):
            parser.error('Calibration JSON must be a nonempty list of objects with string input and target fields')
        import torch
        if not torch.cuda.is_available():
            parser.error('GRQ quantization requires CUDA')
        from rknn.quantization.api import RKQuantizer
    output = Path(args.output_path).expanduser().resolve()
    if output.suffix != '.onnx':
        parser.error('--output_path must point to an .onnx file')
    source = Path(args.model_path).expanduser()
    if source.is_dir():
        model_path = str(source.resolve())
    elif args.model_path.startswith(('/', '.', '~')):
        parser.error('Local model directory does not exist')
    elif args.modelscope:
        from modelscope import snapshot_download
        model_path = snapshot_download(args.model_path)
    else:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(args.model_path)

    import torch
    import onnx
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    from py_utils.export_llm_helper import causal_llm_to_onnx, export_llm_config, export_embed_weight

    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if (config.model_type, config.hidden_size, config.num_hidden_layers) != ('llama', 2048, 42):
        raise ValueError('This exporter only supports the MiniCPM5-2B Llama configuration')
    config.use_cache = False
    torch.set_num_threads(min(16, os.cpu_count() or 1))
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, local_files_only=True, attn_implementation='eager').float().eval()
    if args.quant:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        quantizer = RKQuantizer(verbose=True)
        ret = quantizer.load_model(model=model, tokenizer=tokenizer, device='cuda',
                                   system_prompt=None, tools=None, system_role='system', user_role='user')
        if ret != 0:
            raise RuntimeError(f'GRQ model loading failed with return code {ret}')
        model = quantizer.quantize(quantized_dtype='w4a16', quantized_method='group32',
                                   quantized_algorithm='grq', dataset=str(dataset))
        if model is None:
            raise RuntimeError('GRQ quantization did not return a model')
        model = model.cpu().eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stage files in a sibling temporary directory until all exports succeed.
    with tempfile.TemporaryDirectory(prefix='.minicpm5-', dir=output.parent) as work:
        temp = Path(work) / output.name
        causal_llm_to_onnx(model, SimpleNamespace(export_llm_path=str(temp)))
        onnx.checker.check_model(str(temp))
        chat = {'messages': [{'role': 'user', 'content': 'RKLLM'}],
                'add_generation_prompt': True, 'enable_thinking': False}
        export_llm_config(model_path, str(temp.with_suffix('.config.pkl')), chat, 'RKLLM')
        export_embed_weight(model.model.embed_tokens.weight, str(temp.with_suffix('.embed.bin')))
        tokenizer = temp.with_suffix('.tokenizer.gguf')
        # Use the same interpreter and propagate tokenizer export failures.
        subprocess.run([sys.executable, str(repo / 'tokenizer/thirdparty/llama_vocab/convert_hf_to_gguf.py'),
                        '--vocab-only', '--outtype', 'f16', '--outfile', str(tokenizer), model_path], check=True)
        with tokenizer.open('rb') as handle:
            if handle.read(4) != b'GGUF':
                raise ValueError('Invalid tokenizer GGUF file')
        for artifact in Path(work).iterdir():
            if artifact.name != output.name:
                artifact.replace(output.parent / artifact.name)
        temp.replace(output)
    print(f'Export complete: {output}')


if __name__ == '__main__':
    main()
