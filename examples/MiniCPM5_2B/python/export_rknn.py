"""Convert MiniCPM5-2B ONNX to an 8-core RKNN model."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile


def check_ret(ret, stage):
    if ret != 0:
        raise RuntimeError(f'{stage} failed with return code {ret}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx_path', required=True, help='Input .onnx file with a matching .config.pkl in the same directory')
    parser.add_argument('--output_path', required=True, help='Output .rknn file path')
    parser.add_argument('--dtype', choices=['w4a16'], default='w4a16',
                        help='Weight quantization: w4a16/group32 (default)')
    parser.add_argument('--target', type=str.lower, default='rk1828',
                        help='Target chip (default: rk1828)')
    args = parser.parse_args()
    onnx = Path(args.onnx_path).expanduser().resolve()
    output = Path(args.output_path).expanduser().resolve()
    config = onnx.with_suffix('.config.pkl')
    if onnx.suffix != '.onnx' or not onnx.is_file() or not config.is_file():
        parser.error('A valid .onnx file and matching .config.pkl are required')
    if output.suffix != '.rknn':
        parser.error('--output_path must point to an .rknn file')
    from rknn.api import RKNN, DEFAULT_RKNN_LLM_CONFIG

    context = 2048
    method = 'group32'
    llm = deepcopy(DEFAULT_RKNN_LLM_CONFIG)
    llm['attention_config'][0].update(kvcache_buffer_len=context, max_position_embeddings=context,
                                     position_embeddings_host_storage=False, kvcache_dtype='Float16')
    llm['llm_head'][0].update(device=args.target.upper(), quantized_dtype=args.dtype, quantized_method=method)
    report = Path.cwd() / 'tmp/model_report.html'
    if report.exists():
        report.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        check_ret(rknn.config(target_platform=args.target, core_num=8,
                             quantized_dtype=args.dtype, quantized_algorithm='normal',
                             quantized_method=method, llm_config=llm), 'Configuration')
        check_ret(rknn.load_llm(model=str(onnx), config=str(config)), 'Loading')
        # W4A16 normal quantization is weight-only and does not use calibration data.
        check_ret(rknn.build(do_quantization=True), 'Build')
        with tempfile.TemporaryDirectory(prefix='.minicpm5-', dir=output.parent) as work:
            temp = Path(work) / output.name
            check_ret(rknn.export_rknn(str(temp)), 'Export')
            for artifact in [temp, temp.with_suffix('.weight'), report]:
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise RuntimeError(f'Missing or empty output artifact: {artifact}')
            temp.with_suffix('.weight').replace(output.with_suffix('.weight'))
            temp.replace(output)
            shutil.copy2(report, output.parent / 'model_report.html')
    finally:
        rknn.release()
    print(f'Export complete: {output}; context={context}; quantization={args.dtype}/{method}/normal')


if __name__ == '__main__':
    main()
