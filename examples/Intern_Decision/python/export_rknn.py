"""Convert Intern-Decision components for RK1820, RK1828 or RK3572."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import onnx
from rknn.api import DEFAULT_RKNN_LLM_CONFIG, RKNN


def check(ret, operation):
    if ret != 0:
        raise RuntimeError(f'{operation} failed: {ret}')


def find_models(source, stages):
    models = {}
    for stage in stages:
        candidates = sorted(path for path in source.glob(f'*-{stage}.onnx')
                            if path.is_file())
        if len(candidates) != 1:
            raise ValueError(
                f'Expected exactly one *-{stage}.onnx in {source}, '
                f'found {len(candidates)}')
        models[stage] = candidates[0]
    names = {path.stem.removesuffix(f'-{stage}')
             for stage, path in models.items()}
    if len(names) != 1:
        raise ValueError('Vision and LLM ONNX files must share the model name')
    return models


def convert(stage, model_path, output, platform):
    source = model_path.parent
    prefix = model_path.stem
    core_num = 1 if platform == 'rk3572' else 8
    resources = []
    if stage == 'llm':
        resources = [source / f'{prefix}.{suffix}' for suffix in
                     ('tokenizer.gguf', 'embed.bin')]
        for resource in [source / f'{prefix}.config.pkl', *resources]:
            if not resource.is_file():
                raise FileNotFoundError(resource)
    report = Path('tmp/model_report.html')
    if report.exists():
        report.unlink()
    rknn = RKNN(verbose=True)
    try:
        if stage == 'vision':
            check(rknn.config(target_platform=platform, core_num=core_num,
                              # Official (pixel / 255 - 0.5) / 0.5 normalization.
                              mean_values=[[127.5, 127.5, 127.5]],
                              std_values=[[127.5, 127.5, 127.5]],
                              input_attrs={'pixel': {'dtype': 'uint8',
                                                     'layout': 'NHWC'}}), 'config')
            check(rknn.load_onnx(model=str(model_path)), 'load_onnx')
            check(rknn.build(do_quantization=False), 'build')
        else:
            graph = onnx.load(str(model_path), load_external_data=False)
            shapes = [[d.dim_value or 1 for d in v.type.tensor_type.shape.dim]
                      for v in graph.graph.input]
            dynamic = [[[1, 128], [1, 128], [1, 128]] + shapes[3:], shapes]
            initial = [[None] * (len(s) - 1) + [np.zeros(s[-1], np.float32)]
                       for s in dynamic]
            remaining = [v for v in graph.graph.output if v.name != 'conv_state_out']
            del graph.graph.output[:]
            graph.graph.output.extend(remaining)
            # Preserve external-weight relative paths next to the source ONNX.
            converted = source / f'{prefix}_streaming.onnx'
            onnx.save(graph, str(converted))
            config = deepcopy(DEFAULT_RKNN_LLM_CONFIG)
            config['attention_config'][0]['kvcache_buffer_len'] = 4096
            config['attention_config'][0]['max_position_embeddings'] = 4096
            # load_llm replaces token IDs with the embedding input.
            attrs = {'input_embeds': {'dtype': 'float16'}}
            check(rknn.config(target_platform=platform, core_num=core_num,
                              dynamic_input=dynamic, input_attrs=attrs,
                              quantized_dtype='w4a16', quantized_method='group32',
                              quantized_algorithm='normal', llm_config=config,
                              linear_attn_out_project_dtype=None,
                              input_initial_value=initial,
                              cvt_conv_streaming=[n.output[0] for n in graph.graph.node
                                                  if n.op_type == 'Conv']), 'config')
            check(rknn.load_llm(model=str(converted),
                               config=str(source / f'{prefix}.config.pkl'),
                               seq=[1, 128]), 'load_llm')
            check(rknn.build(do_quantization=True), 'build')
        check(rknn.export_rknn(str(output / f'{prefix}.rknn')), 'export_rknn')
        for suffix in ('rknn', 'weight'):
            if not (output / f'{prefix}.{suffix}').is_file():
                raise FileNotFoundError(output / f'{prefix}.{suffix}')
        if not report.is_file():
            raise FileNotFoundError(report)
        report_path = output / f'{prefix}_report'
        report_path.mkdir(exist_ok=True)
        shutil.copy2(report, report_path / 'model_report.html')
        for resource in resources:
            shutil.copy2(resource, output / resource.name)
    finally:
        rknn.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx_path', required=True, help='Exported ONNX directory')
    parser.add_argument('--output_path', required=True, help='Deployment output directory')
    parser.add_argument('--stage', choices=['vision', 'llm', 'all'], default='all')
    parser.add_argument('--platform', choices=['rk1820', 'rk1828', 'rk3572'],
                        default='rk1820',
                        help='Target platform (default: rk1820; '
                             'RK1820/RK1828: 8 cores, RK3572: 1 core)')
    args = parser.parse_args()
    source = Path(args.onnx_path).resolve()
    output = Path(args.output_path).resolve()
    stages = ('vision', 'llm') if args.stage == 'all' else (args.stage,)
    models = find_models(source, stages)
    if source == output:
        raise ValueError('ONNX and deployment output directories must differ')
    output.mkdir(parents=True, exist_ok=True)
    for stage, model_path in models.items():
        convert(stage, model_path, output, args.platform)


if __name__ == '__main__':
    main()
