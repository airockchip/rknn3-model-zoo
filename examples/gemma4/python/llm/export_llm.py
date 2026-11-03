import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_quantize_dataset
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from export_onnx import causal_gemma4_to_onnx

prompt = "RKLLM"
chat_context = {
    "messages":[
        {"role": "user", "content": prompt}
    ],
    "add_generation_prompt": True,
}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export google/gemma-4 llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="google/gemma-4-E2B-it")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/gemma-4-e2b-it.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='../../../../datasets/llm_quant.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if "E4B" in args.model_path:
        args.export_llm_path = "../../model/llm/gemma-4-e4b-it.onnx"

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    fakeq_state = None
    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer

        # gemma4-e4b 的 global（non-sliding）attention 层 head_dim=global_head_dim=512，
        # 而 flash-attn 2.x 内核硬限制 head_dim<=256，无法跑 global 层（会报"FlashAttention forward only supports head dimension at most 256"），
        # 因此无法使用flashattention。
        model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                trust_remote_code=True
            )

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.model.language_model, tokenizer=tokenizer, device='cuda',
                                   system_prompt=None, tools=None, system_role='system', user_role='user')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        model.model.language_model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm='grq', dataset=dataset)
        
        model = model.cpu()
        fakeq_state = model.state_dict()
        del model
        torch.cuda.empty_cache()


    # Get the custom module path
    custom_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gemma4'))
    print(f"Custom gemma4 module path: {custom_path}")

    # Load our custom modeling_gemma4 module using importlib
    import importlib.util
    modeling_spec = importlib.util.spec_from_file_location(
        'transformers.models.gemma4.modeling_gemma4',
        os.path.join(custom_path, 'modeling_gemma4.py')
    )
    modeling_module = importlib.util.module_from_spec(modeling_spec)
    sys.modules['transformers.models.gemma4.modeling_gemma4'] = modeling_module
    modeling_spec.loader.exec_module(modeling_module)
    print(f"Loaded custom modeling_gemma4 from: {modeling_module.__file__}")

    # NOTE: quant 阶段按设计用原生 transformers 源码加载模型，这会让 transformers 把
    # 原生 gemma4 类缓存到两处：(1) transformers.models.gemma4 包对象（__init__.py 执行
    # 了 `from .modeling_gemma4 import *`）；(2) _LazyAutoMapping._modules["gemma4"]。
    # 仅替换 sys.modules 子模块无效——因为自定义 Gemma4Model.__init__ 内部会用
    # AutoModel.from_config(...) 构造 language_model，仍走 _LazyAutoMapping 缓存拿到原生
    # Gemma4ForCausalLM，其 text model 的 embed_tokens 是原生 Gemma4TextScaledWordEmbedding
    # （没有 fuse_scale 方法），导致导出 onnx 时报 AttributeError。
    # 因此必须把自定义类写回包对象并清掉所有 _LazyAutoMapping 缓存，让顶层和内部
    # AutoModel.from_config 都解析到自定义类（embed_tokens 为 _RK 版本，带 fuse_scale）。
    import transformers.models.gemma4 as _gemma4_pkg
    for _name in getattr(modeling_module, '__all__', []):
        if hasattr(modeling_module, _name):
            setattr(_gemma4_pkg, _name, getattr(modeling_module, _name))
    import transformers.models.auto as _auto_pkg
    for _attr in dir(_auto_pkg):
        _mapping = getattr(_auto_pkg, _attr, None)
        if hasattr(_mapping, '_modules') and isinstance(getattr(_mapping, '_modules', None), dict):
            _mapping._modules.pop('gemma4', None)

    kwargs = {
        'attn_implementation': 'eager',
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    # update_config(config, ['use_cache'], False)
    kwargs['config'] = config
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    if fakeq_state is not None:
        model.load_state_dict(
                fakeq_state,
                strict=False
            )

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
            os.makedirs(export_dirname)

    # Export llm to onnx
    causal_gemma4_to_onnx(model, args)

    # # Export LLM configuration 
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.language_model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')
    per_layer_scale = config.get_text_config().hidden_size_per_layer_input ** 0.5
    export_embed_weight(model.model.language_model.embed_tokens_per_layer.weight*per_layer_scale, os.path.splitext(args.export_llm_path)[0] + '_per_layer_inputs.embed.bin')
