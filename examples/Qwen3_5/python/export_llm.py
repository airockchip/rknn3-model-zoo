import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
import torch
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from py_utils.export_llm_helper import causal_qwen_3_5_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


## 避免flashatten报错
try:
    import transformers.modeling_flash_attention_utils as _fa_utils
    _orig_is_packed_sequence = _fa_utils._is_packed_sequence
    def _is_packed_sequence(position_ids, batch_size):
        return False
    _fa_utils._is_packed_sequence = _is_packed_sequence
except:
    pass

prompt = "RKLLM"
chat_context = {"role": "user", "content": prompt}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen3_5 llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='../../../datasets/llm_quant.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../model/llm/Qwen3.5-0.8B-llm.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()
    
    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    fakeq_state = None
    if args.quant and torch.cuda.is_available():
        ATTN_IMPL = None
        # Try to import FlashAttention2
        try:
            ## 使用flashattention可以极大减少grq运行显存占用，请自行安装，注意flashattention要求模型加载类型为bf16/fp16，建议使用bf16。
            import flash_attn
            ATTN_IMPL = "flash_attention_2"
            print("Use FlashAttention2")
        except Exception as e:
            print(
                f"FlashAttention2 is unavailable ({e}), "
                "falling back to standard attention."
                "It is recommended to install FlashAttention2, as standard attention requires a large amount of video memory."
        )
        from rknn.quantization.api import RKQuantizer
        model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                _attn_implementation=ATTN_IMPL if ATTN_IMPL is not None else "eager", ## 开启flashattention可以减少grq显存占用，请自行安装
                trust_remote_code=True
            )
        
        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model, tokenizer=tokenizer, device='cuda',
                                   system_prompt=None, tools=None, system_role='system', user_role='user')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        quant_model = QuantTool.quantize(quantized_dtype="w4a16",
                              quantized_method="group32",
                              quantized_algorithm='grq',
                              dataset=dataset,
                            #   layer_quant_config="layer_quant_config.json", ##指定量化位宽
                              auto_hybrid_rate=0.) ##auto_hybrid_rate>0会开启自动量化位宽分配

        quant_model = quant_model.cpu()
        fakeq_state = quant_model.state_dict()
        del quant_model

                
    
    # 动态加载自定义的 modeling_qwen3_5_export 模块
    custom_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    modeling_spec = importlib.util.spec_from_file_location(
        'transformers.models.qwen3_5.modeling_qwen3_5',
        os.path.join(custom_path, 'modeling_qwen3_5_export.py')
    )
    modeling_module = importlib.util.module_from_spec(modeling_spec)
    sys.modules['transformers.models.qwen3_5.modeling_qwen3_5'] = modeling_module
    modeling_spec.loader.exec_module(modeling_module)

    def pack_dynamic_cache(conv_states, recurrent_states, key_cache, value_cache, model_config):
        past_key_values = modeling_module.Qwen3_5DynamicCache(model_config)
        assert conv_states.shape[0] == past_key_values.layer_types.count('linear_attention'), "conv_states mismatch"
        
        conv_states = list(torch.split(conv_states, dim=0, split_size_or_sections=1))
        
        for i, layer_types in enumerate(past_key_values.layer_types):
            if layer_types == 'linear_attention':
                past_key_values.conv_states[i] = conv_states.pop(0)
        return past_key_values

    class Qwen3_5_export_model_cst(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.config = model.config

        def forward(self, input_ids, position_ids, attention_mask, Tc, logits_to_keep, conv_status, recurrent_status=None, key_cache=None, value_cache=None):
            past_key_values = pack_dynamic_cache(conv_status, recurrent_status, key_cache, value_cache, self.model.config)
            model_out = self.model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, past_key_values=past_key_values, Tc=Tc, logits_to_keep=logits_to_keep)
            logits = model_out.logits
            conv_status = model_out.past_key_values.conv_states
            recurrent_status = model_out.past_key_values.recurrent_states
            key_cache = model_out.past_key_values.key_cache
            value_cache = model_out.past_key_values.value_cache
            
            conv_status = [v for v in conv_status if v is not None]
            
            recurrent_status = None
            key_cache = None
            value_cache = None
            
            conv_status = torch.concat(conv_status, dim=0)
            
            return logits, conv_status, recurrent_status, key_cache, value_cache

    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    # 直接用注入的 export 版 modeling 类加载模型，避免 AutoModelForCausalLM 复用 quant 阶段
    # 缓存的原生 Qwen3_5ForCausalLM（其 linear attn 走 fla triton kernel，在 CPU 上无法导出 onnx）
    model = modeling_module.Qwen3_5ForCausalLM.from_pretrained(args.model_path, torch_dtype="float32", device_map="cpu", attn_implementation="eager")

    if fakeq_state is not None:
        model.load_state_dict(
                fakeq_state,
                strict=False
            )

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
        os.makedirs(export_dirname)
    
    exported_model = Qwen3_5_export_model_cst(model)

    # Add tokenizer to args
    args.tokenizer = tokenizer

    # Add chat_context to args
    args.chat_context = [chat_context] # 注意：apply_chat_template 通常需要列表格式
    
    # Export llm to onnx
    causal_qwen_3_5_llm_to_onnx(exported_model, args)
    
    if args.quant and torch.cuda.is_available(): ##开启混合量化需要执行下述export_op_quantized_dtype
        QuantTool.export_op_quantized_dtype(args.export_llm_path, op_dtype_path='layer_bit.json')

    # Export LLM configuration 
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl', chat_context, prompt)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.get_input_embeddings().weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')
