import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
import torch
import importlib.util

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_qwen_3_5_llm_to_onnx, export_tokenizer, export_llm_config, export_embed_weight
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

    parser = ArgumentParser(description="Export openbmb/MiniCPM-V-4.7-1B llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="./openbmb/MiniCPM-V-4.7-1B")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="../../model/llm/MiniCPM-V-4.7-1B-llm.onnx")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='./quant_data/model_inputs.json', help="some samples for grq quantized_algorithm")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    fakeq_state = None
    if args.quant and torch.cuda.is_available():
        ## 量化用原生模型（含 fla kernel 的版本），量化完成后再用导出版模型加载权重
        from rknn.quantization.api import RKQuantizer
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager",
            trust_remote_code=True,
        )

        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)

        ## 量化工具加载模型
        ret = QuantTool.load_model(model=model.model.language_model, tokenizer=None, device='cuda',
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
                                         auto_hybrid_rate=0.)

        quant_model = quant_model.cpu()
        fakeq_state = quant_model.state_dict()
        ## 子模块量化返回的 state_dict 缺少 'model.' 前缀，补上前缀以便后续 load_state_dict(strict=False) 正确写回完整 VLM 的语言骨干
        fakeq_state = {f"model.{k}": v for k, v in fakeq_state.items()}
        del quant_model

    # 动态加载自定义的 modeling_qwen3_5_export 模块（qwen3_5_text 架构的导出版：
    # 关闭 fla/causal_conv1d kernel，linear attention 走 rknn::exLinearAttention 自定义算子）
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

    class MiniCPMV4_7_export_model_cst(torch.nn.Module):
        """只导出纯文本 LLM 部分：language_model（qwen3_5_text）+ lm_head。
           linear_attention层conv_state作为图输入/输出，full attention 层由内部管理 KV cache。"""
        def __init__(self, language_model, lm_head):
            super().__init__()
            self.model = language_model
            self.lm_head = lm_head
            self.config = language_model.config

        def forward(self, input_ids, position_ids, attention_mask, Tc, logits_to_keep, conv_status, recurrent_status=None, key_cache=None, value_cache=None):
            past_key_values = pack_dynamic_cache(conv_status, recurrent_status, key_cache, value_cache, self.model.config)
            model_out = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                Tc=Tc,
                logits_to_keep=logits_to_keep,
                use_cache=True,
            )
            hidden_states = model_out.last_hidden_state
            logits = self.lm_head(hidden_states[:, logits_to_keep, :])

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
        'attn_implementation': 'eager',
        'trust_remote_code': True,
    }
    # 直接用注入的 export 版 modeling 类加载模型，避免 AutoModelForCausalLM 复用 quant 阶段
    # 缓存的原生 Qwen3_5ForCausalLM（其 linear attn 走 fla triton kernel，在 CPU 上无法导出 onnx）
    model = modeling_module.Qwen3_5ForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32, **kwargs)

    if fakeq_state is not None:
        model.load_state_dict(
            fakeq_state,
            strict=False
        )

    export_dirname = os.path.dirname(args.export_llm_path)
    if not os.path.exists(export_dirname):
        os.makedirs(export_dirname)

    exported_model = MiniCPMV4_7_export_model_cst(model.model, model.lm_head)

    # Add tokenizer to args
    args.tokenizer = tokenizer

    # Add chat_context to args
    args.chat_context = [chat_context] # 注意：apply_chat_template 通常需要列表格式

    # Export llm to onnx
    causal_qwen_3_5_llm_to_onnx(exported_model, args)

    if args.quant and torch.cuda.is_available(): ##开启混合量化需要执行下述export_op_quantized_dtype
        QuantTool.export_op_quantized_dtype(args.export_llm_path, op_dtype_path='layer_bit.json')

    # Export LLM configuration
    config_pkl_path = os.path.splitext(args.export_llm_path)[0] + '.config.pkl'
    user_config = {'model_type': "qwen3_5_text", 'spatial_merge_size': 4}
    export_llm_config(args.model_path, config_pkl_path, chat_context, prompt, user_config=user_config)

    # Export tokenizer
    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(model.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')
