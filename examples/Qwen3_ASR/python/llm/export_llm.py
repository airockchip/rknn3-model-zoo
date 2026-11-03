import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import causal_llm_to_onnx, update_config, export_tokenizer, export_llm_config, export_embed_weight
from py_utils.tools import clear_llm_external_weight_in_dir, gen_quantize_dataset
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import Cache
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from qwen_asr import Qwen3ASRModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    MoeCausalLMOutputWithPast,
)

class Qwen3ASRThinkerCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    r"""
    Args:
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    rope_deltas: Optional[torch.LongTensor] = None

class Qwen3TextOnlyCausalLM(nn.Module):
    def __init__(self, text_model, lm_head, config):
        super().__init__()
        self.model = text_model
        self.lm_head = lm_head
        self.config = config
        self.rope_deltas = 0

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None
    ):
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        loss = None

        # 保持与原输出类同字段，方便上层代码兼容
        return Qwen3ASRThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
        )

prompt = "RKLLM"
chat_context = {
    "messages":[
        {"role": "user", "content": prompt}
    ],
    "add_generation_prompt": True,
    "enable_thinking": True,
}

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Qwen/Qwen3 llm configuration and onnx model for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False, default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False, default="llm.onnx")
    parser.add_argument("--modelscope", action='store_true', help="Whether download model from www.modelscope.cn")
    parser.add_argument("--quant", action='store_true', help="Whether use GRQ quantization")
    parser.add_argument("--cali_dataset", default='quant_data/model_input.json', help="some samples for grq quantized_algorithm")
    args = parser.parse_args()

    if args.modelscope:
        from modelscope import snapshot_download
        args.model_path = snapshot_download(args.model_path)

    qwen3_asr_model = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cpu",
        max_inference_batch_size=1,
        max_new_tokens=128,
        attn_implementation="eager",
    )
    qwen3_asr_model.model.thinker.model.config._attn_implementation = "eager"
    kwargs = {
        'trust_remote_code': True,
    }
    config = AutoConfig.from_pretrained(args.model_path, **kwargs)
    config = config.thinker_config
    
    thinker = qwen3_asr_model.model.thinker
    text_model = thinker.model
    lm_head = thinker.lm_head
    
    if args.quant and torch.cuda.is_available():
        from rknn.quantization.api import RKQuantizer
        ## 初始化量化工具
        QuantTool = RKQuantizer(verbose=True)
        
        ## 量化工具加载模型
        ret = QuantTool.load_model(model=text_model, tokenizer=None, device='cuda')
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        
        ## 执行量化算法
        dataset = args.cali_dataset
        text_model = QuantTool.quantize(quantized_dtype="w4a16", quantized_method="group32", quantized_algorithm="grq", dataset=dataset)

        text_model = text_model.cpu()
        
    model = Qwen3TextOnlyCausalLM(text_model, lm_head, config)
    model.eval()
    
    # Export llm to onnx
    args.hidden_size = config.text_config.hidden_size
    args.arch = "Qwen3-ASR"
    causal_llm_to_onnx(model, args)

    # Export LLM configuration 
    llm_config = {
    'vocab_size' : config.text_config.vocab_size,
    'hidden_size': config.text_config.hidden_size,
    }
    config_path = os.path.splitext(args.export_llm_path)[0] + '.config.pkl'
    with open(config_path, "wb") as f:
        pickle.dump(llm_config, f)
    print(f"config_path exported to {config_path}")

    # Export tokenizer
    # Using Qwen3 tokenizer (Special tokens should be added manually in the C demo. Please refer to C demo for details). 
    qwen3_model_path = "Qwen/Qwen3-0.6B"
    if args.modelscope:
        from modelscope import snapshot_download
        qwen3_model_path = snapshot_download(qwen3_model_path)
    # qwen3_model_path = '/mnt/nfs_client/public/data/CKPT/Qwen3-0.6B/'
    export_tokenizer(qwen3_model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    # Export embedding weight
    export_embed_weight(qwen3_asr_model.model.thinker.model.embed_tokens.weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

