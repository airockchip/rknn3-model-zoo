import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
import sys
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from py_utils.export_llm_helper import export_llm_config, export_embed_weight, export_tokenizer
from transformers import AutoConfig

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import ChineseCLIPModel

SEQ_LEN = 352


def _get_vector_norm(tensor: torch.Tensor) -> torch.Tensor:
    square_tensor = torch.pow(tensor, 2)
    sum_tensor = torch.sum(square_tensor, dim=-1, keepdim=True)
    normed_tensor = torch.pow(sum_tensor, 0.5)
    return normed_tensor


class ChineseCLIPTextModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config.text_config

    def forward(self, input_embeds, attention_mask=None, position_ids=None):
        # cpp 端写的是 INT32 position_ids，ONNX 输入保持 INT32，模型内部再转 long
        if position_ids is not None:
            position_ids = position_ids.to(torch.long)

        text_outputs = self.model.text_model(
            inputs_embeds=input_embeds, attention_mask=attention_mask,
            position_ids=position_ids, return_dict=True
        )
        sequence_output = text_outputs[0]
        pooled = sequence_output[:, 0, :]  # [CLS] 池化

        text_embeds = self.model.text_projection(pooled)
        text_embeds = text_embeds / _get_vector_norm(text_embeds)
        return text_embeds


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Export Chinese-CLIP text model to ONNX for RKNN")
    parser.add_argument("--model_path", type=str, help="model path or name", required=False,
                       default="../../models/QA-CLIP-ViT-L-14")
    parser.add_argument("--export_llm_path", type=str, help="export llm onnx model path", required=False,
                       default="./chinese_clip_text.onnx")
    args = parser.parse_args()

    export_llm_dirname = os.path.dirname(args.export_llm_path)
    if export_llm_dirname:
        os.makedirs(export_llm_dirname, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

    model = ChineseCLIPModel.from_pretrained(args.model_path, torch_dtype=torch.float16, trust_remote_code=True, attn_implementation="eager")
    wrapped_model = ChineseCLIPTextModelWrapper(model)
    wrapped_model.eval()

    fake_input_embeds = torch.zeros(1, SEQ_LEN, config.text_config.hidden_size, dtype=torch.float16)
    fake_attention_mask = torch.ones(1, SEQ_LEN, dtype=torch.float16)
    fake_position_ids = torch.arange(0, SEQ_LEN, dtype=torch.int32).unsqueeze(0)

    print("Exporting Chinese-CLIP text model to ONNX...")
    torch.onnx.export(
        wrapped_model,
        (fake_input_embeds, fake_attention_mask, fake_position_ids),
        args.export_llm_path,
        input_names=['input_embeds', 'attention_mask', 'position_ids'],
        output_names=['output'],
        opset_version=17,
        dynamic_axes={
            'input_embeds': {1: 'seq'},
            'attention_mask': {1: 'seq'},
            'position_ids': {1: 'seq'},
        },
    )
    print("Export completed successfully!")

    # 导出运行时配套文件：config、tokenizer、embedding
    embedding_config = {
        "model_type": "chinese_clip_text",
        "hidden_size": config.text_config.hidden_size,
        "projection_dim": config.projection_dim,
        "max_position_embeddings": config.text_config.max_position_embeddings,
        "vocab_size": config.text_config.vocab_size,
    }
    user_config = {"task_type": 1}
    export_llm_config(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.config.pkl',
                     embedding_config, None, user_config)

    export_tokenizer(args.model_path, os.path.splitext(args.export_llm_path)[0] + '.tokenizer.gguf')

    embed_weight = wrapped_model.model.text_model.embeddings.word_embeddings.weight
    export_embed_weight(embed_weight, os.path.splitext(args.export_llm_path)[0] + '.embed.bin')

    print(f"Chinese-CLIP text model exported successfully to {args.export_llm_path}")
