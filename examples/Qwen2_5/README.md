# Qwen2.5 模型部署说明

模型地址：[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

## 1. 导出 RKNN 模型

### 1.1 导出 LLM ONNX 模型

```bash
cd examples/Qwen2_5/python

# 默认导出 Qwen2.5-3B-Instruct
python export_llm.py \
    --model_path Qwen/Qwen2.5-3B-Instruct \
    --export_llm_path ../model/llm/Qwen2.5-3B-Instruct.onnx

# 如需从 ModelScope 下载模型，增加 --modelscope 参数
python export_llm.py --modelscope
```

执行 `export_llm.py` 后，会在 ONNX 同目录下同步生成以下文件：

```
Qwen2.5-3B-Instruct.onnx
Qwen2.5-3B-Instruct.config.pkl
Qwen2.5-3B-Instruct.tokenizer.gguf
Qwen2.5-3B-Instruct.embed.bin
```

> `--quant` 参数可启用 GRQ 量化（需要 CUDA 环境），校准数据默认使用 `datasets/llm_quant.json`，可通过 `--cali_dataset` 指定。

### 1.2 导出 LLM RKNN 模型

```bash
python export_rknn.py \
    --onnx_path ../model/llm/Qwen2.5-3B-Instruct.onnx \
    --config ../model/llm/Qwen2.5-3B-Instruct.config.pkl \
    --rknn_path ../model/llm/Qwen2.5-3B-Instruct.rknn \
    --platform rk1820
```

RKNN 转换默认使用 W4A16 量化，采用 group32 方式（`do_quantization=True` 时 toolkit 会自动进行量化，无需提供量化数据集）：

```python
rknn.config(target_platform=args.platform,
            quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32')
```

> ⚠️ `--platform` 为必填参数，目前可选 `rk1820`。

## 2. C++ 部署说明

C++ 推理侧需使用 RKNN 模型、权重文件、tokenizer 文件和 embedding 权重文件进行部署。

### 2.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_Qwen2_5_demo/` 目录。

### 2.2 运行示例

```bash
./rknn_qwen2_5_demo \
    <model_path> <weight_path> \
    <tokenizer_path> <embedding_path> \
    <core_mask> <prompt>
```

以 **Qwen2.5-3B-Instruct** 为例：

```bash
./rknn_qwen2_5_demo \
    model/Qwen2.5-3B-Instruct.rknn \
    model/Qwen2.5-3B-Instruct.weight \
    model/Qwen2.5-3B-Instruct.tokenizer.gguf \
    model/Qwen2.5-3B-Instruct.embed.bin \
    0xff \
    "解释相对论"
```

参数说明：

| 参数            | 说明                                          |
| --------------- | --------------------------------------------- |
| model_path      | LLM RKNN 模型路径                             |
| weight_path     | LLM RKNN 权重文件路径                         |
| tokenizer_path  | tokenizer 文件路径（.tokenizer.gguf）         |
| embedding_path  | embedding 权重文件路径（.embed.bin）          |
| core_mask       | NPU 核心掩码，按 16 进制填写，例如 0x1、0xff  |
| prompt          | 用户输入文本，包含空格时需用英文双引号包裹    |

> ⚠️ demo 默认使用 `top_k=1`、`top_p=0.9`、`temperature=1.0`、`repeat_penalty=1.2`。
