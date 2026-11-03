# MiniCPM-V-4.7-1B 模型部署说明

模型地址：[openbmb/MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)

## 1. 部署环境

> ⚠️ 需安装最新版 rknn3-toolkit。
```bash
rknn3-toolkit>=1.1.0
torch==2.7.0
torchvision==0.22.0
transformers==5.18.0
numpy==2.5.2
```

## 2. 导出 RKNN 模型

### 2.1 导出 LLM ONNX 模型

```bash
cd examples/MiniCPM_V_4.7_1B/python/llm

# 生成 LLM 模型量化校准数据（可选）
python make_calidata.py --model_path openbmb/MiniCPM-V-4.7-1B

# 导出 LLM ONNX 模型 
python export_llm.py \
    --model_path openbmb/MiniCPM-V-4.7-1B \
    --export_llm_path ../../model/llm/MiniCPM-V-4.7-1B-llm.onnx

# 导出 LLM ONNX 模型 (GRQ 量化, 需要 CUDA 环境)
python export_llm.py --quant \
    --model_path openbmb/MiniCPM-V-4.7-1B \
    --export_llm_path ../../model/llm/MiniCPM-V-4.7-1B-llm.onnx

# 如需从 ModelScope 下载模型，增加 --modelscope 参数
python export_llm.py --modelscope
```

`make_calidata.py` 参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | HuggingFace 模型路径或名称 | `openbmb/MiniCPM-V-4.7-1B` |
| `--datapath` | 校准数据集 JSON 路径（含图片路径和文本输入） | `../../../../datasets/MMBench/llm/dataset.json` |
| `--export_datapath` | 生成的量化校准数据输出路径 | `./quant_data/model_inputs.json` |
| `--modelscope` | 从 ModelScope 下载模型 | `False` |

> ⚠️ `make_calidata.py` 会加载完整模型并捕获流入语言骨干 `language_model` 的多模态输入，需要较大内存。
> 生成的校准数据（`./quant_data/model_inputs.json` + `model_inputs/sample_N`）由 `export_llm.py --quant`
> 作为 GRQ 量化数据集消费（`--cali_dataset ./quant_data/model_inputs.json`），量化对象为 `language_model` 子模块。

执行 `export_llm.py` 后，会在 ONNX 同目录下同步生成以下文件：

```
MiniCPM-V-4.7-1B-llm.onnx
MiniCPM-V-4.7-1B-llm.config.pkl
MiniCPM-V-4.7-1B-llm.tokenizer.gguf
MiniCPM-V-4.7-1B-llm.embed.bin
```

### 2.2 导出 LLM RKNN 模型

```bash
python export_rknn.py \
    --onnx_path ../../model/llm/MiniCPM-V-4.7-1B-llm.onnx \
    --config ../../model/llm/MiniCPM-V-4.7-1B-llm.config.pkl \
    --rknn_path ../../model/llm/MiniCPM-V-4.7-1B-llm.rknn \
    --platform rk1820

# 如需重新导出（仅修改 profile_mode 或 kvcache 相关参数），可使用 --rebuild 跳过 ONNX 加载和图优化等步骤，加速模型导出：
python export_rknn.py \
    --onnx_path ../../model/llm/MiniCPM-V-4.7-1B-llm.onnx \
    --config ../../model/llm/MiniCPM-V-4.7-1B-llm.config.pkl \
    --rknn_path ../../model/llm/MiniCPM-V-4.7-1B-llm.rknn \
    --platform rk1820 \
    --rebuild
```

> ⚠️ **`--rebuild` 说明**：当前仅支持重置 `profile_mode` 和 `llm_config` 中 kvcache 相关参数（如 `kvcache_buffer_len`或者`max_position_embeddings` 等）。其他参数变更需走完整导出流程。

### 2.3 导出 Vision ONNX 模型

```bash
cd examples/MiniCPM_V_4.7_1B/python/vision

# 生成视觉量化校准数据（可选）
python make_calidata.py --model_path /path/to/MiniCPM-V-4.7-1B \
    --datapath datasets/MMBench/llm/dataset.json

# 导出 Vision ONNX（448x448 图像输入）
python export_vision.py \
    --model_path openbmb/MiniCPM-V-4.7-1B \
    --export_vision_path ../../model/vision/MiniCPM-V-4.7-1B-vision.onnx

# 导出 Vision ONNX（GRQ 量化, 需要 CUDA 环境）
python export_vision.py --quant \
    --model_path openbmb/MiniCPM-V-4.7-1B \
    --export_vision_path ../../model/vision/MiniCPM-V-4.7-1B-vision.onnx
```

> Vision 模型本身是固定 448×448（32×32 patch）的**单 crop 编码器**；
> 大图的切片由 C++ 侧多次调用该模型实现，不需要重新导出模型，见 §4。

### 2.4 导出 Vision RKNN 模型

```bash
python export_rknn.py \
    --onnx_path ../../model/vision/MiniCPM-V-4.7-1B-vision.onnx \
    --rknn_path ../../model/vision/MiniCPM-V-4.7-1B-vision.rknn \
    --platform rk1820
```
## 3. C++ 板端部署

### 3.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d MiniCPM_V_4.7_1B
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_MiniCPM_V_4.7_1B_demo/` 目录：

```
rknn_MiniCPM_V_4.7_1B_demo/
├── lib
│   ├── librga.so
│   └── librknn3_api.so
├── model
│   ├── MiniCPM-V-4.7-1B-llm.embed.bin
│   ├── MiniCPM-V-4.7-1B-llm.rknn
│   ├── MiniCPM-V-4.7-1B-llm.tokenizer.gguf
│   ├── MiniCPM-V-4.7-1B-llm.weight
│   ├── MiniCPM-V-4.7-1B-vision.rknn
│   ├── MiniCPM-V-4.7-1B-vision.weight
│   └── demo.jpg
└── rknn_minicpmv4_7_demo
```

### 3.2 部署到开发板

```bash
# 推送 demo 目录
adb push rknn_MiniCPM_V_4.7_1B_demo /data/

# 推送运行库
adb push rknn_MiniCPM_V_4.7_1B_demo/lib/* /usr/lib/
```

### 3.3 运行示例

```bash
./rknn_minicpmv4_7_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt>
```

**图文多模态模式**（vision 模型路径传实际路径）：

```bash
export LD_LIBRARY_PATH=./lib

./rknn_minicpmv4_7_demo \
    model/MiniCPM-V-4.7-1B-vision.rknn model/MiniCPM-V-4.7-1B-vision.weight \
    model/MiniCPM-V-4.7-1B-llm.rknn model/MiniCPM-V-4.7-1B-llm.weight \
    model/MiniCPM-V-4.7-1B-llm.tokenizer.gguf model/MiniCPM-V-4.7-1B-llm.embed.bin \
    0xff 0xff \
    model/demo.jpg \
    "描述一下这张图片"
```

**纯文本模式**（vision 模型路径传空串 `""` 即可自动降级，仅初始化 LLM）：

```bash
./rknn_minicpmv4_7_demo \
    "" "" \
    model/MiniCPM-V-4.7-1B-llm.rknn model/MiniCPM-V-4.7-1B-llm.weight \
    model/MiniCPM-V-4.7-1B-llm.tokenizer.gguf model/MiniCPM-V-4.7-1B-llm.embed.bin \
    0xff 0xff \
    "" \
    "解释相对论"
```

参数说明：

| 参数              | 说明                                                    |
| ----------------- | ------------------------------------------------------- |
| vision_model_path | Vision RKNN 模型路径；传空串 `""` 则退化为纯文本推理    |
| vision_weight_path| Vision 权重文件路径（纯文本模式下设置为`""`）                 |
| llm_model_path    | LLM RKNN 模型路径                                       |
| llm_weight_path   | LLM RKNN 权重文件路径                                   |
| tokenizer_path    | tokenizer 文件路径（.tokenizer.gguf）                   |
| embedding_path    | embedding 权重文件路径（.embed.bin）                    |
| vision_core_mask  | 视觉模型 NPU 核心掩码，按 16 进制填写，例如 0xff         |
| llm_core_mask     | LLM 模型 NPU 核心掩码，按 16 进制填写，例如 0xff         |
| image_path        | 输入图片路径（纯文本模式下设置为`""`）                         |
| prompt            | 用户输入文本，包含空格时需用英文双引号包裹               |

> ⚠️ demo 默认使用 `top_k=1`、`top_p=0.9`、`temperature=1.0`、`repeat_penalty=1.2`。

## 4. 切片（slice）支持

C++ demo 已支持与模型源码一致的切片流程：一张大图会先拆成 **1 张缩略图 + rows×cols 张切片**，
各 crop 分别编码后按顺序拼进多模态占位符，位置编码使用 canvas M-RoPE（画布坐标系）。

### 4.1 行为说明

**视觉模型不变。** 导出的 vision rknn 仍是"单 crop 448×448 → 64 token"的编码器；
切片由 C++ 侧**逐 crop 调用同一个模型**实现。因为 crop 之间在 ViT 内部完全独立
（注意力按 `cu_seqlens` 分块，互不可见），且每个 crop 都被缩放到同一尺寸（448×448），
所以逐 crop 调用与一次性 batch 等价——**crop 数变化不需要改模型，也不需要动态 shape**。

**位置编码（canvas M-RoPE）**：整张图共用一个"时间"位 T，所有 token 落在同一块
`rows*8 × cols*8` 的画布上（每个 crop 在 LLM 侧是 8×8）：

| token | T | H | W |
|---|---|---|---|
| `<image>` | base | base−1 | base−1 |
| 缩略图 token | base | `linspace` 拉伸铺满画布 | 同左 |
| `</image>` | base | base+canvas_H | base+canvas_W |
| `<slice>` / `</slice>` | base | 切片左上 / 右下角的行 | 切片左上 / 右下角的列 |
| 行间 `\n` | base | 该行最后一行 | 行右外侧 |
| 图之后的文本 | 从 `base + max(canvas_H, canvas_W) + 1` 继续 | 同左 | 同左 |

缩略图是整块画布的低清全局视图（所以用 `linspace` 拉伸），切片是画布上各格的高清局部，
两者共享同一套坐标。**embeds 注入**：缩略图由 `image.*` 模板写入，切片由 `embed_callback`
按 `<slice>`/`</slice>` 标记逐 token 贴入视觉特征。

### 4.2 用法

命令行与之前完全一致，图片面积超过 `448²` 时会自动切片：

```bash
./rknn_minicpmv4_7_demo \
    model/MiniCPM-V-4.7-1B-vision.rknn model/MiniCPM-V-4.7-1B-vision.weight \
    model/MiniCPM-V-4.7-1B-llm.rknn     model/MiniCPM-V-4.7-1B-llm.weight \
    model/MiniCPM-V-4.7-1B-llm.tokenizer.gguf model/MiniCPM-V-4.7-1B-llm.embed.bin \
    0xff 0xff \
    model/demo.jpg \
    "描述一下这张图片"
```

强制单切片（省算力）：

```bash
export MINICPMV4_7_NO_SLICE=1
./rknn_minicpmv4_7_demo \
    model/MiniCPM-V-4.7-1B-vision.rknn model/MiniCPM-V-4.7-1B-vision.weight \
    model/MiniCPM-V-4.7-1B-llm.rknn     model/MiniCPM-V-4.7-1B-llm.weight \
    model/MiniCPM-V-4.7-1B-llm.tokenizer.gguf model/MiniCPM-V-4.7-1B-llm.embed.bin \
    0xff 0xff \
    model/demo.jpg \
    "描述一下这张图片"
```

纯文本模式（vision 路径传空串）不受影响。
