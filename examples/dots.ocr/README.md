# dots.ocr 模型部署说明

dots.ocr 是一个基于 Qwen2 的多模态视觉语言模型，专注于文档版面解析和 OCR 任务。本示例演示如何将 dots.ocr 模型转换为 RKNN 格式并在 RK3588 等设备上部署。

- 原始模型：<https://huggingface.co/rednote-hilab/dots.ocr>
- 官方代码：<https://github.com/rednote-hilab/dots.ocr>

## 1. 模型特点

dots.ocr 与 Qwen2.5-VL 等常见 VLM 有关键差异，部署时需特别注意：

| 特性 | dots.ocr | Qwen2.5-VL |
|------|----------|------------|
| temporal_patch_size | 1 | 2 |
| Patch Embedding | Conv2d | Conv3d |
| RoPE 类型 | 1D RoPE | MRoPE |
| Window Attention | 无 | 有 |
| Vision 模块名 | `vision_tower` | `visual` |
| flatten 维度 | 588 (3×1×14×14) | 1176 (3×2×14×14) |
| chat_template 标记 | `<\|user\|>`/`<\|endofuser\|>`/`<\|assistant\|>` | `<\|im_start\|>`/`<\|im_end\|>` |
| EOS token | 151643, 151673 | 151645 |

### chat_template 格式

dots.ocr 使用自定义 chat_template（非 Qwen 标准格式）：

```
system:    "<|system|>" + content + "<|endofsystem|>\n"
user:      "<|user|>" + content + "<|endofuser|>"
assistant: "<|assistant|>" + content
```

C++ 部署时对应的 `rknn3_session_set_chat_template` 参数为：
- `system_prompt` = `""`
- `prompt_prefix` = `"<|user|>"`
- `prompt_postfix` = `"<|endofuser|><|assistant|>"`

## 2. 模型裁剪策略

### 2.1 Vision 模型裁剪

将部分算子迁移至主控设备 CPU 上运行（prune 模式），减少 NPU 内存占用。默认开启。

### 2.2 LLM 模型裁剪

将 LM Head 独立出来在主控设备上单独运行，此为可选模式。默认开启，导出时可添加 `--no_prune_mode` 参数关闭。

### 2.3 完整模型模式（无裁剪）

RK1828 等内存较大的设备可直接使用完整模型，导出时添加 `--no_prune_mode` 参数：

```bash
python export_rknn.py --no_prune_mode --platform rk1820
```

## 3. 模型导出

### 3.1 模型下载

- modelscope 下载

```
modelscope download --model 'dots-studio/dots.ocr' --local_dir  /path/to/dots.ocr
```

- huggingface 下载

```
hf download rednote-hilab/dots.ocr --local-dir /path/to/dots.ocr
```


### 3.2 环境准备

```bash
pip install -r ../../requirements.txt
pip install -r requirements.txt
```

依次安装仓库通用依赖、RKNN3 Toolkit 和本示例的增量依赖。本示例的 `requirements.txt` 包含 dots.ocr 模型加载、ONNX 导出和量化校准图像预处理所需的额外依赖及版本约束。

将原始模型放置在 `/path/to/dots.ocr` 目录下。

> **注意**：`tmp/` 目录存放原始模型、校准数据等中间产物，不应提交到仓库。

> **`data/` 目录（应提交内容）**：本示例随仓库提交以下测试数据，保证文件夹可独立完成端到端部署：
> - `data/demo.jpg`：板端运行 demo 的测试图片（示例命令中的 `demo.jpg`）
> - `data/vision/datasets_npy.txt` 与 `data/vision/npy/*.npy`：Vision 模型量化校准数据（共 20 张），`export_rknn.py` 在校准文件缺失时会自动回退使用 `../../data/vision/datasets_npy.txt`
>
> 注意：`.gitignore` 忽略了 `data/` 目录，提交时需要 `git add -f data/` 强制添加。

### 3.3 导出 Vision 模型

```bash
cd python/vision

# 导出 ONNX 模型
python export_vision.py \
    --model_path /path/to/dots.ocr \
    --export_vision_path ../../model/vision/dots-ocr-vision.onnx \
    --img_h 448 \
    --img_w 448

# 导出 RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/vision/dots-ocr-vision.onnx \
    --rknn_path ../../model/vision/dots-ocr-vision.rknn \
    --platform rk1820
```

### 3.4 导出 LLM 模型

```bash
cd python/llm

# 生成量化校准数据
python make_calidata.py --model_path /path/to/dots.ocr

# 导出 ONNX 模型
python export_llm.py \
    --model_path /path/to/dots.ocr \
    --export_llm_path ../../model/llm/dots-ocr-llm.onnx

# 导出 RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/llm/dots-ocr-llm.onnx \
    --config ../../model/llm/dots-ocr-llm.config.pkl \
    --rknn_path ../../model/llm/dots-ocr-llm.rknn \
    --platform rk1820
```

> **说明**：dots.ocr 的 `DotsOCRForCausalLM.forward` 参数顺序非标准，导出时需替换为 `Qwen2ForCausalLM.forward` 绕过 vision 部分。`convert_hf_to_gguf.py` 已原生支持 `DotsOCRForCausalLM` 架构（按 Qwen2 处理并跳过 vision tower 权重），直接从原始模型目录导出 tokenizer，不修改原始模型目录。

### 3.5 导出参数说明

**Vision `export_vision.py`**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 原始模型路径 | `/path/to/dots.ocr` |
| `--export_vision_path` | ONNX 输出路径 | `../../model/vision/dots-ocr-vision.onnx` |
| `--img_h` | 输入图像高度（28 的倍数） | 448 |
| `--img_w` | 输入图像宽度（28 的倍数） | 448 |
| `--quant` | 启用 GRQ 量化 | False |

**LLM `export_llm.py`**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 原始模型路径 | `/path/to/dots.ocr` |
| `--export_llm_path` | ONNX 输出路径 | `../../model/llm/dots-ocr-llm.onnx` |
| `--quant` | 启用 GRQ 量化 | False |

**Vision `export_rknn.py`**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--onnx_path` | ONNX 输入路径 | `../../model/vision/dots-ocr-vision.onnx` |
| `--rknn_path` | RKNN 输出路径 | `../../model/vision/dots-ocr-vision.rknn` |
| `--platform` | 目标平台 | `rk1820` |
| `--dataset_path` | 量化校准图片列表 | `../../../../datasets/MMBench/vision/datasets.txt`；若该文件不存在，脚本会自动生成并回退使用随仓库提交的 `../../data/vision/datasets_npy.txt` |
| `--core_num` | NPU 核数 | 8 |
| `--no_prune_mode` | 关闭裁剪模式 | False |

**`make_calidata.py`**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型路径 | `/path/to/dots.ocr` |
| `--datapath` | 校准数据集 JSON 路径 | `../../../../datasets/MMBench/llm/dataset.json` |
| `--export_datapath` | 校准数据输出路径 | `./quant_data/model_inputs.json` |

> **说明**：dots.ocr 的 `modeling_dots_vision.py` 自带 flash_attn 降级逻辑，`make_calidata.py` 可在无 GPU 的 CPU 环境直接运行；请确认已安装 `torch`、`transformers`、`qwen-vl-utils`、`tqdm`。

## 4. Vision 模型分辨率调整

可通过 `--img_h` 和 `--img_w` 参数调整输入分辨率（必须为 28 的倍数）：

```bash
python export_vision.py --img_h 448 --img_w 448
```

> **注意**：
> - 分辨率越大，内存占用越高，会影响 LLM 的最大上下文长度
> - 部分分辨率可能与 RKNN 推理框架不兼容，如遇报错请联系 RKNPU 团队

若修改了分辨率，需同步修改 `cpp/vision/rknn_dots_ocr_vision.h` 中的参数：

```cpp
#define MODEL_WIDTH  <your_width>
#define MODEL_HEIGHT <your_height>
```

## 5. C++ 部署说明

### 5.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d dots.ocr
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_dots.ocr_demo/` 目录。

### 5.2 运行

```bash
./rknn_dots_ocr_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt> \
    [model_width model_height]
```

参数说明：

| 参数 | 说明 |
|------|------|
| `vision_model_path` | Vision RKNN 模型路径 |
| `vision_weight_path` | Vision 权重文件路径 |
| `llm_model_path` | LLM RKNN 模型路径 |
| `llm_weight_path` | LLM 权重文件路径 |
| `tokenizer_path` | Tokenizer GGUF 文件路径 |
| `embedding_path` | Embedding BIN 文件路径 |
| `vision_core_mask` | Vision NPU 核心掩码（如 `0xff`） |
| `llm_core_mask` | LLM NPU 核心掩码（如 `0xff`） |
| `image_path` | 输入图像路径 |
| `prompt` | 文本提示词（如 `"OCR."`） |
| `model_width` | （可选）模型输入宽度，默认 448 |
| `model_height` | （可选）模型输入高度，默认 448 |

### 5.3 示例

```bash
./rknn_dots_ocr_demo \
    model/dots-ocr-vision.rknn model/dots-ocr-vision.weight \
    model/dots-ocr-llm.rknn model/dots-ocr-llm.weight \
    model/dots-ocr-llm.tokenizer.gguf model/dots-ocr-llm.embed.bin \
    0xff 0xff \
    demo.jpg "OCR."
```

> 示例中的 `demo.jpg` 取自本示例随仓库提交的 `data/demo.jpg`，运行前需将该文件复制/推送到 demo 所在目录（或直接传绝对路径）。

## 6. 模型结构

dots.ocr 采用双模型架构：

- **Vision 模型**（`dots-ocr-vision.rknn`）：基于 DotsVisionTransformer，42 层 Transformer blocks，使用 Conv2d patch embedding 和 1D RoPE，输出 1536 维图像嵌入。
- **LLM 模型**（`dots-ocr-llm.rknn`）：基于 Qwen2ForCausalLM，28 层，hidden_size=1536，使用标准 1D RoPE（非 MRoPE），w4a16 量化。

两个模型通过 NPU 内部内存共享机制（`rknn3_set_internal_mem`）高效传递图像嵌入向量。

## 7. 常见问题

### Vision 导出 bf16 类型不匹配

dots.ocr 的 `DotsVisionTransformer.forward` 默认 `bf16=True`，会将输入转为 bfloat16。在 CPU 环境下导出时会报类型不匹配错误。导出脚本已通过 `model.float()` 和 `bf16=False` 参数处理此问题。

### Tokenizer 转换失败

`convert_hf_to_gguf.py` 已原生支持 `DotsOCRForCausalLM` 架构（按 Qwen2 处理并跳过 vision tower 权重）。若仍遇到转换失败，请确认 `tokenizer/thirdparty/llama_vocab/` 目录完整，以及模型目录中的 `tokenizer_config.json` 与 `tokenizer.json` 存在且未被损坏。

### LLM 输出为空

请检查以下配置是否正确：

1. **EOS token**：必须使用 dots.ocr 的 `generation_config.json` 中指定的 `[151643, 151673]`，而非 tokenizer 返回的全部 EOG token（包含 `<|im_end|>` 151645 等，会导致过早停止生成）。
2. **chat_template 标记**：C++ 代码中的 `prompt_prefix`/`prompt_postfix` 必须使用 `<|user|>`/`<|endofuser|><|assistant|>`，不能使用其他字符。
3. **NPU 核心掩码**：RK3588 有 8 个 NPU 核心，需使用 `0xff` 而非 `0x1`/`0x6`。
