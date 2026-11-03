# dots.mocr 模型部署说明

dots.mocr 是一个基于 Qwen2 的多模态视觉语言模型，专注于多语言 OCR 任务。本示例演示如何将 dots.mocr 模型转换为 RKNN 格式并在 RK3588 等设备上部署。

- 原始模型：<https://huggingface.co/rednote-hilab/dots.mocr>
- 官方代码：<https://github.com/rednote-hilab/dots.mocr>

## 1. 模型特点

dots.mocr 与 Qwen2.5-VL 等常见 VLM 有关键差异，部署时需特别注意：

| 特性 | dots.mocr | Qwen2.5-VL |
|------|-----------|------------|
| temporal_patch_size | 1 | 2 |
| Patch Embedding | Conv2d | Conv3d |
| RoPE 类型 | 1D RoPE | MRoPE |
| Window Attention | 无 | 有 |
| Vision 模块名 | `vision_tower` | `visual` |
| flatten 维度 | 588 (3×1×14×14) | 1176 (3×2×14×14) |
| chat_template 标记 | `<\|user\|>`/`<\|endofuser\|>`/`<\|assistant\|>` | `<\|im_start\|>`/`<\|im_end\|>` |
| EOS token | 151643, 151672, 151673 | 151645 |

### chat_template 格式

dots.mocr 使用自定义 chat_template（非 Qwen 标准格式）：

```
system:    "<|system|>" + content + "<|endofsystem|>\n"
user:      "<|user|>" + content + "<|endofuser|>"
assistant: "<|assistant|>" + content
```

C++ 部署时对应的 `rknn3_session_set_chat_template` 参数为：
- `system_prompt` = `""`
- `prompt_prefix` = `"<|user|>"`
- `prompt_postfix` = `"<|endofuser|><|assistant|>"`

图像占位 token 为 `<|img|>` … `<|imgpad|>` … `<|endofimg|>`（对应 C++ 多模态输入的 `image_start`/`image_content`/`image_end`）。

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
modelscope download --model 'dots-studio/dots.mocr' --local_dir /path/to/dots.mocr
```

- huggingface 下载

```
hf download rednote-hilab/dots.mocr --local-dir /path/to/dots.mocr
```

### 3.2 环境准备

```bash
pip install -r ../../requirements.txt
pip install -r requirements.txt
```

依次安装仓库通用依赖、RKNN3 Toolkit 和本示例的增量依赖。本示例的 `requirements.txt` 包含 dots.mocr 模型加载、ONNX 导出和量化校准图像预处理所需的额外依赖及版本约束。

将原始模型放置在 `/path/to/dots.mocr` 目录下。

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
    --model_path /path/to/dots.mocr \
    --export_vision_path ../../model/vision/dots-mocr-vision.onnx \
    --img_h 448 \
    --img_w 448

# 导出 RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/vision/dots-mocr-vision.onnx \
    --rknn_path ../../model/vision/dots-mocr-vision.rknn \
    --platform rk1820
```

### 3.4 导出 LLM 模型

```bash
cd python/llm

# 生成量化校准数据
python make_calidata.py --model_path /path/to/dots.mocr

# 导出 ONNX 模型
python export_llm.py \
    --model_path /path/to/dots.mocr \
    --export_llm_path ../../model/llm/dots-mocr-llm.onnx

# 导出 RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/llm/dots-mocr-llm.onnx \
    --config ../../model/llm/dots-mocr-llm.config.pkl \
    --rknn_path ../../model/llm/dots-mocr-llm.rknn \
    --platform rk1820
```

> **说明**：dots.mocr 的 `DotsOCRForCausalLM.forward` 参数顺序非标准，导出时需替换为 `Qwen2ForCausalLM.forward` 绕过 vision 部分。`convert_hf_to_gguf.py` 已原生支持 `DotsOCRForCausalLM` 架构（按 Qwen2 处理并跳过 vision tower 权重），直接从原始模型目录导出 tokenizer，不修改原始模型目录。

### 3.5 导出参数说明

**Vision `export_vision.py`**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 原始模型路径 | `/path/to/dots.mocr` |
| `--export_vision_path` | ONNX 输出路径 | `../../model/vision/dots-mocr-vision.onnx` |
| `--img_h` | 输入图像高度（28 的倍数） | 448 |
| `--img_w` | 输入图像宽度（28 的倍数） | 448 |
| `--quant` | 启用 GRQ 量化 | False |

**LLM `export_llm.py`**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 原始模型路径 | `/path/to/dots.mocr` |
| `--export_llm_path` | ONNX 输出路径 | `../../model/llm/dots-mocr-llm.onnx` |
| `--quant` | 启用 GRQ 量化 | False |

**Vision `export_rknn.py`**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--onnx_path` | ONNX 输入路径 | `../../model/vision/dots-mocr-vision.onnx` |
| `--rknn_path` | RKNN 输出路径 | `../../model/vision/dots-mocr-vision.rknn` |
| `--platform` | 目标平台 | `rk1820` |
| `--dataset_path` | 量化校准图片列表 | `../../../../datasets/MMBench/vision/datasets.txt`；若该文件不存在，脚本会自动生成并回退使用随仓库提交的 `../../data/vision/datasets_npy.txt` |
| `--core_num` | NPU 核数 | 8 |
| `--no_prune_mode` | 关闭裁剪模式 | False |

**`make_calidata.py`**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型路径 | `/path/to/dots.mocr` |
| `--datapath` | 校准数据集 JSON 路径 | `../../../../datasets/MMBench/llm/dataset.json` |
| `--export_datapath` | 校准数据输出路径 | `./quant_data/model_inputs.json` |

> **说明**：dots.mocr 的 `modeling_dots_vision.py` 硬依赖 `flash_attn`（无 fallback），脚本已注入纯 PyTorch 等价实现并在 CPU 环境可用；若在无 GPU 的机器上运行，请确认已安装 `torch`、`transformers`、`qwen-vl-utils`、`tqdm`。

## 4. Vision 模型分辨率调整

可通过 `--img_h` 和 `--img_w` 参数调整输入分辨率（必须为 28 的倍数）：

```bash
python export_vision.py --img_h 448 --img_w 448
```

> **注意**：
> - 分辨率越大，内存占用越高，会影响 LLM 的最大上下文长度
> - 部分分辨率可能与 RKNN 推理框架不兼容，如遇报错请联系 RKNPU 团队

若修改了分辨率，需同步修改 `cpp/vision/rknn_dots_mocr_vision.h` 中的参数：

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
./build-linux.sh -t rk3588 -a aarch64 -d dots.mocr
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_dots.mocr_demo/` 目录。

### 5.2 运行

```bash
./rknn_dots_mocr_demo \
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
./rknn_dots_mocr_demo \
    model/dots-mocr-vision.rknn model/dots-mocr-vision.weight \
    model/dots-mocr-llm.rknn model/dots-mocr-llm.weight \
    model/dots-mocr-llm.tokenizer.gguf model/dots-mocr-llm.embed.bin \
    0xff 0xff \
    demo.jpg "OCR."
```

> 示例中的 `demo.jpg` 取自本示例随仓库提交的 `data/demo.jpg`，运行前需将该文件复制/推送到 demo 所在目录（或直接传绝对路径）。

运行成功后，流式输出识别结果，例如：

```
Hello World

This is a test of the OCR model for text recognition. The quick brown fox jumps over the lazy dog.
```

### 5.4 性能参考

测试环境：RK3588（8 核 NPU，`core_mask=0xff`），Vision 分辨率 448×448，`rknn3-api` 1.1.0。

测试条件：无 warmup，单次执行（loop=1），NPU 频率未主动固定；统计口径为各阶段 NPU 推理耗时（不含模型初始化与前后处理），Prefill 耗时即生成首 token 前的 prefill 阶段耗时，速度按该阶段 token 数计算。

| 测试图片 | 阶段 | 耗时 (ms) | Token 数 | 速度 |
|----------|------|-----------|----------|------|
| demo.jpg | Vision | 503 | - | 1.99 FPS |
| demo.jpg | Prefill | 456 | 263 | 576 tok/s |
| demo.jpg | Generate | 452 | 25 | 55.3 tok/s |

## 6. 模型结构

dots.mocr 采用双模型架构：

- **Vision 模型**（`dots-mocr-vision.rknn`）：基于 DotsVisionTransformer，42 层 Transformer blocks，embed_dim=1536，使用 Conv2d patch embedding（patch_size=14，temporal_patch_size=1，spatial_merge_size=2）和 1D RoPE，无 window attention。输入 [1024, 588]（448×448 图像的 patch tokens），输出 [256, 1536] 图像嵌入。
- **LLM 模型**（`dots-mocr-llm.rknn`）：基于 Qwen2ForCausalLM 风格，28 层，hidden_size=1536，12 个注意力头 / 2 个 KV 头（GQA），使用标准 1D RoPE（非 MRoPE），vocab_size=151936，w4a16 量化。

两个模型通过 NPU 内部内存共享机制（`rknn3_set_internal_mem`）高效传递图像嵌入向量。

## 7. 常见问题

### LLM 输出乱码或不停生成

若导出 LLM RKNN 模型时在 `rknn.config()` 中设置了 `dynamic_input`（例如仅包含一个较短的 seq_len，如 64），模型会被编译为只支持该长度附近的动态形状。而实际 OCR 推理的 prefill 长度通常远超该值（如 448×448 图像对应 256 个图像 token + 文本 token，共 260+ tokens），超出编译时的 shape 范围会导致位置编码/attention 错乱，输出乱码且持续生成至 `max_new_tokens` 上限。

**解决方案**：导出时不要设置 `dynamic_input`，让编译器自动支持任意序列长度（与本仓库 `export_rknn.py` 保持一致）。

### Vision 导出 bf16 类型不匹配

dots.mocr 的 `DotsVisionTransformer.forward` 默认 `bf16=True`，会将输入转为 bfloat16。在 CPU 环境下导出时会报类型不匹配错误。导出脚本已通过 `model.float()` 和 `bf16=False` 参数处理此问题。

### Tokenizer 转换失败

`convert_hf_to_gguf.py` 已原生支持 `DotsOCRForCausalLM` 架构（按 Qwen2 处理并跳过 vision tower 权重）。若仍遇到转换失败，请确认 `tokenizer/thirdparty/llama_vocab/` 目录完整，以及模型目录中的 `tokenizer_config.json` 与 `tokenizer.json` 存在且未被损坏。

### LLM 输出为空

请检查以下配置是否正确：

1. **EOS token**：必须使用 `[151643, 151672, 151673]`（`</s>`、`<|assistant|>` 与 `<|endofassistant|>`，与 `generation_config.json` 一致），而非 tokenizer 返回的全部 EOG token（包含 `<|im_end|>` 151645、`[PAD]` 151680 等，会导致过早停止生成）。C++ 代码中已显式指定，无需修改。
2. **chat_template 标记**：C++ 代码中的 `prompt_prefix`/`prompt_postfix` 必须使用 `"<|user|>"`/`"<|endofuser|><|assistant|>"`，不能使用 Qwen 标准的 `<|im_start|>` 等标记。
3. **NPU 核心掩码**：RK3588 有 8 个 NPU 核心，需使用 `0xff` 而非 `0x1`/`0x6`。

### 提示词选择

dots.mocr 针对不同任务使用不同的提示词，常用：

| 任务 | 提示词 |
|------|--------|
| 文本识别（OCR） | `OCR.` |
| 版面解析 | `Parse the reading order of all elements in this image.`（输出 JSON） |
| 表格转 Markdown | `Parse the table in this image.` |
