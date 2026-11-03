# HunyuanOCR-1.5 模型部署说明

模型地址：[tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR)

HunyuanOCR-1.5 是腾讯推出的多模态 OCR 模型，由 Vision Encoder（SigLIP）和 LLM Decoder（HunyuanVL 架构，使用 XD-RoPE 位置编码）两部分组成。

## 1. 部署环境

由于 HunyuanOCR-1.5 模型对依赖库版本有特殊要求，与本仓库 `requirements.txt` 中的版本不兼容，请手动安装以下依赖：

```
transformers==5.13.0
```

> ⚠️ 使用默认依赖会导致模型转换失败，请务必按上述版本安装。

导出和校准脚本支持 `--hf_download` 或 `--modelscope` 下载模型，两个参数不能同时使用。使用 ModelScope 时需先安装可选依赖 `modelscope`。

- 目标平台：RK1820（RK3588）

## 2. 模型裁剪策略

### 2.1 Vision 模型裁剪

由于 RKNN 工具链不支持高维 Transpose 算子，完整版在 NPU 上执行该算子会导致精度损失。因此**必须使用裁剪版（prune mode）**来保证模型精度。

裁剪版将 patch 展平、归一化等预处理操作从 Vision 模型中剥离，在 CPU 侧完成，具有以下特点：

- 避免了 NPU 上高维 Transpose 算子的精度损失，保证模型推理精度
- 减小模型体积和 NPU 推理开销
- 输入变为展平后的 patches（形状 `[grid_h * grid_w, 768]`）和位置编码（`[grid_h * grid_w, 1152]`），而非原始像素
- C++ 运行时自动检测模型类型（完整版/裁剪版），无需手动指定模式

### 2.2 KVCache 量化

HunyuanOCR-1.5 的 LLM 在 `export_rknn.py` 中配置了 KV Cache 参数：

```python
llm_config['attention_config'][0]['kvcache_buffer_len'] = 1024
llm_config['attention_config'][0]['max_position_embeddings'] = 1024
```

若需支持更长的上下文长度，可调整 `kvcache_buffer_len` 和 `max_position_embeddings` 参数，并考虑启用 KV Cache INT4 量化（`Int4_to_F16`）以进一步压缩内存：

```python
llm_config['attention_config'][0]['kvcache_dtype'] = 'Int4_to_F16'
```

修改后需重新执行 RKNN 转换。

> ⚠️ 开启 KVCache 量化后精度可能有所下降，请根据实际效果决定是否启用。

## 3. 模型导出

### 3.1 导出 LLM 模型

```bash
cd examples/HY_OCR_1_5/python/llm

# 生成 LLM 模型量化校准数据
python make_calidata.py --model_path tencent/HunyuanOCR

# 导出 LLM ONNX 模型（支持 GRQ 量化）
python export_llm.py \
    --model_path tencent/HunyuanOCR \
    --quant

# 导出 LLM RKNN 模型（默认 W4A16，默认 8 核）
python export_rknn.py \
    --onnx_path ../../model/llm/HunyuanOCR-1.5-llm.onnx \
    --config ../../model/llm/HunyuanOCR-1.5-llm.config.pkl \
    --rknn_path ../../model/llm/HunyuanOCR-1.5-llm.rknn \
    --core_num 8
```

导出 LLM ONNX 时会同步生成以下文件：
- `.config.pkl`：LLM 模型配置
- `.tokenizer.gguf`：Tokenizer 文件
- `.embed.bin`：Embedding 权重（float16，hidden dim 1024）

LLM RKNN 转换默认使用 W4A16 量化，采用 normal + group32 方式。如需 W8A16 量化，修改 `export_rknn.py` 中的 `quantized_dtype` 为 `w8a16` 即可。

默认推荐 W4A16 双 8 核配置（Vision 与 LLM 均为 8 核，运行时 core mask 传 `0xff`），板端实测性能见 4.4 节。W8A16 亦可转换，但权重体积约为 W4A16 的两倍。

HunyuanOCR-1.5 使用 XD-RoPE 位置编码，已在 `export_rknn.py` 中通过 `llm_config` 配置好：

```python
llm_config['attention_config'][0]['mrope_type'] = 'HunyuanVL'
llm_config['attention_config'][0]['mrope_section'] = [16, 16, 16, 16]
llm_config['attention_config'][0]['mrope_new_id_name'] = 'xd_rope_id_input'
```

`make_calidata.py` 参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型路径或 HuggingFace 名称 | `tencent/HunyuanOCR` |
| `--datapath` | 校准数据集 JSON 路径（含图片路径和文本输入） | `../../../../datasets/OmniDocBench_ROI/llm/dataset.json` |
| `--export_datapath` | 生成的量化校准数据输出路径 | `./quant_data/model_inputs.json` |
| `--hf_download` | 从 HuggingFace 下载模型 | 关闭 |
| `--modelscope` | 从 ModelScope 下载模型 | 关闭 |

> ⚠️ `make_calidata.py` 会加载完整模型并捕获模块输入，需要较大内存。生成的校准数据用于 `export_llm.py --quant` 时的 GRQ 量化。

### 3.2 导出 Vision 模型

Vision 模型固定使用 512×512 分辨率。导出时会同步生成 ONNX 模型、pos_embed 预计算文件和 vision_config.json。

```bash
cd examples/HY_OCR_1_5/python/vision

# 生成 Vision 模型量化校准数据
python make_calidata.py --model_path tencent/HunyuanOCR

# 导出 Vision ONNX 模型（支持 GRQ 量化）
python export_vision.py \
    --model_path tencent/HunyuanOCR \
    --img_h 512 \
    --img_w 512 \
    --quant

# 导出 Vision RKNN 模型（默认裁剪版，使用 --no_prune 导出完整版）
python export_rknn.py \
    --onnx_path ../../model/vision/HunyuanOCR-1.5-vision.onnx \
    --rknn_path ../../model/vision/HunyuanOCR-1.5-vision.rknn \
    --core_num 8
```

导出文件说明：

| 文件 | 说明 |
|------|------|
| `HunyuanOCR-1.5-vision.onnx` | Vision ONNX 模型（裁剪版，2 输入：flatten_patches + pos_embed） |
| `pos_embed_512x512.bin` | 预计算的位置编码（float16，[1024, 1152]），C++ 运行时加载 |
| `vision_config.json` | Vision 配置（分辨率、grid 尺寸等） |
| `HunyuanOCR-1.5-vision.rknn` | Vision RKNN 模型 |
| `HunyuanOCR-1.5-vision.weight` | Vision RKNN 权重文件 |

> ⚠️ **注意**：
> - 由于 RKNN 工具链不支持高维 Transpose 算子，必须使用裁剪版（默认行为）导出 Vision 模型
> - `pos_embed_512x512.bin` 由 `export_vision.py` 自动生成，C++ 侧从模型同目录加载
> - 若需修改分辨率，需同步修改 `rknn_hy_ocr_vision.h` 中的 `MODEL_WIDTH`/`MODEL_HEIGHT` 默认值

`make_calidata.py` 参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型路径或 HuggingFace 名称 | `tencent/HunyuanOCR` |
| `--datapath` | 校准数据集 JSON 路径（含图片路径和文本输入） | `../../../../datasets/OmniDocBench_ROI/llm/dataset.json` |
| `--export_datapath` | 生成的量化校准数据输出路径 | `./quant_data/model_inputs.json` |
| `--hf_download` | 从 HuggingFace 下载模型 | 关闭 |
| `--modelscope` | 从 ModelScope 下载模型 | 关闭 |

> ⚠️ `make_calidata.py` 会加载完整模型并捕获模块输入，需要较大内存。生成的校准数据用于 `export_vision.py --quant` 时的 GRQ 量化。

## 4. C++ 部署说明

### 4.1 编译

```bash
cd rknn3-model-zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d HY_OCR_1_5
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_hy_ocr_demo/` 目录。目录结构如下：

```
- install/rk3588_linux_aarch64/rknn_hy_ocr_demo
  - lib
    - librknn3_api.so
    - librga.so
  - model
    - HunyuanOCR-1.5-vision.rknn
    - HunyuanOCR-1.5-vision.weight
    - pos_embed_512x512.bin
    - HunyuanOCR-1.5-llm.rknn
    - HunyuanOCR-1.5-llm.weight
    - HunyuanOCR-1.5-llm.tokenizer.gguf
    - HunyuanOCR-1.5-llm.embed.bin
    - demo.jpg
  - rknn_hy_ocr_demo
```

### 4.2 运行

```bash
./rknn_hy_ocr_demo \
    model/HunyuanOCR-1.5-vision.rknn \
    model/HunyuanOCR-1.5-vision.weight \
    model/HunyuanOCR-1.5-llm.rknn \
    model/HunyuanOCR-1.5-llm.weight \
    model/HunyuanOCR-1.5-llm.tokenizer.gguf \
    model/HunyuanOCR-1.5-llm.embed.bin \
    0xff 0xff \
    model/demo.jpg \
    "Detect and recognize text in the image, and output the text coordinates in a formatted manner."
```

> `model_width`/`model_height` 参数可省略，默认使用 `rknn_hy_ocr_vision.h` 中的 `MODEL_WIDTH=512`、`MODEL_HEIGHT=512`。如需指定，在末尾追加 `512 512` 即可。

### 4.3 参数说明

| 参数 | 说明 |
|------|------|
| vision_model_path | Vision RKNN 模型路径 |
| vision_weight_path | Vision RKNN 权重文件路径 |
| llm_model_path | LLM RKNN 模型路径 |
| llm_weight_path | LLM RKNN 权重文件路径 |
| tokenizer_path | Tokenizer 文件路径（.tokenizer.gguf） |
| embedding_path | Embedding 权重文件路径（.embed.bin） |
| vision_core_mask | Vision 模型 NPU 核心掩码，需与 `--core_num` 匹配，建议 `0xff`（8 核） |
| llm_core_mask | LLM 模型 NPU 核心掩码，需与 `--core_num` 匹配，建议 `0xff`（8 核） |
| image_path | 输入图像路径 |
| prompt | 用户输入文本，包含空格时需用双引号包裹 |
| model_width | （可选）Vision 模型输入宽度，默认 512 |
| model_height | （可选）Vision 模型输入高度，默认 512 |

> ⚠️ **NPU 核心配置注意事项**：
> - Vision 与 LLM 均建议使用 8 核（`0xff`）。两个模型在板端串行执行（先 Vision 后 LLM），可以同时占满全部 8 个 NPU 核心，internal memory 也可完全共享
> - 核心掩码的置位数必须等于模型 build 时的 `--core_num`，否则初始化会报错（例如 `core_num=8` 时必须传 `0xff`，传 `0x0f` 会因置位数不符而失败）
> - LLM **禁止单核**运行（单核可能超内存并导致 NPU 通信异常，需板端重启恢复）
> - 运行前需设置环境变量 `RKNN3_SKIP_PROTO_CHECK=1`

运行成功后，会打印出识别结果及推理性能，例如：

```
--> init hy_ocr vision model
--> init hy_ocr llm model
--> init internal share
--> inference hy_ocr vision model
--> inference hy_ocr llm model
rknn_session_run
识别结果...

--------------------Finished--------------------

--------------------------------------------------------------------------------------
 Stage         Total Time (ms)  Tokens    Time per Token (ms)      Tokens per Second
--------------------------------------------------------------------------------------
 Prefill       107.78           297       0.36                     2755.6
 Generate      648.70           103       6.30                     158.77
--------------------------------------------------------------------------------------
 Vision latency = 553.74 ms, FPS = 1.81
```

### 4.4 性能数据

**W4A16 双 8 核（推荐配置）**：测试条件：`demo.jpg`（即 long_text2.jpg）+ 官方英文 prompt，W4A16 LLM（8 核 `0xff`）+ W4A16 Vision（8 核 `0xff`），板端 RK1820 NPU 1GHz，无 warmup 单次推理。

| 阶段 | Total Time (ms) | Tokens | Tokens per Second |
|------|----------------|--------|-------------------|
| Prefill | 107.78 | 297 | 2755.6 |
| Generate | 648.70 | 103 | 158.77 |
| Vision | 553.74 | — | FPS 1.81 |

#### Vision resize 模式

C++ demo 通过 `HY_OCR_RESIZE_MODE` 选择预处理策略：

| 模式 | 处理方式 | 适用场景 |
|------|----------|----------|
| `lanczos`（默认） | 优化 CPU Lanczos 直接缩放至 512×512 | 精度优先 |
| `hybrid` | 大图先由 RGA 预缩小，再以优化 CPU Lanczos 精修；小图直接使用 Lanczos | 性能与精度平衡（推荐） |
| `rga` | RGA 双线性直接缩放至 512×512 | 普通文本极速模式；密集表格可能退化 |

```bash
export HY_OCR_RESIZE_MODE=hybrid
```

纯 `lanczos` 模式会分配约 `src_height × model_width × 3 × sizeof(float)` 的中间缓冲；处理超大图或板端内存紧张时建议使用 `hybrid` 模式。


## 5. 模型接口

| 子模型 | 输入 | 输出 | 配置 |
|--------|------|------|------|
| Vision（裁剪版 512×512） | `flatten_patches [1024, 768]` FP32<br>`pos_embed [1024, 1152]` FP16 | `merged [1, 1025, 1024]` FP16 | W4A16, 8 核, grid 32×32 |
| LLM | vision/text embedding 融合输入 `[1, seq_len, 1024]`<br>`xd_rope_id_input`（input_callback 注入） | token logits → OCR 文本+坐标 | W4A16, 8 核, KV cache 1024 |

> LLM image token 布局：`begin + content(grid_h×(grid_w+1) 含 newline) + end = n_image_tokens`，image content token id=120120，512×512 时 content 1024 tokens（32 行 × 33 列）。grid 32×32 时输出含 begin/end 共 1025 tokens。


## 6. 常见问题

### Vision 预处理 transpose 顺序

HunyuanOCR-1.5 使用 PIL backend（`use_fast=False`）预处理，patch flatten 的 transpose perm 必须为 `{2, 3, 5, 6, 1, 0, 4, 7}`（对应 PIL backend 的 `patches.transpose(0, 1, 4, 5, 7, 8, 3, 2, 6, 9)`）。C++ 侧已正确实现，修改时请勿更改此顺序，否则会导致 OCR 结果错误（仅输出点号）。

### pos_embed 文件

C++ 侧从 Vision 模型同目录加载 `pos_embed_{width}x{height}.bin` 文件（如 `pos_embed_512x512.bin`）。该文件由 `export_vision.py` 导出 ONNX 时自动生成，包含经 bilinear 插值后的位置编码（float16）。若修改了 Vision 分辨率，需重新导出并确保该文件存在。

### LLM Embedding 文件

正确的 LLM Embedding 文件是 `HunyuanOCR-1.5-llm.embed.bin`（float16，hidden dim 1024）。注意 `model.language_model.embed_tokens.weight` 是 float32，按 float16 读取会错误推导 hidden dim 为 2048，触发 `invalid embed buffer` 错误。请使用 `export_llm.py` 生成的 `.embed.bin` 文件。

### Chat template

C++ 运行时的 `prompt_prefix`、`prompt_postfix` 和 `system_prompt` 与原模型 `chat_template.jinja` 保持一致。若更新模型或修改 chat template，必须同步更新 `rknn_hy_ocr_llm.cc` 中的这些常量并重新验证输出。
