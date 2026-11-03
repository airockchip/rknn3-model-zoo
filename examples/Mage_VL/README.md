# Mage-VL

本目录提供 Microsoft Mage-VL 的 Vision 与 Qwen3-4B LLM 转换脚本，支持将
Hugging Face 模型转换为 ONNX，并进一步转换为 RKNN3 模型。

## 模型配置

| 项目 | 配置 |
|---|---|
| 原始模型 | `microsoft/Mage-VL` |
| Vision 输入 | `pixel_values [576, 768]`、`rotary_pos_emb [1, 576, 64]` |
| Vision 输出 | `image_embeddings [144, 2560]` |
| Vision 量化 | GRQ W4A16，group32 |
| LLM | Qwen3-4B，`input_ids + RKNN3Image` |
| LLM 量化 | 主体 W4A16，LM Head W6A16，group32 |
| KV Cache | Int4-to-F16，group16，最大长度 4096 |
| 目标平台 | RK1820、RK1828 |

默认 Vision 输入对应一张 384×384 图片：每个 Patch 为 16×16 像素，共
576 个 Patch；经过 2×2 Patch Merge 后输出 144 个视觉 Token。

## 环境准备

在仓库根目录执行：

```bash
pip install -r examples/Mage_VL/requirements.txt
```

GRQ 量化要求 CUDA 可用，并需要安装与当前 SDK 配套的 RKNN3 Toolkit。

## 生成校准数据

仓库中的 `datasets/MMBench/llm/dataset.json` 可作为默认校准数据源。建议生成
不少于 200 条样本，正式模型使用 256 条样本。

```bash
cd examples/Mage_VL/python/vision
python make_calidata.py \
  --model_path microsoft/Mage-VL \
  --export_datapath ./quant_data/model_inputs.json \
  --limit 256

cd ../llm
python make_calidata.py --model_path microsoft/Mage-VL --limit 256
```

## 转换 Vision 模型

```bash
cd examples/Mage_VL/python/vision

python export_vision.py \
  --model_path microsoft/Mage-VL \
  --patch_num 576 \
  --quant \
  --cali_dataset ./quant_data/model_inputs.json

python export_rknn.py \
  --platform rk1820 \
  --core_num 8 \
  --save_ctx
```

默认输出：

```text
examples/Mage_VL/model/vision/
├── mage_vision_patch576_grq_w4a16.onnx
├── mage_vision_patch576_grq_w4a16.rknn
└── mage_vision_patch576_grq_w4a16.weight
```

## 转换 LLM 模型

```bash
cd examples/Mage_VL/python/llm

python export_llm.py \
  --model_path microsoft/Mage-VL \
  --quant \
  --cali_dataset ./quant_data/model_inputs.json

python export_rknn.py \
  --platform rk1820 \
  --context_len 4096 \
  --seq_lens 1,128 \
  --save_ctx
```

ONNX 导出同时生成 RKNN3 LLM 所需的配置、Tokenizer 和 Embedding 文件：

```text
examples/Mage_VL/model/llm/
├── model_llm.onnx
├── model_llm.config.pkl
├── model_llm.embed.bin
├── model_llm.tokenizer.gguf
├── model_llm_w4a16.rknn
└── model_llm_w4a16.weight
```

`.rknn` 文件与同名 `.weight` 文件必须配套使用。

## C++ Demo

C++ Demo 通过 `--input-mode auto|image|frame|codec` 选择推理模式，四种模式
使用同一套 Vision、LLM、Tokenizer 和 Embedding 模型文件：

- `image`：输入一张图片。
- `frame`：输入视频帧目录，最多选择 16 帧。
- `codec`：输入由 Codec 预处理工具生成的资产目录。
- `auto`：根据输入文件或目录自动选择模式。

### Frame 与 Codec 模式区别

Mage-VL 的视频理解特点主要体现在 Codec 模式。它不是简单地从视频中均匀抽取
若干完整帧，而是利用视频编码侧的块级 bit-cost 信息判断画面不同区域的重要
程度，优先保留包含运动和新增细节的 Patch。在相同的 Vision 输入预算下，
Codec 模式可以覆盖更多候选帧。

| 项目 | Frame 模式 | Codec 模式 |
|---|---|---|
| 时间采样 | 从视频中均匀抽取完整帧 | 将更多候选帧分组后进行 Patch 筛选 |
| 空间处理 | 保留每张入选帧的全部 Patch | 按编码代价选择高价值的 2×2 Patch Block，并重新拼成 Canvas |
| 位置信息 | 根据当前完整帧的 Patch 网格生成 3D RoPE | 记录每个 Patch 的原始 `[帧号, 行, 列]`，据此生成 3D RoPE |
| 默认输入规模 | 最多 16 个完整帧 | 从 4×32 个候选帧生成 4×4 个 Canvas |
| 适用场景 | 通用视频理解和功能验证 | 在固定视觉 Token 预算下分析更长时间范围内的运动和内容变化 |

默认 Codec 配置将 128 个候选帧划分为 4 组，每组从 32 个候选帧中生成 4 个
384×384 Canvas，最终仍只执行 16 次 Vision 推理。每个 Canvas 包含 576 个
Patch，经 2×2 Patch Merge 后输出 144 个视觉 Token，因此 LLM 共接收 2304 个
视觉 Token。与把 128 个候选帧全部作为完整帧送入相比，视觉 Token 数量减少
87.5%。

Codec 预处理会生成以下关键资产：

- `canvas_*.png`：由筛选出的 Patch Block 重新排列得到的固定尺寸 Canvas。
- `src_patch_position.npy`：每个 Patch 在原视频中的 `[帧号, 行, 列]`。
- `rotary_pos_emb.fp16.bin`：根据原始 Patch 位置生成的 3D RoPE。
- `codec_metadata.json`：候选帧、Canvas、Patch 和位置编码等配置记录。

编码信息筛选由 `decode_video.py` 调用 `codec-video-prep` 提供的 `cv-preinfer`
预先完成。C++ Demo 不直接解析 H.264/HEVC 码流，而是加载生成的 Canvas 和
3D RoPE 后依次运行 Vision，再将全部视觉 Embedding 送入 LLM。

### 编译

C++ Demo 依赖仓库 `3rdparty` 中随 RKNN3 SDK 提供的 RKNN Runtime、RGA 和
图像解码库，版本应与当前 SDK 保持一致。板端的 `librknn3_api.so` 与
`rknn3_transfer_proxy` 也必须来自同一 SDK 版本。视频预处理脚本的 Python
依赖由本目录的 `requirements.txt` 管理。

```bash
export GCC_COMPILER=/path/to/toolchain/bin/aarch64-none-linux-gnu
./build-linux.sh -t rk3588 -a aarch64 -d Mage_VL
```

### 图片理解

图片模式支持 JPG、JPEG、PNG 和 BMP 格式：

```bash
./rknn_mage_vl_demo \
  <vision.rknn> <vision.weight> \
  <llm.rknn> <llm.weight> \
  <tokenizer.gguf> <embed.bin> \
  <vision_core_mask> <llm_core_mask> \
  /path/to/image.jpg \
  "Describe this image." \
  --input-mode image
```

### 普通帧模式视频理解

仓库提供了一个 15 秒的 CC0 行人街景视频用于功能验证：
`examples/Mage_VL/data/pedestrian_area_cc0.mp4`。视频来源、许可、转码参数和
SHA-256 记录在 `examples/Mage_VL/data/README.md` 中。

先使用视频预处理脚本均匀抽取视频帧：

```bash
python examples/Mage_VL/python/decode_video.py \
  examples/Mage_VL/data/pedestrian_area_cc0.mp4 \
  /tmp/mage_vl_frame_assets \
  --mode frame
```

```bash
./rknn_mage_vl_demo \
  <vision.rknn> <vision.weight> \
  <llm.rknn> <llm.weight> \
  <tokenizer.gguf> <embed.bin> \
  <vision_core_mask> <llm_core_mask> \
  /tmp/mage_vl_frame_assets \
  "Describe this video." \
  --input-mode frame
```

帧目录中的图片按照文件名排序。当前 Demo 最多选择 16 帧；如需分析特定视频
片段，请先裁剪视频，再执行预处理。

### Codec 模式视频理解

```bash
python examples/Mage_VL/python/decode_video.py \
  examples/Mage_VL/data/pedestrian_area_cc0.mp4 \
  /tmp/mage_vl_codec_assets \
  --mode codec \
  --codec-score official
```

`official` 为默认且推荐的 Codec 筛选方式，使用 H.264/HEVC 编码侧的块级
bit-cost 信息，而不是仅根据解码后相邻帧的像素差异判断 Patch 重要性。

然后将生成的资产目录传给 C++ Demo：

```bash
./rknn_mage_vl_demo \
  <vision.rknn> <vision.weight> \
  <llm.rknn> <llm.weight> \
  <tokenizer.gguf> <embed.bin> \
  <vision_core_mask> <llm_core_mask> \
  /tmp/mage_vl_codec_assets \
  "Describe this video." \
  --input-mode codec
```
