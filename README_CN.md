[English](README.md)

# RKNN3-MODEL-ZOO

## 目录

- [1. 简介](#1-简介)
- [2. 支持的模型](#2-支持的模型)
  - [2.1 LLM](#21-llm)
  - [2.2 VLM](#22-vlm)
  - [2.3 全模态（Omni）](#23-全模态omni)
  - [2.4 ASR（语音识别）](#24-asr语音识别)
  - [2.5 TTS（语音合成）](#25-tts语音合成)
  - [2.6 Embedding / Reranker](#26-embedding--reranker语义向量与排序)
  - [2.7 翻译](#27-翻译)
  - [2.8 OCR](#28-ocr)
  - [2.9 JEV（决策模型）](#29-jev决策模型)
  - [2.10 CV（计算机视觉）](#210-cv计算机视觉)
  - [2.11 预转换 RKNN 模型](#211-预转换-rknn-模型)
- [3. 支持的平台](#3-支持的平台)
- [4. 快速开始](#4-快速开始)
  - [4.1 环境准备](#41-环境准备)
  - [4.2 通用部署流程](#42-通用部署流程)
  - [4.3 常用参数](#43-常用参数)
- [5. 高级功能](#5-高级功能)
  - [5.1 SpeedUP 推理加速](#51-speedup-推理加速)
  - [5.2 多卡推理](#52-多卡推理)
  - [5.3 LoRA 支持](#53-lora-支持)
- [6. 模型适配指南](#6-模型适配指南)
- [7. 注意事项](#7-注意事项)
- [8. 其他说明](#8-其他说明)


## 1. 简介

RKNN3 SDK 提供了将 AI 模型部署到 RK1820/RK1828 协处理器所需的完整软件栈，包括：

- **[RKNN3-Toolkit](https://github.com/airockchip/rknn3-toolkit)**：PC 端软件开发套件，支持模型转换、推理和性能评估等。
- **RKNN3 Runtime**：板端运行时库，提供 C/C++ 编程接口，用于部署 RKNN 模型并加速 AI 应用。
- **[RKNN3 Model Zoo](https://github.com/airockchip/rknn3-model-zoo)**：模型转换与部署示例仓库，包含 CNN / LLM / VLM 等多种模型的参考实现。
- **[RKNN3 MCP](https://github.com/airockchip/rknn3-toolkit/blob/main/doc/CN/09_RKNN3_MCP_%E5%BF%AB%E9%80%9F%E5%85%A5%E9%97%A8_V0.1.0.pdf)**：RKNN3 MCP 是面向瑞芯微 RKNN3 平台的模型部署助手。用户可以通过支持 MCP 的 AI 开发工具，以自然语言描述需求，由 Agent 在 RKNN3 MCP 的指导下，协助完成模型分析、ONNX 导出、RKNN3 模型转换、精度分析和板端部署等任务，与本仓库的导出及部署示例配套使用。RKNN3 MCP 覆盖 CNN、ViT、LLM、VLM、ASR 等模型工作流，并支持 LoRA 转换与部署、多卡部署与推理等功能。

本仓库提供了完整的模型部署流程：
- **模型导出**：将 HuggingFace / PyTorch 模型导出为 ONNX 格式
- **模型转换**：使用 RKNN3 Toolkit 将 ONNX 等模型转换为 RKNN 格式
- **板端部署**：提供 C++ 推理示例代码

## 2. 支持的模型

### 2.1 LLM

| 模型 | 示例目录 | 说明 |
|------|---------|------|
| [GLM-Edge](examples/glm_edge/README.md) | `examples/glm_edge` | 智谱 GLM Edge 系列端侧大语言模型 |
| [MiniCPM5](examples/MiniCPM5/README.md) | `examples/MiniCPM5` | 面壁智能 MiniCPM5 大语言模型 |
| [MiniCPM5-2B](examples/MiniCPM5_2B/README.md) | `examples/MiniCPM5_2B` | 面壁智能 MiniCPM5-2B 大语言模型 |
| [Nanbeige4.2](examples/Nanbeige4_2/README.md) | `examples/Nanbeige4_2` | 南北阁 Looped Transformer 大语言模型 |
| [Qwen2.5](examples/Qwen2_5/README.md) | `examples/Qwen2_5` | 通义千问 2.5 系列大语言模型 |
| [Qwen3](examples/Qwen3/README.md) | `examples/Qwen3` | 通义千问 3 系列大语言模型 |
| [Qwen3.5](examples/Qwen3_5/README.md) | `examples/Qwen3_5` | 通义千问 3.5 系列大语言模型 |
| [FunctionGemma](examples/functiongemma/README.md) | `examples/functiongemma` | Google FunctionGemma 函数调用模型 |
| [LFM2.5](examples/LFM2_5/README.md) | `examples/LFM2_5` | LiquidAI LFM2.5 混合架构语言模型（Full Attention + 流式短卷积） |
| [Spark-X2.5](examples/Spark_X2_5/README.md) | `examples/Spark_X2_5` | Spark-X2.5 混合注意力大语言模型（full attention + 滑动窗口，1.7B / 4B） |

### 2.2 VLM

| 模型 | 示例目录 | 说明 |
|------|---------|------|
| [FastVLM](examples/FastVLM/README.md) | `examples/FastVLM` | 苹果轻量级视觉语言模型 |
| [LFM2.5-VL](examples/LFM2_5_VL/README.md) | `examples/LFM2_5_VL` | LiquidAI LFM2.5-VL 视觉语言模型（1.6B / 3B） |
| [SmolVLM](examples/SmolVLM/README.md) | `examples/SmolVLM` | HuggingFace 轻量级视觉语言模型 |
| [SmolVLM2](examples/SmolVLM2/README.md) | `examples/SmolVLM2` | HuggingFace SmolVLM 第二代视觉语言模型 |
| [GME-Qwen2-VL](examples/GME-Qwen2-VL/README.md) | `examples/GME-Qwen2-VL` | 通义千问视觉嵌入模型 |
| [InternVLM](examples/InternVLM/README.md) | `examples/InternVLM` | 书生视觉语言模型 |
| [Janus-Pro](examples/Janus_Pro/README.md) | `examples/Janus_Pro` | DeepSeek 多模态理解与生成模型 |
| [Mage-VL](examples/Mage_VL/README.md) | `examples/Mage_VL` | 微软 Mage-VL 视觉语言模型 |
| [LocateAnything](examples/LocateAnything/README.md) | `examples/LocateAnything` | NVIDIA 视觉定位多模态模型（目标检测/短语定位/GUI 定位） |
| [MiniCPM-V](examples/MiniCPM_V/README.md) | `examples/MiniCPM_V` | 面壁智能 MiniCPM-V 视觉语言模型 |
| [MiniCPM-V-4](examples/MiniCPM_V_4/README.md) | `examples/MiniCPM_V_4` | 面壁智能 MiniCPM-V-4 视觉语言模型 | 
| [MiniCPM-V-4.7-1B](examples/MiniCPM_V_4.7_1B/README.md) | `examples/MiniCPM_V_4.7_1B` | 面壁智能 MiniCPM-V-4.7-1B 视觉语言模型 |
| [Qwen2.5-VL](examples/Qwen2_5_VL/README.md) | `examples/Qwen2_5_VL` | 通义千问 2.5 视觉语言模型 |
| [Qwen3-VL](examples/Qwen3_VL/README.md) | `examples/Qwen3_VL` | 通义千问 3 视觉语言模型 |
| [Qwen3-VL-LoRA](examples/Qwen3_VL_LoRA/README.md) | `examples/Qwen3_VL_LoRA` | 通义千问 3-VL LoRA 微调模型 |
| [Qwen3.5-VL](examples/Qwen3_5_VL/README.md) | `examples/Qwen3_5_VL` | 通义千问 3.5 视觉语言模型 |
| [UI-TARS](examples/UI_TARS/README.md) | `examples/UI_TARS` | 字节跳动 GUI Agent 视觉语言模型 |

### 2.3 全模态（Omni）

| 模型 | 示例目录 | 说明 |
|------|---------|------|
| [Gemma-4](examples/gemma4/README.md) | `examples/gemma4` | Google Gemma-4 多模态模型（文本 + 音频） |
| [Qwen2.5-Omni](examples/Qwen2_5_Omni/README.md) | `examples/Qwen2_5_Omni` | 通义千问 2.5 全模态模型 |
| Qwen3-Omni | 商业闭源模型 | 通义千问 3 全模态模型 |
| Qwen3.5-Omni | 商业闭源模型 | 通义千问 3.5 全模态模型 |

### 2.4 ASR（语音识别）

| 模型 | 示例目录 | 说明 |
|------|---------|------|
| [Paraformer](examples/paraformer/README.md) | `examples/paraformer` | FunASR SeaCo-Paraformer 中文离线语音识别模型 |
| [Qwen3-ASR](examples/Qwen3_ASR/README.md) | `examples/Qwen3_ASR` | 通义千问语音识别（流式/非流式） |
| [Qwen3-ForcedAligner](examples/Qwen3_ForcedAligner/README.md) | `examples/Qwen3_ForcedAligner` | 音频文本强制对齐，输出中文逐字、英文逐词的起止时间 |
| [SenseVoiceSmall](examples/sensevoice_small/README.md) | `examples/sensevoice_small` | 阿里 FunAudioLLM 语音识别模型 |
| [WeNet (Conformer)](examples/wenet/README.md) | `examples/wenet` | WeNet U2++ Conformer 流式中文语音识别 |
| [Whisper](examples/whisper/README.md) | `examples/whisper` | OpenAI 多语言语音识别模型 |
| [Zipformer](examples/zipformer/README.md) | `examples/zipformer` | Zipformer 流式语音识别 |

### 2.5 TTS（语音合成）

| 模型 | 示例目录 | 说明 |
|------|---------|------|
| [Qwen3-TTS](examples/Qwen3_TTS/README.md) | `examples/Qwen3_TTS` | 通义千问语音合成 |
| [VITS](examples/vits/README.md) | `examples/vits` | VITS 语音合成（LJSpeech / VCTK） |

### 2.6 Embedding / Reranker（语义向量与排序）

| 模型 | 示例目录 | 说明 |
|------|---------|------|
| [Qwen3-Embedding](examples/Qwen3_Embedding/README.md) | `examples/Qwen3_Embedding` | 通义千问文本向量模型 |
| [Qwen3-Reranker](examples/Qwen3_Reranker/README.md) | `examples/Qwen3_Reranker` | 通义千问重排序模型 |
| [WeMM-Embedding-2B](examples/WeMM-Embedding/README.md) | `examples/WeMM-Embedding` | 腾讯多模态向量模型（文本/图片/视频输出 2048 维向量） |

### 2.7 翻译

| 模型 | 示例目录 | 说明 |
|------|---------|------|
| [HY-MT1.5](examples/HY_MT_1_5/README.md) | `examples/HY_MT_1_5` | 混元多语言翻译模型 |

### 2.8 OCR

| 模型 | 示例目录 | 说明 |
|------|---------|------|
| [dots.mocr](examples/dots.mocr/README.md) | `examples/dots.mocr` | 小红书 dots.mocr 多语言 OCR 视觉语言模型 |
| [dots.ocr](examples/dots.ocr/README.md) | `examples/dots.ocr` | 小红书 dots.ocr 文档版面解析与 OCR 视觉语言模型 |
| [HunyuanOCR-1.5](examples/HY_OCR_1_5/README.md) | `examples/HY_OCR_1_5` | 腾讯混元 HunyuanOCR-1.5 多模态 OCR 模型 |
| [PaddleOCR-VL](examples/paddleocr_vl/README.md) | `examples/paddleocr_vl` | 百度 PaddleOCR-VL 视觉 OCR 模型 |
| [Unlimited-OCR](examples/Unlimited-OCR/README.md) | `examples/Unlimited-OCR` | Unlimited-OCR 模型（SAM + Vision + LLM 拆分部署） |

### 2.9 JEV（决策模型）

JEV 模型本质上是决策（Decision）模型：不生成自由文本，而是将文本或图文输入直接转换为结构化的决策结果，如分类（选择）、是非判断（含概率）和评分。

| 模型 | 示例目录 | 说明 |
|------|---------|------|
| [Intern-Decision](examples/Intern_Decision/README.md) | `examples/Intern_Decision` | 书生 Intern-Decision 文本/图文决策模型（分类、是非判断、评分） |
| [Laya](examples/Laya/README.md) | `examples/Laya` | Laya 决策模型，用于选择、评分、真假判断（任务配置驱动） |

### 2.10 CV（计算机视觉）

| 模型 | 示例目录 | 说明 |
|------|---------|------|
| [MobileNetV2](examples/mobilenet_v2/README.md) | `examples/mobilenet_v2` | 轻量级图像分类模型 |
| [ResNet](examples/resnet/README.md) | `examples/resnet` | 经典残差图像分类模型 |
| [YOLOv5](examples/yolov5/README.md) | `examples/yolov5` | 目标检测模型 |
| [YOLOv6](examples/yolov6/README.md) | `examples/yolov6` | 目标检测模型 |
| [YOLOv8](examples/yolov8/README.md) | `examples/yolov8` | 目标检测模型 |
| [YOLO26](examples/yolo26/README.md) | `examples/yolo26` | Ultralytics YOLO26 目标检测模型（yolo26n/s/m） |
| [YOLO26-Segment](examples/yolo26_segment/README.md) | `examples/yolo26_segment` | Ultralytics YOLO26 实例分割模型 |
| [YOLO26-Pose](examples/yolo26_pose/README.md) | `examples/yolo26_pose` | Ultralytics YOLO26 人体姿态估计模型 |
| [QA-CLIP](examples/QAClip/README.md) | `examples/QAClip` | 中英文图文相似度模型 |
| [Depth Anything V3](examples/depth_anything_v3/README.md) | `examples/depth_anything_v3` | 立体深度估计模型 |
| [RAFT-Stereo](examples/RAFT-Stereo/README.md) | `examples/RAFT-Stereo` | 立体匹配模型，从左右目图像对估计视差图 |

### 2.11 预转换 RKNN 模型

用户可以从 [RKNN3_SDK 网盘](https://console.box.lenovo.com/l/H1fig1) 下载预先转换好的 RKNN 模型（提取码：`rknn`）。本次发布的模型位于 `RKNN3_SDK/rknn3_models/v1.1.0` 目录。

## 3. 支持的平台

| 主芯片 | 协处理器 | 操作系统 |
|--------|----------|----------|
| RK3588 系列 | RK1820 / RK1828 | Linux / Android |
| RK3576 系列 | RK1820 / RK1828 | Linux / Android |
| RK3572 系列 | - | Linux / Android |

> **构建与运行时库说明**：
> - 顶层构建脚本 `build-linux.sh` / `build-android.sh` 的 `-t` 参数支持 `rk3588`、`rk3576`、`rk3572`、`x86`。
> - 示例安装目录 `lib/` 中的 RKNN3 Runtime 库按 SoC 区分：
>   - `RK3588` / `RK3576`：安装 `librknn3_api.so` 和 `librknn3_api_rkcp.so`
>   - `RK3572`：安装 `librknn3_api.so` 和 `librknn3_api_native.so`

## 4. 快速开始

### 4.1 环境准备

> **环境要求**：Python 3.10

```bash
cd rknn3_model_zoo/
pip install -r requirements.txt
export PYTHONPATH=./
```

### 4.2 通用部署流程

所有模型遵循统一的四步流程：**导出 ONNX → 转换 RKNN → 编译 → 运行**。

#### LLM 模型（以 Qwen2.5-3B 为例）

```bash
# Step 1: 导出 ONNX 模型（--quant 启用 GRQ 量化）
cd examples/Qwen2_5/python/
python export_llm.py --quant

# Step 2: 转换 RKNN 模型
python export_rknn.py

# Step 3: 编译
cd ../../../
export GCC_COMPILER=<GCC_COMPILER_PATH>     # 可选：指定交叉编译器路径
./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5

# Step 4: 推送并运行
adb push install/rk3588_linux_aarch64/rknn_Qwen2_5_demo/ /data/
adb shell
cd /data/rknn_Qwen2_5_demo
export LD_LIBRARY_PATH=./lib
./rknn_qwen2_5_demo \
    model/Qwen2.5-0.5B-Instruct.rknn \
    model/Qwen2.5-0.5B-Instruct.weight \
    model/Qwen2.5-0.5B-Instruct.tokenizer.gguf \
    model/Qwen2.5-0.5B-Instruct.embed.bin \
    0xff \
    "Who are you?"
```

#### VLM 模型（以 Qwen3-VL-4B 为例）

```bash
# Step 1: 导出 ONNX 模型（Vision + LLM 分别导出，--quant 需先生成量化校准数据）
cd examples/Qwen3_VL/python/llm/
python make_calidata.py --model_path Qwen/Qwen3-VL-4B-Instruct --modelscope
python export_llm.py --quant \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_llm_path Qwen3-VL-4B-llm.onnx --modelscope
cd ../vision/
python make_calidata.py --model_path Qwen/Qwen3-VL-4B-Instruct --modelscope
python export_vision.py --quant \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_vision_path Qwen3-VL-4B-vision.onnx --modelscope

# Step 2: 转换 RKNN 模型（Vision + LLM 分别转换）
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-vision.onnx \
    --rknn_path Qwen3-VL-4B-vision.rknn --platform rk1820   # Vision RKNN
cd ../llm/
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm.rknn --platform rk1820       # LLM RKNN

# Step 3: 编译
cd ../../../
export GCC_COMPILER=<GCC_COMPILER_PATH>     # 可选
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_VL

# Step 4: 推送并运行
adb push install/rk3588_linux_aarch64/rknn_Qwen3_VL_demo/ /data/
adb shell
cd /data/rknn_Qwen3_VL_demo
export LD_LIBRARY_PATH=./lib
./rknn_qwen3_vl_demo \
    model/Qwen3-VL-4B-vision.rknn model/Qwen3-VL-4B-vision.weight \
    model/Qwen3-VL-4B-llm.rknn model/Qwen3-VL-4B-llm.weight \
    model/Qwen3-VL-4B-llm.tokenizer.gguf model/Qwen3-VL-4B-llm.embed.bin \
    0xff 0xff \
    model/demo.jpg \
    "请描述这张图片"
```

> 各模型的具体参数和差异请参考对应示例目录下的 README。

### 4.3 常用参数

各模型的导出脚本共享以下常用参数（具体默认值因模型而异，详见各示例目录下的 README）：

| 参数 | 说明 |
|------|------|
| `--model_path` | 模型路径或 HuggingFace 名称 |
| `--quant` | 启用 GRQ 量化算法（需 CUDA 环境） |
| `--modelscope` | 从 ModelScope 下载模型（国内用户推荐） |
| `--export_llm_path` / `--export_vision_path` | ONNX 导出路径 |
| `--platform` / `--target_platform` | RKNN 目标平台（`rk1820` / `rk1828` / `rk3572`） |
| `--load_weight` | 是否加载模型权重（`False` 时仅导出模型结构） |

> **说明**：
> - 使用 GRQ 量化时，模型自带量化参数，转换 RKNN 时无需量化数据集
> - RKNN 转换采用权重分离模式，生成 `.rknn` 和 `.weight` 两个文件
> - LLM 模型导出还包括 Config（`.config.pkl`）、Tokenizer（`.tokenizer.gguf`）、Embed（`.embed.bin`）文件

## 5. 高级功能

### 5.1 SpeedUP 推理加速

Qwen2.5-VL、Qwen3-VL 和 Qwen3.5-VL 示例可链接 SpeedUP 库，用于推理加速。

#### 文件位置

发布包中需要保留以下文件：

```text
3rdparty/SpeedUP/
├── include/speedup.h
├── Linux/aarch64/libSpeedUP.so
└── Android/arm64-v8a/libSpeedUP.so
```

#### 编译

```bash
# Qwen2.5-VL
./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5_VL --enable_speedup

# Qwen3-VL
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_VL --enable_speedup

# Qwen3.5-VL
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_5_VL --enable_speedup

# Android（先将 ANDROID_NDK_PATH 设置为 NDK 安装目录）
./build-android.sh -t rk3588 -a arm64-v8a -d Qwen3_5_VL --enable_speedup
```

编译完成后，`libSpeedUP.so` 会安装到对应 demo 的 `lib/` 目录。

默认编译普通推理版本。添加 `--enable_speedup` 启用 SpeedUP，启用前需准备好 `speedup.h` 和对应平台的 `libSpeedUP.so`。不添加该参数或使用 `--disable_speedup`，则编译普通推理版本。

#### 运行参数

```bash
./rknn_qwen2_5_vl_demo \    # 或 ./rknn_qwen3_vl_demo、./rknn_qwen3_5_vl_demo
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt> \
    [model_width model_height] [speedup_ratio]
```

`speedup_ratio` 为可选参数：

| 参数值 | 模式 |
|--------|------|
| `1.0` | 自动模式 |
| `0.0` | 关闭 |
| `(0.0, 1.0)` | 手动模式 |

RKNN3 多核设备通常可使用：

```bash
0xff 0xff
```

#### 运行示例

Qwen2.5-VL：

```bash
cd /userdata/rknn3-model-zoo/install/rk3588_linux_aarch64/rknn_Qwen2_5_VL_demo
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

./rknn_qwen2_5_vl_demo \
    /userdata/Qwen2.5-VL-3B/Qwen2.5-VL-3B-vision.rknn \
    /userdata/Qwen2.5-VL-3B/Qwen2.5-VL-3B-vision.weight \
    /userdata/Qwen2.5-VL-3B/Qwen2.5-VL-3B-llm.rknn \
    /userdata/Qwen2.5-VL-3B/Qwen2.5-VL-3B-llm.weight \
    /userdata/Qwen2.5-VL-3B/Qwen2.5-VL-3B-llm.tokenizer.gguf \
    /userdata/Qwen2.5-VL-3B/Qwen2.5-VL-3B-llm.embed.bin \
    0xff 0xff \
    /userdata/rknn3-model-zoo/examples/Qwen2_5_VL/data/vision/demo.jpg \
    "请描述这张图片" \
    392 392 \
    1.0
```

Qwen3-VL：

```bash
cd /userdata/rknn3-model-zoo/install/rk3588_linux_aarch64/rknn_Qwen3_VL_demo
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

./rknn_qwen3_vl_demo \
    /userdata/Qwen3-VL-model/Qwen3-VL-4B-vision_384_384.rknn \
    /userdata/Qwen3-VL-model/Qwen3-VL-4B-vision_384_384.weight \
    /userdata/Qwen3-VL-model/Qwen3-VL-4B-llm.rknn \
    /userdata/Qwen3-VL-model/Qwen3-VL-4B-llm.weight \
    /userdata/Qwen3-VL-model/Qwen3-VL-4B-llm.tokenizer.gguf \
    /userdata/Qwen3-VL-model/Qwen3-VL-4B-llm.embed.bin \
    0xff 0xff \
    /userdata/rknn3-model-zoo/examples/Qwen2_5_VL/data/vision/demo.jpg \
    "请描述这张图片" \
    384 384 \
    1.0
```

Qwen3.5-VL：

```bash
cd /userdata/rknn3-model-zoo/install/rk3588_linux_aarch64/rknn_Qwen3_5_VL_demo
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

./rknn_qwen3_5_vl_demo \
    /userdata/Qwen3.5-VL-model/Qwen3.5-2B-vision.rknn \
    /userdata/Qwen3.5-VL-model/Qwen3.5-2B-vision.weight \
    /userdata/Qwen3.5-VL-model/Qwen3.5-2B-llm.rknn \
    /userdata/Qwen3.5-VL-model/Qwen3.5-2B-llm.weight \
    /userdata/Qwen3.5-VL-model/Qwen3.5-2B-llm.tokenizer.gguf \
    /userdata/Qwen3.5-VL-model/Qwen3.5-2B-llm.embed.bin \
    0xff 0xff \
    /userdata/rknn3-model-zoo/examples/Qwen3_5_VL/data/vision/demo.jpg \
    "请描述这张图片" \
    384 384 \
    1.0
```

### 5.2 多卡推理

本示例将 LLM 模型按 Transformer layer 边界切分为多段（segment），每段部署在一张 RK182X 加速卡上，通过流水线并行实现多卡协同推理，支持更大参数量的模型并提高 prefill 性能。

当前支持以下模型：

| 模型 | RK182X 加速卡数量 |
|------|-------------------|
| `Qwen/Qwen3.5-9B` | 2 |
| `google/gemma-4-12B-it` | 2 |
| `Qwen/Qwen3.5-27B` | 4 |
| `Qwen/Qwen3.8-27B` | 4 |
| `google/gemma-4-31B-it` | 4 |

#### 模型切分原理

分段在 layer 之间切分，不改变层内计算顺序；最后一段除分配的 Transformer layer 外还固定包含最终 `norm` 和 `lm_head`。导出脚本并非简单按 `总层数 / 段数` 平均分配，而是估算每个 layer 的权重（Transformer layer 按 W4A16/group32、`lm_head` 按 W6A16/group32、最终 `norm` 按 FP16 估算），自动寻找更均衡的切分边界，使各段的估算总权重尽量接近。`--num_segments N` 用于指定段数。

#### 模型转换

```bash
# 以 Qwen3.5-9B 拆分为 2 段为例
cd examples/multicard/python/qwen3_5

# 导出分段 ONNX 模型
python export_llm_segment.py --model_path /path/to/Qwen3.5-9B --multi_segment --num_segments 2

# 导出分段 RKNN 模型
python export_rknn_segment.py --multi_segment --num_segments 2
```

Qwen3.5 和 Gemma-4 默认使用外置 rope cache，每个分段都会生成内容相同的 `.safetensors` 文件，运行时通过 `--rope-tensor` 传入任意一个即可。

#### 编译与运行

```bash
# 编译
./build-linux.sh -t rk3588 -a aarch64 -d multicard

# 运行（2 段 Qwen3.5-9B）
cd /data/rknn_multicard_demo
export LD_LIBRARY_PATH=./lib
taskset f0 ./rknn_multicard_demo \
    --model /data/models/multicard/Qwen3.5-9B-llm_seg0.rknn \
    --weight /data/models/multicard/Qwen3.5-9B-llm_seg0.weight \
    --vocab /data/models/multicard/Qwen3.5-9B-llm.tokenizer.gguf \
    --embed /data/models/multicard/Qwen3.5-9B-llm.embed.bin \
    -c 4096 \
    --core-mask 0xff \
    --stage-count 2 \
    --bucket-size 128 \
    --rope-tensor /data/models/multicard/Qwen3.5-9B-llm_seg0.safetensors \
    --prompt "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n" \
    --predict 128
```

只需传入 `seg0` 的模型/权重路径，`seg1..segN` 的路径会按 `_segN` 后缀自动推导。

> 完整的参数说明、Gemma-4 示例、KV Cache 重建及多卡推理代码逻辑详见 [`examples/multicard/README.md`](examples/multicard/README.md)。

### 5.3 LoRA 支持

RKNN3 支持 LoRA（Low-Rank Adaptation）适配器加载，可在不修改基座模型权重的前提下，将 LoRA 权重叠加到 RKNN 模型中实现特定任务的微调推理。目前以 Qwen3-VL 为示例。

#### 模型导出

LoRA 权重（如 `adapter_model.safetensors`）无需单独转换为 ONNX，在 `export_rknn.py` 中直接传入即可：

```bash
cd examples/Qwen3_VL_LoRA/python/llm

# 导出含 LoRA 的 RKNN 模型
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm-lora.rknn \
    --lora_path /path/to/lora/adapter_model.safetensors \
    --lora_config_path /path/to/lora/adapter_config.json
```

导出后生成 `.rknn` 和 `.lora_weight` 两个文件，C++ 推理时需同时提供。

#### 编译与运行

```bash
# 编译
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_VL_LoRA

# 运行（Base + LoRA 双路对比）
cd /data/rknn_Qwen3_VL_LoRA_demo
export LD_LIBRARY_PATH=./lib
./rknn_qwen3_vl_demo \
    ./model/vision.rknn ./model/vision.weight \
    ./model/llm.rknn ./model/llm.weight \
    ./model/tokenizer.gguf ./model/embed.bin \
    0x3 0x3 \
    ./model/demo.jpg "描述这张图片" \
    384 384 \
    2048 3072 \
    ./model/llm_lora.weight
```

程序会依次输出 Base model 和 LoRA model 的推理结果与性能统计。

#### LoRA 相关 API

| 接口 | 说明 |
|------|------|
| `rknn3_lora_init(ctx, lora_weight_path)` | 从文件初始化 LoRA 环境 |
| `rknn3_lora_load(ctx, lora)` | 将 LoRA 适配器加载进 context |
| `rknn3_session_enable_lora(session, lora)` | 为指定 session 启用 LoRA（自动清空 kvcache） |
| `rknn3_session_disable_lora(session, lora)` | 为指定 session 关闭 LoRA |
| `rknn3_lora_unload(ctx, lora)` | 卸载 LoRA 适配器 |

**推荐调用顺序**：`lora_init` -> `query(LORA_NUM)` -> `query(LORA_INFO)` -> `lora_load` -> `session_enable_lora` -> `lora_unload`

> 完整的参数说明、API 详情及双路推理实现详见 [`examples/Qwen3_VL_LoRA/README.md`](examples/Qwen3_VL_LoRA/README.md)。

## 6. 模型适配指南

- **同系列模型兼容**：同一系列的示例程序相互兼容。例如，Qwen2.5-0.5B 的示例可直接用于 Qwen2.5-7B，只需修改模型加载路径。

- **新模型适配**：对于本仓库未收录的 LLM 模型，请参考 [LLM 模型适配教程](LLM_model_modification_guide_CN.md) 进行 ONNX 导出和部署移植。

## 7. 注意事项

- **Transformers 版本**：不同模型依赖的 `transformers` 版本可能不同，导出 ONNX 前请安装对应版本。版本信息可从模型的 config.json（例如 https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json ）中的 `transformers_version` 字段获取。部分模型有特殊版本要求，见各示例目录下的 `requirements.txt`。

- **PyTorch 版本**：建议使用 PyTorch <= 2.8.0（Qwen3-VL、Gemma-4 等模型需 PyTorch >= 2.9.0，PaddleOCR-VL 需 transformers == 4.55.0，具体看对应模型下的 requirements.txt）

- **模块兼容性**：Gemma-4 的 Audio 模型与 LLM 模型需使用同一版本（同为 E2B 或同为 E4B），不可混用。

## 8. 其他说明

本仓库默认使用以下镜像站点获取模型：
- [ModelScope](https://www.modelscope.cn)（国内推荐）
- [HuggingFace Mirror](https://hf-mirror.com)
