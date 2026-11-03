[中文](README_CN.md)

# RKNN3-MODEL-ZOO

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Supported Models](#2-supported-models)
  - [2.1 LLM](#21-llm)
  - [2.2 VLM](#22-vlm)
  - [2.3 Omni](#23-omni)
  - [2.4 ASR (Speech Recognition)](#24-asr-speech-recognition)
  - [2.5 TTS (Text-to-Speech)](#25-tts-text-to-speech)
  - [2.6 Embedding / Reranker](#26-embedding--reranker)
  - [2.7 Translation](#27-translation)
  - [2.8 OCR](#28-ocr)
  - [2.9 JEV (Decision Models)](#29-jev-decision-models)
  - [2.10 CV (Computer Vision)](#210-cv-computer-vision)
  - [2.11 Pre-converted RKNN Models](#211-pre-converted-rknn-models)
- [3. Supported Platforms](#3-supported-platforms)
- [4. Quick Start](#4-quick-start)
  - [4.1 Environment Setup](#41-environment-setup)
  - [4.2 General Deployment Flow](#42-general-deployment-flow)
  - [4.3 Common Parameters](#43-common-parameters)
- [5. Advanced Features](#5-advanced-features)
  - [5.1 SpeedUP Inference Acceleration](#51-speedup-inference-acceleration)
  - [5.2 Multi-card Inference](#52-multi-card-inference)
  - [5.3 LoRA Support](#53-lora-support)
- [6. Model Adaptation Guide](#6-model-adaptation-guide)
- [7. Important Notes](#7-important-notes)
- [8. Additional Notes](#8-additional-notes)


## 1. Introduction

RKNN3 SDK provides the complete software stack for deploying AI models on RK1820/RK1828 coprocessors, including:

- **[RKNN3-Toolkit](https://github.com/airockchip/rknn3-toolkit)**: PC-side software development kit for model conversion, inference, performance evaluation, etc.
- **RKNN3 Runtime**: On-board runtime library providing C/C++ programming interfaces for deploying RKNN models and accelerating AI applications.
- **[RKNN3 Model Zoo](https://github.com/airockchip/rknn3-model-zoo)**: Model conversion and deployment example repository, including reference implementations for CNN / LLM / VLM and other models.
- **[RKNN3 MCP](https://github.com/airockchip/rknn3-toolkit/blob/main/doc/EN/09_RKNN3_MCP_QuickStart_V0.1.0.pdf)**: RKNN3 MCP is a model deployment assistant for the Rockchip RKNN3 platform. Through AI development tools that support MCP, users can describe their requirements in natural language, and the Agent — guided by RKNN3 MCP — helps complete model analysis, ONNX export, RKNN3 model conversion, accuracy analysis, and on-board deployment, working alongside the export and deployment examples in this repository. RKNN3 MCP covers CNN, ViT, LLM, VLM, ASR, and other model workflows, and supports LoRA conversion and deployment as well as multi-NPU deployment and inference.

This repository provides a complete model deployment workflow:
- **Model Export**: Export HuggingFace / PyTorch models to ONNX format
- **Model Conversion**: Convert ONNX models to RKNN format using RKNN3 Toolkit
- **On-board Deployment**: Provide C++ inference example code

## 2. Supported Models

### 2.1 LLM

| Model | Example Directory | Description |
|-------|-------------------|-------------|
| [GLM-Edge](examples/glm_edge/README.md) | `examples/glm_edge` | GLM Edge series on-device LLM |
| [MiniCPM5](examples/MiniCPM5/README.md) | `examples/MiniCPM5` | MiniCPM5 LLM |
| [MiniCPM5-2B](examples/MiniCPM5_2B/README.md) | `examples/MiniCPM5_2B` | MiniCPM5-2B LLM |
| [Nanbeige4.2](examples/Nanbeige4_2/README.md) | `examples/Nanbeige4_2` | Nanbeige Looped Transformer LLM |
| [Qwen2.5](examples/Qwen2_5/README.md) | `examples/Qwen2_5` | Tongyi Qianwen 2.5 series LLM |
| [Qwen3](examples/Qwen3/README.md) | `examples/Qwen3` | Tongyi Qianwen 3 series LLM |
| [Qwen3.5](examples/Qwen3_5/README.md) | `examples/Qwen3_5` | Tongyi Qianwen 3.5 series LLM |
| [FunctionGemma](examples/functiongemma/README.md) | `examples/functiongemma` | Google FunctionGemma function-calling model |
| [LFM2.5](examples/LFM2_5/README.md) | `examples/LFM2_5` | LiquidAI LFM2.5 hybrid-architecture LLM (Full Attention + streaming short convolution) |
| [Spark-X2.5](examples/Spark_X2_5/README.md) | `examples/Spark_X2_5` | Spark-X2.5 hybrid-attention LLM (full attention + sliding window, 1.7B / 4B) |

### 2.2 VLM

| Model | Example Directory | Description |
|-------|-------------------|-------------|
| [FastVLM](examples/FastVLM/README.md) | `examples/FastVLM` | Apple lightweight vision-language model |
| [LFM2.5-VL](examples/LFM2_5_VL/README.md) | `examples/LFM2_5_VL` | LiquidAI LFM2.5-VL vision-language model (1.6B / 3B) |
| [SmolVLM](examples/SmolVLM/README.md) | `examples/SmolVLM` | HuggingFace lightweight vision-language model |
| [SmolVLM2](examples/SmolVLM2/README.md) | `examples/SmolVLM2` | HuggingFace SmolVLM second-generation vision-language model |
| [GME-Qwen2-VL](examples/GME-Qwen2-VL/README.md) | `examples/GME-Qwen2-VL` | Tongyi Qianwen vision embedding model |
| [InternVLM](examples/InternVLM/README.md) | `examples/InternVLM` | InternVL vision-language model |
| [Janus-Pro](examples/Janus_Pro/README.md) | `examples/Janus_Pro` | DeepSeek multimodal understanding and generation model |
| [Mage-VL](examples/Mage_VL/README.md) | `examples/Mage_VL` | Microsoft Mage-VL vision-language model |
| [LocateAnything](examples/LocateAnything/README.md) | `examples/LocateAnything` | NVIDIA visual grounding multimodal model (object detection / phrase grounding / GUI grounding) |
| [MiniCPM-V](examples/MiniCPM_V/README.md) | `examples/MiniCPM_V` | OpenBMB MiniCPM-V vision-language model |
| [MiniCPM-V-4](examples/MiniCPM_V_4/README.md) | `examples/MiniCPM_V_4` | MiniCPM-V-4 vision-language model |
| [MiniCPM-V-4.7-1B](examples/MiniCPM_V_4.7_1B/README.md) | `examples/MiniCPM_V_4.7_1B` | MiniCPM-V-4.7-1B vision-language model |
| [Qwen2.5-VL](examples/Qwen2_5_VL/README.md) | `examples/Qwen2_5_VL` | Tongyi Qianwen 2.5 vision-language model |
| [Qwen3-VL](examples/Qwen3_VL/README.md) | `examples/Qwen3_VL` | Tongyi Qianwen 3 vision-language model |
| [Qwen3-VL-LoRA](examples/Qwen3_VL_LoRA/README.md) | `examples/Qwen3_VL_LoRA` | Tongyi Qianwen 3-VL LoRA fine-tuned model |
| [Qwen3.5-VL](examples/Qwen3_5_VL/README.md) | `examples/Qwen3_5_VL` | Tongyi Qianwen 3.5 vision-language model |
| [UI-TARS](examples/UI_TARS/README.md) | `examples/UI_TARS` | ByteDance GUI Agent vision-language model |

### 2.3 Omni

| Model | Example Directory | Description |
|-------|-------------------|-------------|
| [Gemma-4](examples/gemma4/README.md) | `examples/gemma4` | Google Gemma-4 multimodal model (text + audio) |
| [Qwen2.5-Omni](examples/Qwen2_5_Omni/README.md) | `examples/Qwen2_5_Omni` | Tongyi Qianwen 2.5 omni model |
| Qwen3-Omni | Commercial closed-source model | Tongyi Qianwen 3 omni model |
| Qwen3.5-Omni | Commercial closed-source model | Tongyi Qianwen 3.5 omni model |

### 2.4 ASR (Speech Recognition)

| Model | Example Directory | Description |
|-------|-------------------|-------------|
| [Paraformer](examples/paraformer/README.md) | `examples/paraformer` | FunASR SeaCo-Paraformer Chinese offline speech recognition |
| [Qwen3-ASR](examples/Qwen3_ASR/README.md) | `examples/Qwen3_ASR` | Tongyi Qianwen speech recognition (streaming / non-streaming) |
| [Qwen3-ForcedAligner](examples/Qwen3_ForcedAligner/README.md) | `examples/Qwen3_ForcedAligner` | Audio-text forced alignment with per-character (Chinese) / per-word (English) timestamps |
| [SenseVoiceSmall](examples/sensevoice_small/README.md) | `examples/sensevoice_small` | Alibaba FunAudioLLM speech recognition model |
| [WeNet (Conformer)](examples/wenet/README.md) | `examples/wenet` | WeNet U2++ Conformer streaming Chinese speech recognition |
| [Whisper](examples/whisper/README.md) | `examples/whisper` | OpenAI multilingual speech recognition model |
| [Zipformer](examples/zipformer/README.md) | `examples/zipformer` | Zipformer streaming speech recognition |

### 2.5 TTS (Text-to-Speech)

| Model | Example Directory | Description |
|-------|-------------------|-------------|
| [Qwen3-TTS](examples/Qwen3_TTS/README.md) | `examples/Qwen3_TTS` | Tongyi Qianwen text-to-speech |
| [VITS](examples/vits/README.md) | `examples/vits` | VITS speech synthesis (LJSpeech / VCTK) |

### 2.6 Embedding / Reranker

| Model | Example Directory | Description |
|-------|-------------------|-------------|
| [Qwen3-Embedding](examples/Qwen3_Embedding/README.md) | `examples/Qwen3_Embedding` | Tongyi Qianwen text embedding model |
| [Qwen3-Reranker](examples/Qwen3_Reranker/README.md) | `examples/Qwen3_Reranker` | Tongyi Qianwen reranker model |
| [WeMM-Embedding-2B](examples/WeMM-Embedding/README.md) | `examples/WeMM-Embedding` | Tencent multimodal embedding model (text / image / video to 2048-dim vector) |

### 2.7 Translation

| Model | Example Directory | Description |
|-------|-------------------|-------------|
| [HY-MT1.5](examples/HY_MT_1_5/README.md) | `examples/HY_MT_1_5` | Hunyuan multilingual translation model |

### 2.8 OCR

| Model | Example Directory | Description |
|-------|-------------------|-------------|
| [dots.mocr](examples/dots.mocr/README.md) | `examples/dots.mocr` | rednote dots.mocr multilingual OCR vision-language model |
| [dots.ocr](examples/dots.ocr/README.md) | `examples/dots.ocr` | rednote dots.ocr document layout parsing and OCR vision-language model |
| [HunyuanOCR-1.5](examples/HY_OCR_1_5/README.md) | `examples/HY_OCR_1_5` | Tencent HunyuanOCR-1.5 multimodal OCR model |
| [PaddleOCR-VL](examples/paddleocr_vl/README.md) | `examples/paddleocr_vl` | Baidu PaddleOCR-VL visual OCR model |
| [Unlimited-OCR](examples/Unlimited-OCR/README.md) | `examples/Unlimited-OCR` | Unlimited-OCR model (SAM + Vision + LLM split deployment) |

### 2.9 JEV (Decision Models)

JEV models are essentially decision models: instead of generating free-form text, they turn text or image-text input into structured decisions such as classification (choice), yes/no judgment (with probability), and scoring.

| Model | Example Directory | Description |
|-------|-------------------|-------------|
| [Intern-Decision](examples/Intern_Decision/README.md) | `examples/Intern_Decision` | Intern-Decision text / image-text decision model (classification, yes/no judgment, scoring) |
| [Laya](examples/Laya/README.md) | `examples/Laya` | Laya decision model for choice / scoring / true-false questions (task-config driven) |

### 2.10 CV (Computer Vision)

| Model | Example Directory | Description |
|-------|-------------------|-------------|
| [MobileNetV2](examples/mobilenet_v2/README.md) | `examples/mobilenet_v2` | Lightweight image classification model |
| [ResNet](examples/resnet/README.md) | `examples/resnet` | Classic residual image classification model |
| [YOLOv5](examples/yolov5/README.md) | `examples/yolov5` | Object detection model |
| [YOLOv6](examples/yolov6/README.md) | `examples/yolov6` | Object detection model |
| [YOLOv8](examples/yolov8/README.md) | `examples/yolov8` | Object detection model |
| [YOLO26](examples/yolo26/README.md) | `examples/yolo26` | Ultralytics YOLO26 object detection model (yolo26n/s/m) |
| [YOLO26-Segment](examples/yolo26_segment/README.md) | `examples/yolo26_segment` | Ultralytics YOLO26 instance segmentation model |
| [YOLO26-Pose](examples/yolo26_pose/README.md) | `examples/yolo26_pose` | Ultralytics YOLO26 human pose estimation model |
| [QA-CLIP](examples/QAClip/README.md) | `examples/QAClip` | Chinese-English image-text similarity model |
| [Depth Anything V3](examples/depth_anything_v3/README.md) | `examples/depth_anything_v3` | Multi-view stereo depth estimation model |
| [RAFT-Stereo](examples/RAFT-Stereo/README.md) | `examples/RAFT-Stereo` | Stereo matching model for disparity estimation from left/right image pairs |

### 2.11 Pre-converted RKNN Models

Users can download pre-converted RKNN models from the [RKNN3_SDK cloud drive](https://console.box.lenovo.com/l/H1fig1) (access code: `rknn`). The models for this release are available under `RKNN3_SDK/rknn3_models/v1.1.0`.

## 3. Supported Platforms

| Host SoC | Coprocessor | OS |
|----------|-------------|-----|
| RK3588 Series | RK1820 / RK1828 | Linux / Android |
| RK3576 Series | RK1820 / RK1828 | Linux / Android |
| RK3572 Series | - | Linux / Android |

> **Build and runtime library notes**:
> - The top-level build scripts `build-linux.sh` / `build-android.sh` accept `-t` with `rk3588`, `rk3576`, `rk3572`, and `x86`.
> - RKNN3 runtime libraries installed into each demo's `lib/` directory are SoC-specific:
>   - `RK3588` / `RK3576`: `librknn3_api.so` and `librknn3_api_rkcp.so`
>   - `RK3572`: `librknn3_api.so` and `librknn3_api_native.so`

## 4. Quick Start

### 4.1 Environment Setup

> **Requirements**: Python 3.10

```bash
cd rknn3_model_zoo/
pip install -r requirements.txt
export PYTHONPATH=./
```

### 4.2 General Deployment Flow

All models follow a unified four-step flow: **Export ONNX -> Convert RKNN -> Build -> Run**.

#### LLM Model (Example: Qwen2.5-3B)

```bash
# Step 1: Export ONNX model (--quant enables GRQ quantization)
cd examples/Qwen2_5/python/
python export_llm.py --quant

# Step 2: Convert to RKNN model
python export_rknn.py

# Step 3: Build
cd ../../../
export GCC_COMPILER=<GCC_COMPILER_PATH>     # Optional: specify cross-compiler path
./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5

# Step 4: Push and run
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

#### VLM Model (Example: Qwen3-VL-4B)

```bash
# Step 1: Export ONNX models (Vision + LLM exported separately; --quant requires calibration data generated first)
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

# Step 2: Convert to RKNN models (Vision + LLM converted separately)
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-vision.onnx \
    --rknn_path Qwen3-VL-4B-vision.rknn --platform rk1820   # Vision RKNN
cd ../llm/
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm.rknn --platform rk1820       # LLM RKNN

# Step 3: Build
cd ../../../
export GCC_COMPILER=<GCC_COMPILER_PATH>     # Optional
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_VL

# Step 4: Push and run
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
    "Please describe this image"
```

> For model-specific parameters and differences, refer to the README in each example directory.

### 4.3 Common Parameters

The export scripts of each model share the following common parameters (specific default values vary by model; see the README in each example directory):

| Parameter | Description |
|-----------|-------------|
| `--model_path` | Model path or HuggingFace name |
| `--quant` | Enable GRQ quantization algorithm (requires CUDA environment) |
| `--modelscope` | Download model from ModelScope (recommended for China users) |
| `--export_llm_path` / `--export_vision_path` | ONNX export path |
| `--platform` / `--target_platform` | RKNN target platform (`rk1820` / `rk1828` / `rk3572`) |
| `--load_weight` | Whether to load model weights (`False` exports structure only) |

> **Notes**:
> - When using GRQ quantization, the model contains quantization parameters; no quantization dataset is needed for RKNN conversion
> - RKNN conversion uses weight-separated mode, generating both `.rknn` and `.weight` files
> - LLM model export also includes Config (`.config.pkl`), Tokenizer (`.tokenizer.gguf`), and Embed (`.embed.bin`) files

## 5. Advanced Features

### 5.1 SpeedUP Inference Acceleration

The Qwen2.5-VL, Qwen3-VL, and Qwen3.5-VL examples can link against the SpeedUP library for inference acceleration.

#### File Location

Keep the following files in the release package:

```text
3rdparty/SpeedUP/
├── include/speedup.h
├── Linux/aarch64/libSpeedUP.so
└── Android/arm64-v8a/libSpeedUP.so
```

#### Build

```bash
# Qwen2.5-VL
./build-linux.sh -t rk3588 -a aarch64 -d Qwen2_5_VL --enable_speedup

# Qwen3-VL
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_VL --enable_speedup

# Qwen3.5-VL
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_5_VL --enable_speedup

# Android (set ANDROID_NDK_PATH to your NDK installation first)
./build-android.sh -t rk3588 -a arm64-v8a -d Qwen3_5_VL --enable_speedup
```

After installation, `libSpeedUP.so` is copied into the corresponding demo `lib/` directory.

SpeedUP is disabled by default. Add `--enable_speedup` to enable it; provide `speedup.h` and `libSpeedUP.so` for the target platform before building. Omit this flag or use `--disable_speedup` to build for normal inference.

#### Runtime Arguments

```bash
./rknn_qwen2_5_vl_demo \    # or ./rknn_qwen3_vl_demo, ./rknn_qwen3_5_vl_demo
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt> \
    [model_width model_height] [speedup_ratio]
```

`speedup_ratio` is optional:

| Value | Mode |
|-------|------|
| `1.0` | Auto |
| `0.0` | Disabled |
| `(0.0, 1.0)` | Manual |

For RKNN3 multi-core devices, this is usually suitable:

```bash
0xff 0xff
```

#### Examples

Qwen2.5-VL:

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
    "Describe this image" \
    392 392 \
    1.0
```

Qwen3-VL:

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
    "Describe this image" \
    384 384 \
    1.0
```

Qwen3.5-VL:

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
    "Describe this image" \
    384 384 \
    1.0
```

### 5.2 Multi-card Inference

This example splits an LLM model into multiple segments (stages) at Transformer layer boundaries, with each segment deployed on one RK182X accelerator card. Through pipeline parallelism, it enables multi-card collaborative inference, supporting larger models and improving prefill performance.

Currently supported models:

| Model | RK182X Accelerator Cards |
|-------|--------------------------|
| `Qwen/Qwen3.5-9B` | 2 |
| `google/gemma-4-12B-it` | 2 |
| `Qwen/Qwen3.5-27B` | 4 |
| `Qwen/Qwen3.8-27B` | 4 |
| `google/gemma-4-31B-it` | 4 |

#### Model Splitting Principle

Segments are cut between layers without changing the intra-layer computation order; the last segment always contains the final `norm` and `lm_head` on top of its assigned Transformer layers. The export script does not simply split by `total_layers / num_segments` — it estimates each layer's weight (Transformer layers as W4A16/group32, `lm_head` as W6A16/group32, final `norm` as FP16) and automatically searches for balanced split boundaries so the estimated total weight of every segment is as close as possible. `--num_segments N` specifies the number of segments.

#### Model Conversion

```bash
# Example: splitting Qwen3.5-9B into 2 segments
cd examples/multicard/python/qwen3_5

# Export segmented ONNX model
python export_llm_segment.py --model_path /path/to/Qwen3.5-9B --multi_segment --num_segments 2

# Export segmented RKNN model
python export_rknn_segment.py --multi_segment --num_segments 2
```

Qwen3.5 and Gemma-4 use an external rope cache by default. Each segment produces an identical `.safetensors` file; pass any one of them via `--rope-tensor` at runtime.

#### Build and Run

```bash
# Build
./build-linux.sh -t rk3588 -a aarch64 -d multicard

# Run (2-segment Qwen3.5-9B)
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
    --prompt "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n" \
    --predict 128
```

Only the seg0 model/weight paths need to be passed; the paths of `seg1..segN` are derived automatically from the `_segN` suffix.

> For complete parameter descriptions, Gemma-4 examples, KV Cache rebuild, and multi-card inference code logic, see [`examples/multicard/README.md`](examples/multicard/README.md).

### 5.3 LoRA Support

RKNN3 supports LoRA (Low-Rank Adaptation) adapter loading, which overlays LoRA weights onto the RKNN model without modifying the base model weights, enabling task-specific fine-tuned inference. Currently, Qwen3-VL is provided as an example.

#### Model Export

LoRA weights (e.g., `adapter_model.safetensors`) do not need to be converted to ONNX separately; they can be passed directly to `export_rknn.py`:

```bash
cd examples/Qwen3_VL_LoRA/python/llm

# Export RKNN model with LoRA
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm-lora.rknn \
    --lora_path /path/to/lora/adapter_model.safetensors \
    --lora_config_path /path/to/lora/adapter_config.json
```

After export, two files are generated: `.rknn` and `.lora_weight`. Both must be provided during C++ inference.

#### Build and Run

```bash
# Build
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_VL_LoRA

# Run (Base + LoRA dual-path comparison)
cd /data/rknn_Qwen3_VL_LoRA_demo
export LD_LIBRARY_PATH=./lib
./rknn_qwen3_vl_demo \
    ./model/vision.rknn ./model/vision.weight \
    ./model/llm.rknn ./model/llm.weight \
    ./model/tokenizer.gguf ./model/embed.bin \
    0x3 0x3 \
    ./model/demo.jpg "Describe this image" \
    384 384 \
    2048 3072 \
    ./model/llm_lora.weight
```

The program will sequentially output inference results and performance statistics for both the Base model and LoRA model.

#### LoRA-related API

| API | Description |
|-----|-------------|
| `rknn3_lora_init(ctx, lora_weight_path)` | Initialize LoRA environment from file |
| `rknn3_lora_load(ctx, lora)` | Load LoRA adapter into context |
| `rknn3_session_enable_lora(session, lora)` | Enable LoRA for the specified session (automatically clears kvcache) |
| `rknn3_session_disable_lora(session, lora)` | Disable LoRA for the specified session |
| `rknn3_lora_unload(ctx, lora)` | Unload LoRA adapter |

**Recommended call sequence**: `lora_init` -> `query(LORA_NUM)` -> `query(LORA_INFO)` -> `lora_load` -> `session_enable_lora` -> `lora_unload`

> For complete parameter descriptions, API details, and dual-path inference implementation, see [`examples/Qwen3_VL_LoRA/README.md`](examples/Qwen3_VL_LoRA/README.md).

## 6. Model Adaptation Guide

- **Same-series Compatibility**: Examples within the same model series are interchangeable. For instance, the Qwen2.5-0.5B example works directly with Qwen2.5-7B by simply changing the model loading path.

- **New Model Adaptation**: For LLM models not included in this repository, refer to the [LLM Model Adaptation Tutorial](LLM_model_modification_guide_EN.md) for ONNX export and deployment porting.

## 7. Important Notes

- **Transformers Version**: Different models may require different `transformers` versions. Before exporting to ONNX, install the correct version. Version info can be found in the `transformers_version` field of the model's config.json (e.g., https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json ). Some models have special version requirements; refer to the `requirements.txt` in each example directory.

- **PyTorch Version**: Recommended PyTorch <= 2.8.0 (Qwen3-VL, Gemma-4, and other models require PyTorch >= 2.9.0; PaddleOCR-VL requires transformers == 4.55.0. See the `requirements.txt` under the corresponding model for details.)

- **Module Compatibility**: Gemma-4 Audio and LLM models must use the same version (both E2B or both E4B); mixing versions is not supported.

## 8. Additional Notes

This repository uses the following mirror sites by default to obtain model files:
- [ModelScope](https://www.modelscope.cn) (recommended for China users)
- [HuggingFace Mirror](https://hf-mirror.com)
