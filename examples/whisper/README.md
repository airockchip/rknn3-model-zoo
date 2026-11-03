# Whisper 模型部署说明

Whisper 是 OpenAI 发布的通用语音识别模型，支持多语言语音识别与翻译。本示例基于 RKNN3 将 Whisper 拆分为 Encoder、Decode0 和 Decode1 三个部分部署：

- Encoder：将 log-mel 音频特征编码为隐藏状态。
- Decode0：根据 Encoder 输出生成 cross-attention 的 key/value cache。
- Decode1：基于 RKNN3 LLM 运行时进行自回归 token 生成。

当前 C++ demo 支持英文和中文识别任务。

## 1. 项目结构

```
whisper/
├── cpp
│   ├── CMakeLists.txt
│   ├── easy_timer.h
│   ├── main_raw.cc                  # RKNN3 Whisper 推理入口
│   ├── process.cc
│   ├── process.h
│   ├── rknpu3
│   │   └── whisper.cc
│   └── whisper.h
├── model
│   ├── long_en.wav
│   ├── mel_128_filters.txt
│   ├── mel_80_filters.txt
│   ├── position_embed.bin
│   ├── prompt_embed.bin
│   ├── prompt_embed_en.bin
│   ├── prompt_embed_zh.bin
│   ├── test_en.wav
│   ├── test_zh.wav
│   ├── token_embed.bin
│   ├── vocab_en.txt
│   └── vocab_zh.txt
├── python
│   ├── export_onnx.py               # 导出 Encoder / Decode0 / Decode1 ONNX
│   └── export_rknn.py               # 导出 Encoder / Decode0 / Decode1 RKNN
└── README.md
```

## 2. ONNX 模型导出

### 模型导出环境

建议直接使用同一个 Python 环境完成 ONNX 导出和 RKNN 导出，避免在两个环境之间切换：

- 该环境需要同时满足：
  - `export_onnx.py` 所需的 `transformers`、`onnx`、`numpy`等
  - `export_onnx.py` 和 `export_rknn.py` 所需的 `rknn3-toolkit`

推荐按 RKNN Toolkit 发布包安装 `rknn3-toolkit` 后，再补齐其余 Python 依赖：

```bash
cd python
pip install -r requirements.txt
```

> **注意**：`export_onnx.py` 也会导入 `rknn.utils.onnx_edit`，因此 ONNX 导出阶段同样需要 `rknn3-toolkit` 可用。

### 2.1 导出 Whisper ONNX 模型

```bash
cd examples/whisper/python
python export_onnx.py
```

导出后会在 `python/whisper-base-model` 下生成：

```text
whisper_encoder_base.onnx
whisper_encoder_base.config.pkl
whisper_decode0_base.onnx
whisper_decode0_base.config.pkl
whisper_decode1_base.onnx
whisper_decode1_base.config.pkl
```

> **注意**：如果使用modelscope ，应该运行‘python export_onnx.py --modelscope --model_path openai-mirror/whisper-base’

#### 参数说明

| 参数               | 类型 | 说明                                                          | 默认值                      |
| ------------------ | ---- | ------------------------------------------------------------- | --------------------------- |
| `--model_path`   | str  | Whisper 模型路径或 HuggingFace 名称                           | `openai/whisper-base`     |
| `--output_dir`   | str  | ONNX 和 config 输出目录                                       | `whisper-base-model` |
| `--modelscope`   | bool | 从 ModelScope 下载模型                                        | `False`                   |
| `--model`        | str  | 导出部分，可选 `encoder`、`decode0`、`decode1`、`all` | `all`                     |
| `--seq_len`      | int  | Decode0 导出使用的 dummy decoder 序列长度                     | `4`                       |
| `--past_seq_len` | int  | Decode1 导出使用的 dummy position id                          | `4`                       |

> **注意**：输出文件名中的 `base` 来自模型名 `openai/whisper-base`。如果使用 `openai/whisper-large-v3-turbo`，suffix 会变成 `large-v3-turbo`，后续 RKNN 导出和运行命令中的文件名也需要相应修改。当前支持 whisper-base 和 whisper-large-v3-turbo；Turbo 运行时需添加 `--model-type turbo`。

## 3. RKNN 模型导出

### 3.1 导出全部 RKNN 模型

```bash
cd examples/whisper/python
python export_rknn.py
```

导出后会生成：

```text
whisper_encoder_base.rknn
whisper_encoder_base.weight
whisper_decode0_base.rknn
whisper_decode0_base.weight
whisper_decode1_base.rknn
whisper_decode1_base.weight
```

其中 Decode1 使用 `load_llm` 导出，会生成 `.rknn` 和 `.weight` 两个文件。

### 3.2 单独导出某一部分

```bash
# Encoder
python export_rknn.py --suffix base --model encoder

# Decode0
python export_rknn.py --suffix base --model decode0

# Decode1
python export_rknn.py --suffix base --model decode1
```

#### 参数说明

| 参数                  | 类型 | 说明                                                          | 默认值                      |
| --------------------- | ---- | ------------------------------------------------------------- | --------------------------- |
| `--onnx_dir`        | str  | ONNX 和 config 文件所在目录                                   | `whisper-base-model` |
| `--suffix`          | str  | 模型文件名 suffix，例如 `base`、`large-v3`                | `base`                    |
| `--model`           | str  | 导出部分，可选 `encoder`、`decode0`、`decode1`、`all` | `all`                     |
| `--target_platform` | str  | RKNN 目标平台                                                 | `rk1820`                  |
| `--max_seq_len`     | int  | Decode1 动态输入最大序列长度，0 表示从 config 读取            | `0`                       |

## 4. C++ 部署

### 4.1 编译

在仓库根目录执行：

```bash
./build-linux.sh -t rk3588 -a aarch64 -d whisper -b Release
```

当前默认编译目标为：

```text
rknn_whisper_raw_demo
```

### 4.2 推送到开发板

```bash
adb push install/rk3588_linux_aarch64/rknn_whisper_demo/ /data/
```

如果导出的 RKNN 模型没有被拷贝到 install 目录，需要手动推送：

```bash
adb push examples/whisper/python/whisper-base-model/whisper_encoder_base.rknn /data/rknn_whisper_demo/model/
adb push examples/whisper/python/whisper-base-model/whisper_encoder_base.weight /data/rknn_whisper_demo/model/
adb push examples/whisper/python/whisper-base-model/whisper_decode0_base.rknn /data/rknn_whisper_demo/model/
adb push examples/whisper/python/whisper-base-model/whisper_decode0_base.weight /data/rknn_whisper_demo/model/
adb push examples/whisper/python/whisper-base-model/whisper_decode1_base.rknn /data/rknn_whisper_demo/model/
adb push examples/whisper/python/whisper-base-model/whisper_decode1_base.weight /data/rknn_whisper_demo/model/
```

### 4.3 运行

> **注意**：当前 demo 暂不支持 30s 以上语音，请使用 30s 以内的 WAV 音频进行测试。

```bash
adb shell
cd /data/rknn_whisper_demo


./rknn_whisper_raw_demo \
    model/whisper_encoder_base.rknn \
    model/whisper_encoder_base.weight \
    model/whisper_decode0_base.rknn \
    model/whisper_decode0_base.weight \
    model/whisper_decode1_base.rknn \
    model/whisper_decode1_base.weight \
    en \
    model/test_en.wav \
    model/token_embed.bin \
    0xff \
    0xff \
    0xff \
    128
```

中文音频可将任务参数改为 `zh`，并使用中文测试音频：

```bash
./rknn_whisper_raw_demo \
    model/whisper_encoder_base.rknn \
    model/whisper_encoder_base.weight \
    model/whisper_decode0_base.rknn \
    model/whisper_decode0_base.weight \
    model/whisper_decode1_base.rknn \
    model/whisper_decode1_base.weight \
    zh \
    model/test_zh.wav \
    model/token_embed.bin \
    0xff \
    0xff \
    0xff \
    128
```

#### 运行参数说明

```text
./rknn_whisper_raw_demo \
    <encoder.rknn> \
    <encoder.weight> \
    <decode0.rknn> \
    <decode0.weight> \
    <decode1.rknn> \
    <decode1.weight> \
    <task:en|zh> \
    <audio.wav> \
    <token_embed.bin> \
    <encoder_core_mask_hex> \
    <decode0_core_mask_hex> \
    <decode1_core_mask_hex> \
    [max_new_tokens]
```

| 参数                                  | 说明                                  |
| ------------------------------------- | ------------------------------------- |
| `encoder.rknn` / `encoder.weight` | Encoder RKNN 模型和分离权重           |
| `decode0.rknn` / `decode0.weight` | Decode0 RKNN 模型和分离权重           |
| `decode1.rknn` / `decode1.weight` | Decode1 RKNN 模型和分离权重           |
| `task`                              | 识别语言，目前支持 `en` 和 `zh`   |
| `audio.wav`                         | 输入 WAV 音频                         |
| `token_embed.bin`                   | Whisper decoder token embedding       |
| `*_core_mask_hex`                   | NPU core mask，例如 `0xff`          |
| `max_new_tokens`                    | 最大生成 token 数，可选，默认 `64` |

运行成功后会输出识别文本，例如：

```text
---- whisper raw decode ----
Whisper output:  "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
Inference RTF: 1.369 / 5.855 = 0.234

---- whisper raw decode ----
Whisper output:  Mm-hmm. Oh yeah, yeah. He wasn't even that big when I started listening to him. But, and his solo music didn't do overly well, but he did very well when he started writing for other people.
Inference RTF: 1.974 / 15.051 = 0.131

---- whisper raw decode ----
Whisper output: 对我做了介绍我想说的是呢大家如果对我的研究感兴趣
Inference RTF: 1.358 / 5.611 = 0.242
```
