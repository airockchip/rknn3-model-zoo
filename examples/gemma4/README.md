# Gemma-4 模型部署说明

模型地址：
- E2B: [Gemma-4-E2B-IT](https://huggingface.co/google/gemma-4-E2B-it) (2B 参数)
- E4B: [Gemma-4-E4B-IT](https://huggingface.co/google/gemma-4-E4B-it) (4B 参数)

## 1. 部署环境

首先安装最新版 rknn3-toolkit， `transformers/torch/torchvision` 的版本需要更新以支持 Gemma-4 模型导出，请使用以下命令安装：

```bash
pip install transformers==5.5.0
pip install torch==2.9.0 torchvision==0.24.0 
```

验证命令：

```bash
pip list | grep -E "torch|vision|transformers"
# 预期输出如下
torch                     2.9.0
torchvision               0.24.0
transformers              5.5.0
```

## 2. 导出 RKNN 模型

### 2.1 LLM 模型

```bash
cd rknn3_model_zoo/

# 配置python环境变量为当前rknn3-model-zoo的路径
export PYTHONPATH="/path/to/rknn3-model-zoo"

cd examples/gemma4/python/llm

# 导出 E2B ONNX 模型
python export_llm.py

# 导出 E4B ONNX 模型
python export_llm.py --model_path google/gemma-4-E4B-it

# 导出 E2B RKNN 模型
python export_rknn.py

# 导出 E4B RKNN 模型
python export_rknn.py --model_type e4b
```

### 2.2 Audio 模型

```bash
cd examples/gemma4/python/audio

# 导出 E2B ONNX 模型
python export_audio.py

# 导出 E4B ONNX 模型
python export_audio.py --model_path google/gemma-4-E4B-it

# 导出 E2B RKNN 模型
python export_rknn.py

# 导出 E4B RKNN 模型
python export_rknn.py --model_type e4b
```

> **注意**：Audio 模型与 LLM 模型需使用同一版本（同为 E2B 或同为 E4B），不可混用。

流式 Audio 使用显式状态反馈，并通过 RKNN3 动态输入配置生成多个 Mel 长度桶。原有命令仍导出离线 Audio Tower；仅在增加 `--streaming` 时导出流式模型。

```bash
# 导出流式 ONNX（示例跟踪长度 720，时间轴保持动态）
python export_audio.py \
    --streaming \
    --export_audio_path ../../model/audio/gemma-4-e2b-it-audio-stream.onnx

# 转换为 240/480/720 三桶 RKNN3 模型
python export_streaming_rknn.py \
    --onnx_path ../../model/audio/gemma-4-e2b-it-audio-stream.onnx \
    --rknn_path ../../model/audio/gemma-4-e2b-it-audio-stream.rknn \
    --buckets 240,480,720 \
    --core_num 8
```

### 2.3 Vision 模型

```bash
cd examples/gemma4/python/vision

# 导出 E2B Vision ONNX 模型
python export_vision.py --model_path google/gemma-4-E2B-it --export_vision_path ./gemma-4-e2b-vision.onnx

# 导出 E2B Vision RKNN 模型
python export_rknn.py --onnx_path ./gemma-4-e2b-vision.onnx --rknn_path ./gemma-4-e2b-vision.rknn
```

> **注意**：Vision 模型与 LLM 模型需使用同一版本。`export_vision.py` 会同时生成 `gemma4_vision_rknn_load.json` 和 `pixel_position_ids` 的 `.npy` 文件，`export_rknn.py` 默认通过 `--rknn_load_json` 读取这些配置。默认输入图像大小为 `384x384`，输入高宽需为 48 的倍数。

## 3. C++ 板端部署

### 3.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d gemma4
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_gemma4_demo/` 目录：

```
rknn_gemma4_demo/
├── lib/
│   ├── librga.so
│   └── librknn3_api.so
├── model/
│   ├── gemma-4-{e2b,e4b}-it-audio.rknn
│   ├── gemma-4-{e2b,e4b}-it-audio.weight
│   ├── gemma-4-{e2b,e4b}-vision.rknn
│   ├── gemma-4-{e2b,e4b}-vision.weight
│   ├── gemma-4-{e2b,e4b}-it.rknn
│   ├── gemma-4-{e2b,e4b}-it.weight
│   ├── gemma-4-{e2b,e4b}-it.embed.bin
│   ├── gemma-4-{e2b,e4b}-it.tokenizer.gguf
│   ├── gemma-4-{e2b,e4b}-it_per_layer_inputs.embed.bin
│   └── gemma-4-{e2b,e4b}-it.safetensors
├── demo_16k_mono_f32.wav
├── demo.jpg
└── rknn_gemma4_demo
```

### 3.2 部署到开发板

```bash
# 推送 demo 目录
adb push install/rk3588_linux_aarch64/rknn_gemma4_demo /data/

# 推送运行库
adb push install/rk3588_linux_aarch64/rknn_gemma4_demo/lib/* /usr/lib/
```

### 3.3 运行示例

运行程序时采用固定参数顺序，会根据 Audio 与 Vision 模型路径是否为空自动判断加载方式：

- Audio 模型路径和 Audio Weight 路径均非空：加载 Audio 模型。
- Vision 模型路径和 Vision Weight 路径均非空：加载 Vision 模型。
- 某一模态不需要加载时，该模态的 `.rknn` 和 `.weight` 两个参数均填写空字符串 `""`。

固定参数格式如下：

```bash
./rknn_gemma4_demo \
    <llm_model_path> <llm_weight_path> <llm_core_mask> \
    <tokenizer_path> <embedding_path> \
    <max_context_len> <max_new_tokens> \
    <per_layer_embed_path> <safetensors_path> \
    <audio_model_path> <audio_weight_path> <audio_core_mask> \
    <vision_model_path> <vision_weight_path> <vision_core_mask> \
    <audio_path> <image_path> [prompt] [audio_mode]
```

进入运行目录并赋予可执行权限：

```bash
adb shell
cd /data/rknn_gemma4_demo
chmod +x rknn_gemma4_demo
```

#### 3.3.1 Audio + Vision + LLM

同时加载 Audio、Vision 和 LLM 模型时，Audio 与 Vision 的 `.rknn`、`.weight` 均填写实际模型路径。

```bash
./rknn_gemma4_demo \
    model/gemma-4-e2b-it.rknn \
    model/gemma-4-e2b-it.weight \
    0xff \
    model/gemma-4-e2b-it.tokenizer.gguf \
    model/gemma-4-e2b-it.embed.bin \
    4096 \
    4096 \
    model/gemma-4-e2b-it_per_layer_inputs.embed.bin \
    model/gemma-4-e2b-it.safetensors \
    model/gemma-4-e2b-it-audio.rknn \
    model/gemma-4-e2b-it-audio.weight \
    0xff \
    model/gemma-4-e2b-vision.rknn \
    model/gemma-4-e2b-vision.weight \
    0xff \
    demo_16k_mono_f32.wav \
    demo.jpg \
    "<audio>请将这段语音转为文本，并描述<image>这张图片"
```

#### 3.3.2 Audio + LLM

仅加载 Audio 和 LLM 模型时，Vision 的 `.rknn` 与 `.weight` 均填写空字符串 `""`，图片路径也填写空字符串 `""`。

```bash
./rknn_gemma4_demo \
    model/gemma-4-e2b-it.rknn \
    model/gemma-4-e2b-it.weight \
    0xff \
    model/gemma-4-e2b-it.tokenizer.gguf \
    model/gemma-4-e2b-it.embed.bin \
    4096 \
    4096 \
    model/gemma-4-e2b-it_per_layer_inputs.embed.bin \
    model/gemma-4-e2b-it.safetensors \
    model/gemma-4-e2b-it-audio.rknn \
    model/gemma-4-e2b-it-audio.weight \
    0xff \
    "" \
    "" \
    0 \
    demo_16k_mono_f32.wav \
    "" \
    "<audio>将这段语音转为文本"
```

#### 3.3.3 Vision + LLM

仅加载 Vision 和 LLM 模型时，Audio 的 `.rknn` 与 `.weight` 均填写空字符串 `""`，音频路径也填写空字符串 `""`。

```bash
./rknn_gemma4_demo \
    model/gemma-4-e2b-it.rknn \
    model/gemma-4-e2b-it.weight \
    0xff \
    model/gemma-4-e2b-it.tokenizer.gguf \
    model/gemma-4-e2b-it.embed.bin \
    4096 \
    4096 \
    model/gemma-4-e2b-it_per_layer_inputs.embed.bin \
    model/gemma-4-e2b-it.safetensors \
    "" \
    "" \
    0 \
    model/gemma-4-e2b-vision.rknn \
    model/gemma-4-e2b-vision.weight \
    0xff \
    "" \
    demo.jpg \
    "<image>请描述图片"
```

流式 Audio + LLM 与 3.3.2 使用相同参数，仅将 Audio 模型替换为流式多桶模型，并在命令末尾增加 `streaming`：

```bash
./rknn_gemma4_demo <其余参数> \
    demo_16k_mono_f32.wav "" "<audio>将这段语音转为文本" streaming
```

不指定 `[audio_mode]` 或指定 `offline` 时，仍使用原有离线 Audio Tower。

> **e4b 只需将上述命令中的 e2b 对应模型文件替换为 e4b 对应模型文件即可。Audio、Vision 与 LLM 模型必须来自同一版本，不可混用。**

### 3.4 参数说明

| 参数 | 说明 |
|------|------|
| `<llm_model_path>` | LLM 模型路径，`.rknn` 文件 |
| `<llm_weight_path>` | LLM Weight 路径，`.weight` 文件 |
| `<llm_core_mask>` | LLM 模型使用的 NPU 核心掩码（十六进制），如 `0xff` |
| `<tokenizer_path>` | Tokenizer 文件路径，`.gguf` 文件 |
| `<embedding_path>` | Token Embedding 文件路径，`.bin` 文件 |
| `<max_context_len>` | 最大上下文长度 |
| `<max_new_tokens>` | 最大生成 token 数 |
| `<per_layer_embed_path>` | Per-layer Embedding 文件路径 |
| `<safetensors_path>` | Rope Cache 文件路径，使用 host rope cache 时填写 `.safetensors` 文件 |
| `<audio_model_path>` | Audio 模型路径，`.rknn` 文件；不加载 Audio 时填写 `""` |
| `<audio_weight_path>` | Audio Weight 路径，`.weight` 文件；不加载 Audio 时填写 `""` |
| `<audio_core_mask>` | Audio 模型使用的 NPU 核心掩码（十六进制），如 `0xff`；不加载 Audio 时填写 `0` |
| `<vision_model_path>` | Vision 模型路径，`.rknn` 文件；不加载 Vision 时填写 `""` |
| `<vision_weight_path>` | Vision Weight 路径，`.weight` 文件；不加载 Vision 时填写 `""` |
| `<vision_core_mask>` | Vision 模型使用的 NPU 核心掩码（十六进制），如 `0xff`；不加载 Vision 时填写 `0` |
| `<audio_path>` | 音频文件路径；不使用 Audio 时填写 `""` |
| `<image_path>` | 图片文件路径；不使用 Vision 时填写 `""` |
| `[prompt]` | 可选输入提示词。音频输入需包含 `<audio>`，图片输入需包含 `<image>` |
| `[audio_mode]` | 可选 Audio 执行方式：`offline`（默认）或 `streaming` |

### 3.5 加载约束
- Audio 与 Vision 是否加载只由对应 `.rknn` 和 `.weight` 两个参数决定。
- 不加载某一模态时，该模态的 `.rknn` 和 `.weight` 必须同时填写空字符串 `""`。
- 加载某一模态时，该模态的 `.rknn` 和 `.weight` 必须同时填写实际路径。
- Audio、Vision 与 LLM 模型需保持同一模型版本，例如均为 E2B 或均为 E4B。
- `streaming` 模式要求 Audio RKNN 模型为 8 输入、7 输出的显式状态接口，并包含动态 Mel 桶。

## 4. 音频格式要求

Gemma-4 的音频预处理模块要求输入为 **16kHz 单声道 float32** PCM 格式。

如果您的音频文件不是此格式，请使用 `convert_wav_for_cpp.py` 进行转换：

```bash
cd examples/gemma4/python/audio
python convert_wav_for_cpp.py \
    --src /path/to/your/audio.wav \
    --dst /path/to/output_16k_mono_f32.wav \
    --target_sr 16000
```

支持的输入格式：任意采样率、任意声道数的 WAV 文件。
