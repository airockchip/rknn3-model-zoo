# Qwen3-ForcedAligner-0.6B

## 模型说明

本示例在 RK1828 上将音频与已知文本对齐，输出中文逐字、英文逐词的起止时间。采用 AuT FP16、decoder W4A16，使用 8 核运行。

原始模型：[Qwen3-ForcedAligner-0.6B（Hugging Face）](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/tree/main)。

## Python 环境

需要已安装支持 RK182X 的 RKNN3 Toolkit 的 Python 环境。安装其他依赖：

```bash
pip install -r requirements.txt
```

## 导出模型

### 导出 ONNX

```bash
python python/export_onnx.py \
  --model_path Qwen/Qwen3-ForcedAligner-0.6B \
  --output_path outputs/onnx \
  --stage all
```

远端模型默认从 Hugging Face 下载；添加 `--modelscope` 可改用 ModelScope。本地模型目录直接使用。

### 导出 RKNN

```bash
python python/export_rknn.py \
  --onnx_path outputs/onnx \
  --output_path model \
  --stage all
```

ONNX 导出时自动生成板端文本资源，RKNN 转换时自动复制到 `model/`。部署时保留整个 `model/` 目录。导出时请使用未存在的输出文件。

## 编译

从仓库根目录执行；`rk3588` 是 Host，模型目标仍为 RK1828：

```bash
export GCC_COMPILER=/path/to/bin/aarch64-linux-gnu
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_ForcedAligner -b Release
```

安装目录为 `install/rk3588_linux_aarch64/rknn_Qwen3_ForcedAligner_demo`。构建会安装 Mel 滤波器以及 `model/` 下的模型和文本资源。

## 板端运行

从仓库根目录执行：

```bash
adb push install/rk3588_linux_aarch64/rknn_Qwen3_ForcedAligner_demo /userdata/
adb push /path/to/audio.wav /userdata/rknn_Qwen3_ForcedAligner_demo/audio.wav
```

板端执行：

```bash
cd /userdata/rknn_Qwen3_ForcedAligner_demo
export LD_LIBRARY_PATH=./lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
./rknn_qwen3_forced_aligner_demo \
  --model_path ./model \
  --audio_path ./audio.wav \
  --text '与音频对应的文本' \
  --language Chinese \
  --mel_filters_path ./mel_128_filters.txt \
  --output_path ./result.tsv
```

以上六个参数均为必填项。`--text` 是与音频对应的已知文本，含空格时应加引号。`--language` 支持 `Chinese` 和 `English`。使用 `--help` 查看完整参数说明。

输出的文字及起止时间会打印到终端，并保存到 `--output_path` 指定的 TSV 文件。

## 输出示例

以约 14 秒的 AISHELL 音频 `BAC009S0906W0219.wav` 和文本“健全农作物种质资源和畜禽遗传资源保存体系”为例，时间戳输出如下：

```text
Input text: 健全农作物种质资源和畜禽遗传资源保存体系
Word/character timestamps (s):
  [0.480 - 0.720] 健
  [0.720 - 0.960] 全
  [0.960 - 1.280] 农
  [1.280 - 1.440] 作
  [1.440 - 1.840] 物
  [2.880 - 3.360] 种
  [5.040 - 5.520] 质
  [5.600 - 5.840] 资
  [5.840 - 6.080] 源
  [6.080 - 6.560] 和
  [8.000 - 8.560] 畜
  [8.560 - 9.040] 禽
  [10.080 - 10.240] 遗
  [10.240 - 10.720] 传
  [11.440 - 11.680] 资
  [11.680 - 12.000] 源
  [12.400 - 12.560] 保
  [12.560 - 12.800] 存
  [12.800 - 12.960] 体
  [12.960 - 13.200] 系
```
