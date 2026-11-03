# SeaCo-Paraformer RKNN3

## 模型说明

- FunASR 官方工程：<https://github.com/modelscope/FunASR>
- 模型下载：<https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch>

本示例将 SeaCo-Paraformer 中文离线语音识别模型部署到 RK1820。基础识别链路拆分为以下阶段：

1. Encoder 在 RK1820 上提取声学特征，并同时输出 Predictor alpha。
2. CIF 在 CPU 上根据 alpha 生成长度可变的 acoustic embedding。
3. Masked Decoder 在 RK1820 上输出 token logits，CPU 完成 argmax 和文本解码。

```text
16 kHz WAV -> FBank/LFR/CMVN -> Encoder + Alpha (RK1820)
           -> CIF (CPU) -> Masked Decoder (RK1820) -> 文本
```

热词模式在主 Decoder 后增加 SeaCo Decoder。可选 ASF Scorer 用于从较大的热词表中筛选候选词，BiasEncoder 在初始化阶段由 CPU 计算热词 embedding。

```text
hotwords.txt -> BiasEncoder (CPU) -> [ASF Scorer (RK1820)]
             -> SeaCo Decoder (RK1820) -> Posterior Fusion -> 文本
```

## 模型接口

所有 RKNN3 子模型的输入输出均为 FP16。下表中，`T` 表示 Encoder 时间长度，`E` 表示 Decoder 使用的 Encoder 时间长度，`N` 表示 token 长度，`H` 表示热词数量。

### 基础识别

| 子模型 | 输入 | 输出 |
|---|---|---|
| Encoder + Alpha | `speech [1,T,560]`<br>`enc_mask [1,T,1]` | `enc [1,T,512]`<br>`alphas [1,T,1]` |
| Masked Decoder | `enc [1,E,512]`<br>`pre_acoustic_embeds [1,N,512]`<br>`token_mask [1,N,1]`<br>`enc_mask [1,E,1]` | `decoder_out [1,N,8404]` |

### 热词增强

| 子模型 | 输入 | 输出 |
|---|---|---|
| Hotword Decoder | 与 Masked Decoder 相同 | `decoder_out [1,N,8404]`<br>`decoder_hidden [1,N,512]` |
| SeaCo Decoder | `contextual_info [1,H,512]`<br>`pre_acoustic_embeds [1,N,512]`<br>`decoder_hidden [1,N,512]`<br>`token_mask [1,N,1]`<br>`hotword_mask [1,H]` | `seaco_logits [1,N,8404]` |
| ASF Scorer | `contextual_info [1,H,512]`<br>`decoder_hidden [1,N,512]`<br>`token_mask [1,N,1]`<br>`hotword_mask [1,H]` | 每个热词的 attention score |

## Python 环境

```bash
pip install -r requirements.txt
```

## 导出 ONNX

在当前目录执行：

```bash
python3 python/export_onnx.py \
  --output_path model \
  --stage all
```

| 参数 | 说明 |
|---|---|
| `--model_path` | ModelScope、Hugging Face 模型 ID 或本地模型目录；默认使用官方 ModelScope 模型 |
| `--modelscope` | 从 ModelScope 下载自定义的非本地 `--model_path` |
| `--output_path` | ONNX 和运行资源的输出目录 |
| `--stage` | 导出阶段：`all`、`encoder`、`decoder` 或 `hotword`；默认为 `all` |

默认从 ModelScope 自动下载官方模型。自定义远程模型 ID 默认从 Hugging Face 下载，增加 `--modelscope` 后改为从 ModelScope 下载；如果 `--model_path` 指向本地目录，则直接读取且不会触发下载。

## 导出 RKNN

```bash
python3 python/export_rknn.py \
  --onnx_path model \
  --output_path model \
  --platform rk1820 \
  --stage all
```

| 参数 | 说明 |
|---|---|
| `--onnx_path` | 包含所选阶段 ONNX 文件的目录 |
| `--output_path` | RKNN、weight 和运行资源的部署输出目录 |
| `--platform` | RKNN 目标平台；默认为 `rk1820` |
| `--stage` | 可选 `encoder`、`decoder`、`decoder_hotword`、`seaco_decoder`、`asf_scorer`、`hotword` 或 `all`；默认为 `all` |

## 编译

从 model-zoo 根目录执行：

```bash
export GCC_COMPILER=/path/to/aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu
./build-linux.sh -t rk3588 -a aarch64 -d paraformer -b Release
```

输出目录：

```text
install/rk3588_linux_aarch64/rknn_paraformer_demo/
├── rknn_paraformer_demo
├── lib/
└── model/
```

## 板端运行

将编译结果和模型目录推送到板端：

```bash
adb push install/rk3588_linux_aarch64/rknn_paraformer_demo /userdata/
adb push examples/paraformer/model/. /userdata/rknn_paraformer_demo/model/
```

每个 `.weight` 必须与对应的 `.rknn` 位于同一目录并使用相同文件名。Demo 只需要传入 `.rknn` 路径，会自动查找同名 `.weight`。

### 基础识别

```bash
adb shell
cd /userdata/rknn_paraformer_demo
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH

./rknn_paraformer_demo \
  --encoder_path model/paraformer_encoder.rknn \
  --decoder_path model/paraformer_decoder.rknn \
  --tokens_path model/tokens.json \
  --audio_path model/asr_example.wav
```

Demo 默认查询模型核数并生成 core mask。只有需要显式校验运行核时才传入 `--core_mask`；当前单核模型对应 `--core_mask 0x1`。

### SeaCo 热词

热词文件使用 UTF-8 编码，每行一个热词。例如：

```text
麋鹿
```

运行命令：

```bash
./rknn_paraformer_demo \
  --encoder_path model/paraformer_encoder.rknn \
  --decoder_path model/paraformer_decoder_hotword.rknn \
  --tokens_path model/tokens.json \
  --audio_path model/asr_example_hotword.wav \
  --seaco_path model/seaco_decoder.rknn \
  --hotwords_path model/hotwords.txt
```

Demo 根据 `--seaco_path` 自动查找同目录下的 `seaco_decoder.weight`、`bias_encoder.bin` 和 `seg_dict`。

如需启用 ASF，在上述命令末尾追加：

```bash
  --asf_path model/asf_scorer.rknn \
  --asf_nfilter 50
```

常用可选参数：

| 参数 | 说明 |
|---|---|
| `--core_mask` | NPU core mask；省略时根据模型核数自动生成 |
| `--seaco_weight` | SeaCo posterior 融合权重，范围为 `[0,1]`，默认 `1.0` |
| `--max_hotwords` | 包含 `NO_BIAS` 的热词总数上限，默认 `128` |
| `--max_hotword_len` | 单个热词的 token 数上限，默认 `16` |
| `--asf_nfilter` | ASF 保留的真实热词数量，必须与 `--asf_path` 同时使用 |

所有 NPU 子模型按串行方式运行，并在首次推理前自动复用同一组 RKNN3 internal memory。模型权重和输入输出 tensor buffer 不属于该共享范围。

## 结果输出

### 基础识别

```text
Recognition Result: 用彩色水稻种出了一幅中国地图
```

### SeaCo 热词识别

热词：`麋鹿`

```text
未启用热词：大丰麋露保护区和边防派出所请来施工单位
启用 SeaCo 热词：大丰麋鹿保护区和边防派出所请来施工单位
```
