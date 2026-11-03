# Unlimited-OCR 模型部署说明

## 1. 部署环境

建议使用 Python 3.10 环境执行模型导出和 RKNN 转换。

Unlimited-OCR 使用 HuggingFace 自定义模型代码，加载本地模型时需要开启 `trust_remote_code=True`。本仓库脚本已在代码中处理。

## 2. 模型拆分策略

Unlimited-OCR 部署时拆成三部分：

1. SAM 模型：输入原图，输出 `sam_features`。
2. Vision 模型：输入 `sam_features`，输出 `image_embeds`。
3. LLM 模型：输入文本和 `image_embeds`，输出 OCR 结果。

### 2.1 SAM 与 Vision 拆分

SAM 和 Vision 分开导出。这样可以单独控制 SAM 精度，避免 SAM 使用低比重量化后影响最终 LLM 输出。

当前策略：

- SAM ONNX：不做 GRQ 量化。
- SAM RKNN：不做量化。
- Vision ONNX：CLIP/Vision 分支使用 GRQ W4A16 group32。
- Vision RKNN：使用 W4A16 normal group32，因为 GRQ 已经在 ONNX 侧完成。

### 2.2 Vision 模式

Vision 支持两种模式：

- `base`：固定输入分辨率，当前 C++ demo 已实现。
- `gundam`：支持大图裁剪和动态输入，当前 C++ demo 未实现。

### 2.3 LLM MoE

LLM 导出 ONNX 时，MoE 由 RKNN 自定义 OP 接管。相关权重、配置和 tokenizer 文件由 `python/llm/export_llm.py` 一并导出。

## 3. 导出模型流程

以下命令默认在仓库根目录执行。

### 3.1 生成校准数据

LLM 校准数据：

```bash
cd examples/Unlimited-OCR/python/llm

python make_calidata.py \
  --model_path ../../model/Unlimited-OCR \
  --datapath ../../../../datasets/OmniDocBench/llm/dataset.json \
  --export_datapath ../../data/llm/model_inputs.json \
  --mode base
```

Vision 校准数据：

```bash
cd examples/Unlimited-OCR/python/vision

python make_calidata.py \
  --model_path ../../model/Unlimited-OCR \
  --datapath ../../../../datasets/OmniDocBench/llm/dataset.json \
  --export_datapath ../../data/vision/model_inputs.json \
  --mode base
```

### 3.2 导出 LLM ONNX

```bash
cd examples/Unlimited-OCR/python/llm

python export_llm.py \
  --model_path ../../model/Unlimited-OCR \
  --export_llm_path ../../model/llm/Unlimited-OCR-llm.onnx \
  --quant \
  --cali_dataset ../../data/llm/model_inputs.json
```

执行后会生成：

```bash
model/llm/Unlimited-OCR-llm.onnx
model/llm/Unlimited-OCR-llm.config.pkl
model/llm/Unlimited-OCR-llm.moe.json
model/llm/Unlimited-OCR-llm.tokenizer.gguf
model/llm/Unlimited-OCR-llm.embed.bin
```

### 3.3 导出 LLM RKNN

SlidingAttention：

```bash
cd examples/Unlimited-OCR/python/llm

python export_rknn.py \
  --onnx_path ../../model/llm/Unlimited-OCR-llm.onnx \
  --config ../../model/llm/Unlimited-OCR-llm.config.pkl \
  --rknn_path ../../model/llm/Unlimited-OCR-llm.rknn \
  --attention sliding \
  --sliding_window 128 \
  --prefill_seq_len 128 \
  --core_num 8
```

FullAttention：

```bash
cd examples/Unlimited-OCR/python/llm

python export_rknn.py \
  --onnx_path ../../model/llm/Unlimited-OCR-llm.onnx \
  --config ../../model/llm/Unlimited-OCR-llm.config.pkl \
  --rknn_path ../../model/llm/Unlimited-OCR-llm-full.rknn \
  --attention full \
  --prefill_seq_len 128 \
  --core_num 8
```

### 3.4 导出 SAM 和 Vision ONNX

```bash
cd examples/Unlimited-OCR/python/vision

python export_vision.py \
  --model_path ../../model/Unlimited-OCR \
  --mode base \
  --export_sam_path ../../model/vision/Unlimited-OCR-sam.onnx \
  --export_vision_path ../../model/vision/Unlimited-OCR-vision.onnx \
  --vision_config_path vision_config.json \
  --quant \
  --cali_dataset ../../data/vision/model_inputs.json
```

执行后会生成：

```bash
model/vision/Unlimited-OCR-sam.onnx
model/vision/Unlimited-OCR-vision.onnx
python/vision/vision_config.json
```

### 3.5 导出 SAM 和 Vision RKNN

```bash
cd examples/Unlimited-OCR/python/vision

python export_rknn.py \
  --part all \
  --sam_input_dtype uint8 \
  --quantized_dtype w4a16 \
  --quantized_algorithm normal \
  --quantized_method group32
```

也可以单独导出：

```bash
# 只导 SAM，SAM 不量化
python export_rknn.py \
  --part sam \
  --sam_input_dtype uint8

# 只导 Vision，Vision 使用 W4A16 normal group32
python export_rknn.py \
  --part vision \
  --quantized_dtype w4a16 \
  --quantized_algorithm normal \
  --quantized_method group32
```

## 4. 生成量化校准数据

Unlimited-OCR 的 LLM 与 Vision 在导出 ONNX 时可使用 GRQ W4A16 量化。量化需要校准数据。

仓库在 `python/llm/` 与 `python/vision/` 下各提供一个 `make_calidata.py`：

- LLM 脚本捕获 LLM 主干输入。
- Vision 脚本捕获 Vision 分支输入。
- SAM 当前不做量化，不需要 SAM 校准数据。

### 4.1 LLM 校准数据参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--model_path` | str | 模型路径或 HuggingFace 名称 | `baidu/Unlimited-OCR` |
| `--datapath` | str | 校准样本列表 JSON 路径 | 脚本默认值 |
| `--export_datapath` | str | 输出索引 JSON 路径 | 脚本默认值 |
| `--mode` | str | Vision 预处理模式 | `base` |
| `--base_size` | int | 原图基础尺寸 | `1024` |
| `--image_size` | int | 输入图尺寸 | base 默认 `1024`，gundam 默认 `640` |
| `--max_samples` | int | 最大样本数 | 脚本默认值 |
| `--max_new_tokens` | int | 生成 token 数 | 脚本默认值 |

### 4.2 Vision 校准数据参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--model_path` | str | 模型路径或 HuggingFace 名称 | `baidu/Unlimited-OCR` |
| `--datapath` | str | 校准样本列表 JSON 路径 | 脚本默认值 |
| `--export_datapath` | str | 输出索引 JSON 路径 | 脚本默认值 |
| `--mode` | str | Vision 预处理模式 | `base` |
| `--base_size` | int | 原图基础尺寸 | `1024` |
| `--image_size` | int | 输入图尺寸 | base 默认 `1024`，gundam 默认 `640` |
| `--max_samples` | int | 最大样本数 | 脚本默认值 |

### 4.3 数据说明

校准数据建议使用真实文档图片和真实提示词。样本数量过少会影响 GRQ 量化效果。

## 5. KV Cache 配置

LLM RKNN 导出脚本会生成 RKNN LLM 配置，并传给 `rknn.load_llm()`。

默认配置：

```bash
--max_ctx_len 8192
--kvcache_buffer_len 8192
--attention sliding
--sliding_window 128
--prefill_seq_len 128
```

如需使用全量注意力，可设置：

```bash
--attention full
```

## 6. Vision 模型分辨率调整

base 模式默认使用固定分辨率：

```bash
--mode base --image_size 1024
```

gundam 模式可设置裁剪网格：

```bash
--mode gundam \
  --base_size 1024 \
  --image_size 640 \
  --crop_width_num 2 \
  --crop_height_num 2
```

注意：当前 C++ demo 只实现 base 模式。

## 7. C++ 部署说明

C++ demo 代码位置：

```bash
examples/Unlimited-OCR/cpp/
```

### 7.1 编译 C++ demo

```bash
cd rknn3-model-zoo

./build-linux.sh -t rk3588 -a aarch64 -d Unlimited-OCR -b Debug
```

编译成功后，demo 会安装到：

```bash
install/rk3588_linux_aarch64/rknn_unlimited_ocr_demo
```

目录结构：

```bash
- install/rk3588_linux_aarch64/rknn_unlimited_ocr_demo
  - lib
    - librknn3_api.so
    - librga.so
  - model
    - Unlimited-OCR-sam.rknn
    - Unlimited-OCR-sam.weight
    - Unlimited-OCR-vision.rknn
    - Unlimited-OCR-vision.weight
    - Unlimited-OCR-llm.rknn
    - Unlimited-OCR-llm.weight
    - Unlimited-OCR-llm.tokenizer.gguf
    - Unlimited-OCR-llm.embed.bin
  - test.png
  - rknn_unlimited_ocr_demo
```

### 7.2 推送 demo 到 RK3588 开发板

```bash
adb push install/rk3588_linux_aarch64/rknn_unlimited_ocr_demo /userdata/
```

### 7.3 运行 C++ demo

```bash
adb shell
cd /userdata/rknn_unlimited_ocr_demo

./rknn_unlimited_ocr_demo \
  model/Unlimited-OCR-sam.rknn \
  model/Unlimited-OCR-sam.weight \
  model/Unlimited-OCR-vision.rknn \
  model/Unlimited-OCR-vision.weight \
  model/Unlimited-OCR-llm.rknn \
  model/Unlimited-OCR-llm.weight \
  model/Unlimited-OCR-llm.tokenizer.gguf \
  model/Unlimited-OCR-llm.embed.bin \
  0xff 0xff 0xff \
  test.png \
  "<image>document parsing." \
  base \
  1024
```

运行成功后，会打印 OCR 结果和推理性能。板端输出如下：

```bash
Unlimited-OCR preprocess latency = 18.87 ms
--> inference Unlimited-OCR SAM
Unlimited-OCR SAM latency = 1700.66 ms
Unlimited-OCR SAM->Vision copy latency = 0.27 ms
--> inference Unlimited-OCR vision
Unlimited-OCR vision latency = 40.90 ms
Unlimited-OCR vision output copy latency = 0.06 ms
Unlimited-OCR vision stack total latency = 1760.80 ms
Unlimited-OCR image_embeds tokens=273, bytes=698880
--> rknn3_session_run llm
header [720, 48, 808, 62]Solutions
page_number [847, 49, 881, 61]105
text [113, 79, 882, 133]The last two expectations in this expression are known from the basic result about E[U] so that the only really new part concerns the expectation of the product UV. Now, by the definition of U, V
equation [298, 697]\[
\mathbf {U} \mathbf {V} = \exp (\mathbf {X}) \cdot \exp (\mathbf {Y)} = \exp (\mathbf {X} + \mathbf {Y)}
\]
text [113, 180, 884, 235]Let the rv S = X + Y. Clearly, S is normally distributed, with mean \(\mu_{s} = \mu_{\mathcal{I}} + \mu_{y}\) and variance \(\sigma_s^2 = \sigma_x^2 +\sigma_y^2 +2g\sigma_x\sigma_y\). It then follows from the basic result about E[U] that
equation [368, 248, 624, 310]\[
\begin{array}{l} \mathrm{E} [ \mathbf {U V} ] = \mathrm{E} [ \exp (\mathbf {S}) \\ = \exp \left(\mu_ {s} + \frac {1}{2} \sigma_ {s} ^ {2}\right) \\ \end{array}
\]
text [113, 322, 882, 392]Putting together this result — with  \( \mu_{s}, \sigma_{s} \)  expressed in terms of the original parameters a scalar — and the previously known facts about the means and variances of U, V, the correlation coefficient is, after slight simplification, found to be
equation [277, 406, 720, 458]\[
\operatorname{corr} (\mathbf {U}, \mathbf {V}) = \frac {\exp (\varrho \sigma_ {x} {\sqrt {[ \exp (\sigma_ {s} ^ {2}) - 1 ] [ \exp (\sigma_ {y} ^ {2})}}}
\]
text [113, 468, 884, 594]Note that the correlation between U and V is completely independent of the means of X and Y. As explained earlier, the exponentiation turns the location parameters \(\mu_{x},\mu_{y}\) of X and Y into scaling factors of U and V. Because variations of the scaling factors generally do not change linear correlations, the result was to be expected. On the other hand, the standard deviations \(\sigma_{x},\sigma_{y}\) of X and Y turn into powers for U and V, which generally do influence the linear correlation coefficient.
text [113, 595, 884, 647]The result about the correlation of U and V may also be used to verify the properties we already discussed in part c. Note, in particular, that in the equal-variance case \(\sigma_{x} = \sigma_{y} = \sigma\) we get
equation [361, 660, 637, 700]\[
\operatorname{corr} (\mathbf {U}, \mathbf {V}) = \frac {\exp (\varrho \sigma^ {2}) - 1}{\exp (\sigma^ {2}) - 1}
\]
text [113, 711, 884, 784]The correlation of U and V is shown in Figure 3.25 as a function of the correlation \(\varrho\) of the original rvs X and Y, for three values of \(\sigma\). For \(\varrho = +1\), the two correlations are identical, but for any other value the correlation between U and V is weaker than that between X and Y.
equation [361, 662, 634, 700]\[
\operatorname{corr} (\mathbf {U}, \mathbf {V}) = \frac {\exp (\varrho \sigma^ {2}) - 1}{\exp (\sigma^ {2}) - 1}
\]

--------------------Finished--------------------

--------------------------------------------------------------------------------------
 Stage         Total Time (ms)  Tokens    Time per Token (ms)      Tokens per Second      
--------------------------------------------------------------------------------------
 Prefill       327.13           278       1.18                     849.82                 
 Generate      8141.50          893       9.12                     109.68                 
--------------------------------------------------------------------------------------
 Vision latency = 1760.81 ms, FPS = 0.57
```

## 8. 内存占用

| 模型部分 | 量化方式 | 内存占用 |
|----------|----------|----------|
| SAM | W16A16 | 1877.80 MB |
| Vision | W4A16 GRQ group32 | 193.45 MB |
| LLM | W4A16 GRQ group32 | 2408.92 MB |
