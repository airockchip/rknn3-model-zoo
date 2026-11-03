# LFM2.5-VL 模型部署说明

支持以下模型：

| 模型             | 参数量  | LLM 层数 | Conv 层数 | 词表大小   | 模型地址                                                                                                                                   |
| -------------- | ---- | ------ | ------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| LFM2.5-VL-1.6B | 1.6B | 16     | 10      | 65536  | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) / [ModelScope](https://www.modelscope.cn/models/LiquidAI/LFM2.5-VL-1.6B) |
| LFM2.5-VL-3B   | 3.1B | 30     | 22      | 128000 | [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-VL-3B) / [ModelScope](https://www.modelscope.cn/models/LiquidAI/LFM2.5-VL-3B)     |

## 模型简介

LFM2.5-VL 是 LiquidAI 推出的视觉语言模型，由三部分组成：

- **LLM 骨干**：LFM2.5 混合架构（Full Attention + LIV 短卷积层）
- **视觉编码器**：SigLIP2 NaFlex shape-optimized 400M（两个模型完全相同）
- **多模态投影器**：2 层 MLP + pixel unshuffle（两个模型完全相同）

### 1.6B vs 3B 架构差异

| 参数                         | 1.6B        | 3B                |
| -------------------------- | ----------- | ----------------- |
| hidden\_size               | 2048        | 2048（相同）          |
| num\_hidden\_layers        | 16          | 30                |
| Conv 层数                    | 10          | 22                |
| Full Attention 层数          | 6           | 8                 |
| vocab\_size                | 65536       | 128000            |
| intermediate\_size         | 12288       | 10752             |
| image\_token\_id           | 396         | 124907            |
| tie\_word\_embeddings      | -           | true              |
| num\_key\_value\_heads     | 8           | 8（相同）             |
| conv\_dim / conv\_L\_cache | 2048 / 3    | 2048 / 3（相同）      |
| Vision encoder             | SigLIP2 27层 | SigLIP2 27层（完全相同） |

RKNN3 部署将模型拆分为 Vision 和 LLM 两个部分分别导出。LLM 部分通过 RKNN3 的 `cvt_conv_streaming` 技术实现板端流式卷积推理。

Vision 部分采用 **pos\_emb 外部输入**方案（`export_vision.py`）：RKNN 图为全静态 3 输入（`pixel_values` / `pixel_attention_mask` / `pos_emb`），位置编码表 `pos_emb_base.bin`（16×16 基础网格）由板端 C++ 用 antialias 双线性插值到实际图片的 patch grid，支持任意宽高比的动态分辨率输入（上限 32×32 = 1024 patches）。经 2×2 pixel unshuffle 投影后每张图最多输出 256 个视觉 token，有效行在模型外提取。

## 1. 部署环境

```bash
pip install -r requirements.txt
```

> ⚠️ 还需安装最新版 rknn3-toolkit

> ⚠️ LFM2.5 模型需要 `transformers>=5.0.0`，请单独安装：
>
> ```bash
> pip install "transformers>=5.0.0" pillow torch
> ```

## 2. 项目结构

```
LFM2_5_VL/
├── cpp/                          # C++ 推理代码
│   ├── main.cc                   # 程序入口，多模态输入组装与调度
│   ├── lfm2_vl.cc/.h             # VL 模型调度逻辑
│   ├── vision/                   # Vision 推理封装
│   ├── llm/                      # LLM 推理封装
│   └── CMakeLists.txt
├── python/
│   ├── vision/                   # Vision ONNX/RKNN 导出 + 校准数据生成
│   │   ├── export_vision.py           # ONNX 导出（pos_emb 外部输入，支持动态分辨率）
│   │   ├── export_rknn.py             # RKNN 转换（默认 w4a16 量化）+ simulator 验证
│   │   └── make_calidata.py          # 生成量化校准数据
│   └── llm/                      # LLM ONNX/RKNN 导出 + 校准数据生成
│       ├── export_llm.py
│       ├── export_rknn.py
│       └── make_calidata.py
└── README.md
```

## 3. ONNX 模型导出

所有脚本通过 `--model_size` 参数选择模型变体（`1.6B` 或 `3B`），默认 `1.6B`（Vision 与 LLM 脚本一致，见下文各脚本参数表）。

### 3.1 导出 Vision 模型

```bash
# 1.6B（默认）
cd python/vision

# 1) 生成校准数据（首次或模型变更时运行，需 MMBench 数据集）
python make_calidata.py --modelscope --model_size 1.6B

# 2) 导出 ONNX（默认 fp32；需 GRQ 时加 --quant，需 CUDA）
python export_vision.py --modelscope --model_size 1.6B

# 3B
python make_calidata.py --modelscope --model_size 3B
python export_vision.py --modelscope --model_size 3B

# 开启 GRQ 权重微调（默认 w4a16，可设 --quant_dtype w4a16）
python export_vision.py --modelscope --model_size 1.6B --quant

```

> Vision 编码器在 1.6B 和 3B 中完全相同，但模型权重不同（因 projector 权重绑定到不同的 LLM）。
> 默认导出 fp32 ONNX。加 `--quant` 开启 GRQ 权重微调（需 CUDA + 校准数据），用于提升后续量化精度；`--quant_dtype` 选择 GRQ 精度（w8a16/w4a16）。最终的位宽压缩在 RKNN 转换阶段（4.1 节 `--quant_dtype`）完成。

导出产物（均生成在 `model/vision/` 下）：

| 文件                       | 说明                                                                                       |
| ------------------------ | ---------------------------------------------------------------------------------------- |
| `<basename>-vision.onnx` | 3 输入静态图：`pixel_values`(1,1024,768)、`pixel_attention_mask`(1,1024)、`pos_emb`(1,1024,1152) |
| `vision_config.json`     | 板端预处理/网格参数（pos\_emb 方案参数 + `image_mean`/`image_std` 等，C++ 启动时自动读取）                       |
| `pos_emb_base.bin`       | 基础位置编码表（16×16×1152 fp32），板端插值到动态 grid 后作为图输入                                             |

#### 参数说明

| 参数                     | 类型   | 说明                  | 默认值                                         |
| ---------------------- | ---- | ------------------- | ------------------------------------------- |
| `--model_size`         | str  | 模型变体                | `1.6B`                                      |
| `--model_path`         | str  | 模型路径或名称             | 由 `--model_size` 决定                         |
| `--export_vision_path` | str  | 输出 ONNX 路径          | `../../model/vision/<basename>-vision.onnx` |
| `--modelscope`         | bool | 从 ModelScope 下载     | `False`                                     |
| `--quant`              | bool | 开启 GRQ 权重微调（需 CUDA） | `False`                                     |
| `--quant_dtype`        | str  | `w8a16`/`w4a16`     | `w4a16`                                     |
| `--cali_dataset`       | str  | GRQ 校准数据路径          | `./quant_data/model_inputs.json`            |

### 3.2 导出 LLM 模型

```bash
# 1.6B（默认）
cd python/llm
python export_llm.py --modelscope

# 3B
python export_llm.py --modelscope --model_size 3B

# GRQ 量化导出（需 CUDA GPU，w4a16，精度最高）
python export_llm.py --modelscope --model_size 3B --quant \
    --cali_dataset ../quant_data_cn/dataset.json
```

LLM 导出会自动处理 Conv Streaming：将 ShortConv 的 padding 从 `[2,2]` 改为 `[2,0]`，生成 `*-convstream.onnx`。Conv 层数自动适配（1.6B=10, 3B=22）。

> `--quant` 开启 GRQ 权重微调（w4a16，需 CUDA GPU，加载用 bfloat16）。校准数据格式为 `[{"input":"...","target":"..."}]`（默认 `datasets/CMMLU/dataset.json`，也可用 `quant_data_cn/dataset.json`）。建议使用独立训练集，避免测试集泄漏。

#### 参数说明

| 参数                  | 类型   | 说明                      | 默认值                                   |
| ------------------- | ---- | ----------------------- | ------------------------------------- |
| `--model_size`      | str  | 模型变体                    | `1.6B`                                |
| `--model_path`      | str  | 模型路径或名称                 | 由 `--model_size` 决定                   |
| `--export_llm_path` | str  | 输出 ONNX 路径              | `../../model/llm/<basename>-llm.onnx` |
| `--modelscope`      | bool | 从 ModelScope 下载         | `False`                               |
| `--dtype`           | str  | 模型加载精度（非量化时）            | `fp16`                                |
| `--quant`           | bool | 开启 GRQ w4a16 量化（需 CUDA） | `False`                               |
| `--cali_dataset`    | str  | GRQ 校准数据路径              | `datasets/CMMLU/dataset.json`         |

> **注意**：LLM 导出除 ONNX 外，还会生成配套的 `.config.pkl`、`.tokenizer.gguf`、`.embed.bin` 等文件。

## 4. RKNN 模型导出

### 4.1 导出 Vision RKNN 模型

Vision 量化分两个正交阶段，可独立或组合使用：

- **GRQ 权重微调（可选，ONNX 阶段）**：3.1 导出 ONNX 时用 `--quant`（需 CUDA + 校准数据），对权重做微调以提升量化精度。
- **normal 位宽压缩（RKNN 阶段）**：本步骤用 `--quant_dtype` 决定最终权重位宽，默认 `w4a16`。

```bash
cd python/vision

# 正常量化（默认 w4a16）
python export_rknn.py --modelscope --model_size 1.6B

# w8a16
python export_rknn.py --modelscope --model_size 1.6B --quant_dtype w8a16

# 不量化（fp16 格式转换）
python export_rknn.py --modelscope --model_size 1.6B --quant_dtype fp16
```

> `--quant_dtype` 决定 RKNN 阶段是否压缩权重：`w4a16`（默认）、`w8a16` → `do_quantization=True`（normal 算法）；`fp16` → `do_quantization=False`（不量化）。GRQ（ONNX 阶段）与 normal（RKNN 阶段）正交，可组合获得最佳精度。

转换完成后自动执行 simulator 验证（与 torch 参考输出对比，`COS > 0.99` 判定通过，`--skip_verify` 可跳过）。输入形状自动从 `model/vision/vision_config.json` 读取。

#### 参数说明

| 参数                  | 类型   | 说明                               | 默认值                                         |
| ------------------- | ---- | -------------------------------- | ------------------------------------------- |
| `--model_size`      | str  | 模型变体                             | `1.6B`                                      |
| `--onnx`            | str  | 输入 ONNX 路径                       | `../../model/vision/<basename>-vision.onnx` |
| `--output`          | str  | 输出 RKNN 路径                       | `../../model/vision/<basename>-vision.rknn` |
| `--target_platform` | str  | 目标平台                             | `rk1820`                                    |
| `--core_num`        | int  | NPU 核心数                          | `8`                                         |
| `--quant_dtype`     | str  | RKNN 量化精度：`w4a16`/`w8a16`/`fp16` | `w4a16`                                     |
| `--model_path`      | str  | 验证用模型路径                          | 由 `--model_size` 决定                         |
| `--modelscope`      | bool | 从 ModelScope 下载（验证用）             | `False`                                     |
| `--verify_image`    | str  | 验证图片路径                           | `../../data/demo.jpg`                       |
| `--skip_verify`     | bool | 跳过 simulator 验证                  | `False`                                     |
| `--overwrite`       | bool | RKNN 已存在时强制重新转换                  | `False`                                     |

### 4.2 导出 LLM RKNN 模型

```bash
# 正常量化（默认 w4a16）
cd python/llm
python export_rknn.py --model_size 3B --seq_len 128 --target_platform rk1820

# w8a16
python export_rknn.py --model_size 3B --seq_len 128 --target_platform rk1820 \
    --quant_type w8a16

# 不量化（fp16）
python export_rknn.py --model_size 3B --seq_len 128 --target_platform rk1820 --no_quant
```

LLM RKNN 转换通过 `cvt_conv_streaming` 将 ShortConv 转换为流式卷积，Runtime 内部自动维护 Conv Cache。Conv 节点数自动适配（1.6B=10, 3B=22）。

> `--quant_type` 决定权重位宽：`w4a16`（默认，group32）/ `w8a16`（channel）；`--no_quant` 导出 fp16（`do_quantization=False`）。若 3.2 用 `--quant` 做了 GRQ，此处照常转换即可（`load_llm` 自动识别 GRQ 权重）。

#### 参数说明

| 参数                  | 类型   | 说明                              | 默认值                                          |
| ------------------- | ---- | ------------------------------- | -------------------------------------------- |
| `--model_size`      | str  | 模型变体                            | `1.6B`                                       |
| `--onnx_path`       | str  | Conv streaming ONNX 路径          | `../../model/llm/<basename>-convstream.onnx` |
| `--config`          | str  | LLM config.pkl 路径               | `../../model/llm/<basename>.config.pkl`      |
| `--rknn_path`       | str  | 输出 RKNN 路径                      | `../../model/llm/<basename>-convstream.rknn` |
| `--seq_len`         | int  | Prefill seq\_len                | `128`                                        |
| `--target_platform` | str  | 目标平台                            | `rk1820`                                     |
| `--quant_type`      | str  | 权重量化：`w4a16`/`w8a16`            | `w4a16`                                      |
| `--dataset_path`    | str  | 量化校准数据索引（`make_calidata.py` 生成） | `None`                                       |
| `--no_quant`        | bool | 导出 fp16（不量化）                    | `False`                                      |

## 5. C++ 板端部署

### 5.1 编译

```bash
cd rknn3_model_zoo/

export GCC_COMPILER=/path/to/aarch64-linux-gnu

./build-linux.sh -t rk3588 -a aarch64 -d LFM2_5_VL
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_LFM2_5_VL_demo/` 目录。

### 5.2 运行示例

```bash
./rknn_lfm2_vl_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt>
```

以 **LFM2.5-VL-1.6B** 为例：

```bash
./rknn_lfm2_vl_demo \
    model/vision/LFM2.5-VL-1.6B-vision-posemb.rknn \
    model/vision/LFM2.5-VL-1.6B-vision-posemb.weight \
    model/llm/LFM2.5-VL-1.6B-convstream.rknn \
    model/llm/LFM2.5-VL-1.6B-convstream.weight \
    model/llm/LFM2.5-VL-1.6B.tokenizer.gguf \
    model/llm/LFM2.5-VL-1.6B.embed.bin \
    0xff 0xff \
    demo.jpg \
    "Describe this image in detail, including the main objects, the colors and the overall atmosphere."
```

以 **LFM2.5-VL-3B** 为例：

```bash
./rknn_lfm2_vl_demo \
    model/vision/LFM2.5-VL-3B-vision-posemb.rknn \
    model/vision/LFM2.5-VL-3B-vision-posemb.weight \
    model/llm/LFM2.5-VL-3B-convstream.rknn \
    model/llm/LFM2.5-VL-3B-convstream.weight \
    model/llm/LFM2.5-VL-3B.tokenizer.gguf \
    model/llm/LFM2.5-VL-3B.embed.bin \
    0xff 0xff \
    demo.jpg \
    "Describe this image in detail, including the main objects, the colors and the overall atmosphere."
```

> **注意**：`vision_config.json` 和 `pos_emb_base.bin` 必须与 vision `.rknn` 模型放在同一目录（如上例的 `model/vision/`），板端启动时自动加载。二者由 `export_vision.py` 导出时生成（详见 3.1 节产物清单）。

参数说明：

| 参数                   | 说明                              |
| -------------------- | ------------------------------- |
| vision\_model\_path  | Vision RKNN 模型路径                |
| vision\_weight\_path | Vision 权重文件路径                   |
| llm\_model\_path     | LLM RKNN 模型路径（convstream）       |
| llm\_weight\_path    | LLM 权重文件路径                      |
| tokenizer\_path      | Tokenizer 文件路径（.tokenizer.gguf） |
| embedding\_path      | Embedding 文件路径（.embed.bin）      |
| vision\_core\_mask   | Vision NPU 核心掩码（16 进制）          |
| llm\_core\_mask      | LLM NPU 核心掩码（16 进制）             |
| image\_path          | 输入图片路径                          |
| prompt               | 提示词                             |

> **注意**：C++ 代码通过 tokenizer 动态获取 image token ID（1.6B=396, 3B=124907），无需手动配置。

## 6. Conv Streaming 技术说明

LFM2.5 的 LIV 层使用短卷积（kernel\_size=3, depthwise），在自回归生成中需要维护 Conv State。

- **ONNX 导出**：标准 4 输入 ONNX，将 Conv padding 从 `[2,2]` 改为 `[2,0]`
- **RKNN 转换**：使用 `cvt_conv_streaming` 将 Conv 节点转换为流式卷积（1.6B=10 个, 3B=22 个）
- **Session 推理**：`RKNN3_LLM_INPUT_MULTIMODAL` 高级 API 自动处理 Prefill/Decode 调度、Conv Streaming Cache 和 Attention KV Cache

## 7. 支持的模型

| 模型                      | 状态    |
| ----------------------- | ----- |
| LiquidAI/LFM2.5-VL-1.6B | ✅ 已支持 |
| LiquidAI/LFM2.5-VL-3B   | ✅ 已支持 |

