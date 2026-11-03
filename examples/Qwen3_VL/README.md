# Qwen3-VL 模型部署说明

## 1. 部署环境

由于 Qwen3-VL 模型对依赖库版本有特殊要求，与本仓库 `requirements.txt` 中的版本不兼容，请手动安装以下依赖：

```
torch >= 2.9.0
transformers == 4.57.1
onnxruntime >= 1.23.2
```

> ⚠️ 使用默认依赖会导致模型转换失败，请务必按上述版本安装。

## 2. 模型裁剪策略

为了支持更大的上下文长度，部署多模态模型时需进行适当裁剪。

### 2.1 Vision 模型裁剪

将部分算子迁移至 RK3588 等主控设备的 CPU 上运行。

### 2.2 LLM 模型裁剪

将 LM Head 独立出来在主控设备上单独运行以减少协处理器内存占用，此为可选模式且**默认关闭**，导出时添加 `--prune_mode` 参数即可开启：

```bash
python export_rknn.py --prune_mode --platform rk1820
```

### 2.3 完整模型模式（无裁剪）

默认即为完整模型模式，无需额外参数。RK1828 等内存较大的设备可直接使用：

```bash
python export_rknn.py --platform rk1820
```

### 2.4 KVCache 量化

如需进一步降低内存占用，可开启 KVCache INT 量化（将 KVCache 从 F16 量化为 Int4）。在 `export_rknn.py` 中取消以下行的注释即可：

```python
llm_config['attention_config'][0]['kvcache_dtype'] = 'Int4_to_F16'
```

修改后需重新导出 RKNN 模型。若仅修改了 kvcache 相关参数，可使用 `--rebuild` 跳过 ONNX 加载步骤加速导出：

```bash
python export_rknn.py \
    --onnx_path Qwen3-VL-2B-llm.onnx \
    --config Qwen3-VL-2B-llm.config.pkl \
    --rknn_path Qwen3-VL-2B-llm.rknn \
    --platform rk1820 \
    --rebuild
```

> ⚠️ 开启 KVCache 量化后精度可能有所下降，请根据实际效果决定是否启用。

## 3. 支持的模型

目前支持 Qwen3-VL 2B 和 4B 等模型。导出时请指定对应的模型路径。

以 **Qwen3-VL-4B** 为例：

```bash
# 生成 LLM 模型量化校准数据
python make_calidata.py --model_path Qwen/Qwen3-VL-4B-Instruct

# 导出 ONNX 模型
python export_llm.py --quant\
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_llm_path Qwen3-VL-4B-llm.onnx \
    --modelscope

# 导出 RKNN 模型
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm.rknn \
    --platform rk1820

# 如需重新导出（仅修改 profile_mode 或 kvcache 相关参数），可使用 --rebuild 跳过 ONNX 加载和图优化等步骤，加速模型导出：
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm.rknn \
    --platform rk1820 \
    --rebuild
```

> ⚠️ **`--rebuild` 说明**：当前仅支持重置 `profile_mode` 和 `llm_config` 中 kvcache 相关参数（如 `kvcache_buffer_len`、`max_position_embeddings`、`kvcache_dtype` 等）。其他参数变更需走完整导出流程。

`make_calidata.py` 参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | HuggingFace 模型路径或名称 | `Qwen/Qwen3-VL-2B-Instruct` |
| `--datapath` | 校准数据集 JSON 路径（含图片路径和文本输入） | `../../../../datasets/MMBench/llm/dataset.json` |
| `--export_datapath` | 生成的量化校准数据输出路径 | `./quant_data/model_inputs.json` |

> ⚠️ `make_calidata.py` 会加载完整模型并捕获模块输入，需要较大内存。生成的校准数据用于 `export_llm.py --quant` 时的 GRQ 量化。

## 4. Vision 模型分辨率调整

可通过 `--img_h` 和 `--img_w` 参数调整输入分辨率（必须为 32 的倍数）：

```bash
# 生成 Vision 模型量化校准数据
python make_calidata.py --model_path Qwen/Qwen3-VL-4B-Instruct

# 导出 Vision ONNX 模型
python export_vision.py --quant \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_vision_path Qwen3-VL-4B-vision.onnx \
    --img_h 384 --img_w 384 \
    --modelscope

# 导出 Vision RKNN 模型
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-vision.onnx \
    --rknn_path Qwen3-VL-4B-vision.rknn \
    --platform rk1820
```

> ⚠️ **注意**：
> - 分辨率越大，内存占用越高，会影响 LLM 的最大上下文长度
> - 部分分辨率可能与 RKNN 推理框架不兼容，如遇报错请联系 RKNPU 团队

`make_calidata.py` 参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | HuggingFace 模型路径或名称 | `Qwen/Qwen3-VL-2B-Instruct` |
| `--datapath` | 校准数据集 JSON 路径（含图片路径和文本输入） | `../../../../datasets/MMBench/llm/dataset.json` |
| `--export_datapath` | 生成的量化校准数据输出路径 | `./quant_data/model_inputs.json` |

> ⚠️ `make_calidata.py` 会加载完整模型并捕获模块输入，需要较大内存。生成的校准数据用于 `export_vision.py --quant` 时的 GRQ 量化。

## 5. C++ 部署说明

### 5.1 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# 编译
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_VL
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_Qwen3_VL_demo/` 目录。

### 5.2 运行

C++ 推理代码已实现模型格式自动识别，无需修改代码即可兼容裁剪版与完整版模型。

C++ demo 保持原有命令行参数兼容。使用 SpeedUP 时，请先准备好 SpeedUP 头文件及对应平台的库，并在 `build-linux.sh` 命令末尾添加 `--enable_speedup`。运行时可在命令末尾增加可选参数
`speedup_ratio`：

```bash
./rknn_qwen3_vl_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt> \
    [model_width model_height] [speedup_ratio]
```

`speedup_ratio` 取值说明：
- `1.0`：单图自动模式，多图不支持 Auto。
- `0.0`：关闭 SpeedUP。
- `(0.0, 1.0)`：手动模式。

若修改了 Vision 模型的分辨率，需同步调整 `rknn_qwen3_vl_vision.h` 中的参数：

```cpp
#define MODEL_WIDTH  <your_width>
#define MODEL_HEIGHT <your_height>
```

## 6. 常见问题

### ONNX 文件路径问题

使用 PyTorch ≥ 2.9.0 导出的模型会生成 `xxx.onnx` 和 `xxx.onnx.data` 两个文件。执行 `rknn.load_llm` 时，必须确保这两个文件在同一目录下，否则会报错：

```
RUNTIME_EXCEPTION: Exception during initialization: filesystem error: 
cannot get file size: No such file or directory [Qwen3-VL-4B-llm.onnx.data]
```

### PyTorch 版本不兼容

若使用 PyTorch < 2.9.0，执行 `rknn.load_llm` 时会报错：

```
RUNTIME EXCEPTION : Non-zero status code, returned while running Reshape node. 
...
The input tensor cannot be reshaped to the requested shape. Input shape:{384}, requested shape:{64,1}
```

请升级 PyTorch 至 ≥ 2.9.0 版本。
