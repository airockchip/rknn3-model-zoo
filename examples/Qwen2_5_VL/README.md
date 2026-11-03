# Qwen2.5-VL 模型部署说明

## 1. 模型裁剪策略

为了支持更大的上下文长度，部署多模态模型时可进行适当裁剪。

### 1.1 Vision 模型裁剪

将部分算子迁移至 RK3588 等主控设备的 CPU 上运行。

### 1.2 LLM 模型裁剪

将 LLM Head 独立出来，在主控设备上单独运行，从而减少协处理器的内存占用。（可选）

### 1.3 完整模型模式（无裁剪）

RK1828 等内存较大的设备可直接使用完整模型，导出时添加 `--no_prune_mode` 参数：

```bash
python export_rknn.py --no_prune_mode
```

## 2. 支持的模型

目前支持 Qwen2.5-VL 3B 和 7B 等模型。导出时请指定对应的模型路径。

以 **Qwen2.5-VL-7B** 为例：

```bash
# 导出 LLM ONNX 模型
python export_llm.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --export_llm_path ../../model/llm/Qwen2.5-VL-7B-llm.onnx \
    --modelscope

# 导出 LLM RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/llm/Qwen2.5-VL-7B-llm.onnx \
    --config ../../model/llm/Qwen2.5-VL-7B-llm.config.pkl \
    --rknn_path ../../model/llm/Qwen2.5-VL-7B-llm.rknn \
    --platform rk1820

# 导出 Vision ONNX 模型
python export_vision.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --export_vision_path ../../model/vision/Qwen2.5-VL-7B-vision.onnx \
    --modelscope \
    --img_h 392 \
    --img_w 392

# 导出 Vision RKNN 模型
python export_rknn.py \
    --onnx_path ../../model/vision/Qwen2.5-VL-7B-vision.onnx \
    --rknn_path ../../model/vision/Qwen2.5-VL-7B-vision.rknn \
    --platform rk1820
```

## 3. KV Cache INT4 量化
在大规模语言模型推理过程中，KV Cache（Key/Value Cache）用于存储历史的注意力键值，以避免重
复计算，从而提高推理速度。随着序列长度增长，KV Cache 的内存占用会快速增加。为了减少 KV
Cache 的存储带宽与内存访问开销，可以采用量化方式将其从 FP16/FP32 转换为 INT8 或更低位宽表
示。但由于 KV Cache 数值分布随时间逐 token 动态变化，如果对整段 KV 使用统一的量化参数，会导致
量化误差累积从而影响推理精度。因此通常采用分组量化（Group Quantization）来降低精度损失。
目前，RKNN 的LLM支持两种 KV Cache 量化模式：
Int8_to_F16（默认）：以 INT8 格式存储，计算时转换回 FP16；
Int4_to_F16（适用于更长上下文场景,有一定精度损失）：以 INT4 格式存储，计算时转换回 FP16。
若需支持更长的上下文长度并进一步压缩 KV Cache 内存，建议启用 Int4_to_F16 模式。

> ⚠️ **注意**：当前 `python/llm/export_rknn.py` 的 `rknn.config()` 仅配置了 MRoPE 相关参数
> （`mrope_type`、`mrope_section`、`mrope_new_id_name`），并未启用 KV Cache 量化。实际配置如下：

```python
llm_config = DEFAULT_RKNN_LLM_CONFIG.copy()
llm_config['attention_config'][0]['mrope_type'] = 'Qwen2.5-VL'
llm_config['attention_config'][0]['mrope_section'] = [16, 24, 24]
llm_config['attention_config'][0]['mrope_new_id_name'] = 'mrope_id_input'

rknn.config(target_platform=args.platform,
            quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
            llm_config=llm_config)
```

若需启用 Int4_to_F16 以支持更长上下文，可在 `llm_config['attention_config'][0]` 中增加：

```python
llm_config['attention_config'][0]['kvcache_dtype'] = 'Int4_to_F16'
```

修改后需重新执行 RKNN 转换。


## 4. Vision 模型分辨率调整

可通过 `--img_h` 和 `--img_w` 参数调整输入分辨率（必须为 28 的倍数）：

```bash
python export_vision.py --img_h 392 --img_w 392
```

> ⚠️ **注意**：
> - 裁剪版本不支持修改分辨率
> - 分辨率越大，内存占用越高，会影响 LLM 的最大上下文长度
> - 部分分辨率可能与 RKNN 推理框架不兼容，如遇报错请联系 RKNPU 团队

## 5. C++ 部署说明

C++ 推理代码已实现模型格式自动识别，无需修改代码即可兼容裁剪版与完整版模型。

C++ demo 保持原有命令行参数兼容。使用 SpeedUP 时，请先准备好 SpeedUP 头文件及对应平台的库，并在 `build-linux.sh` 命令末尾添加 `--enable_speedup`。运行时可在命令末尾增加可选参数
`speedup_ratio`：

```bash
./rknn_qwen2_5_vl_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt> \
    [model_width model_height] [speedup_ratio]
```

`speedup_ratio` 取值说明：
- `1.0`：自动模式。
- `0.0`：关闭 SpeedUP。
- `(0.0, 1.0)`：手动模式。

若修改了 Vision 模型的分辨率，需同步调整 `rknn_qwen2_5_vl_vision.h` 中的参数：

```cpp
#define MODEL_WIDTH  <your_width>
#define MODEL_HEIGHT <your_height>
```
