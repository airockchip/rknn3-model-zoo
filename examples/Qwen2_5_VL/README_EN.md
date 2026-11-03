# Qwen2.5-VL Model Deployment Guide

## 1. Model Pruning Strategy

To support larger context lengths, appropriate pruning can be applied when deploying multimodal models.

### 1.1 Vision Model Pruning

Some operators are offloaded to the CPU of the host device (e.g., RK3588).

### 1.2 LLM Model Pruning

The LLM Head is separated and runs independently on the host device, reducing memory usage on the coprocessor. (Optional)

### 1.3 Full Model Mode (No Pruning)

Devices with larger memory (e.g., RK1828) can use the full model directly. Add the `--no_prune_mode` parameter when exporting:

```bash
python export_rknn.py --no_prune_mode
```

## 2. Supported Models

Currently supports Qwen2.5-VL 3B, 7B, and other variants. Specify the corresponding model path when exporting.

Example with **Qwen2.5-VL-7B**:

```bash
# Export LLM ONNX model
python export_llm.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --export_llm_path ../../model/llm/Qwen2.5-VL-7B-llm.onnx \
    --modelscope

# Export LLM RKNN model
python export_rknn.py \
    --onnx_path ../../model/llm/Qwen2.5-VL-7B-llm.onnx \
    --config ../../model/llm/Qwen2.5-VL-7B-llm.config.pkl \
    --rknn_path ../../model/llm/Qwen2.5-VL-7B-llm.rknn \
    --platform rk1820

# Export Vision ONNX model
python export_vision.py \
    --model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --export_vision_path ../../model/vision/Qwen2.5-VL-7B-vision.onnx \
    --modelscope \
    --img_h 392 \
    --img_w 392

# Export Vision RKNN model
python export_rknn.py \
    --onnx_path ../../model/vision/Qwen2.5-VL-7B-vision.onnx \
    --rknn_path ../../model/vision/Qwen2.5-VL-7B-vision.rknn \
    --platform rk1820
```

## 3. KV Cache INT4 Quantization

In large-scale language model inference, a KV (Key/Value Cache) is used to store historical attention keys to avoid redundant calculations and thus improve inference speed. As sequence length increases, the memory footprint of the KV Cache increases rapidly. To reduce the storage bandwidth and memory access overhead of the KV Cache, quantization can be used to convert it from FP16/FP32 to INT8 or a lower bit width representation. However, since the KV Cache value distribution changes dynamically token by token over time, using a uniform quantization parameter for the entire KV segment can lead to accumulated quantization errors, affecting inference accuracy. Therefore, group quantization is typically used to reduce accuracy loss.

Currently, RKNN's LLM supports two KV cache quantization modes:

Int8_to_F16 (default): Stores in INT8 format, converts back to FP16 during computation;

Int4_to_F16 (suitable for longer contexts, with some precision loss): Stores in INT4 format, converts back to FP16 during computation.

For support of longer context lengths and further compression of KV cache memory, it is recommended to enable the Int4_to_F16 mode.

The configuration for enabling Int4_to_F16 RKNN model transformation is as follows:

```python
rknn.config(target_platform='rk1820', 
          quantized_dtype='w4a16', quantized_algorithm='normal', quantized_method='group32',
          max_ctx_len           =2048,
          max_position_embeddings=2048,
          kvcache_store_method='GroupQuant', kvcache_dtype='Int4_to_F16', 
          kvcache_group_size=16, kvcache_residual_depth=64,
          )
```
- Note: The above configuration is located in the python/llm/export_rknn.py file. Please adjust the relevant parameters according to your actual needs.


## 4. Vision Model Resolution Adjustment

Use `--img_h` and `--img_w` parameters to adjust input resolution (must be a multiple of 28):

```bash
python export_vision.py --img_h 392 --img_w 392
```

> ⚠️ **Note**:
> - Pruned version does not support resolution modification
> - Higher resolution increases memory usage and affects the maximum LLM context length
> - Some resolutions may be incompatible with the RKNN inference framework; contact the RKNPU team if errors occur

## 5. C++ Deployment

The C++ inference code automatically detects the model format and is compatible with both pruned and full models without code modification.

The C++ demo remains compatible with the original command line. To use SpeedUP, provide its header and library for the target platform, then add `--enable_speedup` to the `build-linux.sh` command. At runtime, append the optional `speedup_ratio` argument:

```bash
./rknn_qwen2_5_vl_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt> \
    [model_width model_height] [speedup_ratio]
```

`speedup_ratio` values:
- `1.0`: automatic mode.
- `0.0`: disabled.
- `(0.0, 1.0)`: manual mode.

If you modified the Vision model resolution, update the parameters in `rknn_qwen2_5_vl_vision.h`:

```cpp
#define MODEL_WIDTH  <your_width>
#define MODEL_HEIGHT <your_height>
```
