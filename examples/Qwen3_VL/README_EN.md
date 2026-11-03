# Qwen3-VL Model Deployment Guide

## 1. Environment Requirements

Qwen3-VL has specific dependency requirements that are incompatible with the `requirements.txt` in this repository. Please install the following dependencies manually:

```
torch >= 2.9.0
transformers == 4.57.1
onnxruntime >= 1.23.2
```

> ⚠️ Using the default dependencies will cause model conversion to fail. Please install the versions specified above.

## 2. Model Pruning Strategy

To support larger context lengths, appropriate pruning is required when deploying large-scale multimodal models.

### 2.1 Vision Model Pruning

Some operators are offloaded to the CPU of the host device (e.g., RK3588).

### 2.2 LLM Model Pruning

The LLM Head is separated and runs independently on the host device to reduce memory usage on the coprocessor, this is an optional mode that is **disabled by default**. Add the `--prune_mode` parameter when exporting to enable it:

```bash
python export_rknn.py --prune_mode --platform rk1820
```

### 2.3 Full Model Mode (No Pruning)

The full model mode is used by default, no extra parameters are needed. Devices with larger memory (e.g., RK1828) can use the full model directly:

```bash
python export_rknn.py --platform rk1820
```

### 2.4 KVCache Quantization

To further reduce memory usage, KVCache INT quantization can be enabled (quantizing KVCache from F16 to Int4). Uncomment the following line in `export_rknn.py`:

```python
llm_config['attention_config'][0]['kvcache_dtype'] = 'Int4_to_F16'
```

After modifying, re-export the RKNN model. If only kvcache-related parameters were changed, use `--rebuild` to skip ONNX loading and speed up the export:

```bash
python export_rknn.py \
    --onnx_path Qwen3-VL-2B-llm.onnx \
    --config Qwen3-VL-2B-llm.config.pkl \
    --rknn_path Qwen3-VL-2B-llm.rknn \
    --platform rk1820 \
    --rebuild
```

> ⚠️ Enabling KVCache quantization may reduce precision. Please evaluate the actual results before enabling it.

## 3. Supported Models

Currently supports Qwen3-VL 2B, 4B, and other variants. Specify the corresponding model path when exporting.

Example with **Qwen3-VL-4B**:

```bash
# Generate calibration data for LLM model quantization
python make_calidata.py --model_path Qwen/Qwen3-VL-4B-Instruct

# Export ONNX model
python export_llm.py --quant \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_llm_path Qwen3-VL-4B-llm.onnx \
    --modelscope

# Export RKNN model
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm.rknn \
    --platform rk1820

# To re-export with only profile_mode or kvcache-related parameter changes, use --rebuild
# to skip ONNX loading and graph optimization, speeding up the export:
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-llm.onnx \
    --config Qwen3-VL-4B-llm.config.pkl \
    --rknn_path Qwen3-VL-4B-llm.rknn \
    --platform rk1820 \
    --rebuild
```

> ⚠️ **`--rebuild` note**: Currently only supports resetting `profile_mode` and kvcache-related parameters in `llm_config` (e.g., `kvcache_buffer_len`, `max_position_embeddings`, `kvcache_dtype`, etc.). Other parameter changes require a full export.

`make_calidata.py` parameter description:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_path` | HuggingFace model path or name | `Qwen/Qwen3-VL-2B-Instruct` |
| `--datapath` | Calibration dataset JSON path (contains image paths and text inputs) | `../../../../datasets/MMBench/llm/dataset.json` |
| `--export_datapath` | Output path for generated calibration data | `./quant_data/model_inputs.json` |

> ⚠️ `make_calidata.py` loads the full model and captures module inputs, requiring significant memory. The generated calibration data is used for GRQ quantization in `export_llm.py --quant`.

## 4. Vision Model Resolution Adjustment

Use `--img_h` and `--img_w` parameters to adjust input resolution (must be a multiple of 32):

```bash

# Generate calibration data for Vision model quantization
python make_calidata.py --model_path Qwen/Qwen3-VL-4B-Instruct

# Export Vision ONNX model
python export_vision.py --quant \
    --model_path Qwen/Qwen3-VL-4B-Instruct \
    --export_vision_path Qwen3-VL-4B-vision.onnx \
    --img_h 384 --img_w 384 \
    --modelscope

# Export Vision RKNN model
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-vision.onnx \
    --rknn_path Qwen3-VL-4B-vision.rknn \
    --platform rk1820
```

> ⚠️ **Note**:
> - Higher resolution increases memory usage and affects the maximum LLM context length
> - Some resolutions may be incompatible with the RKNN inference framework; contact the RKNPU team if errors occur

`make_calidata.py` parameter description:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_path` | HuggingFace model path or name | `Qwen/Qwen3-VL-2B-Instruct` |
| `--datapath` | Calibration dataset JSON path (contains image paths and text inputs) | `../../../../datasets/MMBench/llm/dataset.json` |
| `--export_datapath` | Output path for generated calibration data | `./quant_data/model_inputs.json` |

> ⚠️ `make_calidata.py` loads the full model and captures module inputs, requiring significant memory. The generated calibration data is used for GRQ quantization in `export_vision.py --quant`.

## 5. C++ Deployment

### 5.1 Build

```bash
cd rknn3_model_zoo/

# Set cross-compilation toolchain
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# Build
./build-linux.sh -t rk3588 -a aarch64 -d Qwen3_VL
```

After building, files are generated in `install/rk3588_linux_aarch64/rknn_Qwen3_VL_demo/`.

### 5.2 Run

The C++ demo remains compatible with the original command line. To use SpeedUP, provide its header and library for the target platform, then add `--enable_speedup` to the `build-linux.sh` command. At runtime, append the optional `speedup_ratio` argument:

```bash
./rknn_qwen3_vl_demo \
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

If you modified the Vision model resolution, update the parameters in `rknn_qwen3_vl_vision.h`:

```cpp
#define MODEL_WIDTH  <your_width>
#define MODEL_HEIGHT <your_height>
```

## 6. Troubleshooting

### ONNX File Path Issue

Models exported with PyTorch ≥ 2.9.0 generate both `xxx.onnx` and `xxx.onnx.data` files. When running `rknn.load_llm`, ensure both files are in the same directory, otherwise you'll get:

```
RUNTIME_EXCEPTION: Exception during initialization: filesystem error: 
cannot get file size: No such file or directory [Qwen3-VL-4B-llm.onnx.data]
```

### PyTorch Version Incompatibility

If using PyTorch < 2.9.0, `rknn.load_llm` will fail with:

```
RUNTIME EXCEPTION : Non-zero status code, returned while running Reshape node. 
...
The input tensor cannot be reshaped to the requested shape. Input shape:{384}, requested shape:{64,1}
```

Please upgrade PyTorch to ≥ 2.9.0.
