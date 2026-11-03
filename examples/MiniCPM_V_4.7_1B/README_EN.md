# MiniCPM-V-4.7-1B Model Deployment Guide

Model: [openbmb/MiniCPM-V-4.7-1B](https://huggingface.co/openbmb/MiniCPM-V-4.7-1B)

## 1. Deployment Environment

> ⚠️ The latest version of rknn3-toolkit is required.
```bash
rknn3-toolkit>=1.1.0
torch==2.7.0
torchvision==0.22.0
transformers==5.18.0
numpy==2.5.2
```

## 2. Exporting RKNN Models

### 2.1 Exporting the LLM ONNX Model

```bash
cd examples/MiniCPM_V_4.7_1B/python/llm

# Generate LLM quantization calibration data (optional)
python make_calidata.py --model_path openbmb/MiniCPM-V-4.7-1B

# Export LLM ONNX model
python export_llm.py \
    --model_path openbmb/MiniCPM-V-4.7-1B \
    --export_llm_path ../../model/llm/MiniCPM-V-4.7-1B-llm.onnx

# Export LLM ONNX model (GRQ quantization, requires a CUDA environment)
python export_llm.py --quant \
    --model_path openbmb/MiniCPM-V-4.7-1B \
    --export_llm_path ../../model/llm/MiniCPM-V-4.7-1B-llm.onnx

# Add the --modelscope flag to download the model from ModelScope
python export_llm.py --modelscope
```

`make_calidata.py` arguments:

| Argument | Description | Default |
|------|------|--------|
| `--model_path` | HuggingFace model path or name | `openbmb/MiniCPM-V-4.7-1B` |
| `--datapath` | Calibration dataset JSON path (contains image paths and text inputs) | `../../../../datasets/MMBench/llm/dataset.json` |
| `--export_datapath` | Output path for the generated quantization calibration data | `./quant_data/model_inputs.json` |
| `--modelscope` | Download the model from ModelScope | `False` |

> ⚠️ `make_calidata.py` loads the full model and captures the multimodal inputs flowing into the
> language backbone `language_model`, so it requires a large amount of memory.
> The generated calibration data (`./quant_data/model_inputs.json` + `model_inputs/sample_N`) is
> consumed by `export_llm.py --quant` as the GRQ quantization dataset
> (`--cali_dataset ./quant_data/model_inputs.json`); the `language_model` submodule is quantized.

After running `export_llm.py`, the following files are generated alongside the ONNX file:

```
MiniCPM-V-4.7-1B-llm.onnx
MiniCPM-V-4.7-1B-llm.config.pkl
MiniCPM-V-4.7-1B-llm.tokenizer.gguf
MiniCPM-V-4.7-1B-llm.embed.bin
```

### 2.2 Exporting the LLM RKNN Model

```bash
python export_rknn.py \
    --onnx_path ../../model/llm/MiniCPM-V-4.7-1B-llm.onnx \
    --config ../../model/llm/MiniCPM-V-4.7-1B-llm.config.pkl \
    --rknn_path ../../model/llm/MiniCPM-V-4.7-1B-llm.rknn \
    --platform rk1820

# To re-export (only for changes to profile_mode or kvcache-related parameters), use --rebuild
# to skip ONNX loading and graph optimization steps for faster export:
python export_rknn.py \
    --onnx_path ../../model/llm/MiniCPM-V-4.7-1B-llm.onnx \
    --config ../../model/llm/MiniCPM-V-4.7-1B-llm.config.pkl \
    --rknn_path ../../model/llm/MiniCPM-V-4.7-1B-llm.rknn \
    --platform rk1820 \
    --rebuild
```

> ⚠️ **About `--rebuild`**: Currently only `profile_mode` and kvcache-related parameters in
> `llm_config` (such as `kvcache_buffer_len` or `max_position_embeddings`) can be reset. Any other
> parameter change requires the full export flow.

### 2.3 Exporting the Vision ONNX Model

```bash
cd examples/MiniCPM_V_4.7_1B/python/vision

# Generate vision quantization calibration data (optional)
python make_calidata.py --model_path /path/to/MiniCPM-V-4.7-1B \
    --datapath datasets/MMBench/llm/dataset.json

# Export Vision ONNX (448x448 image input)
python export_vision.py \
    --model_path openbmb/MiniCPM-V-4.7-1B \
    --export_vision_path ../../model/vision/MiniCPM-V-4.7-1B-vision.onnx

# Export Vision ONNX (GRQ quantization, requires a CUDA environment)
python export_vision.py --quant \
    --model_path openbmb/MiniCPM-V-4.7-1B \
    --export_vision_path ../../model/vision/MiniCPM-V-4.7-1B-vision.onnx
```

> The vision model itself is a **single-crop encoder** with a fixed 448×448 input (32×32 patches).
> Slicing of large images is implemented by calling this model multiple times on the C++ side;
> re-exporting the model is not required. See §4.

### 2.4 Exporting the Vision RKNN Model

```bash
python export_rknn.py \
    --onnx_path ../../model/vision/MiniCPM-V-4.7-1B-vision.onnx \
    --rknn_path ../../model/vision/MiniCPM-V-4.7-1B-vision.rknn \
    --platform rk1820
```

## 3. C++ Board Deployment

### 3.1 Build

```bash
cd rknn3_model_zoo/

# Set the cross-compilation toolchain
export GCC_COMPILER=/path/to/aarch64-linux-gnu

# Build
./build-linux.sh -t rk3588 -a aarch64 -d MiniCPM_V_4.7_1B
```

After the build completes, the files are generated in the
`install/rk3588_linux_aarch64/rknn_MiniCPM_V_4.7_1B_demo/` directory:

```
rknn_MiniCPM_V_4.7_1B_demo/
├── lib
│   ├── librga.so
│   └── librknn3_api.so
├── model
│   ├── MiniCPM-V-4.7-1B-llm.embed.bin
│   ├── MiniCPM-V-4.7-1B-llm.rknn
│   ├── MiniCPM-V-4.7-1B-llm.tokenizer.gguf
│   ├── MiniCPM-V-4.7-1B-llm.weight
│   ├── MiniCPM-V-4.7-1B-vision.rknn
│   ├── MiniCPM-V-4.7-1B-vision.weight
│   └── demo.jpg
└── rknn_minicpmv4_7_demo
```

### 3.2 Deploy to the Development Board

```bash
# Push the demo directory
adb push rknn_MiniCPM_V_4.7_1B_demo /data/

# Push the runtime libraries
adb push rknn_MiniCPM_V_4.7_1B_demo/lib/* /usr/lib/
```

### 3.3 Run the Example

```bash
./rknn_minicpmv4_7_demo \
    <vision_model_path> <vision_weight_path> \
    <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> \
    <vision_core_mask> <llm_core_mask> \
    <image_path> <prompt>
```

**Image-text multimodal mode** (pass the actual vision model paths):

```bash
export LD_LIBRARY_PATH=./lib

./rknn_minicpmv4_7_demo \
    model/MiniCPM-V-4.7-1B-vision.rknn model/MiniCPM-V-4.7-1B-vision.weight \
    model/MiniCPM-V-4.7-1B-llm.rknn model/MiniCPM-V-4.7-1B-llm.weight \
    model/MiniCPM-V-4.7-1B-llm.tokenizer.gguf model/MiniCPM-V-4.7-1B-llm.embed.bin \
    0xff 0xff \
    model/demo.jpg \
    "Describe this image"
```

**Text-only mode** (pass an empty string `""` for the vision model paths to automatically fall back
and initialize only the LLM):

```bash
./rknn_minicpmv4_7_demo \
    "" "" \
    model/MiniCPM-V-4.7-1B-llm.rknn model/MiniCPM-V-4.7-1B-llm.weight \
    model/MiniCPM-V-4.7-1B-llm.tokenizer.gguf model/MiniCPM-V-4.7-1B-llm.embed.bin \
    0xff 0xff \
    "" \
    "Explain the theory of relativity"
```

Arguments:

| Argument           | Description                                                           |
| ------------------ | --------------------------------------------------------------------- |
| vision_model_path  | Vision RKNN model path; pass an empty string `""` for text-only inference |
| vision_weight_path | Vision weight file path (set `""` in text-only mode)                   |
| llm_model_path     | LLM RKNN model path                                                   |
| llm_weight_path    | LLM RKNN weight file path                                             |
| tokenizer_path     | Tokenizer file path (.tokenizer.gguf)                                 |
| embedding_path     | Embedding weight file path (.embed.bin)                               |
| vision_core_mask   | NPU core mask for the vision model, in hexadecimal, e.g. 0xff         |
| llm_core_mask      | NPU core mask for the LLM model, in hexadecimal, e.g. 0xff            |
| image_path         | Input image path (set `""` in text-only mode)                          |
| prompt             | User input text; enclose in double quotes if it contains spaces       |

> ⚠️ The demo uses `top_k=1`, `top_p=0.9`, `temperature=1.0`, and `repeat_penalty=1.2` by default.

## 4. Slice Support

The C++ demo supports the same slicing flow as the model source: a large image is first split into
**1 thumbnail + rows×cols slices**, each crop is encoded separately and then concatenated in order
into the multimodal placeholders, with position encoding using canvas M-RoPE (canvas coordinate
system).

### 4.1 Behavior

**The vision model is unchanged.** The exported vision rknn is still a "single crop 448×448 → 64
tokens" encoder; slicing is implemented by the C++ side **calling the same model once per crop**.
Because crops are completely independent inside the ViT (attention is blocked by `cu_seqlens` and
crops cannot see each other), and every crop is resized to the same size (448×448), per-crop calls
are equivalent to a single batched call—**changing the number of crops neither requires changing the
model nor dynamic shapes**.

**Position encoding (canvas M-RoPE)**: The whole image shares one "time" position T, and all tokens
lie on a single `rows*8 × cols*8` canvas (each crop is 8×8 on the LLM side):

| token | T | H | W |
|---|---|---|---|
| `<image>` | base | base−1 | base−1 |
| Thumbnail tokens | base | `linspace` stretched to fill the canvas | same as left |
| `</image>` | base | base+canvas_H | base+canvas_W |
| `<slice>` / `</slice>` | base | Row of the slice's top-left / bottom-right corner | Column of the slice's top-left / bottom-right corner |
| Line break `\n` | base | Last row of that line | Right outside the line |
| Text after the image | Continues from `base + max(canvas_H, canvas_W) + 1` | same as left | same as left |

The thumbnail is a low-resolution global view of the entire canvas (hence the `linspace` stretch),
while slices are high-resolution local views of cells on the canvas; both share the same coordinate
system. **Embed injection**: the thumbnail is written by the `image.*` template, while slices are
injected token by token from the visual features via `embed_callback` according to the
`<slice>`/`</slice>` markers.

### 4.2 Usage

The command line is exactly the same as before; slicing is performed automatically when the image
area exceeds `448²`:

```bash
./rknn_minicpmv4_7_demo \
    model/MiniCPM-V-4.7-1B-vision.rknn model/MiniCPM-V-4.7-1B-vision.weight \
    model/MiniCPM-V-4.7-1B-llm.rknn     model/MiniCPM-V-4.7-1B-llm.weight \
    model/MiniCPM-V-4.7-1B-llm.tokenizer.gguf model/MiniCPM-V-4.7-1B-llm.embed.bin \
    0xff 0xff \
    model/demo.jpg \
    "Describe this image"
```

Force a single slice (saves compute):

```bash
export MINICPMV4_7_NO_SLICE=1
./rknn_minicpmv4_7_demo \
    model/MiniCPM-V-4.7-1B-vision.rknn model/MiniCPM-V-4.7-1B-vision.weight \
    model/MiniCPM-V-4.7-1B-llm.rknn     model/MiniCPM-V-4.7-1B-llm.weight \
    model/MiniCPM-V-4.7-1B-llm.tokenizer.gguf model/MiniCPM-V-4.7-1B-llm.embed.bin \
    0xff 0xff \
    model/demo.jpg \
    "Describe this image"
```

Text-only mode (pass an empty vision path) is unaffected.
