# Qwen3-VL LoRA Model Deployment Guide

## 1. Environment Requirements

Please refer to `requirements.txt` for the Python environment.

## 2. Model Pruning Strategy

To support larger context lengths, appropriate pruning is required when deploying large-scale multimodal models.

### 2.1 Vision Model Pruning

Some operators are offloaded to the CPU of the host device (e.g., RK3588).

### 2.2 LLM Model Pruning

The LLM Head is separated and runs independently on the host device, reducing memory usage on the coprocessor. (Optional)

### 2.3 Full Model Mode (No Pruning)

Devices with larger memory (e.g., RK1828) can use the full model directly. Add the `--no_prune_mode` parameter when exporting the Vision model:

```bash
python export_rknn.py --no_prune_mode
```

## 3. Supported Models

Currently supports Qwen3-VL 2B, 4B, and other variants. Specify the corresponding model path when exporting.

### 3.1 Base Model Download

Example with **Qwen3-VL-4B**:

| Model | Download Link |
|-------|---------------|
| Qwen3-VL-4B-Instruct | [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct) |

### 3.2 LoRA Model Download

This demo supports loading LoRA weights. The following are community-provided LoRA model examples:

| Model | Download Link |
|-------|---------------|
| qwen3-vl-4b-ui-confidence-lora | [HuggingFace](https://huggingface.co/bobbyzhong/qwen3-vl-4b-ui-confidence-lora) |
| qwen3-4b-vl-lora-vision-connector | [HuggingFace](https://huggingface.co/prashanth058/qwen3-4b-vl-lora-vision-connector) |

> ⚠️ **Note**:
> - Vision LoRA only changes the vision output when the adapter actually trained the vision tower, i.e. the `lora_B` of the vision layers in `adapter_model.safetensors` is non-zero.
> - `qwen3-4b-vl-lora-vision-connector` trains both the llm and vision parts and can be used directly for the Vision LoRA demo.
> - `qwen3-vl-4b-ui-confidence-lora` did not train the vision part; use it for the LLM LoRA example, not to verify a Vision output change.

### 3.3 Model Export Commands

Example with **Qwen3-VL-4B**:

```bash
# Replace these model, LoRA, and output directories with actual paths.
BASE_HF=/path/to/Qwen3-VL-4B-Instruct
LORA_DIR=/path/to/qwen3-vl-4b-ui-confidence-lora
VISION_LORA_DIR=/path/to/qwen3-4b-vl-lora-vision-connector
OUT_DIR=/path/to/qwen3_vl_4b_lora_rknn
REPO=/path/to/rknn3-model-zoo

cd /path/to/rknn3-model-zoo/examples/Qwen3_VL_LoRA/python/llm

# Export ONNX and generate the HF-weight-to-ONNX-op mapping.
python export_llm.py \
    --model_path "$BASE_HF" \
    --export_llm_path "$OUT_DIR/Qwen3-VL-4B-llm.onnx" \
    --hf_onnx_map "$OUT_DIR/Qwen3-VL-4B-llm.hf_onnx_map.json"

# Export RKNN model without LoRA.
python export_rknn.py \
    --onnx_path "$OUT_DIR/Qwen3-VL-4B-llm.onnx" \
    --config "$OUT_DIR/Qwen3-VL-4B-llm.config.pkl" \
    --rknn_path "$OUT_DIR/Qwen3-VL-4B-llm.rknn"

# Export RKNN model with LoRA.
python export_rknn.py \
    --onnx_path "$OUT_DIR/Qwen3-VL-4B-llm.onnx" \
    --config "$OUT_DIR/Qwen3-VL-4B-llm.config.pkl" \
    --rknn_path "$OUT_DIR/Qwen3-VL-4B-llm-lora.rknn" \
    --lora_path "$LORA_DIR/adapter_model.safetensors" \
    --lora_config_path "$LORA_DIR/adapter_config.json" \
    --hf_onnx_map "$OUT_DIR/Qwen3-VL-4B-llm.hf_onnx_map.json"

cd /path/to/rknn3-model-zoo/examples/Qwen3_VL_LoRA/python/vision

# Export Vision ONNX and generate the HF-weight-to-ONNX-op mapping.
python export_vision.py \
    --model_path "$BASE_HF" \
    --export_vision_path "$OUT_DIR/Qwen3-VL-4B-vision.onnx" \
    --img_h 384 --img_w 384 \
    --hf_onnx_map "$OUT_DIR/Qwen3-VL-4B-vision.hf_onnx_map.json"

# Export RKNN model with Vision LoRA.
python export_rknn.py \
    --onnx_path "$OUT_DIR/Qwen3-VL-4B-vision.onnx" \
    --rknn_path "$OUT_DIR/Qwen3-VL-4B-vision-lora.rknn" \
    --lora_path "$VISION_LORA_DIR/adapter_model.safetensors" \
    --lora_config_path "$VISION_LORA_DIR/adapter_config.json" \
    --hf_onnx_map "$OUT_DIR/Qwen3-VL-4B-vision.hf_onnx_map.json" \
    --dataset_path "$REPO/datasets/MMBench/vision/datasets.txt"
```

> ⚠️ **Note**:
> - LoRA weights (for example, `adapter_model.safetensors`) do not need to be converted to ONNX separately. The script reads the LoRA keys, builds an ONNX-node-based `lora_node_map` from `hf_onnx_map.json`, and calls `rknn.load_lora()`.
> - If an adapter contains both language and vision LoRA modules, the script creates an additional `*.llm.safetensors` containing only weights mapped to the LLM. The source `adapter_model.safetensors` is not overwritten.
> - If the model structure and quantization parameters are unchanged and you only need to re-export the `.rknn` file, use `--rebuild` for a quick rebuild. It depends on intermediate artifacts generated during the previous `build` phase in the `./tmp` directory.
> - `--rebuild` reuses the previous build state and cannot be combined with new `--lora_path` / `--lora_config_path` inputs; run a full export when changing the LoRA.
> - When exporting an RKNN model with LoRA, an additional `.lora_weight` file is generated alongside the `.rknn` file. Both files are required for C++ inference (`.rknn` as the model path, `.lora_weight` as the LoRA weight path).
> - Make sure `--lora_path` points to the correct LoRA weight file (e.g., `.safetensors`) and `--lora_config_path` points to the corresponding config file (e.g., `adapter_config.json`).
> - Both the LLM and Vision `export_rknn.py` accept `--platform` (`rk1820` / `rk1828` / `rk3572`, default `rk1820`).

### 3.4 HF Weight to ONNX Operator Mapping

After saving the RKNN-compatible ONNX, `export_llm.py` and `export_vision.py` write `*.hf_onnx_map.json`. The file records the ONNX initializer, consumer node name, operator type, transpose status, and L1 matching result for each HF key:

```json
{
  "mapping": {
    "model.language_model.layers.0.mlp.down_proj.weight": {
      "onnx_tensor": "...",
      "onnx_node": ["/language_model/layers.0/mlp/down_proj/MatMul"],
      "onnx_op_type": ["MatMul"],
      "transpose": false,
      "clean": true
    }
  }
}
```

Matching algorithm reference:

1. Filter candidates by shape, including transposed layouts for 2-D weights.
2. Use the absolute sum of the first 1024 elements as a fingerprint pre-filter.
3. Compare candidates with chunked L1 distance and greedily select an unused initializer.
4. Record the initializer's consumer node name and `op_type`, preferring `MatMul/Gemm` nodes.

> **Matching boundary:** The fingerprint is only a candidate pre-filter, and `clean=true` means only that the chunked L1 distance is at most 1.0; it is not a unique identifier or a cryptographic proof. If the graph contains same-shaped initializers with identical or near-identical values, duplicated weights, or shared weights, tensor values alone cannot distinguish their graph semantics. The current greedy matcher selects one candidate in traversal order, so an ambiguous mapping is possible. Before using a mapping, inspect `approx_matches` and `unmatched` in the top-level JSON `stats`, review each item's `l1` and `transpose`, and make sure the HF base model, LoRA adapter, and ONNX file come from the same revision. For ambiguous results, review the key/graph structure and run a Base/LoRA output regression check.
>
> MD5/SHA256 can audit whether values are exactly identical, but cannot disambiguate two initializers whose contents are identical while their graph semantics differ. A hash alone therefore cannot guarantee absolute node-mapping correctness.

In the LLM mapping, Vision weights are not present in the LLM ONNX; conversely, LLM weights are not present in the Vision ONNX. They remain in the mapping JSON and are marked as unmatched, which is expected.

### 3.5 LoRA Node Map Conversion and Loading

`export_rknn.py` first reads `--lora_path`, groups `.lora_A.weight` and `.lora_B.weight` into pairs, removes the PEFT wrapper prefix, and resolves each pair through the HF/ONNX mapping. The resulting RKNN map has the following form:

```python
{
    "/language_model/layers.0/mlp/down_proj/MatMul": [
        [[
            "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight",
            "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight",
            0,
        ]]
    ]
}
```

The actual call is:

```python
rknn.load_lora(
    lora_paths=[[lora_load_path]],
    lora_config_paths=[[args.lora_config_path]],
    lora_patterns=[["lora0_pattern0"]],
    lora_quantized_dtype="int4",
    lora_quantized_method="group32",
    lora_distribute_strategy="less_mem",
    lora_node_map=lora_node_map,
)
```

When using `lora_node_map`, do not pass `lora_prefix`, `lora_postfix_a`, or `lora_postfix_b` at the same time. The script generates these auxiliary files by default:

```text
<rknn_stem>.lora_node_map.json
<rknn_stem>.lora_node_map.report.json
<rknn_stem>.llm.safetensors       # generated when unmapped tensors exist
```

Output paths can be specified explicitly:

```bash
--lora_node_map_output /path/to/model.lora_node_map.json \
--lora_node_map_report /path/to/model.lora_node_map.report.json \
--lora_filtered_path /path/to/model.llm.safetensors
```

### 3.6 Mapping Validation

For the Qwen3-VL-4B base model and a 4B adapter containing Vision targets, the expected result is approximately:

| Item | Count |
|------|------:|
| Source LoRA tensors | 696 |
| Source A/B pairs | 348 |
| Pairs mapped to the LLM ONNX | 252 |
| Unmapped vision pairs | 96 |
| Tensors retained after filtering | 504 |
| ONNX nodes in the node map | 252 |
| `rknn.load_llm()` return value | 0 |
| `rknn.load_lora(lora_node_map=...)` return value | 0 |

If the mapped-pair count changes significantly, first check that the HF base model, LoRA adapter, and ONNX model come from the same model revision. Do not manually rename keys as a first workaround.

### 3.7 Vision LoRA Node Assignment

Vision follows exactly the same flow as the LLM and reuses the same matcher (`python/lora_node_map.py`):

```text
export_vision.py  ──►  *.hf_onnx_map.json      (HF key -> ONNX initializer/op)
export_rknn.py    ──►  *.lora_node_map.json    (exact ONNX node -> LoRA A/B keys)
                  ──►  *.lora_node_map.report.json
                  ──►  *.vision.safetensors    (vision-only weights, when unmapped tensors exist)
                  ──►  *.rknn + *.lora_weight  (model with Vision LoRA)
```

**1. Export the Vision ONNX and its HF/ONNX mapping**

```bash
cd /path/to/rknn3-model-zoo/examples/Qwen3_VL_LoRA/python/vision

python export_vision.py \
    --model_path "$BASE_HF" \
    --export_vision_path "$OUT_DIR/Qwen3-VL-4B-vision.onnx" \
    --img_h 384 --img_w 384 \
    --hf_onnx_map "$OUT_DIR/Qwen3-VL-4B-vision.hf_onnx_map.json"
```

`export_vision.py` builds the mapping from `model.visual.state_dict()`. The keys look like `model.visual.blocks.0.attn.qkv.weight`, which is exactly what a PEFT adapter key such as `base_model.model.model.visual....lora_A.weight` resolves to after the `base_model.model.` wrapper prefix is stripped, so no manual renaming is needed.

**2. Export the RKNN model with Vision LoRA**

```bash
python export_rknn.py \
    --onnx_path "$OUT_DIR/Qwen3-VL-4B-vision.onnx" \
    --rknn_path "$OUT_DIR/Qwen3-VL-4B-vision-lora.rknn" \
    --lora_path "$VISION_LORA_DIR/adapter_model.safetensors" \
    --lora_config_path "$VISION_LORA_DIR/adapter_config.json" \
    --hf_onnx_map "$OUT_DIR/Qwen3-VL-4B-vision.hf_onnx_map.json" \
    --dataset_path "$REPO/datasets/MMBench/vision/datasets.txt"
```

> ⚠️ `--dataset_path` is the quantization calibration **image list** (one image path per line; relative paths are resolved against the list file's directory). The bundled `datasets/MMBench/vision/datasets.txt` works directly; the script caches the pruned patches under `<example>/data/vision/`. Without this option it falls back to `../../data/vision/datasets_npy.txt`.

`export_rknn.py` groups `.lora_A.weight` / `.lora_B.weight` into pairs, resolves each pair through `--hf_onnx_map` to a real Vision ONNX node, and keeps only the pairs that belong to the vision tower. Language-model pairs are filtered out into `<rknn_stem>.vision.safetensors` (the source `adapter_model.safetensors` is never overwritten). The generated map looks like:

```python
{
    "/vl_m/blocks.0/attn/qkv/Gemm": [
        [[
            "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight",
            "base_model.model.model.visual.blocks.0.attn.qkv.lora_B.weight",
            0,
        ]]
    ]
}
```

The actual call is:

```python
rknn.load_lora(
    lora_paths=[[lora_load_path]],
    lora_config_paths=[[args.lora_config_path]],
    lora_patterns=[["lora0_pattern0"]],
    lora_quantized_dtype="int4",
    lora_quantized_method="group32",
    lora_distribute_strategy="less_mem",
    lora_node_map=lora_node_map,
)
```

As with the LLM, exporting a LoRA-enabled Vision RKNN also produces a `.lora_weight` file. At runtime it must be loaded into the vision context (`rknn3_lora_init` → `query` → `lora_load` → `lora_enable`).

Output paths can be specified explicitly:

```bash
--lora_node_map_output /path/to/model.lora_node_map.json \
--lora_node_map_report /path/to/model.lora_node_map.report.json \
--lora_filtered_path /path/to/model.vision.safetensors
```

**3. Mapping Validation**

For the Qwen3-VL-4B base model and the `qwen3-vl-4b-ui-confidence-lora` adapter, the expected result is approximately:

| Item | Count |
|------|------:|
| Source LoRA tensors | 696 |
| Source A/B pairs | 348 |
| Pairs mapped to the Vision ONNX | 96 |
| LLM pairs filtered out | 252 |
| Tensors retained after filtering | 192 |
| ONNX nodes in the node map | 96 |
| `rknn.load_onnx()` return value | 0 |
| `rknn.load_lora(lora_node_map=...)` return value | 0 |

Vision LoRA normally targets the four linear layers `blocks.*/attn/qkv`, `blocks.*/attn/proj`, `blocks.*/mlp/linear_fc1`, and `blocks.*/mlp/linear_fc2` (24 blocks x 4 = 96 pairs for the 4B model). If the mapped-pair count changes significantly, first check that the HF base model, LoRA adapter, and Vision ONNX come from the same model revision.

If you use `qwen3-vl-4b-ui-confidence-lora`, mapped Vision pairs do not necessarily mean that the Vision output changes: that adapter did not train the Vision part. Use `qwen3-4b-vl-lora-vision-connector` to verify the Vision LoRA effect.

> ⚠️ **Note**: `--no_prune_mode` (full model) and the pruned model share the same ONNX and the same node map. Pruning only removes operators before the input; node names inside the Vision blocks are unchanged.

## 4. Vision Model Resolution Adjustment

Use `--img_h` and `--img_w` parameters to adjust input resolution (must be a multiple of 32):

```bash
# Export Vision ONNX model (resolution specified via --img_h/--img_w)
python export_vision.py \
    --model_path /path/to/Qwen3-VL-4B-Instruct \
    --export_vision_path Qwen3-VL-4B-vision.onnx \
    --img_h 384 --img_w 384

# Export Vision RKNN model (resolution auto-read from vision_config.json)
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-vision.onnx \
    --rknn_path Qwen3-VL-4B-vision.rknn
```

> ⚠️ **Note**:
> - Higher resolution increases memory usage and affects the maximum LLM context length
> - Some resolutions may be incompatible with the RKNN inference framework; contact the RKNPU team if errors occur
> - Vision RKNN export resolution is controlled by `vision_config.json`; no need to pass `--img_h`/`--img_w` again

## 5. C++ Deployment

The C++ inference code automatically detects the model format and is compatible with both pruned and full models without code modification. When running the demo, you **must** pass `model_width` and `model_height` via command line so they match the Vision export resolution (must be multiples of 32).

### 5.1 Base and LoRA Dual-Path Inference

The demo creates the Base LLM session at initialization and toggles LoRA on a single vision context on demand:

| Session / Module | Description |
|------------------|-------------|
| **Vision (base)** | Vision inference without LoRA, producing the base vision embedding. |
| **Vision (LoRA)** | Vision inference with LoRA enabled on the vision context, producing the LoRA vision embedding (requires a vision LoRA weight). |
| **Base LLM**  | Loads only the base model weights and consumes the base vision embedding. |
| **LoRA LLM**  | Loads LoRA weights on top of the base model and consumes the LoRA vision embedding. |

**Inference flow** for the same image and prompt:

1. Vision runs once in **base** mode (LoRA disabled); the result goes into the base deepstack buffers.
2. If a vision LoRA weight is provided, Vision runs again with LoRA enabled; the result goes into dedicated LoRA deepstack buffers.
3. **Base LLM** runs with the base vision embedding.
4. **LoRA LLM** runs with the LoRA vision embedding.

Each pass prints its own generated text and performance (Prefill/Generate, Vision latency, etc.). Without a vision LoRA weight, Vision runs only once and both LLM passes share the same vision embedding.

> ⚠️ The LoRA session is created only after the base pass, and the base session is destroyed first — two sessions coexisting on one context exhaust the runtime transfer buffer and make `rknn_session_run` fail.

> **Scope and performance boundary:** The demo performs one Base → LoRA A/B comparison per process; it does not repeatedly create two LLM sessions in the same process. If the current lifecycle is adapted for high-frequency Base/LoRA alternation, repeatedly creating, initializing, and destroying sessions adds latency and may clear the KV cache or other session state. A high-frequency service should keep the context resident and separately evaluate context-level `rknn3_lora_enable()` / `rknn3_lora_disable()` switching. The demo's LLM path currently uses session-level `rknn3_session_enable_lora()`; such a change must be validated against the SDK and model. The Vision path already toggles `rknn3_lora_enable()` / `rknn3_lora_disable()` on one context and does not recreate the Vision session.

### 5.2 Command-Line Arguments and Usage

Usage:

```text
./rknn_qwen3_vl_demo <vision_model_path> <vision_weight_path> <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> <vision_core_mask> <llm_core_mask> <image_path> <prompt> \
    <model_width> <model_height> <max_context_len1> <max_context_len2> \
    [llm_lora_weight_path] [vision_lora_weight_path]
```

| Argument count (excluding executable) | Meaning |
|--------------------------------------|---------|
| **14**    | Required arguments (including `model_width`, `model_height`, `max_context_len1`, `max_context_len2`). |
| **15**    | The 14 required arguments plus `llm_lora_weight_path` for LoRA weights. |
| **16**    | The 15 arguments plus `vision_lora_weight_path` for the Vision LoRA weights. |

> The count above excludes the executable itself; therefore the corresponding C++ `argc` values remain 15/16/17. Passing `-` (or an empty string) for either LoRA path disables LoRA for that part (it falls back to base behaviour). This lets you enable Vision LoRA only: pass `-` as the 15th argument and the vision `.lora_weight` as the 16th.

New context-length arguments:

| Argument | Description |
|----------|-------------|
| `max_context_len1` | Maximum context length for the **Base session** (`session_base params.max_context_len`). |
| `max_context_len2` | Maximum context length for the **LoRA session** (`session_lora params.max_context_len`). |

> Both arguments are required and must be greater than 0. You can set them independently based on device memory, e.g., Base = 2048 and LoRA = 3072.

**Example (14 args; resolution must match Vision export, e.g., 384x384)**:

```bash
./rknn_qwen3_vl_demo ... 384 384 2048 3072
```

**Compare Base and LoRA (15 args, LLM LoRA only)**:

```bash
./rknn_qwen3_vl_demo \
    /path/to/model/vision.rknn /path/to/model/vision.weight \
    /path/to/model/llm.rknn /path/to/model/llm.weight \
    /path/to/model/tokenizer.gguf /path/to/model/embed.bin \
    0x3 0x3 \
    /path/to/model/demo.jpg "Describe this image." \
    384 384 \
    2048 3072 \
    /path/to/model/llm_lora.weight
```

**LoRA on both Vision and LLM (16 args)**:

```bash
./rknn_qwen3_vl_demo \
    /path/to/model/vision.rknn /path/to/model/vision.weight \
    /path/to/model/llm.rknn /path/to/model/llm.weight \
    /path/to/model/tokenizer.gguf /path/to/model/embed.bin \
    0x3 0x3 \
    /path/to/model/demo.jpg "Describe this image." \
    384 384 \
    2048 3072 \
    /path/to/model/llm_lora.weight /path/to/model/vision_lora.weight
```

> The LoRA paths must point to the `*.lora_weight` files produced during export (e.g. `Qwen3-VL-4B-vision-lora_lora0_pattern0.lora_weight`).

The program runs **Vision(base) → Vision(LoRA) → Base model → LoRA model** in order and prints both sets of results and metrics.

### 5.3 LoRA API Usage

The following describes the RKNN3 LoRA APIs and how this demo uses them.

#### API List and Call Order

| API | Description |
|-----|-------------|
| `rknn3_lora_init(context, lora_weight_path)` | Initialize LoRA from file; must be called before any load/enable. |
| `rknn3_lora_init_from_data(context, weight_data, weight_size)` | Initialize LoRA from memory buffer for direct loading from memory. |
| `rknn3_query(context, RKNN3_QUERY_LORA_NUM, &n_lora, sizeof(n_lora))` | Query the number of LoRAs available in the context. |
| `rknn3_query(context, RKNN3_QUERY_LORA_INFO, lora_list, sizeof(lora_list))` | Query the list of LoRAs available in the context, including lora_name and scale. |
| `rknn3_lora_load(context, lora)` | Load the given LoRA adapter into the context; different sessions under the same context share the same LoRA weights. |
| `rknn3_lora_enable(context, lora)` | Enable a loaded LoRA at the context level. |
| `rknn3_lora_disable(context, lora)` | Disable an enabled LoRA at the context level. |
| `rknn3_lora_unload(context, lora)` | Unload a loaded LoRA; this will automatically disable all enabled LoRAs. |
| `rknn3_session_enable_lora(session, lora)` | Enable LoRA for a specific session; this will automatically clear all KV caches. |
| `rknn3_session_disable_lora(session, lora)` | Disable LoRA for a specific session; this will automatically clear all KV caches. |

**Recommended order**: `lora_init` → `query(RKNN3_QUERY_LORA_NUM)` → `query(RKNN3_QUERY_LORA_INFO)` → `lora_load` → `session_enable_lora` → `lora_unload`.

#### Where this demo uses them

LoRA setup is done in `cpp/llm/rknn_qwen3_vl_llm.cc` inside `setup_context_lora()`, which is called from `init_qwen3_vl_llm()` when `lora_weight_path` is non-null:

1. `rknn3_lora_init(ctx, lora_weight_path)` — set LoRA weight path (file or directory).
2. `rknn3_query(ctx, RKNN3_QUERY_LORA_NUM, &n_lora, sizeof(n_lora))` — get LoRA count.
3. `rknn3_query(ctx, RKNN3_QUERY_LORA_INFO, lora_list, sizeof(lora_list))` — get LoRA list.
4. `rknn3_lora_load(ctx, &lora_list[0])` — load that LoRA into the context.
5. `rknn3_session_enable_lora(session_lora, &lora_list[0])` — enable it for the LoRA session.

Inference uses `inference_qwen3_vl_llm_base()` for the base session and `inference_qwen3_vl_llm_lora()` for the LoRA session.

Vision has no session, so it uses **context-level LoRA** in `cpp/vision/rknn_qwen3_vl_vision.cc`:

1. `setup_qwen3_vl_vision_lora()` — called once the internal memory is configured; it runs `rknn3_lora_init()` → `rknn3_query()` → `rknn3_lora_load()` and loads without enabling, so the first inference is the base pass.
2. `set_vision_lora_enabled()` — `inference_qwen3_vl_vision()` calls `rknn3_lora_enable()` / `rknn3_lora_disable()` right before `rknn3_run()` according to its `use_lora` argument, producing base and LoRA outputs from the same context.
3. `release_qwen3_vl_vision()` — `disable` then `unload` before destroying the context.

Two sets of deepstack buffers are used (`deepstack_tensor` and `deepstack_tensor_lora`): base vision writes into the former for the Base LLM session, LoRA vision writes into the latter for the LoRA LLM session.
