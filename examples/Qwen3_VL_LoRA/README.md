# Qwen3-VL LoRA模型部署说明

## 1. 部署环境

python环境参考 `requirements.txt` 

## 2. 模型裁剪策略

为了支持更大的上下文长度，部署多模态模型时需进行适当裁剪。

### 2.1 Vision 模型裁剪

将部分算子迁移至 RK3588 等主控设备的 CPU 上运行。

### 2.2 LLM 模型裁剪

将 LLM Head 独立出来，在主控设备上单独运行，从而减少协处理器的内存占用。（可选）

### 2.3 完整模型模式（无裁剪）

RK1828 等内存较大的设备可直接使用完整模型。Vision 模型导出时添加 `--no_prune_mode` 参数即可关闭裁剪：

```bash
python export_rknn.py --no_prune_mode
```

## 3. 支持的模型

目前支持 Qwen3-VL 2B 和 4B 等模型。导出时请指定对应的模型路径。

### 3.1 基座模型下载

以 **Qwen3-VL-4B** 为例：

| 模型 | 下载链接 |
|------|----------|
| Qwen3-VL-4B-Instruct | [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-4B-Instruct) |

### 3.2 LoRA 模型下载

本demo支持加载LoRA权重。以下为社区提供的 LoRA 模型示例：

| 模型 | 下载链接 |
|------|----------|
| qwen3-vl-4b-ui-confidence-lora | [HuggingFace](https://huggingface.co/bobbyzhong/qwen3-vl-4b-ui-confidence-lora) |
| qwen3-4b-vl-lora-vision-connector | [HuggingFace](https://huggingface.co/prashanth058/qwen3-4b-vl-lora-vision-connector) |

> ⚠️ **注意**：
> - Vision LoRA 只有 adapter 真正训练过 vision tower 时才会改变 vision 输出，即 `adapter_model.safetensors` 中 vision 层的 `lora_B` 非零。
> - `qwen3-4b-vl-lora-vision-connector` 同时训练了 llm 与 vision 部分，可直接用于 Vision LoRA 演示。
> - `qwen3-vl-4b-ui-confidence-lora` 未训练 vision 部分；它适合用于 LLM LoRA 示例，不适合验证 Vision 输出变化。

### 3.3 模型导出命令

以 **Qwen3-VL-4B** 为例：

```bash
# 模型、LoRA 和输出目录均替换为实际路径
BASE_HF=/path/to/Qwen3-VL-4B-Instruct
LORA_DIR=/path/to/qwen3-vl-4b-ui-confidence-lora
VISION_LORA_DIR=/path/to/qwen3-4b-vl-lora-vision-connector
OUT_DIR=/path/to/qwen3_vl_4b_lora_rknn
REPO=/path/to/rknn3-model-zoo

cd /path/to/rknn3-model-zoo/examples/Qwen3_VL_LoRA/python/llm

# 导出 ONNX，同时生成 HF 权重名到 ONNX initializer/op 的映射
python export_llm.py \
    --model_path "$BASE_HF" \
    --export_llm_path "$OUT_DIR/Qwen3-VL-4B-llm.onnx" \
    --hf_onnx_map "$OUT_DIR/Qwen3-VL-4B-llm.hf_onnx_map.json"

# 导出 RKNN 模型（不含 LoRA）
python export_rknn.py \
    --onnx_path "$OUT_DIR/Qwen3-VL-4B-llm.onnx" \
    --config "$OUT_DIR/Qwen3-VL-4B-llm.config.pkl" \
    --rknn_path "$OUT_DIR/Qwen3-VL-4B-llm.rknn"

# 导出 RKNN 模型（含 LoRA）
python export_rknn.py \
    --onnx_path "$OUT_DIR/Qwen3-VL-4B-llm.onnx" \
    --config "$OUT_DIR/Qwen3-VL-4B-llm.config.pkl" \
    --rknn_path "$OUT_DIR/Qwen3-VL-4B-llm-lora.rknn" \
    --lora_path "$LORA_DIR/adapter_model.safetensors" \
    --lora_config_path "$LORA_DIR/adapter_config.json" \
    --hf_onnx_map "$OUT_DIR/Qwen3-VL-4B-llm.hf_onnx_map.json"

cd /path/to/rknn3-model-zoo/examples/Qwen3_VL_LoRA/python/vision

# 导出 Vision ONNX，同时生成 HF 权重名到 ONNX op 的映射
python export_vision.py \
    --model_path "$BASE_HF" \
    --export_vision_path "$OUT_DIR/Qwen3-VL-4B-vision.onnx" \
    --img_h 384 --img_w 384 \
    --hf_onnx_map "$OUT_DIR/Qwen3-VL-4B-vision.hf_onnx_map.json"

# 导出 RKNN 模型（含 Vision LoRA）
python export_rknn.py \
    --onnx_path "$OUT_DIR/Qwen3-VL-4B-vision.onnx" \
    --rknn_path "$OUT_DIR/Qwen3-VL-4B-vision-lora.rknn" \
    --lora_path "$VISION_LORA_DIR/adapter_model.safetensors" \
    --lora_config_path "$VISION_LORA_DIR/adapter_config.json" \
    --hf_onnx_map "$OUT_DIR/Qwen3-VL-4B-vision.hf_onnx_map.json" \
    --dataset_path "$REPO/datasets/MMBench/vision/datasets.txt"
```

> ⚠️ **注意**：
> - LoRA 权重（如 `adapter_model.safetensors`）无需单独转换为 ONNX。脚本会读取 LoRA key，结合 `hf_onnx_map.json` 生成基于真实 ONNX 节点名的 `lora_node_map`，再调用 `rknn.load_lora()`。
> - 如果一个 adapter 同时包含语言和视觉 LoRA，脚本会额外生成只包含 LLM 权重的 `*.llm.safetensors`；源 `adapter_model.safetensors` 不会被覆盖。
> - 导出含 LoRA 的 RKNN 模型时，除 `.rknn` 文件外还会额外生成一个 `.lora_weight` 文件。C++ 推理时需同时提供这两个文件（`.rknn` 作为模型路径，`.lora_weight` 作为 LoRA 权重路径）。
> - 导出时需确保 `--lora_path` 指向正确的 LoRA 权重文件（如 `.safetensors`），`--lora_config_path` 指向对应的配置文件（如 `adapter_config.json`）。
> - 如果模型结构和量化参数未变，仅需重新导出 `.rknn` 文件，可使用 `--rebuild` 快速重建。重建依赖上次 `build` 阶段在 `./tmp` 目录下生成的中间产物。
> - `--rebuild` 会复用上一次 build 的中间状态，不能与本次新的 `--lora_path` / `--lora_config_path` 一起使用；更换 LoRA 时请执行完整导出。
> - LLM 与 Vision 的 `export_rknn.py` 都支持 `--platform` 指定目标平台（`rk1820` / `rk1828` / `rk3572`），默认 `rk1820`。

### 3.4 HF 权重到 ONNX 算子映射

`export_llm.py` 和 `export_vision.py` 在保存 RKNN 兼容的 ONNX 后，都会生成 `*.hf_onnx_map.json`。文件中会记录每个 HF key 对应的 ONNX initializer、消费者节点名、算子类型、是否转置以及 L1 匹配结果，例如：

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

匹配算法参考：

1. 按 shape 筛选候选 initializer，二维权重同时支持转置布局。
2. 使用前 1024 个元素的绝对值和做 fingerprint 预筛选。
3. 对候选 initializer 做分块 L1 比较，贪心选取最相近且未使用的 initializer。
4. 记录 initializer 的消费者节点名和 `op_type`，优先选择 `MatMul/Gemm` 节点。

> **匹配边界**：fingerprint 只用于候选预筛选，`clean=true` 只表示分块 L1 距离不超过 1.0，不是唯一标识或密码学证明。若图中存在 shape 相同且数值相同/接近的 initializer、重复权重或共享权重，单靠张量数值无法区分它们的图语义；当前贪心流程会按候选顺序选择一个结果，可能产生歧义。使用映射前请检查 JSON 顶层 `stats` 中的 `approx_matches`、`unmatched`，并核对每项的 `l1`、`transpose` 以及 HF 基座、LoRA adapter 和 ONNX 是否来自同一版本。对存在歧义的映射，应结合 key/图结构人工复核并做 Base/LoRA 输出回归验证。
>
> MD5/SHA256 可以作为“数值是否完全一致”的审计信息，但无法解决两个 initializer 内容完全相同却对应不同节点的语义歧义，因此不能单独保证节点映射绝对正确。

在 LLM 映射中，Vision 权重不在 LLM ONNX 内；在 Vision 映射中，LLM 权重不在 Vision ONNX 内。它们都会保留在映射 JSON 中并标记为 unmatched，这是正常现象。

### 3.5 LoRA node map 转换和加载

`export_rknn.py` 会先加载 `--lora_path`，按 `.lora_A.weight` 和 `.lora_B.weight` 成对解析 key，再去除 PEFT 的包装前缀，与 HF/ONNX 映射进行匹配。最终传入 RKNN 的 map 格式如下：

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

实际调用为：

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

使用 `lora_node_map` 时，不再同时传 `lora_prefix`、`lora_postfix_a`、`lora_postfix_b`。脚本默认生成以下辅助文件：

```text
<rknn_stem>.lora_node_map.json
<rknn_stem>.lora_node_map.report.json
<rknn_stem>.llm.safetensors       # 存在未映射 tensor 时生成
```

也可以通过以下参数指定输出路径：

```bash
--lora_node_map_output /path/to/model.lora_node_map.json \
--lora_node_map_report /path/to/model.lora_node_map.report.json \
--lora_filtered_path /path/to/model.llm.safetensors
```

### 3.6 映射结果检查

对于 Qwen3-VL-4B base 和包含 Vision target 的 4B adapter，正常结果约为：

| 项目 | 数量 |
|---|---:|
| LoRA 源 tensor | 696 |
| LoRA 源 A/B 对 | 348 |
| 映射到 LLM ONNX 的 pair | 252 |
| 未映射的视觉 pair | 96 |
| 过滤后保留 tensor | 504 |
| node map 中的 ONNX 节点 | 252 |
| `rknn.load_llm()` 返回值 | 0 |
| `rknn.load_lora(lora_node_map=...)` 返回值 | 0 |

如果 mapped pair 数量明显变化，应优先检查 base HF、LoRA adapter 和 ONNX 是否来自同一模型版本，不要直接手工改 key 名称。

### 3.7 Vision LoRA 节点分配

Vision 走与 LLM 完全相同的流程，复用同一套匹配算法（`python/lora_node_map.py`）：

```text
export_vision.py  ──►  *.hf_onnx_map.json      （HF key → ONNX initializer/op）
export_rknn.py    ──►  *.lora_node_map.json    （精确 ONNX 节点 → LoRA A/B key）
                  ──►  *.lora_node_map.report.json
                  ──►  *.vision.safetensors    （只保留 Vision LoRA 权重，存在未映射 tensor 时生成）
                  ──►  *.rknn + *.lora_weight  （含 Vision LoRA 的模型）
```

**1. 导出 Vision ONNX 并生成 HF/ONNX 映射**

```bash
cd /path/to/rknn3-model-zoo/examples/Qwen3_VL_LoRA/python/vision

python export_vision.py \
    --model_path "$BASE_HF" \
    --export_vision_path "$OUT_DIR/Qwen3-VL-4B-vision.onnx" \
    --img_h 384 --img_w 384 \
    --hf_onnx_map "$OUT_DIR/Qwen3-VL-4B-vision.hf_onnx_map.json"
```

`export_vision.py` 使用 `model.visual.state_dict()` 生成映射，HF key 形如 `model.visual.blocks.0.attn.qkv.weight`，与 PEFT adapter 中 `base_model.model.model.visual....lora_A.weight` 去掉 `base_model.model.` 包装前缀后的 key 完全对应，因此无需手工改 key 名称。

**2. 导出含 Vision LoRA 的 RKNN 模型**

```bash
python export_rknn.py \
    --onnx_path "$OUT_DIR/Qwen3-VL-4B-vision.onnx" \
    --rknn_path "$OUT_DIR/Qwen3-VL-4B-vision-lora.rknn" \
    --lora_path "$VISION_LORA_DIR/adapter_model.safetensors" \
    --lora_config_path "$VISION_LORA_DIR/adapter_config.json" \
    --hf_onnx_map "$OUT_DIR/Qwen3-VL-4B-vision.hf_onnx_map.json" \
    --dataset_path "$REPO/datasets/MMBench/vision/datasets.txt"
```

> ⚠️ `--dataset_path` 是 Vision 量化校准用的**图片路径列表**（每行一个图片路径，相对路径以列表文件所在目录为基准）。可直接使用仓库自带的 `datasets/MMBench/vision/datasets.txt`，脚本会把裁剪版输入 patch 缓存到 `<example>/data/vision/`；不传该参数时默认读取 `../../data/vision/datasets_npy.txt`。

`export_rknn.py` 先按 `.lora_A.weight` / `.lora_B.weight` 成对解析 key，再通过 `--hf_onnx_map` 解析出对应的 Vision ONNX 节点，只保留能匹配到 Vision ONNX 的 pair，语言模型的 pair 会被过滤掉并写入 `<rknn_stem>.vision.safetensors`（源 `adapter_model.safetensors` 不会被覆盖）。生成的 map 形如：

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

实际调用为：

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

与 LLM 相同，导出含 LoRA 的 Vision RKNN 时会额外生成一个 `.lora_weight` 文件；运行期需要把它加载到 vision context（`rknn3_lora_init` → `query` → `lora_load` → `lora_enable`）。

默认输出路径可通过以下参数指定：

```bash
--lora_node_map_output /path/to/model.lora_node_map.json \
--lora_node_map_report /path/to/model.lora_node_map.report.json \
--lora_filtered_path /path/to/model.vision.safetensors
```

**3. 映射结果检查**

以 Qwen3-VL-4B base 和 `qwen3-vl-4b-ui-confidence-lora` adapter 为例，正常结果约为：

| 项目 | 数量 |
|---|---:|
| LoRA 源 tensor | 696 |
| LoRA 源 A/B 对 | 348 |
| 映射到 Vision ONNX 的 pair | 96 |
| 被过滤掉的 LLM pair | 252 |
| 过滤后保留 tensor | 192 |
| node map 中的 ONNX 节点 | 96 |
| `rknn.load_onnx()` 返回值 | 0 |
| `rknn.load_lora(lora_node_map=...)` 返回值 | 0 |

Vision LoRA 一般作用在 `blocks.*/attn/qkv`、`blocks.*/attn/proj`、`blocks.*/mlp/linear_fc1`、`blocks.*/mlp/linear_fc2` 四类线性层上（4B 模型 24 个 block × 4 = 96 个 pair）。若 mapped pair 数量明显变化，同样应先检查 base HF、LoRA adapter 与 Vision ONNX 是否来自同一模型版本。

如果使用 `qwen3-vl-4b-ui-confidence-lora`，即使能映射出 Vision pair，也不代表视觉输出会变化；该 adapter 未训练 Vision 部分，建议使用 `qwen3-4b-vl-lora-vision-connector` 验证 Vision LoRA 效果。

> ⚠️ **注意**：`--no_prune_mode` 完整版与裁剪版使用同一份 ONNX 和同一份 node map；裁剪只截断输入前的算子，不改变 Vision block 内的节点名。

## 4. Vision 模型分辨率调整

可通过 `--img_h` 和 `--img_w` 参数调整输入分辨率（必须为 32 的倍数）：

```bash
# 导出 Vision ONNX 模型（分辨率通过 --img_h/--img_w 指定）
python export_vision.py \
    --model_path /path/to/Qwen3-VL-4B-Instruct \
    --export_vision_path Qwen3-VL-4B-vision.onnx \
    --img_h 384 --img_w 384

# 导出 Vision RKNN 模型（分辨率从 vision_config.json 自动读取）
python export_rknn.py \
    --onnx_path Qwen3-VL-4B-vision.onnx \
    --rknn_path Qwen3-VL-4B-vision.rknn
```

> ⚠️ **注意**：
> - 分辨率越大，内存占用越高，会影响 LLM 的最大上下文长度
> - 部分分辨率可能与 RKNN 推理框架不兼容，如遇报错请联系 RKNPU 团队
> - Vision RKNN 导出时分辨率由 `vision_config.json` 控制，无需再次传入 `--img_h`/`--img_w`

## 5. C++ 部署说明

C++ 推理代码已实现模型格式自动识别，无需修改代码即可兼容裁剪版与完整版模型。运行 demo 时**必须**通过命令行传入与 Vision 模型导出分辨率一致的 `model_width`、`model_height`（须为 32 的倍数）。

### 5.1 Base 与 LoRA 双路推理说明

本示例在初始化时创建 Base LLM session，并在同一个 vision context 上按需切换 LoRA：

| Session / 模块 | 说明 |
|---------------|------|
| **Vision（base）** | 未启用 LoRA 的 vision 推理，输出 base vision embedding。 |
| **Vision（LoRA）** | 在 vision context 上启用 LoRA 后的 vision 推理，输出带 LoRA 的 vision embedding（需传入 vision LoRA 权重）。 |
| **Base LLM**  | 仅加载基座模型权重，消费 base vision embedding。 |
| **LoRA LLM**  | 在基座基础上加载 LoRA 权重，消费带 LoRA 的 vision embedding。 |

**推理流程**：对同一张图片与同一 prompt，

1. Vision 跑一次 **base**（不启用 LoRA），结果写入 base 专用 deepstack 缓冲区；
2. 若传入 vision LoRA 权重，Vision 再跑一次并启用 LoRA，结果写入 lora 专用 deepstack 缓冲区；
3. **Base LLM** 使用 base vision embedding 推理；
4. **LoRA LLM** 使用带 LoRA 的 vision embedding 推理。

两段分别输出生成结果与性能（Prefill/Generate、Vision 耗时等）。未传 vision LoRA 权重时 Vision 只跑一次，两段 LLM 共用同一份 vision embedding。

> ⚠️ LoRA session 在 base 段推理结束后才创建，创建前会先释放 base session——同一 context 同时存在两个 session 会挤占运行期传输缓冲，导致 `rknn_session_run` 分配失败。

> **适用范围与性能边界**：当前 demo 是每个进程执行一次 Base → LoRA 的 A/B 对比流程；它不会在同一进程内反复交替创建两个 LLM session。若按当前生命周期改造成高频 Base/LoRA 交替推理，反复创建、初始化和销毁 session 会带来额外延迟，并可能清空 KV cache 或其他 session 状态。高频服务应保持 context 常驻，并单独评估 context 级 `rknn3_lora_enable()` / `rknn3_lora_disable()` 的切换方案；当前 demo 的 LLM 路径使用的是 session 级 `rknn3_session_enable_lora()`，该改造需要结合 SDK 和模型验证。Vision 路径已经在同一 context 上使用 `rknn3_lora_enable()` / `rknn3_lora_disable()`，不需要重建 Vision session。

### 5.2 命令行参数与用法

可执行文件用法：

```text
./rknn_qwen3_vl_demo <vision_model_path> <vision_weight_path> <llm_model_path> <llm_weight_path> \
    <tokenizer_path> <embedding_path> <vision_core_mask> <llm_core_mask> <image_path> <prompt> \
    <model_width> <model_height> <max_context_len1> <max_context_len2> \
    [llm_lora_weight_path] [vision_lora_weight_path]
```

| 参数个数（不含可执行文件名） | 含义 |
|------------------------------|------|
| **14**   | 必选参数（包含 `model_width`、`model_height`、`max_context_len1`、`max_context_len2`）。 |
| **15**   | 在 14 个必选参数基础上增加 `llm_lora_weight_path`，用于 LoRA 权重路径。 |
| **16**   | 在 15 个参数基础上增加 `vision_lora_weight_path`，用于 Vision 的 LoRA 权重路径。 |

> 这里按用户实际传入的参数计数，不包含可执行文件本身；因此 C++ 内部对应的 `argc` 仍为 15/16/17。`llm_lora_weight_path` / `vision_lora_weight_path` 支持传 `-` 或空串表示该段不使用 LoRA（此时分别回退为 base 行为），因此可以只开 Vision LoRA：第 15 个参数传 `-`，第 16 个参数传 vision 的 `.lora_weight`。

新增上下文长度参数说明：

| 参数 | 说明 |
|------|------|
| `max_context_len1` | **Base session** 的最大上下文长度（`session_base params.max_context_len`）。 |
| `max_context_len2` | **LoRA session** 的最大上下文长度（`session_lora params.max_context_len`）。 |

> 两个参数均为必填，且必须大于 0。可根据设备内存情况分别设置，例如 Base 设为 2048、LoRA 设为 3072。

**示例（14 参数，分辨率需与导出 Vision 时一致，如 384×384）**：

```bash
./rknn_qwen3_vl_demo ... 384 384 2048 3072
```

**同时对比 Base 与 LoRA（15 参数，仅 LLM 带 LoRA）**：

```bash
./rknn_qwen3_vl_demo \
    /path/to/model/vision.rknn /path/to/model/vision.weight \
    /path/to/model/llm.rknn /path/to/model/llm.weight \
    /path/to/model/tokenizer.gguf /path/to/model/embed.bin \
    0x3 0x3 \
    /path/to/model/demo.jpg "描述这张图片" \
    384 384 \
    2048 3072 \
    /path/to/model/llm_lora.weight
```

**Vision + LLM 都带 LoRA（16 参数）**：

```bash
./rknn_qwen3_vl_demo \
    /path/to/model/vision.rknn /path/to/model/vision.weight \
    /path/to/model/llm.rknn /path/to/model/llm.weight \
    /path/to/model/tokenizer.gguf /path/to/model/embed.bin \
    0x3 0x3 \
    /path/to/model/demo.jpg "描述这张图片" \
    384 384 \
    2048 3072 \
    /path/to/model/llm_lora.weight /path/to/model/vision_lora.weight
```

> 注意 LoRA 权重路径要指向导出时生成的 `*.lora_weight` 文件（如 `Qwen3-VL-4B-vision-lora_lora0_pattern0.lora_weight`）。

程序会按 **Vision(base) → Vision(LoRA) → Base model → LoRA model** 的顺序执行，并输出两段结果与性能。

### 5.3 LoRA 相关 API 使用说明

以下为 RKNN3 LoRA 接口及在本 demo 中的使用方式。

#### 接口列表与调用顺序

| 接口 | 说明 |
|------|------|
| `rknn3_lora_init(context, lora_weight_path)` | 从文件初始化 LoRA 环境，必须在 load/enable 之前调用。 |
| `rknn3_lora_init_from_data(context, weight_data, weight_size)` | 从内存数据初始化 LoRA 环境，用于直接从内存加载 LoRA 权重。 |
| `rknn3_query(context, RKNN3_QUERY_LORA_NUM, &n_lora, sizeof(n_lora))` | 查询当前 context 中可用的 LoRA 数量。 |
| `rknn3_query(context, RKNN3_QUERY_LORA_INFO, lora_list, sizeof(lora_list))` | 查询当前 context 中可用的 LoRA 信息列表，包括lora_name和scale |
| `rknn3_lora_load(context, lora)` | 将指定 LoRA 适配器加载进 context，相同context的不同session共享同一个LoRA权重。 |
| `rknn3_lora_enable(context, lora)` | 在 context 级别启用已加载的 LoRA。 |
| `rknn3_lora_disable(context, lora)` | 在 context 级别关闭已启用的 LoRA。 |
| `rknn3_lora_unload(context, lora)` | 卸载已加载的 LoRA，会自动disable所有启用的LoRA |
| `rknn3_session_enable_lora(session, lora)` | 为指定 session 启用 LoRA，调用后会自动清空所有kvcache |
| `rknn3_session_disable_lora(session, lora)` | 为指定 session 关闭 LoRA，调用后会自动清空所有kvcache |

**推荐顺序**：`lora_init` → `query(RKNN3_QUERY_LORA_NUM)` → `query(RKNN3_QUERY_LORA_INFO)` → `lora_load` → `session_enable_lora` → `lora_unload`。

#### 本 demo 中的实现位置

LoRA 初始化集中在 `cpp/llm/rknn_qwen3_vl_llm.cc` 的 `setup_context_lora()` 中（由 `init_qwen3_vl_llm()` 在传入 `lora_weight_path` 时调用）：

1. `rknn3_lora_init(ctx, lora_weight_path)` — 指定 LoRA 权重路径（文件或目录）。
2. `rknn3_query(ctx, RKNN3_QUERY_LORA_NUM, &n_lora, sizeof(n_lora))` — 获取 LoRA 数量。
3. `rknn3_query(ctx, RKNN3_QUERY_LORA_INFO, lora_list, sizeof(lora_list))` — 获取 LoRA 列表。
4. `rknn3_lora_load(ctx, &lora_list[0])` — 在 context 上加载该 LoRA。
5. `rknn3_session_enable_lora(session_lora, &lora_list[0])` — 为 LoRA session 启用该 LoRA。

推理时通过 `inference_qwen3_vl_llm_base()` 使用 Base session，通过 `inference_qwen3_vl_llm_lora()` 使用已加载 LoRA 的 session。

Vision 侧没有 session，使用 **context 级 LoRA**，实现在 `cpp/vision/rknn_qwen3_vl_vision.cc`：

1. `setup_qwen3_vl_vision_lora()`：在内部内存配置完成后执行 `rknn3_lora_init()` → `rknn3_query()` → `rknn3_lora_load()`，只加载不启用，保证第一次推理是 base。
2. `set_vision_lora_enabled()`：`inference_qwen3_vl_vision()` 在 `rknn3_run()` 前根据 `use_lora` 调用 `rknn3_lora_enable()` / `rknn3_lora_disable()`，在同一 context 上得到 base / LoRA 两套输出。
3. `release_qwen3_vl_vision()`：退出前 `disable` 并 `unload`。

deepstack 使用两套缓冲区（`deepstack_tensor` / `deepstack_tensor_lora`）：base vision 写前者供 Base LLM session，带 LoRA 的 vision 写后者供 LoRA LLM session。
