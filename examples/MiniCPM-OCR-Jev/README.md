# MiniCPM-OCR Jev（OCR 正确性判别）模型部署说明

本示例把 **[MiniCPM-OCR Jev](https://huggingface.co/openbmb/MiniCPM-OCR-Jev)** 部署到 RKNN3。MiniCPM-OCR Jev 是一个 **2B 的 OCR 正确性判别模型（correctness decision
model）**，面向文档中的文本 / 公式 / 表格区域：给定「区域原图 + 候选 OCR 结果」，输出 **0–1 的分数**，分数越高表示该候选结果越可能正确。可用于自动放行可信区域、把其余区域送人工复核，或在多个 OCR 模型的结果中择优。

本示例的输入是「原图 + OCR 文本渲染图 + OCR 文本」三元组，输出是「一致 / 不一致」的二分类打分。

**模型输入契约**

```text
<|im_start|>user\n
<image>OCR文本：\n{ocr_text}\n<image>第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。
请判断该 OCR 文本及其对应渲染图是否与原图内容一致。
这是一个二分类任务：一致标为 1，不一致标为 0。<|im_end|>\n
```

## 1. 环境要求

MiniCPM-OCR Jev 的导出依赖固定版本环境。请优先使用如下版本，避免因 transformers、ONNX 或 ONNXRuntime 版本差异导致模型结构、动态维度或算子导出不一致。

```bash
torch                             2.7.0
transformers                      5.3.0
onnx                              1.18.0
onnxruntime                       1.22.1
```

> ⚠️ **注意**：
>
> - 分类头（`score.weight`）必须存在于 `model.safetensors` 中；脚本只加载**文本主干** `Qwen3_5TextModel`，不加载视觉塔，也不需要 `visual.*` 权重。
> - LLM 与 Vision 是两个独立模型，需要**分别**导出 ONNX、分别转 RKNN，运行时由 C++ 串起来。
> - 所有命令都假定当前目录是 `examples/MiniCPM-OCR-Jev`。模型**只从网络下载**（默认 HuggingFace，`--modelscope` 切到 ModelScope）。

## 2. 支持的模型

**MiniCPM-OCR Jev**（HuggingFace 仓库：`openbmb/MiniCPM-OCR-Jev`，Apache-2.0）。

- **定位**：2B 的 OCR **正确性判别**模型，不是生成模型。对文本 / 公式 / 表格区域给出候选结果的可信度分数（0–1，越高越可能正确）。
- **结构**：从 Qwen3.5-2B 训练而来。文本主干 24 层（18 层 `linear_attention` + 6 层全注意力），hidden=2048，vocab=248320；`lm_head` 被替换为二分类头 `score.weight [2, 2048]`，视觉编码器与语言主干保留。

本示例只导出**判别所需**的两部分：LLM（文本主干 + 分类头）与 Vision（视觉塔）。

| 站点 | 触发方式 | 行为 |
|------|----------|------|
| HuggingFace（默认） | 直接运行 | `huggingface_hub.snapshot_download`，遵循 `HF_ENDPOINT`（默认已切到 `https://hf-mirror.com/`） |
| ModelScope | 加 `--modelscope` | `modelscope.snapshot_download`，无需 `HF_ENDPOINT` |

> 💡 直连 huggingface.co 不通时，用 `--hf_endpoint` 指定镜像（默认已经是 `https://hf-mirror.com/`），或提前导出 `HF_ENDPOINT` 环境变量；也可以直接用 `--modelscope` 走国内站点。

```bash
# 从 HuggingFace 下载（默认，等价于 --model_path openbmb/MiniCPM-OCR-Jev）
python export_onnx.py --output_dir ../../model/llm --name value_llm

# 从 ModelScope 下载
python export_onnx.py --modelscope \
    --output_dir ../../model/llm --name value_llm
```

### 2.1 导出 Vision ONNX 模型

```bash
# 进入脚本目录（--output_dir 的相对路径是相对当前工作目录的）
cd examples/MiniCPM-OCR-Jev/python/vision

# 导出 320x832 视觉塔（默认分辨率）
# --model_path 省略时默认从 HuggingFace 下载 openbmb/MiniCPM-OCR-Jev；
# 加 --modelscope 则改从 ModelScope 下载。
python export_onnx.py \
    --height 320 --width 832 \
    --output_dir ../../model/vision
```

`export_onnx.py` 参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型仓库 id，会下载到本地缓存 | `openbmb/MiniCPM-OCR-Jev` |
| `--modelscope` | 从 ModelScope 下载（默认走 HuggingFace） | 关闭 |
| `--output_dir` | 仅给目录时使用；会**自动追加 `{H}x{W}` 子目录** | `../../model/vision` |
| `--output_path` | 直接指定 ONNX 全路径，优先级高于 `--output_dir` | `None` |
| `--name` | 输出文件名前缀 | `vision` |
| `--height` / `--width` | 输入分辨率，必须是 32 的倍数且 ≥ 64 | `320` / `832` |
| `--zoo_root` | rknn3-model-zoo 根目录（需要其中的 `py_utils/`） | 自动探测 |
| `--export_modeling` | 导出用 modeling 文件 | `../llm/modeling_qwen3_5_export.py` |
| `--hf_endpoint` | 走 HuggingFace 下载时的镜像地址（`--modelscope` 时忽略） | `https://hf-mirror.com/` |

> ⚠️ **注意**：
>
> - `--output_dir` 会**自动追加 `320x832` 这样的分辨率子目录**（`{H}x{W}` 顺序），目的是避免不同 shape 的 external data 互相覆盖。上面的命令最终写到 `model/vision/320x832/vision.onnx`。
> - 分辨率是**编译期常量**，换 H/W 必须重新导出 ONNX 并重新转 RKNN，不能只改运行时参数。
> - 视觉塔强制使用 `_attn_implementation="eager"`：SDPA 路径在 ONNX 里会走到 `enable_gqa=True` 的符号化分支而断言失败。
> - 只导出 `pooler_output`，输入固定为 `pixel [1, 3, H, W]`（NCHW float32）。

### 2.2 导出 Vision RKNN 模型

```bash
cd examples/MiniCPM-OCR-Jev/python/vision

python export_rknn.py \
    --onnx_path ../../model/vision/320x832/vision.onnx \
    --target_platform rk1820 \
    --core_num 8 \
    --quantized_dtype w4a16
```

`export_rknn.py` 参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--onnx_path` | 输入视觉 ONNX（必填） | — |
| `--output_path` | 输出 `.rknn`；默认与 ONNX 同名同目录 | `None` |
| `--target_platform` | `rk1820` / `rk1828` | `rk1820` |
| `--core_num` | NPU 核数，1–8；**必须与运行时 core_mask 的 popcount 一致** | `8` |
| `--quantized_dtype` | `w4a16` / `w6a16` / `w8a16` | `w4a16` |
| `--fp_input` | 声明 `pixel` 为 float32 NCHW（归一化放到 host 侧） | 关闭 |
| `--calib` | 校准集 `dataset.txt`；**w4a16 下无效**（见下） | `None` |
| `--report_source` | RKNN3 生成的 `tmp/model_report.html` 位置 | `./tmp/model_report.html` |

> ⚠️ **注意**：
>
> - 默认把预处理**搬进 NPU**：`input_attrs={"pixel":{"dtype":"uint8","layout":"NHWC"}}` 加 `mean_values = std_values = 127.5`，正好等价于 `(x/255 - 0.5) / 0.5`，与检查点里 `image_mean=[0.5,0.5,0.5]`、`image_std=[0.5,0.5,0.5]`、`rescale_factor=1/255` 一致。运行时直接喂 uint8 NHWC 图即可。
> - `--calib` 在 `w4a16/w6a16/w8a16` 下**只量化权重、激活保持 16-bit**，toolkit 会明确忽略校准集（产物与随机噪声校准逐字节相同）。保留该选项是为了将来切到 `w8a8/i8`。
> - `rknn.build()` 会写 `./tmp/`。**多分辨率转换必须串行执行**，否则 `tmp/` 互相踩踏。
> - 输入 `pixel` 的 `frames` 只能是 1 或 2（本示例恒为 1）。

### 2.3 导出 LLM ONNX 模型

```bash
cd examples/MiniCPM-OCR-Jev/python/llm

# 默认从 HuggingFace 下载 openbmb/MiniCPM-OCR-Jev；加 --modelscope 改从 ModelScope
python export_onnx.py \
    --output_dir ../../model/llm \
    --name value_llm \
    --head_key score.weight \
    --num_labels 2 \
    --task_type 1
```

`export_onnx.py` 参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model_path` | 模型仓库 id，会下载到本地缓存 | `openbmb/MiniCPM-OCR-Jev` |
| `--modelscope` | 从 ModelScope 下载（默认走 HuggingFace） | 关闭 |
| `--output_dir` | 产物输出目录 | `../../model/llm` |
| `--name` | 产物文件名前缀 | `value_llm` |
| `--head_key` | safetensors 中分类头权重 key | `score.weight` |
| `--num_labels` | 分类头输出维度（二分类 = 2） | `2` |
| `--seq_len` | ONNX 追踪用的 dummy 序列长度（动态轴由 `dynamic_axes` 决定） | `64` |
| `--task_type` | 写进 `config.pkl` 的 task_type；`1` = 向量/分类任务，`-1` = 不写 | `1` |
| `--skip_embed_bin` | 不导出 `.embed.bin`（仅调试用） | 关闭 |
| `--zoo_root` | rknn3-model-zoo 根目录（需要其中的 `py_utils/`） | 自动探测 |
| `--export_modeling` | 导出用 modeling 文件 | `python/llm/modeling_qwen3_5_export.py` |
| `--hf_endpoint` | 走 HuggingFace 下载时的镜像地址（`--modelscope` 时忽略） | `https://hf-mirror.com/` |

> ⚠️ **注意**：
>
> - 该脚本导出的是 **数值化的 ONNX**：`conv_state` 输入必须传**全 0**，代表「没有历史」。用 warm 过的 `conv_state` 会让输出漂移 2.2e+00（全 0 时 max|Δ| = 3.9e-07）。
> - ONNX opset 为 **19**，并带 `custom_opsets={"rknn": 1}`；只用标准算子，RKNN3 可直接吃。
> - 权重超过 2GB，`torch.onnx.export` 会自动切成 external data：`value_llm.onnx` 本身只有几百 KB，真正的权重在同目录的 `onnx__*` / `*.weight` 旁挂文件里。**拷贝时必须整目录拷贝**，只复制 `.onnx` 会加载失败。
> - 校验 `onnx.checker.check_model` 时传的是**路径**（不是 model 对象），避免把几 GB 权重读进内存。

ONNX I/O 契约：

| 名称 | 方向 | shape | 说明 |
|------|------|-------|------|
| `input_ids` | in | `[1, sequence]` | 动态轴 |
| `position_ids` | in | `[1, sequence]` | 动态轴 |
| `attention_mask` | in | `[1, sequence]` | 动态轴 |
| `Linear_Tc` | in | `[1, ...]` | 线性注意力卷积状态 |
| `num_logits_to_keep` | in | `[1]` | 只算最后 N 个 token 的 logits |
| `conv_state` | in | `[...]` | **必须全 0** |
| `output` | out | `[1, num_logits_to_keep, 2]` | 分类头 logits |
| `conv_state_out` | out | `[...]` | 转换 RKNN 时会被剥离 |

同时生成三个附属产物：

```bash
value_llm.onnx
value_llm.config.pkl        # rknn3 LLM 配置（含 mrope / vocab / chat template）
value_llm.tokenizer.gguf    # 供 C++ Tokenizer 使用
value_llm.embed.bin         # token embedding 表，[vocab_size, 2048] fp16
```

### 2.4 导出 LLM RKNN 模型

```bash
cd examples/MiniCPM-OCR-Jev/python/llm

python export_rknn.py \
    --onnx_path ../../model/llm/value_llm.onnx \
    --target_platform rk1820 \
    --core_num 8 \
    --quantized_dtype w4a16 \
    --context_length 4096
```

`export_rknn.py` 参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--onnx_path` | 输入 LLM ONNX（与 `.config.pkl` 同名同目录） | 必填 |
| `--output_path` | 输出 `.rknn`；默认与 ONNX 同名同目录 | `None` |
| `--target_platform` | `rk1820` / `rk1828` | `rk1820` |
| `--core_num` | NPU 核数，1–8 | `8` |
| `--quantized_dtype` | `w4a16` / `w6a16` / `w8a16` | `w4a16` |
| `--context_length` | 最大上下文长度，≥ 128 | `4096` |
| `--keep_all_logits` | 不裁剪 logits 输出 | 关闭 |
| `--report_source` | 量化报告位置 | `./tmp/model_report.html` |

> ⚠️ **注意**：
>
> - 转换前脚本会把前 3 个输入（`input_ids` / `position_ids` / `attention_mask`）实例化成 `[1, 128]`、剥离 `conv_state_out` 输出，并给 `conv_state` 填全 0 初值。
> - **默认（推荐）保留 `keep_one_logit` 配置**：让 RKNN3 复用 `num_logits_to_keep` 只算最后 1 个 token 的分类结果。MiniCPM-OCR Jev 本来就是「取最后一个 token」，清零字节开销。`--keep_all_logits` 只在需要逐 token logits 时才用。
> - `rknn.build()` 共享 `tmp/`，**不同分辨率/不同模型的转换请串行执行**。

## 3. RKNN LLM 量化配置

MiniCPM-OCR Jev 的 LLM 转换默认使用 **W4A16 + group32 + normal**，目标平台 `rk1820`。与纯文本 Qwen3.5 不同，这里仍需要 **MRoPE** 参数（因为模型是多模态训练的）：

```python
llm_config = copy.deepcopy(DEFAULT_RKNN_LLM_CONFIG)
llm_config['attention_config'][0]['kvcache_buffer_len'] = 4096
llm_config['attention_config'][0]['max_position_embeddings'] = 4096
llm_config['attention_config'][0]['mrope_type'] = 'Qwen3.5'
llm_config['attention_config'][0]['mrope_section'] = [11, 11, 10]
llm_config['attention_config'][0]['mrope_new_id_name'] = 'mrope_id_input'
llm_config['attention_config'][0]['attention_type'] = 'FullAttention'
llm_config['attention_config'][0]['keep_one_logit'] = []   # 只有 --keep_all_logits 时清空

rknn.config(
    target_platform='rk1820',
    dynamic_input=dynamic_shapes,
    quantized_dtype='w4a16',
    quantized_algorithm='normal',
    quantized_method='group32',
    llm_config=llm_config,
    input_initial_value=input_initial_value,
    cvt_conv_streaming=conv_streaming,
)
```

> ⚠️ **注意**：
>
> - `mrope_section` 必须与 HuggingFace `config.json` 中的配置一致（本模型是 `[11, 11, 10]`）。
> - `rknn.config(...)` 的 `llm_config` 必须是 **deepcopy**：`DEFAULT_RKNN_LLM_CONFIG` 含嵌套 `list`/`dict`，浅拷贝会在同进程多次转换时互相污染。
> - `cvt_conv_streaming` 取自图中所有 `Conv` 节点的输出名（线性注意力需要流式改写）。
> - `input_initial_value` 的最后一个元素（`conv_state`）必须是全 0 的 `float32`。

## 4. 打分口径说明

### 4.1 Prompt 组装


```text
<|im_start|>user\n
<image>                 ← 原图（先）
OCR文本：\n
{ocr_text}              ← 原样插入，不做转义/清洗，只去掉结尾的 \r\n
\n
<image>                 ← OCR 文本渲染图（后）
第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。\n
请判断该 OCR 文本及其对应渲染图是否与原图内容一致。\n
这是一个二分类任务：一致标为 1，不一致标为 0。
<|im_end|>\n
```

**两个容易踩的坑**

- **图像顺序**：原图在前、渲染图在后。反过来会显著改变 `prob_1`。
- **没有 assistant 轮次**。模型取的是 user 轮最后一个 token 的 pooled 表示，加 `<|im_start|>assistant` 会取错位置。

C++ 里用的是运行时的 `<image>` 标记，`rknn3_session_run` 会把它展开成 `<|vision_start|>` + N×`<|image_pad|>` + `<|vision_end|>`，并按出现顺序消费 `image_embed` 里的图像块。N 由视觉塔输出决定（320x832 时 N=260）。

### 4.2 打分与判定

```cpp
// 稳定 softmax，避免 exp 溢出
const float m  = std::max(l0, l1);
const float e0 = std::exp(l0 - m);
const float e1 = std::exp(l1 - m);
prob_0 = e0 / (e0 + e1);
prob_1 = e1 / (e0 + e1);
label  = (prob_1 >= 0.5606135129928589f) ? 1 : 0;
```

阈值 `0.5606135129928589` 来自训练侧在验证集上扫出的最优点，已作为 `VALUE_HEAD_DEFAULT_THRESHOLD` 写进 `cpp/minicpm_ocr_jev.h`，可用 `--threshold` 覆盖。

> ⚠️ 该阈值对应的标签语义是 **`1 = 一致`**。`manifest.json` 里的 `label` 字段用的也是同一语义，可以直接对齐。

### 4.3 推理约束

- **prefill only**：`infer_param.prefill_only = true`、`disable_sampling = true`、`max_new_tokens = 1`，不做任何解码。
- **单次会话**：一次 `rknn3_session_run` 只处理一个三元组，连续打分需要复用同一个 session（本 demo 每次进程启动打分一次）。
- **输出张量**：MiniCPM-OCR Jev 的分数来自 `output` 张量（`n_elems = 2`）。C++ 在 `init_value_head_llm` 里通过 `RKNN3_QUERY_OUTPUT_ATTR` + `rknn3_create_mem` 预分配，并在 `rknn3_session_set_callback` 之前挂到 `callback.output_tensors` 上；回调只在 `RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED` 状态取数。
- **`result_callback` 必须有**：即使分数不走 `result_callback`，`rknn3_session_run()` 在它和 `tokenizer_callback` 都为 `NULL` 时会直接失败（logcat 里报 `no tokenizer_callback or result_callback provided!`）。本 demo 因此注册了一个空实现，见 §5.5。

### 4.4 图像预处理

MiniCPM-OCR Jev 在 `transformers` 的 `Qwen2VLImageProcessorFast` 输出上训练，其预处理是**保留长宽比**的 `smart_resize`，随后切 patch。OCR 块裁图常常是极扁的细条（本示例数据里 `s01` 是 455×46 ≈ 9.9:1，`s10` 是 1346×230 ≈ 5.9:1），**直接拉伸到方形画布会把这些字形压扁**（如 256×256，实测让 `prob_1` 偏移最多 ~0.4）。默认的 320x832（2.6:1 宽高比）与参考 demo 输入图（411×156 ≈ 2.6:1）的 `smart_resize` 原生比例一致，比方形画布更贴近训练分布。因此 C++ demo 按下面的顺序预处理：

1. **极端长宽比补白**：若 `max(w,h)/min(w,h) > 199`（`transformers` 的硬上限），先贴到白底画布上居中，把比例拉回合法范围。常规块裁图这一步是 no-op。
2. **`smart_resize`**：两侧各自四舍五入到 `factor = 32`（`patch_size 16 × merge 2`）的整数倍，再把总像素数钳到 `[65536, 16777216]`。
3. **letterbox**：把 `smart_resize` 的结果等比缩放进 320×832 画布并居中，四周用灰底 `127` 填充。`transformers` 本身没有填充步骤，这一步是**移植侧的取舍**——静态单分辨率的视觉塔必须有固定输入尺寸。


> 💡 归一化仍在 RKNN 内完成。默认导出把 `pixel` 声明为 **`uint8` NHWC**，`mean_values = std_values = [127.5, 127.5, 127.5]`，即 `(x - 127.5) / 127.5` 由 RKNN 内执行；用户只负责把 RGB888 画布 `memcpy` 进去。若用 `--fp_input` 导出成 `float32 NCHW`，用户才需要自己做归一化和转置。

## 5. C++ 部署说明

### 5.1 编译

```bash
cd rknn3-model-zoo

# 交叉编译器（按实际安装路径修改）
export GCC_COMPILER=/path/to/aarch64-linux-gnu

./build-linux.sh -t rk3588 -a aarch64 -d MiniCPM-OCR-Jev -b Release
```

产物目录：

```bash
install/rk3588_linux_aarch64/rknn_MiniCPM-OCR-Jev_demo/
├── rknn_minicpm_ocr_jev_demo               # aarch64 ELF
├── lib/{librknn3_api.so, librknn3_api_rkcp.so, librga.so}
├── data/{manifest.json, images/, texts/}
└── model/
    ├── llm/{value_llm.rknn, value_llm.weight, value_llm.tokenizer.gguf, value_llm.embed.bin}
    └── vision/320x832/{vision.rknn, vision.weight}
```

> ⚠️ 可执行文件已设置 **RPATH = `$ORIGIN/lib`**，正常情况下不需要 `LD_LIBRARY_PATH`；若把 `lib/` 移走，则需要显式设置。

### 5.2 所需文件

| 文件 | 说明 |
|------|------|
| `model/vision/<H>x<W>/vision.rknn` | 视觉塔 RKNN 模型 |
| `model/vision/<H>x<W>/vision.weight` | 视觉塔权重 |
| `model/llm/value_llm.rknn` | LLM + 分类头 RKNN 模型 |
| `model/llm/value_llm.weight` | LLM 权重 |
| `model/llm/value_llm.tokenizer.gguf` | 分词器（llama backend） |
| `model/llm/value_llm.embed.bin` | token embedding 表（fp16） |
| 原图 + 渲染图 | PNG/JPG，RGB |
| OCR 文本 | 字符串或文本文件 |

### 5.3 命令行参数

```text
./rknn_minicpm_ocr_jev_demo \
    <vision_model> <vision_weight> \
    <llm_model> <llm_weight> \
    <tokenizer> <embedding> \
    <vision_core_mask> <llm_core_mask> \
    <orig_image> <render_image> \
    (--ocr-text <text> | --ocr-text-file <path>) \
    [--threshold <float>] [--model-width <int>] [--model-height <int>] \
    [--print-prompt] [--json]
```

| 参数 | 说明 |
|------|------|
| `vision_model` | 视觉 `.rknn` 路径 |
| `vision_weight` | 视觉 `.weight` 路径 |
| `llm_model` | LLM `.rknn` 路径 |
| `llm_weight` | LLM `.weight` 路径 |
| `tokenizer` | `.tokenizer.gguf` 路径 |
| `embedding` | `.embed.bin` 路径 |
| `vision_core_mask` | 视觉 NPU 核心掩码，按 **16 进制**填写，如 `0x1`、`0xff` |
| `llm_core_mask` | LLM NPU 核心掩码，按 16 进制填写 |
| `orig_image` | 原图路径 |
| `render_image` | OCR 文本渲染图路径 |
| `--ocr-text` | OCR 文本（与 `--ocr-text-file` 二选一） |
| `--ocr-text-file` | 从文件读 OCR 文本，只去掉结尾 `\r\n`，保留中间换行 |
| `--threshold` | 判定阈值，默认 `0.5606135129928589` |
| `--model-width` / `--model-height` | **Prune 导出**专用的画布尺寸；完整导出会从张量 shape 自动读取 |
| `--print-prompt` | 打印实际送入模型的完整 prompt |
| `--json` | 单行 JSON 输出，便于脚本批处理 |

### 5.4 运行示例

```bash
cd install/rk3588_linux_aarch64/rknn_MiniCPM-OCR-Jev_demo

./rknn_minicpm_ocr_jev_demo \
    model/vision/320x832/vision.rknn \
    model/vision/320x832/vision.weight \
    model/llm/value_llm.rknn \
    model/llm/value_llm.weight \
    model/llm/value_llm.tokenizer.gguf \
    model/llm/value_llm.embed.bin \
    0xff 0xff \
    data/images/s01_orig.png \
    data/images/s01_render.png \
    --ocr-text-file data/texts/s01.txt
```

输出格式如下（数值为 `s01` 在 320x832 下的**实测**结果，RK1820）：

```text
---------------- value head ----------------
 logit_0        : 2.06054688
 logit_1        : -2.66015625
 prob_0         : 0.99116981
 prob_1         : 0.00883024
 threshold      : 0.5606135129928589
 label          : 0 (不一致)
 prompt tokens  : 612
 vision tokens  : 260 x 2
 vision latency : 364.837 ms (x2)
 llm latency    : 415.883 ms
--------------------------------------------
```
