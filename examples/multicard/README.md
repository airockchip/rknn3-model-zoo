# 多卡推理 Demo 说明

本 demo 将 LLM 模型按层数切分为多段（segment），每段部署在一张 RK182X 加速卡上，通过流水线并行（pipeline parallelism）实现多卡协同推理，旨在支持更大参数量的模型并能通过流水提高prefill性能。

当前支持以下模型：

| 模型 | RK182X 加速卡数量 |
|------|-------------------|
| `Qwen/Qwen3.5-9B` | 2 |
| `google/gemma-4-12B-it` | 2 |
| `Qwen/Qwen3.5-27B` | 4 |
| `Qwen/Qwen3.8-27B` | 4 |
| `google/gemma-4-31B-it` | 4 |

## 1. 模型切分原理

模型按照 Transformer layer 的顺序进行连续切分，切分边界位于 layer 之间，不改变模型层内的计算顺序。最后一段除了分配到末尾的 Transformer layer 外，还固定包含最终 `norm` 和 `lm_head`，负责输出 logits。

```text
输入 hidden states
        |
        v
+-------------------------------+
| stage0: layer 0 ... layer k0  |
+-------------------------------+
        |
        v
+-------------------------------+
| stage1: layer k0+1 ... layer k1|
+-------------------------------+
        |
       ...
        |
        v
+-----------------------------------------+
| stageN-1: layer kN-2 ... layer L-1      |
|           + norm + lm_head -> logits   |
+-----------------------------------------+
```

导出 ONNX 时，脚本不会简单地按 `总层数 / 段数` 平均分配，而是根据权重大小自动寻找更均衡的切分边界：

1. 计算每个 Transformer layer 的估算权重大小。
2. 将最后一段固定需要的 `norm + lm_head` 权重加入最后一段的权重预算。
3. 在保持 layer 顺序和每段至少包含一层的前提下，选择连续层区间，使各段的估算总权重尽量接近。
4. 按得到的 layer 区间分别导出 ONNX。非最后一段输出中间 hidden states，最后一段执行 `norm + lm_head` 并输出 logits。

权重估算使用导出配置对应的量化规格：Transformer layer 按 W4A16/group32 估算，`lm_head` 按 W6A16/group32 估算，最终 `norm` 按 FP16 估算。因此最后一段通常会少分配一些 Transformer layer，用来抵消 `lm_head` 带来的额外权重。

例如分成 2 段时，逻辑结构如下：

```text
stage0: layer 0 ... layer k
stage1: layer k+1 ... layer L-1 + norm + lm_head
```

`--num_segments N` 用于指定段数；未指定或设置为 `0` 时，导出脚本按默认策略自动推导段数。段数确定后，实际的 layer 数量分配仍由上述权重均衡算法决定。

## 2. 多卡推理流程说明

本节对应 `examples/multicard/cpp/main.cc` 的实际执行流程，便于排查行为与性能。

### 2.1 整体框架流程图

```
启动与参数解析
  │  解析 --stage-count / --bucket-size / --prompt 等命名参数
  ▼
路径推导与校验
  │  seg0 → seg1..segN
  ▼
初始化 Tokenizer + Embedding
  │  mmap embed.bin
  ▼
自动发现设备
  │  rknn3_find_devices
  ▼
逐 stage 初始化
  │  init_stage: init → load → model_init → session → set_callback
  ▼
是否使用外置 rope（Qwen3.5/Gemma-4 默认需要）？
  │
  ├── 是 ──> load_safetensors (mmap 外部 rope cache)
  │              │
  │              ▼
  │          注册 input_callback
  │
  └── 否（仅适用于不需要外置 rope 的模型）
         │
         ▼
    进入推理
         │
         ▼
Prefill: run_pipeline_once(prompt)
         │
         ▼
stage0 产出中间张量 → stage_output_callback 入队
         │
         ▼
stage1..stageN worker 线程
  │  wait_stage_batch → session_run
         │
         ▼
最后 stage 且最后一桶?
  │
  ├── 否 ──> disable_sampling = true  (仅传播)
  │              │
  └── 是 ──> disable_sampling = false (执行采样并输出 token)
         │
         ▼
本轮 prefill 完成
         │
         ▼
读取 next_token
         │
         ▼
  ┌── decode 循环是否继续?
  │         │
  │    ┌── 是 ──> 单 token 输入: run_pipeline_once(token)
  │    │              │
  │    │              ▼
  │    │         result_callback 更新 next_token
  │    └──────────────┘
  │
  └── 否 ──> 输出性能统计 + 清理 KV cache + 释放资源
```

### 2.2 初始化阶段

程序启动后，主要完成以下初始化：

1. 参数解析与路径推导。
  - 只输入 `seg0` 的模型/权重路径，自动推导 `seg1..segN`。
  - 校验所有分段文件是否存在。
2. Tokenizer 与 Embedding 初始化。
  - `init_tokenizer_and_embedding` 打开 tokenizer 与 `embed.bin`。
  - `embed.bin` 通过 mmap 映射，后续 `embed_callback` 按 token id 直接查表拷贝 embedding。
3. 设备与 Stage 初始化。
  - `rknn3_find_devices` 自动发现设备，要求设备数 >= `stage_count`。
  - 每个 stage 调用 `init_stage`，完成 `rknn3_init -> load_model -> model_init -> session_init -> set_callback`。
4. 外部 rope cache 初始化（Qwen3.5/Gemma-4 必需，其他模型可选）。
  - 如果传入 `--rope-tensor`，调用 `load_safetensors` mmap 外部 rope cache，并在各 stage 注册 `input_callback`。

### 2.3 Prefill Pipeline 推理

`run_pipeline_once(..., prompt, nullptr, ...)` 用于 prefill：

1. 统一单步配置：
  - `infer_param.prefill_only = true`
  - 初始 `infer_param.disable_sampling = true`
2. Stage0 输入 prompt（或 token），由 `embed_callback` 统计本轮输入 token 总量（`expected_tokens`）。
3. 非 stage0 由 worker 线程执行 `run_stage_worker`，从前一段队列取批次并调用 `rknn3_session_run`。
4. 非最后段通过 `stage_output_callback` 把输出 tensor 拷贝到 `StageBatch`，推入下游队列。

说明：prefill 是“分桶流水”。每段输出会按 bucket 向下一段输送，直到最后一段处理完最后一桶。

### 2.4 Decode 循环推理

主流程先做一次 prefill 拿到首个 `next_token`，然后进入 decode 循环：

1. 每步把上一步 token 作为单 token 输入（`RKNN3_LLM_INPUT_TOKEN`）。
2. 再次调用 `run_pipeline_once`，复用同一套 pipeline 机制。
3. `result_callback` 返回并打印当前步输出，同时更新 `next_token`。
4. 满足下列任一条件结束：
  - 达到 `max_new_tokens`
  - 推理失败
  - 命中 EOS（且未传入 `--ignore-eos`）

### 2.5 多卡之间的数据流

多卡通信在 host 侧通过 `StageSlot` 队列完成，核心数据流如下：

1. Stage0 运行后触发 callback，产出 hidden states（等中间张量）。
2. `stage_output_callback` 将输出复制为 `TensorBlob`，封装为 `StageBatch` 放入当前 stage 对应队列。
3. 下游 stage 线程 `wait_stage_batch` 阻塞等待，取到 batch 后作为 `RKNN3_LLM_INPUT_EMBED` 输入本段。
4. 重复该流程直到最后 stage；最后 stage 不再向下游转发。

### 2.6 采样逻辑：仅最后 stage 的最后一桶采样

采样控制在 `run_stage_worker` 中对最后 stage 单独处理：

1. 先由 stage0 的 `embed_callback` 统计本轮总 token 数 `total_tokens`。
2. 最后 stage 处理每个 batch 时维护 `consumed_tokens`。
3. 若当前 batch 不是最后一桶：
  - `disable_sampling = true`（不采样，只做中间传播）
4. 仅当 `consumed_tokens + batch.n_tokens == total_tokens` 时：
  - `disable_sampling = false`（开启采样，产出本轮 token）

这保证了 prefill 期间只有最后一段、最后一桶触发真实采样，避免中间桶重复采样。

### 2.7 单步推理控制（prefill_only 与 disable_sampling）

本 demo 的一次 `run_pipeline_once` 固定为“单步执行”：

1. `prefill_only = true`：
  - 表示本次只执行当前输入对应的一次 prefill 计算路径，不在单次 run 中连续生成多个 token。
2. `disable_sampling`：
  - 默认 true。
  - 在最后 stage 根据“是否最后一桶”动态切换。

组合效果：

- prefill 阶段：同一轮 prompt 会被分桶流过多段，但只在最后一桶采样 1 次。
- decode 阶段：每次输入 1 个 token，再单步采样 1 个 token，通过外层 for 循环推进生成。

### 2.8 Qwen3.5/Gemma-4 外部 rope cache（默认用法）

Qwen3.5 和 Gemma-4 默认通过 `--rope-tensor <safetensors>` 使用外部 rope cache。程序会自动识别两种模型的 tensor 命名格式：

- Qwen3.5：`rope_cos_cache`、`rope_sin_cache`
- Gemma-4：`rope_cos_cache_0`、`rope_sin_cache_0`、`rope_cos_cache_1`、`rope_sin_cache_1`

识别格式后，统一通过以下流程提供外置 rope：

1. `load_safetensors`
  - 根据模型格式，从 `.safetensors` 读取 Qwen3.5 的 `rope_cos_cache` / `rope_sin_cache`，或 Gemma-4 的编号 rope cache 元数据与偏移。
  - 通过 mmap 映射文件，避免把整份 rope cache 再复制到大块 host 内存。
2. `query_ext_input_indices`
  - 找到模型中 rope cache 对应的 ext input 索引。
3. `input_callback`
  - 每次推理按当前位置 `param.pos`，把当前 step 需要的 rope 片段拷贝到对应输入 tensor。

这样做的目的：

- 将大体积 rope cache 常驻在host端中，减少device端内存占用。
- 按 step 提供所需片段，而不是一次性构造完整 rope 输入。

### 2.9 分段内存占用分析

不同分段模型的内存占用可通过 `tmp` 目录下的 `model_report.html` 查看。

1. 在 RKNN 转换完成后，`tmp_seg{idx}` 目录下会生成 `model_report.html`：
   - 用浏览器打开该文件，可查看各 NPU core 的 `total_size`、`weights`、`kvcache` 等明细。
2. `kvcache_buffer_len` 配置越大，`kvcache` 和 `internal` 占用越多，整体内存也越大。


## 3. 模型转换

分段模型的导出脚本统一放在 `examples/multicard/python/` 目录下，按模型分为 `qwen3_5/` 和 `gemma4/` 两个子目录。

### 3.1 Qwen3.5 分段转换

以 Qwen3.5-9B 拆分为 2 段为例。

**环境要求**：

```bash
pip install -r requirements.txt
```

> 具体版本依赖见 `examples/multicard/python/qwen3_5/requirements.txt`。

**第一步：导出分段 ONNX 模型**

```bash
cd examples/multicard/python/qwen3_5

python export_llm_segment.py \
    --model_path /path/to/Qwen3.5-9B \
    --multi_segment --num_segments 2
```

执行后，会在输出目录下生成分段子目录和对应文件：

```bash
seg0/
  Qwen3.5-9B-llm_seg0.onnx
seg1/
  Qwen3.5-9B-llm_seg1.onnx
Qwen3.5-9B-llm.config.pkl
Qwen3.5-9B-llm.tokenizer.gguf
Qwen3.5-9B-llm.embed.bin
```

**第二步：导出分段 RKNN 模型**

```bash
python export_rknn_segment.py --multi_segment --num_segments 2
```

导出完成后，生成的文件如下：

```bash
Qwen3.5-9B-llm_seg0.rknn
Qwen3.5-9B-llm_seg0.weight
Qwen3.5-9B-llm_seg1.rknn
Qwen3.5-9B-llm_seg1.weight
Qwen3.5-9B-llm_seg0.safetensors
Qwen3.5-9B-llm_seg1.safetensors
```

Qwen3.5 默认使用外置 rope cache。不同分段均会生成一个内容相同的 `.safetensors` 文件，运行时传入任意一个即可。

### 3.2 Gemma-4 分段转换

以 Gemma-4-12B-it 拆分为 2 段为例。

> ⚠️ **特殊版本要求**：依赖特定的 transformers/torch 版本，请使用以下命令安装：
>
> ```bash
> pip install -r requirements.txt
> ```
>
> 具体版本依赖见 `examples/multicard/python/gemma4/requirements.txt`。

**第一步：导出分段 ONNX 模型**

```bash
cd examples/multicard/python/gemma4

python export_llm_segment.py \
    --model_path google/gemma-4-12B-it \
    --multi_segment --num_segments 2
```

执行后，会在指定输出目录下生成分段子目录和对应文件：

```bash
seg_0/
  gemma-4-12b-it_seg0.onnx
seg_1/
  gemma-4-12b-it_seg1.onnx
gemma-4-12b-it.config.pkl
gemma-4-12b-it.tokenizer.gguf
gemma-4-12b-it.embed.bin
```

**第二步：导出分段 RKNN 模型**

```bash
python export_rknn_segment.py --multi_segment --num_segments 2
```

导出完成后，生成的文件如下：

```bash
gemma-4-12b-it_seg0.rknn
gemma-4-12b-it_seg0.weight
gemma-4-12b-it_seg1.rknn
gemma-4-12b-it_seg1.weight
gemma-4-12b-it_seg0.safetensors
gemma-4-12b-it_seg1.safetensors
```

Gemma-4 同样默认使用外置 rope cache。不同分段均会生成一个内容相同的 `.safetensors` 文件，运行时传入任意一个即可。

### 3.3 分段导出参数配置

#### `--max_context_len`

 `export_rknn_segment.py` 支持通过 `--max_context_len` 从命令行覆盖脚本内的 `MAX_CONTEXT_LEN` 默认值。不传该参数时使用默认值 `4096`；该值会同时用于 `kvcache_buffer_len` 和 `max_position_embeddings`。

例如将最大上下文长度设置为 8192：

```bash
cd examples/multicard/python/qwen3_5
python export_rknn_segment.py --multi_segment --num_segments 2 --max_context_len 8192
```

该参数应设置为正整数，并结合目标设备的内存容量选择合适的上下文长度。

#### `--segment_layer_counts`

`--segment_layer_counts c1 c2 ...` 仅用于 LLM/ONNX 分段导出脚本，用于手动指定每个分段包含的连续 Transformer block 数量。列表长度就是分段数，所有数值之和必须等于模型的 block 总数；配置后会覆盖 `--num_segments` 的自动切分策略。参数使用空格分隔整数，不要写成 Python 列表或逗号分隔字符串。Qwen3、Qwen3.5、Gemma4、MiniCPM5 四种模型均支持。

例如，假设模型共有 32 个 Transformer block，手动切成 3 段：

```bash
cd examples/multicard/python/qwen3_5
python export_llm_segment.py \
    --multi_segment \
    --segment_layer_counts 8 8 16
```

上述配置对应：

```text
seg0: block 0    - 7
seg1: block 8    - 15
seg2: block 16   - 31 + norm + lm_head
```

不提供 `--segment_layer_counts` 时，`export_llm_segment.py --multi_segment --num_segments N` 会按目标卡内存估算，自动规划 N 个连续分段（默认约每 8 层一段）。`--card` 选择估算卡型，默认 `rk1828`；默认可用容量按标称容量的 98% 计算：rk1828 为 4900 MiB、rk1820 为 2450 MiB。实际可用内存不同时，可用 `--card_usable_mib` 指定每卡容量。

```bash
cd examples/multicard/python/qwen3_5
python export_llm_segment.py \
    --multi_segment --num_segments 4 \
    --card rk1828 --card_usable_mib 4800
```

导出成功后，脚本会在 `--export_llm_path` 同目录生成同名前缀的 `.segment_plan.json`，包含实际使用的 `segment_layer_counts`、建议的 `MAX_CONTEXT_LEN`（JSON 字段 `suggested_MAX_CONTEXT_LEN`）以及各卡的 block 权重、末段 norm/lm_head 权重、KV cache、内部 scratch、固定开销、总内存和剩余内存估算。手动指定分段时，JSON 中的上限是该固定分法可承受的上限；`--stop_after_segment` 只导出前缀时，JSON 仍规划完整模型，并用 `exported_segment_indices` 标出已导出的段。

估算沿用 `qwen3_5/estimate_plan_v2.py` 的 KV 对齐、每卡 6144 B/token scratch 和 32 MiB 固定开销；Qwen3.5 经过 RKNN 报告标定，其余模型是外推。Gemma4 的 1024-token FP16 滑动窗口及 Qwen3.5 的线性注意力状态分别计入 KV。规划值仅供参考：将 JSON 中的 `suggested_MAX_CONTEXT_LEN` 作为对应 `export_rknn_segment.py --max_context_len` 的起点，最终以每卡实际 RKNN `model_report` 和板端运行结果校验。若 JSON 的 `feasible` 为 `false` / 上限为 0，当前卡数和容量的估算下不能部署该分段组合。

#### `--profile_mode` 和 `--save_ctx`

这两个参数仅用于 `export_rknn_segment.py`，可以独立使用，默认均关闭：

| 参数 | 作用 |
|------|------|
| `--profile_mode` | 开启 RKNN profile 模式，为逐层精度/性能分析保留所需的 profile 信息 |
| `--save_ctx` | 导出 RKNN context 文件，供运行时 `load_rknn(load_ctx=True)` 使用 |

以 Qwen3.5 两段模型为例：

```bash
cd examples/multicard/python/qwen3_5
python export_rknn_segment.py \
    --multi_segment \
    --num_segments 2 \
    --profile_mode \
    --save_ctx
```

### 3.4 使用 `--rebuild` 修改 KV Cache 配置

如果已经完成过 RKNN 导出，不需要重新执行完整的 ONNX 构建流程即可调整 KV Cache 长度。以两段模型为例，首次完整导出后，当前导出目录下会保留每一段对应的构建临时目录：

```text
tmp_seg0/
tmp_seg1/
```

通过 `--max_context_len` 设置新的上下文长度，再用 `--rebuild` 复用上述目录中的构建结果并重新导出 RKNN；不需要修改脚本常量：

Qwen3.5：

```bash
cd examples/multicard/python/qwen3_5
python export_rknn_segment.py \
    --multi_segment --num_segments 2 --max_context_len 8192 --rebuild
```

Gemma-4：

```bash
cd examples/multicard/python/gemma4
python export_rknn_segment.py \
    --multi_segment --num_segments 2 --max_context_len 8192 --rebuild
```

`--rebuild` 会依次执行以下操作：

1. 将 `tmp_seg0`、`tmp_seg1` 依次恢复为 `tmp`。
2. 使用 `--max_context_len` 设置的 KV Cache 长度调用 `rknn.rebuild("./tmp")`。
3. 导出新的 `*_seg0.rknn`、`*_seg1.rknn`，并将新的临时目录分别保存回 `tmp_seg0`、`tmp_seg1`。

注意事项：

- `--rebuild` 依赖已有的 `tmp_segN` 目录，必须先完成一次不带 `--rebuild` 的完整导出。
- `--rebuild` 适用于修改 KV Cache 等 RKNN 配置；如果修改了 ONNX、层切分方式或段数，需要重新执行完整导出。
- `kvcache_buffer_len` 通常需要和 `max_position_embeddings` 保持一致，具体数值按目标上下文长度和设备内存设置。

## 4. 编译

```bash
cd rknn3_model_zoo/

# 设置交叉编译工具链（可选）
export GCC_COMPILER=<GCC_COMPILER_PATH>

# 编译 multicard demo
./build-linux.sh -t rk3588 -a aarch64 -d multicard
```

编译完成后，文件生成在 `install/rk3588_linux_aarch64/rknn_multicard_demo/` 目录。

为使以下示例命令可运行，确保转换完成的文件齐全（以 2 段为例）：

```
Qwen3.5-9B-llm_seg0.rknn      gemma-4-12b-it_seg0.rknn
Qwen3.5-9B-llm_seg0.weight     gemma-4-12b-it_seg0.weight
Qwen3.5-9B-llm_seg1.rknn      gemma-4-12b-it_seg1.rknn
Qwen3.5-9B-llm_seg1.weight     gemma-4-12b-it_seg1.weight
Qwen3.5-9B-llm.tokenizer.gguf  gemma-4-12b-it.tokenizer.gguf
Qwen3.5-9B-llm.embed.bin       gemma-4-12b-it.embed.bin
Qwen3.5-9B-llm_seg0.safetensors gemma-4-12b-it_seg0.safetensors
```

## 5. 部署到开发板

```bash
# 推送整个 demo 目录到开发板
adb push install/rk3588_linux_aarch64/rknn_multicard_demo/ /data/

# 推送模型文件（以 Qwen3.5-9B 为例，包含外置 rope cache）
adb shell mkdir -p /data/models/multicard/
adb push Qwen3.5-9B-llm_seg0.rknn /data/models/multicard/
adb push Qwen3.5-9B-llm_seg0.weight /data/models/multicard/
adb push Qwen3.5-9B-llm_seg1.rknn /data/models/multicard/
adb push Qwen3.5-9B-llm_seg1.weight /data/models/multicard/
adb push Qwen3.5-9B-llm.tokenizer.gguf /data/models/multicard/
adb push Qwen3.5-9B-llm.embed.bin /data/models/multicard/
adb push Qwen3.5-9B-llm_seg0.safetensors /data/models/multicard/
```

## 6. 命令行参数

```bash
./rknn_multicard_demo \
    --model <stage0_model.rknn> \
    --weight <stage0_weight> \
    --vocab <tokenizer.gguf> \
    --embed <embedding.bin> \
    --ctx-size <tokens> \
    --core-mask <mask> \
    --stage-count <count> \
    --bucket-size <tokens> \
    [--prompt <text-or-file>] [--predict <count>] \
    [--verbose] [--ignore-eos] [--rope-tensor <safetensors>] \
    [--custom-op-path <plugin.so>] \
    [--device-id <id0#id1#...>] [--perf <input_tokens> <output_tokens>] \
    [--dump-tensors <dir>]
```

### 必选参数

| 参数 | 含义 | 示例 |
|------|------|------|
| `-m, --model <path>` | 第 0 段（seg0）的 RKNN 模型路径，程序会自动推导出 seg1、seg2... 的路径 | `/data/models/multicard/Qwen3.5-9B-llm_seg0.rknn` |
| `--weight <path>` | 第 0 段（seg0）的权重文件路径，程序会自动推导出 seg1、seg2... 的路径 | `/data/models/multicard/Qwen3.5-9B-llm_seg0.weight` |
| `--vocab <path>` | Tokenizer/Vocabulary 文件路径 | `/data/models/multicard/Qwen3.5-9B-llm.tokenizer.gguf` |
| `--embed <path>` | Token Embedding 权重文件路径 | `/data/models/multicard/Qwen3.5-9B-llm.embed.bin` |
| `-c, --ctx-size <tokens>` | 最大上下文长度（token 数） | `4096` |
| `--core-mask <mask>` | NPU 核心掩码，支持十六进制或十进制，如 `0x1`、`0x2`、`0xff` | `0xff` |
| `--stage-count <count>` | 段数（即使用的 RK182X 加速卡数） | `2` |
| `--bucket-size <tokens>` | 每次推理处理的 token 桶大小，通常设为 `128` | `128` |
| `--rope-tensor <safetensors>` | 外置 Rope Cache 文件路径（`.safetensors`）。Qwen3.5 和 Gemma-4 默认需要，其他模型按需使用 | Qwen3.5/Gemma-4 必填 |

### 可选参数

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--prompt <text-or-file>` | 用户输入。可直接输入文本，也可传入 `.txt` 文件路径 | 默认 Qwen 格式问候语 |
| `-n, --predict, --n-predict <count>` | 最大生成 token 数 | `512` |
| `--verbose` | 开启详细日志；不传则关闭 | 关闭 |
| `--ignore-eos` | 忽略 EOS 终止符；不传则正常响应 EOS | 不忽略 |
| `--device-id STRING` | 设备 ID，可不设置；多卡推理时使用 `#` 分隔多个设备 ID | 自动分配 |
| `--perf <input_tokens> <output_tokens>` | 启用性能测试模式，指定输入 token 总数和生成 token 总数 | 无 |
| `--dump-tensors <dir>` | 保存 `embed_callback` 和 `input_callback` 的 tensor 二进制及元数据 | 无 |
| `--custom-op-path <plugin.so>` | 自定义算子插件路径；程序会对每个 stage 的 RKNN context 分别调用 `rknn3_register_custom_ops_plugins` 注册 | 无；模型包含自定义算子时必填 |

性能测试模式直接使用词表中的有效 token 构造 `input_tokens`，以获得精确输入长度且不受 prompt 大小影响。`output_tokens` 包含 prefill 产生的首 token，程序随后固定执行 `output_tokens - 1` 次 decode；测试期间忽略 EOS，并关闭生成文本打印。输入和 decode 输入占用的总长度必须满足 `input_tokens + output_tokens - 1 <= max_context_len`。

例如，在自动分配设备时测试 512 token 输入、128 token 输出：

```bash
./rknn_multicard_demo \
    --model model_seg0.rknn \
    --weight model_seg0.weight \
    --vocab tokenizer.gguf \
    --embed embedding.bin \
    --ctx-size 4096 \
    --core-mask 0xff \
    --stage-count 2 \
    --bucket-size 128 \
    --rope-tensor model_seg0.safetensors \
    --perf 512 128
```

开启 tensor dump 的示例：

```bash
./rknn_multicard_demo \
    --model model_seg0.rknn --weight model_seg0.weight \
    --vocab tokenizer.gguf --embed embedding.bin \
    --ctx-size 4096 --core-mask 0xff \
    --stage-count 2 --bucket-size 128 \
    --rope-tensor model_seg0.safetensors \
    --dump-tensors ./tensor_dump
```

输出目录中每个 tensor 都有一个 `.bin` 原始数据文件和一个 `.txt` 元数据文件。

### 关于 device_id

程序默认通过 `rknn3_find_devices` 自动发现设备并分配。如需手动指定每张卡的 device_id，可通过 `rknn3_transfer_proxy devices` 命令查询可用设备列表：

```bash
$ rknn3_transfer_proxy devices
List of ntb devices attached
180B6D6D60646C601B535051564A4B4034d11892    USB_DEVICE
18086D6F6C616C6266535051564D4B4471b465d1    USB_DEVICE
180B6D6D60646C601B53505156487379cc92a983    USB_DEVICE
180B6D6D60646C601B535051564E404B94caeca     USB_DEVICE
```

多卡推理时将设备 ID 按 stage 顺序用 `#` 连接后传给 `--device-id`。例如 2 段推理指定前两个设备：

```bash
./rknn_multicard_demo \
    --model model_seg0.rknn \
    --weight model_seg0.weight \
    --vocab tokenizer.gguf --embed embedding.bin \
    --ctx-size 4096 --core-mask 0xff \
    --stage-count 2 --bucket-size 128 \
    --prompt "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n" \
    --predict 128 \
    --rope-tensor model_seg0.safetensors \
    --device-id 180B6D6D60646C601B535051564A4B4034d11892#18086D6F6C616C6266535051564D4B4471b465d1
```

### 关于 stage0 路径自动推导

程序只需要传入 `stage0` 的模型和权重路径，会自动根据 `_segN` 后缀推导出后续段的路径：

```
输入:  model_seg0.rknn   →  自动推导 model_seg1.rknn  model_seg2.rknn ...
输入:  model_seg0.weight  →  自动推导 model_seg1.weight model_seg2.weight ...
```

## 7. 运行示例

### 7.1 Qwen3.5-9B（2 段，默认使用外置 rope）

```bash
adb shell
cd /data/rknn_multicard_demo

export LD_LIBRARY_PATH=./lib

# 使用 taskset 绑定 CPU 核心，避免多线程竞争
taskset f0 ./rknn_multicard_demo \
    --model /data/models/multicard/Qwen3.5-9B-llm_seg0.rknn \
    --weight /data/models/multicard/Qwen3.5-9B-llm_seg0.weight \
    --vocab /data/models/multicard/Qwen3.5-9B-llm.tokenizer.gguf \
    --embed /data/models/multicard/Qwen3.5-9B-llm.embed.bin \
    -c 4096 \
    --core-mask 0xff \
    --stage-count 2 \
    --bucket-size 128 \
    --rope-tensor /data/models/multicard/Qwen3.5-9B-llm_seg0.safetensors \
    --prompt "<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n" \
    --predict 128
```

参数解释：

| 参数值 | 含义 |
|--------|------|
| `-c 4096` | 最大上下文长度 4096 token |
| `--core-mask 0xff` | 使用全部 NPU 核心 |
| `--stage-count 2` | 2 段（2 张卡） |
| `--bucket-size 128` | bucket_size = 128 token |
| `--prompt "<...>"` | Qwen3.5 格式的文本输入 |
| `--predict 128` | 最多生成 128 个 token |
| `--rope-tensor /data/models/...safetensors` | 外置 Rope Cache 文件 |

### 7.2 Gemma-4-12B-it（2 段，默认使用外置 rope）

```bash
adb shell
cd /data/rknn_multicard_demo

export LD_LIBRARY_PATH=./lib

# Gemma-4 与 Qwen3.5 一样默认使用外置 safetensors rope cache
taskset f0 ./rknn_multicard_demo \
    --model /data/models/multicard/gemma-4-12b-it_seg0.rknn \
    --weight /data/models/multicard/gemma-4-12b-it_seg0.weight \
    --vocab /data/models/multicard/gemma-4-12b-it.tokenizer.gguf \
    --embed /data/models/multicard/gemma-4-12b-it.embed.bin \
    -c 4096 \
    --core-mask 0xff \
    --stage-count 2 \
    --bucket-size 128 \
    --prompt "<|turn>user\n你好\n<|turn>model" \
    --predict 128 \
    --rope-tensor /data/models/multicard/gemma-4-12b-it_seg0.safetensors
```

参数解释：

| 参数值 | 含义 |
|--------|------|
| `-c 4096` | 最大上下文长度 4096 token |
| `--core-mask 0xff` | 使用全部 NPU 核心 |
| `--stage-count 2` | 2 段（2 张卡） |
| `--bucket-size 128` | bucket_size = 128 token |
| `--prompt "<...>"` | Gemma-4 格式的 prompt |
| `--predict 128` | 最多生成 128 个 token |
| `--rope-tensor /data/models/...safetensors` | **外置 Rope Cache 文件**（Gemma-4 默认必填） |

## 8. 性能统计

程序运行结束后会输出 Prefill 和 Decode 阶段的总体性能以及每个stage的性能：

```
Performance Statistics:
-----------------------------------------------------------------------------------------
 Stage      | Total Time (ms)  | Tokens   | Time per Token (ms)  | Tokens per Second
-----------------------------------------------------------------------------------------
 Prefill    | 5985.07          | 5120     | 1.17                 | 855.46
 Decode     | 3569.87          | 127      | 28.11                | 35.58
-----------------------------------------------------------------------------------------

Per-Stage Performance Statistics:
----------------------------------------------------------------------------------------------------------------------
 Stage      | Phase      | Runs     | Total Time (ms)  | Tokens   | Time per Token (ms)  | Tokens per Second
----------------------------------------------------------------------------------------------------------------------
 stage0     | Prefill    | 1        | 5881.76          | 5120     | 1.15                 | 870.49
 stage0     | Decode     | 127      | 1903.59          | 127      | 14.99                | 66.72
 stage1     | Prefill    | 40       | 3931.82          | 5120     | 0.77                 | 1302.20
 stage1     | Decode     | 127      | 1663.64          | 127      | 13.10                | 76.34
----------------------------------------------------------------------------------------------------------------------
```
