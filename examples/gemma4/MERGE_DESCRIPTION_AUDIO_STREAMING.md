# Gemma4 Audio：新增显式状态流式推理与动态多桶支持

## 变更背景

原有 Gemma4 Audio Tower 使用固定长度离线模型，最大约支持 7.5 秒音频。超过该长度时，需要扩大离线模型输入 shape，模型内存也会随最大输入长度增加。

本次变更新增真正的有状态 Audio streaming 路径，通过显式回灌 Attention KV、卷积状态和历史 mask，使模型工作集不随整段音频长度增长；同时使用 `240/480/720` Mel-frame 动态桶减少 RKNN 调用次数。原有离线 Audio Tower 完整保留，默认仍走 `offline`，不会改变已有调用行为。

## 主要修改

- 新增 Gemma4 Audio streaming ONNX 导出：
  - 将两层 subsampling convolution、12 层 Audio Attention KV、LightConv state 和历史 mask 显式暴露为模型输入/输出。
  - 保持 Gemma4 原有局部 Attention、relative position bias 和 softcap 计算语义。
  - 使用可逆的 `past KV * 0.5` / `current KV * 2.0` 编码，避免 RKNN3 将 Audio KV 误识别并改写为 LLM native KV cache；该编码只用于保持普通模型 IO，不改变 Attention 数值语义。
- 新增 RKNN3 动态多桶转换脚本：
  - 默认桶为 `240/480/720` Mel frames，对应约 `2.4/4.8/7.2 s` 输入和 `60/120/180` 个 Audio token。
  - 使用 8 核 FP16 转换，`do_quantization=False`。
  - 转换前校验固定的 8 输入、7 输出接口，避免转换错误的 ONNX 图。
- 新增板端 streaming Audio 实现：
  - 查询 RKNN3 动态 shape，并按全部桶的最大 aligned size 复用一套 IO memory。
  - 长音频优先使用 720-frame 主桶；尾块选择可容纳有效帧的最小桶。
  - 每块完成后显式回灌 Attention KV、卷积状态和历史 mask，状态长度固定，不随音频总长度增长。
  - Mel 前处理沿用离线 Audio Tower 参数与 `pocketfft::r2c` 实现：16 kHz mono、Hann 320、hop 160、FFT 512、128 Mel bins。
  - 支持连续处理任意长度 Audio Tower 输入；工程上限由宿主 embedding 缓冲、LLM context 和应用策略决定，不再由 Audio RKNN 最大桶决定。
- 完整 Gemma4 demo 新增可选 `audio_mode`：
  - `offline`：默认值，保持原有离线 Audio Tower 路径。
  - `streaming`：加载新的动态多桶流式 Audio 模型。

## 模型接口与状态管理

流式模型包含 8 个输入和 7 个输出：

```text
Inputs
  input_features
  input_features_mask
  conv_state_0
  conv_state_1
  audio_history_mask
  audio_history_key
  audio_history_value
  past_conv

Outputs
  hidden_states
  output_mask
  next_conv_state_0
  next_conv_state_1
  audio_current_key
  audio_current_value
  next_conv
```

这里的 KV cache 由 demo 显式管理，不使用 RKNN3 session 内置的 LLM KV cache。每次推理后仅保留固定最近 12 个 Audio token 的状态，因此 cache 占用不随整段音频长度增长。

## 兼容性

- 原离线 Audio Tower 的导出、初始化、推理和释放流程均保留。
- 未指定 `audio_mode` 时默认使用 `offline`，已有命令行无需修改。
- 只有显式传入 `streaming` 时才启用新模型和新运行时。
- Audio 与 LLM 的 embedding 拼接接口保持不变，流式路径只替换 Audio Tower 的生成方式。

## 验证环境与口径

以下均为本次开发期间已有的 RK1820 板端实测记录，并非本次文档整理时重新测试：

```text
NPU：RK1820，8 cores，1000 MHz
CPU：performance mode
DDR：2112 MHz
模型：FP16，dynamic buckets = 240/480/720
长音频：journal2，16 kHz mono float32，11.221 s
```

Audio latency/Audio-only 包含 Mel 前处理、输入输出 `mem_sync`、`rknn3_run`、状态回灌和宿主调度，不包含模型初始化。

## 多桶生效验证

三个动态桶均已在板端独立命中：

| 音频长度 | 命中桶 | 有效 Audio token | NPU run | Audio 全流程 |
|---:|---:|---:|---:|---:|
| 1.5 s | 240 | 38 | 43.280 ms | 67.956 ms |
| 3.5 s | 480 | 88 | 61.928 ms | 101.606 ms |
| 6.5 s | 720 | 163 | 82.192 ms | 145.080 ms |

11.221 秒长音频按 `720 + 480` 调度：首块输出 180 个有效 token，尾块实际使用 402 帧并补零到 480，输出 101 个有效 token，合计 281 个 token。

## 性能对比

### 11.221 秒：固定 48-frame 流式与动态多桶

两种实现均为 8 核 FP16、显式管理相同类型的 KV/卷积/mask 状态，输入相同，均输出 281 个有效 Audio token。

| 指标 | 固定 48-frame | 动态 `720+480` | 对比 |
|---|---:|---:|---:|
| RKNN 调用次数 | 24 | 2 | 减少 91.67% |
| NPU run 累计 | 646.608 ms | 144.408 ms | 4.48 倍加速 |
| chunk stage 累计 | 866.112 ms | 174.074 ms | 4.98 倍加速 |
| Audio-only | 950.517 ms | 255.917 ms | 3.71 倍加速 |
| 完整 demo Audio latency | 962.010 ms | 258.150 ms | 3.73 倍加速 |
| 有效 Audio token | 281 | 281 | 一致 |

完整 demo 端到端结果：

```text
Audio latency：258.15 ms，RTF 约 0.023
LLM Prefill：  411.18 ms / 298 tokens
LLM Generate： 350.73 ms / 22 tokens
ASR：Spent the afternoon at the park. It was a perfect day for walk and I even
     spotted a few caterpillars.
```

动态多桶与此前高精度固定 48-frame 流式版本的 ASR 文本一致。该结果验证的是两条流式实现的端到端一致性；音频中 `cherry blossoms` 被模型识别为 `caterpillars`，因此不声明为逐字正确的参考转写。

### 7.0/7.5 秒边界对比原离线模型

| 音频 | 离线 756-frame | 动态流式 | 结论 |
|---:|---:|---:|---|
| 7.0 s | 155.434 ms | 148.582 ms，单次 720 桶 | 流式快 6.852 ms（4.41%） |
| 7.5 s | 158.492 ms | 208.491 ms，720+240 | 离线快 50.000 ms；流式多一次 RKNN 调用 |

流式性能在最大桶边界处不连续：7.0 秒可由一个 720 桶覆盖，性能与离线相近且略快；7.5 秒产生 749 个有效 Mel frame，需要额外执行一次 240 尾桶，因此慢于单次执行的离线 756-frame 模型；超过原离线模型约 7.5 秒上限后，应使用流式路径。

## 显式 KV cache 的 `mem_sync` 耗时

`input_sync_us` 和 `output_sync_us` 是一次 RKNN 调用中全部 8 个输入或全部 7 个输出的同步耗时，包含 features、mask、卷积状态和 KV 等 tensor，不能解释为“仅 KV cache 的拷贝耗时”。`feedback_us` 才是输出状态在宿主侧重排并写回下一块输入 buffer 的耗时。

11.221 秒动态 `720+480` 的逐块实测：

| Chunk | NPU run | Input mem_sync | Output mem_sync | State feedback | Chunk total |
|---|---:|---:|---:|---:|---:|
| 720 frames | 82.350 ms | 7.779 ms | 7.530 ms | 0.223 ms | 97.883 ms |
| 480 frames | 62.058 ms | 6.637 ms | 7.340 ms | 0.155 ms | 76.191 ms |
| 合计 | 144.408 ms | **14.416 ms** | **14.870 ms** | **0.378 ms** | **174.074 ms** |

因此，最终多桶方案在整段 11.221 秒音频上的全部输入同步约 14.4 ms、全部输出同步约 14.9 ms，显式状态回灌约 0.38 ms。同步耗时约占两个 chunk stage 的 16.8%，状态回灌本身约占 0.22%。

固定 48-frame 基线的 72 个 chunk 统计，平均每块为：

| 阶段 | 平均每 chunk | 占 chunk total |
|---|---:|---:|
| `rknn3_run` | 26.942 ms | 74.66% |
| Input mem_sync | 4.162 ms | 11.53% |
| Output mem_sync | 4.828 ms | 13.38% |
| State feedback | 0.155 ms | 0.43% |
| Chunk total | 36.088 ms | 100% |

换算到 11.221 秒的 24 个小块，Input/Output mem_sync 分别累计约 99.888/115.872 ms。此前出现的几十毫秒甚至约 60 ms 级同步结论属于多个小 chunk 的累计值，不是单次同步，也不是 KV-only 耗时。动态大桶的主要收益正是将调用和同步轮数从 24 次降为 2 次。

## 模型内存

最终 `240/480/720` 流式模型的 `model_report.html`：

| 项目 | 内存 |
|---|---:|
| Weight | 563.800 MiB |
| Command | 27.765 MiB |
| Internal | 45.970 MiB |
| 总计 | **637.535 MiB** |
| 最大单核总内存 | **81.172 MiB** |

与 8 核离线 `240/528/768` 报告对比：

| 模型 | 总内存 | 最大单核 |
|---|---:|---:|
| 离线 `240/528/768` | 664.655 MiB | 97.745 MiB |
| 流式 `240/480/720` | 637.535 MiB | 81.172 MiB |

流式模型总内存减少 27.121 MiB（4.08%），最大单核减少 16.573 MiB，并解除 Audio Tower 的 7.5 秒输入上限。

## 精度结果与已知限制

- 多桶 ONNX 使用 `720+480` 处理 11.221 秒音频，与整段 PyTorch Audio Tower 的 281 个有效 embedding 对比：cosine `1.000000`、MAE `3.12017e-6`、RMSE `7.09746e-6`。
- 板端 RKNN 与相同多桶 ONNX 的整段有效 embedding 对比：cosine `0.997539`、MAE `0.110702`、RMSE `0.191837`。
- RKNN 数值误差主要来自大块流式图的 ONNX 到 RKNN 编译/FP16 执行，而不是 C++ Mel 前处理或 chunk/state 拼接；当前端到端 ASR 文本与固定 48-frame 高精度基线一致。
- 如果后续要求 RKNN embedding cosine 必须达到 `0.999`，需要继续逐层 accuracy analysis，或将最大主桶缩小到 480，在性能、内存和精度之间重新取点。
- 流式 Audio Tower 可持续输出任意长度的 embedding，但完整 Audio+LLM 仍受 LLM context 长度和宿主 embedding 累积内存约束。

## 验收结论

- 原离线 Audio Tower 默认流程保持兼容。
- `240/480/720` 三个动态桶已分别上板命中。
- 11.221 秒输入可通过 `720+480` 正确生成 281 个有效 Audio token，并完成 Audio+LLM 推理。
- 模型总内存和最大单核内存均未超过现有 7.5 秒离线多桶方案。
- 相比固定 48-frame 流式基线，11.221 秒 Audio latency 从 962.010 ms 降至 258.150 ms，提升 3.73 倍。
- 显式状态的宿主反馈不是主要耗时；主要额外开销来自每次 RKNN 调用的输入/输出 `mem_sync`，动态大桶已通过显著减少调用次数降低该开销。

