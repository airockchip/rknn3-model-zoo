// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Whisper 单模型 ASR 演示程序:
// 音频 -> log-mel 特征提取 -> 单个 RKNN 模型推理 -> ASR 识别结果输出
//
// 整体流程:
//   1. 加载 encoder / decode0 / decoder1 三个 RKNN 模型
//   2. 读取 WAV 音频并提取 log-mel 频谱特征
//   3. encoder 编码音频特征
//   4. decode0 生成 cross-attention 的 key/value
//   5. decoder1 自回归生成 token，最终解码为文本

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>
#include <vector>

#include "audio_utils.h"
#include "easy_timer.h"
#include "float16.h"
#include "process.h"
#include "rknn3_api.h"
#include "whisper.h"
// Model config: max_target_positions=448 for both supported variants.
static constexpr int WHISPER_MAX_TARGET_POSITIONS = 448;
static constexpr int WHISPER_PROMPT_TOKENS = 4;
// Prefill produces the first new token without an additional decoder position.
static constexpr int WHISPER_MAX_NEW_TOKENS = WHISPER_MAX_TARGET_POSITIONS - WHISPER_PROMPT_TOKENS + 1;
static constexpr uint32_t WHISPER_KVCACHE_RESIDUAL_DEPTH = 32;
struct WhisperModelConfig
{
  int vocab_size;
  int n_mels;
  int32_t transcribe_token;
  int32_t no_timestamps_token;
  const char *mel_filters_path;
};

static const WhisperModelConfig base_config = {
    51865, 80, 50359, 50363, "./model/mel_80_filters.txt"
};
static const WhisperModelConfig turbo_config = {
    51866, 128, 50360, 50364, "./model/mel_128_filters.txt"
};

static bool parse_max_new_tokens(const char *value, int *result)
{
  if (!value || !value[0]) return false;
  for (const char *p = value; *p; ++p)
  {
    if (*p < '0' || *p > '9') return false;
  }
  errno = 0;
  char *end = NULL;
  long parsed = strtol(value, &end, 10);
  if (errno == ERANGE || *end != '\0' || parsed < 1 || parsed > WHISPER_MAX_NEW_TOKENS)
  {
    return false;
  }
  *result = (int)parsed;
  return true;
}

// 判断字符串 value 是否以 prefix 开头
static bool starts_with(const char *value, const char *prefix)
{
  return strncmp(value, prefix, strlen(prefix)) == 0;
}

enum Decoder1ShapeMode
{
  DECODER1_SHAPE_AUTO = 0,   // prefill 用大 shape，decode 用小 shape,由于音频为30s,结果与非流式一致
  DECODER1_SHAPE0_ONLY = 1,  // 非流式
  DECODER1_SHAPE1_ONLY = 2,  // 流式
};

static Decoder1ShapeMode g_decoder1_shape_mode = DECODER1_SHAPE1_ONLY; // 默认为流式模式

static const char *decoder1_shape_mode_name(Decoder1ShapeMode mode)
{
  switch (mode)
  {
  case DECODER1_SHAPE0_ONLY:
    return "shape0_only";
  case DECODER1_SHAPE1_ONLY:
    return "shape1_only";
  case DECODER1_SHAPE_AUTO:
  default:
    return "auto";
  }
}

static int sync_all_outputs_from_device(rknn_app_context_t *ctx, const char *tag)
{
  if (!ctx || !tag)
  {
    printf("sync_all_outputs_from_device invalid args tag=%s ctx=%p\n", tag ? tag : "<null>", ctx);
    return -1;
  }

  for (uint32_t i = 0; i < ctx->io_num.n_output; ++i)
  {
    if (!ctx->outputs[i].mem)
    {
      printf("[%s] output mem is null index=%u name=%s\n",
             tag,
             i,
             ctx->outputs[i].attr ? ctx->outputs[i].attr->name : "<null>");
      if (ctx->outputs[i].attr)
      {
        whisper_print_tensor_shape_brief("  output", ctx->outputs[i].attr);
      }
      return -1;
    }
    int ret = rknn3_mem_sync(ctx->rknn_ctx, ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
    if (ret != RKNN3_SUCCESS)
    {
      printf("[%s] output sync failed index=%u name=%s ret=%d mem_size=%lu\n",
             tag,
             i,
             ctx->outputs[i].attr ? ctx->outputs[i].attr->name : "<null>",
             ret,
             ctx->outputs[i].mem ? (unsigned long)ctx->outputs[i].mem->size : 0UL);
      if (ctx->outputs[i].attr)
      {
        whisper_print_tensor_shape_brief("  output", ctx->outputs[i].attr);
      }
      return ret;
    }
  }

  return 0;
}
// ============================================================================
// ============================================================================

// ============================================================================
// 数据类型工具函数
// ============================================================================

// 解析十六进制 core mask 参数，例如 0xff / ff / 0x1。
// 返回 false 表示参数非法；不再支持 auto，也不允许 0，避免走自动 core mask。
static bool parse_core_mask_hex(const char *value, uint32_t *out)
{
  if (!value || !out || value[0] == '\0')
  {
    return false;
  }

  if (value[0] == '-' || value[0] == '+' || isspace((unsigned char)value[0]))
  {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  unsigned long mask = strtoul(value, &end, 16);
  if (errno == ERANGE || end == value || *end != '\0' || mask == 0 || mask > UINT32_MAX)
  {
    return false;
  }

  *out = (uint32_t)mask;
  return true;
}

static uint32_t calc_valid_encoder_frames(const audio_buffer_t *audio, uint32_t max_audio_length)
{
  if (!audio || audio->num_frames <= 0)
  {
    return 0;
  }

  uint32_t valid_audio_samples = (uint32_t)audio->num_frames;
  if (valid_audio_samples > max_audio_length)
  {
    valid_audio_samples = max_audio_length;
  }

  // process.cc 里有效 mel 帧数是 audio_length / HOP_LENGTH
  uint32_t valid_mel_frames = valid_audio_samples / HOP_LENGTH;

  // Whisper encoder conv2 stride=2: 3000 mel frames -> 1500 encoder frames
  uint32_t valid_encoder_frames = (valid_mel_frames + 1) / 2;

  uint32_t max_encoder_frames = max_audio_length / HOP_LENGTH / 2;
  if (valid_encoder_frames > max_encoder_frames)
  {
    valid_encoder_frames = max_encoder_frames;
  }

  return valid_encoder_frames;
}

static uint32_t get_encoder_mask_seq_len(const rknn3_tensor_attr *attr)
{
  if (!attr || attr->n_dims == 0)
  {
    return 0;
  }

  // 原始 ONNX mask: [1, 1, 1, S]
  if (attr->n_dims == 4)
  {
    return attr->shape[3];
  }

  // RKNN 可能把 [1,1,1,S] 打包成 [1,1,1,S,16]
  // 此时最后一维 16 是 C2 对齐，不是 seq_len。
  if (attr->n_dims == 5 && attr->shape[4] == 16)
  {
    return attr->shape[3];
  }

  return attr->shape[attr->n_dims - 1];
}

static void fill_encoder_mask_values(std::vector<float> &mask, const rknn3_tensor_attr *attr,
                                     uint32_t mask_start, uint32_t seq_len, float pad_value)
{
  std::fill(mask.begin(), mask.end(), 0.0f);

  if (!attr)
  {
    return;
  }

  // 原始 4D: [1, 1, 1, S]
  if (attr->n_dims == 4)
  {
    for (uint32_t i = mask_start; i < seq_len && i < mask.size(); ++i)
    {
      mask[i] = pad_value;
    }
    return;
  }

  // packed 5D: [1, 1, 1, S, 16]
  // shape[3] 才是真正的时间长度，shape[4] 是 C2 对齐。
  if (attr->n_dims == 5 && attr->shape[4] == 16)
  {
    uint32_t c1 = attr->shape[1];
    uint32_t h = attr->shape[2];
    uint32_t w = attr->shape[3];
    uint32_t c2 = attr->shape[4];

    for (uint32_t s = mask_start; s < seq_len && s < w; ++s)
    {
      for (uint32_t lane = 0; lane < c2; ++lane)
      {
        size_t off = ((((size_t)0 * c1 + 0) * h + 0) * w + s) * c2 + lane;
        if (off < mask.size())
        {
          mask[off] = pad_value;
        }
      }
    }
    return;
  }

  // fallback
  for (uint32_t i = mask_start; i < seq_len && i < mask.size(); ++i)
  {
    mask[i] = pad_value;
  }
}

static int fill_encoder_attention_mask(rknn_app_context_t *encoder_ctx, const audio_buffer_t *audio,
                                       uint32_t max_audio_length, bool is_turbo)
{
  if (!encoder_ctx)
  {
    printf("stage=preprocess fill_encoder_attention_mask invalid encoder_ctx=%p\n", encoder_ctx);
    return -1;
  }

  int mask_index = whisper_find_input_index_by_name(encoder_ctx, "encoder_attention_mask");
  if (mask_index < 0)
  {
    // 旧 encoder 没有 mask 输入，直接兼容通过
    return 0;
  }

  rknn3_tensor *mask_tensor = &encoder_ctx->inputs[mask_index];
  if (!mask_tensor->attr || !mask_tensor->mem || !mask_tensor->mem->virt_addr)
  {
    printf("stage=preprocess encoder_attention_mask tensor invalid index=%d ret=-1\n", mask_index);
    if (mask_tensor->attr)
    {
      whisper_print_tensor_shape_brief("  encoder_attention_mask", mask_tensor->attr);
    }
    return -1;
  }

  uint32_t mask_elems = whisper_shape_count(mask_tensor->attr);
  std::vector<float> mask(mask_elems, 0.0f);

  enum EncoderMaskMode
  {
    ENCODER_MASK_ZERO = 0,
    ENCODER_MASK_HARD = 1,
    ENCODER_MASK_DELAY_HARD = 2,
  };

  const EncoderMaskMode encoder_mask_mode = is_turbo ? ENCODER_MASK_ZERO : ENCODER_MASK_DELAY_HARD;
  const uint32_t encoder_mask_margin_frames = 250;  // 约 5 秒，因为 encoder 50fps
  const float encoder_mask_pad_value = -10.0f;

  uint32_t valid_encoder_frames = calc_valid_encoder_frames(audio, max_audio_length);
  uint32_t encoder_seq_len = get_encoder_mask_seq_len(mask_tensor->attr);

  if (valid_encoder_frames > encoder_seq_len)
  {
    valid_encoder_frames = encoder_seq_len;
  }

  uint32_t mask_start = encoder_seq_len;

  if (encoder_mask_mode == ENCODER_MASK_ZERO)
  {
    mask_start = encoder_seq_len;
  }
  else if (encoder_mask_mode == ENCODER_MASK_HARD)
  {
    mask_start = valid_encoder_frames;
  }
  else if (encoder_mask_mode == ENCODER_MASK_DELAY_HARD)
  {
    mask_start = valid_encoder_frames + encoder_mask_margin_frames;
    if (mask_start > encoder_seq_len)
    {
      mask_start = encoder_seq_len;
    }
  }

  fill_encoder_mask_values(mask, mask_tensor->attr, mask_start, encoder_seq_len, encoder_mask_pad_value);

  printf("[encoder mask] mode=%d valid_frames=%u mask_start=%u seq_len=%u pad_value=%.1f\n",
       (int)encoder_mask_mode,
       valid_encoder_frames,
       mask_start,
       encoder_seq_len,
       encoder_mask_pad_value);
  whisper_print_tensor_shape_brief("[encoder mask attr]", mask_tensor->attr);

  memset(mask_tensor->mem->virt_addr, 0, mask_tensor->mem->size);

  int ret = whisper_convert_fp32_to_tensor(mask.data(), mask_tensor->mem->virt_addr, mask_elems, mask_tensor->attr->dtype);
  if (ret != 0)
  {
    printf("stage=preprocess fill encoder_attention_mask failed ret=%d input=encoder_attention_mask index=%d\n", ret, mask_index);
    return ret;
  }

  ret = rknn3_mem_sync(encoder_ctx->rknn_ctx, mask_tensor->mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
  if (ret != RKNN3_SUCCESS)
  {
    printf("stage=preprocess rknn3_mem_sync encoder_attention_mask failed ret=%d index=%d name=encoder_attention_mask mem_size=%lu\n",
           ret, mask_index, (unsigned long)mask_tensor->mem->size);
    whisper_print_tensor_shape_brief("  encoder_attention_mask", mask_tensor->attr);
    return ret;
  }

  return 0;
}

// ============================================================================
// 文件 I/O 辅助函数
// ============================================================================

// 读取二进制文件到 vector<uint8_t> 中
static int read_binary_file(const char *path, std::vector<uint8_t> &data)
{
  if (!path)
  {
    printf("read_binary_file invalid path=null ret=-1\n");
    return -1;
  }
  FILE *fp = fopen(path, "rb");
  if (!fp)
  {
    printf("read_binary_file open failed path=%s ret=-1\n", path ? path : "<null>");
    return -1;
  }
  fseek(fp, 0, SEEK_END);
  long size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  if (size <= 0)
  {
    fclose(fp);
    printf("read_binary_file empty path=%s size=%ld ret=-1\n", path ? path : "<null>", size);
    return -1;
  }
  data.resize((size_t)size);
  size_t n = fread(data.data(), 1, data.size(), fp);
  fclose(fp);
  if (n != data.size())
  {
    printf("read_binary_file read failed path=%s expected=%lu actual=%lu ret=-1\n",
           path ? path : "<null>", (unsigned long)data.size(), (unsigned long)n);
    return -1;
  }
  return 0;
}

// 保存 Whisper 解码过程中“流式输出”的状态
// 记录词表指针、已生成的 token ID 列表，以及拼接出来的文本
struct WhisperStreamState
{
  VocabEntry *vocab;              // 词表指针
  std::vector<int32_t> token_ids; // 已生成的 token ID 序列
  std::string text;               // 拼接后的输出文本
};

// 把 decoder1 和 decode0 两个模型上下文打包在一起
// 保存 decoder1 在某一种动态 shape 下的输入/输出 tensor 属性信息
struct Decoder1ShapeAttrs
{
  std::vector<rknn3_tensor_attr> inputs;   // 该 shape 下的输入属性列表
  std::vector<rknn3_tensor_attr> outputs;  // 该 shape 下的输出属性列表
};

// 保存 decoder1 手动 raw 解码流程中的动态 shape 运行信息
// 用于区分 prefill 阶段和 decode 阶段应该使用哪一种 shape_id
struct Decoder1RawRuntime
{
  std::vector<Decoder1ShapeAttrs> shapes;

  uint32_t prefill_shape_id;
  uint32_t decode_shape_id;
  uint32_t prefill_seq_len;
  uint32_t decode_seq_len;

  Decoder1ShapeMode shape_mode;
  bool has_current_shape;
  uint32_t current_shape_id;

  bool static_inputs_valid;
  uint32_t static_inputs_shape_id;

  bool attention_mask_valid;
  uint32_t attention_mask_shape_id;
  uint32_t attention_mask_n_tokens;

  bool scalar_valid[5];
  int32_t scalar_values[5];

  int input_embed_index;
  int attention_mask_index;
  int position_ids_index;
  int scalar_indices[5];
};

// ============================================================================
// Whisper 特殊 Token 定义
// ============================================================================

static const char *DECODER1_SCALAR_INPUT_NAMES[5] = {
    "Th",
    "Tc",
    "Ts",
    "Tsr",
    "num_logits_to_keep",
};
static void init_decoder1_runtime_input_indices(rknn_app_context_t *ctx, Decoder1RawRuntime *runtime)
{
  if (!ctx || !runtime)
  {
    return;
  }
  runtime->input_embed_index = whisper_find_input_index_by_name(ctx, "input_embeds");
  if (runtime->input_embed_index < 0)
  {
    runtime->input_embed_index = whisper_find_input_index_by_name(ctx, "inputs_embeds");
  }
  if (runtime->input_embed_index < 0)
  {
    printf("stage=decoder1_prepare missing input name=input_embeds/inputs_embeds fallback_index=0\n");
  }
  runtime->attention_mask_index = whisper_find_input_index_by_name(ctx, "attention_mask");
  if (runtime->attention_mask_index < 0)
  {
    printf("stage=decoder1_prepare missing input name=attention_mask fallback_index=0\n");
  }
  runtime->position_ids_index = whisper_find_input_index_by_name(ctx, "position_ids");
  if (runtime->position_ids_index < 0)
  {
    printf("stage=decoder1_prepare missing input name=position_ids fallback_index=0\n");
  }
  for (uint32_t i = 0; i < sizeof(runtime->scalar_indices) / sizeof(runtime->scalar_indices[0]); ++i)
  {
    runtime->scalar_indices[i] = whisper_find_input_index_by_name(ctx, DECODER1_SCALAR_INPUT_NAMES[i]);
  }
}
// ============================================================================
// Decoder1 动态 Shape 初始化与管理
// ============================================================================

// 读取 decoder1 模型的动态 shape 信息
// 判断哪个 shape 用于 prompt/prefill（序列最长），哪个 shape 用于单 token decode（序列最短）
// 并提前给所有输入/输出 tensor 分配足够大的内存（取所有 shape 中的最大值）
static int prepare_decoder1_raw_runtime(rknn_app_context_t *ctx, Decoder1RawRuntime *runtime)
{
  if (!ctx || !runtime)
  {
    printf("stage=decoder1_prepare invalid args ctx=%p runtime=%p\n", ctx, runtime);
    return -1;
  }

  runtime->shapes.clear();
  runtime->prefill_shape_id = 0;
  runtime->decode_shape_id = 0;
  runtime->prefill_seq_len = 0;
  runtime->decode_seq_len = 0;
  runtime->shape_mode = g_decoder1_shape_mode;
  runtime->has_current_shape = false;
  runtime->current_shape_id = 0;
  runtime->static_inputs_valid = false;
  runtime->static_inputs_shape_id = 0;
  runtime->attention_mask_valid = false;
  runtime->attention_mask_shape_id = 0;
  runtime->attention_mask_n_tokens = 0;
  memset(runtime->scalar_valid, 0, sizeof(runtime->scalar_valid));
  memset(runtime->scalar_values, 0, sizeof(runtime->scalar_values));
  runtime->input_embed_index = -1;
  runtime->attention_mask_index = -1;
  runtime->position_ids_index = -1;
  for (uint32_t i = 0; i < sizeof(runtime->scalar_indices) / sizeof(runtime->scalar_indices[0]); ++i)
  {
    runtime->scalar_indices[i] = -1;
  }

  init_decoder1_runtime_input_indices(ctx, runtime);

  rknn3_shape_config shape_cfg;
  memset(&shape_cfg, 0, sizeof(shape_cfg));
  int ret = rknn3_query(ctx->rknn_ctx, RKNN3_QUERY_DYNAMIC_SHAPE_CONFIG, &shape_cfg, sizeof(shape_cfg));
  if (ret != RKNN3_SUCCESS)
  {
    printf("stage=decoder1_prepare query dynamic shape config failed ret=%d shape_mode=%s\n",
           ret, decoder1_shape_mode_name(runtime->shape_mode));
    return ret;
  }
  if (shape_cfg.n_shapes == 0)
  {
    printf("stage=decoder1_prepare decoder1 has no dynamic shape info, use current attrs n_shapes=%u shape_mode=%s\n",
           shape_cfg.n_shapes, decoder1_shape_mode_name(runtime->shape_mode));
    return 0;
  }

  uint32_t n_shapes = shape_cfg.n_shapes;
  std::vector<rknn3_shape_info> infos(n_shapes);
  std::vector<std::vector<rknn3_tensor_attr>> input_attrs(n_shapes);
  std::vector<std::vector<rknn3_tensor_attr>> output_attrs(n_shapes);
  for (uint32_t s = 0; s < n_shapes; ++s)
  {
    input_attrs[s].resize(ctx->io_num.n_input);
    output_attrs[s].resize(ctx->io_num.n_output);
    memset(input_attrs[s].data(), 0, sizeof(rknn3_tensor_attr) * ctx->io_num.n_input);
    memset(output_attrs[s].data(), 0, sizeof(rknn3_tensor_attr) * ctx->io_num.n_output);
    infos[s].shape_id = s;
    infos[s].input_attrs = input_attrs[s].data();
    infos[s].output_attrs = output_attrs[s].data();
  }

  ret = rknn3_query(ctx->rknn_ctx, RKNN3_QUERY_DYNAMIC_SHAPE_INFO, infos.data(), sizeof(rknn3_shape_info) * n_shapes);
  if (ret != RKNN3_SUCCESS)
  {
    printf("RKNN3_QUERY_DYNAMIC_SHAPE_INFO failed: %d\n", ret);
    return ret;
  }

  runtime->shapes.resize(n_shapes);
  uint32_t max_seq = 0;
  uint32_t min_seq = 0xffffffffu;
  for (uint32_t s = 0; s < n_shapes; ++s)
  {
    runtime->shapes[s].inputs = input_attrs[s];
    runtime->shapes[s].outputs = output_attrs[s];
    uint32_t seq = whisper_decoder_embed_seq_len(runtime->shapes[s].inputs);
    if (seq > max_seq)
    {
      max_seq = seq;
      runtime->prefill_shape_id = s;
    }
    if (seq > 0 && seq < min_seq)
    {
      min_seq = seq;
      runtime->decode_shape_id = s;
    }
  }

  runtime->prefill_seq_len = max_seq;
  runtime->decode_seq_len = min_seq == 0xffffffffu ? 0 : min_seq;

  for (uint32_t i = 0; i < ctx->io_num.n_input; ++i)
  {
    uint64_t max_size = 0;
    uint32_t core_id = ctx->inputs[i].attr ? ctx->inputs[i].attr->core_id : 0;
    if (runtime->shape_mode == DECODER1_SHAPE1_ONLY)
    {
      max_size = runtime->shapes[runtime->decode_shape_id].inputs[i].aligned_size;
      core_id = runtime->shapes[runtime->decode_shape_id].inputs[i].core_id;
      ret = whisper_recreate_tensor_mem_exact(ctx->rknn_ctx, &ctx->inputs[i], max_size, core_id);
    }
    else
    {
      for (uint32_t s = 0; s < n_shapes; ++s)
      {
        if (runtime->shapes[s].inputs[i].aligned_size > max_size)
        {
          max_size = runtime->shapes[s].inputs[i].aligned_size;
          core_id = runtime->shapes[s].inputs[i].core_id;
        }
      }
      ret = whisper_ensure_tensor_mem_size(ctx->rknn_ctx, &ctx->inputs[i], max_size, core_id);
    }
    if (ret != 0)
    {
      printf("stage=decoder1_prepare input mem alloc failed index=%u name=%s ret=%d shape_mode=%s shape_id=%u aligned_size=%lu core_id=%u\n",
             i,
             ctx->inputs[i].attr ? ctx->inputs[i].attr->name : "<null>",
             ret,
             decoder1_shape_mode_name(runtime->shape_mode),
             runtime->shape_mode == DECODER1_SHAPE1_ONLY ? runtime->decode_shape_id : runtime->prefill_shape_id,
             (unsigned long)max_size,
             core_id);
      if (ctx->inputs[i].attr)
      {
        whisper_print_tensor_shape_brief("  input", ctx->inputs[i].attr);
      }
      return ret;
    }
  }

for (uint32_t i = 0; i < ctx->io_num.n_output; ++i)
{
  uint64_t mem_size = 0;
  uint32_t core_id = ctx->outputs[i].attr ? ctx->outputs[i].attr->core_id : 0;

  if (runtime->shape_mode == DECODER1_SHAPE1_ONLY)
  {
    mem_size = runtime->shapes[runtime->decode_shape_id].outputs[i].aligned_size;
    core_id = runtime->shapes[runtime->decode_shape_id].outputs[i].core_id;
    ret = whisper_recreate_tensor_mem_exact(ctx->rknn_ctx, &ctx->outputs[i], mem_size, core_id);
  }
  else
  {
    for (uint32_t s = 0; s < n_shapes; ++s)
    {
      if (runtime->shapes[s].outputs[i].aligned_size > mem_size)
      {
        mem_size = runtime->shapes[s].outputs[i].aligned_size;
        core_id = runtime->shapes[s].outputs[i].core_id;
      }
    }
    ret = whisper_ensure_tensor_mem_size(ctx->rknn_ctx, &ctx->outputs[i], mem_size, core_id);
  }

  if (ret != 0)
  {
    printf("stage=decoder1_prepare output mem alloc failed index=%u name=%s ret=%d shape_mode=%s shape_id=%u aligned_size=%lu core_id=%u\n",
           i,
           ctx->outputs[i].attr ? ctx->outputs[i].attr->name : "<null>",
           ret,
           decoder1_shape_mode_name(runtime->shape_mode),
           runtime->shape_mode == DECODER1_SHAPE1_ONLY ? runtime->decode_shape_id : runtime->prefill_shape_id,
           (unsigned long)mem_size,
           core_id);
    if (ctx->outputs[i].attr)
    {
      whisper_print_tensor_shape_brief("  output", ctx->outputs[i].attr);
    }
    return ret;
  }
}

  return 0;
}

// 打印 decoder1 所有输入/输出 tensor 的内存摘要信息（形状、对齐大小、core_id 等）
// 把 decoder1 切换到指定的 dynamic shape，并同步更新当前输入/输出 tensor 的 attr 和内存大小
static int switch_decoder1_raw_shape(rknn_app_context_t *ctx, Decoder1RawRuntime *runtime, uint32_t shape_id)
{
  if (!ctx || !runtime)
  {
    printf("stage=decoder1_switch invalid args ctx=%p runtime=%p shape_id=%u\n", ctx, runtime, shape_id);
    return -1;
  }
  if (runtime->shapes.empty())
  {
    return 0;
  }
  if (shape_id >= runtime->shapes.size())
  {
    printf("stage=decoder1_switch invalid shape_id=%u n_shapes=%zu shape_mode=%s\n",
           shape_id, runtime->shapes.size(), decoder1_shape_mode_name(runtime->shape_mode));
    return -1;
  }
  if (runtime->has_current_shape && runtime->current_shape_id == shape_id)
  {
    return 0;
  }

  int ret = rknn3_set_shape(ctx->rknn_ctx, shape_id);
  if (ret != RKNN3_SUCCESS)
  {
    printf("stage=decoder1_switch rknn3_set_shape failed ret=%d shape_id=%u shape_mode=%s\n",
           ret, shape_id, decoder1_shape_mode_name(runtime->shape_mode));
    return ret;
  }

  for (uint32_t i = 0; i < ctx->io_num.n_input; ++i)
  {
    *ctx->inputs[i].attr = runtime->shapes[shape_id].inputs[i];
    if (!ctx->inputs[i].mem || ctx->inputs[i].mem->size < ctx->inputs[i].attr->aligned_size)
    {
      printf("stage=decoder1_switch input mem too small index=%u name=%s need=%lu actual=%lu shape_id=%u shape_mode=%s\n",
             i, ctx->inputs[i].attr->name, ctx->inputs[i].attr->aligned_size,
             ctx->inputs[i].mem ? ctx->inputs[i].mem->size : 0,
             shape_id,
             decoder1_shape_mode_name(runtime->shape_mode));
      whisper_print_tensor_shape_brief("  input", ctx->inputs[i].attr);
      return -1;
    }
  }

for (uint32_t i = 0; i < ctx->io_num.n_output; ++i)
{
  *ctx->outputs[i].attr = runtime->shapes[shape_id].outputs[i];

  // A previous decode shape may have shrunk the output allocation. Grow it
  // again before switching back to a larger prefill shape.
  if (!ctx->outputs[i].mem || ctx->outputs[i].mem->size < ctx->outputs[i].attr->aligned_size)
  {
    ret = whisper_recreate_tensor_mem_exact(ctx->rknn_ctx,
                                    &ctx->outputs[i],
                                    ctx->outputs[i].attr->aligned_size,
                                    ctx->outputs[i].attr->core_id);
    if (ret != 0)
    {
      printf("stage=decoder1_switch grow output mem failed index=%u ret=%d shape_id=%u\n", i, ret, shape_id);
      return ret;
    }
    continue;
  }

  if (i == 0 &&
      shape_id == runtime->decode_shape_id &&
      (runtime->shape_mode == DECODER1_SHAPE_AUTO ||
       runtime->shape_mode == DECODER1_SHAPE1_ONLY) &&
      ctx->outputs[i].mem->size != ctx->outputs[i].attr->aligned_size)
  {
    ret = whisper_recreate_tensor_mem_exact(ctx->rknn_ctx,
                                    &ctx->outputs[i],
                                    ctx->outputs[i].attr->aligned_size,
                                    ctx->outputs[i].attr->core_id);
    if (ret != 0)
    {
      printf("stage=decoder1_switch recreate output mem failed index=%u name=%s ret=%d shape_id=%u shape_mode=%s aligned_size=%lu\n",
             i,
             ctx->outputs[i].attr ? ctx->outputs[i].attr->name : "<null>",
             ret,
             shape_id,
             decoder1_shape_mode_name(runtime->shape_mode),
             ctx->outputs[i].attr ? (unsigned long)ctx->outputs[i].attr->aligned_size : 0UL);
      if (ctx->outputs[i].attr)
      {
        whisper_print_tensor_shape_brief("  output", ctx->outputs[i].attr);
      }
      return ret;
    }
    continue;
  }

}

  runtime->has_current_shape = true;
  runtime->current_shape_id = shape_id;
  runtime->static_inputs_valid = false;
  runtime->attention_mask_valid = false;
  memset(runtime->scalar_valid, 0, sizeof(runtime->scalar_valid));
  return 0;
}

// ============================================================================
// Decoder1 输入填充函数
// ============================================================================

// 给 decoder1 的 INT32 标量控制输入（Th/Tc/Ts/Tsr/num_logits_to_keep）赋值并同步到 NPU
// Th - 历史 token 数, Tc - 当前 token 数, Ts/Tsr - 序列总长度
// num_logits_to_keep - 需要保留的 logits 数量（prefill 阶段通常只需最后一个）
static int fill_decoder1_raw_scalar_inputs(rknn_app_context_t *decoder1_ctx, Decoder1RawRuntime *runtime,
                                           int32_t th, int32_t tc, int32_t ts, int32_t tsr, int32_t num_logits)
{
  int control_index = whisper_find_input_index_by_name(decoder1_ctx, "control_variables");
  if (control_index >= 0)
  {
    rknn3_tensor *tensor = &decoder1_ctx->inputs[control_index];
    // Sequential decoding, no pinned prefix or cache rollback. The FP16
    // residual holds the last incomplete 32-token block.
    if (decoder1_ctx->kvcache_residual_depth != WHISPER_KVCACHE_RESIDUAL_DEPTH || th < 0 || th + tc > WHISPER_MAX_TARGET_POSITIONS)
    {
      return -1;
    }
    int32_t stored = (th / 32) * 32;
    const int32_t values[9] = {th, tc, 0, 0, stored ? 0 : -1, stored, th % 32, th, 1};
    if (!tensor->attr || !tensor->mem || tensor->attr->dtype != RKNN3_TENSOR_INT32 ||
        tensor->attr->n_dims != 2 || tensor->attr->shape[0] != 1 ||
        tensor->attr->shape[1] != 9)
    {
      return -1;
    }
    memset(tensor->mem->virt_addr, 0, tensor->mem->size);
    int32_t *data = reinterpret_cast<int32_t *>(tensor->mem->virt_addr);
    for (uint32_t i = 0; i < 9; ++i)
    {
      data[i * tensor->attr->stride[1]] = values[i];
    }
    int ret = rknn3_mem_sync(decoder1_ctx->rknn_ctx, tensor->mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN3_SUCCESS) return ret;
    int index = whisper_find_input_index_by_name(decoder1_ctx, "num_logits_to_keep");
    return index < 0 ? 0 : whisper_fill_int32_scalar_tensor(
        decoder1_ctx->rknn_ctx, &decoder1_ctx->inputs[index], num_logits);
  }

  struct ScalarInput
  {
    int32_t value;
  } scalar_inputs[] = {
      {th},
      {tc},
      {ts},
      {tsr},
      {num_logits},
  };

  for (uint32_t i = 0; i < sizeof(scalar_inputs) / sizeof(scalar_inputs[0]); ++i)
  {
    int index = runtime ? runtime->scalar_indices[i] : whisper_find_input_index_by_name(decoder1_ctx, DECODER1_SCALAR_INPUT_NAMES[i]);
    if (index < 0)
    {
      printf("decoder1 scalar input missing: name=%s shape_mode=%s shape_id=%u th=%d tc=%d ts=%d tsr=%d num_logits_to_keep=%d\n",
             DECODER1_SCALAR_INPUT_NAMES[i],
             runtime ? decoder1_shape_mode_name(runtime->shape_mode) : "unknown",
             runtime && runtime->has_current_shape ? runtime->current_shape_id : 0,
             th, tc, ts, tsr, num_logits);
      return -1;
    }
    if (runtime && runtime->scalar_valid[i] && runtime->scalar_values[i] == scalar_inputs[i].value)
    {
      continue;
    }
    int ret = whisper_fill_int32_scalar_tensor(decoder1_ctx->rknn_ctx, &decoder1_ctx->inputs[index], scalar_inputs[i].value);
    if (ret != 0)
    {
      printf("fill raw scalar input failed: name=%s index=%d ret=%d shape_mode=%s shape_id=%u th=%d tc=%d ts=%d tsr=%d num_logits_to_keep=%d\n",
             DECODER1_SCALAR_INPUT_NAMES[i],
             index,
             ret,
             runtime ? decoder1_shape_mode_name(runtime->shape_mode) : "unknown",
             runtime && runtime->has_current_shape ? runtime->current_shape_id : 0,
             th, tc, ts, tsr, num_logits);
      return ret;
    }
    if (runtime)
    {
      runtime->scalar_valid[i] = true;
      runtime->scalar_values[i] = scalar_inputs[i].value;
    }
  }
  return 0;
}

// ============================================================================
// Decoder1 输入数据填充函数
// ============================================================================

// 根据 token ID 序列查 token_embed 表，生成 inputs_embeds tensor 数据
// 支持 FP16 和 FP32 两种输出格式，写入后同步到 NPU 设备
static int fill_decoder1_embed_tensor(rknn3_context ctx, rknn3_tensor *tensor, const int32_t *tokens, uint32_t n_tokens,
                                      const float16 *token_embed, uint32_t hidden_size, Decoder1RawRuntime *runtime)
{
  (void)runtime;
  if (!tensor || !tensor->mem || !tensor->attr || !tokens || !token_embed)
  {
    printf("fill_decoder1_embed_tensor invalid args tensor=%p mem=%p attr=%p tokens=%p token_embed=%p n_tokens=%u\n",
           tensor, tensor ? tensor->mem : NULL, tensor ? tensor->attr : NULL, tokens, token_embed, n_tokens);
    return -1;
  }

  uint32_t max_tokens = tensor->attr->n_dims >= 2 ? tensor->attr->shape[1] : n_tokens;
  if (n_tokens > max_tokens)
  {
    printf("fill_decoder1_embed_tensor n_tokens exceed capacity tensor=%s n_tokens=%u capacity=%u\n",
           tensor->attr->name, n_tokens, max_tokens);
    return -1;
  }

  memset(tensor->mem->virt_addr, 0, tensor->mem->size);
  if (n_tokens == 1 &&
      tensor->attr->dtype == RKNN3_TENSOR_FLOAT16 &&
      tensor->attr->n_dims == 3 &&
      tensor->attr->stride[2] == 1)
  {
    memcpy(tensor->mem->virt_addr,
           token_embed + (size_t)tokens[0] * hidden_size,
           hidden_size * sizeof(float16));
    int ret = rknn3_mem_sync(ctx, tensor->mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN3_SUCCESS)
    {
      printf("fill_decoder1_embed_tensor mem_sync failed tensor=%s ret=%d n_tokens=%u hidden_size=%u\n",
             tensor->attr->name, ret, n_tokens, hidden_size);
    }
    return ret;
  }

  if (tensor->attr->dtype == RKNN3_TENSOR_FLOAT16)
  {
    float16 *dst = reinterpret_cast<float16 *>(tensor->mem->virt_addr);
    for (uint32_t t = 0; t < n_tokens; ++t)
    {
      const float16 *src = token_embed + (size_t)tokens[t] * hidden_size;
      for (uint32_t d = 0; d < hidden_size; ++d)
      {
        dst[whisper_tensor_offset_2d(tensor->attr, t, d, hidden_size)] = src[d];
      }
    }
  }
  else if (tensor->attr->dtype == RKNN3_TENSOR_FLOAT32)
  {
    float *dst = reinterpret_cast<float *>(tensor->mem->virt_addr);
    for (uint32_t t = 0; t < n_tokens; ++t)
    {
      const float16 *src = token_embed + (size_t)tokens[t] * hidden_size;
      for (uint32_t d = 0; d < hidden_size; ++d)
      {
        dst[whisper_tensor_offset_2d(tensor->attr, t, d, hidden_size)] = fp16_to_fp32(src[d]);
      }
    }
  }
  else
  {
    printf("fill_decoder1_embed_tensor unsupported dtype tensor=%s dtype=%s n_tokens=%u hidden_size=%u\n",
           tensor->attr->name, rknn3_get_type_string(tensor->attr->dtype), n_tokens, hidden_size);
    return -1;
  }

  int ret = rknn3_mem_sync(ctx, tensor->mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
  if (ret != RKNN3_SUCCESS)
  {
    printf("fill_decoder1_embed_tensor mem_sync failed tensor=%s ret=%d n_tokens=%u hidden_size=%u\n",
           tensor->attr->name, ret, n_tokens, hidden_size);
  }
  return ret;
}

// 填充 position_ids tensor：从 start_position 开始递增写入 n_tokens 个位置编号
// 支持 INT32 和 INT64 两种数据类型
static int fill_decoder1_position_ids(rknn3_context ctx, rknn3_tensor *tensor, uint32_t n_tokens, uint32_t start_position)
{
  if (!tensor || !tensor->mem || !tensor->attr)
  {
    printf("fill_decoder1_position_ids invalid tensor=%p mem=%p attr=%p n_tokens=%u start_position=%u\n",
           tensor, tensor ? tensor->mem : NULL, tensor ? tensor->attr : NULL, n_tokens, start_position);
    return -1;
  }
  uint32_t max_tokens = tensor->attr->n_dims >= 2 ? tensor->attr->shape[1] : n_tokens;
  if (n_tokens > max_tokens)
  {
    printf("fill_decoder1_position_ids n_tokens exceed capacity tensor=%s n_tokens=%u capacity=%u start_position=%u\n",
           tensor->attr->name, n_tokens, max_tokens, start_position);
    return -1;
  }
  memset(tensor->mem->virt_addr, 0, tensor->mem->size);
  uint32_t count = n_tokens;
  if (tensor->attr->dtype == RKNN3_TENSOR_INT64)
  {
    int64_t *dst = reinterpret_cast<int64_t *>(tensor->mem->virt_addr);
    for (uint32_t i = 0; i < count; ++i)
    {
      dst[whisper_tensor_offset_2d_int(tensor->attr, 0, i, max_tokens)] = (int64_t)(start_position + i);
    }
  }
  else if (tensor->attr->dtype == RKNN3_TENSOR_INT32)
  {
    int32_t *dst = reinterpret_cast<int32_t *>(tensor->mem->virt_addr);
    for (uint32_t i = 0; i < count; ++i)
    {
      dst[whisper_tensor_offset_2d_int(tensor->attr, 0, i, max_tokens)] = (int32_t)(start_position + i);
    }
  }
  else
  {
    printf("fill_decoder1_position_ids unsupported dtype tensor=%s dtype=%s n_tokens=%u start_position=%u\n",
           tensor->attr->name, rknn3_get_type_string(tensor->attr->dtype), n_tokens, start_position);
    return -1;
  }
  int ret = rknn3_mem_sync(ctx, tensor->mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
  if (ret != RKNN3_SUCCESS)
  {
    printf("fill_decoder1_position_ids mem_sync failed tensor=%s ret=%d n_tokens=%u start_position=%u\n",
           tensor->attr->name, ret, n_tokens, start_position);
  }
  return ret;
}
static int fill_decoder1_raw_attention_mask(rknn_app_context_t *decoder1_ctx, Decoder1RawRuntime *runtime,
                                            uint32_t shape_id, uint32_t n_tokens, int mask_index)
{
  if (!decoder1_ctx || mask_index < 0)
  {
    printf("fill_decoder1_raw_attention_mask invalid args mask_index=%d shape_id=%u n_tokens=%u shape_mode=%s\n",
           mask_index, shape_id, n_tokens, runtime ? decoder1_shape_mode_name(runtime->shape_mode) : "unknown");
    return -1;
  }
  bool cache_shape1_mask = runtime &&
                           shape_id == runtime->decode_shape_id &&
                           n_tokens == 1;
  if (cache_shape1_mask && runtime->attention_mask_valid &&
      runtime->attention_mask_shape_id == shape_id &&
      runtime->attention_mask_n_tokens == n_tokens)
  {
    return 0;
  }

  int ret = whisper_fill_causal_attention_mask_tensor(decoder1_ctx->rknn_ctx, &decoder1_ctx->inputs[mask_index]);
  if (ret != 0)
  {
    printf("fill_decoder1_raw_attention_mask failed ret=%d mask_index=%d name=%s shape_id=%u n_tokens=%u shape_mode=%s\n",
           ret,
           mask_index,
           decoder1_ctx->inputs[mask_index].attr ? decoder1_ctx->inputs[mask_index].attr->name : "<null>",
           shape_id,
           n_tokens,
           runtime ? decoder1_shape_mode_name(runtime->shape_mode) : "unknown");
    return ret;
  }
  if (cache_shape1_mask)
  {
    runtime->attention_mask_valid = true;
    runtime->attention_mask_shape_id = shape_id;
    runtime->attention_mask_n_tokens = n_tokens;
  }
  return 0;
}

// 填充 decoder1 的静态输入：cross_key/cross_value 从 decode0 复制，rope 缓存填零
static int fill_decoder1_raw_static_inputs(rknn_app_context_t *decoder1_ctx, rknn_app_context_t *decode0_ctx,
                                           Decoder1RawRuntime *runtime, uint32_t shape_id)
{
  if (!decoder1_ctx || !decode0_ctx)
  {
    printf("fill_decoder1_raw_static_inputs invalid args decoder1_ctx=%p decode0_ctx=%p shape_id=%u shape_mode=%s\n",
           decoder1_ctx, decode0_ctx, shape_id, runtime ? decoder1_shape_mode_name(runtime->shape_mode) : "unknown");
    return -1;
  }
  if (runtime && runtime->static_inputs_valid && runtime->static_inputs_shape_id == shape_id)
  {
    return 0;
  }

  for (uint32_t i = 0; i < decoder1_ctx->io_num.n_input; ++i)
  {
    if (!decoder1_ctx->inputs[i].attr || !decoder1_ctx->inputs[i].attr->name)
    {
      printf("fill_decoder1_raw_static_inputs invalid input attr index=%u shape_id=%u shape_mode=%s\n",
             i, shape_id, runtime ? decoder1_shape_mode_name(runtime->shape_mode) : "unknown");
      return -1;
    }
    const char *name = decoder1_ctx->inputs[i].attr->name;
    if (starts_with(name, "cross_key_") || starts_with(name, "cross_value_"))
    {
      int out_index = whisper_find_output_index_by_name(decode0_ctx, name);
      if (out_index < 0)
      {
        printf("fill_decoder1_raw_static_inputs missing decode0 output name=%s shape_id=%u shape_mode=%s\n",
               name,
               shape_id,
               runtime ? decoder1_shape_mode_name(runtime->shape_mode) : "unknown");
        return -1;
      }
      int ret = whisper_copy_tensor_data(decoder1_ctx->rknn_ctx, &decoder1_ctx->inputs[i], &decode0_ctx->outputs[out_index], false);
      if (ret != 0)
      {
        printf("fill_decoder1_raw_static_inputs copy failed name=%s out_index=%d ret=%d shape_id=%u shape_mode=%s\n",
               name,
               out_index,
               ret,
               shape_id,
               runtime ? decoder1_shape_mode_name(runtime->shape_mode) : "unknown");
        return ret;
      }
      if (!decoder1_ctx->inputs[i].mem)
      {
        printf("fill_decoder1_raw_static_inputs missing input mem name=%s shape_id=%u shape_mode=%s\n",
               name, shape_id, runtime ? decoder1_shape_mode_name(runtime->shape_mode) : "unknown");
        return -1;
      }
      ret = rknn3_mem_sync(decoder1_ctx->rknn_ctx, decoder1_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
      if (ret != RKNN3_SUCCESS)
      {
        printf("fill_decoder1_raw_static_inputs mem_sync failed name=%s ret=%d shape_id=%u shape_mode=%s mem_size=%lu\n",
               name,
               ret,
               shape_id,
               runtime ? decoder1_shape_mode_name(runtime->shape_mode) : "unknown",
               decoder1_ctx->inputs[i].mem ? (unsigned long)decoder1_ctx->inputs[i].mem->size : 0UL);
        return ret;
      }
      continue;
    }

    if (strcmp(name, "rope_cos_cache") == 0 || strcmp(name, "rope_sin_cache") == 0)
    {
      int ret = whisper_fill_zero_tensor(decoder1_ctx->rknn_ctx, &decoder1_ctx->inputs[i], false);
      if (ret != 0)
      {
        printf("fill_decoder1_raw_static_inputs fill zero failed name=%s ret=%d shape_id=%u shape_mode=%s\n",
               name, ret, shape_id, runtime ? decoder1_shape_mode_name(runtime->shape_mode) : "unknown");
        return ret;
      }
      if (!decoder1_ctx->inputs[i].mem)
      {
        printf("fill_decoder1_raw_static_inputs missing rope cache mem name=%s shape_id=%u shape_mode=%s\n",
               name, shape_id, runtime ? decoder1_shape_mode_name(runtime->shape_mode) : "unknown");
        return -1;
      }
      ret = rknn3_mem_sync(decoder1_ctx->rknn_ctx, decoder1_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
      if (ret != RKNN3_SUCCESS)
      {
        printf("fill_decoder1_raw_static_inputs mem_sync failed name=%s ret=%d shape_id=%u shape_mode=%s mem_size=%lu\n",
               name,
               ret,
               shape_id,
               runtime ? decoder1_shape_mode_name(runtime->shape_mode) : "unknown",
               decoder1_ctx->inputs[i].mem ? (unsigned long)decoder1_ctx->inputs[i].mem->size : 0UL);
        return ret;
      }
    }
  }

  if (runtime)
  {
    runtime->static_inputs_valid = true;
    runtime->static_inputs_shape_id = shape_id;
  }
  return 0;
}

// 将 decoder1 的 cross_key/cross_value 和 rope 缓存全部填零
// ============================================================================
// Decoder1 推理函数
// ============================================================================

// 输入: SOT token 的 embedding + attention_mask=1 + position_ids=0
// 从 logits 输出 tensor 中贪心选取分数最高的 token（贪心解码）
// 屏蔽特殊 token：SOT~TIMESTAMP_BEGIN 范围和时间戳 token
// 返回选中的 token ID，失败返回 -1
static int select_next_token_from_tensor(rknn3_tensor *tensor, uint32_t vocab_size, uint32_t seq_index, Decoder1RawRuntime *runtime)
{
  (void)runtime;
  if (!tensor || !tensor->mem || !tensor->attr)
  {
    printf("select_next_token_from_tensor invalid tensor=%p mem=%p attr=%p vocab_size=%u seq_index=%u\n",
           tensor, tensor ? tensor->mem : NULL, tensor ? tensor->attr : NULL, vocab_size, seq_index);
    return -1;
  }
  int max_id = -1;
  float max_value = 0.0f;
  uint32_t scan_vocab = std::min<uint32_t>(vocab_size, (uint32_t)whisper_tokens::SOT);
  if (scan_vocab == 0)
  {
    printf("select_next_token_from_tensor empty vocab tensor=%s vocab_size=%u seq_index=%u\n",
           tensor->attr->name, vocab_size, seq_index);
    whisper_print_tensor_shape_brief("  logits", tensor->attr);
    return -1;
  }

  if (tensor->attr->n_dims == 3 && tensor->attr->stride[2] == 1)
  {
    uint32_t seq = tensor->attr->shape[1] > 0 ? tensor->attr->shape[1] : 1;
    uint32_t row = seq > 1 ? std::min(seq_index, seq - 1) : 0;
    uint32_t scan_limit = tensor->attr->shape[2] > 0 ? std::min(scan_vocab, tensor->attr->shape[2]) : scan_vocab;
    if (tensor->attr->dtype == RKNN3_TENSOR_FLOAT16)
    {
      const float16 *row_ptr = reinterpret_cast<const float16 *>(tensor->mem->virt_addr) + row * tensor->attr->stride[1];
      for (uint32_t i = 0; i < scan_limit; ++i)
      {
        float value = fp16_to_fp32(row_ptr[i]);
        if (max_id < 0 || value > max_value)
        {
          max_value = value;
          max_id = (int)i;
        }
      }
    }
    else if (tensor->attr->dtype == RKNN3_TENSOR_FLOAT32)
    {
      const float *row_ptr = reinterpret_cast<const float *>(tensor->mem->virt_addr) + row * tensor->attr->stride[1];
      for (uint32_t i = 0; i < scan_limit; ++i)
      {
        float value = row_ptr[i];
        if (max_id < 0 || value > max_value)
        {
          max_value = value;
          max_id = (int)i;
        }
      }
    }
    else
    {
      printf("select_next_token_from_tensor unsupported dtype tensor=%s dtype=%s vocab_size=%u scan_vocab=%u seq_index=%u\n",
             tensor->attr->name,
             rknn3_get_type_string(tensor->attr->dtype),
             vocab_size,
             scan_vocab,
             seq_index);
      whisper_print_tensor_shape_brief("  logits", tensor->attr);
      return -1;
    }
    if (max_id < 0)
    {
      printf("select_next_token_from_tensor no token selected tensor=%s vocab_size=%u scan_vocab=%u seq_index=%u\n",
             tensor->attr->name, vocab_size, scan_vocab, seq_index);
      whisper_print_tensor_shape_brief("  logits", tensor->attr);
      return -1;
    }
    return max_id;
  }

  if (tensor->attr->n_dims == 5 && tensor->attr->shape[4] > 0)
  {
    uint32_t seq = tensor->attr->shape[2] > 0 ? tensor->attr->shape[2] : 1;
    uint32_t row = seq > 1 ? std::min(seq_index, seq - 1) : 0;
    uint32_t c2 = tensor->attr->shape[4];
    uint32_t scan_limit = std::min(scan_vocab, tensor->attr->shape[1] * c2);
    uint32_t token_id = 0;
    if (tensor->attr->dtype == RKNN3_TENSOR_FLOAT16)
    {
      const float16 *data = reinterpret_cast<const float16 *>(tensor->mem->virt_addr);
      for (uint32_t c1 = 0; token_id < scan_limit; ++c1)
      {
        uint32_t lane_count = std::min(c2, scan_limit - token_id);
        uint32_t base = c1 * tensor->attr->stride[1] + row * tensor->attr->stride[2];
        for (uint32_t lane = 0; lane < lane_count; ++lane, ++token_id)
        {
          float value = fp16_to_fp32(data[base + lane * tensor->attr->stride[4]]);
          if (max_id < 0 || value > max_value)
          {
            max_value = value;
            max_id = (int)token_id;
          }
        }
      }
    }
    else if (tensor->attr->dtype == RKNN3_TENSOR_FLOAT32)
    {
      const float *data = reinterpret_cast<const float *>(tensor->mem->virt_addr);
      for (uint32_t c1 = 0; token_id < scan_limit; ++c1)
      {
        uint32_t lane_count = std::min(c2, scan_limit - token_id);
        uint32_t base = c1 * tensor->attr->stride[1] + row * tensor->attr->stride[2];
        for (uint32_t lane = 0; lane < lane_count; ++lane, ++token_id)
        {
          float value = data[base + lane * tensor->attr->stride[4]];
          if (max_id < 0 || value > max_value)
          {
            max_value = value;
            max_id = (int)token_id;
          }
        }
      }
    }
    else
    {
      printf("select_next_token_from_tensor unsupported dtype tensor=%s dtype=%s vocab_size=%u scan_vocab=%u seq_index=%u\n",
             tensor->attr->name,
             rknn3_get_type_string(tensor->attr->dtype),
             vocab_size,
             scan_vocab,
             seq_index);
      whisper_print_tensor_shape_brief("  logits", tensor->attr);
      return -1;
    }
    if (max_id < 0)
    {
      printf("select_next_token_from_tensor failed to select token tensor=%s vocab_size=%u scan_vocab=%u seq_index=%u\n",
             tensor->attr->name, vocab_size, scan_vocab, seq_index);
      whisper_print_tensor_shape_brief("  logits", tensor->attr);
      return -1;
    }
    return max_id;
  }

  for (uint32_t i = 0; i < scan_vocab; ++i)
  {
    float value = 0.0f;
    if (tensor->attr->dtype == RKNN3_TENSOR_FLOAT16)
    {
      value = fp16_to_fp32(reinterpret_cast<float16 *>(tensor->mem->virt_addr)[whisper_tensor_offset_logits(tensor->attr, i, seq_index)]);
    }
    else if (tensor->attr->dtype == RKNN3_TENSOR_FLOAT32)
    {
      value = reinterpret_cast<float *>(tensor->mem->virt_addr)[whisper_tensor_offset_logits(tensor->attr, i, seq_index)];
    }
    else
    {
      printf("select_next_token_from_tensor unsupported dtype tensor=%s dtype=%s vocab_size=%u scan_vocab=%u seq_index=%u\n",
             tensor->attr->name,
             rknn3_get_type_string(tensor->attr->dtype),
             vocab_size,
             scan_vocab,
             seq_index);
      whisper_print_tensor_shape_brief("  logits", tensor->attr);
      return -1;
    }
    if (max_id < 0 || value > max_value)
    {
      max_value = value;
      max_id = (int)i;
    }
  }
  (void)runtime;
  if (max_id < 0)
  {
    printf("select_next_token_from_tensor no token selected tensor=%s vocab_size=%u scan_vocab=%u seq_index=%u\n",
           tensor->attr->name, vocab_size, scan_vocab, seq_index);
    whisper_print_tensor_shape_brief("  logits", tensor->attr);
    return -1;
  }
  return max_id;
}


// 根据 n_tokens（当前处理的 token 数）选择最合适的 decoder1 shape：
//   - prefill shape: 序列最长，用于处理多个 prompt token
//   - decode shape: 序列=1，用于逐个生成 token
static int choose_decoder1_shape_id(Decoder1RawRuntime *raw_runtime, uint32_t n_tokens, uint32_t start_position, uint32_t *shape_id)
{
  (void)start_position;
  if (!shape_id || n_tokens == 0)
  {
    printf("choose_decoder1_shape_id invalid args shape_id=%p n_tokens=%u start_position=%u shape_mode=%s\n",
           shape_id, n_tokens, start_position, raw_runtime ? decoder1_shape_mode_name(raw_runtime->shape_mode) : "unknown");
    return -1;
  }

  if (!raw_runtime || raw_runtime->shapes.empty())
  {
    *shape_id = 0;
    return 0;
  }

  uint32_t selected_shape_id = raw_runtime->prefill_shape_id;
  switch (raw_runtime->shape_mode)
  {
  case DECODER1_SHAPE0_ONLY:
    selected_shape_id = raw_runtime->prefill_shape_id;
    break;
  case DECODER1_SHAPE1_ONLY:
    selected_shape_id = raw_runtime->decode_shape_id;
    break;
  case DECODER1_SHAPE_AUTO:
  default:
    selected_shape_id = (n_tokens == 1) ? raw_runtime->decode_shape_id : raw_runtime->prefill_shape_id;
    break;
  }

  if (selected_shape_id >= raw_runtime->shapes.size())
  {
    printf("choose_decoder1_shape_id invalid selected_shape_id=%u n_shapes=%zu shape_mode=%s n_tokens=%u start_position=%u\n",
           selected_shape_id,
           raw_runtime->shapes.size(),
           decoder1_shape_mode_name(raw_runtime->shape_mode),
           n_tokens,
           start_position);
    return -1;
  }

  uint32_t seq_len = whisper_decoder_embed_seq_len(raw_runtime->shapes[selected_shape_id].inputs);
  if (seq_len > 0 && n_tokens > seq_len)
  {
    printf("choose_decoder1_shape_id capacity exceeded shape_id=%u n_tokens=%u capacity=%u shape_mode=%s start_position=%u\n",
           selected_shape_id,
           n_tokens,
           seq_len,
           decoder1_shape_mode_name(raw_runtime->shape_mode),
           start_position);
    return -1;
  }

  *shape_id = selected_shape_id;
  return 0;
}

// Raw 模式核心函数：用 decoder1 推理 n_tokens 个 token
// 流程: 选择 shape → 切换 shape → 填充 inputs_embeds/attention_mask/position_ids/标量参数/静态输入
//       → rknn3_run → 同步输出 → 贪心选取 next_token
// 参数:
//   decoder1_ctx, decode0_ctx: 模型上下文
//   raw_runtime: 动态 shape 信息
//   tokens: 输入的 token ID 数组
//   n_tokens: token 数量
//   start_position: 起始位置编号（用于 position_ids 和 Th）
//   token_embed: token embedding 查表
//   hidden_size: 隐藏层维度
//   num_logits_to_keep: 需要保留的 logits 数（prefill 阶段通常保留最后一个，decode 阶段保留全部=0）
//   next_token: [输出] 推理结果的下一个 token ID
static int run_decoder1_raw_tokens(rknn_app_context_t *decoder1_ctx, rknn_app_context_t *decode0_ctx, Decoder1RawRuntime *raw_runtime,
                                   const int32_t *tokens, uint32_t n_tokens, uint32_t start_position,
                                   const float16 *token_embed, uint32_t hidden_size, int32_t num_logits_to_keep, int *next_token, uint32_t vocab_size)
{
  if (!decoder1_ctx || !decode0_ctx || !tokens || n_tokens == 0 || !next_token || !token_embed || hidden_size == 0)
  {
    printf("run_decoder1_raw_tokens invalid args decoder1_ctx=%p decode0_ctx=%p tokens=%p n_tokens=%u next_token=%p token_embed=%p hidden_size=%u\n",
           decoder1_ctx, decode0_ctx, tokens, n_tokens, next_token, token_embed, hidden_size);
    return -1;
  }

  uint32_t shape_id = 0;
  int ret = choose_decoder1_shape_id(raw_runtime, n_tokens, start_position, &shape_id);
  if (ret != 0)
  {
    printf("run_decoder1_raw_tokens choose shape failed ret=%d shape_mode=%s n_tokens=%u start_position=%u num_logits_to_keep=%d token0=%d\n",
           ret,
           raw_runtime ? decoder1_shape_mode_name(raw_runtime->shape_mode) : "unknown",
           n_tokens,
           start_position,
           num_logits_to_keep,
           tokens[0]);
    return ret;
  }

  ret = switch_decoder1_raw_shape(decoder1_ctx, raw_runtime, shape_id);
  if (ret != 0)
  {
    printf("run_decoder1_raw_tokens switch shape failed ret=%d shape_id=%u shape_mode=%s n_tokens=%u start_position=%u num_logits_to_keep=%d token0=%d\n",
           ret,
           shape_id,
           raw_runtime ? decoder1_shape_mode_name(raw_runtime->shape_mode) : "unknown",
           n_tokens,
           start_position,
           num_logits_to_keep,
           tokens[0]);
    return ret;
  }

  int embed_index = raw_runtime ? raw_runtime->input_embed_index : whisper_find_input_index_by_name(decoder1_ctx, "input_embeds");
  if (!raw_runtime && embed_index < 0)
  {
    embed_index = whisper_find_input_index_by_name(decoder1_ctx, "inputs_embeds");
  }
  int mask_index = raw_runtime ? raw_runtime->attention_mask_index : whisper_find_input_index_by_name(decoder1_ctx, "attention_mask");
  int pos_index = raw_runtime ? raw_runtime->position_ids_index : whisper_find_input_index_by_name(decoder1_ctx, "position_ids");
  if (embed_index < 0 || mask_index < 0 || pos_index < 0)
  {
    printf("run_decoder1_raw_tokens required inputs missing ret=-1 shape_id=%u shape_mode=%s n_tokens=%u start_position=%u num_logits_to_keep=%d token0=%d embed=%d mask=%d pos=%d\n",
           shape_id,
           raw_runtime ? decoder1_shape_mode_name(raw_runtime->shape_mode) : "unknown",
           n_tokens,
           start_position,
           num_logits_to_keep,
           tokens[0],
           embed_index,
           mask_index,
           pos_index);
    return -1;
  }

  ret = fill_decoder1_embed_tensor(decoder1_ctx->rknn_ctx, &decoder1_ctx->inputs[embed_index], tokens, n_tokens, token_embed, hidden_size, raw_runtime);
  if (ret != 0)
  {
    printf("run_decoder1_raw_tokens fill input_embeds failed ret=%d shape_id=%u shape_mode=%s n_tokens=%u start_position=%u num_logits_to_keep=%d token0=%d input_index=%d name=%s mem_size=%lu aligned_size=%lu\n",
           ret,
           shape_id,
           raw_runtime ? decoder1_shape_mode_name(raw_runtime->shape_mode) : "unknown",
           n_tokens,
           start_position,
           num_logits_to_keep,
           tokens[0],
           embed_index,
           decoder1_ctx->inputs[embed_index].attr ? decoder1_ctx->inputs[embed_index].attr->name : "<null>",
           decoder1_ctx->inputs[embed_index].mem ? (unsigned long)decoder1_ctx->inputs[embed_index].mem->size : 0UL,
           decoder1_ctx->inputs[embed_index].attr ? (unsigned long)decoder1_ctx->inputs[embed_index].attr->aligned_size : 0UL);
    whisper_print_tensor_shape_brief("  input_embeds", decoder1_ctx->inputs[embed_index].attr);
    return ret;
  }

  ret = fill_decoder1_raw_attention_mask(decoder1_ctx, raw_runtime, shape_id, n_tokens, mask_index);
  if (ret != 0)
  {
    printf("run_decoder1_raw_tokens fill attention_mask failed ret=%d shape_id=%u shape_mode=%s n_tokens=%u start_position=%u num_logits_to_keep=%d token0=%d input_index=%d name=%s mem_size=%lu aligned_size=%lu\n",
           ret,
           shape_id,
           raw_runtime ? decoder1_shape_mode_name(raw_runtime->shape_mode) : "unknown",
           n_tokens,
           start_position,
           num_logits_to_keep,
           tokens[0],
           mask_index,
           decoder1_ctx->inputs[mask_index].attr ? decoder1_ctx->inputs[mask_index].attr->name : "<null>",
           decoder1_ctx->inputs[mask_index].mem ? (unsigned long)decoder1_ctx->inputs[mask_index].mem->size : 0UL,
           decoder1_ctx->inputs[mask_index].attr ? (unsigned long)decoder1_ctx->inputs[mask_index].attr->aligned_size : 0UL);
    whisper_print_tensor_shape_brief("  attention_mask", decoder1_ctx->inputs[mask_index].attr);
    return ret;
  }

  ret = fill_decoder1_position_ids(decoder1_ctx->rknn_ctx, &decoder1_ctx->inputs[pos_index], n_tokens, start_position);
  if (ret != 0)
  {
    printf("run_decoder1_raw_tokens fill position_ids failed ret=%d shape_id=%u shape_mode=%s n_tokens=%u start_position=%u num_logits_to_keep=%d token0=%d input_index=%d name=%s mem_size=%lu aligned_size=%lu\n",
           ret,
           shape_id,
           raw_runtime ? decoder1_shape_mode_name(raw_runtime->shape_mode) : "unknown",
           n_tokens,
           start_position,
           num_logits_to_keep,
           tokens[0],
           pos_index,
           decoder1_ctx->inputs[pos_index].attr ? decoder1_ctx->inputs[pos_index].attr->name : "<null>",
           decoder1_ctx->inputs[pos_index].mem ? (unsigned long)decoder1_ctx->inputs[pos_index].mem->size : 0UL,
           decoder1_ctx->inputs[pos_index].attr ? (unsigned long)decoder1_ctx->inputs[pos_index].attr->aligned_size : 0UL);
    whisper_print_tensor_shape_brief("  position_ids", decoder1_ctx->inputs[pos_index].attr);
    return ret;
  }

  int32_t total_seq = (int32_t)start_position + (int32_t)n_tokens;
  ret = fill_decoder1_raw_scalar_inputs(decoder1_ctx, raw_runtime, (int32_t)start_position, (int32_t)n_tokens,
                                        total_seq, total_seq, num_logits_to_keep);
  if (ret != 0)
  {
    printf("run_decoder1_raw_tokens fill scalar inputs failed ret=%d shape_id=%u shape_mode=%s n_tokens=%u start_position=%u num_logits_to_keep=%d token0=%d\n",
           ret,
           shape_id,
           raw_runtime ? decoder1_shape_mode_name(raw_runtime->shape_mode) : "unknown",
           n_tokens,
           start_position,
           num_logits_to_keep,
           tokens[0]);
    return ret;
  }

  ret = fill_decoder1_raw_static_inputs(decoder1_ctx, decode0_ctx, raw_runtime, shape_id);
  if (ret != 0)
  {
    printf("run_decoder1_raw_tokens fill static inputs failed ret=%d shape_id=%u shape_mode=%s n_tokens=%u start_position=%u num_logits_to_keep=%d token0=%d\n",
           ret,
           shape_id,
           raw_runtime ? decoder1_shape_mode_name(raw_runtime->shape_mode) : "unknown",
           n_tokens,
           start_position,
           num_logits_to_keep,
           tokens[0]);
    return ret;
  }

  ret = rknn3_run(decoder1_ctx->rknn_ctx, decoder1_ctx->inputs, decoder1_ctx->io_num.n_input,
                  decoder1_ctx->outputs, decoder1_ctx->io_num.n_output);
  if (ret != RKNN3_SUCCESS)
  {
    printf("run_decoder1_raw_tokens rknn3_run failed ret=%d shape_id=%u shape_mode=%s n_tokens=%u start_position=%u num_logits_to_keep=%d token0=%d\n",
           ret,
           shape_id,
           raw_runtime ? decoder1_shape_mode_name(raw_runtime->shape_mode) : "unknown",
           n_tokens,
           start_position,
           num_logits_to_keep,
           tokens[0]);
    return ret;
  }

  ret = rknn3_mem_sync(decoder1_ctx->rknn_ctx, decoder1_ctx->outputs[0].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
  if (ret != RKNN3_SUCCESS)
  {
    printf("run_decoder1_raw_tokens logits sync failed ret=%d shape_id=%u shape_mode=%s n_tokens=%u start_position=%u num_logits_to_keep=%d token0=%d output_index=0 name=%s mem_size=%lu aligned_size=%lu\n",
           ret,
           shape_id,
           raw_runtime ? decoder1_shape_mode_name(raw_runtime->shape_mode) : "unknown",
           n_tokens,
           start_position,
           num_logits_to_keep,
           tokens[0],
           decoder1_ctx->outputs[0].attr ? decoder1_ctx->outputs[0].attr->name : "<null>",
           decoder1_ctx->outputs[0].mem ? (unsigned long)decoder1_ctx->outputs[0].mem->size : 0UL,
           decoder1_ctx->outputs[0].attr ? (unsigned long)decoder1_ctx->outputs[0].attr->aligned_size : 0UL);
    whisper_print_tensor_shape_brief("  logits", decoder1_ctx->outputs[0].attr);
    return ret;
  }

  uint32_t logits_seq_index = n_tokens > 0 ? n_tokens - 1 : 0;
  *next_token = select_next_token_from_tensor(&decoder1_ctx->outputs[0], vocab_size, logits_seq_index, raw_runtime);
  if (*next_token < 0)
  {
    printf("run_decoder1_raw_tokens select_next_token_from_tensor failed ret=-1 shape_id=%u shape_mode=%s n_tokens=%u start_position=%u num_logits_to_keep=%d token0=%d output_index=0 name=%s\n",
           shape_id,
           raw_runtime ? decoder1_shape_mode_name(raw_runtime->shape_mode) : "unknown",
           n_tokens,
           start_position,
           num_logits_to_keep,
           tokens[0],
           decoder1_ctx->outputs[0].attr ? decoder1_ctx->outputs[0].attr->name : "<null>");
    whisper_print_tensor_shape_brief("  logits", decoder1_ctx->outputs[0].attr);
    return -1;
  }
  return 0;
}

// Prefill 分块执行：将 prompt tokens 按 shape 容量分块，逐块推理
// num_logits_to_keep 与 python/test.py 保持一致，传当前 seq_len/chunk，而不是最后一个 token 下标
static int run_decoder1_raw_prefill_chunks(rknn_app_context_t *decoder1_ctx, rknn_app_context_t *decode0_ctx, Decoder1RawRuntime *raw_runtime,
                                           const int32_t *tokens, uint32_t total_tokens, uint32_t start_position,
                                           const float16 *token_embed, uint32_t hidden_size, int *next_token, uint32_t vocab_size)
{
  if (!tokens || total_tokens == 0 || !next_token)
  {
    printf("run_decoder1_raw_prefill_chunks invalid args tokens=%p total_tokens=%u next_token=%p start_position=%u\n",
           tokens, total_tokens, next_token, start_position);
    return -1;
  }

  uint32_t chunk_capacity = total_tokens;
  if (raw_runtime && !raw_runtime->shapes.empty())
  {
    uint32_t shape_id = raw_runtime->prefill_shape_id;
    if (raw_runtime->shape_mode == DECODER1_SHAPE1_ONLY)
    {
      shape_id = raw_runtime->decode_shape_id;
    }
    chunk_capacity = whisper_decoder_embed_seq_len(raw_runtime->shapes[shape_id].inputs);
    if (chunk_capacity == 0)
    {
      chunk_capacity = total_tokens;
    }

    if (raw_runtime && raw_runtime->shape_mode == DECODER1_SHAPE_AUTO && chunk_capacity < 2 && total_tokens > 1)
    {
      chunk_capacity = total_tokens;
    }
  }

  uint32_t remaining = total_tokens;
  uint32_t offset = 0;
  uint32_t position = start_position;
  int ret = 0;

  // shape0: remaining > 1 时走 prefill shape
  while (remaining > 1)
  {
    uint32_t chunk = std::min(chunk_capacity, remaining);

    int32_t num_logits_to_keep = (int32_t)chunk;

    ret = run_decoder1_raw_tokens(decoder1_ctx, decode0_ctx, raw_runtime,
                                  tokens + offset, chunk, position,
                                  token_embed, hidden_size,
                                  num_logits_to_keep, next_token, vocab_size);
    if (ret != 0)
    {
      printf("run_decoder1_raw_prefill_chunks failed ret=%d offset=%u chunk=%u remaining=%u position=%u total_tokens=%u\n",
             ret, offset, chunk, remaining, position, total_tokens);
      return ret;
    }

    offset += chunk;
    position += chunk;
    remaining -= chunk;
  }

  // shape1: 如果最后只剩 1 个 token，用 decode shape 跑一次
  if (remaining == 1)
  {
    int32_t num_logits_to_keep = 1;

    ret = run_decoder1_raw_tokens(decoder1_ctx, decode0_ctx, raw_runtime,
                                  tokens + offset, 1, position,
                                  token_embed, hidden_size,
                                  num_logits_to_keep, next_token, vocab_size);
    if (ret != 0)
    {
      printf("run_decoder1_raw_prefill_chunks tail failed ret=%d offset=%u remaining=%u position=%u total_tokens=%u\n",
             ret, offset, remaining, position, total_tokens);
      return ret;
    }
  }
  return ret;
}

// 打印模型所有输出的摘要信息
// 根据 dtype 分别处理：UINT8/INT8 解码为字符串、INT32 解码为 token、
// FP16/FP32 打印前若干个值
// ============================================================================
// 主函数
// Whisper 三模型 ASR 完整流程:
//   1. 解析命令行参数（模型路径、任务语言、音频文件、token_embed、core_mask 等）
//   2. 加载词表、读取音频、提取 mel 频谱特征
//   3. encoder 推理 → 音频编码
//   4. decode0 推理 → 生成 cross-attention key/value
//   5. decoder1 raw 推理 → prefill prompt tokens → 自回归生成 token 序列
//   6. 将 token 序列解码为文本输出
//   7. 释放所有资源
// ============================================================================
struct WhisperAudioOptions
{
  const char *encoder_model;
  const char *encoder_weight;
  const char *decode0_model;
  const char *decode0_weight;
  const char *decoder1_model;
  const char *token_embed_path;
  const char *task;
  const WhisperModelConfig &model;
  int max_audio_length;
  int max_new_tokens;
  bool is_turbo;
};

// The caller frees audio.data, including when preprocessing or inference fails.
static int process_single_audio(const char *audio_path, audio_buffer_t &audio,
                                rknn_app_context_t &encoder_ctx, rknn_app_context_t &decode0_ctx,
                                rknn_app_context_t &decoder1_ctx, Decoder1RawRuntime &raw_runtime,
                                float *mel_filters, std::vector<VocabEntry> &vocab,
                                const float16 *token_embed, uint32_t hidden_size,
                                const WhisperAudioOptions &options)
{
  int ret = 0;
  WhisperStreamState stream_state = {};
  TIMER infer_timer;
  float infer_time = 0.0f;
  float audio_length = 0.0f;
  ret = read_audio(audio_path, &audio);
  if (ret != 0)
  {
    printf("stage=read_audio failed ret=%d audio_path=%s\n", ret, audio_path);
    return ret;
  }
  if (audio.num_channels == 2)
  {
    ret = convert_channels(&audio);
    if (ret != 0)
    {
      printf("stage=preprocess convert_channels failed ret=%d audio_path=%s channels=%d\n",
             ret, audio_path, audio.num_channels);
      return ret;
    }
  }
  if (audio.sample_rate != SAMPLE_RATE)
  {
    ret = resample_audio(&audio, audio.sample_rate, SAMPLE_RATE);
    if (ret != 0)
    {
      printf("stage=preprocess resample_audio failed ret=%d audio_path=%s src_sr=%d dst_sr=%d\n",
             ret, audio_path, audio.sample_rate, SAMPLE_RATE);
      return ret;
    }
  }
  audio_length = audio.num_frames / (float)SAMPLE_RATE;
  audio_length = std::min(audio_length, ((float)options.max_audio_length / SAMPLE_RATE));

  std::vector<float> mel_data(options.model.n_mels * options.max_audio_length / HOP_LENGTH, 0.0f);
  audio_preprocess(&audio, mel_filters, mel_data, options.model.n_mels, options.max_audio_length);

  if (encoder_ctx.io_num.n_input < 1 || encoder_ctx.io_num.n_output < 1 || decode0_ctx.io_num.n_input < 1 || decode0_ctx.io_num.n_output < 1)
  {
    printf("stage=preprocess encoder/decode0 model IO is invalid ret=-1 encoder_in=%d encoder_out=%d decode0_in=%d decode0_out=%d\n",
           encoder_ctx.io_num.n_input, encoder_ctx.io_num.n_output, decode0_ctx.io_num.n_input, decode0_ctx.io_num.n_output);
    ret = -1;
    return ret;
  }

  infer_timer.tik();

  int encoder_x_index = whisper_require_input_index(&encoder_ctx, "encoder", "x", 0);
  if (encoder_x_index < 0)
  {
    ret = -1;
    printf("stage=encoder missing required input name=x fallback_index=0 ret=%d model=%s weight=%s\n",
           ret, options.encoder_model, options.encoder_weight ? options.encoder_weight : "NULL");
    return ret;
  }

  rknn3_tensor_attr *encoder_in_attr = encoder_ctx.inputs[encoder_x_index].attr;
  if (!encoder_in_attr)
  {
    ret = -1;
    printf("stage=encoder invalid input attr model=%s input_index=%d name=x\n", options.encoder_model, encoder_x_index);
    return ret;
  }
  uint32_t encoder_input_elems = whisper_shape_count(encoder_in_attr);
  uint32_t copy_elems = (uint32_t)mel_data.size() < encoder_input_elems ? (uint32_t)mel_data.size() : encoder_input_elems;

  memset(encoder_ctx.inputs[encoder_x_index].mem->virt_addr, 0, encoder_ctx.inputs[encoder_x_index].mem->size);
  ret = whisper_convert_fp32_to_tensor(mel_data.data(), encoder_ctx.inputs[encoder_x_index].mem->virt_addr, copy_elems, encoder_in_attr->dtype);
  if (ret != 0)
  {
    printf("stage=preprocess convert_fp32_to_tensor failed ret=%d model=%s input_index=%d name=%s dtype=%s copy_elems=%u audio_path=%s\n",
           ret,
           options.encoder_model,
           encoder_x_index,
           encoder_ctx.inputs[encoder_x_index].attr ? encoder_ctx.inputs[encoder_x_index].attr->name : "<null>",
           rknn3_get_type_string(encoder_in_attr->dtype),
           copy_elems,
           audio_path);
    return ret;
  }

  ret = rknn3_mem_sync(encoder_ctx.rknn_ctx, encoder_ctx.inputs[encoder_x_index].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
  if (ret != RKNN3_SUCCESS)
  {
    printf("stage=preprocess encoder input sync failed ret=%d model=%s input_index=%d name=%s mem_size=%lu audio_path=%s\n",
           ret,
           options.encoder_model,
           encoder_x_index,
           encoder_ctx.inputs[encoder_x_index].attr ? encoder_ctx.inputs[encoder_x_index].attr->name : "<null>",
           (unsigned long)encoder_ctx.inputs[encoder_x_index].mem->size,
           audio_path);
    whisper_print_tensor_shape_brief("  encoder input", encoder_ctx.inputs[encoder_x_index].attr);
    return ret;
  }

  ret = fill_encoder_attention_mask(&encoder_ctx, &audio, options.max_audio_length, options.is_turbo);
  if (ret != 0)
  {
    printf("stage=preprocess fill_encoder_attention_mask failed ret=%d model=%s audio_path=%s\n", ret, options.encoder_model, audio_path);
    return ret;
  }

  ret = rknn3_run(encoder_ctx.rknn_ctx, encoder_ctx.inputs, encoder_ctx.io_num.n_input, encoder_ctx.outputs, encoder_ctx.io_num.n_output);
  if (ret != RKNN3_SUCCESS)
  {
    printf("stage=encoder run failed ret=%d model=%s audio_path=%s\n", ret, options.encoder_model, audio_path);
    return ret;
  }
  ret = sync_all_outputs_from_device(&encoder_ctx, "encoder");
  if (ret != 0)
  {
    printf("stage=encoder output sync failed ret=%d model=%s audio_path=%s\n", ret, options.encoder_model, audio_path);
    return ret;
  }

  int decode0_encoder_index = whisper_require_input_index(&decode0_ctx, "decode0", "encoder_outputs", 0);
  if (decode0_encoder_index < 0)
  {
    ret = -1;
    printf("stage=decode0 missing required input name=encoder_outputs fallback_index=0 ret=%d model=%s weight=%s\n",
           ret, options.decode0_model, options.decode0_weight ? options.decode0_weight : "NULL");
    return ret;
  }
  ret = whisper_copy_tensor_data(decode0_ctx.rknn_ctx, &decode0_ctx.inputs[decode0_encoder_index], &encoder_ctx.outputs[0]);
  if (ret != 0)
  {
    printf("stage=decode0 copy encoder output failed ret=%d input_index=%d name=%s src_name=%s model=%s\n",
           ret,
           decode0_encoder_index,
           decode0_ctx.inputs[decode0_encoder_index].attr ? decode0_ctx.inputs[decode0_encoder_index].attr->name : "<null>",
           encoder_ctx.outputs[0].attr ? encoder_ctx.outputs[0].attr->name : "<null>",
           options.decode0_model);
    return ret;
  }

  ret = rknn3_run(decode0_ctx.rknn_ctx, decode0_ctx.inputs, decode0_ctx.io_num.n_input, decode0_ctx.outputs, decode0_ctx.io_num.n_output);
  if (ret != RKNN3_SUCCESS)
  {
    printf("stage=decode0 run failed ret=%d model=%s audio_path=%s\n", ret, options.decode0_model, audio_path);
    return ret;
  }
  ret = sync_all_outputs_from_device(&decode0_ctx, "decode0");
  if (ret != 0)
  {
    printf("stage=decode0 output sync failed ret=%d model=%s audio_path=%s\n", ret, options.decode0_model, audio_path);
    return ret;
  }

  std::vector<int32_t> prompt_tokens = {
      whisper_tokens::SOT,
      strcmp(options.task, "en") == 0 ? whisper_tokens::TASK_EN : whisper_tokens::TASK_ZH,
      options.model.transcribe_token,
      options.model.no_timestamps_token,
  };

  int next_token = -1;
  printf("\n---- whisper raw decode ----\n");
  ret = run_decoder1_raw_prefill_chunks(&decoder1_ctx, &decode0_ctx, &raw_runtime, prompt_tokens.data(), (uint32_t)prompt_tokens.size(), 0, token_embed, hidden_size, &next_token, options.model.vocab_size);
  if (ret != 0)
  {
    printf("stage=decoder1 prefill failed ret=%d model=%s audio_path=%s token_embed_path=%s shape_mode=%s n_tokens=%zu start_position=0\n",
           ret,
           options.decoder1_model,
           audio_path,
           options.token_embed_path,
           decoder1_shape_mode_name(raw_runtime.shape_mode),
           prompt_tokens.size());
    return ret;
  }
  stream_state.token_ids.push_back(next_token);

  uint32_t position = (uint32_t)prompt_tokens.size();
  for (int step = 1; step < options.max_new_tokens && next_token != whisper_tokens::EOT; ++step)
  {
    int32_t token = next_token;
    ret = run_decoder1_raw_tokens(&decoder1_ctx, &decode0_ctx, &raw_runtime, &token, 1, position, token_embed, hidden_size, 1, &next_token, options.model.vocab_size);
    if (ret != 0)
    {
      printf("stage=decoder1 decode failed ret=%d step=%d model=%s audio_path=%s token_embed_path=%s shape_mode=%s token=%d position=%u n_new_tokens=%d\n",
             ret,
             step,
             options.decoder1_model,
             audio_path,
             options.token_embed_path,
             decoder1_shape_mode_name(raw_runtime.shape_mode),
             token,
             position,
             options.max_new_tokens);
      return ret;
    }
    stream_state.token_ids.push_back(next_token);
    ++position;
  }
  stream_state.text = decode_whisper_tokens(vocab.data(), stream_state.token_ids, options.task, options.model.vocab_size);
  infer_timer.tok();
  infer_time = infer_timer.get_time() / 1000.0f;
  printf("Whisper output: %s\n", stream_state.text.c_str());
  if (audio_length > 0.0f)
  {
    printf("Inference RTF: %.3f / %.3f = %.3f\n", infer_time, audio_length, infer_time / audio_length);
  }
  else
  {
    printf("Inference RTF: %.3f / %.3f = n/a\n", infer_time, audio_length);
  }
  return 0;
}

int main(int argc, char **argv)
{
  if (argc < 13)
  {
    printf("main_raw invalid argc=%d\n", argc);
    printf("%s <encoder.rknn> <encoder.weight> <decode0.rknn> <decode0.weight> <decode1.rknn> <decode1.weight> <task:en|zh> <audio.wav> <token_embed.bin> <encoder_core_mask_hex> <decode0_core_mask_hex> <decode1_core_mask_hex> [max_new_tokens] [--model-type base|turbo]\n", argv[0]);
    printf("example core masks: 0xff / 0x1 / 0x2 / 0x4; default max_new_tokens=64\n");
    return -1;
  }

  const char *encoder_model = argv[1];
  const char *encoder_weight = argv[2];
  const char *decode0_model = argv[3];
  const char *decode0_weight = argv[4];
  const char *decoder1_model = argv[5];
  const char *decoder1_weight = argv[6];
  const char *task = argv[7];
  const char *audio_path = argv[8];
  const char *token_embed_path = argv[9];
  uint32_t encoder_core_mask = 0;
  uint32_t decode0_core_mask = 0;
  uint32_t decoder1_core_mask = 0;
  int max_new_tokens = 64;

  if (!parse_core_mask_hex(argv[10], &encoder_core_mask) ||
      !parse_core_mask_hex(argv[11], &decode0_core_mask) ||
      !parse_core_mask_hex(argv[12], &decoder1_core_mask))
  {
    printf("main_raw invalid core mask args encoder=%s decode0=%s decoder1=%s\n", argv[10], argv[11], argv[12]);
    printf("invalid core mask, please input non-zero hex values, for example: 0xff 0xff 0xff\n");
    return -1;
  }
  bool is_turbo = false;
  int option_index = 13;
  if (option_index < argc && argv[option_index][0] != '-')
  {
    if (!parse_max_new_tokens(argv[option_index], &max_new_tokens))
    {
      printf("Invalid max_new_tokens: %s (expected integer in [1, %d])\n",
             argv[option_index], WHISPER_MAX_NEW_TOKENS);
      return -1;
    }
    ++option_index;
  }
  while (option_index < argc)
  {
    if (strcmp(argv[option_index], "--model-type") != 0 || option_index + 1 >= argc)
    {
      printf("Expected --model-type base|turbo\n");
      return -1;
    }
    const char *model_type = argv[option_index + 1];
    if (strcmp(model_type, "base") != 0 && strcmp(model_type, "turbo") != 0)
    {
      printf("Unsupported model type: %s (expected base or turbo)\n", model_type);
      return -1;
    }
    is_turbo = strcmp(model_type, "turbo") == 0;
    option_index += 2;
  }
  const WhisperModelConfig &config = is_turbo ? turbo_config : base_config;
  int max_audio_length = 0;
  printf("Using core masks: encoder=0x%x decode0=0x%x decoder1=0x%x, max_new_tokens=%d\n", encoder_core_mask, decode0_core_mask, decoder1_core_mask, max_new_tokens);

  const char *vocab_path = NULL;
  if (strcmp(task, "en") == 0)
  {
    vocab_path = "./model/vocab_en.txt";
  }
  else if (strcmp(task, "zh") == 0)
  {
    vocab_path = "./model/vocab_zh.txt";
  }
  else
  {
    printf("main_raw invalid task=%s\n", task);
    printf("task must be en or zh\n");
    return -1;
  }

  int ret = 0;
  audio_buffer_t audio;
  memset(&audio, 0, sizeof(audio));
  rknn_app_context_t encoder_ctx;
  rknn_app_context_t decode0_ctx;
  rknn_app_context_t decoder1_ctx;
  memset(&encoder_ctx, 0, sizeof(encoder_ctx));
  memset(&decode0_ctx, 0, sizeof(decode0_ctx));
  memset(&decoder1_ctx, 0, sizeof(decoder1_ctx));
  float *mel_filters = NULL;
  std::vector<VocabEntry> vocab(config.vocab_size, VocabEntry{});
  Decoder1RawRuntime raw_runtime = {};
  std::vector<uint8_t> token_embed_bytes;
  const float16 *token_embed = NULL;
  uint32_t hidden_size = 0;

  do
  {
    ret = init_whisper_model(encoder_model, &encoder_ctx, encoder_weight, encoder_core_mask);
    if (ret != 0)
    {
      printf("stage=load_model encoder failed ret=%d model=%s weight=%s core_mask=0x%x\n",
             ret, encoder_model, encoder_weight ? encoder_weight : "NULL", encoder_core_mask);
      break;
    }

    const rknn3_tensor_attr *mel_attr = encoder_ctx.io_num.n_input ? encoder_ctx.inputs[0].attr : NULL;
    if (!mel_attr || (mel_attr->n_dims != 3 && mel_attr->n_dims != 4) ||
        mel_attr->shape[0] != 1 || (mel_attr->n_dims == 4 && mel_attr->shape[1] != 1) ||
        mel_attr->shape[mel_attr->n_dims - 2] != (uint32_t)config.n_mels)
    {
      printf("Encoder input does not match --model-type %s (%d Mel bins)\n",
             is_turbo ? "turbo" : "base", config.n_mels);
      ret = -1;
      break;
    }
    uint32_t mel_frames = mel_attr->shape[mel_attr->n_dims - 1];
    if (mel_frames == 0 || mel_frames > 3000 || mel_frames % 2 != 0)
    {
      printf("Unsupported encoder Mel frame count: %u\n", mel_frames);
      ret = -1;
      break;
    }
    max_audio_length = mel_frames * HOP_LENGTH;
    printf("Model type=%s, n_mels=%d, vocab_size=%d, audio window=%.2f s\n",
           is_turbo ? "turbo" : "base", config.n_mels, config.vocab_size,
           (float)max_audio_length / SAMPLE_RATE);

    ret = init_whisper_model(decode0_model, &decode0_ctx, decode0_weight, decode0_core_mask);
    if (ret != 0)
    {
      printf("stage=load_model decode0 failed ret=%d model=%s weight=%s core_mask=0x%x\n",
             ret, decode0_model, decode0_weight ? decode0_weight : "NULL", decode0_core_mask);
      break;
    }

    ret = init_whisper_model(decoder1_model, &decoder1_ctx, decoder1_weight, decoder1_core_mask);
    if (ret != 0)
    {
      printf("stage=load_model decoder1 failed ret=%d model=%s weight=%s core_mask=0x%x\n",
             ret, decoder1_model, decoder1_weight ? decoder1_weight : "NULL", decoder1_core_mask);
      break;
    }

    ret = prepare_decoder1_raw_runtime(&decoder1_ctx, &raw_runtime);
    if (ret != 0)
    {
      printf("stage=decoder1_prepare failed ret=%d model=%s weight=%s shape_mode=%s\n",
             ret, decoder1_model, decoder1_weight ? decoder1_weight : "NULL", decoder1_shape_mode_name(raw_runtime.shape_mode));
      break;
    }
    if (!raw_runtime.shapes.empty())
    {
      uint32_t initial_shape_id = raw_runtime.prefill_shape_id;
      if (raw_runtime.shape_mode == DECODER1_SHAPE1_ONLY)
      {
        initial_shape_id = raw_runtime.decode_shape_id;
      }
      ret = switch_decoder1_raw_shape(&decoder1_ctx, &raw_runtime, initial_shape_id);
      if (ret != 0)
      {
        printf("stage=decoder1_switch failed ret=%d model=%s weight=%s shape_id=%u shape_mode=%s\n",
               ret, decoder1_model, decoder1_weight ? decoder1_weight : "NULL", initial_shape_id, decoder1_shape_mode_name(raw_runtime.shape_mode));
        break;
      }
    }

    int embed_input_index = whisper_require_input_index(&decoder1_ctx, "decoder1_preload", "input_embeds", 0);
    if (embed_input_index < 0)
    {
      ret = -1;
      printf("stage=decoder1_preload missing required input name=input_embeds fallback_index=0 ret=%d model=%s weight=%s\n",
             ret, decoder1_model, decoder1_weight ? decoder1_weight : "NULL");
      break;
    }

    ret = read_binary_file(token_embed_path, token_embed_bytes);
    if (ret != 0)
    {
      printf("stage=load_token_embed failed ret=%d path=%s model=%s weight=%s\n",
             ret, token_embed_path, decoder1_model, decoder1_weight ? decoder1_weight : "NULL");
      break;
    }
    if (!decoder1_ctx.inputs[embed_input_index].attr)
    {
      ret = -1;
      printf("stage=load_token_embed invalid input attr model=%s weight=%s input_index=%d\n",
             decoder1_model, decoder1_weight ? decoder1_weight : "NULL", embed_input_index);
      break;
    }
    hidden_size = decoder1_ctx.inputs[embed_input_index].attr->shape[2];
    uint64_t expected_token_bytes = (uint64_t)config.vocab_size * hidden_size * sizeof(float16);
    if (token_embed_bytes.size() != expected_token_bytes)
    {
      printf("stage=load_token_embed token embed size mismatch ret=-1 path=%s expected=%lu actual=%lu hidden_size=%u input_index=%d name=%s\n",
             token_embed_path,
             expected_token_bytes,
             (uint64_t)token_embed_bytes.size(),
             hidden_size,
             embed_input_index,
             decoder1_ctx.inputs[embed_input_index].attr ? decoder1_ctx.inputs[embed_input_index].attr->name : "<null>");
      ret = -1;
      break;
    }
    token_embed = reinterpret_cast<const float16 *>(token_embed_bytes.data());

    ret = read_vocab(vocab_path, vocab.data(), config.vocab_size);
    if (ret != 0)
    {
      printf("stage=load_vocab failed ret=%d vocab_path=%s task=%s\n", ret, vocab_path, task);
      break;
    }
    mel_filters = (float *)malloc(config.n_mels * MELS_FILTERS_SIZE * sizeof(float));
    if (!mel_filters)
    {
      ret = -1;
      printf("stage=preprocess mel_filters alloc failed ret=-1 size=%u path=%s\n",
             config.n_mels * MELS_FILTERS_SIZE, config.mel_filters_path);
      break;
    }
    ret = read_mel_filters(config.mel_filters_path, mel_filters, config.n_mels * MELS_FILTERS_SIZE);
    if (ret != 0)
    {
      printf("stage=preprocess read_mel_filters failed ret=%d path=%s\n", ret, config.mel_filters_path);
      break;
    }

    const WhisperAudioOptions audio_options = {
        encoder_model, encoder_weight, decode0_model, decode0_weight, decoder1_model,
        token_embed_path, task, config, max_audio_length, max_new_tokens, is_turbo
    };
    ret = process_single_audio(audio_path, audio, encoder_ctx, decode0_ctx, decoder1_ctx,
                               raw_runtime, mel_filters, vocab, token_embed, hidden_size, audio_options);
  } while (0);

  if (mel_filters)
  {
    free(mel_filters);
  }
  release_whisper_model(&encoder_ctx);
  release_whisper_model(&decode0_ctx);
  release_whisper_model(&decoder1_ctx);
  for (int i = 0; i < config.vocab_size; ++i)
  {
    if (vocab[i].token)
    {
      free(vocab[i].token);
    }
  }
  if (audio.data)
  {
    free(audio.data);
  }
  if (ret != 0)
  {
    printf("main_raw failed: ret=%d encoder_model=%s decode0_model=%s decoder1_model=%s audio_path=%s token_embed_path=%s\n",
           ret,
           encoder_model,
           decode0_model,
           decoder1_model,
           audio_path,
           token_embed_path);
  }
  return ret;
}
