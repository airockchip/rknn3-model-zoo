// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#include "Tokenizer.h"
#include "float16.h"
#include "rknn3_api.h"

#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <deque>
#include <functional>
#include <inttypes.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "nlohmann/json.hpp"

#define LOGE(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#define LOGW(fmt, ...) fprintf(stderr, "\033[33m" fmt "\033[0m", ##__VA_ARGS__)
#define VLOG(fmt, ...) do { if (g_verbose) printf(fmt, ##__VA_ARGS__); } while (0)

static size_t g_stage_count = 2;
static uint64_t g_bucket_size = 128;
static bool g_verbose = false;
static bool g_ignore_eos = false;
static bool g_performance_mode = false;
static bool g_tensor_dump_enabled = false;
static std::string g_tensor_dump_dir;
static std::mutex g_tensor_dump_mutex;
static uint64_t g_embed_dump_count = 0;
static uint64_t g_input_dump_count = 0;

enum class InferencePhase
{
  PREFILL,
  DECODE,
};

struct TensorBlob
{
  rknn3_tensor_attr    attr;
  std::vector<uint8_t> data;
};

struct StageBatch
{
  std::vector<TensorBlob> tensors;
  uint64_t                n_tokens = 0;
};

struct StageSlot
{
  std::mutex              mutex;
  std::condition_variable cv;
  std::deque<StageBatch>  batches;
  uint64_t                expected_tokens = 0;
  uint64_t                emitted_tokens = 0;
  uint64_t                active_input_tokens = 0;
  bool                    producer_done = false;
  bool                    failed = false;
};

struct EmbedCallbackContext;

struct PipelineState
{
  std::vector<std::unique_ptr<StageSlot>> slots;

  explicit PipelineState(size_t stage_count)
  {
    slots.reserve(stage_count);
    for (size_t i = 0; i < stage_count; ++i) {
      slots.emplace_back(new StageSlot());
    }
  }
};

struct embedding_info
{
  int      fd = -1;
  float16* embedding_data = nullptr;
  int      embedding_dim = 0;
  int      vocab_size = 0;
};

struct EmbedCallbackContext
{
  embedding_info* embed_info = nullptr;
  PipelineState*  pipeline = nullptr;
};

struct StageCallbackContext
{
  PipelineState* pipeline = nullptr;
  size_t         stage_index = 0;
  int32_t        embedding_dim = 0;
};

struct PhasePerformanceStatistics
{
  uint64_t run_count = 0;
  uint64_t token_count = 0;
  double   total_time_ms = 0.0;
};

struct StagePerformanceStatistics
{
  PhasePerformanceStatistics prefill;
  PhasePerformanceStatistics decode;
};

struct StageRuntime
{
  std::string      name;
  std::string      model_path;
  std::string      weight_path;
  rknn3_context    ctx = 0;
  rknn3_session*   session = nullptr;
  int32_t          embedding_dim = 0;
  int32_t          vocab_size = 0;
  int32_t          max_ctx_len = 0;
  rknn3_tensor*    output_tensors = nullptr;
  int              n_output_tensors = 0;
  int*             ext_input_indices = nullptr;
  int              n_ext_inputs = 0;
  StageCallbackContext callback_ctx;
  StagePerformanceStatistics performance;
};

struct LastStageResultState
{
  Tokenizer* tokenizer = nullptr;
  std::mutex mutex;
  bool       has_token = false;
  int32_t    next_token = -1;
};

struct rope_cache_tensor
{
  void* data = nullptr;
  int   n_dims = 0;
  int   shape[5] = {};
  int   dtype = 0;
  int   layout = 0;
};

enum class RopeCacheFormat
{
  NONE = 0,
  GEMMA4,
  QWEN35,
};

static const char* GEMMA4_ROPE_CACHE_NAMES[4] = {
    "rope_cos_cache_0", "rope_sin_cache_0",
    "rope_cos_cache_1", "rope_sin_cache_1"
};

static const char* QWEN35_ROPE_CACHE_NAMES[2] = {
    "rope_cos_cache", "rope_sin_cache"
};

struct InputCbUserdata
{
  rope_cache_tensor rope_caches[4];
  RopeCacheFormat   rope_format = RopeCacheFormat::NONE;
  int               rope_fd = -1;
  void*             rope_mmap_base = nullptr;
  size_t            rope_mmap_size = 0;
};

static int find_rope_cache_index(RopeCacheFormat format, const char* name)
{
  const char* const* names = nullptr;
  int                count = 0;
  if (format == RopeCacheFormat::GEMMA4) {
    names = GEMMA4_ROPE_CACHE_NAMES;
    count = 4;
  } else if (format == RopeCacheFormat::QWEN35) {
    names = QWEN35_ROPE_CACHE_NAMES;
    count = 2;
  }

  for (int i = 0; i < count; ++i) {
    if (strcmp(name, names[i]) == 0) {
      return i;
    }
  }
  return -1;
}

static LastStageResultState g_last_stage_result;

static size_t get_dtype_elem_size(int dtype)
{
  switch (dtype) {
  case 0:  return 4;   /* FLOAT32   */
  case 1:  return 2;   /* FLOAT16   */
  case 2:  return 1;   /* INT8      */
  case 3:  return 1;   /* UINT8     */
  case 4:  return 2;   /* INT16     */
  case 5:  return 2;   /* UINT16    */
  case 6:  return 4;   /* INT32     */
  case 7:  return 4;   /* UINT32    */
  case 8:  return 8;   /* INT64     */
  case 9:  return 8;   /* UINT64    */
  case 10: return 1;   /* BOOL      */
  case 11: return 1;   /* INT4      */
  case 12: return 1;   /* FLOAT8E4M3FN */
  case 13: return 2;   /* BFLOAT16  */
  case 14: return 1;   /* FLOAT8E8M0   */
  case 15: return 1;   /* FLOAT4E2M1   */
  default: return 1;
  }
}

static double elapsed_us(const timeval& start, const timeval& end)
{
  return (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_usec - start.tv_usec);
}

static void print_performance_statistics(uint64_t prefill_tokens, float prefill_ms,
                uint64_t decode_tokens, float decode_ms)
{
  float prefill_s = prefill_ms / 1e3f;
  float prefill_tpt = prefill_tokens == 0 ? 0.0f : prefill_ms / (float)prefill_tokens;
  float prefill_tps = prefill_tokens == 0 ? 0.0f : (float)prefill_tokens / prefill_s;

  float decode_s = decode_ms / 1e3f;
  float decode_tpt = decode_tokens == 0 ? 0.0f : decode_ms / (float)decode_tokens;
  float decode_tps = decode_tokens == 0 ? 0.0f : (float)decode_tokens / decode_s;

  printf("\n\nPerformance Statistics: ");
  printf("\n-----------------------------------------------------------------------------------------\n");
  printf(" %-10s | %-16s | %-8s | %-20s | %-20s \n",
    "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
  printf("-----------------------------------------------------------------------------------------\n");
  printf(" %-10s | %-16.2f | %-8llu | %-20.2f | %-20.2f \n",
    "Prefill", prefill_ms, (unsigned long long)prefill_tokens, prefill_tpt, prefill_tps);
  printf(" %-10s | %-16.2f | %-8llu | %-20.2f | %-20.2f \n",
    "Decode", decode_ms, (unsigned long long)decode_tokens, decode_tpt, decode_tps);
  printf("-----------------------------------------------------------------------------------------\n");
}

static void record_stage_performance(StageRuntime& stage, InferencePhase phase,
                                     uint64_t token_count, double total_time_ms)
{
  PhasePerformanceStatistics& statistics = phase == InferencePhase::PREFILL
                                                ? stage.performance.prefill
                                                : stage.performance.decode;
  statistics.run_count++;
  statistics.token_count += token_count;
  statistics.total_time_ms += total_time_ms;
}

static void print_stage_performance_statistics(const std::vector<StageRuntime>& stages)
{
  printf("\nPer-Stage Performance Statistics: ");
  printf("\n----------------------------------------------------------------------------------------------------------------------\n");
  printf(" %-10s | %-10s | %-8s | %-16s | %-8s | %-20s | %-20s \n",
         "Stage", "Phase", "Runs", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
  printf("----------------------------------------------------------------------------------------------------------------------\n");

  for (const auto& stage : stages) {
    const PhasePerformanceStatistics* phase_statistics[] = {
        &stage.performance.prefill,
        &stage.performance.decode,
    };
    const char* phase_names[] = {"Prefill", "Decode"};

    for (size_t i = 0; i < 2; ++i) {
      const PhasePerformanceStatistics& statistics = *phase_statistics[i];
      double time_per_token = statistics.token_count == 0
                                  ? 0.0
                                  : statistics.total_time_ms / (double)statistics.token_count;
      double tokens_per_second = statistics.total_time_ms <= 0.0
                                     ? 0.0
                                     : (double)statistics.token_count * 1e3 / statistics.total_time_ms;
      printf(" %-10s | %-10s | %-8llu | %-16.2f | %-8llu | %-20.2f | %-20.2f \n",
             stage.name.c_str(), phase_names[i],
             (unsigned long long)statistics.run_count,
             statistics.total_time_ms,
             (unsigned long long)statistics.token_count,
             time_per_token, tokens_per_second);
    }
  }
  printf("----------------------------------------------------------------------------------------------------------------------\n");
}

static void release_safetensors(InputCbUserdata* cb_data)
{
  if (!cb_data) return;
  if (cb_data->rope_mmap_base && cb_data->rope_mmap_base != MAP_FAILED) {
    munmap(cb_data->rope_mmap_base, cb_data->rope_mmap_size);
    cb_data->rope_mmap_base = nullptr;
  }
  if (cb_data->rope_fd >= 0) {
    close(cb_data->rope_fd);
    cb_data->rope_fd = -1;
  }
}

static void release_output_tensors(StageRuntime& stage)
{
  if (!stage.output_tensors) {
    return;
  }

  for (int i = 0; i < stage.n_output_tensors; ++i) {
    if (stage.output_tensors[i].mem) {
      rknn3_destroy_mem(stage.ctx, stage.output_tensors[i].mem);
      stage.output_tensors[i].mem = nullptr;
    }
    if (stage.output_tensors[i].attr) {
      free(stage.output_tensors[i].attr);
      stage.output_tensors[i].attr = nullptr;
    }
  }

  free(stage.output_tensors);
  stage.output_tensors = nullptr;
  stage.n_output_tensors = 0;
}

static void destroy_stage(StageRuntime& stage)
{
  if (stage.session) {
    rknn3_session_destroy(stage.session);
    stage.session = nullptr;
  }

  release_output_tensors(stage);

  if (stage.ext_input_indices) {
    free(stage.ext_input_indices);
    stage.ext_input_indices = nullptr;
  }
  stage.n_ext_inputs = 0;

  if (stage.ctx) {
    rknn3_destroy(stage.ctx);
    stage.ctx = 0;
  }
}

static void destroy_stages(std::vector<StageRuntime>& stages)
{
  for (auto& stage : stages) {
    destroy_stage(stage);
  }
}

static void release_resources(std::vector<StageRuntime>& stages, InputCbUserdata* input_cb_data,
                              embedding_info* embed_info, size_t embedding_size, Tokenizer* tokenizer)
{
  destroy_stages(stages);
  release_safetensors(input_cb_data);
  if (embed_info->embedding_data) {
    munmap(embed_info->embedding_data, embedding_size);
    embed_info->embedding_data = nullptr;
  }
  if (embed_info->fd != -1) {
    close(embed_info->fd);
    embed_info->fd = -1;
  }
  delete tokenizer;
}

static void reset_stage_slot(StageSlot& slot)
{
  std::lock_guard<std::mutex> lock(slot.mutex);
  slot.batches.clear();
  slot.expected_tokens = 0;
  slot.emitted_tokens = 0;
  slot.active_input_tokens = 0;
  slot.producer_done = false;
  slot.failed = false;
}

static void reset_pipeline(PipelineState& pipeline)
{
  for (auto& slot : pipeline.slots) {
    reset_stage_slot(*slot);
  }
}

static void close_stage_slot(StageSlot& slot)
{
  {
    std::lock_guard<std::mutex> lock(slot.mutex);
    slot.producer_done = true;
  }
  slot.cv.notify_all();
}

static void fail_pipeline(PipelineState& pipeline)
{
  for (auto& slot_ptr : pipeline.slots) {
    StageSlot& slot = *slot_ptr;
    {
      std::lock_guard<std::mutex> lock(slot.mutex);
      slot.failed = true;
    }
    slot.cv.notify_all();
  }
}

static bool pipeline_failed(PipelineState& pipeline)
{
  for (auto& slot_ptr : pipeline.slots) {
    std::lock_guard<std::mutex> lock(slot_ptr->mutex);
    if (slot_ptr->failed) {
      return true;
    }
  }
  return false;
}

static void reset_last_stage_result()
{
  std::lock_guard<std::mutex> lock(g_last_stage_result.mutex);
  g_last_stage_result.has_token = false;
  g_last_stage_result.next_token = -1;
}

static bool get_last_stage_token(int32_t* token)
{
  std::lock_guard<std::mutex> lock(g_last_stage_result.mutex);
  if (!g_last_stage_result.has_token) {
    return false;
  }
  *token = g_last_stage_result.next_token;
  return true;
}

static bool parse_positive_u64(const char* value, uint64_t* result)
{
  if (!value || !result || value[0] == '\0' || value[0] == '-') {
    return false;
  }

  errno = 0;
  char* end = nullptr;
  unsigned long long parsed = strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
    return false;
  }

  *result = (uint64_t)parsed;
  return true;
}

static bool parse_positive_i32(const char* value, int32_t* result)
{
  uint64_t parsed = 0;
  if (!result || !parse_positive_u64(value, &parsed) || parsed > 0x7fffffffU) {
    return false;
  }
  *result = (int32_t)parsed;
  return true;
}

static bool parse_bool01(const char* value, bool* result)
{
  if (!value || !result) {
    return false;
  }
  if (strcmp(value, "0") == 0) {
    *result = false;
    return true;
  }
  if (strcmp(value, "1") == 0) {
    *result = true;
    return true;
  }
  return false;
}

static bool parse_core_mask(const char* value, uint32_t* result)
{
  if (!value || !result || value[0] == '\0' || value[0] == '-') {
    return false;
  }

  errno = 0;
  char* end = nullptr;
  unsigned long long parsed = strtoull(value, &end, 0);
  if (errno != 0 || end == value || *end != '\0') {
    errno = 0;
    end = nullptr;
    parsed = strtoull(value, &end, 16);
  }
  if (errno != 0 || end == value || *end != '\0' || parsed > 0xffffffffULL) {
    return false;
  }

  *result = (uint32_t)parsed;
  return true;
}

struct CommandLineOptions
{
  const char* stage0_model = nullptr;
  const char* stage0_weight = nullptr;
  const char* tokenizer = nullptr;
  const char* embedding = nullptr;
  int32_t     max_context_len = 0;
  uint32_t    run_core_mask = 0;
  size_t      stage_count = 0;
  uint64_t    bucket_size = 0;
  bool        has_max_context_len = false;
  bool        has_run_core_mask = false;
  bool        has_stage_count = false;
  bool        has_bucket_size = false;

  const char* prompt = nullptr;
  int         max_new_tokens = 512;
  bool        verbose = false;
  bool        ignore_eos = false;
  const char* rope_path = nullptr;
  const char* custom_op_path = nullptr;
  const char* tensor_dump_dir = nullptr;
  std::vector<std::string> device_ids;

  bool     performance_mode = false;
  uint64_t performance_input_length = 0;
  uint64_t performance_output_length = 0;
};

static void print_usage(const char* program)
{
  printf("Usage:\n");
  printf("  %s \\\n", program);
  printf("    --model <path> --weight <path> \\\n"
         "    --vocab <path> --embed <path> \\\n"
         "    -c, --ctx-size <tokens> --core-mask <mask> \\\n"
         "    --stage-count <count> --bucket-size <tokens> [options]\n");
  printf("Options:\n");
  printf("  --prompt <text-or-file>       prompt text or a .txt prompt file\n");
  printf("  -n/--predict/--n-predict <count>  maximum generated tokens (default: 512)\n");
  printf("  --verbose                     enable verbose logs\n");
  printf("  --ignore-eos                  ignore EOS during generation\n");
  printf("  --rope-tensor <safetensors>   external rope cache\n");
  printf("  --custom-op-path <plugin.so>  custom op plugin; registered for every stage ctx\n");
  printf("  --device-id <id[#id...]>      device IDs separated by '#'; optional\n");
  printf("  --perf <input> <output>       performance test mode\n");
  printf("  --dump-tensors <dir>          dump callback tensors to this directory\n");
  printf("  --help                        show this message\n");
  printf("Legacy positional arguments remain supported for compatibility.\n");
}

static bool take_option_value(int argc, char** argv, int* index,
                              const char* option, const char** value)
{
  if (!index || !value || *index + 1 >= argc || argv[*index + 1][0] == '\0' ||
      strncmp(argv[*index + 1], "--", 2) == 0) {
    LOGE("%s requires a value\n", option);
    return false;
  }
  *value = argv[++(*index)];
  return true;
}

static bool append_device_ids(const char* value, std::vector<std::string>* device_ids)
{
  if (!value || !device_ids || value[0] == '\0') {
    return false;
  }

  std::string all_ids = value;
  size_t start = 0;
  while (start <= all_ids.size()) {
    size_t end = all_ids.find('#', start);
    if (end == start) {
      return false;
    }
    device_ids->push_back(all_ids.substr(start, end == std::string::npos
                                                   ? std::string::npos
                                                   : end - start));
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
    if (start == all_ids.size()) {
      return false;
    }
  }
  return true;
}

static bool validate_command_line_options(const CommandLineOptions& options)
{
  if (!options.stage0_model || !options.stage0_weight || !options.tokenizer ||
      !options.embedding || !options.has_max_context_len ||
      !options.has_run_core_mask || !options.has_stage_count ||
      !options.has_bucket_size) {
    LOGE("missing required command-line option\n");
    return false;
  }
  if (options.max_context_len <= 0 || options.stage_count == 0 || options.bucket_size == 0) {
    LOGE("max-context-len, stage-count and bucket-size must be positive\n");
    return false;
  }
  if (!options.device_ids.empty() && options.device_ids.size() != options.stage_count) {
    LOGE("expected %zu --device-id values, got %zu\n",
           options.stage_count, options.device_ids.size());
    return false;
  }
  if (options.custom_op_path) {
    struct stat st;
    if (stat(options.custom_op_path, &st) != 0 || !S_ISREG(st.st_mode)) {
      LOGE("custom op plugin file not found or not a regular file: %s\n",
             options.custom_op_path);
      return false;
    }
  }
  return true;
}

static bool parse_named_command_line(int argc, char** argv, CommandLineOptions* options)
{
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    const char* value = nullptr;

    if (strcmp(arg, "-m") == 0 || strcmp(arg, "--model") == 0 ||
        strcmp(arg, "--stage0-model") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->stage0_model = value;
    } else if (strcmp(arg, "--weight") == 0 || strcmp(arg, "--stage0-weight") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->stage0_weight = value;
    } else if (strcmp(arg, "--vocab") == 0 || strcmp(arg, "--tokenizer") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->tokenizer = value;
    } else if (strcmp(arg, "--embed") == 0 || strcmp(arg, "--embedding") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value)) return false;
      options->embedding = value;
    } else if (strcmp(arg, "-c") == 0 || strcmp(arg, "--ctx-size") == 0 ||
               strcmp(arg, "--max-context-len") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value) ||
          !parse_positive_i32(value, &options->max_context_len)) {
        LOGE("%s requires a positive integer\n", arg);
        return false;
      }
      options->has_max_context_len = true;
    } else if (strcmp(arg, "--core-mask") == 0 || strcmp(arg, "--run-core-mask") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value) ||
          !parse_core_mask(value, &options->run_core_mask)) {
        LOGE("%s requires a valid hexadecimal mask\n", arg);
        return false;
      }
      options->has_run_core_mask = true;
    } else if (strcmp(arg, "--stage-count") == 0) {
      uint64_t parsed = 0;
      if (!take_option_value(argc, argv, &i, arg, &value) ||
          !parse_positive_u64(value, &parsed) || parsed > (uint64_t)(size_t)-1) {
        LOGE("%s requires a positive integer\n", arg);
        return false;
      }
      options->stage_count = (size_t)parsed;
      options->has_stage_count = true;
    } else if (strcmp(arg, "--bucket-size") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value) ||
          !parse_positive_u64(value, &options->bucket_size)) {
        LOGE("%s requires a positive integer\n", arg);
        return false;
      }
      options->has_bucket_size = true;
    } else if (strcmp(arg, "--prompt") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &options->prompt)) return false;
    } else if (strcmp(arg, "-n") == 0 || strcmp(arg, "--predict") == 0 ||
               strcmp(arg, "--n-predict") == 0 || strcmp(arg, "--max-new-tokens") == 0) {
      int32_t parsed = 0;
      if (!take_option_value(argc, argv, &i, arg, &value) ||
          !parse_positive_i32(value, &parsed)) {
        LOGE("%s requires a positive integer\n", arg);
        return false;
      }
      options->max_new_tokens = (int)parsed;
    } else if (strcmp(arg, "--verbose") == 0) {
      options->verbose = true;
    } else if (strcmp(arg, "--no-verbose") == 0) {
      options->verbose = false;
    } else if (strcmp(arg, "--ignore-eos") == 0) {
      options->ignore_eos = true;
    } else if (strcmp(arg, "--no-ignore-eos") == 0) {
      options->ignore_eos = false;
    } else if (strcmp(arg, "--rope-tensor") == 0 || strcmp(arg, "--rope") == 0 ||
               strcmp(arg, "--rope-path") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &options->rope_path)) return false;
    } else if (strcmp(arg, "--custom-op-path") == 0 ||
               strcmp(arg, "--custom_op_path") == 0) {
      if (options->custom_op_path ||
          !take_option_value(argc, argv, &i, arg, &options->custom_op_path)) {
        LOGE("%s requires a value and can only be specified once\n", arg);
        return false;
      }
    } else if (strcmp(arg, "--device-id") == 0) {
      if (!take_option_value(argc, argv, &i, arg, &value) ||
          !append_device_ids(value, &options->device_ids)) {
        LOGE("%s requires one or more device IDs separated by '#': id0#id1#...\n", arg);
        return false;
      }
    } else if (strcmp(arg, "--perf") == 0 || strcmp(arg, "--performance") == 0) {
      const char* input_value = nullptr;
      const char* output_value = nullptr;
      if (options->performance_mode) {
        LOGE("performance mode can only be specified once\n");
        return false;
      }
      if (!take_option_value(argc, argv, &i, arg, &input_value) ||
          !take_option_value(argc, argv, &i, arg, &output_value) ||
          !parse_positive_u64(input_value, &options->performance_input_length) ||
          !parse_positive_u64(output_value, &options->performance_output_length)) {
        LOGE("%s requires positive <input_tokens> and <output_tokens>\n", arg);
        return false;
      }
      options->performance_mode = true;
    } else if (strcmp(arg, "--dump-tensors") == 0) {
      if (options->tensor_dump_dir ||
          !take_option_value(argc, argv, &i, arg, &options->tensor_dump_dir)) {
        LOGE("%s requires a value and can only be specified once\n", arg);
        return false;
      }
    } else if (strcmp(arg, "--help") == 0) {
      return false;
    } else {
      LOGE("unknown or positional argument in named mode: %s\n", arg);
      return false;
    }
  }

  return validate_command_line_options(*options);
}

static bool parse_legacy_command_line(int argc, char** argv, CommandLineOptions* options)
{
  if (argc < 9) {
    LOGE("legacy mode requires 8 positional arguments\n");
    return false;
  }

  options->stage0_model = argv[1];
  options->stage0_weight = argv[2];
  options->tokenizer = argv[3];
  options->embedding = argv[4];
  if (!parse_positive_i32(argv[5], &options->max_context_len) ||
      !parse_core_mask(argv[6], &options->run_core_mask)) {
    LOGE("invalid legacy max_context_len or run_core_mask\n");
    return false;
  }
  uint64_t parsed_stage_count = 0;
  if (!parse_positive_u64(argv[7], &parsed_stage_count) ||
      parsed_stage_count > (uint64_t)(size_t)-1 ||
      !parse_positive_u64(argv[8], &options->bucket_size)) {
    LOGE("invalid legacy stage_count or bucket_size\n");
    return false;
  }
  options->stage_count = (size_t)parsed_stage_count;
  options->has_max_context_len = true;
  options->has_run_core_mask = true;
  options->has_stage_count = true;
  options->has_bucket_size = true;

  std::vector<const char*> optional_args;
  for (int i = 9; i < argc; ++i) {
    const char* arg = argv[i];
    if (strcmp(arg, "--perf") == 0 || strcmp(arg, "--performance") == 0) {
      const char* input_value = nullptr;
      const char* output_value = nullptr;
      if (options->performance_mode) {
        LOGE("performance mode can only be specified once\n");
        return false;
      }
      if (!take_option_value(argc, argv, &i, arg, &input_value) ||
          !take_option_value(argc, argv, &i, arg, &output_value) ||
          !parse_positive_u64(input_value, &options->performance_input_length) ||
          !parse_positive_u64(output_value, &options->performance_output_length)) {
        LOGE("%s requires positive <input_tokens> and <output_tokens>\n", arg);
        return false;
      }
      options->performance_mode = true;
    } else if (strcmp(arg, "--dump-tensors") == 0) {
      if (options->tensor_dump_dir ||
          !take_option_value(argc, argv, &i, arg, &options->tensor_dump_dir)) {
        LOGE("%s requires a value and can only be specified once\n", arg);
        return false;
      }
    } else if (strcmp(arg, "--custom-op-path") == 0 ||
               strcmp(arg, "--custom_op_path") == 0) {
      if (options->custom_op_path ||
          !take_option_value(argc, argv, &i, arg, &options->custom_op_path)) {
        LOGE("%s requires a value and can only be specified once\n", arg);
        return false;
      }
    } else if (arg[0] == '-') {
      LOGE("unknown option in legacy mode: %s\n", arg);
      return false;
    } else {
      optional_args.push_back(arg);
    }
  }

  if (optional_args.size() > 5 + options->stage_count) {
    LOGE("too many legacy optional arguments\n");
    return false;
  }
  if (optional_args.size() >= 1) options->prompt = optional_args[0];
  if (optional_args.size() >= 2) {
    int32_t parsed = 0;
    if (!parse_positive_i32(optional_args[1], &parsed)) {
      LOGE("invalid legacy max_new_tokens: %s\n", optional_args[1]);
      return false;
    }
    options->max_new_tokens = (int)parsed;
  }
  if (optional_args.size() >= 3 && !parse_bool01(optional_args[2], &options->verbose)) {
    LOGE("legacy verbose must be 0 or 1\n");
    return false;
  }
  if (optional_args.size() >= 4 && !parse_bool01(optional_args[3], &options->ignore_eos)) {
    LOGE("legacy ignore_eos must be 0 or 1\n");
    return false;
  }
  if (optional_args.size() >= 5) options->rope_path = optional_args[4];
  for (size_t i = 5; i < optional_args.size(); ++i) {
    if (!append_device_ids(optional_args[i], &options->device_ids)) {
      LOGE("invalid legacy device ID list: %s\n", optional_args[i]);
      return false;
    }
  }

  return validate_command_line_options(*options);
}

static bool parse_command_line(int argc, char** argv, CommandLineOptions* options)
{
  if (!options || argc < 2) {
    return false;
  }
  if (argv[1][0] == '-') {
    return parse_named_command_line(argc, argv, options);
  }
  return parse_legacy_command_line(argc, argv, options);
}

static void build_performance_input_tokens(const VocabInfo& vocab_info, uint64_t input_length,
                                           std::vector<int32_t>* input_tokens)
{
  int32_t bos_token = -1;
  if (vocab_info.n_special_bos_id > 0 &&
      vocab_info.special_bos_id[0] >= 0 &&
      vocab_info.special_bos_id[0] < vocab_info.vocab_size) {
    bos_token = vocab_info.special_bos_id[0];
  }

  int32_t fill_token = vocab_info.linefeed_id;
  if (fill_token < 0 || fill_token >= vocab_info.vocab_size) {
    fill_token = bos_token >= 0 ? bos_token : 0;
  }
  input_tokens->assign((size_t)input_length, fill_token);
  if (bos_token >= 0 && !input_tokens->empty()) {
    input_tokens->front() = bos_token;
  }
}

static bool name_contains(const char* name, const char* needle)
{
  return name && needle && strstr(name, needle) != nullptr;
}

static void dump_tensor_blob(const TensorBlob& blob, size_t index, const char* prefix)
{
  VLOG("%s tensor[%zu]: name=%s, dtype=%d, n_elems=%u, aligned_size=%llu\n",
       prefix,
       index,
       blob.attr.name,
       (int)blob.attr.dtype,
       blob.attr.n_elems,
       (unsigned long long)blob.attr.aligned_size);
}

static bool configure_tensor_dump(const char* path)
{
  if (!path || path[0] == '\0') {
    return false;
  }

  struct stat st;
  if (stat(path, &st) == 0) {
    if (!S_ISDIR(st.st_mode)) {
      LOGE("tensor dump path is not a directory: %s\n", path);
      return false;
    }
  } else {
    if (errno != ENOENT || mkdir(path, 0755) != 0) {
      LOGE("failed to create tensor dump directory: %s, error=%s\n",
             path, strerror(errno));
      return false;
    }
  }

  g_tensor_dump_dir = path;
  g_tensor_dump_enabled = true;
  printf("tensor dump enabled: %s\n", g_tensor_dump_dir.c_str());
  return true;
}

static std::string sanitize_tensor_dump_name(const char* name)
{
  std::string result = name ? name : "unnamed";
  for (size_t i = 0; i < result.size(); ++i) {
    unsigned char c = (unsigned char)result[i];
    if (!isalnum(c) && c != '_' && c != '-' && c != '.') {
      result[i] = '_';
    }
  }
  if (result.empty()) {
    result = "unnamed";
  }
  return result;
}

static std::string make_tensor_dump_base_path(const char* callback_name,
                                              uint64_t call_index,
                                              uint32_t tensor_index,
                                              const char* tensor_name)
{
  char prefix[128];
  snprintf(prefix, sizeof(prefix), "%s_%08llu_tensor_%03u_",
           callback_name,
           (unsigned long long)call_index,
           tensor_index);

  std::string path = g_tensor_dump_dir;
  if (!path.empty() && path.back() != '/') {
    path.push_back('/');
  }
  path += prefix;
  path += sanitize_tensor_dump_name(tensor_name);
  return path;
}

// The caller holds g_tensor_dump_mutex while this function runs.
static void dump_tensor_data_locked(const char* callback_name,
                                    uint64_t call_index,
                                    uint32_t tensor_index,
                                    const char* tensor_name,
                                    const void* data,
                                    size_t data_size,
                                    const rknn3_tensor_attr* attr,
                                    const char* extra_metadata)
{
  std::string base_path = make_tensor_dump_base_path(callback_name, call_index,
                                                     tensor_index, tensor_name);
  std::string data_path = base_path + ".bin";
  FILE* data_file = fopen(data_path.c_str(), "wb");
  if (!data_file) {
    LOGE("failed to open tensor dump file: %s, error=%s\n",
           data_path.c_str(), strerror(errno));
    return;
  }

  bool write_ok = true;
  if (data_size > 0 && (!data || fwrite(data, 1, data_size, data_file) != data_size)) {
    write_ok = false;
  }
  if (fclose(data_file) != 0) {
    write_ok = false;
  }
  if (!write_ok) {
    LOGE("failed to write tensor dump file: %s\n", data_path.c_str());
    return;
  }

  std::string metadata_path = base_path + ".txt";
  FILE* metadata_file = fopen(metadata_path.c_str(), "w");
  if (!metadata_file) {
    LOGE("failed to open tensor dump metadata: %s, error=%s\n",
           metadata_path.c_str(), strerror(errno));
    return;
  }

  fprintf(metadata_file, "callback=%s\n", callback_name);
  fprintf(metadata_file, "call_index=%llu\n", (unsigned long long)call_index);
  fprintf(metadata_file, "tensor_index=%u\n", tensor_index);
  fprintf(metadata_file, "name=%s\n", tensor_name ? tensor_name : "");
  fprintf(metadata_file, "bytes=%zu\n", data_size);
  fprintf(metadata_file, "data_file=%s\n", data_path.c_str());
  if (attr) {
    fprintf(metadata_file, "attr_index=%u\n", attr->index);
    fprintf(metadata_file, "dtype=%d\n", (int)attr->dtype);
    fprintf(metadata_file, "layout=%d\n", (int)attr->layout);
    fprintf(metadata_file, "n_dims=%u\n", attr->n_dims);
    fprintf(metadata_file, "shape=");
    for (uint32_t i = 0; i < attr->n_dims; ++i) {
      fprintf(metadata_file, "%s%u", i == 0 ? "" : ",", attr->shape[i]);
    }
    fprintf(metadata_file, "\n");
    fprintf(metadata_file, "n_elems=%u\n", attr->n_elems);
    fprintf(metadata_file, "aligned_size=%llu\n",
            (unsigned long long)attr->aligned_size);
    fprintf(metadata_file, "n_stride=%u\n", attr->n_stride);
    fprintf(metadata_file, "stride=");
    for (uint32_t i = 0; i < attr->n_stride; ++i) {
      fprintf(metadata_file, "%s%llu", i == 0 ? "" : ",",
              (unsigned long long)attr->stride[i]);
    }
    fprintf(metadata_file, "\n");
  }
  if (extra_metadata && extra_metadata[0] != '\0') {
    fprintf(metadata_file, "%s\n", extra_metadata);
  }
  fclose(metadata_file);

  VLOG("[tensor_dump] %s\n", data_path.c_str());
}

static void dump_embed_callback_tensor(const void* embed, uint64_t num_tokens,
                                       uint64_t len, int embedding_dim)
{
  if (!g_tensor_dump_enabled) {
    return;
  }

  std::lock_guard<std::mutex> lock(g_tensor_dump_mutex);
  uint64_t call_index = g_embed_dump_count++;
  char extra_metadata[256];
  snprintf(extra_metadata, sizeof(extra_metadata),
           "dtype=1\nshape=%llu,%d\nnum_tokens=%llu\nembedding_dim=%d",
           (unsigned long long)num_tokens,
           embedding_dim,
           (unsigned long long)num_tokens,
           embedding_dim);
  dump_tensor_data_locked("embed_callback", call_index, 0, "embedding",
                          embed, (size_t)len, nullptr, extra_metadata);
}

static void dump_input_callback_tensors(rknn3_tensor* input_tensors,
                                         uint32_t n_input_tensors,
                                         LLMInputCallbackParam param)
{
  if (!g_tensor_dump_enabled) {
    return;
  }

  std::lock_guard<std::mutex> lock(g_tensor_dump_mutex);
  uint64_t call_index = g_input_dump_count++;
  for (uint32_t i = 0; i < n_input_tensors; ++i) {
    const rknn3_tensor_attr* attr = input_tensors[i].attr;
    const rknn3_tensor_mem* mem = input_tensors[i].mem;
    char fallback_name[32];
    snprintf(fallback_name, sizeof(fallback_name), "tensor_%u", i);
    const char* tensor_name = attr ? attr->name : fallback_name;

    size_t data_size = attr ? (size_t)attr->aligned_size : 0;
    if (mem && mem->size > 0 && (data_size == 0 || mem->size < data_size)) {
      data_size = (size_t)mem->size;
    }

    char extra_metadata[256];
    snprintf(extra_metadata, sizeof(extra_metadata),
             "pos=%d\nshape_id=%d\nnum_tokens=%d\nmrope_pos=%d\n"
             "mrope_start=%d\nsystem_prompt_seqlen=%d\n"
             "valid_system_prompt_seqlen=%d\nmax_position_embeddings=%d",
             param.pos,
             param.shape_id,
             param.num_tokens,
             param.mrope_pos,
             param.mrope_start,
             param.system_prompt_seqlen,
             param.valid_system_prompt_seqlen,
             param.max_position_embeddings);
    dump_tensor_data_locked("input_callback", call_index, i, tensor_name,
                            mem ? mem->virt_addr : nullptr, data_size,
                            attr, extra_metadata);
  }
}

static const TensorBlob* pick_embed_tensor(const std::vector<TensorBlob>& tensors)
{
  if (tensors.empty()) {
    return nullptr;
  }

  for (const auto& tensor : tensors) {
    if (name_contains(tensor.attr.name, "hidden") || name_contains(tensor.attr.name, "last_hidden") ||
        name_contains(tensor.attr.name, "output")) {
      return &tensor;
    }
  }

  return &tensors.front();
}

static int tokenizer_callback(void* userdata, const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max)
{
  Tokenizer* tokenizer = (Tokenizer*)userdata;
  if (!tokenizer || !text || !tokens || n_tokens_max <= 0) {
    return -1;
  }

  int n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);
  VLOG("[tokenizer_callback] text=%s, text_len=%d, n_tokens=%d\n", text, text_len, n_tokens);
  if (n_tokens <= 0) {
    LOGE("tokenizer failed for input text\n");
  }
  return n_tokens;
}

static int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len)
{
  EmbedCallbackContext* ctx = (EmbedCallbackContext*)userdata;
  embedding_info* info = ctx ? ctx->embed_info : nullptr;
  if (!info || !tokens || !embed || info->embedding_dim <= 0 || !info->embedding_data) {
    return -1;
  }

  if (len != num_tokens * (uint64_t)info->embedding_dim * sizeof(float16)) {
    LOGE("invalid embed buffer size\n");
    return -1;
  }

  for (uint64_t n = 0; n < num_tokens; ++n) {
    memcpy((unsigned char*)embed + n * info->embedding_dim * sizeof(float16),
           info->embedding_data + tokens[n] * info->embedding_dim,
           info->embedding_dim * sizeof(float16));
  }

  dump_embed_callback_tensor(embed, num_tokens, len, info->embedding_dim);

  // 统计 stage0 输入 token 数，用于 prefill 性能统计。
  if (ctx->pipeline && !ctx->pipeline->slots.empty()) {
    StageSlot& slot = *ctx->pipeline->slots.front();
    std::lock_guard<std::mutex> lock(slot.mutex);
    slot.expected_tokens += num_tokens;
    VLOG("[embed_callback] num_tokens=%llu, total=%llu, token_id=%d\n",
    (unsigned long long)num_tokens, (unsigned long long)slot.expected_tokens, tokens[num_tokens - 1]);
  }

  return 0;
}

static int result_callback(void* userdata, RKLLMResult* result, LLMCallState state)
{
  LastStageResultState* result_state = (LastStageResultState*)userdata;
  Tokenizer* tokenizer = result_state ? result_state->tokenizer : nullptr;

  if (state == RKLLM_RUN_NORMAL && result && tokenizer) {
    int32_t next_token = -1;
    if (result->num_tokens > 0) {
      next_token = result->token_ids[result->num_tokens - 1];
      std::lock_guard<std::mutex> lock(result_state->mutex);
      result_state->next_token = next_token;
      result_state->has_token = true;
    }

    if (g_performance_mode) {
      return 0;
    }

    std::string piece;
    if (result->num_tokens == 1) {
      piece = tokenizer->TokenToPiece(result->token_ids[0]);
    } else {
      piece = tokenizer->Decode(result->token_ids, result->num_tokens);
    }
    VLOG("[result_callback] %s, next_token=%d\n", piece.c_str(), next_token);
    printf("%s", piece.c_str());
    fflush(stdout);
  }

  return 0;
}

static int stage_output_callback(void* userdata, rknn3_tensor* output_tensors, uint32_t n_output_tensors, LLMOutputCallbackState state)
{
  auto* cb_ctx = reinterpret_cast<StageCallbackContext*>(userdata);
  if (!cb_ctx || !cb_ctx->pipeline || cb_ctx->stage_index >= cb_ctx->pipeline->slots.size()) {
    return -1;
  }

  StageSlot& slot = *cb_ctx->pipeline->slots[cb_ctx->stage_index];

  VLOG("[Stage %zu] output_callback: state=%d, n_outputs=%u\n", cb_ctx->stage_index, state, n_output_tensors);

  StageBatch batch;
  batch.tensors.reserve(n_output_tensors);

  for (uint32_t i = 0; i < n_output_tensors; ++i) {
    if (!output_tensors[i].attr || !output_tensors[i].mem || !output_tensors[i].mem->virt_addr) {
      continue;
    }

    TensorBlob blob;
    blob.attr = *output_tensors[i].attr;
    blob.data.resize((size_t)blob.attr.aligned_size);
    memcpy(blob.data.data(), output_tensors[i].mem->virt_addr, blob.data.size());
    dump_tensor_blob(blob, i, "  [captured]");
    batch.tensors.push_back(std::move(blob));
  }

  if (!batch.tensors.empty()) {
    {
      std::lock_guard<std::mutex> lock(slot.mutex);
      uint64_t remaining_tokens = slot.expected_tokens > slot.emitted_tokens
                                      ? slot.expected_tokens - slot.emitted_tokens
                                      : 0;
      if (slot.active_input_tokens > 0) {
        batch.n_tokens = slot.active_input_tokens;
      } else if (state == RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED) {
        batch.n_tokens = remaining_tokens;
      } else {
        batch.n_tokens = remaining_tokens > g_bucket_size ? g_bucket_size : remaining_tokens;
      }

      if (batch.n_tokens == 0) {
        const TensorBlob* embed = pick_embed_tensor(batch.tensors);
        if (embed && embed->attr.n_elems > 0 && cb_ctx->embedding_dim > 0) {
          batch.n_tokens = embed->attr.n_elems / (uint64_t)cb_ctx->embedding_dim;
        }
      }
      slot.emitted_tokens += batch.n_tokens;
      slot.batches.push_back(std::move(batch));
    }
    slot.cv.notify_one();
  }

  return 0;
}

static int input_callback(void* userdata, rknn3_tensor* input_tensors, uint32_t n_input_tensors,
                          LLMInputCallbackParam param)
{
  InputCbUserdata* cb_data = (InputCbUserdata*)userdata;

  for (uint32_t i = 0; i < n_input_tensors; ++i) {
    int cache_index = find_rope_cache_index(cb_data->rope_format, input_tensors[i].attr->name);
    if (cache_index < 0) {
      continue;
    }

    const rope_cache_tensor* cache      = &cb_data->rope_caches[cache_index];
    const size_t             elem_sz    = get_dtype_elem_size(cache->dtype);
    const int                C1         = cache->shape[1];
    const size_t             c2_bytes   = (size_t)cache->shape[4] * elem_sz;
    const size_t             src_stride = (size_t)cache->shape[3] * c2_bytes;
    const size_t             dst_stride = (size_t)input_tensors[i].attr->shape[3] * c2_bytes;
    // 取 min 防止 src 越界读取
    const size_t             copy_stride = src_stride < dst_stride ? src_stride : dst_stride;
    const uint8_t*           src = (const uint8_t*)cache->data + (size_t)param.pos * c2_bytes;
    uint8_t*                 dst = (uint8_t*)input_tensors[i].mem->virt_addr;
    for (int c1 = 0; c1 < C1; c1++, src += src_stride, dst += dst_stride) {
      memcpy(dst, src, copy_stride);
    }
  }

  // Dump after the callback has filled the rope/input buffers.
  dump_input_callback_tensors(input_tensors, n_input_tensors, param);

  return 0;
}

static int load_safetensors(const char* path, rope_cache_tensor caches[4],
                            RopeCacheFormat* format_out,
                            int* fd_out, void** mmap_base_out, size_t* mmap_size_out)
{
  int         fd          = -1;
  void*       map         = MAP_FAILED;
  uint64_t    header_size = 0;
  struct stat st;
  int         ret         = -1;

  *format_out = RopeCacheFormat::NONE;

  fd = open(path, O_RDONLY);
  if (fd < 0) {
    LOGE("Failed to open safetensors file: %s\n", path);
    goto err;
  }
  if (fstat(fd, &st) < 0) {
    LOGE("Failed to stat safetensors file: %s\n", path);
    goto err;
  }

  if (read(fd, &header_size, 8) != 8) {
    LOGE("Failed to read safetensors header size\n");
    goto err;
  }
  if (header_size == 0 || header_size > (uint64_t)st.st_size - 8) {
    LOGE("Invalid safetensors header size: %" PRIu64 "\n", header_size);
    goto err;
  }

  map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    LOGE("Failed to mmap safetensors file: %s\n", path);
    goto err;
  }

  {
    const char*    json_ptr  = (const char*)map + 8;
    const uint8_t* data_base = (const uint8_t*)map + 8 + header_size;
    try {
      nlohmann::json j = nlohmann::json::parse(json_ptr, json_ptr + header_size);

      nlohmann::json meta_index = nlohmann::json::parse(
          j.at("__metadata__").at("index").get<std::string>());

      const char* const* rope_names = nullptr;
      int                n_rope_caches = 0;
      RopeCacheFormat    rope_format = RopeCacheFormat::NONE;
      auto has_rope_tensor = [&meta_index, &j](const char* name) {
        return meta_index.find(name) != meta_index.end() && j.find(name) != j.end();
      };

      bool has_gemma4_names = true;
      for (int i = 0; i < 4; ++i) {
        if (!has_rope_tensor(GEMMA4_ROPE_CACHE_NAMES[i])) {
          has_gemma4_names = false;
          break;
        }
      }
      bool has_qwen35_names = true;
      for (int i = 0; i < 2; ++i) {
        if (!has_rope_tensor(QWEN35_ROPE_CACHE_NAMES[i])) {
          has_qwen35_names = false;
          break;
        }
      }

      if (has_gemma4_names) {
        rope_names = GEMMA4_ROPE_CACHE_NAMES;
        n_rope_caches = 4;
        rope_format = RopeCacheFormat::GEMMA4;
      } else if (has_qwen35_names) {
        rope_names = QWEN35_ROPE_CACHE_NAMES;
        n_rope_caches = 2;
        rope_format = RopeCacheFormat::QWEN35;
      } else {
        LOGE("safetensors must contain either Gemma4 rope caches ");
        LOGE("(rope_cos_cache_0/1, rope_sin_cache_0/1) or Qwen3.5 ");
        LOGE("rope caches (rope_cos_cache, rope_sin_cache)\n");
        ret = -1;
      }

      ret = rope_format == RopeCacheFormat::NONE ? -1 : 0;
      for (int i = 0; i < n_rope_caches && ret == 0; i++) {
        const auto& meta_t = meta_index.at(rope_names[i]);
        int         dtype  = meta_t.at("dtype").get<int>();
        int         layout = meta_t.at("layout").get<int>();

        const auto& t      = j.at(rope_names[i]);
        auto shape_v   = t.at("shape").get<std::vector<int>>();
        auto offsets_v = t.at("data_offsets").get<std::vector<int64_t>>();

        int n_dims = (int)shape_v.size();
        if (n_dims != 5 || layout != 3) {
          LOGE("Tensor '%s': expected 5-D NC1HWC2 (layout=%d, n_dims=%d)\n",
                 rope_names[i], layout, n_dims);
          ret = -1;
          break;
        }
        caches[i].data   = (void*)(data_base + offsets_v[0]);
        caches[i].n_dims = n_dims;
        caches[i].dtype  = dtype;
        caches[i].layout = layout;
        for (int d = 0; d < n_dims; d++) caches[i].shape[d] = shape_v[d];
        printf("Loaded %-24s  dtype=%-2d  shape=[%d,%d,%d,%d,%d]\n",
               rope_names[i], dtype,
               caches[i].shape[0], caches[i].shape[1], caches[i].shape[2],
               caches[i].shape[3], caches[i].shape[4]);
      }
      if (ret == 0) {
        *format_out = rope_format;
      }
    } catch (const nlohmann::json::exception& e) {
      LOGE("Failed to parse safetensors JSON: %s\n", e.what());
      ret = -1;
    }
  }

err:
  if (ret != 0) {
    if (map != MAP_FAILED) munmap(map, (size_t)st.st_size);
    if (fd >= 0) close(fd);
    return ret;
  }
  *fd_out        = fd;
  *mmap_base_out = map;
  *mmap_size_out = (size_t)st.st_size;
  return 0;
}

static int init_tokenizer_and_embedding(const char* tokenizer_path, const char* embedding_path, VocabInfo* vocab_info,
                                        Tokenizer** tokenizer, embedding_info* embed_info, struct stat* emb_st)
{
  *tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
  (*tokenizer)->GetVocabInfo(vocab_info);

  embed_info->fd = open(embedding_path, O_RDONLY);
  if (embed_info->fd == -1) {
    LOGE("Failed to open embedding file: %s\n", embedding_path);
    delete *tokenizer;
    *tokenizer = nullptr;
    return -1;
  }

  if (fstat(embed_info->fd, emb_st) == -1) {
    LOGE("Failed to get embedding file size\n");
    close(embed_info->fd);
    embed_info->fd = -1;
    delete *tokenizer;
    *tokenizer = nullptr;
    return -1;
  }

  embed_info->embedding_data = (float16*)mmap(NULL, emb_st->st_size, PROT_READ, MAP_PRIVATE, embed_info->fd, 0);
  if (embed_info->embedding_data == MAP_FAILED) {
    LOGE("Failed to mmap embedding file\n");
    embed_info->embedding_data = nullptr;
    close(embed_info->fd);
    embed_info->fd = -1;
    delete *tokenizer;
    *tokenizer = nullptr;
    return -1;
  }

  embed_info->vocab_size = vocab_info->vocab_size;
  embed_info->embedding_dim = (emb_st->st_size / vocab_info->vocab_size) / sizeof(float16);
  return 0;
}

static int init_output_tensors(StageRuntime& stage)
{
  rknn3_input_output_num io_num;
  memset(&io_num, 0, sizeof(io_num));
  int ret = rknn3_query(stage.ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN3_SUCCESS) {
    LOGE("[%s] query io num failed, ret=%d\n", stage.name.c_str(), ret);
    return -1;
  }

  stage.n_output_tensors = (int)io_num.n_output;
  stage.output_tensors = (rknn3_tensor*)calloc(io_num.n_output, sizeof(rknn3_tensor));
  if (!stage.output_tensors) {
    return -1;
  }

  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    stage.output_tensors[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
    if (!stage.output_tensors[i].attr) {
      // 释放前序已分配的 attr 和 output_tensors
      for (uint32_t j = 0; j < i; ++j) {
        if (stage.output_tensors[j].mem) {
          rknn3_destroy_mem(stage.ctx, stage.output_tensors[j].mem);
          stage.output_tensors[j].mem = nullptr;
        }
        free(stage.output_tensors[j].attr);
        stage.output_tensors[j].attr = nullptr;
      }
      free(stage.output_tensors);
      stage.output_tensors = nullptr;
      stage.n_output_tensors = 0;
      return -1;
    }
    memset(stage.output_tensors[i].attr, 0, sizeof(rknn3_tensor_attr));
    stage.output_tensors[i].attr->index = i;

    ret = rknn3_query(stage.ctx, RKNN3_QUERY_OUTPUT_ATTR, stage.output_tensors[i].attr, sizeof(rknn3_tensor_attr));
    if (ret != RKNN3_SUCCESS) {
      LOGE("[%s] query output attr[%u] failed, ret=%d\n", stage.name.c_str(), i, ret);
      // 释放前序已分配的资源
      for (uint32_t j = 0; j < i; ++j) {
        if (stage.output_tensors[j].mem) {
          rknn3_destroy_mem(stage.ctx, stage.output_tensors[j].mem);
          stage.output_tensors[j].mem = nullptr;
        }
        free(stage.output_tensors[j].attr);
        stage.output_tensors[j].attr = nullptr;
      }
      free(stage.output_tensors[i].attr);
      stage.output_tensors[i].attr = nullptr;
      free(stage.output_tensors);
      stage.output_tensors = nullptr;
      stage.n_output_tensors = 0;
      return -1;
    }

    stage.output_tensors[i].mem = rknn3_create_mem(stage.ctx,
                                                    stage.output_tensors[i].attr->aligned_size,
                                                    stage.output_tensors[i].attr->core_id,
                                                    RKNN3_FLAG_MEMORY_CACHEABLE);
    if (!stage.output_tensors[i].mem) {
      LOGE("[%s] create output mem[%u] failed\n", stage.name.c_str(), i);
      // 释放前序已分配的资源
      for (uint32_t j = 0; j < i; ++j) {
        if (stage.output_tensors[j].mem) {
          rknn3_destroy_mem(stage.ctx, stage.output_tensors[j].mem);
          stage.output_tensors[j].mem = nullptr;
        }
        free(stage.output_tensors[j].attr);
        stage.output_tensors[j].attr = nullptr;
      }
      free(stage.output_tensors[i].attr);
      stage.output_tensors[i].attr = nullptr;
      free(stage.output_tensors);
      stage.output_tensors = nullptr;
      stage.n_output_tensors = 0;
      return -1;
    }
  }

  return 0;
}

static int query_ext_input_indices(StageRuntime& stage)
{
  rknn3_input_output_num io_num;
  memset(&io_num, 0, sizeof(io_num));
  int ret = rknn3_query(stage.ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN3_SUCCESS) {
    LOGE("[%s] query io num for ext inputs failed, ret=%d\n", stage.name.c_str(), ret);
    return -1;
  }

  int n_ext = 0;
  for (uint32_t i = 0; i < io_num.n_input; ++i) {
    rknn3_tensor_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.index = i;
    ret = rknn3_query(stage.ctx, RKNN3_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    if (ret != RKNN3_SUCCESS) {
      LOGE("[%s] query input attr[%u] failed, ret=%d\n", stage.name.c_str(), i, ret);
      return -1;
    }
    if (strcmp(attr.name, "per_layer_inputs") == 0) {
      n_ext++;
    } else if (strstr(attr.name, "rope_cos_cache") || strstr(attr.name, "rope_sin_cache")) {
      n_ext++;
    }
  }

  if (n_ext == 0) {
    return 0;
  }

  stage.ext_input_indices = (int*)malloc(n_ext * sizeof(int));
  if (!stage.ext_input_indices) {
    LOGE("[%s] malloc ext_input_indices failed\n", stage.name.c_str());
    return -1;
  }
  stage.n_ext_inputs = 0;

  for (uint32_t i = 0; i < io_num.n_input; ++i) {
    rknn3_tensor_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.index = i;
    ret = rknn3_query(stage.ctx, RKNN3_QUERY_INPUT_ATTR, &attr, sizeof(attr));
    if (ret != RKNN3_SUCCESS) {
      return -1;
    }
    if (strcmp(attr.name, "per_layer_inputs") == 0) {
      stage.ext_input_indices[stage.n_ext_inputs++] = (int)i;
    } else if (strstr(attr.name, "rope_cos_cache") || strstr(attr.name, "rope_sin_cache")) {
      stage.ext_input_indices[stage.n_ext_inputs++] = (int)i;
    }
  }

  printf("[%s] found %d ext input tensors (per_layer_inputs/rope_caches)\n", stage.name.c_str(), stage.n_ext_inputs);
  return 0;
}

static bool init_stage(StageRuntime& stage, PipelineState& pipeline, size_t stage_idx, const char* device_id,
                       const char* model_path, const char* weight_path, const rknn3_llm_param& session_param,
                       uint32_t run_core_mask,
                       const char* custom_op_path,
                       bool is_last_stage, Tokenizer* tokenizer,
                       EmbedCallbackContext* embed_ctx,
                       InputCbUserdata* input_cb_data)
{
  stage.model_path = model_path;
  stage.weight_path = weight_path;
  stage.callback_ctx.pipeline = &pipeline;
  stage.callback_ctx.stage_index = stage_idx;

  printf("[%s] init stage: model=%s, weight=%s, device_id=%s\n",
         stage.name.c_str(), model_path, weight_path, device_id);

  rknn3_init_extend ext;
  memset(&ext, 0, sizeof(ext));
  ext.device_id = const_cast<char*>(device_id);

  int ret = rknn3_init(&stage.ctx, &ext);
  if (ret != RKNN3_SUCCESS) {
    LOGE("[%s] rknn3_init failed, ret=%d\n", stage.name.c_str(), ret);
    return false;
  }

  ret = rknn3_load_model_from_path(stage.ctx, stage.model_path.c_str(), stage.weight_path.c_str());
  if (ret != RKNN3_SUCCESS) {
    LOGE("[%s] rknn3_load_model_from_path failed, ret=%d\n", stage.name.c_str(), ret);
    destroy_stage(stage);
    return false;
  }

  rknn3_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.run_core_mask = run_core_mask;
  ret = rknn3_model_init(stage.ctx, &cfg);
  if (ret != RKNN3_SUCCESS) {
    LOGE("[%s] rknn3_model_init failed, ret=%d\n", stage.name.c_str(), ret);
    destroy_stage(stage);
    return false;
  }

  if (custom_op_path && custom_op_path[0] != '\0') {
    printf("[%s] registering custom ops plugin: %s\n",
           stage.name.c_str(), custom_op_path);
    // Custom-op plugins are context-local. Register the plugin for every stage
    // after model_init and before querying/creating the LLM session.
    ret = rknn3_register_custom_ops_plugins(stage.ctx, custom_op_path, strlen(custom_op_path));
    if (ret != RKNN3_SUCCESS) {
      LOGE("[%s] rknn3_register_custom_ops_plugins failed, ret=%d\n",
             stage.name.c_str(), ret);
      destroy_stage(stage);
      return false;
    }
    printf("[%s] rknn3_register_custom_ops_plugins success\n", stage.name.c_str());
  }

  // rknn3_set_input_name_alias(stage.ctx, "inputs_embeds", "input_embeds");
  // rknn3_set_input_name_alias(stage.ctx, "hidden_states", "input_embeds");
  rknn3_llm_config llm_cfg;
  memset(&llm_cfg, 0, sizeof(llm_cfg));
  ret = rknn3_query(stage.ctx, RKNN3_QUERY_LLM_CONFIG, &llm_cfg, sizeof(llm_cfg));
  if (ret != RKNN3_SUCCESS) {
    LOGE("[%s] query llm config failed, ret=%d\n", stage.name.c_str(), ret);
    destroy_stage(stage);
    return false;
  }
  stage.embedding_dim = (int32_t)llm_cfg.embedding_dim;
  stage.vocab_size = (int32_t)llm_cfg.vocab_size;
  stage.max_ctx_len = (int32_t)llm_cfg.max_ctx_len;
  stage.callback_ctx.embedding_dim = stage.embedding_dim;

  if (init_output_tensors(stage) != 0) {
    LOGE("[%s] init output tensors failed\n", stage.name.c_str());
    destroy_stage(stage);
    return false;
  }

  rknn3_llm_param local_param = session_param;

  stage.session = rknn3_session_init(stage.ctx, &local_param, 1);
  if (!stage.session) {
    LOGE("[%s] rknn3_session_init failed\n", stage.name.c_str());
    destroy_stage(stage);
    return false;
  }
  rknn3_session_set_chat_template(stage.session, "", "", "");
  RKLLMCallback callback;
  memset(&callback, 0, sizeof(callback));

  if (!is_last_stage) {
    callback.output_callback = stage_output_callback;
    callback.output_userdata = &stage.callback_ctx;
    callback.output_tensors = stage.output_tensors;
    callback.n_output_tensors = stage.n_output_tensors;
  }

  callback.tokenizer_callback = tokenizer_callback; 
  callback.tokenizer_userdata = tokenizer;
  callback.embed_callback = embed_callback;
  callback.embed_userdata = embed_ctx;
  callback.result_callback = result_callback;
  callback.result_userdata = &g_last_stage_result;

  // 如果提供了 safetensors (rope caches)，则注册 input_callback 并设置 ext input indices
  if (input_cb_data && input_cb_data->rope_mmap_base) {
    if (query_ext_input_indices(stage) != 0) {
      LOGE("[%s] query_ext_input_indices failed\n", stage.name.c_str());
      destroy_stage(stage);
      return false;
    }
    callback.input_callback = input_callback;
    callback.input_userdata = input_cb_data;
    callback.input_tensors_index = stage.ext_input_indices;
    callback.n_input_tensors     = stage.n_ext_inputs;
    for (int i = 0; i < stage.n_ext_inputs; ++i) {
      VLOG("[%s] ext input tensor[%d] index=%d\n", stage.name.c_str(), i, stage.ext_input_indices[i]);
    }
  }

  ret = rknn3_session_set_callback(stage.session, &callback);
  if (ret != RKNN3_SUCCESS) {
    LOGE("[%s] rknn3_session_set_callback failed, ret=%d\n", stage.name.c_str(), ret);
    destroy_stage(stage);
    return false;
  }

  printf("[%s] init done: embedding_dim=%d vocab_size=%d max_ctx_len=%d outputs=%d\n",
         stage.name.c_str(), stage.embedding_dim, stage.vocab_size, stage.max_ctx_len, stage.n_output_tensors);
  return true;
}

static bool wait_stage_batch(StageSlot& slot, StageBatch* batch)
{
  std::unique_lock<std::mutex> lock(slot.mutex);
  slot.cv.wait(lock, [&slot]() {
    return !slot.batches.empty() || slot.producer_done || slot.failed;
  });

  if (slot.failed || slot.batches.empty()) {
    return false;
  }
  *batch = std::move(slot.batches.front());
  slot.batches.pop_front();
  return true;
}

static void run_stage_worker(size_t stage_idx, std::vector<StageRuntime>& stages,
                             PipelineState& pipeline, const rknn3_llm_infer_param& infer_param,
                             InferencePhase phase)
{
  StageRuntime& stage = stages[stage_idx];
  StageSlot& input_slot = *pipeline.slots[stage_idx - 1];
  StageSlot* output_slot = stage_idx + 1 < stages.size() ? pipeline.slots[stage_idx].get() : nullptr;
  bool is_last_stage = (stage_idx == stages.size() - 1);
  uint64_t consumed_tokens = 0;

  while (true) {
    StageBatch batch;
    if (!wait_stage_batch(input_slot, &batch)) {
      std::lock_guard<std::mutex> lock(input_slot.mutex);
      if (!input_slot.failed && input_slot.producer_done && input_slot.batches.empty()) {
        break;
      }
      if (output_slot) {
        close_stage_slot(*output_slot);
      }
      return;
    }

    const TensorBlob* embed = pick_embed_tensor(batch.tensors);
    if (!embed || batch.n_tokens == 0 || stage.embedding_dim <= 0) {
      LOGE("[stage%zu] invalid pipeline batch\n", stage_idx);
      fail_pipeline(pipeline);
      return;
    }

    size_t embed_bytes = batch.n_tokens * (uint64_t)stage.embedding_dim * sizeof(float16);
    if (embed_bytes > embed->data.size()) {
      LOGE("[stage%zu] embed buffer too small: need=%zu, got=%zu\n",
             stage_idx, embed_bytes, embed->data.size());
      fail_pipeline(pipeline);
      return;
    }

    if (output_slot) {
      std::lock_guard<std::mutex> lock(output_slot->mutex);
      output_slot->expected_tokens += batch.n_tokens;
      output_slot->active_input_tokens = batch.n_tokens;
    }

    rknn3_llm_input embed_input;
    memset(&embed_input, 0, sizeof(embed_input));
    embed_input.input_type = RKNN3_LLM_INPUT_EMBED;
    embed_input.llm_input.embed = (float16*)embed->data.data();
    embed_input.llm_input.n_tokens = batch.n_tokens;
    rknn3_llm_infer_param local_param = infer_param;
    if (is_last_stage) {
      // 最后一段：根据 embed_callback 中统计的总 token 数判断是否最后一桶
      uint64_t total_tokens;
      {
        std::lock_guard<std::mutex> lock(pipeline.slots[0]->mutex);
        total_tokens = pipeline.slots[0]->expected_tokens;
      }
      local_param.disable_sampling = (consumed_tokens + batch.n_tokens < total_tokens);
    }
    consumed_tokens += batch.n_tokens;
    VLOG("[stage%zu] consume batch: n_tokens=%llu, bytes=%zu, disable_sampling=%d\n",
         stage_idx, (unsigned long long)batch.n_tokens, embed_bytes,
         (int)local_param.disable_sampling);
    timeval run_start;
    timeval run_end;
    gettimeofday(&run_start, NULL);
    int ret = rknn3_session_run(stage.session, &embed_input, 1, &local_param);
    gettimeofday(&run_end, NULL);
    record_stage_performance(stage, phase, batch.n_tokens,
                             elapsed_us(run_start, run_end) / 1e3);

    if (output_slot) {
      std::lock_guard<std::mutex> lock(output_slot->mutex);
      output_slot->active_input_tokens = 0;
    }
    if (ret != RKNN3_SUCCESS) {
      LOGE("[stage%zu] run failed ret=%d\n", stage_idx, ret);
      fail_pipeline(pipeline);
      return;
    }
  }

  if (output_slot) {
    close_stage_slot(*output_slot);
  }
}

static bool run_pipeline_once(std::vector<StageRuntime>& stages, PipelineState& pipeline,
                              const char* prompt, const std::vector<int32_t>* input_tokens,
                              InferencePhase phase, uint64_t* stage0_input_tokens)
{
  if (stages.empty()) {
    return false;
  }

  rknn3_llm_infer_param infer_param;
  memset(&infer_param, 0, sizeof(infer_param));
  infer_param.keep_history = 1;
  infer_param.max_new_tokens = 1;
  // 使用 prefill_only 模式进行单步推理
  infer_param.prefill_only = true;
  // 只有最后一段才进行采样，且prefill时只有最后一桶结束才采样
  infer_param.disable_sampling = true;
  if (stages.size() == 1) {
    // 单卡时 stage0 同时是最后一段，没有 worker 负责打开采样。
    infer_param.disable_sampling = false;
  }

  reset_pipeline(pipeline);
  reset_last_stage_result();

  rknn3_llm_input first_input;
  memset(&first_input, 0, sizeof(first_input));
  if (prompt) {
    first_input.input_type = RKNN3_LLM_INPUT_PROMPT;
    first_input.llm_input.prompt = prompt;
  } else {
    if (!input_tokens || input_tokens->empty()) {
      return false;
    }
    first_input.input_type = RKNN3_LLM_INPUT_TOKEN;
    first_input.llm_input.tokens = const_cast<int32_t*>(input_tokens->data());
    first_input.llm_input.n_tokens = input_tokens->size();
  }

  std::vector<std::thread> workers;
  workers.reserve(stages.size() - 1);
  for (size_t i = 1; i < stages.size(); ++i) {
    workers.emplace_back(run_stage_worker, i, std::ref(stages), std::ref(pipeline),
                         std::cref(infer_param), phase);
  }

  timeval stage0_start;
  timeval stage0_end;
  gettimeofday(&stage0_start, NULL);
  int ret = rknn3_session_run(stages[0].session, &first_input, 1, &infer_param);
  gettimeofday(&stage0_end, NULL);
  if (ret != RKNN3_SUCCESS) {
    LOGE("[stage0] run failed ret=%d\n", ret);
    fail_pipeline(pipeline);
  }
  close_stage_slot(*pipeline.slots[0]);

  for (auto& worker : workers) {
    worker.join();
  }

  uint64_t current_stage0_input_tokens = 0;
  {
    std::lock_guard<std::mutex> lock(pipeline.slots[0]->mutex);
    current_stage0_input_tokens = pipeline.slots[0]->expected_tokens;
  }
  record_stage_performance(stages[0], phase, current_stage0_input_tokens,
                           elapsed_us(stage0_start, stage0_end) / 1e3);

  if (stage0_input_tokens) {
    *stage0_input_tokens = current_stage0_input_tokens;
  }

  return ret == RKNN3_SUCCESS && !pipeline_failed(pipeline);
}

int main(int argc, char** argv)
{
  if (argc == 2 && strcmp(argv[1], "--help") == 0) {
    print_usage(argv[0]);
    return 0;
  }

  CommandLineOptions options;
  if (!parse_command_line(argc, argv, &options)) {
    print_usage(argv[0]);
    return -1;
  }

  const char* base_model_path = options.stage0_model;
  const char* base_weight_path = options.stage0_weight;
  const char* tokenizer_path = options.tokenizer;
  const char* embedding_path = options.embedding;
  const char* custom_op_path = options.custom_op_path;
  int32_t max_context_len = options.max_context_len;
  uint32_t run_core_mask = options.run_core_mask;
  g_stage_count = options.stage_count;
  g_bucket_size = options.bucket_size;
  g_performance_mode = options.performance_mode;

  uint64_t performance_input_length = options.performance_input_length;
  uint64_t performance_output_length = options.performance_output_length;
  const char* prompt_arg = options.prompt;
  std::string prompt_buf;
  const char* prompt = nullptr;
  if (prompt_arg) {
    // 如果以 .txt 结尾，读取文件内容作为 prompt
    size_t arg_len = strlen(prompt_arg);
    if (arg_len >= 4 && strcmp(prompt_arg + arg_len - 4, ".txt") == 0) {
      int fd = open(prompt_arg, O_RDONLY);
      if (fd < 0) {
        LOGE("Failed to open prompt file: %s\n", prompt_arg);
        return -1;
      }
      struct stat st;
      if (fstat(fd, &st) != 0) {
        LOGE("Failed to stat prompt file: %s\n", prompt_arg);
        close(fd);
        return -1;
      }
      prompt_buf.resize(st.st_size);
      ssize_t n = read(fd, &prompt_buf[0], st.st_size);
      close(fd);
      if (n != st.st_size) {
        LOGE("Failed to read prompt file: %s\n", prompt_arg);
        return -1;
      }
      prompt = prompt_buf.c_str();
      printf("Loaded prompt from file: %s (%lld bytes)\n", prompt_arg, (long long)st.st_size);
    } else {
      prompt = prompt_arg;
    }
  }
  if (!prompt) {
    prompt = "system\n You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n";
  }
  int max_new_tokens = options.max_new_tokens;
  g_verbose = options.verbose;
  g_ignore_eos = options.ignore_eos;
  const char* rope_path = options.rope_path;

  if (options.tensor_dump_dir && !configure_tensor_dump(options.tensor_dump_dir)) {
    return -1;
  }

  std::vector<std::string> ext_device_ids = options.device_ids;

  if (g_stage_count < 1) {
    LOGE("stage_count must be >= 1, got %zu\n", g_stage_count);
    return -1;
  }
  if (max_context_len <= 0) {
    LOGE("max_context_len must be > 0, got %d\n", max_context_len);
    return -1;
  }
  if (g_performance_mode) {
    uint64_t max_context_tokens = (uint64_t)max_context_len;
    uint64_t decode_input_tokens = performance_output_length - 1;
    if (performance_input_length > max_context_tokens ||
        decode_input_tokens > max_context_tokens - performance_input_length) {
      LOGE("performance lengths exceed max_context_len: input=%llu, output=%llu, max_context_len=%d\n",
             (unsigned long long)performance_input_length,
             (unsigned long long)performance_output_length,
             max_context_len);
      return -1;
    }
  }

  // 工具: 将 "model_seg0.rknn" 替换为 "model_segN.rknn"
  auto replace_seg_suffix = [](const std::string& base, size_t seg_idx) -> std::string {
    // 找到最后一个 _seg 出现的位置
    size_t pos = base.rfind("_seg");
    if (pos == std::string::npos) {
      // 没有 _seg 后缀，直接在 .rknn 前插入 _segN
      size_t dot = base.rfind(".rknn");
      if (dot != std::string::npos) {
        return base.substr(0, dot) + "_seg" + std::to_string(seg_idx) + ".rknn";
      }
      return base + "_seg" + std::to_string(seg_idx);
    }
    // 有 _seg 后缀，替换其中的数字
    size_t num_start = pos + 4; // 跳过 "_seg"
    size_t num_end = num_start;
    while (num_end < base.size() && isdigit(base[num_end])) {
      ++num_end;
    }
    return base.substr(0, num_start) + std::to_string(seg_idx) + base.substr(num_end);
  };

  // 生成所有段的模型 & 权重路径
  std::vector<std::string> model_paths(g_stage_count);
  std::vector<std::string> weight_paths(g_stage_count);
  for (size_t i = 0; i < g_stage_count; ++i) {
    model_paths[i] = replace_seg_suffix(base_model_path, i);
    weight_paths[i] = replace_seg_suffix(base_weight_path, i);
  }

  // 校验所有段文件存在
  bool all_exist = true;
  for (size_t i = 0; i < g_stage_count; ++i) {
    struct stat st;
    if (stat(model_paths[i].c_str(), &st) != 0) {
      LOGE("ERROR: model file not found: %s\n", model_paths[i].c_str());
      all_exist = false;
    }
    if (stat(weight_paths[i].c_str(), &st) != 0) {
      LOGE("ERROR: weight file not found: %s\n", weight_paths[i].c_str());
      all_exist = false;
    }
  }
  if (!all_exist) {
    return -1;
  }

  PipelineState pipeline(g_stage_count);
  std::vector<StageRuntime> stages(g_stage_count);
  for (size_t i = 0; i < stages.size(); ++i) {
    stages[i].name = "stage" + std::to_string(i);
  }

  Tokenizer* tokenizer = nullptr;
  VocabInfo vocab_info;
  memset(&vocab_info, 0, sizeof(vocab_info));
  embedding_info embed_info;
  struct stat emb_st;
  memset(&emb_st, 0, sizeof(emb_st));

  if (init_tokenizer_and_embedding(tokenizer_path, embedding_path, &vocab_info, &tokenizer, &embed_info, &emb_st) != 0) {
    return -1;
  }

  g_last_stage_result.tokenizer = tokenizer;
  reset_last_stage_result();

  rknn3_llm_param session_param;
  memset(&session_param, 0, sizeof(session_param));
  session_param.logits_name = (char*)"output";
  session_param.max_context_len = max_context_len;
  session_param.sampling_param.temperature = 1.0f;
  session_param.sampling_param.top_k = 1;
  session_param.sampling_param.top_p = 0.9f;
  session_param.sampling_param.repeat_penalty = 1.0f;
  session_param.sampling_param.frequency_penalty = 0.0f;
  session_param.sampling_param.presence_penalty = 0.0f;
  session_param.vocab_info.vocab_size = vocab_info.vocab_size;
  session_param.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
  session_param.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
  memcpy(session_param.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
  memcpy(session_param.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));
  session_param.vocab_info.linefeed_id = vocab_info.linefeed_id;
  session_param.vocab_info.ignore_eos_token = g_ignore_eos ? 1 : 0;

  EmbedCallbackContext embed_ctx;
  embed_ctx.embed_info = &embed_info;
  embed_ctx.pipeline = &pipeline;

  InputCbUserdata input_cb_data;
  memset(&input_cb_data, 0, sizeof(input_cb_data));

  // 自动查找设备，检查设备数是否足够
  rknn3_devices devs;
  memset(&devs, 0, sizeof(devs));
  int ret = rknn3_find_devices(&devs);
  if (ret != RKNN3_SUCCESS) {
    LOGE("find devices failed: ret=%d\n", ret);
    release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);
    return -1;
  }
  printf("found %d devices:\n", devs.n_devices);
  for (int i = 0; i < devs.n_devices; ++i) {
    printf("  [%d] type=%s, id=%s\n", i, devs.devices[i].type, devs.devices[i].id);
  }

  if (ext_device_ids.empty()) {
    // 未指定外部 device_id，自动分配
    if (devs.n_devices < (int)g_stage_count) {
      LOGE("auto-detect failed: found=%d devices, need=%zu\n", devs.n_devices, g_stage_count);
      release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);
      return -1;
    }
    printf("auto-assigning %zu devices\n", g_stage_count);
  } else if ((int)ext_device_ids.size() < (int)g_stage_count) {
    LOGE("not enough device_id arguments: need=%zu, got=%zu\n", g_stage_count, ext_device_ids.size());
    release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);
    return -1;
  } else {
    // 校验外部指定的 device_id 是否在可用设备列表中
    printf("using external device_id list:\n");
    for (const auto& id : ext_device_ids) {
      printf("  %s\n", id.c_str());
    }
  }

    // 如果提供了 rope 路径，加载 rope caches 并注册 input_callback
  if (rope_path && rope_path[0] != '\0') {
    printf("loading rope cache: %s\n", rope_path);
    if (load_safetensors(rope_path, input_cb_data.rope_caches,
                         &input_cb_data.rope_format,
                         &input_cb_data.rope_fd, &input_cb_data.rope_mmap_base,
                         &input_cb_data.rope_mmap_size) != 0) {
      LOGE("load_safetensors failed\n");
      release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);
      return -1;
    }
  }

  bool ok = true;
  for (size_t i = 0; i < stages.size(); ++i) {
    const char* device_id = ext_device_ids.empty() ? devs.devices[i].id : ext_device_ids[i].c_str();
    ok = init_stage(stages[i], pipeline, i, device_id,
                    model_paths[i].c_str(), weight_paths[i].c_str(), session_param, run_core_mask,
                    custom_op_path,
                    i + 1 == stages.size(), tokenizer, &embed_ctx, &input_cb_data);
    if (!ok) {
      break;
    }
  }
  if (!ok) {
    release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);
    return -1;
  }

  std::vector<int32_t> performance_input_tokens;
  if (g_performance_mode) {
    uint64_t required_context_tokens = performance_input_length + performance_output_length - 1;
    for (const auto& stage : stages) {
      if (stage.max_ctx_len > 0 && required_context_tokens > (uint64_t)stage.max_ctx_len) {
        LOGE("performance lengths exceed %s max context: required=%llu, max=%d\n",
               stage.name.c_str(),
               (unsigned long long)required_context_tokens,
               stage.max_ctx_len);
        release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);
        return -1;
      }
    }

    build_performance_input_tokens(vocab_info, performance_input_length,
                                   &performance_input_tokens);
    printf("\n=== Performance Test Mode ===\n");
    printf("Input tokens: %llu, output tokens: %llu\n",
           (unsigned long long)performance_input_length,
           (unsigned long long)performance_output_length);
  }

  printf("\n=== Prefill Pipeline ===\n");
  timeval prefill_start;
  timeval prefill_end;
  gettimeofday(&prefill_start, NULL);
  uint64_t prefill_tokens = 0;
  const char* prefill_prompt = g_performance_mode ? nullptr : prompt;
  const std::vector<int32_t>* prefill_input_tokens =
      g_performance_mode ? &performance_input_tokens : nullptr;
  if (!run_pipeline_once(stages, pipeline, prefill_prompt, prefill_input_tokens,
                         InferencePhase::PREFILL, &prefill_tokens)) {
    LOGE("prefill failed\n");
    release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);
    return -1;
  }
  gettimeofday(&prefill_end, NULL);

  int next_token = -1;
  get_last_stage_token(&next_token);

  printf("\n=== Decode Loop ===\n");
  if (next_token < 0) {
    LOGE("prefill did not return token from result_callback\n");
    release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);
    return -1;
  }
  timeval decode_start;
  timeval decode_end;
  gettimeofday(&decode_start, NULL);
  uint64_t decode_tokens = 0;
  uint64_t decode_steps = g_performance_mode
                              ? performance_output_length - 1
                              : (max_new_tokens > 0 ? (uint64_t)max_new_tokens : 0);
  for (uint64_t step = 0; step < decode_steps && next_token >= 0; ++step) {
    std::vector<int32_t> token_vec(1, next_token);
    VLOG("[Decode %llu] token=%d\n", (unsigned long long)(step + 1), next_token);

    if (!run_pipeline_once(stages, pipeline, nullptr, &token_vec,
                           InferencePhase::DECODE, nullptr)) {
      LOGE("decode step %llu failed\n", (unsigned long long)(step + 1));
      break;
    }

    if (!get_last_stage_token(&next_token)) {
      LOGE("decode step %llu did not return token from result_callback\n",
             (unsigned long long)(step + 1));
      break;
    }
    decode_tokens += 1;
    if (!g_performance_mode && !g_ignore_eos &&
        next_token == vocab_info.special_eos_id[0]) {
      VLOG("\ndecode step %llu reached EOS token\n", (unsigned long long)(step + 1));
      break;
    }
  }
  gettimeofday(&decode_end, NULL);

  float prefill_ms = elapsed_us(prefill_start, prefill_end) / 1e3f;
  float decode_ms  = elapsed_us(decode_start, decode_end) / 1e3f;
  if (g_performance_mode) {
    printf("\nPerformance Test Lengths: input=%llu/%llu, output=%llu/%llu\n",
           (unsigned long long)prefill_tokens,
           (unsigned long long)performance_input_length,
           (unsigned long long)(decode_tokens + 1),
           (unsigned long long)performance_output_length);
  }
  print_performance_statistics(prefill_tokens, prefill_ms, decode_tokens, decode_ms);
  print_stage_performance_statistics(stages);
  for (size_t i = 0; i < stages.size(); ++i) {
    rknn3_session_clear_kvcache(stages[i].session, RKNN3_KVCACHE_CLEAR_ALL, nullptr);
  }

  release_resources(stages, &input_cb_data, &embed_info, emb_st.st_size, tokenizer);

  return 0;
}
