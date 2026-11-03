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

#ifndef _RKNN_DEMO_SPARK_H_
#define _RKNN_DEMO_SPARK_H_

#ifdef __cplusplus

#include <stdint.h>
#include <stddef.h>

#include "common.h"
#include "rknn3_api.h"
#include "rknn_spark_llm.h"
#include "Tokenizer.h"
#include "time_utils.h"

// Forward declarations
typedef struct {
  int      fd;
  float16* embedding_data;
  int      embedding_dim;
  int      vocab_size;
  size_t   data_size;
} embedding_info;

typedef struct {
  void* data;
  int   n_dims;    // number of valid dimensions in shape[]
  int   shape[5];  // full 5-D shape: [N, C1, H, W, C2] for NC1HWC2
  int   dtype;     // rknn3_tensor_type value
  int   layout;    // rknn3_tensor_layout value
} rope_cache_tensor;

static const char* ROPE_CACHE_NAMES[4] = {
    "rope_cos_cache_0", "rope_sin_cache_0",
    "rope_cos_cache_1", "rope_sin_cache_1"
};

typedef struct {
  rope_cache_tensor rope_caches[4];
  int               rope_fd;
  void*             rope_mmap_base;
  size_t            rope_mmap_size;
} input_cb_userdata;

typedef struct {
    rknn_spark_llm_context llm;
    int n_internal_mems;
    rknn3_tensor_mem** internal_mems;
} rknn_spark_app_context;

int init_spark_model(rknn_spark_app_context* app_ctx,
                      const char* llm_model_path, const char* llm_weight_path,
                      rknn3_llm_param* params, int n_params,
                      RKLLMCallback& callback, uint32_t llm_core_mask,
                      const char* safetensors_path,
                      Tokenizer* tokenizer, embedding_info* token_embedding,
                      input_cb_userdata* input_cb_data, bool enable_thinking);

int release_spark_model(rknn_spark_app_context* app_ctx);

int inference_spark_model(rknn_spark_app_context* app_ctx,
                           rknn3_llm_multimodal_tensor tensor,
                           int32_t max_new_tokens,
                           rknn_perf_metrics_t* perf);

#endif

#endif // _RKNN_DEMO_SPARK_H_