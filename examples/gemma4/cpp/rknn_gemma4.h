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

#ifndef _RKNN_DEMO_GEMMA4_H_
#define _RKNN_DEMO_GEMMA4_H_

#ifdef __cplusplus

#include <stdint.h>
#include <stddef.h>

#include "common.h"
#include "rknn3_api.h"
#include "rknn_gemma4_llm.h"
#include "rknn_gemma4_audio.h"
#include "rknn_gemma4_audio_stream.h"
#include "rknn_gemma4_vision.h"
#include "Tokenizer.h"
#include "time_utils.h"

#define MAX_NEW_TOKENS 1024
#define MAX_CONTEXT_LEN 1024

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
  embedding_info    per_layer_embed;
  rope_cache_tensor rope_caches[4];
  int               rope_fd;
  void*             rope_mmap_base;
  size_t            rope_mmap_size;
} input_cb_userdata;

typedef struct {
    rknn_gemma4_llm_context llm;
    rknn_gemma4_audio_context audio;
    rknn_gemma4_audio_stream_context* audio_stream;
    rknn_gemma4_vision_context vision;
    bool enable_audio;
    bool audio_streaming;
    bool enable_vision;
    int n_internal_mems;
    rknn3_tensor_mem** internal_mems;
} rknn_gemma4_app_context;

int init_gemma4_model(rknn_gemma4_app_context* app_ctx,
                      const char* llm_model_path, const char* llm_weight_path,
                      const char* audio_model_path, const char* audio_weight_path,
                      bool enable_audio, bool audio_streaming, bool enable_vision,
                      rknn3_llm_param* params, int n_params,
                      RKLLMCallback& callback, uint32_t llm_core_mask,
                      uint32_t audio_core_mask, uint32_t vision_core_mask,
                      const char* per_layer_embed_path, const char* safetensors_path,
                      Tokenizer* tokenizer, embedding_info* token_embedding,
                      input_cb_userdata* input_cb_data,
                      const char* vision_model_path = NULL, const char* vision_weight_path = NULL);

int release_gemma4_model(rknn_gemma4_app_context* app_ctx);

int inference_gemma4_model(rknn_gemma4_app_context* app_ctx,
                           rknn3_llm_multimodal_tensor tensor,
                           audio_buffer_t* audio,
                           image_buffer_t* image,
                           float16* audio_embeds,
                           float16* image_embeds,
                           int32_t max_new_tokens,
                           rknn_perf_metrics_t* perf);

#endif

#endif // _RKNN_DEMO_GEMMA4_H_
