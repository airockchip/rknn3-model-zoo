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


#ifndef _RKNN_DEMO_QWEN3_5_VL_H_
#define _RKNN_DEMO_QWEN3_5_VL_H_

#include "rknn3_api.h"
#include "common.h"
#include "rknn_qwen3_5_vl_vision.h"
#include "rknn_qwen3_5_vl_llm.h"
#include "time_utils.h"

#ifdef ENABLE_SPEEDUP
#include "speedup.h"
#endif

// llm
extern const rknn3_sampling_params SAMPLE_PARAMS;

typedef struct {
    rknn_qwen3_5_vl_llm_context llm;
    rknn_qwen3_5_vl_vision_context vision;
    int n_internal_mems;
    rknn3_tensor_mem** internal_mems;
    uint32_t model_width;
    uint32_t model_height;
#ifdef ENABLE_SPEEDUP
    SpeedUPHandle speedup;
    RKLLMCallback base_callback;
#endif
} rknn_app_context_t;

int init_qwen3_5_vl_model(rknn_app_context_t* app_ctx, const char* llm_model_path, const char* llm_weight_path, const char* vision_model_path, const char* vision_weight_path, 
                        rknn3_llm_param* params, int n_params, RKLLMCallback* callback, uint32_t vision_core_mask, uint32_t llm_core_mask,
                        const char* vision_lora_weight_path = nullptr);

int release_qwen3_5_vl_model(rknn_app_context_t* app_ctx);

// Load the optional LLM adapter after Base inference. The Base session is
// destroyed first to avoid runtime transfer-buffer pressure.
int prepare_qwen3_5_vl_lora_session(rknn_app_context_t* app_ctx, const char* llm_lora_weight_path);

int inference_qwen3_5_vl_model_vision(rknn_app_context_t* app_ctx, image_buffer_t* img, float16* img_embeds, rknn_perf_metrics_t* perf, bool use_lora = false);
int inference_qwen3_5_vl_model_llm(rknn_app_context_t* app_ctx, rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf, bool use_lora = false, float speedup_ratio = 0.0f);

// SpeedUP mode: 1.0 single-image Auto, 0.0 disabled, (0, 1) Manual.
int inference_qwen3_5_vl_model(rknn_app_context_t* app_ctx, image_buffer_t* img, float16* img_embeds, rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf, float speedup_ratio = 1.0f);

#endif //_RKNN_DEMO_QWEN3_5_VL_H_
