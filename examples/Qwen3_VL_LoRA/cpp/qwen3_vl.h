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


#ifndef _RKNN_DEMO_QWEN3_VL_H_
#define _RKNN_DEMO_QWEN3_VL_H_

#include "rknn3_api.h"
#include "common.h"
#include "vision/rknn_qwen3_vl_vision.h"
#include "llm/rknn_qwen3_vl_llm.h"
#include "time_utils.h"

// llm
extern const rknn3_sampling_params SAMPLE_PARAMS;

typedef struct {
    rknn_qwen3_vl_llm_context llm;
    rknn_qwen3_vl_vision_context vision;
    int n_internal_mems;
    rknn3_tensor_mem** internal_mems;
    uint32_t model_width;
    uint32_t model_height;
} rknn_app_context_t;

int init_qwen3_vl_model(rknn_app_context_t* app_ctx, const char* llm_model_path, const char* llm_weight_path, const char* vision_model_path, const char* vision_weight_path,
                        rknn3_llm_param* params, int n_params, RKLLMCallback& callback, uint32_t vision_core_mask, uint32_t llm_core_mask, int max_context_len1, int max_context_len2,
                        char* lora_weight_path, char* vision_lora_weight_path);

int release_qwen3_vl_model(rknn_app_context_t* app_ctx);

// 仅跑 vision，填充 deepstack，供后续分别跑 base / lora 两段 LLM 推理。
// use_lora=true 时启用 vision LoRA，并把结果写入 lora 专用的 deepstack 缓冲区。
int inference_qwen3_vl_model_vision(rknn_app_context_t* app_ctx, image_buffer_t* img, float16* img_embeds, rknn_perf_metrics_t* perf, bool use_lora = false);

#endif //_RKNN_DEMO_QWEN3_VL_H_
