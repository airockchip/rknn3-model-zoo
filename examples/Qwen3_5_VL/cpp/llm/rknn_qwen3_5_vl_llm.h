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


#ifndef RKNN_DEMO_QWEN3_5_VL_LLM_UTILS_H
#define RKNN_DEMO_QWEN3_5_VL_LLM_UTILS_H

#include "rknn3_api.h"
#include "Tokenizer.h"
#include "common.h"

extern const char* system_prompt;
extern const char* prompt_prefix;
extern const char* prompt_postfix;

#define MAX_NEW_TOKENS 1024
#define MAX_CONTEXT_LEN 1024

typedef struct {
    rknn3_context   rknn_ctx;
    // Keep the Base session under the original name for SpeedUP integration.
    rknn3_session* rknn_sess;
    rknn3_session* rknn_sess_lora;
    rknn3_lora      lora;
    bool            lora_loaded;
    bool            lora_enabled;
    rknn3_llm_param params_lora;
    RKLLMCallback*  callback;
} rknn_qwen3_5_vl_llm_context;

int init_qwen3_5_vl_llm(rknn_qwen3_5_vl_llm_context* llm_ctx, const char* model_path, const char* weight_path, rknn3_llm_param* params, int n_params, RKLLMCallback* callback, uint32_t core_mask);

// The Base session is destroyed before the optional LoRA session is created.
int destroy_qwen3_5_vl_llm_base_session(rknn_qwen3_5_vl_llm_context* llm_ctx);
int create_qwen3_5_vl_llm_lora_session(rknn_qwen3_5_vl_llm_context* llm_ctx, const char* lora_weight_path);

int release_qwen3_5_vl_llm(rknn_qwen3_5_vl_llm_context* llm_ctx);

int inference_qwen3_5_vl_llm(rknn_qwen3_5_vl_llm_context* llm_ctx, rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf);
int inference_qwen3_5_vl_llm_base(rknn_qwen3_5_vl_llm_context* llm_ctx, rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf);
int inference_qwen3_5_vl_llm_lora(rknn_qwen3_5_vl_llm_context* llm_ctx, rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf);

#endif // RKNN_DEMO_QWEN3_5_VL_LLM_UTILS_H
