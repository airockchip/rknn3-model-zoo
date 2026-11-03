// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Licensed under the Apache License, Version 2.0

#ifndef _RKNN_DEMO_LFM2_VL_LLM_UTILS_H_
#define _RKNN_DEMO_LFM2_VL_LLM_UTILS_H_

#include "rknn3_api.h"
#include "Tokenizer.h"
#include "common.h"

// Max generated tokens per request. 128 matches the board-validated protocol
// class; larger values change the runtime's prefill buffer layout and
// reintroduce empty outputs for short prompts.
#define LFM2_VL_MAX_NEW_TOKENS   256
#define LFM2_VL_MAX_CONTEXT_LEN  4096

typedef struct {
    rknn3_context   rknn_ctx;
    rknn3_session*  rknn_sess;
    bool initialized;
} rknn_lfm2_vl_llm_context;

int init_lfm2_vl_llm(rknn_lfm2_vl_llm_context* llm_ctx,
                     const char* model_path,
                     const char* weight_path,
                     rknn3_llm_param* params,
                     int n_params,
                     RKLLMCallback& callback,
                     uint32_t core_mask);

int release_lfm2_vl_llm(rknn_lfm2_vl_llm_context* llm_ctx);

int inference_lfm2_vl_llm(rknn_lfm2_vl_llm_context* llm_ctx,
                          rknn3_llm_multimodal_tensor tensor,
                          int n_inputs,
                          rknn_perf_metrics_t* perf);

int clear_lfm2_vl_llm_cache(rknn_lfm2_vl_llm_context* llm_ctx);

#endif //_RKNN_DEMO_LFM2_VL_LLM_UTILS_H_
