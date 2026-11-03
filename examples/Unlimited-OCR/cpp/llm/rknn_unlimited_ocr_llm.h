// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0.

#ifndef _RKNN_DEMO_UNLIMITED_OCR_LLM_H_
#define _RKNN_DEMO_UNLIMITED_OCR_LLM_H_

#include "Tokenizer.h"
#include "common.h"
#include "rknn3_api.h"

#define UNLIMITED_OCR_MAX_NEW_TOKENS 4096
#define UNLIMITED_OCR_MAX_CONTEXT_LEN 8192

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_session* rknn_sess;
} rknn_unlimited_ocr_llm_context;

extern const char* unlimited_ocr_system_prompt;
extern const char* unlimited_ocr_prompt_prefix;
extern const char* unlimited_ocr_prompt_postfix;

int init_unlimited_ocr_llm(rknn_unlimited_ocr_llm_context* llm_ctx,
                           const char* model_path,
                           const char* weight_path,
                           rknn3_llm_param* params,
                           int n_params,
                           RKLLMCallback& callback,
                           uint32_t core_mask);

int release_unlimited_ocr_llm(rknn_unlimited_ocr_llm_context* llm_ctx);

int inference_unlimited_ocr_llm(rknn_unlimited_ocr_llm_context* llm_ctx,
                                rknn3_llm_multimodal_tensor tensor,
                                int n_inputs,
                                rknn_perf_metrics_t* perf);

#endif
