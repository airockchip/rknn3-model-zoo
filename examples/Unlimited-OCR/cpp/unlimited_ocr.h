// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0.

#ifndef _RKNN_DEMO_UNLIMITED_OCR_H_
#define _RKNN_DEMO_UNLIMITED_OCR_H_

#include "common.h"
#include "llm/rknn_unlimited_ocr_llm.h"
#include "time_utils.h"
#include "vision/rknn_unlimited_ocr_vision.h"

typedef struct {
    rknn_unlimited_ocr_llm_context llm;
    rknn_unlimited_ocr_vision_context vision;
    int n_internal_mems;
    rknn3_tensor_mem** internal_mems;
} rknn_unlimited_ocr_context_t;

extern const rknn3_sampling_params UNLIMITED_OCR_SAMPLE_PARAMS;

int init_unlimited_ocr_model(rknn_unlimited_ocr_context_t* app_ctx,
                             const char* llm_model_path,
                             const char* llm_weight_path,
                             const char* sam_model_path,
                             const char* sam_weight_path,
                             const char* vision_model_path,
                             const char* vision_weight_path,
                             rknn3_llm_param* params,
                             int n_params,
                             RKLLMCallback& callback,
                             uint32_t sam_core_mask,
                             uint32_t vision_core_mask,
                             uint32_t llm_core_mask,
                             unlimited_ocr_vision_mode_t mode,
                             int base_size,
                             int image_size);

int release_unlimited_ocr_model(rknn_unlimited_ocr_context_t* app_ctx);

int inference_unlimited_ocr_model(rknn_unlimited_ocr_context_t* app_ctx,
                                  image_buffer_t* img,
                                  float16* image_embeds,
                                  rknn3_llm_multimodal_tensor tensor,
                                  int n_inputs,
                                  rknn_perf_metrics_t* perf);

#endif
