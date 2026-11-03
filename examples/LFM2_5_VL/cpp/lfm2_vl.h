// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Licensed under the Apache License, Version 2.0

#ifndef _RKNN_DEMO_LFM2_VL_H_
#define _RKNN_DEMO_LFM2_VL_H_

#include "rknn3_api.h"
#include "common.h"
#include "rknn_lfm2_vl_vision.h"
#include "rknn_lfm2_vl_llm.h"
#include "time_utils.h"

extern const rknn3_sampling_params SAMPLE_PARAMS;

typedef struct {
    rknn_lfm2_vl_llm_context llm;
    rknn_lfm2_vl_vision_context vision;
    int n_internal_mems;
    rknn3_tensor_mem** internal_mems;
    rknn3_context* internal_mems_ctx;  // parallel array: which ctx each mem was created with
} rknn_app_context_t;

int init_lfm2_vl_model(rknn_app_context_t* app_ctx,
                       const char* llm_model_path,
                       const char* llm_weight_path,
                       const char* vision_model_path,
                       const char* vision_weight_path,
                       rknn3_llm_param* params,
                       int n_params,
                       RKLLMCallback& callback,
                       uint32_t vision_core_mask,
                       uint32_t llm_core_mask);

int release_lfm2_vl_model(rknn_app_context_t* app_ctx);

int inference_lfm2_vl_model(rknn_app_context_t* app_ctx,
                            image_buffer_t* img,
                            float16* img_embeds,
                            rknn3_llm_multimodal_tensor tensor,
                            int n_inputs,
                            rknn_perf_metrics_t* perf);

#endif //_RKNN_DEMO_LFM2_VL_H_
