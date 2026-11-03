// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "unlimited_ocr.h"
#include "image_utils.h"

const rknn3_sampling_params UNLIMITED_OCR_SAMPLE_PARAMS = {
    .top_k = 1,
    .top_p = 0.9,
    .temperature = 0.0f,
    .repeat_penalty = 1.1f,
    .frequency_penalty = 0.0f,
    .presence_penalty = 0.0f,
};

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
                             int image_size)
{
    if (!app_ctx) return -1;
    memset(app_ctx, 0, sizeof(*app_ctx));

    printf("--> init Unlimited-OCR vision stack\n");
    int ret = init_unlimited_ocr_vision(&app_ctx->vision,
                                        sam_model_path,
                                        sam_weight_path,
                                        vision_model_path,
                                        vision_weight_path,
                                        sam_core_mask,
                                        vision_core_mask,
                                        mode,
                                        base_size,
                                        image_size);
    if (ret < 0) {
        printf("init Unlimited-OCR vision failed! ret=%d\n", ret);
        return ret;
    }

    printf("--> init Unlimited-OCR LLM\n");
    ret = init_unlimited_ocr_llm(&app_ctx->llm, llm_model_path, llm_weight_path,
                                 params, n_params, callback, llm_core_mask);
    if (ret < 0) {
        printf("init Unlimited-OCR llm failed! ret=%d\n", ret);
        return ret;
    }
    return 0;
}

int release_unlimited_ocr_model(rknn_unlimited_ocr_context_t* app_ctx)
{
    if (!app_ctx) return 0;
    release_unlimited_ocr_vision(&app_ctx->vision);
    release_unlimited_ocr_llm(&app_ctx->llm);
    return 0;
}

int inference_unlimited_ocr_model(rknn_unlimited_ocr_context_t* app_ctx,
                                  image_buffer_t* img,
                                  float16* image_embeds,
                                  rknn3_llm_multimodal_tensor tensor,
                                  int n_inputs,
                                  rknn_perf_metrics_t* perf)
{
    if (!app_ctx || !img || !image_embeds || !perf) {
        printf("invalid Unlimited-OCR inference arguments\n");
        return -1;
    }

    int n_image_tokens = 0;
    int64_t start_us = getCurrentTimeUs();
    int ret = inference_unlimited_ocr_vision_base(&app_ctx->vision, img, image_embeds, &n_image_tokens);
    perf->vision_latency = getCurrentTimeUs() - start_us;
    if (ret < 0) {
        printf("inference Unlimited-OCR vision failed! ret=%d\n", ret);
        return ret;
    }

    tensor.image.image_embed = image_embeds;
    tensor.image.n_image = 1;
    tensor.image.n_image_tokens = n_image_tokens;
    tensor.image.image_width = app_ctx->vision.image_size;
    tensor.image.image_height = app_ctx->vision.image_size;

    ret = inference_unlimited_ocr_llm(&app_ctx->llm, tensor, n_inputs, perf);
    if (ret < 0) {
        printf("inference Unlimited-OCR llm failed! ret=%d\n", ret);
        return ret;
    }
    return 0;
}
