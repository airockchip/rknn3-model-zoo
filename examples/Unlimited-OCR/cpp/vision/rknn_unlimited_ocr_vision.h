// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0.

#ifndef _RKNN_DEMO_UNLIMITED_OCR_VISION_H_
#define _RKNN_DEMO_UNLIMITED_OCR_VISION_H_

#include <stdint.h>
#include "common.h"
#include "rknn3_api.h"

#define UNLIMITED_OCR_BASE_IMAGE_SIZE 1024
#define UNLIMITED_OCR_PATCH_SIZE 16
#define UNLIMITED_OCR_DOWNSAMPLE_RATIO 4
#define UNLIMITED_OCR_HIDDEN_SIZE 1280
#define UNLIMITED_OCR_SAM_HIDDEN_SIZE 1024

typedef enum {
    UNLIMITED_OCR_VISION_BASE = 0,
    UNLIMITED_OCR_VISION_GUNDAM = 1,
} unlimited_ocr_vision_mode_t;

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;
} unlimited_ocr_rknn_model_t;

typedef struct {
    unlimited_ocr_rknn_model_t sam;
    unlimited_ocr_rknn_model_t vision;
    unlimited_ocr_vision_mode_t mode;
    int base_size;
    int image_size;
    int patch_size;
    int downsample_ratio;
    int hidden_size;
    int sam_hidden_size;
    int image_tokens;
} rknn_unlimited_ocr_vision_context;

int init_unlimited_ocr_vision(rknn_unlimited_ocr_vision_context* vision_ctx,
                              const char* sam_model_path,
                              const char* sam_weight_path,
                              const char* vision_model_path,
                              const char* vision_weight_path,
                              uint32_t sam_core_mask,
                              uint32_t vision_core_mask,
                              unlimited_ocr_vision_mode_t mode,
                              int base_size,
                              int image_size);

int release_unlimited_ocr_vision(rknn_unlimited_ocr_vision_context* vision_ctx);

int inference_unlimited_ocr_vision_base(rknn_unlimited_ocr_vision_context* vision_ctx,
                                        image_buffer_t* image,
                                        float16* image_embeds,
                                        int* n_image_tokens);

size_t unlimited_ocr_model_output_bytes(const unlimited_ocr_rknn_model_t* model, int output_index);

#endif
