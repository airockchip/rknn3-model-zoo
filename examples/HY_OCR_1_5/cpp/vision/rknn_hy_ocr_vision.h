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


#ifndef _RKNN_DEMO_HY_OCR_VISION_UTILS_H_
#define _RKNN_DEMO_HY_OCR_VISION_UTILS_H_

#include "rknn3_api.h"
#include "common.h"

// HunyuanOCR-1.5 vision 默认输入尺寸 (pixel 输入, 非 prune 模式)
#define MODEL_WIDTH 512
#define MODEL_HEIGHT 512

// HunyuanOCR-1.5 常量 (与 preprocessor_config.json / config.json 一致)
#define HY_OCR_PATCH_SIZE     16
#define HY_OCR_MERGE_SIZE     2
#define HY_OCR_FACTOR         (HY_OCR_PATCH_SIZE * HY_OCR_MERGE_SIZE)  // 32
#define HY_OCR_MIN_PIXELS     (MODEL_WIDTH * MODEL_HEIGHT)  // shortest_edge
#define HY_OCR_MAX_PIXELS     (2048 * 2048)     // longest_edge

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;

    int model_channel;
    int model_height;
    int model_width;

    uint32_t* embeds_shape;
    uint32_t embeds_ndims;
    int pruned_version_flag;

    // 裁剪版 2 输入模式: pos_embed 预计算数据
    float16* pos_embed_data;
    int pos_embed_size;  // 元素个数
} rknn_hy_ocr_vision_context;

int init_hy_ocr_vision(rknn_hy_ocr_vision_context* vision_ctx, const char* model_path, const char* weight_path, uint32_t core_mask, uint32_t model_width, uint32_t model_height);

int release_hy_ocr_vision(rknn_hy_ocr_vision_context* vision_ctx);

int inference_hy_ocr_vision(rknn_hy_ocr_vision_context* vision_ctx, image_buffer_t* img, float16* img_embeds, int n_image_tokens);

#endif //_RKNN_DEMO_HY_OCR_VISION_UTILS_H_
