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

#ifndef _RKNN_DEMO_DOTS_MOCR_VISION_UTILS_H_
#define _RKNN_DEMO_DOTS_MOCR_VISION_UTILS_H_

#include <stddef.h>

#include "rknn3_api.h"
#include "common.h"

#define MODEL_WIDTH 448
#define MODEL_HEIGHT 448
#define PATCH_SIZE 14
#define SPATIAL_MERGE_SIZE 2
#define PATCH_MERGE_SIZE (SPATIAL_MERGE_SIZE * PATCH_SIZE) // 28

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
} rknn_dots_mocr_vision_context;

int init_dots_mocr_vision(rknn_dots_mocr_vision_context* vision_ctx, const char* model_path, const char* weight_path, uint32_t core_mask, uint32_t model_width, uint32_t model_height);

int release_dots_mocr_vision(rknn_dots_mocr_vision_context* vision_ctx);

int inference_dots_mocr_vision(rknn_dots_mocr_vision_context* vision_ctx, image_buffer_t* img, float16* img_embeds, size_t img_embeds_size);

#endif //_RKNN_DEMO_DOTS_MOCR_VISION_UTILS_H_
