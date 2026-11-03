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


#ifndef _RKNN_DEMO_QWEN3_5_VL_VISION_UTILS_H_
#define _RKNN_DEMO_QWEN3_5_VL_VISION_UTILS_H_

#include "rknn3_api.h"
#include "common.h"

#define MODEL_WIDTH 384 // 由于vision模型被裁剪了，需要手动添加这些参数配置
#define MODEL_HEIGHT 384

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

    rknn3_lora lora;
    bool       lora_loaded;
    bool       lora_enabled;
} rknn_qwen3_5_vl_vision_context;

int init_qwen3_5_vl_vision(rknn_qwen3_5_vl_vision_context* vision_ctx, const char* model_path, const char* weight_path, uint32_t core_mask, uint32_t model_width, uint32_t model_height);

// Called after shared internal memory is configured. It loads the adapter but
// leaves it disabled so the first vision inference remains the Base pass.
int setup_qwen3_5_vl_vision_lora(rknn_qwen3_5_vl_vision_context* vision_ctx, const char* lora_weight_path);

int release_qwen3_5_vl_vision(rknn_qwen3_5_vl_vision_context* vision_ctx);

int inference_qwen3_5_vl_vision(rknn_qwen3_5_vl_vision_context* vision_ctx, image_buffer_t* img, float16* img_embeds, bool use_lora = false);

#endif //_RKNN_DEMO_QWEN3_5_VL_VISION_UTILS_H_
