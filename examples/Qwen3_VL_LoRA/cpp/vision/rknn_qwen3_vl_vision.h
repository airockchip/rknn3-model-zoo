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


#ifndef _RKNN_DEMO_QWEN3_VL_VISION_UTILS_H_
#define _RKNN_DEMO_QWEN3_VL_VISION_UTILS_H_

#include "rknn3_api.h"
#include "common.h"

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

    rknn3_lora lora;         // vision LoRA 适配器（由 vision *.lora_weight 加载）
    bool       lora_loaded;  // 是否已成功加载 LoRA
    bool       lora_enabled; // 当前 context 上是否已启用 LoRA
} rknn_qwen3_vl_vision_context;

int init_qwen3_vl_vision(rknn_qwen3_vl_vision_context* vision_ctx, const char* model_path, const char* weight_path, uint32_t core_mask, int* deepstack_aligned_size, uint32_t model_width, uint32_t model_height, const char* lora_weight_path = nullptr);

// 加载 vision LoRA。注意：rknn3_lora_init 要求 rknn3_model_init 之后调用；
// 本 demo 在 model_init 之后还会调用 rknn3_set_internal_mem() 重配内部内存，
// 因此 LoRA 的加载放在所有内存配置完成之后（init_internal_share 之后）再执行。
int setup_qwen3_vl_vision_lora(rknn_qwen3_vl_vision_context* vision_ctx, const char* lora_weight_path);

int release_qwen3_vl_vision(rknn_qwen3_vl_vision_context* vision_ctx);

// use_lora=true 时在 vision context 上启用 LoRA，false 时确保关闭，从而得到 base / lora 两套 vision embedding
int inference_qwen3_vl_vision(rknn_qwen3_vl_vision_context* vision_ctx, image_buffer_t* img, float16* img_embeds, float16* deepstack_data0, float16* deepstack_data1, float16* deepstack_data2, bool use_lora = false);

#endif //_RKNN_DEMO_QWEN3_VL_VISION_UTILS_H_
