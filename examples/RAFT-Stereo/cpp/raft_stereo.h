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

#ifndef _RKNN_DEMO_RAFT_STEREO_H_
#define _RKNN_DEMO_RAFT_STEREO_H_

#include "rknn3_api.h"
#include "common.h"

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;

    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;

} rknn_app_context_t;

int init_raft_stereo_model(const char *model_path, const char* weight_path,
                           rknn_app_context_t *app_ctx, uint32_t core_mask);
int inference_raft_stereo_model(rknn_app_context_t *app_ctx, image_buffer_t *img_l,
                                image_buffer_t *img_r);
int release_raft_stereo_model(rknn_app_context_t *app_ctx);
int save_raft_stereo_output(rknn_app_context_t *app_ctx);

#endif /* _RKNN_DEMO_RAFT_STEREO_H_ */