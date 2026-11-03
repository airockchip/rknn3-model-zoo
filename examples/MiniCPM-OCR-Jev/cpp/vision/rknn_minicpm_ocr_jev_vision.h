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

#ifndef _RKNN_DEMO_VALUE_HEAD_VISION_UTILS_H_
#define _RKNN_DEMO_VALUE_HEAD_VISION_UTILS_H_

#include <stdint.h>

#include "rknn3_api.h"
#include "common.h"

// The pruned (patch-sequence) vision export does not carry the image shape in
// the graph, so the runtime has to be told the canvas size. These defaults only
// apply to that variant; the normal NHWC/NCHW export is self-describing and
// ignores them. The default is the smart_resize native resolution of the demo
// input image (832 wide x 320 tall, 2.6:1, 260 vision tokens).
#define MODEL_WIDTH  832
#define MODEL_HEIGHT 320

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor *inputs;
    rknn3_tensor *outputs;

    int model_channel;
    int model_height;
    int model_width;

    uint32_t *embeds_shape;
    uint32_t embeds_ndims;
    int pruned_version_flag;
} rknn_value_head_vision_context;

// model_width / model_height are only consulted when the export is the pruned
// variant (undefined input layout). Pass 0 to use MODEL_WIDTH / MODEL_HEIGHT.
int init_value_head_vision(rknn_value_head_vision_context *vision_ctx,
                           const char *model_path,
                           const char *weight_path,
                           uint32_t core_mask,
                           uint32_t model_width,
                           uint32_t model_height);

int release_value_head_vision(rknn_value_head_vision_context *vision_ctx);

// Total number of float16 elements of one vision output (tokens * hidden).
size_t value_head_vision_embed_elems(const rknn_value_head_vision_context *vision_ctx);

// Number of visual tokens produced for one image
// (= (H / 32) * (W / 32) for the Qwen3.5 tower, i.e. patch16 + merge2).
int value_head_vision_token_count(const rknn_value_head_vision_context *vision_ctx);

// Runs the tower on `img` and copies `embed_elems` float16 values into
// `img_embeds`. The image is stretched into the compiled canvas; call sites
// that care about aspect ratio must pre-render the canvas themselves.
int inference_value_head_vision(rknn_value_head_vision_context *vision_ctx,
                                image_buffer_t *img,
                                float16 *img_embeds,
                                rknn_perf_metrics_t *perf);

#endif // _RKNN_DEMO_VALUE_HEAD_VISION_UTILS_H_
