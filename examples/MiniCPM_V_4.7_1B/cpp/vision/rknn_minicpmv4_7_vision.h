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


#ifndef _RKNN_DEMO_MINICPMV4_7_VISION_UTILS_H_
#define _RKNN_DEMO_MINICPMV4_7_VISION_UTILS_H_

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
} rknn_minicpmv4_7_vision_context;

/* ---- slice (crop) geometry ----------------------------------------------
 * The slice layout assumes every crop is resized to the vision model's input
 * size (448x448 in this export): thumbnail and slices all use that one size.
 * The upstream processor instead keeps each crop's own aspect ratio; using a
 * single size is what allows this static single-crop rknn to be reused per crop.
 */
#define MINICPMV4_7_MAX_SLICE_NUMS 9
#define MINICPMV4_7_MAX_CROPS      (1 + MINICPMV4_7_MAX_SLICE_NUMS) /* thumbnail + slices */

/* Slice grid for an image.  Returns 0 (and rows=cols=0) when the image must not
 * be sliced (area <= scale_resolution^2), otherwise 1 with rows*cols >= 2. */
int minicpmv4_7_get_sliced_grid(int img_h, int img_w, int max_slice_nums,
                                int scale_resolution, int* rows, int* cols);

/* boxes[0] = whole image (thumbnail), boxes[1..n) = slices in row-major order.
 * Returns the number of crops written (>= 1), or -1 on bad arguments.
 * NOTE: image_rect_t right/bottom are INCLUSIVE (convert_image uses right-left+1). */
int minicpmv4_7_build_crop_boxes(int img_h, int img_w, int max_slice_nums,
                                 int scale_resolution, image_rect_t* boxes, int max_boxes);

/* Number of float16 elements produced by ONE crop (product of the output shape). */
uint64_t minicpmv4_7_vision_embeds_per_crop(const rknn_minicpmv4_7_vision_context* vision_ctx);

/* ---- canvas mrope position fill -----------------------------------------
 * The prefill is CHUNKED by the runtime (mrope_id_input is [128,3], so at most
 * 128 tokens per forward), therefore the canvas position of a token is computed
 * from its offset inside the visual group alone -- no cross-chunk scan of the
 * token markers is needed.  Only the absolute index of <image> has to be
 * remembered, which the caller keeps in `canvas_base`.
 */
typedef struct {
    int32_t im_start_id;      /* <image>        */
    int32_t im_end_id;        /* </image>       */
    int32_t slice_start_id;   /* <slice>        */
    int32_t slice_end_id;     /* </slice>       */
    int32_t crop_sq;          /* LLM grid side of ONE crop (e.g. 8 for 64 tokens) */
    int32_t rows;             /* slice grid rows (1 when there are no slices) */
    int32_t cols;             /* slice grid cols */
    int32_t n_slices;         /* number of slice blocks (0 => thumbnail only) */
    int32_t canvas_base;      /* IN/OUT: absolute token index of <image>,
                               *         -1 while it has not been seen yet */
} minicpmv4_7_canvas_geom_t;

/* Fill (T,H,W) for the tokens of one forward into dst (3 int32 per token, at
 * most cap rows).  `first_pos` is the absolute index of the first token of this
 * forward (LLMInputCallbackParam.pos).
 *
 * Returns 0 when canvas positions were written, -1 when no <image> has been seen
 * yet (text-only sequence, or a chunk that starts before the image); the caller
 * then writes plain 1-D positions.  `geom->canvas_base` is updated when <image>
 * is found in this chunk, so later chunks keep using the canvas. */
int minicpmv4_7_canvas_fill(const int32_t* tokens, int n_tokens, int first_pos,
                           minicpmv4_7_canvas_geom_t* geom,
                           int32_t* dst, int cap);

int init_minicpmv4_7_vision(rknn_minicpmv4_7_vision_context* vision_ctx, const char* model_path, const char* weight_path, uint32_t core_mask);

int release_minicpmv4_7_vision(rknn_minicpmv4_7_vision_context* vision_ctx);

/* Encode several crops with the same single-crop rknn model and concatenate the
 * embeddings in crop order into embeds_out[c*per_crop .. (c+1)*per_crop).
 * Equivalent to a batched run: crops never interact inside the ViT. */
int inference_minicpmv4_7_vision_multi(rknn_minicpmv4_7_vision_context* vision_ctx,
                                       image_buffer_t* src,
                                       image_rect_t* boxes, int n_crops,
                                       float16* embeds_out);

#endif //_RKNN_DEMO_MINICPMV4_7_VISION_UTILS_H_
