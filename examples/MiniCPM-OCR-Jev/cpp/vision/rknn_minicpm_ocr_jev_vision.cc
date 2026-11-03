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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <string>

#include "rknn_minicpm_ocr_jev_vision.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "time_utils.h"

// ---------------------------------------------------------------------------
// HF-compatible smart_resize() for the reward-model path.
// ---------------------------------------------------------------------------
// The score head was trained on pixels produced by transformers'
// Qwen2VLImageProcessorFast, which snaps both sides to multiples of
// patch_size * merge_size (16 * 2 = 32) while preserving the aspect ratio and
// clamps the pixel count into [shortest_edge, longest_edge]. The last step is a
// letterbox into the compiled H x W canvas.
//
// Stretch-resizing straight into the canvas instead (which is what a plain
// `convert_image(img, &dst, NULL, NULL, 0)` does) destroys the glyph geometry
// the score head was trained on: on these OCR block crops that shifts prob_1 by
// up to ~0.4.
//
// These bounds come from the checkpoint's preprocessor_config.json:
//   shortest_edge = 65536, longest_edge = 16777216, patch 16, merge 2
#define VH_IMAGE_FACTOR        32
#define VH_IMAGE_MIN_PIXELS    65536LL
#define VH_IMAGE_MAX_PIXELS    16777216LL
// transformers asserts on aspect ratios beyond ~199:1; pad onto white first.
#define VH_IMAGE_MAX_ASPECT    199.0
// Letterbox fill colour. transformers has no padding step at all, so both the
// reference server and this demo use the same neutral grey.
#define VH_IMAGE_PAD_COLOR     127

static int vh_round_by_factor(int x, int factor)
{
    return (int)(((long long)x + factor / 2) / factor) * factor;
}

static int vh_floor_by_factor(int x, int factor)
{
    return (int)((long long)x / factor) * factor;
}

static int vh_ceil_by_factor(int x, int factor)
{
    return (int)(((long long)x + factor - 1) / factor) * factor;
}

// Mirrors transformers' smart_resize(): round each side to the nearest multiple
// of `factor` (never below `factor`), clamp the pixel count into
// [min_pixels, max_pixels], then re-round. The clamp uses the ORIGINAL pixel
// count, exactly as transformers does.
static void vh_smart_resize(int width, int height, int factor,
                            long long min_pixels, long long max_pixels,
                            int *out_width, int *out_height)
{
    const int w = width > 0 ? width : factor;
    const int h = height > 0 ? height : factor;

    int h_bar = std::max(factor, vh_round_by_factor(h, factor));
    int w_bar = std::max(factor, vh_round_by_factor(w, factor));

    const long long px = (long long)h_bar * (long long)w_bar;
    const long long px_orig = (long long)h * (long long)w;

    if (px > max_pixels) {
        const double beta = sqrt((double)max_pixels / (double)px_orig);
        h_bar = std::max(factor, vh_floor_by_factor((int)floor((double)h * beta), factor));
        w_bar = std::max(factor, vh_floor_by_factor((int)floor((double)w * beta), factor));
    } else if (px < min_pixels) {
        const double beta = sqrt((double)min_pixels / (double)px_orig);
        h_bar = std::max(factor, vh_ceil_by_factor((int)ceil((double)h * beta), factor));
        w_bar = std::max(factor, vh_ceil_by_factor((int)ceil((double)w * beta), factor));
    }

    *out_height = h_bar;
    *out_width = w_bar;
}

// smart_resize + letterbox into canvas_w x canvas_h. `dst` must already have
// its width/height/format/size/virt_addr set.
static int vh_preprocess_aspect(image_buffer_t *src, image_buffer_t *dst)
{
    unsigned char *pixels = src->virt_addr;
    int src_w = src->width;
    int src_h = src->height;
    unsigned char *canvas = NULL;

    if (pixels == NULL || src_w <= 0 || src_h <= 0) {
        return -1;
    }

    // 1) Ultra-wide / ultra-tall: pad onto a white canvas so the resize below
    //    cannot collapse the short side. A no-op for normal crops.
    double ar = (double)std::max(src_w, src_h) / (double)std::min(src_w, src_h);
    if (ar > VH_IMAGE_MAX_ASPECT) {
        int pad = (int)ceil((double)std::max(src_w, src_h) / VH_IMAGE_MAX_ASPECT);
        int cw = src_w >= src_h ? src_w : pad;
        int ch = src_w >= src_h ? pad : src_h;
        canvas = (unsigned char *)malloc((size_t)cw * (size_t)ch * 3);
        if (canvas == NULL) {
            return -1;
        }
        memset(canvas, 255, (size_t)cw * (size_t)ch * 3);
        int ox = (cw - src_w) / 2;
        int oy = (ch - src_h) / 2;
        for (int y = 0; y < src_h; y++) {
            memcpy(canvas + ((size_t)(y + oy) * cw + ox) * 3,
                   pixels + (size_t)y * src_w * 3,
                   (size_t)src_w * 3);
        }
        pixels = canvas;
        src_w = cw;
        src_h = ch;
    }

    // 2) smart_resize(): keep the aspect ratio, snap to `factor` multiples.
    int rw = 0;
    int rh = 0;
    vh_smart_resize(src_w, src_h, VH_IMAGE_FACTOR,
                    VH_IMAGE_MIN_PIXELS, VH_IMAGE_MAX_PIXELS, &rw, &rh);

    // 3) Letterbox the smart-resized image into the model canvas.
    int dst_w = dst->width;
    int dst_h = dst->height;

    double scale = std::min((double)dst_w / (double)rw, (double)dst_h / (double)rh);
    int fit_w = std::max(1, (int)lround((double)rw * scale));
    int fit_h = std::max(1, (int)lround((double)rh * scale));
    if (fit_w > dst_w) fit_w = dst_w;
    if (fit_h > dst_h) fit_h = dst_h;

    int box_x = (dst_w - fit_w) / 2;
    int box_y = (dst_h - fit_h) / 2;

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    src_image.width = src_w;
    src_image.height = src_h;
    src_image.width_stride = src_w;
    src_image.height_stride = src_h;
    src_image.format = IMAGE_FORMAT_RGB888;
    src_image.virt_addr = pixels;

    // dst_box is inclusive on both ends in convert_image_cpu().
    image_rect_t dst_box;
    dst_box.left = box_x;
    dst_box.top = box_y;
    dst_box.right = box_x + fit_w - 1;
    dst_box.bottom = box_y + fit_h - 1;

    int ret = convert_image(&src_image, dst, NULL, &dst_box, VH_IMAGE_PAD_COLOR);

    if (canvas != NULL) {
        free(canvas);
    }
    return ret;
}

static void dump_tensor_attr(rknn3_tensor_attr *attrs)
{
    std::string shape_str = "";
    for (int j = 0; j < attrs->n_dims; j++) {
        shape_str += std::to_string(attrs->shape[j]);
        if (j < attrs->n_dims - 1) {
            shape_str += ", ";
        }
    }

    std::string stride_str = "";
    for (int j = 0; j < attrs->n_stride; j++) {
        stride_str += std::to_string(attrs->stride[j]);
        if (j < attrs->n_stride - 1) {
            stride_str += ", ";
        }
    }

    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, "
           "qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}

int init_value_head_vision(rknn_value_head_vision_context *vision_ctx,
                           const char *model_path,
                           const char *weight_path,
                           uint32_t core_mask,
                           uint32_t model_width,
                           uint32_t model_height)
{
    if (!vision_ctx) {
        printf("init_value_head_vision: vision_ctx is NULL\n");
        return -1;
    }
    int ret;
    rknn3_context ctx = 0;
    rknn3_config config;
    memset(&config, 0, sizeof(config));
    memset(vision_ctx, 0, sizeof(*vision_ctx));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1; // 使用用户管理的 internal 内存

    ret = rknn3_init(&ctx, NULL);
    if (ret < 0) {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        return ret;
    }

    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        return ret;
    }

    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    printf("vision model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("vision input tensors:\n");
    rknn3_tensor_attr input_attrs[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0) {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    printf("vision output tensors:\n");
    rknn3_tensor_attr output_attrs[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0) {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    vision_ctx->pruned_version_flag = 0;
    vision_ctx->inputs = (rknn3_tensor *)malloc(io_num.n_input * sizeof(rknn3_tensor));
    vision_ctx->outputs = (rknn3_tensor *)malloc(io_num.n_output * sizeof(rknn3_tensor));
    vision_ctx->rknn_ctx = ctx;
    vision_ctx->io_num = io_num;
    for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
        vision_ctx->inputs[i].mem = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id,
                                                     RKNN3_FLAG_MEMORY_CACHEABLE);
        vision_ctx->inputs[i].attr = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr));
        memcpy(vision_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
        vision_ctx->outputs[i].mem = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id,
                                                      RKNN3_FLAG_MEMORY_CACHEABLE);
        vision_ctx->outputs[i].attr = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr));
        memcpy(vision_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    if (input_attrs[0].layout == RKNN3_TENSOR_NCHW) {
        printf("vision model is NCHW input layout\n");
        vision_ctx->model_channel = input_attrs[0].shape[1];
        vision_ctx->model_height = input_attrs[0].shape[2];
        vision_ctx->model_width = input_attrs[0].shape[3];
    } else if (input_attrs[0].layout == RKNN3_TENSOR_NHWC) {
        printf("vision model is NHWC input layout\n");
        vision_ctx->model_channel = input_attrs[0].shape[3];
        vision_ctx->model_height = input_attrs[0].shape[1];
        vision_ctx->model_width = input_attrs[0].shape[2];
    } else {
        // Pruned export: the graph only takes the flattened patch sequence, so
        // the canvas size is not recoverable from the tensor shape. Validate the
        // caller-supplied size against the patch count instead.
        int feature_size = MODEL_WIDTH * MODEL_HEIGHT / 16 / 16;
        if (model_width != 0 && model_height != 0) {
            feature_size = (int)(model_width * model_height / 16 / 16);
        }
        if ((int)input_attrs[0].shape[0] != feature_size) {
            printf("Please note: model_width=%u and model_height=%u do not match the model's expected input dimensions. "
                   "Please either update the default parameters or explicitly specify the correct model_width and "
                   "model_height arguments in the command line.\n",
                   model_width == 0 ? MODEL_WIDTH : model_width,
                   model_height == 0 ? MODEL_HEIGHT : model_height);
            return -1;
        }
        vision_ctx->pruned_version_flag = 1;
        vision_ctx->model_channel = 3;
        if (model_width != 0 && model_height != 0) {
            vision_ctx->model_width = (int)model_width;
            vision_ctx->model_height = (int)model_height;
        } else {
            vision_ctx->model_width = MODEL_WIDTH;
            vision_ctx->model_height = MODEL_HEIGHT;
        }
        printf("vision model is UNDEFINED input layout (pruned version)\n");
    }
    printf("vision input image height=%d, width=%d, channel=%d\n",
           vision_ctx->model_height, vision_ctx->model_width, vision_ctx->model_channel);

    if (output_attrs[0].layout != RKNN3_TENSOR_UNDEFINED) {
        printf("vision model output layout must be UNDEFINED, got %s\n",
               rknn3_get_layout_string(output_attrs[0].layout));
        return -1;
    }
    vision_ctx->embeds_shape = vision_ctx->outputs[0].attr->shape;
    vision_ctx->embeds_ndims = vision_ctx->outputs[0].attr->n_dims;
    printf("vision token count: %d\n", value_head_vision_token_count(vision_ctx));

    return 0;
}

size_t value_head_vision_embed_elems(const rknn_value_head_vision_context *vision_ctx)
{
    if (!vision_ctx || !vision_ctx->outputs || !vision_ctx->outputs[0].attr) {
        return 0;
    }
    // Derive the element count from the shape: for fp16 outputs `n_elems` is
    // reported in bytes (aligned_size / elem_size also equals it), so using it
    // directly would double the buffer size.
    const rknn3_tensor_attr *attr = vision_ctx->outputs[0].attr;
    size_t elems = 1;
    for (uint32_t i = 0; i < attr->n_dims; i++) {
        if (attr->shape[i] == 0) {
            return 0;
        }
        elems *= (size_t)attr->shape[i];
    }
    return attr->n_dims == 0 ? 0 : elems;
}

int value_head_vision_token_count(const rknn_value_head_vision_context *vision_ctx)
{
    if (!vision_ctx) {
        return 0;
    }
    // Qwen3.5 merges spatial_merge_size=2 neighbours, so one token covers a
    // 32x32 pixel block (patch_size=16).
    return (vision_ctx->model_height / 32) * (vision_ctx->model_width / 32);
}

int release_value_head_vision(rknn_value_head_vision_context *vision_ctx)
{
    if (!vision_ctx) {
        return -1;
    }
    for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
        if (vision_ctx->inputs[i].mem) {
            rknn3_destroy_mem(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem);
        }
        if (vision_ctx->inputs[i].attr != NULL) {
            free(vision_ctx->inputs[i].attr);
            vision_ctx->inputs[i].attr = NULL;
        }
    }
    for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
        if (vision_ctx->outputs[i].mem) {
            rknn3_destroy_mem(vision_ctx->rknn_ctx, vision_ctx->outputs[i].mem);
        }
        if (vision_ctx->outputs[i].attr != NULL) {
            free(vision_ctx->outputs[i].attr);
            vision_ctx->outputs[i].attr = NULL;
        }
    }
    free(vision_ctx->inputs);
    vision_ctx->inputs = NULL;
    free(vision_ctx->outputs);
    vision_ctx->outputs = NULL;
    if (vision_ctx->rknn_ctx != 0) {
        rknn3_destroy(vision_ctx->rknn_ctx);
        vision_ctx->rknn_ctx = 0;
    }
    return 0;
}

// NHWC uint8 RGB888 -> NCHW float, normalised with the Qwen3.5 image
// statistics (mean = std = 0.5, rescale 1/255), i.e. (x - 127.5) / 127.5.
// Only used when the RKNN model was exported with --fp_input, i.e. when the
// graph does NOT carry the normalisation itself.
static void normalize_and_to_nchw(const unsigned char *src, void *dst, int width, int height,
                                  rknn3_tensor_type dtype)
{
    float mean[3] = {127.5f, 127.5f, 127.5f};
    float std[3] = {127.5f, 127.5f, 127.5f};
    int num_pixels = width * height;

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int nhwc_idx = (h * width + w) * 3 + c;
                int nchw_idx = c * num_pixels + h * width + w;
                float value = ((float)src[nhwc_idx] - mean[c]) / std[c];
                if (dtype == RKNN3_TENSOR_FLOAT16) {
                    ((float16 *)dst)[nchw_idx] = fp32_to_fp16(value);
                } else {
                    ((float *)dst)[nchw_idx] = value;
                }
            }
        }
    }
}

// Repack NCHW float16 into the flat patch sequence the pruned export expects:
//   [t, c, h/m/p, m, p, w/m/p, m, p] -> [h/p * w/p, m, m, c, t, p, p]
// Both temporal frames are identical, so the input is duplicated in place by
// the caller before this runs.
static void transpose(float16 *in, float16 *out, int model_width, int model_height)
{
    int64_t dims[8] = {2, 3, int(model_height / 32), 2, 16, int(model_width / 32), 2, 16};
    int perm[8] = {2, 5, 3, 6, 1, 0, 4, 7};
    int64_t new_dims[8];
    for (int k = 0; k < 8; k++) {
        new_dims[k] = dims[perm[k]];
    }
    int64_t strides[8];
    strides[7] = 1;
    for (int d = 6; d >= 0; d--) {
        strides[d] = strides[d + 1] * dims[d + 1];
    }

    int64_t out_idx = 0;
    for (int64_t o0 = 0; o0 < new_dims[0]; o0++) {
        for (int64_t o1 = 0; o1 < new_dims[1]; o1++) {
            for (int64_t o2 = 0; o2 < new_dims[2]; o2++) {
                for (int64_t o3 = 0; o3 < new_dims[3]; o3++) {
                    for (int64_t o4 = 0; o4 < new_dims[4]; o4++) {
                        for (int64_t o5 = 0; o5 < new_dims[5]; o5++) {
                            for (int64_t o6 = 0; o6 < new_dims[6]; o6++) {
                                for (int64_t o7 = 0; o7 < new_dims[7]; o7++) {
                                    int64_t i[8];
                                    i[perm[0]] = o0;
                                    i[perm[1]] = o1;
                                    i[perm[2]] = o2;
                                    i[perm[3]] = o3;
                                    i[perm[4]] = o4;
                                    i[perm[5]] = o5;
                                    i[perm[6]] = o6;
                                    i[perm[7]] = o7;
                                    int64_t flat_in = i[0] * strides[0] + i[1] * strides[1] +
                                                      i[2] * strides[2] + i[3] * strides[3] +
                                                      i[4] * strides[4] + i[5] * strides[5] +
                                                      i[6] * strides[6] + i[7] * strides[7];
                                    out[out_idx++] = in[flat_in];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

int inference_value_head_vision(rknn_value_head_vision_context *vision_ctx,
                                image_buffer_t *img,
                                float16 *img_embeds,
                                rknn_perf_metrics_t *perf)
{
    if ((!vision_ctx) || (!img) || (!img_embeds)) {
        printf("inference_value_head_vision: invalid argument\n");
        return -1;
    }

    int ret = 0;
    int64_t start_us = 0;
    int64_t end_us = 0;
    image_buffer_t dst_img;
    // Declared up front: the `goto out` error paths below must not jump over
    // their initialisation (C++ forbids that).
    rknn3_tensor_attr *in_attr = NULL;
    size_t canvas_elems = 0;
    memset(&dst_img, 0, sizeof(image_buffer_t));

    dst_img.width = vision_ctx->model_width;
    dst_img.height = vision_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL) {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    // Preprocess the way the checkpoint was trained: smart_resize keeps the
    // aspect ratio and the letterbox centres the result in the canvas. A plain
    // stretch would distort the glyphs and move the score.
    ret = vh_preprocess_aspect(img, &dst_img);
    if (ret < 0) {
        printf("preprocess (smart_resize + letterbox) fail! ret=%d\n", ret);
        goto out;
    }

    in_attr = vision_ctx->inputs[0].attr;
    canvas_elems = (size_t)dst_img.size;

    if (vision_ctx->pruned_version_flag == 1) {
        size_t single_bytes = canvas_elems * sizeof(float16);
        // Two identical temporal frames back to back.
        float16 *expand_data = (float16 *)malloc(single_bytes * 2);
        if (expand_data == NULL) {
            printf("malloc expanded vision input failed\n");
            ret = -1;
            goto out;
        }
        if (single_bytes * 2 > (size_t)vision_ctx->inputs[0].attr->aligned_size) {
            printf("pruned vision input too large: %zu bytes > tensor %llu bytes\n",
                   single_bytes * 2, (unsigned long long)vision_ctx->inputs[0].attr->aligned_size);
            free(expand_data);
            ret = -1;
            goto out;
        }
        normalize_and_to_nchw(dst_img.virt_addr, expand_data, vision_ctx->model_width,
                              vision_ctx->model_height, RKNN3_TENSOR_FLOAT16);
        memcpy((char *)expand_data + single_bytes, expand_data, single_bytes);
        transpose(expand_data, (float16 *)(vision_ctx->inputs[0].mem->virt_addr),
                  vision_ctx->model_width, vision_ctx->model_height);
        free(expand_data);
    } else {
        if ((size_t)in_attr->n_elems != canvas_elems) {
            printf("vision input element count mismatch: tensor has %u, canvas needs %zu\n",
                   in_attr->n_elems, canvas_elems);
            ret = -1;
            goto out;
        }

        if (in_attr->dtype == RKNN3_TENSOR_UINT8 && in_attr->layout == RKNN3_TENSOR_NHWC) {
            // Default export path: the graph carries the uint8 -> fp32 cast and
            // the (x - 127.5) / 127.5 normalisation (mean_values/std_values are
            // baked in at conversion time), so hand over the RGB888 canvas
            // verbatim instead of pre-normalising it on the host.
            memcpy(vision_ctx->inputs[0].mem->virt_addr, dst_img.virt_addr, canvas_elems);
        } else if ((in_attr->dtype == RKNN3_TENSOR_FLOAT16 || in_attr->dtype == RKNN3_TENSOR_FLOAT32) &&
                   in_attr->layout == RKNN3_TENSOR_NCHW) {
            // --fp_input export path: normalise on the host into NCHW floats.
            size_t elem_bytes = in_attr->dtype == RKNN3_TENSOR_FLOAT16 ? sizeof(float16) : sizeof(float);
            if (canvas_elems * elem_bytes > (size_t)in_attr->aligned_size) {
                printf("vision input too large: %zu bytes > tensor %llu bytes\n",
                       canvas_elems * elem_bytes, (unsigned long long)in_attr->aligned_size);
                ret = -1;
                goto out;
            }
            normalize_and_to_nchw(dst_img.virt_addr, vision_ctx->inputs[0].mem->virt_addr,
                                  vision_ctx->model_width, vision_ctx->model_height, in_attr->dtype);
        } else {
            printf("unsupported vision input: dtype=%d layout=%d\n",
                   (int)in_attr->dtype, (int)in_attr->layout);
            ret = -1;
            goto out;
        }
    }

    for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
        ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_mem_sync input[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }

    start_us = getCurrentTimeUs();
    ret = rknn3_run(vision_ctx->rknn_ctx, vision_ctx->inputs, vision_ctx->io_num.n_input,
                    vision_ctx->outputs, vision_ctx->io_num.n_output);
    end_us = getCurrentTimeUs();
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
        ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_mem_sync output[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }

    memcpy(img_embeds, vision_ctx->outputs[0].mem->virt_addr, value_head_vision_embed_elems(vision_ctx) * sizeof(float16));
    if (perf) {
        perf->vision_latency = end_us - start_us;
    }

out:
    if (dst_img.virt_addr != NULL) {
        free(dst_img.virt_addr);
    }
    return ret;
}
