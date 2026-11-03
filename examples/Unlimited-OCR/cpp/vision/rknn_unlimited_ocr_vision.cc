// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0.

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "rknn_unlimited_ocr_vision.h"
#include "image_utils.h"
#include "time_utils.h"

static void dump_tensor_attr(rknn3_tensor_attr* attr)
{
    std::string shape;
    for (int i = 0; i < attr->n_dims; ++i) {
        shape += std::to_string(attr->shape[i]);
        if (i + 1 < attr->n_dims) shape += ", ";
    }
    std::string stride;
    for (int i = 0; i < attr->n_stride; ++i) {
        stride += std::to_string(attr->stride[i]);
        if (i + 1 < attr->n_stride) stride += ", ";
    }
    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, qnt_type=%s, scale=%f, zero_point=%d\n",
           attr->name, attr->n_dims, shape.c_str(), stride.c_str(), attr->aligned_size,
           rknn3_get_layout_string(attr->layout), rknn3_get_type_string(attr->dtype),
           attr->core_id, rknn3_get_qnt_type_string(attr->qnt_type),
           attr->qnt_info.scale, attr->qnt_info.zero_point);
}

static size_t tensor_attr_bytes(const rknn3_tensor_attr* attr)
{
    if (!attr) return 0;
    switch (attr->dtype) {
    case RKNN3_TENSOR_FLOAT32:
        return (size_t)attr->n_elems * sizeof(float);
    case RKNN3_TENSOR_FLOAT16:
    case RKNN3_TENSOR_BFLOAT16:
        return (size_t)attr->n_elems * sizeof(float16);
    case RKNN3_TENSOR_INT64:
        return (size_t)attr->n_elems * sizeof(int64_t);
    case RKNN3_TENSOR_INT32:
        return (size_t)attr->n_elems * sizeof(int32_t);
    case RKNN3_TENSOR_UINT8:
    case RKNN3_TENSOR_INT8:
        return (size_t)attr->n_elems;
    default:
        return attr->aligned_size;
    }
}

size_t unlimited_ocr_model_output_bytes(const unlimited_ocr_rknn_model_t* model, int output_index)
{
    if (!model || output_index < 0 || output_index >= model->io_num.n_output) return 0;
    return tensor_attr_bytes(model->outputs[output_index].attr);
}

static int init_rknn_model(unlimited_ocr_rknn_model_t* model,
                           const char* model_path,
                           const char* weight_path,
                           uint32_t core_mask,
                           const char* tag)
{
    if (!model || !model_path || !weight_path) return -1;
    memset(model, 0, sizeof(*model));

    int ret;
    rknn3_context ctx = 0;
    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 0;

    ret = rknn3_init(&ctx, NULL);
    if (ret < 0) {
        printf("rknn3_init %s failed! ret=%d\n", tag, ret);
        return ret;
    }

    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn3_load_model_from_path %s failed! ret=%d\n", tag, ret);
        rknn3_destroy(ctx);
        return ret;
    }

    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn3_model_init %s failed! ret=%d\n", tag, ret);
        rknn3_destroy(ctx);
        return ret;
    }

    rknn3_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn3_query %s io num failed! ret=%d\n", tag, ret);
        rknn3_destroy(ctx);
        return ret;
    }
    printf("%s input num: %d, output num: %d\n", tag, io_num.n_input, io_num.n_output);

    std::vector<rknn3_tensor_attr> input_attrs(io_num.n_input);
    std::vector<rknn3_tensor_attr> output_attrs(io_num.n_output);
    printf("%s input tensors:\n", tag);
    for (int i = 0; i < io_num.n_input; ++i) {
        memset(&input_attrs[i], 0, sizeof(rknn3_tensor_attr));
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn3_tensor_attr));
        if (ret < 0) {
            printf("rknn3_query %s input[%d] failed! ret=%d\n", tag, i, ret);
            rknn3_destroy(ctx);
            return ret;
        }
        dump_tensor_attr(&input_attrs[i]);
    }
    printf("%s output tensors:\n", tag);
    for (int i = 0; i < io_num.n_output; ++i) {
        memset(&output_attrs[i], 0, sizeof(rknn3_tensor_attr));
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn3_tensor_attr));
        if (ret < 0) {
            printf("rknn3_query %s output[%d] failed! ret=%d\n", tag, i, ret);
            rknn3_destroy(ctx);
            return ret;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

    model->inputs = (rknn3_tensor*)calloc(io_num.n_input, sizeof(rknn3_tensor));
    model->outputs = (rknn3_tensor*)calloc(io_num.n_output, sizeof(rknn3_tensor));
    if (!model->inputs || !model->outputs) {
        printf("calloc %s tensors failed\n", tag);
        rknn3_destroy(ctx);
        return -1;
    }

    model->rknn_ctx = ctx;
    model->io_num = io_num;
    for (int i = 0; i < io_num.n_input; ++i) {
        model->inputs[i].mem = rknn3_create_mem(ctx, input_attrs[i].aligned_size,
                                                input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        model->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (!model->inputs[i].mem || !model->inputs[i].attr) {
            printf("create %s input[%d] mem failed\n", tag, i);
            return -1;
        }
        memcpy(model->inputs[i].attr, &input_attrs[i], sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < io_num.n_output; ++i) {
        model->outputs[i].mem = rknn3_create_mem(ctx, output_attrs[i].aligned_size,
                                                 output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        model->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (!model->outputs[i].mem || !model->outputs[i].attr) {
            printf("create %s output[%d] mem failed\n", tag, i);
            return -1;
        }
        memcpy(model->outputs[i].attr, &output_attrs[i], sizeof(rknn3_tensor_attr));
    }
    return 0;
}

static int release_rknn_model(unlimited_ocr_rknn_model_t* model)
{
    if (!model) return 0;
    for (int i = 0; i < model->io_num.n_input; ++i) {
        if (model->inputs && model->inputs[i].mem) rknn3_destroy_mem(model->rknn_ctx, model->inputs[i].mem);
        if (model->inputs && model->inputs[i].attr) free(model->inputs[i].attr);
    }
    for (int i = 0; i < model->io_num.n_output; ++i) {
        if (model->outputs && model->outputs[i].mem) rknn3_destroy_mem(model->rknn_ctx, model->outputs[i].mem);
        if (model->outputs && model->outputs[i].attr) free(model->outputs[i].attr);
    }
    if (model->inputs) free(model->inputs);
    if (model->outputs) free(model->outputs);
    if (model->rknn_ctx != 0) rknn3_destroy(model->rknn_ctx);
    memset(model, 0, sizeof(*model));
    return 0;
}

static int run_rknn_model(unlimited_ocr_rknn_model_t* model, const char* tag)
{
    int ret;
    for (int i = 0; i < model->io_num.n_input; ++i) {
        ret = rknn3_mem_sync(model->rknn_ctx, model->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_mem_sync %s input[%d] failed! ret=%d\n", tag, i, ret);
            return ret;
        }
    }
    ret = rknn3_run(model->rknn_ctx, model->inputs, model->io_num.n_input, model->outputs, model->io_num.n_output);
    if (ret < 0) {
        printf("rknn3_run %s failed! ret=%d\n", tag, ret);
        return ret;
    }
    for (int i = 0; i < model->io_num.n_output; ++i) {
        ret = rknn3_mem_sync(model->rknn_ctx, model->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_mem_sync %s output[%d] failed! ret=%d\n", tag, i, ret);
            return ret;
        }
    }
    return 0;
}

static int image_format_channels(image_format_t format)
{
    switch (format) {
    case IMAGE_FORMAT_GRAY8: return 1;
    case IMAGE_FORMAT_RGB888: return 3;
    case IMAGE_FORMAT_RGBA8888: return 4;
    default: return 0;
    }
}

static int clamp_int(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

static float sam_normalize_u8(int pixel_u8)
{
    return (float)pixel_u8 / 127.5f - 1.0f;
}

static int8_t quantize_f32_to_i8(float value, const rknn3_tensor_attr* attr)
{
    const float scale = attr->qnt_info.scale > 0.0f ? attr->qnt_info.scale : 1.0f;
    const int zp = attr->qnt_info.zero_point;
    const int q = (int)floorf(value / scale + (float)zp + 0.5f);
    return (int8_t)clamp_int(q, -128, 127);
}

static int write_sam_input_pixel(rknn3_tensor* input, size_t index, int pixel_u8)
{
    pixel_u8 = clamp_int(pixel_u8, 0, 255);
    const float normalized = sam_normalize_u8(pixel_u8);
    if (input->attr->dtype == RKNN3_TENSOR_UINT8) {
        ((unsigned char*)input->mem->virt_addr)[index] = (unsigned char)pixel_u8;
        return 0;
    }
    if (input->attr->dtype == RKNN3_TENSOR_INT8) {
        ((int8_t*)input->mem->virt_addr)[index] = quantize_f32_to_i8(normalized, input->attr);
        return 0;
    }
    if (input->attr->dtype == RKNN3_TENSOR_FLOAT16) {
        ((float16*)input->mem->virt_addr)[index] = fp32_to_fp16(normalized);
        return 0;
    }
    if (input->attr->dtype == RKNN3_TENSOR_FLOAT32) {
        ((float*)input->mem->virt_addr)[index] = normalized;
        return 0;
    }
    return -1;
}

static float read_rgb_channel(const unsigned char* pixel, int src_channels, int channel)
{
    if (src_channels == 1) return (float)pixel[0];
    return (float)pixel[channel];
}

static int fill_base_pixel_values(image_buffer_t* src_img, rknn3_tensor* input, int image_size)
{
    if (!src_img || !src_img->virt_addr || !input || !input->mem || !input->attr) return -1;
    const int src_channels = image_format_channels(src_img->format);
    if (src_channels != 1 && src_channels != 3 && src_channels != 4) {
        printf("unsupported input image format=%d\n", src_img->format);
        return -1;
    }
    if (input->attr->dtype != RKNN3_TENSOR_UINT8 &&
        input->attr->dtype != RKNN3_TENSOR_INT8 &&
        input->attr->dtype != RKNN3_TENSOR_FLOAT16 &&
        input->attr->dtype != RKNN3_TENSOR_FLOAT32) {
        printf("SAM input must be int8/uint8/float16/float32, got %s\n", rknn3_get_type_string(input->attr->dtype));
        return -1;
    }

    const int dst_h = image_size;
    const int dst_w = image_size;
    const int src_stride = src_img->width_stride > 0 ? src_img->width_stride : src_img->width;
    const float scale = std::min((float)dst_w / (float)src_img->width, (float)dst_h / (float)src_img->height);
    const int resized_w = std::max(1, (int)floorf(src_img->width * scale + 0.5f));
    const int resized_h = std::max(1, (int)floorf(src_img->height * scale + 0.5f));
    const int pad_x = (dst_w - resized_w) / 2;
    const int pad_y = (dst_h - resized_h) / 2;

    const bool nchw = (input->attr->layout == RKNN3_TENSOR_NCHW || input->attr->layout == RKNN3_TENSOR_UNDEFINED);
    const bool uint8_input = input->attr->dtype == RKNN3_TENSOR_UINT8;
    const bool int8_input = input->attr->dtype == RKNN3_TENSOR_INT8;
    const bool fp16_input = input->attr->dtype == RKNN3_TENSOR_FLOAT16;
    unsigned char* out8 = (unsigned char*)input->mem->virt_addr;
    int8_t* outi8 = (int8_t*)input->mem->virt_addr;
    float16* out16 = (float16*)input->mem->virt_addr;
    float* out32 = (float*)input->mem->virt_addr;

    const int pad_value = 128;
    const size_t elem_count = (size_t)3 * dst_h * dst_w;
    if (uint8_input) {
        memset(out8, pad_value, elem_count);
    } else if (int8_input) {
        const int8_t qpad = quantize_f32_to_i8(sam_normalize_u8(pad_value), input->attr);
        memset(outi8, qpad, elem_count);
    } else if (fp16_input) {
        for (size_t i = 0; i < elem_count; ++i) out16[i] = fp32_to_fp16(sam_normalize_u8(pad_value));
    } else {
        for (size_t i = 0; i < elem_count; ++i) out32[i] = sam_normalize_u8(pad_value);
    }

    for (int y = 0; y < resized_h; ++y) {
        float src_y = ((float)y + 0.5f) / scale - 0.5f;
        int y0 = clamp_int((int)floorf(src_y), 0, src_img->height - 1);
        int y1 = clamp_int(y0 + 1, 0, src_img->height - 1);
        float wy = src_y - floorf(src_y);
        for (int x = 0; x < resized_w; ++x) {
            float src_x = ((float)x + 0.5f) / scale - 0.5f;
            int x0 = clamp_int((int)floorf(src_x), 0, src_img->width - 1);
            int x1 = clamp_int(x0 + 1, 0, src_img->width - 1);
            float wx = src_x - floorf(src_x);
            const unsigned char* p00 = src_img->virt_addr + ((size_t)y0 * src_stride + x0) * src_channels;
            const unsigned char* p01 = src_img->virt_addr + ((size_t)y0 * src_stride + x1) * src_channels;
            const unsigned char* p10 = src_img->virt_addr + ((size_t)y1 * src_stride + x0) * src_channels;
            const unsigned char* p11 = src_img->virt_addr + ((size_t)y1 * src_stride + x1) * src_channels;
            for (int c = 0; c < 3; ++c) {
                float v0 = read_rgb_channel(p00, src_channels, c) * (1.0f - wx) + read_rgb_channel(p01, src_channels, c) * wx;
                float v1 = read_rgb_channel(p10, src_channels, c) * (1.0f - wx) + read_rgb_channel(p11, src_channels, c) * wx;
                float pixel = v0 * (1.0f - wy) + v1 * wy;
                int pixel_u8 = clamp_int((int)floorf(pixel + 0.5f), 0, 255);
                int dx = pad_x + x;
                int dy = pad_y + y;
                size_t index = nchw ? ((size_t)c * dst_h + dy) * dst_w + dx : ((size_t)dy * dst_w + dx) * 3 + c;
                if (uint8_input) out8[index] = (unsigned char)pixel_u8;
                else if (int8_input) outi8[index] = quantize_f32_to_i8(sam_normalize_u8(pixel_u8), input->attr);
                else if (fp16_input) out16[index] = fp32_to_fp16(sam_normalize_u8(pixel_u8));
                else out32[index] = sam_normalize_u8(pixel_u8);
            }
        }
    }
    return 0;
}

static int copy_tensor_data(void* dst, rknn3_tensor_attr* dst_attr, void* src, rknn3_tensor_attr* src_attr)
{
    if (!dst || !dst_attr || !src || !src_attr) return -1;
    if (dst_attr->n_elems != src_attr->n_elems) {
        printf("tensor elem mismatch: dst=%d src=%d\n", dst_attr->n_elems, src_attr->n_elems);
        return -1;
    }

    if (dst_attr->layout == RKNN3_TENSOR_NC1HWC2 &&
        src_attr->layout == RKNN3_TENSOR_NCHW &&
        dst_attr->dtype == RKNN3_TENSOR_FLOAT16 &&
        src_attr->dtype == RKNN3_TENSOR_FLOAT16 &&
        dst_attr->n_dims == 5 && src_attr->n_dims == 4) {
        const int n = src_attr->shape[0];
        const int c = src_attr->shape[1];
        const int h = src_attr->shape[2];
        const int w = src_attr->shape[3];
        const int c1 = dst_attr->shape[1];
        const int c2 = dst_attr->shape[4];
        if (dst_attr->shape[0] != n || dst_attr->shape[2] != h || dst_attr->shape[3] != w || c1 * c2 != c) {
            printf("NCHW->NC1HWC2 shape mismatch: src=[%d,%d,%d,%d] dst=[%d,%d,%d,%d,%d]\n",
                   src_attr->shape[0], src_attr->shape[1], src_attr->shape[2], src_attr->shape[3],
                   dst_attr->shape[0], dst_attr->shape[1], dst_attr->shape[2], dst_attr->shape[3], dst_attr->shape[4]);
            return -1;
        }
        const float16* s = (const float16*)src;
        float16* d = (float16*)dst;
        for (int ni = 0; ni < n; ++ni) {
            for (int ci = 0; ci < c; ++ci) {
                const int c1i = ci / c2;
                const int c2i = ci % c2;
                for (int hi = 0; hi < h; ++hi) {
                    for (int wi = 0; wi < w; ++wi) {
                        const size_t src_index = ((size_t)ni * c + ci) * h * w + (size_t)hi * w + wi;
                        const size_t dst_index = ((((size_t)ni * c1 + c1i) * h + hi) * w + wi) * c2 + c2i;
                        d[dst_index] = s[src_index];
                    }
                }
            }
        }
        return 0;
    }

    if (dst_attr->dtype == src_attr->dtype) {
        memcpy(dst, src, tensor_attr_bytes(dst_attr));
        return 0;
    }
    if (dst_attr->dtype == RKNN3_TENSOR_FLOAT32 && src_attr->dtype == RKNN3_TENSOR_FLOAT16) {
        float* d = (float*)dst;
        float16* s = (float16*)src;
        for (int i = 0; i < src_attr->n_elems; ++i) d[i] = fp16_to_fp32(s[i]);
        return 0;
    }
    if (dst_attr->dtype == RKNN3_TENSOR_FLOAT16 && src_attr->dtype == RKNN3_TENSOR_FLOAT32) {
        float16* d = (float16*)dst;
        float* s = (float*)src;
        for (int i = 0; i < src_attr->n_elems; ++i) d[i] = fp32_to_fp16(s[i]);
        return 0;
    }
    printf("unsupported tensor dtype/layout conversion: %s/%s -> %s/%s\n",
           rknn3_get_type_string(src_attr->dtype), rknn3_get_layout_string(src_attr->layout),
           rknn3_get_type_string(dst_attr->dtype), rknn3_get_layout_string(dst_attr->layout));
    return -1;
}

static int copy_output_to_fp16(float16* dst, rknn3_tensor* output)
{
    if (!dst || !output || !output->mem || !output->attr) return -1;
    if (output->attr->dtype == RKNN3_TENSOR_FLOAT16) {
        memcpy(dst, output->mem->virt_addr, tensor_attr_bytes(output->attr));
        return 0;
    }
    if (output->attr->dtype == RKNN3_TENSOR_FLOAT32) {
        float* src = (float*)output->mem->virt_addr;
        for (int i = 0; i < output->attr->n_elems; ++i) dst[i] = fp32_to_fp16(src[i]);
        return 0;
    }
    printf("vision output must be float16/float32, got %s\n", rknn3_get_type_string(output->attr->dtype));
    return -1;
}

int init_unlimited_ocr_vision(rknn_unlimited_ocr_vision_context* vision_ctx,
                              const char* sam_model_path,
                              const char* sam_weight_path,
                              const char* vision_model_path,
                              const char* vision_weight_path,
                              uint32_t sam_core_mask,
                              uint32_t vision_core_mask,
                              unlimited_ocr_vision_mode_t mode,
                              int base_size,
                              int image_size)
{
    if (!vision_ctx) return -1;
    memset(vision_ctx, 0, sizeof(*vision_ctx));
    if (mode != UNLIMITED_OCR_VISION_BASE) {
        printf("Only Unlimited-OCR base vision mode is implemented in this demo.\n");
        return -1;
    }

    vision_ctx->mode = mode;
    vision_ctx->base_size = base_size > 0 ? base_size : UNLIMITED_OCR_BASE_IMAGE_SIZE;
    vision_ctx->image_size = image_size > 0 ? image_size : UNLIMITED_OCR_BASE_IMAGE_SIZE;
    vision_ctx->patch_size = UNLIMITED_OCR_PATCH_SIZE;
    vision_ctx->downsample_ratio = UNLIMITED_OCR_DOWNSAMPLE_RATIO;
    vision_ctx->hidden_size = UNLIMITED_OCR_HIDDEN_SIZE;
    vision_ctx->sam_hidden_size = UNLIMITED_OCR_SAM_HIDDEN_SIZE;
    const int grid = vision_ctx->image_size / vision_ctx->patch_size / vision_ctx->downsample_ratio;
    vision_ctx->image_tokens = grid * (grid + 1) + 1;

    printf("--> init Unlimited-OCR SAM model\n");
    int ret = init_rknn_model(&vision_ctx->sam, sam_model_path, sam_weight_path, sam_core_mask, "sam");
    if (ret < 0) return ret;
    if (vision_ctx->sam.io_num.n_input != 1 || vision_ctx->sam.io_num.n_output < 1) {
        printf("SAM RKNN must have one input and at least one output\n");
        return -1;
    }

    printf("--> init Unlimited-OCR vision model\n");
    ret = init_rknn_model(&vision_ctx->vision, vision_model_path, vision_weight_path, vision_core_mask, "vision");
    if (ret < 0) return ret;
    if (vision_ctx->vision.io_num.n_input != 1 || vision_ctx->vision.io_num.n_output < 1) {
        printf("base vision RKNN must have one SAM-feature input and at least one output\n");
        return -1;
    }
    return 0;
}

int release_unlimited_ocr_vision(rknn_unlimited_ocr_vision_context* vision_ctx)
{
    if (!vision_ctx) return 0;
    release_rknn_model(&vision_ctx->sam);
    release_rknn_model(&vision_ctx->vision);
    return 0;
}

int inference_unlimited_ocr_vision_base(rknn_unlimited_ocr_vision_context* vision_ctx,
                                        image_buffer_t* image,
                                        float16* image_embeds,
                                        int* n_image_tokens)
{
    if (!vision_ctx || !image || !image_embeds || !n_image_tokens) {
        printf("invalid vision inference arguments\n");
        return -1;
    }
    if (vision_ctx->mode != UNLIMITED_OCR_VISION_BASE) {
        printf("Only base vision inference is implemented\n");
        return -1;
    }

    int64_t total_start_us = getCurrentTimeUs();

    int64_t preprocess_start_us = getCurrentTimeUs();
    int ret = fill_base_pixel_values(image, &vision_ctx->sam.inputs[0], vision_ctx->image_size);
    int64_t preprocess_latency = getCurrentTimeUs() - preprocess_start_us;
    if (ret < 0) {
        printf("fill SAM pixel_values failed\n");
        return ret;
    }
    printf("Unlimited-OCR preprocess latency = %.2f ms\n", preprocess_latency / 1000.0f);

    printf("--> inference Unlimited-OCR SAM\n");
    int64_t sam_start_us = getCurrentTimeUs();
    ret = run_rknn_model(&vision_ctx->sam, "sam");
    int64_t sam_latency = getCurrentTimeUs() - sam_start_us;
    if (ret < 0) return ret;
    printf("Unlimited-OCR SAM latency = %.2f ms\n", sam_latency / 1000.0f);

    int64_t copy_start_us = getCurrentTimeUs();
    ret = copy_tensor_data(vision_ctx->vision.inputs[0].mem->virt_addr,
                           vision_ctx->vision.inputs[0].attr,
                           vision_ctx->sam.outputs[0].mem->virt_addr,
                           vision_ctx->sam.outputs[0].attr);
    int64_t copy_latency = getCurrentTimeUs() - copy_start_us;
    if (ret < 0) {
        printf("copy SAM features to vision input failed\n");
        return ret;
    }
    printf("Unlimited-OCR SAM->Vision copy latency = %.2f ms\n", copy_latency / 1000.0f);

    printf("--> inference Unlimited-OCR vision\n");
    int64_t vision_start_us = getCurrentTimeUs();
    ret = run_rknn_model(&vision_ctx->vision, "vision");
    int64_t vision_latency = getCurrentTimeUs() - vision_start_us;
    if (ret < 0) return ret;
    printf("Unlimited-OCR vision latency = %.2f ms\n", vision_latency / 1000.0f);

    int64_t output_start_us = getCurrentTimeUs();
    ret = copy_output_to_fp16(image_embeds, &vision_ctx->vision.outputs[0]);
    int64_t output_latency = getCurrentTimeUs() - output_start_us;
    if (ret < 0) return ret;
    printf("Unlimited-OCR vision output copy latency = %.2f ms\n", output_latency / 1000.0f);

    *n_image_tokens = vision_ctx->image_tokens;
    int64_t total_latency = getCurrentTimeUs() - total_start_us;
    printf("Unlimited-OCR vision stack total latency = %.2f ms\n", total_latency / 1000.0f);
    printf("Unlimited-OCR image_embeds tokens=%d, bytes=%zu\n",
           *n_image_tokens, unlimited_ocr_model_output_bytes(&vision_ctx->vision, 0));
    return 0;
}
