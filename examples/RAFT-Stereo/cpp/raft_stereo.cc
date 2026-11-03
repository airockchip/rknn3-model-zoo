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
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sys/time.h>

#include "rknn3_api.h"
#include "raft_stereo.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "plasma_lut.h"
#include "float16.h"

/* ---------------Image Process for RAFT Stereo----------------- */
/* resize image by keeping ratio */
static void resize_image(const unsigned char *src_data, int src_w, int src_h, int src_c,
                         unsigned char *dst_data, int target_w, int target_h,
                         float &out_scale, int &out_new_w, int &out_new_h)
{
    if (target_w <= 0 || target_h <= 0) { return; }

    float scale_x = static_cast<float>(target_w) / src_w;
    float scale_y = static_cast<float>(target_h) / src_h;
    out_scale = std::min(scale_x, scale_y);

    out_new_w = static_cast<int>(src_w * out_scale);
    out_new_h = static_cast<int>(src_h * out_scale);

    // Ensure calculated new size doesn't exceed target due to rounding
    out_new_w = std::min(out_new_w, target_w);
    out_new_h = std::min(out_new_h, target_h);

    float x_ratio = (out_new_w > 1) ? static_cast<float>(src_w - 1) / (out_new_w - 1) : 0.0f;
    float y_ratio = (out_new_h > 1) ? static_cast<float>(src_h - 1) / (out_new_h - 1) : 0.0f;

    for (int dy = 0; dy < out_new_h; ++dy) {
        for (int dx = 0; dx < out_new_w; ++dx) {
            float sx = dx * x_ratio;
            float sy = dy * y_ratio;

            int x_low = static_cast<int>(sx);
            int y_low = static_cast<int>(sy);
            int x_high = (x_low + 1 < src_w) ? x_low + 1 : x_low;
            int y_high = (y_low + 1 < src_h) ? y_low + 1 : y_low;

            float x_lerp = sx - x_low;
            float y_lerp = sy - y_low;

            for (int c = 0; c < src_c; ++c) {
                float tl = src_data[(y_low * src_w + x_low) * src_c + c];
                float tr = src_data[(y_low * src_w + x_high) * src_c + c];
                float bl = src_data[(y_high * src_w + x_low) * src_c + c];
                float br = src_data[(y_high * src_w + x_high) * src_c + c];

                float top = tl * (1 - x_lerp) + tr * x_lerp;
                float bot = bl * (1 - x_lerp) + br * x_lerp;
                float val = top * (1 - y_lerp) + bot * y_lerp;

                dst_data[(dy * out_new_w + dx) * src_c + c] = static_cast<unsigned char>(
                    std::max(0.0f, std::min(255.0f, val + 0.5f))
                );
            }
        }
    }
}

/* pad image for right & bottom */
static void pad_image(const unsigned char* src_data, int src_h, int src_w, int src_c,
                      unsigned char* dst_data, int dst_h, int dst_w, uint8_t pad_value)
{
    // Assumes dst_data is pre-zeroed or initialized with pad_value
    size_t src_row_stride = static_cast<size_t>(src_w) * src_c;
    for (int y = 0; y < src_h; ++y) {
        size_t src_offset = y * src_row_stride;
        size_t dst_offset = y * static_cast<size_t>(dst_w) * src_c;
        memcpy(dst_data + dst_offset, src_data + src_offset, src_row_stride);
    }
}

static unsigned char* convert_channels(const unsigned char* src_data, int width, int height,
                                       int src_c, int dst_c, int& out_dst_c)
{
    if (src_c == dst_c) {
        // If channels are already the same, return a copy
        size_t size = static_cast<size_t>(width) * height * src_c * sizeof(unsigned char);
        unsigned char* copy = static_cast<unsigned char*>(malloc(size));
        if (copy) {
            memcpy(copy, src_data, size);
        }
        out_dst_c = src_c;
        return copy;
    }

    size_t src_size = static_cast<size_t>(width) * height * src_c;
    size_t temp_size = static_cast<size_t>(width) * height * dst_c;
    unsigned char* dst_data = static_cast<unsigned char*>(malloc(temp_size));

    if (!dst_data) {
        std::cerr << "OOM in convertChannels\n";
        return nullptr;
    }

    // Common conversions
    if (src_c == 3 && dst_c == 1) {
        // RGB to Grayscale (Luminance formula: 0.299*R + 0.587*G + 0.114*B)
        for (int i = 0; i < width * height; ++i) {
            float r = src_data[i * 3 + 0];
            float g = src_data[i * 3 + 1];
            float b = src_data[i * 3 + 2];
            dst_data[i] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        }
        out_dst_c = 1;
    } else if (src_c == 1 && dst_c == 3) {
        // Grayscale to RGB (Replicate the single channel)
        for (int i = 0; i < width * height; ++i) {
            unsigned char gray = src_data[i];
            dst_data[i * 3 + 0] = gray;
            dst_data[i * 3 + 1] = gray;
            dst_data[i * 3 + 2] = gray;
        }
        out_dst_c = 3;
    } else if (src_c == 4 && dst_c == 3) {
        // RGBA to RGB (Discard Alpha)
        for (int i = 0; i < width * height; ++i) {
            dst_data[i * 3 + 0] = src_data[i * 4 + 0]; // R
            dst_data[i * 3 + 1] = src_data[i * 4 + 1]; // G
            dst_data[i * 3 + 2] = src_data[i * 4 + 2]; // B
        }
        out_dst_c = 3;
    } else if (src_c == 3 && dst_c == 4) {
        // RGB to RGBA (Add Alpha = 255)
        for (int i = 0; i < width * height; ++i) {
            dst_data[i * 4 + 0] = src_data[i * 3 + 0]; // R
            dst_data[i * 4 + 1] = src_data[i * 3 + 1]; // G
            dst_data[i * 4 + 2] = src_data[i * 3 + 2]; // B
            dst_data[i * 4 + 3] = 255;                 // A
        }
        out_dst_c = 4;
    } else {
        // Add other specific conversions as needed, or implement a more generic approach
        std::cerr << "Warning: Unsupported channel conversion " << src_c << " -> " << dst_c << ". Returning nullptr.\n";
        free(dst_data);
        out_dst_c = 0;
        return nullptr; 
    }

    return dst_data;
}


static void dump_tensor_attr(rknn3_tensor_attr* attrs)
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

    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, "
           "layout=%s, dtype=%s, core_id=%d, qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(),
           attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id,
           rknn3_get_qnt_type_string(attrs->qnt_type));
}

/**
 * 1. scale the image to a specified size proportionally
 * 2. fill the lower right corner with black
 */
static int convert_img_input(rknn_app_context_t *app_ctx, image_buffer_t *img_in,
                             unsigned char **img_out, size_t *size)
{
    if (!app_ctx || !img_in)
        return RKNN3_ERR_ARGUMENT_INVALID;

    int src_w = img_in->width, src_h = img_in->height;
    int src_c = img_in->size / (src_w * src_h);
    /**
     * assume that src channels are equal to model channels
     */
    if (src_c != app_ctx->model_channel) {
        printf("error: src img channels[%d] doesn't match to model channels[%d]!\n",
               src_c, app_ctx->model_channel);
        return RKNN3_ERR_ARGUMENT_INVALID;
    }

    /* Resize Keep Aspect Ratio (to fit inside target) */
    float scale;
    int new_w, new_h;
    /* Use max possible size */
    size_t temp_size = static_cast<size_t>(app_ctx->model_height) * app_ctx->model_width *
                       app_ctx->model_channel * sizeof(unsigned char);
    unsigned char *temp_data = static_cast<unsigned char*>(malloc(temp_size));
    if (!temp_data) {
        printf("malloc buffer for temp_data failed\n");
        return RKNN3_ERR_OUT_OF_MEMORY;
    }

    resize_image(img_in->virt_addr, src_w, src_h, src_c, temp_data, app_ctx->model_width,
                 app_ctx->model_height, scale, new_w, new_h);

    /* Crop the oversized buffer to the actual new size */
    size_t actual_size = static_cast<size_t>(new_h) * new_w * app_ctx->model_channel * sizeof(unsigned char);
    unsigned char *actual_data = static_cast<unsigned char*>(malloc(actual_size));
    if (!actual_data) {
        printf("malloc buffer for actual_data failed\n");
        free(temp_data);
        return RKNN3_ERR_OUT_OF_MEMORY;
    }

    memcpy(actual_data, temp_data, actual_size);
    free(temp_data);

    /* Pad Right Bottom to Reach Target Size */
    size_t dst_size = static_cast<size_t>(app_ctx->model_height) * app_ctx->model_width *
                      app_ctx->model_channel * sizeof(unsigned char);
    unsigned char *dst_data = static_cast<unsigned char*>(malloc(dst_size));
    if (!dst_data) {
        printf("malloc buffer for dst_data failed\n");
        free(actual_data);
        return RKNN3_ERR_OUT_OF_MEMORY;
    }

    /* Initialize with pad value (black) */
    memset(dst_data, 0, dst_size);
    pad_image(actual_data, new_h, new_w, app_ctx->model_channel, dst_data,
              app_ctx->model_height, app_ctx->model_width, 0);
    free(actual_data);

    if (!dst_data) {
        printf("%s: nullptr\n", __func__);
    }

    *img_out = dst_data;
    *size = dst_size;
    return RKNN3_SUCCESS;
}

/* ---------------get outputs for RAFT Stereo----------------- */

static int save_rknn_disp_as_png(void *virt_addr, int H, int W, const char *out_path) {
    if (!virt_addr || H <= 0 || W <= 0) {
        printf("[vis] Invalid parameters\n");
        return -1;
    }

    float16 *fp16_data = static_cast<float16 *>(virt_addr);
    int total = H * W;

    std::vector<float> disp(total);
    float vmin = 1e10f, vmax = -1e10f;
    for (int i = 0; i < total; i++) {
        float val = fp16_to_fp32(fp16_data[i]);
        disp[i] = val;
        if (val < vmin) vmin = val;
        if (val > vmax) vmax = val;
    }
    printf("[vis] Disp stats: min=%.4f, max=%.4f\n", vmin, vmax);

    std::vector<unsigned char> rgb(total * 3);
    float range = vmax - vmin + 1e-6f;
    for (int i = 0; i < total; i++) {
        int idx = (int)((disp[i] - vmin) / range * 255.0f);
        idx = std::min(255, std::max(0, idx));
        rgb[i * 3 + 0] = plasma_lut[idx][0];
        rgb[i * 3 + 1] = plasma_lut[idx][1];
        rgb[i * 3 + 2] = plasma_lut[idx][2];
    }

    int ret = stbi_write_png(out_path, W, H, 3, rgb.data(), W * 3);
    if (ret) {
        printf("[vis] Saved disparity visualization -> %s (%dx%d)\n", out_path, W, H);
    } else {
        printf("[vis] Failed to save PNG: %s\n", out_path);
    }
    return ret ? 0 : -1;
}

int save_raft_stereo_output(rknn_app_context_t *app_ctx)
{
    rknn3_tensor_attr output_attrs0;
    int ret = RKNN3_SUCCESS;

    output_attrs0.index = 0;
    ret = rknn3_query(app_ctx->rknn_ctx, RKNN3_QUERY_OUTPUT_ATTR, &output_attrs0, sizeof(rknn3_tensor_attr));
    if (ret) {
        printf("rknn_query output tensors fail! ret=%d\n", ret);
        return ret;
    }

    size_t disp_size = static_cast<size_t>(output_attrs0.shape[0]) * output_attrs0.shape[1] *
                       output_attrs0.shape[2] * output_attrs0.shape[3] * sizeof(unsigned short);

    std::ofstream file("out.data", std::ios::binary);
    if (!file.is_open()) {
        printf("Failed to open file for writing\n");
        return RKNN3_ERR_OUTPUT_INVALID;
    }
    file.write(static_cast<const char*>(app_ctx->outputs[0].mem->virt_addr), disp_size);
    file.close();

    save_rknn_disp_as_png(app_ctx->outputs[0].mem->virt_addr, output_attrs0.shape[2],
                          output_attrs0.shape[3], "./out.png");
    printf("save data success\n");
    return RKNN3_SUCCESS;
}

/* ---------------Integrating RAFT into the rknn model framework----------------- */
int inference_raft_stereo_model(rknn_app_context_t *app_ctx, image_buffer_t *img_l,
                                image_buffer_t *img_r)
{
    unsigned char *dst_img_l, *dst_img_r;
    size_t dst_img_l_sz, dst_img_r_sz;

    struct timeval start_time, end_time;
    double preprocess_time, inference_time, postprocess_time;

    int ret = RKNN3_SUCCESS;

    if (!app_ctx || !img_l || !img_r)
        return RKNN3_ERR_ARGUMENT_INVALID;

    /* record preprocess time */
    gettimeofday(&start_time, NULL);

    /* convert input image */
    ret = convert_img_input(app_ctx, img_l, &dst_img_l, &dst_img_l_sz);
    if (ret) {
        printf("convert left img failed\n");
        goto out;
    }

    ret = convert_img_input(app_ctx, img_r, &dst_img_r, &dst_img_r_sz);
    if (ret) {
        printf("convert right img failed\n");
        goto out;
    }

    /* set input data for model */
    memcpy(app_ctx->inputs[0].mem->virt_addr, dst_img_l, dst_img_l_sz);
    memcpy(app_ctx->inputs[1].mem->virt_addr, dst_img_r, dst_img_r_sz);
    /* calculate preprocess time */
    gettimeofday(&end_time, NULL);
    preprocess_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                      (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    printf("Pre-process time: %.2f ms\n", preprocess_time);
    /* record reasoning start time */
    gettimeofday(&start_time, NULL);

    stbi_write_jpg("convert_left.jpg", app_ctx->model_width, app_ctx->model_height,
                    app_ctx->model_channel, app_ctx->inputs[0].mem->virt_addr, 95);

    stbi_write_jpg("convert_right.jpg", app_ctx->model_width, app_ctx->model_height,
                    app_ctx->model_channel, app_ctx->inputs[1].mem->virt_addr, 95);
    /* sync inputs */
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret) {
            printf("rknn3_mem_sync input[%d] fail! ret=%d\n", i, ret);
            goto out;
        }
    }

    /* running */
    ret = rknn3_run(app_ctx->rknn_ctx, app_ctx->inputs, app_ctx->io_num.n_input,
                    app_ctx->outputs, app_ctx->io_num.n_output);
    if (ret) {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    /* sync outputs */
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        ret = rknn3_mem_sync(app_ctx->rknn_ctx, app_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret) {
            printf("rknn3_mem_sync output[%d] fail! ret=%d\n", i, ret);
            goto out;
        }
    }

    /* calculate reasoning time */
    gettimeofday(&end_time, NULL);
    inference_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    printf("Reasoning time: %.2f ms\n", inference_time);

out:
    if (dst_img_l)
        free(dst_img_l);

    if (dst_img_r)
        free(dst_img_r);

    return ret;
}

int init_raft_stereo_model(const char *model_path, const char* weight_path,
                           rknn_app_context_t *app_ctx, uint32_t core_mask)
{
    int ret;
    rknn3_context ctx = 0;
    rknn3_config config;

    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;

    /* RKNN init */
    ret = rknn3_init(&ctx, NULL);
    if (ret) {
        printf("rknn_init fail! ret=%d\n", ret);
        return ret;
    }

    /* Load RKNN Model */
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret) {
        printf("rknn load model fail! ret=%d\n", ret);
        return ret;
    }

    /* Init RKNN Model */
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        return ret;
    }

    /* Get Model Input Output Number */
    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret) {
        printf("rknn_query io_num fail! ret=%d\n", ret);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    /* Get Model Input info */
    printf("input tensors:\n");
    rknn3_tensor_attr input_attrs[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret) {
            printf("rknn_query input tensors fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    /* Get Model Output info */
    rknn3_tensor_attr output_attrs[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret) {
            printf("rknn_query output tensors fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    /* config context */
    app_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    app_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    app_ctx->io_num = io_num;
    app_ctx->rknn_ctx = ctx;
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->inputs[i].mem  = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id,
                                                   RKNN3_FLAG_MEMORY_CACHEABLE);
        app_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(app_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->outputs[i].mem  = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id,
                                                    RKNN3_FLAG_MEMORY_CACHEABLE);
        app_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(app_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    /* TODO */
    if (output_attrs[0].qnt_type == RKNN3_TENSOR_PER_LAYER_ASYMMETRIC &&
        output_attrs[0].dtype == RKNN3_TENSOR_INT8) {
        app_ctx->is_quant = true;
    } else {
        app_ctx->is_quant = false;
    }

    if (input_attrs[0].layout == RKNN3_TENSOR_NHWC) {
        printf("model is NHWC input fmt\n");
        app_ctx->model_channel = input_attrs[0].shape[3];
        app_ctx->model_height = input_attrs[0].shape[1];
        app_ctx->model_width = input_attrs[0].shape[2];
    } else {
        printf("model is NCHW input fmt\n");
        app_ctx->model_height = input_attrs[0].shape[1];
        app_ctx->model_width = input_attrs[0].shape[2];
        app_ctx->model_channel = input_attrs[0].shape[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return RKNN3_SUCCESS;
}

int release_raft_stereo_model(rknn_app_context_t *app_ctx)
{
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        if (app_ctx->inputs[i].mem) {
            rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->inputs[i].mem);
            app_ctx->inputs[i].mem = NULL;
        }
        if (app_ctx->inputs[i].attr != NULL) {
            free(app_ctx->inputs[i].attr);
            app_ctx->inputs[i].attr = NULL;
        }
    }

    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        if (app_ctx->outputs[i].mem) {
            rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->outputs[i].mem);
            app_ctx->outputs[i].mem = NULL;
        }
        if (app_ctx->outputs[i].attr != NULL) {
            free(app_ctx->outputs[i].attr);
            app_ctx->outputs[i].attr = NULL;
        }
    }

    if (app_ctx->rknn_ctx) {
        rknn3_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }

    return RKNN3_SUCCESS;
}