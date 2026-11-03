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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "rknn_dots_ocr_vision.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

static rknn3_init_extend make_rknn3_init_extend(const char* tag)
{
    // NOTE: intentionally not thread-safe; the demo initializes model contexts
    // sequentially on a single thread, so a static buffer avoids per-call allocation.
    static char selected_device_id[RKNN3_MAX_DEV_LEN] = {0};
    rknn3_init_extend init_extend;
    memset(&init_extend, 0, sizeof(init_extend));

    const char* env_device_id = getenv("RKNN3_DEVICE_ID");
    if (env_device_id && env_device_id[0] != '\0') {
        if (strlen(env_device_id) >= sizeof(selected_device_id)) {
            // Reject over-long values instead of silently truncating them
            printf("[%s] RKNN3_DEVICE_ID too long (>= %d bytes), ignored\n", tag, (int)sizeof(selected_device_id));
        } else {
            snprintf(selected_device_id, sizeof(selected_device_id), "%s", env_device_id);
            init_extend.device_id = selected_device_id;
            printf("[%s] use RKNN3_DEVICE_ID=%s\n", tag, selected_device_id);
            return init_extend;
        }
    }

    rknn3_devices devices;
    memset(&devices, 0, sizeof(devices));
    int ret = rknn3_find_devices(&devices);
    if (ret == 0) {
        if (devices.n_devices > 1) {
            printf("[%s] multiple RKNN3 devices found, using device_id=%s\n",
                   tag, devices.devices[0].id);
            snprintf(selected_device_id, sizeof(selected_device_id), "%s", devices.devices[0].id);
            init_extend.device_id = selected_device_id;
        }
    }
    return init_extend;
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

    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, "
           "qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}


int init_dots_ocr_vision(rknn_dots_ocr_vision_context* vision_ctx, const char* model_path, const char* weight_path, uint32_t core_mask, uint32_t model_width, uint32_t model_height)
{
    int ret;
    rknn3_context ctx = 0;
    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1;

    rknn3_input_output_num io_num;
    std::vector<rknn3_tensor_attr> input_attrs;
    std::vector<rknn3_tensor_attr> output_attrs;

    rknn3_init_extend init_extend = make_rknn3_init_extend("VisionInit");
    ret = rknn3_init(&ctx, init_extend.device_id ? &init_extend : NULL);
    if (ret < 0)
    {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        goto cleanup;
    }

    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        goto cleanup;
    }

    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        goto cleanup;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    input_attrs.resize(io_num.n_input);
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            goto cleanup;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    printf("output tensors:\n");
    output_attrs.resize(io_num.n_output);
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            goto cleanup;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    vision_ctx->pruned_version_flag = 0;
    vision_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    if (!vision_ctx->inputs) {
        printf("malloc vision inputs failed!\n");
        ret = -1;
        goto cleanup;
    }
    memset(vision_ctx->inputs, 0, io_num.n_input * sizeof(rknn3_tensor));
    vision_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    if (!vision_ctx->outputs) {
        printf("malloc vision outputs failed!\n");
        ret = -1;
        goto cleanup;
    }
    memset(vision_ctx->outputs, 0, io_num.n_output * sizeof(rknn3_tensor));
    vision_ctx->rknn_ctx = ctx;
    vision_ctx->io_num = io_num;
    for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
        vision_ctx->inputs[i].mem  = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        vision_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (!vision_ctx->inputs[i].mem || !vision_ctx->inputs[i].attr) {
            printf("Failed to create vision input tensor %d\n", i);
            ret = -1;
            goto cleanup;
        }
        memcpy(vision_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
        vision_ctx->outputs[i].mem  = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        vision_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (!vision_ctx->outputs[i].mem || !vision_ctx->outputs[i].attr) {
            printf("Failed to create vision output tensor %d\n", i);
            ret = -1;
            goto cleanup;
        }
        memcpy(vision_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    if (input_attrs[0].layout == RKNN3_TENSOR_NCHW)
    {
        printf("model is NCHW input layout\n");
        vision_ctx->model_channel = input_attrs[0].shape[1];
        vision_ctx->model_height = input_attrs[0].shape[2];
        vision_ctx->model_width = input_attrs[0].shape[3];
        printf("input image height=%d, input image width=%d, input image channel=%d\n",
                vision_ctx->model_height, vision_ctx->model_width, vision_ctx->model_channel);
    }
    else if (input_attrs[0].layout == RKNN3_TENSOR_NHWC)
    {
        printf("model is NHWC input layout\n");
        vision_ctx->model_channel = input_attrs[0].shape[3];
        vision_ctx->model_height = input_attrs[0].shape[1];
        vision_ctx->model_width = input_attrs[0].shape[2];
        printf("input image height=%d, input image width=%d, input image channel=%d\n",
                vision_ctx->model_height, vision_ctx->model_width, vision_ctx->model_channel);
    }
    else
    {
        int feature_size = MODEL_WIDTH * MODEL_HEIGHT / PATCH_SIZE / PATCH_SIZE;
        if(model_width != 0 && model_height != 0) {
            feature_size = model_width * model_height / PATCH_SIZE / PATCH_SIZE;
        }
        if(input_attrs[0].shape[0] != feature_size){
            printf("Please note: model_width=%d and model_height=%d do not match the model's expected input dimensions.\n", model_width == 0 ? MODEL_WIDTH : model_width, model_height == 0 ? MODEL_HEIGHT : model_height);
            ret = -1;
            goto cleanup;
        }
        vision_ctx->pruned_version_flag = 1;
        vision_ctx->model_channel = 3;
        if(model_width != 0 && model_height != 0) {
            vision_ctx->model_width = (int)model_width;
            vision_ctx->model_height = (int)model_height;
        } else {
            vision_ctx->model_width = (int)MODEL_WIDTH;
            vision_ctx->model_height = (int)MODEL_HEIGHT;
        }
        printf("model is UNDEFINED!\n");
        printf("model_width=%d model_height=%d \n", vision_ctx->model_width, vision_ctx->model_height);
    }

    if (output_attrs[0].layout == RKNN3_TENSOR_UNDEFINED)
    {
        printf("model is UNDEFINED output layout\n");
        vision_ctx->embeds_shape = vision_ctx->outputs[0].attr->shape;
        vision_ctx->embeds_ndims = vision_ctx->outputs[0].attr->n_dims;
    }
    else
    {
        printf("model is not UNDEFINED output layout, model output error!\n");
        ret = -1;
        goto cleanup;
    }

    return ret;

cleanup:
    if (vision_ctx->inputs) {
        free(vision_ctx->inputs);
        vision_ctx->inputs = NULL;
    }
    if (vision_ctx->outputs) {
        free(vision_ctx->outputs);
        vision_ctx->outputs = NULL;
    }
    if (ctx) {
        rknn3_destroy(ctx);
        ctx = 0;
    }
    return ret;
}

int release_dots_ocr_vision(rknn_dots_ocr_vision_context* vision_ctx)
{
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
    if (vision_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(vision_ctx->rknn_ctx);
        vision_ctx->rknn_ctx = 0;
    }
    return 0;
}

void normalize_and_to_nchw(const unsigned char *src, float16 *dst, int width, int height) {
    float mean[3] = {122.7f, 116.74f, 104.09f};
    float std[3] = {68.5f, 66.6f, 70.3f};
    int num_pixels = width * height;

    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int nhwc_idx = (h * width + w) * 3 + c;
                int nchw_idx = c * num_pixels + h * width + w;
                dst[nchw_idx] = fp32_to_fp16(((float)src[nhwc_idx] - mean[c]) / std[c]);
            }
        }
    }
}

void transpose(float16 *in, float16 *out, int model_width, int model_height) {
    int64_t dims[8] = {1, 3, int(model_height/PATCH_MERGE_SIZE), SPATIAL_MERGE_SIZE, PATCH_SIZE, int(model_width/PATCH_MERGE_SIZE), SPATIAL_MERGE_SIZE, PATCH_SIZE};
    //[temporal_patch_size, channel, grid_h // merge_size, merge_size, patch_size, grid_w // merge_size, merge_size, patch_size]
    int perm[8] = {2, 5, 3, 6, 1, 0, 4, 7};
    int64_t new_dims[8];
    for (int k = 0; k < 8; k++) {
        new_dims[k] = dims[perm[k]];
    }
    int64_t total = 1;
    for (int k = 0; k < 8; k++) {
        total *= dims[k];
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
        int64_t flat_in = i[0] * strides[0] + i[1] * strides[1] + i[2] * strides[2] +
                          i[3] * strides[3] + i[4] * strides[4] + i[5] * strides[5] +
                          i[6] * strides[6] + i[7] * strides[7];
        out[out_idx++] = in[flat_in];
    }}}}}}}}
}

int inference_dots_ocr_vision(rknn_dots_ocr_vision_context* vision_ctx, image_buffer_t* img, float16* img_embeds)
{
    if ((!vision_ctx) || (!img))
    {
        printf("vision_ctx or img is NULL\n");
        return -1;
    }
    int ret;
    image_buffer_t dst_img;
    float16* norm_data;
    memset(&dst_img, 0, sizeof(image_buffer_t));

    // Pre Process
    dst_img.width     = vision_ctx->model_width;
    dst_img.height    = vision_ctx->model_height;
    dst_img.format    = IMAGE_FORMAT_RGB888;
    dst_img.size      = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);

    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        goto out;
    }

    ret = convert_image(img, &dst_img, NULL, NULL, 0);
    if (ret < 0)
    {
        printf("convert_image fail! ret=%d\n", ret);
        goto out;
    }

    if(vision_ctx->pruned_version_flag == 1) {
        norm_data = (float16*)malloc(dst_img.size * 2); // float16 = 2 * uint8
        if (!norm_data) {
            printf("malloc norm_data failed!\n");
            ret = -1;
            goto out;
        }
        normalize_and_to_nchw(dst_img.virt_addr, norm_data, vision_ctx->model_width, vision_ctx->model_height);
        transpose((float16 *)norm_data, (float16 *)(vision_ctx->inputs[0].mem->virt_addr), vision_ctx->model_width, vision_ctx->model_height);
        free(norm_data);
        norm_data = NULL;
    } else {
        memcpy(vision_ctx->inputs[0].mem->virt_addr, (uint8_t*)dst_img.virt_addr, dst_img.size);
    }

    // Sync Inputs
    for (int i = 0; i < vision_ctx->io_num.n_input; i++)
    {
        ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS)
        {
            printf("rknn3_mem_sync input[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }

    // Run
    ret = rknn3_run(vision_ctx->rknn_ctx, vision_ctx->inputs, vision_ctx->io_num.n_input, vision_ctx->outputs, vision_ctx->io_num.n_output);
    if (ret < 0)
    {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Sync Outputs
    for (int i = 0; i < vision_ctx->io_num.n_output; i++)
    {
        ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS)
        {
            printf("rknn3_mem_sync output[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }

    // Get Output
    memcpy(img_embeds, (float16*)vision_ctx->outputs[0].mem->virt_addr, vision_ctx->outputs[0].mem->size);

out:
    if (dst_img.virt_addr != NULL)
    {
        free(dst_img.virt_addr);
    }

    return ret;
}
