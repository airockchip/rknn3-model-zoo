// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <math.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>
#include <memory>
#include <new>
#include <string>

#include "common.h"
#include "image_utils.h"
#include "rknn_mage_vl_vision.h"

// Runtime versions before orig_shape was added use a 512-byte tensor attr.
// The fields through qnt_info have identical offsets; core_id is at byte 504.
#define LEGACY_TENSOR_ATTR_SIZE 512
#define LEGACY_CORE_ID_OFFSET 504

static int tensor_attr_core_id(rknn3_tensor_attr* attr, int legacy_abi)
{
    if (legacy_abi) {
        unsigned char* raw_attr = reinterpret_cast<unsigned char*>(attr);
        return *reinterpret_cast<int32_t*>(raw_attr + LEGACY_CORE_ID_OFFSET);
    }
    return attr->core_id;
}

static int query_tensor_attr(rknn3_context ctx, rknn3_query_cmd cmd,
                             rknn3_tensor_attr* attr, int* legacy_abi)
{
    int index = attr->index;
    int ret = rknn3_query(ctx, cmd, attr, sizeof(rknn3_tensor_attr));
    if (ret >= 0) {
        *legacy_abi = 0;
        return ret;
    }
    memset(attr, 0, sizeof(rknn3_tensor_attr));
    attr->index = index;
    ret = rknn3_query(ctx, cmd, attr, LEGACY_TENSOR_ATTR_SIZE);
    if (ret >= 0) {
        *legacy_abi = 1;
        printf("Use legacy 512-byte rknn3_tensor_attr ABI\n");
    }
    return ret;
}

static void dump_tensor_attr(rknn3_tensor_attr* attrs, int legacy_abi)
{
    std::string shape_str;
    for (int i = 0; i < attrs->n_dims; i++) {
        shape_str += std::to_string(attrs->shape[i]);
        if (i + 1 < attrs->n_dims) {
            shape_str += ", ";
        }
    }
    printf("Tensor: name=%s, shape=[%s], aligned_size=%" PRIu64
           ", layout=%s, dtype=%s, core_id=%d\n",
           attrs->name, shape_str.c_str(),
           static_cast<uint64_t>(attrs->aligned_size),
           rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype),
           tensor_attr_core_id(attrs, legacy_abi));
}

static int find_tensor_index(rknn3_tensor_attr* attrs, int count, const char* name)
{
    for (int i = 0; i < count; i++) {
        if (strcmp(attrs[i].name, name) == 0) {
            return i;
        }
    }
    return -1;
}

static void fill_pixel_values(const unsigned char* rgb, float16* pixel_values,
                              int width, int height)
{
    const float mean[3] = {0.48145466f, 0.45782750f, 0.40821073f};
    const float std[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    const int grid_h = height / MAGE_VL_PATCH_SIZE;
    const int grid_w = width / MAGE_VL_PATCH_SIZE;
    size_t output_index = 0;

    // Match Qwen2VLImageProcessor block layout:
    // [grid_h/2, grid_w/2, merge_h, merge_w, channel, patch_h, patch_w].
    for (int block_h = 0; block_h < grid_h / MAGE_VL_MERGE_SIZE; block_h++) {
        for (int block_w = 0; block_w < grid_w / MAGE_VL_MERGE_SIZE; block_w++) {
            for (int merge_h = 0; merge_h < MAGE_VL_MERGE_SIZE; merge_h++) {
                for (int merge_w = 0; merge_w < MAGE_VL_MERGE_SIZE; merge_w++) {
                    int patch_h = block_h * MAGE_VL_MERGE_SIZE + merge_h;
                    int patch_w = block_w * MAGE_VL_MERGE_SIZE + merge_w;
                    for (int channel = 0; channel < 3; channel++) {
                        for (int y = 0; y < MAGE_VL_PATCH_SIZE; y++) {
                            for (int x = 0; x < MAGE_VL_PATCH_SIZE; x++) {
                                int image_y = patch_h * MAGE_VL_PATCH_SIZE + y;
                                int image_x = patch_w * MAGE_VL_PATCH_SIZE + x;
                                int image_index = (image_y * width + image_x) * 3 + channel;
                                float value = rgb[image_index] / 255.0f;
                                pixel_values[output_index++] = fp32_to_fp16(
                                    (value - mean[channel]) / std[channel]);
                            }
                        }
                    }
                }
            }
        }
    }
}

static void fill_rotary_pos_emb(float16* rotary_pos_emb, int width, int height)
{
    const int section_sizes[3] = {8, 12, 12};
    const int grid_h = height / MAGE_VL_PATCH_SIZE;
    const int grid_w = width / MAGE_VL_PATCH_SIZE;
    size_t patch_index = 0;

    for (int block_h = 0; block_h < grid_h / MAGE_VL_MERGE_SIZE; block_h++) {
        for (int block_w = 0; block_w < grid_w / MAGE_VL_MERGE_SIZE; block_w++) {
            for (int merge_h = 0; merge_h < MAGE_VL_MERGE_SIZE; merge_h++) {
                for (int merge_w = 0; merge_w < MAGE_VL_MERGE_SIZE; merge_w++) {
                    const int positions[3] = {
                        0,
                        block_h * MAGE_VL_MERGE_SIZE + merge_h,
                        block_w * MAGE_VL_MERGE_SIZE + merge_w
                    };
                    float half[MAGE_VL_ROTARY_DIM / 2];
                    int offset = 0;
                    for (int axis = 0; axis < 3; axis++) {
                        for (int i = 0; i < section_sizes[axis]; i++) {
                            float inv_freq = 1.0f / powf(10000.0f,
                                (float)i / (float)section_sizes[axis]);
                            half[offset++] = positions[axis] * inv_freq;
                        }
                    }
                    for (int i = 0; i < MAGE_VL_ROTARY_DIM / 2; i++) {
                        float16 value = fp32_to_fp16(half[i]);
                        rotary_pos_emb[patch_index * MAGE_VL_ROTARY_DIM + i] = value;
                        rotary_pos_emb[patch_index * MAGE_VL_ROTARY_DIM +
                                       MAGE_VL_ROTARY_DIM / 2 + i] = value;
                    }
                    patch_index++;
                }
            }
        }
    }
}

int init_mage_vl_vision(rknn_mage_vl_vision_context* vision_ctx,
                        const char* model_path, const char* weight_path,
                        uint32_t core_mask, uint32_t model_width,
                        uint32_t model_height)
{
    return init_mage_vl_vision_ex(vision_ctx, model_path, weight_path,
                                  core_mask, model_width, model_height, 1);
}

int init_mage_vl_vision_ex(rknn_mage_vl_vision_context* vision_ctx,
                           const char* model_path, const char* weight_path,
                           uint32_t core_mask, uint32_t model_width,
                           uint32_t model_height, int user_mem_internal)
{
    if (vision_ctx == NULL || model_path == NULL || weight_path == NULL) {
        printf("init_mage_vl_vision: context or model path is NULL\n");
        return -1;
    }

    int ret = 0;
    rknn3_context ctx = 0;
    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = user_mem_internal;

    ret = rknn3_init(&ctx, NULL);
    if (ret < 0) {
        printf("rknn3_init failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn3_load_model_from_path failed! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn3_model_init failed! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }

    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0 || io_num.n_input != 2 || io_num.n_output != 1) {
        printf("Mage-VL Vision expects 2 inputs and 1 output\n");
        rknn3_destroy(ctx);
        return -1;
    }

    rknn3_tensor_attr input_attrs[2];
    rknn3_tensor_attr output_attrs[1];
    int legacy_abi = 0;
    memset(input_attrs, 0, sizeof(input_attrs));
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = query_tensor_attr(ctx, RKNN3_QUERY_INPUT_ATTR, &input_attrs[i],
                                &legacy_abi);
        if (ret < 0) {
            rknn3_destroy(ctx);
            return ret;
        }
        dump_tensor_attr(&input_attrs[i], legacy_abi);
    }
    output_attrs[0].index = 0;
    ret = query_tensor_attr(ctx, RKNN3_QUERY_OUTPUT_ATTR, &output_attrs[0],
                            &legacy_abi);
    if (ret < 0) {
        rknn3_destroy(ctx);
        return ret;
    }
    dump_tensor_attr(&output_attrs[0], legacy_abi);

    int pixel_index = find_tensor_index(input_attrs, 2, "pixel_values");
    int rotary_index = find_tensor_index(input_attrs, 2, "rotary_pos_emb");
    if (pixel_index < 0 || rotary_index < 0) {
        printf("Cannot find pixel_values or rotary_pos_emb input\n");
        rknn3_destroy(ctx);
        return -1;
    }

    int width = model_width == 0 ? MODEL_WIDTH : model_width;
    int height = model_height == 0 ? MODEL_HEIGHT : model_height;
    if (width % (MAGE_VL_PATCH_SIZE * MAGE_VL_MERGE_SIZE) != 0 ||
        height % (MAGE_VL_PATCH_SIZE * MAGE_VL_MERGE_SIZE) != 0) {
        printf("model width and height must be multiples of 32\n");
        rknn3_destroy(ctx);
        return -1;
    }
    int spatial_patch_num = width / MAGE_VL_PATCH_SIZE * height / MAGE_VL_PATCH_SIZE;
    int patch_num = (int)input_attrs[pixel_index].shape[0];
    if ((patch_num != spatial_patch_num && patch_num != 4 * spatial_patch_num) ||
        input_attrs[pixel_index].shape[1] != 3 * MAGE_VL_PATCH_SIZE * MAGE_VL_PATCH_SIZE) {
        printf("pixel_values shape does not match one or four %dx%d frames\n",
               width, height);
        rknn3_destroy(ctx);
        return -1;
    }

    vision_ctx->rknn_ctx = ctx;
    vision_ctx->io_num = io_num;
    vision_ctx->pixel_values_index = pixel_index;
    vision_ctx->rotary_pos_emb_index = rotary_index;
    vision_ctx->model_channel = 3;
    vision_ctx->model_width = width;
    vision_ctx->model_height = height;
    vision_ctx->patch_num = patch_num;
    vision_ctx->legacy_tensor_attr_abi = legacy_abi;
    vision_ctx->inputs = new (std::nothrow) rknn3_tensor[io_num.n_input]();
    vision_ctx->outputs = new (std::nothrow) rknn3_tensor[io_num.n_output]();
    if (vision_ctx->inputs == NULL || vision_ctx->outputs == NULL) {
        release_mage_vl_vision(vision_ctx);
        return -1;
    }

    for (int i = 0; i < io_num.n_input; i++) {
        vision_ctx->inputs[i].attr =
            new (std::nothrow) rknn3_tensor_attr;
        if (vision_ctx->inputs[i].attr == NULL) {
            release_mage_vl_vision(vision_ctx);
            return -1;
        }
        memcpy(vision_ctx->inputs[i].attr, &input_attrs[i], sizeof(rknn3_tensor_attr));
        vision_ctx->inputs[i].mem = rknn3_create_mem(ctx, input_attrs[i].aligned_size,
            tensor_attr_core_id(&input_attrs[i], legacy_abi),
            RKNN3_FLAG_MEMORY_CACHEABLE);
        if (vision_ctx->inputs[i].mem == NULL) {
            release_mage_vl_vision(vision_ctx);
            return -1;
        }
    }
    vision_ctx->outputs[0].attr = new (std::nothrow) rknn3_tensor_attr;
    if (vision_ctx->outputs[0].attr == NULL) {
        release_mage_vl_vision(vision_ctx);
        return -1;
    }
    memcpy(vision_ctx->outputs[0].attr, &output_attrs[0], sizeof(rknn3_tensor_attr));
    vision_ctx->outputs[0].mem = rknn3_create_mem(ctx, output_attrs[0].aligned_size,
        tensor_attr_core_id(&output_attrs[0], legacy_abi),
        RKNN3_FLAG_MEMORY_CACHEABLE);
    if (vision_ctx->outputs[0].mem == NULL) {
        release_mage_vl_vision(vision_ctx);
        return -1;
    }
    vision_ctx->embeds_shape = vision_ctx->outputs[0].attr->shape;
    vision_ctx->embeds_ndims = vision_ctx->outputs[0].attr->n_dims;
    return 0;
}

int release_mage_vl_vision(rknn_mage_vl_vision_context* vision_ctx)
{
    if (vision_ctx == NULL) {
        return 0;
    }
    for (int i = 0; vision_ctx->inputs != NULL && i < vision_ctx->io_num.n_input; i++) {
        if (vision_ctx->inputs[i].mem != NULL) {
            rknn3_destroy_mem(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem);
        }
        delete vision_ctx->inputs[i].attr;
        vision_ctx->inputs[i].attr = NULL;
    }
    for (int i = 0; vision_ctx->outputs != NULL && i < vision_ctx->io_num.n_output; i++) {
        if (vision_ctx->outputs[i].mem != NULL) {
            rknn3_destroy_mem(vision_ctx->rknn_ctx, vision_ctx->outputs[i].mem);
        }
        delete vision_ctx->outputs[i].attr;
        vision_ctx->outputs[i].attr = NULL;
    }
    delete[] vision_ctx->inputs;
    delete[] vision_ctx->outputs;
    vision_ctx->inputs = NULL;
    vision_ctx->outputs = NULL;
    if (vision_ctx->rknn_ctx != 0) {
        rknn3_destroy(vision_ctx->rknn_ctx);
        vision_ctx->rknn_ctx = 0;
    }
    return 0;
}

static int inference_mage_vl_vision_impl(
    rknn_mage_vl_vision_context* vision_ctx, image_buffer_t* img,
    const float16* external_rotary_pos_emb, size_t rotary_element_count,
    float16* img_embeds)
{
    if (vision_ctx == NULL || img == NULL || img_embeds == NULL) {
        printf("vision_ctx, img or img_embeds is NULL\n");
        return -1;
    }

    int ret = 0;
    image_buffer_t resized;
    memset(&resized, 0, sizeof(resized));
    int target_width = vision_ctx->model_width;
    int target_height = vision_ctx->model_height;
    if (external_rotary_pos_emb != NULL) {
        // cv-preinfer preserves aspect ratio (for example 288x512) while keeping
        // the same Patch576 budget. The RKNN input is a flat [576, 768] patch
        // tensor, so Codec canvases need not be forced to the image-mode 384x384.
        if (img->width % (MAGE_VL_PATCH_SIZE * MAGE_VL_MERGE_SIZE) != 0 ||
            img->height % (MAGE_VL_PATCH_SIZE * MAGE_VL_MERGE_SIZE) != 0 ||
            (img->width / MAGE_VL_PATCH_SIZE) *
                    (img->height / MAGE_VL_PATCH_SIZE) != vision_ctx->patch_num) {
            printf("Codec canvas must be Patch32 aligned and contain %d patches; "
                   "got %dx%d\n", vision_ctx->patch_num, img->width, img->height);
            return -1;
        }
        target_width = img->width;
        target_height = img->height;
    }
    resized.width = target_width;
    resized.height = target_height;
    resized.format = IMAGE_FORMAT_RGB888;
    resized.size = get_image_size(&resized);
    std::unique_ptr<unsigned char[]> resized_data(
        new (std::nothrow) unsigned char[resized.size]);
    if (!resized_data) {
        return -1;
    }
    resized.virt_addr = resized_data.get();
    ret = convert_image(img, &resized, NULL, NULL, 0);
    if (ret < 0) {
        printf("convert_image failed! ret=%d\n", ret);
        return ret;
    }

    fill_pixel_values(resized.virt_addr,
        static_cast<float16*>(vision_ctx->inputs[
            vision_ctx->pixel_values_index].mem->virt_addr),
        resized.width, resized.height);
    rknn3_tensor* rotary_input =
        &vision_ctx->inputs[vision_ctx->rotary_pos_emb_index];
    if (external_rotary_pos_emb != NULL) {
        if (rotary_element_count != rotary_input->attr->n_elems) {
            printf("external rotary element count mismatch: got=%zu expected=%u\n",
                   rotary_element_count, rotary_input->attr->n_elems);
            return -1;
        }
        memcpy(rotary_input->mem->virt_addr, external_rotary_pos_emb,
               rotary_element_count * sizeof(float16));
    } else {
        fill_rotary_pos_emb(static_cast<float16*>(rotary_input->mem->virt_addr),
                            resized.width, resized.height);
    }
    for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
        ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem,
                             RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            return ret;
        }
    }
    ret = rknn3_run(vision_ctx->rknn_ctx, vision_ctx->inputs,
                    vision_ctx->io_num.n_input, vision_ctx->outputs,
                    vision_ctx->io_num.n_output);
    if (ret < 0) {
        return ret;
    }
    ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->outputs[0].mem,
                         RKNN3_MEMORY_SYNC_FROM_DEVICE);
    if (ret != RKNN3_SUCCESS) {
        return ret;
    }
    memcpy(img_embeds, vision_ctx->outputs[0].mem->virt_addr,
           (size_t)vision_ctx->outputs[0].attr->n_elems * sizeof(float16));
    return 0;
}

int inference_mage_vl_vision(rknn_mage_vl_vision_context* vision_ctx,
                             image_buffer_t* img, float16* img_embeds)
{
    return inference_mage_vl_vision_impl(vision_ctx, img, NULL, 0,
                                         img_embeds);
}

int inference_mage_vl_vision_with_rotary(
    rknn_mage_vl_vision_context* vision_ctx, image_buffer_t* img,
    const float16* rotary_pos_emb, size_t rotary_element_count,
    float16* img_embeds)
{
    if (rotary_pos_emb == NULL) {
        printf("external rotary_pos_emb is NULL\n");
        return -1;
    }
    return inference_mage_vl_vision_impl(
        vision_ctx, img, rotary_pos_emb, rotary_element_count, img_embeds);
}
