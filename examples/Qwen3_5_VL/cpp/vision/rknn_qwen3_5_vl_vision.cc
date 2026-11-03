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

#include "rknn_qwen3_5_vl_vision.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"


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

static int setup_context_lora(rknn3_context ctx,
                              const char* lora_weight_path,
                              rknn3_lora* lora_out)
{
    if (!lora_weight_path || !lora_out) {
        return -1;
    }

    int ret = rknn3_lora_init(ctx, lora_weight_path);
    if (ret < 0) {
        printf("Failed to initialize Vision LoRA: ret=%d\n", ret);
        return ret;
    }

    // RKNN3 has no lora_deinit API. Registration remains owned by ctx and is
    // released by rknn3_destroy; before lora_load succeeds there is no
    // descriptor that can be passed to rknn3_lora_unload.

    uint32_t n_lora = 0;
    ret = rknn3_query(ctx, RKNN3_QUERY_LORA_NUM, &n_lora, sizeof(n_lora));
    if (ret < 0 || n_lora == 0) {
        printf("No Vision LoRA found: ret=%d count=%u\n", ret, n_lora);
        return ret < 0 ? ret : -1;
    }
    if (n_lora > RKNN3_MAX_LORA_NUM) {
        printf("Vision LoRA count exceeds maximum limit: %u > %u\n",
               n_lora, (unsigned)RKNN3_MAX_LORA_NUM);
        return -1;
    }

    rknn3_lora lora_list[RKNN3_MAX_LORA_NUM] = {0};
    ret = rknn3_query(ctx, RKNN3_QUERY_LORA_INFO, lora_list, sizeof(lora_list));
    if (ret < 0) {
        printf("Failed to query Vision LoRA info: ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_lora_load(ctx, &lora_list[0]);
    if (ret < 0) {
        printf("Failed to load Vision LoRA: ret=%d\n", ret);
        return ret;
    }
    *lora_out = lora_list[0];
    printf("Vision LoRA loaded: name=%s scale=%.3f\n", lora_out->lora_name, lora_out->scale);
    return 0;
}

static int set_vision_lora_enabled(rknn_qwen3_5_vl_vision_context* vision_ctx,
                                   bool enable)
{
    if (!vision_ctx || !vision_ctx->lora_loaded ||
        vision_ctx->lora_enabled == enable) {
        return 0;
    }
    int ret = enable ? rknn3_lora_enable(vision_ctx->rknn_ctx, &vision_ctx->lora)
                     : rknn3_lora_disable(vision_ctx->rknn_ctx, &vision_ctx->lora);
    if (ret < 0) {
        printf("Failed to %s Vision LoRA: ret=%d\n",
               enable ? "enable" : "disable", ret);
        return ret;
    }
    vision_ctx->lora_enabled = enable;
    printf("Vision LoRA %s\n", enable ? "enabled" : "disabled");
    return 0;
}


int init_qwen3_5_vl_vision(rknn_qwen3_5_vl_vision_context* vision_ctx, const char* model_path, const char* weight_path, uint32_t core_mask, uint32_t model_width, uint32_t model_height)
{
    if (!vision_ctx) {
        printf("init_qwen3_5_vl_vision: vision_ctx is NULL\n");
        return -1;
    }
    int ret;
    rknn3_context ctx = 0;
    rknn3_config config;
    memset(&config, 0, sizeof(config));
    memset(vision_ctx, 0, sizeof(*vision_ctx));
    vision_ctx->lora_loaded = false;
    vision_ctx->lora_enabled = false;
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1; // 使用用户管理的internal内存

    // RKNN Init
    ret = rknn3_init(&ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        return ret;
    }

    //Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        return ret;
    }

    // Get Model Input Output Number
    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn3_tensor_attr input_attrs[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn3_tensor_attr output_attrs[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    vision_ctx->pruned_version_flag = 0;
    vision_ctx->inputs = (rknn3_tensor*)malloc(io_num.n_input * sizeof(rknn3_tensor));
    vision_ctx->outputs = (rknn3_tensor*)malloc(io_num.n_output * sizeof(rknn3_tensor));
    vision_ctx->rknn_ctx = ctx;
    vision_ctx->io_num = io_num;
    for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
        vision_ctx->inputs[i].mem  = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        vision_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        memcpy(vision_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
        vision_ctx->outputs[i].mem  = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        vision_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
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
        int feature_size = MODEL_WIDTH * MODEL_HEIGHT / 16 / 16;
        if(model_width != 0 && model_height != 0) {
            feature_size = model_width * model_height / 16 / 16;
        }
        if(input_attrs[0].shape[0] != feature_size){
            printf("Please note: model_width=%d and model_height=%d do not match the model's expected input dimensions. Please either update the default parameters or explicitly specify the correct model_width and model_height arguments in the command line.\n", model_width == 0 ? MODEL_WIDTH : model_width, model_height == 0 ? MODEL_HEIGHT : model_height);
            return -1;
        }
        vision_ctx->pruned_version_flag = 1;
        vision_ctx->model_channel = 3;
        // 由于vision模型被裁剪了，需要手动添加这些参数配置
        if(model_width != 0 && model_height != 0) {
            vision_ctx->model_width = (int)model_width;
            vision_ctx->model_height = (int)model_height;
        } else {
            vision_ctx->model_width = (int)MODEL_WIDTH;
            vision_ctx->model_height = (int)MODEL_HEIGHT;
        }
        printf("model is UNDEFINED!\n");
        printf("model_width=%d model_height=%d \n", vision_ctx->model_width,vision_ctx->model_height);
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
        return -1;
    }

    return ret;
}

int setup_qwen3_5_vl_vision_lora(rknn_qwen3_5_vl_vision_context* vision_ctx,
                                const char* lora_weight_path)
{
    if (!vision_ctx || !lora_weight_path) {
        return 0;
    }
    rknn3_lora lora = {0};
    int ret = setup_context_lora(vision_ctx->rknn_ctx, lora_weight_path, &lora);
    if (ret < 0) {
        return ret;
    }
    vision_ctx->lora = lora;
    vision_ctx->lora_loaded = true;
    vision_ctx->lora_enabled = false;
    return 0;
}

int release_qwen3_5_vl_vision(rknn_qwen3_5_vl_vision_context* vision_ctx)
{
    if (!vision_ctx) {
        return -1;
    }
    if (vision_ctx->lora_loaded) {
        if (vision_ctx->lora_enabled) {
            rknn3_lora_disable(vision_ctx->rknn_ctx, &vision_ctx->lora);
            vision_ctx->lora_enabled = false;
        }
        rknn3_lora_unload(vision_ctx->rknn_ctx, &vision_ctx->lora);
        vision_ctx->lora_loaded = false;
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
    if (vision_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(vision_ctx->rknn_ctx);
        vision_ctx->rknn_ctx = 0;
    }
    return 0;
}

void normalize_and_to_nchw(const unsigned char *src, float16 *dst, int width, int height) {
    float mean[3] = {127.5f, 127.5f, 127.5f};
    float std[3] = {127.5f, 127.5f, 127.5f};
    int num_pixels = width * height;
    
    // 转换到NCHW
    for (int c = 0; c < 3; ++c) {  // 遍历R,G,B通道
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int nhwc_idx = (h * width + w) * 3 + c;  // NHWC索引
                int nchw_idx = c * num_pixels + h * width + w;  // NCHW索引
                dst[nchw_idx] = fp32_to_fp16(((float)src[nhwc_idx] - mean[c]) / std[c]); // 归一化
            }
        }
    }   
}

void transpose(float16 *in, float16 *out, int model_width, int model_height) {
    int64_t dims[8] = {2,3,int(model_height/32),2,16,int(model_width/32),2,16};    
    //[temporal_patch_size,channel, grid_h // merge_size,  merge_size, patch_size, 
    // grid_w // merge_size, merge_size, patch_size]
    int perm[8] = {2, 5, 3, 6, 1, 0, 4, 7};
    int64_t new_dims[8];
    for (int k = 0; k < 8; k++) {
        new_dims[k] = dims[perm[k]];
    }
    int64_t total = 1;
    for (int k = 0; k < 8; k++) {
        total *= dims[k];
    }
    // 计算原始strides
    int64_t strides[8];
    strides[7] = 1;
    for (int d = 6; d >= 0; d--) {
        strides[d] = strides[d + 1] * dims[d + 1];
    }

    // 使用嵌套循环遍历输出索引，避免内循环计算
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


bool save_as_bin(const char* filename, void* data, size_t size) {
    FILE* fp = fopen(filename, "wb");
    if (fp == NULL) {
        printf("Error: Could not open file %s for writing.\n", filename);
        return false;
    }

    size_t written = fwrite(data, 1, size, fp);
    fclose(fp);

    if (written != size) {
        printf("Error: Expected to write %zu bytes, but wrote %zu.\n", size, written);
        return false;
    }

    printf("Successfully saved %zu bytes to %s\n", size, filename);
    return true;
}

int inference_qwen3_5_vl_vision(rknn_qwen3_5_vl_vision_context* vision_ctx, image_buffer_t* img, float16* img_embeds, bool use_lora)
{
    if ((!vision_ctx) || (!img))
    {
        printf("vision_ctx or img is NULL");
        return -1;
    }
    int ret;
    image_buffer_t dst_img;
    float16* expand_data;
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

    ret = set_vision_lora_enabled(vision_ctx, use_lora);
    if (ret < 0) {
        goto out;
    }

    if(vision_ctx->pruned_version_flag == 1) {
        expand_data = (float16*)malloc(dst_img.size * 2 * sizeof(float16)); //expand
        normalize_and_to_nchw(dst_img.virt_addr, expand_data, vision_ctx->model_width, vision_ctx->model_height);
        memcpy((char *)expand_data + dst_img.size * 2 , expand_data, dst_img.size * 2); //expand
        transpose((float16 *)expand_data, (float16 *)(vision_ctx->inputs[0].mem->virt_addr), vision_ctx->model_width, vision_ctx->model_height);
        free(expand_data);
    } else {
        // The Qwen3.5 full graph is exported as NCHW float16. Convert the
        // RGB888 image to the same [-1, 1] representation used by the
        // original PyTorch export before copying it to RKNN memory.
        size_t pixel_count = (size_t)vision_ctx->model_width *
                             (size_t)vision_ctx->model_height * 3;
        float16* nchw_data = (float16*)malloc(pixel_count * sizeof(float16));
        if (!nchw_data) {
            printf("malloc normalized vision input failed\n");
            ret = -1;
            goto out;
        }
        normalize_and_to_nchw(dst_img.virt_addr, nchw_data,
                              vision_ctx->model_width, vision_ctx->model_height);
        memcpy(vision_ctx->inputs[0].mem->virt_addr, nchw_data,
               pixel_count * sizeof(float16));
        free(nchw_data);
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

    // 方案 A：直接保存输出内存（推荐，效率高）
    save_as_bin("img_embeds.bin", vision_ctx->outputs[0].mem->virt_addr, vision_ctx->outputs[0].mem->size);

out:
    if (dst_img.virt_addr != NULL)
    {
        free(dst_img.virt_addr);
    }

    return ret;
}
