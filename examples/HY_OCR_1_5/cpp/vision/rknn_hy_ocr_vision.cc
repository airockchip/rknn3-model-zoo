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
#include <math.h>
#include <ctime>

#include "rknn_hy_ocr_vision.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"


// ---------------------------------------------------------------------------
// LANCZOS 重采样实现 (与 PIL LANCZOS/resample=1 一致)
// 用于替换 RGA/双线性 resize, 消除预处理精度差异
// ---------------------------------------------------------------------------

struct LanczosFilter {
    int first;
    int count;
    size_t weights_offset;
};

struct LanczosFilterTable {
    std::vector<LanczosFilter> filters;
    std::vector<float> weights;
};

// 系数只依赖输入/输出尺寸；预计算并归一化后，像素内循环只做 float32 乘加。
static inline float lanczos_kernel_f32(float x, int a)
{
    static const float kPi = 3.14159265358979323846f;
    if (x == 0.0f) return 1.0f;
    if (x <= -a || x >= a) return 0.0f;
    const float px = kPi * x;
    const float pa = px / static_cast<float>(a);
    return (sinf(px) / px) * (sinf(pa) / pa);
}

static LanczosFilterTable build_lanczos_filter_table(int src_size, int dst_size, int a)
{
    LanczosFilterTable table;
    table.filters.resize(dst_size);
    const float scale = static_cast<float>(src_size) / static_cast<float>(dst_size);
    const float support = scale > 1.0f ? static_cast<float>(a) * scale : static_cast<float>(a);

    for (int dst_pos = 0; dst_pos < dst_size; ++dst_pos) {
        const float center = (static_cast<float>(dst_pos) + 0.5f) * scale - 0.5f;
        const int first = static_cast<int>(floorf(center - support + 1e-6f));
        const int last = static_cast<int>(ceilf(center + support - 1e-6f));
        LanczosFilter& filter = table.filters[dst_pos];
        filter.first = first;
        filter.count = last - first + 1;
        filter.weights_offset = table.weights.size();

        float weight_sum = 0.0f;
        for (int src_pos = first; src_pos <= last; ++src_pos) {
            const float distance = static_cast<float>(src_pos) - center;
            const float arg = scale > 1.0f ? distance / scale : distance;
            const float weight = lanczos_kernel_f32(arg, a);
            table.weights.push_back(weight);
            weight_sum += weight;
        }
        if (weight_sum != 0.0f) {
            const float inv_sum = 1.0f / weight_sum;
            for (size_t i = filter.weights_offset; i < table.weights.size(); ++i) {
                table.weights[i] *= inv_sum;
            }
        }
    }
    return table;
}

static inline int clamp_index(int value, int size)
{
    if (value < 0) return 0;
    if (value >= size) return size - 1;
    return value;
}

// RGB888 Lanczos resize：RGB 三通道联合处理，避免拆分、合并和平面临时缓冲。
static int lanczos_resize_rgb(const uint8_t* src, int src_w, int src_h,
                              uint8_t* dst, int dst_w, int dst_h)
{
    const int a = 3; // Lanczos-3
    const int channels = 3;
    const LanczosFilterTable horizontal = build_lanczos_filter_table(src_w, dst_w, a);
    const LanczosFilterTable vertical = build_lanczos_filter_table(src_h, dst_h, a);

    // 步骤 1：水平 resize，中间结果保持交错 RGB float32 布局。
    size_t tmp_h_size = (size_t)src_h * dst_w * channels;
    std::vector<float> tmp_h(tmp_h_size);
    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            const LanczosFilter& filter = horizontal.filters[x];
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
            for (int tap = 0; tap < filter.count; ++tap) {
                const int src_x = clamp_index(filter.first + tap, src_w);
                const float weight = horizontal.weights[filter.weights_offset + tap];
                const uint8_t* pixel = src + ((size_t)y * src_w + src_x) * channels;
                sum_r += static_cast<float>(pixel[0]) * weight;
                sum_g += static_cast<float>(pixel[1]) * weight;
                sum_b += static_cast<float>(pixel[2]) * weight;
            }
            float* output = tmp_h.data() + ((size_t)y * dst_w + x) * channels;
            output[0] = sum_r;
            output[1] = sum_g;
            output[2] = sum_b;
        }
    }

    // 步骤 2：按输出行进行垂直 resize，目标写入连续，三通道共享索引和权重。
    for (int y = 0; y < dst_h; ++y) {
        const LanczosFilter& filter = vertical.filters[y];
        for (int x = 0; x < dst_w; ++x) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
            for (int tap = 0; tap < filter.count; ++tap) {
                const int src_y = clamp_index(filter.first + tap, src_h);
                const float weight = vertical.weights[filter.weights_offset + tap];
                const float* pixel = tmp_h.data() + ((size_t)src_y * dst_w + x) * channels;
                sum_r += pixel[0] * weight;
                sum_g += pixel[1] * weight;
                sum_b += pixel[2] * weight;
            }
            float values[3] = {sum_r, sum_g, sum_b};
            uint8_t* output = dst + ((size_t)y * dst_w + x) * channels;
            for (int c = 0; c < channels; ++c) {
                float value = values[c];
                if (value < 0.0f) value = 0.0f;
                if (value > 255.0f) value = 255.0f;
                output[c] = static_cast<uint8_t>(value + 0.5f);
            }
        }
    }

    return 0;
}


// 精度测试辅助: 通过环境变量 HY_OCR_DUMP_DIR 导出 vision 模型的实际输入/输出张量
// 用于板端 RKNN 输出与 PyTorch/ONNX/Simulator 的同输入精度对比 (仅调试用途)
static size_t tensor_type_size(rknn3_tensor_type dtype)
{
    switch (dtype) {
    case RKNN3_TENSOR_FLOAT32:
    case RKNN3_TENSOR_INT32:
    case RKNN3_TENSOR_UINT32:
        return 4;
    case RKNN3_TENSOR_INT64:
    case RKNN3_TENSOR_UINT64:
        return 8;
    case RKNN3_TENSOR_BOOL:
    case RKNN3_TENSOR_INT8:
    case RKNN3_TENSOR_UINT8:
    case RKNN3_TENSOR_FLOAT8E4M3FN:
    case RKNN3_TENSOR_FLOAT8E8M0:
    case RKNN3_TENSOR_FLOAT4E2M1:
        return 1;
    case RKNN3_TENSOR_FLOAT16:
    case RKNN3_TENSOR_BFLOAT16:
    case RKNN3_TENSOR_INT16:
    case RKNN3_TENSOR_UINT16:
        return 2;
    case RKNN3_TENSOR_INT4:
        return 1; // packed int4 按 1 字节写出
    default:
        return 0;
    }
}

static void dump_tensor_raw(const char* dir, const char* name, rknn3_tensor* tensor)
{
    if (!dir || !name || !tensor || !tensor->mem || !tensor->mem->virt_addr || !tensor->attr) {
        return;
    }
    uint32_t n_elems = tensor->attr->n_elems;
    size_t type_size = tensor_type_size(tensor->attr->dtype);
    if (n_elems == 0 || type_size == 0) {
        return;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE* fp = fopen(path, "wb");
    if (!fp) {
        printf("dump_tensor_raw: cannot open %s\n", path);
        return;
    }
    size_t wrote = fwrite(tensor->mem->virt_addr, type_size, n_elems, fp);
    fclose(fp);
    printf("[dump] %s: %u elems, %zu bytes\n", path, n_elems, wrote * type_size);
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
  
    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%lu, layout=%s, dtype=%s, core_id=%d, "
           "qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), (unsigned long)attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}


int init_hy_ocr_vision(rknn_hy_ocr_vision_context* vision_ctx, const char* model_path, const char* weight_path, uint32_t core_mask, uint32_t model_width, uint32_t model_height)
{
    int ret;
    rknn3_context ctx = 0;
    rknn3_config config;
    memset(&config, 0, sizeof(config));
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
        rknn3_destroy(ctx);
        return ret;
    }

    //Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }

    // Get Model Input Output Number
    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    std::vector<rknn3_tensor_attr> input_attrs(io_num.n_input);
    std::vector<rknn3_tensor_attr> output_attrs(io_num.n_output);

    // Set to context early so release_hy_ocr_vision can clean up on failure
    vision_ctx->rknn_ctx = ctx;
    vision_ctx->pruned_version_flag = 0;
    vision_ctx->inputs = (rknn3_tensor*)calloc(io_num.n_input, sizeof(rknn3_tensor));
    vision_ctx->outputs = (rknn3_tensor*)calloc(io_num.n_output, sizeof(rknn3_tensor));
    if (!vision_ctx->inputs || !vision_ctx->outputs) {
        printf("calloc inputs/outputs fail!\n");
        ret = -1;
        goto out;
    }
    vision_ctx->io_num = io_num;

    // Get Model Input Info
    printf("input tensors:\n");
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            goto out;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            goto out;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
        vision_ctx->inputs[i].mem  = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!vision_ctx->inputs[i].mem || !vision_ctx->inputs[i].mem->virt_addr) {
            printf("rknn3_create_mem input[%d] fail!\n", i);
            ret = -1;
            goto out;
        }
        vision_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (!vision_ctx->inputs[i].attr) {
            printf("malloc input attr fail!\n");
            ret = -1;
            goto out;
        }
        memcpy(vision_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
        vision_ctx->outputs[i].mem  = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!vision_ctx->outputs[i].mem || !vision_ctx->outputs[i].mem->virt_addr) {
            printf("rknn3_create_mem output[%d] fail!\n", i);
            ret = -1;
            goto out;
        }
        vision_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (!vision_ctx->outputs[i].attr) {
            printf("malloc output attr fail!\n");
            ret = -1;
            goto out;
        }
        memcpy(vision_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    if (input_attrs[0].layout == RKNN3_TENSOR_NCHW)
    {
        printf("model is NCHW input layout (full pixel input)\n");
        vision_ctx->pruned_version_flag = 0;
        vision_ctx->model_channel = input_attrs[0].shape[1];
        vision_ctx->model_height = input_attrs[0].shape[2];
        vision_ctx->model_width = input_attrs[0].shape[3];
        printf("input image height=%d, input image width=%d, input image channel=%d\n",
                vision_ctx->model_height, vision_ctx->model_width, vision_ctx->model_channel);
    }
    else if (input_attrs[0].layout == RKNN3_TENSOR_NHWC)
    {
        printf("model is NHWC input layout (full pixel input)\n");
        vision_ctx->pruned_version_flag = 0;
        vision_ctx->model_channel = input_attrs[0].shape[3];
        vision_ctx->model_height = input_attrs[0].shape[1];
        vision_ctx->model_width = input_attrs[0].shape[2];
        printf("input image height=%d, input image width=%d, input image channel=%d\n",
                vision_ctx->model_height, vision_ctx->model_width, vision_ctx->model_channel);
    }
    else 
    {
        // prune 模式：输入是 flatten_patches [grid_h*grid_w, PATCH_DIM]
        const uint32_t effective_width = model_width != 0 ? model_width : MODEL_WIDTH;
        const uint32_t effective_height = model_height != 0 ? model_height : MODEL_HEIGHT;
        int feature_size = effective_width * effective_height /
                           HY_OCR_PATCH_SIZE / HY_OCR_PATCH_SIZE;
        if(input_attrs[0].shape[0] != feature_size){
            printf("Please note: model_width=%d and model_height=%d do not match the model's expected input dimensions. Please either update the default parameters or explicitly specify the correct model_width and model_height arguments in the command line.\n",
                   model_width == 0 ? MODEL_WIDTH : model_width, model_height == 0 ? MODEL_HEIGHT : model_height);
            ret = -1;
            goto out;
        }
        vision_ctx->pruned_version_flag = 1;
        printf("model is UNDEFINED (pruned version)!\n");
        vision_ctx->model_channel = 3;
        if(model_width != 0 && model_height != 0) {
            vision_ctx->model_width = (int)model_width;
            vision_ctx->model_height = (int)model_height;
        } else {
            vision_ctx->model_width = (int)MODEL_WIDTH;
            vision_ctx->model_height = (int)MODEL_HEIGHT;
        }
        printf("model_width=%d model_height=%d \n", vision_ctx->model_width, vision_ctx->model_height);
    }

    // 裁剪版 2 输入模式: 加载预计算的 pos_embed
    vision_ctx->pos_embed_data = NULL;
    vision_ctx->pos_embed_size = 0;
    if (vision_ctx->pruned_version_flag == 1 && io_num.n_input == 2) {
        // 从模型同目录推导 pos_embed 文件路径
        std::string model_path_str(model_path);
        std::string dir = model_path_str.substr(0, model_path_str.find_last_of("/"));
        char pos_embed_path[512];
        snprintf(pos_embed_path, sizeof(pos_embed_path), "%s/pos_embed_%dx%d.bin",
                 dir.c_str(), vision_ctx->model_width, vision_ctx->model_height);
        printf("Loading pos_embed from: %s\n", pos_embed_path);

        FILE* fp = fopen(pos_embed_path, "rb");
        if (!fp) {
            printf("Error: cannot open pos_embed file: %s\n", pos_embed_path);
            ret = -1;
            goto out;
        }
        fseek(fp, 0, SEEK_END);
        long file_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        const size_t expected_elements = input_attrs[1].n_elems;
        const size_t expected_bytes = expected_elements * sizeof(float16);
        if (file_size <= 0 || (size_t)file_size != expected_bytes) {
            printf("Error: pos_embed size mismatch: file=%ld bytes, expected=%lu bytes (%lu elements)\n",
                   file_size, (unsigned long)expected_bytes, (unsigned long)expected_elements);
            fclose(fp);
            ret = -1;
            goto out;
        }
        vision_ctx->pos_embed_size = file_size / sizeof(float16);
        vision_ctx->pos_embed_data = (float16*)malloc(file_size);
        if (!vision_ctx->pos_embed_data) {
            printf("Error: malloc pos_embed_data fail!\n");
            fclose(fp);
            ret = -1;
            goto out;
        }
        if (fread(vision_ctx->pos_embed_data, 1, file_size, fp) != (size_t)file_size) {
            printf("Error: read pos_embed file incomplete!\n");
            fclose(fp);
            ret = -1;
            goto out;
        }
        fclose(fp);
        printf("pos_embed loaded: %d elements (%ld bytes)\n", vision_ctx->pos_embed_size, file_size);
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
        goto out;
    }

    return ret;

out:
    release_hy_ocr_vision(vision_ctx);
    return ret;
}

int release_hy_ocr_vision(rknn_hy_ocr_vision_context* vision_ctx)
{
    if (!vision_ctx) {
        return 0;
    }

    if (vision_ctx->inputs) {
        for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
            if (vision_ctx->inputs[i].mem) {
                rknn3_destroy_mem(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem);
                vision_ctx->inputs[i].mem = NULL;
            }
            if (vision_ctx->inputs[i].attr != NULL) {
                free(vision_ctx->inputs[i].attr);
                vision_ctx->inputs[i].attr = NULL;
            }
        }
        free(vision_ctx->inputs);
        vision_ctx->inputs = NULL;
    }

    if (vision_ctx->outputs) {
        for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
            if (vision_ctx->outputs[i].mem) {
                rknn3_destroy_mem(vision_ctx->rknn_ctx, vision_ctx->outputs[i].mem);
                vision_ctx->outputs[i].mem = NULL;
            }
            if (vision_ctx->outputs[i].attr != NULL) {
                free(vision_ctx->outputs[i].attr);
                vision_ctx->outputs[i].attr = NULL;
            }
        }
        free(vision_ctx->outputs);
        vision_ctx->outputs = NULL;
    }

    if (vision_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(vision_ctx->rknn_ctx);
        vision_ctx->rknn_ctx = 0;
    }

    if (vision_ctx->pos_embed_data) {
        free(vision_ctx->pos_embed_data);
        vision_ctx->pos_embed_data = NULL;
    }
    return 0;
}

// HunyuanOCR-1.5 归一化 + NHWC→NCHW 转换
// mean = [0.48145466*255, 0.4578275*255, 0.40821073*255]
// std  = [0.26862954*255, 0.26130258*255, 0.27577711*255]
static void normalize_and_to_nchw(const unsigned char* src, float16* dst, int width, int height)
{
    float mean[3] = {122.771f, 116.746f, 104.094f};
    float std[3]  = {68.501f, 66.632f, 70.323f};
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

// HunyuanOCR-1.5 patch 展平: 对齐 processor PIL backend 的 reshape+transpose 逻辑
// 输入: NCHW 格式 (temporal_patch_size=1 不需要 repeat)
// dims: [temporal_patch, channel, grid_h//merge, merge, patch, grid_w//merge, merge, patch]
//       = [1, 3, model_height/factor, merge, patch, model_width/factor, merge, patch]
// perm:  {2, 3, 5, 6, 1, 0, 4, 7}  → [gh//merge, merge_h, gw//merge, merge_w, ch, temporal, patch, patch]
// 与 PIL backend (image_processing_pil_hunyuan_vl.py) 的 transpose(0,1,4,5,7,8,3,2,6,9) 一致
static void transpose(float16* in, float16* out, int model_width, int model_height)
{
    int64_t dims[8] = {1, 3, (int64_t)model_height / HY_OCR_FACTOR,
                       HY_OCR_MERGE_SIZE, HY_OCR_PATCH_SIZE,
                       (int64_t)model_width / HY_OCR_FACTOR,
                       HY_OCR_MERGE_SIZE, HY_OCR_PATCH_SIZE};
    int perm[8] = {2, 3, 5, 6, 1, 0, 4, 7};
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
        i[perm[0]] = o0; i[perm[1]] = o1; i[perm[2]] = o2; i[perm[3]] = o3;
        i[perm[4]] = o4; i[perm[5]] = o5; i[perm[6]] = o6; i[perm[7]] = o7;
        int64_t flat_in = i[0] * strides[0] + i[1] * strides[1] + i[2] * strides[2] +
                          i[3] * strides[3] + i[4] * strides[4] + i[5] * strides[5] +
                          i[6] * strides[6] + i[7] * strides[7];
        out[out_idx++] = in[flat_in];
    }}}}}}}}
}


int inference_hy_ocr_vision(rknn_hy_ocr_vision_context* vision_ctx, image_buffer_t* img, float16* img_embeds, int n_image_tokens)
{
    if ((!vision_ctx) || (!img))
    {
        printf("vision_ctx or img is NULL");
        return -1;
    }
    int ret = -1;
    const int patch_size = HY_OCR_PATCH_SIZE;
    const int merge_size = HY_OCR_MERGE_SIZE;
    // 性能统计: 预处理(读图/resize/归一化/展平)与 NPU 推理分别计时
    struct timespec ts_;
    int64_t t_start_us = 0, t_npu_us = 0, t_end_us = 0;
    clock_gettime(CLOCK_MONOTONIC, &ts_);
    t_start_us = ts_.tv_sec * 1000000LL + ts_.tv_nsec / 1000;
    int hidden_dim = vision_ctx->embeds_shape[vision_ctx->embeds_ndims - 1];
    image_buffer_t dst_img;
    memset(&dst_img, 0, sizeof(image_buffer_t));

    // Pre Process: resize to model input size (LANCZOS, 与 PIL resample=1 一致)
    dst_img.width     = vision_ctx->model_width;
    dst_img.height    = vision_ctx->model_height;
    dst_img.format    = IMAGE_FORMAT_RGB888;
    dst_img.size      = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char*)malloc(dst_img.size);

    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    // resize 策略(环境变量 HY_OCR_RESIZE_MODE):
    //   lanczos(默认): CPU LANCZOS, 与 PIL resample=1 一致(精度基准, 大图慢)
    //   hybrid: 仅大图(边>2*model)启用两级: RGA 硬件下采样到中间尺寸(边 clamp(src/2, model, 2*model))
    //           + CPU LANCZOS 精修到目标尺寸(精度/速度折中); 小图行为与 lanczos 相同
    //   rga: RGA 硬件直接 resize 到目标尺寸(最快, 双线性插值, 与 PIL LANCZOS 有精度差)
    const char* resize_mode_env = getenv("HY_OCR_RESIZE_MODE");
    const char* resize_mode = (resize_mode_env && resize_mode_env[0]) ? resize_mode_env : "lanczos";
    int64_t t_rga_us = 0, t_lanczos_us = 0;
    const int mid_min = vision_ctx->model_width;
    const int mid_max = vision_ctx->model_width * 2;

    if (strcmp(resize_mode, "rga") == 0) {
        // RGA 硬件直接 resize
        clock_gettime(CLOCK_MONOTONIC, &ts_);
        t_rga_us -= ts_.tv_sec * 1000000LL + ts_.tv_nsec / 1000;
        ret = convert_image(img, &dst_img, NULL, NULL, 0);
        clock_gettime(CLOCK_MONOTONIC, &ts_);
        t_rga_us += ts_.tv_sec * 1000000LL + ts_.tv_nsec / 1000;
        if (ret < 0) {
            printf("convert_image(rga) fail! ret=%d\n", ret);
            goto out;
        }
    } else if (strcmp(resize_mode, "hybrid") == 0 && img->format == IMAGE_FORMAT_RGB888
               && (img->width > mid_max || img->height > mid_max)) {
        // 大图两级: RGA 下采样到中间尺寸, 再 CPU LANCZOS 精修
        image_buffer_t mid_img;
        memset(&mid_img, 0, sizeof(image_buffer_t));
        int mid_w = img->width / 2;
        int mid_h = img->height / 2;
        if (mid_w < mid_min) mid_w = mid_min; else if (mid_w > mid_max) mid_w = mid_max;
        if (mid_h < mid_min) mid_h = mid_min; else if (mid_h > mid_max) mid_h = mid_max;
        mid_img.width     = mid_w;
        mid_img.height    = mid_h;
        mid_img.format    = IMAGE_FORMAT_RGB888;
        mid_img.size      = get_image_size(&mid_img);
        mid_img.virt_addr = (unsigned char*)malloc(mid_img.size);
        if (mid_img.virt_addr == NULL) {
            printf("malloc mid buffer(size=%d) fail!\n", mid_img.size);
            ret = -1;
            goto out;
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_);
        t_rga_us -= ts_.tv_sec * 1000000LL + ts_.tv_nsec / 1000;
        ret = convert_image(img, &mid_img, NULL, NULL, 0);
        clock_gettime(CLOCK_MONOTONIC, &ts_);
        t_rga_us += ts_.tv_sec * 1000000LL + ts_.tv_nsec / 1000;
        if (ret < 0) {
            printf("convert_image(hybrid rga) fail! ret=%d\n", ret);
            free(mid_img.virt_addr);
            goto out;
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_);
        t_lanczos_us -= ts_.tv_sec * 1000000LL + ts_.tv_nsec / 1000;
        ret = lanczos_resize_rgb(mid_img.virt_addr, mid_img.width, mid_img.height,
                                 dst_img.virt_addr, dst_img.width, dst_img.height);
        clock_gettime(CLOCK_MONOTONIC, &ts_);
        t_lanczos_us += ts_.tv_sec * 1000000LL + ts_.tv_nsec / 1000;
        free(mid_img.virt_addr);
        if (ret < 0) {
            printf("lanczos_resize_rgb(hybrid) fail! ret=%d\n", ret);
            goto out;
        }
    } else {
        // 默认: CPU LANCZOS (与 PIL resample=1 一致); 非 RGB888 格式回退 convert_image
        clock_gettime(CLOCK_MONOTONIC, &ts_);
        t_lanczos_us -= ts_.tv_sec * 1000000LL + ts_.tv_nsec / 1000;
        if (img->format == IMAGE_FORMAT_RGB888) {
            ret = lanczos_resize_rgb(img->virt_addr, img->width, img->height,
                                     dst_img.virt_addr, dst_img.width, dst_img.height);
        } else {
            ret = convert_image(img, &dst_img, NULL, NULL, 0);
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_);
        t_lanczos_us += ts_.tv_sec * 1000000LL + ts_.tv_nsec / 1000;
        if (ret < 0) {
            printf("lanczos_resize_rgb fail! ret=%d\n", ret);
            goto out;
        }
    }
    printf("[resize] mode=%s src=%dx%d rga=%.2f ms lanczos=%.2f ms\n",
           resize_mode, img->width, img->height, t_rga_us / 1000.0, t_lanczos_us / 1000.0);

    if (vision_ctx->pruned_version_flag == 1) {
        // prune 模式: 归一化 + NHWC→NCHW → transpose → flatten_patches
        if (!vision_ctx->inputs[0].mem || !vision_ctx->inputs[0].mem->virt_addr) {
            printf("inputs[0].mem is NULL!\n");
            ret = -1;
            goto out;
        }
        std::vector<float16> nchw_data((size_t)dst_img.size);
        normalize_and_to_nchw(dst_img.virt_addr, nchw_data.data(), vision_ctx->model_width, vision_ctx->model_height);

        // temporal_patch_size=1 不需要 repeat
        transpose(nchw_data.data(), (float16*)(vision_ctx->inputs[0].mem->virt_addr),
                  vision_ctx->model_width, vision_ctx->model_height);

        // 2 输入模式: 拷贝预计算的 pos_embed 到 inputs[1]
        if (vision_ctx->io_num.n_input == 2 && vision_ctx->pos_embed_data != NULL) {
            if (!vision_ctx->inputs[1].mem || !vision_ctx->inputs[1].mem->virt_addr) {
                printf("inputs[1].mem is NULL!\n");
                ret = -1;
                goto out;
            }
            memcpy(vision_ctx->inputs[1].mem->virt_addr,
                   vision_ctx->pos_embed_data,
                   vision_ctx->pos_embed_size * sizeof(float16));
        }
    } else {
        // 完整版: 直接拷贝像素数据
        if (!vision_ctx->inputs[0].mem || !vision_ctx->inputs[0].mem->virt_addr) {
            printf("inputs[0].mem is NULL!\n");
            ret = -1;
            goto out;
        }
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
    clock_gettime(CLOCK_MONOTONIC, &ts_);
    t_npu_us = ts_.tv_sec * 1000000LL + ts_.tv_nsec / 1000;
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

    // 精度测试: HY_OCR_DUMP_DIR 设置时导出实际输入/输出张量
    // 性能统计: 区分 CPU 预处理与 NPU 推理耗时
    clock_gettime(CLOCK_MONOTONIC, &ts_);
    t_end_us = ts_.tv_sec * 1000000LL + ts_.tv_nsec / 1000;
    printf("[perf] vision: preprocess=%.2f ms, npu_run=%.2f ms\n",
           (t_npu_us - t_start_us) / 1000.0, (t_end_us - t_npu_us) / 1000.0);
    {
        const char* dump_dir = getenv("HY_OCR_DUMP_DIR");
        if (dump_dir) {
            dump_tensor_raw(dump_dir, "flatten_patches_f16.bin", &vision_ctx->inputs[0]);
            if (vision_ctx->io_num.n_input > 1) {
                dump_tensor_raw(dump_dir, "pos_embed_f16.bin", &vision_ctx->inputs[1]);
            }
            dump_tensor_raw(dump_dir, "vision_out_f16.bin", &vision_ctx->outputs[0]);
        }
    }

    // Get Output: copy all vision merger outputs. The first/last rows are the
    // vision image_begin/image_end embeddings; they map to the first/last
    // repeated image_token placeholders, not to the outer template wrappers.
    // vision merger 输出布局 (见 modeling_hunyuan_vl.py HunYuanVisionMerger.forward):
    //   [image_begin, row0(pw个patch), \n, row1(pw个patch), \n, ..., row_{ph-1}(pw个patch), \n, image_end]
    //   = 1 + ph*(pw+1) + 1 个 token
    // Python uses masked_scatter to replace every repeated image_token placeholder.
    {
        int ph = vision_ctx->model_height / patch_size / merge_size;
        int pw = vision_ctx->model_width / patch_size / merge_size;
        int n_content_tokens = ph * (pw + 1);
        int n_out_tokens = vision_ctx->embeds_shape[vision_ctx->embeds_ndims - 2];
        if (!vision_ctx->outputs[0].mem || !vision_ctx->outputs[0].mem->virt_addr) {
            printf("outputs[0].mem is NULL!\n");
            ret = -1;
            goto out;
        }
        float16* src = (float16*)vision_ctx->outputs[0].mem->virt_addr;

        // n_image_tokens includes begin + content + end = n_content_tokens + 2
        if (n_content_tokens + 2 != n_image_tokens) {
            printf("vision content mismatch! n_content_tokens=%d, n_image_tokens=%d (expected %d)\n",
                   n_content_tokens, n_image_tokens, n_content_tokens + 2);
            ret = -1;
            goto out;
        }
        if (n_out_tokens != n_content_tokens + 2) {
            printf("unexpected vision output token num! expect=%d, got=%d\n", n_content_tokens + 2, n_out_tokens);
            ret = -1;
            goto out;
        }

        // 拷贝全部 token (begin + content + end)，与 Python masked_scatter 一致
        memcpy(img_embeds, src, (uint64_t)n_out_tokens * hidden_dim * sizeof(float16));
    }

out:
    if (dst_img.virt_addr != NULL)
    {
        free(dst_img.virt_addr);
    }

    return ret;
}
