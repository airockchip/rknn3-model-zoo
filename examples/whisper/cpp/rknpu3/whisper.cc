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

#include "whisper.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "audio_utils.h"
#include "easy_timer.h"
#include "file_utils.h"
#include "float16.h"
#include "process.h"
static void dump_tensor_attr(rknn3_tensor_attr *attrs) {
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

    printf(
        "Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, "
        "qnt_type=%s\n",
        attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
        rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}

int init_whisper_model(const char *model_path, rknn_app_context_t *app_ctx, const char *weight_path, uint32_t core_mask) {
    int ret;
    int core_number = 0;
    rknn3_context ctx = 0;
    rknn3_config config;
    memset(&config, 0, sizeof(config));

    // RKNN Init
    ret = rknn3_init(&ctx, NULL);
    if (ret != RKNN3_SUCCESS) {
        printf("init_whisper_model stage=init model=%s weight=%s core_mask=0x%x ret=%d\n",
               model_path, weight_path ? weight_path : "NULL", core_mask, ret);
        return -1;
    }

    app_ctx->rknn_ctx = ctx;

    // Load RKNN Model
    printf("Loading model: %s, weight: %s\n", model_path, weight_path ? weight_path : "NULL");
    // 检查权重文件是否存在
    if (weight_path != NULL) {
        FILE *fp = fopen(weight_path, "rb");
        if (fp == NULL) {
            printf("Warning: Weight file %s cannot be opened!\n", weight_path);
        } else {
            fseek(fp, 0, SEEK_END);
            long weight_size = ftell(fp);
            fclose(fp);
            printf("Weight file size: %ld bytes\n", weight_size);
        }
    }
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret != RKNN3_SUCCESS) {
        printf("init_whisper_model stage=load_model model=%s weight=%s ret=%d\n",
               model_path, weight_path ? weight_path : "NULL", ret);
        return -1;
    }
    printf("Model loaded successfully\n");

    // Init RKNN Model
    ret = rknn3_query(ctx, RKNN3_QUERY_CORE_NUMBER, &core_number, sizeof(core_number));
    if (ret != RKNN3_SUCCESS || core_number <= 0 || core_number > 32) {
        printf("init_whisper_model stage=query_core_number model=%s weight=%s ret=%d core_number=%d\n",
               model_path, weight_path ? weight_path : "NULL", ret, core_number);
        return -1;
    }
    if (core_mask == 0) {
        config.run_core_mask = (uint32_t)((1ULL << core_number) - 1ULL);
        printf("Initializing model with auto core_mask: 0x%x (core_number=%d)\n", config.run_core_mask, core_number);
    } else {
        config.run_core_mask = core_mask;
        printf("Initializing model with core_mask: 0x%x\n", core_mask);
    }
    ret = rknn3_model_init(ctx, &config);
    if (ret != RKNN3_SUCCESS) {
        printf("init_whisper_model stage=model_init model=%s weight=%s ret=%d core_mask=0x%x\n",
               model_path, weight_path ? weight_path : "NULL", ret, config.run_core_mask);
        return ret;
    }
    printf("Model initialized successfully\n");

    std::vector<rknn3_allocation_info> allocation_info(core_number);
    memset(allocation_info.data(), 0, allocation_info.size() * sizeof(rknn3_allocation_info));
    ret = rknn3_query(ctx, RKNN3_QUERY_ALLOCATION_INFO, allocation_info.data(),
                      allocation_info.size() * sizeof(rknn3_allocation_info));
    if (ret != RKNN3_SUCCESS) {
        printf("init_whisper_model stage=query_allocation_info model=%s ret=%d core_number=%d\n",
               model_path, ret, core_number);
        return ret;
    }

    app_ctx->n_kvcache_mems = 0;
#ifdef RKNN3_MAX_CACHE_PAGE_TYPES
    app_ctx->n_kvcache_bindings = 0;
    bool paged_cache = false;
    for (int i = 0; i < core_number; ++i) {
        rknn3_kvcache_page_info info = {};
        info.core_id = allocation_info[i].core_id;
        ret = rknn3_query(ctx, RKNN3_QUERY_KVCACHE_PAGE_INFO, &info, sizeof(info));
        if (ret != RKNN3_SUCCESS) {
            return ret;
        }
        if (info.n_cache_types != 0 && !paged_cache) {
            rknn3_llm_config llm = {};
            ret = rknn3_query(ctx, RKNN3_QUERY_LLM_CONFIG, &llm, sizeof(llm));
            if (ret != RKNN3_SUCCESS) return ret;
            app_ctx->kvcache_residual_depth = llm.kvcache_residual_depth;
            printf("KV residual depth: %u\n", app_ctx->kvcache_residual_depth);
            if (app_ctx->kvcache_residual_depth != 32) return -1;
        }
        paged_cache = paged_cache || info.n_cache_types != 0;
        if (info.n_cache_types > RKNN3_MAX_CACHE_PAGE_TYPES) {
            return -1;
        }
        for (uint32_t request = 0; request < info.max_request_num; ++request) {
            for (uint32_t type = 0; type < info.n_cache_types; ++type) {
                const rknn3_cache_page_attr &attr = info.caches[type];
                if (app_ctx->n_kvcache_bindings >= 32 || attr.max_page_num == 0 ||
                    attr.max_page_num > 32 - app_ctx->n_kvcache_mems) {
                    return -1;
                }
                std::vector<const rknn3_tensor_mem *> pages;
                for (uint32_t page = 0; page < attr.max_page_num; ++page) {
                    rknn3_tensor_mem *mem = rknn3_create_mem(
                        ctx, attr.page_size, info.core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
                    if (!mem || !mem->virt_addr) {
                        if (mem) rknn3_destroy_mem(ctx, mem);
                        return -1;
                    }
                    app_ctx->kvcache_mems[app_ctx->n_kvcache_mems++] = mem;
                    memset(mem->virt_addr, 0, mem->size);
                    ret = rknn3_mem_sync(ctx, mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
                    if (ret != RKNN3_SUCCESS) return ret;
                    pages.push_back(mem);
                }
                rknn3_kvcache_page_binding binding = {};
                binding.core_id = info.core_id;
                binding.cache_type = attr.cache_type;
                binding.request_index = request;
                binding.n_pages = attr.max_page_num;
                binding.pages = pages.data();
                ret = rknn3_set_kvcache_pages(ctx, &binding, 1);
                if (ret != RKNN3_SUCCESS) return ret;
                printf("Bound paged KV cache: core=%d type=%d pages=%u page_size=%lu\n",
                       info.core_id, attr.cache_type, attr.max_page_num,
                       (unsigned long)attr.page_size);
                binding.n_pages = 0;
                binding.pages = NULL;
                app_ctx->kvcache_bindings[app_ctx->n_kvcache_bindings++] = binding;
            }
        }
    }
    if (!paged_cache) {
#endif
    for (int i = 0; i < core_number; ++i) {
        uint64_t kvcache_size = allocation_info[i].kvcache_mem.size;
        if (kvcache_size == 0) {
            continue;
        }
        rknn3_tensor_mem *mem = rknn3_create_mem(ctx, kvcache_size, allocation_info[i].core_id,
                                                 RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!mem) {
            printf("init_whisper_model stage=create_kvcache_mem model=%s core_id=%d size=%lu ret=-1\n",
                   model_path, allocation_info[i].core_id, (unsigned long)kvcache_size);
            return -1;
        }
        memset(mem->virt_addr, 0, mem->size);
        ret = rknn3_mem_sync(ctx, mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            rknn3_destroy_mem(ctx, mem);
            printf("init_whisper_model stage=sync_kvcache_mem model=%s core_id=%d size=%lu ret=%d\n",
                   model_path, allocation_info[i].core_id, (unsigned long)kvcache_size, ret);
            return ret;
        }
        uint32_t index = app_ctx->n_kvcache_mems++;
        app_ctx->kvcache_mems[index] = mem;
        app_ctx->kvcache_core_ids[index] = allocation_info[i].core_id;
        printf("Created KV cache memory: core_id=%d size=%lu\n",
               allocation_info[i].core_id, (unsigned long)kvcache_size);
    }
    if (app_ctx->n_kvcache_mems > 0) {
        ret = rknn3_set_kvcache_mem(ctx, app_ctx->kvcache_mems, app_ctx->kvcache_core_ids,
                                    (int)app_ctx->n_kvcache_mems);
        if (ret != RKNN3_SUCCESS) {
            printf("init_whisper_model stage=set_kvcache_mem model=%s ret=%d n_core=%u\n",
                   model_path, ret, app_ctx->n_kvcache_mems);
            return ret;
        }
        printf("Bound KV cache memory for %u cores\n", app_ctx->n_kvcache_mems);
    }

#ifdef RKNN3_MAX_CACHE_PAGE_TYPES
    }
#endif

    // Get Model Input Output Number
    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN3_SUCCESS) {
        printf("init_whisper_model stage=query_io_num model=%s weight=%s ret=%d\n",
               model_path, weight_path ? weight_path : "NULL", ret);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn3_tensor_attr input_attrs[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret != RKNN3_SUCCESS) {
            printf("init_whisper_model stage=query_input_attr model=%s weight=%s index=%d ret=%d\n",
                   model_path, weight_path ? weight_path : "NULL", i, ret);
            return ret;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn3_tensor_attr output_attrs[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret != RKNN3_SUCCESS) {
            printf("init_whisper_model stage=query_output_attr model=%s weight=%s index=%d ret=%d\n",
                   model_path, weight_path ? weight_path : "NULL", i, ret);
            return ret;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->inputs = (rknn3_tensor *)malloc(io_num.n_input * sizeof(rknn3_tensor));
    app_ctx->outputs = (rknn3_tensor *)malloc(io_num.n_output * sizeof(rknn3_tensor));
    if (!app_ctx->inputs || !app_ctx->outputs) {
        printf("init_whisper_model stage=alloc_tensor_array model=%s weight=%s ret=-1 n_input=%d n_output=%d\n",
               model_path, weight_path ? weight_path : "NULL", io_num.n_input, io_num.n_output);
        release_whisper_model(app_ctx);
        return -1;
    }
    memset(app_ctx->inputs, 0, io_num.n_input * sizeof(rknn3_tensor));
    memset(app_ctx->outputs, 0, io_num.n_output * sizeof(rknn3_tensor));
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    for (int i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->inputs[i].mem = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        app_ctx->inputs[i].attr = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr));
        if (!app_ctx->inputs[i].mem || !app_ctx->inputs[i].attr) {
            printf("init_whisper_model stage=alloc_input_tensor failed model=%s weight=%s index=%d ret=-1 name=%s aligned_size=%ld core_id=%d mem=%p attr=%p\n",
                   model_path,
                   weight_path ? weight_path : "NULL",
                   i,
                   input_attrs[i].name,
                   input_attrs[i].aligned_size,
                   input_attrs[i].core_id,
                   app_ctx->inputs[i].mem,
                   app_ctx->inputs[i].attr);
            release_whisper_model(app_ctx);
            return -1;
        }
        memcpy(app_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->outputs[i].mem = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        app_ctx->outputs[i].attr = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr));
        if (!app_ctx->outputs[i].mem || !app_ctx->outputs[i].attr) {
            printf("init_whisper_model stage=alloc_output_tensor failed model=%s weight=%s index=%d ret=-1 name=%s aligned_size=%ld core_id=%d mem=%p attr=%p\n",
                   model_path,
                   weight_path ? weight_path : "NULL",
                   i,
                   output_attrs[i].name,
                   output_attrs[i].aligned_size,
                   output_attrs[i].core_id,
                   app_ctx->outputs[i].mem,
                   app_ctx->outputs[i].attr);
            release_whisper_model(app_ctx);
            return -1;
        }
        memcpy(app_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    if (output_attrs[0].qnt_type == RKNN3_TENSOR_PER_LAYER_ASYMMETRIC && output_attrs[0].dtype == RKNN3_TENSOR_INT8) {
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

    return 0;
}

int release_whisper_model(rknn_app_context_t *app_ctx) {
    if (!app_ctx) {
        printf("release_whisper_model invalid app_ctx=null\n");
        return -1;
    }
    if (app_ctx && app_ctx->inputs) {
        for (int i = 0; i < app_ctx->io_num.n_input; i++) {
            if (app_ctx->inputs[i].mem) {
                rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->inputs[i].mem);
            }
            if (app_ctx->inputs[i].attr != NULL) {
                free(app_ctx->inputs[i].attr);
                app_ctx->inputs[i].attr = NULL;
            }
        }
    }
    if (app_ctx && app_ctx->outputs) {
        for (int i = 0; i < app_ctx->io_num.n_output; i++) {
            if (app_ctx->outputs[i].mem) {
                rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->outputs[i].mem);
            }
            if (app_ctx->outputs[i].attr != NULL) {
                free(app_ctx->outputs[i].attr);
                app_ctx->outputs[i].attr = NULL;
            }
        }
    }
#ifdef RKNN3_MAX_CACHE_PAGE_TYPES
    if (app_ctx->n_kvcache_bindings > 0) {
        int ret = rknn3_set_kvcache_pages(app_ctx->rknn_ctx, app_ctx->kvcache_bindings,
                                         app_ctx->n_kvcache_bindings);
        if (ret != RKNN3_SUCCESS) return ret;
        app_ctx->n_kvcache_bindings = 0;
    }
#endif
    for (uint32_t i = 0; i < app_ctx->n_kvcache_mems; ++i) {
        if (app_ctx->kvcache_mems[i]) {
            rknn3_destroy_mem(app_ctx->rknn_ctx, app_ctx->kvcache_mems[i]);
            app_ctx->kvcache_mems[i] = NULL;
        }
    }
    app_ctx->n_kvcache_mems = 0;
    if (app_ctx->rknn_ctx != 0) {
        rknn3_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    if (app_ctx->inputs) {
        free(app_ctx->inputs);
        app_ctx->inputs = NULL;
    }
    if (app_ctx->outputs) {
        free(app_ctx->outputs);
        app_ctx->outputs = NULL;
    }
    return 0;
}

uint32_t whisper_shape_count(const rknn3_tensor_attr *attr) {
    uint32_t elems = 1;
    for (uint32_t i = 0; i < attr->n_dims; ++i) {
        elems *= attr->shape[i];
    }
    return elems;
}

uint64_t whisper_tensor_active_aligned_size(const rknn3_tensor *tensor) {
    if (!tensor || !tensor->mem || !tensor->attr) {
        return 0;
    }
    return tensor->attr->aligned_size < tensor->mem->size ? tensor->attr->aligned_size : tensor->mem->size;
}

int whisper_find_input_index_by_name(rknn_app_context_t *ctx, const char *name) {
    if (!ctx || !ctx->inputs || !name) {
        printf("whisper_find_input_index_by_name invalid ctx=%p inputs=%p name=%s\n",
               ctx, ctx ? ctx->inputs : NULL, name ? name : "<null>");
        return -1;
    }
    for (uint32_t i = 0; i < ctx->io_num.n_input; ++i) {
        if (ctx->inputs[i].attr && strcmp(ctx->inputs[i].attr->name, name) == 0) {
            return (int)i;
        }
    }
    printf("whisper_find_input_index_by_name missing input name=%s\n", name ? name : "<null>");
    return -1;
}

int whisper_require_input_index(rknn_app_context_t *ctx, const char *stage, const char *name, int fallback_index) {
    if (!ctx || !name) {
        printf("[%s] required input lookup invalid ctx=%p name=%s fallback_index=%d\n",
               stage ? stage : "unknown", ctx, name ? name : "<null>", fallback_index);
        return -1;
    }
    int index = whisper_find_input_index_by_name(ctx, name);
    if (index >= 0) {
        return index;
    }
    if (fallback_index >= 0 && fallback_index < (int)ctx->io_num.n_input) {
        printf("[%s] input missing name=%s fallback_index=%d using input[%d]=%s\n",
               stage ? stage : "unknown",
               name ? name : "<null>",
               fallback_index,
               fallback_index,
               ctx->inputs[fallback_index].attr ? ctx->inputs[fallback_index].attr->name : "<null>");
        return fallback_index;
    }
    printf("[%s] required input missing name=%s fallback_index=%d n_input=%u\n",
           stage ? stage : "unknown",
           name ? name : "<null>",
           fallback_index,
           ctx ? ctx->io_num.n_input : 0);
    return -1;
}

int whisper_find_output_index_by_name(rknn_app_context_t *ctx, const char *name) {
    if (!ctx || !ctx->outputs || !name) {
        printf("whisper_find_output_index_by_name invalid ctx=%p outputs=%p name=%s\n",
               ctx, ctx ? ctx->outputs : NULL, name ? name : "<null>");
        return -1;
    }
    for (uint32_t i = 0; i < ctx->io_num.n_output; ++i) {
        if (ctx->outputs[i].attr && strcmp(ctx->outputs[i].attr->name, name) == 0) {
            return (int)i;
        }
    }
    printf("whisper_find_output_index_by_name missing output name=%s\n", name ? name : "<null>");
    return -1;
}

uint32_t whisper_tensor_dtype_size(rknn3_tensor_type type) {
    switch (type) {
        case RKNN3_TENSOR_FLOAT32:
        case RKNN3_TENSOR_INT32:
        case RKNN3_TENSOR_UINT32:
            return 4;
        case RKNN3_TENSOR_FLOAT16:
        case RKNN3_TENSOR_INT16:
        case RKNN3_TENSOR_UINT16:
            return 2;
        case RKNN3_TENSOR_INT8:
        case RKNN3_TENSOR_UINT8:
            return 1;
        case RKNN3_TENSOR_INT64:
            return 8;
        default:
            return 0;
    }
}

bool whisper_same_shape(const rknn3_tensor_attr *a, const rknn3_tensor_attr *b) {
    if (!a || !b || a->n_dims != b->n_dims) {
        return false;
    }
    for (uint32_t i = 0; i < a->n_dims; ++i) {
        if (a->shape[i] != b->shape[i]) {
            return false;
        }
    }
    return true;
}

void whisper_print_tensor_shape_brief(const char *prefix, const rknn3_tensor_attr *attr) {
    if (!attr) {
        printf("%s <null>\n", prefix);
        return;
    }

    printf("%s name=%s n_dims=%u shape=[", prefix, attr->name, attr->n_dims);
    for (uint32_t i = 0; i < attr->n_dims; ++i) {
        printf("%u%s", attr->shape[i], i + 1 == attr->n_dims ? "" : ",");
    }
    printf("] dtype=%s\n", rknn3_get_type_string(attr->dtype));
}

int whisper_convert_fp32_to_tensor(const float *src, void *dst, uint32_t n_elems, rknn3_tensor_type type) {
    switch (type) {
        case RKNN3_TENSOR_FLOAT16:
            for (uint32_t i = 0; i < n_elems; i++) {
                ((float16 *)dst)[i] = fp32_to_fp16(src[i]);
            }
            break;
        case RKNN3_TENSOR_FLOAT32:
            memcpy(dst, src, n_elems * sizeof(float));
            break;
        case RKNN3_TENSOR_INT32:
            for (uint32_t i = 0; i < n_elems; i++) {
                ((int32_t *)dst)[i] = (int32_t)src[i];
            }
            break;
        case RKNN3_TENSOR_INT64:
            for (uint32_t i = 0; i < n_elems; i++) {
                ((int64_t *)dst)[i] = (int64_t)src[i];
            }
            break;
        default:
            printf("whisper_convert_fp32_to_tensor unsupported dtype=%s n_elems=%u\n",
                   rknn3_get_type_string(type), n_elems);
            return -1;
    }
    return 0;
}

int whisper_convert_tensor_to_fp32(const void *src, float *dst, int n_elems, rknn3_tensor_type type) {
    switch (type) {
        case RKNN3_TENSOR_FLOAT16:
            for (int i = 0; i < n_elems; i++) {
                dst[i] = fp16_to_fp32(((const float16 *)src)[i]);
            }
            break;
        case RKNN3_TENSOR_FLOAT32:
            memcpy(dst, src, n_elems * sizeof(float));
            break;
        case RKNN3_TENSOR_INT64:
            for (int i = 0; i < n_elems; i++) {
                dst[i] = (float)((const int64_t *)src)[i];
            }
            break;
        case RKNN3_TENSOR_INT32:
            for (int i = 0; i < n_elems; i++) {
                dst[i] = (float)((const int32_t *)src)[i];
            }
            break;
        case RKNN3_TENSOR_INT16:
            for (int i = 0; i < n_elems; i++) {
                dst[i] = (float)((const int16_t *)src)[i];
            }
            break;
        case RKNN3_TENSOR_INT8:
            for (int i = 0; i < n_elems; i++) {
                dst[i] = (float)((const int8_t *)src)[i];
            }
            break;
        case RKNN3_TENSOR_UINT8:
            for (int i = 0; i < n_elems; i++) {
                dst[i] = (float)((const uint8_t *)src)[i];
            }
            break;
        default:
            printf("whisper_convert_tensor_to_fp32 unsupported dtype=%s n_elems=%d\n",
                   rknn3_get_type_string(type), n_elems);
            return -1;
    }
    return 0;
}

int whisper_fill_zero_tensor(rknn3_context ctx, rknn3_tensor *tensor, bool sync_to_device) {
    if (!tensor || !tensor->mem || !tensor->attr) {
        printf("whisper_fill_zero_tensor invalid tensor=%p mem=%p attr=%p sync_to_device=%d\n",
               tensor, tensor ? tensor->mem : NULL, tensor ? tensor->attr : NULL, (int)sync_to_device);
        return -1;
    }
    memset(tensor->mem->virt_addr, 0, whisper_tensor_active_aligned_size(tensor));
    if (!sync_to_device) {
        return 0;
    }
    if (!ctx) {
        printf("whisper_fill_zero_tensor invalid ctx=null tensor=%s\n", tensor->attr->name);
        return -1;
    }
    int ret = rknn3_mem_sync(ctx, tensor->mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN3_SUCCESS) {
        printf("whisper_fill_zero_tensor mem_sync fail tensor=%s ret=%d size=%lu\n",
               tensor->attr->name, ret, (unsigned long)tensor->mem->size);
    }
    return ret;
}

int whisper_fill_int32_scalar_tensor(rknn3_context ctx, rknn3_tensor *tensor, int32_t value) {
    if (!tensor || !tensor->mem || !tensor->attr || tensor->attr->dtype != RKNN3_TENSOR_INT32) {
        printf("whisper_fill_int32_scalar_tensor invalid tensor=%p mem=%p attr=%p dtype=%s value=%d\n",
               tensor,
               tensor ? tensor->mem : NULL,
               tensor ? tensor->attr : NULL,
               (tensor && tensor->attr) ? rknn3_get_type_string(tensor->attr->dtype) : "<null>",
               value);
        return -1;
    }
    int32_t *data = reinterpret_cast<int32_t *>(tensor->mem->virt_addr);
    uint32_t elems = whisper_shape_count(tensor->attr);
    memset(tensor->mem->virt_addr, 0, tensor->mem->size);
    if (elems == 1) {
        data[0] = value;
    } else {
        for (uint32_t i = 0; i < elems; ++i) {
            data[i] = value;
        }
    }
    if (!ctx) {
        printf("whisper_fill_int32_scalar_tensor invalid ctx=null tensor=%s value=%d\n", tensor->attr->name, value);
        return -1;
    }
    int ret = rknn3_mem_sync(ctx, tensor->mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN3_SUCCESS) {
        printf("whisper_fill_int32_scalar_tensor mem_sync fail tensor=%s ret=%d size=%lu\n",
               tensor->attr->name, ret, (unsigned long)tensor->mem->size);
    }
    return ret;
}

int whisper_fill_causal_attention_mask_tensor(rknn3_context ctx, rknn3_tensor *tensor) {
    if (!tensor || !tensor->mem || !tensor->attr) {
        printf("whisper_fill_causal_attention_mask_tensor invalid tensor=%p mem=%p attr=%p\n",
               tensor, tensor ? tensor->mem : NULL, tensor ? tensor->attr : NULL);
        return -1;
    }
    if (tensor->attr->dtype != RKNN3_TENSOR_FLOAT16 && tensor->attr->dtype != RKNN3_TENSOR_FLOAT32) {
        printf("whisper_fill_causal_attention_mask_tensor invalid dtype tensor=%s dtype=%s\n",
               tensor->attr->name, rknn3_get_type_string(tensor->attr->dtype));
        return -1;
    }

    const float mask_value = -65504.0f;
    memset(tensor->mem->virt_addr, 0, tensor->mem->size);

    if (tensor->attr->n_dims == 5) {
        uint32_t n = tensor->attr->shape[0];
        uint32_t c1 = tensor->attr->shape[1];
        uint32_t h = tensor->attr->shape[2];
        uint32_t w = tensor->attr->shape[3];
        uint32_t c2 = tensor->attr->shape[4];
        for (uint32_t ni = 0; ni < n; ++ni) {
            for (uint32_t ci = 0; ci < c1; ++ci) {
                for (uint32_t qi = 0; qi < h; ++qi) {
                    for (uint32_t ki = 0; ki < w; ++ki) {
                        if (ki <= qi) {
                            continue;
                        }
                        uint32_t base = ni * tensor->attr->stride[0] + ci * tensor->attr->stride[1] +
                                        qi * tensor->attr->stride[2] + ki * tensor->attr->stride[3];
                        for (uint32_t lane = 0; lane < c2; ++lane) {
                            uint32_t index = base + lane * tensor->attr->stride[4];
                            if (tensor->attr->dtype == RKNN3_TENSOR_FLOAT16) {
                                reinterpret_cast<float16 *>(tensor->mem->virt_addr)[index] = fp32_to_fp16(mask_value);
                            } else {
                                reinterpret_cast<float *>(tensor->mem->virt_addr)[index] = mask_value;
                            }
                        }
                    }
                }
            }
        }
    } else if (tensor->attr->n_dims == 4) {
        uint32_t n = tensor->attr->shape[0];
        uint32_t c = tensor->attr->shape[1];
        uint32_t h = tensor->attr->shape[2];
        uint32_t w = tensor->attr->shape[3];
        for (uint32_t ni = 0; ni < n; ++ni) {
            for (uint32_t ci = 0; ci < c; ++ci) {
                for (uint32_t qi = 0; qi < h; ++qi) {
                    for (uint32_t ki = qi + 1; ki < w; ++ki) {
                        uint32_t index = ni * tensor->attr->stride[0] + ci * tensor->attr->stride[1] +
                                         qi * tensor->attr->stride[2] + ki * tensor->attr->stride[3];
                        if (tensor->attr->dtype == RKNN3_TENSOR_FLOAT16) {
                            reinterpret_cast<float16 *>(tensor->mem->virt_addr)[index] = fp32_to_fp16(mask_value);
                        } else {
                            reinterpret_cast<float *>(tensor->mem->virt_addr)[index] = mask_value;
                        }
                    }
                }
            }
        }
    } else {
        printf("whisper_fill_causal_attention_mask_tensor unsupported dims tensor=%s n_dims=%u\n",
               tensor->attr->name, tensor->attr->n_dims);
        return -1;
    }

    if (!ctx) {
        printf("whisper_fill_causal_attention_mask_tensor invalid ctx=null tensor=%s\n", tensor->attr->name);
        return -1;
    }
    int ret = rknn3_mem_sync(ctx, tensor->mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN3_SUCCESS) {
        printf("whisper_fill_causal_attention_mask_tensor mem_sync fail tensor=%s ret=%d size=%lu\n",
               tensor->attr->name, ret, (unsigned long)tensor->mem->size);
        return ret;
    }
    return 0;
}

bool whisper_is_cross_kv_hdns_4d(const rknn3_tensor_attr *attr) {
    if (!attr || attr->n_dims != 4) {
        return false;
    }

    // HDNS: [H, D, N, S]. S follows the exported audio chunk length.
    return attr->shape[0] > 1 &&
           attr->shape[1] > 1 &&
           attr->shape[2] == 1 &&
           attr->shape[3] > 1;
}

bool whisper_is_cross_kv_nhsd_4d(const rknn3_tensor_attr *attr) {
    if (!attr || attr->n_dims != 4) {
        return false;
    }

    // NHSD: [N, H, S, D]
    // Whisper base 常见: [1, 8, 1500, 64]
    return attr->shape[0] == 1 &&
           attr->shape[1] > 1 &&
           attr->shape[2] > 1 &&
           attr->shape[3] > 1;
}

bool whisper_is_cross_kv_hdns_c1hwc2_5d(const rknn3_tensor_attr *attr) {
    if (!attr || attr->n_dims != 5) {
        return false;
    }

    // HDNS packed: [H, D/16, N, S, 16]
    // Whisper base 常见: [8, 4, 1, 1500, 16]
    return attr->shape[0] > 1 &&
           attr->shape[1] > 0 &&
           attr->shape[2] == 1 &&
           attr->shape[3] > 1 &&
           attr->shape[4] == 16;
}

bool whisper_is_cross_kv_nc1hwc2_5d(const rknn3_tensor_attr *attr) {
    if (!attr || attr->n_dims != 5) {
        return false;
    }

    // NC1HWC2: [N, C1, S, D, C2]
    // Whisper base 常见: [1, 1, 1500, 64, 16]
    return attr->shape[0] == 1 &&
           attr->shape[1] == 1 &&
           attr->shape[2] > 1 &&
           attr->shape[3] > 1 &&
           attr->shape[4] >= 8;
}

int whisper_repack_hdns_to_hdns_c1hwc2(rknn3_context dst_ctx, rknn3_tensor *dst, const rknn3_tensor *src, bool sync_to_device) {
    if (!dst || !src || !dst->attr || !src->attr || !dst->mem || !src->mem) {
        printf("whisper_repack_hdns_to_hdns_c1hwc2 invalid dst=%p src=%p dst_mem=%p src_mem=%p\n",
               dst, src, dst ? dst->mem : NULL, src ? src->mem : NULL);
        return -1;
    }

    if (src->attr->dtype != RKNN3_TENSOR_FLOAT16 || dst->attr->dtype != RKNN3_TENSOR_FLOAT16) {
        printf("whisper_repack_hdns_to_hdns_c1hwc2 only supports FP16 dst=%s src=%s\n",
               dst->attr->name, src->attr->name);
        return -1;
    }

    const uint32_t H = src->attr->shape[0];
    const uint32_t D = src->attr->shape[1];
    const uint32_t N = src->attr->shape[2];
    const uint32_t S = src->attr->shape[3];

    const uint32_t dst_H  = dst->attr->shape[0];
    const uint32_t dst_D1 = dst->attr->shape[1];
    const uint32_t dst_N  = dst->attr->shape[2];
    const uint32_t dst_S  = dst->attr->shape[3];
    const uint32_t dst_D2 = dst->attr->shape[4];

    if (H != dst_H || N != dst_N || S != dst_S || dst_D2 != 16 || dst_D1 * dst_D2 < D) {
        printf("HDNS repack shape mismatch\n");
        whisper_print_tensor_shape_brief("  src", src->attr);
        whisper_print_tensor_shape_brief("  dst", dst->attr);
        return -1;
    }

    memset(dst->mem->virt_addr, 0, dst->mem->size);

    const float16 *src_data = reinterpret_cast<const float16 *>(src->mem->virt_addr);
    float16 *dst_data = reinterpret_cast<float16 *>(dst->mem->virt_addr);
    const uint64_t *src_stride = src->attr->stride;
    const uint64_t *dst_stride = dst->attr->stride;

    for (uint32_t h = 0; h < H; ++h) {
        for (uint32_t d = 0; d < D; ++d) {
            const uint32_t d1 = d / 16;
            const uint32_t d2 = d % 16;

            for (uint32_t n = 0; n < N; ++n) {
                for (uint32_t s = 0; s < S; ++s) {
                    size_t src_off = (size_t)h * src_stride[0] + (size_t)d * src_stride[1]
                                   + (size_t)n * src_stride[2] + (size_t)s * src_stride[3];
                    size_t dst_off = (size_t)h * dst_stride[0] + (size_t)d1 * dst_stride[1]
                                   + (size_t)n * dst_stride[2] + (size_t)s * dst_stride[3]
                                   + (size_t)d2 * dst_stride[4];
                    dst_data[dst_off] = src_data[src_off];
                }
            }
        }
    }

    if (sync_to_device) {
        if (!dst_ctx) {
            printf("whisper_repack_hdns_to_hdns_c1hwc2 invalid dst_ctx=null dst=%s src=%s\n",
                   dst->attr->name, src->attr->name);
            return -1;
        }
        int ret = rknn3_mem_sync(dst_ctx, dst->mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("whisper_repack_hdns_to_hdns_c1hwc2 mem_sync fail ret=%d dst=%s src=%s\n",
                   ret,
                   dst->attr ? dst->attr->name : "<null>",
                   src->attr ? src->attr->name : "<null>");
            return ret;
        }
    }

    return 0;
}

int whisper_copy_tensor_data(rknn3_context dst_ctx, rknn3_tensor *dst, const rknn3_tensor *src, bool sync_to_device) {
    if (!dst || !src || !dst->attr || !src->attr || !dst->mem || !src->mem) {
        printf("whisper_copy_tensor_data invalid dst=%p src=%p dst_mem=%p src_mem=%p\n",
               dst, src, dst ? dst->mem : NULL, src ? src->mem : NULL);
        return -1;
    }

    bool src_hdns = whisper_is_cross_kv_hdns_4d(src->attr);
    bool dst_hdns = whisper_is_cross_kv_hdns_4d(dst->attr);
    bool src_nhsd = whisper_is_cross_kv_nhsd_4d(src->attr);
    bool dst_nhsd = whisper_is_cross_kv_nhsd_4d(dst->attr);
    bool dst_hdns_c1hwc2 = whisper_is_cross_kv_hdns_c1hwc2_5d(dst->attr);

    // 新 kv_hdns 模式 1：decode0 output [H,D,N,S] -> decode1 input [H,D,N,S]
    if (src_hdns && dst_hdns) {
        if (!whisper_same_shape(src->attr, dst->attr) ||
            (src->attr->dtype != RKNN3_TENSOR_FLOAT16 && src->attr->dtype != RKNN3_TENSOR_FLOAT32) ||
            (dst->attr->dtype != RKNN3_TENSOR_FLOAT16 && dst->attr->dtype != RKNN3_TENSOR_FLOAT32)) {
            printf("whisper_copy_tensor_data HDNS shape/dtype mismatch dst=%s src=%s\n",
                   dst->attr->name, src->attr->name);
            whisper_print_tensor_shape_brief("  src", src->attr);
            whisper_print_tensor_shape_brief("  dst", dst->attr);
            return -1;
        }

        memset(dst->mem->virt_addr, 0, dst->mem->size);
        for (uint32_t h = 0; h < src->attr->shape[0]; ++h) {
            for (uint32_t d = 0; d < src->attr->shape[1]; ++d) {
                for (uint32_t t = 0; t < src->attr->shape[3]; ++t) {
                    size_t src_offset = h * src->attr->stride[0] + d * src->attr->stride[1]
                                      + t * src->attr->stride[3];
                    size_t dst_offset = h * dst->attr->stride[0] + d * dst->attr->stride[1]
                                      + t * dst->attr->stride[3];
                    float value = src->attr->dtype == RKNN3_TENSOR_FLOAT16
                        ? fp16_to_fp32(reinterpret_cast<const float16 *>(src->mem->virt_addr)[src_offset])
                        : reinterpret_cast<const float *>(src->mem->virt_addr)[src_offset];
                    if (dst->attr->dtype == RKNN3_TENSOR_FLOAT16) {
                        reinterpret_cast<float16 *>(dst->mem->virt_addr)[dst_offset] = fp32_to_fp16(value);
                    } else {
                        reinterpret_cast<float *>(dst->mem->virt_addr)[dst_offset] = value;
                    }
                }
            }
        }

        if (sync_to_device) {
            if (!dst_ctx) {
                printf("whisper_copy_tensor_data invalid dst_ctx=null dst=%s src=%s\n",
                       dst->attr->name, src->attr->name);
                return -1;
            }
            int ret = rknn3_mem_sync(dst_ctx, dst->mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
            if (ret != RKNN3_SUCCESS) {
                printf("whisper_copy_tensor_data mem_sync fail dst=%s src=%s ret=%d\n",
                       dst->attr->name, src->attr->name, ret);
                return ret;
            }
        }

        return 0;
    }

    // 新 kv_hdns 模式 2：decode0 output [H,D,N,S] -> decode1 input [H,D/16,N,S,16]
    if (src_hdns && dst_hdns_c1hwc2) {
        return whisper_repack_hdns_to_hdns_c1hwc2(dst_ctx, dst, src, sync_to_device);
    }

    // 如果一边是 HDNS，另一边不是可支持格式，才报错
    if (src_hdns || dst_hdns || dst_hdns_c1hwc2) {
        printf("whisper_copy_tensor_data cross_kv layout mismatch dst=%s src=%s\n",
               dst->attr->name, src->attr->name);
        whisper_print_tensor_shape_brief("  src", src->attr);
        whisper_print_tensor_shape_brief("  dst", dst->attr);
        return -1;
    }

    // 旧模式：decode0 output 是 [1,H,S,D]，decode1 input 是 [1,1,S,D,16]。
    // 这个允许继续走下面旧的 NCHW/NHSD -> NC1HWC2 重排。
    // 但如果 src 是旧 NHSD，dst 却不是 NC1HWC2，也不要直接 memcpy。
    if (src_nhsd && dst->attr->n_dims == 4 && !whisper_same_shape(src->attr, dst->attr)) {
        printf("whisper_copy_tensor_data old NHSD cannot copy to non-matching 4D dst=%s src=%s\n",
               dst->attr->name, src->attr->name);
        whisper_print_tensor_shape_brief("  src", src->attr);
        whisper_print_tensor_shape_brief("  dst", dst->attr);
        return -1;
    }

    uint32_t dst_elems = whisper_shape_count(dst->attr);
    uint32_t src_elems = whisper_shape_count(src->attr);
    if (dst_elems != src_elems) {
        // 元素数不一致时，尝试 NCHW(4D) → NC1HWC2(5D) 布局重排
        bool can_repack_nchw_to_nc1hwc2 =
            whisper_is_cross_kv_nhsd_4d(src->attr) &&
            whisper_is_cross_kv_nc1hwc2_5d(dst->attr) &&
            src->attr->dtype == dst->attr->dtype &&
            src->attr->dtype == RKNN3_TENSOR_FLOAT16 &&
            src->attr->shape[0] == dst->attr->shape[0] &&
            dst->attr->shape[1] == 1 &&
            src->attr->shape[2] == dst->attr->shape[2] &&
            src->attr->shape[3] == dst->attr->shape[3] &&
            src->attr->shape[1] <= dst->attr->shape[4];

        if (!can_repack_nchw_to_nc1hwc2) {
            printf("whisper_copy_tensor_data tensor elem mismatch dst=%s elems=%u src=%s elems=%u\n",
                   dst->attr->name, dst_elems, src->attr->name, src_elems);
            return -1;
        }

        const float16 *src_data = reinterpret_cast<const float16 *>(src->mem->virt_addr);
        float16 *dst_data = reinterpret_cast<float16 *>(dst->mem->virt_addr);
        memset(dst->mem->virt_addr, 0, dst->mem->size);
        uint32_t batch = src->attr->shape[0];
        uint32_t heads = src->attr->shape[1];
        uint32_t seq = src->attr->shape[2];
        uint32_t head_dim = src->attr->shape[3];
        for (uint32_t n = 0; n < batch; ++n) {
            for (uint32_t h = 0; h < heads; ++h) {
                for (uint32_t s = 0; s < seq; ++s) {
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        uint32_t src_index = n * src->attr->stride[0] + h * src->attr->stride[1] + s * src->attr->stride[2] + d * src->attr->stride[3];
                        uint32_t dst_index = n * dst->attr->stride[0] + s * dst->attr->stride[2] + d * dst->attr->stride[3] + h * dst->attr->stride[4];
                        dst_data[dst_index] = src_data[src_index];
                    }
                }
            }
        }

        if (sync_to_device) {
            if (!dst_ctx) {
                printf("whisper_copy_tensor_data invalid dst_ctx=null dst=%s src=%s\n",
                       dst->attr->name, src->attr->name);
                return -1;
            }
            int ret = rknn3_mem_sync(dst_ctx, dst->mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
            if (ret != RKNN3_SUCCESS) {
                printf("whisper_copy_tensor_data mem_sync fail dst=%s src=%s ret=%d\n",
                       dst->attr->name, src->attr->name, ret);
                return ret;
            }
        }
        return 0;
    }

    // cross_kv 的元素数相同不代表 layout 相同。
    // 例如 [1,8,1500,64] 和 [8,64,1,1500] 元素数相同，但不能直接 memcpy。
    if ((whisper_is_cross_kv_nhsd_4d(src->attr) && whisper_is_cross_kv_hdns_4d(dst->attr)) ||
        (whisper_is_cross_kv_hdns_4d(src->attr) && whisper_is_cross_kv_nhsd_4d(dst->attr)) ||
        (whisper_is_cross_kv_nhsd_4d(src->attr) && whisper_is_cross_kv_nhsd_4d(dst->attr) && !whisper_same_shape(src->attr, dst->attr)) ||
        (whisper_is_cross_kv_hdns_4d(src->attr) && whisper_is_cross_kv_hdns_4d(dst->attr) && !whisper_same_shape(src->attr, dst->attr))) {
        printf("whisper_copy_tensor_data same-element layout mismatch dst=%s src=%s\n",
               dst->attr->name, src->attr->name);
        whisper_print_tensor_shape_brief("  src", src->attr);
        whisper_print_tensor_shape_brief("  dst", dst->attr);
        return -1;
    }

    // 元素数相同：直接逐元素复制（同类型直接 memcpy，不同类型经 FP32 中转）
    if (dst->attr->dtype == src->attr->dtype) {
        uint32_t elem_size = whisper_tensor_dtype_size(dst->attr->dtype);
        if (elem_size == 0) {
            printf("whisper_copy_tensor_data unsupported direct copy dtype dst=%s dtype=%s src=%s\n",
                   dst->attr->name,
                   rknn3_get_type_string(dst->attr->dtype),
                   src->attr->name);
            return -1;
        }
        memcpy(dst->mem->virt_addr, src->mem->virt_addr, (size_t)dst_elems * elem_size);
    } else {
        std::vector<float> tmp(src_elems);
        int ret = whisper_convert_tensor_to_fp32(src->mem->virt_addr, tmp.data(), src_elems, src->attr->dtype);
        if (ret != 0) {
            return ret;
        }
        ret = whisper_convert_fp32_to_tensor(tmp.data(), dst->mem->virt_addr, dst_elems, dst->attr->dtype);
        if (ret != 0) {
            return ret;
        }
    }

    if (sync_to_device) {
        if (!dst_ctx) {
            printf("whisper_copy_tensor_data invalid dst_ctx=null dst=%s src=%s\n",
                   dst->attr->name, src->attr->name);
            return -1;
        }
        int ret = rknn3_mem_sync(dst_ctx, dst->mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("whisper_copy_tensor_data mem_sync fail dst=%s src=%s ret=%d\n",
                   dst->attr->name, src->attr->name, ret);
            return ret;
        }
    }
    return 0;
}

int whisper_find_input_index_in_attrs(const std::vector<rknn3_tensor_attr> &attrs, const char *name) {
    if (!name) {
        printf("whisper_find_input_index_in_attrs invalid name=null attr_count=%zu\n", attrs.size());
        return -1;
    }
    for (size_t i = 0; i < attrs.size(); ++i) {
        if (strcmp(attrs[i].name, name) == 0) {
            return (int)i;
        }
    }
    printf("whisper_find_input_index_in_attrs missing input name=%s attr_count=%zu\n",
           name ? name : "<null>", attrs.size());
    return -1;
}

int whisper_find_decoder_embed_index_in_attrs(const std::vector<rknn3_tensor_attr> &attrs) {
    int index = whisper_find_input_index_in_attrs(attrs, "input_embeds");
    if (index < 0) {
        index = whisper_find_input_index_in_attrs(attrs, "inputs_embeds");
    }
    if (index < 0) {
        printf("whisper_find_decoder_embed_index_in_attrs missing input_embeds/inputs_embeds attr_count=%zu\n", attrs.size());
    }
    return index;
}

uint32_t whisper_decoder_embed_seq_len(const std::vector<rknn3_tensor_attr> &attrs) {
    int embed_index = whisper_find_decoder_embed_index_in_attrs(attrs);
    if (embed_index < 0 || attrs[embed_index].n_dims < 2) {
        return 0;
    }
    return attrs[embed_index].shape[1];
}

int whisper_ensure_tensor_mem_size(rknn3_context ctx, rknn3_tensor *tensor, uint64_t size, uint32_t core_id) {
    if (!tensor || !tensor->attr) {
        printf("whisper_ensure_tensor_mem_size invalid tensor=%p attr=%p size=%lu core_id=%u\n",
               tensor, tensor ? tensor->attr : NULL, size, core_id);
        return -1;
    }

    // 当前内存足够且 core_id 匹配 → 直接复用
    if (tensor->mem && tensor->mem->size >= size && tensor->mem->core_id == (int32_t)core_id) {
        return 0;
    }

    // 需要重新分配：销毁旧内存
    if (!ctx) {
        printf("whisper_ensure_tensor_mem_size invalid ctx=null tensor=%s size=%lu core_id=%u\n",
               tensor->attr->name, size, core_id);
        return -1;
    }
    if (tensor->mem) {
        rknn3_destroy_mem(ctx, tensor->mem);
        tensor->mem = NULL;
    }
    // 创建新内存
    tensor->mem = rknn3_create_mem(ctx, size, core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
    if (!tensor->mem) {
        printf("rknn3_create_mem failed: tensor=%s size=%lu core_id=%u\n", tensor->attr->name, size, core_id);
        return -1;
    }
    memset(tensor->mem->virt_addr, 0, tensor->mem->size);
    return 0;
}

int whisper_recreate_tensor_mem_exact(rknn3_context ctx, rknn3_tensor *tensor, uint64_t size, uint32_t core_id) {
    if (!tensor || !tensor->attr) {
        printf("whisper_recreate_tensor_mem_exact invalid tensor=%p attr=%p size=%lu core_id=%u\n",
               tensor, tensor ? tensor->attr : NULL, size, core_id);
        return -1;
    }

    if (tensor->mem && tensor->mem->size == size && tensor->mem->core_id == (int32_t)core_id) {
        return 0;
    }

    if (!ctx) {
        printf("whisper_recreate_tensor_mem_exact invalid ctx=null tensor=%s size=%lu core_id=%u\n",
               tensor->attr->name, size, core_id);
        return -1;
    }
    if (tensor->mem) {
        rknn3_destroy_mem(ctx, tensor->mem);
        tensor->mem = NULL;
    }
    tensor->mem = rknn3_create_mem(ctx, size, core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
    if (!tensor->mem) {
        printf("rknn3_create_mem exact failed: tensor=%s size=%lu core_id=%u\n",
               tensor->attr->name, size, core_id);
        return -1;
    }

    memset(tensor->mem->virt_addr, 0, tensor->mem->size);
    return 0;
}

uint32_t whisper_tensor_offset_2d(const rknn3_tensor_attr *attr, uint32_t row, uint32_t col, uint32_t col_size) {
    if (!attr) {
        return row * col_size + col;
    }
    if (attr->n_dims == 3) {
        // 3D: [1, seq, hidden] 布局
        return row * attr->stride[1] + col * attr->stride[2];
    }
    if (attr->n_dims == 5) {
        // 5D NC1HWC2: [1, C1, seq, 1, C2] 布局
        uint32_t lane = attr->shape[4] ? col % attr->shape[4] : 0;
        uint32_t c1 = attr->shape[4] ? col / attr->shape[4] : 0;
        return c1 * attr->stride[1] + row * attr->stride[2] + lane * attr->stride[4];
    }
    return row * col_size + col;
}

uint32_t whisper_tensor_offset_2d_int(const rknn3_tensor_attr *attr, uint32_t row, uint32_t col, uint32_t col_size) {
    if (!attr) {
        return row * col_size + col;
    }
    if (attr->n_dims == 2) {
        return row * attr->stride[0] + col * attr->stride[1];
    }
    if (attr->n_dims == 3) {
        return row * attr->stride[1] + col * attr->stride[2];
    }
    return row * col_size + col;
}

uint32_t whisper_tensor_offset_logits(const rknn3_tensor_attr *attr, uint32_t vocab_index, uint32_t seq_index) {
    if (!attr) {
        return vocab_index;
    }
    if (attr->n_dims == 3) {
        uint32_t seq = attr->shape[1] > 0 ? attr->shape[1] : 1;
        uint32_t row = seq > 1 ? std::min(seq_index, seq - 1) : 0;
        return row * attr->stride[1] + vocab_index * attr->stride[2];
    }
    if (attr->n_dims == 5) {
        uint32_t seq = attr->shape[2] > 0 ? attr->shape[2] : 1;
        uint32_t row = seq > 1 ? std::min(seq_index, seq - 1) : 0;
        uint32_t lane = attr->shape[4] ? vocab_index % attr->shape[4] : 0;
        uint32_t c1 = attr->shape[4] ? vocab_index / attr->shape[4] : 0;
        return c1 * attr->stride[1] + row * attr->stride[2] + lane * attr->stride[4];
    }
    return vocab_index;
}
