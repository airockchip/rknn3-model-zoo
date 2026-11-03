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
#include <algorithm>

#include "hy_ocr_vl.h"
#include "common.h"
#include "image_utils.h"
#include "rknn3_api.h"
#include "time_utils.h"


int init_internal_share(rknn_app_context_t* app_ctx, uint32_t core_mask_vision, uint32_t core_mask_llm)
{
    int ret = -1;

    rknn3_core_mem_size* core_mem_sizes_vision = NULL;
    rknn3_core_mem_size* core_mem_sizes_llm = NULL;
    rknn3_tensor_mem** internal_mems_vision = NULL;
    rknn3_tensor_mem** internal_mems_llm = NULL;
    int* llm_to_vision = NULL;

    uint32_t core_num_vision = 0;
    uint32_t core_num_llm = 0;
    uint32_t core_num_vision_mask = 0;
    uint32_t core_num_llm_mask = 0;

    int core_num_same = 0;
    int idx = 0;

    ret = rknn3_query(app_ctx->vision.rknn_ctx, RKNN3_QUERY_CORE_NUMBER,
                      &core_num_vision, sizeof(core_num_vision));
    if (ret < 0) {
        printf("rknn3_query failed! ret=%d\n", ret);
        goto cleanup;
    }

    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_NUMBER,
                      &core_num_llm, sizeof(core_num_llm));
    if (ret < 0) {
        printf("rknn3_query failed! ret=%d\n", ret);
        goto cleanup;
    }

    for (int i = 0; i < 32; i++) {
        if (core_mask_vision & (1U << i)) {
            core_num_vision_mask++;
        }
    }

    for (int i = 0; i < 32; i++) {
        if (core_mask_llm & (1U << i)) {
            core_num_llm_mask++;
        }
    }

    if (core_num_vision_mask != core_num_vision) {
        printf("the core_mask_vision = %x is not match the core_num_vision = %u!\n",
               core_mask_vision, core_num_vision);
        ret = -1;
        goto cleanup;
    }

    if (core_num_llm_mask != core_num_llm) {
        printf("the core_mask_llm = %x is not match the core_num_llm = %u!\n",
               core_mask_llm, core_num_llm);
        ret = -1;
        goto cleanup;
    }

    core_mem_sizes_vision = (rknn3_core_mem_size*)malloc(sizeof(rknn3_core_mem_size) * core_num_vision);
    if (!core_mem_sizes_vision) {
        printf("Failed to allocate memory for core_mem_sizes_vision\n");
        ret = -1;
        goto cleanup;
    }

    core_mem_sizes_llm = (rknn3_core_mem_size*)malloc(sizeof(rknn3_core_mem_size) * core_num_llm);
    if (!core_mem_sizes_llm) {
        printf("Failed to allocate memory for core_mem_sizes_llm\n");
        ret = -1;
        goto cleanup;
    }

    ret = rknn3_query(app_ctx->vision.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE,
                      core_mem_sizes_vision,
                      sizeof(rknn3_core_mem_size) * core_num_vision);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        goto cleanup;
    }

    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE,
                      core_mem_sizes_llm,
                      sizeof(rknn3_core_mem_size) * core_num_llm);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        goto cleanup;
    }

    llm_to_vision = (int*)malloc(sizeof(int) * core_num_llm);
    if (!llm_to_vision) {
        printf("Failed to allocate memory for llm_to_vision\n");
        ret = -1;
        goto cleanup;
    }

    for (uint32_t i = 0; i < core_num_llm; i++) {
        llm_to_vision[i] = -1;
    }

    for (uint32_t i = 0; i < core_num_vision; i++) {
        for (uint32_t j = 0; j < core_num_llm; j++) {
            if (llm_to_vision[j] == -1 && core_mem_sizes_vision[i].core_id == core_mem_sizes_llm[j].core_id) {
                uint64_t internal_size = std::max(core_mem_sizes_vision[i].internal_size,
                                                  core_mem_sizes_llm[j].internal_size);
                core_mem_sizes_vision[i].internal_size = internal_size;
                core_mem_sizes_llm[j].internal_size = internal_size;
                core_num_same++;
                llm_to_vision[j] = i;
                break;
            }
        }
    }

    app_ctx->n_internal_mems = core_num_vision + core_num_llm - core_num_same;
    app_ctx->internal_mems = (rknn3_tensor_mem**)calloc(app_ctx->n_internal_mems,
                                                        sizeof(rknn3_tensor_mem*));
    if (!app_ctx->internal_mems) {
        printf("Failed to allocate memory for app_ctx->internal_mems array\n");
        ret = -1;
        goto cleanup;
    }

    app_ctx->internal_mem_owner = (int*)calloc(app_ctx->n_internal_mems, sizeof(int));
    if (!app_ctx->internal_mem_owner) {
        printf("Failed to allocate memory for app_ctx->internal_mem_owner array\n");
        ret = -1;
        goto cleanup;
    }

    internal_mems_vision = (rknn3_tensor_mem**)calloc(core_num_vision,
                                                      sizeof(rknn3_tensor_mem*));
    if (!internal_mems_vision) {
        printf("Failed to allocate memory for internal_mems_vision array\n");
        ret = -1;
        goto cleanup;
    }

    internal_mems_llm = (rknn3_tensor_mem**)calloc(core_num_llm,
                                                   sizeof(rknn3_tensor_mem*));
    if (!internal_mems_llm) {
        printf("Failed to allocate memory for internal_mems_llm array\n");
        ret = -1;
        goto cleanup;
    }

    for (uint32_t i = 0; i < core_num_vision; i++) {
        internal_mems_vision[i] = rknn3_create_mem(
            app_ctx->vision.rknn_ctx,
            core_mem_sizes_vision[i].internal_size,
            core_mem_sizes_vision[i].core_id,
            RKNN3_FLAG_MEMORY_CACHEABLE
        );

        if (!internal_mems_vision[i]) {
            printf("rknn3_create_mem failed for vision core %d\n",
                   core_mem_sizes_vision[i].core_id);
            ret = -1;
            goto cleanup;
        }

#ifdef DEBUG
        printf("Created user internal memory for core %d: size=%lu, virt_addr=%p, phys_addr=0x%lx\n",
               core_mem_sizes_vision[i].core_id,
               internal_mems_vision[i]->size,
               internal_mems_vision[i]->virt_addr,
               internal_mems_vision[i]->phys_addr);
#endif

        app_ctx->internal_mems[idx++] = internal_mems_vision[i];
        app_ctx->internal_mem_owner[idx - 1] = 0;  // 0: vision context
    }

    for (uint32_t i = 0; i < core_num_llm; i++) {
        if (llm_to_vision[i] != -1) {
            internal_mems_llm[i] = internal_mems_vision[llm_to_vision[i]];
            continue;
        }

        internal_mems_llm[i] = rknn3_create_mem(
            app_ctx->llm.rknn_ctx,
            core_mem_sizes_llm[i].internal_size,
            core_mem_sizes_llm[i].core_id,
            RKNN3_FLAG_MEMORY_CACHEABLE
        );

        if (!internal_mems_llm[i]) {
            printf("rknn3_create_mem failed for llm core %d\n",
                   core_mem_sizes_llm[i].core_id);
            ret = -1;
            goto cleanup;
        }

#ifdef DEBUG
        printf("Created user internal memory for core %d: size=%lu, virt_addr=%p, phys_addr=0x%lx\n",
               core_mem_sizes_llm[i].core_id,
               internal_mems_llm[i]->size,
               internal_mems_llm[i]->virt_addr,
               internal_mems_llm[i]->phys_addr);
#endif

        app_ctx->internal_mems[idx++] = internal_mems_llm[i];
        app_ctx->internal_mem_owner[idx - 1] = 1;  // 1: llm context
    }

    ret = rknn3_set_internal_mem(app_ctx->vision.rknn_ctx,
                                 internal_mems_vision,
                                 core_num_vision);
    if (ret < 0) {
        printf("rknn3_set_internal_mem vision failed! ret=%d\n", ret);
        goto cleanup;
    }

    ret = rknn3_set_internal_mem(app_ctx->llm.rknn_ctx,
                                 internal_mems_llm,
                                 core_num_llm);
    if (ret < 0) {
        printf("rknn3_set_internal_mem llm failed! ret=%d\n", ret);
        goto cleanup;
    }

cleanup:
    if (ret != 0) {
        // 释放已创建的 internal_mem 对象
        // Vision 全部为独立创建的 mem，直接释放
        if (internal_mems_vision) {
            for (uint32_t i = 0; i < core_num_vision; i++) {
                if (internal_mems_vision[i]) {
                    rknn3_destroy_mem(app_ctx->vision.rknn_ctx, internal_mems_vision[i]);
                    internal_mems_vision[i] = NULL;
                }
            }
        }
        // LLM 仅释放非共享的 mem（共享的指向 vision 的 mem，在上一循环已释放）
        if (internal_mems_llm) {
            for (uint32_t i = 0; i < core_num_llm; i++) {
                if (internal_mems_llm[i] && llm_to_vision && (llm_to_vision[i] == -1)) {
                    rknn3_destroy_mem(app_ctx->llm.rknn_ctx, internal_mems_llm[i]);
                    internal_mems_llm[i] = NULL;
                }
            }
        }
        // 释放 app_ctx->internal_mems
        if (app_ctx->internal_mems) {
            free(app_ctx->internal_mems);
            app_ctx->internal_mems = NULL;
        }
        if (app_ctx->internal_mem_owner) {
            free(app_ctx->internal_mem_owner);
            app_ctx->internal_mem_owner = NULL;
        }
    }

    if (llm_to_vision) {
        free(llm_to_vision);
        llm_to_vision = NULL;
    }

    if (internal_mems_vision) {
        free(internal_mems_vision);
        internal_mems_vision = NULL;
    }

    if (internal_mems_llm) {
        free(internal_mems_llm);
        internal_mems_llm = NULL;
    }

    if (core_mem_sizes_vision) {
        free(core_mem_sizes_vision);
        core_mem_sizes_vision = NULL;
    }

    if (core_mem_sizes_llm) {
        free(core_mem_sizes_llm);
        core_mem_sizes_llm = NULL;
    }

    return ret;
}

int release_internal_share(rknn_app_context_t* app_ctx)
{
    if (!app_ctx) {
        return 0;
    }

    if (app_ctx->internal_mems) {
        for (int i = 0; i < app_ctx->n_internal_mems; i++) {
            if (app_ctx->internal_mems[i]) {
                // Use the context that created the mem for destruction
                rknn3_context ctx = (app_ctx->internal_mem_owner && app_ctx->internal_mem_owner[i] == 1)
                                    ? app_ctx->llm.rknn_ctx
                                    : app_ctx->vision.rknn_ctx;
                rknn3_destroy_mem(ctx, app_ctx->internal_mems[i]);
                app_ctx->internal_mems[i] = NULL;
            }
        }
        free(app_ctx->internal_mems);
        app_ctx->internal_mems = NULL;
    }
    if (app_ctx->internal_mem_owner) {
        free(app_ctx->internal_mem_owner);
        app_ctx->internal_mem_owner = NULL;
    }
    app_ctx->n_internal_mems = 0;
    return 0;
}


int init_hy_ocr_vl_model(rknn_app_context_t* app_ctx, const char* llm_model_path, const char* llm_weight_path, const char* vision_model_path, const char* vision_weight_path, rknn3_llm_param* params, int n_params, RKLLMCallback* callback, uint32_t vision_core_mask, uint32_t llm_core_mask)
{
    int ret = 0;

    printf("--> init hy_ocr vision model\n");
    ret = init_hy_ocr_vision(&(app_ctx->vision), vision_model_path, vision_weight_path, vision_core_mask, app_ctx->model_width, app_ctx->model_height);
    if (ret < 0)
    {
        printf("rknn_init hy_ocr vision model fail! ret=%d\n", ret);
        return ret;
    }

    printf("--> init hy_ocr llm model\n");
    ret = init_hy_ocr_llm(&(app_ctx->llm), llm_model_path, llm_weight_path, params, n_params, callback, llm_core_mask);
    if (ret < 0)
    {
        printf("rknn_init hy_ocr llm model fail! ret=%d\n", ret);
        release_hy_ocr_vision(&(app_ctx->vision));
        return ret;
    }

    printf("--> init internal share\n");
    ret = init_internal_share(app_ctx, vision_core_mask, llm_core_mask);
    if (ret < 0)
    {
        printf("hy_ocr vl llm/vision internal memory share fail! ret=%d\n", ret);
        release_hy_ocr_llm(&(app_ctx->llm));
        release_hy_ocr_vision(&(app_ctx->vision));
        return ret;
    }

    return ret;
}

int release_hy_ocr_vl_model(rknn_app_context_t* app_ctx)
{
    if (!app_ctx) {
        return 0;
    }
    release_internal_share(app_ctx);
    release_hy_ocr_vision(&(app_ctx->vision));
    release_hy_ocr_llm(&(app_ctx->llm));
    return 0;
}

int inference_hy_ocr_vl_model(rknn_app_context_t* app_ctx, image_buffer_t* img, float16* img_embeds, const rknn3_llm_multimodal_tensor& tensor, int n_inputs, rknn_perf_metrics_t* perf)
{
    int ret;

    if ((!app_ctx) || (!img) || (!img_embeds))
    {
        printf("app_ctx, img or img_embeds is NULL\n");
        return -1;
    }

    if (!perf) {
        printf("inference_hy_ocr_vl_model perf metrics pointer is NULL\n");
        return -1;
    }

    if (app_ctx->vision.model_height < HY_OCR_PATCH_SIZE * HY_OCR_MERGE_SIZE ||
        app_ctx->vision.model_width < HY_OCR_PATCH_SIZE * HY_OCR_MERGE_SIZE) {
        printf("inference_hy_ocr_vl_model invalid model size: %dx%d (min %dx%d)\n",
               app_ctx->vision.model_width, app_ctx->vision.model_height,
               HY_OCR_PATCH_SIZE * HY_OCR_MERGE_SIZE, HY_OCR_PATCH_SIZE * HY_OCR_MERGE_SIZE);
        return -1;
    }

    printf("--> inference hy_ocr vision model\n");
    int64_t start_us = getCurrentTimeUs();
    // Compute n_image_tokens locally (runtime requires n_image_tokens=0 in tensor when n_image=0)
    // HY_OCR_PATCH_SIZE and HY_OCR_MERGE_SIZE defined in rknn_hy_ocr_vision.h (from preprocessor_config.json)
    int ph = app_ctx->vision.model_height / HY_OCR_PATCH_SIZE / HY_OCR_MERGE_SIZE;
    int pw = app_ctx->vision.model_width / HY_OCR_PATCH_SIZE / HY_OCR_MERGE_SIZE;
    int n_image_tokens = ph * (pw + 1) + 2;  // +1 for newline token per row, +2 for begin/end tokens
    ret = inference_hy_ocr_vision(&(app_ctx->vision), img, img_embeds, n_image_tokens);
    perf->vision_latency = getCurrentTimeUs() - start_us;
    if (ret != 0)
    {
        printf("inference hy_ocr vision model fail! ret=%d\n", ret);
        return ret;
    }

    printf("--> inference hy_ocr llm model\n");
    ret = inference_hy_ocr_llm(&(app_ctx->llm), tensor, n_inputs, perf);

    if (ret != 0)
    {
        printf("inference hy_ocr llm model fail! ret=%d\n", ret);
        return ret;
    }

    return ret;
}
