// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Licensed under the Apache License, Version 2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#include "lfm2_vl.h"
#include "common.h"
#include "image_utils.h"
#include "rknn3_api.h"
#include "time_utils.h"

int init_internal_share(rknn_app_context_t* app_ctx, uint32_t core_mask_vision, uint32_t core_mask_llm)
{
    int ret = -1;
    rknn3_core_mem_size* core_mem_sizes_vision = nullptr;
    rknn3_core_mem_size* core_mem_sizes_llm = nullptr;
    rknn3_tensor_mem** internal_mems_vision = nullptr;
    rknn3_tensor_mem** internal_mems_llm = nullptr;
    uint32_t core_num_vision = 0;
    uint32_t core_num_llm = 0;
    uint32_t core_num_vision_ = 0;
    uint32_t core_num_llm_ = 0;
    int core_num_same = 0;
    int idx = 0;
    std::vector<int> llm_to_vision;

    ret = rknn3_query(app_ctx->vision.rknn_ctx, RKNN3_QUERY_CORE_NUMBER, &core_num_vision, sizeof(core_num_vision));
    if (ret < 0) {
        printf("rknn3_query failed! ret=%d\n", ret);
        goto cleanup;
    }
    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_NUMBER, &core_num_llm, sizeof(core_num_llm));
    if (ret < 0) {
        printf("rknn3_query failed! ret=%d\n", ret);
        goto cleanup;
    }

    for (int i = 0; i < 32; i++) {
        if (core_mask_vision & (1 << i))    core_num_vision_++;
    }
    for (int i = 0; i < 32; i++) {
        if (core_mask_llm & (1 << i))    core_num_llm_++;
    }
    if (core_num_vision_ != core_num_vision) {
        printf("the core_mask_vision = %x is not match the core_num_vision = %d!\n", core_mask_vision, core_num_vision);
        ret = -1;
        goto cleanup;
    }
    if (core_num_llm_ != core_num_llm) {
        printf("the core_mask_llm = %x is not match the core_num_llm = %d!\n", core_mask_llm, core_num_llm);
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
    ret = rknn3_query(app_ctx->vision.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_vision, sizeof(rknn3_core_mem_size) * core_num_vision);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        goto cleanup;
    }
    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_llm, sizeof(rknn3_core_mem_size) * core_num_llm);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        goto cleanup;
    }

    // Use std::vector instead of a VLA for C++ standard compliance
    llm_to_vision.resize(core_num_llm, -1);

    for (int i = 0; i < core_num_vision; i++) {
        for (int j = 0; j < core_num_llm; j++) {
            if (core_mem_sizes_vision[i].core_id == core_mem_sizes_llm[j].core_id) {
                uint64_t internal_size = std::max(core_mem_sizes_vision[i].internal_size, core_mem_sizes_llm[j].internal_size);
                core_mem_sizes_vision[i].internal_size = internal_size;
                core_mem_sizes_llm[j].internal_size = internal_size;
                core_num_same ++;
                llm_to_vision[j] = i;
                break;
            }
        }
    }

    app_ctx->n_internal_mems = core_num_vision + core_num_llm - core_num_same;
    app_ctx->internal_mems = (rknn3_tensor_mem**)calloc(app_ctx->n_internal_mems, sizeof(rknn3_tensor_mem*));
    app_ctx->internal_mems_ctx = (rknn3_context*)calloc(app_ctx->n_internal_mems, sizeof(rknn3_context));
    if (!app_ctx->internal_mems || !app_ctx->internal_mems_ctx) {
        printf("Failed to allocate memory for app_ctx->internal_mems array\n");
        ret = -1;
        goto cleanup;
    }
    internal_mems_vision = (rknn3_tensor_mem**)calloc(core_num_vision, sizeof(rknn3_tensor_mem*));
    if (!internal_mems_vision) {
        printf("Failed to allocate memory for internal_mems_vision array\n");
        ret = -1;
        goto cleanup;
    }
    internal_mems_llm = (rknn3_tensor_mem**)calloc(core_num_llm, sizeof(rknn3_tensor_mem*));
    if (!internal_mems_llm) {
        printf("Failed to allocate memory for internal_mems_llm array\n");
        ret = -1;
        goto cleanup;
    }

    for (uint32_t i = 0; i < core_num_vision; i++) {
        printf("[internal share] vision create_mem: core_id=%d, internal_size=%llu (0x%llx)\n",
               core_mem_sizes_vision[i].core_id,
               (unsigned long long)core_mem_sizes_vision[i].internal_size,
               (unsigned long long)core_mem_sizes_vision[i].internal_size);
        internal_mems_vision[i] = rknn3_create_mem(app_ctx->vision.rknn_ctx, core_mem_sizes_vision[i].internal_size, core_mem_sizes_vision[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!internal_mems_vision[i]) {
            ret = -1;
            goto cleanup;
        }
        app_ctx->internal_mems[idx] = internal_mems_vision[i];
        app_ctx->internal_mems_ctx[idx] = app_ctx->vision.rknn_ctx;
        idx++;
    }
    for (uint32_t i = 0; i < core_num_llm; i++) {
        if (llm_to_vision[i] != -1) {
            internal_mems_llm[i] = internal_mems_vision[llm_to_vision[i]];
            continue;
        }
        // Dedicated LLM core mem must be created from the LLM context
        printf("[internal share] llm(dedicated) create_mem: core_id=%d, internal_size=%llu (0x%llx)\n",
               core_mem_sizes_llm[i].core_id,
               (unsigned long long)core_mem_sizes_llm[i].internal_size,
               (unsigned long long)core_mem_sizes_llm[i].internal_size);
        internal_mems_llm[i] = rknn3_create_mem(app_ctx->llm.rknn_ctx, core_mem_sizes_llm[i].internal_size, core_mem_sizes_llm[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!internal_mems_llm[i]) {
            ret = -1;
            goto cleanup;
        }
        app_ctx->internal_mems[idx] = internal_mems_llm[i];
        app_ctx->internal_mems_ctx[idx] = app_ctx->llm.rknn_ctx;
        idx++;
    }

    ret = rknn3_set_internal_mem(app_ctx->vision.rknn_ctx, internal_mems_vision, core_num_vision);
    if (ret < 0) {
        printf("rknn3_set_internal_mem failed! ret=%d\n", ret);
        goto cleanup;
    }
    ret = rknn3_set_internal_mem(app_ctx->llm.rknn_ctx, internal_mems_llm, core_num_llm);
    if (ret < 0) {
        printf("rknn3_set_internal_mem failed! ret=%d\n", ret);
        goto cleanup;
    }

    // Success: free temporary arrays.
    // Mem objects are owned by app_ctx->internal_mems (created mems) and
    // app_ctx->internal_mems_ctx (the context that created each mem, used
    // by release_internal_share to call rknn3_destroy_mem with correct ctx).
    free(internal_mems_vision);
    free(internal_mems_llm);
    free(core_mem_sizes_vision);
    free(core_mem_sizes_llm);
    return ret;

cleanup:
    // Unified cleanup: only LLM-dedicated mems are destroyed here.
    // Mems shared with vision (llm_to_vision[i] != -1) are destroyed in the vision loop below.
    if (internal_mems_llm) {
        for (uint32_t i = 0; i < core_num_llm; i++) {
            if (internal_mems_llm[i] && llm_to_vision[i] == -1) {
                rknn3_destroy_mem(app_ctx->llm.rknn_ctx, internal_mems_llm[i]);
            }
        }
        free(internal_mems_llm);
    }
    if (internal_mems_vision) {
        for (uint32_t i = 0; i < core_num_vision; i++) {
            if (internal_mems_vision[i]) {
                rknn3_destroy_mem(app_ctx->vision.rknn_ctx, internal_mems_vision[i]);
            }
        }
        free(internal_mems_vision);
    }
    if (app_ctx->internal_mems) {
        // Note: mems already destroyed above via internal_mems_llm/vision arrays
        // Just free the pointer array, do NOT destroy mems again (double-free risk)
        free(app_ctx->internal_mems);
        app_ctx->internal_mems = nullptr;
    }
    if (app_ctx->internal_mems_ctx) {
        free(app_ctx->internal_mems_ctx);
        app_ctx->internal_mems_ctx = nullptr;
    }
    app_ctx->n_internal_mems = 0;
    if (core_mem_sizes_llm) free(core_mem_sizes_llm);
    if (core_mem_sizes_vision) free(core_mem_sizes_vision);
    return ret;
}

int release_internal_share(rknn_app_context_t* app_ctx)
{
    if (app_ctx->internal_mems) {
        for (int i = 0; i < app_ctx->n_internal_mems; i++) {
            if (app_ctx->internal_mems[i]) {
                // Use the correct context that was used to create this mem
                rknn3_context ctx = app_ctx->internal_mems_ctx[i];
                rknn3_destroy_mem(ctx, app_ctx->internal_mems[i]);
                app_ctx->internal_mems[i] = NULL;
            }
        }
        free(app_ctx->internal_mems);
        app_ctx->internal_mems = NULL;
    }
    if (app_ctx->internal_mems_ctx) {
        free(app_ctx->internal_mems_ctx);
        app_ctx->internal_mems_ctx = NULL;
    }
    app_ctx->n_internal_mems = 0;
    return 0;
}

int init_lfm2_vl_model(rknn_app_context_t* app_ctx,
                       const char* llm_model_path,
                       const char* llm_weight_path,
                       const char* vision_model_path,
                       const char* vision_weight_path,
                       rknn3_llm_param* params,
                       int n_params,
                       RKLLMCallback& callback,
                       uint32_t vision_core_mask,
                       uint32_t llm_core_mask)
{
    int ret = 0;

    printf("--> init lfm2_vl vision model\n");
    ret = init_lfm2_vl_vision(&(app_ctx->vision), vision_model_path, vision_weight_path, vision_core_mask);
    if (ret < 0) {
        printf("rknn_init lfm2_vl vision model fail! ret=%d\n", ret);
        return ret;
    }

    printf("--> init lfm2_vl llm model\n");
    ret = init_lfm2_vl_llm(&(app_ctx->llm), llm_model_path, llm_weight_path, params, n_params, callback, llm_core_mask);
    if (ret < 0) {
        printf("rknn_init lfm2_vl llm model fail! ret=%d\n", ret);
        release_lfm2_vl_vision(&(app_ctx->vision));
        return ret;
    }

    printf("--> init internal share\n");
    ret = init_internal_share(app_ctx, vision_core_mask, llm_core_mask);
    if (ret < 0) {
        printf("llm/vision internal memory share fail! ret=%d\n", ret);
        release_lfm2_vl_llm(&(app_ctx->llm));
        release_lfm2_vl_vision(&(app_ctx->vision));
        return ret;
    }

    // Clear KV cache before first inference.
    // Not fatal: a freshly initialized session has an empty KV cache, so a
    // failure here only skips the clear and inference can still proceed.
    if (clear_lfm2_vl_llm_cache(&(app_ctx->llm)) != 0) {
        printf("[WARN] clear lfm2_vl llm cache fail, continue anyway\n");
    }

    return 0;
}

int release_lfm2_vl_model(rknn_app_context_t* app_ctx)
{
    release_internal_share(app_ctx);
    release_lfm2_vl_vision(&(app_ctx->vision));
    release_lfm2_vl_llm(&(app_ctx->llm));
    return 0;
}

int inference_lfm2_vl_model(rknn_app_context_t* app_ctx,
                            image_buffer_t* img,
                            float16* img_embeds,
                            rknn3_llm_multimodal_tensor tensor,
                            int n_inputs,
                            rknn_perf_metrics_t* perf)
{
    int ret;

    if ((!app_ctx) || (!img) || (!img_embeds) || (!perf)) {
        printf("app_ctx, img, img_embeds or perf is NULL\n");
        return -1;
    }

    printf("--> inference lfm2_vl vision model\n");
    int64_t start_us = getCurrentTimeUs();
    ret = inference_lfm2_vl_vision(&(app_ctx->vision), img, img_embeds);
    perf->vision_latency = getCurrentTimeUs() - start_us;
    if (ret != 0) {
        printf("inference lfm2_vl vision model fail! ret=%d\n", ret);
        return ret;
    }

    printf("--> inference lfm2_vl llm model\n");
    ret = inference_lfm2_vl_llm(&(app_ctx->llm), tensor, n_inputs, perf);
    if (ret != 0) {
        printf("inference lfm2_vl llm model fail! ret=%d\n", ret);
        return ret;
    }

    return ret;
}
