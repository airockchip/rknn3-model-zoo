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
#include <algorithm>
#include <vector>

#include "dots_mocr.h"
#include "common.h"
#include "image_utils.h"
#include "rknn3_api.h"
#include "time_utils.h"

int init_internal_share(rknn_app_context_t* app_ctx, uint32_t core_mask_vision, uint32_t core_mask_llm)
{
    int ret = -1;

    if (!app_ctx) {
        printf("app_ctx is NULL\n");
        return -1;
    }

    uint32_t core_num_vision = 0;
    uint32_t core_num_llm = 0;
    ret = rknn3_query(app_ctx->vision.rknn_ctx, RKNN3_QUERY_CORE_NUMBER, &core_num_vision, sizeof(core_num_vision));
    if (ret < 0) {
        printf("rknn3_query failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_NUMBER, &core_num_llm, sizeof(core_num_llm));
    if (ret < 0) {
        printf("rknn3_query failed! ret=%d\n", ret);
        return ret;
    }
    if (core_num_vision == 0 || core_num_llm == 0) {
        printf("invalid core_num: vision=%u llm=%u\n", core_num_vision, core_num_llm);
        return -1;
    }

    // Number of cores expected from the core masks set by the caller
    uint32_t expected_core_num_vision = 0;
    uint32_t expected_core_num_llm = 0;
    for (uint32_t i = 0; i < MAX_NPU_CORES; i++) {
        if (core_mask_vision & (1 << i))    expected_core_num_vision++;
    }
    for (uint32_t i = 0; i < MAX_NPU_CORES; i++) {
        if (core_mask_llm & (1 << i))    expected_core_num_llm++;
    }
    if (expected_core_num_vision != core_num_vision) {
        printf("the core_mask_vision = %x is not match the core_num_vision = %d!\n", core_mask_vision, core_num_vision);
        return -1;
    }
    if (expected_core_num_llm != core_num_llm) {
        printf("the core_mask_llm = %x is not match the core_num_llm = %d!\n", core_mask_llm, core_num_llm);
        return -1;
    }

    std::vector<rknn3_core_mem_size> core_mem_sizes_vision(core_num_vision);
    std::vector<rknn3_core_mem_size> core_mem_sizes_llm(core_num_llm);
    ret = rknn3_query(app_ctx->vision.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_vision.data(), sizeof(rknn3_core_mem_size) * core_num_vision);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_llm.data(), sizeof(rknn3_core_mem_size) * core_num_llm);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        return ret;
    }

    // Map each LLM core to the matching vision core (-1 when unmatched/shared mem created by vision side)
    std::vector<int> llm_to_vision(core_num_llm, -1);

    int core_num_same = 0;
    for (uint32_t i = 0; i < core_num_vision; i++) {
        for (uint32_t j = 0; j < core_num_llm; j++) {
            if (llm_to_vision[j] == -1 && core_mem_sizes_vision[i].core_id == core_mem_sizes_llm[j].core_id) {
                uint64_t internal_size = std::max<uint64_t>(core_mem_sizes_vision[i].internal_size, core_mem_sizes_llm[j].internal_size);
                core_mem_sizes_vision[i].internal_size = internal_size;
                core_mem_sizes_llm[j].internal_size = internal_size;
                core_num_same ++;
                llm_to_vision[j] = i;
                break;
            }
        }
    }

    app_ctx->n_internal_mems = core_num_vision + core_num_llm - core_num_same;
    app_ctx->n_vision_internal_mems = core_num_vision;
    app_ctx->internal_mems = (rknn3_tensor_mem**)calloc(app_ctx->n_internal_mems, sizeof(rknn3_tensor_mem*));
    if (!app_ctx->internal_mems) {
        printf("Failed to allocate memory for app_ctx->internal_mems array\n");
        return -1;
    }
    std::vector<rknn3_tensor_mem*> internal_mems_vision(core_num_vision, nullptr);
    std::vector<rknn3_tensor_mem*> internal_mems_llm(core_num_llm, nullptr);

    int idx = 0;
    for (uint32_t i = 0; i < core_num_vision; i++) {
        internal_mems_vision[i] = rknn3_create_mem(app_ctx->vision.rknn_ctx, core_mem_sizes_vision[i].internal_size, core_mem_sizes_vision[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!internal_mems_vision[i]) {
            printf("rknn3_create_mem vision[%d] failed!\n", i);
            ret = -1;
            goto cleanup;
        }
        app_ctx->internal_mems[idx++] = internal_mems_vision[i];
    }
    for (uint32_t i = 0; i < core_num_llm; i++) {
        if (llm_to_vision[i] != -1) {
            internal_mems_llm[i] = internal_mems_vision[llm_to_vision[i]];
            continue;
        }
        internal_mems_llm[i] = rknn3_create_mem(app_ctx->llm.rknn_ctx, core_mem_sizes_llm[i].internal_size, core_mem_sizes_llm[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!internal_mems_llm[i]) {
            printf("rknn3_create_mem llm[%d] failed!\n", i);
            ret = -1;
            goto cleanup;
        }
        app_ctx->internal_mems[idx++] = internal_mems_llm[i];
    }

    ret = rknn3_set_internal_mem(app_ctx->vision.rknn_ctx, internal_mems_vision.data(), core_num_vision);
    if (ret < 0) {
        printf("rknn3_set_internal_mem failed! ret=%d\n", ret);
        goto cleanup;
    }
    ret = rknn3_set_internal_mem(app_ctx->llm.rknn_ctx, internal_mems_llm.data(), core_num_llm);
    if (ret < 0) {
        printf("rknn3_set_internal_mem failed! ret=%d\n", ret);
        goto cleanup;
    }

    return ret;

cleanup:
    // Destroy mems owned by each side: vision destroys all mems it created,
    // llm destroys only the ones it created itself (llm_to_vision[i] == -1);
    // shared mems appear in both vectors but are destroyed exactly once via
    // the vision context, so ownership does not depend on array ordering.
    for (uint32_t i = 0; i < core_num_vision; i++) {
        if (internal_mems_vision[i]) {
            rknn3_destroy_mem(app_ctx->vision.rknn_ctx, internal_mems_vision[i]);
            internal_mems_vision[i] = NULL;
        }
    }
    for (uint32_t i = 0; i < core_num_llm; i++) {
        if (internal_mems_llm[i] && llm_to_vision[i] == -1) {
            rknn3_destroy_mem(app_ctx->llm.rknn_ctx, internal_mems_llm[i]);
            internal_mems_llm[i] = NULL;
        }
    }
    free(app_ctx->internal_mems);
    app_ctx->internal_mems = NULL;
    app_ctx->n_internal_mems = 0;
    app_ctx->n_vision_internal_mems = 0;
    return ret;
}

int release_internal_share(rknn_app_context_t* app_ctx)
{
    if (app_ctx->internal_mems) {
        for (int i = 0; i < app_ctx->n_internal_mems; i++) {
            if (app_ctx->internal_mems[i]) {
                if (i < app_ctx->n_vision_internal_mems) {
                    rknn3_destroy_mem(app_ctx->vision.rknn_ctx, app_ctx->internal_mems[i]);
                } else {
                    rknn3_destroy_mem(app_ctx->llm.rknn_ctx, app_ctx->internal_mems[i]);
                }
                app_ctx->internal_mems[i] = NULL;
            }
        }
        free(app_ctx->internal_mems);
        app_ctx->internal_mems = NULL;
        app_ctx->n_internal_mems = 0;
        app_ctx->n_vision_internal_mems = 0;
    }
    return 0;
}

int init_dots_mocr_model(rknn_app_context_t* app_ctx, const char* llm_model_path, const char* llm_weight_path, const char* vision_model_path, const char* vision_weight_path, rknn3_llm_param* params, int n_params, RKLLMCallback& callback, uint32_t vision_core_mask, uint32_t llm_core_mask)
{
    int ret = 0;

    if (!app_ctx) {
        printf("app_ctx is NULL\n");
        return -1;
    }

    printf("--> init dots_mocr vision model\n");
    ret = init_dots_mocr_vision(&(app_ctx->vision), vision_model_path, vision_weight_path, vision_core_mask, app_ctx->model_width, app_ctx->model_height);
    if (ret < 0) {
        printf("rknn_init dots_mocr vision model fail! ret=%d\n", ret);
        return ret;
    }

    printf("--> init dots_mocr llm model\n");
    ret = init_dots_mocr_llm(&(app_ctx->llm), llm_model_path, llm_weight_path, params, n_params, callback, llm_core_mask);
    if (ret < 0) {
        printf("rknn_init dots_mocr llm model fail! ret=%d\n", ret);
        release_dots_mocr_vision(&(app_ctx->vision));
        return ret;
    }

    printf("--> init internal share\n");
    ret = init_internal_share(app_ctx, vision_core_mask, llm_core_mask);
    if (ret < 0) {
        printf("dots_mocr llm/vision internal memeory share fail! ret=%d\n", ret);
        release_dots_mocr_llm(&(app_ctx->llm));
        release_dots_mocr_vision(&(app_ctx->vision));
        return ret;
    }

    return ret;
}

int release_dots_mocr_model(rknn_app_context_t* app_ctx)
{
    release_internal_share(app_ctx);
    release_dots_mocr_vision(&(app_ctx->vision));
    release_dots_mocr_llm(&(app_ctx->llm));
    return 0;
}

int inference_dots_mocr_model(
    rknn_app_context_t* app_ctx,
    image_buffer_t* img,
    float16* img_embeds,
    size_t img_embeds_size,
    const rknn3_llm_multimodal_tensor& tensor,
    int n_inputs,
    rknn_perf_metrics_t* perf)
{
    int ret;

    if ((!app_ctx) || (!img) || (!img_embeds) || (!perf)) {
        printf("app_ctx, img, img_embeds or perf is NULL\n");
        return -1;
    }

    printf("--> inference dots_mocr vision model\n");
    int start_us = getCurrentTimeUs();
    ret = inference_dots_mocr_vision(&(app_ctx->vision), img, img_embeds, img_embeds_size);
    perf->vision_latency = getCurrentTimeUs() - start_us;
    if (ret < 0) {
        printf("inference dots_mocr vision model fail! ret=%d\n", ret);
        return ret;
    }

    printf("--> inference dots_mocr llm model\n");
    ret = inference_dots_mocr_llm(&(app_ctx->llm), tensor, n_inputs, perf);
    if (ret < 0) {
        printf("inference dots_mocr llm model fail! ret=%d\n", ret);
        return ret;
    }

    return ret;
}
