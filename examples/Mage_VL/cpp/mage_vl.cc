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
#include <algorithm>
#include <inttypes.h>
#include <vector>

#include "mage_vl.h"
#include "common.h"
#include "image_utils.h"
#include "rknn3_api.h"
#include "time_utils.h"


static void destroy_internal_mems(rknn_app_context_t* app_ctx)
{
    if (app_ctx == NULL) {
        return;
    }

    for (rknn3_tensor_mem*& internal_mem : app_ctx->internal_mems) {
        if (internal_mem != NULL) {
            rknn3_destroy_mem(app_ctx->vision.rknn_ctx,
                              internal_mem);
            internal_mem = NULL;
        }
    }
    app_ctx->internal_mems.clear();
}

int init_internal_share(rknn_app_context_t* app_ctx, uint32_t core_mask_vision, uint32_t core_mask_llm)
{
    int ret = -1;
    uint32_t core_num_vision = 0;
    uint32_t core_num_llm = 0;
    uint32_t core_num_vision_mask = 0;
    uint32_t core_num_llm_mask = 0;
    uint32_t queried_vision_mask = 0;
    uint32_t queried_llm_mask = 0;
    int core_num_same = 0;
    std::vector<rknn3_core_mem_size> core_mem_sizes_vision;
    std::vector<rknn3_core_mem_size> core_mem_sizes_llm;
    std::vector<int> llm_to_vision;
    std::vector<rknn3_tensor_mem*> internal_mems_vision;
    std::vector<rknn3_tensor_mem*> internal_mems_llm;

    if (app_ctx == NULL) {
        printf("init_internal_share: app_ctx is NULL\n");
        return -1;
    }
    if (!app_ctx->internal_mems.empty()) {
        printf("init_internal_share: internal memory has already been initialized\n");
        return -1;
    }

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

    core_num_vision_mask =
        static_cast<uint32_t>(__builtin_popcount(core_mask_vision));
    core_num_llm_mask =
        static_cast<uint32_t>(__builtin_popcount(core_mask_llm));

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

    core_mem_sizes_vision.resize(core_num_vision);
    core_mem_sizes_llm.resize(core_num_llm);

    ret = rknn3_query(app_ctx->vision.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE,
                      core_mem_sizes_vision.data(),
                      sizeof(rknn3_core_mem_size) * core_num_vision);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        goto cleanup;
    }

    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE,
                      core_mem_sizes_llm.data(),
                      sizeof(rknn3_core_mem_size) * core_num_llm);
    if (ret < 0) {
        printf("rknn3_query core memory size failed! ret=%d\n", ret);
        goto cleanup;
    }

    for (uint32_t i = 0; i < core_num_vision; i++) {
        int core_id = core_mem_sizes_vision[i].core_id;
        if (core_id < 0 || core_id >= 32) {
            printf("vision returned invalid core id: %d\n", core_id);
            ret = -1;
            goto cleanup;
        }
        queried_vision_mask |= UINT32_C(1) << core_id;
    }
    for (uint32_t i = 0; i < core_num_llm; i++) {
        int core_id = core_mem_sizes_llm[i].core_id;
        if (core_id < 0 || core_id >= 32) {
            printf("llm returned invalid core id: %d\n", core_id);
            ret = -1;
            goto cleanup;
        }
        queried_llm_mask |= UINT32_C(1) << core_id;
    }
    if (queried_vision_mask != core_mask_vision) {
        printf("vision core mask mismatch: requested=0x%x queried=0x%x\n",
               core_mask_vision, queried_vision_mask);
        ret = -1;
        goto cleanup;
    }
    if (queried_llm_mask != core_mask_llm) {
        printf("llm core mask mismatch: requested=0x%x queried=0x%x\n",
               core_mask_llm, queried_llm_mask);
        ret = -1;
        goto cleanup;
    }

    llm_to_vision.assign(core_num_llm, -1);

    for (uint32_t i = 0; i < core_num_vision; i++) {
        for (uint32_t j = 0; j < core_num_llm; j++) {
            if (core_mem_sizes_vision[i].core_id == core_mem_sizes_llm[j].core_id) {
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

    internal_mems_vision.assign(core_num_vision, NULL);
    internal_mems_llm.assign(core_num_llm, NULL);

    app_ctx->internal_mems.reserve(core_num_vision + core_num_llm -
                                   core_num_same);

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

        printf("Created user internal memory for core %d: size=%" PRIu64
               ", virt_addr=%p, phys_addr=0x%" PRIx64 "\n",
               core_mem_sizes_vision[i].core_id,
               static_cast<uint64_t>(internal_mems_vision[i]->size),
               internal_mems_vision[i]->virt_addr,
               static_cast<uint64_t>(internal_mems_vision[i]->phys_addr));

        app_ctx->internal_mems.push_back(internal_mems_vision[i]);
    }

    for (uint32_t i = 0; i < core_num_llm; i++) {
        if (llm_to_vision[i] != -1) {
            internal_mems_llm[i] = internal_mems_vision[llm_to_vision[i]];
            continue;
        }

        // 都使用 vision.rknn_ctx 分配，方便后面统一释放
        internal_mems_llm[i] = rknn3_create_mem(
            app_ctx->vision.rknn_ctx,
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

        printf("Created user internal memory for core %d: size=%" PRIu64
               ", virt_addr=%p, phys_addr=0x%" PRIx64 "\n",
               core_mem_sizes_llm[i].core_id,
               static_cast<uint64_t>(internal_mems_llm[i]->size),
               internal_mems_llm[i]->virt_addr,
               static_cast<uint64_t>(internal_mems_llm[i]->phys_addr));

        app_ctx->internal_mems.push_back(internal_mems_llm[i]);
    }

    ret = rknn3_set_internal_mem(app_ctx->vision.rknn_ctx,
                                 internal_mems_vision.data(),
                                 core_num_vision);
    if (ret < 0) {
        printf("rknn3_set_internal_mem vision failed! ret=%d\n", ret);
        goto cleanup;
    }

    ret = rknn3_set_internal_mem(app_ctx->llm.rknn_ctx,
                                 internal_mems_llm.data(),
                                 core_num_llm);
    if (ret < 0) {
        printf("rknn3_set_internal_mem llm failed! ret=%d\n", ret);
        goto cleanup;
    }

cleanup:
    if (ret < 0) {
        destroy_internal_mems(app_ctx);
    }
    return ret;
}

int release_internal_share(rknn_app_context_t* app_ctx)
{
    destroy_internal_mems(app_ctx);
    return 0;
}




int init_mage_vl_model(rknn_app_context_t* app_ctx, const char* llm_model_path, const char* llm_weight_path, const char* vision_model_path, const char* vision_weight_path, rknn3_llm_param* params, int n_params, RKLLMCallback* callback, uint32_t vision_core_mask, uint32_t llm_core_mask)
{
    int ret = 0;
    bool vision_initialized = false;
    bool llm_initialized = false;

    if (app_ctx == NULL || llm_model_path == NULL ||
        llm_weight_path == NULL || vision_model_path == NULL ||
        vision_weight_path == NULL || params == NULL || n_params <= 0 ||
        callback == NULL) {
        printf("init_mage_vl_model: invalid argument\n");
        return -1;
    }

    printf("--> init mage_vl vision model\n");
    ret = init_mage_vl_vision(&(app_ctx->vision), vision_model_path, vision_weight_path, vision_core_mask, app_ctx->model_width, app_ctx->model_height);
    if (ret < 0)
    {
        printf("rknn_init mage_vl vision model fail! ret=%d\n", ret);
        goto cleanup;
    }
    vision_initialized = true;

    printf("--> init mage_vl llm model\n");
    ret = init_mage_vl_llm(&(app_ctx->llm), llm_model_path, llm_weight_path, params, n_params, callback, llm_core_mask);
    if (ret < 0)
    {
        printf("rknn_init mage_vl llm model fail! ret=%d\n", ret);
        goto cleanup;
    }
    llm_initialized = true;

    printf("--> init internal share\n");
    ret = init_internal_share(app_ctx, vision_core_mask, llm_core_mask);
    if (ret < 0)
    {
        printf("mage_vl llm/vision internal memory share fail! ret=%d\n", ret);
        goto cleanup;
    }

    return 0;

cleanup:
    release_internal_share(app_ctx);
    if (llm_initialized) {
        release_mage_vl_llm(&(app_ctx->llm));
    }
    if (vision_initialized) {
        release_mage_vl_vision(&(app_ctx->vision));
    }
    return ret;
}

int release_mage_vl_model(rknn_app_context_t *app_ctx)
{
    release_internal_share(app_ctx);
    release_mage_vl_vision(&(app_ctx->vision));
    release_mage_vl_llm(&(app_ctx->llm));
    return 0;
}
