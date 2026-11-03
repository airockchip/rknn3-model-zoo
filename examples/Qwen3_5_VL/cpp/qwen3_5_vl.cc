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

#include "qwen3_5_vl.h"
#include "common.h"
#include "image_utils.h"
#include "rknn3_api.h"
#include "time_utils.h"

#ifdef ENABLE_SPEEDUP
static const SpeedUPConfig g_speedup_config = {
    .tau = 0.90f,
    .min_ratio = 0.6f,
    .max_ratio = 0.75f,
    .stride = 16
};

#endif

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

    app_ctx->n_internal_mems = core_num_vision + core_num_llm - core_num_same;
    app_ctx->internal_mems = (rknn3_tensor_mem**)calloc(app_ctx->n_internal_mems,
                                                        sizeof(rknn3_tensor_mem*));
    if (!app_ctx->internal_mems) {
        printf("Failed to allocate memory for app_ctx->internal_mems array\n");
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

        printf("Created user internal memory for core %d: size=%lu, virt_addr=%p, phys_addr=0x%lx\n",
               core_mem_sizes_vision[i].core_id,
               internal_mems_vision[i]->size,
               internal_mems_vision[i]->virt_addr,
               internal_mems_vision[i]->phys_addr);

        app_ctx->internal_mems[idx++] = internal_mems_vision[i];
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

        printf("Created user internal memory for core %d: size=%lu, virt_addr=%p, phys_addr=0x%lx\n",
               core_mem_sizes_llm[i].core_id,
               internal_mems_llm[i]->size,
               internal_mems_llm[i]->virt_addr,
               internal_mems_llm[i]->phys_addr);

        app_ctx->internal_mems[idx++] = internal_mems_llm[i];
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

    if (app_ctx->internal_mems) {
        for (int i = 0; i < app_ctx->n_internal_mems; i++) {
            if (app_ctx->internal_mems[i]) {
                rknn3_destroy_mem(app_ctx->vision.rknn_ctx, app_ctx->internal_mems[i]);
                app_ctx->internal_mems[i] = NULL;
            }
        }
        free(app_ctx->internal_mems);
        app_ctx->internal_mems = NULL;
        app_ctx->n_internal_mems = 0;
    }
    return 0;
}




int init_qwen3_5_vl_model(rknn_app_context_t* app_ctx, const char* llm_model_path, const char* llm_weight_path, const char* vision_model_path, const char* vision_weight_path, rknn3_llm_param* params, int n_params, RKLLMCallback* callback, uint32_t vision_core_mask, uint32_t llm_core_mask, const char* vision_lora_weight_path)
{
    int ret = 0;

    printf("--> init qwen3_5_vl vision model\n");
    ret = init_qwen3_5_vl_vision(&(app_ctx->vision), vision_model_path, vision_weight_path, vision_core_mask, app_ctx->model_width, app_ctx->model_height);
    if (ret < 0)
    {
        printf("rknn_init qwen3_5_vl vision model fail! ret=%d\n", ret);
        return ret;
    }

    printf("--> init qwen3_5_vl llm model\n");
    ret = init_qwen3_5_vl_llm(&(app_ctx->llm), llm_model_path, llm_weight_path, params, n_params, callback, llm_core_mask);
    if (ret < 0)
    {
        printf("rknn_init qwen3_5_vl llm model fail! ret=%d\n", ret);
        release_qwen3_5_vl_vision(&(app_ctx->vision));
        return ret;
    }

    printf("--> init internal share\n");
    ret = init_internal_share(app_ctx, vision_core_mask, llm_core_mask);
    if (ret < 0)
    {
        printf("qwen3_5_vl llm/vision internal memeory share fail! ret=%d\n", ret);
        release_qwen3_5_vl_llm(&(app_ctx->llm));
        release_qwen3_5_vl_vision(&(app_ctx->vision));
        return ret;
    }

    // rknn3_lora_init must run after shared internal memory is configured;
    // otherwise rknn3_set_internal_mem can invalidate the loaded adapter.
    if (vision_lora_weight_path != nullptr) {
        printf("--> setup qwen3_5_vl vision LoRA\n");
        ret = setup_qwen3_5_vl_vision_lora(&(app_ctx->vision), vision_lora_weight_path);
        if (ret < 0) {
            printf("setup qwen3_5_vl vision LoRA failed! ret=%d\n", ret);
            return ret;
        }
    }

#ifdef ENABLE_SPEEDUP
    app_ctx->base_callback = *callback;
    app_ctx->speedup = speedup_create(&g_speedup_config);
    if (!app_ctx->speedup) {
        printf("[SpeedUP] create failed; fallback to normal inference without SpeedUP\n");
    } else {
        ret = speedup_attach_mrope_callback(app_ctx->speedup,
                                           app_ctx->llm.rknn_ctx,
                                           app_ctx->llm.rknn_sess,
                                           &app_ctx->base_callback);
        if (ret != 0) {
            printf("[SpeedUP] initialization failed, ret=%d; fallback to normal inference without SpeedUP\n", ret);
            ret = speedup_detach_mrope_callback(app_ctx->speedup, app_ctx->llm.rknn_sess);
            if (ret != 0) {
                printf("[SpeedUP] fallback failed! ret=%d; stop inference\n", ret);
                return ret;
            }
            speedup_destroy(app_ctx->speedup);
            app_ctx->speedup = NULL;
        } else {
            printf("[SpeedUP] ready, version=%s\n",
                   speedup_get_version());
        }
    }
#endif
    return ret;
}

int prepare_qwen3_5_vl_lora_session(rknn_app_context_t* app_ctx,
                                    const char* llm_lora_weight_path)
{
    if (!app_ctx) {
        return -1;
    }

#ifdef ENABLE_SPEEDUP
    // SpeedUP is attached to the Base session. It cannot survive the session
    // replacement and is intentionally disabled for the LoRA comparison pass.
    if (app_ctx->speedup) {
        speedup_cleanup(app_ctx->speedup, app_ctx->llm.rknn_sess);
        app_ctx->speedup = NULL;
        printf("[SpeedUP] disabled for LoRA session\n");
    }
#endif

    int ret = destroy_qwen3_5_vl_llm_base_session(&(app_ctx->llm));
    if (ret < 0) {
        return ret;
    }
    return create_qwen3_5_vl_llm_lora_session(&(app_ctx->llm), llm_lora_weight_path);
}

int release_qwen3_5_vl_model(rknn_app_context_t *app_ctx)
{
#ifdef ENABLE_SPEEDUP
    if (app_ctx->speedup) {
        speedup_cleanup(app_ctx->speedup, app_ctx->llm.rknn_sess);
        app_ctx->speedup = NULL;
    }
#endif
    release_internal_share(app_ctx);
    release_qwen3_5_vl_vision(&(app_ctx->vision));
    release_qwen3_5_vl_llm(&(app_ctx->llm));
    return 0;
}

int inference_qwen3_5_vl_model_vision(rknn_app_context_t* app_ctx,
                                      image_buffer_t* img,
                                      float16* img_embeds,
                                      rknn_perf_metrics_t* perf,
                                      bool use_lora)
{
    if (!app_ctx || !img || !img_embeds || !perf) {
        printf("app_ctx, img, img_embeds or perf is NULL\n");
        return -1;
    }
    printf("--> inference qwen3_5_vl vision model (%s)\n", use_lora ? "lora" : "base");
    int64_t start_us = getCurrentTimeUs();
    int ret = inference_qwen3_5_vl_vision(&(app_ctx->vision), img, img_embeds, use_lora);
    perf->vision_latency = getCurrentTimeUs() - start_us;
    if (ret != 0) {
        printf("inference qwen3_5_vl vision model fail! ret=%d\n", ret);
        return ret;
    }
    return 0;
}

int inference_qwen3_5_vl_model_llm(rknn_app_context_t* app_ctx,
                                   rknn3_llm_multimodal_tensor tensor,
                                   int n_inputs,
                                   rknn_perf_metrics_t* perf,
                                   bool use_lora,
                                   float speedup_ratio)
{
    if (!app_ctx || !perf || n_inputs != 1) {
        printf("Qwen3.5-VL requires one LLM input descriptor\n");
        return -1;
    }

    int ret = 0;

#ifdef ENABLE_SPEEDUP
    // SpeedUP is bound to the Base session and is not used for a LoRA pass.
    if (use_lora) {
        if (app_ctx->speedup) {
            printf("[SpeedUP] LoRA session requested; SpeedUP should have been released\n");
        }
    } else if (!app_ctx->speedup) {
        printf("[SpeedUP] handle is NULL, run normal inference without SpeedUP\n");
    } else {
        int original_tokens = tensor.image.n_image_tokens;
        int output_count_per_input = original_tokens;
        ret = speedup_prepare_rknn_multimodal_tensor(app_ctx->speedup,
                                                   tensor.image.image_embed,
                                                   app_ctx->vision.embeds_shape,
                                                   app_ctx->vision.embeds_ndims,
                                                   &tensor,
                                                   app_ctx->vision.model_width,
                                                   app_ctx->vision.model_height,
                                                   speedup_ratio,
                                                   app_ctx->llm.rknn_sess,
                                                   &output_count_per_input);
        if (ret != 0) {
            printf("[SpeedUP] processing failed! ret=%d; fallback to normal inference without SpeedUP\n", ret);
            ret = speedup_detach_mrope_callback(app_ctx->speedup, app_ctx->llm.rknn_sess);
            if (ret != 0) {
                printf("[SpeedUP] fallback failed! ret=%d; stop inference\n", ret);
                return ret;
            }
            speedup_destroy(app_ctx->speedup);
            app_ctx->speedup = NULL;
        }
    }
#endif
    printf("--> inference qwen3_5_vl llm model\n");
    ret = use_lora
        ? inference_qwen3_5_vl_llm_lora(&(app_ctx->llm), tensor, n_inputs, perf)
        : inference_qwen3_5_vl_llm_base(&(app_ctx->llm), tensor, n_inputs, perf);

    if (ret != 0) {
        printf("inference qwen3_5_vl llm model fail! ret=%d\n", ret);
        return ret;
    }
    return 0;
}

int inference_qwen3_5_vl_model(rknn_app_context_t *app_ctx,
                               image_buffer_t *img,
                               float16 *img_embeds,
                               rknn3_llm_multimodal_tensor tensor,
                               int n_inputs,
                               rknn_perf_metrics_t *perf,
                               float speedup_ratio)
{
    if (!app_ctx || !img || !img_embeds || !perf || n_inputs != 1) {
        printf("invalid qwen3.5-vl inference argument\n");
        return -1;
    }

    int ret = inference_qwen3_5_vl_model_vision(app_ctx, img, img_embeds, perf, false);
    if (ret != 0) {
        return ret;
    }
    return inference_qwen3_5_vl_model_llm(app_ctx, tensor, n_inputs, perf, false, speedup_ratio);

}
