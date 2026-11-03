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
#include <cmath>
#include <vector>

#include "minicpm_ocr_jev.h"
#include "image_utils.h"
#include "time_utils.h"

// The user turn text, reproduced byte for byte from the reference demo.
static const char *kUserTail =
    "第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。\n"
    "请判断该 OCR 文本及其对应渲染图是否与原图内容一致。\n"
    "这是一个二分类任务：一致标为 1，不一致标为 0。";

// Runtime-side multimodal marker: rknn3_session_run rewrites every occurrence
// into image_start + n_image_tokens * image_content + image_end, and consumes
// one image block from image_embed per occurrence.
static const char *kImageMarker = "<image>";

std::string build_value_head_prompt(const std::string &ocr_text)
{
    std::string prompt;
    prompt.reserve(ocr_text.size() + 512);
    prompt += "<|im_start|>user\n";
    prompt += kImageMarker;          // original image
    prompt += "OCR文本：\n";
    prompt += ocr_text;
    prompt += "\n";
    prompt += kImageMarker;          // rendered OCR image
    prompt += kUserTail;
    prompt += "<|im_end|>\n";
    return prompt;
}

/*-------------------------------------------
        Shared internal memory management
-------------------------------------------*/

static int init_internal_share(rknn_value_head_context *app_ctx,
                               uint32_t core_mask_vision,
                               uint32_t core_mask_llm)
{
    int ret = -1;

    rknn3_core_mem_size *core_mem_sizes_vision = NULL;
    rknn3_core_mem_size *core_mem_sizes_llm = NULL;
    rknn3_tensor_mem **internal_mems_vision = NULL;
    rknn3_tensor_mem **internal_mems_llm = NULL;
    int *llm_to_vision = NULL;

    uint32_t core_num_vision = 0;
    uint32_t core_num_llm = 0;
    uint32_t core_num_vision_mask = 0;
    uint32_t core_num_llm_mask = 0;
    int core_num_same = 0;
    int idx = 0;

    ret = rknn3_query(app_ctx->vision.rknn_ctx, RKNN3_QUERY_CORE_NUMBER, &core_num_vision, sizeof(core_num_vision));
    if (ret < 0) {
        printf("rknn3_query vision core number failed! ret=%d\n", ret);
        goto cleanup;
    }
    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_NUMBER, &core_num_llm, sizeof(core_num_llm));
    if (ret < 0) {
        printf("rknn3_query llm core number failed! ret=%d\n", ret);
        goto cleanup;
    }

    for (int i = 0; i < 32; i++) {
        if (core_mask_vision & (1U << i)) {
            core_num_vision_mask++;
        }
        if (core_mask_llm & (1U << i)) {
            core_num_llm_mask++;
        }
    }
    if (core_num_vision_mask != core_num_vision) {
        printf("core_mask_vision = %x does not match core_num_vision = %u\n", core_mask_vision, core_num_vision);
        ret = -1;
        goto cleanup;
    }
    if (core_num_llm_mask != core_num_llm) {
        printf("core_mask_llm = %x does not match core_num_llm = %u\n", core_mask_llm, core_num_llm);
        ret = -1;
        goto cleanup;
    }

    core_mem_sizes_vision = (rknn3_core_mem_size *)malloc(sizeof(rknn3_core_mem_size) * core_num_vision);
    core_mem_sizes_llm = (rknn3_core_mem_size *)malloc(sizeof(rknn3_core_mem_size) * core_num_llm);
    if (!core_mem_sizes_vision || !core_mem_sizes_llm) {
        printf("Failed to allocate core_mem_sizes arrays\n");
        ret = -1;
        goto cleanup;
    }

    ret = rknn3_query(app_ctx->vision.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_vision,
                      sizeof(rknn3_core_mem_size) * core_num_vision);
    if (ret < 0) {
        printf("rknn3_query vision core memory size failed! ret=%d\n", ret);
        goto cleanup;
    }
    ret = rknn3_query(app_ctx->llm.rknn_ctx, RKNN3_QUERY_CORE_MEM_SIZE, core_mem_sizes_llm,
                      sizeof(rknn3_core_mem_size) * core_num_llm);
    if (ret < 0) {
        printf("rknn3_query llm core memory size failed! ret=%d\n", ret);
        goto cleanup;
    }

    llm_to_vision = (int *)malloc(sizeof(int) * core_num_llm);
    if (!llm_to_vision) {
        printf("Failed to allocate llm_to_vision\n");
        ret = -1;
        goto cleanup;
    }
    for (uint32_t i = 0; i < core_num_llm; i++) {
        llm_to_vision[i] = -1;
    }

    // Cores used by both modules share one physical internal pool sized to the
    // larger of the two requirements.
    for (uint32_t i = 0; i < core_num_vision; i++) {
        for (uint32_t j = 0; j < core_num_llm; j++) {
            if (core_mem_sizes_vision[i].core_id == core_mem_sizes_llm[j].core_id) {
                uint64_t internal_size = std::max(core_mem_sizes_vision[i].internal_size,
                                                  core_mem_sizes_llm[j].internal_size);
                core_mem_sizes_vision[i].internal_size = internal_size;
                core_mem_sizes_llm[j].internal_size = internal_size;
                core_num_same++;
                llm_to_vision[j] = (int)i;
                break;
            }
        }
    }

    app_ctx->n_internal_mems = (int)(core_num_vision + core_num_llm - core_num_same);
    app_ctx->internal_mems = (rknn3_tensor_mem **)calloc(app_ctx->n_internal_mems, sizeof(rknn3_tensor_mem *));
    internal_mems_vision = (rknn3_tensor_mem **)calloc(core_num_vision, sizeof(rknn3_tensor_mem *));
    internal_mems_llm = (rknn3_tensor_mem **)calloc(core_num_llm, sizeof(rknn3_tensor_mem *));
    if (!app_ctx->internal_mems || !internal_mems_vision || !internal_mems_llm) {
        printf("Failed to allocate internal memory arrays\n");
        ret = -1;
        goto cleanup;
    }

    for (uint32_t i = 0; i < core_num_vision; i++) {
        internal_mems_vision[i] = rknn3_create_mem(app_ctx->vision.rknn_ctx,
                                                   core_mem_sizes_vision[i].internal_size,
                                                   core_mem_sizes_vision[i].core_id,
                                                   RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!internal_mems_vision[i]) {
            printf("rknn3_create_mem failed for vision core %d\n", core_mem_sizes_vision[i].core_id);
            ret = -1;
            goto cleanup;
        }
        printf("internal mem core %d: size=%lu, virt_addr=%p\n",
               core_mem_sizes_vision[i].core_id, internal_mems_vision[i]->size,
               internal_mems_vision[i]->virt_addr);
        app_ctx->internal_mems[idx++] = internal_mems_vision[i];
    }

    for (uint32_t i = 0; i < core_num_llm; i++) {
        if (llm_to_vision[i] != -1) {
            internal_mems_llm[i] = internal_mems_vision[llm_to_vision[i]];
            continue;
        }
        // Allocated from the vision context so that release is uniform.
        internal_mems_llm[i] = rknn3_create_mem(app_ctx->vision.rknn_ctx,
                                                core_mem_sizes_llm[i].internal_size,
                                                core_mem_sizes_llm[i].core_id,
                                                RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!internal_mems_llm[i]) {
            printf("rknn3_create_mem failed for llm core %d\n", core_mem_sizes_llm[i].core_id);
            ret = -1;
            goto cleanup;
        }
        printf("internal mem core %d: size=%lu, virt_addr=%p\n",
               core_mem_sizes_llm[i].core_id, internal_mems_llm[i]->size,
               internal_mems_llm[i]->virt_addr);
        app_ctx->internal_mems[idx++] = internal_mems_llm[i];
    }

    ret = rknn3_set_internal_mem(app_ctx->vision.rknn_ctx, internal_mems_vision, core_num_vision);
    if (ret < 0) {
        printf("rknn3_set_internal_mem vision failed! ret=%d\n", ret);
        goto cleanup;
    }
    ret = rknn3_set_internal_mem(app_ctx->llm.rknn_ctx, internal_mems_llm, core_num_llm);
    if (ret < 0) {
        printf("rknn3_set_internal_mem llm failed! ret=%d\n", ret);
        goto cleanup;
    }

cleanup:
    free(llm_to_vision);
    free(internal_mems_vision);
    free(internal_mems_llm);
    free(core_mem_sizes_vision);
    free(core_mem_sizes_llm);
    return ret;
}

static void release_internal_share(rknn_value_head_context *app_ctx)
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
}

/*-------------------------------------------
        Model lifecycle
-------------------------------------------*/

int init_value_head_model(rknn_value_head_context *app_ctx,
                          const char *llm_model_path,
                          const char *llm_weight_path,
                          const char *vision_model_path,
                          const char *vision_weight_path,
                          rknn3_llm_param *params,
                          int n_params,
                          RKLLMCallback *callback,
                          uint32_t vision_core_mask,
                          uint32_t llm_core_mask,
                          uint32_t model_width,
                          uint32_t model_height)
{
    int ret = 0;

    printf("--> init vision model\n");
    ret = init_value_head_vision(&(app_ctx->vision), vision_model_path, vision_weight_path,
                                 vision_core_mask, model_width, model_height);
    if (ret < 0) {
        printf("init value head vision model failed! ret=%d\n", ret);
        return ret;
    }

    printf("--> init llm model\n");
    ret = init_value_head_llm(&(app_ctx->llm), llm_model_path, llm_weight_path,
                              params, n_params, callback, llm_core_mask,
                              VALUE_HEAD_NUM_LABELS);
    if (ret < 0) {
        printf("init value head llm model failed! ret=%d\n", ret);
        release_value_head_vision(&(app_ctx->vision));
        return ret;
    }

    printf("--> init internal share\n");
    ret = init_internal_share(app_ctx, vision_core_mask, llm_core_mask);
    if (ret < 0) {
        printf("llm/vision internal memory share failed! ret=%d\n", ret);
        release_value_head_llm(&(app_ctx->llm));
        release_value_head_vision(&(app_ctx->vision));
        return ret;
    }

    app_ctx->sink.expected_n_elems = VALUE_HEAD_NUM_LABELS;
    app_ctx->model_width = app_ctx->vision.model_width;
    app_ctx->model_height = app_ctx->vision.model_height;
    return 0;
}

int release_value_head_model(rknn_value_head_context *app_ctx)
{
    release_internal_share(app_ctx);
    release_value_head_vision(&(app_ctx->vision));
    release_value_head_llm(&(app_ctx->llm));
    return 0;
}

/*-------------------------------------------
        Scoring
-------------------------------------------*/

int score_value_head(rknn_value_head_context *app_ctx,
                     Tokenizer *tokenizer,
                     image_buffer_t *orig_image,
                     image_buffer_t *render_image,
                     const std::string &ocr_text,
                     float threshold,
                     bool print_prompt,
                     value_head_result *result)
{
    if (!app_ctx || !tokenizer || !orig_image || !render_image || !result) {
        printf("score_value_head: invalid argument\n");
        return -1;
    }

    int ret = 0;
    size_t embed_elems = value_head_vision_embed_elems(&(app_ctx->vision));
    int n_vision_tokens = value_head_vision_token_count(&(app_ctx->vision));
    if (embed_elems == 0 || n_vision_tokens <= 0) {
        printf("invalid vision output shape\n");
        return -1;
    }

    std::string prompt = build_value_head_prompt(ocr_text);
    if (print_prompt) {
        printf("========== prompt (%zu bytes) ==========\n%s", prompt.size(), prompt.c_str());
        printf("=======================================\n");
    }

    // Two images -> one embedding block each, laid out back to back.
    std::vector<float16> img_embeds(embed_elems * 2);
    rknn_perf_metrics_t perf;
    memset(&perf, 0, sizeof(perf));

    // Order matters: the prompt says "the first image is the original".
    ret = inference_value_head_vision(&(app_ctx->vision), orig_image, img_embeds.data(), &perf);
    if (ret != 0) {
        printf("vision inference (original image) failed! ret=%d\n", ret);
        return ret;
    }
    int64_t t_vision_1 = perf.vision_latency;

    ret = inference_value_head_vision(&(app_ctx->vision), render_image, img_embeds.data() + embed_elems, &perf);
    if (ret != 0) {
        printf("vision inference (rendered image) failed! ret=%d\n", ret);
        return ret;
    }
    int64_t t_vision_2 = perf.vision_latency;

    rknn3_llm_multimodal_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.name = "input_embeds";
    tensor.prompt = prompt.c_str();
    tensor.enable_thinking = false;
    tensor.image.image_embed = img_embeds.data();
    tensor.image.n_image = 2;
    tensor.image.n_image_tokens = (uint64_t)n_vision_tokens;
    tensor.image.image_start = "<|vision_start|>";
    tensor.image.image_end = "<|vision_end|>";
    tensor.image.image_content = "<|image_pad|>";
    tensor.image.image_width = (uint64_t)app_ctx->vision.model_width;
    tensor.image.image_height = (uint64_t)app_ctx->vision.model_height;

    ret = score_value_head_llm(&(app_ctx->llm), tensor, &(app_ctx->sink), 300000);
    if (ret != 0) {
        printf("value head scoring failed! ret=%d\n", ret);
        return ret;
    }

    float l0 = 0.0f;
    float l1 = 0.0f;
    {
        std::lock_guard<std::mutex> lock(app_ctx->sink.mutex);
        l0 = app_ctx->sink.logits[0];
        l1 = app_ctx->sink.logits[1];
        result->n_prompt_tokens = app_ctx->sink.n_prefill_tokens;
        result->llm_latency_us = app_ctx->sink.run_end_us - app_ctx->sink.run_start_us;
    }

    // Numerically stable softmax over the two logits, matching torch.softmax.
    const float m = std::max(l0, l1);
    const float e0 = std::exp(l0 - m);
    const float e1 = std::exp(l1 - m);
    const float sum = e0 + e1;

    result->logit_0 = l0;
    result->logit_1 = l1;
    result->prob_0 = e0 / sum;
    result->prob_1 = e1 / sum;
    result->threshold = threshold;
    result->label = (result->prob_1 >= threshold) ? 1 : 0;
    result->n_vision_tokens = n_vision_tokens;
    result->vision_latency_us = t_vision_1 + t_vision_2;
    return 0;
}
