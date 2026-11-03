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
#include <string.h>
#include <vector>
#include "time_utils.h"

#include "rknn_mage_vl_llm.h"

int init_mage_vl_llm(rknn_mage_vl_llm_context* llm_ctx, const char* model_path, const char* weight_path, rknn3_llm_param* params, int n_params, RKLLMCallback* callback, uint32_t core_mask)
{
    int ret = 0;
    rknn3_context  ctx     = 0;
    rknn3_session* session = NULL;

    if (llm_ctx == NULL)
    {
        printf("init_mage_vl_llm: llm_ctx is NULL\n");
        return -1;
    }
    if (model_path == NULL || weight_path == NULL) {
        printf("init_mage_vl_llm: model_path or weight_path is NULL\n");
        return -1;
    }
    if (params == NULL || n_params <= 0 || callback == NULL) {
        printf("init_mage_vl_llm: params or callback is invalid\n");
        return -1;
    }

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1; // 使用用户管理的internal内存

    // RKNN Init
    ret = rknn3_init(&ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail ret=%d\n", ret);
        goto out;
    }

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        goto out;
    }

    //Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        goto out;
    }

    // RKNN Session Init
    session = rknn3_session_init(ctx, params, n_params);
    if (!session)
    {
        printf("Failed to initialize test session\n");
        ret = -1;
        goto out;
    }

    // Set Chat Template
    // ret = rknn3_session_set_chat_template(session, system_prompt, prompt_prefix, prompt_postfix);
    // if (ret < 0)
    // {
    //     printf("Failed to set chat template\n");
    //     goto out;
    // }

    // Set Callback
    ret = rknn3_session_set_callback(session, callback);
    if (ret < 0)
    {
        printf("Failed to set callback\n");
        goto out;
    }

    // Set to context
    llm_ctx->rknn_ctx = ctx;
    llm_ctx->rknn_sess = session;

    return ret;

out:
    if (session) {
        rknn3_session_destroy(session);
    }
    if (ctx) {
        rknn3_destroy(ctx);
    }
    return ret;
}

int release_mage_vl_llm(rknn_mage_vl_llm_context* llm_ctx)
{
    if (!llm_ctx)
    {
        printf("release_mage_vl_llm: llm_ctx is NULL\n");
        return -1;
    }

    if (llm_ctx->rknn_sess) {
        rknn3_session_destroy(llm_ctx->rknn_sess);
        llm_ctx->rknn_sess = NULL;
    }

    if (llm_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(llm_ctx->rknn_ctx);
        llm_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_mage_vl_llm(rknn_mage_vl_llm_context* llm_ctx, rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf)
{
    if ((!llm_ctx) || !(llm_ctx->rknn_sess))
    {
        printf("llm_ctx or rknn_session is NULL");
        return -1;
    }

    if (!perf) {
        printf("inference_mage_vl_llm perf metrics pointer is NULL\n");
        return -1;
    }
    if (n_inputs != MAGE_VL_LLM_INPUT_COUNT) {
        printf("inference_mage_vl_llm supports exactly %d multimodal input "
               "for the current model, got %d\n",
               MAGE_VL_LLM_INPUT_COUNT, n_inputs);
        return -1;
    }

    int ret;
    std::vector<rknn3_llm_input> inputs(n_inputs);
    rknn3_llm_infer_param llm_infer_param;

    memset(&(llm_infer_param), 0, sizeof(llm_infer_param));

    llm_infer_param.keep_history = 0;
    llm_infer_param.max_new_tokens = MAX_NEW_TOKENS;


    // Set Input Data
    inputs[0].input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;

    // Run
    printf("rknn_session_run\n");
    perf->llm_start_time = getCurrentTimeUs();
    ret = rknn3_session_run(llm_ctx->rknn_sess, inputs.data(), n_inputs,
                            &llm_infer_param);
    perf->llm_end_time = getCurrentTimeUs();
    if (ret < 0)
    {
        printf("rknn_session_run fail! ret=%d\n", ret);
        return ret;
    }

    // Query State
    RKLLMRunState state = {0};
    ret = rknn3_session_query_state(llm_ctx->rknn_sess, &state);
    if (ret < 0)
    {
        printf("rknn_session_query_state fail! ret=%d\n", ret);
        return ret;
    }
    perf->n_decode_tokens = state.n_decode_tokens;
    perf->n_prefill_tokens = state.n_prefill_tokens;

    return ret;
}
