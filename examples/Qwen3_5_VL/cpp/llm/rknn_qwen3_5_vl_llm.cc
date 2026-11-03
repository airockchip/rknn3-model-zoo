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
#include "time_utils.h"

#include "rknn_qwen3_5_vl_llm.h"

static int setup_context_lora(rknn3_context ctx,
                              rknn3_session* session,
                              const char* lora_weight_path,
                              rknn3_lora* lora_out)
{
    if (!lora_weight_path || !session || !lora_out) {
        return -1;
    }

    int ret = rknn3_lora_init(ctx, lora_weight_path);
    if (ret < 0) {
        printf("Failed to initialize LLM LoRA: ret=%d\n", ret);
        return ret;
    }

    // RKNN3 has no lora_deinit API. Registration remains owned by ctx and is
    // released by rknn3_destroy; before lora_load succeeds there is no
    // descriptor that can be passed to rknn3_lora_unload.

    uint32_t n_lora = 0;
    ret = rknn3_query(ctx, RKNN3_QUERY_LORA_NUM, &n_lora, sizeof(n_lora));
    if (ret < 0 || n_lora == 0) {
        printf("No LLM LoRA found: ret=%d count=%u\n", ret, n_lora);
        return ret < 0 ? ret : -1;
    }
    if (n_lora > RKNN3_MAX_LORA_NUM) {
        printf("LLM LoRA count exceeds maximum limit: %u > %u\n",
               n_lora, (unsigned)RKNN3_MAX_LORA_NUM);
        return -1;
    }

    rknn3_lora lora_list[RKNN3_MAX_LORA_NUM] = {0};

    ret = rknn3_query(ctx, RKNN3_QUERY_LORA_INFO, lora_list, sizeof(lora_list));
    if (ret < 0) {
        printf("Failed to query LLM LoRA info: ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_lora_load(ctx, &lora_list[0]);
    if (ret < 0) {
        printf("Failed to load LLM LoRA: ret=%d\n", ret);
        return ret;
    }
    ret = rknn3_session_enable_lora(session, &lora_list[0]);
    if (ret < 0) {
        printf("Failed to enable LLM LoRA on session: ret=%d\n", ret);
        rknn3_lora_unload(ctx, &lora_list[0]);
        return ret;
    }
    *lora_out = lora_list[0];
    printf("LLM LoRA loaded: name=%s scale=%.3f\n", lora_out->lora_name, lora_out->scale);
    return 0;
}

int init_qwen3_5_vl_llm(rknn_qwen3_5_vl_llm_context* llm_ctx, const char* model_path, const char* weight_path, rknn3_llm_param* params, int n_params, RKLLMCallback* callback, uint32_t core_mask)
{
    int ret = 0;
    rknn3_context  ctx     = 0;
    rknn3_session* session = NULL;

    if (!llm_ctx || !params || n_params <= 0 || !callback)
    {
        printf("init_qwen3_5_vl_llm: invalid argument\n");
        return -1;
    }

    memset(llm_ctx, 0, sizeof(*llm_ctx));

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
    llm_ctx->params_lora = params[0];
    llm_ctx->callback = callback;
    llm_ctx->lora_loaded = false;
    llm_ctx->lora_enabled = false;

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

int destroy_qwen3_5_vl_llm_base_session(rknn_qwen3_5_vl_llm_context* llm_ctx)
{
    if (!llm_ctx) {
        return -1;
    }
    if (llm_ctx->rknn_sess) {
        rknn3_session_destroy(llm_ctx->rknn_sess);
        llm_ctx->rknn_sess = NULL;
        printf("Base LLM session released\n");
    }
    return 0;
}

int create_qwen3_5_vl_llm_lora_session(rknn_qwen3_5_vl_llm_context* llm_ctx,
                                       const char* lora_weight_path)
{
    if (!llm_ctx || !llm_ctx->rknn_ctx || !llm_ctx->callback) {
        return -1;
    }
    if (llm_ctx->rknn_sess_lora) {
        return 0;
    }

    llm_ctx->rknn_sess_lora = rknn3_session_init(llm_ctx->rknn_ctx,
                                                &llm_ctx->params_lora,
                                                1);
    if (!llm_ctx->rknn_sess_lora) {
        printf("Failed to initialize LoRA LLM session\n");
        return -1;
    }

    int ret = rknn3_session_set_callback(llm_ctx->rknn_sess_lora, llm_ctx->callback);
    if (ret < 0) {
        printf("Failed to set LoRA LLM callback: ret=%d\n", ret);
        rknn3_session_destroy(llm_ctx->rknn_sess_lora);
        llm_ctx->rknn_sess_lora = NULL;
        return ret;
    }

    memset(&llm_ctx->lora, 0, sizeof(llm_ctx->lora));
    llm_ctx->lora_loaded = false;
    llm_ctx->lora_enabled = false;
    if (lora_weight_path) {
        ret = setup_context_lora(llm_ctx->rknn_ctx,
                                 llm_ctx->rknn_sess_lora,
                                 lora_weight_path,
                                 &llm_ctx->lora);
        if (ret < 0) {
            // RKNN3 has no lora_deinit API. If lora_init succeeded but the
            // adapter was not loaded, destroying the owning context below
            // is the SDK-supported cleanup path.
            rknn3_session_destroy(llm_ctx->rknn_sess_lora);
            llm_ctx->rknn_sess_lora = NULL;
            return ret;
        }
        llm_ctx->lora_loaded = true;
        llm_ctx->lora_enabled = true;
    } else {
        printf("No LLM LoRA weight provided; LoRA session runs Base weights\n");
    }
    return 0;
}

int release_qwen3_5_vl_llm(rknn_qwen3_5_vl_llm_context* llm_ctx)
{
    if (!llm_ctx)
    {
        printf("release_qwen3_5_vl_llm: llm_ctx is NULL\n");
        return -1;
    }

    if (llm_ctx->rknn_sess) {
        rknn3_session_destroy(llm_ctx->rknn_sess);
        llm_ctx->rknn_sess = NULL;
    }

    if (llm_ctx->rknn_sess_lora) {
        if (llm_ctx->lora_loaded) {
            if (llm_ctx->lora_enabled) {
                rknn3_session_disable_lora(llm_ctx->rknn_sess_lora, &llm_ctx->lora);
                llm_ctx->lora_enabled = false;
            }
            rknn3_lora_unload(llm_ctx->rknn_ctx, &llm_ctx->lora);
            llm_ctx->lora_loaded = false;
        }
        rknn3_session_destroy(llm_ctx->rknn_sess_lora);
        llm_ctx->rknn_sess_lora = NULL;
    }

    if (llm_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(llm_ctx->rknn_ctx);
        llm_ctx->rknn_ctx = 0;
    }
    llm_ctx->callback = NULL;
    return 0;
}

static int inference_with_session(rknn_qwen3_5_vl_llm_context* llm_ctx,
                                  rknn3_session* session,
                                  rknn3_llm_multimodal_tensor tensor,
                                  int n_inputs,
                                  rknn_perf_metrics_t* perf,
                                  const char* session_name)
{
    if ((!llm_ctx) || !session || n_inputs <= 0)
    {
        printf("llm_ctx, rknn_session or n_inputs is invalid\n");
        return -1;
    }

    if (!perf) {
        printf("inference_qwen3_5_vl_llm perf metrics pointer is NULL\n");
        return -1;
    }
    
    int ret;
    rknn3_llm_input inputs[n_inputs];
    rknn3_llm_infer_param llm_infer_param;

    memset(inputs, 0, sizeof(inputs));
    memset(&(llm_infer_param), 0, sizeof(llm_infer_param));

    llm_infer_param.keep_history = 0;
    llm_infer_param.max_new_tokens = MAX_NEW_TOKENS;


    // Set Input Data
    inputs[0].input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;

    // Run
    printf("rknn_session_run (%s)\n", session_name);
    perf->llm_start_time = getCurrentTimeUs();
    ret = rknn3_session_run(session, inputs, n_inputs, &llm_infer_param);
    perf->llm_end_time = getCurrentTimeUs();
    if (ret < 0)
    {
        printf("rknn_session_run fail! ret=%d\n", ret);
        return ret;
    }
    
    // Query State
    RKLLMRunState state = {0};
    ret = rknn3_session_query_state(session, &state);
    if (ret < 0)
    {
        printf("rknn_session_query_state fail! ret=%d\n", ret);
        return ret;
    }   
    perf->n_decode_tokens = state.n_decode_tokens;
    perf->n_prefill_tokens = state.n_prefill_tokens;

    return ret;
}

int inference_qwen3_5_vl_llm_base(rknn_qwen3_5_vl_llm_context* llm_ctx,
                                  rknn3_llm_multimodal_tensor tensor,
                                  int n_inputs,
                                  rknn_perf_metrics_t* perf)
{
    return inference_with_session(llm_ctx,
                                  llm_ctx ? llm_ctx->rknn_sess : NULL,
                                  tensor, n_inputs, perf, "base");
}

int inference_qwen3_5_vl_llm_lora(rknn_qwen3_5_vl_llm_context* llm_ctx,
                                  rknn3_llm_multimodal_tensor tensor,
                                  int n_inputs,
                                  rknn_perf_metrics_t* perf)
{
    return inference_with_session(llm_ctx,
                                  llm_ctx ? llm_ctx->rknn_sess_lora : NULL,
                                  tensor, n_inputs, perf, "lora");
}

int inference_qwen3_5_vl_llm(rknn_qwen3_5_vl_llm_context* llm_ctx,
                             rknn3_llm_multimodal_tensor tensor,
                             int n_inputs,
                             rknn_perf_metrics_t* perf)
{
    return inference_qwen3_5_vl_llm_base(llm_ctx, tensor, n_inputs, perf);
}
