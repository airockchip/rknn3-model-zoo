// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rknn_unlimited_ocr_llm.h"
#include "time_utils.h"

const char* unlimited_ocr_system_prompt = "";
const char* unlimited_ocr_prompt_prefix = "";
const char* unlimited_ocr_prompt_postfix = "";

int init_unlimited_ocr_llm(rknn_unlimited_ocr_llm_context* llm_ctx,
                           const char* model_path,
                           const char* weight_path,
                           rknn3_llm_param* params,
                           int n_params,
                           RKLLMCallback& callback,
                           uint32_t core_mask)
{
    if (!llm_ctx || !model_path || !weight_path || !params) {
        printf("invalid llm init arguments\n");
        return -1;
    }

    int ret;
    rknn3_context ctx = 0;
    rknn3_session* session = NULL;

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 0;

    ret = rknn3_init(&ctx, NULL);
    if (ret < 0) {
        printf("rknn3_init llm failed! ret=%d\n", ret);
        return ret;
    }

    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn3_load_model_from_path llm failed! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }

    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn3_model_init llm failed! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }

    session = rknn3_session_init(ctx, params, n_params);
    if (!session) {
        printf("rknn3_session_init llm failed\n");
        rknn3_destroy(ctx);
        return -1;
    }

    ret = rknn3_session_set_chat_template(session, unlimited_ocr_system_prompt,
                                          unlimited_ocr_prompt_prefix,
                                          unlimited_ocr_prompt_postfix);
    if (ret < 0) {
        printf("rknn3_session_set_chat_template failed! ret=%d\n", ret);
        rknn3_session_destroy(session);
        rknn3_destroy(ctx);
        return ret;
    }

    ret = rknn3_session_set_callback(session, &callback);
    if (ret < 0) {
        printf("rknn3_session_set_callback failed! ret=%d\n", ret);
        rknn3_session_destroy(session);
        rknn3_destroy(ctx);
        return ret;
    }

    llm_ctx->rknn_ctx = ctx;
    llm_ctx->rknn_sess = session;
    return 0;
}

int release_unlimited_ocr_llm(rknn_unlimited_ocr_llm_context* llm_ctx)
{
    if (!llm_ctx) {
        return 0;
    }
    if (llm_ctx->rknn_sess) {
        rknn3_session_destroy(llm_ctx->rknn_sess);
        llm_ctx->rknn_sess = NULL;
    }
    if (llm_ctx->rknn_ctx != 0) {
        rknn3_destroy(llm_ctx->rknn_ctx);
        llm_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_unlimited_ocr_llm(rknn_unlimited_ocr_llm_context* llm_ctx,
                                rknn3_llm_multimodal_tensor tensor,
                                int n_inputs,
                                rknn_perf_metrics_t* perf)
{
    if (!llm_ctx || !llm_ctx->rknn_sess || !perf || n_inputs <= 0) {
        printf("invalid llm inference arguments\n");
        return -1;
    }

    int ret;
    rknn3_llm_input inputs[n_inputs];
    rknn3_llm_infer_param infer_param;
    memset(inputs, 0, sizeof(inputs));
    memset(&infer_param, 0, sizeof(infer_param));

    infer_param.keep_history = 0;
    infer_param.max_new_tokens = UNLIMITED_OCR_MAX_NEW_TOKENS;

    inputs[0].input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;

    printf("--> rknn3_session_run llm\n");
    perf->llm_start_time = getCurrentTimeUs();
    ret = rknn3_session_run(llm_ctx->rknn_sess, inputs, n_inputs, &infer_param);
    perf->llm_end_time = getCurrentTimeUs();
    if (ret < 0) {
        printf("rknn3_session_run llm failed! ret=%d\n", ret);
        return ret;
    }

    RKLLMRunState state;
    memset(&state, 0, sizeof(state));
    ret = rknn3_session_query_state(llm_ctx->rknn_sess, &state);
    if (ret < 0) {
        printf("rknn3_session_query_state failed! ret=%d\n", ret);
        return ret;
    }
    perf->n_decode_tokens = state.n_decode_tokens;
    perf->n_prefill_tokens = state.n_prefill_tokens;
    return 0;
}
