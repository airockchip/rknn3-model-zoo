// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Licensed under the Apache License, Version 2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include "time_utils.h"

#include "rknn_lfm2_5_llm.h"

int init_lfm2_llm(rknn_lfm2_llm_context *ctx, const char *model_path, const char *weight_path,
                  rknn3_llm_param *params, int n_params, RKLLMCallback &callback, uint32_t core_mask)
{
    // --- parameter validation ---
    if (!ctx || !model_path || !weight_path || !params || n_params <= 0 || core_mask == 0) {
        printf("[ERROR] init_lfm2_llm: invalid parameters\n");
        return -1;
    }

    int ret = -1;
    rknn3_context rknn_ctx = 0;
    rknn3_session *session = NULL;

    memset(ctx, 0, sizeof(*ctx));

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;

    // 1. RKNN Init
    ret = rknn3_init(&rknn_ctx, NULL);
    if (ret != 0) {
        printf("[ERROR] rknn3_init fail ret=%d (stage: init)\n", ret);
        return ret;
    }

    // 2. Load Model
    ret = rknn3_load_model_from_path(rknn_ctx, model_path, weight_path);
    if (ret != 0) {
        printf("[ERROR] rknn3_load_model_from_path fail ret=%d (stage: load_model) model=%s weight=%s\n",
               ret, model_path, weight_path);
        goto fail_init;
    }

    // 3. Model Init
    ret = rknn3_model_init(rknn_ctx, &config);
    if (ret != 0) {
        printf("[ERROR] rknn3_model_init fail ret=%d (stage: model_init)\n", ret);
        goto fail_init;
    }

    // 4. Session Init
    session = rknn3_session_init(rknn_ctx, params, n_params);
    if (!session) {
        printf("[ERROR] rknn3_session_init failed (stage: session_init)\n");
        ret = -1;
        goto fail_init;
    }

    // 5. Set Callback
    ret = rknn3_session_set_callback(session, &callback);
    if (ret != 0) {
        printf("[ERROR] rknn3_session_set_callback fail ret=%d (stage: set_callback)\n", ret);
        goto fail_session;
    }

    ctx->rknn_ctx = rknn_ctx;
    ctx->rknn_sess = session;
    ctx->initialized = true;
    return 0;

fail_session:
    rknn3_session_destroy(session);
fail_init:
    ctx->rknn_ctx = 0;
    ctx->rknn_sess = NULL;
    ctx->initialized = false;
    rknn3_destroy(rknn_ctx);
    return ret;
}

int inference_lfm2_llm(rknn_lfm2_llm_context *ctx, rknn3_llm_input *inputs, uint32_t n_inputs,
                       uint32_t max_new_tokens, bool keep_history, rknn_perf_metrics_t *perf)
{
    if (!ctx || !ctx->initialized || !ctx->rknn_sess || !inputs || n_inputs == 0 ||
        max_new_tokens == 0 || !perf) {
        printf("[ERROR] inference_lfm2_llm: invalid parameters\n");
        return -1;
    }

    int ret = -1;
    memset(perf, 0, sizeof(*perf));

    rknn3_llm_infer_param infer_param;
    memset(&infer_param, 0, sizeof(infer_param));
    infer_param.keep_history = keep_history ? 1 : 0;
    infer_param.max_new_tokens = max_new_tokens;

    printf("rknn3_session_run\n");
    perf->llm_start_time = getCurrentTimeUs();
    ret = rknn3_session_run(ctx->rknn_sess, inputs, n_inputs, &infer_param);
    perf->llm_end_time = getCurrentTimeUs();
    if (ret != 0) {
        printf("[ERROR] rknn3_session_run fail ret=%d (stage: session_run)\n", ret);
        return ret;
    }

    // Query State (only after successful session_run)
    RKLLMRunState state;
    memset(&state, 0, sizeof(state));
    ret = rknn3_session_query_state(ctx->rknn_sess, &state);
    if (ret != 0) {
        printf("[ERROR] rknn3_session_query_state fail ret=%d (stage: query_state)\n", ret);
        return ret;
    }
    perf->n_decode_tokens = state.n_decode_tokens;
    perf->n_prefill_tokens = state.n_prefill_tokens;

    return 0;
}

int clear_lfm2_cache(rknn_lfm2_llm_context *ctx)
{
    if (!ctx || !ctx->initialized || !ctx->rknn_sess) {
        printf("[ERROR] clear_lfm2_cache: ctx not initialized\n");
        return -1;
    }
    int ret = rknn3_session_clear_kvcache(ctx->rknn_sess, RKNN3_KVCACHE_CLEAR_ALL, nullptr);
    if (ret != 0) {
        printf("[ERROR] rknn3_session_clear_kvcache fail ret=%d (stage: clear_kvcache)\n", ret);
    }
    return ret;
}

int release_lfm2_llm(rknn_lfm2_llm_context *ctx)
{
    if (!ctx)
        return 0;

    if (ctx->rknn_sess) {
        rknn3_session_destroy(ctx->rknn_sess);
        ctx->rknn_sess = NULL;
    }
    if (ctx->rknn_ctx != 0) {
        rknn3_destroy(ctx->rknn_ctx);
        ctx->rknn_ctx = 0;
    }
    ctx->initialized = false;
    return 0;
}
