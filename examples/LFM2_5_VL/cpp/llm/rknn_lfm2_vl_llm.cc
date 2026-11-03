// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Licensed under the Apache License, Version 2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include "time_utils.h"

#include "rknn_lfm2_vl_llm.h"

// NOTE: This function is intended to be called only during single-threaded
// initialization. The static buffer is safe under that constraint.
static rknn3_init_extend make_rknn3_init_extend(const char* tag)
{
    static char selected_device_id[RKNN3_MAX_DEV_LEN] = {0};
    rknn3_init_extend init_extend;
    memset(&init_extend, 0, sizeof(init_extend));

    const char* env_device_id = getenv("RKNN3_DEVICE_ID");
    if (env_device_id && env_device_id[0] != '\0') {
        snprintf(selected_device_id, sizeof(selected_device_id), "%s", env_device_id);
        init_extend.device_id = selected_device_id;
        printf("[%s] use RKNN3_DEVICE_ID=%s\n", tag, selected_device_id);
        return init_extend;
    }

    rknn3_devices devices;
    memset(&devices, 0, sizeof(devices));
    int ret = rknn3_find_devices(&devices);
    if (ret == 0) {
        if (devices.n_devices > 1) {
            printf("[%s] multiple RKNN3 devices found, using device_id=%s\n",
                   tag, devices.devices[0].id);
            snprintf(selected_device_id, sizeof(selected_device_id), "%s", devices.devices[0].id);
            init_extend.device_id = selected_device_id;
        }
    }
    return init_extend;
}

int init_lfm2_vl_llm(rknn_lfm2_vl_llm_context* llm_ctx,
                     const char* model_path,
                     const char* weight_path,
                     rknn3_llm_param* params,
                     int n_params,
                     RKLLMCallback& callback,
                     uint32_t core_mask)
{
    int ret;
    rknn3_context ctx = 0;
    rknn3_session* session = NULL;

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1;

    // RKNN Init
    rknn3_init_extend init_extend = make_rknn3_init_extend("LLMInit");
    ret = rknn3_init(&ctx, init_extend.device_id ? &init_extend : NULL);
    if (ret < 0) {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }

    // Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        rknn3_destroy(ctx);
        return ret;
    }

    // RKNN Session Init
    session = rknn3_session_init(ctx, params, n_params);
    if (!session) {
        printf("Failed to initialize session\n");
        rknn3_destroy(ctx);
        return -1;
    }

    // Set Chat Template: ChatML wrapper matching the HF chat template
    // (<|im_start|>user\n ... <|im_end|>\n<|im_start|>assistant\n). The
    // MULTIMODAL path passes the assembled prompt (including image tokens)
    // via tensor.prompt, and the runtime wraps it with this template.
    //
    // The template is required, not optional: with an empty template, prompts
    // whose prefill length is below the compiled seq_len hit a runtime issue
    // that reads logits from a padding slot, so the model emits an empty or
    // garbage answer. The ChatML wrapper shifts real tokens to the tail of
    // the prefill window and makes every prompt length work.
    ret = rknn3_session_set_chat_template(session, "", "<|im_start|>user\n",
                                          "<|im_end|>\n<|im_start|>assistant\n");
    if (ret < 0) {
        printf("Failed to set chat template\n");
        rknn3_session_destroy(session);
        rknn3_destroy(ctx);
        return -1;
    }

    // Set Callback
    ret = rknn3_session_set_callback(session, &callback);
    if (ret < 0) {
        printf("Failed to set callback\n");
        rknn3_session_destroy(session);
        rknn3_destroy(ctx);
        return -1;
    }

    // Set to context
    llm_ctx->rknn_ctx = ctx;
    llm_ctx->rknn_sess = session;
    llm_ctx->initialized = true;

    return ret;
}

int release_lfm2_vl_llm(rknn_lfm2_vl_llm_context* llm_ctx)
{
    if (llm_ctx->rknn_sess) {
        rknn3_session_destroy(llm_ctx->rknn_sess);
        llm_ctx->rknn_sess = NULL;
    }

    if (llm_ctx->rknn_ctx != 0) {
        rknn3_destroy(llm_ctx->rknn_ctx);
        llm_ctx->rknn_ctx = 0;
    }
    llm_ctx->initialized = false;
    return 0;
}

int inference_lfm2_vl_llm(rknn_lfm2_vl_llm_context* llm_ctx,
                          rknn3_llm_multimodal_tensor tensor,
                          int n_inputs,
                          rknn_perf_metrics_t* perf)
{
    if ((!llm_ctx) || !(llm_ctx->rknn_sess)) {
        printf("llm_ctx or rknn_session is NULL");
        return -1;
    }

    int ret;
    std::vector<rknn3_llm_input> inputs(n_inputs);
    rknn3_llm_infer_param llm_infer_param;

    memset(inputs.data(), 0, sizeof(rknn3_llm_input) * n_inputs);
    memset(&(llm_infer_param), 0, sizeof(llm_infer_param));

    llm_infer_param.keep_history = 0;
    llm_infer_param.max_new_tokens = LFM2_VL_MAX_NEW_TOKENS;

    // Set Input Data - Multimodal input
    inputs[0].input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;

    // Run
    printf("rknn3_session_run\n");
    perf->llm_start_time = getCurrentTimeUs();
    ret = rknn3_session_run(llm_ctx->rknn_sess, inputs.data(), n_inputs, &llm_infer_param);
    perf->llm_end_time = getCurrentTimeUs();
    if (ret < 0) {
        printf("rknn_session_run fail! ret=%d\n", ret);
        return ret;
    }

    // Query State
    RKLLMRunState state = {0};
    ret = rknn3_session_query_state(llm_ctx->rknn_sess, &state);
    if (ret < 0) {
        printf("rknn_session_query_state fail! ret=%d\n", ret);
        return ret;
    }
    perf->n_decode_tokens = state.n_decode_tokens;
    perf->n_prefill_tokens = state.n_prefill_tokens;

    return ret;
}

int clear_lfm2_vl_llm_cache(rknn_lfm2_vl_llm_context* llm_ctx)
{
    if (!llm_ctx || !llm_ctx->initialized || !llm_ctx->rknn_sess) {
        printf("[ERROR] clear_lfm2_vl_llm_cache: ctx not initialized\n");
        return -1;
    }
    int ret = rknn3_session_clear_kvcache(llm_ctx->rknn_sess, RKNN3_KVCACHE_CLEAR_ALL, nullptr);
    if (ret != 0) {
        printf("[ERROR] rknn3_session_clear_kvcache fail ret=%d\n", ret);
    }
    return ret;
}
