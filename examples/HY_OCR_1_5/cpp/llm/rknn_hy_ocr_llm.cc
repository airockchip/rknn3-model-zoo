// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#include "rknn_hy_ocr_llm.h"

const char* system_prompt  = "";
// chat template (见原模型 chat_template.jinja；模型模板变更时必须同步以下常量):
//   有 system message (Python infer_hf_8gpu.py 总是包含):
//     <bos>{system_message}<｜hy_place▁holder▁no▁3｜>{user_content}<｜hy_User｜>
// 模型在 <｜hy_User｜> (120006) 后开始生成, 输出 <｜hy_Assistant｜> (120007) 时停止
// runtime 构造: prompt_prefix + system_prompt + prompt + prompt_postfix
// prompt_prefix 含 <BOS> + <sep>(120021), 匹配 Python 的有 system message 分支
const char* prompt_prefix  = "<｜hy_begin▁of▁sentence｜><｜hy_place▁holder▁no▁3｜>";
const char* prompt_postfix = "<｜hy_User｜>";

static int query_input_tensors(rknn_hy_ocr_llm_context* llm_ctx)
{
    int ret;
    rknn3_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));

    ret = rknn3_query(llm_ctx->rknn_ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN3_SUCCESS) {
        printf("rknn3_query RKNN3_QUERY_IN_OUT_NUM failed! ret=%d\n", ret);
        return ret;
    }

    llm_ctx->input_tensors = (rknn3_tensor*)calloc(io_num.n_input, sizeof(rknn3_tensor));
    if (!llm_ctx->input_tensors) {
        printf("calloc input tensors failed\n");
        return -1;
    }
    llm_ctx->n_input_tensors = (int)io_num.n_input;

    for (uint32_t i = 0; i < io_num.n_input; ++i) {
        llm_ctx->input_tensors[i].attr = (rknn3_tensor_attr*)calloc(1, sizeof(rknn3_tensor_attr));
        if (!llm_ctx->input_tensors[i].attr) {
            printf("calloc input tensor attr failed\n");
            return -1;
        }

        llm_ctx->input_tensors[i].attr->index = i;
        ret = rknn3_query(llm_ctx->rknn_ctx, RKNN3_QUERY_INPUT_ATTR, llm_ctx->input_tensors[i].attr,
                          sizeof(rknn3_tensor_attr));
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_query RKNN3_QUERY_INPUT_ATTR failed! ret=%d\n", ret);
            return ret;
        }
    }

    // Collect indices of ext inputs: xd_rope_id_input
    int n_ext = 0;
    for (int i = 0; i < llm_ctx->n_input_tensors; ++i) {
        const char* name = llm_ctx->input_tensors[i].attr->name;
        if (strcmp(name, "xd_rope_id_input") == 0) {
            n_ext++;
        }
    }

    if (n_ext == 0) {
        printf("no ext input tensors (xd_rope_id_input) found\n");
        return -1;
    }

    llm_ctx->llm_ext_input_indices = (int*)malloc(n_ext * sizeof(int));
    if (!llm_ctx->llm_ext_input_indices) {
        printf("malloc llm_ext_input_indices failed\n");
        return -1;
    }
    llm_ctx->n_llm_ext_inputs = 0;
    for (int i = 0; i < llm_ctx->n_input_tensors; ++i) {
        const char* name = llm_ctx->input_tensors[i].attr->name;
        if (strcmp(name, "xd_rope_id_input") == 0) {
            llm_ctx->llm_ext_input_indices[llm_ctx->n_llm_ext_inputs++] = i;
        }
    }

    return RKNN3_SUCCESS;
}

int init_hy_ocr_llm(rknn_hy_ocr_llm_context* llm_ctx, const char* model_path, const char* weight_path, rknn3_llm_param* params, int n_params, RKLLMCallback* callback, uint32_t core_mask)
{
    int ret;
    rknn3_context  ctx     = 0;
    rknn3_session* session = NULL;

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1; // 使用用户管理的internal内存

    // RKNN Init
    ret = rknn3_init(&ctx, NULL);
    if (ret < 0)
    {
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

    // Set context early for query_input_tensors
    llm_ctx->rknn_ctx = ctx;

    ret = query_input_tensors(llm_ctx);
    if (ret != RKNN3_SUCCESS) {
        printf("query_input_tensors failed! ret=%d\n", ret);
        rknn3_destroy(ctx);
        llm_ctx->rknn_ctx = 0;
        return ret;
    }

    // Set callback ext input indices before session init
    callback->input_tensors_index = llm_ctx->llm_ext_input_indices;
    callback->n_input_tensors     = llm_ctx->n_llm_ext_inputs;

    // RKNN Session Init
    session = rknn3_session_init(ctx, params, n_params);
    if (!session)
    {
        printf("Failed to initialize test session\n");
        rknn3_destroy(ctx);
        llm_ctx->rknn_ctx = 0;
        return -1;
    }

    // Set Chat Template for HunyuanOCR-1.5
    // 格式: <BOS><sep>{content}<User>, 模型在 <User> 后生成, <Assistant> 为 EOS
    ret = rknn3_session_set_chat_template(session, system_prompt, prompt_prefix, prompt_postfix);
    if (ret < 0)
    {
        printf("Failed to set chat template\n");
        rknn3_session_destroy(session);
        rknn3_destroy(ctx);
        return -1;
    }

    // Set Callback
    ret = rknn3_session_set_callback(session, callback);
    if (ret < 0)
    {
        printf("Failed to set callback\n");
        rknn3_session_destroy(session);
        rknn3_destroy(ctx);
        return -1;
    }

    // Set to context
    llm_ctx->rknn_ctx = ctx;
    llm_ctx->rknn_sess = session;

    return ret;
}

int release_hy_ocr_llm(rknn_hy_ocr_llm_context* llm_ctx)
{
    if (!llm_ctx) {
        return 0;
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

    if (llm_ctx->input_tensors) {
        for (int i = 0; i < llm_ctx->n_input_tensors; ++i) {
            free(llm_ctx->input_tensors[i].attr);
            llm_ctx->input_tensors[i].attr = NULL;
        }
        free(llm_ctx->input_tensors);
        llm_ctx->input_tensors = NULL;
    }

    if (llm_ctx->llm_ext_input_indices) {
        free(llm_ctx->llm_ext_input_indices);
        llm_ctx->llm_ext_input_indices = NULL;
    }

    llm_ctx->n_input_tensors  = 0;
    llm_ctx->n_llm_ext_inputs = 0;
    return 0;
}

int inference_hy_ocr_llm(rknn_hy_ocr_llm_context* llm_ctx, rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf)
{
    if ((!llm_ctx) || !(llm_ctx->rknn_sess))
    {
        printf("llm_ctx or rknn_session is NULL");
        return -1;
    }

    if (n_inputs <= 0 || n_inputs > 16)
    {
        printf("invalid n_inputs=%d\n", n_inputs);
        return -1;
    }

    if (!perf) {
        printf("inference_hy_ocr_llm perf metrics pointer is NULL\n");
        return -1;
    }
    
    int ret;
    rknn3_llm_input inputs[16];
    rknn3_llm_infer_param llm_infer_param;

    memset(inputs, 0, sizeof(inputs));
    memset(&(llm_infer_param), 0, sizeof(llm_infer_param));

    llm_infer_param.keep_history = 0;
    llm_infer_param.max_new_tokens = MAX_NEW_TOKENS;

    // Set Input Data
    inputs[0].input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;

    // Run
    printf("rknn_session_run for HunyuanOCR model\n");
    perf->llm_start_time = getCurrentTimeUs();
    ret = rknn3_session_run(llm_ctx->rknn_sess, inputs, n_inputs, &llm_infer_param);
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
