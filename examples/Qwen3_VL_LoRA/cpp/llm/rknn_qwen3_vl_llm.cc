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

#include "rknn_qwen3_vl_llm.h"

// 多卡/多设备场景下可通过环境变量 RKNN3_DEVICE_ID 指定设备（如 0001:11:00.0）；
// 未设置时沿用 rknn3 默认行为（使用第一个设备）。
static int init_rknn3_context(rknn3_context* ctx)
{
    const char* device_id = getenv("RKNN3_DEVICE_ID");
    if (device_id != nullptr && device_id[0] != '\0') {
        rknn3_init_extend init_extend;
        memset(&init_extend, 0, sizeof(init_extend));
        init_extend.device_id = (char*)device_id;
        printf("Using RKNN3 device: %s\n", device_id);
        return rknn3_init(ctx, &init_extend);
    }
    return rknn3_init(ctx, NULL);
}

// 为 session 设置 chat template 与 callback
static int setup_session(rknn3_session* session, RKLLMCallback& callback)
{
    int ret = rknn3_session_set_chat_template(session, system_prompt, prompt_prefix, prompt_postfix);
    if (ret < 0) {
        printf("Failed to set chat template\n");
        return ret;
    }
    ret = rknn3_session_set_callback(session, &callback);
    if (ret < 0) {
        printf("Failed to set callback\n");
        return ret;
    }
    return 0;
}

// 在 context 级别初始化 LoRA 并为 lora session 启用 LoRA
static int setup_context_lora(rknn3_context ctx, rknn3_session* session_lora, const char* lora_weight_path, rknn3_lora* lora_out)
{
    int ret;
    uint32_t n_lora = 0;
    rknn3_lora lora_list[RKNN3_MAX_LORA_NUM] = {0};

    // 在 context 上初始化 LoRA
    ret = rknn3_lora_init(ctx, lora_weight_path);
    if (ret < 0) {
        printf("Failed to initialize lora on context\n");
        return ret;
    }

    // 使用 rknn3_query 查询 LoRA 数量
    ret = rknn3_query(ctx, RKNN3_QUERY_LORA_NUM, &n_lora, sizeof(n_lora));
    if (ret < 0) {
        printf("Failed to query lora num\n");
        return ret;
    }
    if (n_lora == 0) {
        printf("No lora found in weight file\n");
        return -1;
    }

    // 使用 rknn3_query 查询 LoRA 信息
    ret = rknn3_query(ctx, RKNN3_QUERY_LORA_INFO, lora_list, sizeof(lora_list));
    if (ret < 0) {
        printf("Failed to query lora info\n");
        return ret;
    }

    // 在 context 上加载 LoRA
    ret = rknn3_lora_load(ctx, &lora_list[0]);
    if (ret < 0) {
        printf("Failed to load lora\n");
        return ret;
    }

    // 为 lora session 启用 LoRA
    ret = rknn3_session_enable_lora(session_lora, &lora_list[0]);
    if (ret < 0) {
        printf("Failed to enable lora on session\n");
        return ret;
    }

    *lora_out = lora_list[0];
    return 0;
}

int init_qwen3_vl_llm(rknn_qwen3_vl_llm_context* llm_ctx, const char* model_path, const char* weight_path, rknn3_llm_param* params, int n_params, RKLLMCallback& callback, uint32_t core_mask, int max_context_len1, int max_context_len2, int* deepstack_aligned_size, char* lora_weight_path)
{
    int ret;
    rknn3_context  ctx = 0;
    rknn3_session* session_base = NULL;
    rknn3_session* session_lora = NULL;

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1; // 使用用户管理的internal内存
    rknn3_tensor_attr deepstack_attrs[3];

    // RKNN Init
    ret = init_rknn3_context(&ctx);
    if (ret < 0) {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        return ret;
    }

    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        return ret;
    }

    if (!params || n_params <= 0) {
        printf("invalid llm params\n");
        return -1;
    }

    rknn3_llm_param params_base = params[0];
    rknn3_llm_param params_lora = params[0];
    params_base.max_context_len = max_context_len1;
    params_lora.max_context_len = max_context_len2;

    // Base session: 初始化 -> 设置 chat template 和 callback
    session_base = rknn3_session_init(ctx, &params_base, 1);
    if (!session_base) {
        printf("Failed to initialize base session\n");
        return -1;
    }
    ret = setup_session(session_base, callback);
    if (ret < 0)
        return ret;

    rknn3_lora lora = {0};
    bool lora_enabled = false;
    // LoRA session 不在这里创建：同一 context 同时存在两个 session 会挤占运行期传输缓冲，
    // base 段推理结束后由 create_qwen3_vl_llm_lora_session() 按需创建。
    (void)lora_weight_path;

    // Deepstack aux tensors: base 与 lora 各一套，分别接收 base / lora vision 的输出
    for (int i = 0; i < 3; i++) {
        deepstack_attrs[i].index = 2 + i; //deepstack的index为2、3、 4,可以通过rknn3_query查询所有input_attrs定位到deepstack的index
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(deepstack_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0) {
          printf("rknn_query fail! ret=%d\n", ret);
          return -1;
        }

        llm_ctx->deepstack_tensor[i].mem = rknn3_create_mem(ctx, *deepstack_aligned_size, deepstack_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!llm_ctx->deepstack_tensor[i].mem) {
          printf("fail to create aux_input_tensor.mem!\n");
          return -1;
        }

        llm_ctx->deepstack_tensor_lora[i].mem = rknn3_create_mem(ctx, *deepstack_aligned_size, deepstack_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!llm_ctx->deepstack_tensor_lora[i].mem) {
          printf("fail to create lora aux_input_tensor.mem!\n");
          return -1;
        }
    }

    llm_ctx->rknn_ctx = ctx;
    llm_ctx->rknn_sess_base = session_base;
    llm_ctx->rknn_sess_lora = session_lora;
    llm_ctx->lora = lora;
    llm_ctx->lora_enabled = lora_enabled;
    llm_ctx->lora_deepstack_ready = false;
    llm_ctx->params_lora = params_lora;
    llm_ctx->callback = &callback;
    return 0;
}

// base 段推理结束后释放 base session，保证同一时刻只有一个 session
int destroy_qwen3_vl_llm_base_session(rknn_qwen3_vl_llm_context* llm_ctx)
{
    if (!llm_ctx || !llm_ctx->rknn_sess_base) {
        return 0;
    }
    rknn3_session_destroy(llm_ctx->rknn_sess_base);
    llm_ctx->rknn_sess_base = NULL;
    printf("Base session released\n");
    return 0;
}

// 在 base session 销毁之后再创建带 LoRA 的 session
int create_qwen3_vl_llm_lora_session(rknn_qwen3_vl_llm_context* llm_ctx, const char* lora_weight_path)
{
    if (!llm_ctx) {
        return -1;
    }
    if (llm_ctx->rknn_sess_lora) {
        return 0;
    }

    llm_ctx->rknn_sess_lora = rknn3_session_init(llm_ctx->rknn_ctx, &llm_ctx->params_lora, 1);
    if (!llm_ctx->rknn_sess_lora) {
        printf("Failed to initialize lora session\n");
        return -1;
    }

    rknn3_lora lora = {0};
    int ret = 0;
    if (lora_weight_path == nullptr) {
        // 只跑 vision LoRA 时，第二段 LLM session 不带 LoRA，便于单独对比 vision LoRA 的效果
        printf("No LLM LoRA weight provided, the second session runs without LoRA\n");
        llm_ctx->lora_enabled = false;
    } else {
        ret = setup_context_lora(llm_ctx->rknn_ctx, llm_ctx->rknn_sess_lora, lora_weight_path, &lora);
        if (ret < 0) {
            printf("Warning: LoRA setup failed, lora session will run as base\n");
            llm_ctx->lora_enabled = false;
        } else {
            llm_ctx->lora = lora;
            llm_ctx->lora_enabled = true;
        }
    }

    ret = setup_session(llm_ctx->rknn_sess_lora, *llm_ctx->callback);
    if (ret < 0) {
        return ret;
    }
    return 0;
}

int release_qwen3_vl_llm(rknn_qwen3_vl_llm_context* llm_ctx)
{
    for (int i = 0; i < 3; i++) {
        if (llm_ctx->deepstack_tensor[i].mem) {
            rknn3_destroy_mem(llm_ctx->rknn_ctx, llm_ctx->deepstack_tensor[i].mem);
            llm_ctx->deepstack_tensor[i].mem = nullptr;
        }
        if (llm_ctx->deepstack_tensor_lora[i].mem) {
            rknn3_destroy_mem(llm_ctx->rknn_ctx, llm_ctx->deepstack_tensor_lora[i].mem);
            llm_ctx->deepstack_tensor_lora[i].mem = nullptr;
        }
    }

    if (llm_ctx->rknn_sess_base) {
        rknn3_session_destroy(llm_ctx->rknn_sess_base);
        llm_ctx->rknn_sess_base = NULL;
    }

    if (llm_ctx->rknn_sess_lora) {
        // 禁用 LoRA（如果已启用）
        if (llm_ctx->lora_enabled && strlen(llm_ctx->lora.lora_name) > 0) {
            rknn3_session_disable_lora(llm_ctx->rknn_sess_lora, &llm_ctx->lora);
            rknn3_lora_unload(llm_ctx->rknn_ctx, &llm_ctx->lora);
        }
        rknn3_session_destroy(llm_ctx->rknn_sess_lora);
        llm_ctx->rknn_sess_lora = NULL;
    }

    if (llm_ctx->rknn_ctx != 0) {
        rknn3_destroy(llm_ctx->rknn_ctx);
        llm_ctx->rknn_ctx = 0;
    }
    return 0;
}

static int inference_qwen3_vl_llm_with_session(rknn_qwen3_vl_llm_context* llm_ctx, rknn3_session* session, rknn3_aux_tensor* deepstack, rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf, const char* session_name)
{
    int ret;
    rknn3_llm_input inputs[n_inputs];
    rknn3_llm_infer_param llm_infer_param;

    memset(inputs, 0, sizeof(inputs));
    memset(&llm_infer_param, 0, sizeof(llm_infer_param));
    llm_infer_param.keep_history = 0;
    llm_infer_param.max_new_tokens = MAX_NEW_TOKENS;

    inputs[0].input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;
    for (int i = 0; i < 3; i++) {
        inputs[i + 1].input_type = RKNN3_LLM_INPUT_AUX;
        inputs[i + 1].aux_input  = deepstack[i];
    }

    printf("rknn_session_run (%s)\n", session_name);
    perf->llm_start_time = getCurrentTimeUs();
    ret = rknn3_session_run(session, inputs, n_inputs, &llm_infer_param);
    perf->llm_end_time = getCurrentTimeUs();
    if (ret < 0) {
        printf("rknn_session_run fail! ret=%d\n", ret);
        return ret;
    }

    RKLLMRunState state = {0};
    ret = rknn3_session_query_state(session, &state);
    if (ret < 0) {
        printf("rknn_session_query_state fail! ret=%d\n", ret);
        return ret;
    }
    perf->n_decode_tokens = state.n_decode_tokens;
    perf->n_prefill_tokens = state.n_prefill_tokens;
    return 0;
}

int inference_qwen3_vl_llm_base(rknn_qwen3_vl_llm_context* llm_ctx, rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf)
{
    if (!llm_ctx || !llm_ctx->rknn_sess_base) {
        printf("llm_ctx or rknn_sess_base is NULL\n");
        return -1;
    }
    return inference_qwen3_vl_llm_with_session(llm_ctx, llm_ctx->rknn_sess_base, llm_ctx->deepstack_tensor, tensor, n_inputs, perf, "base");
}

int inference_qwen3_vl_llm_lora(rknn_qwen3_vl_llm_context* llm_ctx, rknn3_llm_multimodal_tensor tensor, int n_inputs, rknn_perf_metrics_t* perf)
{
    if (!llm_ctx || !llm_ctx->rknn_sess_lora) {
        printf("llm_ctx or rknn_sess_lora is NULL\n");
        return -1;
    }
    if (!llm_ctx->lora_enabled) {
        printf("Warning: LoRA is not enabled, running lora session without LoRA\n");
    }
    // 如果跑过带 LoRA 的 vision，则使用 lora vision 的 deepstack，否则复用 base vision 的结果
    rknn3_aux_tensor* deepstack = llm_ctx->lora_deepstack_ready ? llm_ctx->deepstack_tensor_lora : llm_ctx->deepstack_tensor;
    if (!llm_ctx->lora_deepstack_ready) {
        printf("Warning: no vision LoRA deepstack, reusing base vision deepstack\n");
    }
    return inference_qwen3_vl_llm_with_session(llm_ctx, llm_ctx->rknn_sess_lora, deepstack, tensor, n_inputs, perf, "lora");
}
