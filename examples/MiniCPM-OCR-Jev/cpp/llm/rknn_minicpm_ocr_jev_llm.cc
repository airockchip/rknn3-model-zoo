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

#include <chrono>

#include "rknn_minicpm_ocr_jev_llm.h"
#include "time_utils.h"

// Releases everything this module allocated for the pre-allocated output
// tensors: the rknn3 tensor memory first (owned by `ctx`), then the attribute
// blocks (plain heap allocations owned by us). Every slot is reset so the
// function is idempotent and safe to call from both the init failure path and
// release_value_head_llm(). `ctx` may be the context that created the memory;
// it must still be alive when this runs.
static void free_value_head_output_tensors(rknn3_context ctx,
                                           rknn_value_head_llm_context *llm_ctx)
{
    if (!llm_ctx) {
        return;
    }
    for (int i = 0; i < VALUE_HEAD_MAX_OUTPUT_TENSORS; i++) {
        if (llm_ctx->output_mems[i]) {
            if (ctx) {
                rknn3_destroy_mem(ctx, llm_ctx->output_mems[i]);
            }
            llm_ctx->output_mems[i] = NULL;
        }
        if (llm_ctx->output_attrs[i]) {
            free(llm_ctx->output_attrs[i]);
            llm_ctx->output_attrs[i] = NULL;
        }
        // Drop the attr/mem pointers exported to the callback as well, so a
        // caller holding RKLLMCallback cannot dereference freed memory.
        memset(&llm_ctx->output_tensors[i], 0, sizeof(llm_ctx->output_tensors[i]));
    }
    llm_ctx->n_output_tensors = 0;
}

int init_value_head_llm(rknn_value_head_llm_context *llm_ctx,
                        const char *model_path,
                        const char *weight_path,
                        rknn3_llm_param *params,
                        int n_params,
                        RKLLMCallback *callback,
                        uint32_t core_mask,
                        int expected_n_elems)
{
    if (!llm_ctx) {
        printf("init_value_head_llm: llm_ctx is NULL\n");
        return -1;
    }

    memset(llm_ctx, 0, sizeof(*llm_ctx));

    rknn3_context ctx = 0;
    rknn3_session *sess = NULL;
    // 标记 callback 是否已被本函数改写：失败时要把调用方的 callback 复位，
    // 否则它会保留 n_output_tensors=1 但指向已被 zero 的输出数组。
    bool callback_prepared = false;
    int ret = rknn3_init(&ctx, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return ret;
    }

    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        goto fail_ctx;
    }

    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1; // 使用用户管理的 internal 内存
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        goto fail_ctx;
    }

    // 打分模型只有一个 [1, num_labels] 的 logits 输出，需要在
    // rknn3_session_set_callback 之前把输出 tensor 准备好，
    // 否则 prefill 结束时的回调拿不到有效的 mem。
    {
        rknn3_input_output_num io_num;
        memset(&io_num, 0, sizeof(io_num));
        ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        if (ret < 0) {
            printf("rknn_query IN_OUT_NUM failed! ret=%d\n", ret);
            goto fail_ctx;
        }
        printf("llm model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
        if (io_num.n_output < 1) {
            printf("llm model has no output tensor\n");
            ret = -1;
            goto fail_ctx;
        }

        // 当前打分模型只暴露 1 个输出，最大容量仍由 VALUE_HEAD_MAX_OUTPUT_TENSORS 固定
        static_assert(VALUE_HEAD_MAX_OUTPUT_TENSORS >= 1,
                      "value head must reserve at least one output tensor slot");

        const int n_out_tensors = 1;
        if (n_out_tensors > VALUE_HEAD_MAX_OUTPUT_TENSORS) {
            printf("output tensor count %d exceeds VALUE_HEAD_MAX_OUTPUT_TENSORS (%d)\n",
                   n_out_tensors, VALUE_HEAD_MAX_OUTPUT_TENSORS);
            ret = -1;
            goto fail_ctx;
        }

        llm_ctx->n_output_tensors = n_out_tensors;
        for (int i = 0; i < llm_ctx->n_output_tensors; i++) {
            llm_ctx->output_tensor_index[i] = i;
            rknn3_tensor_attr *attr = (rknn3_tensor_attr *)malloc(sizeof(rknn3_tensor_attr));
            if (!attr) {
                printf("malloc output attr failed\n");
                ret = -1;
                goto fail_ctx;
            }
            memset(attr, 0, sizeof(*attr));
            attr->index = llm_ctx->output_tensor_index[i];
            ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, attr, sizeof(rknn3_tensor_attr));
            if (ret < 0) {
                printf("rknn_query OUTPUT_ATTR failed! ret=%d\n", ret);
                free(attr);
                goto fail_ctx;
            }
            llm_ctx->output_attrs[i] = attr;

            rknn3_tensor_mem *mem = rknn3_create_mem(ctx, attr->aligned_size, attr->core_id,
                                                     RKNN3_FLAG_MEMORY_CACHEABLE);
            if (!mem) {
                printf("rknn3_create_mem output %d failed\n", i);
                // attr 已经挂到 llm_ctx->output_attrs[i] 上，fail_ctx 会兜底释放；
                // 这里仍然显式释放并清空槽位，让失败点自身闭合，不再依赖
                // “赋值先于 create_mem” 这一隐含顺序（顺序一旦调整就会泄漏）。
                free(attr);
                llm_ctx->output_attrs[i] = NULL;
                ret = -1;
                goto fail_ctx;
            }
            llm_ctx->output_mems[i] = mem;
            printf("output tensor[%d]: %s, n_elems=%d, expected_n_elems=%d\n",
                   i, attr->name, (int)attr->n_elems, expected_n_elems);
            if (expected_n_elems > 0 && (int)attr->n_elems != expected_n_elems) {
                // 早失败：导出时 num_labels 写错会在这里暴露，而不是返回一个尺寸不对的分数
                printf("output tensor n_elems does not match expected %d\n", expected_n_elems);
                ret = -1;
                goto fail_ctx;
            }
        }
    }

    sess = rknn3_session_init(ctx, params, n_params);
    if (!sess) {
        printf("rknn3_session_init failed!\n");
        ret = -1;
        goto fail_ctx;
    }

    if (callback) {
        callback->output_callback = value_head_output_callback;
        callback->n_output_tensors = (uint32_t)llm_ctx->n_output_tensors;
        callback->output_tensors = llm_ctx->output_tensors;
        callback_prepared = true;
        for (int i = 0; i < llm_ctx->n_output_tensors; i++) {
            llm_ctx->output_tensors[i].attr = llm_ctx->output_attrs[i];
            llm_ctx->output_tensors[i].mem = llm_ctx->output_mems[i];
        }
        ret = rknn3_session_set_callback(sess, callback);
        if (ret < 0) {
            printf("rknn3_session_set_callback failed! ret=%d\n", ret);
            // sess 交给 fail_ctx 统一销毁，避免这里与清理路径重复 destroy。
            goto fail_ctx;
        }
    }

    llm_ctx->rknn_ctx = ctx;
    llm_ctx->rknn_sess = sess;
    llm_ctx->callback = callback;
    return 0;

fail_ctx:
    // 清理顺序与 release_value_head_llm 保持一致：session -> mem -> attr -> ctx。
    if (sess) {
        rknn3_session_destroy(sess);
        sess = NULL;
    }
    free_value_head_output_tensors(ctx, llm_ctx);
    if (callback_prepared) {
        // output_tensors 已被 free_value_head_output_tensors 清零，这里把数量
        // 与指针一并复位，避免调用方误以为还有可用的输出 tensor。
        callback->n_output_tensors = 0;
        callback->output_tensors = NULL;
        callback->output_callback = NULL;
    }
    rknn3_destroy(ctx);
    return ret;
}

int release_value_head_llm(rknn_value_head_llm_context *llm_ctx)
{
    if (!llm_ctx) {
        return -1;
    }
    if (llm_ctx->rknn_sess) {
        rknn3_session_destroy(llm_ctx->rknn_sess);
        llm_ctx->rknn_sess = NULL;
    }
    free_value_head_output_tensors(llm_ctx->rknn_ctx, llm_ctx);
    if (llm_ctx->rknn_ctx != 0) {
        rknn3_destroy(llm_ctx->rknn_ctx);
        llm_ctx->rknn_ctx = 0;
    }
    return 0;
}

int value_head_output_callback(void *userdata,
                               rknn3_tensor *output_tensors,
                               uint32_t n_output_tensors,
                               LLMOutputCallbackState state)
{
    value_head_score_sink *sink = (value_head_score_sink *)userdata;
    if (!sink) {
        return -1;
    }
    if (n_output_tensors != 1) {
        printf("value head expects exactly 1 output tensor, got %u\n", n_output_tensors);
        std::lock_guard<std::mutex> lock(sink->mutex);
        sink->error = true;
        sink->ready = true;
        sink->cv.notify_one();
        return -1;
    }

    // Only the pooled classification row matters, and it is produced at the end
    // of prefill. Decode callbacks (should not happen with prefill_only) are
    // ignored.
    if (state != RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED) {
        return 0;
    }

    int n_elems = (int)output_tensors[0].attr->n_elems;
    if (sink->expected_n_elems > 0 && n_elems != sink->expected_n_elems) {
        printf("value head output has %d elements, expected %d\n", n_elems, sink->expected_n_elems);
        std::lock_guard<std::mutex> lock(sink->mutex);
        sink->error = true;
        sink->ready = true;
        sink->cv.notify_one();
        return -1;
    }

    float16 *p_out = (float16 *)output_tensors[0].mem->virt_addr;
    std::lock_guard<std::mutex> lock(sink->mutex);
    sink->logits.resize(n_elems);
    for (int i = 0; i < n_elems; i++) {
        sink->logits[i] = fp16_to_fp32(p_out[i]);
    }
    sink->ready = true;
    sink->cv.notify_one();
    return 0;
}

int score_value_head_llm(rknn_value_head_llm_context *llm_ctx,
                         rknn3_llm_multimodal_tensor tensor,
                         value_head_score_sink *sink,
                         int timeout_ms)
{
    if (!llm_ctx || !llm_ctx->rknn_sess || !sink) {
        return -1;
    }

    {
        std::lock_guard<std::mutex> lock(sink->mutex);
        sink->logits.clear();
        sink->ready = false;
        sink->error = false;
        sink->n_prefill_tokens = 0;
    }

    rknn3_llm_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    inputs[0].multimodal_input = tensor;

    rknn3_llm_infer_param infer_param;
    memset(&infer_param, 0, sizeof(infer_param));
    infer_param.keep_history = 0;
    // Scoring only needs the pooled logits, so stop right after prefill and
    // skip sampling entirely. Note that setting `disable_sampling` alone has no
    // effect -- `prefill_only` is what actually truncates the run.
    infer_param.prefill_only = true;
    infer_param.disable_sampling = true;
    infer_param.max_new_tokens = 1;

    sink->run_start_us = getCurrentTimeUs();
    int ret = rknn3_session_run(llm_ctx->rknn_sess, inputs, 1, &infer_param);
    sink->run_end_us = getCurrentTimeUs();
    if (ret < 0) {
        printf("rknn3_session_run failed! ret=%d\n", ret);
        return ret;
    }

    RKLLMRunState state;
    memset(&state, 0, sizeof(state));
    if (rknn3_session_query_state(llm_ctx->rknn_sess, &state) == 0) {
        sink->n_prefill_tokens = state.n_prefill_tokens;
    }

    std::unique_lock<std::mutex> lock(sink->mutex);
    if (!sink->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                           [sink] { return sink->ready; })) {
        printf("timed out waiting for the value head output after %d ms\n", timeout_ms);
        return -1;
    }
    if (sink->error) {
        return -1;
    }
    if ((int)sink->logits.size() != VALUE_HEAD_NUM_LABELS) {
        printf("value head returned %zu logits, expected %d\n", sink->logits.size(), VALUE_HEAD_NUM_LABELS);
        return -1;
    }
    return 0;
}
