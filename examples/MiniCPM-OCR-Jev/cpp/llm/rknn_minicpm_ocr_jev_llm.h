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

#ifndef RKNN_DEMO_VALUE_HEAD_LLM_UTILS_H
#define RKNN_DEMO_VALUE_HEAD_LLM_UTILS_H

#include <stdint.h>

#include <condition_variable>
#include <mutex>
#include <vector>

#include "rknn3_api.h"
#include "Tokenizer.h"
#include "common.h"

// The value head is exported with a 4096 context so that a long OCR block
// plus two 64-token image blocks still fit. Only prefill runs, so there is no
// generation budget to configure.
#define VALUE_HEAD_MAX_CONTEXT_LEN 4096

// Number of outputs of the score head (binary: 0 / 1).
#define VALUE_HEAD_NUM_LABELS 2

// The score head exposes a single `[1, num_labels]` output; the runtime needs
// this tensor pre-allocated before `rknn3_session_set_callback` so that the
// prefill-finished callback receives a valid memory handle.
#define VALUE_HEAD_MAX_OUTPUT_TENSORS 1

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_session *rknn_sess;
    RKLLMCallback *callback;

    // Owned output tensor plumbing (see prepare_output_tensor()).
    int n_output_tensors;
    int32_t output_tensor_index[VALUE_HEAD_MAX_OUTPUT_TENSORS];
    rknn3_tensor_attr *output_attrs[VALUE_HEAD_MAX_OUTPUT_TENSORS];
    rknn3_tensor_mem *output_mems[VALUE_HEAD_MAX_OUTPUT_TENSORS];
    rknn3_tensor output_tensors[VALUE_HEAD_MAX_OUTPUT_TENSORS];
} rknn_value_head_llm_context;

// Scores collected from the prefill-finished output callback.
typedef struct {
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<float> logits;
    bool ready;
    bool error;
    int expected_n_elems;
    int64_t run_start_us;
    int64_t run_end_us;
    int n_prefill_tokens;
} value_head_score_sink;

int init_value_head_llm(rknn_value_head_llm_context *llm_ctx,
                        const char *model_path,
                        const char *weight_path,
                        rknn3_llm_param *params,
                        int n_params,
                        RKLLMCallback *callback,
                        uint32_t core_mask,
                        int expected_n_elems);

int release_value_head_llm(rknn_value_head_llm_context *llm_ctx);

// LLMOutputCallback implementation: captures `expected_n_elems` float16 values
// once the prefill stage finishes. `userdata` must point to a
// value_head_score_sink.
int value_head_output_callback(void *userdata,
                               rknn3_tensor *output_tensors,
                               uint32_t n_output_tensors,
                               LLMOutputCallbackState state);

// Runs one prefill-only pass and blocks until the score sink is filled or
// `timeout_ms` elapses. Returns 0 on success.
int score_value_head_llm(rknn_value_head_llm_context *llm_ctx,
                         rknn3_llm_multimodal_tensor tensor,
                         value_head_score_sink *sink,
                         int timeout_ms);

#endif // RKNN_DEMO_VALUE_HEAD_LLM_UTILS_H
