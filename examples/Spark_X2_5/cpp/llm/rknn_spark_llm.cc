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

#include "rknn_spark_llm.h"

#include "time_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int query_input_tensors(rknn_spark_llm_context* llm_ctx)
{
  int                    ret;
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

  // Collect indices of ext inputs: rope_cos_cache/rope_sin_cache
  // only when rope_cache_host_storage == 1. When it is 0 the model has no
  // rope cache inputs, so skip ext input handling entirely (the runtime
  // will not invoke input_callback with n_input_tensors == 0).
  int  n_ext            = 0;
  bool need_rope_cache  = llm_ctx->llm_config.rope_cache_host_storage != 0;
  if (need_rope_cache) {
    for (int i = 0; i < llm_ctx->n_input_tensors; ++i) {
      const char* name = llm_ctx->input_tensors[i].attr->name;
      if (strstr(name, "rope_cos_cache") || strstr(name, "rope_sin_cache")) {
        n_ext++;
      }
    }

    if (n_ext == 0) {
      printf("no ext input tensors (rope_cos_cache/rope_sin_cache) found\n");
      return -1;
    }

    llm_ctx->llm_ext_input_indices = (int*)malloc(n_ext * sizeof(int));
    if (!llm_ctx->llm_ext_input_indices) {
      printf("malloc llm_ext_input_indices failed\n");
      return -1;
    }
    for (int i = 0; i < llm_ctx->n_input_tensors; ++i) {
      const char* name = llm_ctx->input_tensors[i].attr->name;
      if (strstr(name, "rope_cos_cache") || strstr(name, "rope_sin_cache")) {
        llm_ctx->llm_ext_input_indices[llm_ctx->n_llm_ext_inputs++] = i;
      }
    }
  }

  return RKNN3_SUCCESS;
}

int init_spark_llm(rknn_spark_llm_context* llm_ctx, const char* model_path, const char* weight_path,
                    rknn3_llm_param* params, int n_params, RKLLMCallback* callback, uint32_t core_mask,
                    bool enable_thinking)
{
  int           ret;
  rknn3_devices devs;
  rknn3_config  config;

  if (!llm_ctx || !model_path || !weight_path || !params || !callback || n_params <= 0) {
    return -1;
  }

  memset(llm_ctx, 0, sizeof(*llm_ctx));

  memset(&devs, 0, sizeof(devs));
  ret = rknn3_find_devices(&devs);
  if (ret != RKNN3_SUCCESS || devs.n_devices == 0) {
    printf("rknn3_find_devices fail! ret=%d, n_devices=%d\n", ret, devs.n_devices);
    return -1;
  }

  rknn3_init_extend init_extend;
  memset(&init_extend, 0, sizeof(init_extend));
  init_extend.device_id = devs.devices[0].id;
  printf("device id: %s, type: %s\n",  devs.devices[0].id, devs.devices[0].type);
  ret = rknn3_init(&llm_ctx->rknn_ctx, &init_extend);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_init fail! ret=%d\n", ret);
    return ret;
  }

  ret = rknn3_load_model_from_path(llm_ctx->rknn_ctx, model_path, weight_path);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_load_model_from_path failed! ret=%d\n", ret);
    release_spark_llm(llm_ctx);
    return ret;
  }

  memset(&config, 0, sizeof(config));
  config.run_core_mask = core_mask;
  ret = rknn3_model_init(llm_ctx->rknn_ctx, &config);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_model_init failed! ret=%d\n", ret);
    release_spark_llm(llm_ctx);
    return ret;
  }

  ret = rknn3_query(llm_ctx->rknn_ctx, RKNN3_QUERY_LLM_CONFIG, &llm_ctx->llm_config, sizeof(llm_ctx->llm_config));
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_query RKNN3_QUERY_LLM_CONFIG failed! ret=%d\n", ret);
    release_spark_llm(llm_ctx);
    return ret;
  }

  ret = query_input_tensors(llm_ctx);
  if (ret != RKNN3_SUCCESS) {
    release_spark_llm(llm_ctx);
    return ret;
  }

  if (params[0].max_context_len <= 0) {
    params[0].max_context_len = llm_ctx->llm_config.max_ctx_len;
  }

  llm_ctx->rknn_sess = rknn3_session_init(llm_ctx->rknn_ctx, params, n_params);
  if (!llm_ctx->rknn_sess) {
    printf("Failed to initialize RKNN3 session\n");
    release_spark_llm(llm_ctx);
    return -1;
  }

  callback->input_tensors_index = llm_ctx->llm_ext_input_indices;
  callback->n_input_tensors     = llm_ctx->n_llm_ext_inputs;
  ret = rknn3_session_set_callback(llm_ctx->rknn_sess, callback);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_set_callback failed! ret=%d\n", ret);
    release_spark_llm(llm_ctx);
    return ret;
  }

  // Spark-X2.5 chat template (model/Spark-X2.5-1.7B/chat_template.jinja):
  //   <｜start▁of▁sentence｜><|System|>\n{system}<｜end▁of▁sentence｜>
  //   <｜start▁of▁sentence｜><|User|>{user}<｜end▁of▁sentence｜>
  //   <｜start▁of▁sentence｜><|Bot|>{assistant}...
  // 非思考模式 generation prompt 为 <|Bot|></think>，思考模式为 <|Bot|><think>
  const char* prompt_postfix = enable_thinking
      ? "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|><think>"
      : "<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|></think>";
  ret = rknn3_session_set_chat_template(llm_ctx->rknn_sess,
                                        "<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.<｜end▁of▁sentence｜>",
                                        "<｜start▁of▁sentence｜><|User|>",
                                        prompt_postfix);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_set_chat_template failed! ret=%d\n", ret);
    release_spark_llm(llm_ctx);
    return ret;
  }
  

  return RKNN3_SUCCESS;
}

int release_spark_llm(rknn_spark_llm_context* llm_ctx)
{
  if (!llm_ctx) {
    return 0;
  }

  if (llm_ctx->rknn_sess) {
    rknn3_session_destroy(llm_ctx->rknn_sess);
    llm_ctx->rknn_sess = NULL;
  }

  if (llm_ctx->rknn_ctx) {
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

int inference_spark_llm(rknn_spark_llm_context* llm_ctx, rknn3_llm_multimodal_tensor tensor, int32_t max_new_tokens,
                         rknn_perf_metrics_t* perf)
{
  int ret;

  if (!llm_ctx || !llm_ctx->rknn_sess || !perf) {
    return -1;
  }

  rknn3_llm_infer_param infer_param;
  rknn3_llm_input       input;
  // rknn3_llm_tensor      input_tensor;
  RKLLMRunState         state;
  memset(&infer_param, 0, sizeof(infer_param));
  memset(&input, 0, sizeof(input));
  // memset(&input_tensor, 0, sizeof(input_tensor));
  memset(&state, 0, sizeof(state));

  infer_param.keep_history   = 0;
  infer_param.max_new_tokens = max_new_tokens;

  input.input_type = RKNN3_LLM_INPUT_MULTIMODAL;
  input.multimodal_input = tensor;

  perf->llm_start_time = getCurrentTimeUs();
  ret                  = rknn3_session_run(llm_ctx->rknn_sess, &input, 1, &infer_param);
  perf->llm_end_time   = getCurrentTimeUs();
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_run failed! ret=%d\n", ret);
    return ret;
  }

  ret = rknn3_session_query_state(llm_ctx->rknn_sess, &state);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_query_state failed! ret=%d\n", ret);
    return ret;
  }

  ret = rknn3_session_clear_kvcache(llm_ctx->rknn_sess, RKNN3_KVCACHE_KEEP_SYSTEM_PROMPT, nullptr);
  if (ret != RKNN3_SUCCESS) {
    printf("rknn3_session_clear_kvcache failed! ret=%d\n", ret);
    return ret;
  }

  perf->n_prefill_tokens = (int)state.n_prefill_tokens;
  perf->n_decode_tokens  = (int)state.n_decode_tokens;
  return RKNN3_SUCCESS;
}
