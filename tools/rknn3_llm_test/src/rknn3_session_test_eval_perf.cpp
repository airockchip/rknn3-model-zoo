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

#include "Tokenizer.h"
#include "float16.h"
#include "rknn3_api.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#define LOGW(fmt, ...) printf("\033[33m" fmt "\033[0m", ##__VA_ARGS__)

rknn3_session *session = nullptr;

struct timeval start, first_token, end;

static bool first_decode = true;

struct embedding_info
{
  int fd;
  float16 *embedding_data;
  int embedding_dim;
  int vocab_size;
};

int argmax(const float16 *data, int size)
{
  if (size <= 0 || !data)
    return -1;

  int max_id = 0;
  for (int i = 1; i < size; i++)
  {
    if (fp16_to_fp32(data[i]) > fp16_to_fp32(data[max_id]))
    {
      max_id = i;
    }
  }
  return max_id;
}

int sampling_callback(void *userdata, float16 *logits, char *logits_name)
{
  struct embedding_info *embed_info = (struct embedding_info *)userdata;
  int size = embed_info->vocab_size;
  int max_id = argmax(logits, size);

  return max_id;
}

int result_callback(void *userdata, RKLLMResult *result, LLMCallState state)
{
  Tokenizer *tokenizer = (Tokenizer *)userdata;

  if (state == RKLLM_RUN_ERROR)
  {
    // printf("\n\nError occurred during inference\n");
    fflush(stdout);
    return 0;
  }
  else if (state == RKLLM_RUN_FINISH)
  {
    // printf("\n\n--------------------Finished-------------------- \n");
    fflush(stdout);
    return 0;
  }
  else if (state == RKLLM_RUN_WAITING)
  {
    // printf("\n\nWaiting for UTF-8 encoded character\n");
    fflush(stdout);
    return 0;
  }
  else if (state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED)
  {
    // printf("\n\n--------------Max new token reached------------- \n");
    fflush(stdout);
    return 0;
  }
  else if (state == RKLLM_RUN_STOP)
  {
    // printf("\n\n-----------------------Stop--------------------- \n");
    fflush(stdout);
    return 0;
  }
  else if (state == RKLLM_RUN_NORMAL)
  {
    // Get token text
    std::string piece;
    if (result->num_tokens == 1)
    {
      piece = tokenizer->TokenToPiece(result->token_ids[0]);
    }
    else
    {
      piece = tokenizer->Decode(result->token_ids, result->num_tokens);
    }

    // Print token text
    // printf("%s", piece.c_str());

    if (first_decode)
    {
      gettimeofday(&first_token, NULL);
      first_decode = false;
    }
    fflush(stdout);
  }
  return 0;
}

int tokenizer_callback(void *userdata, const char *text, int32_t text_len, int32_t *tokens, int32_t n_tokens_max)
{
  Tokenizer *tokenizer = (Tokenizer *)userdata;

  int n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);

  if (n_tokens <= 0)
  {
    printf("tokenizer failed for %s\n", text);
    return n_tokens;
  }

  return n_tokens;
}

int embed_callback(void *userdata, int32_t *tokens, uint64_t num_tokens, void *embed, uint64_t len)
{
  struct embedding_info *embed_info = (struct embedding_info *)userdata;

  if (len != num_tokens * embed_info->embedding_dim * sizeof(float16))
  {
    printf("invalid embed buffer\n");
    return -1;
  }

  for (int n = 0; n < num_tokens; n++)
  {
    memcpy((unsigned char *)embed + n * embed_info->embedding_dim * sizeof(float16),
           embed_info->embedding_data + tokens[n] * embed_info->embedding_dim, embed_info->embedding_dim * sizeof(float16));
  }

  return 0;
}

uint64_t read_tokens_from_bin(const char *binPath, int32_t *tokens)
{
  FILE *fp = fopen(binPath, "rb");
  if (!fp)
  {
    printf("Failed to open file: %s\n", binPath);
    return 0;
  }

  fseek(fp, 0, SEEK_END);
  long fileSize = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  uint64_t n_tokens = fileSize / sizeof(int32_t);
  size_t readSize = fread(tokens, sizeof(int32_t), n_tokens, fp);

  if (readSize != n_tokens)
  {
    printf("Failed to read file, only read %zu elements\n", readSize);
    fclose(fp);
    return 0;
  }

  fclose(fp);
  return n_tokens;
}

rknn3_context init_context_and_model(const char *model_path, const char *weight_path, uint32_t core_mask)
{
  int ret = 0;
  rknn3_config config;
  rknn3_context ctx = 0;

  // init context
  ret = rknn3_init(&ctx, nullptr);
  if (ret < 0)
  {
    printf("rknn3_init fail! ret=%d\n", ret);
    return -1;
  }

  // load model
  ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
  if (ret != RKNN3_SUCCESS)
  {
    printf("rknn3_load_model_from_data failed! ret=%d\n", ret);
    rknn3_destroy(ctx);
    return -1;
  }

  // set config
  memset(&config, 0, sizeof(config));
  config.run_core_mask = core_mask;

  // init model
  ret = rknn3_model_init(ctx, &config);
  if (ret != RKNN3_SUCCESS)
  {
    printf("rknn3_model_init failed! ret=%d\n", ret);
    rknn3_destroy(ctx);
    return -1;
  }

  return ctx;
}

int get_tokenizer_and_embedding(const char *tokenizer_path, VocabInfo *vocab_info, Tokenizer **tokenizer,
                                struct embedding_info *embedding_info, const char *embedding_path, struct stat *emb_st)
{
  *tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
  (*tokenizer)->GetVocabInfo(vocab_info);

  memset(embedding_info, 0x00, sizeof(struct embedding_info));
  embedding_info->fd = open(embedding_path, O_RDONLY);
  if (embedding_info->fd == -1)
  {
    printf("Failed to open embedding file: %s\n", embedding_path);
    delete *tokenizer;
    *tokenizer = nullptr;
    return -1;
  }

  if (fstat(embedding_info->fd, emb_st) == -1)
  {
    printf("Failed to get file size\n");
    close(embedding_info->fd);
    delete *tokenizer;
    *tokenizer = nullptr;
    return -1;
  }

  embedding_info->embedding_data = (float16 *)mmap(NULL, emb_st->st_size, PROT_READ, MAP_PRIVATE, embedding_info->fd, 0);
  if (embedding_info->embedding_data == MAP_FAILED)
  {
    printf("Failed to mmap file\n");
    close(embedding_info->fd);
    delete *tokenizer;
    *tokenizer = nullptr;
    return -1;
  }

  embedding_info->vocab_size = vocab_info->vocab_size;
  embedding_info->embedding_dim = (emb_st->st_size / vocab_info->vocab_size) / sizeof(float16);

  return 0;
}

int main(int argc, char **argv)
{
  if (argc != 9)
  {
    LOGW("Usage: %s <rknn_path> <weight_path> <tokenizer.gguf> <embedding.bin> <max_context_len> <n_input_tokens> <max_new_tokens> <core_mask>\n ", argv[0]);
    LOGW("Such as : %s Qwen2.5-0.5B.rknn Qwen2.5-0.5B.weight Qwen2.5-0.5B.tokenizer.gguf Qwen2.5-0.5B.embed.bin 1024 128 128 0xff\n", argv[0]);
    return -1;
  }

  int ret = 0;
  char *model_path = argv[1];
  char *weight_path = argv[2];
  const char *tokenizer_path = argv[3];
  const char *embedding_path = argv[4];
  int32_t max_context_len = atoi(argv[5]);
  uint64_t n_input_tokens = atoi(argv[6]);
  int32_t max_new_tokens = atoi(argv[7]);
  uint32_t core_mask = strtoul(argv[8], nullptr, 16);

  rknn3_devices devs;
  rknn3_context ctx = 0;
  memset(&devs, 0, sizeof(rknn3_devices));

  VocabInfo vocab_info;
  Tokenizer *tokenizer = nullptr;
  struct stat emb_st;
  struct embedding_info embedding_info;
  RKLLMCallback callback;
  rknn3_llm_param params;
  RKLLMRunState state;
  rknn3_llm_config llm_config;
  memset(&vocab_info, 0, sizeof(VocabInfo));
  memset(&embedding_info, 0, sizeof(struct embedding_info));
  memset(&callback, 0, sizeof(RKLLMCallback));
  memset(&params, 0, sizeof(rknn3_llm_param));
  memset(&state, 0, sizeof(RKLLMRunState));
  memset(&llm_config, 0, sizeof(rknn3_llm_config));

  int test_num = 3;

  printf("*******************************NEW TEST**********************************\n");

  // find devices
  memset(&devs, 0, sizeof(devs));
  ret = rknn3_find_devices(&devs);
  if (ret != RKNN3_SUCCESS || devs.n_devices == 0)
  {
    printf("rknn3_find_devices fail! ret=%d\n", ret);
    return -1;
  }

  // init context and model
  ctx = init_context_and_model(model_path, weight_path, core_mask);
  if (ctx < 0)
  {
    printf("rknn3_session_init_model fail! ctx=%lu\n", ctx);
    goto exit;
  }

  // get_tokenizer_and_embedding
  ret = get_tokenizer_and_embedding(tokenizer_path, &vocab_info, &tokenizer, &embedding_info, embedding_path, &emb_st);
  if (ret < 0)
  {
    printf("get_tokenizer_and_embedding fail! ret=%d\n", ret);
    goto exit;
  }

  ret = rknn3_query(ctx, RKNN3_QUERY_LLM_CONFIG, &llm_config, sizeof(rknn3_llm_config));
  if (ret != RKNN3_SUCCESS)
  {
    printf("rknn3_query llm config failed! ret=%d", ret);
    goto exit;
  }

  // set basic session parameters
  params.logits_name = (char *)"output";
  params.max_context_len = llm_config.max_ctx_len;
  params.sampling_param.temperature = 1.0f;
  params.sampling_param.top_k = 1; // topk=1 sampling
  params.sampling_param.top_p = 0.9f;
  params.sampling_param.repeat_penalty = 1.1f; // repetition penalty
  params.sampling_param.frequency_penalty = 0.0f;
  params.sampling_param.presence_penalty = 0.0f;
  params.vocab_info.vocab_size = vocab_info.vocab_size;
  params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
  params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
  memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
  memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));
  params.vocab_info.linefeed_id = vocab_info.linefeed_id;
  params.vocab_info.ignore_eos_token = 1;

  // init session
  session = rknn3_session_init(ctx, &params, 1);
  if (!session)
  {
    printf("Failed to initialize test session, ret=%d\n", ret);
    goto exit;
  }

  // set callback
  callback.result_callback = result_callback;
  callback.result_userdata = tokenizer;
  callback.embed_callback = embed_callback;
  callback.embed_userdata = &embedding_info;
  callback.tokenizer_callback = tokenizer_callback;
  callback.tokenizer_userdata = tokenizer;
  // callback.sampling_callback  = sampling_callback;
  // callback.sampling_userdata  = &embedding_info;

  // set session callback
  ret = rknn3_session_set_callback(session, &callback);
  if (ret != RKNN3_SUCCESS)
  {
    printf("rknn3_session_set_callback failed, ret=%d\n", ret);
    goto exit;
  }

  if (max_context_len != llm_config.max_ctx_len)
  {
    if (max_context_len < llm_config.max_ctx_len)
    {
      LOGW("Warning: max_context_len (%d) is less than llm_config.max_ctx_len (%d).\n", max_context_len, llm_config.max_ctx_len);
      LOGW("It's recommended to set <max_context_len> to %d.\n", llm_config.max_ctx_len);
    }
    else if (max_context_len > llm_config.max_ctx_len)
    {
      LOGW("Error: max_context_len (%d) is greater than llm_config.max_ctx_len (%d).\n", max_context_len, llm_config.max_ctx_len);
      LOGW("Please set <max_context_len> to %d.\n", llm_config.max_ctx_len);
      goto exit;
    }
  }

  printf("\n");
  printf("=============================================================\n");
  printf("%-32s: %-8d\n", "Max Context Length", llm_config.max_ctx_len);
  printf("%-32s: %-8d\n", "Max Position Embeddings", llm_config.max_position_embeddings);
  printf("%-32s: %s\n", "Model Type", llm_config.model_type);
  printf("%-32s: %s\n", "Task Type", llm_config.task_type == RKNN3_LLM_TASK_GENERATE ? "RKNN3_LLM_TASK_GENERATE" : "RKNN3_LLM_TASK_EMBEDDING");
  printf("%-32s: %-8lu\n", "Number of Input Tokens", n_input_tokens);
  printf("%-32s: %-8d\n", "Max New Tokens", max_new_tokens);
  printf("=============================================================\n\n");

  for (int i = 0; i < test_num; i++)
  {
    int llm_n_inputs = 1;
    rknn3_llm_infer_param llm_infer_param;
    rknn3_llm_input llm_inputs[llm_n_inputs];
    rknn3_llm_input input;
    rknn3_llm_tensor input_tensor;
    memset(&llm_infer_param, 0, sizeof(rknn3_llm_infer_param));
    memset(&llm_inputs, 0, sizeof(rknn3_llm_input) * llm_n_inputs);
    memset(&input, 0, sizeof(rknn3_llm_input));
    memset(&input_tensor, 0, sizeof(rknn3_llm_tensor));

    // Generate fixed tokens for performance testing
    std::vector<int32_t> tokens(n_input_tokens);
    srand(42); // Use fixed seed
    for (uint64_t i = 0; i < n_input_tokens; i++)
    {
      tokens[i] = rand() % 1024;
    }

    llm_infer_param = {.keep_history = 0, .max_new_tokens = max_new_tokens};
    input_tensor = {.name = NULL, .prompt = NULL, .embed = NULL, .tokens = tokens.data(), .n_tokens = n_input_tokens, .enable_thinking = false};
    input.input_type = RKNN3_LLM_INPUT_TOKEN;
    input.llm_input = input_tensor;
    llm_inputs[0] = input;

    gettimeofday(&start, NULL);
    first_decode = true;
    ret = rknn3_session_run(session, llm_inputs, llm_n_inputs, &llm_infer_param);
    if (ret != RKNN3_SUCCESS)
    {
      printf("rknn3_session_run failed, ret=%d\n", ret);
      goto exit;
    }
    gettimeofday(&end, NULL);

    ret = rknn3_session_query_state(session, &state);
    if (ret != RKNN3_SUCCESS)
    {
      printf("rknn3_session_query_state failed, ret=%d\n", ret);
      goto exit;
    }

    ret = rknn3_session_clear_kvcache(session, RKNN3_KVCACHE_KEEP_SYSTEM_PROMPT, nullptr);
    if (ret != RKNN3_SUCCESS)
    {
      printf("rknn3_session_clear_kvcache failed, ret=%d\n", ret);
      goto exit;
    }

    // calculate Prefill Time
    int prefill_n_tokens = state.n_prefill_tokens;
    float prefill_us = (first_token.tv_sec - start.tv_sec) * 1e6f + (first_token.tv_usec - start.tv_usec);
    float prefill_ms = prefill_us / 1e3f;
    float prefill_s = prefill_us / 1e6f;
    float prefill_tpt = prefill_n_tokens == 0 ? 0.0f : prefill_ms / prefill_n_tokens;
    float prefill_tps = prefill_n_tokens == 0 ? 0.0f : prefill_n_tokens / prefill_s;

    // calculate Generate Time
    int decode_n_tokens = state.n_decode_tokens;
    float decode_time_us = ((end.tv_sec - first_token.tv_sec) * 1e6f) + (end.tv_usec - first_token.tv_usec);
    float decode_ms = decode_time_us / 1e3f;
    float decode_s = decode_time_us / 1e6f;
    float decode_tpt = decode_n_tokens == 0 ? 0.0f : decode_ms / decode_n_tokens;
    float decode_tps = decode_n_tokens == 0 ? 0.0f : decode_n_tokens / decode_s;

    printf("\nPerformance Statistics for Run %d: ", i);
    printf("\n-----------------------------------------------------------------------------------------\n");
    printf(" %-10s | %-16s | %-8s | %-20s | %-20s \n", "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    printf("-----------------------------------------------------------------------------------------\n");
    printf(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f \n", "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);
    printf(" %-10s | %-16.2f | %-8d | %-20.2f | %-20.2f \n", "Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps);
    printf("-----------------------------------------------------------------------------------------\n");

    if (i == test_num - 1)
    {
      printf("\n\n====================================================================================================================\n");
      printf("%*s\n", 60 + (int)strlen("Performance Summary") / 2, "Performance Summary");
      printf("====================================================================================================================\n");
      printf("--------------------------------------------------------------------------------------------------------------------\n");
      printf(" %-15s | %-11s | %-13s | %-11s | %-9s | %-10s | %-13s | %-10s \n", "Model Context", "Rope Length", "Input Tokens",
             "New Tokens", "TTFT(ms)", "TPOT(ms)", "Prefill TPS", "Decode TPS");
      printf("--------------------------------------------------------------------------------------------------------------------\n");
      printf(" %-15d | %-11d | %-13d | %-11d | %-9.2f | %-10.2f | %-13.2f | %-10.2f \n", llm_config.max_ctx_len,
             llm_config.max_position_embeddings, n_input_tokens, decode_n_tokens + 1, prefill_ms, decode_tpt, prefill_tps, decode_tps);
      printf("--------------------------------------------------------------------------------------------------------------------\n");
    }

    fflush(stdout);
  }

exit:
  if (session)
  {
    rknn3_session_destroy(session);
  }

  if (ctx)
  {
    rknn3_destroy(ctx);
  }

  if (embedding_info.embedding_data)
  {
    munmap(embedding_info.embedding_data, emb_st.st_size);
  }

  if (embedding_info.fd != -1)
  {
    close(embedding_info.fd);
  }

  if (tokenizer)
  {
    delete tokenizer;
  }

  printf("\n*******************************END TEST**********************************\n");

  fflush(stdout);

  return 0;
}
