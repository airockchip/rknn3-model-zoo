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

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <cmath>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include "qwen3_vl.h"
#include "image_utils.h"


#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include "time_utils.h"

int64_t first_token;
bool first_decode = true;

struct embedding_info
{
  int      fd;
  float16* embedding_data;
  int      embedding_dim;
  int      vocab_size;
};

const rknn3_sampling_params SAMPLE_PARAMS = {
    .top_k = 1,
    .top_p = 0.9,
    .temperature = 0.7f,
    .repeat_penalty = 1.0f,   // Please refer to generation_config.json to configure the corresponding parameters. eg, //huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/main/generation_config.json
    .frequency_penalty = 0.0f,
    .presence_penalty = 0.0f
};

const char* system_prompt  = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
const char* prompt_prefix  = "<|im_start|>user\n";
const char* prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n";
// '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|>*x<|vision_end|>这张图片是关于什么的<|im_end|>\n<|im_start|>assistant\n'
/*-------------------------------------------
                Callback Function
-------------------------------------------*/
int result_callback(void *userdata, RKLLMResult *result, LLMCallState state)
{
    Tokenizer *tokenizer = (Tokenizer *)userdata;

    if (state == RKLLM_RUN_ERROR)
    {
        printf("\n\nError occurred during inference\n");
        return 0;
    }
    else if (state == RKLLM_RUN_FINISH)
    {
        printf("\n\n--------------------Finished-------------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_WAITING)
    {
        printf("\n\nWaiting for UTF-8 encoded character\n");
        return 0;
    }
    else if (state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED)
    {
        printf("\n\n--------------Max new token reached------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_STOP)
    {
        printf("\n\n-----------------------Stop--------------------- \n");
        return 0;
    }
    else if (state == RKLLM_RUN_NORMAL)
    {   
        // Get token text
        std::string piece;
        if (result->num_tokens == 1) {
          piece = tokenizer->TokenToPiece(result->token_ids[0]);
        } else {
          piece = tokenizer->Decode(result->token_ids, result->num_tokens);
        }

        // Print token text
        printf("%s", piece.c_str());

        if (first_decode) {
            first_token = getCurrentTimeUs();
            first_decode = false;
        }
        fflush(stdout);
    }
    return 0;
}


int tokenizer_callback(void *userdata, const char *text, int32_t text_len, int32_t *tokens, int32_t n_tokens_max)
{

    int n_tokens = 0;
    Tokenizer *tokenizer = (Tokenizer *)userdata;
    n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);

    if (n_tokens <= 0)
    {
        printf("tokenizer failed for %s\n", text);
        return n_tokens;
    }

    return n_tokens;
}

int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len)
{

    struct embedding_info* embed_info = (struct embedding_info*)userdata;

    if (len != num_tokens * embed_info->embedding_dim * sizeof(float16)) {
        printf("invalid embed buffer\n");
        return -1;
    }

    for (int n = 0; n < num_tokens; n++) {
        memcpy((unsigned char*)embed + n * embed_info->embedding_dim * sizeof(float16), embed_info->embedding_data + tokens[n] * embed_info->embedding_dim,
            embed_info->embedding_dim * sizeof(float16));
    }

    return 0;
}

void printf_perf(rknn_perf_metrics_t *p) 
{

    printf("\n--------------------------------------------------------------------------------------\n");
    printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n", 
           "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    printf("--------------------------------------------------------------------------------------\n");


    float ttft_us = (float)(first_token - p->llm_start_time);
    int prefill_n_tokens = p->n_prefill_tokens;
    float prefill_ms = ttft_us / 1000.0;
    float prefill_tpt = prefill_n_tokens == 0 ? 0.0f : prefill_ms / prefill_n_tokens;  
    float prefill_tps = prefill_n_tokens == 0 ? 0.0f : 1e3f / prefill_ms * prefill_n_tokens; 
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);

    float decode_time_us = (float)(p->llm_end_time - first_token);
    float decode_ms = decode_time_us / 1000.0;
    int decode_n_tokens = p->n_decode_tokens;
    float decode_tpt = decode_n_tokens == 0 ? 0.0f : decode_ms / decode_n_tokens;
    float decode_tps = decode_n_tokens == 0 ? 0.0f : 1e3f / decode_ms * decode_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps);

    printf("--------------------------------------------------------------------------------------\n");
    
    printf(" Vision latency = %.2f ms, FPS = %.2f\n", 
           (int)p->vision_latency / 1000.f, 1000.f * 1000.f / (int)p->vision_latency);
}


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 11 && argc != 13 && argc != 14)
    {
        printf("%s <vision_model_path> <vision_weight_path> <llm_model_path> <llm_weight_path> <tokenizer_path> <embedding_path> <vision_core_mask> <llm_core_mask> <image_path> <prompt> [model_width model_height] [speedup_ratio]\n", argv[0]);
        printf("  speedup_ratio: 1.0=Auto, 0.0=Disable, (0,1)=Manual (default: 1.0)\n");
        return -1;
    }

    const char *vision_model_path  = argv[1];
    const char *vision_weight_path = argv[2];
    const char *llm_model_path     = argv[3];
    const char *llm_weight_path    = argv[4];
    const char *tokenizer_path     = argv[5];
    const char *embedding_path     = argv[6];
    uint32_t    vision_core_mask   = strtoul(argv[7], nullptr, 16);
    uint32_t    llm_core_mask      = strtoul(argv[8], nullptr, 16);
    const char *img_path           = argv[9];
    const char *prompt             = argv[10];
    uint32_t    model_width        = 0;
    uint32_t    model_height       = 0;
    float       speedup_ratio = 1.0f;
    if (argc >= 13) {
        uint32_t* dimensions[] = {&model_width, &model_height};
        for (int i = 0; i < 2; ++i) {
            char* end = NULL;
            errno = 0;
            unsigned long value = strtoul(argv[11 + i], &end, 0);
            if (errno != 0 || end == argv[11 + i] || *end != '\0' ||
                strchr(argv[11 + i], '-') || value > UINT32_MAX) {
                printf("model_width and model_height must be integers in [0, UINT32_MAX]\n");
                return -1;
            }
            *dimensions[i] = static_cast<uint32_t>(value);
        }
    }
    if (argc == 14) {
        char* end = NULL;
        speedup_ratio = strtof(argv[13], &end);
        if (end == argv[13] || *end != '\0' || !std::isfinite(speedup_ratio) ||
            speedup_ratio < 0.0f || speedup_ratio > 1.0f) {
            printf("speedup_ratio must be a number in [0, 1]\n");
            return -1;
        }
    }
    std::string prompt_with_image;

    int ret = -1;
    rknn_perf_metrics_t perf;


    // RKNN Context
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    rknn_app_ctx.model_width = model_width;
    rknn_app_ctx.model_height = model_height;

    // Tokenizer
    Tokenizer* tokenizer = NULL;
    VocabInfo vocab_info;

    // Embedding
    struct embedding_info embedding_info;
    struct stat           emb_st;
    memset(&embedding_info, 0x00, sizeof(embedding_info));
    embedding_info.fd = -1;

    // LLM Param
    int n_params = 1;
    rknn3_llm_param params;
    memset(&params, 0, sizeof(rknn3_llm_param));

    // Input Image
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));

    // Image Embed
    size_t embed_elems = 1;
    float16* img_embeds = NULL;

    // LLM Multi Model Tensor
    int n_inputs = 4;
    rknn3_llm_multimodal_tensor tensor;
    memset(&tensor, 0, sizeof(rknn3_llm_multimodal_tensor));

    // Callback
    RKLLMCallback callback;
    memset(&callback, 0, sizeof(RKLLMCallback));

    // Load Tokenizer
    tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
    if (!tokenizer)
    {
        printf("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
        goto out;
    }
    
    tokenizer->GetVocabInfo(&(vocab_info));
    printf("vocab_info: vocab_size=%d, special_bos_id=[", vocab_info.vocab_size);
    for (int i = 0; i < vocab_info.n_special_bos_id; ++i)
    {
        printf("%d%s", vocab_info.special_bos_id[i], (i + 1 < vocab_info.n_special_bos_id) ? ", " : "");
    }
    printf("], special_eos_id=[");
    for (int i = 0; i < vocab_info.n_special_eos_id; ++i)
    {
        printf("%d%s", vocab_info.special_eos_id[i], (i + 1 < vocab_info.n_special_eos_id) ? ", " : "");
    }
    printf("]\n");

    // Read Embedding
    embedding_info.fd = open(embedding_path, O_RDONLY);
    if (embedding_info.fd == -1) {
        printf("Failed to open embedding file: %s\n", embedding_path);
        goto out;
    }

    if (fstat(embedding_info.fd, &emb_st) == -1) {
        printf("Failed to get embedding file size\n");
        goto out;
    }

    embedding_info.embedding_data = (float16*)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, embedding_info.fd, 0);
    if (embedding_info.embedding_data == MAP_FAILED) {
        printf("Failed to mmap embedding file\n");
        goto out;
    }

    embedding_info.vocab_size    = vocab_info.vocab_size;
    embedding_info.embedding_dim = (emb_st.st_size / vocab_info.vocab_size) / sizeof(float16);

    // Set LLM parameters
    params.logits_name               = "logits";
    params.max_context_len           = MAX_CONTEXT_LEN;
    // params.max_new_tokens            = MAX_NEW_TOKENS;
    params.sampling_param            = SAMPLE_PARAMS;
    params.vocab_info.vocab_size     = vocab_info.vocab_size;
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));
    params.vocab_info.linefeed_id    = vocab_info.linefeed_id;

    // LLM Callback
    callback.result_callback    = result_callback;
    callback.result_userdata    = tokenizer;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = tokenizer;
    callback.embed_callback     = embed_callback;
    callback.embed_userdata     = &embedding_info;

    ret = init_qwen3_vl_model(&rknn_app_ctx, llm_model_path, llm_weight_path, vision_model_path, vision_weight_path, &params, n_params, callback, vision_core_mask, llm_core_mask);
    if (ret != 0)
    {
        printf("init_qwen3_vl_model fail! ret=%d llm_model_path=%s vision_model_path=%s\n", ret, llm_model_path, vision_model_path);
        goto out;
    }

    // Image Embed
    for (size_t i = 0; i < rknn_app_ctx.vision.embeds_ndims; i++)
    {
      embed_elems *= rknn_app_ctx.vision.embeds_shape[i];
    }
    img_embeds = (float16*)malloc((embed_elems) * sizeof(float16));
    if (img_embeds == NULL) {
        printf("malloc image embeds failed! elems=%zu\n", embed_elems);
        ret = -1;
        goto out;
    }

    // Read Image
    ret = read_image(img_path, &src_image);
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, img_path);
        goto out;
    }

    // LLM Input
    tensor.name = "input_embeds";
    // Add image start tags to the prompt
    prompt_with_image = "<image> " + std::string(prompt);
    tensor.prompt = (prompt_with_image).c_str();
    tensor.image.image_embed = img_embeds;
    if(rknn_app_ctx.vision.embeds_ndims == 2) {
        tensor.image.n_image_tokens = rknn_app_ctx.vision.embeds_shape[0];
        tensor.image.n_image        = 1;
    } else {
        tensor.image.n_image_tokens = rknn_app_ctx.vision.embeds_shape[1];
        tensor.image.n_image        = rknn_app_ctx.vision.embeds_shape[0];
    }
    tensor.image.image_width   = rknn_app_ctx.vision.model_width;
    tensor.image.image_height  = rknn_app_ctx.vision.model_height;
    tensor.image.image_start   = "<|vision_start|>";
    tensor.image.image_end     = "<|vision_end|>";
    tensor.image.image_content = "<|image_pad|>";
    tensor.enable_thinking     = false;

    ret = inference_qwen3_vl_model(&rknn_app_ctx, &src_image, img_embeds, tensor, n_inputs, &perf, speedup_ratio);
    if (ret != 0)
    {
        printf("inference qwen3_vl model fail! ret=%d\n", ret);
        goto out;
    }

    printf_perf(&perf);
out:
    int release_ret = release_qwen3_vl_model(&rknn_app_ctx);

    if (release_ret != 0)
    {
        printf("release qwen3_vl model fail! ret=%d\n", release_ret);
        if (ret == 0) ret = release_ret;
    }

    if (embedding_info.fd != -1) {
        if (embedding_info.embedding_data != MAP_FAILED && embedding_info.embedding_data != NULL) {
            munmap((void*)embedding_info.embedding_data, emb_st.st_size);
            embedding_info.embedding_data = NULL;
        }
        close(embedding_info.fd);
        embedding_info.fd = -1;
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }

    if (tokenizer != NULL)
    {
        delete tokenizer;
        tokenizer = NULL;
    }

    if (img_embeds != NULL)
    {
        free(img_embeds);
    }

    return ret;
}
