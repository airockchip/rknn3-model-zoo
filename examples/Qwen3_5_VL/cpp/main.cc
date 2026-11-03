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

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <cmath>
#include <memory>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

#include "qwen3_5_vl.h"
#include "image_utils.h"
#include "time_utils.h"

int64_t first_token = 0;
bool first_decode = true;

struct embedding_info
{
    int      fd;
    float16* embedding_data;
    int      embedding_dim;
    int      vocab_size;
};

struct free_deleter
{
    void operator()(void* ptr) const
    {
        free(ptr);
    }
};

const rknn3_sampling_params SAMPLE_PARAMS = {
    .top_k = 1,
    .top_p = 0.9,
    .temperature = 1.0f,
    .repeat_penalty = 1.0f,   
    .frequency_penalty = 0.0f, 
    .presence_penalty = 1.5f
};

// 如有需要可以自行启用system_prompt等参数，配合rknn3_session_set_chat_template()函数使用
// const char* system_prompt  = "";
// const char* prompt_prefix  = "";
// const char* prompt_postfix = "";

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
        fputs(piece.c_str(), stdout);

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
        printf("tokenizer failed for %.*s\n", text_len > 0 ? text_len : 0, text != NULL ? text : "");
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

    for (uint64_t n = 0; n < num_tokens; n++) {
        memcpy((unsigned char*)embed + n * embed_info->embedding_dim * sizeof(float16), embed_info->embedding_data + tokens[n] * embed_info->embedding_dim,
            embed_info->embedding_dim * sizeof(float16));
    }

    return 0;
}


static void print_latency_fps(const char *name, int64_t latency_us)
{
    if (latency_us > 0) {
        double latency_ms = (double)latency_us / 1000.0;
        double fps = 1000000.0 / (double)latency_us;

        printf(" %s latency = %.2f ms, FPS = %.2f\n",
               name, latency_ms, fps);
    } else {
        printf(" %s latency is invalid, skip FPS.\n", name);
    }
}


void printf_perf(rknn_perf_metrics_t *p)
{
    if (p == NULL) {
        printf("Invalid perf metrics pointer.\n");
        return;
    }

    printf("\n--------------------------------------------------------------------------------------\n");
    printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n",
           "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    printf("--------------------------------------------------------------------------------------\n");

    double ttft_us = (double)(first_token - p->llm_start_time);
    double prefill_ms = ttft_us / 1000.0;
    int prefill_n_tokens = p->n_prefill_tokens;

    double prefill_tpt =
        (prefill_n_tokens > 0 && prefill_ms > 0.0)
            ? prefill_ms / (double)prefill_n_tokens
            : 0.0;

    double prefill_tps =
        (prefill_n_tokens > 0 && prefill_ms > 0.0)
            ? (double)prefill_n_tokens * 1000.0 / prefill_ms
            : 0.0;

    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);

    double decode_time_us = (double)(p->llm_end_time - first_token);
    double decode_ms = decode_time_us / 1000.0;
    int decode_n_tokens = p->n_decode_tokens;

    double decode_tpt =
        (decode_n_tokens > 0 && decode_ms > 0.0)
            ? decode_ms / (double)decode_n_tokens
            : 0.0;

    double decode_tps =
        (decode_n_tokens > 0 && decode_ms > 0.0)
            ? (double)decode_n_tokens * 1000.0 / decode_ms
            : 0.0;

    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps);

    printf("--------------------------------------------------------------------------------------\n");

    print_latency_fps("Vision", p->vision_latency);
}

static bool parse_model_dimension(const char* text, const char* name, uint32_t* value)
{
    if (!text || !value || text[0] == '\0' || text[0] == '-') {
        printf("%s must be an integer in [0, UINT32_MAX]\n", name);
        return false;
    }

    char* end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || parsed > UINT32_MAX) {
        printf("%s must be an integer in [0, UINT32_MAX]\n", name);
        return false;
    }

    *value = static_cast<uint32_t>(parsed);
    return true;
}

static bool parse_speedup_ratio(const char* text, float* value)
{
    if (!text || !value || text[0] == '\0') {
        printf("speedup_ratio must be a number in [0, 1]\n");
        return false;
    }

    char* end = NULL;
    errno = 0;
    float parsed = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' ||
        !std::isfinite(parsed) || parsed < 0.0f || parsed > 1.0f) {
        printf("speedup_ratio must be a number in [0, 1]\n");
        return false;
    }

    *value = parsed;
    return true;
}

static void parse_lora_path(const char* value, const char** output)
{
    *output = (value && value[0] != '\0' && strcmp(value, "-") != 0)
        ? value : nullptr;
}

static int parse_legacy_optional_args(int argc, char** argv,
                                      uint32_t* model_width,
                                      uint32_t* model_height,
                                      float* speedup_ratio,
                                      const char** llm_lora_weight_path,
                                      const char** vision_lora_weight_path)
{
    int optional_count = argc - 11;
    if (optional_count != 2 && optional_count != 3 &&
        optional_count != 4 && optional_count != 5) {
        printf("Legacy optional arguments must be: [model_width model_height] "
               "[speedup_ratio] [llm_lora_weight_path] [vision_lora_weight_path]\n");
        return -1;
    }

    if (!parse_model_dimension(argv[11], "model_width", model_width) ||
        !parse_model_dimension(argv[12], "model_height", model_height)) {
        return -1;
    }
    if (optional_count >= 3 &&
        !parse_speedup_ratio(argv[13], speedup_ratio)) {
        return -1;
    }
    if (optional_count >= 4) {
        parse_lora_path(argv[14], llm_lora_weight_path);
    }
    if (optional_count >= 5) {
        parse_lora_path(argv[15], vision_lora_weight_path);
    }
    return 0;
}

static bool is_named_option(const char* arg)
{
    return arg && arg[0] == '-' && arg[1] == '-';
}

static bool option_is(const char* arg, const char* a, const char* b = nullptr,
                      const char* c = nullptr)
{
    return (arg && a && strcmp(arg, a) == 0) ||
           (arg && b && strcmp(arg, b) == 0) ||
           (arg && c && strcmp(arg, c) == 0);
}

static int parse_named_optional_args(int argc, char** argv,
                                     uint32_t* model_width,
                                     uint32_t* model_height,
                                     float* speedup_ratio,
                                     const char** llm_lora_weight_path,
                                     const char** vision_lora_weight_path)
{
    for (int i = 11; i < argc; ++i) {
        const char* option = argv[i];
        if (strcmp(option, "--help") == 0 || strcmp(option, "-h") == 0) {
            return 1;
        }
        if (!is_named_option(option)) {
            printf("Named optional arguments cannot be mixed with legacy positional arguments: %s\n",
                   option);
            return -1;
        }
        if (i + 1 >= argc) {
            printf("Missing value for option %s\n", option);
            return -1;
        }

        const char* value = argv[++i];
        if (option_is(option, "--model-width", "--model_width")) {
            if (!parse_model_dimension(value, "model_width", model_width)) {
                return -1;
            }
        } else if (option_is(option, "--model-height", "--model_height")) {
            if (!parse_model_dimension(value, "model_height", model_height)) {
                return -1;
            }
        } else if (option_is(option, "--speedup-ratio", "--speedup_ratio")) {
            if (!parse_speedup_ratio(value, speedup_ratio)) {
                return -1;
            }
        } else if (option_is(option, "--llm-lora", "--llm_lora_path", "--llm-lora-path")) {
            parse_lora_path(value, llm_lora_weight_path);
        } else if (option_is(option, "--vision-lora", "--vision_lora_path", "--vision-lora-path")) {
            parse_lora_path(value, vision_lora_weight_path);
        } else {
            printf("Unknown option: %s\n", option);
            return -1;
        }
    }
    return 0;
}

static void print_usage(const char* program)
{
    printf("%s <vision_model_path> <vision_weight_path> <llm_model_path> "
           "<llm_weight_path> <tokenizer_path> <embedding_path> "
           "<vision_core_mask> <llm_core_mask> <image_path> <prompt> "
           "[model_width model_height] [speedup_ratio] "
           "[llm_lora_weight_path] [vision_lora_weight_path]\n", program);
    printf("  speedup_ratio: 1.0=Auto, 0.0=Disable, (0,1)=Manual (default: 1.0)\n");
    printf("  LoRA paths are optional; use '-' for the adapter that is not enabled.\n");
    printf("  Named options may be supplied independently, for example:\n");
    printf("    --llm_lora_path <path> --vision_lora_path <path>\n");
}


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc < 11) {
        print_usage(argv[0]);
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
    uint32_t    model_width         = 0;
    uint32_t    model_height        = 0;
    float speedup_ratio = 1.0f;
    const char* llm_lora_weight_path = nullptr;
    const char* vision_lora_weight_path = nullptr;
    if (argc > 11) {
        bool has_named_option = false;
        for (int i = 11; i < argc; ++i) {
            if (is_named_option(argv[i])) {
                has_named_option = true;
                break;
            }
        }
        int parse_ret = has_named_option
            ? parse_named_optional_args(argc, argv, &model_width, &model_height,
                                        &speedup_ratio, &llm_lora_weight_path,
                                        &vision_lora_weight_path)
            : parse_legacy_optional_args(argc, argv, &model_width, &model_height,
                                         &speedup_ratio, &llm_lora_weight_path,
                                         &vision_lora_weight_path);
        if (parse_ret != 0) {
            if (parse_ret > 0) {
                print_usage(argv[0]);
                return 0;
            }
            print_usage(argv[0]);
            return -1;
        }
    }
    std::string prompt_with_image;

    int ret = 0;
    bool lora_requested = (llm_lora_weight_path != nullptr ||
                           vision_lora_weight_path != nullptr);
    rknn_perf_metrics_t perf;
    memset(&perf, 0, sizeof(perf));
    first_token = 0;
    first_decode = true;


    // RKNN Context
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));
    rknn_app_ctx.model_width = model_width;
    rknn_app_ctx.model_height = model_height;

    // Tokenizer
    VocabInfo vocab_info;
    int max_special_ids = sizeof(vocab_info.special_eos_id) / sizeof(vocab_info.special_eos_id[0]);

    // Embedding
    struct embedding_info embedding_info;
    struct stat           emb_st;
    memset(&embedding_info, 0x00, sizeof(embedding_info));
    memset(&emb_st, 0x00, sizeof(emb_st));
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
    std::unique_ptr<float16, free_deleter> img_embeds;
    std::unique_ptr<float16, free_deleter> img_embeds_lora;

    // LLM Multi Model Tensor
    int n_inputs = 1;
    rknn3_llm_multimodal_tensor tensor;
    memset(&tensor, 0, sizeof(rknn3_llm_multimodal_tensor));

    // Callback
    RKLLMCallback callback;
    memset(&callback, 0, sizeof(RKLLMCallback));

    // Load Toenizer
    std::unique_ptr<Tokenizer> tokenizer(new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path));
    if (!tokenizer)
    {
        printf("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
        ret = -1;
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
        ret = -1;
        goto out;
    }

    if (fstat(embedding_info.fd, &emb_st) == -1) {
        printf("Failed to get embedding file size\n");
        ret = -1;
        goto out;
    }

    if (vocab_info.vocab_size <= 0) {
        printf("invalid vocab size\n");
        ret = -1;
        goto out;
    }

    if (emb_st.st_size <= 0 || emb_st.st_size % (sizeof(float16) * (size_t)vocab_info.vocab_size) != 0) {
        printf("embedding file size does not match vocab size\n");
        ret = -1;
        goto out;
    }

    embedding_info.embedding_data = (float16*)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, embedding_info.fd, 0);
    if (embedding_info.embedding_data == MAP_FAILED) {
        printf("Failed to mmap embedding file\n");
        ret = -1;
        goto out;
    }

    embedding_info.vocab_size    = vocab_info.vocab_size;
    embedding_info.embedding_dim = (emb_st.st_size / vocab_info.vocab_size) / sizeof(float16);

    // Set LLM parameters
    params.logits_name               = const_cast<char*>("output");
    params.max_context_len           = MAX_CONTEXT_LEN;
    // params.max_new_tokens            = MAX_NEW_TOKENS;
    params.sampling_param            = SAMPLE_PARAMS;
    params.vocab_info.vocab_size     = vocab_info.vocab_size;
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    if (vocab_info.n_special_eos_id > max_special_ids) {
        printf("n_special_eos_id (%d) exceeds max capacity (%d)\n", vocab_info.n_special_eos_id, max_special_ids);
        ret = -1;
        goto out;
    }
    if (vocab_info.n_special_bos_id > max_special_ids) {
        printf("n_special_bos_id (%d) exceeds max capacity (%d)\n", vocab_info.n_special_bos_id, max_special_ids);
        ret = -1;
        goto out;
    }
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, vocab_info.n_special_eos_id * sizeof(vocab_info.special_eos_id[0]));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, vocab_info.n_special_bos_id * sizeof(vocab_info.special_bos_id[0]));    
    params.vocab_info.linefeed_id    = vocab_info.linefeed_id;

    // LLM Callback
    callback.result_callback    = result_callback;
    callback.result_userdata    = tokenizer.get();
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = tokenizer.get();
    callback.embed_callback     = embed_callback;
    callback.embed_userdata     = &embedding_info;

    ret = init_qwen3_5_vl_model(&rknn_app_ctx, llm_model_path, llm_weight_path, vision_model_path, vision_weight_path, &params, n_params, &callback, vision_core_mask, llm_core_mask, vision_lora_weight_path);
    if (ret != 0)
    {
        printf("init_qwen3_5_vl_model fail! ret=%d llm_model_path=%s vision_model_path=%s\n", ret, llm_model_path, vision_model_path);
        goto out;
    }

    // Image Embed
    for (size_t i = 0; i < rknn_app_ctx.vision.embeds_ndims; i++)
    {
      embed_elems *= rknn_app_ctx.vision.embeds_shape[i];
    }
    img_embeds.reset((float16*)malloc((embed_elems) * sizeof(float16)));
    if (!img_embeds) {
        printf("malloc img_embeds fail!\n");
        ret = -1;
        goto out;
    }
    // A separate output buffer is required only when Vision LoRA is enabled;
    // otherwise the LoRA LLM pass reuses the Base vision embedding.
    if (vision_lora_weight_path != nullptr) {
        img_embeds_lora.reset((float16*)malloc((embed_elems) * sizeof(float16)));
        if (!img_embeds_lora) {
            printf("malloc img_embeds_lora fail!\n");
            ret = -1;
            goto out;
        }
    }

    // Read Image
    ret = read_image(img_path, &src_image);
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, img_path);
        goto out;
    }

    // LLM Input
    tensor.name           = "input_embeds";
    // Add image start tags to the prompt
    prompt_with_image = "<image> " + std::string(prompt);
    tensor.prompt         = (prompt_with_image).c_str();
    tensor.image.image_embed    = img_embeds.get();
    if(rknn_app_ctx.vision.embeds_ndims == 2) {
        tensor.image.n_image_tokens = rknn_app_ctx.vision.embeds_shape[0];
        tensor.image.n_image        = 1;
    } else {
        tensor.image.n_image_tokens = rknn_app_ctx.vision.embeds_shape[1];
        tensor.image.n_image        = rknn_app_ctx.vision.embeds_shape[0];
    }
    tensor.image.image_width    = rknn_app_ctx.vision.model_width;
    tensor.image.image_height   = rknn_app_ctx.vision.model_height;
    tensor.image.image_start      = "<|vision_start|>";
    tensor.image.image_end        = "<|vision_end|>";
    tensor.image.image_content    = "<|image_pad|>";
    tensor.enable_thinking = false;

    if (!lora_requested) {
        ret = inference_qwen3_5_vl_model(&rknn_app_ctx, &src_image,
                                         img_embeds.get(), tensor, n_inputs,
                                         &perf, speedup_ratio);
        if (ret != 0) {
            printf("inference qwen3_5_vl model fail! ret=%d\n", ret);
            goto out;
        }
        printf_perf(&perf);
    } else {
        rknn_perf_metrics_t perf_base;
        rknn_perf_metrics_t perf_lora;
        memset(&perf_base, 0, sizeof(perf_base));
        memset(&perf_lora, 0, sizeof(perf_lora));

        printf("\n========== Base model ==========\n");
        first_decode = true;
        ret = inference_qwen3_5_vl_model_vision(&rknn_app_ctx, &src_image,
                                                img_embeds.get(), &perf_base,
                                                false);
        if (ret != 0) {
            printf("inference qwen3_5_vl vision (base) fail! ret=%d\n", ret);
            goto out;
        }
        tensor.image.image_embed = img_embeds.get();
        ret = inference_qwen3_5_vl_model_llm(&rknn_app_ctx, tensor, n_inputs,
                                             &perf_base, false, speedup_ratio);
        if (ret != 0) {
            printf("inference qwen3_5_vl llm (base) fail! ret=%d\n", ret);
            goto out;
        }
        printf_perf(&perf_base);

        // Vision LoRA is context-level and can be toggled without recreating
        // the Vision session. LLM LoRA uses a new session after Base ends.
        if (vision_lora_weight_path != nullptr) {
            first_decode = true;
            ret = inference_qwen3_5_vl_model_vision(&rknn_app_ctx, &src_image,
                                                    img_embeds_lora.get(),
                                                    &perf_lora, true);
            if (ret != 0) {
                printf("inference qwen3_5_vl vision (lora) fail! ret=%d\n", ret);
                goto out;
            }
        }

        tensor.image.image_embed = vision_lora_weight_path != nullptr
            ? img_embeds_lora.get() : img_embeds.get();
        perf_lora.vision_latency = vision_lora_weight_path != nullptr
            ? perf_lora.vision_latency : perf_base.vision_latency;

        if (llm_lora_weight_path != nullptr) {
            ret = prepare_qwen3_5_vl_lora_session(&rknn_app_ctx,
                                                  llm_lora_weight_path);
            if (ret != 0) {
                printf("prepare qwen3_5_vl LoRA session fail! ret=%d\n", ret);
                goto out;
            }

            printf("\n========== LoRA model (LLM%s) ==========\n",
                   vision_lora_weight_path != nullptr ? "+Vision" : "");
            first_decode = true;
            ret = inference_qwen3_5_vl_model_llm(&rknn_app_ctx, tensor, n_inputs,
                                                 &perf_lora, true, 0.0f);
            if (ret != 0) {
                printf("inference qwen3_5_vl llm (lora) fail! ret=%d\n", ret);
                goto out;
            }
            printf_perf(&perf_lora);
        } else if (vision_lora_weight_path != nullptr) {
            // With only a Vision adapter, keep the Base LLM session and run a
            // second pass using the Vision-LoRA embeddings.
            printf("\n========== LoRA model (Vision only) ==========\n");
            first_decode = true;
            ret = inference_qwen3_5_vl_model_llm(&rknn_app_ctx, tensor, n_inputs,
                                                 &perf_lora, false, 0.0f);
            if (ret != 0) {
                printf("inference qwen3_5_vl llm (vision lora) fail! ret=%d\n", ret);
                goto out;
            }
            printf_perf(&perf_lora);
        }
    }
out:
    int release_ret = release_qwen3_5_vl_model(&rknn_app_ctx);

    if (release_ret != 0)
    {
        printf("release qwen3_5_vl model fail! ret=%d\n", release_ret);
        if (ret == 0) {
            ret = release_ret;
        }
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

    return ret;
}
