// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0.

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <string>

#include "Tokenizer.h"
#include "image_utils.h"
#include "time_utils.h"
#include "unlimited_ocr.h"

static int64_t first_token = 0;
static bool first_decode = true;

typedef struct {
    int fd;
    float16* embedding_data;
    int embedding_dim;
    int vocab_size;
} embedding_info_t;

static void printf_usage(const char* prog)
{
    printf("Usage:\n");
    printf("  %s <sam_model_path> <sam_weight_path> <vision_model_path> <vision_weight_path> <llm_model_path> <llm_weight_path> <tokenizer_path> <embedding_path> <sam_core_mask> <vision_core_mask> <llm_core_mask> <image_path> [prompt] [base|gundam] [image_size]\n", prog);
    printf("\nExample:\n");
    printf("  %s ./model/Unlimited-OCR-sam.rknn ./model/Unlimited-OCR-sam.weight ./model/Unlimited-OCR-vision.rknn ./model/Unlimited-OCR-vision.weight ./model/Unlimited-OCR-llm.rknn ./model/Unlimited-OCR-llm.weight ./model/Unlimited-OCR-llm.tokenizer.gguf ./model/Unlimited-OCR-llm.embed.bin 0xff 0xff 0xff ./test.png '<image>document parsing.' base 1024\n", prog);
}

static std::string normalize_prompt(const char* prompt)
{
    std::string text = (prompt && prompt[0] != '\0') ? prompt : "document parsing.";
    if (text.find("<image>") == std::string::npos) {
        text = "<image>" + text;
    }
    return text;
}


static int result_callback(void* userdata, RKLLMResult* result, LLMCallState state)
{
    Tokenizer* tokenizer = (Tokenizer*)userdata;
    if (state == RKLLM_RUN_ERROR) {
        printf("\n\nError occurred during inference\n");
        return 0;
    }
    if (state == RKLLM_RUN_FINISH) {
        printf("\n\n--------------------Finished--------------------\n");
        return 0;
    }
    if (state == RKLLM_RUN_WAITING) {
        printf("\n\nWaiting for UTF-8 encoded character\n");
        return 0;
    }
    if (state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED) {
        printf("\n\n--------------Max new token reached-------------\n");
        return 0;
    }
    if (state == RKLLM_RUN_STOP) {
        printf("\n\n-----------------------Stop---------------------\n");
        return 0;
    }
    if (state == RKLLM_RUN_NORMAL) {
        std::string piece;
        if (result->num_tokens == 1) {
            piece = tokenizer->TokenToPiece(result->token_ids[0]);
        } else {
            piece = tokenizer->Decode(result->token_ids, result->num_tokens);
        }
        printf("%s", piece.c_str());
        if (first_decode) {
            first_token = getCurrentTimeUs();
            first_decode = false;
        }
        fflush(stdout);
    }
    return 0;
}

static int tokenizer_callback(void* userdata, const char* text, int32_t text_len, int32_t* tokens, int32_t n_tokens_max)
{
    Tokenizer* tokenizer = (Tokenizer*)userdata;
    int n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);
    if (n_tokens <= 0) {
        printf("tokenizer failed for %.*s\n", text_len, text);
    }
    return n_tokens;
}

static int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len)
{
    embedding_info_t* embed_info = (embedding_info_t*)userdata;
    if (!embed_info || !embed_info->embedding_data || !embed) return -1;
    if (len != num_tokens * (uint64_t)embed_info->embedding_dim * sizeof(float16)) {
        printf("invalid embed buffer, len=%lu tokens=%lu dim=%d\n", len, num_tokens, embed_info->embedding_dim);
        return -1;
    }
    for (uint64_t n = 0; n < num_tokens; ++n) {
        if (tokens[n] < 0 || tokens[n] >= embed_info->vocab_size) {
            printf("token id out of range: %d\n", tokens[n]);
            return -1;
        }
        memcpy((unsigned char*)embed + n * embed_info->embedding_dim * sizeof(float16),
               embed_info->embedding_data + tokens[n] * embed_info->embedding_dim,
               embed_info->embedding_dim * sizeof(float16));
    }
    return 0;
}

static void printf_perf(const rknn_perf_metrics_t* p)
{
    printf("\n--------------------------------------------------------------------------------------\n");
    printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n",
           "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    printf("--------------------------------------------------------------------------------------\n");

    float ttft_us = first_token > 0 ? (float)(first_token - p->llm_start_time) : 0.0f;
    int prefill_tokens = p->n_prefill_tokens;
    float prefill_ms = ttft_us / 1000.0f;
    float prefill_tpt = prefill_tokens == 0 ? 0.0f : prefill_ms / prefill_tokens;
    float prefill_tps = prefill_ms <= 0.0f ? 0.0f : 1000.0f * prefill_tokens / prefill_ms;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Prefill", prefill_ms, prefill_tokens, prefill_tpt, prefill_tps);

    float decode_us = first_token > 0 ? (float)(p->llm_end_time - first_token) : 0.0f;
    int decode_tokens = p->n_decode_tokens;
    float decode_ms = decode_us / 1000.0f;
    float decode_tpt = decode_tokens == 0 ? 0.0f : decode_ms / decode_tokens;
    float decode_tps = decode_ms <= 0.0f ? 0.0f : 1000.0f * decode_tokens / decode_ms;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Generate", decode_ms, decode_tokens, decode_tpt, decode_tps);
    printf("--------------------------------------------------------------------------------------\n");
    if (p->vision_latency > 0) {
        printf(" Vision latency = %.2f ms, FPS = %.2f\n",
               p->vision_latency / 1000.0f, 1000.0f * 1000.0f / p->vision_latency);
    }
}

int main(int argc, char** argv)
{
    if (argc < 13 || argc > 16) {
        printf_usage(argv[0]);
        return -1;
    }

    const char* sam_model_path = argv[1];
    const char* sam_weight_path = argv[2];
    const char* vision_model_path = argv[3];
    const char* vision_weight_path = argv[4];
    const char* llm_model_path = argv[5];
    const char* llm_weight_path = argv[6];
    const char* tokenizer_path = argv[7];
    const char* embedding_path = argv[8];
    uint32_t sam_core_mask = strtoul(argv[9], NULL, 16);
    uint32_t vision_core_mask = strtoul(argv[10], NULL, 16);
    uint32_t llm_core_mask = strtoul(argv[11], NULL, 16);
    const char* image_path = argv[12];
    std::string prompt = normalize_prompt(argc >= 14 ? argv[13] : NULL);
    const char* mode_str = argc >= 15 ? argv[14] : "base";
    int image_size = argc >= 16 ? atoi(argv[15]) : UNLIMITED_OCR_BASE_IMAGE_SIZE;
    unlimited_ocr_vision_mode_t mode = UNLIMITED_OCR_VISION_BASE;
    if (strcmp(mode_str, "gundam") == 0) {
        printf("gundam mode is not implemented yet in this C++ demo. Use base.\n");
        return -1;
    } else if (strcmp(mode_str, "base") != 0) {
        printf("unsupported mode: %s\n", mode_str);
        return -1;
    }

    int ret = 0;
    rknn_unlimited_ocr_context_t app_ctx;
    memset(&app_ctx, 0, sizeof(app_ctx));
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(src_image));
    rknn_perf_metrics_t perf;
    memset(&perf, 0, sizeof(perf));

    Tokenizer* tokenizer = NULL;
    VocabInfo vocab_info;
    memset(&vocab_info, 0, sizeof(vocab_info));
    embedding_info_t embedding_info;
    memset(&embedding_info, 0, sizeof(embedding_info));
    embedding_info.fd = -1;
    struct stat emb_st;
    memset(&emb_st, 0, sizeof(emb_st));

    float16* image_embeds = NULL;
    size_t image_embeds_bytes = 0;

    printf("--> prepare Unlimited-OCR runtime params\n");
    tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
    if (!tokenizer) {
        printf("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
        ret = -1;
        goto out;
    }
    tokenizer->GetVocabInfo(&vocab_info);
    printf("vocab_info: vocab_size=%d\n", vocab_info.vocab_size);

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
    embedding_info.embedding_data = (float16*)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, embedding_info.fd, 0);
    if (embedding_info.embedding_data == MAP_FAILED) {
        printf("Failed to mmap embedding file\n");
        embedding_info.embedding_data = NULL;
        ret = -1;
        goto out;
    }
    embedding_info.vocab_size = vocab_info.vocab_size;
    embedding_info.embedding_dim = (emb_st.st_size / vocab_info.vocab_size) / sizeof(float16);
    printf("embedding_dim=%d\n", embedding_info.embedding_dim);

    rknn3_llm_param params;
    memset(&params, 0, sizeof(params));
    params.logits_name = (char*)"logits";
    params.max_context_len = UNLIMITED_OCR_MAX_CONTEXT_LEN;
    params.sampling_param = UNLIMITED_OCR_SAMPLE_PARAMS;
    params.vocab_info.vocab_size = vocab_info.vocab_size;
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));
    params.vocab_info.linefeed_id = vocab_info.linefeed_id;

    RKLLMCallback callback;
    memset(&callback, 0, sizeof(callback));
    callback.result_callback = result_callback;
    callback.result_userdata = tokenizer;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = tokenizer;
    callback.embed_callback = embed_callback;
    callback.embed_userdata = &embedding_info;

    ret = init_unlimited_ocr_model(&app_ctx,
                                   llm_model_path,
                                   llm_weight_path,
                                   sam_model_path,
                                   sam_weight_path,
                                   vision_model_path,
                                   vision_weight_path,
                                   &params,
                                   1,
                                   callback,
                                   sam_core_mask,
                                   vision_core_mask,
                                   llm_core_mask,
                                   mode,
                                   image_size,
                                   image_size);
    if (ret < 0) {
        printf("init Unlimited-OCR model failed! ret=%d\n", ret);
        goto out;
    }

    ret = read_image(image_path, &src_image);
    if (ret != 0) {
        printf("read image failed! ret=%d image_path=%s\n", ret, image_path);
        goto out;
    }

    image_embeds_bytes = unlimited_ocr_model_output_bytes(&app_ctx.vision.vision, 0);
    image_embeds = (float16*)malloc(image_embeds_bytes);
    if (!image_embeds) {
        printf("malloc image_embeds failed, bytes=%zu\n", image_embeds_bytes);
        ret = -1;
        goto out;
    }

    rknn3_llm_multimodal_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.name = "input_embeds";
    tensor.prompt = prompt.c_str();
    tensor.image.image_start = "";
    tensor.image.image_end = "";
    tensor.image.image_content = "<image>";
    tensor.image.image_embed = NULL;
    tensor.image.n_image = 0;
    tensor.image.n_image_tokens = 0;
    tensor.image.image_width = 0;
    tensor.image.image_height = 0;
    tensor.enable_thinking = false;

    first_decode = true;
    first_token = 0;
    ret = inference_unlimited_ocr_model(&app_ctx, &src_image, image_embeds, tensor, 1, &perf);
    if (ret < 0) {
        printf("inference Unlimited-OCR model failed! ret=%d\n", ret);
        goto out;
    }
    printf_perf(&perf);

out:
    release_unlimited_ocr_model(&app_ctx);
    if (src_image.virt_addr) free(src_image.virt_addr);
    if (image_embeds) free(image_embeds);
    if (embedding_info.embedding_data) munmap((void*)embedding_info.embedding_data, emb_st.st_size);
    if (embedding_info.fd != -1) close(embedding_info.fd);
    if (tokenizer) delete tokenizer;
    return ret;
}
