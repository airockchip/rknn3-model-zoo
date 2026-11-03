// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Licensed under the Apache License, Version 2.0

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>

#include "lfm2_vl.h"
#include "image_utils.h"
#include "time_utils.h"

// Fix 1.4: Encapsulate timing state in a struct instead of global variables
struct LlmCallbackContext {
    Tokenizer* tokenizer;
    int64_t first_token_time;
    bool first_decode;
    std::string utf8_buf;  // reassemble partial UTF-8 code points from pieces
};

struct embedding_info
{
    int      fd;
    float16* embedding_data;
    int      embedding_dim;
    int      vocab_size;
};

// LFM2.5-VL uses argmax sampling (top_k=1) for deterministic output
const rknn3_sampling_params SAMPLE_PARAMS = {
    .top_k = 1,
    .top_p = 1.0f,
    .temperature = 1.0f,
    .repeat_penalty = 1.0f,
    .frequency_penalty = 0.0f,
    .presence_penalty = 0.0f
};

/*-------------------------------------------
                Callback Function
-------------------------------------------*/
int result_callback(void *userdata, RKLLMResult *result, LLMCallState state)
{
    LlmCallbackContext *ctx = (LlmCallbackContext *)userdata;
    Tokenizer *tokenizer = ctx->tokenizer;

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
        std::string piece;
        if (result->num_tokens == 1) {
            piece = tokenizer->TokenToPiece(result->token_ids[0]);
        } else {
            piece = tokenizer->Decode(result->token_ids, result->num_tokens);
        }

        // Reassemble multi-byte UTF-8 characters split across token pieces so
        // the streamed output does not contain broken bytes.
        ctx->utf8_buf.append(piece);
        size_t i = 0;
        while (i < ctx->utf8_buf.size()) {
            const unsigned char c = (unsigned char)ctx->utf8_buf[i];
            size_t need = 1;
            if ((c & 0xE0) == 0xC0) need = 2;
            else if ((c & 0xF0) == 0xE0) need = 3;
            else if ((c & 0xF8) == 0xF0) need = 4;
            if (i + need > ctx->utf8_buf.size()) break;
            fwrite(ctx->utf8_buf.data() + i, 1, need, stdout);
            i += need;
        }
        ctx->utf8_buf.erase(0, i);

        if (ctx->first_decode) {
            ctx->first_token_time = getCurrentTimeUs();
            ctx->first_decode = false;
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
        if (tokens[n] < 0 || tokens[n] >= embed_info->vocab_size) {
            printf("embed_callback: token id %d out of range\n", tokens[n]);
            return -1;
        }
        memcpy((unsigned char*)embed + n * embed_info->embedding_dim * sizeof(float16),
               embed_info->embedding_data + tokens[n] * embed_info->embedding_dim,
               embed_info->embedding_dim * sizeof(float16));
    }

    return 0;
}

void printf_perf(rknn_perf_metrics_t *p, LlmCallbackContext* ctx)
{
    printf("\n--------------------------------------------------------------------------------------\n");
    printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n",
           "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    printf("--------------------------------------------------------------------------------------\n");

    float ttft_us = (float)(ctx->first_token_time - p->llm_start_time);
    int prefill_n_tokens = p->n_prefill_tokens;
    float prefill_ms = ttft_us / 1000.0;
    float prefill_tpt = prefill_n_tokens == 0 ? 0.0f : prefill_ms / prefill_n_tokens;
    float prefill_tps = prefill_n_tokens == 0 ? 0.0f : 1e3f / prefill_ms * prefill_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);

    float decode_time_us = (float)(p->llm_end_time - ctx->first_token_time);
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
    if (argc < 11)
    {
        printf("%s <vision_model_path> <vision_weight_path> <llm_model_path> <llm_weight_path> <tokenizer_path> <embedding_path> <vision_core_mask> <llm_core_mask> <image_path> <prompt> [tasks_file]\n", argv[0]);
        printf("  tasks_file (optional): batch mode, one 'image_path\\tprompt' per line; model is loaded once.\n");
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
    std::string prompt_with_image;

    // Batch mode: argv[11] (optional) = tasks file, one "image_path\tprompt" per line.
    // Single process loads the model once and processes all tasks in a loop,
    // avoiding repeated model load/unload which leaks NPU resources and hangs.
    FILE* batch_f = NULL;
    char  batch_line[16384];
    int   n_batch = 0;

    int ret;
    rknn_perf_metrics_t perf;

    // RKNN Context
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

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
    int n_inputs = 1;
    rknn3_llm_multimodal_tensor tensor;
    memset(&tensor, 0, sizeof(rknn3_llm_multimodal_tensor));

    // Callback
    RKLLMCallback callback;
    memset(&callback, 0, sizeof(RKLLMCallback));
    LlmCallbackContext llm_ctx = {};

    // Check files exist
    const char *files[] = {vision_model_path, vision_weight_path, llm_model_path, llm_weight_path, tokenizer_path, embedding_path, img_path};
    for (int i = 0; i < 7; i++) {
        struct stat st;
        if (stat(files[i], &st) != 0) {
            printf("[ERROR] Not found: %s\n", files[i]);
            goto out;
        }
    }

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
    llm_ctx.tokenizer = tokenizer;

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
    params.logits_name               = (char*)"output";
    params.max_context_len           = LFM2_VL_MAX_CONTEXT_LEN;
    params.sampling_param            = SAMPLE_PARAMS;
    params.vocab_info.vocab_size     = vocab_info.vocab_size;
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));
    params.vocab_info.linefeed_id    = vocab_info.linefeed_id;
    params.vocab_info.ignore_eos_token = 0;

    // LLM Callback
    callback.result_callback    = result_callback;
    callback.result_userdata    = &llm_ctx;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = tokenizer;
    callback.embed_callback     = embed_callback;
    callback.embed_userdata     = &embedding_info;

    printf("\n--> init model\n");
    ret = init_lfm2_vl_model(&rknn_app_ctx, llm_model_path, llm_weight_path, vision_model_path, vision_weight_path, &params, n_params, callback, vision_core_mask, llm_core_mask);
    if (ret != 0)
    {
        printf("init_lfm2_vl_model fail! ret=%d\n", ret);
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
        goto out;
    }

    // Batch mode: open tasks file if provided as argv[11]
    if (argc > 11) {
        batch_f = fopen(argv[11], "r");
        if (batch_f) {
            printf("[BATCH] batch mode: tasks file %s\n", argv[11]);
        } else {
            printf("[BATCH] cannot open tasks file %s, single mode\n", argv[11]);
        }
    }

    do {
        // Reset per-task state
        memset(&perf, 0, sizeof(perf));
        memset(&src_image, 0, sizeof(src_image));
        memset(&tensor, 0, sizeof(rknn3_llm_multimodal_tensor));
        llm_ctx.first_token_time = 0;
        llm_ctx.first_decode = true;

        if (batch_f) {
            if (!fgets(batch_line, sizeof(batch_line), batch_f)) break;  // EOF
            char* tab = strchr(batch_line, '\t');
            if (!tab) { printf("[BATCH] skip bad line: %s", batch_line); continue; }
            *tab = 0;
            // trim trailing \n \r from prompt
            char* pe = tab + 1;
            size_t pl = strlen(pe);
            while (pl > 0 && (pe[pl-1] == '\n' || pe[pl-1] == '\r')) { pe[pl-1] = 0; pl--; }
            img_path = batch_line;
            prompt   = pe;
            n_batch++;
            printf("\n[BATCH] task %d: %s\n", n_batch, img_path);
        }

        // Read Image
        ret = read_image(img_path, &src_image);
        if (ret != 0)
        {
            printf("read image fail! ret=%d image_path=%s\n", ret, img_path);
            if (batch_f) continue;
            goto out;
        }

    // LLM Input - Multimodal
    tensor.name = "input_embeds";
    prompt_with_image = "<image>" + std::string(prompt);
    tensor.prompt = (prompt_with_image).c_str();
    // Run vision once; the MULTIMODAL path reuses the result.
    printf("--> inference lfm2_vl vision model\n");
    int64_t vs_us = getCurrentTimeUs();
    ret = inference_lfm2_vl_vision(&(rknn_app_ctx.vision), &src_image, img_embeds);
    perf.vision_latency = getCurrentTimeUs() - vs_us;
    if (ret != 0) {
        printf("vision fail! ret=%d\n", ret);
        if (batch_f) continue;
        goto out;
    }
    tensor.image.image_embed = img_embeds;

    tensor.image.n_image_tokens = rknn_app_ctx.vision.n_valid_embeds;
    tensor.image.n_image        = 1;

    tensor.image.image_width   = rknn_app_ctx.vision.last_grid_w * rknn_app_ctx.vision.encoder_patch_size;
    tensor.image.image_height  = rknn_app_ctx.vision.last_grid_h * rknn_app_ctx.vision.encoder_patch_size;
    tensor.image.image_start   = "<|image_start|>";
    tensor.image.image_end     = "<|image_end|>";
    tensor.image.image_content = "<image>";
    tensor.enable_thinking     = false;

    printf("--> inference\n\n");
    printf("Response:\n");
    llm_ctx.first_token_time = getCurrentTimeUs();
    llm_ctx.first_decode = true;

    // MULTIMODAL path: vision already ran above, call LLM directly
    printf("--> inference lfm2_vl llm model\n");
    ret = inference_lfm2_vl_llm(&(rknn_app_ctx.llm), tensor, n_inputs, &perf);
    if (ret != 0)
    {
        printf("inference lfm2_vl llm model fail! ret=%d\n", ret);
        goto out;
    }

    printf("\n");
    printf_perf(&perf, &llm_ctx);

        // Free per-task image buffer before next iteration
        if (src_image.virt_addr != NULL) {
            free(src_image.virt_addr);
            src_image.virt_addr = NULL;
        }
    } while (batch_f != NULL);

    if (batch_f) { fclose(batch_f); batch_f = NULL; }

out:
    ret = release_lfm2_vl_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release lfm2_vl model fail! ret=%d\n", ret);
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

    printf("done\n");

    return ret;
}
