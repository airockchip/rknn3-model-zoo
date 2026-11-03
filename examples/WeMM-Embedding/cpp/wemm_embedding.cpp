// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "wemm_embedding.h"

#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <new>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "Tokenizer.h"
#include "common.h"
#include "image_utils.h"
#include "rknn3_api.h"
#include "time_utils.h"

// One loaded RKNN model and its explicitly allocated input/output tensors.
typedef struct {
    rknn3_context context;
    uint32_t core_number;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;
} wemm_rknn_model_t;

// Memory-mapped FP16 token embedding table exported beside the LLM model.
typedef struct {
    int fd;
    size_t size;
    float16* data;
    int vocab_size;
    int embedding_dim;
} wemm_token_embedding_t;

// Private state shared by text, image, and video embedding requests.
typedef struct {
    wemm_rknn_model_t llm;
    wemm_rknn_model_t vision_image;
    wemm_rknn_model_t vision_video;
    rknn3_session* session;
    RKLLMCallback callback;
    Tokenizer* tokenizer;
    wemm_token_embedding_t token_embedding;
    rknn3_tensor output_tensor;
    float output[WEMM_EMBEDDING_DIM];
    int output_ready;
    float16* image_embedding;
    uint64_t image_tokens;
    uint64_t image_embedding_elements;
    wemm_embedding_mode_t mode;
} wemm_embedding_impl_t;

// Validate the public uint8 NHWC interface expected by the shared image utilities.
static int validate_vision_model(const wemm_rknn_model_t* model,
                                 uint32_t input_frames,
                                 const char* mode)
{
    if (model->io_num.n_input != 1 || model->io_num.n_output != 1 ||
        model->inputs[0].attr->layout != RKNN3_TENSOR_NHWC ||
        model->inputs[0].attr->dtype != RKNN3_TENSOR_UINT8 ||
        model->inputs[0].attr->shape[0] != input_frames ||
        model->inputs[0].attr->shape[1] == 0 ||
        model->inputs[0].attr->shape[2] == 0 ||
        model->inputs[0].attr->shape[3] != 3 ||
        model->outputs[0].attr->dtype != RKNN3_TENSOR_FLOAT16 ||
        model->outputs[0].attr->n_elems % WEMM_EMBEDDING_DIM != 0) {
        fprintf(stderr, "Unexpected vision-%s model interface\n", mode);
        return -1;
    }
    return 0;
}

static void release_rknn_model(wemm_rknn_model_t* model)
{
    uint32_t i;
    if (model == NULL) {
        return;
    }
    if (model->inputs != NULL) {
        for (i = 0; i < model->io_num.n_input; ++i) {
            if (model->inputs[i].mem != NULL && model->context != 0) {
                rknn3_destroy_mem(model->context, model->inputs[i].mem);
            }
            free(model->inputs[i].attr);
        }
        free(model->inputs);
    }
    if (model->outputs != NULL) {
        for (i = 0; i < model->io_num.n_output; ++i) {
            if (model->outputs[i].mem != NULL && model->context != 0) {
                rknn3_destroy_mem(model->context, model->outputs[i].mem);
            }
            free(model->outputs[i].attr);
        }
        free(model->outputs);
    }
    if (model->context != 0) {
        rknn3_destroy(model->context);
    }
    memset(model, 0, sizeof(*model));
}

// Build and validate one required deployment-file path.
static int make_model_file_path(char* output,
                                size_t output_size,
                                const char* model_path,
                                const char* file_name)
{
    int count = snprintf(output, output_size, "%s/%s", model_path, file_name);
    if (count < 0 || (size_t)count >= output_size || access(output, R_OK) != 0) {
        fprintf(stderr, "Model file is not readable: %s/%s\n", model_path, file_name);
        return -1;
    }
    return 0;
}

// Load one RKNN/weight pair, select all model cores, and optionally allocate I/O.
static int init_rknn_model(wemm_rknn_model_t* model,
                           const char* rknn_path,
                           const char* weight_path,
                           int allocate_io)
{
    rknn3_config config;
    uint32_t i;
    uint32_t core_mask;
    int ret;

    // Query the model core count instead of assuming a fixed deployment mask.
    memset(model, 0, sizeof(*model));
    ret = rknn3_init(&model->context, NULL);
    if (ret != RKNN3_SUCCESS) {
        fprintf(stderr, "rknn3_init failed: %d\n", ret);
        goto fail;
    }
    ret = rknn3_load_model_from_path(model->context, rknn_path, weight_path);
    if (ret != RKNN3_SUCCESS) {
        fprintf(stderr, "Failed to load model: %s, ret=%d\n", rknn_path, ret);
        goto fail;
    }
    ret = rknn3_query(model->context, RKNN3_QUERY_CORE_NUMBER,
                      &model->core_number, sizeof(model->core_number));
    if (ret != RKNN3_SUCCESS || model->core_number == 0 || model->core_number > 32) {
        fprintf(stderr, "Failed to query model core count: ret=%d, core_number=%u\n",
                ret, model->core_number);
        goto fail;
    }
    core_mask = model->core_number == 32
                    ? UINT32_MAX
                    : (uint32_t)((1ULL << model->core_number) - 1ULL);
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    ret = rknn3_model_init(model->context, &config);
    if (ret != RKNN3_SUCCESS) {
        fprintf(stderr, "rknn3_model_init failed: ret=%d, core_mask=0x%x\n",
                ret, core_mask);
        goto fail;
    }
    ret = rknn3_query(model->context, RKNN3_QUERY_IN_OUT_NUM,
                      &model->io_num, sizeof(model->io_num));
    if (ret != RKNN3_SUCCESS) {
        fprintf(stderr, "Failed to query model input/output count: %d\n", ret);
        goto fail;
    }
    printf("Model: %s, cores: %u, core mask: 0x%x, inputs: %u, outputs: %u\n",
           rknn_path, model->core_number, core_mask,
           model->io_num.n_input, model->io_num.n_output);

    if (!allocate_io) {
        return 0;
    }
    model->inputs = (rknn3_tensor*)calloc(model->io_num.n_input, sizeof(*model->inputs));
    model->outputs = (rknn3_tensor*)calloc(model->io_num.n_output, sizeof(*model->outputs));
    if (model->inputs == NULL || model->outputs == NULL) {
        ret = -1;
        goto fail;
    }
    for (i = 0; i < model->io_num.n_input; ++i) {
        model->inputs[i].attr = (rknn3_tensor_attr*)calloc(1, sizeof(rknn3_tensor_attr));
        if (model->inputs[i].attr == NULL) {
            ret = -1;
            goto fail;
        }
        model->inputs[i].attr->index = i;
        ret = rknn3_query(model->context, RKNN3_QUERY_INPUT_ATTR,
                          model->inputs[i].attr, sizeof(rknn3_tensor_attr));
        if (ret != RKNN3_SUCCESS) {
            goto fail;
        }
        model->inputs[i].mem = rknn3_create_mem(
            model->context, model->inputs[i].attr->aligned_size,
            model->inputs[i].attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (model->inputs[i].mem == NULL) {
            ret = -1;
            goto fail;
        }
    }
    for (i = 0; i < model->io_num.n_output; ++i) {
        model->outputs[i].attr = (rknn3_tensor_attr*)calloc(1, sizeof(rknn3_tensor_attr));
        if (model->outputs[i].attr == NULL) {
            ret = -1;
            goto fail;
        }
        model->outputs[i].attr->index = i;
        ret = rknn3_query(model->context, RKNN3_QUERY_OUTPUT_ATTR,
                          model->outputs[i].attr, sizeof(rknn3_tensor_attr));
        if (ret != RKNN3_SUCCESS) {
            goto fail;
        }
        model->outputs[i].mem = rknn3_create_mem(
            model->context, model->outputs[i].attr->aligned_size,
            model->outputs[i].attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (model->outputs[i].mem == NULL) {
            ret = -1;
            goto fail;
        }
    }
    return 0;

fail:
    release_rknn_model(model);
    return ret == RKNN3_SUCCESS ? -1 : ret;
}

// Bridge RKNN3 text callbacks to the shared model-zoo Tokenizer implementation.
static int tokenizer_callback(void* userdata,
                              const char* text,
                              int32_t text_len,
                              int32_t* tokens,
                              int32_t n_tokens_max)
{
    Tokenizer* tokenizer = (Tokenizer*)userdata;
    int token_count = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);
    const int is_image_tag =
        (text_len == 16 && strncmp(text, "<|vision_start|>", 16) == 0) ||
        (text_len == 14 && strncmp(text, "<|vision_end|>", 14) == 0) ||
        (text_len == 13 && strncmp(text, "<|image_pad|>", 13) == 0) ||
        (text_len == 29 && strncmp(text, "<|vision_start|><|image_pad|>", 29) == 0) ||
        (text_len == 27 && strncmp(text, "<|image_pad|><|vision_end|>", 27) == 0) ||
        (text_len == 43 && strncmp(text, "<|vision_start|><|image_pad|><|vision_end|>", 43) == 0);
    const int is_video_tag =
        (text_len == 13 && strncmp(text, "<|video_pad|>", 13) == 0) ||
        (text_len == 29 && strncmp(text, "<|vision_start|><|video_pad|>", 29) == 0) ||
        (text_len == 27 && strncmp(text, "<|video_pad|><|vision_end|>", 27) == 0) ||
        (text_len == 43 && strncmp(text, "<|vision_start|><|video_pad|><|vision_end|>", 43) == 0);
    const int is_vision_tag = is_image_tag || is_video_tag;
    // The GGUF intentionally appends <embedding> to a full prompt. RKNN3 also
    // tokenizes these tags independently while expanding multimodal input; they must
    // remain exactly one token or multimodal replacement cannot locate them.
    if (is_vision_tag && token_count > 1) {
        --token_count;
    }
    // Multimodal expansion renders the embedding template before invoking the
    // tokenizer. The GGUF post-processor appends the same EOS once more.
    if (!is_vision_tag && token_count > 1 &&
        tokens[token_count - 1] == tokens[token_count - 2]) {
        --token_count;
    }
    if (token_count <= 0) {
        fprintf(stderr, "Tokenizer failed\n");
    }
    return token_count;
}

// Supply token embeddings requested by the RKNN3 LLM runtime.
static int embed_callback(void* userdata,
                          int32_t* tokens,
                          uint64_t num_tokens,
                          void* embed,
                          uint64_t len)
{
    // Gather FP16 token embeddings from the memory-mapped vocabulary table.
    wemm_token_embedding_t* table = (wemm_token_embedding_t*)userdata;
    uint64_t expected = num_tokens * (uint64_t)table->embedding_dim * sizeof(float16);
    uint64_t i;
    if (len != expected) {
        fprintf(stderr, "Token embedding buffer size mismatch: %lu != %lu\n",
                (unsigned long)len, (unsigned long)expected);
        return -1;
    }
    for (i = 0; i < num_tokens; ++i) {
        if (tokens[i] < 0 || tokens[i] >= table->vocab_size) {
            fprintf(stderr, "Token ID out of range: %d\n", tokens[i]);
            return -1;
        }
        memcpy((unsigned char*)embed + i * table->embedding_dim * sizeof(float16),
               table->data + (uint64_t)tokens[i] * table->embedding_dim,
               table->embedding_dim * sizeof(float16));
    }
    return 0;
}

// Capture the model's FP16 prefill output as a host float32 embedding.
static int output_callback(void* userdata,
                           rknn3_tensor* output_tensors,
                           uint32_t n_output_tensors,
                           LLMOutputCallbackState state)
{
    // The embedding model returns its result when prefill finishes; no decode is needed.
    wemm_embedding_impl_t* impl = (wemm_embedding_impl_t*)userdata;
    uint32_t i;
    if (state != RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED || n_output_tensors != 1) {
        return 0;
    }
    if (output_tensors[0].attr->n_elems != WEMM_EMBEDDING_DIM ||
        output_tensors[0].attr->dtype != RKNN3_TENSOR_FLOAT16) {
        fprintf(stderr, "Unexpected embedding output interface\n");
        return -1;
    }
    for (i = 0; i < WEMM_EMBEDDING_DIM; ++i) {
        impl->output[i] = fp16_to_fp32(((float16*)output_tensors[0].mem->virt_addr)[i]);
    }
    impl->output_ready = 1;
    return 0;
}

// Propagate asynchronous LLM runtime failures to the session runner.
static int result_callback(void* userdata, RKLLMResult* result, LLMCallState state)
{
    (void)userdata;
    (void)result;
    return state == RKLLM_RUN_ERROR ? -1 : 0;
}

// Initialize the tokenizer and map the exported token embedding table read-only.
static int init_tokenizer_and_embedding(wemm_embedding_impl_t* impl,
                                        const char* tokenizer_path,
                                        const char* embedding_path)
{
    struct stat file_stat;
    VocabInfo vocab_info;
    size_t embedding_dim;
    memset(&vocab_info, 0, sizeof(vocab_info));
    impl->tokenizer = new (std::nothrow) Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
    if (impl->tokenizer == NULL) {
        fprintf(stderr, "Failed to load tokenizer: %s\n", tokenizer_path);
        return -1;
    }
    impl->tokenizer->GetVocabInfo(&vocab_info);
    if (vocab_info.vocab_size <= 0) {
        fprintf(stderr, "Invalid tokenizer vocabulary size: %d\n", vocab_info.vocab_size);
        goto fail;
    }
    // mmap avoids copying the approximately 1 GB token embedding table into heap memory.
    impl->token_embedding.fd = open(embedding_path, O_RDONLY);
    if (impl->token_embedding.fd < 0) {
        fprintf(stderr, "Failed to open token embedding: %s\n", embedding_path);
        goto fail;
    }
    if (fstat(impl->token_embedding.fd, &file_stat) != 0) {
        fprintf(stderr, "Failed to stat token embedding: %s\n", embedding_path);
        goto fail;
    }
    impl->token_embedding.size = (size_t)file_stat.st_size;
    impl->token_embedding.data = (float16*)mmap(
        NULL, impl->token_embedding.size, PROT_READ, MAP_PRIVATE,
        impl->token_embedding.fd, 0);
    if (impl->token_embedding.data == MAP_FAILED) {
        impl->token_embedding.data = NULL;
        fprintf(stderr, "Failed to mmap token embedding: %s\n", embedding_path);
        goto fail;
    }
    if (impl->token_embedding.size %
            ((size_t)vocab_info.vocab_size * sizeof(float16)) != 0) {
        fprintf(stderr, "Token embedding file size does not match vocabulary\n");
        goto fail;
    }
    impl->token_embedding.vocab_size = vocab_info.vocab_size;
    embedding_dim = impl->token_embedding.size /
                    (size_t)vocab_info.vocab_size / sizeof(float16);
    if (embedding_dim != WEMM_EMBEDDING_DIM || embedding_dim > INT_MAX) {
        fprintf(stderr, "Unexpected token embedding dimension: %lu\n",
                (unsigned long)embedding_dim);
        goto fail;
    }
    impl->token_embedding.embedding_dim = (int)embedding_dim;
    return 0;

fail:
    // Roll back only resources allocated by this function so a failed init
    // never leaks the tokenizer, the mapping, or the file descriptor.
    delete impl->tokenizer;
    impl->tokenizer = NULL;
    if (impl->token_embedding.data != NULL) {
        munmap(impl->token_embedding.data, impl->token_embedding.size);
        impl->token_embedding.data = NULL;
    }
    if (impl->token_embedding.fd >= 0) {
        close(impl->token_embedding.fd);
        impl->token_embedding.fd = -1;
    }
    return -1;
}

// Configure the single-request LLM session and register all callback bridges.
static int init_llm_session(wemm_embedding_impl_t* impl)
{
    static char output_name[] = "output";
    rknn3_llm_config llm_config;
    rknn3_llm_param params;
    rknn3_sampling_params sampling;
    VocabInfo vocab_info;
    int ret;

    memset(&llm_config, 0, sizeof(llm_config));
    memset(&params, 0, sizeof(params));
    memset(&sampling, 0, sizeof(sampling));
    memset(&impl->callback, 0, sizeof(impl->callback));
    memset(&vocab_info, 0, sizeof(vocab_info));

    impl->output_tensor.attr = (rknn3_tensor_attr*)calloc(1, sizeof(rknn3_tensor_attr));
    if (impl->output_tensor.attr == NULL) {
        return -1;
    }
    impl->output_tensor.attr->index = 0;
    ret = rknn3_query(impl->llm.context, RKNN3_QUERY_OUTPUT_ATTR,
                      impl->output_tensor.attr, sizeof(rknn3_tensor_attr));
    if (ret != RKNN3_SUCCESS ||
        impl->output_tensor.attr->n_elems != WEMM_EMBEDDING_DIM) {
        fprintf(stderr, "Failed to query LLM embedding output: %d\n", ret);
        return -1;
    }
    impl->output_tensor.mem = rknn3_create_mem(
        impl->llm.context, impl->output_tensor.attr->aligned_size,
        impl->output_tensor.attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
    if (impl->output_tensor.mem == NULL) {
        return -1;
    }
    ret = rknn3_query(impl->llm.context, RKNN3_QUERY_LLM_CONFIG,
                      &llm_config, sizeof(llm_config));
    if (ret != RKNN3_SUCCESS) {
        return ret;
    }
    impl->tokenizer->GetVocabInfo(&vocab_info);
    // One output token is requested only to trigger the prefill output callback.
    sampling.top_k = 1;
    sampling.temperature = 1.0f;
    sampling.repeat_penalty = 1.0f;
    params.logits_name = output_name;
    params.max_context_len = llm_config.max_ctx_len;
    params.sampling_param = sampling;
    params.vocab_info.vocab_size = vocab_info.vocab_size;
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id,
           sizeof(params.vocab_info.special_eos_id));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id,
           sizeof(params.vocab_info.special_bos_id));
    params.vocab_info.linefeed_id = vocab_info.linefeed_id;

    impl->callback.result_callback = result_callback;
    impl->callback.result_userdata = impl;
    impl->callback.tokenizer_callback = tokenizer_callback;
    impl->callback.tokenizer_userdata = impl->tokenizer;
    impl->callback.embed_callback = embed_callback;
    impl->callback.embed_userdata = &impl->token_embedding;
    impl->callback.output_callback = output_callback;
    impl->callback.output_userdata = impl;
    impl->callback.output_tensors = &impl->output_tensor;
    impl->callback.n_output_tensors = 1;

    impl->session = rknn3_session_init(impl->llm.context, &params, 1);
    if (impl->session == NULL) {
        fprintf(stderr, "Failed to initialize LLM session\n");
        return -1;
    }
    ret = rknn3_session_set_callback(impl->session, &impl->callback);
    if (ret != RKNN3_SUCCESS) {
        fprintf(stderr, "Failed to set LLM callbacks: %d\n", ret);
        return ret;
    }
    return 0;
}

// Initialize the LLM and the one optional vision tower required by mode.
int init_wemm_embedding(wemm_embedding_context_t* context,
                        const char* model_path,
                        wemm_embedding_mode_t mode)
{
    char llm_rknn[1024];
    char llm_weight[1024];
    char vision_image_rknn[1024];
    char vision_image_weight[1024];
    char vision_video_rknn[1024];
    char vision_video_weight[1024];
    char tokenizer_path[1024];
    char embedding_path[1024];
    wemm_embedding_impl_t* impl;
    int ret;

    if (context == NULL || model_path == NULL ||
        (mode != WEMM_EMBEDDING_MODE_TEXT &&
         mode != WEMM_EMBEDDING_MODE_IMAGE &&
         mode != WEMM_EMBEDDING_MODE_VIDEO)) {
        return -1;
    }
    memset(context, 0, sizeof(*context));
    if (make_model_file_path(llm_rknn, sizeof(llm_rknn), model_path,
                             "wemm-embedding-2b-llm.rknn") != 0 ||
        make_model_file_path(llm_weight, sizeof(llm_weight), model_path,
                             "wemm-embedding-2b-llm.weight") != 0 ||
        make_model_file_path(tokenizer_path, sizeof(tokenizer_path), model_path,
                             "wemm-embedding-2b-llm.tokenizer.gguf") != 0 ||
        make_model_file_path(embedding_path, sizeof(embedding_path), model_path,
                             "wemm-embedding-2b-llm.embed.bin") != 0) {
        return -1;
    }
    if (mode == WEMM_EMBEDDING_MODE_IMAGE &&
        (make_model_file_path(vision_image_rknn, sizeof(vision_image_rknn), model_path,
                              "wemm-embedding-2b-vision-image.rknn") != 0 ||
         make_model_file_path(vision_image_weight, sizeof(vision_image_weight), model_path,
                              "wemm-embedding-2b-vision-image.weight") != 0)) {
        return -1;
    }
    if (mode == WEMM_EMBEDDING_MODE_VIDEO &&
        (make_model_file_path(vision_video_rknn, sizeof(vision_video_rknn), model_path,
                              "wemm-embedding-2b-vision-video.rknn") != 0 ||
         make_model_file_path(vision_video_weight, sizeof(vision_video_weight), model_path,
                              "wemm-embedding-2b-vision-video.weight") != 0)) {
        return -1;
    }

    impl = (wemm_embedding_impl_t*)calloc(1, sizeof(*impl));
    if (impl == NULL) {
        return -1;
    }
    impl->token_embedding.fd = -1;
    impl->mode = mode;
    context->impl = impl;

    ret = init_tokenizer_and_embedding(impl, tokenizer_path, embedding_path);
    if (ret != 0) {
        release_wemm_embedding(context);
        return ret;
    }
    ret = init_rknn_model(&impl->llm, llm_rknn, llm_weight, 0);
    if (ret != 0) {
        release_wemm_embedding(context);
        return ret;
    }
    ret = init_llm_session(impl);
    if (ret != 0) {
        release_wemm_embedding(context);
        return ret;
    }
    if (mode == WEMM_EMBEDDING_MODE_IMAGE) {
        ret = init_rknn_model(&impl->vision_image,
                              vision_image_rknn, vision_image_weight, 1);
        if (ret == 0) {
            ret = validate_vision_model(&impl->vision_image, 1, "image");
        }
        if (ret != 0) {
            release_wemm_embedding(context);
            return ret;
        }
        impl->image_embedding_elements = impl->vision_image.outputs[0].attr->n_elems;
    } else if (mode == WEMM_EMBEDDING_MODE_VIDEO) {
        ret = init_rknn_model(&impl->vision_video,
                              vision_video_rknn, vision_video_weight, 1);
        if (ret == 0) {
            ret = validate_vision_model(&impl->vision_video, 2, "video");
        }
        if (ret != 0) {
            release_wemm_embedding(context);
            return ret;
        }
        impl->image_embedding_elements = impl->vision_video.outputs[0].attr->n_elems;
    } else {
        printf("WeMM initialized: text mode, embedding_dim=%d\n", WEMM_EMBEDDING_DIM);
        return 0;
    }
    impl->image_tokens = impl->image_embedding_elements / WEMM_EMBEDDING_DIM;
    impl->image_embedding =
        (float16*)malloc(impl->image_embedding_elements * sizeof(float16));
    if (impl->image_embedding == NULL) {
        release_wemm_embedding(context);
        return -1;
    }
    printf("WeMM initialized: mode=%s, vision_tokens=%lu, embedding_dim=%d\n",
           mode == WEMM_EMBEDDING_MODE_IMAGE ? "image" : "video",
           (unsigned long)impl->image_tokens, WEMM_EMBEDDING_DIM);
    return 0;
}

// Run one prefill-only LLM request and return its embedding and timing data.
static int run_llm_input(wemm_embedding_impl_t* impl,
                         rknn3_llm_input* input,
                         float* embedding,
                         wemm_embedding_perf_t* perf)
{
    rknn3_llm_infer_param infer_param;
    RKLLMRunState state;
    int64_t start_us;
    int64_t end_us;
    int ret;

    memset(&infer_param, 0, sizeof(infer_param));
    memset(&state, 0, sizeof(state));
    infer_param.keep_history = 0;
    infer_param.max_new_tokens = 1;
    impl->output_ready = 0;
    start_us = getCurrentTimeUs();
    ret = rknn3_session_run(impl->session, input, 1, &infer_param);
    end_us = getCurrentTimeUs();
    if (ret != RKNN3_SUCCESS || !impl->output_ready) {
        fprintf(stderr, "LLM embedding inference failed: ret=%d, output_ready=%d\n",
                ret, impl->output_ready);
        return ret == RKNN3_SUCCESS ? -1 : ret;
    }
    ret = rknn3_session_query_state(impl->session, &state);
    if (ret != RKNN3_SUCCESS) {
        return ret;
    }
    memcpy(embedding, impl->output, sizeof(impl->output));
    perf->llm_ms = (double)(end_us - start_us) / 1000.0;
    perf->input_tokens = state.n_prefill_tokens;
    return 0;
}

// Run the text-only path without invoking either vision model.
int run_wemm_text_embedding(wemm_embedding_context_t* context,
                            const char* text,
                            float* embedding,
                            wemm_embedding_perf_t* perf)
{
    wemm_embedding_impl_t* impl;
    rknn3_llm_input input;
    if (context == NULL || context->impl == NULL || text == NULL ||
        embedding == NULL || perf == NULL) {
        return -1;
    }
    impl = (wemm_embedding_impl_t*)context->impl;
    if (impl->mode != WEMM_EMBEDDING_MODE_TEXT) {
        return -1;
    }
    memset(perf, 0, sizeof(*perf));
    memset(&input, 0, sizeof(input));
    input.role = "user";
    input.input_type = RKNN3_LLM_INPUT_PROMPT;
    input.llm_input.name = "input";
    input.llm_input.prompt = text;
    input.llm_input.enable_thinking = 0;
    return run_llm_input(impl, &input, embedding, perf);
}

// Decode and resize with rknn3-model-zoo's shared image_utils implementation.
static int load_resized_rgb(const char* image_path,
                            int width,
                            int height,
                            unsigned char* output)
{
    image_buffer_t source;
    image_buffer_t resized;
    int ret = -1;

    memset(&source, 0, sizeof(source));
    memset(&resized, 0, sizeof(resized));
    ret = read_image(image_path, &source);
    if (ret != 0) {
        fprintf(stderr, "Failed to read image: %s\n", image_path);
        goto out;
    }
    resized.width = width;
    resized.height = height;
    resized.format = IMAGE_FORMAT_RGB888;
    resized.size = get_image_size(&resized);
    resized.virt_addr = (unsigned char*)malloc(resized.size);
    if (resized.virt_addr == NULL) {
        ret = -1;
        goto out;
    }
    ret = convert_image(&source, &resized, NULL, NULL, 0);
    if (ret != 0) {
        goto out;
    }
    memcpy(output, resized.virt_addr, resized.size);

out:
    free(source.virt_addr);
    free(resized.virt_addr);
    return ret;
}

static int run_vision(wemm_rknn_model_t* vision,
                      const char* first_path,
                      const char* second_path,
                      uint64_t output_elements,
                      float16* output,
                      double* latency_ms)
{
    const int frame_height = (int)vision->inputs[0].attr->shape[1];
    const int frame_width = (int)vision->inputs[0].attr->shape[2];
    const size_t frame_size = (size_t)frame_width * (size_t)frame_height * 3U;
    unsigned char* input = (unsigned char*)vision->inputs[0].mem->virt_addr;
    int64_t start_us;
    int64_t end_us;
    int ret;

    memset(input, 0, vision->inputs[0].attr->aligned_size);
    // Frames are stored consecutively because the RKNN input is NHWC.
    ret = load_resized_rgb(first_path, frame_width, frame_height, input);
    if (ret != 0) {
        return ret;
    }
    if (second_path != NULL) {
        ret = load_resized_rgb(second_path, frame_width, frame_height,
                               input + frame_size);
        if (ret != 0) {
            return ret;
        }
    }
    ret = rknn3_mem_sync(vision->context, vision->inputs[0].mem,
                         RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN3_SUCCESS) {
        return ret;
    }
    start_us = getCurrentTimeUs();
    ret = rknn3_run(vision->context,
                    vision->inputs, vision->io_num.n_input,
                    vision->outputs, vision->io_num.n_output);
    end_us = getCurrentTimeUs();
    if (ret != RKNN3_SUCCESS) {
        return ret;
    }
    ret = rknn3_mem_sync(vision->context, vision->outputs[0].mem,
                         RKNN3_MEMORY_SYNC_FROM_DEVICE);
    if (ret != RKNN3_SUCCESS) {
        return ret;
    }
    memcpy(output,
           vision->outputs[0].mem->virt_addr,
           output_elements * sizeof(float16));
    *latency_ms = (double)(end_us - start_us) / 1000.0;
    return 0;
}

// Run the one-frame vision model and inject its tokens at the image marker.
int run_wemm_image_embedding(wemm_embedding_context_t* context,
                             const char* image_path,
                             const char* text,
                             float* embedding,
                             wemm_embedding_perf_t* perf)
{
    static const char image_marker[] = "<image>";
    wemm_embedding_impl_t* impl;
    rknn3_llm_input input;
    char* prompt;
    size_t marker_length;
    size_t text_length;
    int ret;
    if (context == NULL || context->impl == NULL || image_path == NULL ||
        text == NULL || embedding == NULL || perf == NULL) {
        return -1;
    }
    impl = (wemm_embedding_impl_t*)context->impl;
    if (impl->mode != WEMM_EMBEDDING_MODE_IMAGE) {
        return -1;
    }
    memset(perf, 0, sizeof(*perf));
    ret = run_vision(&impl->vision_image, image_path, NULL,
                     impl->image_embedding_elements,
                     impl->image_embedding, &perf->vision_ms);
    if (ret != 0) {
        return ret;
    }
    // The RKNN3 multimodal API replaces <image> with the supplied vision tokens.
    marker_length = strlen(image_marker);
    text_length = strlen(text);
    if (text_length > SIZE_MAX - marker_length - 1U) {
        return -1;
    }
    prompt = (char*)malloc(marker_length + text_length + 1U);
    if (prompt == NULL) {
        return -1;
    }
    memcpy(prompt, image_marker, marker_length);
    memcpy(prompt + marker_length, text, text_length + 1U);
    memset(&input, 0, sizeof(input));
    input.role = "user";
    input.input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    input.multimodal_input.name = "input_embeds";
    input.multimodal_input.prompt = prompt;
    input.multimodal_input.enable_thinking = 0;
    input.multimodal_input.image.image_embed = impl->image_embedding;
    input.multimodal_input.image.n_image_tokens = impl->image_tokens;
    input.multimodal_input.image.n_image = 1;
    input.multimodal_input.image.image_start = "<|vision_start|>";
    input.multimodal_input.image.image_end = "<|vision_end|>";
    input.multimodal_input.image.image_content = "<|image_pad|>";
    input.multimodal_input.image.image_width =
        impl->vision_image.inputs[0].attr->shape[2];
    input.multimodal_input.image.image_height =
        impl->vision_image.inputs[0].attr->shape[1];
    ret = run_llm_input(impl, &input, embedding, perf);
    free(prompt);
    return ret;
}

// Run adjacent frame pairs and pass timestamped video tokens to the LLM runtime.
int run_wemm_video_embedding(wemm_embedding_context_t* context,
                             const char* const* frame_paths,
                             uint64_t frame_count,
                             float frame_fps,
                             const char* text,
                             float* embedding,
                             wemm_embedding_perf_t* perf)
{
    wemm_embedding_impl_t* impl;
    rknn3_llm_input input;
    char* prompt;
    float16* video_embedding;
    uint64_t pair_count;
    uint64_t pair_index;
    size_t prompt_capacity;
    size_t prompt_length;
    size_t text_length;
    size_t video_elements;
    int ret = 0;

    if (context == NULL || context->impl == NULL || frame_paths == NULL ||
        frame_count == 0 || !isfinite(frame_fps) || frame_fps <= 0.0f || text == NULL ||
        embedding == NULL || perf == NULL) {
        return -1;
    }
    for (pair_index = 0; pair_index < frame_count; ++pair_index) {
        if (frame_paths[pair_index] == NULL || frame_paths[pair_index][0] == '\0') {
            return -1;
        }
    }
    // The video vision model consumes adjacent frame pairs. Duplicate the last
    // frame when an odd number of sampled frames is provided. Twenty-six pairs
    // leave room for text and special tokens in the verified 4096-token context.
    pair_count = (frame_count + 1) / 2;
    if (pair_count > 26 ||
        pair_count > SIZE_MAX / sizeof(float16) /
                         ((size_t)WEMM_EMBEDDING_DIM * 144U)) {
        fprintf(stderr, "Too many video frames: %lu (maximum: 52)\n",
                (unsigned long)frame_count);
        return -1;
    }
    impl = (wemm_embedding_impl_t*)context->impl;
    if (impl->mode != WEMM_EMBEDDING_MODE_VIDEO) {
        return -1;
    }
    if (impl->image_tokens != 144) {
        fprintf(stderr, "Unexpected video vision token count: %lu\n",
                (unsigned long)impl->image_tokens);
        return -1;
    }
    memset(perf, 0, sizeof(*perf));
    video_elements = (size_t)pair_count * impl->image_embedding_elements;
    video_embedding = (float16*)malloc(video_elements * sizeof(float16));
    if (video_embedding == NULL) {
        return -1;
    }
    for (pair_index = 0; pair_index < pair_count; ++pair_index) {
        const uint64_t first = pair_index * 2;
        const uint64_t second = first + 1 < frame_count ? first + 1 : first;
        double pair_ms = 0.0;
        ret = run_vision(
            &impl->vision_video, frame_paths[first], frame_paths[second],
            impl->image_embedding_elements,
            video_embedding + pair_index * impl->image_embedding_elements,
            &pair_ms);
        if (ret != 0) {
            free(video_embedding);
            return ret;
        }
        perf->vision_ms += pair_ms;
    }

    // Match WeMM's processor by placing one timestamped marker around each
    // temporal unit. All markers still belong to the single video described by
    // n_video=1 and n_frame_per_video=pair_count below.
    text_length = strlen(text);
    if (pair_count > (SIZE_MAX - text_length - 1U) / 64U) {
        free(video_embedding);
        return -1;
    }
    prompt_capacity = text_length + (size_t)pair_count * 64U + 1U;
    prompt = (char*)malloc(prompt_capacity);
    if (prompt == NULL) {
        free(video_embedding);
        return -1;
    }
    prompt[0] = '\0';
    prompt_length = 0;
    for (pair_index = 0; pair_index < pair_count; ++pair_index) {
        const uint64_t first = pair_index * 2;
        const uint64_t second = first + 1 < frame_count ? first + 1 : first;
        const double timestamp = ((double)first + (double)second) /
                                 (2.0 * (double)frame_fps);
        int count = snprintf(prompt + prompt_length,
                             prompt_capacity - prompt_length,
                             "<%.1f seconds><video>", timestamp);
        if (count < 0 || (size_t)count >= prompt_capacity - prompt_length) {
            free(prompt);
            free(video_embedding);
            return -1;
        }
        prompt_length += (size_t)count;
    }
    memcpy(prompt + prompt_length, text, text_length + 1U);
    memset(&input, 0, sizeof(input));
    input.role = "user";
    input.input_type = RKNN3_LLM_INPUT_MULTIMODAL;
    input.multimodal_input.name = "input_embeds";
    input.multimodal_input.prompt = prompt;
    input.multimodal_input.enable_thinking = 0;
    input.multimodal_input.video.video_embed = video_embedding;
    input.multimodal_input.video.n_frame_tokens = impl->image_tokens;
    input.multimodal_input.video.n_frame_per_video = pair_count;
    input.multimodal_input.video.n_video = 1;
    input.multimodal_input.video.video_start = "<|vision_start|>";
    input.multimodal_input.video.video_end = "<|vision_end|>";
    input.multimodal_input.video.video_content = "<|video_pad|>";
    input.multimodal_input.video.frame_width =
        impl->vision_video.inputs[0].attr->shape[2];
    input.multimodal_input.video.frame_height =
        impl->vision_video.inputs[0].attr->shape[1];
    ret = run_llm_input(impl, &input, embedding, perf);
    free(prompt);
    free(video_embedding);
    return ret;
}

// Tear down resources in reverse initialization order; safe after partial init.
void release_wemm_embedding(wemm_embedding_context_t* context)
{
    wemm_embedding_impl_t* impl;
    if (context == NULL || context->impl == NULL) {
        return;
    }
    impl = (wemm_embedding_impl_t*)context->impl;
    if (impl->session != NULL) {
        rknn3_session_destroy(impl->session);
    }
    if (impl->output_tensor.mem != NULL && impl->llm.context != 0) {
        rknn3_destroy_mem(impl->llm.context, impl->output_tensor.mem);
    }
    free(impl->output_tensor.attr);
    release_rknn_model(&impl->vision_video);
    release_rknn_model(&impl->vision_image);
    release_rknn_model(&impl->llm);
    if (impl->token_embedding.data != NULL) {
        munmap(impl->token_embedding.data, impl->token_embedding.size);
    }
    if (impl->token_embedding.fd >= 0) {
        close(impl->token_embedding.fd);
    }
    delete impl->tokenizer;
    free(impl->image_embedding);
    free(impl);
    context->impl = NULL;
}
