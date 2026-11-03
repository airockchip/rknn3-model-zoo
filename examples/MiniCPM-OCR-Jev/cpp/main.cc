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
    对 (原图, 渲染图, OCR文本) 三元组打分的最小 C++ 演示
-------------------------------------------*/
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "image_utils.h"
#include "time_utils.h"
#include "minicpm_ocr_jev.h"

struct free_deleter {
    void operator()(void *ptr) const { free(ptr); }
};

/*-------------------------------------------
                  Callbacks
-------------------------------------------*/

// The runtime expands every `<image>` marker in `multimodal_input.prompt` into
// image_start + n_image_tokens * image_content + image_end, wraps the whole
// text in the model's chat template, and only then tokenizes it through this
// callback. So the text received here is neither the host string nor a plain
// expansion of it -- it is the fully templated prompt.
static int tokenizer_callback(void *userdata, const char *text, int32_t text_len,
                              int32_t *tokens, int32_t n_tokens_max)
{
    Tokenizer *tokenizer = (Tokenizer *)userdata;
    int n_tokens = tokenizer->Tokenize(text, text_len, tokens, n_tokens_max);
    if (n_tokens <= 0) {
        printf("tokenizer failed for %.*s\n", text_len > 0 ? text_len : 0, text != NULL ? text : "");
        return n_tokens;
    }
    return n_tokens;
}

// Looks up the float16 embedding of every token from the mapped bank.
static int embed_callback(void *userdata, int32_t *tokens, uint64_t num_tokens,
                          void *embed, uint64_t len)
{
    value_head_embedding_info *info = (value_head_embedding_info *)userdata;
    if (len != num_tokens * (uint64_t)info->embedding_dim * sizeof(float16)) {
        printf("invalid embed buffer\n");
        return -1;
    }
    for (uint64_t n = 0; n < num_tokens; n++) {
        memcpy((unsigned char *)embed + n * (uint64_t)info->embedding_dim * sizeof(float16),
               info->data + (size_t)tokens[n] * (size_t)info->embedding_dim,
               (size_t)info->embedding_dim * sizeof(float16));
    }
    return 0;
}

// The runtime refuses to start a run (`no tokenizer_callback or
// result_callback provided!`) unless one of the two is registered. This demo
// drives everything through `prompt` + `tokenizer_callback`, but the sampling
// stage still needs a result sink, so register a no-op one: the score itself
// arrives through `output_callback` (see value_head_output_callback), not here.
static int result_callback(void *userdata, RKLLMResult *result, LLMCallState state)
{
    (void)userdata;
    (void)result;
    (void)state;
    return 0;
}

/*-------------------------------------------
                  Arguments
-------------------------------------------*/

typedef struct {
    const char *vision_model_path;
    const char *vision_weight_path;
    const char *llm_model_path;
    const char *llm_weight_path;
    const char *tokenizer_path;
    const char *embedding_path;
    uint32_t vision_core_mask;
    uint32_t llm_core_mask;
    const char *orig_image;
    const char *render_image;
    std::string ocr_text;
    float threshold;
    uint32_t model_width;
    uint32_t model_height;
    bool print_prompt;
    bool json_output;
    bool has_prompt;
} value_head_args;

static void print_usage(const char *program)
{
    printf("Usage:\n");
    printf("  %s <vision_model> <vision_weight> <llm_model> <llm_weight> \\\n", program);
    printf("     <tokenizer> <embedding> <vision_core_mask> <llm_core_mask> \\\n");
    printf("     <orig_image> <render_image> --ocr-text <text>\n\n");
    printf("Positional arguments:\n");
    printf("  vision_model        vision .rknn path\n");
    printf("  vision_weight       vision .weight path\n");
    printf("  llm_model           llm .rknn path\n");
    printf("  llm_weight          llm .weight path\n");
    printf("  tokenizer           .tokenizer.gguf path\n");
    printf("  embedding           .embed.bin path\n");
    printf("  vision_core_mask    vision NPU core mask (hex), e.g. 1\n");
    printf("  llm_core_mask       llm NPU core mask (hex), e.g. fe\n");
    printf("  orig_image          original document image\n");
    printf("  render_image        rendered OCR text image\n\n");
    printf("Named arguments:\n");
    printf("  --ocr-text <text>       OCR text (mutually exclusive with --ocr-text-file)\n");
    printf("  --ocr-text-file <path>  read OCR text from file (trailing CR/LF removed)\n");
    printf("  --threshold <float>     decision threshold (default %.16f)\n", VALUE_HEAD_DEFAULT_THRESHOLD);
    printf("  --model-width <int>     vision canvas width for pruned exports (default %d)\n", MODEL_WIDTH);
    printf("  --model-height <int>    vision canvas height for pruned exports (default %d)\n", MODEL_HEIGHT);
    printf("  --print-prompt          print the assembled prompt\n");
    printf("  --json                  print one JSON object per sample\n");
    printf("  --help                  show this message\n");
    printf("\nThe two images must already be preprocessed the way the reference\n");
    printf("pipeline does it (aspect-preserving resize + letterbox to the compiled\n");
    printf("canvas). This demo only stretches if the image size differs.\n");
}

static bool parse_uint32(const char *text, const char *name, uint32_t *value)
{
    if (!text || !value || text[0] == '\0' || text[0] == '-') {
        printf("%s must be an unsigned integer\n", name);
        return false;
    }
    char *end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || parsed > 0xFFFFFFFFUL) {
        printf("%s must be an unsigned integer\n", name);
        return false;
    }
    *value = (uint32_t)parsed;
    return true;
}

static bool parse_float_arg(const char *text, const char *name, float *value)
{
    if (!text || !value || text[0] == '\0') {
        printf("%s must be a number\n", name);
        return false;
    }
    char *end = NULL;
    errno = 0;
    float parsed = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0') {
        printf("%s must be a number\n", name);
        return false;
    }
    *value = parsed;
    return true;
}

static bool read_text_file(const char *path, std::string *out)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        printf("failed to open text file: %s\n", path);
        return false;
    }
    std::ostringstream ss;
    ss << ifs.rdbuf();
    *out = ss.str();
    // 与 demo_value_head.py 一致：只去掉结尾的换行符，保留中间的换行/空行
    while (!out->empty() && (out->back() == '\n' || out->back() == '\r')) {
        out->pop_back();
    }
    return true;
}

static int parse_args(int argc, char **argv, value_head_args *args)
{
    if (argc < 2) {
        return -1;
    }
    // value_head_args holds a std::string, so it must be value-initialized
    // rather than memset.
    *args = value_head_args();
    args->threshold = VALUE_HEAD_DEFAULT_THRESHOLD;
    args->vision_core_mask = 0;
    args->llm_core_mask = 0;

    std::vector<const char *> positional;
    const char *ocr_text_arg = NULL;
    const char *ocr_text_file = NULL;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            return 1;
        }
        if (strncmp(arg, "--", 2) != 0) {
            positional.push_back(arg);
            continue;
        }
        if (strcmp(arg, "--print-prompt") == 0) {
            args->print_prompt = true;
            continue;
        }
        if (strcmp(arg, "--json") == 0) {
            args->json_output = true;
            continue;
        }
        if (i + 1 >= argc) {
            printf("missing value for option %s\n", arg);
            return -1;
        }
        const char *value = argv[++i];
        if (strcmp(arg, "--ocr-text") == 0) {
            ocr_text_arg = value;
        } else if (strcmp(arg, "--ocr-text-file") == 0) {
            ocr_text_file = value;
        } else if (strcmp(arg, "--threshold") == 0) {
            if (!parse_float_arg(value, "threshold", &args->threshold)) {
                return -1;
            }
        } else if (strcmp(arg, "--model-width") == 0) {
            if (!parse_uint32(value, "model_width", &args->model_width)) {
                return -1;
            }
        } else if (strcmp(arg, "--model-height") == 0) {
            if (!parse_uint32(value, "model_height", &args->model_height)) {
                return -1;
            }
        } else {
            printf("unknown option: %s\n", arg);
            return -1;
        }
    }

    if (positional.size() != 10) {
        printf("expected 10 positional arguments, got %zu\n", positional.size());
        return -1;
    }
    args->vision_model_path = positional[0];
    args->vision_weight_path = positional[1];
    args->llm_model_path = positional[2];
    args->llm_weight_path = positional[3];
    args->tokenizer_path = positional[4];
    args->embedding_path = positional[5];
    if (!parse_uint32(positional[6], "vision_core_mask", &args->vision_core_mask) ||
        !parse_uint32(positional[7], "llm_core_mask", &args->llm_core_mask)) {
        return -1;
    }
    args->orig_image = positional[8];
    args->render_image = positional[9];

    if (ocr_text_arg && ocr_text_file) {
        printf("--ocr-text and --ocr-text-file are mutually exclusive\n");
        return -1;
    }
    if (ocr_text_file) {
        if (!read_text_file(ocr_text_file, &args->ocr_text)) {
            return -1;
        }
        args->has_prompt = true;
    } else if (ocr_text_arg) {
        args->ocr_text = ocr_text_arg;
        args->has_prompt = true;
    } else {
        printf("one of --ocr-text or --ocr-text-file is required\n");
        return -1;
    }
    return 0;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/

int main(int argc, char **argv)
{
    // Unbuffered stdout: the progress prints double as the crash trail when a
    // run is killed by a signal (a block-buffered stdout would drop them).
    setvbuf(stdout, NULL, _IONBF, 0);

    value_head_args args;
    int parse_ret = parse_args(argc, argv, &args);
    if (parse_ret != 0) {
        print_usage(argv[0]);
        return parse_ret > 0 ? 0 : -1;
    }

    int ret = 0;

    // app_ctx owns a std::mutex / std::condition_variable, so it cannot be
    // memset. Zero only the trivially-copyable sub-structures and let the
    // sink's own default constructor take care of the synchronization members.
    rknn_value_head_context app_ctx;
    memset(&(app_ctx.vision), 0, sizeof(app_ctx.vision));
    memset(&(app_ctx.llm), 0, sizeof(app_ctx.llm));
    app_ctx.sink.expected_n_elems = VALUE_HEAD_NUM_LABELS;
    app_ctx.sink.ready = false;
    app_ctx.sink.error = false;
    app_ctx.sink.run_start_us = 0;
    app_ctx.sink.run_end_us = 0;
    app_ctx.sink.n_prefill_tokens = 0;
    app_ctx.n_internal_mems = 0;
    app_ctx.internal_mems = NULL;
    app_ctx.model_width = 0;
    app_ctx.model_height = 0;

    VocabInfo vocab_info;
    memset(&vocab_info, 0, sizeof(vocab_info));
    int max_special_ids = (int)(sizeof(vocab_info.special_eos_id) / sizeof(vocab_info.special_eos_id[0]));

    struct value_head_embedding_holder {
        value_head_embedding_info info;
        off_t size;
    } embedding;
    memset(&embedding, 0, sizeof(embedding));
    embedding.info.fd = -1;
    rknn3_llm_param params;
    memset(&params, 0, sizeof(params));

    RKLLMCallback callback;
    memset(&callback, 0, sizeof(callback));

    image_buffer_t orig_image;
    image_buffer_t render_image;
    memset(&orig_image, 0, sizeof(orig_image));
    memset(&render_image, 0, sizeof(render_image));

    Tokenizer *tokenizer = NULL;

    // Load tokenizer
    tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, args.tokenizer_path);
    if (!tokenizer) {
        printf("load tokenizer failed! tokenizer_path=%s\n", args.tokenizer_path);
        ret = -1;
        goto out;
    }
    tokenizer->GetVocabInfo(&vocab_info);
    printf("vocab_info: vocab_size=%d, n_special_eos_id=%d\n",
           vocab_info.vocab_size, vocab_info.n_special_eos_id);
    if (vocab_info.vocab_size <= 0) {
        printf("invalid vocab size\n");
        ret = -1;
        goto out;
    }

    // Map the embedding bank
    embedding.info.fd = open(args.embedding_path, O_RDONLY);
    if (embedding.info.fd == -1) {
        printf("failed to open embedding file: %s\n", args.embedding_path);
        ret = -1;
        goto out;
    }
    {
        struct stat emb_st;
        memset(&emb_st, 0, sizeof(emb_st));
        if (fstat(embedding.info.fd, &emb_st) == -1) {
            printf("failed to get embedding file size\n");
            ret = -1;
            goto out;
        }
        if (emb_st.st_size <= 0 ||
            emb_st.st_size % ((off_t)sizeof(float16) * (off_t)vocab_info.vocab_size) != 0) {
            printf("embedding file size does not match vocab size\n");
            ret = -1;
            goto out;
        }
        embedding.size = emb_st.st_size;
        embedding.info.data = (float16 *)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, embedding.info.fd, 0);
        if (embedding.info.data == MAP_FAILED) {
            printf("failed to mmap embedding file\n");
            embedding.info.data = NULL;
            ret = -1;
            goto out;
        }
    }
    embedding.info.vocab_size = vocab_info.vocab_size;
    embedding.info.embedding_dim = (int)((embedding.size / vocab_info.vocab_size) / (off_t)sizeof(float16));
    embedding.info.size = (long long)embedding.size;

    // Session parameters. Only prefill runs, so the sampling fields are inert
    // but still validated by the runtime.
    params.logits_name = const_cast<char *>("output");
    params.max_context_len = VALUE_HEAD_MAX_CONTEXT_LEN;
    params.sampling_param.top_k = 1;
    params.sampling_param.top_p = 0.9f;
    params.sampling_param.temperature = 1.0f;
    params.sampling_param.repeat_penalty = 1.0f;
    params.sampling_param.frequency_penalty = 0.0f;
    params.sampling_param.presence_penalty = 0.0f;
    params.vocab_info.vocab_size = vocab_info.vocab_size;
    if (vocab_info.n_special_eos_id > max_special_ids || vocab_info.n_special_bos_id > max_special_ids) {
        printf("special id count exceeds capacity\n");
        ret = -1;
        goto out;
    }
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id,
           vocab_info.n_special_eos_id * sizeof(vocab_info.special_eos_id[0]));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id,
           vocab_info.n_special_bos_id * sizeof(vocab_info.special_bos_id[0]));
    params.vocab_info.linefeed_id = vocab_info.linefeed_id;

    // The score head output tensor is allocated inside init_value_head_llm,
    // which also arms callback->output_callback / n_output_tensors /
    // output_tensors before rknn3_session_set_callback runs.
    //
    // result_callback must be non-NULL: rknn3_session_run() aborts with
    // "no tokenizer_callback or result_callback provided!" otherwise. The
    // score does not come through it, so a no-op sink is enough.
    callback.result_callback = result_callback;
    callback.result_userdata = NULL;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = tokenizer;
    callback.embed_callback = embed_callback;
    callback.embed_userdata = &(embedding.info);
    callback.output_userdata = &(app_ctx.sink);
    callback.n_output_tensors = 0;
    callback.output_tensors = NULL;

    ret = init_value_head_model(&app_ctx,
                                args.llm_model_path, args.llm_weight_path,
                                args.vision_model_path, args.vision_weight_path,
                                &params, 1, &callback,
                                args.vision_core_mask, args.llm_core_mask,
                                args.model_width, args.model_height);
    if (ret != 0) {
        printf("init_value_head_model failed! ret=%d\n", ret);
        goto out;
    }

    ret = read_image(args.orig_image, &orig_image);
    if (ret != 0) {
        printf("read image failed! ret=%d path=%s\n", ret, args.orig_image);
        goto out;
    }
    ret = read_image(args.render_image, &render_image);
    if (ret != 0) {
        printf("read image failed! ret=%d path=%s\n", ret, args.render_image);
        goto out;
    }

    value_head_result result;
    memset(&result, 0, sizeof(result));
    ret = score_value_head(&app_ctx, tokenizer, &orig_image, &render_image,
                           args.ocr_text, args.threshold, args.print_prompt, &result);
    if (ret != 0) {
        printf("score_value_head failed! ret=%d\n", ret);
        goto out;
    }

    if (args.json_output) {
        printf("{\"prob_1\": %.8f, \"prob_0\": %.8f, \"logit_1\": %.8f, \"logit_0\": %.8f, "
               "\"label\": %d, \"threshold\": %.8f, \"n_prompt_tokens\": %d, "
               "\"n_vision_tokens\": %d, \"vision_ms\": %.3f, \"llm_ms\": %.3f}\n",
               result.prob_1, result.prob_0, result.logit_1, result.logit_0,
               result.label, result.threshold, result.n_prompt_tokens,
               result.n_vision_tokens,
               (double)result.vision_latency_us / 1000.0,
               (double)result.llm_latency_us / 1000.0);
    } else {
        printf("\n---------------- value head ----------------\n");
        printf(" logit_0        : %.8f\n", result.logit_0);
        printf(" logit_1        : %.8f\n", result.logit_1);
        printf(" prob_0         : %.8f\n", result.prob_0);
        printf(" prob_1         : %.8f\n", result.prob_1);
        printf(" threshold      : %.16f\n", result.threshold);
        printf(" label          : %d (%s)\n", result.label,
               result.label == 1 ? "一致" : "不一致");
        printf(" prompt tokens  : %d\n", result.n_prompt_tokens);
        printf(" vision tokens  : %d x 2\n", result.n_vision_tokens);
        printf(" vision latency : %.3f ms (x2)\n", (double)result.vision_latency_us / 1000.0);
        printf(" llm latency    : %.3f ms\n", (double)result.llm_latency_us / 1000.0);
        printf("--------------------------------------------\n");
    }

out:
    if (app_ctx.vision.rknn_ctx != 0 || app_ctx.llm.rknn_ctx != 0) {
        release_value_head_model(&app_ctx);
    }
    if (embedding.info.data != NULL && embedding.info.data != MAP_FAILED) {
        munmap((void *)embedding.info.data, embedding.size);
        embedding.info.data = NULL;
    }
    if (embedding.info.fd != -1) {
        close(embedding.info.fd);
        embedding.info.fd = -1;
    }
    if (orig_image.virt_addr != NULL) {
        free(orig_image.virt_addr);
    }
    if (render_image.virt_addr != NULL) {
        free(render_image.virt_addr);
    }
    delete tokenizer;
    return ret;
}
