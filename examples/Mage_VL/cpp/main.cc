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
#include <string.h>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <memory>
#include <new>
#include <string>
#include <vector>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

#include "mage_vl.h"
#include "image_utils.h"
#include "time_utils.h"

static const size_t MAX_INPUT_FRAMES = 16;
static const int PROMPT_TOKEN_RESERVE = 256;
static const char* CODEC_ROTARY_FILE = "rotary_pos_emb.fp16.bin";
static const char* CODEC_POSITION_FILE = "src_patch_position.npy";
static const char* CODEC_METADATA_FILE = "codec_metadata.json";

struct EmbeddingInfo
{
    int      fd;
    float16* embedding_data;
    int      embedding_dim;
    int      vocab_size;
};

struct ResultCallbackContext
{
    Tokenizer* tokenizer;
    int64_t first_token;
    bool first_decode;
};

class ImageDataGuard
{
public:
    explicit ImageDataGuard(image_buffer_t* image) : image_(image) {}

    ~ImageDataGuard()
    {
        if (image_ != NULL && image_->virt_addr != NULL) {
            free(image_->virt_addr);
            image_->virt_addr = NULL;
        }
    }

    ImageDataGuard(const ImageDataGuard&) = delete;
    ImageDataGuard& operator=(const ImageDataGuard&) = delete;

private:
    image_buffer_t* image_;
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
    ResultCallbackContext* callback_ctx =
        static_cast<ResultCallbackContext*>(userdata);
    if (callback_ctx == NULL || callback_ctx->tokenizer == NULL) {
        printf("result_callback context is NULL\n");
        return -1;
    }
    Tokenizer* tokenizer = callback_ctx->tokenizer;

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
        if (result == NULL || result->token_ids == NULL ||
            result->num_tokens <= 0) {
            printf("invalid inference result\n");
            return -1;
        }
        // Get token text
        std::string piece;
        if (result->num_tokens == 1) {
          piece = tokenizer->TokenToPiece(result->token_ids[0]);
        } else {
          piece = tokenizer->Decode(result->token_ids, result->num_tokens);
        }

        // Print token text
        fputs(piece.c_str(), stdout);

        if (callback_ctx->first_decode) {
            callback_ctx->first_token = getCurrentTimeUs();
            callback_ctx->first_decode = false;
        }
        fflush(stdout);
    }
    return 0;
}


int tokenizer_callback(void *userdata, const char *text, int32_t text_len, int32_t *tokens, int32_t n_tokens_max)
{
    if (userdata == NULL || text == NULL || text_len < 0 || tokens == NULL ||
        n_tokens_max <= 0) {
        printf("invalid tokenizer callback argument\n");
        return -1;
    }
    int n_tokens = 0;
    Tokenizer* tokenizer = static_cast<Tokenizer*>(userdata);
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

    EmbeddingInfo* embed_info = static_cast<EmbeddingInfo*>(userdata);
    if (embed_info == NULL || tokens == NULL || embed == NULL ||
        embed_info->embedding_data == NULL ||
        embed_info->embedding_dim <= 0 || embed_info->vocab_size <= 0) {
        printf("invalid embedding callback argument\n");
        return -1;
    }

    const uint64_t token_bytes =
        static_cast<uint64_t>(embed_info->embedding_dim) * sizeof(float16);
    if (len % token_bytes != 0 || len / token_bytes != num_tokens) {
        printf("invalid embed buffer\n");
        return -1;
    }

    for (uint64_t n = 0; n < num_tokens; n++) {
        if (tokens[n] < 0 || tokens[n] >= embed_info->vocab_size) {
            printf("invalid token id: %d\n", tokens[n]);
            return -1;
        }
        memcpy(static_cast<unsigned char*>(embed) + n * embed_info->embedding_dim * sizeof(float16), embed_info->embedding_data + tokens[n] * embed_info->embedding_dim,
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


void printf_perf(rknn_perf_metrics_t *p, size_t visual_count,
                 const ResultCallbackContext* callback_ctx)
{
    if (p == NULL || callback_ctx == NULL) {
        printf("Invalid perf metrics or callback context pointer.\n");
        return;
    }

    printf("\n--------------------------------------------------------------------------------------\n");
    printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n",
           "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    printf("--------------------------------------------------------------------------------------\n");

    const bool has_first_token =
        callback_ctx->first_token >= p->llm_start_time &&
        callback_ctx->first_token <= p->llm_end_time;
    double ttft_us = has_first_token
                         ? (double)(callback_ctx->first_token -
                                    p->llm_start_time)
                         : 0.0;
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

    double decode_time_us = has_first_token
                                ? (double)(p->llm_end_time -
                                           callback_ctx->first_token)
                                : 0.0;
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

    printf(" Vision total latency = %.2f ms for %zu visual(s)\n",
           (double)p->vision_latency / 1000.0, visual_count);
    if (visual_count > 0) {
        print_latency_fps("Vision/visual",
                          p->vision_latency / (int64_t)visual_count);
    }
}

static bool is_supported_image(const std::string& name)
{
    size_t dot = name.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }
    std::string suffix = name.substr(dot);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
                   [](unsigned char value) {
                       return static_cast<char>(std::tolower(value));
                   });
    return suffix == ".jpg" || suffix == ".jpeg" || suffix == ".png" ||
           suffix == ".bmp";
}

static std::string join_path(const std::string& directory,
                             const std::string& name)
{
    if (!directory.empty() && directory[directory.size() - 1] == '/') {
        return directory + name;
    }
    return directory + "/" + name;
}

static bool is_regular_file(const std::string& path)
{
    struct stat file_stat;
    return stat(path.c_str(), &file_stat) == 0 &&
           S_ISREG(file_stat.st_mode);
}

static int read_frame_directory(const char* directory_path,
                                std::vector<std::string>* frame_paths,
                                size_t* discovered_frames)
{
    if (directory_path == NULL || frame_paths == NULL ||
        discovered_frames == NULL) {
        return -1;
    }

    DIR* directory = opendir(directory_path);
    if (directory == NULL) {
        printf("Failed to open frame directory: %s\n", directory_path);
        return -1;
    }

    std::vector<std::string> all_frames;
    struct dirent* entry = NULL;
    while ((entry = readdir(directory)) != NULL) {
        const std::string name(entry->d_name);
        if (name == "." || name == ".." || !is_supported_image(name)) {
            continue;
        }

        const std::string path = join_path(directory_path, name);
        struct stat file_stat;
        if (stat(path.c_str(), &file_stat) == 0 && S_ISREG(file_stat.st_mode)) {
            all_frames.push_back(path);
        }
    }
    closedir(directory);

    std::sort(all_frames.begin(), all_frames.end());
    *discovered_frames = all_frames.size();
    if (all_frames.empty()) {
        printf("Frame directory contains no supported image: %s\n",
               directory_path);
        return -1;
    }

    frame_paths->clear();
    if (all_frames.size() <= MAX_INPUT_FRAMES) {
        frame_paths->swap(all_frames);
        return 0;
    }

    // Preserve the first and last images and sample the remainder uniformly.
    frame_paths->reserve(MAX_INPUT_FRAMES);
    for (size_t index = 0; index < MAX_INPUT_FRAMES; ++index) {
        const size_t denominator = MAX_INPUT_FRAMES - 1;
        size_t source_index = (index * (all_frames.size() - 1) +
                               denominator / 2) / denominator;
        frame_paths->push_back(all_frames[source_index]);
    }
    return 0;
}

static int read_codec_directory(const char* directory_path,
                                std::vector<std::string>* canvas_paths,
                                size_t* discovered_canvases)
{
    if (directory_path == NULL || canvas_paths == NULL ||
        discovered_canvases == NULL) {
        return -1;
    }

    const std::string directory(directory_path);
    const std::string rotary_path = join_path(directory, CODEC_ROTARY_FILE);
    const std::string position_path = join_path(directory, CODEC_POSITION_FILE);
    const std::string metadata_path = join_path(directory, CODEC_METADATA_FILE);
    if (!is_regular_file(rotary_path) || !is_regular_file(position_path) ||
        !is_regular_file(metadata_path)) {
        printf("Codec directory must contain %s, %s and %s: %s\n",
               CODEC_ROTARY_FILE, CODEC_POSITION_FILE, CODEC_METADATA_FILE,
               directory_path);
        return -1;
    }

    DIR* directory_handle = opendir(directory_path);
    if (directory_handle == NULL) {
        printf("Failed to open Codec directory: %s\n", directory_path);
        return -1;
    }

    std::vector<std::string> all_canvases;
    struct dirent* entry = NULL;
    while ((entry = readdir(directory_handle)) != NULL) {
        const std::string name(entry->d_name);
        if (name.compare(0, 7, "canvas_") != 0 ||
            !is_supported_image(name)) {
            continue;
        }
        const std::string path = join_path(directory, name);
        if (is_regular_file(path)) {
            all_canvases.push_back(path);
        }
    }
    closedir(directory_handle);

    std::sort(all_canvases.begin(), all_canvases.end());
    *discovered_canvases = all_canvases.size();
    if (all_canvases.empty() || all_canvases.size() > MAX_INPUT_FRAMES) {
        printf("Codec directory requires 1..%zu canvas images, got %zu: %s\n",
               MAX_INPUT_FRAMES, all_canvases.size(), directory_path);
        return -1;
    }
    canvas_paths->swap(all_canvases);
    return 0;
}

static int read_visual_input(const char* input_path,
                             const std::string& requested_mode,
                             std::vector<std::string>* visual_paths,
                             size_t* discovered_visuals,
                             std::string* resolved_mode)
{
    if (input_path == NULL || visual_paths == NULL ||
        discovered_visuals == NULL || resolved_mode == NULL) {
        return -1;
    }

    struct stat input_stat;
    if (stat(input_path, &input_stat) != 0) {
        printf("Visual input does not exist: %s\n", input_path);
        return -1;
    }
    std::string mode = requested_mode;
    if (mode == "auto") {
        if (S_ISREG(input_stat.st_mode)) {
            mode = "image";
        } else if (S_ISDIR(input_stat.st_mode) &&
                   is_regular_file(join_path(input_path, CODEC_ROTARY_FILE))) {
            mode = "codec";
        } else if (S_ISDIR(input_stat.st_mode)) {
            mode = "frame";
        }
    }

    if (mode == "image") {
        if (!S_ISREG(input_stat.st_mode)) {
            printf("image mode requires an image file: %s\n", input_path);
            return -1;
        }
        if (!is_supported_image(input_path)) {
            printf("Unsupported image file: %s\n", input_path);
            return -1;
        }
        visual_paths->assign(1, input_path);
        *discovered_visuals = 1;
        *resolved_mode = mode;
        return 0;
    }
    if (!S_ISDIR(input_stat.st_mode)) {
        printf("%s mode requires a directory: %s\n", mode.c_str(),
               input_path);
        return -1;
    }

    int ret = -1;
    if (mode == "frame") {
        ret = read_frame_directory(input_path, visual_paths,
                                   discovered_visuals);
    } else if (mode == "codec") {
        ret = read_codec_directory(input_path, visual_paths,
                                   discovered_visuals);
    } else {
        printf("Unsupported input mode: %s\n", mode.c_str());
        return -1;
    }
    if (ret == 0) {
        *resolved_mode = mode;
    }
    return ret;
}

static int load_float16_file(const std::string& path, size_t expected_elements,
                             std::vector<float16>* values)
{
    if (values == NULL) {
        return -1;
    }
    std::ifstream input(path.c_str(), std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        printf("Failed to open FP16 file: %s\n", path.c_str());
        return -1;
    }
    const std::streamoff actual_bytes = input.tellg();
    const size_t expected_bytes = expected_elements * sizeof(float16);
    if (actual_bytes < 0 || static_cast<size_t>(actual_bytes) != expected_bytes) {
        printf("FP16 file size mismatch: %s got=%lld expected=%zu\n",
               path.c_str(), static_cast<long long>(actual_bytes),
               expected_bytes);
        return -1;
    }
    values->resize(expected_elements);
    input.seekg(0, std::ios::beg);
    input.read(reinterpret_cast<char*>(values->data()), actual_bytes);
    if (!input) {
        printf("Failed to read FP16 file: %s\n", path.c_str());
        values->clear();
        return -1;
    }
    return 0;
}

static std::string make_multimodal_prompt(size_t visual_count,
                                          const std::string& input_mode,
                                          const char* prompt)
{
    if (input_mode == "image") {
        return "<image> " + std::string(prompt);
    }

    std::string result;
    const char* label = input_mode == "codec" ? "Canvas" : "Frame";
    for (size_t i = 0; i < visual_count; ++i) {
        result += std::string(label) + " " + std::to_string(i + 1) +
                  ": <image>\n";
    }
    result += prompt;
    return result;
}


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc < 11)
    {
        printf("%s <vision_model_path> <vision_weight_path> <llm_model_path> <llm_weight_path> <tokenizer_path> <embedding_path> <vision_core_mask> <llm_core_mask> <image_or_asset_directory> <prompt> [model_width model_height] [--input-mode auto|image|frame|codec]\n", argv[0]);
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
    const char *visual_input_path  = argv[9];
    const char *prompt             = argv[10];
    uint32_t    model_width         = 0;
    uint32_t    model_height        = 0;
    std::string requested_input_mode = "auto";
    int argument_index = 11;
    if (argument_index < argc &&
        strcmp(argv[argument_index], "--input-mode") != 0) {
        if (argument_index + 1 >= argc) {
            printf("model_width and model_height must be provided together\n");
            return -1;
        }
        model_width = strtoul(argv[argument_index], nullptr, 0);
        model_height = strtoul(argv[argument_index + 1], nullptr, 0);
        argument_index += 2;
    }
    if (argument_index < argc) {
        if (argument_index + 1 >= argc ||
            strcmp(argv[argument_index], "--input-mode") != 0) {
            printf("Expected --input-mode auto|image|frame|codec\n");
            return -1;
        }
        requested_input_mode = argv[argument_index + 1];
        argument_index += 2;
    }
    if (argument_index != argc ||
        (requested_input_mode != "auto" &&
         requested_input_mode != "image" &&
         requested_input_mode != "frame" &&
         requested_input_mode != "codec")) {
        printf("Invalid arguments or input mode\n");
        return -1;
    }
    std::string prompt_with_images;
    std::vector<std::string> frame_paths;
    std::vector<float16> codec_rotary;
    size_t discovered_frames = 0;
    size_t rotary_elements_per_canvas = 0;
    std::string resolved_input_mode;

    int ret = 0;
    rknn_perf_metrics_t perf;
    memset(&perf, 0, sizeof(perf));
    if (read_visual_input(visual_input_path, requested_input_mode,
                          &frame_paths, &discovered_frames,
                          &resolved_input_mode) != 0) {
        return -1;
    }
    printf("Input mode: %s (requested: %s)\n", resolved_input_mode.c_str(),
           requested_input_mode.c_str());
    if (resolved_input_mode == "frame") {
        printf("Found %zu ordered images in %s; selected %zu frame(s)\n",
               discovered_frames, visual_input_path, frame_paths.size());
    } else if (resolved_input_mode == "codec") {
        printf("Loaded %zu precomputed Codec canvas(es) from %s\n",
               frame_paths.size(), visual_input_path);
    } else {
        printf("Loaded image: %s\n", visual_input_path);
    }


    // RKNN Context
    rknn_app_context_t rknn_app_ctx = {};
    rknn_app_ctx.model_width = model_width;
    rknn_app_ctx.model_height = model_height;

    // Tokenizer
    VocabInfo vocab_info;
    int max_special_ids = sizeof(vocab_info.special_eos_id) / sizeof(vocab_info.special_eos_id[0]);

    // Embedding
    EmbeddingInfo embedding_info = {};
    struct stat           emb_st;
    memset(&emb_st, 0x00, sizeof(emb_st));
    embedding_info.fd = -1;

    // LLM Param
    int n_params = 1;
    rknn3_llm_param params;
    memset(&params, 0, sizeof(rknn3_llm_param));

    // One input image is decoded at a time to keep host memory bounded.
    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));

    // Consecutive frame embeddings passed to the LLM as multiple images.
    size_t embed_elems = 1;
    size_t image_tokens = 0;
    std::unique_ptr<float16[]> frame_embeds;

    // LLM Multi Model Tensor
    int n_inputs = 1;
    rknn3_llm_multimodal_tensor tensor;
    memset(&tensor, 0, sizeof(rknn3_llm_multimodal_tensor));

    // Callback
    RKLLMCallback callback;
    memset(&callback, 0, sizeof(RKLLMCallback));
    ResultCallbackContext result_context;
    result_context.tokenizer = NULL;
    result_context.first_token = 0;
    result_context.first_decode = true;

    // Load Tokenizer
    std::unique_ptr<Tokenizer> tokenizer(new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path));
    if (!tokenizer)
    {
        printf("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
        ret = -1;
        goto out;
    }
    result_context.tokenizer = tokenizer.get();

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

    embedding_info.embedding_data = static_cast<float16*>(
        mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE,
             embedding_info.fd, 0));
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
    callback.result_userdata    = &result_context;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = tokenizer.get();
    callback.embed_callback     = embed_callback;
    callback.embed_userdata     = &embedding_info;

    ret = init_mage_vl_model(&rknn_app_ctx, llm_model_path, llm_weight_path, vision_model_path, vision_weight_path, &params, n_params, &callback, vision_core_mask, llm_core_mask);
    if (ret != 0)
    {
        printf("init_mage_vl_model fail! ret=%d llm_model_path=%s vision_model_path=%s\n", ret, llm_model_path, vision_model_path);
        goto out;
    }
    // All modes use the single-canvas Patch576 Vision model once per visual.
    for (size_t i = 0; i < rknn_app_ctx.vision.embeds_ndims; i++)
    {
      embed_elems *= rknn_app_ctx.vision.embeds_shape[i];
    }
    if (rknn_app_ctx.vision.embeds_ndims == 2) {
        image_tokens = rknn_app_ctx.vision.embeds_shape[0];
    } else if (rknn_app_ctx.vision.embeds_ndims == 3 &&
               rknn_app_ctx.vision.embeds_shape[0] == 1) {
        image_tokens = rknn_app_ctx.vision.embeds_shape[1];
    } else {
        printf("Demo requires one canvas per Vision run\n");
        ret = -1;
        goto out;
    }
    if (embed_elems != rknn_app_ctx.vision.outputs[0].attr->n_elems) {
        printf("Vision output shape and element count do not match\n");
        ret = -1;
        goto out;
    }
    if (rknn_app_ctx.vision.patch_num !=
        rknn_app_ctx.vision.model_width / MAGE_VL_PATCH_SIZE *
        rknn_app_ctx.vision.model_height / MAGE_VL_PATCH_SIZE) {
        printf("Demo requires a single-canvas Vision model\n");
        ret = -1;
        goto out;
    }
    if (resolved_input_mode == "codec") {
        rotary_elements_per_canvas =
            rknn_app_ctx.vision.inputs[
                rknn_app_ctx.vision.rotary_pos_emb_index].attr->n_elems;
        const std::string rotary_path =
            join_path(visual_input_path, CODEC_ROTARY_FILE);
        if (load_float16_file(rotary_path,
                              frame_paths.size() *
                                  rotary_elements_per_canvas,
                              &codec_rotary) != 0) {
            ret = -1;
            goto out;
        }
        printf("Loaded Codec rotary tensor: [%zu, %zu] FP16 elements\n",
               frame_paths.size(), rotary_elements_per_canvas);
    }
    if (frame_paths.size() * image_tokens + MAX_NEW_TOKENS +
            PROMPT_TOKEN_RESERVE > MAX_CONTEXT_LEN) {
        printf("Visual input needs %zu visual tokens and exceeds ctx%d "
               "after output/prompt reserve\n",
               frame_paths.size() * image_tokens, MAX_CONTEXT_LEN);
        ret = -1;
        goto out;
    }

    frame_embeds.reset(
        new (std::nothrow) float16[frame_paths.size() * embed_elems]);
    if (!frame_embeds) {
        printf("allocate frame_embeds failed\n");
        ret = -1;
        goto out;
    }

    perf.vision_latency = 0;
    for (size_t frame_index = 0; frame_index < frame_paths.size(); ++frame_index) {
        memset(&src_image, 0, sizeof(src_image));
        ret = read_image(frame_paths[frame_index].c_str(), &src_image);
        if (ret != 0) {
            printf("read frame fail! ret=%d frame=%zu path=%s\n", ret,
                   frame_index, frame_paths[frame_index].c_str());
            goto out;
        }
        ImageDataGuard source_image_guard(&src_image);

        printf("--> inference Mage-VL Vision %s %zu/%zu: %s\n",
               resolved_input_mode == "codec" ? "canvas" : "frame",
               frame_index + 1, frame_paths.size(),
               frame_paths[frame_index].c_str());
        int64_t vision_start = getCurrentTimeUs();
        if (resolved_input_mode == "codec") {
            ret = inference_mage_vl_vision_with_rotary(
                &(rknn_app_ctx.vision), &src_image,
                codec_rotary.data() +
                    frame_index * rotary_elements_per_canvas,
                rotary_elements_per_canvas,
                frame_embeds.get() + frame_index * embed_elems);
        } else {
            ret = inference_mage_vl_vision(
                &(rknn_app_ctx.vision), &src_image,
                frame_embeds.get() + frame_index * embed_elems);
        }
        perf.vision_latency += getCurrentTimeUs() - vision_start;

        if (ret != 0) {
            printf("inference Mage-VL Vision failed! ret=%d frame=%zu\n",
                   ret, frame_index);
            goto out;
        }
    }

    // LLM Input
    tensor.name           = "input_embeds";
    prompt_with_images = make_multimodal_prompt(
        frame_paths.size(), resolved_input_mode, prompt);
    tensor.prompt         = prompt_with_images.c_str();
    tensor.image.image_embed    = frame_embeds.get();
    tensor.image.n_image_tokens = image_tokens;
    tensor.image.n_image        = frame_paths.size();
    tensor.image.image_width    = rknn_app_ctx.vision.model_width;
    tensor.image.image_height   = rknn_app_ctx.vision.model_height;
    tensor.image.image_start      = "<|vision_start|>";
    tensor.image.image_end        = "<|vision_end|>";
    tensor.image.image_content    = "<|image_pad|>";
    tensor.enable_thinking = false;

    printf("--> inference Mage-VL LLM in %s mode with %zu visual(s) and "
           "%zu visual tokens\n", resolved_input_mode.c_str(),
           frame_paths.size(), frame_paths.size() * image_tokens);
    ret = inference_mage_vl_llm(&(rknn_app_ctx.llm), tensor, n_inputs, &perf);
    if (ret != 0)
    {
        printf("inference mage_vl model fail! ret=%d\n", ret);
        goto out;
    }

    printf_perf(&perf, frame_paths.size(), &result_context);
out:
    int release_ret = release_mage_vl_model(&rknn_app_ctx);

    if (release_ret != 0)
    {
        printf("release mage_vl model fail! ret=%d\n", release_ret);
        if (ret == 0) {
            ret = release_ret;
        }
    }

    if (embedding_info.fd != -1) {
        if (embedding_info.embedding_data != MAP_FAILED && embedding_info.embedding_data != NULL) {
            munmap(embedding_info.embedding_data, emb_st.st_size);
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
