// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "Tokenizer.h"
#include "image_drawing.h"
#include "image_utils.h"
#include "locate_anything.h"
#include "time_utils.h"

static int64_t first_token = 0;
static bool first_decode = true;
static std::string g_generated_text;

struct locate_box {
    int x1;
    int y1;
    int x2;
    int y2;
    std::string label;
};

struct locate_point {
    int x;
    int y;
    std::string label;
};

struct embedding_info {
    int fd;
    float16* embedding_data;
    int embedding_dim;
    int vocab_size;
};

const rknn3_sampling_params SAMPLE_PARAMS = {
    .top_k = 1,
    .top_p = 0.9f,
    .temperature = 0.000001f,
    .repeat_penalty = 1.05f,
    .frequency_penalty = 0.0f,
    .presence_penalty = 0.0f,
};

const char* system_prompt = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>";
const char* prompt_prefix = "<|im_start|>user\n<image 1>";
const char* prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n";

static std::string locate_anything_token_to_piece(Tokenizer* tokenizer, int32_t token)
{
    // LocateAnything encodes refs/boxes/coordinates as special tokens.  The
    // generic llama tokenizer may suppress those control tokens during
    // detokenization, so render them explicitly for human-readable detection
    // output such as <ref>person</ref><box><43><475><137><865></box>.
    switch (token) {
    case 151668:
        return "<box>";
    case 151669:
        return "</box>";
    case 151670:
        return "<quad>";
    case 151671:
        return "</quad>";
    case 151672:
        return "<ref>";
    case 151673:
        return "</ref>";
    case 151674:
        return "<interval>";
    case 151675:
        return "</interval>";
    default:
        break;
    }
    if (token >= 151677 && token <= 152677) {
        return "<" + std::to_string(token - 151677) + ">";
    }
    return tokenizer->TokenToPiece(token);
}

static int result_callback(void* userdata, RKLLMResult* result, LLMCallState state)
{
    Tokenizer* tokenizer = (Tokenizer*)userdata;

    if (state == RKLLM_RUN_ERROR) {
        printf("\n\nError occurred during inference\n");
        return 0;
    } else if (state == RKLLM_RUN_FINISH) {
        printf("\n\n--------------------Finished--------------------\n");
        return 0;
    } else if (state == RKLLM_RUN_WAITING) {
        printf("\n\nWaiting for UTF-8 encoded character\n");
        return 0;
    } else if (state == RKLLM_RUN_MAX_NEW_TOKEN_REACHED) {
        printf("\n\n--------------Max new token reached-------------\n");
        return 0;
    } else if (state == RKLLM_RUN_STOP) {
        printf("\n\n-----------------------Stop---------------------\n");
        return 0;
    } else if (state == RKLLM_RUN_NORMAL) {
        std::string piece;
        for (int32_t i = 0; i < result->num_tokens; i++) {
            piece += locate_anything_token_to_piece(tokenizer, result->token_ids[i]);
        }
        printf("%s", piece.c_str());
        g_generated_text += piece;
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
        printf("tokenizer failed for %s\n", text);
    }
    return n_tokens;
}

static int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len)
{
    embedding_info* embed_info = (embedding_info*)userdata;
    if (len != num_tokens * embed_info->embedding_dim * sizeof(float16)) {
        printf("invalid embed buffer len=%lu, num_tokens=%lu, embedding_dim=%d\n",
               (unsigned long)len, (unsigned long)num_tokens, embed_info->embedding_dim);
        return -1;
    }

    for (uint64_t n = 0; n < num_tokens; n++) {
        int32_t token = tokens[n];
        if (token < 0 || token >= embed_info->vocab_size) {
            printf("token id out of range: %d (vocab_size=%d)\n", token, embed_info->vocab_size);
            return -1;
        }
        memcpy((unsigned char*)embed + n * embed_info->embedding_dim * sizeof(float16),
               embed_info->embedding_data + token * embed_info->embedding_dim,
               embed_info->embedding_dim * sizeof(float16));
    }
    return 0;
}

static int clamp_int(int value, int min_value, int max_value)
{
    return std::max(min_value, std::min(value, max_value));
}

static bool parse_coord_token(const std::string& text, size_t* pos, int* value)
{
    if (*pos >= text.size() || text[*pos] != '<') {
        return false;
    }

    size_t num_begin = *pos + 1;
    size_t num_end = num_begin;
    while (num_end < text.size() && text[num_end] >= '0' && text[num_end] <= '9') {
        num_end++;
    }
    if (num_end == num_begin || num_end >= text.size() || text[num_end] != '>') {
        return false;
    }

    *value = atoi(text.substr(num_begin, num_end - num_begin).c_str());
    *pos = num_end + 1;
    return true;
}

// LocateAnything emits <ref>label</ref><box>coords</box>; find the nearest <ref>...</ref> before box_pos.
static std::string extract_label(const std::string& answer, size_t box_pos)
{
    size_t ref_end = answer.rfind("</ref>", box_pos);
    if (ref_end == std::string::npos) {
        return "";
    }
    size_t ref_begin = answer.rfind("<ref>", ref_end);
    if (ref_begin == std::string::npos) {
        return "";
    }
    size_t label_start = ref_begin + strlen("<ref>");
    if (label_start > ref_end) {
        return "";
    }
    return answer.substr(label_start, ref_end - label_start);
}

static void parse_locate_output(const std::string& answer,
                                std::vector<locate_box>* boxes,
                                std::vector<locate_point>* points)
{
    size_t pos = 0;
    while ((pos = answer.find("<box>", pos)) != std::string::npos) {
        size_t coord_pos = pos + strlen("<box>");
        int coords[4] = {0, 0, 0, 0};
        int coord_count = 0;
        while (coord_count < 4 && parse_coord_token(answer, &coord_pos, &coords[coord_count])) {
            coord_count++;
        }

        if (answer.compare(coord_pos, strlen("</box>"), "</box>") == 0) {
            std::string label = extract_label(answer, pos);
            if (coord_count == 4) {
                locate_box box;
                box.x1 = coords[0];
                box.y1 = coords[1];
                box.x2 = coords[2];
                box.y2 = coords[3];
                box.label = label;
                boxes->push_back(box);
            } else if (coord_count == 2) {
                locate_point point;
                point.x = coords[0];
                point.y = coords[1];
                point.label = label;
                points->push_back(point);
            }
            pos = coord_pos + strlen("</box>");
        } else {
            pos += strlen("<box>");
        }
    }
}

static int norm_to_pixel(int value, int size)
{
    return clamp_int((value * size + 500) / 1000, 0, size - 1);
}

static const unsigned int CATEGORY_PALETTE[] = {
    0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFF00, 0xFFFF00FF,
    0xFF00FFFF, 0xFFFFA500, 0xFF800080, 0xFF008000, 0xFF000080,
    0xFF800000, 0xFF808000, 0xFF008080, 0xFFFF1493, 0xFF4B0082,
    0xFFD2691E, 0xFF2E8B57, 0xFFDC143C, 0xFF8A2BE2, 0xFFFFD700,
};
static const size_t PALETTE_SIZE = sizeof(CATEGORY_PALETTE) / sizeof(CATEGORY_PALETTE[0]);

static unsigned int pick_category_color(std::map<std::string, unsigned int>* color_map,
                                        const std::string& label)
{
    auto it = color_map->find(label);
    if (it != color_map->end()) {
        return it->second;
    }
    unsigned int color = CATEGORY_PALETTE[color_map->size() % PALETTE_SIZE];
    (*color_map)[label] = color;
    return color;
}

static int save_visualization(const char* img_path, const std::string& answer, int image_width, int image_height)
{
    std::vector<locate_box> boxes;
    std::vector<locate_point> points;
    parse_locate_output(answer, &boxes, &points);
    if (boxes.empty() && points.empty()) {
        printf("No boxes or points found in LocateAnything output, skip visualization.\n");
        return 0;
    }

    std::map<std::string, unsigned int> category_colors;

    image_buffer_t vis_image;
    memset(&vis_image, 0, sizeof(vis_image));
    int ret = read_image(img_path, &vis_image);
    if (ret != 0) {
        printf("read image for visualization failed! ret=%d image_path=%s\n", ret, img_path);
        return ret;
    }

    for (size_t i = 0; i < boxes.size(); i++) {
        int x1 = norm_to_pixel(boxes[i].x1, image_width);
        int y1 = norm_to_pixel(boxes[i].y1, image_height);
        int x2 = norm_to_pixel(boxes[i].x2, image_width);
        int y2 = norm_to_pixel(boxes[i].y2, image_height);
        int left = std::min(x1, x2);
        int top = std::min(y1, y2);
        int right = std::max(x1, x2);
        int bottom = std::max(y1, y2);
        int width = std::max(1, right - left);
        int height = std::max(1, bottom - top);
        unsigned int color = pick_category_color(&category_colors, boxes[i].label);
        draw_rectangle(&vis_image, left, top, width, height, color, 3);
        if (!boxes[i].label.empty()) {
            int text_y = std::max(0, top - 20);
            draw_text(&vis_image, boxes[i].label.c_str(), left, text_y, color, 14);
        }
    }

    for (size_t i = 0; i < points.size(); i++) {
        int x = norm_to_pixel(points[i].x, image_width);
        int y = norm_to_pixel(points[i].y, image_height);
        unsigned int color = pick_category_color(&category_colors, points[i].label);
        draw_circle(&vis_image, x, y, 4, color, -1);
        if (!points[i].label.empty()) {
            int text_y = std::max(0, y - 10);
            draw_text(&vis_image, points[i].label.c_str(), x, text_y, color, 14);
        }
    }

    const char* out_path = "locate_anything_vis.jpg";
    ret = write_image(out_path, &vis_image);
    if (ret == 0) {
        printf("Saved visualization: %s (boxes=%zu points=%zu categories=%zu)\n",
               out_path, boxes.size(), points.size(), category_colors.size());
    } else {
        printf("write visualization failed! ret=%d path=%s\n", ret, out_path);
    }

    if (vis_image.virt_addr) {
        free(vis_image.virt_addr);
    }
    return ret;
}

static void printf_perf(rknn_perf_metrics_t* p)
{
    printf("\n--------------------------------------------------------------------------------------\n");
    printf(" %-12s  %-15s  %-8s  %-23s  %-23s\n",
           "Stage", "Total Time (ms)", "Tokens", "Time per Token (ms)", "Tokens per Second");
    printf("--------------------------------------------------------------------------------------\n");

    float ttft_us = (float)(first_token - p->llm_start_time);
    int prefill_n_tokens = p->n_prefill_tokens;
    float prefill_ms = ttft_us / 1000.0f;
    float prefill_tpt = prefill_n_tokens == 0 ? 0.0f : prefill_ms / prefill_n_tokens;
    float prefill_tps = prefill_ms == 0.0f ? 0.0f : 1000.0f / prefill_ms * prefill_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Prefill", prefill_ms, prefill_n_tokens, prefill_tpt, prefill_tps);

    float decode_time_us = (float)(p->llm_end_time - first_token);
    int decode_n_tokens = p->n_decode_tokens;
    float decode_ms = decode_time_us / 1000.0f;
    float decode_tpt = decode_n_tokens == 0 ? 0.0f : decode_ms / decode_n_tokens;
    float decode_tps = decode_ms == 0.0f ? 0.0f : 1000.0f / decode_ms * decode_n_tokens;
    printf(" %-12s  %-15.2f  %-8d  %-23.2f  %-23.2f\n",
           "Generate", decode_ms, decode_n_tokens, decode_tpt, decode_tps);
    printf("--------------------------------------------------------------------------------------\n");
    printf(" Vision latency = %.2f ms, FPS = %.2f\n",
           (int)p->vision_latency / 1000.0f, 1000.0f * 1000.0f / (int)p->vision_latency);
}

int main(int argc, char** argv)
{
    if (argc != 11 && argc != 13) {
        printf("%s <vision_model_path> <vision_weight_path> <llm_model_path> <llm_weight_path> <tokenizer_path> <embedding_path> <vision_core_mask> <llm_core_mask> <image_path> <prompt> [model_width model_height]\n", argv[0]);
        printf("example prompt: \"Locate all the instances that matches the following description: person</c>car.\"\n");
        return -1;
    }

    const char* vision_model_path = argv[1];
    const char* vision_weight_path = argv[2];
    const char* llm_model_path = argv[3];
    const char* llm_weight_path = argv[4];
    const char* tokenizer_path = argv[5];
    const char* embedding_path = argv[6];
    uint32_t vision_core_mask = strtoul(argv[7], NULL, 16);
    uint32_t llm_core_mask = strtoul(argv[8], NULL, 16);
    const char* img_path = argv[9];
    const char* prompt = argv[10];
    uint32_t model_width = 0;
    uint32_t model_height = 0;
    if (argc == 13) {
        model_width = strtoul(argv[11], NULL, 0);
        model_height = strtoul(argv[12], NULL, 0);
    }

    int ret = 0;
    rknn_app_context_t app_ctx;
    memset(&app_ctx, 0, sizeof(app_ctx));
    app_ctx.model_width = model_width;
    app_ctx.model_height = model_height;

    Tokenizer* tokenizer = NULL;
    VocabInfo vocab_info;
    memset(&vocab_info, 0, sizeof(vocab_info));

    embedding_info emb_info;
    struct stat emb_st;
    memset(&emb_info, 0, sizeof(emb_info));
    memset(&emb_st, 0, sizeof(emb_st));
    emb_info.fd = -1;

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(src_image));

    float16* img_embeds = NULL;
    size_t embed_elems = 1;
    rknn_perf_metrics_t perf;
    memset(&perf, 0, sizeof(perf));

    rknn3_llm_param params;
    memset(&params, 0, sizeof(params));
    RKLLMCallback callback;
    memset(&callback, 0, sizeof(callback));

    rknn3_llm_multimodal_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    int n_inputs = 1;
    std::string prompt_with_image;
    int vis_width = 0;
    int vis_height = 0;

    tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
    if (!tokenizer) {
        printf("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
        goto out;
    }
    tokenizer->GetVocabInfo(&vocab_info);
    printf("vocab_info: vocab_size=%d\n", vocab_info.vocab_size);

    emb_info.fd = open(embedding_path, O_RDONLY);
    if (emb_info.fd == -1) {
        printf("failed to open embedding file: %s\n", embedding_path);
        goto out;
    }
    if (fstat(emb_info.fd, &emb_st) == -1) {
        printf("failed to get embedding file size\n");
        goto out;
    }
    emb_info.embedding_data = (float16*)mmap(NULL, emb_st.st_size, PROT_READ, MAP_PRIVATE, emb_info.fd, 0);
    if (emb_info.embedding_data == MAP_FAILED) {
        printf("failed to mmap embedding file\n");
        emb_info.embedding_data = NULL;
        goto out;
    }
    emb_info.vocab_size = vocab_info.vocab_size;
    emb_info.embedding_dim = (emb_st.st_size / vocab_info.vocab_size) / sizeof(float16);
    printf("embedding_dim=%d\n", emb_info.embedding_dim);

    static char logits_name[] = "logits";
    params.logits_name = logits_name;
    params.max_context_len = 0;
    params.sampling_param = SAMPLE_PARAMS;
    params.vocab_info.vocab_size = vocab_info.vocab_size;
    params.vocab_info.n_special_eos_id = vocab_info.n_special_eos_id;
    params.vocab_info.n_special_bos_id = vocab_info.n_special_bos_id;
    memcpy(params.vocab_info.special_eos_id, vocab_info.special_eos_id, sizeof(vocab_info.special_eos_id));
    memcpy(params.vocab_info.special_bos_id, vocab_info.special_bos_id, sizeof(vocab_info.special_bos_id));
    params.vocab_info.linefeed_id = vocab_info.linefeed_id;

    callback.result_callback = result_callback;
    callback.result_userdata = tokenizer;
    callback.tokenizer_callback = tokenizer_callback;
    callback.tokenizer_userdata = tokenizer;
    callback.embed_callback = embed_callback;
    callback.embed_userdata = &emb_info;

    ret = init_locate_anything_model(&app_ctx,
                                     llm_model_path,
                                     llm_weight_path,
                                     vision_model_path,
                                     vision_weight_path,
                                     &params,
                                     1,
                                     callback,
                                     vision_core_mask,
                                     llm_core_mask);
    if (ret != 0) {
        printf("init LocateAnything model failed! ret=%d\n", ret);
        goto out;
    }

    for (uint32_t i = 0; i < app_ctx.vision.embeds_ndims; i++) {
        embed_elems *= app_ctx.vision.embeds_shape[i];
    }
    img_embeds = (float16*)malloc(embed_elems * sizeof(float16));
    if (!img_embeds) {
        printf("malloc image embeds failed! elems=%zu\n", embed_elems);
        goto out;
    }

    ret = read_image(img_path, &src_image);
    if (ret != 0) {
        printf("read image failed! ret=%d image_path=%s\n", ret, img_path);
        goto out;
    }
    vis_width = src_image.width;
    vis_height = src_image.height;

    tensor.name = "input_embeds";
    prompt_with_image = "<image>" + std::string(prompt);
    tensor.prompt = prompt_with_image.c_str();
    tensor.image.image_embed = img_embeds;
    if (app_ctx.vision.embeds_ndims == 2) {
        tensor.image.n_image_tokens = app_ctx.vision.embeds_shape[0];
        tensor.image.n_image = 1;
    } else {
        tensor.image.n_image_tokens = app_ctx.vision.embeds_shape[1];
        tensor.image.n_image = app_ctx.vision.embeds_shape[0];
    }
    tensor.image.image_width = app_ctx.vision.model_width;
    tensor.image.image_height = app_ctx.vision.model_height;
    tensor.image.image_start = "<img>";
    tensor.image.image_end = "</img>";
    tensor.image.image_content = "<IMG_CONTEXT>";
    tensor.enable_thinking = false;

    printf("image tokens=%lu, prompt=%s\n", (unsigned long)tensor.image.n_image_tokens, prompt);
    first_decode = true;
    g_generated_text.clear();
    ret = inference_locate_anything_model(&app_ctx, &src_image, img_embeds, tensor, n_inputs, &perf);
    if (ret != 0) {
        printf("inference LocateAnything model failed! ret=%d\n", ret);
        goto out;
    }
    save_visualization(img_path, g_generated_text, vis_width, vis_height);
    printf_perf(&perf);

out:
    release_locate_anything_model(&app_ctx);
    if (emb_info.fd != -1) {
        if (emb_info.embedding_data) {
            munmap((void*)emb_info.embedding_data, emb_st.st_size);
            emb_info.embedding_data = NULL;
        }
        close(emb_info.fd);
        emb_info.fd = -1;
    }
    if (src_image.virt_addr) {
        free(src_image.virt_addr);
    }
    if (img_embeds) {
        free(img_embeds);
    }
    if (tokenizer) {
        delete tokenizer;
    }
    return ret;
}
