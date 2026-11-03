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
#include <stdlib.h>
#include <string.h>
#include <memory>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

#include "hy_ocr_vl.h"
#include "image_utils.h"
#include "time_utils.h"

struct result_callback_context
{
    Tokenizer* tokenizer;
    int64_t    first_token;
    bool       first_decode;
};

struct embedding_info
{
    int      fd;
    float16* embedding_data;
    int      embedding_dim;
    int      vocab_size;
    float16* img_embeds;         // vision encoder 的全部输出
    int      img_n_tokens;       // vision encoder 输出 token 数
    int      img_token_id;       // image_content 的 token id (120120)
    int      img_idx;            // 跨 callback 调用累计的 image embedding 索引
};

struct free_deleter
{
    void operator()(void* ptr) const
    {
        free(ptr);
    }
};

// 采样参数参考 HunyuanOCR generation_config (v1.0):
//   top_k=1, top_p=1.0, temperature=0.0, repetition_penalty=1.08
// top_k=1 即贪心解码; repetition_penalty=1.08 与 Python 推理脚本一致
const rknn3_sampling_params SAMPLE_PARAMS = {
    .top_k = 1,
    .top_p = 1.0f,
    .temperature = 0.0f,
    .repeat_penalty = 1.08f,
    .frequency_penalty = 0.0f,
    .presence_penalty = 0.0f
};

// HunyuanOCR-1.5 图像相关特殊 token (vocab id 120118/120120/120119)
const char* IMAGE_START_TAG = "<｜hy_place▁holder▁no▁100｜>"; // 120118 image_start
const char* IMAGE_TOKEN_TAG = "<｜hy_place▁holder▁no▁102｜>"; // 120120 image_content
const char* IMAGE_END_TAG   = "<｜hy_place▁holder▁no▁101｜>"; // 120119 image_end

/*-------------------------------------------
                Callback Function
-------------------------------------------*/
int result_callback(void *userdata, RKLLMResult *result, LLMCallState state)
{
    result_callback_context* context = (result_callback_context*)userdata;
    if (!context || !context->tokenizer) {
        printf("invalid result callback context\n");
        return -1;
    }
    Tokenizer *tokenizer = context->tokenizer;

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

        if (context->first_decode) {
            context->first_token = getCurrentTimeUs();
            context->first_decode = false;
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

    if (!embed_info || !tokens || !embed || !embed_info->embedding_data ||
        embed_info->embedding_dim <= 0 || embed_info->vocab_size <= 0) {
        printf("invalid embedding callback arguments\n");
        return -1;
    }

    if (len != num_tokens * embed_info->embedding_dim * sizeof(float16)) {
        printf("invalid embed buffer: len=%lu, expected=%lu (num_tokens=%lu, embedding_dim=%d)\n",
               (unsigned long)len, (unsigned long)(num_tokens * embed_info->embedding_dim * sizeof(float16)),
               (unsigned long)num_tokens, embed_info->embedding_dim);
        return -1;
    }

    unsigned char* dst = (unsigned char*)embed;
    uint64_t stride = (uint64_t)embed_info->embedding_dim * sizeof(float16);

    for (uint64_t n = 0; n < num_tokens; n++) {
        if (embed_info->img_embeds && tokens[n] == embed_info->img_token_id) {
            if (embed_info->img_idx < 0 || embed_info->img_idx >= embed_info->img_n_tokens) {
                printf("image embedding index out of range: index=%d, count=%d\n",
                       embed_info->img_idx, embed_info->img_n_tokens);
                return -1;
            }
            memcpy(dst + n * stride,
                   embed_info->img_embeds + (uint64_t)embed_info->img_idx * embed_info->embedding_dim,
                   stride);
            embed_info->img_idx++;
        } else {
            if (tokens[n] < 0 || tokens[n] >= embed_info->vocab_size) {
                printf("token id out of range: token=%d, vocab_size=%d\n",
                       tokens[n], embed_info->vocab_size);
                return -1;
            }
            memcpy(dst + n * stride,
                   embed_info->embedding_data + (uint64_t)tokens[n] * embed_info->embedding_dim,
                   stride);
        }
    }

    return 0;
}

// XD-RoPE (HunyuanVL) 4 轴位置编码上下文, 语义参考 modeling_hunyuan_vl.py 的 get_rope_index:
//   轴0: 1D 序列位置 (所有 token)
//   轴1: 宽度坐标 (图像 token 为所在列 c, 文本 token 为序列位置)
//   轴2: 高度坐标 (图像 token 为所在行 r, 文本 token 为序列位置)
//   轴3: 图像序号 (图像 token 为图像在序列中的序号, 文本 token 为序列位置)
typedef struct {
    int32_t content_token_id; // repeated image placeholder id (120120)
    int     image_pw;         // 图像 token 网格列数 (含 newline 列 = pw+1)
    int     image_index;      // 当前图像序号 (从 0 开始)
    int     img_t;            // 跨 chunk 累计的 vision token 计数
    int     total_vision_tokens; // vision 输出数（首尾为 image_begin/image_end）
} xd_rope_ctx_t;

static int input_callback(void* userdata, rknn3_tensor* input_tensors, uint32_t n_input_tensors,
                          LLMInputCallbackParam param)
{
    xd_rope_ctx_t* ctx = (xd_rope_ctx_t*)userdata;

    for (uint32_t i = 0; i < n_input_tensors; ++i) {
        if (strcmp(input_tensors[i].attr->name, "xd_rope_id_input") != 0) {
            continue;
        }

        // xd_rope_id_input shape: [num_tokens, 4] (4D position for HunyuanVL XD-RoPE)
        // mrope_section = [16, 16, 16, 16], total 4 position dimensions
        int32_t* dst = (int32_t*)input_tensors[i].mem->virt_addr;
        int32_t base_pos = param.pos;

        // 新的一次推理从 pos 0 开始, 重置图像 token 计数
        if (base_pos <= 0) {
            ctx->img_t = 0;
        }

        int img_count = 0;
        for (int t = 0; t < param.num_tokens; ++t) {
            int32_t pos = base_pos + t;
            dst[t * 4 + 0] = pos;
            if (ctx->image_pw > 0 && param.tokens[t] == ctx->content_token_id) {
                if (ctx->img_t < 0 || ctx->img_t >= ctx->total_vision_tokens) {
                    printf("vision position index out of range: index=%d, count=%d\n",
                           ctx->img_t, ctx->total_vision_tokens);
                    return -1;
                }
                // Python get_rope_index: the first and last image tokens use 1D
                // positions; tokens between them use 2D grid positions.
                if (ctx->img_t == 0 || ctx->img_t == ctx->total_vision_tokens - 1) {
                    dst[t * 4 + 1] = pos;
                    dst[t * 4 + 2] = pos;
                    dst[t * 4 + 3] = pos;
                } else {
                    int content_idx = ctx->img_t - 1;
                    dst[t * 4 + 1] = content_idx % ctx->image_pw;  // width
                    dst[t * 4 + 2] = content_idx / ctx->image_pw;  // height
                    dst[t * 4 + 3] = ctx->image_index;            // image index
                }
                ctx->img_t++;
                img_count++;
            } else {
                dst[t * 4 + 1] = pos;
                dst[t * 4 + 2] = pos;
                dst[t * 4 + 3] = pos;
            }
        }
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


void printf_perf(rknn_perf_metrics_t *p, int64_t first_token)
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


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 11 && argc != 13)
    {
        printf("Usage: %s <vision_model_path> <vision_weight_path> <llm_model_path> <llm_weight_path> <tokenizer_path> <embedding_path> <vision_core_mask> <llm_core_mask> <image_path> <prompt> [model_width] [model_height]\n", argv[0]);
        printf("Example:\n");
        printf("  %s HunyuanOCR-1.5-vision.rknn HunyuanOCR-1.5-vision.weight HunyuanOCR-1.5-llm.rknn HunyuanOCR-1.5-llm.weight HunyuanOCR-1.5-llm.tokenizer.gguf HunyuanOCR-1.5-llm.embed.bin 0xff 0xff demo.jpg \"请识别图片中的文字\" %d %d\n",
               argv[0], MODEL_WIDTH, MODEL_HEIGHT);
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
    if (argc == 13) {
        model_width  = strtoul(argv[11], nullptr, 0);
        model_height = strtoul(argv[12], nullptr, 0);
    }
    std::string prompt_with_image;

    int ret = 0;
    rknn_perf_metrics_t perf;
    memset(&perf, 0, sizeof(perf));
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
    std::unique_ptr<float16, free_deleter> img_embeds;

    // LLM Multi Model Tensor
    int n_inputs = 1;
    rknn3_llm_multimodal_tensor tensor;
    memset(&tensor, 0, sizeof(rknn3_llm_multimodal_tensor));

    xd_rope_ctx_t xd_rope_ctx;
    memset(&xd_rope_ctx, 0, sizeof(xd_rope_ctx));

    // Image token grid parameters (declared before any goto to avoid crossing initialization)
    const int patch_size = HY_OCR_PATCH_SIZE;
    const int spatial_merge_size = HY_OCR_MERGE_SIZE;
    int image_ph = 0;
    int image_pw = 0;
    int n_image_tokens = 0;
    int hidden_dim = 0;
    size_t n_embed_image = 0;

    // Callback
    RKLLMCallback callback;
    memset(&callback, 0, sizeof(RKLLMCallback));

    // Load Tokenizer
    std::unique_ptr<Tokenizer> tokenizer(new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path));
    if (!tokenizer)
    {
        printf("load tokenizer failed! tokenizer_path=%s\n", tokenizer_path);
        ret = -1;
        goto out;
    }
    
    tokenizer->GetVocabInfo(&(vocab_info));

    result_callback_context result_context;
    result_context.tokenizer = tokenizer.get();
    result_context.first_token = 0;
    result_context.first_decode = true;
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
    callback.input_callback     = input_callback;
    callback.input_userdata     = &xd_rope_ctx;

    // Read Image
    ret = read_image(img_path, &src_image);
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, img_path);
        goto out;
    }

    ret = init_hy_ocr_vl_model(&rknn_app_ctx, llm_model_path, llm_weight_path, vision_model_path, vision_weight_path, &params, n_params, &callback, vision_core_mask, llm_core_mask);
    if (ret != 0)
    {
        printf("init_hy_ocr_vl_model fail! ret=%d llm_model_path=%s vision_model_path=%s\n", ret, llm_model_path, vision_model_path);
        goto out;
    }

    // 从 model 尺寸计算 image token 网格
    image_ph = rknn_app_ctx.vision.model_height / patch_size / spatial_merge_size;
    image_pw = rknn_app_ctx.vision.model_width / patch_size / spatial_merge_size;
    n_image_tokens = image_ph * (image_pw + 1) + 2;

    // LLM Input
    tensor.name           = "input_embeds";
    // 手动构建 prompt，与 processor.replace_image_token 的结果一致：
    //   runtime 最终构造: <BOS><sep><image_start><image_token × N><image_end>{prompt_text}<User>
    //   其中 <BOS><sep> 由 prompt_prefix 添加, <User> 由 prompt_postfix 添加
    // N 等于 vision 输出数；首尾两个 image_token 分别接收 vision merger 的
    // image_begin/image_end embedding。外围 image_start/image_end 是文本模板包装 token。
    // 不使用 runtime 的 <image> 展开, 因为 runtime 要求 n_image_tokens == image_ph * image_pw,
    // 而 HunyuanOCR vision merger 输出含换行 token, n_image_tokens = ph*(pw+1)+2 ≠ ph*pw。
    // image token 的 embedding 注入由 embed_callback 处理 (与 Python masked_scatter 一致)。
    {
        prompt_with_image = IMAGE_START_TAG;
        for (int i = 0; i < n_image_tokens; i++) {
            prompt_with_image += IMAGE_TOKEN_TAG;
        }
        prompt_with_image += IMAGE_END_TAG;
        prompt_with_image += std::string(prompt);
    }
    tensor.prompt = prompt_with_image.c_str();

    // n_image = 0, n_image_tokens = 0: 不使用 runtime 的 image 展开,
    // image embedding 由 embed_callback 注入 (与 Python masked_scatter 一致)。
    // runtime 即使 n_image=0 也检查 n_image_tokens, 必须设为 0 避免报错。
    tensor.image.n_image        = 0;
    tensor.image.n_image_tokens = 0;
    tensor.image.image_embed    = NULL;
    tensor.enable_thinking = false;

    // Allocate image embeds buffer for all vision output tokens.
    hidden_dim = rknn_app_ctx.vision.embeds_shape[rknn_app_ctx.vision.embeds_ndims - 1];
    n_embed_image = (size_t)n_image_tokens * hidden_dim;

    // 填充 XD-RoPE callback 上下文: image token id + 网格尺寸
    {
        int32_t ids[4];
        int n_ids = tokenizer->Tokenize(IMAGE_TOKEN_TAG, (int32_t)strlen(IMAGE_TOKEN_TAG), ids, 4);
        if (n_ids != 1) {
            printf("image_content tag should be a single token, got %d tokens!\n", n_ids);
            ret = -1;
            goto out;
        }
        xd_rope_ctx.content_token_id = ids[0];
        xd_rope_ctx.image_pw         = image_pw + 1; // +1: newline 列
        xd_rope_ctx.image_index      = 0;
        xd_rope_ctx.img_t            = 0;
        xd_rope_ctx.total_vision_tokens = n_image_tokens;
    }

    img_embeds.reset((float16*)malloc(n_embed_image * sizeof(float16)));
    if (!img_embeds) {
        printf("malloc img_embeds fail!\n");
        ret = -1;
        goto out;
    }
    // embed_callback 配置: 替换 image token 为 vision encoder 输出
    embedding_info.img_embeds    = img_embeds.get();
    embedding_info.img_n_tokens  = n_image_tokens;
    embedding_info.img_token_id  = xd_rope_ctx.content_token_id;

    // The callback context may be reused by callers that run more than one session.
    embedding_info.img_idx = 0;
    ret = inference_hy_ocr_vl_model(&rknn_app_ctx, &src_image, img_embeds.get(), tensor, n_inputs, &perf);
    if (ret != 0)
    {
        printf("inference hy_ocr_vl model fail! ret=%d\n", ret);
        goto out;
    }

    printf_perf(&perf, result_context.first_token);
out:
    int release_ret = release_hy_ocr_vl_model(&rknn_app_ctx);

    if (release_ret != 0)
    {
        printf("release hy_ocr_vl model fail! ret=%d\n", release_ret);
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
