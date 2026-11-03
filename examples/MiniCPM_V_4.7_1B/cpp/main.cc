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
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <string>

#include "minicpmv4_7.h"
#include "image_utils.h"
#include "time_utils.h"

int64_t first_token;
bool first_decode = true;

const rknn3_sampling_params SAMPLE_PARAMS = {
    .top_k = 1,
    .top_p = 0.9,
    .temperature = 1.0f,
    .repeat_penalty = 1.2f,
    .frequency_penalty = 0.0f,
    .presence_penalty = 0.0f
};

const char* system_prompt  = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
const char* prompt_prefix  = "<|im_start|>user\n";
const char* prompt_postfix = "<|im_end|>\n<|im_start|>assistant\n<think>\n";

struct embedding_info
{
  int      fd;
  float16* embedding_data;
  int      embedding_dim;
  int      vocab_size;
};

// Slice mode geometry (rows/cols/crop_sq) is set by main() before inference:
// the canvas is rows*cols slices of crop_sq x crop_sq LLM tokens, plus the
// thumbnail which also occupies crop_sq x crop_sq.
struct input_cb_data
{
  /* Canvas geometry, set by main() before inference (crop_sq == 0 => disabled). */
  int32_t crop_sq;      // LLM grid side of ONE crop (8 for 64 tokens)
  int32_t rows;         // slice grid rows (1 when there are no slices)
  int32_t cols;         // slice grid cols
  int32_t n_slices;     // number of slice blocks (0 => thumbnail only)

  /* Absolute token index of <image>.  -1 until it shows up; kept across the
   * prefill chunks (the runtime calls us at most 128 tokens at a time). */
  int32_t canvas_base;
};

// MiniCPM-V-4.7 canvas special token ids.
// Fallback values; resolve_special_ids() overrides them from the tokenizer.
static int32_t IM_START_ID    = 248078; // <image>
static int32_t IM_END_ID      = 248079; // </image>
static int32_t SLICE_START_ID = 248088; // <slice>
static int32_t SLICE_END_ID   = 248089; // </slice>
static int32_t IMAGE_PAD_ID   = 248056; // <|image_pad|>

/* Resolve one special token to a single id; keep the fallback when the
 * tokenizer splits it or is unavailable. */
static int32_t resolve_special_id(Tokenizer* tokenizer, const char* tok, int32_t fallback)
{
    if (tokenizer == NULL) {
        return fallback;
    }
    int32_t ids[8];
    int n = tokenizer->Tokenize(tok, (int32_t)strlen(tok), ids, 8);
    if (n == 1) {
        return ids[0];
    }
    printf("[warn] tokenize(\"%s\") -> %d token(s), keep fallback id %d\n", tok, n, fallback);
    return fallback;
}

static void resolve_special_ids(Tokenizer* tokenizer)
{
    IM_START_ID    = resolve_special_id(tokenizer, "<image>",       IM_START_ID);
    IM_END_ID      = resolve_special_id(tokenizer, "</image>",      IM_END_ID);
    SLICE_START_ID = resolve_special_id(tokenizer, "<slice>",       SLICE_START_ID);
    SLICE_END_ID   = resolve_special_id(tokenizer, "</slice>",      SLICE_END_ID);
    IMAGE_PAD_ID   = resolve_special_id(tokenizer, "<|image_pad|>", IMAGE_PAD_ID);
    printf("[token ids] <image>=%d </image>=%d <slice>=%d </slice>=%d <|image_pad|>=%d\n",
           IM_START_ID, IM_END_ID, SLICE_START_ID, SLICE_END_ID, IMAGE_PAD_ID);
}


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

        if (first_decode)
        {
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

// embed_callback context: text embeddings from the mmap'ed table, plus vision
// features spliced in for the visual placeholders.
struct embed_ctx
{
  struct embedding_info* info;
  const float16* vis;        // vision embeds, crop order [thumbnail, slice0, ...]
  uint64_t       tokens_per_crop;
  int32_t        pad_id;

  // per-forward progress (reset before every prefill)
  int      in_image;         // inside <image>..</image>
  int      in_slice;         // inside <slice>..</slice>
  uint64_t span_pad;         // pads seen inside the current span
  uint64_t slice_idx;        // slice span index (0-based)
};

// Splice the vision features of one placeholder token.
//   <image> span  -> crop 0  (thumbnail; identical to what the image.* template writes)
//   <slice> span  -> crop 1 + slice_idx
// The decision is made purely from the marker tokens, so it does not depend on
// whether the runtime expands <image> itself or hands the pads to this callback.
static const float16* pick_visual_embed(struct embed_ctx* ec, int32_t tok)
{
    if (ec->vis == NULL || tok != ec->pad_id) {
        return NULL;
    }
    if (!ec->in_image && !ec->in_slice) {
        return NULL;                       // pad outside any visual span
    }
    const uint64_t crop  = ec->in_slice ? (1 + ec->slice_idx) : 0;
    const uint64_t off   = crop * ec->tokens_per_crop + ec->span_pad;
    if (ec->span_pad >= ec->tokens_per_crop) {
        printf("[embed] span overflow (crop=%llu pad=%llu per_crop=%llu)\n",
               (unsigned long long)crop, (unsigned long long)ec->span_pad,
               (unsigned long long)ec->tokens_per_crop);
        return NULL;
    }
    ec->span_pad++;
    return ec->vis + off * ec->info->embedding_dim;
}

int embed_callback(void* userdata, int32_t* tokens, uint64_t num_tokens, void* embed, uint64_t len)
{
    struct embed_ctx* ec = (struct embed_ctx*)userdata;
    struct embedding_info* embed_info = ec->info;
    const int dim = embed_info->embedding_dim;

    if (len != num_tokens * dim * sizeof(float16)) {
        printf("invalid embed buffer\n");
        return -1;
    }

    for (uint64_t n = 0; n < num_tokens; n++) {
        float16* dst = (float16*)((unsigned char*)embed + n * dim * sizeof(float16));
        const int32_t tok = tokens[n];

        if (tok == IM_START_ID)      { ec->in_image = 1; ec->span_pad = 0; }
        else if (tok == IM_END_ID)   { ec->in_image = 0; ec->span_pad = 0; }
        else if (tok == SLICE_START_ID) { ec->in_slice = 1; ec->span_pad = 0; }
        else if (tok == SLICE_END_ID)   { ec->in_slice = 0; ec->span_pad = 0; ec->slice_idx++; }

        const float16* vis = pick_visual_embed(ec, tok);
        if (vis != NULL) {
            memcpy(dst, vis, dim * sizeof(float16));
            continue;
        }

        if (tok >= 0 && tok < embed_info->vocab_size) {
            memcpy(dst, embed_info->embedding_data + (uint64_t)tok * dim, dim * sizeof(float16));
        } else {
            memset(dst, 0, dim * sizeof(float16));
        }
    }

    return 0;
}

static int input_callback(void* userdata, rknn3_tensor* input_tensors, uint32_t n_input_tensors,
                          LLMInputCallbackParam param)
{
    input_cb_data* ud = (input_cb_data*)userdata;
    const int32_t* tk = param.tokens;
    const int      N  = param.num_tokens;
    if (!ud || !tk || N <= 0) {
        return 0;
    }

    minicpmv4_7_canvas_geom_t gm;
    gm.im_start_id    = IM_START_ID;
    gm.im_end_id      = IM_END_ID;
    gm.slice_start_id = SLICE_START_ID;
    gm.slice_end_id   = SLICE_END_ID;
    gm.crop_sq        = ud->crop_sq;
    gm.rows           = ud->rows;
    gm.cols           = ud->cols;
    gm.n_slices       = ud->n_slices;
    gm.canvas_base    = ud->canvas_base;

    for (uint32_t i = 0; i < n_input_tensors; ++i) {
        rknn3_tensor_attr* a = input_tensors[i].attr;
        rknn3_tensor_mem*  m = input_tensors[i].mem;
        if (!a || !m || !m->virt_addr) {
            continue;
        }
        if (strstr(a->name, "mrope") == NULL) {
            continue; // leave runtime-managed tensors untouched
        }

        int32_t* dst = (int32_t*)m->virt_addr;
        const int cap = (int)(a->n_elems / 3); // capacity in (T,H,W) rows

        const int rc = minicpmv4_7_canvas_fill(tk, N, param.pos, &gm, dst, cap);

        if (rc != 0) {
            // No <image> seen yet: plain 1-D positions.  For a pure-text sequence the
            // position is just the absolute token index, and a text-only session keeps
            // going the same way.
            for (int k = 0; k < N && k < cap; ++k) {
                const int32_t p = param.pos + k;
                dst[3 * k + 0] = p;
                dst[3 * k + 1] = p;
                dst[3 * k + 2] = p;
            }
        }
    }

    ud->canvas_base = gm.canvas_base;   // keep <image> for later chunks / decode
    return 0;
}


void printf_perf(rknn_perf_metrics_t *p, bool show_vision) 
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
    
    if (show_vision) {
        printf(" Vision latency = %.2f ms, FPS = %.2f\n", 
               (int)p->vision_latency / 1000.f, 1000.f * 1000.f / (int)p->vision_latency);
    }
}


/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 11)
    {
        printf("%s <vision_model_path> <vision_weight_path> <llm_model_path> <llm_weight_path> <tokenizer_path> <embedding_path> <vision_core_mask> <llm_core_mask> <image_path> <prompt>\n", argv[0]);
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

    bool text_only = (vision_model_path[0] == '\0') || (vision_weight_path[0] == '\0') || (img_path[0] == '\0'); 

    int ret;
    rknn_perf_metrics_t perf;

    // RKNN Context
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    // Tokenizer
    Tokenizer* tokenizer;
    VocabInfo vocab_info;

    // Embedding
    struct embedding_info embedding_info;
    struct stat           emb_st;
    memset(&embedding_info, 0x00, sizeof(embedding_info));
    embedding_info.fd = -1;

    // Canvas mrope host state (input_callback userdata)
    input_cb_data input_cb_state;
    memset(&input_cb_state, 0x00, sizeof(input_cb_state));

    // Embed callback context (text embedding table + optional vision splice)
    struct embed_ctx emb_ctx;
    memset(&emb_ctx, 0x00, sizeof(emb_ctx));
    emb_ctx.info = &embedding_info;

    // Slice mode (image mode only)
    int          slice_rows = 0, slice_cols = 0, n_crops = 1, tokens_per_crop = 0;
    image_rect_t crop_boxes[MINICPMV4_7_MAX_CROPS];
    memset(crop_boxes, 0x00, sizeof(crop_boxes));

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

    std::string prompt_new = "<image>";
    prompt_new += prompt;

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

    // Resolve the canvas special token ids from the tokenizer, then publish the pad id to the embed callback.
    resolve_special_ids(tokenizer);
    emb_ctx.pad_id = IMAGE_PAD_ID;

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
    params.logits_name               = (char*)"logits";
    params.max_context_len           = MAX_CONTEXT_LEN;
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
    callback.embed_userdata     = &emb_ctx;
    callback.input_callback     = input_callback;
    callback.input_userdata     = &input_cb_state;

    memset(&perf, 0, sizeof(perf));

    if (text_only)
    {
        // 纯文本模式：只初始化 LLM（不启用用户管理 internal 内存），跳过视觉
        printf("[warning] text-only mode: vision model disabled\n");
        ret = init_minicpmv4_7_llm_text(&(rknn_app_ctx.llm), llm_model_path, llm_weight_path, &params, n_params, callback, llm_core_mask);
        if (ret != 0)
        {
            printf("init_minicpmv4_7_llm_text fail! ret=%d llm_model_path=%s\n", ret, llm_model_path);
            goto out;
        }

        // LLM Input（纯文本 prompt）
        rknn3_llm_tensor tensor_text;
        memset(&tensor_text, 0, sizeof(tensor_text));
        tensor_text.name           = "input_embeds";
        tensor_text.prompt         = prompt;
        tensor_text.embed          = NULL;
        tensor_text.tokens         = NULL;
        tensor_text.n_tokens       = 0;
        tensor_text.enable_thinking = false;

        printf("--> inference minicpmv4_7 llm model (text-only)\n");
        first_token = getCurrentTimeUs();
        memset(&input_cb_state, 0x00, sizeof(input_cb_state));
        input_cb_state.canvas_base = -1;   // no canvas: 1-D text positions
        ret = inference_minicpmv4_7_llm_text(&(rknn_app_ctx.llm), tensor_text, n_inputs, &perf);
        if (ret != 0)
        {
            printf("inference minicpmv4_7 llm text fail! ret=%d\n", ret);
            goto out;
        }

        printf_perf(&perf, false);
    }
    else
    {
        ret = init_minicpmv4_7_model(&rknn_app_ctx, llm_model_path, llm_weight_path, vision_model_path, vision_weight_path, &params, n_params, callback, vision_core_mask, llm_core_mask);
        if (ret != 0)
        {
            printf("init_minicpmv4_7_model fail! ret=%d llm_model_path=%s vision_model_path=%s\n", ret, llm_model_path, vision_model_path);
            goto out;
        }

        // Read Image
        ret = read_image(img_path, &src_image);
        if (ret != 0)
        {
            printf("read image fail! ret=%d image_path=%s\n", ret, img_path);
            goto out;
        }

        // ---- slice decision ----
        tokens_per_crop = (int)rknn_app_ctx.vision.embeds_shape[1];   // 64 for a 448x448 crop
        {
            const int scale_res = rknn_app_ctx.vision.model_width;   // 448
            const char* env = getenv("MINICPMV4_7_NO_SLICE");
            const bool no_slice = (env != NULL && std::string(env) != "0");
            if (!no_slice) {
                minicpmv4_7_get_sliced_grid(src_image.height, src_image.width,
                                            MINICPMV4_7_MAX_SLICE_NUMS, scale_res,
                                            &slice_rows, &slice_cols);
            } else {
                printf("[crop] MINICPMV4_7_NO_SLICE set -> single crop\n");
            }
            n_crops = minicpmv4_7_build_crop_boxes(src_image.height, src_image.width,
                                                   MINICPMV4_7_MAX_SLICE_NUMS, scale_res,
                                                   crop_boxes, MINICPMV4_7_MAX_CROPS);
            if (n_crops < 1) {
                printf("build crop boxes fail!\n");
                goto out;
            }
            if (no_slice) {
                slice_rows = slice_cols = 0;
                n_crops = 1;
            }
        }

        const uint64_t per_crop = minicpmv4_7_vision_embeds_per_crop(&rknn_app_ctx.vision);
        embed_elems = (size_t)per_crop * (size_t)n_crops;
        img_embeds  = (float16*)malloc(embed_elems * sizeof(float16));
        if (img_embeds == NULL) {
            printf("malloc vision embeds (%zu elem) fail!\n", embed_elems);
            goto out;
        }
        printf("[crop] image %dx%d -> thumbnail + %dx%d slices = %d crop(s), %d tok/crop\n",
               src_image.width, src_image.height, slice_rows, slice_cols, n_crops, tokens_per_crop);
        for (int i = 0; i < n_crops; ++i) {
            printf("[crop]  #%d rect [l=%d t=%d r=%d b=%d] %dx%d\n", i,
                   crop_boxes[i].left, crop_boxes[i].top, crop_boxes[i].right, crop_boxes[i].bottom,
                   crop_boxes[i].right - crop_boxes[i].left + 1,
                   crop_boxes[i].bottom - crop_boxes[i].top + 1);
        }

        // ---- prompt: thumbnail via the image.* template, slices as literal tokens ----
        // Slices within a row are adjacent, rows are joined by "\n".
        //
        // The thumbnail is emitted as a BARE "<image>" only: the runtime expands that
        // placeholder into "<image>" + n_image_tokens x image_content + "</image>"
        // (that is how the pre-slice example worked with prompt "<image>"+prompt).
        // Writing the pads here as well would make the runtime insert a second copy.
        //
        // <slice> / </slice> cannot be produced by the runtime (the tensor struct only
        // carries one start/end/content triple), so the slice blocks are literal text
        // and their embeddings are spliced by embed_callback.
        if (n_crops > 1) {
            std::string pad_str;
            pad_str.reserve(tokens_per_crop * 13);
            for (int i = 0; i < tokens_per_crop; ++i) {
                pad_str += "<|image_pad|>";
            }
            std::string ph = "<image>";          // expanded by the runtime (crop 0)
            for (int r = 0; r < slice_rows; ++r) {
                for (int c = 0; c < slice_cols; ++c) {
                    ph += "<slice>" + pad_str + "</slice>";
                }
                if (r + 1 < slice_rows) {
                    ph += "\n";
                }
            }
            prompt_new = ph + prompt;
        } else {
            prompt_new = "<image>";
            prompt_new += prompt;
        }

        // ---- canvas geometry handed to the mrope callback ----
        // crop_sq is the LLM grid side of ONE crop (thumbnail and slices alike);
        // every crop is resized to the model input, so all crops share it.
        {
            int sq = (int)llround(sqrt((double)tokens_per_crop));
            if (sq * sq != tokens_per_crop) {
                printf("[crop] tokens_per_crop=%d is not a square -> canvas mrope disabled\n",
                       tokens_per_crop);
                sq = 0;
            }
            const bool use_slices = (n_crops > 1 && sq > 0);
            input_cb_state.crop_sq     = sq;
            input_cb_state.n_slices    = use_slices ? (slice_rows * slice_cols) : 0;
            input_cb_state.rows        = use_slices ? slice_rows : 1;
            input_cb_state.cols        = use_slices ? slice_cols : 1;
            input_cb_state.canvas_base = -1;   // <image> not seen yet
            printf("[crop] canvas: crop_grid=%dx%d slices=%d canvas=%dx%d LLM tokens\n",
                   sq, sq, input_cb_state.n_slices,
                   input_cb_state.rows * sq, input_cb_state.cols * sq);
        }


        // LLM Input.  The thumbnail is crop 0 and is fed by the image.* template;
        // the remaining crops are spliced by embed_callback (which also writes the
        // same crop-0 data, so either mechanism winning yields identical embeds).
        tensor.name                 = "input_embeds";
        tensor.prompt               = prompt_new.c_str();
        tensor.image.image_embed    = img_embeds;
        tensor.image.n_image_tokens = (uint64_t)tokens_per_crop;
        tensor.image.n_image        = 1;
        tensor.image.image_width    = rknn_app_ctx.vision.model_width;
        tensor.image.image_height   = rknn_app_ctx.vision.model_height;
        tensor.image.image_start    = "<image>";
        tensor.image.image_end      = "</image>";
        tensor.image.image_content  = "<|image_pad|>";
        tensor.enable_thinking = false;

        emb_ctx.vis             = (n_crops > 1) ? img_embeds : NULL;
        emb_ctx.tokens_per_crop = (uint64_t)tokens_per_crop;
        emb_ctx.pad_id          = IMAGE_PAD_ID;
        emb_ctx.in_image = emb_ctx.in_slice = 0;
        emb_ctx.span_pad = emb_ctx.slice_idx = 0;

        ret = inference_minicpmv4_7_model(&rknn_app_ctx, &src_image, img_embeds,
                                          crop_boxes, n_crops, tensor, n_inputs, &perf);
        if (ret != 0)
        {
            printf("inference minicpmv4_7 model fail! ret=%d\n", ret);
            goto out;
        }

        printf_perf(&perf, true);
    }

out:
    ret = release_minicpmv4_7_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release minicpmv4_7 model fail! ret=%d\n", ret);
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
