#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "qaclipvit.h"
#include "file_utils.h"
#include "float16.h"
#include <math.h>

// 计算两个嵌入向量的余弦相似度
// 直接计算点积（不进行归一化）
float calculate_dot_product(float16* text_embed, float16* image_embed, int embed_dim) {
    float dot_product = 0.0f;
    
    for (int i = 0; i < embed_dim; i++) {
        float text_val = fp16_to_fp32(text_embed[i]);
        float image_val = fp16_to_fp32(image_embed[i]);
        
        dot_product += text_val * image_val;
    }
    
    return dot_product;
}

// 计算带logit_scale的点积
static float calculate_single_logit_with_scale(float16* text_embed, float16* image_embed, 
                                                   int embed_dim, float logit_scale) {
    float dot_product = calculate_dot_product(text_embed, image_embed, embed_dim);
    return dot_product * logit_scale;
}

// 释放相似度结果内存
void free_similarity_result(qaclipvit_similarity_result_t* result) {
    if (result->text_embedding) {
        free(result->text_embedding);
        result->text_embedding = NULL;
    }
    if (result->image_embedding) {
        free(result->image_embedding);
        result->image_embedding = NULL;
    }
    result->embedding_size = 0;
    result->logit_one_image = 0.0f;
}

// 释放输入数据内存
static void free_input_data(qaclipvit_input_data_t* input_data) {
    if (input_data->input_embeds) {
        free(input_data->input_embeds);
        input_data->input_embeds = NULL;
    }
    if (input_data->attention_mask) {
        free(input_data->attention_mask);
        input_data->attention_mask = NULL;
    }
    if (input_data->position_ids) {
        free(input_data->position_ids);
        input_data->position_ids = NULL;
    }
}

// 从已分词的 token 生成输入数据（含 padding 补零），并记录有效 token 数 ntok 供动态选桶
static int fill_text_input_data(qaclipvit_input_data_t* input_data,
                                embedding_info_t* embedding_info,
                                const int32_t* tokens, int token_size,
                                int seq_len) {
    memset(input_data, 0, sizeof(qaclipvit_input_data_t));

    // 检查 token 数量是否超过序列长度
    if (token_size > seq_len) {
        printf("Warning: Token count %d exceeds sequence length %d, truncating\n",
               token_size, seq_len);
        token_size = seq_len;
    }
    input_data->ntok = token_size;
    
    // 生成input_embeds
    input_data->input_embeds = (float16*)malloc(1 * seq_len * embedding_info->embedding_dim * sizeof(float16));
    if (!input_data->input_embeds) {
        printf("Failed to allocate memory for input_embeds\n");
        return -1;
    }
    
    // 从token生成embedding
    for (int i = 0; i < token_size && i < seq_len; i++) {
        int token_id = tokens[i];
        if (token_id < 0 || token_id >= embedding_info->vocab_size) {
            printf("Error: token_id %d out of range\n", token_id);
            return -1;
        }
        
        const float16* src_embed = embedding_info->embedding_data + 
                                  token_id * embedding_info->embedding_dim;
        float16* dst_embed = input_data->input_embeds + i * embedding_info->embedding_dim;
        memcpy(dst_embed, src_embed, embedding_info->embedding_dim * sizeof(float16));
    }
    
    // 填充剩余部分
    if (token_size < seq_len) {
        int remaining = seq_len - token_size;
        float16* padding_start = input_data->input_embeds + token_size * embedding_info->embedding_dim;
        memset(padding_start, 0, remaining * embedding_info->embedding_dim * sizeof(float16));
    }
    // 设置attention_mask
    input_data->attention_mask = (float16*)malloc(1 * seq_len * sizeof(float16));
    for (int i = 0; i < seq_len; i++) {
        input_data->attention_mask[i] = fp32_to_fp16(i < token_size ? 1.0f : 0.0f);
    }
    
    // 设置position_ids
    input_data->position_ids = (int32_t*)malloc(1 * seq_len * sizeof(int32_t));
    for (int i = 0; i < seq_len; i++) {
        input_data->position_ids[i] = i;
    }
    
    return 0;
}

// 初始化chinese_clip完整模型
int init_qaclipvit_model(const char* text_model_path, const char* text_weight_path, const char* embedding_path, const char* tokenizer_path,
                        const char* vision_model_path, const char* vision_weight_path,
                        uint32_t core_mask, rknn_qaclipvit_context_t* ctx) {
    int ret = 0;
    
    printf("--> Initializing chinese_clip complete model\n");
    
    // 初始化文本模型
    printf("--> Loading text model\n");
    ret = init_qaclipvit_embedding_model(text_model_path, text_weight_path, 
                                        &ctx->text_ctx, core_mask);
    if (ret != 0) {
        printf("Failed to initialize text model\n");
        return ret;
    }
    
    // 加载分词器（从外部GGUF文件）
    printf("--> Loading tokenizer from %s\n", tokenizer_path);
    ctx->tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA, tokenizer_path);
    if (!ctx->tokenizer) {
        printf("Failed to load tokenizer\n");
        return -1;
    }

    VocabInfo vocab_info;
    memset(&vocab_info, 0, sizeof(vocab_info));
    if (!ctx->tokenizer->GetVocabInfo(&vocab_info)) {
        printf("Failed to get vocab info from tokenizer\n");
        delete ctx->tokenizer;
        ctx->tokenizer = NULL;
        return -1;
    }
    printf("vocab_info: vocab_size=%d\n", vocab_info.vocab_size);

    // 初始化嵌入信息
    printf("--> Loading embeddings\n");
    ctx->embedding_info.fd = open(embedding_path, O_RDONLY);
    if (ctx->embedding_info.fd == -1) {
        printf("Failed to open embedding file: %s\n", embedding_path);
        return -1;
    }

    struct stat emb_st;
    if (fstat(ctx->embedding_info.fd, &emb_st) == -1) {
        printf("Failed to get embedding file size\n");
        close(ctx->embedding_info.fd);
        return -1;
    }

    ctx->embedding_info.file_size = emb_st.st_size;
    ctx->embedding_info.embedding_data = (float16 *)mmap(NULL, emb_st.st_size,
                                                        PROT_READ, MAP_PRIVATE,
                                                        ctx->embedding_info.fd, 0);
    if (ctx->embedding_info.embedding_data == MAP_FAILED) {
        printf("Failed to mmap embedding file\n");
        close(ctx->embedding_info.fd);
        return -1;
    }

    ctx->embedding_info.vocab_size = vocab_info.vocab_size;
    ctx->embedding_info.embedding_dim = (emb_st.st_size / vocab_info.vocab_size) / sizeof(float16);
    
    printf("Embedding info: vocab_size=%d, embedding_dim=%d\n", 
           ctx->embedding_info.vocab_size, ctx->embedding_info.embedding_dim);
    
    // 初始化视觉模型
    // printf("--> Loading vision model\n");
    ret = init_qaclipvit_vision_model(vision_model_path, vision_weight_path, 
                                     &ctx->vision_ctx, core_mask);
    if (ret != 0) {
        printf("Failed to initialize vision model\n");
        // 清理已初始化的资源
        release_qaclipvit_embedding_model(&ctx->text_ctx);
        delete ctx->tokenizer;
        munmap(ctx->embedding_info.embedding_data, ctx->embedding_info.file_size);
        close(ctx->embedding_info.fd);
        return ret;
    }
    
    printf("--> chinese_clip model initialization completed successfully\n");
    return 0;
}

// 释放chinese_clip模型资源
int release_qaclipvit_model(rknn_qaclipvit_context_t* ctx) {
    printf("--> Releasing chinese_clip model resources\n");
    
    int ret = 0;
    int ret_val = 0;
    
    // 释放视觉模型
    if (ctx->vision_ctx.rknn_ctx) {
        ret = release_qaclipvit_vision_model(&ctx->vision_ctx);
        if (ret != 0) {
            printf("Failed to release vision model\n");
            ret_val = ret;
        }
    }
    
    // 释放文本模型
    if (ctx->text_ctx.rknn_ctx) {
        ret = release_qaclipvit_embedding_model(&ctx->text_ctx);
        if (ret != 0) {
            printf("Failed to release text model\n");
            ret_val = ret;
        }
    }
    
    // 释放分词器
    if (ctx->tokenizer) {
        delete ctx->tokenizer;
        ctx->tokenizer = NULL;
    }
    
    // 释放嵌入信息
    if (ctx->embedding_info.embedding_data != MAP_FAILED && 
        ctx->embedding_info.embedding_data != NULL) {
        munmap(ctx->embedding_info.embedding_data, ctx->embedding_info.file_size);
        ctx->embedding_info.embedding_data = NULL;
    }
    
    if (ctx->embedding_info.fd != -1) {
        close(ctx->embedding_info.fd);
        ctx->embedding_info.fd = -1;
    }
    
    printf("--> chinese_clip model resources released\n");
    return ret_val;
}

// chinese_clip完整推理
int inference_qaclipvit_model(rknn_qaclipvit_context_t* ctx, 
                             const char* text, 
                             image_buffer_t* image_input,
                             qaclipvit_similarity_result_t* result) {
    int ret = 0;
    
    printf("--> Running chinese_clip complete inference\n");
    printf("Text input: %s\n", text);
    
    // 文本推理
    printf("--> Text model inference\n");
    qaclipvit_input_data_t text_input_data;
    qaclipvit_result_t text_result;
    memset(&text_input_data, 0, sizeof(qaclipvit_input_data_t));
    memset(&text_result, 0, sizeof(qaclipvit_result_t));
    
    // 先分词得到 token 数，再按 token 数选桶（128/256/352），短文本减少 padding 无效计算
    int32_t tokens[MAX_SEQ_LEN];
    int token_size = ctx->tokenizer->Tokenize(text, strlen(text), tokens, MAX_SEQ_LEN);
    if (token_size <= 0) {
        printf("Error: tokenize failed.\n");
        return -1;
    }
    printf("Token size: %d\n", token_size);
    int bucket_idx = select_qaclipvit_text_bucket(&ctx->text_ctx, token_size);
    if (bucket_idx < 0) {
        printf("Failed to select text bucket for %d tokens\n", token_size);
        return -1;
    }
    int seq_len = ctx->text_ctx.buckets[bucket_idx].seq;
    printf("Selected text bucket[%d] seq=%d\n", bucket_idx, seq_len);

    ret = fill_text_input_data(&text_input_data, &ctx->embedding_info,
                               tokens, token_size, seq_len);
    if (ret != 0) {
        printf("Failed to fill text input data\n");
        return ret;
    }
    ret = inference_qaclipvit_embedding_model(&ctx->text_ctx, &text_input_data, &text_result);
    if (ret != 0) {
        printf("Text model inference failed\n");
        // 清理文本输入数据
        free(text_result.embedding);
        free_input_data(&text_input_data);
        return ret;
    }
    
    // printf("Text embedding size: %d\n", text_result.embedding_size);
    
    // 视觉推理
    printf("--> Vision model inference\n");
    qaclipvit_vision_result_t vision_result;
    memset(&vision_result, 0, sizeof(qaclipvit_vision_result_t));
    
    ret = inference_qaclipvit_vision_model(&ctx->vision_ctx, image_input, &vision_result);
    if (ret != 0) {
        printf("Vision model inference failed\n");
        free(vision_result.image_embeddings);
        free(text_result.embedding);
        free_input_data(&text_input_data);
        return ret;
    }
    
    // printf("Vision embedding size: %d\n", vision_result.embedding_size);
    // 分配结果内存
    result->embedding_size = ctx->embedding_info.embedding_dim;
    result->text_embedding = (float16*)malloc(result->embedding_size * sizeof(float16));
    result->image_embedding = (float16*)malloc(result->embedding_size * sizeof(float16));
    
    if (!result->text_embedding || !result->image_embedding) {
        printf("Failed to allocate memory for similarity result\n");
        free(text_result.embedding);
        free(vision_result.image_embeddings);
        free_input_data(&text_input_data);
        return -1;
    }
    
    // 复制文本嵌入（转换为float16）
    memcpy(result->text_embedding, text_result.embedding, 
           result->embedding_size * sizeof(float16));
    memcpy(result->image_embedding, vision_result.image_embeddings, 
           result->embedding_size * sizeof(float16));
    
    // 计算相似度得分
    result->logit_one_image = calculate_single_logit_with_scale(text_result.embedding, 
                                                   vision_result.image_embeddings, 
                                                   result->embedding_size,LOGIT_SCALE_EXP);

    printf("logit_one_image: %.6f\n", result->logit_one_image);
    
    // 清理临时结果
    free(text_result.embedding);
    free(vision_result.image_embeddings);
    free_input_data(&text_input_data);
    
    printf("--> chinese_clip inference completed successfully\n");
    return 0;
}