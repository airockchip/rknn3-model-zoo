#ifndef _QACLIPVIT_EMBEDDING_H_
#define _QACLIPVIT_EMBEDDING_H_

#include "rknn3_api.h"
#include "common.h"
#define MAX_SEQ_LEN 352
#define EMBEDDING_DIM 768
#define MAX_TEXT_BUCKETS 8
#ifdef __cplusplus
extern "C" {
#endif

// 动态多桶：每个桶记录 shape_id 与该桶下各输入/输出的 tensor attr
typedef struct {
    int32_t shape_id;
    int seq;                     // 该桶的序列长度（input_embeds 的 shape[1]）
    rknn3_tensor_attr in_attrs[3];
    rknn3_tensor_attr out_attrs[1];
} qaclipvit_text_bucket_t;

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;

    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
    // 输入张量下标，init 时按名字解析
    int idx_input_embeds;      // input_embeds [1, seq, dim] FP16
    int idx_attention_mask;    // attention_mask [1, seq] FP16
    int idx_position_ids;      // position_ids [1, seq] INT32

    // 动态多桶
    qaclipvit_text_bucket_t buckets[MAX_TEXT_BUCKETS];
    int n_buckets;             // 可用桶数量
    int cur_bucket;            // 当前推理选中的桶下标
} qaclipvit_text_context_t;

// 嵌入结果结构体
typedef struct {
    float16* embedding;        // 768维嵌入向量
    int embedding_size;        // 嵌入向量大小
} qaclipvit_result_t;

// 输入数据结构体
typedef struct {
    float16* input_embeds;     // [1, seq, dim] 输入嵌入，padding 部分补零
    float16* attention_mask;   // [1, seq] 注意力掩码，有效 token 为 1、padding 为 0
    int32_t* position_ids;     // [1, seq] 位置ID
    int ntok;                  // 实际有效 token 数，用于动态选桶
} qaclipvit_input_data_t;

// 函数声明
int init_qaclipvit_embedding_model(const char *model_path, const char* weight_path, 
                                  qaclipvit_text_context_t *app_ctx, uint32_t core_mask);
int release_qaclipvit_embedding_model(qaclipvit_text_context_t *app_ctx);
int select_qaclipvit_text_bucket(qaclipvit_text_context_t *app_ctx, int ntok);
int inference_qaclipvit_embedding_model(qaclipvit_text_context_t *app_ctx, 
                                       qaclipvit_input_data_t *input_data, 
                                       qaclipvit_result_t *result);
int set_qaclipvit_input_data(qaclipvit_text_context_t *app_ctx, 
                            qaclipvit_input_data_t *input_data);
int process_qaclipvit_output(qaclipvit_text_context_t *app_ctx, 
                            qaclipvit_result_t *result);

#ifdef __cplusplus
}
#endif

#endif // _QACLIPVIT_EMBEDDING_H_
