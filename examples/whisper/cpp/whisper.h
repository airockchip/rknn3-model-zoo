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

#ifndef _RKNN_DEMO_WHISPER_H_
#define _RKNN_DEMO_WHISPER_H_

#include "rknn3_api.h"
#include "audio_utils.h"
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include "process.h"

typedef struct
{
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    // rknn3_tensor_attr *input_attrs;
    // rknn3_tensor_attr *output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;
    rknn3_tensor_mem* kvcache_mems[32];
    int kvcache_core_ids[32];
    uint32_t n_kvcache_mems;
    uint32_t kvcache_residual_depth;
#ifdef RKNN3_MAX_CACHE_PAGE_TYPES
    rknn3_kvcache_page_binding kvcache_bindings[32];
    uint32_t n_kvcache_bindings;
#endif
} rknn_app_context_t;

// typedef struct
// {
//     rknn_app_context_t encoder_context;
//     rknn_app_context_t decoder_context;
// } rknn_whisper_context_t;

int init_whisper_model(const char *model_path, rknn_app_context_t *app_ctx, const char *weight_path, uint32_t core_mask);
int release_whisper_model(rknn_app_context_t *app_ctx);
uint32_t whisper_shape_count(const rknn3_tensor_attr *attr);
uint64_t whisper_tensor_active_aligned_size(const rknn3_tensor *tensor);
int whisper_find_input_index_by_name(rknn_app_context_t *ctx, const char *name);
int whisper_require_input_index(rknn_app_context_t *ctx, const char *stage, const char *name, int fallback_index);
int whisper_find_output_index_by_name(rknn_app_context_t *ctx, const char *name);
uint32_t whisper_tensor_dtype_size(rknn3_tensor_type type);
bool whisper_same_shape(const rknn3_tensor_attr *a, const rknn3_tensor_attr *b);
void whisper_print_tensor_shape_brief(const char *prefix, const rknn3_tensor_attr *attr);
int whisper_convert_fp32_to_tensor(const float *src, void *dst, uint32_t n_elems, rknn3_tensor_type type);
int whisper_convert_tensor_to_fp32(const void *src, float *dst, int n_elems, rknn3_tensor_type type);
int whisper_fill_zero_tensor(rknn3_context ctx, rknn3_tensor *tensor, bool sync_to_device = true);
int whisper_fill_int32_scalar_tensor(rknn3_context ctx, rknn3_tensor *tensor, int32_t value);
int whisper_fill_causal_attention_mask_tensor(rknn3_context ctx, rknn3_tensor *tensor);
bool whisper_is_cross_kv_hdns_4d(const rknn3_tensor_attr *attr);
bool whisper_is_cross_kv_nhsd_4d(const rknn3_tensor_attr *attr);
bool whisper_is_cross_kv_hdns_c1hwc2_5d(const rknn3_tensor_attr *attr);
bool whisper_is_cross_kv_nc1hwc2_5d(const rknn3_tensor_attr *attr);
int whisper_repack_hdns_to_hdns_c1hwc2(rknn3_context dst_ctx, rknn3_tensor *dst, const rknn3_tensor *src, bool sync_to_device);
int whisper_copy_tensor_data(rknn3_context dst_ctx, rknn3_tensor *dst, const rknn3_tensor *src, bool sync_to_device = true);
int whisper_find_input_index_in_attrs(const std::vector<rknn3_tensor_attr> &attrs, const char *name);
int whisper_find_decoder_embed_index_in_attrs(const std::vector<rknn3_tensor_attr> &attrs);
uint32_t whisper_decoder_embed_seq_len(const std::vector<rknn3_tensor_attr> &attrs);
int whisper_ensure_tensor_mem_size(rknn3_context ctx, rknn3_tensor *tensor, uint64_t size, uint32_t core_id);
int whisper_recreate_tensor_mem_exact(rknn3_context ctx, rknn3_tensor *tensor, uint64_t size, uint32_t core_id);
uint32_t whisper_tensor_offset_2d(const rknn3_tensor_attr *attr, uint32_t row, uint32_t col, uint32_t col_size);
uint32_t whisper_tensor_offset_2d_int(const rknn3_tensor_attr *attr, uint32_t row, uint32_t col, uint32_t col_size);
uint32_t whisper_tensor_offset_logits(const rknn3_tensor_attr *attr, uint32_t vocab_index, uint32_t seq_index);
// int inference_whisper_model(rknn_whisper_context_t *app_ctx, std::vector<float> audio_data, float *mel_filters, VocabEntry *vocab, int task_code, std::vector<std::string> &recognized_text);

#endif //_RKNN_DEMO_WHISPER_H_
