// Copyright (c) 2026 Rockchip Electronics Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef RKNN_PARAFORMER_HOTWORD_H
#define RKNN_PARAFORMER_HOTWORD_H

#include <stddef.h>
#include <stdint.h>

#include "rknn3_api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct paraformer_hotword_context paraformer_hotword_context_t;

/**
 * Creates the optional SeaCo hotword pipeline and precomputes embeddings for
 * one UTF-8 hotword list. ASF is enabled when both ASF paths are non-NULL and
 * `asf_nfilter` is positive.
 */
int paraformer_hotword_create(const char* seaco_model_path, const char* seaco_weight_path,
                              const char* bias_weights_path,
                              const char* hotwords_path, const char* tokens_path, const char* seg_dict_path,
                              int max_hotwords, int max_hotword_len, float seaco_weight, const char* asf_model_path,
                              const char* asf_weight_path, int asf_nfilter, uint32_t core_mask,
                              paraformer_hotword_context_t** context);

/**
 * Runs optional ASF filtering and the SeaCo decoder, then replaces the main
 * decoder logits with fused log probabilities in place.
 */
int paraformer_hotword_apply(paraformer_hotword_context_t* context, const float* predictor_embeds,
                             size_t predictor_embed_count, const float* decoder_hidden, size_t decoder_hidden_count,
                             int token_num, float* decoder_logits, size_t decoder_logit_count, int token_len,
                             int vocab_size);

/** Returns the initialized SeaCo and optional ASF contexts for memory sharing. */
int paraformer_hotword_get_npu_contexts(const paraformer_hotword_context_t* context,
                                        rknn3_context* model_contexts, uint32_t capacity,
                                        uint32_t* model_count);

/** Releases NPU models, CPU BiasEncoder weights, and cached hotword data. */
void paraformer_hotword_release(paraformer_hotword_context_t* context);

#ifdef __cplusplus
}
#endif

#endif
