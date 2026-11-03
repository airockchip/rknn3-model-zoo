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

#ifndef _RKNN_DEMO_VALUE_HEAD_H_
#define _RKNN_DEMO_VALUE_HEAD_H_

#include <stdint.h>
#include <string>

#include "rknn3_api.h"
#include "Tokenizer.h"
#include "common.h"
#include "rknn_minicpm_ocr_jev_llm.h"
#include "rknn_minicpm_ocr_jev_vision.h"

// Default decision boundary. This is the threshold that maximises balanced
// accuracy on the held-out split of the reference checkpoint; it is not 0.5.
#define VALUE_HEAD_DEFAULT_THRESHOLD 0.5606135129928589f

// Resolved token embedding bank (.embed.bin, vocab x dim float16).
typedef struct {
    int fd;
    float16 *data;
    int embedding_dim;
    int vocab_size;
    long long size;
} value_head_embedding_info;

typedef struct {
    float logit_0;
    float logit_1;
    float prob_0;
    float prob_1;
    int label;
    float threshold;
    int n_prompt_tokens;  // tokens after image-pad expansion
    int n_vision_tokens;  // visual tokens per image
    int64_t vision_latency_us;
    int64_t llm_latency_us;
} value_head_result;

typedef struct {
    rknn_value_head_vision_context vision;
    rknn_value_head_llm_context llm;
    value_head_score_sink sink;
    int n_internal_mems;
    rknn3_tensor_mem **internal_mems;
    uint32_t model_width;
    uint32_t model_height;
} rknn_value_head_context;

// Builds the exact evaluation prompt used by the reference implementation:
//   <|im_start|>user\n
//   <|vision_start|><|image_pad|><|vision_end|>OCR文本：\n{ocr}\n
//   <|vision_start|><|image_pad|><|vision_end|>
//   <user tail><|im_end|>\n
// Note there is deliberately no assistant turn, so the pooled representation is
// the last token of the user turn.
std::string build_value_head_prompt(const std::string &ocr_text);

int init_value_head_model(rknn_value_head_context *app_ctx,
                          const char *llm_model_path,
                          const char *llm_weight_path,
                          const char *vision_model_path,
                          const char *vision_weight_path,
                          rknn3_llm_param *params,
                          int n_params,
                          RKLLMCallback *callback,
                          uint32_t vision_core_mask,
                          uint32_t llm_core_mask,
                          uint32_t model_width,
                          uint32_t model_height);

int release_value_head_model(rknn_value_head_context *app_ctx);

// Scores one (original image, rendered image, OCR text) triple.
// `print_prompt` dumps the assembled prompt and the token layout for debugging.
int score_value_head(rknn_value_head_context *app_ctx,
                     Tokenizer *tokenizer,
                     image_buffer_t *orig_image,
                     image_buffer_t *render_image,
                     const std::string &ocr_text,
                     float threshold,
                     bool print_prompt,
                     value_head_result *result);

#endif // _RKNN_DEMO_VALUE_HEAD_H_
