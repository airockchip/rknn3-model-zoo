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

#include "rknn_gemma4.h"
#include "common.h"
#include "image_utils.h"
#include "audio_utils.h"
#include "nlohmann/json.hpp"

#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    float16* output;
    int capacity_tokens;
    int n_tokens;
    int embedding_dim;
} gemma4_audio_stream_accumulator;

static int append_gemma4_audio_stream_embeddings(const float16* embeddings,
                                                  int n_tokens,
                                                  void* userdata)
{
    gemma4_audio_stream_accumulator* accumulator =
        (gemma4_audio_stream_accumulator*)userdata;
    if (accumulator == NULL || embeddings == NULL || n_tokens < 0 ||
        accumulator->n_tokens + n_tokens > accumulator->capacity_tokens) {
        if (accumulator != NULL) {
            printf("audio stream output overflow: %d + %d > %d tokens\n",
                   accumulator->n_tokens, n_tokens, accumulator->capacity_tokens);
        }
        return -1;
    }
    memcpy(accumulator->output +
               (size_t)accumulator->n_tokens * accumulator->embedding_dim,
           embeddings,
           (size_t)n_tokens * accumulator->embedding_dim * sizeof(float16));
    accumulator->n_tokens += n_tokens;
    return 0;
}

int init_gemma4_model(rknn_gemma4_app_context* app_ctx,
                      const char* llm_model_path, const char* llm_weight_path,
                      const char* audio_model_path, const char* audio_weight_path,
                      bool enable_audio, bool audio_streaming, bool enable_vision,
                      rknn3_llm_param* params, int n_params,
                      RKLLMCallback& callback, uint32_t llm_core_mask, 
                      uint32_t audio_core_mask, uint32_t vision_core_mask,
                      const char* per_layer_embed_path, const char* safetensors_path,
                      Tokenizer* tokenizer, embedding_info* token_embedding,
                      input_cb_userdata* input_cb_data,
                      const char* vision_model_path, const char* vision_weight_path)
{
    (void)per_layer_embed_path;
    (void)safetensors_path;
    (void)tokenizer;
    (void)token_embedding;
    (void)input_cb_data;

    int ret = 0;
    app_ctx->enable_audio = enable_audio;
    app_ctx->audio_streaming = audio_streaming;
    app_ctx->enable_vision = enable_vision;

    if (enable_audio) {
        printf("--> init gemma4 audio model%s\n", audio_streaming ? " (streaming)" : "");
        if (audio_streaming) {
            ret = init_gemma4_audio_stream(&app_ctx->audio_stream, audio_model_path,
                                           audio_weight_path, audio_core_mask);
        } else {
            ret = init_gemma4_audio(&app_ctx->audio, audio_model_path,
                                    audio_weight_path, audio_core_mask);
        }
        if (ret < 0) {
            printf("init_gemma4_audio%s failed! ret=%d\n",
                   audio_streaming ? "_stream" : "", ret);
            return ret;
        }
    }

    if (enable_vision) {
        printf("--> init gemma4 vision model\n");
        ret = init_gemma4_vision(&app_ctx->vision, vision_model_path, vision_weight_path, vision_core_mask);
        if (ret < 0) {
            printf("init_gemma4_vision failed! ret=%d\n", ret);
            if (enable_audio) {
                if (audio_streaming) release_gemma4_audio_stream(app_ctx->audio_stream);
                else release_gemma4_audio(&app_ctx->audio);
            }
            return ret;
        }
    }

    printf("--> init gemma4 llm model\n");
    ret = init_gemma4_llm(&app_ctx->llm, llm_model_path, llm_weight_path, params, n_params, &callback, llm_core_mask);
    if (ret < 0) {
        printf("init_gemma4_llm failed! ret=%d\n", ret);
        if (enable_audio) {
            if (audio_streaming) release_gemma4_audio_stream(app_ctx->audio_stream);
            else release_gemma4_audio(&app_ctx->audio);
        }
        if (enable_vision) {
            release_gemma4_vision(&app_ctx->vision);
        }
        return ret;
    }

    return ret;
}


int release_gemma4_model(rknn_gemma4_app_context* app_ctx)
{
    release_gemma4_llm(&app_ctx->llm);
    if (app_ctx->enable_audio) {
        if (app_ctx->audio_streaming) release_gemma4_audio_stream(app_ctx->audio_stream);
        else release_gemma4_audio(&app_ctx->audio);
    }
    if (app_ctx->enable_vision) {
        release_gemma4_vision(&app_ctx->vision);
    }
    return 0;
}


int inference_gemma4_model(rknn_gemma4_app_context* app_ctx,
                           rknn3_llm_multimodal_tensor tensor,
                           audio_buffer_t* audio,
                           image_buffer_t* image,
                           float16* audio_embeds,
                           float16* image_embeds,
                           int32_t max_new_tokens,
                           rknn_perf_metrics_t* perf)
{
    int ret = 0;

    if (app_ctx->enable_audio && tensor.audio.n_audio) {
        int n_valid_audio = 0;
        int64_t start_us = getCurrentTimeUs();
        if (audio_embeds == NULL) {
            audio_embeds = (float16*)tensor.audio.audio_embed;
        }
        if (audio_embeds == NULL) {
            printf("audio_embeds is NULL\n");
            return -1;
        }
        tensor.audio.audio_embed = audio_embeds;
        if (app_ctx->audio_streaming) {
            if (!audio || audio->sample_rate != 16000 || audio->num_channels != 1) {
                printf("streaming audio input must be mono 16 kHz PCM\n");
                return -1;
            }
            if (tensor.audio.n_audio_tokens > INT_MAX) {
                printf("streaming audio token capacity is too large: %llu\n",
                       (unsigned long long)tensor.audio.n_audio_tokens);
                return -1;
            }
            gemma4_audio_stream_accumulator accumulator = {
                audio_embeds,
                (int)tensor.audio.n_audio_tokens,
                0,
                get_gemma4_audio_stream_embedding_dim()
            };

            ret = reset_gemma4_audio_stream(app_ctx->audio_stream);
            // Feed 100 ms of PCM at a time. NPU inference consumes the largest
            // configured Mel bucket, while the final tail uses the smallest
            // bucket that can contain it.
            static const int kPcmPushSamples = 1600;
            for (int offset = 0; ret == 0 && offset < audio->num_frames;
                 offset += kPcmPushSamples) {
                int remaining = audio->num_frames - offset;
                int count = (remaining < kPcmPushSamples) ? remaining : kPcmPushSamples;
                ret = push_gemma4_audio_stream(app_ctx->audio_stream,
                                               audio->data + offset, count,
                                               append_gemma4_audio_stream_embeddings,
                                               &accumulator);
            }
            if (ret == 0) {
                ret = flush_gemma4_audio_stream(app_ctx->audio_stream,
                                                append_gemma4_audio_stream_embeddings,
                                                &accumulator);
            }
            n_valid_audio = accumulator.n_tokens;
        } else {
            ret = inference_gemma4_audio(&app_ctx->audio, audio, audio_embeds, &n_valid_audio);
        }
        perf->audio_latency = getCurrentTimeUs() - start_us;
        if (ret != 0) {
            printf("inference_gemma4_audio%s failed! ret=%d\n",
                   app_ctx->audio_streaming ? "_stream" : "", ret);
            return ret;
        }
        tensor.audio.n_audio_tokens = n_valid_audio;
        printf("audio n_audio_tokens: %d\n", tensor.audio.n_audio_tokens);
    }

    if (app_ctx->enable_vision && tensor.image.n_image) {
        int64_t start_us = getCurrentTimeUs();
        if (image_embeds == NULL) {
            image_embeds = (float16*)tensor.image.image_embed;
        }
        if (image_embeds == NULL) {
            printf("image_embeds is NULL\n");
            return -1;
        }
        tensor.image.image_embed = image_embeds;
        ret = inference_gemma4_vision(&app_ctx->vision, image, image_embeds);
        perf->vision_latency = getCurrentTimeUs() - start_us;
        if (ret != 0) {
            printf("inference_gemma4_vision failed! ret=%d\n", ret);
            return ret;
        }
        printf("image n_image_tokens: %d\n", tensor.image.n_image_tokens);
    }

    printf("--> inference gemma4 llm model\n");
    ret = inference_gemma4_llm(&app_ctx->llm, tensor, max_new_tokens, perf);
    if (ret != 0) {
        printf("inference_gemma4_llm failed! ret=%d\n", ret);
        return ret;
    }

    return ret;
}