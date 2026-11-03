// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.

#ifndef _RKNN_DEMO_GEMMA4_AUDIO_STREAM_H_
#define _RKNN_DEMO_GEMMA4_AUDIO_STREAM_H_

#include "rknn3_api.h"
#include "common.h"

typedef struct rknn_gemma4_audio_stream_context rknn_gemma4_audio_stream_context;

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*gemma4_audio_stream_callback)(const float16* embeddings,
                                            int n_tokens,
                                            void* userdata);

int init_gemma4_audio_stream(rknn_gemma4_audio_stream_context** stream,
                             const char* model_path,
                             const char* weight_path,
                             uint32_t core_mask);

int reset_gemma4_audio_stream(rknn_gemma4_audio_stream_context* stream);

// Return the exact number of valid audio tokens produced for a waveform with
// n_samples mono 16 kHz samples, and the fixed embedding width of the stream
// model. These helpers let the full Gemma4 demo allocate its LLM input buffer
// before streaming starts.
int get_n_audio_stream(int n_samples);
int get_gemma4_audio_stream_embedding_dim(void);

// Push mono 16 kHz float PCM samples. For a dynamic-bucket model, full chunks
// use the largest configured bucket; flush() pads the tail to the smallest
// bucket that can contain it.
int push_gemma4_audio_stream(rknn_gemma4_audio_stream_context* stream,
                             const float* samples,
                             int n_samples,
                             gemma4_audio_stream_callback callback,
                             void* userdata);

// Flush the final partial block by zero-padding it to a configured bucket. The
// callback receives only the valid output tokens reported by the model mask.
int flush_gemma4_audio_stream(rknn_gemma4_audio_stream_context* stream,
                              gemma4_audio_stream_callback callback,
                              void* userdata);

int release_gemma4_audio_stream(rknn_gemma4_audio_stream_context* stream);

#ifdef __cplusplus
}
#endif

#endif  // _RKNN_DEMO_GEMMA4_AUDIO_STREAM_H_
