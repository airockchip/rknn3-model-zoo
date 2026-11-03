// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#ifndef RKNN_WEMM_EMBEDDING_H
#define RKNN_WEMM_EMBEDDING_H

#include <stdint.h>

#define WEMM_EMBEDDING_DIM 2048

// Per-request accelerator latency and the number of LLM prefill tokens.
typedef struct {
    double vision_ms;
    double llm_ms;
    uint64_t input_tokens;
} wemm_embedding_perf_t;

// Opaque owner of RKNN models, LLM session, tokenizer, and mapped embeddings.
typedef struct {
    void* impl;
} wemm_embedding_context_t;

// Select the single modality whose resources are loaded into the context.
typedef enum {
    WEMM_EMBEDDING_MODE_TEXT = 0,
    WEMM_EMBEDDING_MODE_IMAGE = 1,
    WEMM_EMBEDDING_MODE_VIDEO = 2,
} wemm_embedding_mode_t;

// Load the LLM and only the vision model required by mode from model_path.
// Returns 0 on success and a non-zero error code on failure.
int init_wemm_embedding(wemm_embedding_context_t* context,
                        const char* model_path,
                        wemm_embedding_mode_t mode);

// Compute a text embedding and write WEMM_EMBEDDING_DIM float values to embedding.
int run_wemm_text_embedding(wemm_embedding_context_t* context,
                            const char* text,
                            float* embedding,
                            wemm_embedding_perf_t* perf);

// Resize one image to the model input, run the image tower, then run the LLM.
int run_wemm_image_embedding(wemm_embedding_context_t* context,
                             const char* image_path,
                             const char* text,
                             float* embedding,
                             wemm_embedding_perf_t* perf);

// Pair sampled frames in order, run the video tower, and construct temporal mRoPE.
// frame_fps must match the sampling rate used to produce frame_paths.
int run_wemm_video_embedding(wemm_embedding_context_t* context,
                             const char* const* frame_paths,
                             uint64_t frame_count,
                             float frame_fps,
                             const char* text,
                             float* embedding,
                             wemm_embedding_perf_t* perf);

// Release all resources owned by context. A zeroed context is also accepted.
void release_wemm_embedding(wemm_embedding_context_t* context);

#endif
