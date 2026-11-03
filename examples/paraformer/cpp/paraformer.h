// Copyright (c) 2026 Rockchip Electronics Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef RKNN_PARAFORMER_H
#define RKNN_PARAFORMER_H

#include <stdint.h>

#include "rknn3_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PARAFORMER_INPUT_FEAT_DIM 560
#define PARAFORMER_ENCODER_DIM 512
#define PARAFORMER_VOCAB_SIZE 8404
#define PARAFORMER_MAX_ENCODER_SHAPES 16
#define PARAFORMER_MAX_DECODER_SHAPES 16
#define PARAFORMER_MAX_NPU_CORES 32
#define PARAFORMER_PREDICTOR_CIF_THRESHOLD 1.0f
#define PARAFORMER_PREDICTOR_CIF_TAIL_THRESHOLD 0.45f
#define PARAFORMER_PREDICTOR_MAX_TOKENS 256
#define PARAFORMER_FBANK_N_MELS 80
#define PARAFORMER_FBANK_SAMPLE_RATE 16000
#define PARAFORMER_LFR_M 7
#define PARAFORMER_LFR_N 6

typedef enum {
    PARAFORMER_INTERNAL_MEM_FAILED = -1,
    PARAFORMER_INTERNAL_MEM_UNINITIALIZED = 0,
    PARAFORMER_INTERNAL_MEM_READY = 1,
} paraformer_internal_mem_state_t;

typedef struct {
    rknn3_context encoder_ctx;
    rknn3_tensor_mem* encoder_input_mem[2];
    rknn3_tensor_mem* encoder_output_mem[2];
    rknn3_tensor_attr encoder_input_attrs[2];
    rknn3_tensor_attr encoder_output_attrs[2];
    uint32_t encoder_n_inputs;
    uint32_t encoder_n_outputs;

    rknn3_shape_config encoder_shape_config;
    rknn3_tensor_attr encoder_shape_input_attrs[PARAFORMER_MAX_ENCODER_SHAPES];
    rknn3_tensor_attr encoder_shape_mask_attrs[PARAFORMER_MAX_ENCODER_SHAPES];
    rknn3_tensor_attr encoder_shape_output_attrs[PARAFORMER_MAX_ENCODER_SHAPES];
    rknn3_tensor_attr encoder_shape_alpha_attrs[PARAFORMER_MAX_ENCODER_SHAPES];
    int32_t encoder_shape_ids[PARAFORMER_MAX_ENCODER_SHAPES];

    rknn3_context decoder_ctx;
    rknn3_tensor_mem* decoder_input_mem[4];
    rknn3_tensor_mem* decoder_output_mem[2];
    rknn3_tensor_attr decoder_input_attrs[4];
    rknn3_tensor_attr decoder_output_attrs[2];
    uint32_t decoder_n_inputs;
    uint32_t decoder_n_outputs;

    rknn3_shape_config decoder_shape_config;
    rknn3_tensor_attr decoder_shape_in0_attrs[PARAFORMER_MAX_DECODER_SHAPES];
    rknn3_tensor_attr decoder_shape_in1_attrs[PARAFORMER_MAX_DECODER_SHAPES];
    rknn3_tensor_attr decoder_shape_in2_attrs[PARAFORMER_MAX_DECODER_SHAPES];
    rknn3_tensor_attr decoder_shape_in3_attrs[PARAFORMER_MAX_DECODER_SHAPES];
    rknn3_tensor_attr decoder_shape_out0_attrs[PARAFORMER_MAX_DECODER_SHAPES];
    rknn3_tensor_attr decoder_shape_out1_attrs[PARAFORMER_MAX_DECODER_SHAPES];
    int32_t decoder_shape_ids[PARAFORMER_MAX_DECODER_SHAPES];

    uint32_t core_mask;
    uint32_t encoder_time;
    void* hotword_context;
    rknn3_tensor_mem* internal_mems[PARAFORMER_MAX_NPU_CORES];
    paraformer_internal_mem_state_t internal_mem_state;
} rknn_app_context_t;

/**
 * Loads the fused encoder-alpha model and the masked decoder model.
 *
 * A zero `core_mask` selects all cores declared by both RKNN models. A
 * nonzero mask must exactly match their compiled core count. The initialized
 * context must be released with `release_paraformer_model()`. RKNN3 internal
 * memory is allocated once per core and shared by the serial model stages on
 * the first inference request.
 *
 * Returns zero on success and -1 on validation, allocation, or RKNN3 errors.
 */
int init_paraformer_model(
    const char* encoder_model_path,
    const char* decoder_model_path,
    const char* encoder_weight_path,
    const char* decoder_weight_path,
    rknn_app_context_t* context,
    uint32_t core_mask);

/**
 * Enables SeaCo contextual biasing for a model initialized with the two-output
 * hotword decoder. ASF is optional and enabled by non-NULL model paths plus a
 * positive `asf_nfilter` value. Hotword mode must be enabled before the first
 * inference request so every active NPU model can join the shared memory pool.
 */
int init_paraformer_hotword(
    const char* seaco_model_path,
    const char* seaco_weight_path,
    const char* bias_weights_path,
    const char* hotwords_path,
    const char* tokens_path,
    const char* seg_dict_path,
    int max_hotwords,
    int max_hotword_len,
    float seaco_weight,
    const char* asf_model_path,
    const char* asf_weight_path,
    int asf_nfilter,
    rknn_app_context_t* context);

/**
 * Runs FBank/LFR/CMVN, encoder-alpha, CPU CIF, masked decoder, and token
 * decoding for one mono 16 kHz PCM waveform.
 *
 * The recognition text is NUL-terminated in `output_text`. Returns zero on
 * success and -1 when preprocessing, inference, or decoding fails.
 */
int run_paraformer_inference(
    rknn_app_context_t* context,
    const int16_t* audio_data,
    int audio_samples,
    const char* tokens_path,
    char* output_text,
    int output_size);

/** Releases every RKNN3 handle and tensor buffer owned by the context. */
void release_paraformer_model(rknn_app_context_t* context);

#ifdef __cplusplus
}
#endif

#endif
