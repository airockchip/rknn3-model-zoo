// Copyright (c) 2026 Rockchip Electronics Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include "paraformer.h"
#include "hotword.h"
#include "paraformer_cmvn.h"
#include "paraformer_utils.h"
#include "time_utils.h"

#include <ctype.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kaldi-native-fbank/csrc/online-feature.h"

typedef struct {
    float* data;
    size_t size;
} float_buffer_t;

typedef struct {
    char* storage;
    char** items;
    int count;
} token_table_t;

typedef struct {
    rknn3_context context;
    const char* name;
    uint32_t core_num;
    rknn3_core_mem_size core_sizes[PARAFORMER_MAX_NPU_CORES];
    rknn3_tensor_mem* bindings[PARAFORMER_MAX_NPU_CORES];
} internal_mem_model_t;

#define PARAFORMER_MAX_INTERNAL_MODELS 4

/** Returns the smaller of two buffer sizes. */
static size_t min_size(size_t a, size_t b) { return a < b ? a : b; }

/** Returns the smaller of two signed dimensions. */
static int min_int(int a, int b) { return a < b ? a : b; }

/** Releases a heap-backed float buffer and clears its ownership fields. */
static void release_float_buffer(float_buffer_t* buffer) {
    if (!buffer)
        return;
    free(buffer->data);
    buffer->data = NULL;
    buffer->size = 0;
}

/** Computes the element count represented by an RKNN3 tensor shape. */
static size_t tensor_numel(const rknn3_tensor_attr* attr) {
    size_t elems = 1;
    for (uint32_t i = 0; i < attr->n_dims; ++i) {
        elems *= attr->shape[i];
    }
    return elems;
}

/** Releases token strings parsed in place from the vocabulary JSON file. */
static void release_token_table(token_table_t* table) {
    if (!table)
        return;
    free(table->items);
    free(table->storage);
    memset(table, 0, sizeof(*table));
}

/**
 * Parses the vocabulary JSON string array into pointers backed by one mutable
 * storage allocation. The resulting table owns both allocations.
 */
static int parse_tokens_json_simple(const char* tokens_path, token_table_t* table) {
    FILE* fp = NULL;
    long file_size = 0;
    size_t capacity = 256;
    int in_quote = 0;
    int escaping = 0;
    char* token_start = NULL;
    char* write_ptr = NULL;

    if (!tokens_path || !table)
        return -1;
    memset(table, 0, sizeof(*table));

    fp = fopen(tokens_path, "rb");
    if (!fp || fseek(fp, 0, SEEK_END) != 0)
        goto fail;
    file_size = ftell(fp);
    if (file_size <= 0 || fseek(fp, 0, SEEK_SET) != 0)
        goto fail;

    table->storage = (char*)malloc((size_t)file_size + 1);
    table->items = (char**)malloc(capacity * sizeof(char*));
    if (!table->storage || !table->items)
        goto fail;
    if (fread(table->storage, 1, (size_t)file_size, fp) != (size_t)file_size) {
        goto fail;
    }
    fclose(fp);
    fp = NULL;
    table->storage[file_size] = '\0';

    for (long i = 0; i < file_size; ++i) {
        char c = table->storage[i];
        if (!in_quote) {
            if (c == '"') {
                in_quote = 1;
                escaping = 0;
                token_start = table->storage + i + 1;
                write_ptr = token_start;
            }
            continue;
        }

        if (escaping) {
            *write_ptr++ = c;
            escaping = 0;
            continue;
        }
        if (c == '\\') {
            escaping = 1;
            continue;
        }
        if (c == '"') {
            char** expanded_items = NULL;
            *write_ptr = '\0';
            if ((size_t)table->count == capacity) {
                capacity *= 2;
                expanded_items = (char**)realloc(table->items, capacity * sizeof(char*));
                if (!expanded_items)
                    goto fail;
                table->items = expanded_items;
            }
            table->items[table->count++] = token_start;
            in_quote = 0;
            continue;
        }
        *write_ptr++ = c;
    }

    if (in_quote || table->count == 0)
        goto fail;
    return 0;

fail:
    if (fp)
        fclose(fp);
    release_token_table(table);
    return -1;
}

/** Returns whether one token consists only of Chinese characters or digits. */
static int token_is_chinese_or_digit(const char* token) {
    const unsigned char* cursor = (const unsigned char*)token;
    int codepoint_count = 0;
    while (cursor && *cursor) {
        uint32_t codepoint = 0;
        if (*cursor < 0x80) {
            codepoint = *cursor++;
        } else if ((*cursor & 0xe0) == 0xc0 && cursor[1]) {
            codepoint = ((uint32_t)(cursor[0] & 0x1f) << 6) | (uint32_t)(cursor[1] & 0x3f);
            cursor += 2;
        } else if ((*cursor & 0xf0) == 0xe0 && cursor[1] && cursor[2]) {
            codepoint = ((uint32_t)(cursor[0] & 0x0f) << 12) | ((uint32_t)(cursor[1] & 0x3f) << 6) |
                        (uint32_t)(cursor[2] & 0x3f);
            cursor += 3;
        } else {
            return 0;
        }
        if (!((codepoint >= 0x4e00 && codepoint <= 0x9fff) ||
              (codepoint >= (uint32_t)'0' && codepoint <= (uint32_t)'9') || codepoint == (uint32_t)'@')) {
            return 0;
        }
        ++codepoint_count;
    }
    return codepoint_count > 0;
}

/** Returns whether one token is an ASCII English word or apostrophe. */
static int token_is_alpha(const char* token) {
    int character_count = 0;
    if (!token)
        return 0;
    for (const unsigned char* cursor = (const unsigned char*)token; *cursor; ++cursor) {
        if (!(isalpha(*cursor) || *cursor == '\''))
            return 0;
        ++character_count;
    }
    return character_count > 0;
}

/** Appends text while optionally removing BPE `@@` continuation markers. */
static void append_decoded_text(char* output, int output_size, int* output_length, const char* token,
                                int remove_join_markers) {
    const char* cursor = token;
    while (cursor && *cursor && *output_length < output_size - 1) {
        if (remove_join_markers && cursor[0] == '@' && cursor[1] == '@') {
            cursor += 2;
            continue;
        }
        output[(*output_length)++] = *cursor++;
    }
}

/**
 * Maps token IDs to text and mirrors FunASR's BPE-oriented
 * `sentence_postprocess()` word reconstruction.
 */
static int decode_tokens(const int32_t* token_ids, int token_num, const char* tokens_path, char* output_text,
                         int output_size) {
    if (!token_ids || token_num <= 0 || !output_text || output_size <= 0) {
        return -1;
    }

    token_table_t token_table;
    int output_length = 0;
    memset(&token_table, 0, sizeof(token_table));
    output_text[0] = '\0';
    if (parse_tokens_json_simple(tokens_path, &token_table) != 0) {
        return -1;
    }

    int all_chinese = 1;
    int all_alpha = 1;
    int valid_tokens = 0;
    for (int i = 0; i < token_num; ++i) {
        int id = token_ids[i];
        if (id <= 2 || id >= token_table.count)
            continue;
        const char* token = token_table.items[id];
        if (!token[0] || token[0] == '<')
            continue;
        ++valid_tokens;
        all_chinese = all_chinese && token_is_chinese_or_digit(token);
        all_alpha = all_alpha && token_is_alpha(token);
    }
    if (valid_tokens == 0) {
        release_token_table(&token_table);
        return 0;
    }

    int alpha_blank = 0;
    for (int i = 0; i < token_num; ++i) {
        int id = token_ids[i];
        if (id <= 2 || id >= token_table.count)
            continue;
        const char* token = token_table.items[id];
        if (!token[0] || token[0] == '<')
            continue;

        if (all_chinese) {
            append_decoded_text(output_text, output_size, &output_length, token, 0);
        } else if (all_alpha) {
            append_decoded_text(output_text, output_size, &output_length, token, 0);
            if (output_length < output_size - 1)
                output_text[output_length++] = ' ';
        } else if (token_is_chinese_or_digit(token)) {
            if (alpha_blank && output_length > 0 && output_text[output_length - 1] == ' ')
                --output_length;
            append_decoded_text(output_text, output_size, &output_length, token, 0);
            alpha_blank = 0;
        } else if (strstr(token, "@@")) {
            append_decoded_text(output_text, output_size, &output_length, token, 1);
            alpha_blank = 0;
        } else if (token_is_alpha(token)) {
            append_decoded_text(output_text, output_size, &output_length, token, 0);
            if (output_length < output_size - 1)
                output_text[output_length++] = ' ';
            alpha_blank = 1;
        } else {
            append_decoded_text(output_text, output_size, &output_length, token, 0);
        }
    }
    while (output_length > 0 && output_text[output_length - 1] == ' ')
        --output_length;
    output_text[output_length] = '\0';

    release_token_table(&token_table);
    return 0;
}

/** Computes Kaldi-compatible FBank features into a caller-owned flat buffer. */
static int compute_fbank(const float* waveform, int audio_samples, int sample_rate, int n_mels, float** frames,
                         int* frame_count) {
    if (!waveform || audio_samples <= 0 || sample_rate <= 0 || n_mels <= 0 || !frames || !frame_count) {
        return -1;
    }
    *frames = NULL;
    *frame_count = 0;

    knf::FbankOptions options;
    options.frame_opts.samp_freq = sample_rate;
    options.frame_opts.frame_shift_ms = 10.0f;
    options.frame_opts.frame_length_ms = 25.0f;
    options.frame_opts.dither = 0.0f;
    options.frame_opts.preemph_coeff = 0.97f;
    options.frame_opts.remove_dc_offset = true;
    options.frame_opts.window_type = "hamming";
    options.frame_opts.snip_edges = true;
    options.mel_opts.num_bins = n_mels;
    options.mel_opts.low_freq = 20.0f;
    options.mel_opts.high_freq = 0.0f;

    knf::OnlineFbank fbank(options);
    fbank.AcceptWaveform(sample_rate, waveform, audio_samples);
    fbank.InputFinished();

    int count = fbank.NumFramesReady();
    if (count <= 0) {
        printf("[Fbank] Error: 0 frames extracted from %d samples\n", audio_samples);
        return -1;
    }

    float* output = (float*)malloc((size_t)count * (size_t)n_mels * sizeof(float));
    if (!output)
        return -1;
    for (int index = 0; index < count; ++index) {
        memcpy(output + (size_t)index * n_mels, fbank.GetFrame(index), (size_t)n_mels * sizeof(float));
    }

    *frames = output;
    *frame_count = count;
    return 0;
}

/**
 * Converts mono PCM to the 560-dimensional FBank/LFR/CMVN features expected
 * by the encoder. On success, `features` is caller-owned and freed with
 * `free()`.
 */
static int extract_fbank(const int16_t* audio_data, int audio_samples, float** features, int* n_frames) {
    if (!audio_data || audio_samples <= 0 || !features || !n_frames) {
        return -1;
    }

    const int n_mels = PARAFORMER_FBANK_N_MELS;
    const int lfr_m = PARAFORMER_LFR_M;
    const int lfr_n = PARAFORMER_LFR_N;
    const int feat_dim = n_mels * lfr_m;

    float* waveform = NULL;
    float* fbank_frames = NULL;
    float* padded_fbank = NULL;
    float* out = NULL;
    // Preserve the raw PCM magnitude expected by Kaldi FBank.
    waveform = (float*)malloc((size_t)audio_samples * sizeof(float));
    if (!waveform)
        return -1;
    for (int i = 0; i < audio_samples; ++i) {
        waveform[i] = (float)audio_data[i];
    }

    int num_fbank_frames = 0;
    if (compute_fbank(waveform, audio_samples, PARAFORMER_FBANK_SAMPLE_RATE, n_mels, &fbank_frames,
                      &num_fbank_frames) != 0) {
        free(waveform);
        return -1;
    }

    printf("[Fbank] Extracted %d frames (80-dim) from %d samples (%.2fs)\n", num_fbank_frames, audio_samples,
           (float)audio_samples / PARAFORMER_FBANK_SAMPLE_RATE);

    // Match Python apply_lfr(): left-pad three frames, stack seven frames, and
    // advance six frames for each output.
    int T = num_fbank_frames;
    int left_pad = (lfr_m - 1) / 2;
    int T_lfr = (int)ceilf((float)T / (float)lfr_n);

    padded_fbank = (float*)malloc((size_t)(left_pad + T) * n_mels * sizeof(float));
    if (!padded_fbank) {
        free(fbank_frames);
        free(waveform);
        return -1;
    }
    const float* first_frame = fbank_frames;
    for (int p = 0; p < left_pad; ++p) {
        memcpy(padded_fbank + (size_t)p * n_mels, first_frame, n_mels * sizeof(float));
    }
    for (int t = 0; t < T; ++t) {
        const float* frame = fbank_frames + (size_t)t * n_mels;
        memcpy(padded_fbank + (size_t)(left_pad + t) * n_mels, frame, n_mels * sizeof(float));
    }

    int total_padded = left_pad + T;
    int needed_frames = (T_lfr - 1) * lfr_n + lfr_m;
    int num_padding = needed_frames > total_padded ? needed_frames - total_padded : 0;
    if (num_padding > 0) {
        float last_frame[PARAFORMER_FBANK_N_MELS];
        size_t old_elems = (size_t)total_padded * n_mels;
        float* expanded_fbank = NULL;
        memcpy(last_frame, padded_fbank + (size_t)(total_padded - 1) * n_mels, sizeof(last_frame));
        expanded_fbank = (float*)realloc(padded_fbank, (old_elems + (size_t)num_padding * n_mels) * sizeof(float));
        if (!expanded_fbank) {
            free(padded_fbank);
            free(fbank_frames);
            free(waveform);
            return -1;
        }
        padded_fbank = expanded_fbank;
        for (int p = 0; p < num_padding; ++p) {
            memcpy(padded_fbank + old_elems + (size_t)p * n_mels, last_frame, sizeof(last_frame));
        }
        total_padded += num_padding;
    }

    out = (float*)malloc((size_t)T_lfr * feat_dim * sizeof(float));
    if (!out) {
        free(padded_fbank);
        free(fbank_frames);
        free(waveform);
        return -1;
    }

    for (int t = 0; t < T_lfr; ++t) {
        float* dst = out + (size_t)t * feat_dim;
        for (int m = 0; m < lfr_m; ++m) {
            int src_t = t * lfr_n + m;
            memcpy(dst + m * n_mels, padded_fbank + (size_t)src_t * n_mels, n_mels * sizeof(float));
        }
    }

    // Apply the exported additive mean and multiplicative variance values.
    for (int t = 0; t < T_lfr; ++t) {
        float* frame = out + (size_t)t * feat_dim;
        for (int d = 0; d < feat_dim; ++d) {
            frame[d] += cmvn_means[d];
            frame[d] *= cmvn_vars[d];
        }
    }

    *features = out;
    *n_frames = T_lfr;

    free(padded_fbank);
    free(fbank_frames);
    free(waveform);

    printf("[Fbank+LFR+CMVN] %d fbank frames -> %d LFR frames (560-dim)\n", num_fbank_frames, T_lfr);

    return 0;
}

/** Returns the time dimension from a rank-two-or-higher tensor attribute. */
static inline int get_time_dim_from_attr(const rknn3_tensor_attr* attr) {
    if (!attr || attr->n_dims < 2)
        return 1;
    return attr->shape[1];
}

/** Returns the final feature dimension from an RKNN3 tensor attribute. */
static inline int get_last_dim_from_attr(const rknn3_tensor_attr* attr) {
    if (!attr || attr->n_dims < 1)
        return 1;
    return attr->shape[attr->n_dims - 1];
}

/**
 * Validates the rank, batch, feature size, FP16 dtype, and undefined layout
 * required by the direct tensor-memory path.
 */
static int validate_fp16_tensor_attr(const rknn3_tensor_attr* attr, const char* name, uint32_t last_dim) {
    if (attr->n_dims != 3 || attr->shape[0] != 1 || attr->shape[2] != last_dim || attr->dtype != RKNN3_TENSOR_FLOAT16 ||
        attr->layout != RKNN3_TENSOR_UNDEFINED) {
        printf("Error: invalid %s attr: n_dims=%u shape=[%u,%u,%u] "
               "dtype=%d layout=%d\n",
               name, attr->n_dims, attr->shape[0], attr->shape[1], attr->shape[2], attr->dtype, attr->layout);
        return -1;
    }
    return 0;
}

/** Finds the smallest encoder dynamic bucket that can hold `frames`. */
static int find_best_encoder_shape(rknn_app_context_t* ctx, int frames) {
    int best_idx = -1;
    int best_time = INT32_MAX;
    for (uint32_t i = 0; i < ctx->encoder_shape_config.n_shapes; i++) {
        int t = get_time_dim_from_attr(&ctx->encoder_shape_input_attrs[i]);
        if (t >= frames && t < best_time) {
            best_time = t;
            best_idx = (int)i;
        }
    }
    return best_idx;
}

/** Finds the smallest decoder bucket that fits encoder and token lengths. */
static int find_best_decoder_shape(rknn_app_context_t* ctx, int enc_time, int token_num) {
    int best_idx = -1;
    int best_score = INT32_MAX;
    for (uint32_t i = 0; i < ctx->decoder_shape_config.n_shapes; i++) {
        int enc_t = get_time_dim_from_attr(&ctx->decoder_shape_in0_attrs[i]);
        int tok_t = get_time_dim_from_attr(&ctx->decoder_shape_in1_attrs[i]);
        if (enc_t >= enc_time && tok_t >= token_num) {
            int score = enc_t + tok_t;
            if (score < best_score) {
                best_score = score;
                best_idx = (int)i;
            }
        }
    }
    return best_idx;
}

/**
 * Selects an encoder bucket, writes zero-padded FP16 features, runs RKNN3,
 * and returns the bucket-sized output as caller-owned FP32 storage.
 */
static int run_encoder(rknn_app_context_t* ctx, const float* feats, int frames, float_buffer_t* encoder_out,
                       float_buffer_t* encoder_alpha) {
    if (!ctx || !feats || frames <= 0 || !encoder_out || !encoder_alpha) {
        return -1;
    }

    int shape_idx = find_best_encoder_shape(ctx, frames);
    if (shape_idx < 0) {
        printf("[Encoder] Error: No bucket found for frames=%d\n", frames);
        return -1;
    }

    int32_t shape_id = ctx->encoder_shape_ids[shape_idx];
    rknn3_tensor_attr* cur_input_attr = &ctx->encoder_shape_input_attrs[shape_idx];
    rknn3_tensor_attr* cur_mask_attr = &ctx->encoder_shape_mask_attrs[shape_idx];
    rknn3_tensor_attr* cur_output_attr = &ctx->encoder_shape_output_attrs[shape_idx];
    rknn3_tensor_attr* cur_alpha_attr = &ctx->encoder_shape_alpha_attrs[shape_idx];

    printf("[Encoder] frames=%d, bucket=%d, speech=[1,%u,%u], "
           "mask=[1,%u,%u], enc=[1,%u,%u], alphas=[1,%u,%u]\n",
           frames, shape_idx, cur_input_attr->shape[1], cur_input_attr->shape[2], cur_mask_attr->shape[1],
           cur_mask_attr->shape[2], cur_output_attr->shape[1], cur_output_attr->shape[2], cur_alpha_attr->shape[1],
           cur_alpha_attr->shape[2]);

    int ret = rknn3_set_shape(ctx->encoder_ctx, shape_id);
    if (ret != RKNN3_SUCCESS) {
        printf("[Encoder] rknn3_set_shape failed: %d\n", ret);
        return -1;
    }

    const int input_time = get_time_dim_from_attr(cur_input_attr);
    const int feat_dim = get_last_dim_from_attr(cur_input_attr);
    const int out_time = get_time_dim_from_attr(cur_output_attr);
    const int out_dim = get_last_dim_from_attr(cur_output_attr);

    const size_t max_input_elems = (size_t)input_time * feat_dim;
    const size_t memory_elems = ctx->encoder_input_mem[0]->size / sizeof(uint16_t);
    const size_t copy_elems = min_size((size_t)frames * feat_dim, min_size(max_input_elems, memory_elems));

    memset(ctx->encoder_input_mem[0]->virt_addr, 0, ctx->encoder_input_mem[0]->size);
    paraformer_fp32_to_fp16(feats, (uint16_t*)ctx->encoder_input_mem[0]->virt_addr, copy_elems);

    const size_t mask_elems = tensor_numel(cur_mask_attr);
    const size_t valid_mask_elems = min_size((size_t)frames, mask_elems);
    memset(ctx->encoder_input_mem[1]->virt_addr, 0, ctx->encoder_input_mem[1]->size);
    if (cur_mask_attr->dtype == RKNN3_TENSOR_FLOAT16) {
        uint16_t* mask = (uint16_t*)ctx->encoder_input_mem[1]->virt_addr;
        for (size_t i = 0; i < valid_mask_elems; ++i) {
            mask[i] = 0x3c00;
        }
    } else {
        float* mask = (float*)ctx->encoder_input_mem[1]->virt_addr;
        for (size_t i = 0; i < valid_mask_elems; ++i) {
            mask[i] = 1.0f;
        }
    }

    for (uint32_t i = 0; i < ctx->encoder_n_inputs; ++i) {
        ret = rknn3_mem_sync(ctx->encoder_ctx, ctx->encoder_input_mem[i], RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("[Encoder] mem_sync input%u failed: %d\n", i, ret);
            return -1;
        }
    }

    rknn3_tensor input_tensors[2];
    memset(input_tensors, 0, sizeof(input_tensors));
    input_tensors[0].mem = ctx->encoder_input_mem[0];
    input_tensors[0].attr = cur_input_attr;
    input_tensors[1].mem = ctx->encoder_input_mem[1];
    input_tensors[1].attr = cur_mask_attr;

    rknn3_tensor output_tensors[2];
    memset(output_tensors, 0, sizeof(output_tensors));
    output_tensors[0].mem = ctx->encoder_output_mem[0];
    output_tensors[0].attr = cur_output_attr;
    output_tensors[1].mem = ctx->encoder_output_mem[1];
    output_tensors[1].attr = cur_alpha_attr;

    int64_t enc_start_us = getCurrentTimeUs();
    ret = rknn3_run(ctx->encoder_ctx, input_tensors, ctx->encoder_n_inputs, output_tensors, ctx->encoder_n_outputs);
    if (ret != RKNN3_SUCCESS) {
        printf("[Encoder] rknn3_run failed: %d\n", ret);
        return -1;
    }
    double enc_ms = (getCurrentTimeUs() - enc_start_us) / 1000.0;
    printf("[Encoder] rknn3_run done (%.2f ms)\n", enc_ms);

    for (uint32_t i = 0; i < ctx->encoder_n_outputs; ++i) {
        ret = rknn3_mem_sync(ctx->encoder_ctx, ctx->encoder_output_mem[i], RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("[Encoder] mem_sync output%u failed: %d\n", i, ret);
            return -1;
        }
    }

    release_float_buffer(encoder_out);
    encoder_out->size = (size_t)out_time * out_dim;
    encoder_out->data = (float*)malloc(encoder_out->size * sizeof(float));
    if (!encoder_out->data) {
        encoder_out->size = 0;
        return -1;
    }
    if (cur_output_attr->dtype == RKNN3_TENSOR_FLOAT16) {
        paraformer_fp16_to_fp32((const uint16_t*)ctx->encoder_output_mem[0]->virt_addr, encoder_out->data,
                                encoder_out->size);
    } else {
        memcpy(encoder_out->data, ctx->encoder_output_mem[0]->virt_addr, encoder_out->size * sizeof(float));
    }

    release_float_buffer(encoder_alpha);
    const int alpha_time = get_time_dim_from_attr(cur_alpha_attr);
    const int alpha_dim = get_last_dim_from_attr(cur_alpha_attr);
    encoder_alpha->size = (size_t)alpha_time * alpha_dim;
    encoder_alpha->data = (float*)malloc(encoder_alpha->size * sizeof(float));
    if (!encoder_alpha->data) {
        encoder_alpha->size = 0;
        return -1;
    }
    if (cur_alpha_attr->dtype == RKNN3_TENSOR_FLOAT16) {
        paraformer_fp16_to_fp32((const uint16_t*)ctx->encoder_output_mem[1]->virt_addr, encoder_alpha->data,
                                encoder_alpha->size);
    } else {
        memcpy(encoder_alpha->data, ctx->encoder_output_mem[1]->virt_addr, encoder_alpha->size * sizeof(float));
    }

    ctx->encoder_time = (uint32_t)min_int(frames, out_time);
    return 0;
}

/**
 * Selects a decoder bucket, writes encoder features, CIF embeddings, and the
 * explicit token mask, then returns bucket-sized logits as FP32 storage.
 */
static int run_decoder(rknn_app_context_t* ctx, const float_buffer_t* encoder_out, int enc_time,
                       const float_buffer_t* predictor_embeds, int predictor_token_num, float_buffer_t* decoder_logits,
                       float_buffer_t* decoder_hidden, int* token_len, int* vocab_size) {
    if (!ctx || !encoder_out || !encoder_out->data || enc_time <= 0 || !predictor_embeds || !predictor_embeds->data ||
        predictor_token_num <= 0 || !decoder_logits || !token_len || !vocab_size) {
        return -1;
    }

    int shape_idx = find_best_decoder_shape(ctx, enc_time, predictor_token_num);
    if (shape_idx < 0) {
        printf("[Decoder] Error: No bucket found for enc_time=%d\n", enc_time);
        return -1;
    }

    int32_t shape_id = ctx->decoder_shape_ids[shape_idx];
    rknn3_tensor_attr* cur_in0_attr = &ctx->decoder_shape_in0_attrs[shape_idx];
    rknn3_tensor_attr* cur_in1_attr = &ctx->decoder_shape_in1_attrs[shape_idx];
    rknn3_tensor_attr* cur_in2_attr = &ctx->decoder_shape_in2_attrs[shape_idx];
    rknn3_tensor_attr* cur_in3_attr = &ctx->decoder_shape_in3_attrs[shape_idx];
    rknn3_tensor_attr* cur_out0_attr = &ctx->decoder_shape_out0_attrs[shape_idx];
    rknn3_tensor_attr* cur_out1_attr = ctx->decoder_n_outputs == 2 ? &ctx->decoder_shape_out1_attrs[shape_idx] : NULL;

    printf("[Decoder] enc_time=%d, selected bucket=%d (shape_id=%d), "
           "in0=[1,%u,%u], in1=[1,%u,%u]",
           enc_time, shape_idx, shape_id, cur_in0_attr->shape[1], cur_in0_attr->shape[2], cur_in1_attr->shape[1],
           cur_in1_attr->shape[2]);
    printf(", token_mask=[1,%u,%u], enc_mask=[1,%u,%u]", cur_in2_attr->shape[1], cur_in2_attr->shape[2],
           cur_in3_attr->shape[1], cur_in3_attr->shape[2]);
    printf(", out0=[1,%u,%u]", cur_out0_attr->shape[1], cur_out0_attr->shape[2]);
    if (cur_out1_attr) {
        printf(", out1=[1,%u,%u]", cur_out1_attr->shape[1], cur_out1_attr->shape[2]);
    }
    printf("\n");

    int ret = rknn3_set_shape(ctx->decoder_ctx, shape_id);
    if (ret != RKNN3_SUCCESS) {
        printf("[Decoder] rknn3_set_shape failed: %d\n", ret);
        return -1;
    }

    const int enc_dim = get_last_dim_from_attr(cur_in0_attr);
    const int in0_time = get_time_dim_from_attr(cur_in0_attr);
    const int in1_time = get_time_dim_from_attr(cur_in1_attr);
    const int emb_dim = get_last_dim_from_attr(cur_in1_attr);

    memset(ctx->decoder_input_mem[0]->virt_addr, 0, ctx->decoder_input_mem[0]->size);
    size_t copy0 = min_size((size_t)enc_time * enc_dim, (size_t)in0_time * enc_dim);
    copy0 = min_size(copy0, encoder_out->size);
    if (cur_in0_attr->dtype == RKNN3_TENSOR_FLOAT16) {
        paraformer_fp32_to_fp16(encoder_out->data, (uint16_t*)ctx->decoder_input_mem[0]->virt_addr, copy0);
    } else {
        memcpy(ctx->decoder_input_mem[0]->virt_addr, encoder_out->data, copy0 * sizeof(float));
    }

    // Copy predictor embeds directly (zero-padding to bucket size, no
    // resampling).
    const int available_time = (int)(predictor_embeds->size / (size_t)emb_dim);
    const int copy_time = min_int(min_int(predictor_token_num, in1_time), available_time);
    memset(ctx->decoder_input_mem[1]->virt_addr, 0, ctx->decoder_input_mem[1]->size);
    if (cur_in1_attr->dtype == RKNN3_TENSOR_FLOAT16) {
        paraformer_fp32_to_fp16(predictor_embeds->data, (uint16_t*)ctx->decoder_input_mem[1]->virt_addr,
                                (size_t)copy_time * emb_dim);
    } else {
        memcpy(ctx->decoder_input_mem[1]->virt_addr, predictor_embeds->data,
               (size_t)copy_time * emb_dim * sizeof(float));
    }

    ret = rknn3_mem_sync(ctx->decoder_ctx, ctx->decoder_input_mem[0], RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN3_SUCCESS) {
        printf("[Decoder] sync input0 failed: %d\n", ret);
        return -1;
    }
    ret = rknn3_mem_sync(ctx->decoder_ctx, ctx->decoder_input_mem[1], RKNN3_MEMORY_SYNC_TO_DEVICE);
    if (ret != RKNN3_SUCCESS) {
        printf("[Decoder] sync input1 failed: %d\n", ret);
        return -1;
    }
    {
        memset(ctx->decoder_input_mem[2]->virt_addr, 0, ctx->decoder_input_mem[2]->size);
        int32_t token_len_i32 = (int32_t)predictor_token_num;
        size_t in2_elems = tensor_numel(cur_in2_attr);
        if (in2_elems > 1) {
            size_t valid_elems = min_size((size_t)predictor_token_num, in2_elems);
            if (cur_in2_attr->dtype == RKNN3_TENSOR_FLOAT32) {
                float* token_mask = (float*)ctx->decoder_input_mem[2]->virt_addr;
                for (size_t i = 0; i < valid_elems; ++i)
                    token_mask[i] = 1.0f;
            } else if (cur_in2_attr->dtype == RKNN3_TENSOR_FLOAT16) {
                uint16_t* token_mask = (uint16_t*)ctx->decoder_input_mem[2]->virt_addr;
                for (size_t i = 0; i < valid_elems; ++i)
                    token_mask[i] = 0x3c00;
            } else {
                int32_t* token_mask = (int32_t*)ctx->decoder_input_mem[2]->virt_addr;
                for (size_t i = 0; i < valid_elems; ++i)
                    token_mask[i] = 1;
            }
        } else if (cur_in2_attr->dtype == RKNN3_TENSOR_FLOAT32) {
            float token_len_f32 = (float)predictor_token_num;
            memcpy(ctx->decoder_input_mem[2]->virt_addr, &token_len_f32, sizeof(token_len_f32));
        } else if (cur_in2_attr->dtype == RKNN3_TENSOR_FLOAT16) {
            float token_len_f32 = (float)predictor_token_num;
            uint16_t token_len_fp16 = 0;
            paraformer_fp32_to_fp16(&token_len_f32, &token_len_fp16, 1);
            memcpy(ctx->decoder_input_mem[2]->virt_addr, &token_len_fp16, sizeof(token_len_fp16));
        } else {
            memcpy(ctx->decoder_input_mem[2]->virt_addr, &token_len_i32, sizeof(token_len_i32));
        }
        ret = rknn3_mem_sync(ctx->decoder_ctx, ctx->decoder_input_mem[2], RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("[Decoder] sync input2 failed: %d\n", ret);
            return -1;
        }
    }

    {
        size_t in3_elems = tensor_numel(cur_in3_attr);
        size_t valid_elems = min_size((size_t)enc_time, in3_elems);
        memset(ctx->decoder_input_mem[3]->virt_addr, 0, ctx->decoder_input_mem[3]->size);
        if (cur_in3_attr->dtype == RKNN3_TENSOR_FLOAT32) {
            float* enc_mask = (float*)ctx->decoder_input_mem[3]->virt_addr;
            for (size_t i = 0; i < valid_elems; ++i)
                enc_mask[i] = 1.0f;
        } else {
            uint16_t* enc_mask = (uint16_t*)ctx->decoder_input_mem[3]->virt_addr;
            for (size_t i = 0; i < valid_elems; ++i)
                enc_mask[i] = 0x3c00;
        }
        ret = rknn3_mem_sync(ctx->decoder_ctx, ctx->decoder_input_mem[3], RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("[Decoder] sync input3 failed: %d\n", ret);
            return -1;
        }
    }

    int64_t dec_start_us = getCurrentTimeUs();

    rknn3_tensor inputs[4];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].mem = ctx->decoder_input_mem[0];
    inputs[0].attr = cur_in0_attr;
    inputs[1].mem = ctx->decoder_input_mem[1];
    inputs[1].attr = cur_in1_attr;
    inputs[2].mem = ctx->decoder_input_mem[2];
    inputs[2].attr = cur_in2_attr;
    inputs[3].mem = ctx->decoder_input_mem[3];
    inputs[3].attr = cur_in3_attr;

    rknn3_tensor outputs[2];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].mem = ctx->decoder_output_mem[0];
    outputs[0].attr = cur_out0_attr;
    if (cur_out1_attr) {
        outputs[1].mem = ctx->decoder_output_mem[1];
        outputs[1].attr = cur_out1_attr;
    }

    ret = rknn3_run(ctx->decoder_ctx, inputs, ctx->decoder_n_inputs, outputs, ctx->decoder_n_outputs);
    if (ret != RKNN3_SUCCESS)
        return -1;

    double dec_ms = (getCurrentTimeUs() - dec_start_us) / 1000.0;
    printf("[Decoder] rknn3_run done (%.2f ms), bucket=%d (in0=[1,%d,512], "
           "in1=[1,%d,512], n_inputs=%u)\n",
           dec_ms, shape_idx, in0_time, in1_time, ctx->decoder_n_inputs);

    ret = rknn3_mem_sync(ctx->decoder_ctx, ctx->decoder_output_mem[0], RKNN3_MEMORY_SYNC_FROM_DEVICE);
    if (ret != RKNN3_SUCCESS) {
        printf("[Decoder] sync output0 failed: %d\n", ret);
        return -1;
    }
    if (decoder_hidden && cur_out1_attr) {
        ret = rknn3_mem_sync(ctx->decoder_ctx, ctx->decoder_output_mem[1], RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("[Decoder] sync output1 failed: %d\n", ret);
            return -1;
        }
    }

    const int out_time = get_time_dim_from_attr(cur_out0_attr);
    *vocab_size = get_last_dim_from_attr(cur_out0_attr);
    release_float_buffer(decoder_logits);
    decoder_logits->size = (size_t)out_time * *vocab_size;
    decoder_logits->data = (float*)malloc(decoder_logits->size * sizeof(float));
    if (!decoder_logits->data) {
        decoder_logits->size = 0;
        return -1;
    }

    if (cur_out0_attr->dtype == RKNN3_TENSOR_FLOAT16) {
        paraformer_fp16_to_fp32((const uint16_t*)ctx->decoder_output_mem[0]->virt_addr, decoder_logits->data,
                                decoder_logits->size);
    } else {
        memcpy(decoder_logits->data, ctx->decoder_output_mem[0]->virt_addr, decoder_logits->size * sizeof(float));
    }

    if (decoder_hidden) {
        int hidden_time = 0;
        int hidden_dimension = 0;
        if (!cur_out1_attr) {
            printf("Error: hotword inference requires decoder_hidden output\n");
            return -1;
        }
        hidden_time = get_time_dim_from_attr(cur_out1_attr);
        hidden_dimension = get_last_dim_from_attr(cur_out1_attr);
        release_float_buffer(decoder_hidden);
        decoder_hidden->size = (size_t)hidden_time * hidden_dimension;
        decoder_hidden->data = (float*)malloc(decoder_hidden->size * sizeof(float));
        if (!decoder_hidden->data) {
            decoder_hidden->size = 0;
            return -1;
        }
        if (cur_out1_attr->dtype == RKNN3_TENSOR_FLOAT16) {
            paraformer_fp16_to_fp32((const uint16_t*)ctx->decoder_output_mem[1]->virt_addr, decoder_hidden->data,
                                    decoder_hidden->size);
        } else {
            memcpy(decoder_hidden->data, ctx->decoder_output_mem[1]->virt_addr, decoder_hidden->size * sizeof(float));
        }
    }

    // Use predictor token count, not bucket output length.
    *token_len = min_int(predictor_token_num, out_time);
    return 0;
}

/**
 * Integrates frame-level encoder alphas into acoustic token embeddings.
 * The encoder runs Conv1D, ReLU, linear projection, and sigmoid on the NPU;
 * this function keeps only the data-dependent integrate-and-fire loop on CPU.
 */
static int run_cif_from_alpha(const float_buffer_t* encoder_out, const float_buffer_t* encoder_alpha, int enc_time,
                              int enc_dim, float_buffer_t* predictor_embeds, int* predictor_token_num) {
    const float threshold = PARAFORMER_PREDICTOR_CIF_THRESHOLD;
    const float tail_threshold = PARAFORMER_PREDICTOR_CIF_TAIL_THRESHOLD;
    float* frame = NULL;
    float integrate = 0.0f;
    float alpha_sum = 0.0f;
    int output_count = 0;
    int64_t cif_start_us = 0;

    if (!encoder_out || !encoder_out->data || !encoder_alpha || !encoder_alpha->data || enc_time <= 0 ||
        enc_dim != PARAFORMER_ENCODER_DIM || !predictor_embeds || !predictor_token_num ||
        encoder_out->size < (size_t)enc_time * enc_dim || encoder_alpha->size < (size_t)enc_time) {
        return -1;
    }

    release_float_buffer(predictor_embeds);
    cif_start_us = getCurrentTimeUs();
    frame = (float*)calloc((size_t)enc_dim, sizeof(float));
    predictor_embeds->data = (float*)calloc((size_t)PARAFORMER_PREDICTOR_MAX_TOKENS * enc_dim, sizeof(float));
    if (!frame || !predictor_embeds->data) {
        goto fail;
    }
    for (int t = 0; t < enc_time; ++t) {
        const float alpha = encoder_alpha->data[t];
        const float integrate_new = integrate + alpha;
        const int fire = integrate_new >= threshold;
        const float current = fire ? (threshold - integrate) : alpha;
        const float remainder = alpha - current;
        const float* enc_t = encoder_out->data + (size_t)t * enc_dim;

        alpha_sum += alpha;
        for (int d = 0; d < enc_dim; ++d) {
            frame[d] += current * enc_t[d];
        }

        if (fire) {
            if (output_count >= PARAFORMER_PREDICTOR_MAX_TOKENS) {
                goto fail;
            }
            memcpy(predictor_embeds->data + (size_t)output_count * enc_dim, frame, (size_t)enc_dim * sizeof(float));
            ++output_count;
            for (int d = 0; d < enc_dim; ++d) {
                frame[d] = remainder * enc_t[d];
            }
            integrate = remainder;
        } else {
            integrate = integrate_new;
        }
    }

    if (integrate + tail_threshold >= threshold) {
        if (output_count >= PARAFORMER_PREDICTOR_MAX_TOKENS) {
            goto fail;
        }
        memcpy(predictor_embeds->data + (size_t)output_count * enc_dim, frame, (size_t)enc_dim * sizeof(float));
        ++output_count;
    }

    if (output_count <= 0) {
        output_count = 1;
    }
    predictor_embeds->size = (size_t)output_count * enc_dim;
    *predictor_token_num = output_count;
    printf("[CIF] enc_time=%d, alpha_sum=%.2f, token_num=%d, dim=%d "
           "(%.2f ms)\n",
           enc_time, alpha_sum, output_count, enc_dim, (getCurrentTimeUs() - cif_start_us) / 1000.0);
    free(frame);
    return 0;

fail:
    free(frame);
    release_float_buffer(predictor_embeds);
    *predictor_token_num = 0;
    return -1;
}

/** Computes one vocabulary argmax per valid decoder timestep. */
static int argmax_decode(const float_buffer_t* logits, int token_len, int vocab_size, int32_t** token_ids,
                         int* token_count) {
    if (!logits || !logits->data || token_len <= 0 || vocab_size <= 0 || !token_ids || !token_count ||
        logits->size < (size_t)token_len * vocab_size) {
        return -1;
    }
    *token_ids = (int32_t*)malloc((size_t)token_len * sizeof(int32_t));
    if (!*token_ids)
        return -1;
    *token_count = token_len;

    // Argmax is invariant under softmax, so no temporary probability buffer
    // is needed.
    for (int t = 0; t < token_len; ++t) {
        const float* row = logits->data + (size_t)t * vocab_size;
        int best_i = 0;
        float best_v = row[0];
        for (int v = 1; v < vocab_size; ++v) {
            if (row[v] > best_v) {
                best_v = row[v];
                best_i = v;
            }
        }
        (*token_ids)[t] = best_i;
    }
    return 0;
}

/**
 * Queries a model's compiled core count, derives its full mask, and validates
 * any explicit user mask against that value.
 */
static int resolve_core_mask(rknn3_context model_context, const char* model_name, uint32_t* core_mask) {
    uint32_t core_num = 0;
    int ret = rknn3_query(model_context, RKNN3_QUERY_CORE_NUMBER, &core_num, sizeof(core_num));
    if (ret != RKNN3_SUCCESS || core_num == 0 || core_num > 32) {
        printf("Error: RKNN3_QUERY_CORE_NUMBER failed for %s: ret=%d core_num=%u\n", model_name, ret, core_num);
        return -1;
    }

    uint32_t model_core_mask = core_num == 32 ? UINT32_MAX : (uint32_t)((1ULL << core_num) - 1ULL);
    if (*core_mask == 0) {
        *core_mask = model_core_mask;
    } else if (*core_mask != model_core_mask) {
        printf("Error: core mask 0x%x does not match %s %u-core model "
               "(expected 0x%x)\n",
               *core_mask, model_name, core_num, model_core_mask);
        return -1;
    }
    printf("[%s] core_num=%u, core_mask=0x%x\n", model_name, core_num, *core_mask);
    return 0;
}

/** Releases the per-core RKNN3 internal memory owned by the encoder context. */
static void release_internal_memory(rknn_app_context_t* ctx) {
    if (!ctx)
        return;
    for (uint32_t core_id = 0; core_id < PARAFORMER_MAX_NPU_CORES; ++core_id) {
        if (ctx->internal_mems[core_id]) {
            int ret = rknn3_destroy_mem(ctx->encoder_ctx, ctx->internal_mems[core_id]);
            if (ret != RKNN3_SUCCESS) {
                printf("Warning: destroy shared internal memory for core%u failed: %d\n", core_id, ret);
            }
            ctx->internal_mems[core_id] = NULL;
        }
    }
    ctx->internal_mem_state = PARAFORMER_INTERNAL_MEM_UNINITIALIZED;
}

/**
 * Allocates one maximum-sized internal buffer per core and binds it to every
 * serial RKNN3 stage in the active baseline or hotword pipeline.
 */
static int init_shared_internal_memory(rknn_app_context_t* ctx) {
    internal_mem_model_t models[PARAFORMER_MAX_INTERNAL_MODELS];
    rknn3_context hotword_contexts[2];
    uint64_t max_size_by_core[PARAFORMER_MAX_NPU_CORES];
    uint64_t unshared_total = 0;
    uint64_t shared_total = 0;
    uint32_t model_count = 2;
    uint32_t hotword_count = 0;
    int ret = -1;

    if (!ctx || !ctx->encoder_ctx || !ctx->decoder_ctx ||
        ctx->internal_mem_state != PARAFORMER_INTERNAL_MEM_UNINITIALIZED) {
        return -1;
    }
    memset(models, 0, sizeof(models));
    memset(hotword_contexts, 0, sizeof(hotword_contexts));
    memset(max_size_by_core, 0, sizeof(max_size_by_core));
    models[0].context = ctx->encoder_ctx;
    models[0].name = "Encoder";
    models[1].context = ctx->decoder_ctx;
    models[1].name = "Decoder";

    if (ctx->hotword_context) {
        if (paraformer_hotword_get_npu_contexts(
                (const paraformer_hotword_context_t*)ctx->hotword_context, hotword_contexts,
                sizeof(hotword_contexts) / sizeof(hotword_contexts[0]), &hotword_count) != 0 ||
            model_count + hotword_count > PARAFORMER_MAX_INTERNAL_MODELS) {
            printf("Error: cannot collect hotword NPU contexts for internal memory sharing\n");
            ctx->internal_mem_state = PARAFORMER_INTERNAL_MEM_FAILED;
            return -1;
        }
        for (uint32_t index = 0; index < hotword_count; ++index) {
            models[model_count].context = hotword_contexts[index];
            models[model_count].name = index == 0 ? "SeaCo" : "ASF";
            ++model_count;
        }
    }

    for (uint32_t model_index = 0; model_index < model_count; ++model_index) {
        internal_mem_model_t* model = &models[model_index];
        ret = rknn3_query(model->context, RKNN3_QUERY_CORE_NUMBER, &model->core_num,
                          sizeof(model->core_num));
        if (ret != RKNN3_SUCCESS || model->core_num == 0 ||
            model->core_num > PARAFORMER_MAX_NPU_CORES) {
            printf("Error: query %s core count for internal memory failed: ret=%d count=%u\n",
                   model->name, ret, model->core_num);
            goto failed;
        }
        ret = rknn3_query(model->context, RKNN3_QUERY_CORE_MEM_SIZE, model->core_sizes,
                          sizeof(model->core_sizes[0]) * model->core_num);
        if (ret != RKNN3_SUCCESS) {
            printf("Error: query %s internal memory size failed: %d\n", model->name, ret);
            goto failed;
        }
        for (uint32_t core_index = 0; core_index < model->core_num; ++core_index) {
            int32_t core_id = model->core_sizes[core_index].core_id;
            uint64_t internal_size = model->core_sizes[core_index].internal_size;
            if (core_id < 0 || core_id >= PARAFORMER_MAX_NPU_CORES || internal_size == 0) {
                printf("Error: invalid %s internal memory entry: core=%d size=%" PRIu64 "\n",
                       model->name, core_id, internal_size);
                goto failed;
            }
            unshared_total += internal_size;
            if (internal_size > max_size_by_core[core_id]) {
                max_size_by_core[core_id] = internal_size;
            }
            printf("[InternalMem] %s core%d requires %" PRIu64 " bytes\n", model->name,
                   core_id, internal_size);
        }
    }

    for (uint32_t core_id = 0; core_id < PARAFORMER_MAX_NPU_CORES; ++core_id) {
        if (max_size_by_core[core_id] == 0)
            continue;
        ctx->internal_mems[core_id] =
            rknn3_create_mem(ctx->encoder_ctx, max_size_by_core[core_id], (int32_t)core_id,
                             RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!ctx->internal_mems[core_id]) {
            printf("Error: create shared internal memory failed: core=%u size=%" PRIu64 "\n",
                   core_id, max_size_by_core[core_id]);
            goto failed;
        }
        shared_total += max_size_by_core[core_id];
    }

    for (uint32_t model_index = 0; model_index < model_count; ++model_index) {
        internal_mem_model_t* model = &models[model_index];
        for (uint32_t core_index = 0; core_index < model->core_num; ++core_index) {
            int32_t core_id = model->core_sizes[core_index].core_id;
            model->bindings[core_index] = ctx->internal_mems[core_id];
        }
        ret = rknn3_set_internal_mem(model->context, model->bindings, model->core_num);
        if (ret != RKNN3_SUCCESS) {
            printf("Error: bind shared internal memory to %s failed: %d\n", model->name, ret);
            goto failed;
        }
    }

    ctx->internal_mem_state = PARAFORMER_INTERNAL_MEM_READY;
    printf("[InternalMem] shared %" PRIu64 " bytes across %u models; independent total=%" PRIu64
           " bytes; saved=%" PRIu64 " bytes\n",
           shared_total, model_count, unshared_total, unshared_total - shared_total);
    return 0;

failed:
    ctx->internal_mem_state = PARAFORMER_INTERNAL_MEM_FAILED;
    return -1;
}

/** Releases encoder tensor memory and its RKNN3 context in dependency order. */
static void release_encoder_resources(rknn_app_context_t* ctx) {
    for (uint32_t i = 0; i < ctx->encoder_n_inputs; ++i) {
        if (ctx->encoder_input_mem[i]) {
            int ret = rknn3_destroy_mem(ctx->encoder_ctx, ctx->encoder_input_mem[i]);
            if (ret != RKNN3_SUCCESS) {
                printf("Warning: destroy encoder input%u failed: %d\n", i, ret);
            }
            ctx->encoder_input_mem[i] = NULL;
        }
    }
    for (uint32_t i = 0; i < ctx->encoder_n_outputs; ++i) {
        if (ctx->encoder_output_mem[i]) {
            int ret = rknn3_destroy_mem(ctx->encoder_ctx, ctx->encoder_output_mem[i]);
            if (ret != RKNN3_SUCCESS) {
                printf("Warning: destroy encoder output%u failed: %d\n", i, ret);
            }
            ctx->encoder_output_mem[i] = NULL;
        }
    }
    if (ctx->encoder_ctx) {
        int ret = rknn3_destroy(ctx->encoder_ctx);
        if (ret != RKNN3_SUCCESS) {
            printf("Warning: destroy encoder context failed: %d\n", ret);
        }
        ctx->encoder_ctx = 0;
    }
    ctx->encoder_n_inputs = 0;
    ctx->encoder_n_outputs = 0;
}

/** Releases decoder tensor memory and its RKNN3 context in dependency order. */
static void release_decoder_resources(rknn_app_context_t* ctx) {
    for (uint32_t i = 0; i < ctx->decoder_n_inputs; ++i) {
        if (ctx->decoder_input_mem[i]) {
            int ret = rknn3_destroy_mem(ctx->decoder_ctx, ctx->decoder_input_mem[i]);
            if (ret != RKNN3_SUCCESS) {
                printf("Warning: destroy decoder input%u failed: %d\n", i, ret);
            }
            ctx->decoder_input_mem[i] = NULL;
        }
    }
    for (uint32_t i = 0; i < ctx->decoder_n_outputs; ++i) {
        if (ctx->decoder_output_mem[i]) {
            int ret = rknn3_destroy_mem(ctx->decoder_ctx, ctx->decoder_output_mem[i]);
            if (ret != RKNN3_SUCCESS) {
                printf("Warning: destroy decoder output%u failed: %d\n", i, ret);
            }
            ctx->decoder_output_mem[i] = NULL;
        }
    }
    if (ctx->decoder_ctx) {
        int ret = rknn3_destroy(ctx->decoder_ctx);
        if (ret != RKNN3_SUCCESS) {
            printf("Warning: destroy decoder context failed: %d\n", ret);
        }
        ctx->decoder_ctx = 0;
    }
    ctx->decoder_n_inputs = 0;
    ctx->decoder_n_outputs = 0;
}

/**
 * Loads and initializes the encoder, validates its interface and dynamic
 * buckets, and allocates reusable input/output tensor memory.
 */
static int init_encoder_model(const char* model_path, const char* weight_path, rknn_app_context_t* ctx) {
    int ret;

    ret = rknn3_init(&ctx->encoder_ctx, NULL);
    if (ret != RKNN3_SUCCESS) {
        printf("Error: rknn3_init failed for encoder: %d\n", ret);
        return -1;
    }

    ret = rknn3_load_model_from_path(ctx->encoder_ctx, model_path, weight_path);
    if (ret != RKNN3_SUCCESS) {
        printf("Error: rknn3_load_model_from_path failed for encoder: %d\n", ret);
        release_encoder_resources(ctx);
        return -1;
    }

    if (resolve_core_mask(ctx->encoder_ctx, "Encoder", &ctx->core_mask) != 0) {
        release_encoder_resources(ctx);
        return -1;
    }

    rknn3_config rknn_config;
    memset(&rknn_config, 0, sizeof(rknn_config));
    rknn_config.run_core_mask = ctx->core_mask;
    rknn_config.user_mem_internal = 1;

    ret = rknn3_model_init(ctx->encoder_ctx, &rknn_config);
    if (ret != RKNN3_SUCCESS) {
        printf("Error: rknn3_model_init failed for encoder: %d\n", ret);
        release_encoder_resources(ctx);
        return -1;
    }

    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx->encoder_ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN3_SUCCESS) {
        printf("Error: rknn3_query IN_OUT_NUM failed for encoder: %d\n", ret);
        release_encoder_resources(ctx);
        return -1;
    }

    if (io_num.n_input != 2 || io_num.n_output != 2) {
        printf("Error: unsupported encoder I/O count: inputs=%u outputs=%u "
               "(expected 2/2)\n",
               io_num.n_input, io_num.n_output);
        release_encoder_resources(ctx);
        return -1;
    }
    ctx->encoder_n_inputs = io_num.n_input;
    ctx->encoder_n_outputs = io_num.n_output;

    for (uint32_t i = 0; i < ctx->encoder_n_inputs; ++i) {
        ctx->encoder_input_attrs[i].index = i;
        ret = rknn3_query(ctx->encoder_ctx, RKNN3_QUERY_INPUT_ATTR, &ctx->encoder_input_attrs[i],
                          sizeof(rknn3_tensor_attr));
        if (ret != RKNN3_SUCCESS) {
            printf("Error: rknn3_query INPUT_ATTR[%u] failed for encoder: %d\n", i, ret);
            release_encoder_resources(ctx);
            return -1;
        }
    }

    for (uint32_t i = 0; i < ctx->encoder_n_outputs; ++i) {
        ctx->encoder_output_attrs[i].index = i;
        ret = rknn3_query(ctx->encoder_ctx, RKNN3_QUERY_OUTPUT_ATTR, &ctx->encoder_output_attrs[i],
                          sizeof(rknn3_tensor_attr));
        if (ret != RKNN3_SUCCESS) {
            printf("Error: rknn3_query OUTPUT_ATTR[%u] failed for encoder: %d\n", i, ret);
            release_encoder_resources(ctx);
            return -1;
        }
    }

    for (uint32_t i = 0; i < ctx->encoder_n_inputs; ++i) {
        ctx->encoder_input_mem[i] = rknn3_create_mem(ctx->encoder_ctx, ctx->encoder_input_attrs[i].aligned_size,
                                                     ctx->encoder_input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!ctx->encoder_input_mem[i]) {
            printf("Error: rknn3_create_mem failed for encoder input %u\n", i);
            release_encoder_resources(ctx);
            return -1;
        }
    }

    for (uint32_t i = 0; i < ctx->encoder_n_outputs; ++i) {
        ctx->encoder_output_mem[i] =
            rknn3_create_mem(ctx->encoder_ctx, ctx->encoder_output_attrs[i].aligned_size,
                             ctx->encoder_output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!ctx->encoder_output_mem[i]) {
            printf("Error: rknn3_create_mem failed for encoder output %u\n", i);
            release_encoder_resources(ctx);
            return -1;
        }
    }

    for (uint32_t i = 0; i < ctx->encoder_n_inputs; ++i) {
        printf("[Encoder] input%u: n_dims=%u, layout=%d, shape=[%u,%u,%u,%u,%u], "
               "aligned_size=%" PRIu64 ", dtype=%d\n",
               i, ctx->encoder_input_attrs[i].n_dims, ctx->encoder_input_attrs[i].layout,
               ctx->encoder_input_attrs[i].shape[0], ctx->encoder_input_attrs[i].shape[1],
               ctx->encoder_input_attrs[i].shape[2], ctx->encoder_input_attrs[i].shape[3],
               ctx->encoder_input_attrs[i].shape[4], ctx->encoder_input_attrs[i].aligned_size,
               ctx->encoder_input_attrs[i].dtype);
    }
    for (uint32_t i = 0; i < ctx->encoder_n_outputs; ++i) {
        printf("[Encoder] output%u: n_dims=%u, layout=%d, shape=[%u,%u,%u,%u,%u], "
               "aligned_size=%" PRIu64 ", dtype=%d\n",
               i, ctx->encoder_output_attrs[i].n_dims, ctx->encoder_output_attrs[i].layout,
               ctx->encoder_output_attrs[i].shape[0], ctx->encoder_output_attrs[i].shape[1],
               ctx->encoder_output_attrs[i].shape[2], ctx->encoder_output_attrs[i].shape[3],
               ctx->encoder_output_attrs[i].shape[4], ctx->encoder_output_attrs[i].aligned_size,
               ctx->encoder_output_attrs[i].dtype);
    }

    memset(&ctx->encoder_shape_config, 0, sizeof(ctx->encoder_shape_config));
    ret = rknn3_query(ctx->encoder_ctx, RKNN3_QUERY_DYNAMIC_SHAPE_CONFIG, &ctx->encoder_shape_config,
                      sizeof(ctx->encoder_shape_config));
    if (ret != RKNN3_SUCCESS) {
        printf("Error: query encoder dynamic shapes failed: %d\n", ret);
        release_encoder_resources(ctx);
        return -1;
    }
    if (ctx->encoder_shape_config.n_shapes > PARAFORMER_MAX_ENCODER_SHAPES) {
        printf("Error: encoder shape count %u exceeds limit %u\n", ctx->encoder_shape_config.n_shapes,
               PARAFORMER_MAX_ENCODER_SHAPES);
        release_encoder_resources(ctx);
        return -1;
    }
    if (ctx->encoder_shape_config.n_shapes == 0) {
        printf("Error: encoder model does not contain dynamic shapes\n");
        release_encoder_resources(ctx);
        return -1;
    }
    {
        uint32_t n_shapes = ctx->encoder_shape_config.n_shapes;
        printf("[Encoder] %u dynamic shapes:\n", n_shapes);
        for (uint32_t s = 0; s < n_shapes; s++) {
            int32_t sid = (int32_t)s;
            ret = rknn3_set_shape(ctx->encoder_ctx, sid);
            if (ret != RKNN3_SUCCESS) {
                printf("  shape[%d]: set_shape failed: %d\n", s, ret);
                release_encoder_resources(ctx);
                return -1;
            }

            rknn3_tensor_attr in_attrs[2];
            rknn3_tensor_attr out_attrs[2];
            memset(in_attrs, 0, sizeof(in_attrs));
            memset(out_attrs, 0, sizeof(out_attrs));
            for (uint32_t i = 0; i < ctx->encoder_n_inputs; ++i) {
                in_attrs[i].index = i;
                ret = rknn3_query(ctx->encoder_ctx, RKNN3_QUERY_INPUT_ATTR, &in_attrs[i], sizeof(rknn3_tensor_attr));
                if (ret != RKNN3_SUCCESS) {
                    printf("  shape[%u]: query input%u failed: %d\n", s, i, ret);
                    release_encoder_resources(ctx);
                    return -1;
                }
            }
            for (uint32_t i = 0; i < ctx->encoder_n_outputs; ++i) {
                out_attrs[i].index = i;
                ret = rknn3_query(ctx->encoder_ctx, RKNN3_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(rknn3_tensor_attr));
                if (ret != RKNN3_SUCCESS) {
                    printf("  shape[%u]: query output%u failed: %d\n", s, i, ret);
                    release_encoder_resources(ctx);
                    return -1;
                }
            }
            if (validate_fp16_tensor_attr(&in_attrs[0], "encoder speech input", PARAFORMER_INPUT_FEAT_DIM) != 0 ||
                validate_fp16_tensor_attr(&in_attrs[1], "encoder mask input", 1) != 0 ||
                validate_fp16_tensor_attr(&out_attrs[0], "encoder feature output", PARAFORMER_ENCODER_DIM) != 0 ||
                validate_fp16_tensor_attr(&out_attrs[1], "encoder alpha output", 1) != 0) {
                release_encoder_resources(ctx);
                return -1;
            }

            ctx->encoder_shape_ids[s] = sid;
            ctx->encoder_shape_input_attrs[s] = in_attrs[0];
            ctx->encoder_shape_mask_attrs[s] = in_attrs[1];
            ctx->encoder_shape_output_attrs[s] = out_attrs[0];
            ctx->encoder_shape_alpha_attrs[s] = out_attrs[1];
            printf("  shape[%u]: id=%d, speech=[1,%u,%u], mask=[1,%u,%u], "
                   "enc=[1,%u,%u], alphas=[1,%u,%u]\n",
                   s, sid, in_attrs[0].shape[1], in_attrs[0].shape[2], in_attrs[1].shape[1], in_attrs[1].shape[2],
                   out_attrs[0].shape[1], out_attrs[0].shape[2], out_attrs[1].shape[1], out_attrs[1].shape[2]);
        }
    }

    return 0;
}

/**
 * Loads the masked decoder. Baseline models expose logits only, while the
 * hotword variant also exposes decoder_hidden for SeaCo and ASF.
 */
static int init_decoder_model(const char* model_path, const char* weight_path, rknn_app_context_t* ctx) {
    int ret;

    ret = rknn3_init(&ctx->decoder_ctx, NULL);
    if (ret != RKNN3_SUCCESS) {
        printf("Error: rknn3_init failed for decoder: %d\n", ret);
        return -1;
    }

    ret = rknn3_load_model_from_path(ctx->decoder_ctx, model_path, weight_path);
    if (ret != RKNN3_SUCCESS) {
        printf("Error: rknn3_load_model_from_path failed for decoder: %d\n", ret);
        release_decoder_resources(ctx);
        return -1;
    }

    if (resolve_core_mask(ctx->decoder_ctx, "Decoder", &ctx->core_mask) != 0) {
        release_decoder_resources(ctx);
        return -1;
    }

    rknn3_config rknn_config;
    memset(&rknn_config, 0, sizeof(rknn_config));
    rknn_config.run_core_mask = ctx->core_mask;
    rknn_config.user_mem_internal = 1;

    ret = rknn3_model_init(ctx->decoder_ctx, &rknn_config);
    if (ret != RKNN3_SUCCESS) {
        printf("Error: rknn3_model_init failed for decoder: %d\n", ret);
        release_decoder_resources(ctx);
        return -1;
    }

    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx->decoder_ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN3_SUCCESS) {
        printf("Error: rknn3_query IN_OUT_NUM failed for decoder: %d\n", ret);
        release_decoder_resources(ctx);
        return -1;
    }

    if (io_num.n_input != 4 || (io_num.n_output != 1 && io_num.n_output != 2)) {
        printf("Error: unsupported decoder I/O count: inputs=%u outputs=%u "
               "(expected masked decoder 4/1 or hotword decoder 4/2)\n",
               io_num.n_input, io_num.n_output);
        release_decoder_resources(ctx);
        return -1;
    }
    ctx->decoder_n_inputs = io_num.n_input;
    ctx->decoder_n_outputs = io_num.n_output;

    // Inputs are enc, pre_acoustic_embeds, token_mask, and enc_mask.
    for (uint32_t i = 0; i < ctx->decoder_n_inputs; i++) {
        ctx->decoder_input_attrs[i].index = i;
        ret = rknn3_query(ctx->decoder_ctx, RKNN3_QUERY_INPUT_ATTR, &ctx->decoder_input_attrs[i],
                          sizeof(rknn3_tensor_attr));
        if (ret != RKNN3_SUCCESS) {
            printf("Error: rknn3_query INPUT_ATTR failed for decoder input %u: %d\n", i, ret);
            release_decoder_resources(ctx);
            return -1;
        }

        ctx->decoder_input_mem[i] = rknn3_create_mem(ctx->decoder_ctx, ctx->decoder_input_attrs[i].aligned_size,
                                                     ctx->decoder_input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!ctx->decoder_input_mem[i]) {
            printf("Error: rknn3_create_mem failed for decoder input %u\n", i);
            release_decoder_resources(ctx);
            return -1;
        }
    }

    for (uint32_t i = 0; i < ctx->decoder_n_outputs; i++) {
        ctx->decoder_output_attrs[i].index = i;
        ret = rknn3_query(ctx->decoder_ctx, RKNN3_QUERY_OUTPUT_ATTR, &ctx->decoder_output_attrs[i],
                          sizeof(rknn3_tensor_attr));
        if (ret != RKNN3_SUCCESS) {
            printf("Error: rknn3_query OUTPUT_ATTR failed for decoder output %u: %d\n", i, ret);
            release_decoder_resources(ctx);
            return -1;
        }

        ctx->decoder_output_mem[i] =
            rknn3_create_mem(ctx->decoder_ctx, ctx->decoder_output_attrs[i].aligned_size,
                             ctx->decoder_output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (!ctx->decoder_output_mem[i]) {
            printf("Error: rknn3_create_mem failed for decoder output %u\n", i);
            release_decoder_resources(ctx);
            return -1;
        }
    }

    printf("[Decoder] input0: n_dims=%u, layout=%d, shape=[%u,%u,%u,%u,%u], "
           "size=%" PRIu64 "\n",
           ctx->decoder_input_attrs[0].n_dims, ctx->decoder_input_attrs[0].layout, ctx->decoder_input_attrs[0].shape[0],
           ctx->decoder_input_attrs[0].shape[1], ctx->decoder_input_attrs[0].shape[2],
           ctx->decoder_input_attrs[0].shape[3], ctx->decoder_input_attrs[0].shape[4],
           ctx->decoder_input_attrs[0].aligned_size);
    printf("[Decoder] input1: n_dims=%u, layout=%d, shape=[%u,%u,%u,%u,%u], "
           "size=%" PRIu64 "\n",
           ctx->decoder_input_attrs[1].n_dims, ctx->decoder_input_attrs[1].layout, ctx->decoder_input_attrs[1].shape[0],
           ctx->decoder_input_attrs[1].shape[1], ctx->decoder_input_attrs[1].shape[2],
           ctx->decoder_input_attrs[1].shape[3], ctx->decoder_input_attrs[1].shape[4],
           ctx->decoder_input_attrs[1].aligned_size);
    printf("[Decoder] input2: n_dims=%u, layout=%d, shape=[%u,%u,%u,%u,%u], "
           "size=%" PRIu64 ", dtype=%d\n",
           ctx->decoder_input_attrs[2].n_dims, ctx->decoder_input_attrs[2].layout, ctx->decoder_input_attrs[2].shape[0],
           ctx->decoder_input_attrs[2].shape[1], ctx->decoder_input_attrs[2].shape[2],
           ctx->decoder_input_attrs[2].shape[3], ctx->decoder_input_attrs[2].shape[4],
           ctx->decoder_input_attrs[2].aligned_size, ctx->decoder_input_attrs[2].dtype);
    printf("[Decoder] input3: n_dims=%u, layout=%d, shape=[%u,%u,%u,%u,%u], "
           "size=%" PRIu64 ", dtype=%d\n",
           ctx->decoder_input_attrs[3].n_dims, ctx->decoder_input_attrs[3].layout, ctx->decoder_input_attrs[3].shape[0],
           ctx->decoder_input_attrs[3].shape[1], ctx->decoder_input_attrs[3].shape[2],
           ctx->decoder_input_attrs[3].shape[3], ctx->decoder_input_attrs[3].shape[4],
           ctx->decoder_input_attrs[3].aligned_size, ctx->decoder_input_attrs[3].dtype);
    for (uint32_t i = 0; i < ctx->decoder_n_outputs; i++) {
        printf("[Decoder] output%u: n_dims=%u, layout=%d, shape=[%u,%u,%u,%u,%u], "
               "size=%" PRIu64 ", dtype=%d\n",
               i, ctx->decoder_output_attrs[i].n_dims, ctx->decoder_output_attrs[i].layout,
               ctx->decoder_output_attrs[i].shape[0], ctx->decoder_output_attrs[i].shape[1],
               ctx->decoder_output_attrs[i].shape[2], ctx->decoder_output_attrs[i].shape[3],
               ctx->decoder_output_attrs[i].shape[4], ctx->decoder_output_attrs[i].aligned_size,
               ctx->decoder_output_attrs[i].dtype);
    }

    memset(&ctx->decoder_shape_config, 0, sizeof(ctx->decoder_shape_config));
    ret = rknn3_query(ctx->decoder_ctx, RKNN3_QUERY_DYNAMIC_SHAPE_CONFIG, &ctx->decoder_shape_config,
                      sizeof(ctx->decoder_shape_config));
    if (ret != RKNN3_SUCCESS) {
        printf("Error: query decoder dynamic shapes failed: %d\n", ret);
        release_decoder_resources(ctx);
        return -1;
    }
    if (ctx->decoder_shape_config.n_shapes > PARAFORMER_MAX_DECODER_SHAPES) {
        printf("Error: decoder shape count %u exceeds limit %u\n", ctx->decoder_shape_config.n_shapes,
               PARAFORMER_MAX_DECODER_SHAPES);
        release_decoder_resources(ctx);
        return -1;
    }
    if (ctx->decoder_shape_config.n_shapes == 0) {
        printf("Error: decoder model does not contain dynamic shapes\n");
        release_decoder_resources(ctx);
        return -1;
    }
    {
        uint32_t n_shapes = ctx->decoder_shape_config.n_shapes;
        printf("[Decoder] %u dynamic shapes:\n", n_shapes);
        for (uint32_t s = 0; s < n_shapes; s++) {
            int32_t sid = (int32_t)s;
            ret = rknn3_set_shape(ctx->decoder_ctx, sid);
            if (ret != RKNN3_SUCCESS) {
                printf("  shape[%d]: set_shape failed: %d\n", s, ret);
                release_decoder_resources(ctx);
                return -1;
            }

            rknn3_tensor_attr in_attrs[4], out_attrs[2];
            memset(in_attrs, 0, sizeof(in_attrs));
            memset(out_attrs, 0, sizeof(out_attrs));
            for (uint32_t input_i = 0; input_i < 4; ++input_i) {
                in_attrs[input_i].index = input_i;
                ret = rknn3_query(ctx->decoder_ctx, RKNN3_QUERY_INPUT_ATTR, &in_attrs[input_i],
                                  sizeof(rknn3_tensor_attr));
                if (ret != RKNN3_SUCCESS) {
                    printf("  shape[%d]: query input%u failed: %d\n", s, input_i, ret);
                    release_decoder_resources(ctx);
                    return -1;
                }
            }
            for (uint32_t output_i = 0; output_i < ctx->decoder_n_outputs; ++output_i) {
                out_attrs[output_i].index = output_i;
                ret = rknn3_query(ctx->decoder_ctx, RKNN3_QUERY_OUTPUT_ATTR, &out_attrs[output_i],
                                  sizeof(rknn3_tensor_attr));
                if (ret != RKNN3_SUCCESS) {
                    printf("  shape[%d]: query output%u failed: %d\n", s, output_i, ret);
                    release_decoder_resources(ctx);
                    return -1;
                }
            }
            if (validate_fp16_tensor_attr(&in_attrs[0], "decoder encoder input", PARAFORMER_ENCODER_DIM) != 0 ||
                validate_fp16_tensor_attr(&in_attrs[1], "decoder acoustic input", PARAFORMER_ENCODER_DIM) != 0 ||
                validate_fp16_tensor_attr(&in_attrs[2], "decoder token mask input", 1) != 0 ||
                validate_fp16_tensor_attr(&in_attrs[3], "decoder encoder mask input", 1) != 0 ||
                validate_fp16_tensor_attr(&out_attrs[0], "decoder output", PARAFORMER_VOCAB_SIZE) != 0 ||
                (ctx->decoder_n_outputs == 2 &&
                 validate_fp16_tensor_attr(&out_attrs[1], "decoder hidden output", PARAFORMER_ENCODER_DIM) != 0)) {
                release_decoder_resources(ctx);
                return -1;
            }

            ctx->decoder_shape_ids[s] = sid;
            ctx->decoder_shape_in0_attrs[s] = in_attrs[0];
            ctx->decoder_shape_in1_attrs[s] = in_attrs[1];
            ctx->decoder_shape_in2_attrs[s] = in_attrs[2];
            ctx->decoder_shape_in3_attrs[s] = in_attrs[3];
            ctx->decoder_shape_out0_attrs[s] = out_attrs[0];
            if (ctx->decoder_n_outputs == 2) {
                ctx->decoder_shape_out1_attrs[s] = out_attrs[1];
            }
            printf("  shape[%d]: id=%d, in0=[1,%u,%u], in1=[1,%u,%u]", s, sid, in_attrs[0].shape[1],
                   in_attrs[0].shape[2], in_attrs[1].shape[1], in_attrs[1].shape[2]);
            printf(", token_mask=[1,%u,%u], enc_mask=[1,%u,%u]", in_attrs[2].shape[1], in_attrs[2].shape[2],
                   in_attrs[3].shape[1], in_attrs[3].shape[2]);
            printf(", out0=[1,%u,%u]", out_attrs[0].shape[1], out_attrs[0].shape[2]);
            if (ctx->decoder_n_outputs == 2) {
                printf(", out1=[1,%u,%u]", out_attrs[1].shape[1], out_attrs[1].shape[2]);
            }
            printf("\n");
        }
    }

    return 0;
}

/** Initializes every model stage and rolls back all resources on failure. */
int init_paraformer_model(const char* encoder_model_path, const char* decoder_model_path,
                          const char* encoder_weight_path, const char* decoder_weight_path,
                          rknn_app_context_t* ctx, uint32_t core_mask) {
    if (!encoder_model_path || !decoder_model_path || !encoder_weight_path || !decoder_weight_path || !ctx) {
        printf("Error: Invalid parameters for init_paraformer_model\n");
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));

    ctx->core_mask = core_mask;
    if (init_encoder_model(encoder_model_path, encoder_weight_path, ctx) != 0) {
        printf("Error: Failed to initialize encoder model\n");
        return -1;
    }

    if (init_decoder_model(decoder_model_path, decoder_weight_path, ctx) != 0) {
        printf("Error: Failed to initialize decoder model\n");
        release_paraformer_model(ctx);
        return -1;
    }

    printf("Paraformer model initialized successfully\n");
    printf("  Encoder: %s\n", encoder_model_path);
    printf("  Decoder: %s\n", decoder_model_path);
    printf("  Predictor: NPU alpha head + CPU CIF\n");
    printf("  Core mask: 0x%x\n", ctx->core_mask);

    return 0;
}

int init_paraformer_hotword(const char* seaco_model_path, const char* seaco_weight_path,
                            const char* bias_weights_path, const char* hotwords_path, const char* tokens_path,
                            const char* seg_dict_path, int max_hotwords, int max_hotword_len, float seaco_weight,
                            const char* asf_model_path, const char* asf_weight_path, int asf_nfilter,
                            rknn_app_context_t* ctx) {
    paraformer_hotword_context_t* hotword_context = NULL;
    if (!ctx || !ctx->decoder_ctx || ctx->decoder_n_outputs != 2 || ctx->hotword_context ||
        ctx->internal_mem_state != PARAFORMER_INTERNAL_MEM_UNINITIALIZED) {
        printf("Error: hotword mode requires an initialized two-output decoder "
               "and must be enabled once before the first inference\n");
        return -1;
    }
    if (paraformer_hotword_create(seaco_model_path, seaco_weight_path, bias_weights_path, hotwords_path, tokens_path,
                                  seg_dict_path, max_hotwords, max_hotword_len, seaco_weight, asf_model_path,
                                  asf_weight_path, asf_nfilter, ctx->core_mask, &hotword_context) != 0) {
        return -1;
    }
    ctx->hotword_context = hotword_context;
    return 0;
}

/** Releases optional models, shared internal memory, and owner resources. */
void release_paraformer_model(rknn_app_context_t* ctx) {
    if (!ctx)
        return;

    paraformer_hotword_release((paraformer_hotword_context_t*)ctx->hotword_context);
    ctx->hotword_context = NULL;
    release_decoder_resources(ctx);
    release_internal_memory(ctx);
    release_encoder_resources(ctx);

    memset(ctx, 0, sizeof(*ctx));
}

/**
 * Executes one complete offline recognition request and releases all
 * request-scoped feature, embedding, logit, and token buffers before return.
 */
int run_paraformer_inference(rknn_app_context_t* ctx, const int16_t* audio_data, int audio_samples,
                             const char* tokens_path, char* output_text, int output_size) {
    float* feats = NULL;
    int n_frames = 0;
    float_buffer_t enc_out;
    float_buffer_t encoder_alpha;
    float_buffer_t predictor_embeds;
    float_buffer_t decoder_logits;
    float_buffer_t decoder_hidden;
    int predictor_token_num = 0;
    int token_len = 0;
    int vocab_size = 0;
    int32_t* token_ids = NULL;
    int token_count = 0;
    int ret = -1;
    int result = -1;

    if (!ctx || !audio_data || !tokens_path || !output_text) {
        printf("Error: Invalid parameters for run_paraformer_inference\n");
        return -1;
    }
    if (ctx->internal_mem_state == PARAFORMER_INTERNAL_MEM_FAILED ||
        (ctx->internal_mem_state == PARAFORMER_INTERNAL_MEM_UNINITIALIZED &&
         init_shared_internal_memory(ctx) != 0)) {
        printf("Error: shared RKNN3 internal memory initialization failed\n");
        return -1;
    }

    memset(&enc_out, 0, sizeof(enc_out));
    memset(&encoder_alpha, 0, sizeof(encoder_alpha));
    memset(&predictor_embeds, 0, sizeof(predictor_embeds));
    memset(&decoder_logits, 0, sizeof(decoder_logits));
    memset(&decoder_hidden, 0, sizeof(decoder_hidden));
    ret = extract_fbank(audio_data, audio_samples, &feats, &n_frames);
    if (ret != 0 || !feats || n_frames <= 0) {
        printf("Error: extract_fbank failed\n");
        goto cleanup;
    }

    printf("[Inference] fbank: frames=%d dim=%d\n", n_frames, PARAFORMER_INPUT_FEAT_DIM);

    ret = run_encoder(ctx, feats, n_frames, &enc_out, &encoder_alpha);
    free(feats);
    feats = NULL;
    if (ret != 0) {
        printf("Error: run_encoder failed\n");
        goto cleanup;
    }
    ret = run_cif_from_alpha(&enc_out, &encoder_alpha, (int)ctx->encoder_time, PARAFORMER_ENCODER_DIM,
                             &predictor_embeds, &predictor_token_num);
    if (ret != 0) {
        printf("Error: run_cif_from_alpha failed\n");
        goto cleanup;
    }

    ret = run_decoder(ctx, &enc_out, (int)ctx->encoder_time, &predictor_embeds, predictor_token_num, &decoder_logits,
                      ctx->hotword_context ? &decoder_hidden : NULL, &token_len, &vocab_size);
    if (ret != 0) {
        printf("Error: run_decoder failed\n");
        goto cleanup;
    }
    if (ctx->hotword_context &&
        paraformer_hotword_apply((paraformer_hotword_context_t*)ctx->hotword_context, predictor_embeds.data,
                                 predictor_embeds.size, decoder_hidden.data, decoder_hidden.size, predictor_token_num,
                                 decoder_logits.data, decoder_logits.size, token_len, vocab_size) != 0) {
        printf("Error: hotword pipeline failed\n");
        goto cleanup;
    }
    printf("[Perf] encoder=%dframes, predictor=%dtokens, decoder_bucket=%d\n", (int)ctx->encoder_time,
           predictor_token_num, find_best_decoder_shape(ctx, (int)ctx->encoder_time, predictor_token_num));

    ret = argmax_decode(&decoder_logits, token_len, vocab_size, &token_ids, &token_count);
    if (ret != 0) {
        printf("Error: argmax_decode failed\n");
        goto cleanup;
    }

    ret = decode_tokens(token_ids, token_count, tokens_path, output_text, output_size);
    if (ret != 0) {
        printf("Error: Failed to decode tokens with %s\n", tokens_path);
        goto cleanup;
    }
    result = 0;

cleanup:
    free(token_ids);
    release_float_buffer(&decoder_hidden);
    release_float_buffer(&decoder_logits);
    release_float_buffer(&predictor_embeds);
    release_float_buffer(&encoder_alpha);
    release_float_buffer(&enc_out);
    free(feats);
    return result;
}
