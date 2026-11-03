#include "rknn_gemma4_audio_stream.h"

#include "pocketfft_hdronly.h"

#include <complex>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GEMMA4_STREAM_SAMPLE_RATE 16000
#define GEMMA4_STREAM_FRAME_LENGTH 320
#define GEMMA4_STREAM_FFT_LENGTH 512
#define GEMMA4_STREAM_HOP_LENGTH 160
#define GEMMA4_STREAM_MEL_BINS 128
#define GEMMA4_STREAM_STATE_TOKENS 12
#define GEMMA4_STREAM_EMBEDDING_DIM 1536
#define GEMMA4_STREAM_PI 3.14159265358979323846f

struct rknn_gemma4_audio_stream_context {
    rknn3_context ctx;
    rknn3_input_output_num io_num;
    uint32_t n_buckets;
    rknn3_shape_info* shape_infos;
    rknn3_tensor_attr* bucket_input_attrs;
    rknn3_tensor_attr* bucket_output_attrs;
    int* bucket_frames;
    int current_bucket;
    uint64_t chunk_index;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;

    float* pcm;
    size_t pcm_size;
    size_t pcm_capacity;
    float* mel;
    float* mel_mask;
    size_t mel_frames;
    size_t mel_capacity_frames;

    float mel_filters[(GEMMA4_STREAM_FFT_LENGTH / 2 + 1) * GEMMA4_STREAM_MEL_BINS];
    float window[GEMMA4_STREAM_FRAME_LENGTH];
};

static rknn3_tensor_attr* get_bucket_input_attrs(
    rknn_gemma4_audio_stream_context* stream, uint32_t bucket)
{
    return stream->bucket_input_attrs + (size_t)bucket * stream->io_num.n_input;
}

static rknn3_tensor_attr* get_bucket_output_attrs(
    rknn_gemma4_audio_stream_context* stream, uint32_t bucket)
{
    return stream->bucket_output_attrs + (size_t)bucket * stream->io_num.n_output;
}

static size_t get_dtype_size(rknn3_tensor_type dtype)
{
    if (dtype == RKNN3_TENSOR_FLOAT16) {
        return sizeof(float16);
    }
    if (dtype == RKNN3_TENSOR_FLOAT32) {
        return sizeof(float);
    }
    return 0;
}

static int reserve_float_buffer(float** buffer, size_t* capacity, size_t required)
{
    float* new_buffer;
    size_t new_capacity;

    if (required <= *capacity) {
        return 0;
    }
    new_capacity = (*capacity == 0) ? 1024 : *capacity;
    while (new_capacity < required) {
        if (new_capacity > SIZE_MAX / 2) {
            return -1;
        }
        new_capacity *= 2;
    }
    new_buffer = (float*)realloc(*buffer, new_capacity * sizeof(float));
    if (new_buffer == NULL) {
        return -1;
    }
    *buffer = new_buffer;
    *capacity = new_capacity;
    return 0;
}

static int reserve_mel_buffer(rknn_gemma4_audio_stream_context* stream,
                              size_t required_frames)
{
    float* new_mel;
    float* new_mask;
    size_t new_capacity;

    if (required_frames <= stream->mel_capacity_frames) {
        return 0;
    }
    new_capacity = (stream->mel_capacity_frames == 0) ? 48 : stream->mel_capacity_frames;
    while (new_capacity < required_frames) {
        if (new_capacity > SIZE_MAX / 2) {
            return -1;
        }
        new_capacity *= 2;
    }

    new_mel = (float*)realloc(
        stream->mel, new_capacity * GEMMA4_STREAM_MEL_BINS * sizeof(float));
    if (new_mel == NULL) {
        return -1;
    }
    stream->mel = new_mel;

    new_mask = (float*)realloc(stream->mel_mask, new_capacity * sizeof(float));
    if (new_mask == NULL) {
        return -1;
    }
    stream->mel_mask = new_mask;
    stream->mel_capacity_frames = new_capacity;
    return 0;
}

static void build_mel_filters(float* filters)
{
    const int n_freqs = GEMMA4_STREAM_FFT_LENGTH / 2 + 1;
    const float mel_max = 2595.0f * log10f(1.0f + 8000.0f / 700.0f);
    float points[GEMMA4_STREAM_MEL_BINS + 2];
    int i;
    int k;
    int m;

    memset(filters, 0, (size_t)n_freqs * GEMMA4_STREAM_MEL_BINS * sizeof(float));
    for (i = 0; i < GEMMA4_STREAM_MEL_BINS + 2; ++i) {
        float mel = mel_max * (float)i / (float)(GEMMA4_STREAM_MEL_BINS + 1);
        points[i] = 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
    }
    for (k = 0; k < n_freqs; ++k) {
        float freq = (float)k * GEMMA4_STREAM_SAMPLE_RATE / GEMMA4_STREAM_FFT_LENGTH;
        for (m = 0; m < GEMMA4_STREAM_MEL_BINS; ++m) {
            float lower = (freq - points[m]) / (points[m + 1] - points[m]);
            float upper = (points[m + 2] - freq) / (points[m + 2] - points[m + 1]);
            float value = (lower < upper) ? lower : upper;
            filters[(size_t)k * GEMMA4_STREAM_MEL_BINS + m] =
                (value > 0.0f) ? value : 0.0f;
        }
    }
}

static void init_hann_window(rknn_gemma4_audio_stream_context* stream)
{
    int i;
    for (i = 0; i < GEMMA4_STREAM_FRAME_LENGTH; ++i) {
        stream->window[i] = 0.5f *
            (1.0f - cosf(2.0f * GEMMA4_STREAM_PI * (float)i /
                         GEMMA4_STREAM_FRAME_LENGTH));
    }
}

static int write_float_tensor(const rknn3_tensor_attr* attr, void* dst,
                              const float* src, size_t count)
{
    size_t i;
    if (attr == NULL || dst == NULL || src == NULL || count > attr->n_elems) {
        return -1;
    }
    memset(dst, 0, (size_t)attr->aligned_size);
    if (attr->dtype == RKNN3_TENSOR_FLOAT16) {
        float16* output = (float16*)dst;
        for (i = 0; i < count; ++i) {
            output[i] = fp32_to_fp16(src[i]);
        }
        return 0;
    }
    if (attr->dtype == RKNN3_TENSOR_FLOAT32) {
        memcpy(dst, src, count * sizeof(float));
        return 0;
    }
    return -1;
}

static float read_float_tensor(const rknn3_tensor_attr* attr,
                               const void* src, size_t index)
{
    if (attr->dtype == RKNN3_TENSOR_FLOAT16) {
        return fp16_to_fp32(((const float16*)src)[index]);
    }
    return ((const float*)src)[index];
}

static int nchw_fp16_to_nc1hwc2(const rknn3_tensor* src, rknn3_tensor* dst)
{
    const rknn3_tensor_attr* src_attr = src->attr;
    const rknn3_tensor_attr* dst_attr = dst->attr;
    const float16* input;
    float16* output;
    uint32_t n;
    uint32_t c;
    uint32_t h;
    uint32_t w;
    uint32_t c1;
    uint32_t c2;
    uint32_t batch;
    uint32_t channel;
    uint32_t y;
    uint32_t x;

    if (src_attr->dtype != RKNN3_TENSOR_FLOAT16 ||
        dst_attr->dtype != RKNN3_TENSOR_FLOAT16 ||
        src_attr->layout != RKNN3_TENSOR_NCHW ||
        dst_attr->layout != RKNN3_TENSOR_NC1HWC2 ||
        src_attr->n_dims != 4 || dst_attr->n_dims != 5) {
        return -1;
    }

    n = src_attr->shape[0];
    c = src_attr->shape[1];
    h = src_attr->shape[2];
    w = src_attr->shape[3];
    c1 = dst_attr->shape[1];
    c2 = dst_attr->shape[4];
    if (dst_attr->shape[0] != n || dst_attr->shape[2] != h ||
        dst_attr->shape[3] != w || c > c1 * c2) {
        return -1;
    }

    input = (const float16*)src->mem->virt_addr;
    output = (float16*)dst->mem->virt_addr;
    memset(output, 0, (size_t)dst_attr->aligned_size);
    for (batch = 0; batch < n; ++batch) {
        for (channel = 0; channel < c; ++channel) {
            uint32_t block = channel / c2;
            uint32_t inner = channel % c2;
            for (y = 0; y < h; ++y) {
                for (x = 0; x < w; ++x) {
                    size_t src_index = (((size_t)batch * c + channel) * h + y) * w + x;
                    size_t dst_index =
                        ((((size_t)batch * c1 + block) * h + y) * w + x) * c2 + inner;
                    output[dst_index] = input[src_index];
                }
            }
        }
    }
    return 0;
}

static int feedback_tensor(const rknn3_tensor* src, rknn3_tensor* dst)
{
    size_t bytes;
    if (src->attr->layout == RKNN3_TENSOR_NCHW &&
        dst->attr->layout == RKNN3_TENSOR_NC1HWC2) {
        return nchw_fp16_to_nc1hwc2(src, dst);
    }
    if (src->attr->dtype != dst->attr->dtype ||
        src->attr->n_elems != dst->attr->n_elems) {
        return -1;
    }
    bytes = src->attr->n_elems * get_dtype_size(src->attr->dtype);
    if (bytes == 0 || bytes > src->mem->size || bytes > dst->mem->size) {
        return -1;
    }
    memset(dst->mem->virt_addr, 0, (size_t)dst->attr->aligned_size);
    memcpy(dst->mem->virt_addr, src->mem->virt_addr, bytes);
    return 0;
}

static int feedback_tail_tensor(const rknn3_tensor* src, rknn3_tensor* dst)
{
    size_t element_size;
    size_t src_elements;
    size_t dst_elements;
    size_t bytes;
    const unsigned char* src_data;

    if (src->attr->dtype != dst->attr->dtype ||
        src->attr->n_elems < dst->attr->n_elems) {
        return -1;
    }
    element_size = get_dtype_size(src->attr->dtype);
    src_elements = src->attr->n_elems;
    dst_elements = dst->attr->n_elems;
    bytes = dst_elements * element_size;
    if (element_size == 0 || bytes > dst->mem->size ||
        src_elements * element_size > src->mem->size) {
        return -1;
    }
    src_data = (const unsigned char*)src->mem->virt_addr;
    memset(dst->mem->virt_addr, 0, dst->mem->size);
    memcpy(dst->mem->virt_addr,
           src_data + (src_elements - dst_elements) * element_size,
           bytes);
    return 0;
}

static int validate_stream_model(rknn_gemma4_audio_stream_context* stream)
{
    static const char* input_names[8] = {
        "input_features", "input_features_mask", "conv_state_0", "conv_state_1",
        "audio_history_mask", "audio_history_key", "audio_history_value", "past_conv"
    };
    static const char* output_names[7] = {
        "hidden_states", "output_mask", "next_conv_state_0", "next_conv_state_1",
        "audio_current_key", "audio_current_value", "next_conv"
    };
    uint32_t bucket;

    if (stream->io_num.n_input != 8 || stream->io_num.n_output != 7 ||
        stream->n_buckets == 0) {
        return -1;
    }
    for (bucket = 0; bucket < stream->n_buckets; ++bucket) {
        rknn3_tensor_attr* inputs = get_bucket_input_attrs(stream, bucket);
        rknn3_tensor_attr* outputs = get_bucket_output_attrs(stream, bucket);
        int frames;
        int tokens;
        int i;

        for (i = 0; i < 8; ++i) {
            if (strcmp(inputs[i].name, input_names[i]) != 0) {
                return -1;
            }
        }
        for (i = 0; i < 7; ++i) {
            if (strcmp(outputs[i].name, output_names[i]) != 0) {
                return -1;
            }
        }
        frames = inputs[0].shape[1];
        tokens = frames / 4;
        if (frames <= 0 || frames % 48 != 0 ||
            inputs[0].n_elems != (uint32_t)(frames * GEMMA4_STREAM_MEL_BINS) ||
            inputs[1].n_elems != (uint32_t)frames ||
            inputs[4].n_elems != GEMMA4_STREAM_STATE_TOKENS ||
            outputs[0].dtype != RKNN3_TENSOR_FLOAT16 ||
            outputs[0].n_elems != (uint32_t)(tokens * GEMMA4_STREAM_EMBEDDING_DIM) ||
            outputs[1].n_elems != (uint32_t)tokens) {
            return -1;
        }
    }
    return 0;
}

static int select_bucket(rknn_gemma4_audio_stream_context* stream, int frames)
{
    int best = -1;
    uint32_t i;
    for (i = 0; i < stream->n_buckets; ++i) {
        if (stream->bucket_frames[i] >= frames &&
            (best < 0 || stream->bucket_frames[i] < stream->bucket_frames[best])) {
            best = (int)i;
        }
    }
    return best;
}

static int activate_bucket(rknn_gemma4_audio_stream_context* stream, int bucket)
{
    uint32_t i;
    int ret;
    if (bucket < 0 || bucket >= (int)stream->n_buckets) {
        return -1;
    }
    if (stream->current_bucket != bucket) {
        ret = rknn3_set_shape(stream->ctx, stream->shape_infos[bucket].shape_id);
        if (ret != RKNN3_SUCCESS) {
            return ret;
        }
        stream->current_bucket = bucket;
    }
    for (i = 0; i < stream->io_num.n_input; ++i) {
        stream->inputs[i].attr = &get_bucket_input_attrs(stream, (uint32_t)bucket)[i];
    }
    for (i = 0; i < stream->io_num.n_output; ++i) {
        stream->outputs[i].attr = &get_bucket_output_attrs(stream, (uint32_t)bucket)[i];
    }
    return 0;
}

static int run_chunk(rknn_gemma4_audio_stream_context* stream,
                     const float* features, const float* mask, int frames,
                     gemma4_audio_stream_callback callback, void* userdata)
{
    static const int output_for_input[8] = {-1, -1, 2, 3, 1, 4, 5, 6};
    rknn3_tensor_attr* input_attrs;
    rknn3_tensor_attr* output_attrs;
    float16* valid_embeddings = NULL;
    const float16* embeddings;
    int bucket;
    int chunk_tokens;
    int valid_tokens = 0;
    int ret;
    int i;

    bucket = select_bucket(stream, frames);
    if (bucket < 0 || stream->bucket_frames[bucket] != frames) {
        fprintf(stderr, "stream: no exact bucket for %d frames\n", frames);
        return -1;
    }
    ret = activate_bucket(stream, bucket);
    if (ret != RKNN3_SUCCESS) {
        fprintf(stderr, "stream: set shape %u failed, ret=%d\n",
                stream->shape_infos[bucket].shape_id, ret);
        return ret;
    }

    input_attrs = get_bucket_input_attrs(stream, (uint32_t)bucket);
    output_attrs = get_bucket_output_attrs(stream, (uint32_t)bucket);
    chunk_tokens = (int)output_attrs[1].n_elems;
    if (write_float_tensor(&input_attrs[0], stream->inputs[0].mem->virt_addr,
                           features, (size_t)frames * GEMMA4_STREAM_MEL_BINS) != 0 ||
        write_float_tensor(&input_attrs[1], stream->inputs[1].mem->virt_addr,
                           mask, (size_t)frames) != 0) {
        return -1;
    }

    for (i = 0; i < (int)stream->io_num.n_input; ++i) {
        ret = rknn3_mem_sync(stream->ctx, stream->inputs[i].mem,
                             RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            return ret;
        }
    }
    ret = rknn3_run(stream->ctx, stream->inputs, stream->io_num.n_input,
                    stream->outputs, stream->io_num.n_output);
    if (ret != RKNN3_SUCCESS) {
        return ret;
    }
    for (i = 0; i < (int)stream->io_num.n_output; ++i) {
        ret = rknn3_mem_sync(stream->ctx, stream->outputs[i].mem,
                             RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            return ret;
        }
    }

    valid_embeddings = (float16*)malloc(
        (size_t)chunk_tokens * GEMMA4_STREAM_EMBEDDING_DIM * sizeof(float16));
    if (valid_embeddings == NULL) {
        return -1;
    }
    embeddings = (const float16*)stream->outputs[0].mem->virt_addr;
    for (i = 0; i < chunk_tokens; ++i) {
        if (read_float_tensor(&output_attrs[1], stream->outputs[1].mem->virt_addr,
                              (size_t)i) > 0.5f) {
            memcpy(valid_embeddings + (size_t)valid_tokens * GEMMA4_STREAM_EMBEDDING_DIM,
                   embeddings + (size_t)i * GEMMA4_STREAM_EMBEDDING_DIM,
                   GEMMA4_STREAM_EMBEDDING_DIM * sizeof(float16));
            ++valid_tokens;
        }
    }
    if (callback != NULL && valid_tokens > 0 &&
        callback(valid_embeddings, valid_tokens, userdata) != 0) {
        free(valid_embeddings);
        return -1;
    }
    free(valid_embeddings);

    for (i = 2; i < 8; ++i) {
        int feedback_ret;
        if (i == 4) {
            feedback_ret = feedback_tail_tensor(&stream->outputs[1], &stream->inputs[4]);
        } else {
            feedback_ret = feedback_tensor(&stream->outputs[output_for_input[i]],
                                           &stream->inputs[i]);
        }
        if (feedback_ret != 0) {
            fprintf(stderr, "stream: state feedback failed: output[%d] -> input[%d]\n",
                    output_for_input[i], i);
            return -1;
        }
    }
    printf("AUDIO_STREAM_BUCKET index=%llu shape_id=%u bucket_frames=%d "
           "valid_tokens=%d\n",
           (unsigned long long)stream->chunk_index,
           stream->shape_infos[bucket].shape_id,
           stream->bucket_frames[bucket], valid_tokens);
    stream->chunk_index++;
    return 0;
}

static int emit_mel_frame(rknn_gemma4_audio_stream_context* stream,
                          const float* samples, int valid)
{
    float fft_input[GEMMA4_STREAM_FFT_LENGTH];
    std::complex<float> fft_output[GEMMA4_STREAM_FFT_LENGTH / 2 + 1];
    pocketfft::shape_t fft_shape = {(size_t)GEMMA4_STREAM_FFT_LENGTH};
    pocketfft::stride_t fft_stride_in = {sizeof(float)};
    pocketfft::stride_t fft_stride_out = {sizeof(std::complex<float>)};
    size_t frame_index = stream->mel_frames;
    size_t offset;
    int i;
    int m;

    if (reserve_mel_buffer(stream, frame_index + 1) != 0) {
        return -1;
    }
    offset = frame_index * GEMMA4_STREAM_MEL_BINS;
    memset(stream->mel + offset, 0, GEMMA4_STREAM_MEL_BINS * sizeof(float));
    stream->mel_mask[frame_index] = valid ? 1.0f : 0.0f;
    stream->mel_frames++;
    if (!valid) {
        return 0;
    }

    for (i = 0; i < GEMMA4_STREAM_FRAME_LENGTH; ++i) {
        fft_input[i] = samples[i] * stream->window[i];
    }
    for (i = GEMMA4_STREAM_FRAME_LENGTH; i < GEMMA4_STREAM_FFT_LENGTH; ++i) {
        fft_input[i] = 0.0f;
    }
    pocketfft::r2c(fft_shape, fft_stride_in, fft_stride_out, 0, true,
                   fft_input, fft_output, 1.0f);
    for (m = 0; m < GEMMA4_STREAM_MEL_BINS; ++m) {
        float mel = 0.0f;
        int k;
        for (k = 0; k < GEMMA4_STREAM_FFT_LENGTH / 2 + 1; ++k) {
            float real = fft_output[k].real();
            float imag = fft_output[k].imag();
            mel += sqrtf(real * real + imag * imag) *
                   stream->mel_filters[(size_t)k * GEMMA4_STREAM_MEL_BINS + m];
        }
        stream->mel[offset + m] = logf(mel + 1e-3f);
    }
    return 0;
}

static int get_largest_bucket_frames(rknn_gemma4_audio_stream_context* stream)
{
    int largest = 0;
    uint32_t i;
    for (i = 0; i < stream->n_buckets; ++i) {
        if (stream->bucket_frames[i] > largest) {
            largest = stream->bucket_frames[i];
        }
    }
    return largest;
}

static int drain_chunks(rknn_gemma4_audio_stream_context* stream,
                        gemma4_audio_stream_callback callback, void* userdata)
{
    int frames = get_largest_bucket_frames(stream);
    while (stream->mel_frames >= (size_t)frames) {
        size_t remaining_frames;
        int ret = run_chunk(stream, stream->mel, stream->mel_mask, frames,
                            callback, userdata);
        if (ret != 0) {
            return ret;
        }
        remaining_frames = stream->mel_frames - (size_t)frames;
        if (remaining_frames > 0) {
            memmove(stream->mel,
                    stream->mel + (size_t)frames * GEMMA4_STREAM_MEL_BINS,
                    remaining_frames * GEMMA4_STREAM_MEL_BINS * sizeof(float));
            memmove(stream->mel_mask, stream->mel_mask + frames,
                    remaining_frames * sizeof(float));
        }
        stream->mel_frames = remaining_frames;
    }
    return 0;
}

static int query_dynamic_buckets(rknn_gemma4_audio_stream_context* stream)
{
    rknn3_shape_config shape_config;
    uint32_t bucket;
    int ret;

    memset(&shape_config, 0, sizeof(shape_config));
    ret = rknn3_query(stream->ctx, RKNN3_QUERY_DYNAMIC_SHAPE_CONFIG,
                      &shape_config, sizeof(shape_config));
    if (ret != RKNN3_SUCCESS || shape_config.n_shapes == 0) {
        fprintf(stderr, "stream: dynamic shape config is required, ret=%d shapes=%u\n",
                ret, shape_config.n_shapes);
        return (ret == RKNN3_SUCCESS) ? -1 : ret;
    }

    stream->n_buckets = shape_config.n_shapes;
    stream->shape_infos = (rknn3_shape_info*)calloc(
        stream->n_buckets, sizeof(rknn3_shape_info));
    stream->bucket_input_attrs = (rknn3_tensor_attr*)calloc(
        (size_t)stream->n_buckets * stream->io_num.n_input,
        sizeof(rknn3_tensor_attr));
    stream->bucket_output_attrs = (rknn3_tensor_attr*)calloc(
        (size_t)stream->n_buckets * stream->io_num.n_output,
        sizeof(rknn3_tensor_attr));
    stream->bucket_frames = (int*)calloc(stream->n_buckets, sizeof(int));
    if (stream->shape_infos == NULL || stream->bucket_input_attrs == NULL ||
        stream->bucket_output_attrs == NULL || stream->bucket_frames == NULL) {
        return -1;
    }

    for (bucket = 0; bucket < stream->n_buckets; ++bucket) {
        stream->shape_infos[bucket].shape_id = bucket;
        stream->shape_infos[bucket].input_attrs = get_bucket_input_attrs(stream, bucket);
        stream->shape_infos[bucket].output_attrs = get_bucket_output_attrs(stream, bucket);
    }
    ret = rknn3_query(stream->ctx, RKNN3_QUERY_DYNAMIC_SHAPE_INFO,
                      stream->shape_infos,
                      sizeof(rknn3_shape_info) * stream->n_buckets);
    if (ret != RKNN3_SUCCESS) {
        return ret;
    }
    for (bucket = 0; bucket < stream->n_buckets; ++bucket) {
        rknn3_tensor_attr* inputs = get_bucket_input_attrs(stream, bucket);
        if (stream->shape_infos[bucket].n_inputs != stream->io_num.n_input ||
            stream->shape_infos[bucket].n_outputs != stream->io_num.n_output ||
            inputs[0].n_dims != 3) {
            return -1;
        }
        stream->bucket_frames[bucket] = (int)inputs[0].shape[1];
    }
    printf("audio stream dynamic buckets:");
    for (bucket = 0; bucket < stream->n_buckets; ++bucket) {
        printf(" %d", stream->bucket_frames[bucket]);
    }
    printf(" mel frames\n");
    return 0;
}

static int allocate_tensor_memory(rknn_gemma4_audio_stream_context* stream)
{
    uint32_t i;
    uint32_t bucket;

    stream->inputs = (rknn3_tensor*)calloc(stream->io_num.n_input,
                                           sizeof(rknn3_tensor));
    stream->outputs = (rknn3_tensor*)calloc(stream->io_num.n_output,
                                            sizeof(rknn3_tensor));
    if (stream->inputs == NULL || stream->outputs == NULL) {
        return -1;
    }

    for (i = 0; i < stream->io_num.n_input; ++i) {
        uint64_t max_size = 0;
        int32_t core_id = get_bucket_input_attrs(stream, 0)[i].core_id;
        for (bucket = 0; bucket < stream->n_buckets; ++bucket) {
            rknn3_tensor_attr* attr = &get_bucket_input_attrs(stream, bucket)[i];
            if (attr->aligned_size > max_size) {
                max_size = attr->aligned_size;
                core_id = attr->core_id;
            }
        }
        stream->inputs[i].mem = rknn3_create_mem(
            stream->ctx, max_size, core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (stream->inputs[i].mem == NULL) {
            return -1;
        }
        memset(stream->inputs[i].mem->virt_addr, 0, stream->inputs[i].mem->size);
    }

    for (i = 0; i < stream->io_num.n_output; ++i) {
        uint64_t max_size = 0;
        int32_t core_id = get_bucket_output_attrs(stream, 0)[i].core_id;
        for (bucket = 0; bucket < stream->n_buckets; ++bucket) {
            rknn3_tensor_attr* attr = &get_bucket_output_attrs(stream, bucket)[i];
            if (attr->aligned_size > max_size) {
                max_size = attr->aligned_size;
                core_id = attr->core_id;
            }
        }
        stream->outputs[i].mem = rknn3_create_mem(
            stream->ctx, max_size, core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (stream->outputs[i].mem == NULL) {
            return -1;
        }
    }
    return 0;
}

int init_gemma4_audio_stream(rknn_gemma4_audio_stream_context** output_stream,
                             const char* model_path, const char* weight_path,
                             uint32_t core_mask)
{
    rknn_gemma4_audio_stream_context* stream;
    rknn3_devices devices;
    rknn3_init_extend init_extend;
    rknn3_config config;
    int ret = -1;

    if (output_stream == NULL || model_path == NULL || weight_path == NULL) {
        return -1;
    }
    *output_stream = NULL;
    stream = (rknn_gemma4_audio_stream_context*)calloc(
        1, sizeof(rknn_gemma4_audio_stream_context));
    if (stream == NULL) {
        return -1;
    }
    stream->current_bucket = -1;

    memset(&devices, 0, sizeof(devices));
    ret = rknn3_find_devices(&devices);
    if (ret != RKNN3_SUCCESS || devices.n_devices == 0) {
        goto fail;
    }
    memset(&init_extend, 0, sizeof(init_extend));
    init_extend.device_id = devices.devices[0].id;
    ret = rknn3_init(&stream->ctx, &init_extend);
    if (ret != RKNN3_SUCCESS) {
        goto fail;
    }
    ret = rknn3_load_model_from_path(stream->ctx, model_path, weight_path);
    if (ret != RKNN3_SUCCESS) {
        goto fail;
    }
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    ret = rknn3_model_init(stream->ctx, &config);
    if (ret != RKNN3_SUCCESS) {
        goto fail;
    }
    ret = rknn3_query(stream->ctx, RKNN3_QUERY_IN_OUT_NUM,
                      &stream->io_num, sizeof(stream->io_num));
    if (ret != RKNN3_SUCCESS) {
        goto fail;
    }
    ret = query_dynamic_buckets(stream);
    if (ret != 0) {
        goto fail;
    }
    ret = validate_stream_model(stream);
    if (ret != 0) {
        fprintf(stderr, "stream: model does not match the dynamic external-state interface\n");
        goto fail;
    }
    ret = allocate_tensor_memory(stream);
    if (ret != 0) {
        goto fail;
    }

    build_mel_filters(stream->mel_filters);
    init_hann_window(stream);
    if (reserve_float_buffer(&stream->pcm, &stream->pcm_capacity,
                             GEMMA4_STREAM_FRAME_LENGTH) != 0) {
        ret = -1;
        goto fail;
    }
    memset(stream->pcm, 0, (GEMMA4_STREAM_FRAME_LENGTH / 2) * sizeof(float));
    stream->pcm_size = GEMMA4_STREAM_FRAME_LENGTH / 2;
    *output_stream = stream;
    return 0;

fail:
    fprintf(stderr, "stream: init failed, ret=%d\n", ret);
    release_gemma4_audio_stream(stream);
    return (ret == RKNN3_SUCCESS) ? -1 : ret;
}

int reset_gemma4_audio_stream(rknn_gemma4_audio_stream_context* stream)
{
    uint32_t i;
    if (stream == NULL) {
        return -1;
    }
    memset(stream->pcm, 0, (GEMMA4_STREAM_FRAME_LENGTH / 2) * sizeof(float));
    stream->pcm_size = GEMMA4_STREAM_FRAME_LENGTH / 2;
    stream->mel_frames = 0;
    stream->current_bucket = -1;
    stream->chunk_index = 0;
    for (i = 2; i < stream->io_num.n_input; ++i) {
        memset(stream->inputs[i].mem->virt_addr, 0, stream->inputs[i].mem->size);
    }
    return 0;
}

int get_n_audio_stream(int n_samples)
{
    int64_t frame_numerator;
    int64_t valid_frames;
    int64_t valid_tokens;
    if (n_samples < 0) {
        return -1;
    }
    frame_numerator = (int64_t)n_samples +
        GEMMA4_STREAM_FRAME_LENGTH / 2 - GEMMA4_STREAM_FRAME_LENGTH;
    if (frame_numerator <= 0) {
        return 0;
    }
    valid_frames = (frame_numerator + GEMMA4_STREAM_HOP_LENGTH - 1) /
                   GEMMA4_STREAM_HOP_LENGTH;
    valid_tokens = (valid_frames + 3) / 4;
    return (valid_tokens <= INT_MAX) ? (int)valid_tokens : -1;
}

int get_gemma4_audio_stream_embedding_dim(void)
{
    return GEMMA4_STREAM_EMBEDDING_DIM;
}

int push_gemma4_audio_stream(rknn_gemma4_audio_stream_context* stream,
                             const float* samples, int n_samples,
                             gemma4_audio_stream_callback callback,
                             void* userdata)
{
    if (stream == NULL || n_samples < 0 ||
        (samples == NULL && n_samples != 0)) {
        return -1;
    }
    if (n_samples > 0) {
        size_t required = stream->pcm_size + (size_t)n_samples;
        if (reserve_float_buffer(&stream->pcm, &stream->pcm_capacity, required) != 0) {
            return -1;
        }
        memcpy(stream->pcm + stream->pcm_size, samples,
               (size_t)n_samples * sizeof(float));
        stream->pcm_size = required;
    }

    while (stream->pcm_size >= GEMMA4_STREAM_FRAME_LENGTH) {
        if (emit_mel_frame(stream, stream->pcm, 1) != 0) {
            return -1;
        }
        memmove(stream->pcm, stream->pcm + GEMMA4_STREAM_HOP_LENGTH,
                (stream->pcm_size - GEMMA4_STREAM_HOP_LENGTH) * sizeof(float));
        stream->pcm_size -= GEMMA4_STREAM_HOP_LENGTH;
        if (drain_chunks(stream, callback, userdata) != 0) {
            return -1;
        }
    }
    return 0;
}

int flush_gemma4_audio_stream(rknn_gemma4_audio_stream_context* stream,
                              gemma4_audio_stream_callback callback,
                              void* userdata)
{
    int bucket;
    int padded_frames;
    int ret;
    if (stream == NULL) {
        return -1;
    }
    stream->pcm_size = 0;
    if (stream->mel_frames == 0) {
        return 0;
    }
    bucket = select_bucket(stream, (int)stream->mel_frames);
    if (bucket < 0) {
        fprintf(stderr, "stream: tail of %zu frames exceeds largest bucket\n",
                stream->mel_frames);
        return -1;
    }
    padded_frames = stream->bucket_frames[bucket];
    if (reserve_mel_buffer(stream, (size_t)padded_frames) != 0) {
        return -1;
    }
    memset(stream->mel + stream->mel_frames * GEMMA4_STREAM_MEL_BINS, 0,
           ((size_t)padded_frames - stream->mel_frames) *
           GEMMA4_STREAM_MEL_BINS * sizeof(float));
    memset(stream->mel_mask + stream->mel_frames, 0,
           ((size_t)padded_frames - stream->mel_frames) * sizeof(float));
    ret = run_chunk(stream, stream->mel, stream->mel_mask, padded_frames,
                    callback, userdata);
    if (ret == 0) {
        stream->mel_frames = 0;
    }
    return ret;
}

int release_gemma4_audio_stream(rknn_gemma4_audio_stream_context* stream)
{
    uint32_t i;
    if (stream == NULL) {
        return 0;
    }
    if (stream->inputs != NULL) {
        for (i = 0; i < stream->io_num.n_input; ++i) {
            if (stream->inputs[i].mem != NULL && stream->ctx != 0) {
                rknn3_destroy_mem(stream->ctx, stream->inputs[i].mem);
            }
        }
    }
    if (stream->outputs != NULL) {
        for (i = 0; i < stream->io_num.n_output; ++i) {
            if (stream->outputs[i].mem != NULL && stream->ctx != 0) {
                rknn3_destroy_mem(stream->ctx, stream->outputs[i].mem);
            }
        }
    }
    if (stream->ctx != 0) {
        rknn3_destroy(stream->ctx);
    }
    free(stream->inputs);
    free(stream->outputs);
    free(stream->shape_infos);
    free(stream->bucket_input_attrs);
    free(stream->bucket_output_attrs);
    free(stream->bucket_frames);
    free(stream->pcm);
    free(stream->mel);
    free(stream->mel_mask);
    free(stream);
    return 0;
}
