#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <complex>
#include <vector>

#include "pocketfft_hdronly.h"

#define MA_NO_DEVICE_IO
#define MA_NO_THREADING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "audio_utils.h"

#ifndef ENABLE_NEON
#if defined(__ARM_NEON) || defined(__aarch64__)
#define ENABLE_NEON     1
#else
#define ENABLE_NEON     0
#endif
#endif

#if ENABLE_NEON
#include "arm_neon.h"
#endif

int read_audio(const char *path, audio_buffer_t *audio)
{
    const int channels = 1;
    const int sampler_rate = 16000;
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, channels, sampler_rate);

    ma_decoder decoder;
    ma_result result = ma_decoder_init_file(path, &decoder_config, &decoder);
    if (result != MA_SUCCESS) {
        fprintf(stderr, "Error: failed to open audio file (%s)\n", ma_result_description(result));
        return -1;
    }

    ma_uint64 frame_count;
    result = ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count);
    if (result != MA_SUCCESS) {
        ma_decoder_uninit(&decoder);
        fprintf(stderr, "Error: failed to retrieve the length of the audio data (%s)\n", ma_result_description(result));
        return -1;
    }

    audio->data = (float*)malloc(frame_count * channels * sizeof(float));
    if (audio->data == NULL) {
        ma_decoder_uninit(&decoder);
        fprintf(stderr, "Error: failed to allocate memory for audio data\n");
        return -1;
    }

    ma_uint64 frames_read;
    result = ma_decoder_read_pcm_frames(&decoder, audio->data, frame_count, &frames_read);
    if (result != MA_SUCCESS) {
        free(audio->data);
        audio->data = NULL;
        ma_decoder_uninit(&decoder);
        fprintf(stderr, "Error: failed to read the frames of the audio data (%s)\n", ma_result_description(result));
        return -1;
    }

    audio->num_frames = (int)frames_read;
    audio->num_channels = channels;
    audio->sample_rate = sampler_rate;

    ma_decoder_uninit(&decoder);

    return 0;
}

int read_mel_filters(const char *fileName, float *data, int max_lines)
{
    FILE *file;
    int line_count = 0;

    file = fopen(fileName, "r");
    if (file == NULL)
    {
        perror("Error opening file");
        return -1;
    }

    while (line_count < max_lines && fscanf(file, "%f", &data[line_count]) == 1)
    {
        line_count++;
    }

    fclose(file);

    return 0;
}

static void pad_x_mel(const std::vector<float> input, int rows_input, int cols_input, std::vector<float> &output, int cols_output)
{
    for (int i = 0; i < rows_input; ++i)
    {
        std::copy(input.begin() + i * cols_input, input.begin() + (i + 1) * cols_input, output.begin() + i * cols_output);
    }
}

static void hann_window(std::vector<float> &window, int length)
{
    for (int i = 0; i < length; i++)
    {
        window[i] = 0.5 * (1 - cos(2 * M_PI * i / (length - 1)));
    }
}

static void reflect_pad(const std::vector<float> &audio, std::vector<float> &padded_audio, int pad_width)
{
    std::copy(audio.begin(), audio.end(), padded_audio.begin() + pad_width);
    std::reverse_copy(audio.begin(), audio.begin() + pad_width, padded_audio.begin());
    std::reverse_copy(audio.end() - pad_width, audio.end(), padded_audio.end() - pad_width);
}

#if ENABLE_NEON
static void stfts_neon(const std::vector<float> &audio, int audio_length, int window_length, int hop_length, const std::vector<float> &window, std::complex<float> *stft_result, int num_frames)
{
    std::vector<float> input(window_length, 0.0f);
    std::vector<std::complex<float>> output(window_length / 2 + 1);
    pocketfft::shape_t shape = {static_cast<size_t>(window_length)};
    pocketfft::stride_t stride_in = {sizeof(float)};
    pocketfft::stride_t stride_out = {sizeof(std::complex<float>)};
    for (int i = 0; i < num_frames; i++)
    {
        int start = i * hop_length;
        int end = start + window_length;
        for (int j = 0; j < window_length - 3; j += 4)
        {
            if (start + j < audio_length)
            {
                float32x4_t in = vld1q_f32(audio.data() + start + j);
                float32x4_t win = vld1q_f32(window.data() + j);
                float32x4_t out = vmulq_f32(in, win);
                vst1q_f32(input.data() + j, out);
            }
            else
            {
                vst1q_f32(input.data() + j, vdupq_n_f32(0.0f));
            }
        }

        for (int j = window_length - window_length % 4; j < window_length; j++)
        {
            if (start + j < audio_length)
            {
                input[j] = audio[start + j] * window[j];
            }
            else
            {
                input[j] = 0.0f;
            }
        }

        pocketfft::r2c(shape, stride_in, stride_out, 0, true, input.data(), output.data(), 1.0f);
        memcpy(stft_result + i * (window_length / 2 + 1), output.data(), sizeof(std::complex<float>) * (window_length / 2 + 1));
    }
}
#else
static void stfts(const std::vector<float> &audio, int audio_length, int window_length, int hop_length, const std::vector<float> &window, std::complex<float> *stft_result, int num_frames)
{
    std::vector<float> input(window_length, 0.0f);
    std::vector<std::complex<float>> output(window_length / 2 + 1);
    pocketfft::shape_t shape = {static_cast<size_t>(window_length)};
    pocketfft::stride_t stride_in = {sizeof(float)};
    pocketfft::stride_t stride_out = {sizeof(std::complex<float>)};

    for (int i = 0; i < num_frames; i++)
    {
        int start = i * hop_length;
        int end = start + window_length;

        for (int j = 0; j < window_length; j++)
        {
            if (start + j < audio_length)
            {
                input[j] = audio[start + j] * window[j];
            }
            else
            {
                input[j] = 0.0f;
            }
        }

        pocketfft::r2c(shape, stride_in, stride_out, 0, true, input.data(), output.data(), 1.0f);
        memcpy(stft_result + i * (window_length / 2 + 1), output.data(), sizeof(std::complex<float>) * (window_length / 2 + 1));
    }
}
#endif

static float compute_magnitude(const std::complex<float> &value)
{
    return value.real() * value.real() + value.imag() * value.imag();
}

static void compute_magnitudes(std::complex<float> *stft_result, int num_mel_filters, int num_frames, std::vector<float> &magnitudes)
{
    int k = 0;
    for (int i = 0; i < num_mel_filters; i++)
    {
        for (int j = 0; j < num_frames - 1; j++)
        {
            magnitudes[k] = compute_magnitude(stft_result[i * num_frames + j]);
            k++;
        }
    }
}

static void clamp_and_log_max(std::vector<float> &mel_spec, int rows, int cols)
{
    float min_val = 1e-10;
    float scaling_factor = 1.0 / 4.0;
    float shift_value = 4.0;

    float max_val = mel_spec[0];
    for (int i = 0; i < rows * cols; ++i)
    {
        float value = mel_spec[i];
        value = (value < min_val) ? min_val : value;
        mel_spec[i] = log10f(value);

        if (mel_spec[i] > max_val)
            max_val = mel_spec[i];
    }

    float threshold = max_val - 8.0;
    for (int i = 0; i < rows * cols; ++i)
    {
        mel_spec[i] = (std::max(mel_spec[i], threshold) + shift_value) * scaling_factor;
    }
}

void transpose(std::complex<float> *input, int input_rows, int input_cols, std::complex<float> *output)
{
    for (int i = 0; i < input_rows; ++i)
    {
        for (int j = 0; j < input_cols; ++j)
        {
            int input_index = i * input_cols + j;
            int output_index = j * input_rows + i;

            output[output_index] = input[input_index];
        }
    }
}

#if ENABLE_NEON
void matmul_by_neon(float *A, float *B, std::vector<float> &C, int ROWS_A, int COLS_A, int COLS_B)
{
    std::fill(C.begin(), C.end(), 0.0f);

    for (int i = 0; i < ROWS_A; i++)
    {
        int k_start = COLS_A;
        int k_end = -1;
        for (int k = 0; k < COLS_A; k++)
        {
            if (A[i * COLS_A + k] != 0.0f)
            {
                k_start = k;
                break;
            }
        }
        if (k_start == COLS_A)
        {
            continue;
        }
        for (int k = COLS_A - 1; k >= k_start; k--)
        {
            if (A[i * COLS_A + k] != 0.0f)
            {
                k_end = k;
                break;
            }
        }

        float *c_row = &C[i * COLS_B];
        int j = 0;
        for (; j + 4 <= COLS_B; j += 4)
        {
            float32x4_t v_acc = vdupq_n_f32(0.0f);
            for (int k = k_start; k <= k_end; k++)
            {
                float a_val = A[i * COLS_A + k];
                float32x4_t v_b = vld1q_f32(&B[k * COLS_B + j]);
                v_acc = vmlaq_n_f32(v_acc, v_b, a_val);
            }
            vst1q_f32(&c_row[j], v_acc);
        }
        for (; j < COLS_B; j++)
        {
            float sum = 0.0f;
            for (int k = k_start; k <= k_end; k++)
            {
                sum += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
            c_row[j] = sum;
        }
    }
}
#else
void matmul_by_cpu(float *A, float *B, std::vector<float> &C, int ROWS_A, int COLS_A, int COLS_B) 
{
    std::fill(C.begin(), C.end(), 0.0f);

    for (int i = 0; i < ROWS_A; i++) {
        float* c_row = &C[i * COLS_B];
        for (int k = 0; k < COLS_A; k++) {
            float a_val = A[i * COLS_A + k];
            float* b_row = &B[k * COLS_B];
            for (int j = 0; j < COLS_B; j++) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}
#endif

static void log_mel_spectrogram(float *audio_data, int audio_length, int cur_num_frames_of_stfts, float *filters, int n_fft, int hop_length, int n_mels, std::vector<float> &mel_spec)
{
    int mels_filters_size = n_fft / 2 + 1;

    std::vector<float> window(n_fft);
    hann_window(window, n_fft);

    std::vector<float> audio(audio_data, audio_data + audio_length);
    int padded_size = audio_length + n_fft;
    std::vector<float> padded_audio(padded_size);
    reflect_pad(audio, padded_audio, n_fft / 2);

    std::vector<std::complex<float>> stfts_result(mels_filters_size * cur_num_frames_of_stfts);
#if ENABLE_NEON
    stfts_neon(padded_audio, audio_length + n_fft, n_fft, hop_length, window, stfts_result.data(), cur_num_frames_of_stfts);
#else
    stfts(padded_audio, audio_length + n_fft, n_fft, hop_length, window, stfts_result.data(), cur_num_frames_of_stfts);
#endif

    std::vector<std::complex<float>> stfts_result_t(mels_filters_size * cur_num_frames_of_stfts);
    transpose(stfts_result.data(), cur_num_frames_of_stfts, mels_filters_size, stfts_result_t.data());

    std::vector<float> magnitudes(mels_filters_size * (cur_num_frames_of_stfts - 1));
    compute_magnitudes(stfts_result_t.data(), mels_filters_size, cur_num_frames_of_stfts, magnitudes);

    int ROWS_A = n_mels;
    int COLS_A = mels_filters_size;
    int COLS_B = cur_num_frames_of_stfts - 1;
#if ENABLE_NEON
    matmul_by_neon(filters, magnitudes.data(), mel_spec, ROWS_A, COLS_A, COLS_B);
#else
    matmul_by_cpu(filters, magnitudes.data(), mel_spec, ROWS_A, COLS_A, COLS_B);
#endif

    clamp_and_log_max(mel_spec, ROWS_A, COLS_B);

}

void audio_preprocess(audio_buffer_t *audio, float *mel_filters, int n_fft, int hop_length, int n_mels, int max_audio_length, std::vector<float> &x_mel, int *actual_len)
{
    int ret;
    int audio_length = audio->num_frames;
    std::vector<float> ori_audio_data(audio->data, audio->data + audio_length);

    if (audio_length >= max_audio_length)
    {
        std::vector<float> trim_audio_data(max_audio_length);
        std::copy(ori_audio_data.begin(), ori_audio_data.begin() + max_audio_length, trim_audio_data.begin());
        int cur_num_frames_of_stfts = max_audio_length / hop_length + 1;
        log_mel_spectrogram(trim_audio_data.data(), max_audio_length, cur_num_frames_of_stfts, mel_filters, n_fft, hop_length, n_mels, x_mel);
        *actual_len = max_audio_length / hop_length;
    }
    else
    {
        int cur_num_frames_of_stfts = audio_length / hop_length + 1;
        int x_mel_rows = n_mels;
        int x_mel_cols = cur_num_frames_of_stfts - 1;
        int x_mel_cols_pad = max_audio_length / hop_length;
        std::vector<float> cur_x_mel(x_mel_rows * x_mel_cols, 0.0f);
        log_mel_spectrogram(ori_audio_data.data(), audio_length, cur_num_frames_of_stfts, mel_filters, n_fft, hop_length, n_mels, cur_x_mel);
        pad_x_mel(cur_x_mel, x_mel_rows, x_mel_cols, x_mel, x_mel_cols_pad);
        *actual_len = x_mel_cols;
    }
}

int resample_audio(audio_buffer_t *audio, int original_sample_rate, int desired_sample_rate)
{
    int original_length = audio->num_frames;
    int out_length = round(original_length * (double)desired_sample_rate / (double)original_sample_rate);
    printf("resample_audio: %d HZ -> %d HZ \n", original_sample_rate, desired_sample_rate);

    float *resampled_data = (float *)malloc(out_length * sizeof(float));
    if (!resampled_data)
    {
        return -1;
    }

    for (int i = 0; i < out_length; ++i)
    {
        double src_index = i * (double)original_sample_rate / (double)desired_sample_rate;
        int left_index = (int)floor(src_index);
        int right_index = (left_index + 1 < original_length) ? left_index + 1 : left_index;
        double fraction = src_index - left_index;
        resampled_data[i] = (1.0f - fraction) * audio->data[left_index] + fraction * audio->data[right_index];
    }

    audio->num_frames = out_length;
    free(audio->data);
    audio->data = resampled_data;

    return 0;
}

int convert_channels(audio_buffer_t *audio)
{

    int original_num_channels = audio->num_channels;
    printf("convert_channels: %d -> %d \n", original_num_channels, 1);

    float *converted_data = (float *)malloc(audio->num_frames * sizeof(float));
    if (!converted_data)
    {
        return -1;
    }

    for (int i = 0; i < audio->num_frames; ++i)
    {
        float left = audio->data[i * 2];
        float right = audio->data[i * 2 + 1];
        converted_data[i] = (left + right) / 2.0f;
    }

    audio->num_channels = 1;
    free(audio->data);
    audio->data = converted_data;

    return 0;
}
