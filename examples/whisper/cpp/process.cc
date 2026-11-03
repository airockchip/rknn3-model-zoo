// Copyright (c) 2024 by Rockchip Electronics Co., Ltd. All Rights Reserved.
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

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <complex>
#include <iostream>

#include "pocketfft_hdronly.h"
#include "process.h"

#define ENABLE_NEON 1

#if ENABLE_NEON
#include "arm_neon.h"
#endif

static void hann_window(std::vector<float> &window, int length)
{
    for (int i = 0; i < length; i++)
    {
        // Periodic Hann Window (matches PyTorch/librosa)
        window[i] = 0.5 * (1 - cos(2 * M_PI * i / length));
    }
}

static void reflect_pad(const std::vector<float> &audio, std::vector<float> &padded_audio, int pad_width)
{
    const int n = (int)audio.size();

    std::copy(audio.begin(), audio.end(), padded_audio.begin() + pad_width);

    // Match PyTorch reflect padding: reflect around the edge without repeating it.
    for (int i = 0; i < pad_width; ++i)
    {
        padded_audio[i] = audio[pad_width - i];
    }

    for (int i = 0; i < pad_width; ++i)
    {
        padded_audio[pad_width + n + i] = audio[n - 2 - i];
    }
}

#if ENABLE_NEON
static void stfts_neon(const std::vector<float> &audio, int audio_length, int window_length, int hop_length, const std::vector<float> &window, std::vector<std::complex<float>> &stft_result, int num_frames)
{
    std::vector<float> input(window_length);
    std::vector<std::complex<float>> output(window_length / 2 + 1);

    pocketfft::shape_t shape = {static_cast<size_t>(window_length)};
    pocketfft::stride_t stride_in = {sizeof(float)};
    pocketfft::stride_t stride_out = {sizeof(std::complex<float>)};

    for (int i = 0; i < num_frames; i++)
    {
        int start = i * hop_length;
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
        memcpy(&stft_result[(size_t)i * (window_length / 2 + 1)], output.data(), sizeof(std::complex<float>) * (window_length / 2 + 1));
    }
}
#else
static void stfts(const std::vector<float> &audio, int audio_length, int window_length, int hop_length, const std::vector<float> &window, std::vector<std::complex<float>> &stft_result, int num_frames)
{
    std::vector<float> input(window_length);
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
        memcpy(&stft_result[(size_t)i * (window_length / 2 + 1)], output.data(), sizeof(std::complex<float>) * (window_length / 2 + 1));
    }
}
#endif

static float compute_magnitude(const std::complex<float> &value)
{
    return value.real() * value.real() + value.imag() * value.imag();
}

static void compute_magnitudes(const std::vector<std::complex<float>> &stft_result, int num_mel_filters, int num_frames, std::vector<float> &magnitudes)
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
    float max_val = -INFINITY;
    int total = rows * cols;

    // Compute log10 first, then clamp by global max - 8.0, matching Whisper preprocessing.
    for (int i = 0; i < total; ++i)
    {
        float value = mel_spec[i];
        value = (value < min_val) ? min_val : value;
        mel_spec[i] = log10f(value);

        if (mel_spec[i] > max_val)
            max_val = mel_spec[i];
    }

    float threshold = max_val - 8.0;
    for (int i = 0; i < total; ++i)
    {
        mel_spec[i] = (std::max(mel_spec[i], threshold) + shift_value) * scaling_factor;
    }
}

static void whisper_transpose(const std::vector<std::complex<float>> &input, int input_rows, int input_cols, std::vector<std::complex<float>> &output)
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

static void matmul_scalar(float *A, float *B, std::vector<float> &C, int ROWS_A, int COLS_A, int COLS_B)
{
    for (int i = 0; i < ROWS_A; ++i)
    {
        for (int j = 0; j < COLS_B; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < COLS_A; ++k)
            {
                sum += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
            C[i * COLS_B + j] = sum;
        }
    }
}

#if ENABLE_NEON
static void matmul_by_neon(float *A, float *B, std::vector<float> &C, int ROWS_A, int COLS_A, int COLS_B)
{
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    // B is row-major [COLS_A, COLS_B]; vectorize across output columns j.
    for (int i = 0; i < ROWS_A; ++i)
    {
        int j = 0;
        for (; j + 3 < COLS_B; j += 4)
        {
            float32x4_t acc = vdupq_n_f32(0.0f);

            for (int k = 0; k < COLS_A; ++k)
            {
                float a = A[i * COLS_A + k];
                float32x4_t b_vec = vld1q_f32(&B[k * COLS_B + j]);
                acc = vmlaq_n_f32(acc, b_vec, a);
            }

            vst1q_f32(&C[i * COLS_B + j], acc);
        }

        for (; j < COLS_B; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < COLS_A; ++k)
            {
                sum += A[i * COLS_A + k] * B[k * COLS_B + j];
            }
            C[i * COLS_B + j] = sum;
        }
    }
#else
    matmul_scalar(A, B, C, ROWS_A, COLS_A, COLS_B);
#endif
}
#else
static void matmul_by_opencv(float *A, float *B, std::vector<float> &C, int ROWS_A, int COLS_A, int COLS_B)
{
    cv::Mat mat_A(ROWS_A, COLS_A, CV_32F, A);
    cv::Mat mat_B(COLS_A, COLS_B, CV_32F, B);
    cv::Mat mat_C(ROWS_A, COLS_B, CV_32F);
    cv::gemm(mat_A, mat_B, 1.0, cv::Mat(), 0.0, mat_C);
    memcpy(C.data(), mat_C.data, ROWS_A * COLS_B * sizeof(float));
}
#endif

static void log_mel_spectrogram(float *audio_data, int audio_length, int cur_num_frames_of_stfts, float *filters, std::vector<float> &mel_spec, int n_mels)
{
    std::vector<float> window(N_FFT);
    hann_window(window, N_FFT);

    std::vector<float> audio(audio_data, audio_data + audio_length);
    int padded_size = audio_length + N_FFT;
    std::vector<float> padded_audio(padded_size);
    reflect_pad(audio, padded_audio, N_FFT / 2);

    std::vector<std::complex<float>> stfts_result((size_t)MELS_FILTERS_SIZE * cur_num_frames_of_stfts);
#if ENABLE_NEON
    stfts_neon(padded_audio, audio_length + N_FFT, N_FFT, HOP_LENGTH, window, stfts_result, cur_num_frames_of_stfts);
#else
    stfts(padded_audio, audio_length + N_FFT, N_FFT, HOP_LENGTH, window, stfts_result, cur_num_frames_of_stfts);
#endif

    std::vector<std::complex<float>> stfts_result_t((size_t)MELS_FILTERS_SIZE * cur_num_frames_of_stfts);
    whisper_transpose(stfts_result, cur_num_frames_of_stfts, MELS_FILTERS_SIZE, stfts_result_t);

    std::vector<float> magnitudes(MELS_FILTERS_SIZE * (cur_num_frames_of_stfts - 1));
    compute_magnitudes(stfts_result_t, MELS_FILTERS_SIZE, cur_num_frames_of_stfts, magnitudes);

    int ROWS_A = n_mels;
    int COLS_A = MELS_FILTERS_SIZE;
    int COLS_B = cur_num_frames_of_stfts - 1;
#if ENABLE_NEON
    matmul_by_neon(filters, magnitudes.data(), mel_spec, ROWS_A, COLS_A, COLS_B);
#else
    matmul_by_opencv(filters, magnitudes.data(), mel_spec, ROWS_A, COLS_A, COLS_B);
#endif

    clamp_and_log_max(mel_spec, ROWS_A, COLS_B);
}

void audio_preprocess(audio_buffer_t *audio, float *mel_filters, std::vector<float> &x_mel,
                      int n_mels, int max_audio_length)
{
    int audio_length = audio->num_frames;
    std::vector<float> ori_audio_data(audio->data, audio->data + audio_length);

    if (audio_length >= max_audio_length)
    {
        std::vector<float> trim_audio_data(max_audio_length);
        std::copy(ori_audio_data.begin(), ori_audio_data.begin() + max_audio_length, trim_audio_data.begin());
        int cur_num_frames_of_stfts = max_audio_length / HOP_LENGTH + 1;
        log_mel_spectrogram(trim_audio_data.data(), max_audio_length, cur_num_frames_of_stfts, mel_filters, x_mel, n_mels);
    }
    else
    {
        std::vector<float> padded_audio_data(max_audio_length, 0.0f);
        std::copy(ori_audio_data.begin(), ori_audio_data.end(), padded_audio_data.begin());

        int cur_num_frames_of_stfts = max_audio_length / HOP_LENGTH + 1;
        log_mel_spectrogram(padded_audio_data.data(), max_audio_length, cur_num_frames_of_stfts, mel_filters, x_mel, n_mels);
    }
}

__attribute__((weak)) int read_mel_filters(const char *fileName, float *data, int max_lines)
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

    return line_count == max_lines ? 0 : -1;
}

int read_vocab(const char *fileName, VocabEntry *vocab, int vocab_size)
{
    FILE *fp;
    char line[512];

    fp = fopen(fileName, "r");
    if (fp == NULL)
    {
        perror("Error opening file");
        return -1;
    }

    int count = 0;
    while (fgets(line, sizeof(line), fp))
    {
        char *piece = strchr(line, ' ');
        if (count >= vocab_size || !piece || atoi(line) != count)
        {
            printf("Invalid vocabulary entry at row %d in %s\n", count, fileName);
            fclose(fp);
            return -1;
        }
        vocab[count].token = strdup(piece + 1);
        if (!vocab[count].token)
        {
            fclose(fp);
            return -1;
        }
        vocab[count].index = count;
        count++;
    }

    fclose(fp);

    return count == vocab_size ? 0 : -1;
}

void replace_substr(std::string &str, const std::string &from, const std::string &to)
{
    if (from.empty())
        return; // Prevent infinite loop if 'from' is empty
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos)
    {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Advance position by length of the replacement
    }
}

static int32_t get_char_index(char c)
{
    if (c >= 'A' && c <= 'Z')
    {
        return c - 'A';
    }
    else if (c >= 'a' && c <= 'z')
    {
        return c - 'a' + ('Z' - 'A') + 1;
    }
    else if (c >= '0' && c <= '9')
    {
        return c - '0' + ('Z' - 'A') + ('z' - 'a') + 2;
    }
    else if (c == '+')
    {
        return 62;
    }
    else if (c == '/')
    {
        return 63;
    }

    std::cerr << "Unknown character " << static_cast<int>(c) << ", " << c << std::endl;
    exit(-1);
}

std::string base64_decode(const std::string &encoded_string)
{
    // see
    // https://github.com/ReneNyffenegger/cpp-base64/blob/master/base64.cpp#L243
    // https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/base64-decode.cc
    if (encoded_string.empty())
    {
        std::cerr << "Empty string!" << std::endl;
        exit(-1);
    }

    int32_t output_length = static_cast<int32_t>(encoded_string.size()) / 4 * 3;

    std::string decoded_string;
    decoded_string.reserve(output_length);

    int32_t index = 0;
    while (index < static_cast<int32_t>(encoded_string.size()))
    {
        if (encoded_string[index] == '=')
        {
            return " ";
        }

        int32_t first_byte = (get_char_index(encoded_string[index]) << 2) + ((get_char_index(encoded_string[index + 1]) & 0x30) >> 4);
        decoded_string.push_back(static_cast<char>(first_byte));

        if (index + 2 < static_cast<int32_t>(encoded_string.size()) && encoded_string[index + 2] != '=')
        {
            int32_t second_byte = ((get_char_index(encoded_string[index + 1]) & 0x0f) << 4) + ((get_char_index(encoded_string[index + 2]) & 0x3c) >> 2);
            decoded_string.push_back(static_cast<char>(second_byte));

            if (index + 3 < static_cast<int32_t>(encoded_string.size()) && encoded_string[index + 3] != '=')
            {
                int32_t third_byte = ((get_char_index(encoded_string[index + 2]) & 0x03) << 6) + get_char_index(encoded_string[index + 3]);
                decoded_string.push_back(static_cast<char>(third_byte));
            }
        }
        index += 4;
    }

    return decoded_string;
}

std::string normalize_piece(const std::string &piece)
{
    std::string normalized = piece;
    replace_substr(normalized, "Ġ", " ");
    replace_substr(normalized, "<|endoftext|>", "");
    replace_substr(normalized, "\n", "");
    return normalized;
}

std::string decode_whisper_tokens(VocabEntry *vocab, const std::vector<int32_t> &token_ids, const char *task, int vocab_size)
{
    std::string text;
    for (size_t i = 0; i < token_ids.size(); ++i)
    {
        int32_t token_id = token_ids[i];
        if (token_id < 0 || token_id >= vocab_size || token_id >= whisper_tokens::EOT)
        {
            continue;
        }
        if (!vocab || !vocab[token_id].token)
        {
            continue;
        }
        std::string piece = normalize_piece(vocab[token_id].token);
        text += task && strcmp(task, "zh") == 0 ? base64_decode(piece) : piece;
    }

    return text;
}
