#ifndef _RKNN_WHISPER_DEMO_PROCESS_H_
#define _RKNN_WHISPER_DEMO_PROCESS_H_

// #define TIMING_DISABLED // if you don't need to print the time used, uncomment this line of code

#include <string>
#include <vector>
#include <cstdint>

#include "audio_utils.h"

#define SAMPLE_RATE 16000
#define N_FFT 400
#define HOP_LENGTH 160
#define MELS_FILTERS_SIZE 201 // (N_FFT / 2 + 1)

namespace whisper_tokens {
static constexpr int32_t EOT = 50257;
static constexpr int32_t SOT = 50258;
static constexpr int32_t TASK_EN = 50259;
static constexpr int32_t TASK_ZH = 50260;
} // namespace whisper_tokens


typedef struct
{
    int index;
    char *token;
} VocabEntry;

void replace_substr(std::string &str, const std::string &from, const std::string &to);
int read_vocab(const char *fileName, VocabEntry *vocab, int vocab_size);
int read_mel_filters(const char *fileName, float *data, int max_lines);
void audio_preprocess(audio_buffer_t *audio, float *mel_filters, std::vector<float> &x_mel,
                      int n_mels, int max_audio_length);
std::string normalize_piece(const std::string &piece);
std::string decode_whisper_tokens(VocabEntry *vocab, const std::vector<int32_t> &token_ids, const char *task, int vocab_size);
std::string base64_decode(const std::string &s);

#endif //_RKNN_WHISPER_DEMO_PROCESS_H_
