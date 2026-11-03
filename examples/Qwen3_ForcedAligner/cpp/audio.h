#ifndef QWEN3_FORCED_ALIGNER_AUDIO_H
#define QWEN3_FORCED_ALIGNER_AUDIO_H

#include <stdint.h>

#include "common.h"

int read_audio(const char *path, audio_buffer_t *audio);
int read_mel_filters(const char *path, float *filters, int count);
int compute_mel(audio_buffer_t *audio, float *filters, float *mel, int frames);

#endif
