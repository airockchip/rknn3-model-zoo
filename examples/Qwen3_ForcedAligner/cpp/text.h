#ifndef QWEN3_FORCED_ALIGNER_TEXT_H
#define QWEN3_FORCED_ALIGNER_TEXT_H

#include <stdint.h>

#include "model_config.h"

struct request {
	int samples;
	int sequence;
	int n_audio;
	int n_units;
	int vocab_size;
	int indices[MODEL_MAX_SEQUENCE];
	int audio_rows[MODEL_MAX_AUDIO_TOKENS];
	int slots[MODEL_MAX_SEQUENCE];
	int32_t tokens[MODEL_MAX_SEQUENCE];
	char text[TEXT_MAX_BYTES];
	char units[MODEL_MAX_UNITS][TEXT_UNIT_BYTES];
};

int prepare_text(const char *model_path, const char *text, const char *language, int samples, struct request *request);

#endif
