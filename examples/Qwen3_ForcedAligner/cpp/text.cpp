#include <exception>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Tokenizer.h"
#include "text.h"
#include "resource.h"

struct unicode_range {
	uint32_t first;
	uint32_t last;
	uint32_t flags;
};

static FILE *open_resource(const char *root, const char *name)
{
	char path[4096];

	if (find_resource(root, name, path, sizeof(path)) != 0)
		return NULL;
	return fopen(path, "rb");
}

static int append_prompt(char *prompt, size_t *used, const char *format, ...)
{
	va_list arguments;
	int written;

	if (*used >= TEXT_PROMPT_BYTES)
		return -1;
	va_start(arguments, format);
	written = vsnprintf(prompt + *used, TEXT_PROMPT_BYTES - *used, format, arguments);
	va_end(arguments);
	if (written < 0 || (size_t)written >= TEXT_PROMPT_BYTES - *used)
		return -1;
	*used += written;
	return 0;
}

static int next_codepoint(const char *text, size_t available, uint32_t *code)
{
	const unsigned char *p = (const unsigned char *)text;
	uint32_t value;
	int count;

	if (!available)
		return -1;
	if (p[0] < 0x80) {
		*code = p[0];
		return 1;
	}
	if (p[0] >= 0xc2 && p[0] <= 0xdf) {
		count = 2;
		value = p[0] & 31;
	} else if (p[0] >= 0xe0 && p[0] <= 0xef) {
		count = 3;
		value = p[0] & 15;
	} else if (p[0] >= 0xf0 && p[0] <= 0xf4) {
		count = 4;
		value = p[0] & 7;
	} else {
		return -1;
	}
	if ((size_t)count > available)
		return -1;
	for (int i = 1; i < count; i++) {
		if ((p[i] & 0xc0) != 0x80)
			return -1;
		value = (value << 6) | (p[i] & 63);
	}
	if ((count == 3 && value < 0x800) || (count == 4 && value < 0x10000) || value > 0x10ffff ||
	    (value >= 0xd800 && value <= 0xdfff))
		return -1;
	*code = value;
	return count;
}

static unsigned unicode_flags(const struct unicode_range *ranges, int count, uint32_t code)
{
	int low = 0, high = count;

	while (low < high) {
		int mid = low + (high - low) / 2;

		if (code < ranges[mid].first)
			high = mid;
		else if (code > ranges[mid].last)
			low = mid + 1;
		else
			return ranges[mid].flags;
	}
	return 0;
}

static int is_cjk(uint32_t code)
{
	return (code >= 0x4e00 && code <= 0x9fff) || (code >= 0x3400 && code <= 0x4dbf) ||
	       (code >= 0x20000 && code <= 0x2a6df) || (code >= 0x2a700 && code <= 0x2b73f) ||
	       (code >= 0x2b740 && code <= 0x2b81f) || (code >= 0x2b820 && code <= 0x2ceaf) ||
	       (code >= 0xf900 && code <= 0xfaff);
}

static int append_unit(struct request *request, const char *unit, size_t length)
{
	if (!length)
		return 0;
	if (request->n_units >= MODEL_MAX_UNITS || length >= sizeof(request->units[0]))
		return -1;
	memcpy(request->units[request->n_units], unit, length);
	request->units[request->n_units++][length] = '\0';
	return 0;
}

/* Match clean_token followed by split_segment_with_chinese, not punctuation
 * splitting: e.g. "hello,world" becomes the single unit "helloworld". */
static int split_text(struct request *request, const struct unicode_range *ranges, int count)
{
	const char *text = request->text;
	char pending[TEXT_UNIT_BYTES];
	size_t used = 0, length = strlen(text), offset = 0;

	while (offset < length) {
		uint32_t code;
		int bytes = next_codepoint(text + offset, length - offset, &code);
		unsigned flags;

		if (bytes < 0)
			return -1;
		flags = unicode_flags(ranges, count, code);
		if ((flags & 1) || ((flags & 2) && is_cjk(code))) {
			if (append_unit(request, pending, used) != 0)
				return -1;
			used = 0;
			if (!(flags & 1) && append_unit(request, text + offset, bytes) != 0)
				return -1;
		} else if ((flags & 2) || code == '\'') {
			if (used + bytes >= sizeof(pending))
				return -1;
			memcpy(pending + used, text + offset, bytes);
			used += bytes;
		}
		offset += bytes;
	}
	return append_unit(request, pending, used);
}

static int tokenize_prompt(const char *path, const char *prompt, struct request *request)
{
	/* C++ stays at the repository tokenizer API boundary. */
	try {
		Tokenizer tokenizer(TOKENIZER_BACKEND_LLAMA, path);
		VocabInfo info = {};

		if (!tokenizer.GetVocabInfo(&info) || info.vocab_size <= 0 || info.vocab_size > request->vocab_size)
			return -1;
		request->sequence = tokenizer.Tokenize(prompt, strlen(prompt), request->tokens, MODEL_MAX_SEQUENCE);
	} catch (const std::exception &error) {
		fprintf(stderr, "Tokenizer failed: %s\n", error.what());
		return -1;
	}
	return request->sequence > 0 && request->sequence <= MODEL_MAX_SEQUENCE ? 0 : -1;
}

int prepare_text(const char *model_path, const char *text, const char *language, int samples, struct request *request)
{
	struct unicode_range *ranges = NULL;
	uint32_t header[2];
	int hidden, audio_start, audio_pad, audio_end, timestamp;
	int n_audio = 0, n_slots = 0, frames, result = -1;
	char path[4096];
	char *prompt = NULL;
	size_t used = 0;
	FILE *file = NULL;

	if ((strcmp(language, "Chinese") != 0 && strcmp(language, "English") != 0) ||
	    strlen(text) >= sizeof(request->text) || samples < MODEL_FFT_SIZE || samples > MODEL_MAX_SAMPLES) {
		fprintf(stderr, "Expected Chinese/English, text <8192 bytes, "
				"25ms-30s audio\n");
		return -1;
	}
	prompt = (char *)malloc(TEXT_PROMPT_BYTES);
	if (!prompt)
		goto cleanup;
	file = open_resource(model_path, "_config.txt");
	if (!file)
		goto cleanup;
	if (fscanf(file, "%d %d %d %d %d %d", &request->vocab_size, &hidden, &audio_start, &audio_pad, &audio_end,
		   &timestamp) != 6 ||
	    hidden != MODEL_HIDDEN_SIZE || request->vocab_size < 1 || request->vocab_size > TEXT_MAX_VOCAB_SIZE)
		goto cleanup;
	fclose(file);
	file = open_resource(model_path, "_rules.bin");
	if (!file || fread(header, sizeof(header), 1, file) != 1 || header[0] != 0x31525551 || header[1] == 0 ||
	    header[1] > TEXT_MAX_UNICODE_RANGES)
		goto cleanup;
	ranges = (struct unicode_range *)malloc(header[1] * sizeof(*ranges));
	if (!ranges || fread(ranges, sizeof(*ranges), header[1], file) != header[1])
		goto cleanup;
	for (unsigned i = 0; i < header[1]; i++)
		if (ranges[i].first > ranges[i].last || ranges[i].last > 0x10ffff || ranges[i].flags == 0 ||
		    ranges[i].flags > 3 || (i > 0 && ranges[i].first <= ranges[i - 1].last))
			goto cleanup;
	strcpy(request->text, text);
	request->samples = samples;
	if (split_text(request, ranges, header[1]) != 0 || request->n_units == 0)
		goto cleanup;
	frames = samples / MODEL_HOP_LENGTH;
	request->n_audio = frames / MODEL_CHUNK_FRAMES * MODEL_CHUNK_TOKENS +
			   (frames % MODEL_CHUNK_FRAMES + MODEL_AUDIO_DOWNSAMPLE - 1) / MODEL_AUDIO_DOWNSAMPLE;
	if (request->n_audio > MODEL_MAX_AUDIO_TOKENS || append_prompt(prompt, &used, "<|audio_start|>") != 0)
		goto cleanup;
	for (int i = 0; i < request->n_audio; i++)
		if (append_prompt(prompt, &used, "<|audio_pad|>") != 0)
			goto cleanup;
	if (append_prompt(prompt, &used, "<|audio_end|>") != 0)
		goto cleanup;
	for (int i = 0; i < request->n_units; i++)
		if (append_prompt(prompt, &used, "%s<timestamp><timestamp>", request->units[i]) != 0)
			goto cleanup;
	if (find_resource(model_path, ".tokenizer.gguf", path, sizeof(path)) != 0 ||
	    tokenize_prompt(path, prompt, request) != 0)
		goto cleanup;
	if (request->sequence <= request->n_audio + 1 || request->tokens[0] != audio_start ||
	    request->tokens[request->n_audio + 1] != audio_end)
		goto cleanup;
	for (int i = 0; i < request->sequence; i++) {
		int token = request->tokens[i];

		if (token < 0 || token >= request->vocab_size)
			goto cleanup;
		if (token == audio_pad) {
			if (n_audio >= request->n_audio)
				goto cleanup;
			request->audio_rows[n_audio++] = i;
		} else if (token == timestamp) {
			if (n_slots >= request->n_units * 2)
				goto cleanup;
			request->slots[n_slots++] = i;
		}
	}
	if (n_audio != request->n_audio || n_slots != request->n_units * 2)
		goto cleanup;
	result = 0;
cleanup:
	if (file)
		fclose(file);
	free(ranges);
	free(prompt);
	if (result != 0)
		fprintf(stderr, "Text preparation failed: resources, UTF-8 or "
				"token limit\n");
	return result;
}
