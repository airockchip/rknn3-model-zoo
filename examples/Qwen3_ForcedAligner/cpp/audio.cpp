#include <math.h>
#include <new>
#include <string.h>
#include <vector>

#include "audio.h"
#include "model_config.h"
#include "audio_utils.h"

/* Adapt the public C++ audio helper to the demo's plain-array interface. */
int compute_mel(audio_buffer_t *audio, float *filters, float *mel, int frames)
{
	int actual = 0;

	if (!audio || !audio->data || !filters || !mel || frames < 2 || frames > MODEL_MAX_FRAMES ||
	    audio->num_frames < frames * MODEL_HOP_LENGTH)
		return -1;
	for (int i = 0; i < frames * MODEL_HOP_LENGTH; i++)
		if (!isfinite(audio->data[i]))
			return -1;
	try {
		std::vector<float> values(MODEL_MEL_BINS * frames);

		audio_preprocess(audio, filters, MODEL_FFT_SIZE, MODEL_HOP_LENGTH, MODEL_MEL_BINS,
				 frames * MODEL_HOP_LENGTH, values, &actual);
		if (actual != frames || values.size() != (size_t)(MODEL_MEL_BINS * frames))
			return -1;
		memcpy(mel, values.data(), values.size() * sizeof(float));
	} catch (const std::bad_alloc &) {
		return -1;
	}
	return 0;
}
