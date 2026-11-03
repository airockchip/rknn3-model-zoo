#ifndef QWEN3_FORCED_ALIGNER_TIMESTAMP_H_
#define QWEN3_FORCED_ALIGNER_TIMESTAMP_H_

#include <stdbool.h>
#include <stdlib.h>

#include "model_config.h"

/*
 * Match Qwen3ForceAlignProcessor.fix_timestamp: retain the first longest
 * nondecreasing subsequence, then fill short gaps or interpolate longer gaps.
 * Input values are milliseconds, not timestamp class indices.
 */
static int fix_timestamps(int *values, int count)
{
	int *length = NULL, *parent = NULL;
	bool *normal = NULL;
	int best = 0;

	if (!values || count < 1 || count > MODEL_MAX_SEQUENCE)
		return -1;
	length = (int *)malloc(count * sizeof(*length));
	parent = (int *)malloc(count * sizeof(*parent));
	normal = (bool *)calloc(count, sizeof(*normal));
	if (!length || !parent || !normal) {
		free(length);
		free(parent);
		free(normal);
		return -1;
	}
	for (int i = 0; i < count; ++i) {
		length[i] = 1;
		parent[i] = -1;
		for (int j = 0; j < i; ++j) {
			if (values[j] <= values[i] && length[j] + 1 > length[i]) {
				length[i] = length[j] + 1;
				parent[i] = j;
			}
		}
		if (length[i] > length[best])
			best = i;
	}
	for (int i = best; i >= 0; i = parent[i])
		normal[i] = true;
	for (int i = 0; i < count;) {
		if (normal[i]) {
			++i;
			continue;
		}
		int end = i;
		while (end < count && !normal[end])
			++end;
		int left = i - 1;
		for (int k = i; k < end; ++k) {
			if (left < 0) {
				values[k] = values[end];
			} else if (end == count) {
				values[k] = values[left];
			} else if (end - i <= 2) {
				values[k] = k - left <= end - k ? values[left] : values[end];
			} else {
				double step = (values[end] - values[left]) / (double)(end - i + 1);
				values[k] = (int)(values[left] + step * (k - i + 1));
			}
		}
		i = end;
	}
	free(length);
	free(parent);
	free(normal);
	return 0;
}

#endif /* QWEN3_FORCED_ALIGNER_TIMESTAMP_H_ */
