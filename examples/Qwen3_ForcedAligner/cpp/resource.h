#ifndef QWEN3_FORCED_ALIGNER_RESOURCE_H
#define QWEN3_FORCED_ALIGNER_RESOURCE_H

#include <stddef.h>

int find_resource(const char *root, const char *suffix, char *path, size_t capacity);

#endif
