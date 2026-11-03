#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#include "resource.h"

int find_resource(const char *root, const char *suffix, char *path, size_t capacity)
{
	DIR *directory = opendir(root);
	struct dirent *entry;
	size_t suffix_length = strlen(suffix);
	int found = 0, result = -1;

	if (!directory) {
		fprintf(stderr, "Cannot open resource directory %s: %s\n", root, strerror(errno));
		return -1;
	}
	for (;;) {
		struct stat status;
		char candidate[4096];
		size_t length;
		int written;

		errno = 0;
		entry = readdir(directory);
		if (!entry) {
			if (errno) {
				fprintf(stderr, "Cannot read resource directory %s: %s\n", root, strerror(errno));
				goto cleanup;
			}
			break;
		}
		length = strlen(entry->d_name);
		if (length < suffix_length || strcmp(entry->d_name + length - suffix_length, suffix) != 0)
			continue;
		written = snprintf(candidate, sizeof(candidate), "%s/%s", root, entry->d_name);
		if (written < 0 || (size_t)written >= sizeof(candidate) || (size_t)written >= capacity) {
			fprintf(stderr, "Resource path is too long: %s/%s\n", root, entry->d_name);
			goto cleanup;
		}
		if (stat(candidate, &status) != 0) {
			fprintf(stderr, "Cannot stat resource %s: %s\n", candidate, strerror(errno));
			goto cleanup;
		}
		if (!S_ISREG(status.st_mode))
			continue;
		if (found) {
			fprintf(stderr, "Multiple files match suffix %s in %s\n", suffix, root);
			goto cleanup;
		}
		strcpy(path, candidate);
		found = 1;
	}
	if (!found) {
		fprintf(stderr, "No file matches suffix %s in %s\n", suffix, root);
		goto cleanup;
	}
	result = 0;
cleanup:
	closedir(directory);
	return result;
}
