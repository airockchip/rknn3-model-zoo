/* RKNN tensor allocation, execution and explicit resource ownership. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <dirent.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "laya.h"
#include "time_utils.h"

static int checked(int ret, const char *step)
{
	if (ret != RKNN3_SUCCESS)
		fprintf(stderr, "%s failed: %d\n", step, ret);
	return ret == RKNN3_SUCCESS ? 0 : -1;
}

int laya_join_path(char *out, const char *root, const char *name)
{
	int n = snprintf(out, PATH_MAX, "%s/%s", root, name);
	return n >= 0 && n < PATH_MAX ? 0 : -1;
}

/* Release tensors before their context; also handles partial initialization. */
int laya_release(struct laya_context *ctx)
{
	int ret = 0;
	for (int i = 0; i < LAYA_TENSOR_COUNT; ++i) {
		if (ctx->tensors[i].mem &&
		    checked(rknn3_destroy_mem(ctx->handle, ctx->tensors[i].mem), "destroy memory") <
			0)
			ret = -1;
	}
	if (ctx->handle && checked(rknn3_destroy(ctx->handle), "destroy context") < 0)
		ret = -1;
	if (ctx->tokenizer)
		tokenizers_free(ctx->tokenizer);
	ctx->tokenizer = NULL;
	free(ctx->mask_token);
	if (ctx->embedding && munmap((void *)ctx->embedding, ctx->embedding_bytes) != 0)
		ret = -1;
	if (ctx->embedding_fd >= 0 && close(ctx->embedding_fd) != 0)
		ret = -1;
	free(ctx->act);
	free(ctx->shape_attrs);
	free(ctx->shape_ids);
	free(ctx->sequence_lengths);
	free(ctx->workspace.head);
	free(ctx->workspace.options);
	free(ctx->workspace.ids);
	free(ctx->workspace.markers);
	free(ctx->workspace.valid);
	free(ctx->workspace.option_lengths);
	free(ctx->workspace.logits);
	free(ctx->workspace.probabilities);
	free(ctx->workspace.features);
	free(ctx->workspace.act_hidden);
	free(ctx->workspace.act_logits);
	memset(ctx, 0, sizeof(*ctx));
	ctx->embedding_fd = -1;
	return ret;
}

/* Validate the queried contract for every bucket before allocating tensors. */
static int validate_shape(const struct laya_context *ctx, rknn3_tensor_attr *attrs, int length)
{
	const char *names[] = {"inputs_embeds", "attention_mask", "marker_pos", "marker_mask",
	    "qtype", "logits", "p", "k", "h0"};
	const int dims[] = {3, 2, 2, 2, 1, 1, 1, 1, 1};
	const int shapes[LAYA_TENSOR_COUNT][3] = {
	    {1, length, ctx->hidden_size}, {1, length, 0}, {1, ctx->max_options, 0},
	    {1, ctx->max_options, 0}, {1, 0, 0}, {ctx->max_options, 0, 0},
	    {ctx->max_options, 0, 0}, {1, 0, 0}, {ctx->hidden_size, 0, 0}};
	for (int i = 0; i < LAYA_TENSOR_COUNT; ++i) {
		rknn3_tensor_attr *attr = &attrs[i];
		rknn3_tensor_type dtype = i != LAYA_INPUT_EMBEDS && i < LAYA_INPUT_COUNT ?
		    RKNN3_TENSOR_INT32 : RKNN3_TENSOR_FLOAT16;
		if (strcmp(attr->name, names[i]) || attr->dtype != dtype ||
		    attr->n_dims != (uint32_t)dims[i] ||
		    (attr->layout != RKNN3_TENSOR_UNDEFINED && attr->layout != RKNN3_TENSOR_NCHW))
			return -1;
		size_t stride = 1;
		for (int j = dims[i] - 1; j >= 0; --j) {
			if (attr->shape[j] != (uint32_t)shapes[i][j] || attr->stride[j] != stride)
				return -1;
			if (stride > SIZE_MAX / (size_t)shapes[i][j])
				return -1;
			stride *= shapes[i][j];
		}
		if (attr->n_elems != stride ||
		    stride > attr->aligned_size /
			(dtype == RKNN3_TENSOR_INT32 ? sizeof(int32_t) : sizeof(float16)))
			return -1;
	}
	return 0;
}

/* Query capacities, rather than assuming that bucket order equals shape ID. */
int laya_select_shape(struct laya_context *ctx, int required_tokens)
{
	int best = -1;
	for (int i = 0; i < ctx->n_shapes; ++i) {
		if (ctx->sequence_lengths[i] >= required_tokens &&
		    (best < 0 || ctx->sequence_lengths[i] < ctx->sequence_lengths[best]))
			best = i;
	}
	if (best < 0)
		return -1;
	if (ctx->active_shape == best)
		return 0;
	/* Tensor allocations depend on the selected bucket's queried sizes. */
	for (int i = 0; i < LAYA_TENSOR_COUNT; ++i) {
		if (ctx->tensors[i].mem) {
			if (checked(rknn3_destroy_mem(ctx->handle, ctx->tensors[i].mem),
				"destroy bucket tensor") < 0)
				return -1;
			ctx->tensors[i].mem = NULL;
		}
	}
	ctx->active_shape = -1;
	if (checked(rknn3_set_shape(ctx->handle, ctx->shape_ids[best]), "set shape") < 0)
		return -1;
	memcpy(ctx->attrs, ctx->shape_attrs + best * LAYA_TENSOR_COUNT, sizeof(ctx->attrs));
	for (int i = 0; i < LAYA_TENSOR_COUNT; ++i) {
		rknn3_tensor_attr *attr = &ctx->attrs[i];
		ctx->tensors[i].attr = attr;
		ctx->tensors[i].mem = rknn3_create_mem(
		    ctx->handle, attr->aligned_size, attr->core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
		if (!ctx->tensors[i].mem)
			return -1;
	}
	ctx->active_shape = best;
	ctx->sequence_length = ctx->sequence_lengths[best];
	fprintf(
	    stderr, "Sequence bucket=%d shape_id=%d\n", ctx->sequence_length, ctx->shape_ids[best]);
	return 0;
}

/* Resolve one RKNN model and its matching weight without depending on directory order. */
static int find_model_files(const char *root, char *model, char *weight)
{
	DIR *dir = opendir(root);
	struct dirent *entry;
	struct stat info;
	char path[PATH_MAX];
	const size_t suffix_length = sizeof(".rknn") - 1;
	size_t size;
	int ret = -1, length;

	model[0] = '\0';
	if (!dir) {
		fprintf(stderr, "Cannot open model directory: %s\n", root);
		return -1;
	}
	for (;;) {
		errno = 0;
		entry = readdir(dir);
		if (!entry) {
			if (errno) {
				fprintf(stderr, "Cannot read model directory: %s\n", root);
				goto out;
			}
			break;
		}
		size = strlen(entry->d_name);
		if (size <= suffix_length ||
		    strcmp(entry->d_name + size - suffix_length, ".rknn"))
			continue;
		if (laya_join_path(path, root, entry->d_name) < 0) {
			fprintf(stderr, "Model path is too long: %s\n", root);
			goto out;
		}
		if (stat(path, &info) != 0 || !S_ISREG(info.st_mode))
			continue;
		if (access(path, R_OK) != 0) {
			fprintf(stderr, "Cannot read model file: %s\n", path);
			goto out;
		}
		if (model[0]) {
			fprintf(stderr, "Expected exactly one .rknn file in: %s\n", root);
			goto out;
		}
		strcpy(model, path);
	}
	if (!model[0]) {
		fprintf(stderr, "No readable .rknn file found in: %s\n", root);
		goto out;
	}
	length = snprintf(weight, PATH_MAX, "%.*s.weight",
		(int)(strlen(model) - suffix_length), model);
	if (length < 0 || length >= PATH_MAX) {
		fprintf(stderr, "Weight path is too long for: %s\n", model);
		goto out;
	}
	if (stat(weight, &info) != 0 || !S_ISREG(info.st_mode) || access(weight, R_OK) != 0) {
		fprintf(stderr, "Missing or unreadable matching weight: %s\n", weight);
		goto out;
	}
	ret = 0;
out:
	closedir(dir);
	return ret;
}

/* Discover the model/weight pair in model_path before creating the runtime context. */
int laya_runtime_init(struct laya_context *ctx, const char *root)
{
	char model[PATH_MAX], weight[PATH_MAX];
	uint32_t cores = 0;
	rknn3_config config = {0};
	rknn3_input_output_num count = {0};
	rknn3_shape_config shape_config = {0};
	rknn3_shape_info *shape_info = NULL;
	int ret = -1;

	ctx->active_shape = -1;
	if (find_model_files(root, model, weight) < 0)
		return -1;
	if (checked(rknn3_init(&ctx->handle, NULL), "init") < 0 ||
	    checked(rknn3_load_model_from_path(ctx->handle, model, weight), "load model") < 0 ||
	    checked(rknn3_query(ctx->handle, RKNN3_QUERY_CORE_NUMBER, &cores, sizeof(cores)),
		"query cores") < 0)
		return -1;
	/* This SDK exposes core IDs 0 through 7 on the supported target. */
	if (cores == 0 || cores > 8)
		return -1;
	config.run_core_mask = (1U << cores) - 1;
	fprintf(stderr, "NPU cores=%u mask=0x%x\n", cores, config.run_core_mask);
	if (checked(rknn3_model_init(ctx->handle, &config), "model init") < 0 ||
	    checked(rknn3_query(ctx->handle, RKNN3_QUERY_IN_OUT_NUM, &count, sizeof(count)),
		"query count") < 0 ||
	    count.n_input != LAYA_INPUT_COUNT || count.n_output != LAYA_OUTPUT_COUNT ||
	    checked(rknn3_query(ctx->handle, RKNN3_QUERY_DYNAMIC_SHAPE_CONFIG, &shape_config,
			sizeof(shape_config)), "query shape config") < 0)
		return -1;
	if (!shape_config.n_shapes || shape_config.n_shapes > RKNN3_MAX_DYNAMIC_SHAPE_NUM)
		return -1;
	ctx->n_shapes = shape_config.n_shapes;
	ctx->shape_attrs = (rknn3_tensor_attr *)calloc(
		ctx->n_shapes * LAYA_TENSOR_COUNT, sizeof(*ctx->shape_attrs));
	ctx->shape_ids = (int *)calloc(ctx->n_shapes, sizeof(*ctx->shape_ids));
	ctx->sequence_lengths = (int *)calloc(ctx->n_shapes, sizeof(*ctx->sequence_lengths));
	shape_info = (rknn3_shape_info *)calloc(ctx->n_shapes, sizeof(*shape_info));
	if (!ctx->shape_attrs || !ctx->shape_ids || !ctx->sequence_lengths || !shape_info)
		goto out;
	for (int i = 0; i < ctx->n_shapes; ++i) {
		shape_info[i].shape_id = i;
		shape_info[i].input_attrs = ctx->shape_attrs + i * LAYA_TENSOR_COUNT;
		shape_info[i].output_attrs = shape_info[i].input_attrs + LAYA_INPUT_COUNT;
	}
	if (checked(rknn3_query(ctx->handle, RKNN3_QUERY_DYNAMIC_SHAPE_INFO, shape_info,
			sizeof(*shape_info) * ctx->n_shapes), "query shape tensors") < 0)
		goto out;
	for (int i = 0; i < ctx->n_shapes; ++i) {
		rknn3_tensor_attr *input = &shape_info[i].input_attrs[LAYA_INPUT_EMBEDS];
		rknn3_tensor_attr *markers = &shape_info[i].input_attrs[LAYA_INPUT_MARKER_POS];
		if (input->n_dims != 3 || input->shape[1] < 1 || input->shape[1] > INT_MAX ||
		    input->shape[2] < 1 || input->shape[2] > INT_MAX || markers->n_dims != 2 ||
		    markers->shape[1] < 2 || markers->shape[1] > INT_MAX)
			goto out;
		/* Infer dimensions once, then require every bucket to share the same contract. */
		if (i == 0) {
			ctx->hidden_size = input->shape[2];
			ctx->max_options = markers->shape[1];
			ctx->max_length = 0;
		}
		int length = input->shape[1];
		if (shape_info[i].n_inputs != LAYA_INPUT_COUNT ||
		    shape_info[i].n_outputs != LAYA_OUTPUT_COUNT ||
		    validate_shape(ctx, shape_info[i].input_attrs, length) < 0) {
			fprintf(stderr, "Invalid or inconsistent Laya RKNN tensor contract\n");
			goto out;
		}
		ctx->shape_ids[i] = shape_info[i].shape_id;
		ctx->sequence_lengths[i] = length;
		if (length > ctx->max_length)
			ctx->max_length = length;
	}
	ret = ctx->max_length >= ctx->max_options ? 0 : -1;
out:
	free(shape_info);
	return ret;
}

/* Synchronize cacheable input/output buffers around one NPU invocation. */
int laya_runtime_run(struct laya_context *ctx)
{
	for (int i = 0; i < LAYA_INPUT_COUNT; ++i) {
		if (checked(rknn3_mem_sync(
				ctx->handle, ctx->tensors[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE),
			"sync input") < 0)
			return -1;
	}
	int64_t start = getCurrentTimeUs();
	int ret = checked(rknn3_run(ctx->handle, ctx->tensors, LAYA_INPUT_COUNT,
	    ctx->tensors + LAYA_INPUT_COUNT, LAYA_OUTPUT_COUNT), "run");
	fprintf(stderr, "rknn3_run: %.3f ms\n", (getCurrentTimeUs() - start) / 1000.0);
	if (ret < 0)
		return ret;
	for (int i = LAYA_INPUT_COUNT; i < LAYA_TENSOR_COUNT; ++i) {
		if (checked(rknn3_mem_sync(
				ctx->handle, ctx->tensors[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE),
			"sync output") < 0)
			return -1;
	}
	return 0;
}
