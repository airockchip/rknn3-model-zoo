#ifndef LAYA_H
#define LAYA_H

#include <stddef.h>
#include "rknn3_api.h"
#include "float16.h"
#include "tokenizers_c.h"

enum laya_tensor_index {
	LAYA_INPUT_EMBEDS,
	LAYA_INPUT_ATTENTION_MASK,
	LAYA_INPUT_MARKER_POS,
	LAYA_INPUT_MARKER_MASK,
	LAYA_INPUT_QTYPE,
	LAYA_INPUT_COUNT,
	LAYA_OUTPUT_LOGITS = LAYA_INPUT_COUNT,
	LAYA_OUTPUT_PROBABILITIES,
	LAYA_OUTPUT_OPTION_COUNT,
	LAYA_OUTPUT_POOLED,
	LAYA_TENSOR_COUNT,
};

enum { LAYA_OUTPUT_COUNT = LAYA_TENSOR_COUNT - LAYA_INPUT_COUNT };
enum laya_question_type { LAYA_CHOICE, LAYA_SCORE, LAYA_NOUL, LAYA_QTYPE_COUNT };

/* Upstream sequence assembly and act features are part of the model contract. */
enum {
	LAYA_OPTION_TEXT_TOKENS = 48,
	LAYA_HEAD_RESERVED_TOKENS = 16,
	LAYA_MIN_HEAD_TOKENS = 8,
	LAYA_MIN_OPTION_TOKENS = 4,
	LAYA_SPECIAL_TOKENS = 4,
	LAYA_ACT_STATS = 4,
};
static const double laya_option_count_scale = 255.0;

struct laya_act_layout {
	int input_size, hidden_size, output_size;
	size_t first_bias, second_weight, second_bias, weight_count;
};

/* Reused across questions; allocated from model/resource sizes and freed with context. */
struct laya_workspace {
	int32_t *head, *options, *ids, *markers, *valid;
	int *option_lengths;
	size_t ids_capacity;
	double *logits, *probabilities, *features, *act_hidden, *act_logits;
};

struct laya_context {
	rknn3_context handle;
	/* All bucket contracts; attrs/tensors below describe the active bucket. */
	rknn3_tensor_attr *shape_attrs;
	int n_shapes, active_shape, sequence_length, max_length;
	int *shape_ids, *sequence_lengths;
	rknn3_tensor_attr attrs[LAYA_TENSOR_COUNT];
	rknn3_tensor tensors[LAYA_TENSOR_COUNT];
	TokenizerHandle tokenizer;
	char *mask_token;
	/* Read-only mapped FP16 embedding table, owned until laya_release(). */
	const float16 *embedding;
	size_t embedding_bytes;
	int embedding_fd;
	float *act;
	struct laya_act_layout act_layout;
	struct laya_workspace workspace;
	int hidden_size, max_options;
	int vocab, cls, sep, mask, pad, head_limit;
};

int laya_runtime_init(struct laya_context *ctx, const char *model_path);
int laya_select_shape(struct laya_context *ctx, int required_tokens);
int laya_runtime_run(struct laya_context *ctx);
int laya_release(struct laya_context *ctx);
int laya_join_path(char *out, const char *root, const char *name);

#endif
