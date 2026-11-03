/* JSON request handling, token preparation and CPU postprocessing. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "laya_text.h"
#include "file_utils.h"

/* Borrow string storage from JSON; callers must not modify it while in use. */
const char *laya_json_string(const json &value)
{
	return value.get_ref<const json::string_t &>().c_str();
}

int laya_load_json(const char *path, json *out)
{
	char *data = NULL;
	int size = read_data_from_file(path, &data);
	if (size <= 0) {
		free(data);
		return -1;
	}
	*out = json::parse(data, data + size, NULL, false);
	free(data);
	return out->is_discarded() ? -1 : 0;
}

/* The shared file writer does not propagate short writes. */
int laya_save_json(const char *path, const json *result)
{
	std::string data = result->dump(2) + "\n";
	FILE *fp = fopen(path, "wb");
	if (!fp)
		return -1;
	int ret = fwrite(data.data(), 1, data.size(), fp) == data.size() ? 0 : -1;
	if (fclose(fp) != 0)
		ret = -1;
	return ret;
}

/* Read bounded dimensions before calculating allocation sizes or binary offsets. */
static int positive_dimension(const json &object, const char *key, int *value)
{
	if (!object.is_object() || !object.contains(key) ||
	    !object[key].is_number_integer() || object[key] < 1 || object[key] > INT_MAX)
		return -1;
	*value = object[key].get<int>();
	return 0;
}

static int load_act_layout(struct laya_context *ctx, const json &config)
{
	struct laya_act_layout *layout = &ctx->act_layout;

	if (ctx->hidden_size > INT_MAX - LAYA_ACT_STATS)
		return -1;
	layout->input_size = ctx->hidden_size + LAYA_ACT_STATS;
	if (!config.contains("act_head") ||
	    positive_dimension(config["act_head"], "hidden_size", &layout->hidden_size) < 0 ||
	    positive_dimension(config["act_head"], "output_size", &layout->output_size) < 0) {
		fprintf(stderr, "Missing or invalid act_head dimensions\n");
		return -1;
	}
	/* The shared reader returns an int byte count; reject oversized resources. */
	uint64_t first = (uint64_t)layout->input_size * layout->hidden_size;
	uint64_t second = first + layout->hidden_size;
	uint64_t bias = second + (uint64_t)layout->hidden_size * layout->output_size;
	uint64_t count = bias + layout->output_size;
	if (count > INT_MAX / sizeof(float))
		return -1;
	layout->first_bias = first;
	layout->second_weight = second;
	layout->second_bias = bias;
	layout->weight_count = count;
	return 0;
}

static int allocate_workspace(struct laya_context *ctx)
{
	struct laya_workspace *work = &ctx->workspace;
	const struct laya_act_layout *act = &ctx->act_layout;
	/* Include the full prefix before state truncation, even for a small bucket. */
	uint64_t capacity = (uint64_t)ctx->head_limit +
	    (uint64_t)ctx->max_options * (LAYA_OPTION_TEXT_TOKENS + 1) + LAYA_SPECIAL_TOKENS;
	if (capacity < (uint64_t)ctx->max_length)
		capacity = ctx->max_length;
	if (capacity > INT_MAX)
		return -1;
	work->ids_capacity = capacity;
	work->head = (int32_t *)calloc(ctx->head_limit, sizeof(*work->head));
	work->options = (int32_t *)calloc(
	    (size_t)ctx->max_options * (LAYA_OPTION_TEXT_TOKENS + 1), sizeof(*work->options));
	work->ids = (int32_t *)calloc(work->ids_capacity, sizeof(*work->ids));
	work->markers = (int32_t *)calloc(ctx->max_options, sizeof(*work->markers));
	work->valid = (int32_t *)calloc(ctx->max_options, sizeof(*work->valid));
	work->option_lengths = (int *)calloc(ctx->max_options, sizeof(*work->option_lengths));
	work->logits = (double *)calloc(ctx->max_options, sizeof(*work->logits));
	work->probabilities = (double *)calloc(ctx->max_options, sizeof(*work->probabilities));
	work->features = (double *)calloc(act->input_size, sizeof(*work->features));
	work->act_hidden = (double *)calloc(act->hidden_size, sizeof(*work->act_hidden));
	work->act_logits = (double *)calloc(act->output_size, sizeof(*work->act_logits));
	return work->head && work->options && work->ids && work->markers && work->valid &&
	    work->option_lengths && work->logits && work->probabilities && work->features &&
	    work->act_hidden && work->act_logits ? 0 : -1;
}

/* Resolve special-token roles through the tokenizer vocabulary, never numeric defaults. */
static int load_tokenizer(struct laya_context *ctx, const char *root)
{
	char path[PATH_MAX];
	char *data = NULL;
	json roles;
	const char *names[] = {"cls_token", "sep_token", "mask_token", "pad_token"};
	int *ids[] = {&ctx->cls, &ctx->sep, &ctx->mask, &ctx->pad};

	if (laya_join_path(path, root, "tokenizer.json") < 0)
		return -1;
	int size = read_data_from_file(path, &data);
	if (size <= 0) {
		free(data);
		return -1;
	}
	ctx->tokenizer = tokenizers_new_from_str(data, size);
	free(data);
	if (!ctx->tokenizer || laya_join_path(path, root, "tokenizer_config.json") < 0 ||
	    laya_load_json(path, &roles) < 0 || !roles.is_object())
		return -1;
	for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
		if (!roles.contains(names[i]))
			return -1;
		const json *value = &roles[names[i]];
		/* Hugging Face stores a role as a string or an AddedToken object. */
		if (value->is_object() && value->contains("content"))
			value = &(*value)["content"];
		if (!value->is_string() || !laya_json_string(*value)[0])
			return -1;
		const char *token = laya_json_string(*value);
		int32_t id = -1;
		tokenizers_token_to_id(ctx->tokenizer, token, strlen(token), &id);
		if (id < 0) {
			fprintf(stderr, "Special token is absent from the tokenizer: %s\n", names[i]);
			return -1;
		}
		*ids[i] = id;
		if (!strcmp(names[i], "mask_token")) {
			ctx->mask_token = strdup(token);
			if (!ctx->mask_token)
				return -1;
		}
	}
	return 0;
}

/* Load CPU-only parameters after RKNN has supplied the input/output dimensions. */
int laya_load_resources(struct laya_context *ctx, const char *root, json *config)
{
	char path[PATH_MAX];
	if (ctx->hidden_size < 1 || ctx->max_options < 2 || ctx->max_length < ctx->max_options ||
	    laya_join_path(path, root, "laya.json") < 0 || laya_load_json(path, config) < 0 ||
	    !config->is_object())
		return -1;
	if (!config->contains("temperature") || !(*config)["temperature"].is_array() ||
	    (*config)["temperature"].size() != LAYA_QTYPE_COUNT ||
	    !config->contains("temperature_by_options") ||
	    !(*config)["temperature_by_options"].is_object())
		return -1;
	for (json::const_iterator v = (*config)["temperature"].begin();
	     v != (*config)["temperature"].end(); ++v) {
		if (!v->is_number() || v->get<double>() <= 0 || !isfinite(v->get<double>()))
			return -1;
	}
	for (json::const_iterator v = (*config)["temperature_by_options"].begin();
	     v != (*config)["temperature_by_options"].end(); ++v) {
		if (!v->is_number() || v->get<double>() <= 0 || !isfinite(v->get<double>()))
			return -1;
	}
	if (positive_dimension(*config, "head_max_len", &ctx->head_limit) < 0 ||
	    ctx->head_limit < LAYA_HEAD_RESERVED_TOKENS || load_act_layout(ctx, *config) < 0 ||
	    load_tokenizer(ctx, root) < 0)
		return -1;
	if (laya_join_path(path, root, "embedding.bin") < 0)
		return -1;
	ctx->embedding_fd = open(path, O_RDONLY);
	struct stat st;
	/* The table may include padded vocabulary rows; infer them from the file size. */
	size_t row_bytes = (size_t)ctx->hidden_size * sizeof(float16);
	if (ctx->embedding_fd < 0 || fstat(ctx->embedding_fd, &st) != 0 ||
	    st.st_size <= 0 || (uint64_t)st.st_size > SIZE_MAX || st.st_size % row_bytes ||
	    (uint64_t)st.st_size / row_bytes > INT_MAX)
		return -1;
	ctx->embedding_bytes = st.st_size;
	ctx->vocab = ctx->embedding_bytes / row_bytes;
	size_t vocab_size = 0;
	tokenizers_get_vocab_size(ctx->tokenizer, &vocab_size);
	if (!vocab_size || vocab_size > (size_t)ctx->vocab)
		return -1;
	int specials[] = {ctx->cls, ctx->sep, ctx->mask, ctx->pad};
	for (size_t i = 0; i < sizeof(specials) / sizeof(specials[0]); ++i) {
		if (specials[i] < 0 || specials[i] >= ctx->vocab)
			return -1;
	}
	void *mapped =
	    mmap(NULL, ctx->embedding_bytes, PROT_READ, MAP_PRIVATE, ctx->embedding_fd, 0);
	if (mapped == MAP_FAILED)
		return -1;
	ctx->embedding = (const float16 *)mapped;
	if (laya_join_path(path, root, "act.bin") < 0)
		return -1;
	char *data = NULL;
	int size = read_data_from_file(path, &data);
	ctx->act = (float *)data;
	if (size != (int)(ctx->act_layout.weight_count * sizeof(float)))
		return -1;
	for (size_t i = 0; i < ctx->act_layout.weight_count; ++i) {
		if (!isfinite(ctx->act[i]))
			return -1;
	}
	return allocate_workspace(ctx);
}

/* Match Python json.dumps(..., ensure_ascii=False), including separator spaces. */
static std::string state_text(const json &value)
{
	if (value.is_string())
		return value.get<std::string>();
	if (!value.is_object() && !value.is_array())
		return value.dump();
	std::string text = value.is_object() ? "{" : "[";
	bool first = true;
	for (json::const_iterator item = value.begin(); item != value.end(); ++item) {
		if (!first)
			text += ", ";
		first = false;
		if (value.is_object())
			text += json(item.key()).dump() + ": ";
		const json &v = item.value();
		text += v.is_string() ? v.dump() : state_text(v);
	}
	return text + (value.is_object() ? "}" : "]");
}

/* Reserve mask tokens for option markers and copy only the required token prefix. */
static int encode(struct laya_context *ctx, const char *prefix, const char *text,
    const char *mask, int32_t *tokens, int capacity)
{
	TokenizerEncodeResult result = {0};
	size_t prefix_len = strlen(prefix), text_len = strlen(text), mask_len = strlen(mask);
	char *buffer, *src, *dst, *match;
	int count = -1;

	if (!mask_len || capacity < 0 || text_len > SIZE_MAX - prefix_len - 1)
		return -1;
	buffer = (char *)malloc(prefix_len + text_len + 1);
	if (!buffer)
		return -1;
	memcpy(buffer, prefix, prefix_len);
	memcpy(buffer + prefix_len, text, text_len + 1);
	src = buffer;
	dst = buffer;
	while ((match = strstr(src, mask)) != NULL) {
		size_t length = match - src;

		memmove(dst, src, length);
		dst += length;
		*dst++ = ' ';
		src = match + mask_len;
	}
	memmove(dst, src, strlen(src) + 1);
	/* Special tokens are inserted by laya_prepare(), not by the tokenizer. */
	tokenizers_encode(ctx->tokenizer, buffer, strlen(buffer), 0, &result);
	if (!result.len || result.token_ids) {
		count = result.len < (size_t)capacity ? (int)result.len : capacity;
		for (int i = 0; i < count; ++i)
			tokens[i] = result.token_ids[i];
	}
	/* The tokenizer owns its result allocation, including any truncated suffix. */
	tokenizers_free_encode_results(&result, 1);
	free(buffer);
	return count;
}

/* Preserve option order because it determines the answer labels. */
static int question_options(const json &question, int max_options, const char **kind,
	json *keys, json *texts)
{
	if (!question.is_object() || !question.contains("type") || !question["type"].is_string() ||
	    !question.contains("instructions") || !question["instructions"].is_string())
		return -1;
	*kind = laya_json_string(question["type"]);
	*keys = json::array();
	*texts = json::array();
	json criteria = question.contains("criteria") ? question["criteria"] : json();
	if (!strcmp(*kind, "choice")) {
		if (criteria.is_array()) {
			json object = json::object();
			for (json::const_iterator v = criteria.begin(); v != criteria.end(); ++v) {
				if (!v->is_string() || !laya_json_string(*v)[0] ||
				    object.contains(laya_json_string(*v)))
					return -1;
				object[laya_json_string(*v)] = json();
			}
			criteria = object;
		}
		if (!criteria.is_object())
			return -1;
		for (json::const_iterator item = criteria.begin(); item != criteria.end(); ++item) {
			if (item.key().empty() ||
			    (!item.value().is_null() && !item.value().is_string()))
				return -1;
			keys->push_back(item.key());
			const char *desc = item.value().is_null() ? "" : laya_json_string(item.value());
			texts->push_back(item.key() + (desc[0] ? std::string(": ") + desc : ""));
		}
	} else if (!strcmp(*kind, "score")) {
		if (!criteria.is_array())
			return -1;
		for (size_t i = 0; i < criteria.size(); ++i) {
			if (!criteria[i].is_string())
				return -1;
			char level[32];
			snprintf(level, sizeof(level), "%zu", i);
			keys->push_back(level);
			texts->push_back(
			    std::string("level ") + level + ": " + laya_json_string(criteria[i]));
		}
	} else if (!strcmp(*kind, "noul")) {
		if (criteria.is_null())
			criteria = json::object();
		if (!criteria.is_object())
			return -1;
		for (json::const_iterator item = criteria.begin(); item != criteria.end(); ++item) {
			if (!item.value().is_null() && !item.value().is_string())
				return -1;
		}
		const char *labels[] = {"false", "true"};
		const char *defaults[] = {
		    "no, the statement does not hold", "yes, the statement holds"};
		for (int i = 0; i < 2; ++i) {
			const char *desc = defaults[i];
			if (criteria.contains(labels[i]) && criteria[labels[i]].is_string() &&
			    laya_json_string(criteria[labels[i]])[0])
				desc = laya_json_string(criteria[labels[i]]);
			keys->push_back(labels[i]);
			texts->push_back(std::string(labels[i]) + ": " + desc);
		}
	} else {
		return -1;
	}
	return keys->size() >= 2 && keys->size() <= (size_t)max_options ? 0 : -1;
}

/* Build the upstream prefix and truncate state within the configured token budget. */
int laya_prepare(struct laya_context *ctx, const json &state, const json &question,
    json *keys, int *tokens, int *qtype)
{
	const char *kind, *mask;
	char prefix[32];
	struct laya_workspace *work = &ctx->workspace;
	int32_t *head = work->head, *ids = work->ids;
	int32_t *markers = work->markers, *valid = work->valid;
	int *opt_lengths = work->option_lengths;
	int head_count, count = 0;
	int budget = ctx->head_limit;
	json texts;

	if (question_options(question, ctx->max_options, &kind, keys, &texts) < 0)
		return -1;
	*qtype = !strcmp(kind, "choice") ? LAYA_CHOICE :
	    !strcmp(kind, "score") ? LAYA_SCORE : LAYA_NOUL;
	memset(markers, 0, ctx->max_options * sizeof(*markers));
	memset(valid, 0, ctx->max_options * sizeof(*valid));
	mask = ctx->mask_token;
	snprintf(prefix, sizeof(prefix), "%s question: ", kind);
	head_count = encode(ctx, prefix, laya_json_string(question["instructions"]), mask,
	    head, ctx->head_limit);
	if (head_count < 0)
		return -1;
	for (size_t i = 0; i < texts.size(); ++i) {
		int32_t *option = work->options + i * (LAYA_OPTION_TEXT_TOKENS + 1);
		int length = encode(ctx, " ", laya_json_string(texts[i]), mask,
		    option + 1, LAYA_OPTION_TEXT_TOKENS);

		if (length < 0)
			return -1;
		option[0] = ctx->mask;
		opt_lengths[i] = length + 1;
		budget -= opt_lengths[i];
	}
	if (budget < LAYA_HEAD_RESERVED_TOKENS) {
		int per = (ctx->head_limit - LAYA_HEAD_RESERVED_TOKENS) / (int)texts.size();
		if (per < LAYA_MIN_OPTION_TOKENS)
			per = LAYA_MIN_OPTION_TOKENS;
		budget = ctx->head_limit;
		for (size_t i = 0; i < texts.size(); ++i) {
			if (opt_lengths[i] > per)
				opt_lengths[i] = per;
			budget -= opt_lengths[i];
		}
	}
	if (budget < LAYA_MIN_HEAD_TOKENS)
		budget = LAYA_MIN_HEAD_TOKENS;
	if (head_count > budget)
		head_count = budget;
	ids[count++] = ctx->cls;
	memcpy(ids + count, head, head_count * sizeof(*head));
	count += head_count;
	ids[count++] = ctx->sep;
	for (size_t i = 0; i < texts.size(); ++i) {
		if ((size_t)count + opt_lengths[i] + 2 > work->ids_capacity)
			return -1;
		markers[i] = count;
		valid[i] = 1;
		memcpy(ids + count, work->options + i * (LAYA_OPTION_TEXT_TOKENS + 1),
		    opt_lengths[i] * sizeof(*ids));
		count += opt_lengths[i];
	}
	ids[count++] = ctx->sep;
	int room = ctx->max_length - count - 1;
	if (room < 0)
		room = 0;
	std::string state_value = state_text(state);
	int state_count = encode(ctx, "", state_value.c_str(), mask, ids + count, room);
	if (state_count < 0)
		return -1;
	count += state_count;
	ids[count++] = ctx->sep;
	if (count > ctx->max_length)
		count = ctx->max_length;
	if (markers[texts.size() - 1] >= count)
		return -1;
	*tokens = count;
	if (laya_select_shape(ctx, *tokens) < 0)
		return -1;
	/* Pad only after selecting a bucket; attention masks exclude padded positions. */
	int length = ctx->sequence_length;
	for (int i = count; i < length; ++i)
		ids[i] = ctx->pad;
	for (int i = 0; i < LAYA_INPUT_COUNT; ++i)
		memset(ctx->tensors[i].mem->virt_addr, 0, ctx->attrs[i].aligned_size);
	float16 *embedding = (float16 *)ctx->tensors[LAYA_INPUT_EMBEDS].mem->virt_addr;
	int32_t *attention = (int32_t *)ctx->tensors[LAYA_INPUT_ATTENTION_MASK].mem->virt_addr;
	for (int i = 0; i < length; ++i) {
		if (ids[i] < 0 || ids[i] >= ctx->vocab)
			return -1;
		memcpy(embedding + (size_t)i * ctx->hidden_size,
		    ctx->embedding + (size_t)ids[i] * ctx->hidden_size,
		    ctx->hidden_size * sizeof(float16));
		attention[i] = i < *tokens ? 1 : 0;
	}
	memcpy(ctx->tensors[LAYA_INPUT_MARKER_POS].mem->virt_addr, markers,
	    ctx->max_options * sizeof(*markers));
	memcpy(ctx->tensors[LAYA_INPUT_MARKER_MASK].mem->virt_addr, valid,
	    ctx->max_options * sizeof(*valid));
	*(int32_t *)ctx->tensors[LAYA_INPUT_QTYPE].mem->virt_addr = *qtype;
	return 0;
}

static double round4(double value)
{
	return round(value * 10000.0) / 10000.0;
}

/* Two linear layers with exact GELU; weights use export_cpu_resources order. */
static double compute_act_probability(struct laya_context *ctx, const double *features)
{
	const struct laya_act_layout *layout = &ctx->act_layout;
	const float *weights = ctx->act;
	double *hidden = ctx->workspace.act_hidden, *act = ctx->workspace.act_logits;
	for (int i = 0; i < layout->hidden_size; ++i) {
		double value = weights[layout->first_bias + i];
		for (int j = 0; j < layout->input_size; ++j)
			value += weights[(size_t)i * layout->input_size + j] * features[j];
		hidden[i] = value * 0.5 * (1 + erf(value / sqrt(2.0)));
	}
	for (int i = 0; i < layout->output_size; ++i) {
		act[i] = weights[layout->second_bias + i];
		for (int j = 0; j < layout->hidden_size; ++j)
			act[i] += weights[layout->second_weight +
			    (size_t)i * layout->hidden_size + j] * hidden[j];
	}
	double max_act = act[0], sum = 0;
	for (int i = 1; i < layout->output_size; ++i)
		max_act = fmax(max_act, act[i]);
	for (int i = 0; i < layout->output_size; ++i)
		sum += exp(act[i] - max_act);
	return exp(act[0] - max_act) / sum;
}

/* Compute the CPU act head from raw probabilities, then calibrate answers. */
int laya_postprocess(struct laya_context *ctx, const json &question, const json &keys,
    const json &config, int qtype, json *answer)
{
	double *logits = ctx->workspace.logits, *p = ctx->workspace.probabilities;
	double *features = ctx->workspace.features;
	int count = keys.size(), best = 0;
	for (int i = LAYA_INPUT_COUNT; i < LAYA_TENSOR_COUNT; ++i) {
		float16 *value = (float16 *)ctx->tensors[i].mem->virt_addr;
		for (uint32_t j = 0; j < ctx->attrs[i].n_elems; ++j) {
			if (!isfinite(fp16_to_fp32(value[j])))
				return -1;
		}
	}
	double k = fp16_to_fp32(*(float16 *)ctx->tensors[LAYA_OUTPUT_OPTION_COUNT].mem->virt_addr);
	if (k != count)
		return -1;
	double first = -1, second = -1, entropy = 0;
	for (int i = 0; i < ctx->max_options; ++i) {
		logits[i] = fp16_to_fp32(((float16 *)ctx->tensors[LAYA_OUTPUT_LOGITS].mem->virt_addr)[i]);
		p[i] = fp16_to_fp32(((float16 *)ctx->tensors[LAYA_OUTPUT_PROBABILITIES].mem->virt_addr)[i]);
		if (p[i] < 0 || p[i] > 1)
			return -1;
		entropy -= p[i] * log(fmax(p[i], 1e-9));
		if (p[i] > first) {
			second = first;
			first = p[i];
		} else if (p[i] > second) {
			second = p[i];
		}
	}
	for (int i = 0; i < ctx->hidden_size; ++i)
		features[i] = fp16_to_fp32(((float16 *)ctx->tensors[LAYA_OUTPUT_POOLED].mem->virt_addr)[i]);
	features[ctx->hidden_size + 0] = first;
	features[ctx->hidden_size + 1] = first - second;
	features[ctx->hidden_size + 2] = entropy / log(k);
	features[ctx->hidden_size + 3] = k / laya_option_count_scale;
	double act_prob = compute_act_probability(ctx, features);
	const char *kind = laya_json_string(question["type"]);
	char bucket[32];
	snprintf(bucket, sizeof(bucket), "%s:%s", kind,
	    count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+");
	double temperature = config["temperature"][qtype].get<double>();
	if (config["temperature_by_options"].contains(bucket))
		temperature = config["temperature_by_options"][bucket].get<double>();
	double sum = 0, confidence = 1, score = 0;
	for (int i = 1; i < count; ++i) {
		if (logits[i] > logits[best])
			best = i;
	}
	for (int i = 0; i < count; ++i) {
		p[i] = exp((logits[i] - logits[best]) / temperature);
		sum += p[i];
	}
	*answer = json::object();
	(*answer)["type"] = kind;
	(*answer)["rl_agent"] = {{"act_probability", act_prob}};
	json probs = json::object(), legend = json::object();
	for (int i = 0; i < count; ++i) {
		p[i] /= sum;
		confidence += p[i] * log(fmax(p[i], 1e-12)) / log((double)count);
		score += i * p[i];
		probs[laya_json_string(keys[i])] = round4(p[i]);
		if (!strcmp(kind, "score")) {
			char level[16];
			snprintf(level, sizeof(level), "%d", i);
			legend[level] = question["criteria"][i];
		}
	}
	if (!strcmp(kind, "noul")) {
		(*answer)["noul"] = round4(p[1]);
	} else {
		(*answer)["probabilities"] = probs;
		(*answer)["confidence"] = round4(confidence);
		if (!strcmp(kind, "choice"))
			(*answer)["choice"] = keys[best];
		else {
			(*answer)["score"] = round4(score);
			(*answer)["legend"] = legend;
		}
	}
	return 0;
}
