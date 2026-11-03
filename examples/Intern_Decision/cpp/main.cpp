/* Intern-Decision: build requests on CPU; encode images and run decisions on RK182X. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <limits.h>
#include <stdint.h>
#include <errno.h>
#include <string>
#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "rknn3_api.h"
#include "float16.h"
#include "Tokenizer.h"
#include "image_utils.h"
#include "file_utils.h"
#include "nlohmann/json.hpp"

/* Keep the existing C++ JSON/tokenizer interfaces; manage resources explicitly. */
using json = nlohmann::ordered_json;
static const char symbols[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
/* Official probability calibration for the Intern-Decision-0.8B checkpoint. */
static const float default_temperature = 2.747760550703f;
static const char system_prompt[] =
	"You are a careful decision assistant. Use the state and decision schema in "
	"the user message to make the requested decisions. For every field, choose "
	"exactly one answer symbol (e.g. A, B, C, ...) from its listed options and "
	"return one valid JSON object mapping each field name to its chosen symbol. "
	"Use the field names and symbols exactly as given. Do not include "
	"explanations, Markdown, or extra text.";

struct model_context {
	rknn3_context handle;
	rknn3_tensor_attr input_attr, output_attr;
	rknn3_tensor input, output;
};

/* Keep image metadata after releasing the vision model between stages. */
struct vision_info {
	uint32_t width, height;
	uint32_t tokens;
	size_t embedding_elements;
};

struct decision_context {
	struct model_context llm, vision;
	rknn3_session *session;
	Tokenizer *tokenizer;
	float16 *embedding;
	size_t embedding_size;
	uint32_t embedding_dim, context_length;
	int32_t decision_token;
	struct vision_info image;
	int embedding_fd;
	float temperature;
	float logits[sizeof(symbols) - 1];
	bool received;
	bool callback_error;
};

struct model_files {
	char llm[PATH_MAX];
	char vision[PATH_MAX];
	char tokenizer[PATH_MAX];
	char embedding[PATH_MAX];
};

static bool path_has_suffix(const char *name, const char *suffix)
{
	size_t length = strlen(name);
	size_t suffix_length = strlen(suffix);

	return length >= suffix_length &&
		strcmp(name + length - suffix_length, suffix) == 0;
}

/* Check path truncation to avoid loading a different file with the same prefix. */
static int path_join(char *path, size_t capacity, const char *base,
		     const char *suffix)
{
	int length = snprintf(path, capacity, "%s%s", base, suffix);

	return length >= 0 && (size_t)length < capacity ? 0 : -1;
}

static int file_is_readable(const char *path)
{
	struct stat info;

	return stat(path, &info) == 0 && S_ISREG(info.st_mode) &&
		access(path, R_OK) == 0;
}

/*
 * The tokenizer basename identifies the text model; the other .rknn is vision.
 * Two passes avoid relying on readdir order or allocating string arrays.
 */
static int intern_model_files_find(const char *directory, bool need_vision,
			   struct model_files *files)
{
	DIR *dir = opendir(directory);
	struct dirent *entry;
	char prefix[PATH_MAX], path[PATH_MAX];
	unsigned int tokenizers = 0, embeddings = 0, models = 0;
	bool text_found = false;
	int ret = -1;

	if (!dir) {
		fprintf(stderr, "Cannot open model directory: %s\n", directory);
		return -1;
	}
	if (path_join(prefix, sizeof(prefix), directory, "/") < 0)
		goto out;
	while ((entry = readdir(dir)) != NULL) {
		if (path_join(path, sizeof(path), prefix, entry->d_name) < 0)
			goto out;
		if (!file_is_readable(path))
			continue;
		if (path_has_suffix(path, ".tokenizer.gguf")) {
			tokenizers++;
			strcpy(files->tokenizer, path);
		} else if (path_has_suffix(path, ".embed.bin")) {
			embeddings++;
			strcpy(files->embedding, path);
		}
	}
	if (tokenizers != 1 || embeddings != 1) {
		fprintf(stderr, "Expected exactly one .tokenizer.gguf and one .embed.bin in %s\n",
			directory);
		goto out;
	}
	strcpy(files->llm, files->tokenizer);
	files->llm[strlen(files->llm) - strlen(".tokenizer.gguf")] = '\0';
	rewinddir(dir);
	while ((entry = readdir(dir)) != NULL) {
		if (path_join(path, sizeof(path), prefix, entry->d_name) < 0)
			goto out;
		if (!file_is_readable(path) || !path_has_suffix(path, ".rknn"))
			continue;
		models++;
		path[strlen(path) - strlen(".rknn")] = '\0';
		if (!strcmp(path, files->llm))
			text_found = true;
		else if (need_vision)
			strcpy(files->vision, path);
	}
	if (!text_found || (need_vision && models != 2)) {
		fprintf(stderr, "Missing text model or ambiguous vision model in %s\n", directory);
		goto out;
	}
	for (int i = 0; i < (need_vision ? 2 : 1); i++) {
		const char *base = i == 0 ? files->llm : files->vision;

		if (path_join(path, sizeof(path), base, ".weight") < 0 || !file_is_readable(path)) {
			fprintf(stderr, "Missing or unreadable model weight for: %s\n", base);
			goto out;
		}
	}
	ret = 0;
out:
	closedir(dir);
	return ret;
}

static int check_rknn_result(int ret, const char *operation)
{
	if (ret < 0)
		fprintf(stderr, "%s failed: %d\n", operation, ret);
	return ret;
}

/* Read through the shared C API; preserve JSON order for stable option symbols. */
static int json_load(const char *path, json *value)
{
	char *data = NULL;
	int size = read_data_from_file(path, &data);

	if (size <= 0) {
		free(data);
		return -1;
	}
	*value = json::parse(data, data + size, NULL, false);
	free(data);
	return value->is_discarded() ? -1 : 0;
}

/* The shared writer ignores short writes; propagate write failures here. */
static int json_save(const char *path, const json *value)
{
	std::string body = value->dump(2) + "\n";
	FILE *fp = fopen(path, "wb");
	int ret = 0;

	if (!fp)
		return -1;
	if (fwrite(body.c_str(), 1, body.size(), fp) != body.size())
		ret = -1;
	if (fclose(fp) != 0)
		ret = -1;
	return ret;
}

/* Use all compiled cores; the caller releases the context on failure. */
static int intern_model_init(struct model_context *ctx, const char *path)
{
	rknn3_config config = {};
	char model_path[PATH_MAX], weight_path[PATH_MAX];
	uint32_t cores = 0;

	if (path_join(model_path, sizeof(model_path), path, ".rknn") < 0 ||
	    path_join(weight_path, sizeof(weight_path), path, ".weight") < 0)
		return -1;
	int ret = rknn3_init(&ctx->handle, NULL);
	if (check_rknn_result(ret, "rknn3_init") < 0)
		return ret;
	ret = rknn3_load_model_from_path(ctx->handle, model_path, weight_path);
	if (check_rknn_result(ret, "load model") < 0)
		return ret;
	ret = rknn3_query(ctx->handle, RKNN3_QUERY_CORE_NUMBER, &cores, sizeof(cores));
	if (check_rknn_result(ret, "query cores") < 0 || cores == 0 || cores > 8)
		return -1;
	config.run_core_mask = (1U << cores) - 1;
	fprintf(stderr, "%s: %u cores, mask=0x%x\n", path, cores, config.run_core_mask);
	return check_rknn_result(rknn3_model_init(ctx->handle, &config), "model init");
}

/* Query tensor attributes and allocate cacheable memory as required by Runtime. */
static int intern_tensor_init(struct model_context *ctx, bool input)
{
	rknn3_tensor_attr *attr = input ? &ctx->input_attr : &ctx->output_attr;
	rknn3_tensor *tensor = input ? &ctx->input : &ctx->output;
	attr->index = 0;
	int ret = rknn3_query(ctx->handle, input ? RKNN3_QUERY_INPUT_ATTR :
			     RKNN3_QUERY_OUTPUT_ATTR, attr, sizeof(*attr));
	if (check_rknn_result(ret, "query tensor") < 0)
		return ret;
	tensor->attr = attr;
	tensor->mem = rknn3_create_mem(ctx->handle, attr->aligned_size, attr->core_id,
				    RKNN3_FLAG_MEMORY_CACHEABLE);
	fprintf(stderr, "tensor %s: elements=%u type=%d bytes=%llu\n", attr->name,
		attr->n_elems, attr->dtype, (unsigned long long)attr->aligned_size);
	return tensor->mem ? 0 : -1;
}

/* Release tensors before the model; allow partially initialized contexts. */
static void intern_model_release(struct model_context *ctx)
{
	if (ctx->input.mem)
		check_rknn_result(rknn3_destroy_mem(ctx->handle, ctx->input.mem), "destroy input");
	if (ctx->output.mem)
		check_rknn_result(rknn3_destroy_mem(ctx->handle, ctx->output.mem), "destroy output");
	if (ctx->handle)
		check_rknn_result(rknn3_destroy(ctx->handle), "destroy model");
	memset(ctx, 0, sizeof(*ctx));
}

static void intern_decision_release(struct decision_context *ctx)
{
	if (ctx->session)
		check_rknn_result(rknn3_session_destroy(ctx->session), "destroy session");
	intern_model_release(&ctx->llm);
	intern_model_release(&ctx->vision);
	delete ctx->tokenizer;
	if (ctx->embedding && ctx->embedding != MAP_FAILED)
		munmap(ctx->embedding, ctx->embedding_size);
	if (ctx->embedding_fd >= 0)
		close(ctx->embedding_fd);
}

/* Copy FP16 vectors by token ID from the read-only mapped embedding table. */
static int intern_embed_callback(void *userdata, int32_t *tokens, uint64_t count,
			  void *embed, uint64_t size)
{
	struct decision_context *ctx = (struct decision_context *)userdata;
	size_t row_size = (size_t)ctx->embedding_dim * sizeof(float16);

	if (count > SIZE_MAX / row_size || size != count * row_size)
		return -1;
	for (uint64_t i = 0; i < count; i++) {
		if (tokens[i] < 0 || (uint64_t)tokens[i] >= ctx->embedding_size / row_size)
			return -1;
		memcpy((char *)embed + i * row_size,
		       (char *)ctx->embedding + (size_t)tokens[i] * row_size, row_size);
	}
	return 0;
}

static int intern_tokenize_callback(void *userdata, const char *text, int32_t length,
			     int32_t *tokens, int32_t capacity)
{
	return ((Tokenizer *)userdata)->Tokenize(text, length, tokens, capacity);
}

/* Accept the 62 candidate-symbol logits only after prefill completes. */
static int intern_output_callback(void *userdata, rknn3_tensor *outputs, uint32_t count,
			   LLMOutputCallbackState state)
{
	struct decision_context *ctx = (struct decision_context *)userdata;
	if (state != RKLLM_OUTPUT_CALLBACK_PREFILL_FINISHED)
		return 0;
	if (count != 1 || outputs[0].attr->n_elems != sizeof(symbols) - 1) {
		ctx->callback_error = true;
		return -1;
	}
	for (int i = 0; i < (int)sizeof(symbols) - 1; i++) {
		if (outputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16)
			ctx->logits[i] = fp16_to_fp32(((float16 *)outputs[0].mem->virt_addr)[i]);
		else if (outputs[0].attr->dtype == RKNN3_TENSOR_FLOAT32)
			ctx->logits[i] = ((float *)outputs[0].mem->virt_addr)[i];
		else {
			ctx->callback_error = true;
			return -1;
		}
		if (!isfinite(ctx->logits[i])) {
			ctx->callback_error = true;
			return -1;
		}
	}
	ctx->received = true;
	return 0;
}

static int intern_result_callback(void *userdata, RKLLMResult *result, LLMCallState state)
{
	(void)result;
	if (state == RKLLM_RUN_ERROR)
		((struct decision_context *)userdata)->callback_error = true;
	return 0;
}

/* Use the existing C++ tokenizer and mmap the embedding table for demand paging. */
static int intern_decision_init(struct decision_context *ctx, const struct model_files *files)
{
	struct stat statbuf = {};
	VocabInfo vocab = {};
	rknn3_llm_config config = {};
	int ret;
	ctx->tokenizer = new Tokenizer(TOKENIZER_BACKEND_LLAMA,
				      files->tokenizer);
	if (!ctx->tokenizer->GetVocabInfo(&vocab))
		return -1;
	if (intern_model_init(&ctx->llm, files->llm) < 0 ||
	    intern_tensor_init(&ctx->llm, false) < 0)
		return -1;
	ret = rknn3_query(ctx->llm.handle, RKNN3_QUERY_LLM_CONFIG, &config, sizeof(config));
	if (check_rknn_result(ret, "query LLM config") < 0 || !config.embedding_dim ||
	    !config.max_ctx_len || config.max_ctx_len > INT_MAX || vocab.vocab_size <= 0)
		return -1;
	ctx->embedding_dim = config.embedding_dim;
	ctx->context_length = config.max_ctx_len;
	if ((size_t)ctx->embedding_dim > SIZE_MAX / sizeof(float16) / vocab.vocab_size)
		return -1;
	ctx->embedding_fd = open(files->embedding, O_RDONLY);
	if (ctx->embedding_fd < 0 || fstat(ctx->embedding_fd, &statbuf) || statbuf.st_size <= 0)
		return -1;
	ctx->embedding_size = statbuf.st_size;
	if (ctx->embedding_size != (size_t)vocab.vocab_size * ctx->embedding_dim * sizeof(float16)) {
		fprintf(stderr, "Embedding table does not match the model and tokenizer\n");
		return -1;
	}
	ctx->embedding = (float16 *)mmap(NULL, ctx->embedding_size, PROT_READ,
				       MAP_PRIVATE, ctx->embedding_fd, 0);
	if (ctx->embedding == MAP_FAILED)
		return -1;
	/* Resolve the marker from the shipped tokenizer instead of a fixed token ID. */
	if (ctx->tokenizer->Tokenize("<decision>", strlen("<decision>"), &ctx->decision_token, 1) != 1)
		return -1;
	fprintf(stderr, "LLM: embedding_dim=%u context=%u decision_token=%d\n",
		ctx->embedding_dim, ctx->context_length, ctx->decision_token);
	return ctx->llm.output_attr.n_elems == sizeof(symbols) - 1 ? 0 : -1;
}

/*
 * Use a fresh session per field to isolate KV, linear-attention and conv state.
 * Run only the prefix before its <decision> token to get logits, not text.
 */
static int intern_decision_run_field(struct decision_context *ctx, int32_t *tokens, int count,
		      float16 *images, int image_count)
{
	rknn3_llm_param params = {};
	RKLLMCallback callbacks = {};
	rknn3_llm_input input = {};
	rknn3_llm_infer_param infer = {};
	VocabInfo vocab = {};
	if (!ctx->tokenizer->GetVocabInfo(&vocab))
		return -1;
	params.logits_name = (char *)"output";
	params.max_context_len = ctx->context_length;
	params.vocab_info.vocab_size = vocab.vocab_size;
	params.vocab_info.n_special_eos_id = vocab.n_special_eos_id;
	params.vocab_info.n_special_bos_id = vocab.n_special_bos_id;
	memcpy(params.vocab_info.special_eos_id, vocab.special_eos_id, sizeof(vocab.special_eos_id));
	memcpy(params.vocab_info.special_bos_id, vocab.special_bos_id, sizeof(vocab.special_bos_id));
	/* Sampling is disabled; checkpoint calibration is applied on CPU after prefill. */
	params.sampling_param.temperature = 1;
	params.sampling_param.repeat_penalty = 1;
	params.sampling_param.top_k = 1;
	params.sampling_param.top_p = 1;
	callbacks.embed_callback = intern_embed_callback;
	callbacks.embed_userdata = ctx;
	callbacks.tokenizer_callback = intern_tokenize_callback;
	callbacks.tokenizer_userdata = ctx->tokenizer;
	callbacks.output_callback = intern_output_callback;
	callbacks.output_userdata = ctx;
	callbacks.output_tensors = &ctx->llm.output;
	callbacks.n_output_tensors = 1;
	callbacks.result_callback = intern_result_callback;
	callbacks.result_userdata = ctx;
	ctx->received = false;
	ctx->callback_error = false;
	ctx->session = rknn3_session_init(ctx->llm.handle, &params, 1);
	if (!ctx->session)
		return -1;
	int ret = rknn3_session_set_callback(ctx->session, &callbacks);
	if (check_rknn_result(ret, "set callback") < 0)
		return ret;
	if (image_count) {
		input.input_type = RKNN3_LLM_INPUT_MULTIMODAL;
		input.multimodal_input.name = "input_embeds";
		input.multimodal_input.tokens = tokens;
		input.multimodal_input.n_tokens = count;
		input.multimodal_input.image.image_embed = images;
		input.multimodal_input.image.n_image = image_count;
		input.multimodal_input.image.n_image_tokens = ctx->image.tokens;
		input.multimodal_input.image.image_width = ctx->image.width;
		input.multimodal_input.image.image_height = ctx->image.height;
		input.multimodal_input.image.image_start = "<|vision_start|>";
		input.multimodal_input.image.image_end = "<|vision_end|>";
		input.multimodal_input.image.image_content = "<|image_pad|>";
	} else {
		input.input_type = RKNN3_LLM_INPUT_TOKEN;
		input.llm_input.name = "input";
		input.llm_input.tokens = tokens;
		input.llm_input.n_tokens = count;
	}
	infer.keep_history = 0;
	infer.max_new_tokens = 1;
	infer.prefill_only = true;
	infer.disable_sampling = true;
	ret = rknn3_session_run(ctx->session, &input, 1, &infer);
	int destroy_ret = rknn3_session_destroy(ctx->session);
	ctx->session = NULL;
	if (check_rknn_result(ret, "session run") < 0 || check_rknn_result(destroy_ret, "destroy session") < 0 ||
	    !ctx->received || ctx->callback_error)
		return -1;
	return 0;
}

/* Normalize choice/score/noul options, preserving their order for A, B, C, etc. */
static int intern_options_parse(const json &q, json *choices)
{
	if (!q.is_object() || !q.contains("type") || !q["type"].is_string())
		return -1;
	std::string type = q["type"];
	json criteria = q.value("criteria", json());
	*choices = json::object();
	if (type == "noul") {
		(*choices)["no"] = "The answer is no (negative, or disagree with the claim).";
		(*choices)["yes"] = "The answer is yes (affirmative, or align with the claim).";
		bool no_set = false, yes_set = false;
		/* Defaults apply only when criteria is omitted, not when it is malformed. */
		if (q.contains("criteria")) {
			if (!criteria.is_object() || criteria.size() != 2)
				return -1;
			for (auto it = criteria.begin(); it != criteria.end(); ++it) {
				if (!it.value().is_string())
					return -1;
				std::string key = it.key();
				if (!no_set && (!strcasecmp(key.c_str(), "no") || !strcasecmp(key.c_str(), "false") || key == "0")) {
					(*choices)["no"] = it.value();
					no_set = true;
				} else if (!yes_set && (!strcasecmp(key.c_str(), "yes") || !strcasecmp(key.c_str(), "true") || key == "1")) {
					(*choices)["yes"] = it.value();
					yes_set = true;
				} else {
					return -1;
				}
			}
			if (!no_set || !yes_set)
				return -1;
		}
	} else if ((type == "choice" || type == "score") && criteria.is_object()) {
		*choices = criteria;
	} else if (type == "score" && criteria.is_array()) {
		for (size_t i = 0; i < criteria.size(); i++)
			(*choices)[std::to_string(i)] = criteria[i];
	} else {
		return -1;
	}
	for (auto it = choices->begin(); it != choices->end(); ++it) {
		if (!it.value().is_string())
			return -1;
		if (type == "score") {
			char *end = NULL;
			const char *start = it.key().c_str();
			errno = 0;
			float value = strtof(start, &end);
			if (end == start || *end || errno == ERANGE || !isfinite(value))
				return -1;
		}
	}
	return !choices->empty() && choices->size() <= sizeof(symbols) - 1 ? 0 : -1;
}

/* Use C++ JSON/string APIs here to preserve the upstream template and field order. */
static int intern_prompt_build(const json &row, uint32_t image_tokens,
			       std::string *prompt, json *all_options)
{
	if (!row.is_object() || !row.contains("questions") || !row["questions"].is_object() ||
	    row["questions"].empty() || row["questions"].size() > 16)
		return -1;
	json skeleton = json::object();
	std::string user = "Return one answer for every field using the supplied answer symbols.\n\n## State\n";
	user += row.value("state", json()).dump(2) + "\n## Decision schema\n";
	*all_options = json::object();
	bool first = true;
	for (auto it = row["questions"].begin(); it != row["questions"].end(); ++it) {
		json choices;
		if (intern_options_parse(it.value(), &choices) < 0 ||
		    (it.value().contains("instructions") && !it.value()["instructions"].is_string()))
			return -1;
		if (!first)
			user += "\n";
		first = false;
		user += it.key() + ": " + it.value().value("instructions", std::string());
		int index = 0;
		for (auto op = choices.begin(); op != choices.end(); ++op)
			user += "\n    " + std::string(1, symbols[index++]) + " = " + op.key() +
				": " + op.value().get<std::string>();
		(*all_options)[it.key()] = choices;
		skeleton[it.key()] = "<decision>";
	}
	if (user.find("<decision>") != std::string::npos)
		return -1;
	*prompt = std::string("<|im_start|>system\n") + system_prompt +
		"<|im_end|>\n<|im_start|>user\n";
	json images = row.value("images", json::array());
	if (!images.is_array() || images.size() > 8)
		return -1;
	for (size_t i = 0; i < images.size(); i++) {
		if (!images[i].is_string())
			return -1;
		*prompt += "Picture " + std::to_string(i + 1) + ": <|vision_start|>";
		for (uint32_t j = 0; j < image_tokens; j++)
			*prompt += "<|image_pad|>";
		*prompt += "<|vision_end|>";
	}
	*prompt += user + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" +
		skeleton.dump(4) + "<|im_end|>\n";
	return 0;
}

/* Direct copies require dense element strides; reject padded layouts explicitly. */
static int intern_tensor_is_dense(const rknn3_tensor_attr *attr, size_t element_size)
{
	uint64_t elements = 1;

	if (!attr->n_dims || attr->n_dims > RKNN3_MAX_DIMS ||
	    attr->n_stride < attr->n_dims || attr->n_stride > RKNN3_MAX_STRIDE_DIMS)
		return 0;
	for (int i = (int)attr->n_dims - 1; i >= 0; i--) {
		if (!attr->shape[i] || attr->stride[attr->n_stride - attr->n_dims + i] != elements ||
		    elements > SIZE_MAX / attr->shape[i])
			return 0;
		elements *= attr->shape[i];
	}
	return elements == attr->n_elems && elements <= SIZE_MAX / element_size &&
		elements * element_size <= attr->aligned_size;
}

/* Validate the deployed interface, not a hard-coded image resolution. */
static int intern_vision_get_info(const rknn3_tensor_attr *input,
				  const rknn3_tensor_attr *output,
				  uint32_t embedding_dim, struct vision_info *info)
{
	uint64_t pixels;
	uint32_t token_axis;

	if (input->dtype != RKNN3_TENSOR_UINT8 || input->layout != RKNN3_TENSOR_NHWC ||
	    input->n_dims != 4 || input->shape[0] != 1 || input->shape[3] != 3 ||
	    !intern_tensor_is_dense(input, 1) || output->dtype != RKNN3_TENSOR_FLOAT16 ||
	    !intern_tensor_is_dense(output, sizeof(float16)))
		return -1;
	/* Accept [tokens, hidden] or [1, tokens, hidden], with identical semantics. */
	if (output->n_dims == 2)
		token_axis = 0;
	else if (output->n_dims == 3 && output->shape[0] == 1)
		token_axis = 1;
	else
		return -1;
	if (!embedding_dim || output->shape[token_axis + 1] != embedding_dim ||
	    output->shape[token_axis] > INT_MAX)
		return -1;
	pixels = (uint64_t)input->shape[1] * input->shape[2];
	/* image_buffer_t stores dimensions and byte count as signed integers. */
	if (pixels > INT_MAX / 3)
		return -1;
	info->height = input->shape[1];
	info->width = input->shape[2];
	info->tokens = output->shape[token_axis];
	info->embedding_elements = output->n_elems;
	return 0;
}

static int intern_vision_init(struct decision_context *ctx, const char *path)
{
	rknn3_input_output_num count = {};
	int ret;

	if (intern_model_init(&ctx->vision, path) < 0)
		return -1;
	ret = rknn3_query(ctx->vision.handle, RKNN3_QUERY_IN_OUT_NUM, &count, sizeof(count));
	if (check_rknn_result(ret, "query vision IO count") < 0 ||
	    count.n_input != 1 || count.n_output != 1 ||
	    intern_tensor_init(&ctx->vision, true) < 0 || intern_tensor_init(&ctx->vision, false) < 0)
		return -1;
	if (intern_vision_get_info(&ctx->vision.input_attr, &ctx->vision.output_attr,
				   ctx->embedding_dim, &ctx->image) < 0) {
		fprintf(stderr, "Vision requires dense uint8 NHWC RGB and matching FP16 embeddings\n");
		return -1;
	}
	fprintf(stderr, "Vision: width=%u height=%u tokens=%u embedding_dim=%u\n",
		ctx->image.width, ctx->image.height, ctx->image.tokens, ctx->embedding_dim);
	return 0;
}

/* Resize RGB to the queried shape; normalization is fused into the model. */
static int intern_vision_run(struct model_context *ctx, const char *path, float16 *embed)
{
	image_buffer_t image = {}, resized = {};
	int ret = read_image(path, &image);
	if (ret < 0)
		return ret;
	resized.width = ctx->input_attr.shape[2];
	resized.height = ctx->input_attr.shape[1];
	resized.format = IMAGE_FORMAT_RGB888;
	resized.size = resized.width * resized.height * 3;
	resized.virt_addr = (unsigned char *)malloc(resized.size);
	if (!resized.virt_addr) {
		free(image.virt_addr);
		return -1;
	}
	ret = convert_image(&image, &resized, NULL, NULL, 0);
	if (ret == 0) {
		/* Copy raw RGB bytes without applying normalization again. */
		memcpy(ctx->input.mem->virt_addr, resized.virt_addr, resized.size);
		ret = rknn3_mem_sync(ctx->handle, ctx->input.mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
		if (check_rknn_result(ret, "vision input sync") == 0)
			ret = check_rknn_result(rknn3_run(ctx->handle, &ctx->input, 1, &ctx->output, 1), "vision run");
		if (ret == 0)
			ret = check_rknn_result(rknn3_mem_sync(ctx->handle, ctx->output.mem,
						   RKNN3_MEMORY_SYNC_FROM_DEVICE), "vision output sync");
		if (ret == 0)
			memcpy(embed, ctx->output.mem->virt_addr, (size_t)ctx->output_attr.n_elems * sizeof(float16));
	}
	free(image.virt_addr);
	free(resized.virt_addr);
	return ret;
}

/* Print questions and top choices; retain all probabilities and logits in JSON. */
static void intern_decision_print(const json &row, const json &all_options,
			    const json &result, const char *output_path)
{
	printf("\n输入：\n");
	json state = row.value("state", json());
	if (state.is_object()) {
		for (auto it = state.begin(); it != state.end(); ++it) {
			std::string value = it.value().is_string() ?
				it.value().get<std::string>() : it.value().dump();
			printf("  %s: %s\n", it.key().c_str(), value.c_str());
		}
	} else {
		std::string value = state.is_string() ? state.get<std::string>() : state.dump();
		printf("  %s\n", value.c_str());
	}
	for (const auto &image : row.value("images", json::array()))
		printf("  图片: %s\n", image.get<std::string>().c_str());

	for (auto it = all_options.begin(); it != all_options.end(); ++it) {
		const json &question = row["questions"][it.key()];
		const json &item = result[it.key()];
		std::string label = item["label"];
		printf("\n问题 [%s]: %s\n", it.key().c_str(),
		       question.value("instructions", std::string()).c_str());
		for (auto op = it.value().begin(); op != it.value().end(); ++op)
			printf("  %s: %s\n", op.key().c_str(),
			       op.value().get<std::string>().c_str());
		printf("结果: %s（%s），概率 %.2f%%\n", label.c_str(),
		       it.value()[label].get<std::string>().c_str(),
		       item["probabilities"][label].get<float>() * 100.0f);
		if (item.contains("score"))
			printf("评分: %.4f\n", item["score"].get<float>());
		if (item.contains("noul"))
			printf("肯定概率: %.2f%%\n", item["noul"].get<float>() * 100.0f);
	}
	if (output_path[0])
		printf("\n完整结果已保存至: %s\n", output_path);
}

/* Apply candidate softmax and upstream temperature calibration in FP32. */
static int intern_probabilities(const float *logits, int count, float temperature,
			        float *probabilities)
{
	float raw[sizeof(symbols) - 1], total = 0;
	float logs[sizeof(symbols) - 1], sum = 0, maximum_log = -INFINITY;
	float maximum;

	if (count <= 0 || count > (int)sizeof(symbols) - 1 ||
	    !isfinite(temperature) || temperature <= 0)
		return -1;
	maximum = logits[0];
	for (int i = 0; i < count; i++) {
		if (!isfinite(logits[i]))
			return -1;
		if (logits[i] > maximum)
			maximum = logits[i];
	}
	for (int i = 0; i < count; i++) {
		raw[i] = expf(logits[i] - maximum);
		total += raw[i];
	}
	for (int i = 0; i < count; i++) {
		raw[i] /= total;
		probabilities[i] = raw[i];
		/* Preserve zero probabilities from FP32 underflow, as upstream does. */
		logs[i] = raw[i] > 0 ? logf(raw[i]) : -INFINITY;
		if (logs[i] > maximum_log)
			maximum_log = logs[i];
	}
	if (temperature == 1.0f)
		return 0;
	for (int i = 0; i < count; i++) {
		probabilities[i] = expf((logs[i] - maximum_log) / temperature);
		sum += probabilities[i];
	}
	for (int i = 0; i < count; i++)
		probabilities[i] /= sum;
	return 0;
}

static int intern_decision_run(struct decision_context *ctx, const struct model_files *files,
			const json &row, const char *output_path)
{
	json all_options, result = json::object();
	json image_paths = row.value("images", json::array());
	std::string prompt;
	int32_t *tokens = NULL;
	int positions[16];
	int position_count = 0, field = 0, ret = -1, count;
	float16 *images = NULL;

	if (!image_paths.is_array() || image_paths.size() > 8)
		goto out;
	/* Query vision before constructing image placeholders or allocating features. */
	if (!image_paths.empty() && intern_vision_init(ctx, files->vision) < 0)
		goto out;
	if (intern_prompt_build(row, ctx->image.tokens, &prompt, &all_options) < 0) {
		fprintf(stderr, "Invalid decision request\n");
		goto out;
	}
	if (prompt.size() > INT_MAX || ctx->context_length > SIZE_MAX / sizeof(*tokens))
		goto out;
	tokens = (int32_t *)calloc(ctx->context_length, sizeof(*tokens));
	if (!tokens)
		goto out;
	count = ctx->tokenizer->Tokenize(prompt.c_str(), prompt.size(), tokens, ctx->context_length);
	if (count <= 0 || (uint32_t)count > ctx->context_length)
		goto out;
	/* Locate each field; exclude its <decision> token from the inference prefix. */
	for (int i = 0; i < count; i++)
		if (tokens[i] == ctx->decision_token) {
			if (position_count >= 16)
				goto out;
			positions[position_count++] = i;
		}
	if ((size_t)position_count != all_options.size()) {
		fprintf(stderr, "Decision token count mismatch\n");
		goto out;
	}
	/* Encode each image once per request and reuse its features across fields. */
	if (!image_paths.empty()) {
		if (ctx->image.embedding_elements > SIZE_MAX / sizeof(float16) / image_paths.size())
			goto out;
		images = (float16 *)calloc(image_paths.size() * ctx->image.embedding_elements, sizeof(float16));
		if (!images)
			goto out;
	}
	if (!image_paths.empty()) {
		for (size_t i = 0; i < image_paths.size(); i++) {
			std::string path = image_paths[i];
			if (path.empty())
				goto out;
			if (intern_vision_run(&ctx->vision, path.c_str(), images + i * ctx->image.embedding_elements) < 0)
				goto out;
		}
		intern_model_release(&ctx->vision);
	}
	for (auto it = all_options.begin(); it != all_options.end(); ++it, field++) {
		if (intern_decision_run_field(ctx, tokens, positions[field], images, image_paths.size()) < 0)
			goto out;
		int n = it.value().size(), best = 0;
		float calibrated[sizeof(symbols) - 1], expected = 0;
		if (intern_probabilities(ctx->logits, n, ctx->temperature, calibrated) < 0)
			goto out;
		json item = json::object(), probabilities = json::object();
		int i = 0;
		std::string kind = row["questions"][it.key()]["type"];
		for (auto op = it.value().begin(); op != it.value().end(); ++op, i++) {
			float probability = calibrated[i];
			probabilities[op.key()] = probability;
			/* Upstream resolves equal probabilities by lexical option label. */
			if (i == 0 || probability > calibrated[best] ||
			    (probability == calibrated[best] && op.key() < item["label"].get<std::string>())) {
				best = i;
				item["label"] = op.key();
			}
			if (kind == "score")
				expected += strtof(op.key().c_str(), NULL) * probability;
		}
		if (!isfinite(expected)) {
			fprintf(stderr, "Score exceeds the finite FP32 range\n");
			goto out;
		}
		item["symbol"] = std::string(1, symbols[best]);
		item["probabilities"] = probabilities;
		item["logits"] = json::array();
		for (int j = 0; j < n; j++)
			item["logits"].push_back(ctx->logits[j]);
		if (kind == "score")
			item["score"] = expected;
		if (kind == "noul")
			item["noul"] = probabilities["yes"];
		result[it.key()] = item;
	}
	{
		json output = {{"decisions", result}, {"temperature", ctx->temperature}};

		if (output_path[0] && json_save(output_path, &output) < 0) {
			fprintf(stderr, "Cannot write output file: %s\n", output_path);
			goto out;
		}
	}
	intern_decision_print(row, all_options, result, output_path);
	ret = 0;
out:
	free(tokens);
	free(images);
	intern_model_release(&ctx->vision);
	return ret;
}

static void print_usage(const char *program)
{
	fprintf(stderr,
		"Usage:\n"
		"  %s --model_path DIR --task_path JSON --text TEXT [--temperature T] [--output_path JSON]\n"
		"  %s --model_path DIR --task_path JSON --image_path IMAGE [--text TEXT] [--temperature T] [--output_path JSON]\n"
		"  --temperature: positive finite value (default: %.9g, Intern-Decision-0.8B)\n",
		program, program, default_temperature);
}

/* Load task questions separately from model resources and validate before NPU init. */
static int intern_request_build(const char *task_path,
			const char *text, const char *image_path, json *row)
{
	json config;

	if ((!text[0] && !image_path[0]) ||
	    (text[0] && strspn(text, " \t\r\n") == strlen(text))) {
		fprintf(stderr, "Provide non-empty --text or --image_path\n");
		return -1;
	}
	if (image_path[0] && !file_is_readable(image_path)) {
		fprintf(stderr, "Cannot read image: %s\n", image_path);
		return -1;
	}
	if (json_load(task_path, &config) < 0 || !config.is_object() ||
	    !config.contains("questions") || !config["questions"].is_object() ||
	    config["questions"].empty() || config["questions"].size() > 16) {
		fprintf(stderr, "Invalid task configuration: %s\n", task_path);
		return -1;
	}
	for (auto it = config["questions"].begin(); it != config["questions"].end(); ++it) {
		json choices;
		if (intern_options_parse(it.value(), &choices) < 0 ||
		    (it.value().contains("instructions") && !it.value()["instructions"].is_string())) {
			fprintf(stderr, "Invalid question '%s' in %s\n", it.key().c_str(), task_path);
			return -1;
		}
	}
	*row = json::object();
	(*row)["state"] = text[0] ? text : "请根据图片回答问题。";
	(*row)["questions"] = config["questions"];
	if (image_path[0])
		(*row)["images"] = json::array({image_path});
	return 0;
}

int main(int argc, char **argv)
{
	const char *model_path = "", *task_path = "", *text = "";
	const char *image_path = "", *output_path = "";
	const char *temperature = "";
	struct model_files files = {};
	struct decision_context ctx = {};
	json row;
	int ret;

	ctx.embedding_fd = -1;
	ctx.temperature = default_temperature;
	if (argc == 2 && !strcmp(argv[1], "--help")) {
		print_usage(argv[0]);
		return 0;
	}
	/* argv strings remain valid until main returns; no allocation or copy is needed. */
	for (int i = 1; i < argc; i += 2) {
		const char **value = NULL;

		if (!strcmp(argv[i], "--model_path"))
			value = &model_path;
		else if (!strcmp(argv[i], "--task_path"))
			value = &task_path;
		else if (!strcmp(argv[i], "--text"))
			value = &text;
		else if (!strcmp(argv[i], "--image_path"))
			value = &image_path;
		else if (!strcmp(argv[i], "--output_path"))
			value = &output_path;
		else if (!strcmp(argv[i], "--temperature"))
			value = &temperature;
		if (!value || i + 1 >= argc || (*value)[0] || !argv[i + 1][0]) {
			fprintf(stderr, "Invalid argument: %s\n", argv[i]);
			print_usage(argv[0]);
			return 1;
		}
		*value = argv[i + 1];
	}
	if (temperature[0]) {
		char *end = NULL;
		errno = 0;
		ctx.temperature = strtof(temperature, &end);
		if (end == temperature || *end || errno == ERANGE ||
		    !isfinite(ctx.temperature) || ctx.temperature <= 0) {
			fprintf(stderr, "Invalid --temperature: expected a positive finite number\n");
			return 1;
		}
	}
	if (!model_path[0] || !task_path[0]) {
		print_usage(argv[0]);
		return 1;
	}
	if (intern_request_build(task_path, text, image_path, &row) < 0)
		return 1;
	if (intern_model_files_find(model_path, image_path[0] != '\0', &files) < 0)
		return 1;
	ret = intern_decision_init(&ctx, &files);
	if (ret == 0)
		ret = intern_decision_run(&ctx, &files, row, output_path);
	/* Use the same cleanup path for partial init/run failures. */
	intern_decision_release(&ctx);
	return ret == 0 ? 0 : 1;
}
