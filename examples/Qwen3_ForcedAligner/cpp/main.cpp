#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include "audio.h"
#include "text.h"
#include "resource.h"
#include "rknn3_api.h"
#include "timestamp.h"

struct network {
	rknn3_context context;
	unsigned n_input;
	rknn3_tensor_attr input_attrs[MODEL_MAX_INPUTS];
	rknn3_tensor_attr output_attr;
	rknn3_tensor inputs[MODEL_MAX_INPUTS];
	rknn3_tensor output;
};

static double now_ms(void)
{
	struct timespec time;
	clock_gettime(CLOCK_MONOTONIC, &time);
	return time.tv_sec * 1000.0 + time.tv_nsec / 1000000.0;
}

static int api_status(int status, const char *operation)
{
	if (status != 0)
		fprintf(stderr, "%s failed: %d\n", operation, status);
	return status;
}

static int release_network(struct network *network)
{
	if (!network)
		return 0;
	/* The SDK does not document context destruction after a memory-release error. */
	for (unsigned i = 0; i < network->n_input; ++i) {
		if (!network->inputs[i].mem)
			continue;
		int status = rknn3_destroy_mem(network->context, network->inputs[i].mem);
		if (status != 0) {
			fprintf(stderr, "Destroy input %u (%s) failed: %d; context retained\n", i,
				network->input_attrs[i].name, status);
			return -1;
		}
		network->inputs[i].mem = NULL;
	}
	if (network->output.mem) {
		if (api_status(rknn3_destroy_mem(network->context, network->output.mem),
			       "destroy output memory; context retained on error") != 0)
			return -1;
		network->output.mem = NULL;
	}
	if (network->context && api_status(rknn3_destroy(network->context), "destroy context") != 0)
		return -1;
	memset(network, 0, sizeof(*network));
	return 0;
}

/* Select by actual tensor capacity, not by assumed shape ID order. */
static int select_shape(struct network *network, const char *name, unsigned required)
{
	rknn3_shape_config config = {0};
	rknn3_shape_info *shapes = NULL;
	rknn3_tensor_attr(*inputs)[MODEL_MAX_INPUTS] = NULL, *outputs = NULL;
	unsigned best_capacity = ~0U;
	int best = -1, result = -1;
	int axis = strcmp(name, "encoder") == 0 ? 0 : 1;

	if (api_status(rknn3_query(network->context, RKNN3_QUERY_DYNAMIC_SHAPE_CONFIG, &config, sizeof(config)),
		       "query shapes") != 0)
		return -1;
	if (!config.n_shapes || config.n_shapes > MODEL_MAX_SHAPES)
		return -1;
	shapes = (rknn3_shape_info *)calloc(config.n_shapes, sizeof(*shapes));
	inputs = (rknn3_tensor_attr(*)[MODEL_MAX_INPUTS])calloc(config.n_shapes, sizeof(*inputs));
	outputs = (rknn3_tensor_attr *)calloc(config.n_shapes, sizeof(*outputs));
	if (!shapes || !inputs || !outputs)
		goto cleanup;
	for (unsigned i = 0; i < config.n_shapes; ++i) {
		shapes[i].shape_id = i;
		shapes[i].input_attrs = inputs[i];
		shapes[i].output_attrs = &outputs[i];
	}
	if (api_status(rknn3_query(network->context, RKNN3_QUERY_DYNAMIC_SHAPE_INFO, shapes,
				   sizeof(shapes[0]) * config.n_shapes),
		       "query shape IO") != 0)
		goto cleanup;
	for (unsigned i = 0; i < config.n_shapes; ++i) {
		if (shapes[i].n_inputs != network->n_input || shapes[i].n_outputs != 1 ||
		    inputs[i][0].n_dims <= (unsigned)axis)
			goto cleanup;
		unsigned capacity = inputs[i][0].shape[axis];
		if (capacity >= required && capacity < best_capacity) {
			best = i;
			best_capacity = capacity;
		}
	}
	if (best < 0) {
		fprintf(stderr, "No %s bucket can hold %u positions\n", name, required);
		goto cleanup;
	}
	if (api_status(rknn3_set_shape(network->context, shapes[best].shape_id), "set shape") != 0)
		goto cleanup;
	memcpy(network->input_attrs, inputs[best], sizeof(network->input_attrs));
	network->output_attr = outputs[best];
	printf("%s bucket: %u, shape ID: %d\n", name, best_capacity, shapes[best].shape_id);
	result = 0;
cleanup:
	free(outputs);
	free(inputs);
	free(shapes);
	return result;
}

static int load_network(struct network *network, const char *root, const char *name)
{
	char *model = (char *)malloc(4096), *weight = (char *)malloc(4096);
	char suffix[64];
	struct stat status;
	int result = -1;

	if (!model || !weight)
		goto cleanup;
	if (snprintf(suffix, sizeof(suffix), "%s.rknn", name) >= (int)sizeof(suffix) ||
	    find_resource(root, suffix, model, 4096) != 0)
		goto cleanup;
	if (snprintf(weight, 4096, "%.*s.weight", (int)strlen(model) - 5, model) >= 4096)
		goto cleanup;
	if (stat(weight, &status) != 0 || !S_ISREG(status.st_mode)) {
		fprintf(stderr, "Missing matching model weight: %s\n", weight);
		goto cleanup;
	}
	if (api_status(rknn3_init(&network->context, NULL), "init") != 0 ||
	    api_status(rknn3_load_model_from_path(network->context, model, weight), "load model") != 0)
		goto cleanup;
	result = 0;
cleanup:
	free(weight);
	free(model);
	return result;
}

static int init_network(struct network *network, const char *root, const char *name, unsigned expected_inputs,
			unsigned required)
{
	uint32_t cores = 0;
	rknn3_config config = {0};
	rknn3_input_output_num count = {0};
	if (load_network(network, root, name) != 0 ||
	    api_status(rknn3_query(network->context, RKNN3_QUERY_CORE_NUMBER, &cores, sizeof(cores)), "query cores") !=
		0)
		return -1;
	if (cores == 0 || cores > MODEL_MAX_CORES) {
		fprintf(stderr, "Unsupported model core count: %u\n", cores);
		return -1;
	}
	config.run_core_mask = (1U << cores) - 1;
	if (api_status(rknn3_model_init(network->context, &config), "model init") != 0 ||
	    api_status(rknn3_query(network->context, RKNN3_QUERY_IN_OUT_NUM, &count, sizeof(count)), "query IO") != 0) {
		return -1;
	}
	if ((count.n_input != expected_inputs && !(expected_inputs == 2 && count.n_input == 1)) ||
	    count.n_output != 1) {
		fprintf(stderr, "Unexpected IO count for %s\n", name);
		return -1;
	}
	network->n_input = count.n_input;
	if (select_shape(network, name, required) != 0)
		return -1;
	for (unsigned i = 0; i < count.n_input; ++i) {
		network->inputs[i].attr = &network->input_attrs[i];
		network->inputs[i].mem = rknn3_create_mem(network->context, network->input_attrs[i].aligned_size,
							  network->input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
		if (!network->inputs[i].mem)
			return -1;
	}

	network->output.attr = &network->output_attr;
	network->output.mem = rknn3_create_mem(network->context, network->output_attr.aligned_size,
					       network->output_attr.core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
	return network->output.mem ? 0 : -1;
}

static size_t tensor_offset(const rknn3_tensor_attr *attr, size_t index)
{
	size_t offset = 0;
	for (int i = attr->n_dims - 1; i >= 0; --i) {
		offset += (index % attr->shape[i]) * attr->stride[i];
		index /= attr->shape[i];
	}
	return offset;
}

static int check_tensor(const rknn3_tensor_attr *attr, const char *name, rknn3_tensor_type type, unsigned rank,
			const unsigned *shape)
{
	size_t max_offset = 0;
	if (strcmp(attr->name, name) != 0 || attr->dtype != type || attr->n_dims != rank || attr->n_stride < rank ||
	    (attr->layout != RKNN3_TENSOR_NCHW && attr->layout != RKNN3_TENSOR_UNDEFINED)) {
		fprintf(stderr, "Unexpected tensor interface: %s\n", attr->name);
		return -1;
	}
	for (unsigned i = 0; i < rank; ++i) {
		if (attr->shape[i] != shape[i]) {
			fprintf(stderr, "Unexpected shape: %s dimension %u\n", name, i);
			return -1;
		}
		max_offset += (shape[i] - 1) * attr->stride[i];
	}
	size_t bytes = type == RKNN3_TENSOR_FLOAT16 ? 2 : 4;
	return (max_offset + 1) * bytes <= attr->aligned_size ? 0 : -1;
}

static int check_interfaces(struct network *encoder, struct network *decoder)
{
	unsigned chunks = encoder->input_attrs[0].shape[0];
	unsigned length = decoder->input_attrs[0].shape[1];
	bool audio_supported = false, sequence_supported = false;
	for (unsigned i = 0; i < sizeof(model_audio_buckets) / sizeof(model_audio_buckets[0]); ++i)
		audio_supported |= chunks == model_audio_buckets[i];
	for (unsigned i = 0; i < sizeof(model_sequence_buckets) / sizeof(model_sequence_buckets[0]); ++i)
		sequence_supported |= length == model_sequence_buckets[i];
	if (!audio_supported || !sequence_supported)
		return -1;
	const unsigned mel[] = {chunks, 1, MODEL_MEL_BINS, MODEL_CHUNK_FRAMES},
		       audio[] = {chunks * MODEL_CHUNK_TOKENS, MODEL_HIDDEN_SIZE};
	const unsigned audio_mask[] = {1, 1, 1, chunks * MODEL_CHUNK_TOKENS};
	if (encoder->n_input == 2 &&
	    check_tensor(&encoder->input_attrs[1], "audio_mask", RKNN3_TENSOR_FLOAT32, 4, audio_mask) != 0)
		return -1;
	const unsigned embed[] = {1, length, MODEL_HIDDEN_SIZE}, mask[] = {1, length};
	const unsigned pos[] = {1, 1, length}, logits[] = {1, length, MODEL_TIMESTAMP_CLASSES};
	const unsigned indices[] = {1, length}, padded_indices[] = {1, 1, length};
	int output_status;

	if (decoder->output_attr.dtype == RKNN3_TENSOR_INT32) {
		unsigned rank = decoder->output_attr.n_dims;

		output_status = check_tensor(&decoder->output_attr, "timestamp_indices", RKNN3_TENSOR_INT32,
					     rank == 2 ? 2 : 3, rank == 2 ? indices : padded_indices);
	} else {
		output_status = check_tensor(&decoder->output_attr, "output", RKNN3_TENSOR_FLOAT16, 3, logits);
	}
	return check_tensor(&encoder->input_attrs[0], "x", RKNN3_TENSOR_FLOAT32, 4, mel) ||
	       check_tensor(&encoder->output_attr, "h", RKNN3_TENSOR_FLOAT16, 2, audio) ||
	       check_tensor(&decoder->input_attrs[0], "input_embeds", RKNN3_TENSOR_FLOAT16, 3, embed) ||
	       check_tensor(&decoder->input_attrs[1], "attention_mask", RKNN3_TENSOR_FLOAT16, 2, mask) ||
	       check_tensor(&decoder->input_attrs[2], "position_ids", RKNN3_TENSOR_INT32, 3, pos) || output_status;
}

/* Text embeddings are model resources; masks and positions are made per call. */
static int prepare_decoder(struct network *decoder, const char *model_path, const struct request *request)
{
	char path[4096];
	struct stat status;
	float16 row[MODEL_HIDDEN_SIZE];
	FILE *file;
	int result = -1;

	if (find_resource(model_path, ".embed.bin", path, sizeof(path)) != 0)
		return -1;
	file = fopen(path, "rb");
	if (!file)
		return -1;
	if (fstat(fileno(file), &status) != 0 ||
	    status.st_size != (off_t)request->vocab_size * MODEL_HIDDEN_SIZE * sizeof(float16))
		goto cleanup;
	for (unsigned i = 0; i < decoder->n_input; i++)
		memset(decoder->inputs[i].mem->virt_addr, 0, decoder->inputs[i].mem->size);
	for (int i = 0; i < request->sequence; i++) {
		float16 *embeds = (float16 *)decoder->inputs[0].mem->virt_addr;
		float16 *mask = (float16 *)decoder->inputs[1].mem->virt_addr;

		if (fseeko(file, (off_t)request->tokens[i] * sizeof(row), SEEK_SET) != 0 ||
		    fread(row, sizeof(row), 1, file) != 1)
			goto cleanup;
		for (int j = 0; j < MODEL_HIDDEN_SIZE; j++)
			embeds[tensor_offset(&decoder->input_attrs[0], i * MODEL_HIDDEN_SIZE + j)] = row[j];
		mask[tensor_offset(&decoder->input_attrs[1], i)] = fp32_to_fp16(1);
	}
	for (unsigned i = 0; i < decoder->input_attrs[0].shape[1]; i++) {
		int32_t *positions = (int32_t *)decoder->inputs[2].mem->virt_addr;

		positions[tensor_offset(&decoder->input_attrs[2], i)] = i;
	}
	result = 0;
cleanup:
	fclose(file);
	return result;
}

static int sync_inputs(struct network *network)
{
	for (unsigned i = 0; i < network->n_input; ++i) {
		if (api_status(rknn3_mem_sync(network->context, network->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE),
			       "sync input") != 0) {
			return -1;
		}
	}
	return 0;
}

static int run_network(struct network *network)
{
	return api_status(rknn3_run(network->context, network->inputs, network->n_input, &network->output, 1), "run");
}

static int sync_output(struct network *network)
{
	return api_status(rknn3_mem_sync(network->context, network->output.mem, RKNN3_MEMORY_SYNC_FROM_DEVICE),
			  "sync output");
}

static int prepare_encoder(struct network *encoder, const struct request *request, audio_buffer_t *wave, float *filters)
{
	int frames = wave->num_frames / MODEL_HOP_LENGTH;
	float *mel = (float *)malloc(MODEL_MEL_BINS * frames * sizeof(*mel));
	if (!mel)
		return -1;
	if (compute_mel(wave, filters, mel, frames) != 0) {
		free(mel);
		return -1;
	}
	float *input = (float *)encoder->inputs[0].mem->virt_addr;
	memset(input, 0, encoder->inputs[0].mem->size);
	const rknn3_tensor_attr *attr = &encoder->input_attrs[0];
	/* check_interfaces validates rank, shape and storage before this loop. */
	for (unsigned b = 0; b < attr->shape[0]; ++b) {
		unsigned first_frame = b * MODEL_CHUNK_FRAMES;
		if (first_frame >= (unsigned)frames)
			break;
		unsigned count = (unsigned)frames - first_frame;
		if (count > MODEL_CHUNK_FRAMES)
			count = MODEL_CHUNK_FRAMES;
		for (unsigned m = 0; m < MODEL_MEL_BINS; ++m) {
			float *destination = input + b * attr->stride[0] + m * attr->stride[2];
			const float *source = mel + m * frames + first_frame;
			for (unsigned k = 0; k < count; ++k) {
				*destination = source[k];
				destination += attr->stride[3];
			}
		}
	}
	free(mel);
	if (encoder->n_input == 2) {
		float *mask = (float *)encoder->inputs[1].mem->virt_addr;
		memset(mask, 0, encoder->inputs[1].mem->size);
		for (unsigned i = 0; i < encoder->output_attr.shape[0]; ++i)
			mask[tensor_offset(&encoder->input_attrs[1], i)] =
			    i < (unsigned)request->n_audio ? 0.0f : -10000.0f;
	}
	return 0;
}

static void copy_audio_embeddings(const struct network *encoder, struct network *decoder, const struct request *request)
{
	const float16 *audio = (const float16 *)encoder->output.mem->virt_addr;
	float16 *embeds = (float16 *)decoder->inputs[0].mem->virt_addr;
	for (int i = 0; i < request->n_audio; ++i) {
		for (int j = 0; j < MODEL_HIDDEN_SIZE; ++j) {
			embeds[tensor_offset(&decoder->input_attrs[0],
					     request->audio_rows[i] * MODEL_HIDDEN_SIZE + j)] =
			    audio[tensor_offset(&encoder->output_attr, i * MODEL_HIDDEN_SIZE + j)];
		}
	}
}

static int decode_timestamps(const struct network *decoder, const struct request *request, int *indices)
{
	const float16 *logits = (const float16 *)decoder->output.mem->virt_addr;
	for (int i = 0; i < request->n_units * 2; ++i) {
		if (decoder->output_attr.dtype == RKNN3_TENSOR_INT32) {
			const int32_t *timestamps = (const int32_t *)decoder->output.mem->virt_addr;

			indices[i] = timestamps[tensor_offset(&decoder->output_attr, request->slots[i])];
			if (indices[i] < 0 || indices[i] >= MODEL_TIMESTAMP_CLASSES) {
				fprintf(stderr, "Invalid timestamp index: %d\n", indices[i]);
				return -1;
			}
			continue;
		}
		float maximum = -INFINITY;
		indices[i] = 0;
		for (int cls = 0; cls < MODEL_TIMESTAMP_CLASSES; ++cls) {
			float value = fp16_to_fp32(logits[tensor_offset(
			    &decoder->output_attr, request->slots[i] * MODEL_TIMESTAMP_CLASSES + cls)]);
			if (!isfinite(value)) {
				fprintf(stderr, "Nonfinite timestamp logits\n");
				return -1;
			}
			if (value > maximum) {
				maximum = value;
				indices[i] = cls;
			}
		}
	}
	for (int i = 0; i < request->n_units * 2; ++i)
		indices[i] *= MODEL_TIMESTAMP_STEP_MS;
	return fix_timestamps(indices, request->n_units * 2);
}

static int write_timestamps(const char *output, const struct request *request, const int *indices)
{
	FILE *file = fopen(output, "w");
	if (!file) {
		fprintf(stderr, "Cannot write result: %s\n", output);
		return -1;
	}
	printf("\nInput text: %s\nWord/character timestamps (s):\n", request->text);
	fprintf(file, "start_ms\tend_ms\ttext\n");
	for (int i = 0; i < request->n_units; ++i) {
		printf("  [%.3f - %.3f] %s\n", indices[2 * i] / 1000.0, indices[2 * i + 1] / 1000.0, request->units[i]);
		fprintf(file, "%d\t%d\t%s\n", indices[2 * i], indices[2 * i + 1], request->units[i]);
	}
	int write_error = ferror(file);
	if (fclose(file) != 0 || write_error)
		return -1;
	return 0;
}

static int align_audio(struct network *encoder, struct network *decoder, struct request *request, audio_buffer_t *wave,
		       float *filters, const char *output)
{
	int *indices = request->indices;
	double start = now_ms();

	if (prepare_encoder(encoder, request, wave, filters) != 0 || sync_inputs(encoder) != 0)
		return -1;
	double audio_start = now_ms();
	if (run_network(encoder) != 0)
		return -1;
	double audio_end = now_ms();
	if (sync_output(encoder) != 0)
		return -1;
	copy_audio_embeddings(encoder, decoder, request);
	if (sync_inputs(decoder) != 0)
		return -1;
	double decoder_start = now_ms();
	if (run_network(decoder) != 0)
		return -1;
	double decoder_end = now_ms();
	if (sync_output(decoder) != 0 || decode_timestamps(decoder, request, indices) != 0)
		return -1;
	double end = now_ms();
	if (write_timestamps(output, request, indices) != 0)
		return -1;
	printf("\nAuT latency: %.2f ms\n", audio_end - audio_start);
	printf("Decoder latency: %.2f ms\n", decoder_end - decoder_start);
	/* Pipeline RTF includes preprocessing, transfers and postprocessing. */
	printf("RTF: %.4f\n", (end - start) / 1000.0 / ((double)wave->num_frames / wave->sample_rate));
	printf("Result saved to: %s\n", output);
	return 0;
}

struct options {
	const char *model_path;
	const char *audio_path;
	const char *text;
	const char *language;
	const char *mel_filters_path;
	const char *output_path;
};

static void print_usage(FILE *stream, const char *program)
{
	fprintf(stream,
		"Usage: %s --model_path DIR --audio_path WAV --text TEXT\n"
		"       --language Chinese|English --mel_filters_path FILE\n"
		"       --output_path TSV\n\n"
		"Align known text to audio; this model does not transcribe audio.\n"
		"All six options are required and may appear in any order.\n\n"
		"  --model_path DIR        RKNN models and text resources "
		"directory\n"
		"  --audio_path WAV        Input audio file (up to 30 seconds)\n"
		"  --text TEXT             Known transcript; quote text with "
		"spaces\n"
		"  --language LANGUAGE     Chinese or English\n"
		"  --mel_filters_path FILE Mel filter file\n"
		"  --output_path TSV       Output timestamp file\n"
		"  --help                  Show this help\n",
		program);
}

/* Return 1 for help, 0 for valid options, and -1 for invalid arguments. */
static int parse_options(int argc, char **argv, struct options *options)
{
	static const struct option long_options[] = {
	    {"model_path", required_argument, NULL, 0},
	    {"audio_path", required_argument, NULL, 0},
	    {"text", required_argument, NULL, 0},
	    {"language", required_argument, NULL, 0},
	    {"mel_filters_path", required_argument, NULL, 0},
	    {"output_path", required_argument, NULL, 0},
	    {"help", no_argument, NULL, 'h'},
	    {NULL, 0, NULL, 0},
	};
	const char **values[] = {
	    &options->model_path, &options->audio_path,	      &options->text,
	    &options->language,	  &options->mel_filters_path, &options->output_path,
	};
	int option, index = 0;

	while ((option = getopt_long(argc, argv, "+", long_options, &index)) != -1) {
		if (option == 'h') {
			print_usage(stdout, argv[0]);
			return 1;
		}
		if (option != 0)
			return -1;
		if (*values[index] || !optarg[0]) {
			fprintf(stderr, "Duplicate or empty option: --%s\n", long_options[index].name);
			return -1;
		}
		*values[index] = optarg;
	}
	if (optind != argc) {
		fprintf(stderr, "Unexpected positional argument: %s\n", argv[optind]);
		return -1;
	}
	for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
		if (!*values[i]) {
			fprintf(stderr, "Missing required option: --%s\n", long_options[i].name);
			return -1;
		}
	}
	if (strcmp(options->language, "Chinese") != 0 && strcmp(options->language, "English") != 0) {
		fprintf(stderr, "--language must be Chinese or English\n");
		return -1;
	}
	return 0;
}

int main(int argc, char **argv)
{
	struct network *encoder = NULL, *decoder = NULL;
	struct request *request = NULL;
	struct options options = {0};
	audio_buffer_t wave = {0};
	float *filters = NULL;
	int result = 1;

	int parsed = parse_options(argc, argv, &options);

	if (parsed > 0)
		return 0;
	if (parsed < 0) {
		print_usage(stderr, argv[0]);
		return 2;
	}
	encoder = (struct network *)calloc(1, sizeof(*encoder));
	decoder = (struct network *)calloc(1, sizeof(*decoder));
	request = (struct request *)calloc(1, sizeof(*request));
	filters = (float *)malloc(MODEL_FILTER_COUNT * sizeof(*filters));
	if (!encoder || !decoder || !request || !filters) {
		fprintf(stderr, "Cannot allocate alignment buffers\n");
		goto cleanup;
	}
	if (read_audio(options.audio_path, &wave) != 0) {
		fprintf(stderr, "Cannot read audio\n");
		goto cleanup;
	}
	if (prepare_text(options.model_path, options.text, options.language, wave.num_frames, request) != 0)
		goto cleanup;
	if (read_mel_filters(options.mel_filters_path, filters, MODEL_FILTER_COUNT) != 0 ||
	    init_network(encoder, options.model_path, "encoder", 2,
			 (wave.num_frames + MODEL_SAMPLE_RATE - 1) / MODEL_SAMPLE_RATE) != 0 ||
	    init_network(decoder, options.model_path, "decoder", 3, request->sequence) != 0 ||
	    check_interfaces(encoder, decoder) != 0 || prepare_decoder(decoder, options.model_path, request) != 0) {
		fprintf(stderr, "Model initialization or input validation failed\n");
		goto cleanup;
	}
	if (align_audio(encoder, decoder, request, &wave, filters, options.output_path) == 0)
		result = 0;
cleanup:
	free(wave.data);
	free(filters);
	free(request);
	if (release_network(decoder) != 0)
		result = 1;
	if (release_network(encoder) != 0)
		result = 1;
	free(decoder);
	free(encoder);
	return result;
}
