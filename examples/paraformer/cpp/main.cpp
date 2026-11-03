// Copyright (c) 2026 Rockchip Electronics Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_utils.h"
#include "paraformer.h"
#include "time_utils.h"

typedef struct {
    const char* encoder_path;
    const char* decoder_path;
    const char* tokens_path;
    const char* audio_path;
    uint32_t core_mask;
    const char* seaco_path;
    const char* hotwords_path;
    const char* asf_path;
    int asf_nfilter;
    int max_hotwords;
    int max_hotword_len;
    float seaco_weight;
    int show_help;
} runtime_options_t;

typedef struct {
    char* encoder;
    char* decoder;
    char* seaco;
    char* asf;
    char* bias_encoder;
    char* seg_dict;
} resolved_paths_t;

/** Prints the named model, resource, audio, and optional hotword arguments. */
static void print_usage(const char* program) {
    printf("Usage: %s --encoder_path <paraformer_encoder.rknn> "
           "--decoder_path <paraformer_decoder.rknn> "
           "--tokens_path <tokens.json> --audio_path <audio.wav> [options]\n"
           "\n"
           "The matching .weight path is derived from every .rknn path.\n"
           "SeaCo also loads sibling bias_encoder.bin and seg_dict.\n"
           "\n"
           "Options:\n"
           "  --core_mask <mask>\n"
           "  --seaco_path <seaco_decoder.rknn>\n"
           "  --hotwords_path <hotwords.txt>\n"
           "  --asf_path <asf_scorer.rknn>\n"
           "  --asf_nfilter <count>\n"
           "  --seaco_weight <0..1>\n"
           "  --max_hotwords <1..128>\n"
           "  --max_hotword_len <1..256>\n"
           "  --help\n",
           program);
}

/** Parses one positive decimal integer without accepting trailing text. */
static int parse_positive_int(const char* text, int* value) {
    char* parse_end = NULL;
    errno = 0;
    long parsed = strtol(text, &parse_end, 10);
    if (errno != 0 || !parse_end || *parse_end != '\0' || parsed <= 0 || parsed > INT32_MAX) {
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

/** Stores one path option and rejects missing values or duplicate options. */
static int parse_path_option(int argc, char** argv, int index, const char* option_name, const char** destination) {
    if (!destination || *destination || index + 1 >= argc || strncmp(argv[index + 1], "--", 2) == 0) {
        printf("Invalid, missing, or duplicate option: %s\n", option_name);
        return -1;
    }
    *destination = argv[index + 1];
    return 0;
}

/** Parses a hexadecimal or decimal core mask into one unsigned 32-bit value. */
static int parse_core_mask(const char* text, uint32_t* core_mask) {
    char* parse_end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(text, &parse_end, 0);
    if (errno != 0 || !parse_end || *parse_end != '\0' || parsed == 0 || parsed > UINT32_MAX) {
        return -1;
    }
    *core_mask = (uint32_t)parsed;
    return 0;
}

/** Parses all public named CLI options and validates option groups. */
static int parse_runtime_options(int argc, char** argv, runtime_options_t* options) {
    int index = 1;
    if (!options)
        return -1;
    memset(options, 0, sizeof(*options));
    options->max_hotwords = 128;
    options->max_hotword_len = 16;
    options->seaco_weight = 1.0f;

    while (index < argc) {
        if (strcmp(argv[index], "--help") == 0 || strcmp(argv[index], "-h") == 0) {
            options->show_help = 1;
            ++index;
        } else if (strcmp(argv[index], "--encoder_path") == 0) {
            if (parse_path_option(argc, argv, index, argv[index], &options->encoder_path) != 0) {
                return -1;
            }
            index += 2;
        } else if (strcmp(argv[index], "--decoder_path") == 0) {
            if (parse_path_option(argc, argv, index, argv[index], &options->decoder_path) != 0) {
                return -1;
            }
            index += 2;
        } else if (strcmp(argv[index], "--tokens_path") == 0) {
            if (parse_path_option(argc, argv, index, argv[index], &options->tokens_path) != 0) {
                return -1;
            }
            index += 2;
        } else if (strcmp(argv[index], "--audio_path") == 0) {
            if (parse_path_option(argc, argv, index, argv[index], &options->audio_path) != 0) {
                return -1;
            }
            index += 2;
        } else if (strcmp(argv[index], "--seaco_path") == 0) {
            if (parse_path_option(argc, argv, index, argv[index], &options->seaco_path) != 0) {
                return -1;
            }
            index += 2;
        } else if (strcmp(argv[index], "--hotwords_path") == 0) {
            if (parse_path_option(argc, argv, index, argv[index], &options->hotwords_path) != 0) {
                return -1;
            }
            index += 2;
        } else if (strcmp(argv[index], "--asf_path") == 0) {
            if (parse_path_option(argc, argv, index, argv[index], &options->asf_path) != 0) {
                return -1;
            }
            index += 2;
        } else if (strcmp(argv[index], "--core_mask") == 0 && index + 1 < argc) {
            if (parse_core_mask(argv[index + 1], &options->core_mask) != 0) {
                printf("Invalid core mask: %s\n", argv[index + 1]);
                return -1;
            }
            index += 2;
        } else if (strcmp(argv[index], "--asf_nfilter") == 0 && index + 1 < argc) {
            if (parse_positive_int(argv[index + 1], &options->asf_nfilter) != 0) {
                printf("Invalid ASF nfilter: %s\n", argv[index + 1]);
                return -1;
            }
            index += 2;
        } else if (strcmp(argv[index], "--seaco_weight") == 0 && index + 1 < argc) {
            char* parse_end = NULL;
            errno = 0;
            options->seaco_weight = strtof(argv[index + 1], &parse_end);
            if (errno != 0 || !parse_end || *parse_end != '\0' || !isfinite(options->seaco_weight) ||
                options->seaco_weight < 0.0f || options->seaco_weight > 1.0f) {
                printf("Invalid SeaCo weight: %s\n", argv[index + 1]);
                return -1;
            }
            index += 2;
        } else if (strcmp(argv[index], "--max_hotwords") == 0 && index + 1 < argc) {
            if (parse_positive_int(argv[index + 1], &options->max_hotwords) != 0 || options->max_hotwords > 128) {
                printf("Invalid max hotwords: %s\n", argv[index + 1]);
                return -1;
            }
            index += 2;
        } else if (strcmp(argv[index], "--max_hotword_len") == 0 && index + 1 < argc) {
            if (parse_positive_int(argv[index + 1], &options->max_hotword_len) != 0 || options->max_hotword_len > 256) {
                printf("Invalid max hotword length: %s\n", argv[index + 1]);
                return -1;
            }
            index += 2;
        } else {
            printf("Unknown or incomplete option: %s\n", argv[index]);
            return -1;
        }
    }

    if (options->show_help)
        return 0;
    if (!options->encoder_path || !options->decoder_path || !options->tokens_path || !options->audio_path) {
        printf("Required options: --encoder_path, --decoder_path, "
               "--tokens_path, --audio_path\n");
        return -1;
    }

    int hotword_options = (options->seaco_path != NULL) + (options->hotwords_path != NULL);
    if (hotword_options != 0 && hotword_options != 2) {
        printf("Hotword mode requires both --seaco_path and --hotwords_path\n");
        return -1;
    }
    if ((options->asf_path != NULL) != (options->asf_nfilter > 0)) {
        printf("ASF requires both --asf_path and --asf_nfilter\n");
        return -1;
    }
    if (options->asf_path && hotword_options == 0) {
        printf("ASF requires the complete hotword option group\n");
        return -1;
    }
    return 0;
}

/** Resolves `<name>.rknn` to a readable sibling `<name>.weight` file. */
static char* resolve_weight_path(const char* model_path) {
    static const char model_suffix[] = ".rknn";
    static const char weight_suffix[] = ".weight";
    size_t model_length;
    size_t stem_length;
    char* weight_path;
    FILE* weight_file;

    if (!model_path)
        return NULL;
    model_length = strlen(model_path);
    if (model_length <= strlen(model_suffix) ||
        strcmp(model_path + model_length - strlen(model_suffix), model_suffix) != 0) {
        printf("RKNN model path must end in .rknn: %s\n", model_path);
        return NULL;
    }

    stem_length = model_length - strlen(model_suffix);
    weight_path = (char*)malloc(stem_length + sizeof(weight_suffix));
    if (!weight_path)
        return NULL;
    memcpy(weight_path, model_path, stem_length);
    memcpy(weight_path + stem_length, weight_suffix, sizeof(weight_suffix));

    weight_file = fopen(weight_path, "rb");
    if (!weight_file) {
        printf("Matching RKNN weight not found: %s\n", weight_path);
        free(weight_path);
        return NULL;
    }
    fclose(weight_file);
    return weight_path;
}

/** Resolves one resource stored beside an RKNN model. */
static char* resolve_sibling_path(const char* model_path, const char* name) {
    const char* separator;
    size_t directory_length;
    size_t name_length;
    char* path;

    if (!model_path || !name)
        return NULL;
    separator = strrchr(model_path, '/');
    directory_length = separator ? (size_t)(separator - model_path + 1) : 0;
    name_length = strlen(name);
    path = (char*)malloc(directory_length + name_length + 1);
    if (!path)
        return NULL;
    if (directory_length > 0) {
        memcpy(path, model_path, directory_length);
    }
    memcpy(path + directory_length, name, name_length + 1);
    return path;
}

/** Releases all automatically resolved weight and runtime resource paths. */
static void release_resolved_paths(resolved_paths_t* paths) {
    if (!paths)
        return;
    free(paths->encoder);
    free(paths->decoder);
    free(paths->seaco);
    free(paths->asf);
    free(paths->bias_encoder);
    free(paths->seg_dict);
    memset(paths, 0, sizeof(*paths));
}

/** Resolves weights and the resources packaged beside the SeaCo RKNN model. */
static int resolve_model_paths(const runtime_options_t* options, resolved_paths_t* paths) {
    if (!options || !paths)
        return -1;
    memset(paths, 0, sizeof(*paths));

    paths->encoder = resolve_weight_path(options->encoder_path);
    paths->decoder = resolve_weight_path(options->decoder_path);
    if (!paths->encoder || !paths->decoder)
        goto failed;

    if (options->seaco_path) {
        paths->seaco = resolve_weight_path(options->seaco_path);
        paths->bias_encoder = resolve_sibling_path(options->seaco_path, "bias_encoder.bin");
        paths->seg_dict = resolve_sibling_path(options->seaco_path, "seg_dict");
        if (!paths->seaco || !paths->bias_encoder || !paths->seg_dict)
            goto failed;
    }
    if (options->asf_path) {
        paths->asf = resolve_weight_path(options->asf_path);
        if (!paths->asf)
            goto failed;
    }
    return 0;

failed:
    release_resolved_paths(paths);
    return -1;
}

/**
 * Loads one audio file, runs the complete Paraformer pipeline, and prints text.
 * Returns zero on success and a nonzero value when validation or inference
 * fails.
 */
int main(int argc, char** argv) {
    runtime_options_t options;
    int parse_result = parse_runtime_options(argc, argv, &options);
    if (parse_result != 0 || options.show_help) {
        print_usage(argv[0]);
        return options.show_help ? 0 : -1;
    }

    resolved_paths_t paths;
    if (resolve_model_paths(&options, &paths) != 0) {
        return -1;
    }

    audio_buffer_t audio;
    memset(&audio, 0, sizeof(audio));
    if (read_audio(options.audio_path, &audio) != 0 || !audio.data) {
        printf("Failed to read audio: %s\n", options.audio_path);
        release_resolved_paths(&paths);
        return -1;
    }

    int16_t* pcm = (int16_t*)malloc((size_t)audio.num_frames * sizeof(int16_t));
    if (!pcm) {
        free(audio.data);
        release_resolved_paths(&paths);
        return -1;
    }
    for (int index = 0; index < audio.num_frames; ++index) {
        float sample = audio.data[index];
        if (sample > 1.0f)
            sample = 1.0f;
        if (sample < -1.0f)
            sample = -1.0f;
        pcm[index] = (int16_t)(sample * 32767.0f);
    }
    free(audio.data);
    audio.data = NULL;

    rknn_app_context_t context;
    memset(&context, 0, sizeof(context));
    int64_t begin_us = getCurrentTimeUs();
    int ret = init_paraformer_model(options.encoder_path, options.decoder_path, paths.encoder, paths.decoder, &context,
                                    options.core_mask);
    if (ret != 0) {
        free(pcm);
        release_resolved_paths(&paths);
        return ret;
    }
    if (options.seaco_path) {
        ret =
            init_paraformer_hotword(options.seaco_path, paths.seaco, paths.bias_encoder, options.hotwords_path,
                                    options.tokens_path, paths.seg_dict, options.max_hotwords, options.max_hotword_len,
                                    options.seaco_weight, options.asf_path, paths.asf, options.asf_nfilter, &context);
        if (ret != 0) {
            release_paraformer_model(&context);
            free(pcm);
            release_resolved_paths(&paths);
            return ret;
        }
    }
    printf("Model initialization: %.2f ms\n", (getCurrentTimeUs() - begin_us) / 1000.0);

    char recognition[4096];
    memset(recognition, 0, sizeof(recognition));
    begin_us = getCurrentTimeUs();
    ret = run_paraformer_inference(&context, pcm, audio.num_frames, options.tokens_path, recognition,
                                   sizeof(recognition));
    double inference_ms = (getCurrentTimeUs() - begin_us) / 1000.0;
    free(pcm);

    if (ret == 0) {
        double duration = (double)audio.num_frames / 16000.0;
        printf("Recognition Result: %s\n", recognition);
        printf("Inference time: %.2f ms\n", inference_ms);
        printf("Audio duration: %.3f s\n", duration);
        printf("RTF: %.4f\n", inference_ms / 1000.0 / duration);
    }

    release_paraformer_model(&context);
    release_resolved_paths(&paths);
    return ret;
}
