// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cnpy.h"
#include "wemm_embedding.h"

// Command-line inputs for exactly one text, image, or video embedding request.
typedef struct {
    const char* model_path;
    const char* text;
    const char* image_path;
    const char* video_frames_path;
    float video_fps;
    const char* output_path;
} command_options_t;

// Print the three supported single-request command forms.
static void print_usage(const char* program)
{
    printf("Single text:\n");
    printf("  %s --model_path <model_directory> --text <text> --output_path <embedding.npy>\n", program);
    printf("Single image:\n");
    printf("  %s --model_path <model_directory> --image_path <image> --text <instruction> --output_path <embedding.npy>\n", program);
    printf("Video frame list:\n");
    printf("  %s --model_path <model_directory> --video_frames_path <frames.txt> [--video_fps <sampled_fps, default 2>] --text <instruction> --output_path <embedding.npy>\n", program);
}

// Parse key-value arguments and reject mixed image/video inputs.
static int parse_options(int argc, char** argv, command_options_t* options)
{
    int i;
    memset(options, 0, sizeof(*options));
    options->video_fps = 2.0f;
    for (i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            return -1;
        }
        if (strcmp(argv[i], "--model_path") == 0) {
            options->model_path = argv[i + 1];
        } else if (strcmp(argv[i], "--text") == 0) {
            options->text = argv[i + 1];
        } else if (strcmp(argv[i], "--image_path") == 0) {
            options->image_path = argv[i + 1];
        } else if (strcmp(argv[i], "--video_frames_path") == 0) {
            options->video_frames_path = argv[i + 1];
        } else if (strcmp(argv[i], "--video_fps") == 0) {
            char* end = NULL;
            options->video_fps = strtof(argv[i + 1], &end);
            if (end == argv[i + 1] || *end != '\0' ||
                !isfinite(options->video_fps) || options->video_fps <= 0.0f) {
                return -1;
            }
        } else if (strcmp(argv[i], "--output_path") == 0) {
            options->output_path = argv[i + 1];
        } else {
            return -1;
        }
    }
    if (options->model_path == NULL || options->output_path == NULL) {
        return -1;
    }
    if (options->text == NULL ||
        (options->image_path != NULL && options->video_frames_path != NULL)) {
        return -1;
    }
    return 0;
}

// Store the normalized 2048-element embedding in NumPy NPY format.
static int save_embedding(const char* output_path, const float* embedding)
{
    uint32_t shape[1] = {WEMM_EMBEDDING_DIM};
    return npy_save_float_buffer_to_file(output_path, (float*)embedding,
                                         WEMM_EMBEDDING_DIM, shape, 1);
}

// Print latency, token count, and a short embedding preview.
static void print_result(const float* embedding,
                         const wemm_embedding_perf_t* perf)
{
    int i;
    printf("Result: tokens=%lu, vision=%.2f ms, llm=%.2f ms, embedding[:8]=[",
           (unsigned long)perf->input_tokens, perf->vision_ms, perf->llm_ms);
    for (i = 0; i < 8; ++i) {
        printf("%s%.6f", i == 0 ? "" : ", ", embedding[i]);
    }
    printf("]\n");
}

// Own all paths loaded from a video frame list.
typedef struct {
    char** paths;
    uint64_t count;
} frame_list_t;

// Release strings and the path array allocated by load_frame_list().
static void release_frame_list(frame_list_t* frames)
{
    uint64_t i;
    for (i = 0; i < frames->count; ++i) {
        free(frames->paths[i]);
    }
    free(frames->paths);
    memset(frames, 0, sizeof(*frames));
}

// Read one decoded frame path per line; blank lines and comments are skipped.
static int load_frame_list(const char* list_path, frame_list_t* frames)
{
    FILE* file;
    char* line = NULL;
    size_t line_capacity = 0;
    size_t path_capacity = 0;
    ssize_t length;
    int ret = -1;

    memset(frames, 0, sizeof(*frames));
    file = fopen(list_path, "r");
    if (file == NULL) {
        fprintf(stderr, "Failed to open video frame list: %s\n", list_path);
        return -1;
    }
    while ((length = getline(&line, &line_capacity, file)) >= 0) {
        char** new_paths;
        char* path;
        while (length > 0 &&
               (line[length - 1] == '\n' || line[length - 1] == '\r')) {
            line[--length] = '\0';
        }
        if (length == 0 || line[0] == '#') {
            continue;
        }
        if (frames->count == path_capacity) {
            size_t new_capacity = path_capacity == 0 ? 8 : path_capacity * 2;
            new_paths = (char**)realloc(frames->paths,
                                        new_capacity * sizeof(*new_paths));
            if (new_paths == NULL) {
                goto out;
            }
            frames->paths = new_paths;
            path_capacity = new_capacity;
        }
        path = strdup(line);
        if (path == NULL) {
            goto out;
        }
        frames->paths[frames->count++] = path;
    }
    if (ferror(file) || frames->count == 0) {
        fprintf(stderr, "Video frame list is empty or unreadable: %s\n", list_path);
        goto out;
    }
    ret = 0;

out:
    free(line);
    fclose(file);
    if (ret != 0) {
        release_frame_list(frames);
    }
    return ret;
}

// Dispatch one request to the text, image, or video pipeline and save its result.
static int run_embedding_request(wemm_embedding_context_t* context,
                                 const char* image_path,
                                 const char* video_frames_path,
                                 float video_fps,
                                 const char* text,
                                 const char* output_path)
{
    float embedding[WEMM_EMBEDDING_DIM];
    wemm_embedding_perf_t perf;
    frame_list_t frames;
    int ret;
    memset(&perf, 0, sizeof(perf));
    memset(&frames, 0, sizeof(frames));
    if (video_frames_path != NULL) {
        ret = load_frame_list(video_frames_path, &frames);
        if (ret == 0) {
            ret = run_wemm_video_embedding(
                context, (const char* const*)frames.paths, frames.count,
                video_fps, text, embedding, &perf);
        }
        release_frame_list(&frames);
    } else if (image_path == NULL) {
        ret = run_wemm_text_embedding(context, text, embedding, &perf);
    } else {
        ret = run_wemm_image_embedding(context, image_path, text, embedding, &perf);
    }
    if (ret != 0) {
        fprintf(stderr, "Inference failed: ret=%d\n", ret);
        return ret;
    }
    ret = save_embedding(output_path, embedding);
    if (ret != 0) {
        fprintf(stderr, "Failed to save embedding: %s\n", output_path);
        return ret;
    }
    print_result(embedding, &perf);
    return 0;
}

// Initialize once, execute one request, and release every model resource.
int main(int argc, char** argv)
{
    command_options_t options;
    wemm_embedding_context_t context;
    wemm_embedding_mode_t mode;
    int ret;

    memset(&context, 0, sizeof(context));
    if (parse_options(argc, argv, &options) != 0) {
        print_usage(argv[0]);
        return 1;
    }
    mode = options.video_frames_path != NULL
               ? WEMM_EMBEDDING_MODE_VIDEO
               : (options.image_path != NULL ? WEMM_EMBEDDING_MODE_IMAGE
                                             : WEMM_EMBEDDING_MODE_TEXT);
    ret = init_wemm_embedding(&context, options.model_path, mode);
    if (ret != 0) {
        fprintf(stderr, "Failed to initialize WeMM-Embedding: %d\n", ret);
        return 1;
    }
    ret = run_embedding_request(&context, options.image_path,
                                options.video_frames_path, options.video_fps,
                                options.text, options.output_path);
    release_wemm_embedding(&context);
    return ret == 0 ? 0 : 1;
}
