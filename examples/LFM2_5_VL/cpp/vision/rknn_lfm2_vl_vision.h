// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Licensed under the Apache License, Version 2.0

#ifndef _RKNN_DEMO_LFM2_VL_VISION_UTILS_H_
#define _RKNN_DEMO_LFM2_VL_VISION_UTILS_H_

#include "rknn3_api.h"
#include "common.h"

#define LFM2_VL_VISION_MODEL_WIDTH  384
#define LFM2_VL_VISION_MODEL_HEIGHT 384

typedef struct {
    rknn3_context rknn_ctx;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs;
    rknn3_tensor* outputs;

    int model_channel;
    int model_height;   // smart_resize target height (e.g. 384)
    int model_width;    // smart_resize target width  (e.g. 384)

    // NaFlex patch-sequence input params
    int num_patches;    // padded patch count of pixel_values (e.g. 1024)
    int patch_flat;     // patch_flat = patch_size*patch_size*channels (e.g. 768)

    // Image-processor params dumped by export_vision.py (vision_config.json),
    // used to reproduce the exact NaFlex preprocessing in C++.
    int    encoder_patch_size;
    int    downsample_factor;
    int    min_image_tokens;
    int    max_image_tokens;
    int    max_num_patches;
    int    do_image_splitting;
    float  image_mean[3];
    float  image_std[3];
    float  rescale_factor;

    // ---- External pos_emb support (方案 B, dynamic grid) ----
    // The position embedding is no longer baked into the ONNX graph as a
    // constant for a fixed grid. Instead the base 16x16 embedding is loaded
    // from pos_emb_base.bin and resized on-device with antialias bilinear
    // (exact torch F.interpolate(antialias=True) replica) to the per-image
    // grid, then padded to pos_emb_rows (1024) with first-row fill. The graph
    // is fully static (fixed reshape 32x32, output always 256 rows); the
    // valid region (r*16+c, r<gh/2, c<gw/2) is extracted in C++.
    float* pos_emb_base;    // base_rows * pos_emb_dim, fp32, from pos_emb_base.bin
    int    grid_base;       // base spatial size (16)
    int    pos_emb_dim;     // hidden dim (1152)
    int    pos_emb_rows;    // padded rows fed to model (1024)
    int    out_rows;        // fixed ONNX output rows (256)
    int    out_dim;         // fixed ONNX output dim (2048)
    int    out_cols;        // out_rows as grid side: 16 (256 = 16x16 after unshuffle)
    int    n_valid_embeds;  // valid embeds rows after extraction (per inference)
    int    last_grid_h;     // smart_resize grid of the latest inference
    int    last_grid_w;
    char   pos_emb_base_path[512];

    uint32_t* embeds_shape;
    uint32_t embeds_ndims;
} rknn_lfm2_vl_vision_context;

int init_lfm2_vl_vision(rknn_lfm2_vl_vision_context* vision_ctx,
                        const char* model_path,
                        const char* weight_path,
                        uint32_t core_mask);

int release_lfm2_vl_vision(rknn_lfm2_vl_vision_context* vision_ctx);

int inference_lfm2_vl_vision(rknn_lfm2_vl_vision_context* vision_ctx,
                             image_buffer_t* img,
                             float16* img_embeds);

#endif //_RKNN_DEMO_LFM2_VL_VISION_UTILS_H_
