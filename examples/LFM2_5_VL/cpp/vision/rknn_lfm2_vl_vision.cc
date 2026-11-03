// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Licensed under the Apache License, Version 2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <math.h>
#include <stdint.h>

#include "rknn_lfm2_vl_vision.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

static rknn3_init_extend make_rknn3_init_extend(const char* tag)
{
    static char selected_device_id[RKNN3_MAX_DEV_LEN] = {0};
    rknn3_init_extend init_extend;
    memset(&init_extend, 0, sizeof(init_extend));

    const char* env_device_id = getenv("RKNN3_DEVICE_ID");
    if (env_device_id && env_device_id[0] != '\0') {
        snprintf(selected_device_id, sizeof(selected_device_id), "%s", env_device_id);
        init_extend.device_id = selected_device_id;
        printf("[%s] use RKNN3_DEVICE_ID=%s\n", tag, selected_device_id);
        return init_extend;
    }

    rknn3_devices devices;
    memset(&devices, 0, sizeof(devices));
    int ret = rknn3_find_devices(&devices);
    if (ret == 0) {
        if (devices.n_devices > 1) {
            printf("[%s] multiple RKNN3 devices found, using device_id=%s\n",
                   tag, devices.devices[0].id);
            snprintf(selected_device_id, sizeof(selected_device_id), "%s", devices.devices[0].id);
            init_extend.device_id = selected_device_id;
        }
    }
    return init_extend;
}

static void dump_tensor_attr(rknn3_tensor_attr* attrs)
{
    std::string shape_str = "";
    for (int j = 0; j < attrs->n_dims; j++) {
        shape_str += std::to_string(attrs->shape[j]);
        if (j < attrs->n_dims - 1) {
            shape_str += ", ";
        }
    }

    std::string stride_str = "";
    for (int j = 0; j < attrs->n_stride; j++) {
        stride_str += std::to_string(attrs->stride[j]);
        if (j < attrs->n_stride - 1) {
            stride_str += ", ";
        }
    }

    printf("Tensor: name=%s, n_dims=%d, shape=[%s], stride=[%s], aligned_size=%ld, layout=%s, dtype=%s, core_id=%d, "
           "qnt_type=%s\n",
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size,
           rknn3_get_layout_string(attrs->layout), rknn3_get_type_string(attrs->dtype), attrs->core_id,
           rknn3_get_qnt_type_string(attrs->qnt_type));
}

/*---------------- Minimal JSON getters for vision_config.json ----------------*/
// Only used to read the preprocess params dumped by export_vision.py.
// Assumes "key": value / "key": [a, b, c] fragments, keys are unique in file.
static bool json_get_int(const std::string& s, const char* key, int* out)
{
    std::string pat = std::string("\"") + key + "\":";
    size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) p++;
    *out = (int)strtol(s.c_str() + p, NULL, 10);
    return true;
}

static bool json_get_double(const std::string& s, const char* key, double* out)
{
    std::string pat = std::string("\"") + key + "\":";
    size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) p++;
    *out = strtod(s.c_str() + p, NULL);
    return true;
}

static bool json_get_float_array(const std::string& s, const char* key, float* out, int n)
{
    std::string pat = std::string("\"") + key + "\":[";
    size_t p = s.find(pat);
    if (p == std::string::npos) return false;
    p += pat.size();
    for (int i = 0; i < n; i++) {
        while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) p++;
        out[i] = (float)strtod(s.c_str() + p, NULL);
        while (p < s.size() && !(s[p] == ',' || s[p] == ']')) p++;
        if (p < s.size() && s[p] == ',') p++;
    }
    return true;
}

static bool json_get_bool(const std::string& s, const char* key, bool* out)
{
    int v = 0;
    if (!json_get_int(s, key, &v)) return false;
    *out = (v != 0);
    return true;
}

static std::string read_file_to_string(const char* path)
{
    FILE* fp = fopen(path, "rb");
    if (!fp) return std::string();
    fseek(fp, 0, SEEK_END);
    long sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    std::string content;
    if (sz > 0) {
        content.resize(sz);
        fread(&content[0], 1, sz, fp);
    }
    fclose(fp);
    return content;
}

// Load image-processor params from vision_config.json (dumped by export_vision.py).
// Falls back to defaults so the demo still works without the file.
static void load_vision_preprocess_config(rknn_lfm2_vl_vision_context* vision_ctx)
{
    // Defaults (LFM2-VL processor defaults)
    vision_ctx->encoder_patch_size = 16;
    vision_ctx->downsample_factor  = 2;
    vision_ctx->min_image_tokens   = 64;
    vision_ctx->max_image_tokens   = 256;
    vision_ctx->max_num_patches    = 1024;
    vision_ctx->do_image_splitting = 0;
    vision_ctx->rescale_factor     = 1.0f / 255.0f;
    vision_ctx->image_mean[0] = vision_ctx->image_mean[1] = vision_ctx->image_mean[2] = 0.5f;
    vision_ctx->image_std[0]  = vision_ctx->image_std[1]  = vision_ctx->image_std[2]  = 0.5f;

    const char* env_cfg = getenv("LFM2_VL_VISION_CONFIG");
    const char* cfg_path = env_cfg ? env_cfg : "./vision_config.json";
    std::string s = read_file_to_string(cfg_path);
    if (s.empty()) {
        printf("[Vision] vision_config.json not found (%s), using default preprocess params\n", cfg_path);
        return;
    }

    int v = 0;
    double d = 0.0;
    bool b = false;
    if (json_get_int(s, "img_h", &v)) vision_ctx->model_height = v;
    if (json_get_int(s, "img_w", &v)) vision_ctx->model_width = v;
    if (json_get_int(s, "num_patches", &v)) vision_ctx->num_patches = v;
    if (json_get_int(s, "patch_flat", &v)) vision_ctx->patch_flat = v;
    if (json_get_int(s, "encoder_patch_size", &v)) vision_ctx->encoder_patch_size = v;
    if (json_get_int(s, "downsample_factor", &v)) vision_ctx->downsample_factor = v;
    if (json_get_int(s, "min_image_tokens", &v)) vision_ctx->min_image_tokens = v;
    if (json_get_int(s, "max_image_tokens", &v)) vision_ctx->max_image_tokens = v;
    if (json_get_int(s, "max_num_patches", &v)) vision_ctx->max_num_patches = v;
    if (json_get_bool(s, "do_image_splitting", &b)) vision_ctx->do_image_splitting = b ? 1 : 0;
    if (json_get_double(s, "rescale_factor", &d)) vision_ctx->rescale_factor = (float)d;
    json_get_float_array(s, "image_mean", vision_ctx->image_mean, 3);
    json_get_float_array(s, "image_std", vision_ctx->image_std, 3);

    // External pos_emb graph params
    if (json_get_int(s, "pos_emb_rows", &v)) vision_ctx->pos_emb_rows = v;
    if (json_get_int(s, "pos_emb_dim", &v)) vision_ctx->pos_emb_dim = v;
    if (json_get_int(s, "grid_base", &v)) vision_ctx->grid_base = v;
    if (json_get_int(s, "out_rows", &v)) vision_ctx->out_rows = v;
    if (json_get_int(s, "out_dim", &v)) vision_ctx->out_dim = v;

    printf("[Vision] preprocess params from %s: img=%dx%d num_patches=%d patch_flat=%d "
           "enc_patch=%d downsample=%d min_tokens=%d max_tokens=%d max_num_patches=%d split=%d "
           "mean=[%.4f,%.4f,%.4f] std=[%.4f,%.4f,%.4f]\n",
           cfg_path, vision_ctx->model_width, vision_ctx->model_height,
           vision_ctx->num_patches, vision_ctx->patch_flat,
           vision_ctx->encoder_patch_size, vision_ctx->downsample_factor,
           vision_ctx->min_image_tokens, vision_ctx->max_image_tokens,
           vision_ctx->max_num_patches, vision_ctx->do_image_splitting,
           vision_ctx->image_mean[0], vision_ctx->image_mean[1], vision_ctx->image_mean[2],
           vision_ctx->image_std[0], vision_ctx->image_std[1], vision_ctx->image_std[2]);
    printf("[Vision] posemb graph: pos_emb_rows=%d pos_emb_dim=%d grid_base=%d out_rows=%d out_dim=%d\n",
           vision_ctx->pos_emb_rows, vision_ctx->pos_emb_dim, vision_ctx->grid_base,
           vision_ctx->out_rows, vision_ctx->out_dim);
}

/* smart_resize of LFM2-VL image processor (do_image_splitting=False path):
 * find largest H,W (multiple of total_factor) that keep W*H within [min,max] pixels.
 * total_factor = encoder_patch_size * downsample_factor
 */
static void smart_resize(int src_h, int src_w, int total_factor,
                         long long min_pixels, long long max_pixels,
                         int* out_h, int* out_w)
{
    int h_bar = (int)(round((double)src_h / total_factor) * total_factor);
    int w_bar = (int)(round((double)src_w / total_factor) * total_factor);
    if (h_bar < total_factor) h_bar = total_factor;
    if (w_bar < total_factor) w_bar = total_factor;

    if ((long long)h_bar * w_bar > max_pixels) {
        double beta = sqrt((double)src_h * src_w / max_pixels);
        h_bar = (int)(floor(src_h / beta / total_factor) * total_factor);
        w_bar = (int)(floor(src_w / beta / total_factor) * total_factor);
        if (h_bar < total_factor) h_bar = total_factor;
        if (w_bar < total_factor) w_bar = total_factor;
    } else if ((long long)h_bar * w_bar < min_pixels) {
        double beta = sqrt((double)min_pixels / (src_h * src_w));
        h_bar = (int)(ceil(src_h * beta / total_factor) * total_factor);
        w_bar = (int)(ceil(src_w * beta / total_factor) * total_factor);
        if (h_bar < total_factor) h_bar = total_factor;
        if (w_bar < total_factor) w_bar = total_factor;
    }
    *out_h = h_bar;
    *out_w = w_bar;
}

/* Resize RGB888 with the same sampling semantics as torchvision/PIL bilinear:
 * half-pixel centers and an antialias triangular kernel when downsampling.
 */
static int resize_rgb_antialias(const image_buffer_t* src,
                                image_buffer_t* dst)
{
    if (!src || !dst || !src->virt_addr || !dst->virt_addr ||
        src->format != IMAGE_FORMAT_RGB888 || dst->format != IMAGE_FORMAT_RGB888) {
        return -1;
    }

    const int src_w = src->width;
    const int src_h = src->height;
    const int dst_w = dst->width;
    const int dst_h = dst->height;
    const unsigned char* in = (const unsigned char*)src->virt_addr;
    unsigned char* out = (unsigned char*)dst->virt_addr;
    const double scale_x = (double)src_w / dst_w;
    const double scale_y = (double)src_h / dst_h;
    const double support_x = scale_x > 1.0 ? scale_x : 1.0;
    const double support_y = scale_y > 1.0 ? scale_y : 1.0;
    const int win_x = (int)floor(2.0 * support_x + 1.0);
    const int win_y = (int)floor(2.0 * support_y + 1.0);

    std::vector<int> x0(dst_w);
    std::vector<int> x1(dst_w);
    std::vector<double> wx(dst_w * win_x);
    for (int ox = 0; ox < dst_w; ox++) {
        const double center = scale_x * (ox + 0.5);
        int left = (int)floor(center - support_x + 0.5);
        if (left < 0) left = 0;
        int right = (int)floor(center + support_x + 0.5);
        if (right > src_w) right = src_w;
        x1[ox] = right;
        double total = 0.0;
        for (int sx = left; sx < right; sx++) {
            const double distance = (sx + 0.5 - center) / support_x;
            const double weight = std::max(0.0, 1.0 - fabs(distance));
            wx[(size_t)ox * win_x + sx - left] = weight;
            total += weight;
        }
        if (total == 0.0) {
            return -1;
        }
        for (int k = 0; k < right - left; k++) {
            wx[(size_t)ox * win_x + k] /= total;
        }
        x0[ox] = left;
    }

    std::vector<int> y0(dst_h);
    std::vector<int> y1(dst_h);
    std::vector<double> wy(dst_h * win_y);
    for (int oy = 0; oy < dst_h; oy++) {
        const double center = scale_y * (oy + 0.5);
        int top = (int)floor(center - support_y + 0.5);
        if (top < 0) top = 0;
        int bottom = (int)floor(center + support_y + 0.5);
        if (bottom > src_h) bottom = src_h;
        y1[oy] = bottom;
        double total = 0.0;
        for (int sy = top; sy < bottom; sy++) {
            const double distance = (sy + 0.5 - center) / support_y;
            const double weight = std::max(0.0, 1.0 - fabs(distance));
            wy[(size_t)oy * win_y + sy - top] = weight;
            total += weight;
        }
        if (total == 0.0) {
            return -1;
        }
        for (int k = 0; k < bottom - top; k++) {
            wy[(size_t)oy * win_y + k] /= total;
        }
        y0[oy] = top;
    }

    for (int oy = 0; oy < dst_h; oy++) {
        const int top = y0[oy];
        const int rows = y1[oy] - top;
        const double* row_weights = &wy[(size_t)oy * win_y];
        for (int ox = 0; ox < dst_w; ox++) {
            const int left = x0[ox];
            const int cols = x1[ox] - left;
            const double* col_weights = &wx[(size_t)ox * win_x];
            double rgb[3] = {0.0, 0.0, 0.0};
            for (int ky = 0; ky < rows; ky++) {
                const unsigned char* row = in + ((size_t)(top + ky) * src_w + left) * 3;
                const double rw = row_weights[ky];
                for (int kx = 0; kx < cols; kx++) {
                    const double weight = rw * col_weights[kx];
                    const unsigned char* pixel = row + (size_t)kx * 3;
                    rgb[0] += weight * pixel[0];
                    rgb[1] += weight * pixel[1];
                    rgb[2] += weight * pixel[2];
                }
            }
            unsigned char* pixel = out + ((size_t)oy * dst_w + ox) * 3;
            for (int c = 0; c < 3; c++) {
                pixel[c] = (unsigned char)(rgb[c] + 0.5);
            }
        }
    }
    return 0;
}

/* Exact replica of torch F.interpolate(pos, size=(gh,gw), mode='bilinear',
 * align_corners=False, antialias=True), for the NaFlex position embedding
 * (base grid_base x grid_base -> gh x gw). Verified against torch output.
 * out must be gh*gw*D floats. base is base_h*base_w*D row-major.
 */
static void aa_bilinear_posemb(const float* base, int base_h, int base_w, int D,
                               int gh, int gw, float* out)
{
    double scale_h = (double)base_h / gh;
    double scale_w = (double)base_w / gw;
    double supp_h  = (scale_h >= 1.0) ? scale_h : 1.0;
    double supp_w  = (scale_w >= 1.0) ? scale_w : 1.0;
    double invsh   = (scale_h >= 1.0) ? 1.0 / scale_h : 1.0;
    double invsw   = (scale_w >= 1.0) ? 1.0 / scale_w : 1.0;
    double wy[32], wx[32];

    for (int oy = 0; oy < gh; oy++) {
        double cy = scale_h * (oy + 0.5);
        int ymin = (int)floor(cy - supp_h + 0.5);
        if (ymin < 0) ymin = 0;
        int ymax = (int)floor(cy + supp_h + 0.5);
        if (ymax > base_h) ymax = base_h;
        double tw = 0.0;
        for (int j = ymin; j < ymax; j++) {
            double d = (j + 0.5 - cy) * invsh;
            double w = 1.0 - fabs(d);
            if (w < 0.0) w = 0.0;
            wy[j - ymin] = w;
            tw += w;
        }
        if (tw != 0.0)
            for (int j = ymin; j < ymax; j++) wy[j - ymin] /= tw;

        for (int ox = 0; ox < gw; ox++) {
            double cx = scale_w * (ox + 0.5);
            int xmin = (int)floor(cx - supp_w + 0.5);
            if (xmin < 0) xmin = 0;
            int xmax = (int)floor(cx + supp_w + 0.5);
            if (xmax > base_w) xmax = base_w;
            double twx = 0.0;
            for (int j = xmin; j < xmax; j++) {
                double d = (j + 0.5 - cx) * invsw;
                double w = 1.0 - fabs(d);
                if (w < 0.0) w = 0.0;
                wx[j - xmin] = w;
                twx += w;
            }
            if (twx != 0.0)
                for (int j = xmin; j < xmax; j++) wx[j - xmin] /= twx;

            float* op = out + ((size_t)oy * gw + ox) * D;
            for (int d = 0; d < D; d++) {
                double acc = 0.0;
                for (int a = ymin; a < ymax; a++) {
                    const float* row = base + ((size_t)a * base_w + xmin) * D + d;
                    double wa = wy[a - ymin];
                    for (int b = xmin; b < xmax; b++) {
                        acc += wa * wx[b - xmin] * row[(b - xmin) * D];
                    }
                }
                op[d] = (float)acc;
            }
        }
    }
}

int init_lfm2_vl_vision(rknn_lfm2_vl_vision_context* vision_ctx,
                        const char* model_path,
                        const char* weight_path,
                        uint32_t core_mask)
{
    int ret;
    rknn3_context ctx = 0;
    rknn3_input_output_num io_num;
    rknn3_tensor* inputs = NULL;
    rknn3_tensor* outputs = NULL;
    // Declared up-front (as heap pointers, not VLAs) so the early `goto fail`
    // paths never jump across an initialized declaration.
    rknn3_tensor_attr* input_attrs = NULL;
    rknn3_tensor_attr* output_attrs = NULL;
    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1;

    // pos_emb external-input defaults
    vision_ctx->pos_emb_base = NULL;
    vision_ctx->grid_base    = 16;
    vision_ctx->pos_emb_dim  = 1152;
    vision_ctx->pos_emb_rows = 1024;
    vision_ctx->out_rows     = 256;
    vision_ctx->out_dim      = 2048;
    vision_ctx->out_cols     = 16;   // 256 = 16x16 (after 32x32 unshuffle)
    vision_ctx->n_valid_embeds = 0;
    vision_ctx->pos_emb_base_path[0] = '\0';

    // RKNN Init
    rknn3_init_extend init_extend = make_rknn3_init_extend("VisionInit");
    ret = rknn3_init(&ctx, init_extend.device_id ? &init_extend : NULL);
    if (ret < 0) {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        goto fail;
    }

    // Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        goto fail;
    }

    // Get Model Input Output Number
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        goto fail;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    input_attrs = (rknn3_tensor_attr*)calloc(io_num.n_input, sizeof(rknn3_tensor_attr));
    output_attrs = (rknn3_tensor_attr*)calloc(io_num.n_output, sizeof(rknn3_tensor_attr));
    if (input_attrs == NULL || output_attrs == NULL) {
        printf("malloc input/output attr array fail!\n");
        ret = -1;
        goto fail;
    }
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0) {
            printf("rknn_query fail! ret=%d\n", ret);
            goto fail;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0) {
            printf("rknn_query fail! ret=%d\n", ret);
            goto fail;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    inputs = (rknn3_tensor*)calloc(io_num.n_input, sizeof(rknn3_tensor));
    outputs = (rknn3_tensor*)calloc(io_num.n_output, sizeof(rknn3_tensor));
    if (inputs == NULL || outputs == NULL) {
        printf("malloc input/output tensor array fail!\n");
        ret = -1;
        goto fail;
    }
    for (int i = 0; i < io_num.n_input; i++) {
        printf("[vision io] input[%d] create_mem: core_id=%d, aligned_size=%llu (0x%llx)\n",
               i,
               input_attrs[i].core_id,
               (unsigned long long)input_attrs[i].aligned_size,
               (unsigned long long)input_attrs[i].aligned_size);
        inputs[i].mem = rknn3_create_mem(ctx, input_attrs[i].aligned_size,
                                         input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (inputs[i].mem == NULL) {
            printf("rknn3_create_mem input[%d] fail!\n", i);
            ret = -1;
            goto fail;
        }
        inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (inputs[i].attr == NULL) {
            printf("malloc input[%d] attr fail!\n", i);
            ret = -1;
            goto fail;
        }
        memcpy(inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < io_num.n_output; i++) {
        printf("[vision io] output[%d] create_mem: core_id=%d, aligned_size=%llu (0x%llx)\n",
               i,
               output_attrs[i].core_id,
               (unsigned long long)output_attrs[i].aligned_size,
               (unsigned long long)output_attrs[i].aligned_size);
        outputs[i].mem = rknn3_create_mem(ctx, output_attrs[i].aligned_size,
                                          output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (outputs[i].mem == NULL) {
            printf("rknn3_create_mem output[%d] fail!\n", i);
            ret = -1;
            goto fail;
        }
        outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (outputs[i].attr == NULL) {
            printf("malloc output[%d] attr fail!\n", i);
            ret = -1;
            goto fail;
        }
        memcpy(outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    vision_ctx->inputs = inputs;
    vision_ctx->outputs = outputs;
    vision_ctx->rknn_ctx = ctx;
    vision_ctx->io_num = io_num;

    // NaFlex: input[0] pixel_values is (1, num_patches, patch_flat)
    vision_ctx->num_patches = 1024;
    vision_ctx->patch_flat = 768;
    vision_ctx->model_channel = 3;
    vision_ctx->model_width = LFM2_VL_VISION_MODEL_WIDTH;
    vision_ctx->model_height = LFM2_VL_VISION_MODEL_HEIGHT;
    if (io_num.n_input >= 1 && input_attrs[0].n_dims == 3) {
        vision_ctx->num_patches = input_attrs[0].shape[1];
        vision_ctx->patch_flat = input_attrs[0].shape[2];
        if (input_attrs[0].n_dims >= 1) {
            vision_ctx->model_channel = 3;
        }
    } else {
        printf("warning: input[0] is not a 3D patch-sequence tensor (n_dims=%d), "
               "using defaults num_patches=%d patch_flat=%d\n",
               input_attrs[0].n_dims, vision_ctx->num_patches, vision_ctx->patch_flat);
    }
    printf("NaFlex input: num_patches=%d, patch_flat=%d\n",
           vision_ctx->num_patches, vision_ctx->patch_flat);

    // Preprocess params (from vision_config.json, overrides model_width/height)
    load_vision_preprocess_config(vision_ctx);

    // Load base position embedding (16x16xD fp32) for on-device antialias resize
    {
        const char* env_pe = getenv("LFM2_VL_POS_EMB_BASE");
        const char* pe_path = env_pe ? env_pe : NULL;
        if (!pe_path) {
            // default: <model_dir>/pos_emb_base.bin; if model_path has no dir
            // part, fall back to plain "pos_emb_base.bin" (cwd)
            snprintf(vision_ctx->pos_emb_base_path, sizeof(vision_ctx->pos_emb_base_path),
                     "%s", model_path);
            char* slash = strrchr(vision_ctx->pos_emb_base_path, '/');
            if (slash) {
                *slash = '\0';
                strncat(vision_ctx->pos_emb_base_path, "/pos_emb_base.bin",
                        sizeof(vision_ctx->pos_emb_base_path) - strlen(vision_ctx->pos_emb_base_path) - 1);
            } else {
                snprintf(vision_ctx->pos_emb_base_path, sizeof(vision_ctx->pos_emb_base_path),
                         "pos_emb_base.bin");
            }
            pe_path = vision_ctx->pos_emb_base_path;
        }
        FILE* fp = fopen(pe_path, "rb");
        if (!fp) {
            printf("[Vision][WARN] cannot open pos_emb_base %s, pos_emb will be zeros!\n", pe_path);
        } else {
            size_t n = (size_t)vision_ctx->grid_base * vision_ctx->grid_base * vision_ctx->pos_emb_dim;
            vision_ctx->pos_emb_base = (float*)malloc(n * sizeof(float));
            if (!vision_ctx->pos_emb_base) {
                printf("[Vision][WARN] malloc pos_emb_base fail\n");
            } else {
                size_t rd = fread(vision_ctx->pos_emb_base, sizeof(float), n, fp);
                if (rd != n)
                    printf("[Vision][WARN] pos_emb_base read %zu/%zu, maybe dtype mismatch (expect fp32)\n", rd, n);
                else
                    printf("[Vision] pos_emb_base loaded: %s (%zu floats)\n", pe_path, n);
            }
            fclose(fp);
        }
    }

    printf("output layout: %d, using shape from output attrs\n", output_attrs[0].layout);
    vision_ctx->embeds_shape = vision_ctx->outputs[0].attr->shape;
    vision_ctx->embeds_ndims = vision_ctx->outputs[0].attr->n_dims;

    return ret;

fail:
    // Release partially initialized resources (mirrors release_lfm2_vl_vision).
    if (input_attrs != NULL) {
        free(input_attrs);
    }
    if (output_attrs != NULL) {
        free(output_attrs);
    }
    if (inputs != NULL) {
        for (int i = 0; i < io_num.n_input; i++) {
            if (inputs[i].mem != NULL) {
                rknn3_destroy_mem(ctx, inputs[i].mem);
            }
            if (inputs[i].attr != NULL) {
                free(inputs[i].attr);
            }
        }
        free(inputs);
    }
    if (outputs != NULL) {
        for (int i = 0; i < io_num.n_output; i++) {
            if (outputs[i].mem != NULL) {
                rknn3_destroy_mem(ctx, outputs[i].mem);
            }
            if (outputs[i].attr != NULL) {
                free(outputs[i].attr);
            }
        }
        free(outputs);
    }
    if (ctx != 0) {
        rknn3_destroy(ctx);
    }
    vision_ctx->inputs = NULL;
    vision_ctx->outputs = NULL;
    vision_ctx->rknn_ctx = 0;
    return ret;
}

int release_lfm2_vl_vision(rknn_lfm2_vl_vision_context* vision_ctx)
{
    if (vision_ctx->pos_emb_base) {
        free(vision_ctx->pos_emb_base);
        vision_ctx->pos_emb_base = NULL;
    }
    // inputs/outputs may be NULL if init failed partway; guard before use.
    if (vision_ctx->inputs != NULL) {
        for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
            if (vision_ctx->inputs[i].mem) {
                rknn3_destroy_mem(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem);
            }
            if (vision_ctx->inputs[i].attr != NULL) {
                free(vision_ctx->inputs[i].attr);
                vision_ctx->inputs[i].attr = NULL;
            }
        }
        free(vision_ctx->inputs);
        vision_ctx->inputs = NULL;
    }
    if (vision_ctx->outputs != NULL) {
        for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
            if (vision_ctx->outputs[i].mem) {
                rknn3_destroy_mem(vision_ctx->rknn_ctx, vision_ctx->outputs[i].mem);
            }
            if (vision_ctx->outputs[i].attr != NULL) {
                free(vision_ctx->outputs[i].attr);
                vision_ctx->outputs[i].attr = NULL;
            }
        }
        free(vision_ctx->outputs);
        vision_ctx->outputs = NULL;
    }
    if (vision_ctx->rknn_ctx != 0) {
        rknn3_destroy(vision_ctx->rknn_ctx);
        vision_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_lfm2_vl_vision(rknn_lfm2_vl_vision_context* vision_ctx,
                             image_buffer_t* img,
                             float16* img_embeds)
{
    if ((!vision_ctx) || (!img)) {
        printf("vision_ctx or img is NULL");
        return -1;
    }

    int ret;
    float* pixel_values = NULL;
    int32_t* mask = NULL;
    float* pos_emb = NULL;
    float* resized_pos = NULL;
    // Declared up-front so the early `goto out` paths never jump across any
    // initialized declaration (C++ forbids jumping over such declarations).
    int grid_h = 0, grid_w = 0, valid_patches = 0;
    int static_grid_w = 1;
    const unsigned char* src = NULL;
    int64_t spatial_shapes[2] = {0, 0};
    image_buffer_t dst_img;
    memset(&dst_img, 0, sizeof(image_buffer_t));

    int ps = vision_ctx->encoder_patch_size;
    int total_factor = ps * vision_ctx->downsample_factor;
    long long min_pixels = (long long)vision_ctx->min_image_tokens * total_factor * total_factor;
    long long max_pixels = (long long)vision_ctx->max_image_tokens * total_factor * total_factor;

    // The b13 vision graph reshapes the padded patch sequence to a fixed
    // square before the 2x2 projector. Keep the real grid's row stride in
    // that padded square instead of writing valid patches contiguously.
    while ((long long)static_grid_w * static_grid_w < vision_ctx->num_patches) {
        static_grid_w++;
    }
    if ((long long)static_grid_w * static_grid_w != vision_ctx->num_patches) {
        printf("num_patches %d is not a square\n", vision_ctx->num_patches);
        return -1;
    }
    int static_out_grid = vision_ctx->out_rows / vision_ctx->out_cols;

    // 1) smart_resize to the per-image dynamic grid (matches HF processor,
    //    grid_h/grid_w depend on the image aspect ratio; graph is static so
    //    the grid only affects preprocessing + pos_emb resize + output crop)
    int new_h = 0, new_w = 0;
    smart_resize(img->height, img->width, total_factor, min_pixels, max_pixels, &new_h, &new_w);

    // Clamp both grid sides to the compiled static graph (static_grid_w x
    // static_grid_w). smart_resize bounds only the total pixel budget, so a
    // wide aspect image (e.g. 1024x640 -> grid 24x40) can stay within
    // max_num_patches yet exceed the graph on one side; scale the aspect
    // ratio down until both sides fit.
    if (new_h / ps > static_grid_w || new_w / ps > static_grid_w) {
        int gh = new_h / ps;
        int gw = new_w / ps;
        int gmax = gh > gw ? gh : gw;
        double k = (double)static_grid_w / gmax;
        int ch = (int)(floor(gh * k)) * ps;
        int cw = (int)(floor(gw * k)) * ps;
        if (ch < ps) ch = ps;
        if (cw < ps) cw = ps;
        printf("[Vision] clamp grid %dx%d -> %dx%d (static graph %dx%d)\n",
               gh, gw, ch / ps, cw / ps, static_grid_w, static_grid_w);
        new_h = ch;
        new_w = cw;
    }

    printf("[Vision] smart_resize %dx%d -> %dx%d (grid %dx%d)\n",
           img->width, img->height, new_w, new_h, new_h / ps, new_w / ps);

    // 2) resize to (new_w, new_h) RGB888
    dst_img.width = new_w;
    dst_img.height = new_h;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL) {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        ret = -1;
        goto out;
    }

    ret = resize_rgb_antialias(img, &dst_img);
    if (ret < 0) {
        printf("resize_rgb_antialias fail! ret=%d\n", ret);
        goto out;
    }

    // 3) normalize + patchify (unfold 16x16) + pad to max_num_patches
    grid_h = new_h / ps;
    grid_w = new_w / ps;
    valid_patches = grid_h * grid_w;
    if (valid_patches > vision_ctx->max_num_patches) {
        printf("valid_patches %d > max_num_patches %d\n", valid_patches, vision_ctx->max_num_patches);
        ret = -1;
        goto out;
    }
    if (grid_h > static_grid_w || grid_w > static_grid_w ||
        grid_h / 2 > static_out_grid || grid_w / 2 > static_out_grid) {
        printf("grid %dx%d exceeds static vision graph %dx%d\n",
               grid_h, grid_w, static_grid_w, static_grid_w);
        ret = -1;
        goto out;
    }

    pixel_values = (float*)calloc(vision_ctx->num_patches * vision_ctx->patch_flat, sizeof(float));
    if (pixel_values == NULL) {
        printf("malloc pixel_values fail!\n");
        ret = -1;
        goto out;
    }

    src = (const unsigned char*)dst_img.virt_addr; // RGB888 row-major
    for (int gy = 0; gy < grid_h; gy++) {
        for (int gx = 0; gx < grid_w; gx++) {
            int patch_idx = gy * static_grid_w + gx;
            for (int py = 0; py < ps; py++) {
                for (int px = 0; px < ps; px++) {
                    const unsigned char* p = src + ((gy * ps + py) * new_w + gx * ps + px) * 3;
                    int off = patch_idx * vision_ctx->patch_flat + (py * ps + px) * 3;
                    pixel_values[off + 0] = (p[0] * vision_ctx->rescale_factor - vision_ctx->image_mean[0]) / vision_ctx->image_std[0];
                    pixel_values[off + 1] = (p[1] * vision_ctx->rescale_factor - vision_ctx->image_mean[1]) / vision_ctx->image_std[1];
                    pixel_values[off + 2] = (p[2] * vision_ctx->rescale_factor - vision_ctx->image_mean[2]) / vision_ctx->image_std[2];
                }
            }
        }
    }

    // pixel_attention_mask: 1 for valid patches, 0 for padding (fp32 for posemb graph)
    mask = (int32_t*)calloc(vision_ctx->num_patches, sizeof(int32_t));
    if (mask == NULL) {
        printf("malloc mask fail!\n");
        ret = -1;
        goto out;
    }
    for (int gy = 0; gy < grid_h; gy++) {
        for (int gx = 0; gx < grid_w; gx++) {
            mask[gy * static_grid_w + gx] = 1;
        }
    }

    // 3b) build pos_emb: antialias-bilinear resize of base (grid_base x grid_base)
    //     to (grid_h, grid_w), then pad to pos_emb_rows with first-row fill
    //     (exact replica of HF Siglip2VisionEmbeddings.resize_positional_embeddings).
    pos_emb = (float*)calloc((size_t)vision_ctx->pos_emb_rows * vision_ctx->pos_emb_dim, sizeof(float));    if (pos_emb == NULL) {
        printf("malloc pos_emb fail!\n");
        ret = -1;
        goto out;
    }
    resized_pos = (float*)calloc((size_t)vision_ctx->pos_emb_rows * vision_ctx->pos_emb_dim, sizeof(float));
    if (resized_pos == NULL) {
        printf("malloc resized pos_emb fail!\n");
        ret = -1;
        goto out;
    }
    if (vision_ctx->pos_emb_base) {
        aa_bilinear_posemb(vision_ctx->pos_emb_base,
                           vision_ctx->grid_base, vision_ctx->grid_base, vision_ctx->pos_emb_dim,
                           grid_h, grid_w, resized_pos);
        for (int gy = 0; gy < grid_h; gy++) {
            for (int gx = 0; gx < grid_w; gx++) {
                size_t src_idx = (size_t)(gy * grid_w + gx) * vision_ctx->pos_emb_dim;
                size_t dst_idx = (size_t)(gy * static_grid_w + gx) * vision_ctx->pos_emb_dim;
                memcpy(pos_emb + dst_idx, resized_pos + src_idx,
                       (size_t)vision_ctx->pos_emb_dim * sizeof(float));
            }
        }
        // Padding positions are masked, but keep them deterministic.
        // mask has num_patches elements (pos_emb_rows >= num_patches is not
        // enforced), so bound the loop by num_patches.
        int mask_rows = vision_ctx->pos_emb_rows < vision_ctx->num_patches
                            ? vision_ctx->pos_emb_rows : vision_ctx->num_patches;
        for (int r = 0; r < mask_rows; r++) {
            if (!mask[r]) {
                memcpy(pos_emb + (size_t)r * vision_ctx->pos_emb_dim, resized_pos,
                       (size_t)vision_ctx->pos_emb_dim * sizeof(float));
            }
        }
    } else {
        printf("[Vision][WARN] pos_emb_base NULL, pos_emb stays zeros\n");
    }

    // 4) fill inputs (0: pixel_values, 1: mask, 2: pos_emb), dtype-adapted
    if (vision_ctx->inputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16) {
        float16* pv_fp16 = (float16*)malloc(vision_ctx->num_patches * vision_ctx->patch_flat * sizeof(float16));
        if (pv_fp16 == NULL) {
            printf("malloc pv_fp16 fail!\n");
            ret = -1;
            goto out;
        }
        for (int i = 0; i < vision_ctx->num_patches * vision_ctx->patch_flat; i++) {
            pv_fp16[i] = fp32_to_fp16(pixel_values[i]);
        }
        memcpy(vision_ctx->inputs[0].mem->virt_addr, pv_fp16,
               vision_ctx->num_patches * vision_ctx->patch_flat * sizeof(float16));
        free(pv_fp16);
    } else {
        memcpy(vision_ctx->inputs[0].mem->virt_addr, pixel_values,
               vision_ctx->num_patches * vision_ctx->patch_flat * sizeof(float));
    }
    // input[1]: pixel_attention_mask (fp32 for posemb graph)
    if (vision_ctx->io_num.n_input >= 2) {
        if (vision_ctx->inputs[1].attr->dtype == RKNN3_TENSOR_FLOAT32) {
            float* mask_f = (float*)malloc(vision_ctx->num_patches * sizeof(float));
            if (mask_f) {
                for (int i = 0; i < vision_ctx->num_patches; i++) mask_f[i] = (float)mask[i];
                memcpy(vision_ctx->inputs[1].mem->virt_addr, mask_f,
                       vision_ctx->num_patches * sizeof(float));
                free(mask_f);
            }
        } else if (vision_ctx->inputs[1].attr->dtype == RKNN3_TENSOR_INT32) {
            memcpy(vision_ctx->inputs[1].mem->virt_addr, mask,
                   vision_ctx->num_patches * sizeof(int32_t));
        } else {
            float16* mask_fp16 = (float16*)malloc(vision_ctx->num_patches * sizeof(float16));
            if (mask_fp16 == NULL) {
                printf("malloc mask_fp16 fail!\n");
                ret = -1;
                goto out;
            }
            for (int i = 0; i < vision_ctx->num_patches; i++) {
                mask_fp16[i] = fp32_to_fp16((float)mask[i]);
            }
            memcpy(vision_ctx->inputs[1].mem->virt_addr, mask_fp16,
                   vision_ctx->num_patches * sizeof(float16));
            free(mask_fp16);
        }
    }
    // input[2]: pos_emb (external)
    if (vision_ctx->io_num.n_input >= 3) {
        size_t pe_n = (size_t)vision_ctx->pos_emb_rows * vision_ctx->pos_emb_dim;
        if (vision_ctx->inputs[2].attr->dtype == RKNN3_TENSOR_FLOAT16) {
            float16* pe_fp16 = (float16*)malloc(pe_n * sizeof(float16));
            if (pe_fp16 == NULL) {
                printf("malloc pe_fp16 fail!\n");
                ret = -1;
                goto out;
            }
            for (size_t i = 0; i < pe_n; i++) pe_fp16[i] = fp32_to_fp16(pos_emb[i]);
            memcpy(vision_ctx->inputs[2].mem->virt_addr, pe_fp16, pe_n * sizeof(float16));
            free(pe_fp16);
        } else {
            memcpy(vision_ctx->inputs[2].mem->virt_addr, pos_emb, pe_n * sizeof(float));
        }
    }

    // Sync inputs
    for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
        ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_mem_sync input[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }

    // Run
    ret = rknn3_run(vision_ctx->rknn_ctx, vision_ctx->inputs, vision_ctx->io_num.n_input,
                    vision_ctx->outputs, vision_ctx->io_num.n_output);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        goto out;
    }

    // Sync outputs
    for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
        ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
        if (ret != RKNN3_SUCCESS) {
            printf("rknn3_mem_sync output[%d] failed! ret=%d\n", i, ret);
            goto out;
        }
    }

    // Get Output: model returns ALL out_rows (256); extract the valid region
    // (rows r*out_cols+c with r<grid_h/2, c<grid_w/2) into img_embeds compactly.
    {
        int n_valid = (grid_h / 2) * (grid_w / 2);
        int oc = vision_ctx->out_cols;
        if (vision_ctx->outputs[0].attr->dtype == RKNN3_TENSOR_FLOAT16) {
            const float16* op = (const float16*)vision_ctx->outputs[0].mem->virt_addr;
            for (int r = 0; r < grid_h / 2; r++)
                for (int c = 0; c < grid_w / 2; c++) {
                    const float16* src_row = op + ((size_t)(r * oc + c)) * vision_ctx->out_dim;
                    memcpy(img_embeds + ((size_t)(r * (grid_w / 2) + c)) * vision_ctx->out_dim,
                           src_row, (size_t)vision_ctx->out_dim * sizeof(float16));
                }
        } else {
            const float* op = (const float*)vision_ctx->outputs[0].mem->virt_addr;
            for (int r = 0; r < grid_h / 2; r++)
                for (int c = 0; c < grid_w / 2; c++) {
                    const float* src_row = op + ((size_t)(r * oc + c)) * vision_ctx->out_dim;
                    for (int d = 0; d < vision_ctx->out_dim; d++)
                        img_embeds[((size_t)(r * (grid_w / 2) + c)) * vision_ctx->out_dim + d] =
                            fp32_to_fp16(src_row[d]);
                }
        }
        vision_ctx->n_valid_embeds = n_valid;
        vision_ctx->last_grid_h = grid_h;
        vision_ctx->last_grid_w = grid_w;
        printf("[Vision] grid %dx%d -> n_valid_embeds=%d (out rows %d x %d)\n",
               grid_h, grid_w, n_valid, grid_h / 2, grid_w / 2);
    }

out:
    if (pixel_values != NULL) {
        free(pixel_values);
    }
    if (mask != NULL) {
        free(mask);
    }
    if (pos_emb != NULL) {
        free(pos_emb);
    }
    if (resized_pos != NULL) {
        free(resized_pos);
    }
    if (dst_img.virt_addr != NULL) {
        free(dst_img.virt_addr);
    }
    return ret;
}
