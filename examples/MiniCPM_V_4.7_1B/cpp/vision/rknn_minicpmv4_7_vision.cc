// Copyright (c) 2026 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <string>

#include "rknn_minicpmv4_7_vision.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"


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
           attrs->name, attrs->n_dims, shape_str.c_str(), stride_str.c_str(), attrs->aligned_size, rknn3_get_layout_string(attrs->layout),
           rknn3_get_type_string(attrs->dtype), attrs->core_id, rknn3_get_qnt_type_string(attrs->qnt_type));
}


/* ---------------------------------------------------------------------------
 * Slice / crop geometry
 * ------------------------------------------------------------------------- */
int minicpmv4_7_get_sliced_grid(int img_h, int img_w, int max_slice_nums,
                                int scale_resolution, int* rows, int* cols)
{
    if (rows) *rows = 0;
    if (cols) *cols = 0;
    if (img_h <= 0 || img_w <= 0 || scale_resolution <= 0 || max_slice_nums < 2) {
        return 0;
    }

    const double log_ratio = log((double)img_w / (double)img_h);
    const double area_ratio =
        (double)img_h * (double)img_w / ((double)scale_resolution * (double)scale_resolution);

    int multiple = (int)ceil(area_ratio);
    if (multiple > max_slice_nums) {
        multiple = max_slice_nums;
    }
    if (multiple <= 1) {
        return 0;   /* area <= scale_resolution^2 -> no slicing */
    }

    int    best_r = 1, best_c = 1;
    double best_err = 1e30;

    for (int num_slices = multiple - 1; num_slices <= multiple + 1; ++num_slices) {
        if (num_slices == 1 || num_slices > max_slice_nums) {
            continue;
        }
        for (int r = 1; r <= num_slices; ++r) {
            if (num_slices % r != 0) {
                continue;
            }
            int    c   = num_slices / r;
            double err = fabs(log_ratio - log((double)c / (double)r));
            if (err < best_err || (err == best_err && r > best_r)) {
                best_r = r;
                best_c = c;
                best_err = err;
            }
        }
    }

    if (rows) *rows = best_r;
    if (cols) *cols = best_c;
    return 1;
}

int minicpmv4_7_build_crop_boxes(int img_h, int img_w, int max_slice_nums,
                                 int scale_resolution, image_rect_t* boxes, int max_boxes)
{
    if (!boxes || max_boxes < 1 || img_h <= 0 || img_w <= 0) {
        return -1;
    }

    int rows = 0, cols = 0;
    minicpmv4_7_get_sliced_grid(img_h, img_w, max_slice_nums, scale_resolution, &rows, &cols);

    int n = 0;
    /* crop 0: thumbnail.  Upstream keeps the resized source image as the
     * thumbnail; here every crop is resized to the model input by convert_image,
     * so the thumbnail is the whole image. */
    boxes[n].left   = 0;
    boxes[n].top    = 0;
    boxes[n].right  = img_w - 1;      /* inclusive */
    boxes[n].bottom = img_h - 1;
    n++;

    if (rows < 1 || cols < 1) {
        return n;
    }

    /* Slices: equal partition of the original image.
     * Upstream resizes the image to (cell_h*rows, cell_w*cols) and then cuts
     * equal (cell_h, cell_w) cells; the cell boundaries map back to
     * (r*H/rows, c*W/cols) of the original image, so an equal partition of the
     * original is equivalent (only the resampling order differs). */
    for (int r = 0; r < rows; ++r) {
        int top    = (int)((int64_t)r * img_h / rows);
        int bottom = (int)((int64_t)(r + 1) * img_h / rows) - 1;
        for (int c = 0; c < cols; ++c) {
            if (n >= max_boxes) {
                printf("[crop] crop count exceeds max_boxes=%d\n", max_boxes);
                return n;
            }
            int left  = (int)((int64_t)c * img_w / cols);
            int right = (int)((int64_t)(c + 1) * img_w / cols) - 1;
            boxes[n].left   = left;
            boxes[n].top    = top;
            boxes[n].right  = right;
            boxes[n].bottom = bottom;
            n++;
        }
    }

    return n;
}

uint64_t minicpmv4_7_vision_embeds_per_crop(const rknn_minicpmv4_7_vision_context* vision_ctx)
{
    if (!vision_ctx || !vision_ctx->embeds_shape) {
        return 0;
    }
    uint64_t n = 1;
    for (uint32_t i = 0; i < vision_ctx->embeds_ndims; ++i) {
        n *= (uint64_t)vision_ctx->embeds_shape[i];
    }
    return n;
}

/* ---------------------------------------------------------------------------
 * Canvas M-RoPE position ids, computed from a token's OFFSET inside the visual group:
 *
 *   d = 0                    <image>
 *   d = 1 .. sq*sq           thumbnail pads  (linspace-stretched over the canvas)
 *   d = sq*sq + 1            </image>
 *   then, for row r = 0..rows-1 and col c = 0..cols-1:
 *       <slice> + sq*sq pads + </slice>     (block size B = sq*sq + 2)
 *     with one "\n" after every row except the last
 *   total = B + n_slices*B + (rows - 1)
 *   d >= total               trailing text (also covers generated tokens)
 *
 * Being offset-based, this needs no scan of the whole visual group, so it also
 * works when the runtime splits the prefill into chunks (mrope_id_input is
 * [128,3] => at most 128 tokens per forward).  Only the absolute index of
 * <image> has to be carried across chunks (`geom->canvas_base`).
 * ------------------------------------------------------------------------- */
static int py_round_half_even(double x)
{
    const double f = floor(x);
    const double d = x - f;
    if (d < 0.5) return (int)f;
    if (d > 0.5) return (int)(f + 1.0);
    const long fi = (long)f;
    return (int)((fi % 2 == 0) ? fi : fi + 1);
}

/* linspace(0, span-1, n).round() -> thumbnail row/col index inside the canvas */
static int canvas_lin_idx(int i, int n, int span)
{
    if (n <= 1) return 0;
    return py_round_half_even((double)i * (double)(span - 1) / (double)(n - 1));
}

static void canvas_pos_from_delta(int d, const minicpmv4_7_canvas_geom_t* gm, int base,
                                  int32_t* T, int32_t* H, int32_t* W)
{
    const int sq     = gm->crop_sq;
    const int B      = sq * sq + 2;
    const int rows   = (gm->n_slices > 0) ? gm->rows : 1;
    const int cols   = (gm->n_slices > 0) ? gm->cols : 1;
    const int canvas_h = rows * sq;
    const int canvas_w = cols * sq;
    const int span     = (canvas_h > canvas_w) ? canvas_h : canvas_w;
    const int total    = B + gm->n_slices * B + ((rows > 1) ? (rows - 1) : 0);

    if (d < B) {
        if (d == 0) {
            /* <image> halo: one step outside the canvas in H/W, same frame in T */
            *T = base; *H = *W = (base > 0) ? base - 1 : 0;
        } else if (d <= sq * sq) {
            /* thumbnail: low-res view of the whole canvas -> linspace stretch when
             * slices exist, plain grid otherwise */
            const int j = d - 1;
            *T = base;
            if (gm->n_slices > 0) {
                *H = base + canvas_lin_idx(j / sq, sq, canvas_h);
                *W = base + canvas_lin_idx(j % sq, sq, canvas_w);
            } else {
                *H = base + j / sq;
                *W = base + j % sq;
            }
        } else {
            /* </image> halo: bottom-right outside the slice canvas */
            *T = base; *H = base + canvas_h; *W = base + canvas_w;
        }
        return;
    }

    if (d >= total) {
        /* trailing text continues after the whole canvas */
        const int v = base + span + 1 + (d - total);
        *T = *H = *W = v;
        return;
    }

    /* inside the slice region */
    const int d2         = d - B;
    const int row_stride = cols * B + 1;              /* row blocks + its "\n" */
    const int r          = d2 / row_stride;
    const int off_in_row = d2 - r * row_stride;

    if (off_in_row >= cols * B) {
        /* "\n": right-outside of the row that just ended */
        *T = base;
        *H = base + (r + 1) * sq - 1;
        *W = base + cols * sq;
        return;
    }

    const int c     = off_in_row / B;
    const int s_off = off_in_row - c * B;
    const int h_off = r * sq;
    const int w_off = c * sq;

    *T = base;
    if (s_off == 0) {
        /* <slice>: top-left corner of the slice cell */
        *H = base + h_off;
        *W = base + w_off;
    } else if (s_off <= sq * sq) {
        /* slice visual tokens */
        const int j = s_off - 1;
        *H = base + h_off + j / sq;
        *W = base + w_off + j % sq;
    } else {
        /* </slice>: bottom-right corner of the slice cell */
        *H = base + h_off + sq - 1;
        *W = base + w_off + sq - 1;
    }
}

int minicpmv4_7_canvas_fill(const int32_t* tk, int n_tokens, int first_pos,
                           minicpmv4_7_canvas_geom_t* gm,
                           int32_t* dst, int cap)
{
    if (!tk || !gm || !dst || n_tokens <= 0 || cap <= 0 || gm->crop_sq <= 0) {
        return -1;
    }

    /* Locate <image> once: it may arrive in any chunk of the prefill. */
    if (gm->canvas_base < 0) {
        for (int k = 0; k < n_tokens; ++k) {
            if (tk[k] == gm->im_start_id) {
                gm->canvas_base = first_pos + k;
                break;
            }
        }
    }
    if (gm->canvas_base < 0) {
        return -1;                       /* no image seen yet -> 1-D positions */
    }

    const int lim = (n_tokens < cap) ? n_tokens : cap;
    for (int k = 0; k < lim; ++k) {
        const int absi = first_pos + k;
        const int d    = absi - gm->canvas_base;
        int32_t T, H, W;
        if (d < 0) {
            T = H = W = absi;            /* text before the image */
        } else {
            canvas_pos_from_delta(d, gm, gm->canvas_base, &T, &H, &W);
        }
        dst[3 * k + 0] = T;
        dst[3 * k + 1] = H;
        dst[3 * k + 2] = W;
    }
    return 0;
}


int init_minicpmv4_7_vision(rknn_minicpmv4_7_vision_context* vision_ctx, const char* model_path, const char* weight_path, uint32_t core_mask)
{
    int ret;
    rknn3_context ctx = 0;
    rknn3_config config;
    memset(&config, 0, sizeof(config));
    config.run_core_mask = core_mask;
    config.user_mem_internal = 1; // 使用用户管理的internal内存

    // RKNN Init
    ret = rknn3_init(&ctx, NULL);
    if (ret < 0)
    {
        printf("rknn_init fail ret=%d\n", ret);
        return ret;
    }
    // Record it immediately so release_minicpmv4_7_vision() can destroy the
    // context on any later failure path.
    vision_ctx->rknn_ctx = ctx;

    // Load RKNN Model
    ret = rknn3_load_model_from_path(ctx, model_path, weight_path);
    if (ret < 0) {
        printf("rknn_load_model failed! ret=%d\n", ret);
        return ret;
    }

    //Init RKNN Model
    ret = rknn3_model_init(ctx, &config);
    if (ret < 0) {
        printf("rknn_model_init failed! ret=%d\n", ret);
        return ret;
    }

    // Get Model Input Output Number
    rknn3_input_output_num io_num;
    ret = rknn3_query(ctx, RKNN3_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_query fail! ret=%d\n", ret);
        return ret;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn3_tensor_attr input_attrs[io_num.n_input];
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn3_tensor_attr output_attrs[io_num.n_output];
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn3_query(ctx, RKNN3_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_query fail! ret=%d\n", ret);
            return ret;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    vision_ctx->inputs = (rknn3_tensor*)calloc(io_num.n_input, sizeof(rknn3_tensor));
    if (vision_ctx->inputs == NULL) {
        printf("calloc vision inputs fail!\n");
        return -1;
    }
    vision_ctx->outputs = (rknn3_tensor*)calloc(io_num.n_output, sizeof(rknn3_tensor));
    if (vision_ctx->outputs == NULL) {
        printf("calloc vision outputs fail!\n");
        return -1;
    }
    vision_ctx->io_num = io_num;
    for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
        vision_ctx->inputs[i].mem  = rknn3_create_mem(ctx, input_attrs[i].aligned_size, input_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (vision_ctx->inputs[i].mem == NULL) {
            printf("create vision input[%d] mem fail!\n", i);
            return -1;
        }
        vision_ctx->inputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (vision_ctx->inputs[i].attr == NULL) {
            printf("malloc vision input[%d] attr fail!\n", i);
            return -1;
        }
        memcpy(vision_ctx->inputs[i].attr, &(input_attrs[i]), sizeof(rknn3_tensor_attr));
    }
    for (int i = 0; i < vision_ctx->io_num.n_output; i++) {
        vision_ctx->outputs[i].mem  = rknn3_create_mem(ctx, output_attrs[i].aligned_size, output_attrs[i].core_id, RKNN3_FLAG_MEMORY_CACHEABLE);
        if (vision_ctx->outputs[i].mem == NULL) {
            printf("create vision output[%d] mem fail!\n", i);
            return -1;
        }
        vision_ctx->outputs[i].attr = (rknn3_tensor_attr*)malloc(sizeof(rknn3_tensor_attr));
        if (vision_ctx->outputs[i].attr == NULL) {
            printf("malloc vision output[%d] attr fail!\n", i);
            return -1;
        }
        memcpy(vision_ctx->outputs[i].attr, &(output_attrs[i]), sizeof(rknn3_tensor_attr));
    }

    if (input_attrs[0].layout == RKNN3_TENSOR_NCHW)
    {
        printf("model is NCHW input layout\n");
        vision_ctx->model_channel = input_attrs[0].shape[1];
        vision_ctx->model_height = input_attrs[0].shape[2];
        vision_ctx->model_width = input_attrs[0].shape[3];
        printf("input image height=%d, input image width=%d, input image channel=%d\n",
                vision_ctx->model_height, vision_ctx->model_width, vision_ctx->model_channel);
    }
    else if (input_attrs[0].layout == RKNN3_TENSOR_NHWC)
    {
        printf("model is NHWC input layout\n");
        vision_ctx->model_channel = input_attrs[0].shape[3];
        vision_ctx->model_height = input_attrs[0].shape[1];
        vision_ctx->model_width = input_attrs[0].shape[2];
        printf("input image height=%d, input image width=%d, input image channel=%d\n",
                vision_ctx->model_height, vision_ctx->model_width, vision_ctx->model_channel);

    }
    else 
    {
        printf("model is not NHWC/NCHW input layout, model input error!\n");
        return -1;
    }

    if (output_attrs[0].layout == RKNN3_TENSOR_UNDEFINED)
    {
        printf("model is UNDEFINED output layout\n");
        vision_ctx->embeds_shape = vision_ctx->outputs[0].attr->shape;
        vision_ctx->embeds_ndims = vision_ctx->outputs[0].attr->n_dims;
    }
    else
    {
        printf("model is not UNDEFINED output layout, model output error!\n");
        return -1;
    }

    return ret;
}

int release_minicpmv4_7_vision(rknn_minicpmv4_7_vision_context* vision_ctx)
{
    if (vision_ctx->inputs != NULL) {
        for (int i = 0; i < vision_ctx->io_num.n_input; i++) {
            if (vision_ctx->inputs[i].mem) {
                rknn3_destroy_mem(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem);
                vision_ctx->inputs[i].mem = NULL;
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
                vision_ctx->outputs[i].mem = NULL;
            }
            if (vision_ctx->outputs[i].attr != NULL) {
                free(vision_ctx->outputs[i].attr);
                vision_ctx->outputs[i].attr = NULL;
            }
        }
        free(vision_ctx->outputs);
        vision_ctx->outputs = NULL;
    }
    if (vision_ctx->rknn_ctx != 0)
    {
        rknn3_destroy(vision_ctx->rknn_ctx);
        vision_ctx->rknn_ctx = 0;
    }
    vision_ctx->io_num.n_input = 0;
    vision_ctx->io_num.n_output = 0;
    return 0;
}

/* Encode n_crops crops with the same static single-crop model.
 *
 * Why this is exact: inside the ViT, crops are independent segments (attention is
 * block-diagonal through cu_seqlens), and every crop is resized to the same model
 * input size, so all per-crop constants (patch grid / position ids / window index)
 * are identical.  Re-running the single-crop model is therefore equivalent to a
 * batched run; only the launch overhead is paid N times.
 */
int inference_minicpmv4_7_vision_multi(rknn_minicpmv4_7_vision_context* vision_ctx,
                                       image_buffer_t* src,
                                       image_rect_t* boxes, int n_crops,
                                       float16* embeds_out)
{
    if ((!vision_ctx) || (!src) || (!boxes) || n_crops <= 0 || (!embeds_out))
    {
        printf("vision_ctx/src/boxes/embeds_out is NULL or n_crops<=0\n");
        return -1;
    }

    const uint64_t per_crop = minicpmv4_7_vision_embeds_per_crop(vision_ctx);
    if (per_crop == 0)
    {
        printf("vision output tensor shape is empty\n");
        return -1;
    }

    int ret = 0;

    // Pre Process buffer (allocated once, reused for every crop)
    image_buffer_t dst_img;
    memset(&dst_img, 0, sizeof(image_buffer_t));
    dst_img.width     = vision_ctx->model_width;
    dst_img.height    = vision_ctx->model_height;
    dst_img.format    = IMAGE_FORMAT_RGB888;
    dst_img.size      = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char *)malloc(dst_img.size);
    if (dst_img.virt_addr == NULL)
    {
        printf("malloc buffer size:%d fail!\n", dst_img.size);
        return -1;
    }

    for (int c = 0; c < n_crops; ++c)
    {
        image_rect_t* box = &boxes[c];
        if (box->right < box->left || box->bottom < box->top)
        {
            printf("crop %d has an empty rect [%d,%d,%d,%d]\n",
                   c, box->left, box->top, box->right, box->bottom);
            ret = -1;
            break;
        }

        /* convert_image crops the (inclusive) src rect and scales it to dst */
        ret = convert_image(src, &dst_img, box, NULL, 0);
        if (ret < 0)
        {
            printf("convert_image fail! crop=%d ret=%d\n", c, ret);
            break;
        }

        // Set Input Data
        memcpy(vision_ctx->inputs[0].mem->virt_addr, (uint8_t*)dst_img.virt_addr, dst_img.size);

        // Sync Inputs
        for (int i = 0; i < vision_ctx->io_num.n_input; i++)
        {
            ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->inputs[i].mem, RKNN3_MEMORY_SYNC_TO_DEVICE);
            if (ret != RKNN3_SUCCESS)
            {
                printf("rknn3_mem_sync input[%d] fail! ret=%d\n", i, ret);
                break;
            }
        }
        if (ret != RKNN3_SUCCESS)
        {
            break;
        }

        // Run
        ret = rknn3_run(vision_ctx->rknn_ctx, vision_ctx->inputs, vision_ctx->io_num.n_input,
                        vision_ctx->outputs, vision_ctx->io_num.n_output);
        if (ret < 0)
        {
            printf("rknn_run fail! ret=%d crop=%d\n", ret, c);
            break;
        }

        // Sync Outputs
        for (int i = 0; i < vision_ctx->io_num.n_output; i++)
        {
            ret = rknn3_mem_sync(vision_ctx->rknn_ctx, vision_ctx->outputs[i].mem, RKNN3_MEMORY_SYNC_FROM_DEVICE);
            if (ret != RKNN3_SUCCESS)
            {
                printf("rknn3_mem_sync output[%d] fail! ret=%d\n", i, ret);
                break;
            }
        }
        if (ret != RKNN3_SUCCESS)
        {
            break;
        }

        // Get Output -> concatenate in crop order
        memcpy(embeds_out + (uint64_t)c * per_crop,
               (float16*)vision_ctx->outputs[0].mem->virt_addr,
               per_crop * sizeof(float16));
    }

    free(dst_img.virt_addr);

    if (ret == 0)
    {
        printf("[vision] encoded %d crop(s), %llu elem/crop -> %llu tokens\n",
               n_crops, (unsigned long long)per_crop,
               (unsigned long long)(per_crop / 1024));
    }

    return ret;
}
