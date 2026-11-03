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

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cerrno>

#include "rknn3_api.h"
#include "raft_stereo.h"
#include "image_utils.h"

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char **argv)
{
    if (argc != 6) {
        printf("%s <model_path> <weight_path> <image0_path> <image1_path> <core_mask>\n", argv[0]);
        return RKNN3_ERR_ARGUMENT_INVALID;
    }

    const char *model_path      = argv[1];
    const char *weight_path     = argv[2];
    const char *left_img_path   = argv[3];
    const char *right_img_path  = argv[4];

    char *endptr = nullptr;
    errno = 0;
    unsigned long parsed_mask = strtoul(argv[5], &endptr, 16);
    if (endptr == argv[5] || *endptr != '\0' || errno == ERANGE) {
        fprintf(stderr, "Error: Invalid core_mask '%s'. Expected a hexadecimal string (e.g., 0xFF).\n", argv[5]);
        return EXIT_FAILURE;
    }

    if (parsed_mask > UINT32_MAX) {
        fprintf(stderr, "Error: core_mask out of uint32_t range.\n");
        return EXIT_FAILURE;
    }
    uint32_t core_mask = static_cast<uint32_t>(parsed_mask);

    int ret = RKNN3_SUCCESS, exit_code = RKNN3_SUCCESS;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    printf("--> init model\n");
    ret = init_raft_stereo_model(model_path, weight_path, &rknn_app_ctx, core_mask);
    if (ret) {
        printf("init_raft_stereo_model fail! ret=%d model_path=%s\n", ret, model_path);
        return ret;
    }

    image_buffer_t src_img_l, src_img_r;
    memset(&src_img_l, 0, sizeof(image_buffer_t));
    memset(&src_img_r, 0, sizeof(image_buffer_t));

    ret = read_image(left_img_path, &src_img_l);
    if (ret != 0) {
        exit_code = ret;
        printf("read image fail! ret=%d image_path=%s\n", ret, left_img_path);
        goto release_img;
    }

    ret = read_image(right_img_path, &src_img_r);
    if (ret != 0) {
        exit_code = ret;
        printf("read image fail! ret=%d image_path=%s\n", ret, right_img_path);
        goto release_img;
    }

    printf("--> inference model\n");
    ret = inference_raft_stereo_model(&rknn_app_ctx, &src_img_l, &src_img_r);
    if (ret) {
        exit_code = ret;
        printf("inference_raft_stereo_model fail! ret=%d\n", ret);
        goto release_img;
    }
    printf("--> inference model done\n");

    /* save image after inference */
    ret = save_raft_stereo_output(&rknn_app_ctx);
    if (ret) {
        exit_code = ret;
        printf("save_raft_stereo_output fail! ret=%d\n", ret);
        goto release_img;
    }
    printf("--> save output done\n");

release_img:
    if (src_img_l.virt_addr != NULL)
        free(src_img_l.virt_addr);

    if (src_img_r.virt_addr != NULL)
        free(src_img_r.virt_addr);

    ret = release_raft_stereo_model(&rknn_app_ctx);
    if (ret) {
        printf("release_raft_stereo_model fail! ret=%d\n", ret);
        if (exit_code == RKNN3_SUCCESS) {
            exit_code = ret;
        }
    }

    return exit_code;
}
