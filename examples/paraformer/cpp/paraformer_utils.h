// Copyright (c) 2026 Rockchip Electronics Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

#ifndef RKNN_PARAFORMER_UTILS_H
#define RKNN_PARAFORMER_UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/** Converts a contiguous FP32 array to IEEE 754 half-precision storage. */
static inline void paraformer_fp32_to_fp16(const float* source, uint16_t* destination, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        uint32_t bits = 0;
        uint32_t sign = 0;
        uint32_t mantissa = 0;
        int32_t exponent = 0;
        memcpy(&bits, source + index, sizeof(bits));
        sign = (bits >> 16) & 0x8000;
        exponent = (int32_t)((bits >> 23) & 0xff) - 127;
        mantissa = bits & 0x7fffff;
        if (exponent <= -15) {
            destination[index] = (uint16_t)sign;
        } else if (exponent >= 16) {
            destination[index] = (uint16_t)(sign | 0x7c00);
        } else {
            destination[index] = (uint16_t)(sign | ((uint32_t)(exponent + 15) << 10) | (mantissa >> 13));
        }
    }
}

/** Converts IEEE 754 half-precision storage to a contiguous FP32 array. */
static inline void paraformer_fp16_to_fp32(const uint16_t* source, float* destination, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        uint16_t half = source[index];
        uint32_t sign = (uint32_t)(half & 0x8000) << 16;
        uint32_t exponent = (half & 0x7c00) >> 10;
        uint32_t mantissa = (uint32_t)(half & 0x03ff) << 13;
        uint32_t bits = 0;
        if (exponent == 0) {
            if (mantissa == 0) {
                bits = sign;
            } else {
                int32_t adjusted = 1 - 15 + 127;
                while (!(mantissa & 0x800000)) {
                    mantissa <<= 1;
                    --adjusted;
                }
                bits = sign | ((uint32_t)adjusted << 23) | (mantissa & 0x7fffff);
            }
        } else if (exponent == 0x1f) {
            bits = sign | 0x7f800000 | mantissa;
        } else {
            bits = sign | ((exponent + 127 - 15) << 23) | mantissa;
        }
        memcpy(destination + index, &bits, sizeof(bits));
    }
}

#endif
