#ifndef CONV2D_INT8_H
#define CONV2D_INT8_H

#include "../core/tensor.h"
#include <stdint.h>

/**
 * INT8 2D convolution with int32 accumulation (for quantization validation).
 * Layout: input int8 NCHW, weight int8 [out_c, in_c, k, k], output acc32 NCHW.
 * No bias in integer path; bias is applied in float after dequant.
 */

typedef struct {
    int32_t out_channels;
    int32_t in_channels;
    int32_t kernel_size;
    int32_t stride;
    int32_t padding;
    int32_t dilation;
} conv2d_int8_params_t;

/**
 * Run INT8 conv (generic k×k), accumulate to int32.
 * q_input: [n, in_c, in_h, in_w], q_weight: [out_c, in_c, k, k], acc32_out: [n, out_c, out_h, out_w].
 * Output size: same as float conv2d (in + 2*pad - d*(k-1) - 1) / stride + 1.
 */
int conv2d_int8_forward(const int8_t* q_input, const int8_t* q_weight,
                       int32_t n, int32_t in_c, int32_t in_h, int32_t in_w,
                       int32_t out_c, int32_t k, int32_t s, int32_t p, int32_t d,
                       int32_t* acc32_out);

#endif /* CONV2D_INT8_H */
