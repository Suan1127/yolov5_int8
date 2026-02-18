#include "conv2d_int8.h"
#include <stdlib.h>
#include <string.h>

/* Output size (same formula as float conv2d). */
static int32_t conv_out_size(int32_t in, int32_t k, int32_t s, int32_t p, int32_t d) {
    return (in + 2 * p - d * (k - 1) - 1) / s + 1;
}

/**
 * INT8 conv: one output pixel (b, oc, oh, ow). acc32 = sum over ic, kh, kw of q_in * q_w.
 * Weight layout: [out_c, in_c, k, k] -> w_oc[ic][kh,kw] = w[oc][ic][kh][kw].
 */
static int32_t conv2d_int8_one_pixel(const int8_t* q_in, const int8_t* q_w,
                                     int32_t b, int32_t oc, int32_t oh, int32_t ow,
                                     int32_t in_c, int32_t in_h, int32_t in_w,
                                     int32_t k, int32_t s, int32_t p, int32_t d) {
    size_t in_hw = (size_t)(in_h * in_w);
    size_t in_chw = (size_t)in_c * in_hw;
    size_t w_kk = (size_t)(k * k);
    size_t w_ic_kk = (size_t)in_c * w_kk;
    const int8_t* w_oc = &q_w[(size_t)oc * w_ic_kk];

    int32_t sum = 0;
    for (int32_t ic = 0; ic < in_c; ic++) {
        const int8_t* in_ic = &q_in[(size_t)b * in_chw + (size_t)ic * in_hw];
        const int8_t* w_ic = &w_oc[(size_t)ic * w_kk];
        for (int32_t kh = 0; kh < k; kh++) {
            for (int32_t kw = 0; kw < k; kw++) {
                int32_t ih = oh * s + kh * d - p;
                int32_t iw = ow * s + kw * d - p;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    size_t in_idx = (size_t)ih * (size_t)in_w + (size_t)iw;
                    sum += (int32_t)in_ic[in_idx] * (int32_t)w_ic[kh * k + kw];
                }
            }
        }
    }
    return sum;
}

int conv2d_int8_forward(const int8_t* q_input, const int8_t* q_weight,
                        int32_t n, int32_t in_c, int32_t in_h, int32_t in_w,
                        int32_t out_c, int32_t k, int32_t s, int32_t p, int32_t d,
                        int32_t* acc32_out) {
    if (!q_input || !q_weight || !acc32_out) return -1;

    int32_t out_h = conv_out_size(in_h, k, s, p, d);
    int32_t out_w = conv_out_size(in_w, k, s, p, d);

    for (int32_t b = 0; b < n; b++) {
        for (int32_t oc = 0; oc < out_c; oc++) {
            for (int32_t oh = 0; oh < out_h; oh++) {
                for (int32_t ow = 0; ow < out_w; ow++) {
                    size_t out_idx = (size_t)b * (size_t)out_c * (size_t)out_h * (size_t)out_w
                                   + (size_t)oc * (size_t)out_h * (size_t)out_w
                                   + (size_t)oh * (size_t)out_w + (size_t)ow;
                    acc32_out[out_idx] = conv2d_int8_one_pixel(
                        q_input, q_weight, b, oc, oh, ow,
                        in_c, in_h, in_w, k, s, p, d);
                }
            }
        }
    }
    return 0;
}
