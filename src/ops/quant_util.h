#ifndef QUANT_UTIL_H
#define QUANT_UTIL_H

#include "../core/tensor.h"
#include <stdint.h>
#include <stddef.h>

/**
 * Symmetric INT8 quantization (range -127..127, zero_point=0).
 * scale = max(|min|, |max|) / 127 so that float value is approximated by scale * q.
 * Used for activation and weight; calibration과 inference에서 동일한 scale 사용해야 함.
 */
typedef struct {
    float scale;   /* dequant: float = scale * (int32_t)acc32 (per-tensor) */
    int8_t zp;    /* symmetric이면 0 */
} quant_scale_t;

/**
 * Compute per-tensor scale from min/max (symmetric).
 * scale = max(|min|, |max|) / 127.f; zp = 0.
 */
void quant_scale_symmetric_from_minmax(float min_val, float max_val, quant_scale_t* out);

/**
 * Quantize float tensor to int8 (symmetric): q = round(x / scale), clamp to [-127, 127].
 * out[i] = clamp(round(data[i] / scale), -127, 127).
 */
void quantize_float_to_int8(const float* data, size_t count, float scale, int8_t* out);

/**
 * Dequantize int32 accumulation to float (NCHW): out = scale_in*scale_w*acc32 + bias[oc].
 * count = n * out_c * out_h * out_w; bias has out_c elements (per-channel).
 */
void dequant_acc32_to_float(const int32_t* acc32, size_t count, int32_t out_c, int32_t out_h, int32_t out_w,
                            float scale_in, float scale_w, const float* bias, float* out);

#endif /* QUANT_UTIL_H */
