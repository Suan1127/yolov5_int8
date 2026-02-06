#include "quant_util.h"
#include <math.h>
#include <stdlib.h>

/* Symmetric scale: scale = max(|min|,|max|)/127, zp=0. 양자화 검증에서 입력/가중치 모두 동일 방식. */
void quant_scale_symmetric_from_minmax(float min_val, float max_val, quant_scale_t* out) {
    if (!out) return;
    float abs_max = (float)fabs((double)min_val);
    float abs_min = (float)fabs((double)max_val);
    if (abs_min > abs_max) abs_max = abs_min;
    if (abs_max < 1e-9f) abs_max = 1e-9f;
    out->scale = abs_max / 127.f;
    out->zp = 0;
}

/* q = round(x/scale), clamp to [-127, 127]. INT8 symmetric. */
void quantize_float_to_int8(const float* data, size_t count, float scale, int8_t* out) {
    if (!data || !out || scale <= 0.f) return;
    for (size_t i = 0; i < count; i++) {
        float qf = data[i] / scale;
        int32_t q = (int32_t)roundf(qf);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        out[i] = (int8_t)q;
    }
}

/* NCHW: out[b,oc,h,w] = scale_in*scale_w*acc32[i] + bias[oc]. */
void dequant_acc32_to_float(const int32_t* acc32, size_t count, int32_t out_c, int32_t out_h, int32_t out_w,
                            float scale_in, float scale_w, const float* bias, float* out) {
    if (!acc32 || !out) return;
    float combined_scale = scale_in * scale_w;
    size_t out_hw = (size_t)out_h * (size_t)out_w;
    for (size_t i = 0; i < count; i++) {
        int32_t oc = (int32_t)((i / out_hw) % (size_t)out_c);
        out[i] = combined_scale * (float)acc32[i] + (bias ? bias[oc] : 0.f);
    }
}
