/**
 * Conv 백엔드: 통합 API + SW/HW 구현 갈아끼우기.
 *
 * - SW 백엔드: conv2d_int8_forward 호출. 내부는 이미 K 무관 generic 루프.
 * - HW 백엔드: 가속기 호출. num_accelerators > 1 이면 OC를 균등 분할해 각 가속기가
 *   담당 구간(oc_start ~ oc_start+oc_count)만 처리하고, 출력은 동일 NCHW(oc→oh→ow)에 쓴다.
 */
#include "conv_backend.h"
#include "conv2d_int8.h"
#include <stddef.h>

static const conv_backend_t* s_default_backend = NULL;
static int s_num_accelerators = 1;

static int sw_run_int8(const conv_backend_params_t* params,
                       const int8_t* q_input,
                       const int8_t* q_weight,
                       int32_t* acc32_out) {
    if (!params || !q_input || !q_weight || !acc32_out) return -1;
    return conv2d_int8_forward(
        q_input, q_weight,
        params->n, params->in_c, params->in_h, params->in_w,
        params->out_c, params->k, params->stride, params->padding, params->dilation,
        acc32_out);
}

static int hw_run_int8(const conv_backend_params_t* params,
                      const int8_t* q_input,
                      const int8_t* q_weight,
                      int32_t* acc32_out) {
    (void)params;
    (void)q_input;
    (void)q_weight;
    (void)acc32_out;
    /* TODO: AXI/레지스터로 K, stride, padding 설정 후 가속기 호출.
     * IP는 하나; 내부 RTL에서 if (K==1) 1x1 path, else if (K==3) 3x3 path 등. */
    return -1;  /* 미구현: 현재는 SW 백엔드 사용 */
}

static const conv_backend_t s_sw = {
    .name = "sw",
    .run_int8 = sw_run_int8,
};

static const conv_backend_t s_hw = {
    .name = "hw",
    .run_int8 = hw_run_int8,
};

const conv_backend_t* conv_backend_get_default(void) {
    if (s_default_backend == NULL)
        return &s_sw;
    return s_default_backend;
}

void conv_backend_set_default(const conv_backend_t* backend) {
    s_default_backend = backend;
}

const conv_backend_t* conv_backend_sw(void) {
    return &s_sw;
}

const conv_backend_t* conv_backend_hw(void) {
    return &s_hw;
}

static int32_t output_dim(int32_t in, int32_t k, int32_t s, int32_t p, int32_t d) {
    return (in + 2 * p - d * (k - 1) - 1) / s + 1;
}

void conv_backend_output_size(const conv_backend_params_t* params, int32_t* out_h, int32_t* out_w) {
    if (!params || !out_h || !out_w) return;
    *out_h = output_dim(params->in_h, params->k, params->stride, params->padding, params->dilation);
    *out_w = output_dim(params->in_w, params->k, params->stride, params->padding, params->dilation);
}

int conv_backend_num_accelerators(void) {
    return s_num_accelerators > 0 ? s_num_accelerators : 1;
}

void conv_backend_set_num_accelerators(int n) {
    s_num_accelerators = (n > 0) ? n : 1;
}

void conv_backend_oc_slice(int32_t out_c, int num_accel, int accel_id,
                           int32_t* oc_start, int32_t* oc_count) {
    if (!oc_start || !oc_count || num_accel <= 0 || accel_id < 0 || accel_id >= num_accel) return;
    int32_t base = out_c / num_accel;
    int32_t rem = out_c % num_accel;
    *oc_count = (int32_t)(accel_id < rem ? base + 1 : base);
    *oc_start = 0;
    for (int i = 0; i < accel_id; i++)
        *oc_start += (int32_t)(i < rem ? base + 1 : base);
}

int conv_backend_run_int8(const conv_backend_params_t* params,
                          const int8_t* q_input,
                          const int8_t* q_weight,
                          int32_t* acc32_out) {
    const conv_backend_t* be = conv_backend_get_default();
    if (!be || !be->run_int8 || !params || !acc32_out) return -1;

    int num_accel = conv_backend_num_accelerators();
    /* 다중 가속기: HW 백엔드일 때만 OC 분할. n=1일 때만 분할 (inference 기본). SW는 항상 1회 호출. */
    if (num_accel <= 1 || be != conv_backend_hw() || params->n != 1) {
        return be->run_int8(params, q_input, q_weight, acc32_out);
    }

    int32_t out_h, out_w;
    conv_backend_output_size(params, &out_h, &out_w);
    int32_t out_c = params->out_c;
    int32_t in_c = params->in_c;
    int32_t k = params->k;
    size_t weight_oc_stride = (size_t)(in_c * k * k);
    size_t out_oc_stride = (size_t)(out_h * out_w);

    for (int accel_id = 0; accel_id < num_accel; accel_id++) {
        int32_t oc_start, oc_count;
        conv_backend_oc_slice(out_c, num_accel, accel_id, &oc_start, &oc_count);
        if (oc_count <= 0) continue;

        conv_backend_params_t slice_params = *params;
        slice_params.out_c = oc_count;

        const int8_t* q_w_slice = q_weight + (size_t)oc_start * weight_oc_stride;
        /* NCHW(oc→oh→ow): 이 가속기가 쓸 구간 = acc32_out + oc_start * (OH*OW) */
        int32_t* acc_slice = acc32_out + (size_t)oc_start * out_oc_stride;

        if (be->run_int8(&slice_params, q_input, q_w_slice, acc_slice) != 0)
            return -1;
    }
    return 0;
}
