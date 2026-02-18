#ifndef CONV_BACKEND_H
#define CONV_BACKEND_H

#include <stdint.h>

/**
 * Conv 백엔드: "하나의 Conv 엔진" API.
 *
 * HW는 하나의 IP로 K=1/3/6을 레지스터(K, stride, pad)로 선택해 처리.
 * C에서는 겉 API를 통일하고, 백엔드만 SW/HW 갈아끼움.
 *
 * - Host: conv2d_*_forward() → 차원 검사, 양자화, 백엔드 호출, 디양자화.
 * - Backend: conv_backend_run_int8() 한 개 → 내부에서 K에 따라 1x1/3x3/6x6 분기 또는 HW 호출.
 */

/** INT8 Conv 한 번 실행에 필요한 파라미터 (레지스터에 대응). */
typedef struct {
    int32_t n;
    int32_t in_c;
    int32_t in_h;
    int32_t in_w;
    int32_t out_c;
    int32_t k;       /* kernel size (1, 3, 6 등) */
    int32_t stride;
    int32_t padding;
    int32_t dilation;
} conv_backend_params_t;

/**
 * INT8 Conv 실행: q_in * q_weight → acc32.
 * 호스트는 차원/주소 계산 후 이 인터페이스만 호출.
 *
 * @param params  K, stride, pad, n, in_c, in_h, in_w, out_c 등
 * @param q_input [n, in_c, in_h, in_w]
 * @param q_weight [out_c, in_c, k, k]
 * @param acc32_out [n, out_c, out_h, out_w]
 * @return 0 success, -1 error
 */
typedef int (*conv_backend_run_int8_fn)(const conv_backend_params_t* params,
                                        const int8_t* q_input,
                                        const int8_t* q_weight,
                                        int32_t* acc32_out);

typedef struct conv_backend_t {
    const char* name;
    conv_backend_run_int8_fn run_int8;
} conv_backend_t;

/** 현재 사용할 백엔드 (기본: SW). */
const conv_backend_t* conv_backend_get_default(void);
void conv_backend_set_default(const conv_backend_t* backend);

/** SW 백엔드: 내부에서 K에 따라 1x1/3x3/일반 분기. 기존 conv2d_int8 로직. */
const conv_backend_t* conv_backend_sw(void);

/** HW 백엔드: 가속기 호출. 구현 시 K, stride, pad를 레지스터로 넘겨 하나의 Conv 엔진 제어. */
const conv_backend_t* conv_backend_hw(void);

/**
 * 통합 실행: params와 버퍼만 넘기면 백엔드(현재 default)가 처리.
 * num_accelerators > 1 이고 HW 백엔드일 때, OC를 가속기 수만큼 나누어 각각 호출 후 결과를
 * 동일한 acc32_out 레이아웃(oc→oh→ow)에 쓴다.
 */
int conv_backend_run_int8(const conv_backend_params_t* params,
                          const int8_t* q_input,
                          const int8_t* q_weight,
                          int32_t* acc32_out);

/** 출력 공간 크기 (params 기준). stride/pad 등과 동일 공식. */
void conv_backend_output_size(const conv_backend_params_t* params, int32_t* out_h, int32_t* out_w);

/** 다중 가속기: 사용할 가속기 개수 (기본 1). HW 백엔드에서 OC 분할에 사용. */
int conv_backend_num_accelerators(void);
void conv_backend_set_num_accelerators(int n);

/** OC 분할: 가속기 번호 accel_id 에 해당하는 [oc_start, oc_start+oc_count) 반환. */
void conv_backend_oc_slice(int32_t out_c, int num_accel, int accel_id,
                           int32_t* oc_start, int32_t* oc_count);

#endif /* CONV_BACKEND_H */
