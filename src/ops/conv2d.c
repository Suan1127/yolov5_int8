#include "conv2d.h"
#include "batchnorm2d.h"
#include "activation.h"
#include "../core/tensor.h"
#include "../validation/quant_util.h"
#include "../validation/conv2d_int8.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

/* ---------------------------------------------------------------------------
 * One-Pixel Compute Kernels (float path only, for conv2d_forward / reference)
 * --------------------------------------------------------------------------- */

/* 입력(activation) 양자화용: 현재 입력 텐서의 min/max로 scale_in 계산.
 * 매 forward마다 호출되므로 "동적 양자화". Calibration으로 scale_in을 고정해 export하면 제거 가능. */
static void tensor_minmax(const float* data, size_t count, float* out_min, float* out_max) {
    if (count == 0) return;
    float mn = data[0], mx = data[0];
    for (size_t i = 1; i < count; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
    }
    *out_min = mn;
    *out_max = mx;
}

int conv2d_init(conv2d_layer_t* layer, int32_t in_channels, const conv2d_params_t* params) {
    if (!layer || !params) return -1;
    layer->params = *params;
    layer->in_channels = in_channels;
    layer->scale_w = 0.f;
    layer->q_weight = NULL;
    size_t weight_size = (size_t)(params->out_channels * in_channels *
                                   params->kernel_size * params->kernel_size);
    layer->weight = (float*)calloc(weight_size, sizeof(float));
    if (!layer->weight) return -1;
    if (params->groups == 1) {
        layer->bias = (float*)calloc(params->out_channels, sizeof(float));
        if (!layer->bias) { free(layer->weight); return -1; }
    } else {
        layer->bias = NULL;
    }
    return 0;
}

void conv2d_free(conv2d_layer_t* layer) {
    if (layer) {
        if (layer->weight) free(layer->weight);
        if (layer->bias) free(layer->bias);
        if (layer->q_weight) free(layer->q_weight);
        memset(layer, 0, sizeof(conv2d_layer_t));
    }
}

int conv2d_load_weights(conv2d_layer_t* layer, const float* weight_buf, const float* bias_buf) {
    if (!layer || !weight_buf) return -1;
    size_t weight_size = (size_t)(layer->params.out_channels * layer->in_channels *
                                   layer->params.kernel_size * layer->params.kernel_size);
    memcpy(layer->weight, weight_buf, weight_size * sizeof(float));
    if (bias_buf && layer->bias)
        memcpy(layer->bias, bias_buf, layer->params.out_channels * sizeof(float));
    /* INT8 가중치: C에서 float→int8 변환 (int8 export 없을 때만). Export 시 int8 쓰면 load_weights_int8 사용. */
    float w_min, w_max;
    tensor_minmax(layer->weight, weight_size, &w_min, &w_max);
    quant_scale_t qs;
    quant_scale_symmetric_from_minmax(w_min, w_max, &qs);
    layer->scale_w = qs.scale;
    layer->q_weight = (int8_t*)malloc(weight_size * sizeof(int8_t));
    if (layer->q_weight)
        quantize_float_to_int8(layer->weight, weight_size, layer->scale_w, layer->q_weight);
    return 0;
}

int conv2d_load_weights_int8(conv2d_layer_t* layer, const int8_t* q_weight_buf, size_t q_weight_numel,
                              float scale_w, const float* bias_buf) {
    if (!layer || !q_weight_buf || scale_w <= 0.f) return -1;
    size_t need = (size_t)(layer->params.out_channels * layer->in_channels *
                           layer->params.kernel_size * layer->params.kernel_size);
    if (q_weight_numel != need) return -1;
    if (layer->weight) {
        free(layer->weight);
        layer->weight = NULL;
    }
    layer->scale_w = scale_w;
    layer->q_weight = (int8_t*)malloc(need * sizeof(int8_t));
    if (!layer->q_weight) return -1;
    memcpy(layer->q_weight, q_weight_buf, need * sizeof(int8_t));
    if (bias_buf && layer->bias)
        memcpy(layer->bias, bias_buf, layer->params.out_channels * sizeof(float));
    return 0;
}

/* 검증용: float weight/bias는 그대로 두고 q_weight/scale_w만 설정. float vs int8 비교 시 사용. */
int conv2d_attach_int8_weights(conv2d_layer_t* layer, const int8_t* q_weight_buf, size_t q_weight_numel, float scale_w) {
    if (!layer || !q_weight_buf || scale_w <= 0.f) return -1;
    size_t need = (size_t)(layer->params.out_channels * layer->in_channels *
                           layer->params.kernel_size * layer->params.kernel_size);
    if (q_weight_numel != need) return -1;
    if (layer->q_weight) free(layer->q_weight);
    layer->scale_w = scale_w;
    layer->q_weight = (int8_t*)malloc(need * sizeof(int8_t));
    if (!layer->q_weight) return -1;
    memcpy(layer->q_weight, q_weight_buf, need * sizeof(int8_t));
    return 0;
}

// Helper: Get output dimensions
static void conv2d_output_size(int32_t in_h, int32_t in_w, const conv2d_params_t* params,
                               int32_t* out_h, int32_t* out_w) {
    *out_h = (in_h + 2 * params->padding - params->dilation * (params->kernel_size - 1) - 1) / params->stride + 1;
    *out_w = (in_w + 2 * params->padding - params->dilation * (params->kernel_size - 1) - 1) / params->stride + 1;
}

/** One-pixel kernel: 1x1 Conv. Computes output[b,oc,h,w] (MAC only). */
static float conv1x1_one_pixel(const conv2d_layer_t* layer, const tensor_t* input,
                                int32_t b, int32_t oc, int32_t h, int32_t w) {
    int32_t in_c = layer->in_channels;
    int32_t in_h = input->h;
    int32_t in_w = input->w;
    size_t in_hw = (size_t)(in_h * in_w);
    size_t in_chw = (size_t)in_c * in_hw;
    size_t in_base = (size_t)b * in_chw + (size_t)h * (size_t)in_w + (size_t)w;
    const float* in_data = input->data;
    const float* pw = &layer->weight[(size_t)oc * (size_t)in_c];
    float sum = layer->bias ? layer->bias[oc] : 0.0f;
    for (int32_t ic = 0; ic < in_c; ic++) {
        sum += in_data[in_base + (size_t)ic * in_hw] * pw[ic];
    }
    return sum;
}

/** One-pixel kernel: 3x3 Conv (k=3, dilation=1, p=1, s=1 or 2). Computes output[b,oc,oh,ow] (MAC only). */
static float conv3x3_one_pixel(const conv2d_layer_t* layer, const tensor_t* input,
                                int32_t b, int32_t oc, int32_t oh, int32_t ow) {
    int32_t s = layer->params.stride;
    int32_t in_c = layer->in_channels;
    int32_t in_h = input->h;
    int32_t in_w = input->w;
    size_t in_hw = (size_t)(in_h * in_w);
    size_t in_chw = (size_t)in_c * in_hw;
    size_t b_in = (size_t)b * in_chw;
    const float* in_data = input->data;
    const float* w_oc = &layer->weight[(size_t)oc * (size_t)in_c * 9u];
    float sum = layer->bias ? layer->bias[oc] : 0.0f;
    int32_t ih0 = oh * s - 1;
    int32_t iw0 = ow * s - 1;
    for (int32_t ic = 0; ic < in_c; ic++) {
        const float* w_ic = &w_oc[(size_t)ic * 9];
        const float* in_ic = &in_data[b_in + (size_t)ic * in_hw];
        int32_t ih1 = ih0, ih2 = ih0 + 1, ih3 = ih0 + 2;
        int32_t iw1 = iw0, iw2 = iw0 + 1, iw3 = iw0 + 2;
        if (ih1 >= 0 && ih1 < in_h && iw1 >= 0 && iw1 < in_w)
            sum += in_ic[(size_t)ih1 * (size_t)in_w + (size_t)iw1] * w_ic[0];
        if (ih1 >= 0 && ih1 < in_h && iw2 >= 0 && iw2 < in_w)
            sum += in_ic[(size_t)ih1 * (size_t)in_w + (size_t)iw2] * w_ic[1];
        if (ih1 >= 0 && ih1 < in_h && iw3 >= 0 && iw3 < in_w)
            sum += in_ic[(size_t)ih1 * (size_t)in_w + (size_t)iw3] * w_ic[2];
        if (ih2 >= 0 && ih2 < in_h && iw1 >= 0 && iw1 < in_w)
            sum += in_ic[(size_t)ih2 * (size_t)in_w + (size_t)iw1] * w_ic[3];
        if (ih2 >= 0 && ih2 < in_h && iw2 >= 0 && iw2 < in_w)
            sum += in_ic[(size_t)ih2 * (size_t)in_w + (size_t)iw2] * w_ic[4];
        if (ih2 >= 0 && ih2 < in_h && iw3 >= 0 && iw3 < in_w)
            sum += in_ic[(size_t)ih2 * (size_t)in_w + (size_t)iw3] * w_ic[5];
        if (ih3 >= 0 && ih3 < in_h && iw1 >= 0 && iw1 < in_w)
            sum += in_ic[(size_t)ih3 * (size_t)in_w + (size_t)iw1] * w_ic[6];
        if (ih3 >= 0 && ih3 < in_h && iw2 >= 0 && iw2 < in_w)
            sum += in_ic[(size_t)ih3 * (size_t)in_w + (size_t)iw2] * w_ic[7];
        if (ih3 >= 0 && ih3 < in_h && iw3 >= 0 && iw3 < in_w)
            sum += in_ic[(size_t)ih3 * (size_t)in_w + (size_t)iw3] * w_ic[8];
    }
    return sum;
}

/** One-pixel kernel: generic k×k Conv. Computes output[b,oc,oh,ow] (MAC only). */
static float conv_generic_one_pixel(const conv2d_layer_t* layer, const tensor_t* input,
                                    int32_t b, int32_t oc, int32_t oh, int32_t ow) {
    int32_t k = layer->params.kernel_size;
    int32_t s = layer->params.stride;
    int32_t p = layer->params.padding;
    int32_t d = layer->params.dilation;
    int32_t in_c = layer->in_channels;
    int32_t in_h = input->h;
    int32_t in_w = input->w;
    size_t in_hw = (size_t)(in_h * in_w);
    size_t in_chw = (size_t)in_c * in_hw;
    size_t w_kk = (size_t)(k * k);
    size_t w_ic_kk = (size_t)in_c * w_kk;
    const float* in_data = input->data;
    const float* w_oc = &layer->weight[(size_t)oc * w_ic_kk];
    float sum = layer->bias ? layer->bias[oc] : 0.0f;
    for (int32_t ic = 0; ic < in_c; ic++) {
        const float* in_ic = &in_data[(size_t)b * in_chw + (size_t)ic * in_hw];
        const float* w_ic = &w_oc[(size_t)ic * w_kk];
        for (int32_t kh = 0; kh < k; kh++) {
            for (int32_t kw = 0; kw < k; kw++) {
                int32_t ih = oh * s + kh * d - p;
                int32_t iw = ow * s + kw * d - p;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                    sum += in_ic[(size_t)ih * (size_t)in_w + (size_t)iw] * w_ic[kh * k + kw];
            }
        }
    }
    return sum;
}

int conv2d_forward(const conv2d_layer_t* layer, const tensor_t* input, tensor_t* output) {
    if (!layer || !input || !output) return -1;
    if (!layer->weight) return -1;  /* int8-only 로드 시 float 경로 불가 */
    
    /* 1) Dimension and boundary validation */
    int32_t out_h, out_w;
    conv2d_output_size(input->h, input->w, &layer->params, &out_h, &out_w);
    if (output->n != input->n || output->c != layer->params.out_channels ||
        output->h != out_h || output->w != out_w) {
        fprintf(stderr, "Error: conv2d_forward: Output tensor size mismatch\n");
        fprintf(stderr, "  Expected: (%d, %d, %d, %d), Got: (%d, %d, %d, %d)\n",
                input->n, layer->params.out_channels, out_h, out_w,
                output->n, output->c, output->h, output->w);
        return -1;
    }
    if (input->c != layer->in_channels) {
        fprintf(stderr, "Error: conv2d_forward: Input channel mismatch\n");
        fprintf(stderr, "  Expected: %d, Got: %d\n", layer->in_channels, input->c);
        return -1;
    }
    if (input->data == output->data) {
        fprintf(stderr, "ERROR: conv2d_forward: input and output share the same memory!\n");
        return -1;
    }
    
    int32_t k = layer->params.kernel_size;
    int32_t s = layer->params.stride;
    int32_t p = layer->params.padding;
    int32_t out_c = layer->params.out_channels;
    int32_t in_h = input->h;
    int32_t in_w = input->w;
    
    /* 2) Loop structure + invocation of one_pixel kernel + write */
    if (k == 1 && s == 1 && p == 0) {
        for (int32_t b = 0; b < input->n; b++) {
            for (int32_t hy = 0; hy < in_h; hy++) {
                for (int32_t wx = 0; wx < in_w; wx++) {
                    for (int32_t oc = 0; oc < out_c; oc++) {
                        float sum = conv1x1_one_pixel(layer, input, b, oc, hy, wx);
                        *tensor_at(output, b, oc, hy, wx) = sum;
                    }
                }
            }
        }
    } else if (k == 3 && layer->params.dilation == 1 && p == 1 && (s == 1 || s == 2)) {
        int32_t out_h = output->h;
        int32_t out_w = output->w;
        for (int32_t b = 0; b < input->n; b++) {
            for (int32_t oc = 0; oc < out_c; oc++) {
                for (int32_t oh = 0; oh < out_h; oh++) {
                    for (int32_t ow = 0; ow < out_w; ow++) {
                        float sum = conv3x3_one_pixel(layer, input, b, oc, oh, ow);
                        *tensor_at(output, b, oc, oh, ow) = sum;
                    }
                }
            }
        }
    } else {
        int32_t out_h = output->h;
        int32_t out_w = output->w;
        for (int32_t b = 0; b < input->n; b++) {
            for (int32_t oc = 0; oc < out_c; oc++) {
                for (int32_t oh = 0; oh < out_h; oh++) {
                    for (int32_t ow = 0; ow < out_w; ow++) {
                        float sum = conv_generic_one_pixel(layer, input, b, oc, oh, ow);
                        *tensor_at(output, b, oc, oh, ow) = sum;
                    }
                }
            }
        }
    }
    return 0;
}

/* Detect head 등 BN 없는 Conv 전용: input(float) → quantize → int8 Conv → dequant → float. (하드웨어에 int8 weight만 올리기 위함) */
int conv2d_quant_forward(const conv2d_layer_t* layer, const tensor_t* input, tensor_t* output) {
    if (!layer || !input || !output || !layer->q_weight || layer->scale_w <= 0.f) return -1;
    int32_t out_h, out_w;
    conv2d_output_size(input->h, input->w, &layer->params, &out_h, &out_w);
    if (output->n != input->n || output->c != layer->params.out_channels ||
        output->h != out_h || output->w != out_w || input->c != layer->in_channels ||
        input->data == output->data)
        return -1;
    int32_t k = layer->params.kernel_size;
    int32_t s = layer->params.stride;
    int32_t p = layer->params.padding;
    int32_t d = layer->params.dilation;
    int32_t out_c = layer->params.out_channels;
    int32_t in_c = layer->in_channels;
    size_t in_count = (size_t)input->n * (size_t)in_c * (size_t)input->h * (size_t)input->w;
    size_t out_count = (size_t)output->n * (size_t)out_c * (size_t)out_h * (size_t)out_w;

    float in_min, in_max;
    tensor_minmax(input->data, in_count, &in_min, &in_max);
    quant_scale_t scale_in_q;
    quant_scale_symmetric_from_minmax(in_min, in_max, &scale_in_q);
    float scale_in = scale_in_q.scale;

    int8_t* q_input = (int8_t*)malloc(in_count * sizeof(int8_t));
    int32_t* acc32 = (int32_t*)malloc(out_count * sizeof(int32_t));
    if (!q_input || !acc32) {
        if (q_input) free(q_input);
        if (acc32) free(acc32);
        return -1;
    }
    quantize_float_to_int8(input->data, in_count, scale_in, q_input);
    if (conv2d_int8_forward(q_input, layer->q_weight, input->n, in_c, input->h, input->w,
                            out_c, k, s, p, d, acc32) != 0) {
        free(q_input);
        free(acc32);
        return -1;
    }
    free(q_input);
    dequant_acc32_to_float(acc32, out_count, out_c, out_h, out_w,
                           scale_in, layer->scale_w, layer->bias, output->data);
    free(acc32);
    return 0;
}

/* 단일 경로: input(float) → quantize → int8 Conv+BN → dequant → float → SiLU. Conv+BN은 가속기용 int8. */
int conv2d_quant_bn_silu_forward(const conv2d_layer_t* layer, const struct batchnorm2d_layer_t* bn,
                                 int bn_fused, const tensor_t* input, tensor_t* output) {
    if (!layer || !input || !output || !layer->q_weight) return -1;
    int32_t out_h, out_w;
    conv2d_output_size(input->h, input->w, &layer->params, &out_h, &out_w);
    if (output->n != input->n || output->c != layer->params.out_channels ||
        output->h != out_h || output->w != out_w || input->c != layer->in_channels ||
        input->data == output->data)
        return -1;
    int32_t k = layer->params.kernel_size;
    int32_t s = layer->params.stride;
    int32_t p = layer->params.padding;
    int32_t d = layer->params.dilation;
    int32_t out_c = layer->params.out_channels;
    int32_t in_c = layer->in_channels;
    size_t in_count = (size_t)input->n * (size_t)in_c * (size_t)input->h * (size_t)input->w;
    size_t out_count = (size_t)output->n * (size_t)out_c * (size_t)out_h * (size_t)out_w;

    /* scale_in: 입력 activation 동적 양자화. Calibration으로 scale_in 고정 시 이 minmax 제거 가능. */
    float in_min, in_max;
    tensor_minmax(input->data, in_count, &in_min, &in_max);
    quant_scale_t scale_in_q;
    quant_scale_symmetric_from_minmax(in_min, in_max, &scale_in_q);
    float scale_in = scale_in_q.scale;

    int8_t* q_input = (int8_t*)malloc(in_count * sizeof(int8_t));
    int32_t* acc32 = (int32_t*)malloc(out_count * sizeof(int32_t));
    if (!q_input || !acc32) {
        if (q_input) free(q_input);
        if (acc32) free(acc32);
        return -1;
    }
    quantize_float_to_int8(input->data, in_count, scale_in, q_input);
    if (conv2d_int8_forward(q_input, layer->q_weight, input->n, in_c, input->h, input->w,
                            out_c, k, s, p, d, acc32) != 0) {
        free(q_input);
        free(acc32);
        return -1;
    }
    free(q_input);
    dequant_acc32_to_float(acc32, out_count, out_c, out_h, out_w,
                           scale_in, layer->scale_w, layer->bias, output->data);
    free(acc32);

    if (bn && !bn_fused) {
        tensor_t* tmp = tensor_create(output->n, output->c, output->h, output->w);
        if (tmp) {
            batchnorm2d_forward(bn, output, tmp);
            tensor_copy(output, tmp);
            tensor_free(tmp);
        }
    }
    activation_silu(output);
    return 0;
}

/* Golden 디버그: 한 점(b,oc,oh,ow)에 대해 conv_raw / bn_out / silu_out 출력 (임베디드 비교용) */
void conv2d_fused_debug_one_pixel(const conv2d_layer_t* layer, const struct batchnorm2d_layer_t* bn,
                                   const tensor_t* input, int32_t b, int32_t oc, int32_t oh, int32_t ow) {
    if (!layer || !input) return;
    int32_t k = layer->params.kernel_size;
    int32_t s = layer->params.stride;
    int32_t p = layer->params.padding;
    float conv_raw;
    if (k == 1 && s == 1 && p == 0)
        conv_raw = conv1x1_one_pixel(layer, input, b, oc, oh, ow);
    else if (k == 3 && layer->params.dilation == 1 && p == 1 && (s == 1 || s == 2))
        conv_raw = conv3x3_one_pixel(layer, input, b, oc, oh, ow);
    else
        conv_raw = conv_generic_one_pixel(layer, input, b, oc, oh, ow);

    float bn_out = conv_raw;
    if (bn) {
        const batchnorm2d_layer_t* b = (const batchnorm2d_layer_t*)bn;
        float inv_std = 1.0f / sqrtf(b->running_var[oc] + b->params.eps);
        bn_out = (conv_raw - b->running_mean[oc]) * inv_std * b->weight[oc] + b->bias[oc];
    }
    float silu_out = bn_out * (1.0f / (1.0f + expf(-bn_out)));

    union { float f; uint32_t u; } u;
    u.f = conv_raw;  printf("  conv_raw\t0x%08X\n", (unsigned)u.u);
    u.f = bn_out;    printf("  bn_out\t0x%08X\n", (unsigned)u.u);
    u.f = silu_out;  printf("  silu_out\t0x%08X  (Layer0 첫 float, LE)\n", (unsigned)u.u);
}

/* Golden 상세 덤프: bias, acc_after_ic0/1/2, w[oc][0] 앞 8개, x[b][0] (0,0)~(1,1) (임베디드 비교용) */
void conv2d_debug_one_pixel_detail(const conv2d_layer_t* layer, const tensor_t* input,
                                   int32_t b, int32_t oc, int32_t oh, int32_t ow) {
    if (!layer || !input) return;
    int32_t k = layer->params.kernel_size;
    int32_t s = layer->params.stride;
    int32_t p = layer->params.padding;
    int32_t d = layer->params.dilation;
    int32_t in_c = layer->in_channels;
    int32_t in_h = input->h;
    int32_t in_w = input->w;
    size_t in_hw = (size_t)(in_h * in_w);
    size_t in_chw = (size_t)in_c * in_hw;
    size_t w_kk = (size_t)(k * k);
    size_t w_ic_kk = (size_t)in_c * w_kk;
    const float* in_data = input->data;
    const float* w_oc = &layer->weight[(size_t)oc * w_ic_kk];
    union { float f; uint32_t u; } u;

    /* bias[oc] */
    float bias_val = layer->bias ? layer->bias[oc] : 0.0f;
    u.f = bias_val;
    printf("  bias[%d]\t0x%08X\n", oc, (unsigned)u.u);

    /* acc_after_ic0, acc_after_ic1, acc_after_ic2 (cumulative over ic) */
    float acc = layer->bias ? layer->bias[oc] : 0.0f;
    for (int32_t ic = 0; ic < in_c && ic < 3; ic++) {
        const float* in_ic = &in_data[(size_t)b * in_chw + (size_t)ic * in_hw];
        const float* w_ic = &w_oc[(size_t)ic * w_kk];
        for (int32_t kh = 0; kh < k; kh++) {
            for (int32_t kw = 0; kw < k; kw++) {
                int32_t ih = oh * s + kh * d - p;
                int32_t iw = ow * s + kw * d - p;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                    acc += in_ic[(size_t)ih * (size_t)in_w + (size_t)iw] * w_ic[kh * k + kw];
            }
        }
        u.f = acc;
        printf("  acc_after_ic%d\t0x%08X\n", ic, (unsigned)u.u);
    }

    /* w[oc=0][ic=0] 6×6 앞 8개 (row-major) */
    const float* w_ic0 = &w_oc[0];
    printf("  w[oc=0][ic=0] 6x6 first8:");
    for (int i = 0; i < 8 && i < (int)(k * k); i++) {
        u.f = w_ic0[i];
        printf(" %02X%02X%02X%02X", (unsigned)(u.u & 0xFF), (unsigned)((u.u >> 8) & 0xFF),
               (unsigned)((u.u >> 16) & 0xFF), (unsigned)((u.u >> 24) & 0xFF));
    }
    printf(" (LE)\n");

    /* x[b=0][ic=0] (0,0), (0,1), (1,0), (1,1) */
    const float* x_ic0 = &in_data[(size_t)b * in_chw + 0 * in_hw];
    printf("  x[b=0][ic=0] (0,0)(0,1)(1,0)(1,1):");
    for (int ih = 0; ih <= 1; ih++) {
        for (int iw = 0; iw <= 1; iw++) {
            if (ih < in_h && iw < in_w) {
                u.f = x_ic0[(size_t)ih * (size_t)in_w + (size_t)iw];
                printf(" %02X%02X%02X%02X", (unsigned)(u.u & 0xFF), (unsigned)((u.u >> 8) & 0xFF),
                       (unsigned)((u.u >> 16) & 0xFF), (unsigned)((u.u >> 24) & 0xFF));
            }
        }
    }
    printf(" (LE)\n");
}
