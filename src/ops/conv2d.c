#include "conv2d.h"
#include "conv_backend.h"
#include "batchnorm2d.h"
#include "activation.h"
#include "../core/tensor.h"
#include "quant_util.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

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
    conv_backend_params_t be_params = {
        .n = input->n, .in_c = in_c, .in_h = input->h, .in_w = input->w,
        .out_c = out_c, .k = k, .stride = s, .padding = p, .dilation = d
    };
    if (conv_backend_run_int8(&be_params, q_input, layer->q_weight, acc32) != 0) {
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
    conv_backend_params_t be_params = {
        .n = input->n, .in_c = in_c, .in_h = input->h, .in_w = input->w,
        .out_c = out_c, .k = k, .stride = s, .padding = p, .dilation = d
    };
    if (conv_backend_run_int8(&be_params, q_input, layer->q_weight, acc32) != 0) {
        free(q_input);
        free(acc32);
        return -1;
    }
    free(q_input);
    dequant_acc32_to_float(acc32, out_count, out_c, out_h, out_w,
                           scale_in, layer->scale_w, layer->bias, output->data);
    free(acc32);
    /* BN은 항상 fused(bias에 반영). 별도 float BN 적용 없음. */
    activation_silu(output);
    return 0;
}
