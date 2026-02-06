#ifndef CONV2D_H
#define CONV2D_H

#include "../core/tensor.h"

/**
 * 2D Convolution parameters
 */
typedef struct {
    int32_t out_channels;
    int32_t kernel_size;   // Assuming square kernel (k×k)
    int32_t stride;
    int32_t padding;
    int32_t groups;        // For depthwise conv (default: 1)
    int32_t dilation;      // Default: 1
} conv2d_params_t;

/**
 * Convolution layer with weights and bias.
 * Inference 경로: input(float) → 양자화 → int8 Conv+BN(가속기) → 디양자화 → float → SiLU.
 */
typedef struct {
    conv2d_params_t params;
    float* weight;         // [out_channels, in_channels, k, k] (float 원본, bias용)
    float* bias;           // [out_channels] or NULL (Conv+BN folded bias)
    int32_t in_channels;
    float scale_w;        // INT8 가중치 scale (symmetric, load 시 설정)
    int8_t* q_weight;     // [out_c, in_c, k, k] INT8 가중치 (가속기 입력용)
} conv2d_layer_t;

/**
 * Initialize convolution layer (scale_w=0, q_weight=NULL; load_weights에서 설정)
 */
int conv2d_init(conv2d_layer_t* layer, int32_t in_channels, const conv2d_params_t* params);

/**
 * Free convolution layer (weight, bias, q_weight 해제)
 */
void conv2d_free(conv2d_layer_t* layer);

/**
 * Forward pass: output = conv2d(input) — float only, 검증/레퍼런스용.
 */
int conv2d_forward(const conv2d_layer_t* layer, const tensor_t* input, tensor_t* output);

/**
 * 단일 경로: input(float32) → quantize → int8 Conv+BN → dequant → float32 → SiLU.
 * Conv+BN은 가속기에서 int8로 수행; SiLU는 float로 적용.
 * bn NULL이면 bias에 BN이 이미 fold된 상태(fused).
 */
#ifndef CONV2D_BN_DECL
struct batchnorm2d_layer_t;
#endif
int conv2d_quant_bn_silu_forward(const conv2d_layer_t* layer, const struct batchnorm2d_layer_t* bn,
                                 int bn_fused, const tensor_t* input, tensor_t* output);

/**
 * Load weights from float buffer (기존; int8 export 사용 시 load_weights_int8 사용)
 */
int conv2d_load_weights(conv2d_layer_t* layer, const float* weight_buf, const float* bias_buf);

/**
 * Load weights from int8 buffer (export 시 int8로 저장한 경우).
 * q_weight 복사, scale_w 설정, bias만 float. float weight는 해제하고 NULL로 둠.
 */
int conv2d_load_weights_int8(conv2d_layer_t* layer, const int8_t* q_weight_buf, size_t q_weight_numel,
                              float scale_w, const float* bias_buf);

#endif // CONV2D_H
