/**
 * Backbone Layer0~8 양자화 검증 (float vs int8)
 *
 * - Layer0: Conv+BN -> SiLU
 * - Layer1: SiLU(layer0) -> Conv+BN
 * - Layer2: SiLU(layer1) -> C3
 * - Layer3: layer2 -> Conv+BN
 * - Layer4: SiLU(layer3) -> C3
 * - Layer5: layer4 -> Conv+BN
 * - Layer6: SiLU(layer5) -> C3
 * - Layer7: layer6 -> Conv+BN
 * - Layer8: SiLU(layer7) -> C3 (backbone 끝)
 * 인자: [weights.bin] [model_meta.json] [input.bin]
 */

#include "conv2d_int8.h"
#include "quant_util.h"
#include "../core/tensor.h"
#include "../ops/conv2d.h"
#include "../ops/batchnorm2d.h"
#include "../ops/activation.h"
#include "../blocks/c3.h"
#include "../blocks/sppf.h"
#include "../ops/upsample.h"
#include "../ops/concat.h"
#include "../models/yolov5n_build.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

static float compute_mse(const float* a, const float* b, size_t count) {
    if (count == 0) return 0.f;
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return (float)(sum / (double)count);
}

static float compute_correlation(const float* a, const float* b, size_t count) {
    if (count < 2) return 1.f;
    double sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0;
    for (size_t i = 0; i < count; i++) {
        double x = (double)a[i], y = (double)b[i];
        sa += x; sb += y;
        saa += x * x; sbb += y * y; sab += x * y;
    }
    double n = (double)count;
    double cov = sab - sa * sb / n;
    double var_a = saa - sa * sa / n;
    double var_b = sbb - sb * sb / n;
    if (var_a <= 0.0 || var_b <= 0.0) return 0.f;
    return (float)(cov / sqrt(var_a * var_b));
}

int main(int argc, char* argv[]) {
    /* 기본: 실행 위치 기준 ../../ (예: build/ 에서 실행 시). float + int8 모두 같은 weights 디렉에서 로드 */
    const char* weights_path = (argc >= 2) ? argv[1] : "../../weights/yolov5n/weights_fused.bin";
    const char* meta_path   = (argc >= 3) ? argv[2] : "../../weights/yolov5n/model_meta_fused.json";
    const char* input_bin   = (argc >= 4) ? argv[3] : "../../data/yolov5n/inputs/bus.bin";

    printf("=== Layer0 quantization validation (Conv+BN fused) ===\n");
    printf("Weights: %s\n", weights_path);
    printf("Meta:    %s\n", meta_path);
    printf("Input:   %s (float NCHW 1x3x640x640)\n", input_bin);

    yolov5n_model_t* model = yolov5n_build(weights_path, meta_path);
    if (!model) {
        fprintf(stderr, "Failed to build model\n");
        return 1;
    }

    /* Layer0: Conv(3->16, 6x6, s=2, p=2) + BN fused (q_weight/scale_w는 weights_int8.bin + scales_int8.json에서 로드) */
    conv2d_layer_t* conv = &model->backbone_convs[0].conv;
    batchnorm2d_layer_t* bn = model->backbone_convs[0].is_fused ? NULL : &model->backbone_convs[0].bn;
    int32_t in_c = conv->in_channels;
    int32_t out_c = conv->params.out_channels;
    int32_t k = conv->params.kernel_size;
    int32_t s = conv->params.stride;
    int32_t p = conv->params.padding;
    int32_t d = conv->params.dilation;

    /* q_weight / scale_w: weights_dir에 weights_int8.bin + scales_int8.json 있으면 load_conv_bn_layer에서 로드됨 */
    int8_t* q_weight = conv->q_weight;
    float scale_w = conv->scale_w;
    if (!q_weight || scale_w <= 0.f) {
        fprintf(stderr, "Layer0 has no q_weight/scale_w. Put weights_int8.bin and scales_int8.json in same dir as weights.\n");
        yolov5n_free(model);
        return 1;
    }
    printf("Layer0 INT8: q_weight and scale_w from weights_int8.bin + scales_int8.json (scale_w=%.6f)\n", scale_w);

    tensor_t* input = tensor_load(input_bin);
    if (!input) {
        fprintf(stderr, "Cannot load input: %s (expected float NCHW 1x3x640x640)\n", input_bin);
        yolov5n_free(model);
        return 1;
    }
    if (input->n != 1 || input->c != 3 || input->h != 640 || input->w != 640) {
        fprintf(stderr, "Input shape must be 1x3x640x640, got %d x %d x %d x %d\n",
                input->n, input->c, input->h, input->w);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    int32_t out_h = (input->h + 2 * p - d * (k - 1) - 1) / s + 1;
    int32_t out_w = (input->w + 2 * p - d * (k - 1) - 1) / s + 1;
    size_t out_count = (size_t)out_c * (size_t)out_h * (size_t)out_w;
    size_t in_count = tensor_size(input);

    tensor_t* ref_float = tensor_create(1, out_c, out_h, out_w);
    if (!ref_float) {
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    /* Float reference: Conv+BN fused (no SiLU) */
    if (bn) {
        tensor_t* conv_out = tensor_create(1, out_c, out_h, out_w);
        if (!conv_out) { tensor_free(ref_float); tensor_free(input); yolov5n_free(model); return 1; }
        if (conv2d_forward(conv, input, conv_out) != 0 ||
            batchnorm2d_forward(bn, conv_out, ref_float) != 0) {
            tensor_free(conv_out); tensor_free(ref_float); tensor_free(input); yolov5n_free(model);
            return 1;
        }
        tensor_free(conv_out);
    } else {
        if (conv2d_forward(conv, input, ref_float) != 0) {
            tensor_free(ref_float); tensor_free(input); yolov5n_free(model);
            return 1;
        }
    }

    /* INT8 path: input float -> dynamic scale_in -> quantize -> int8 conv -> dequant */
    float in_min, in_max;
    tensor_minmax(input->data, in_count, &in_min, &in_max);
    quant_scale_t scale_in;
    quant_scale_symmetric_from_minmax(in_min, in_max, &scale_in);

    int8_t* q_input = (int8_t*)malloc(in_count * sizeof(int8_t));
    int32_t* acc32 = (int32_t*)malloc(out_count * sizeof(int32_t));
    if (!q_input || !acc32) {
        free(q_input);
        free(acc32);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    quantize_float_to_int8(input->data, in_count, scale_in.scale, q_input);

    if (conv2d_int8_forward(q_input, q_weight, 1, in_c, input->h, input->w,
                            out_c, k, s, p, d, acc32) != 0) {
        free(acc32);
        free(q_input);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    tensor_t* out_int8_path = tensor_create(1, out_c, out_h, out_w);
    if (!out_int8_path) {
        free(acc32);
        free(q_input);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    dequant_acc32_to_float(acc32, out_count, out_c, out_h, out_w,
                          scale_in.scale, scale_w, conv->bias, out_int8_path->data);
    if (bn) {
        tensor_t* tmp = tensor_create(1, out_c, out_h, out_w);
        if (tmp) {
            tensor_copy(tmp, out_int8_path);
            batchnorm2d_forward(bn, tmp, out_int8_path);
            tensor_free(tmp);
        }
    }

    float mse0 = compute_mse(ref_float->data, out_int8_path->data, out_count);
    float corr0 = compute_correlation(ref_float->data, out_int8_path->data, out_count);

    /* 디양자화 결과에 SiLU 적용 (layer1 입력으로 사용) */
    activation_silu(ref_float);
    activation_silu(out_int8_path);

    /* Layer1 검증: SiLU(layer0) 출력을 입력으로 Conv+BN fused float vs INT8 */
    conv2d_layer_t* conv1 = &model->backbone_convs[1].conv;
    batchnorm2d_layer_t* bn1 = model->backbone_convs[1].is_fused ? NULL : &model->backbone_convs[1].bn;
    int32_t in_c1 = conv1->in_channels;
    int32_t out_c1 = conv1->params.out_channels;
    int32_t k1 = conv1->params.kernel_size;
    int32_t s1 = conv1->params.stride;
    int32_t p1 = conv1->params.padding;
    int32_t d1 = conv1->params.dilation;

    int8_t* q_weight1 = conv1->q_weight;
    float scale_w1 = conv1->scale_w;
    if (!q_weight1 || scale_w1 <= 0.f) {
        fprintf(stderr, "Layer1 has no q_weight/scale_w.\n");
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    int32_t out_h1 = (out_h + 2 * p1 - d1 * (k1 - 1) - 1) / s1 + 1;
    int32_t out_w1 = (out_w + 2 * p1 - d1 * (k1 - 1) - 1) / s1 + 1;
    size_t out_count1 = (size_t)out_c1 * (size_t)out_h1 * (size_t)out_w1;
    size_t in_count1 = tensor_size(ref_float);  /* ref_float = SiLU(layer0), 1 x out_c x out_h x out_w */

    tensor_t* layer1_ref = tensor_create(1, out_c1, out_h1, out_w1);
    if (!layer1_ref) {
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    /* Layer1 float: Conv+BN (fused면 conv만) */
    if (bn1) {
        tensor_t* conv1_out = tensor_create(1, out_c1, out_h1, out_w1);
        if (!conv1_out) {
            tensor_free(layer1_ref);
            free(acc32);
            free(q_input);
            tensor_free(out_int8_path);
            tensor_free(ref_float);
            tensor_free(input);
            yolov5n_free(model);
            return 1;
        }
        if (conv2d_forward(conv1, ref_float, conv1_out) != 0 ||
            batchnorm2d_forward(bn1, conv1_out, layer1_ref) != 0) {
            tensor_free(conv1_out);
            tensor_free(layer1_ref);
            free(acc32);
            free(q_input);
            tensor_free(out_int8_path);
            tensor_free(ref_float);
            tensor_free(input);
            yolov5n_free(model);
            return 1;
        }
        tensor_free(conv1_out);
    } else {
        if (conv2d_forward(conv1, ref_float, layer1_ref) != 0) {
            tensor_free(layer1_ref);
            free(acc32);
            free(q_input);
            tensor_free(out_int8_path);
            tensor_free(ref_float);
            tensor_free(input);
            yolov5n_free(model);
            return 1;
        }
    }

    /* Layer1 INT8: SiLU(layer0 int8) 출력을 동적 양자화 -> int8 conv -> dequant */
    float in_min1, in_max1;
    tensor_minmax(out_int8_path->data, in_count1, &in_min1, &in_max1);
    quant_scale_t scale_in1;
    quant_scale_symmetric_from_minmax(in_min1, in_max1, &scale_in1);

    int8_t* q_input1 = (int8_t*)malloc(in_count1 * sizeof(int8_t));
    int32_t* acc32_1 = (int32_t*)malloc(out_count1 * sizeof(int32_t));
    if (!q_input1 || !acc32_1) {
        free(q_input1);
        free(acc32_1);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    quantize_float_to_int8(out_int8_path->data, in_count1, scale_in1.scale, q_input1);

    if (conv2d_int8_forward(q_input1, q_weight1, 1, in_c1, (int32_t)out_h, (int32_t)out_w,
                           out_c1, k1, s1, p1, d1, acc32_1) != 0) {
        free(acc32_1);
        free(q_input1);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    tensor_t* layer1_int8 = tensor_create(1, out_c1, out_h1, out_w1);
    if (!layer1_int8) {
        free(acc32_1);
        free(q_input1);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    dequant_acc32_to_float(acc32_1, out_count1, out_c1, out_h1, out_w1,
                          scale_in1.scale, scale_w1, conv1->bias, layer1_int8->data);
    if (bn1) {
        tensor_t* tmp1 = tensor_create(1, out_c1, out_h1, out_w1);
        if (tmp1) {
            tensor_copy(tmp1, layer1_int8);
            batchnorm2d_forward(bn1, tmp1, layer1_int8);
            tensor_free(tmp1);
        }
    }

    float mse1 = compute_mse(layer1_ref->data, layer1_int8->data, out_count1);
    float corr1 = compute_correlation(layer1_ref->data, layer1_int8->data, out_count1);

    free(acc32_1);
    free(q_input1);

    /* Layer2 (C3): 입력 = SiLU(layer1). float 기준은 별도 C3 블록(float만 로드), int8은 model->backbone_c3s[0] */
    activation_silu(layer1_ref);
    activation_silu(layer1_int8);

    c3_block_t c3_float;
    if (c3_init(&c3_float, model->backbone_c3s[0].c1, model->backbone_c3s[0].c2,
                model->backbone_c3s[0].n, model->backbone_c3s[0].shortcut) != 0) {
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    if (c3_load_weights(&c3_float, model->weights, "model.2", NULL) != 0) {
        c3_free(&c3_float);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    int32_t c2_out = model->backbone_c3s[0].c2;
    tensor_t* layer2_ref = tensor_create(1, c2_out, out_h1, out_w1);
    tensor_t* layer2_int8 = tensor_create(1, c2_out, out_h1, out_w1);
    if (!layer2_ref || !layer2_int8) {
        if (layer2_ref) tensor_free(layer2_ref);
        if (layer2_int8) tensor_free(layer2_int8);
        c3_free(&c3_float);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    if (c3_forward_float(&c3_float, layer1_ref, layer2_ref, NULL, NULL) != 0) {
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        c3_free(&c3_float);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    if (c3_forward(&model->backbone_c3s[0].block, layer1_int8, layer2_int8, NULL, NULL) != 0) {
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        c3_free(&c3_float);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    size_t out_count2 = (size_t)c2_out * (size_t)out_h1 * (size_t)out_w1;
    float mse2 = compute_mse(layer2_ref->data, layer2_int8->data, out_count2);
    float corr2 = compute_correlation(layer2_ref->data, layer2_int8->data, out_count2);

    c3_free(&c3_float);

    /* Layer3: Conv+BN (backbone_convs[2]). 입력 = layer2 출력 (C3, SiLU 없음) */
    conv2d_layer_t* conv3 = &model->backbone_convs[2].conv;
    batchnorm2d_layer_t* bn3 = model->backbone_convs[2].is_fused ? NULL : &model->backbone_convs[2].bn;
    int32_t in_c3 = conv3->in_channels;
    int32_t out_c3 = conv3->params.out_channels;
    int32_t k3 = conv3->params.kernel_size;
    int32_t s3 = conv3->params.stride;
    int32_t p3 = conv3->params.padding;
    int32_t d3 = conv3->params.dilation;
    int8_t* q_weight3 = conv3->q_weight;
    float scale_w3 = conv3->scale_w;

    if (!q_weight3 || scale_w3 <= 0.f) {
        fprintf(stderr, "Layer3 has no q_weight/scale_w.\n");
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    int32_t out_h3 = (out_h1 + 2 * p3 - d3 * (k3 - 1) - 1) / s3 + 1;
    int32_t out_w3 = (out_w1 + 2 * p3 - d3 * (k3 - 1) - 1) / s3 + 1;
    size_t out_count3 = (size_t)out_c3 * (size_t)out_h3 * (size_t)out_w3;
    size_t in_count3 = tensor_size(layer2_ref);

    tensor_t* layer3_ref = tensor_create(1, out_c3, out_h3, out_w3);
    tensor_t* layer3_int8 = tensor_create(1, out_c3, out_h3, out_w3);
    if (!layer3_ref || !layer3_int8) {
        if (layer3_ref) tensor_free(layer3_ref);
        if (layer3_int8) tensor_free(layer3_int8);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    if (bn3) {
        tensor_t* conv3_out = tensor_create(1, out_c3, out_h3, out_w3);
        if (!conv3_out ||
            conv2d_forward(conv3, layer2_ref, conv3_out) != 0 ||
            batchnorm2d_forward(bn3, conv3_out, layer3_ref) != 0) {
            if (conv3_out) tensor_free(conv3_out);
            tensor_free(layer3_int8);
            tensor_free(layer3_ref);
            tensor_free(layer2_int8);
            tensor_free(layer2_ref);
            tensor_free(layer1_int8);
            tensor_free(layer1_ref);
            free(acc32);
            free(q_input);
            tensor_free(out_int8_path);
            tensor_free(ref_float);
            tensor_free(input);
            yolov5n_free(model);
            return 1;
        }
        tensor_free(conv3_out);
    } else {
        if (conv2d_forward(conv3, layer2_ref, layer3_ref) != 0) {
            tensor_free(layer3_int8);
            tensor_free(layer3_ref);
            tensor_free(layer2_int8);
            tensor_free(layer2_ref);
            tensor_free(layer1_int8);
            tensor_free(layer1_ref);
            free(acc32);
            free(q_input);
            tensor_free(out_int8_path);
            tensor_free(ref_float);
            tensor_free(input);
            yolov5n_free(model);
            return 1;
        }
    }

    float in_min3, in_max3;
    tensor_minmax(layer2_int8->data, in_count3, &in_min3, &in_max3);
    quant_scale_t scale_in3;
    quant_scale_symmetric_from_minmax(in_min3, in_max3, &scale_in3);

    int8_t* q_input3 = (int8_t*)malloc(in_count3 * sizeof(int8_t));
    int32_t* acc32_3 = (int32_t*)malloc(out_count3 * sizeof(int32_t));
    if (!q_input3 || !acc32_3) {
        free(q_input3);
        free(acc32_3);
        tensor_free(layer3_int8);
        tensor_free(layer3_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    quantize_float_to_int8(layer2_int8->data, in_count3, scale_in3.scale, q_input3);

    if (conv2d_int8_forward(q_input3, q_weight3, 1, in_c3, out_h1, out_w1,
                            out_c3, k3, s3, p3, d3, acc32_3) != 0) {
        free(acc32_3);
        free(q_input3);
        tensor_free(layer3_int8);
        tensor_free(layer3_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    dequant_acc32_to_float(acc32_3, out_count3, out_c3, out_h3, out_w3,
                          scale_in3.scale, scale_w3, conv3->bias, layer3_int8->data);
    if (bn3) {
        tensor_t* tmp3 = tensor_create(1, out_c3, out_h3, out_w3);
        if (tmp3) {
            tensor_copy(tmp3, layer3_int8);
            batchnorm2d_forward(bn3, tmp3, layer3_int8);
            tensor_free(tmp3);
        }
    }

    float mse3 = compute_mse(layer3_ref->data, layer3_int8->data, out_count3);
    float corr3 = compute_correlation(layer3_ref->data, layer3_int8->data, out_count3);

    free(acc32_3);
    free(q_input3);

    /* Layer4: C3. 입력 = SiLU(layer3) */
    activation_silu(layer3_ref);
    activation_silu(layer3_int8);

    int32_t out_h4 = out_h3, out_w4 = out_w3;
    int32_t c4_out = model->backbone_c3s[1].c2;
    size_t out_count4 = (size_t)c4_out * (size_t)out_h4 * (size_t)out_w4;

    c3_block_t c3_float4;
    if (c3_init(&c3_float4, model->backbone_c3s[1].c1, model->backbone_c3s[1].c2,
               model->backbone_c3s[1].n, model->backbone_c3s[1].shortcut) != 0 ||
        c3_load_weights(&c3_float4, model->weights, "model.4", NULL) != 0) {
        c3_free(&c3_float4);
        tensor_free(layer3_int8);
        tensor_free(layer3_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_t* layer4_ref = tensor_create(1, c4_out, out_h4, out_w4);
    tensor_t* layer4_int8 = tensor_create(1, c4_out, out_h4, out_w4);
    if (!layer4_ref || !layer4_int8 ||
        c3_forward_float(&c3_float4, layer3_ref, layer4_ref, NULL, NULL) != 0 ||
        c3_forward(&model->backbone_c3s[1].block, layer3_int8, layer4_int8, NULL, NULL) != 0) {
        if (layer4_ref) tensor_free(layer4_ref);
        if (layer4_int8) tensor_free(layer4_int8);
        c3_free(&c3_float4);
        tensor_free(layer3_int8);
        tensor_free(layer3_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse4 = compute_mse(layer4_ref->data, layer4_int8->data, out_count4);
    float corr4 = compute_correlation(layer4_ref->data, layer4_int8->data, out_count4);
    c3_free(&c3_float4);
    tensor_free(layer3_int8);
    tensor_free(layer3_ref);

    /* Layer5: Conv backbone_convs[3]. 입력 = layer4 (no SiLU after C3) */
    conv2d_layer_t* conv5 = &model->backbone_convs[3].conv;
    batchnorm2d_layer_t* bn5 = model->backbone_convs[3].is_fused ? NULL : &model->backbone_convs[3].bn;
    int32_t out_c5 = conv5->params.out_channels, k5 = conv5->params.kernel_size;
    int32_t s5 = conv5->params.stride, p5 = conv5->params.padding, d5 = conv5->params.dilation;
    int32_t out_h5 = (out_h4 + 2 * p5 - d5 * (k5 - 1) - 1) / s5 + 1;
    int32_t out_w5 = (out_h4 + 2 * p5 - d5 * (k5 - 1) - 1) / s5 + 1;
    size_t out_count5 = (size_t)out_c5 * (size_t)out_h5 * (size_t)out_w5;
    size_t in_count5 = tensor_size(layer4_ref);
    if (!conv5->q_weight || conv5->scale_w <= 0.f) {
        tensor_free(layer4_int8);
        tensor_free(layer4_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_t* layer5_ref = tensor_create(1, out_c5, out_h5, out_w5);
    tensor_t* layer5_int8 = tensor_create(1, out_c5, out_h5, out_w5);
    if (!layer5_ref || !layer5_int8) {
        if (layer5_ref) tensor_free(layer5_ref);
        if (layer5_int8) tensor_free(layer5_int8);
        tensor_free(layer4_int8);
        tensor_free(layer4_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    if (bn5) {
        tensor_t* t5 = tensor_create(1, out_c5, out_h5, out_w5);
        if (!t5 || conv2d_forward(conv5, layer4_ref, t5) != 0 || batchnorm2d_forward(bn5, t5, layer5_ref) != 0) {
            if (t5) tensor_free(t5);
            tensor_free(layer5_int8);
            tensor_free(layer5_ref);
            tensor_free(layer4_int8);
            tensor_free(layer4_ref);
            tensor_free(layer2_int8);
            tensor_free(layer2_ref);
            tensor_free(layer1_int8);
            tensor_free(layer1_ref);
            free(acc32);
            free(q_input);
            tensor_free(out_int8_path);
            tensor_free(ref_float);
            tensor_free(input);
            yolov5n_free(model);
            return 1;
        }
        tensor_free(t5);
    } else if (conv2d_forward(conv5, layer4_ref, layer5_ref) != 0) {
        tensor_free(layer5_int8);
        tensor_free(layer5_ref);
        tensor_free(layer4_int8);
        tensor_free(layer4_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float in_min5, in_max5;
    tensor_minmax(layer4_int8->data, in_count5, &in_min5, &in_max5);
    quant_scale_t scale_in5;
    quant_scale_symmetric_from_minmax(in_min5, in_max5, &scale_in5);
    int8_t* q_input5 = (int8_t*)malloc(in_count5 * sizeof(int8_t));
    int32_t* acc32_5 = (int32_t*)malloc(out_count5 * sizeof(int32_t));
    if (!q_input5 || !acc32_5) {
        free(q_input5);
        free(acc32_5);
        tensor_free(layer5_int8);
        tensor_free(layer5_ref);
        tensor_free(layer4_int8);
        tensor_free(layer4_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    quantize_float_to_int8(layer4_int8->data, in_count5, scale_in5.scale, q_input5);
    if (conv2d_int8_forward(q_input5, conv5->q_weight, 1, conv5->in_channels, out_h4, out_w4,
                            out_c5, k5, s5, p5, d5, acc32_5) != 0) {
        free(acc32_5);
        free(q_input5);
        tensor_free(layer5_int8);
        tensor_free(layer5_ref);
        tensor_free(layer4_int8);
        tensor_free(layer4_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    dequant_acc32_to_float(acc32_5, out_count5, out_c5, out_h5, out_w5,
                          scale_in5.scale, conv5->scale_w, conv5->bias, layer5_int8->data);
    if (bn5) {
        tensor_t* t5b = tensor_create(1, out_c5, out_h5, out_w5);
        if (t5b) {
            tensor_copy(t5b, layer5_int8);
            batchnorm2d_forward(bn5, t5b, layer5_int8);
            tensor_free(t5b);
        }
    }
    float mse5 = compute_mse(layer5_ref->data, layer5_int8->data, out_count5);
    float corr5 = compute_correlation(layer5_ref->data, layer5_int8->data, out_count5);
    free(acc32_5);
    free(q_input5);

    /* Layer4 출력 저장 (Layer16 Concat용) */
    tensor_t* layer4_saved_ref = tensor_create(1, c4_out, out_h4, out_w4);
    tensor_t* layer4_saved_int8 = tensor_create(1, c4_out, out_h4, out_w4);
    if (!layer4_saved_ref || !layer4_saved_int8) {
        if (layer4_saved_ref) tensor_free(layer4_saved_ref);
        if (layer4_saved_int8) tensor_free(layer4_saved_int8);
        tensor_free(layer5_int8);
        tensor_free(layer5_ref);
        tensor_free(layer4_int8);
        tensor_free(layer4_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_copy(layer4_saved_ref, layer4_ref);
    tensor_copy(layer4_saved_int8, layer4_int8);
    tensor_free(layer4_int8);
    tensor_free(layer4_ref);

    /* Layer6: C3. 입력 = SiLU(layer5) */
    activation_silu(layer5_ref);
    activation_silu(layer5_int8);
    int32_t out_h6 = out_h5, out_w6 = out_w5;
    int32_t c6_out = model->backbone_c3s[2].c2;
    size_t out_count6 = (size_t)c6_out * (size_t)out_h6 * (size_t)out_w6;
    c3_block_t c3_float6;
    if (c3_init(&c3_float6, model->backbone_c3s[2].c1, model->backbone_c3s[2].c2,
               model->backbone_c3s[2].n, model->backbone_c3s[2].shortcut) != 0 ||
        c3_load_weights(&c3_float6, model->weights, "model.6", NULL) != 0) {
        c3_free(&c3_float6);
        tensor_free(layer5_int8);
        tensor_free(layer5_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_t* layer6_ref = tensor_create(1, c6_out, out_h6, out_w6);
    tensor_t* layer6_int8 = tensor_create(1, c6_out, out_h6, out_w6);
    if (!layer6_ref || !layer6_int8 ||
        c3_forward_float(&c3_float6, layer5_ref, layer6_ref, NULL, NULL) != 0 ||
        c3_forward(&model->backbone_c3s[2].block, layer5_int8, layer6_int8, NULL, NULL) != 0) {
        if (layer6_ref) tensor_free(layer6_ref);
        if (layer6_int8) tensor_free(layer6_int8);
        c3_free(&c3_float6);
        tensor_free(layer5_int8);
        tensor_free(layer5_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse6 = compute_mse(layer6_ref->data, layer6_int8->data, out_count6);
    float corr6 = compute_correlation(layer6_ref->data, layer6_int8->data, out_count6);
    c3_free(&c3_float6);
    tensor_free(layer5_int8);
    tensor_free(layer5_ref);

    /* Layer7: Conv backbone_convs[4]. 입력 = layer6 */
    conv2d_layer_t* conv7 = &model->backbone_convs[4].conv;
    batchnorm2d_layer_t* bn7 = model->backbone_convs[4].is_fused ? NULL : &model->backbone_convs[4].bn;
    int32_t out_c7 = conv7->params.out_channels, k7 = conv7->params.kernel_size;
    int32_t s7 = conv7->params.stride, p7 = conv7->params.padding, d7 = conv7->params.dilation;
    int32_t out_h7 = (out_h6 + 2 * p7 - d7 * (k7 - 1) - 1) / s7 + 1;
    int32_t out_w7 = (out_h6 + 2 * p7 - d7 * (k7 - 1) - 1) / s7 + 1;
    size_t out_count7 = (size_t)out_c7 * (size_t)out_h7 * (size_t)out_w7;
    size_t in_count7 = tensor_size(layer6_ref);
    if (!conv7->q_weight || conv7->scale_w <= 0.f) {
        tensor_free(layer6_int8);
        tensor_free(layer6_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_t* layer7_ref = tensor_create(1, out_c7, out_h7, out_w7);
    tensor_t* layer7_int8 = tensor_create(1, out_c7, out_h7, out_w7);
    if (!layer7_ref || !layer7_int8) {
        if (layer7_ref) tensor_free(layer7_ref);
        if (layer7_int8) tensor_free(layer7_int8);
        tensor_free(layer6_int8);
        tensor_free(layer6_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    if (bn7) {
        tensor_t* t7 = tensor_create(1, out_c7, out_h7, out_w7);
        if (!t7 || conv2d_forward(conv7, layer6_ref, t7) != 0 || batchnorm2d_forward(bn7, t7, layer7_ref) != 0) {
            if (t7) tensor_free(t7);
            tensor_free(layer7_int8);
            tensor_free(layer7_ref);
            tensor_free(layer6_int8);
            tensor_free(layer6_ref);
            tensor_free(layer2_int8);
            tensor_free(layer2_ref);
            tensor_free(layer1_int8);
            tensor_free(layer1_ref);
            free(acc32);
            free(q_input);
            tensor_free(out_int8_path);
            tensor_free(ref_float);
            tensor_free(input);
            yolov5n_free(model);
            return 1;
        }
        tensor_free(t7);
    } else if (conv2d_forward(conv7, layer6_ref, layer7_ref) != 0) {
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_int8);
        tensor_free(layer6_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float in_min7, in_max7;
    tensor_minmax(layer6_int8->data, in_count7, &in_min7, &in_max7);
    quant_scale_t scale_in7;
    quant_scale_symmetric_from_minmax(in_min7, in_max7, &scale_in7);
    int8_t* q_input7 = (int8_t*)malloc(in_count7 * sizeof(int8_t));
    int32_t* acc32_7 = (int32_t*)malloc(out_count7 * sizeof(int32_t));
    if (!q_input7 || !acc32_7) {
        free(q_input7);
        free(acc32_7);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_int8);
        tensor_free(layer6_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    quantize_float_to_int8(layer6_int8->data, in_count7, scale_in7.scale, q_input7);
    if (conv2d_int8_forward(q_input7, conv7->q_weight, 1, conv7->in_channels, out_h6, out_w6,
                            out_c7, k7, s7, p7, d7, acc32_7) != 0) {
        free(acc32_7);
        free(q_input7);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_int8);
        tensor_free(layer6_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    dequant_acc32_to_float(acc32_7, out_count7, out_c7, out_h7, out_w7,
                          scale_in7.scale, conv7->scale_w, conv7->bias, layer7_int8->data);
    if (bn7) {
        tensor_t* t7b = tensor_create(1, out_c7, out_h7, out_w7);
        if (t7b) {
            tensor_copy(t7b, layer7_int8);
            batchnorm2d_forward(bn7, t7b, layer7_int8);
            tensor_free(t7b);
        }
    }
    float mse7 = compute_mse(layer7_ref->data, layer7_int8->data, out_count7);
    float corr7 = compute_correlation(layer7_ref->data, layer7_int8->data, out_count7);
    free(acc32_7);
    free(q_input7);

    /* Layer6 출력 저장 (Layer12 Concat용) */
    tensor_t* layer6_saved_ref = tensor_create(1, c6_out, out_h6, out_w6);
    tensor_t* layer6_saved_int8 = tensor_create(1, c6_out, out_h6, out_w6);
    if (!layer6_saved_ref || !layer6_saved_int8) {
        if (layer6_saved_ref) tensor_free(layer6_saved_ref);
        if (layer6_saved_int8) tensor_free(layer6_saved_int8);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_int8);
        tensor_free(layer6_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_copy(layer6_saved_ref, layer6_ref);
    tensor_copy(layer6_saved_int8, layer6_int8);
    tensor_free(layer6_int8);
    tensor_free(layer6_ref);

    /* Layer8: C3 (backbone 끝). 입력 = SiLU(layer7) */
    activation_silu(layer7_ref);
    activation_silu(layer7_int8);
    int32_t out_h8 = out_h7, out_w8 = out_w7;
    int32_t c8_out = model->backbone_c3s[3].c2;
    size_t out_count8 = (size_t)c8_out * (size_t)out_h8 * (size_t)out_w8;
    c3_block_t c3_float8;
    if (c3_init(&c3_float8, model->backbone_c3s[3].c1, model->backbone_c3s[3].c2,
               model->backbone_c3s[3].n, model->backbone_c3s[3].shortcut) != 0 ||
        c3_load_weights(&c3_float8, model->weights, "model.8", NULL) != 0) {
        c3_free(&c3_float8);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_t* layer8_ref = tensor_create(1, c8_out, out_h8, out_w8);
    tensor_t* layer8_int8 = tensor_create(1, c8_out, out_h8, out_w8);
    if (!layer8_ref || !layer8_int8 ||
        c3_forward_float(&c3_float8, layer7_ref, layer8_ref, NULL, NULL) != 0 ||
        c3_forward(&model->backbone_c3s[3].block, layer7_int8, layer8_int8, NULL, NULL) != 0) {
        if (layer8_ref) tensor_free(layer8_ref);
        if (layer8_int8) tensor_free(layer8_int8);
        c3_free(&c3_float8);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse8 = compute_mse(layer8_ref->data, layer8_int8->data, out_count8);
    float corr8 = compute_correlation(layer8_ref->data, layer8_int8->data, out_count8);

    c3_free(&c3_float8);

    /* Layer9: SPPF (neck 시작). 입력 = layer8 (no SiLU after C3) */
    sppf_block_t sppf_float;
    if (sppf_init(&sppf_float, model->sppf.c1, model->sppf.c2, 5) != 0 ||
        sppf_load_weights(&sppf_float, model->weights, "model.9", NULL) != 0) {
        sppf_free(&sppf_float);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    int32_t c9_out = model->sppf.c2;
    size_t out_count9 = (size_t)c9_out * (size_t)out_h8 * (size_t)out_w8;
    tensor_t* layer9_ref = tensor_create(1, c9_out, out_h8, out_w8);
    tensor_t* layer9_int8 = tensor_create(1, c9_out, out_h8, out_w8);
    if (!layer9_ref || !layer9_int8 ||
        sppf_forward_float(&sppf_float, layer8_ref, layer9_ref, NULL, NULL, NULL) != 0 ||
        sppf_forward(&model->sppf, layer8_int8, layer9_int8, NULL, NULL, NULL) != 0) {
        if (layer9_ref) tensor_free(layer9_ref);
        if (layer9_int8) tensor_free(layer9_int8);
        sppf_free(&sppf_float);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse9 = compute_mse(layer9_ref->data, layer9_int8->data, out_count9);
    float corr9 = compute_correlation(layer9_ref->data, layer9_int8->data, out_count9);

    sppf_free(&sppf_float);

    /* Layer10: Conv 1x1 (head_convs[0], neck). 입력 = layer9 SPPF */
    conv2d_layer_t* conv10 = &model->head_convs[0].conv;
    batchnorm2d_layer_t* bn10 = model->head_convs[0].is_fused ? NULL : &model->head_convs[0].bn;
    int32_t out_c10 = conv10->params.out_channels;
    int32_t k10 = conv10->params.kernel_size;
    int32_t s10 = conv10->params.stride;
    int32_t p10 = conv10->params.padding;
    int32_t d10 = conv10->params.dilation;
    size_t out_count10 = (size_t)out_c10 * (size_t)out_h8 * (size_t)out_w8;
    size_t in_count10 = tensor_size(layer9_ref);
    if (!conv10->q_weight || conv10->scale_w <= 0.f) {
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_t* layer10_ref = tensor_create(1, out_c10, out_h8, out_w8);
    tensor_t* layer10_int8 = tensor_create(1, out_c10, out_h8, out_w8);
    if (!layer10_ref || !layer10_int8) {
        if (layer10_ref) tensor_free(layer10_ref);
        if (layer10_int8) tensor_free(layer10_int8);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    if (bn10) {
        tensor_t* t10 = tensor_create(1, out_c10, out_h8, out_w8);
        if (!t10 || conv2d_forward(conv10, layer9_ref, t10) != 0 || batchnorm2d_forward(bn10, t10, layer10_ref) != 0) {
            if (t10) tensor_free(t10);
            tensor_free(layer10_int8);
            tensor_free(layer10_ref);
            tensor_free(layer9_int8);
            tensor_free(layer9_ref);
            tensor_free(layer8_int8);
            tensor_free(layer8_ref);
            tensor_free(layer7_int8);
            tensor_free(layer7_ref);
            tensor_free(layer6_saved_int8);
            tensor_free(layer6_saved_ref);
            tensor_free(layer2_int8);
            tensor_free(layer2_ref);
            tensor_free(layer1_int8);
            tensor_free(layer1_ref);
            free(acc32);
            free(q_input);
            tensor_free(out_int8_path);
            tensor_free(ref_float);
            tensor_free(input);
            yolov5n_free(model);
            return 1;
        }
        tensor_free(t10);
    } else if (conv2d_forward(conv10, layer9_ref, layer10_ref) != 0) {
        tensor_free(layer10_int8);
        tensor_free(layer10_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float in_min10, in_max10;
    tensor_minmax(layer9_int8->data, in_count10, &in_min10, &in_max10);
    quant_scale_t scale_in10;
    quant_scale_symmetric_from_minmax(in_min10, in_max10, &scale_in10);
    int8_t* q_input10 = (int8_t*)malloc(in_count10 * sizeof(int8_t));
    int32_t* acc32_10 = (int32_t*)malloc(out_count10 * sizeof(int32_t));
    if (!q_input10 || !acc32_10) {
        free(q_input10);
        free(acc32_10);
        tensor_free(layer10_int8);
        tensor_free(layer10_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    quantize_float_to_int8(layer9_int8->data, in_count10, scale_in10.scale, q_input10);
    if (conv2d_int8_forward(q_input10, conv10->q_weight, 1, conv10->in_channels, out_h8, out_w8,
                            out_c10, k10, s10, p10, d10, acc32_10) != 0) {
        free(acc32_10);
        free(q_input10);
        tensor_free(layer10_int8);
        tensor_free(layer10_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    dequant_acc32_to_float(acc32_10, out_count10, out_c10, out_h8, out_w8,
                           scale_in10.scale, conv10->scale_w, conv10->bias, layer10_int8->data);
    if (bn10) {
        tensor_t* t10b = tensor_create(1, out_c10, out_h8, out_w8);
        if (t10b) {
            tensor_copy(t10b, layer10_int8);
            batchnorm2d_forward(bn10, t10b, layer10_int8);
            tensor_free(t10b);
        }
    }
    float mse10 = compute_mse(layer10_ref->data, layer10_int8->data, out_count10);
    float corr10 = compute_correlation(layer10_ref->data, layer10_int8->data, out_count10);

    free(acc32_10);
    free(q_input10);

    /* Layer11: Upsample x2 (가중치 없음). 입력 = layer10 */
    int32_t out_h11 = out_h8 * 2;
    int32_t out_w11 = out_w8 * 2;
    size_t out_count11 = (size_t)out_c10 * (size_t)out_h11 * (size_t)out_w11;
    tensor_t* layer11_ref = tensor_create(1, out_c10, out_h11, out_w11);
    tensor_t* layer11_int8 = tensor_create(1, out_c10, out_h11, out_w11);
    if (!layer11_ref || !layer11_int8) {
        if (layer11_ref) tensor_free(layer11_ref);
        if (layer11_int8) tensor_free(layer11_int8);
        tensor_free(layer10_int8);
        tensor_free(layer10_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    upsample_params_t upsample_params = { .scale_factor = 2, .mode = "nearest" };
    if (upsample_forward(&upsample_params, layer10_ref, layer11_ref) != 0 ||
        upsample_forward(&upsample_params, layer10_int8, layer11_int8) != 0) {
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer10_int8);
        tensor_free(layer10_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse11 = compute_mse(layer11_ref->data, layer11_int8->data, out_count11);
    float corr11 = compute_correlation(layer11_ref->data, layer11_int8->data, out_count11);

    /* Layer10 출력 저장 (Layer22 Concat용) */
    tensor_t* layer10_saved_ref = tensor_create(1, out_c10, out_h8, out_w8);
    tensor_t* layer10_saved_int8 = tensor_create(1, out_c10, out_h8, out_w8);
    if (!layer10_saved_ref || !layer10_saved_int8) {
        if (layer10_saved_ref) tensor_free(layer10_saved_ref);
        if (layer10_saved_int8) tensor_free(layer10_saved_int8);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer10_int8);
        tensor_free(layer10_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_copy(layer10_saved_ref, layer10_ref);
    tensor_copy(layer10_saved_int8, layer10_int8);
    tensor_free(layer10_int8);
    tensor_free(layer10_ref);

    /* Layer12: Concat(layer11, layer6_saved) along channel */
    int32_t c12_out = out_c10 + c6_out;
    size_t out_count12 = (size_t)c12_out * (size_t)out_h11 * (size_t)out_w11;
    tensor_t* layer12_ref = tensor_create(1, c12_out, out_h11, out_w11);
    tensor_t* layer12_int8 = tensor_create(1, c12_out, out_h11, out_w11);
    if (!layer12_ref || !layer12_int8) {
        if (layer12_ref) tensor_free(layer12_ref);
        if (layer12_int8) tensor_free(layer12_int8);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    const tensor_t* inputs_ref[2] = { layer11_ref, layer6_saved_ref };
    const tensor_t* inputs_int8[2] = { layer11_int8, layer6_saved_int8 };
    if (concat_forward(inputs_ref, 2, layer12_ref) != 0 || concat_forward(inputs_int8, 2, layer12_int8) != 0) {
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse12 = compute_mse(layer12_ref->data, layer12_int8->data, out_count12);
    float corr12 = compute_correlation(layer12_ref->data, layer12_int8->data, out_count12);

    /* Layer13: C3 (head_c3s[0]). 입력 = layer12 Concat 출력 */
    int32_t out_h13 = out_h11, out_w13 = out_w11;
    int32_t c13_out = model->head_c3s[0].c2;
    size_t out_count13 = (size_t)c13_out * (size_t)out_h13 * (size_t)out_w13;
    c3_block_t c3_float13;
    if (c3_init(&c3_float13, model->head_c3s[0].c1, model->head_c3s[0].c2,
                model->head_c3s[0].n, model->head_c3s[0].shortcut) != 0 ||
        c3_load_weights(&c3_float13, model->weights, "model.13", NULL) != 0) {
        c3_free(&c3_float13);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_t* layer13_ref = tensor_create(1, c13_out, out_h13, out_w13);
    tensor_t* layer13_int8 = tensor_create(1, c13_out, out_h13, out_w13);
    if (!layer13_ref || !layer13_int8) {
        if (layer13_ref) tensor_free(layer13_ref);
        if (layer13_int8) tensor_free(layer13_int8);
        c3_free(&c3_float13);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    if (c3_forward_float(&c3_float13, layer12_ref, layer13_ref, NULL, NULL) != 0 ||
        c3_forward(&model->head_c3s[0].block, layer12_int8, layer13_int8, NULL, NULL) != 0) {
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        c3_free(&c3_float13);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse13 = compute_mse(layer13_ref->data, layer13_int8->data, out_count13);
    float corr13 = compute_correlation(layer13_ref->data, layer13_int8->data, out_count13);

    c3_free(&c3_float13);

    /* Layer14: Conv 1x1 (head_convs[1]). 입력 = layer13 C3 */
    conv2d_layer_t* conv14 = &model->head_convs[1].conv;
    batchnorm2d_layer_t* bn14 = model->head_convs[1].is_fused ? NULL : &model->head_convs[1].bn;
    int32_t out_c14 = conv14->params.out_channels;
    int32_t k14 = conv14->params.kernel_size;
    int32_t s14 = conv14->params.stride;
    int32_t p14 = conv14->params.padding;
    int32_t d14 = conv14->params.dilation;
    int32_t out_h14 = out_h13, out_w14 = out_w13;
    size_t out_count14 = (size_t)out_c14 * (size_t)out_h14 * (size_t)out_w14;
    size_t in_count14 = tensor_size(layer13_ref);
    if (!conv14->q_weight || conv14->scale_w <= 0.f) {
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_t* layer14_ref = tensor_create(1, out_c14, out_h14, out_w14);
    tensor_t* layer14_int8 = tensor_create(1, out_c14, out_h14, out_w14);
    if (!layer14_ref || !layer14_int8) {
        if (layer14_ref) tensor_free(layer14_ref);
        if (layer14_int8) tensor_free(layer14_int8);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    if (bn14) {
        tensor_t* t14 = tensor_create(1, out_c14, out_h14, out_w14);
        if (!t14 || conv2d_forward(conv14, layer13_ref, t14) != 0 || batchnorm2d_forward(bn14, t14, layer14_ref) != 0) {
            if (t14) tensor_free(t14);
            tensor_free(layer14_int8);
            tensor_free(layer14_ref);
            tensor_free(layer13_int8);
            tensor_free(layer13_ref);
            tensor_free(layer12_int8);
            tensor_free(layer12_ref);
            tensor_free(layer11_int8);
            tensor_free(layer11_ref);
            tensor_free(layer6_saved_int8);
            tensor_free(layer6_saved_ref);
            tensor_free(layer10_saved_int8);
            tensor_free(layer10_saved_ref);
            tensor_free(layer9_int8);
            tensor_free(layer9_ref);
            tensor_free(layer8_int8);
            tensor_free(layer8_ref);
            tensor_free(layer7_int8);
            tensor_free(layer7_ref);
            tensor_free(layer4_saved_int8);
            tensor_free(layer4_saved_ref);
            tensor_free(layer2_int8);
            tensor_free(layer2_ref);
            tensor_free(layer1_int8);
            tensor_free(layer1_ref);
            free(acc32);
            free(q_input);
            tensor_free(out_int8_path);
            tensor_free(ref_float);
            tensor_free(input);
            yolov5n_free(model);
            return 1;
        }
        tensor_free(t14);
    } else if (conv2d_forward(conv14, layer13_ref, layer14_ref) != 0) {
        tensor_free(layer14_int8);
        tensor_free(layer14_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float in_min14, in_max14;
    tensor_minmax(layer13_int8->data, in_count14, &in_min14, &in_max14);
    quant_scale_t scale_in14;
    quant_scale_symmetric_from_minmax(in_min14, in_max14, &scale_in14);
    int8_t* q_input14 = (int8_t*)malloc(in_count14 * sizeof(int8_t));
    int32_t* acc32_14 = (int32_t*)malloc(out_count14 * sizeof(int32_t));
    if (!q_input14 || !acc32_14) {
        free(q_input14);
        free(acc32_14);
        tensor_free(layer14_int8);
        tensor_free(layer14_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    quantize_float_to_int8(layer13_int8->data, in_count14, scale_in14.scale, q_input14);
    if (conv2d_int8_forward(q_input14, conv14->q_weight, 1, conv14->in_channels, out_h13, out_w13,
                            out_c14, k14, s14, p14, d14, acc32_14) != 0) {
        free(acc32_14);
        free(q_input14);
        tensor_free(layer14_int8);
        tensor_free(layer14_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    dequant_acc32_to_float(acc32_14, out_count14, out_c14, out_h14, out_w14,
                          scale_in14.scale, conv14->scale_w, conv14->bias, layer14_int8->data);
    if (bn14) {
        tensor_t* t14b = tensor_create(1, out_c14, out_h14, out_w14);
        if (t14b) {
            tensor_copy(t14b, layer14_int8);
            batchnorm2d_forward(bn14, t14b, layer14_int8);
            tensor_free(t14b);
        }
    }
    float mse14 = compute_mse(layer14_ref->data, layer14_int8->data, out_count14);
    float corr14 = compute_correlation(layer14_ref->data, layer14_int8->data, out_count14);

    free(acc32_14);
    free(q_input14);

    /* Layer15: Upsample x2 (가중치 없음). 입력 = layer14 */
    int32_t out_h15 = out_h14 * 2;
    int32_t out_w15 = out_w14 * 2;
    size_t out_count15 = (size_t)out_c14 * (size_t)out_h15 * (size_t)out_w15;
    tensor_t* layer15_ref = tensor_create(1, out_c14, out_h15, out_w15);
    tensor_t* layer15_int8 = tensor_create(1, out_c14, out_h15, out_w15);
    if (!layer15_ref || !layer15_int8) {
        if (layer15_ref) tensor_free(layer15_ref);
        if (layer15_int8) tensor_free(layer15_int8);
        tensor_free(layer14_int8);
        tensor_free(layer14_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    upsample_params_t upsample_params_15 = { .scale_factor = 2, .mode = "nearest" };
    if (upsample_forward(&upsample_params_15, layer14_ref, layer15_ref) != 0 ||
        upsample_forward(&upsample_params_15, layer14_int8, layer15_int8) != 0) {
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer14_int8);
        tensor_free(layer14_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse15 = compute_mse(layer15_ref->data, layer15_int8->data, out_count15);
    float corr15 = compute_correlation(layer15_ref->data, layer15_int8->data, out_count15);

    /* Layer14 출력 저장 (Layer19 Concat용) */
    tensor_t* layer14_saved_ref = tensor_create(1, out_c14, out_h14, out_w14);
    tensor_t* layer14_saved_int8 = tensor_create(1, out_c14, out_h14, out_w14);
    if (!layer14_saved_ref || !layer14_saved_int8) {
        if (layer14_saved_ref) tensor_free(layer14_saved_ref);
        if (layer14_saved_int8) tensor_free(layer14_saved_int8);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer14_int8);
        tensor_free(layer14_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_copy(layer14_saved_ref, layer14_ref);
    tensor_copy(layer14_saved_int8, layer14_int8);
    tensor_free(layer14_int8);
    tensor_free(layer14_ref);

    /* Layer16: Concat(layer15, layer4_saved) along channel */
    int32_t c16_out = out_c14 + c4_out;
    size_t out_count16 = (size_t)c16_out * (size_t)out_h15 * (size_t)out_w15;
    tensor_t* layer16_ref = tensor_create(1, c16_out, out_h15, out_w15);
    tensor_t* layer16_int8 = tensor_create(1, c16_out, out_h15, out_w15);
    if (!layer16_ref || !layer16_int8) {
        if (layer16_ref) tensor_free(layer16_ref);
        if (layer16_int8) tensor_free(layer16_int8);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    const tensor_t* inputs16_ref[2] = { layer15_ref, layer4_saved_ref };
    const tensor_t* inputs16_int8[2] = { layer15_int8, layer4_saved_int8 };
    if (concat_forward(inputs16_ref, 2, layer16_ref) != 0 || concat_forward(inputs16_int8, 2, layer16_int8) != 0) {
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse16 = compute_mse(layer16_ref->data, layer16_int8->data, out_count16);
    float corr16 = compute_correlation(layer16_ref->data, layer16_int8->data, out_count16);

    /* Layer17: C3 (head_c3s[1]). 입력 = layer16 Concat 출력 */
    int32_t out_h17 = out_h15, out_w17 = out_w15;
    int32_t c17_out = model->head_c3s[1].c2;
    size_t out_count17 = (size_t)c17_out * (size_t)out_h17 * (size_t)out_w17;
    c3_block_t c3_float17;
    if (c3_init(&c3_float17, model->head_c3s[1].c1, model->head_c3s[1].c2,
                model->head_c3s[1].n, model->head_c3s[1].shortcut) != 0 ||
        c3_load_weights(&c3_float17, model->weights, "model.17", NULL) != 0) {
        c3_free(&c3_float17);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_t* layer17_ref = tensor_create(1, c17_out, out_h17, out_w17);
    tensor_t* layer17_int8 = tensor_create(1, c17_out, out_h17, out_w17);
    if (!layer17_ref || !layer17_int8) {
        if (layer17_ref) tensor_free(layer17_ref);
        if (layer17_int8) tensor_free(layer17_int8);
        c3_free(&c3_float17);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    if (c3_forward_float(&c3_float17, layer16_ref, layer17_ref, NULL, NULL) != 0 ||
        c3_forward(&model->head_c3s[1].block, layer16_int8, layer17_int8, NULL, NULL) != 0) {
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        c3_free(&c3_float17);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse17 = compute_mse(layer17_ref->data, layer17_int8->data, out_count17);
    float corr17 = compute_correlation(layer17_ref->data, layer17_int8->data, out_count17);

    c3_free(&c3_float17);

    /* Layer18: Conv 3x3 s=2 (head_convs[2]). 입력 = layer17 C3 */
    conv2d_layer_t* conv18 = &model->head_convs[2].conv;
    batchnorm2d_layer_t* bn18 = model->head_convs[2].is_fused ? NULL : &model->head_convs[2].bn;
    int32_t out_c18 = conv18->params.out_channels;
    int32_t k18 = conv18->params.kernel_size;
    int32_t s18 = conv18->params.stride;
    int32_t p18 = conv18->params.padding;
    int32_t d18 = conv18->params.dilation;
    int32_t out_h18 = (out_h17 + 2 * p18 - d18 * (k18 - 1) - 1) / s18 + 1;
    int32_t out_w18 = (out_w17 + 2 * p18 - d18 * (k18 - 1) - 1) / s18 + 1;
    size_t out_count18 = (size_t)out_c18 * (size_t)out_h18 * (size_t)out_w18;
    size_t in_count18 = tensor_size(layer17_ref);
    if (!conv18->q_weight || conv18->scale_w <= 0.f) {
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_t* layer18_ref = tensor_create(1, out_c18, out_h18, out_w18);
    tensor_t* layer18_int8 = tensor_create(1, out_c18, out_h18, out_w18);
    if (!layer18_ref || !layer18_int8) {
        if (layer18_ref) tensor_free(layer18_ref);
        if (layer18_int8) tensor_free(layer18_int8);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    if (bn18) {
        tensor_t* t18 = tensor_create(1, out_c18, out_h18, out_w18);
        if (!t18 || conv2d_forward(conv18, layer17_ref, t18) != 0 || batchnorm2d_forward(bn18, t18, layer18_ref) != 0) {
            if (t18) tensor_free(t18);
            tensor_free(layer18_int8);
            tensor_free(layer18_ref);
            tensor_free(layer17_int8);
            tensor_free(layer17_ref);
            tensor_free(layer16_int8);
            tensor_free(layer16_ref);
            tensor_free(layer15_int8);
            tensor_free(layer15_ref);
            tensor_free(layer4_saved_int8);
            tensor_free(layer4_saved_ref);
            tensor_free(layer14_saved_int8);
            tensor_free(layer14_saved_ref);
            tensor_free(layer13_int8);
            tensor_free(layer13_ref);
            tensor_free(layer12_int8);
            tensor_free(layer12_ref);
            tensor_free(layer11_int8);
            tensor_free(layer11_ref);
            tensor_free(layer6_saved_int8);
            tensor_free(layer6_saved_ref);
            tensor_free(layer10_saved_int8);
            tensor_free(layer10_saved_ref);
            tensor_free(layer9_int8);
            tensor_free(layer9_ref);
            tensor_free(layer8_int8);
            tensor_free(layer8_ref);
            tensor_free(layer7_int8);
            tensor_free(layer7_ref);
            tensor_free(layer2_int8);
            tensor_free(layer2_ref);
            tensor_free(layer1_int8);
            tensor_free(layer1_ref);
            free(acc32);
            free(q_input);
            tensor_free(out_int8_path);
            tensor_free(ref_float);
            tensor_free(input);
            yolov5n_free(model);
            return 1;
        }
        tensor_free(t18);
    } else if (conv2d_forward(conv18, layer17_ref, layer18_ref) != 0) {
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float in_min18, in_max18;
    tensor_minmax(layer17_int8->data, in_count18, &in_min18, &in_max18);
    quant_scale_t scale_in18;
    quant_scale_symmetric_from_minmax(in_min18, in_max18, &scale_in18);
    int8_t* q_input18 = (int8_t*)malloc(in_count18 * sizeof(int8_t));
    int32_t* acc32_18 = (int32_t*)malloc(out_count18 * sizeof(int32_t));
    if (!q_input18 || !acc32_18) {
        free(q_input18);
        free(acc32_18);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    quantize_float_to_int8(layer17_int8->data, in_count18, scale_in18.scale, q_input18);
    if (conv2d_int8_forward(q_input18, conv18->q_weight, 1, conv18->in_channels, out_h17, out_w17,
                            out_c18, k18, s18, p18, d18, acc32_18) != 0) {
        free(acc32_18);
        free(q_input18);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    dequant_acc32_to_float(acc32_18, out_count18, out_c18, out_h18, out_w18,
                          scale_in18.scale, conv18->scale_w, conv18->bias, layer18_int8->data);
    if (bn18) {
        tensor_t* t18b = tensor_create(1, out_c18, out_h18, out_w18);
        if (t18b) {
            tensor_copy(t18b, layer18_int8);
            batchnorm2d_forward(bn18, t18b, layer18_int8);
            tensor_free(t18b);
        }
    }
    float mse18 = compute_mse(layer18_ref->data, layer18_int8->data, out_count18);
    float corr18 = compute_correlation(layer18_ref->data, layer18_int8->data, out_count18);

    free(acc32_18);
    free(q_input18);

    /* Layer19: Concat(layer18, layer14_saved) along channel */
    int32_t c19_out = out_c18 + out_c14;
    size_t out_count19 = (size_t)c19_out * (size_t)out_h18 * (size_t)out_w18;
    tensor_t* layer19_ref = tensor_create(1, c19_out, out_h18, out_w18);
    tensor_t* layer19_int8 = tensor_create(1, c19_out, out_h18, out_w18);
    if (!layer19_ref || !layer19_int8) {
        if (layer19_ref) tensor_free(layer19_ref);
        if (layer19_int8) tensor_free(layer19_int8);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    const tensor_t* inputs19_ref[2] = { layer18_ref, layer14_saved_ref };
    const tensor_t* inputs19_int8[2] = { layer18_int8, layer14_saved_int8 };
    if (concat_forward(inputs19_ref, 2, layer19_ref) != 0 || concat_forward(inputs19_int8, 2, layer19_int8) != 0) {
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse19 = compute_mse(layer19_ref->data, layer19_int8->data, out_count19);
    float corr19 = compute_correlation(layer19_ref->data, layer19_int8->data, out_count19);

    /* Layer20: C3 (head_c3s[2]). 입력 = layer19 Concat 출력 */
    int32_t out_h20 = out_h18, out_w20 = out_w18;
    int32_t c20_out = model->head_c3s[2].c2;
    size_t out_count20 = (size_t)c20_out * (size_t)out_h20 * (size_t)out_w20;
    c3_block_t c3_float20;
    if (c3_init(&c3_float20, model->head_c3s[2].c1, model->head_c3s[2].c2,
                model->head_c3s[2].n, model->head_c3s[2].shortcut) != 0 ||
        c3_load_weights(&c3_float20, model->weights, "model.20", NULL) != 0) {
        c3_free(&c3_float20);
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_t* layer20_ref = tensor_create(1, c20_out, out_h20, out_w20);
    tensor_t* layer20_int8 = tensor_create(1, c20_out, out_h20, out_w20);
    if (!layer20_ref || !layer20_int8) {
        if (layer20_ref) tensor_free(layer20_ref);
        if (layer20_int8) tensor_free(layer20_int8);
        c3_free(&c3_float20);
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    if (c3_forward_float(&c3_float20, layer19_ref, layer20_ref, NULL, NULL) != 0 ||
        c3_forward(&model->head_c3s[2].block, layer19_int8, layer20_int8, NULL, NULL) != 0) {
        tensor_free(layer20_int8);
        tensor_free(layer20_ref);
        c3_free(&c3_float20);
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse20 = compute_mse(layer20_ref->data, layer20_int8->data, out_count20);
    float corr20 = compute_correlation(layer20_ref->data, layer20_int8->data, out_count20);

    c3_free(&c3_float20);

    /* Layer21: Conv 3x3 s=2 (head_convs[3]). 입력 = layer20 C3 */
    conv2d_layer_t* conv21 = &model->head_convs[3].conv;
    batchnorm2d_layer_t* bn21 = model->head_convs[3].is_fused ? NULL : &model->head_convs[3].bn;
    int32_t out_c21 = conv21->params.out_channels;
    int32_t k21 = conv21->params.kernel_size;
    int32_t s21 = conv21->params.stride;
    int32_t p21 = conv21->params.padding;
    int32_t d21 = conv21->params.dilation;
    int32_t out_h21 = (out_h20 + 2 * p21 - d21 * (k21 - 1) - 1) / s21 + 1;
    int32_t out_w21 = (out_w20 + 2 * p21 - d21 * (k21 - 1) - 1) / s21 + 1;
    size_t out_count21 = (size_t)out_c21 * (size_t)out_h21 * (size_t)out_w21;
    size_t in_count21 = tensor_size(layer20_ref);
    if (!conv21->q_weight || conv21->scale_w <= 0.f) {
        tensor_free(layer20_int8);
        tensor_free(layer20_ref);
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_t* layer21_ref = tensor_create(1, out_c21, out_h21, out_w21);
    tensor_t* layer21_int8 = tensor_create(1, out_c21, out_h21, out_w21);
    if (!layer21_ref || !layer21_int8) {
        if (layer21_ref) tensor_free(layer21_ref);
        if (layer21_int8) tensor_free(layer21_int8);
        tensor_free(layer20_int8);
        tensor_free(layer20_ref);
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    if (bn21) {
        tensor_t* t21 = tensor_create(1, out_c21, out_h21, out_w21);
        if (!t21 || conv2d_forward(conv21, layer20_ref, t21) != 0 || batchnorm2d_forward(bn21, t21, layer21_ref) != 0) {
            if (t21) tensor_free(t21);
            tensor_free(layer21_int8);
            tensor_free(layer21_ref);
            tensor_free(layer20_int8);
            tensor_free(layer20_ref);
            tensor_free(layer19_int8);
            tensor_free(layer19_ref);
            tensor_free(layer18_int8);
            tensor_free(layer18_ref);
            tensor_free(layer17_int8);
            tensor_free(layer17_ref);
            tensor_free(layer16_int8);
            tensor_free(layer16_ref);
            tensor_free(layer15_int8);
            tensor_free(layer15_ref);
            tensor_free(layer4_saved_int8);
            tensor_free(layer4_saved_ref);
            tensor_free(layer14_saved_int8);
            tensor_free(layer14_saved_ref);
            tensor_free(layer13_int8);
            tensor_free(layer13_ref);
            tensor_free(layer12_int8);
            tensor_free(layer12_ref);
            tensor_free(layer11_int8);
            tensor_free(layer11_ref);
            tensor_free(layer6_saved_int8);
            tensor_free(layer6_saved_ref);
            tensor_free(layer10_saved_int8);
            tensor_free(layer10_saved_ref);
            tensor_free(layer9_int8);
            tensor_free(layer9_ref);
            tensor_free(layer8_int8);
            tensor_free(layer8_ref);
            tensor_free(layer7_int8);
            tensor_free(layer7_ref);
            tensor_free(layer2_int8);
            tensor_free(layer2_ref);
            tensor_free(layer1_int8);
            tensor_free(layer1_ref);
            free(acc32);
            free(q_input);
            tensor_free(out_int8_path);
            tensor_free(ref_float);
            tensor_free(input);
            yolov5n_free(model);
            return 1;
        }
        tensor_free(t21);
    } else if (conv2d_forward(conv21, layer20_ref, layer21_ref) != 0) {
        tensor_free(layer21_int8);
        tensor_free(layer21_ref);
        tensor_free(layer20_int8);
        tensor_free(layer20_ref);
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float in_min21, in_max21;
    tensor_minmax(layer20_int8->data, in_count21, &in_min21, &in_max21);
    quant_scale_t scale_in21;
    quant_scale_symmetric_from_minmax(in_min21, in_max21, &scale_in21);
    int8_t* q_input21 = (int8_t*)malloc(in_count21 * sizeof(int8_t));
    int32_t* acc32_21 = (int32_t*)malloc(out_count21 * sizeof(int32_t));
    if (!q_input21 || !acc32_21) {
        free(q_input21);
        free(acc32_21);
        tensor_free(layer21_int8);
        tensor_free(layer21_ref);
        tensor_free(layer20_int8);
        tensor_free(layer20_ref);
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    quantize_float_to_int8(layer20_int8->data, in_count21, scale_in21.scale, q_input21);
    if (conv2d_int8_forward(q_input21, conv21->q_weight, 1, conv21->in_channels, out_h20, out_w20,
                            out_c21, k21, s21, p21, d21, acc32_21) != 0) {
        free(acc32_21);
        free(q_input21);
        tensor_free(layer21_int8);
        tensor_free(layer21_ref);
        tensor_free(layer20_int8);
        tensor_free(layer20_ref);
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    dequant_acc32_to_float(acc32_21, out_count21, out_c21, out_h21, out_w21,
                          scale_in21.scale, conv21->scale_w, conv21->bias, layer21_int8->data);
    if (bn21) {
        tensor_t* t21b = tensor_create(1, out_c21, out_h21, out_w21);
        if (t21b) {
            tensor_copy(t21b, layer21_int8);
            batchnorm2d_forward(bn21, t21b, layer21_int8);
            tensor_free(t21b);
        }
    }
    float mse21 = compute_mse(layer21_ref->data, layer21_int8->data, out_count21);
    float corr21 = compute_correlation(layer21_ref->data, layer21_int8->data, out_count21);

    free(acc32_21);
    free(q_input21);

    /* Layer22: Concat(layer21, layer10_saved) along channel */
    int32_t c22_out = out_c21 + out_c10;
    size_t out_count22 = (size_t)c22_out * (size_t)out_h21 * (size_t)out_w21;
    tensor_t* layer22_ref = tensor_create(1, c22_out, out_h21, out_w21);
    tensor_t* layer22_int8 = tensor_create(1, c22_out, out_h21, out_w21);
    if (!layer22_ref || !layer22_int8) {
        if (layer22_ref) tensor_free(layer22_ref);
        if (layer22_int8) tensor_free(layer22_int8);
        tensor_free(layer21_int8);
        tensor_free(layer21_ref);
        tensor_free(layer20_int8);
        tensor_free(layer20_ref);
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    const tensor_t* inputs22_ref[2] = { layer21_ref, layer10_saved_ref };
    const tensor_t* inputs22_int8[2] = { layer21_int8, layer10_saved_int8 };
    if (concat_forward(inputs22_ref, 2, layer22_ref) != 0 || concat_forward(inputs22_int8, 2, layer22_int8) != 0) {
        tensor_free(layer22_int8);
        tensor_free(layer22_ref);
        tensor_free(layer21_int8);
        tensor_free(layer21_ref);
        tensor_free(layer20_int8);
        tensor_free(layer20_ref);
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse22 = compute_mse(layer22_ref->data, layer22_int8->data, out_count22);
    float corr22 = compute_correlation(layer22_ref->data, layer22_int8->data, out_count22);

    /* Layer23: C3 (head_c3s[3]). 입력 = layer22 Concat 출력 */
    int32_t out_h23 = out_h21, out_w23 = out_w21;
    int32_t c23_out = model->head_c3s[3].c2;
    size_t out_count23 = (size_t)c23_out * (size_t)out_h23 * (size_t)out_w23;
    c3_block_t c3_float23;
    if (c3_init(&c3_float23, model->head_c3s[3].c1, model->head_c3s[3].c2,
                model->head_c3s[3].n, model->head_c3s[3].shortcut) != 0 ||
        c3_load_weights(&c3_float23, model->weights, "model.23", NULL) != 0) {
        c3_free(&c3_float23);
        tensor_free(layer22_int8);
        tensor_free(layer22_ref);
        tensor_free(layer21_int8);
        tensor_free(layer21_ref);
        tensor_free(layer20_int8);
        tensor_free(layer20_ref);
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    tensor_t* layer23_ref = tensor_create(1, c23_out, out_h23, out_w23);
    tensor_t* layer23_int8 = tensor_create(1, c23_out, out_h23, out_w23);
    if (!layer23_ref || !layer23_int8) {
        if (layer23_ref) tensor_free(layer23_ref);
        if (layer23_int8) tensor_free(layer23_int8);
        c3_free(&c3_float23);
        tensor_free(layer22_int8);
        tensor_free(layer22_ref);
        tensor_free(layer21_int8);
        tensor_free(layer21_ref);
        tensor_free(layer20_int8);
        tensor_free(layer20_ref);
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    if (c3_forward_float(&c3_float23, layer22_ref, layer23_ref, NULL, NULL) != 0 ||
        c3_forward(&model->head_c3s[3].block, layer22_int8, layer23_int8, NULL, NULL) != 0) {
        tensor_free(layer23_int8);
        tensor_free(layer23_ref);
        c3_free(&c3_float23);
        tensor_free(layer22_int8);
        tensor_free(layer22_ref);
        tensor_free(layer21_int8);
        tensor_free(layer21_ref);
        tensor_free(layer20_int8);
        tensor_free(layer20_ref);
        tensor_free(layer19_int8);
        tensor_free(layer19_ref);
        tensor_free(layer18_int8);
        tensor_free(layer18_ref);
        tensor_free(layer17_int8);
        tensor_free(layer17_ref);
        tensor_free(layer16_int8);
        tensor_free(layer16_ref);
        tensor_free(layer15_int8);
        tensor_free(layer15_ref);
        tensor_free(layer4_saved_int8);
        tensor_free(layer4_saved_ref);
        tensor_free(layer14_saved_int8);
        tensor_free(layer14_saved_ref);
        tensor_free(layer13_int8);
        tensor_free(layer13_ref);
        tensor_free(layer12_int8);
        tensor_free(layer12_ref);
        tensor_free(layer11_int8);
        tensor_free(layer11_ref);
        tensor_free(layer6_saved_int8);
        tensor_free(layer6_saved_ref);
        tensor_free(layer10_saved_int8);
        tensor_free(layer10_saved_ref);
        tensor_free(layer9_int8);
        tensor_free(layer9_ref);
        tensor_free(layer8_int8);
        tensor_free(layer8_ref);
        tensor_free(layer7_int8);
        tensor_free(layer7_ref);
        tensor_free(layer2_int8);
        tensor_free(layer2_ref);
        tensor_free(layer1_int8);
        tensor_free(layer1_ref);
        free(acc32);
        free(q_input);
        tensor_free(out_int8_path);
        tensor_free(ref_float);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }
    float mse23 = compute_mse(layer23_ref->data, layer23_int8->data, out_count23);
    float corr23 = compute_correlation(layer23_ref->data, layer23_int8->data, out_count23);

    c3_free(&c3_float23);
    tensor_free(layer23_int8);
    tensor_free(layer23_ref);
    tensor_free(layer22_int8);
    tensor_free(layer22_ref);
    tensor_free(layer21_int8);
    tensor_free(layer21_ref);
    tensor_free(layer20_int8);
    tensor_free(layer20_ref);
    tensor_free(layer19_int8);
    tensor_free(layer19_ref);
    tensor_free(layer18_int8);
    tensor_free(layer18_ref);
    tensor_free(layer17_int8);
    tensor_free(layer17_ref);
    tensor_free(layer16_int8);
    tensor_free(layer16_ref);
    tensor_free(layer15_int8);
    tensor_free(layer15_ref);
    tensor_free(layer4_saved_int8);
    tensor_free(layer4_saved_ref);
    tensor_free(layer14_saved_int8);
    tensor_free(layer14_saved_ref);
    tensor_free(layer13_int8);
    tensor_free(layer13_ref);
    tensor_free(layer12_int8);
    tensor_free(layer12_ref);
    tensor_free(layer11_int8);
    tensor_free(layer11_ref);
    tensor_free(layer6_saved_int8);
    tensor_free(layer6_saved_ref);
    tensor_free(layer10_saved_int8);
    tensor_free(layer10_saved_ref);
    tensor_free(layer9_int8);
    tensor_free(layer9_ref);
    tensor_free(layer8_int8);
    tensor_free(layer8_ref);
    tensor_free(layer7_int8);
    tensor_free(layer7_ref);
    tensor_free(layer2_int8);
    tensor_free(layer2_ref);
    tensor_free(layer1_int8);
    tensor_free(layer1_ref);
    free(acc32);
    free(q_input);
    tensor_free(out_int8_path);
    tensor_free(ref_float);
    tensor_free(input);
    yolov5n_free(model);

    printf("\n--- Layer0 Conv+BN (float vs int8, input: %s) ---\n", input_bin);
    printf("  MSE:   %.6f  corr: %.6f\n", mse0, corr0);
    printf("\n--- Layer1 Conv+BN (input = SiLU(layer0), float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse1, corr1);
    printf("\n--- Layer2 C3 (input = SiLU(layer1), float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse2, corr2);
    printf("\n--- Layer3 Conv+BN (input = layer2 C3, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse3, corr3);
    printf("\n--- Layer4 C3 (input = SiLU(layer3), float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse4, corr4);
    printf("\n--- Layer5 Conv+BN (input = layer4 C3, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse5, corr5);
    printf("\n--- Layer6 C3 (input = SiLU(layer5), float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse6, corr6);
    printf("\n--- Layer7 Conv+BN (input = layer6 C3, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse7, corr7);
    printf("\n--- Layer8 C3 (input = SiLU(layer7), backbone end, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse8, corr8);
    printf("\n--- Layer9 SPPF (input = layer8 C3, neck, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse9, corr9);
    printf("\n--- Layer10 Conv 1x1 (input = layer9 SPPF, neck, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse10, corr10);
    printf("\n--- Layer11 Upsample x2 (input = layer10, no weights, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse11, corr11);
    printf("\n--- Layer12 Concat(layer11, layer6) (float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse12, corr12);
    printf("\n--- Layer13 C3 (input = layer12 Concat, head, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse13, corr13);
    printf("\n--- Layer14 Conv 1x1 (input = layer13 C3, head, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse14, corr14);
    printf("\n--- Layer15 Upsample x2 (input = layer14, no weights, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse15, corr15);
    printf("\n--- Layer16 Concat(layer15, layer4) (float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse16, corr16);
    printf("\n--- Layer17 C3 (input = layer16 Concat, head, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse17, corr17);
    printf("\n--- Layer18 Conv 3x3 s=2 (input = layer17 C3, head, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse18, corr18);
    printf("\n--- Layer19 Concat(layer18, layer14) (float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse19, corr19);
    printf("\n--- Layer20 C3 (input = layer19 Concat, head, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse20, corr20);
    printf("\n--- Layer21 Conv 3x3 s=2 (input = layer20 C3, head, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse21, corr21);
    printf("\n--- Layer22 Concat(layer21, layer10) (float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse22, corr22);
    printf("\n--- Layer23 C3 (input = layer22 Concat, head, float vs int8) ---\n");
    printf("  MSE:   %.6f  corr: %.6f\n", mse23, corr23);
    printf("=== Done ===\n");
    return 0;
}
