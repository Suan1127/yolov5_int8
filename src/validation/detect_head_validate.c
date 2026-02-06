/**
 * Detect head 양자화 검증 (float vs int8)
 *
 * - P3, P4, P5 feature에 대해 Detect head 1x1 Conv를 float / int8 각각 수행 후 비교
 * - Layer0 검증과 동일한 방식: MSE, 상관계수 출력
 * - 인자: [weights.bin] [model_meta.json] [input.bin]
 */

#include "../models/yolov5n_build.h"
#include "../models/yolov5n_infer.h"
#include "../core/tensor.h"
#include "../ops/conv2d.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
    const char* weights_path = (argc >= 2) ? argv[1] : "../../weights/yolov5n/weights_int8.bin";
    const char* meta_path    = (argc >= 3) ? argv[2] : "../../weights/yolov5n/model_meta_fused.json";
    const char* input_bin    = (argc >= 4) ? argv[3] : "../../data/yolov5n/inputs/bus.bin";

    printf("=== Detect head validation (1x1 Conv int8) ===\n");
    printf("Weights: %s\n", weights_path);
    printf("Meta:    %s\n", meta_path);
    printf("Input:   %s\n", input_bin);

    yolov5n_model_t* model = yolov5n_build(weights_path, meta_path);
    if (!model) {
        fprintf(stderr, "Failed to build model\n");
        return 1;
    }

    tensor_t* input = tensor_load(input_bin);
    if (!input) {
        fprintf(stderr, "Cannot load input: %s\n", input_bin);
        yolov5n_free(model);
        return 1;
    }
    if (input->n != 1 || input->c != 3) {
        fprintf(stderr, "Input shape must be (1, 3, H, W), got %d x %d x %d x %d\n",
                input->n, input->c, input->h, input->w);
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    /* Forward까지 수행해 P3, P4, P5 feature 확보 (saved_features 17, 20, 23) */
    tensor_t* outputs[3] = { NULL, NULL, NULL };
    int ret = yolov5n_forward(model, input, outputs);
    if (ret != 0) {
        fprintf(stderr, "yolov5n_forward failed\n");
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    tensor_t* p3 = yolov5n_get_saved_feature(model, 17);
    tensor_t* p4 = yolov5n_get_saved_feature(model, 20);
    tensor_t* p5 = yolov5n_get_saved_feature(model, 23);
    if (!p3 || !p4 || !p5) {
        fprintf(stderr, "Failed to get P3/P4/P5 features (saved_features 17, 20, 23)\n");
        tensor_free(input);
        yolov5n_free(model);
        return 1;
    }

    const tensor_t* features[3] = { p3, p4, p5 };
    int32_t out_c = model->detect_convs[0].conv.params.out_channels; /* 255 */
    const char* names[] = { "P3", "P4", "P5" };
    int has_int8 = 0;

    for (int i = 0; i < 3; i++) {
        const conv2d_layer_t* conv = &model->detect_convs[i].conv;
        const tensor_t* feat = features[i];
        int32_t out_h = feat->h, out_w = feat->w;
        size_t out_count = (size_t)out_c * (size_t)out_h * (size_t)out_w;

        tensor_t* ref_float = tensor_create(1, out_c, out_h, out_w);
        tensor_t* out_int8  = tensor_create(1, out_c, out_h, out_w);
        if (!ref_float || !out_int8) {
            if (ref_float) tensor_free(ref_float);
            if (out_int8) tensor_free(out_int8);
            continue;
        }

        /* Int8 forward (float ref removed: int8-only build) */
        if (conv->q_weight && conv->scale_w > 0.f) {
            has_int8 = 1;
            if (conv2d_quant_forward(conv, feat, out_int8) != 0) {
                fprintf(stderr, "Detect %s int8 forward failed\n", names[i]);
                tensor_free(ref_float);
                tensor_free(out_int8);
                continue;
            }
            /* Optional: float reference if weight loaded (not in int8-only build) */
            if (conv->weight && conv2d_forward(conv, feat, ref_float) == 0) {
                float mse = compute_mse(ref_float->data, out_int8->data, out_count);
                float corr = compute_correlation(ref_float->data, out_int8->data, out_count);
                printf("\n--- Detect head %s (float vs int8) ---\n", names[i]);
                printf("  MSE: %.6e  Correlation: %.6f\n", mse, corr);
            } else {
                printf("\n--- Detect head %s: int8 forward OK ---\n", names[i]);
            }
        } else {
            printf("\n--- Detect head %s: no int8 weights (q_weight/scale_w missing) ---\n", names[i]);
        }

        tensor_free(ref_float);
        tensor_free(out_int8);
    }

    if (!has_int8)
        printf("\n(weights_int8.bin + scales_int8.json with model.24.m.0/1/2.weight 있으면 int8 비교 수행)\n");

    tensor_free(input);
    yolov5n_free(model);

    printf("\nDetect head validation done.\n");
    return 0;
}
