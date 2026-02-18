#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include "../src/core/tensor.h"
#include "../src/ops/conv2d.h"

int test_conv1x1_int8_basic() {
    printf("Testing 1x1 convolution (int8)...\n");

    tensor_t* input = tensor_create(1, 3, 4, 4);
    assert(input != NULL);
    for (int i = 0; i < 3 * 4 * 4; i++)
        input->data[i] = 1.0f;

    conv2d_params_t params = {
        .out_channels = 8,
        .kernel_size = 1,
        .stride = 1,
        .padding = 0,
        .groups = 1,
        .dilation = 1
    };

    conv2d_layer_t layer;
    int ret = conv2d_init(&layer, 3, &params);
    assert(ret == 0);

    /* Int8 weights: 8*3*1*1 = 24. Identity-like: oc==ic -> 64, else 0. scale_w for dequant. */
    size_t numel = (size_t)(8 * 3 * 1 * 1);
    int8_t* q_weight = (int8_t*)malloc(numel * sizeof(int8_t));
    assert(q_weight != NULL);
    for (int oc = 0; oc < 8; oc++)
        for (int ic = 0; ic < 3; ic++)
            q_weight[oc * 3 + ic] = (int8_t)((oc == ic) ? 64 : 0);

    float scale_w = 0.01f;
    float bias[8];
    for (int i = 0; i < 8; i++) bias[i] = 0.0f;

    ret = conv2d_load_weights_int8(&layer, q_weight, numel, scale_w, bias);
    free(q_weight);
    assert(ret == 0);

    tensor_t* output = tensor_create(1, 8, 4, 4);
    assert(output != NULL);

    ret = conv2d_quant_forward(&layer, input, output);
    assert(ret == 0);

    assert(output->n == 1 && output->c == 8 && output->h == 4 && output->w == 4);
    printf("  Output shape: (%d, %d, %d, %d)\n", output->n, output->c, output->h, output->w);
    printf("  First output value: %f\n", output->data[0]);

    conv2d_free(&layer);
    tensor_free(input);
    tensor_free(output);

    printf("  PASSED\n");
    return 0;
}

int main() {
    printf("=== Conv1x1 Int8 Tests ===\n\n");
    test_conv1x1_int8_basic();
    printf("\n=== All tests passed ===\n");
    return 0;
}
