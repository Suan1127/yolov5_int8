#include "yolov5n_build.h"
#include "../core/common.h"
#include "../core/weights_loader_int8.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Helper: make divisible by 8
static int32_t make_divisible(int32_t x, int32_t divisor) {
    return (int32_t)(ceil((double)x / divisor) * divisor);
}

// Helper: calculate actual channels
static int32_t get_actual_channels(int32_t base_channels, float width_multiple) {
    return make_divisible((int32_t)(base_channels * width_multiple), 8);
}

// Helper: calculate actual repeats
static int32_t get_actual_repeats(int32_t base_repeats, float depth_multiple) {
    int32_t result = (int32_t)round(base_repeats * depth_multiple);
    return result > 0 ? result : 1;
}

// Load Conv+BN from int8 package (weight + scale + bias.bin). BN is set to identity.
int load_conv_bn_layer(conv2d_layer_t* conv, batchnorm2d_layer_t* bn,
                       const char* prefix,
                       int32_t in_channels, int32_t out_channels,
                       int32_t kernel_size, int32_t stride, int32_t padding,
                       weights_loader_int8_t* int8_loader) {
    if (!conv || !bn || !prefix || !int8_loader) return -1;
    char name[256];
    conv2d_params_t conv_params = {
        .out_channels = out_channels,
        .kernel_size = kernel_size,
        .stride = stride,
        .padding = padding,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(conv, in_channels, &conv_params) != 0) return -1;
    snprintf(name, sizeof(name), "%s.conv.weight", prefix);

    const int8_t* qptr = NULL;
    float scale_w = 0.f;
    size_t numel = 0;
    const float* bias_ptr = NULL;
    size_t bias_count = 0;
    if (weights_loader_int8_get(int8_loader, name, &qptr, &scale_w, &numel) != 0 ||
        weights_loader_int8_get_bias(int8_loader, name, &bias_ptr, &bias_count) != 0 ||
        !bias_ptr || bias_count != (size_t)out_channels) {
        fprintf(stderr, "Error: int8 load failed for %s (need weight + bias in int8 package)\n", name);
        conv2d_free(conv);
        return -1;
    }
    if (conv2d_load_weights_int8(conv, qptr, numel, scale_w, bias_ptr) != 0) {
        conv2d_free(conv);
        return -1;
    }
    if (batchnorm2d_init(bn, out_channels, &(batchnorm2d_params_t){.num_features = out_channels, .eps = 1e-5f, .momentum = 0.1f}) != 0) {
        conv2d_free(conv);
        return -1;
    }
    for (int i = 0; i < out_channels; i++) {
        bn->weight[i] = 1.0f;
        bn->bias[i] = 0.0f;
        bn->running_mean[i] = 0.0f;
        bn->running_var[i] = 1.0f;
    }
    return 0;
}

yolov5n_model_t* yolov5n_build(const char* weights_path, const char* model_meta_path) {
    if (!weights_path) return NULL;
    
    yolov5n_model_t* model = (yolov5n_model_t*)calloc(1, sizeof(yolov5n_model_t));
    if (!model) return NULL;
    
    // Default parameters (YOLOv5n)
    model->depth_multiple = 0.33f;
    model->width_multiple = 0.25f;
    model->num_classes = 80;
    model->input_size = 640;
    
    // TODO: Load model_meta.json if provided to get actual parameters
    // For now, use defaults
    
    /* weights_dir: 경로에서 디렉터리만 (weights_int8.bin만 있어도 동일 디렉터리 사용) */
    char weights_dir[512];
    strncpy(weights_dir, weights_path, sizeof(weights_dir) - 1);
    weights_dir[sizeof(weights_dir) - 1] = '\0';
    char* last_slash = strrchr(weights_dir, '/');
    if (!last_slash) last_slash = strrchr(weights_dir, '\\');
    if (last_slash) *last_slash = '\0';
    else weights_dir[0] = '\0';
    const char* dir = weights_dir[0] ? weights_dir : ".";

    printf("Loading weights from: %s\n", weights_path);
    model->weights_int8 = weights_loader_int8_create(dir);
    if (model->weights_int8)
        printf("INT8 weights loaded (weights_int8.bin + scales_int8.json + bias.bin)\n");

    if (!model->weights_int8) {
        fprintf(stderr, "Error: Need weights_int8.bin + scales_int8.json (and optional bias.bin) in %s\n", dir);
        free(model);
        return NULL;
    }

    for (int i = 0; i < 12; i++)
        model->saved_features[i] = NULL;
    if (yolov5n_load_weights(model) != 0) {
        yolov5n_free(model);
        return NULL;
    }

    return model;
}

void yolov5n_free(yolov5n_model_t* model) {
    if (!model) return;
    
    // Free backbone conv layers
    for (int i = 0; i < 5; i++) {
        conv2d_free(&model->backbone_convs[i].conv);
        batchnorm2d_free(&model->backbone_convs[i].bn);
    }
    
    // Free backbone C3 blocks
    for (int i = 0; i < 4; i++) {
        c3_free(&model->backbone_c3s[i].block);
    }
    
    // Free SPPF
    sppf_free(&model->sppf);
    
    // Free head conv layers
    for (int i = 0; i < 4; i++) {
        conv2d_free(&model->head_convs[i].conv);
        batchnorm2d_free(&model->head_convs[i].bn);
    }
    
    // Free head C3 blocks
    for (int i = 0; i < 4; i++) {
        c3_free(&model->head_c3s[i].block);
    }
    
    // Free detect head conv layers
    for (int i = 0; i < 3; i++) {
        conv2d_free(&model->detect_convs[i].conv);
    }
    
    // Free saved features
    for (int i = 0; i < 12; i++) {
        if (model->saved_features[i]) {
            tensor_free(model->saved_features[i]);
        }
    }
    
    if (model->weights_int8)
        weights_loader_int8_free((weights_loader_int8_t*)model->weights_int8);
    free(model);
}

int yolov5n_load_weights(yolov5n_model_t* model) {
    if (!model) return -1;
    if (!model->weights_int8) return -1;
    
    float depth_multiple = model->depth_multiple;
    float width_multiple = model->width_multiple;
    char name[256];
    
    // ========== Backbone ==========
    
    // Layer 0: Conv(3→32, 6×6, s=2, p=2)
    model->backbone_convs[0].in_channels = 3;
    model->backbone_convs[0].out_channels = get_actual_channels(64, width_multiple);  // 32
    model->backbone_convs[0].kernel_size = 6;
    model->backbone_convs[0].stride = 2;
    model->backbone_convs[0].padding = 2;
    snprintf(name, sizeof(name), "model.0");
    if (load_conv_bn_layer(&model->backbone_convs[0].conv, &model->backbone_convs[0].bn, name,
                          model->backbone_convs[0].in_channels, model->backbone_convs[0].out_channels,
                          model->backbone_convs[0].kernel_size, model->backbone_convs[0].stride,
                          model->backbone_convs[0].padding, (weights_loader_int8_t*)model->weights_int8) != 0)
        return -1;
    model->backbone_convs[0].is_fused = 1;
    
    // Layer 1: Conv(16→32, 3×3, s=2) for YOLOv5n
    model->backbone_convs[1].in_channels = get_actual_channels(64, width_multiple);  // 16 for YOLOv5n
    model->backbone_convs[1].out_channels = get_actual_channels(128, width_multiple);  // 32 for YOLOv5n
    model->backbone_convs[1].kernel_size = 3;
    model->backbone_convs[1].stride = 2;
    model->backbone_convs[1].padding = 1;
    snprintf(name, sizeof(name), "model.1");
    if (load_conv_bn_layer(&model->backbone_convs[1].conv, &model->backbone_convs[1].bn, name,
                          model->backbone_convs[1].in_channels, model->backbone_convs[1].out_channels,
                          model->backbone_convs[1].kernel_size, model->backbone_convs[1].stride,
                          model->backbone_convs[1].padding, (weights_loader_int8_t*)model->weights_int8) != 0)
        return -1;
    model->backbone_convs[1].is_fused = 1;
    
    // Layer 2: C3(32→32, n=1) for YOLOv5n
    int32_t c3_0_c1 = get_actual_channels(128, width_multiple);  // 32 for YOLOv5n
    int32_t c3_0_c2 = get_actual_channels(128, width_multiple);  // 32 for YOLOv5n
    model->backbone_c3s[0].c1 = c3_0_c1;
    model->backbone_c3s[0].c2 = c3_0_c2;
    model->backbone_c3s[0].n = get_actual_repeats(3, depth_multiple);  // 1
    model->backbone_c3s[0].shortcut = 1;
    if (c3_init(&model->backbone_c3s[0].block, c3_0_c1, c3_0_c2, 1, 1) != 0) return -1;
    snprintf(name, sizeof(name), "model.2");
    if (c3_load_weights(&model->backbone_c3s[0].block, name, model->weights_int8) != 0) {
        c3_free(&model->backbone_c3s[0].block);
        return -1;
    }
    
    // Layer 3: Conv(32→64, 3×3, s=2) for YOLOv5n
    model->backbone_convs[2].in_channels = get_actual_channels(128, width_multiple);  // 32 for YOLOv5n
    model->backbone_convs[2].out_channels = get_actual_channels(256, width_multiple);  // 64 for YOLOv5n
    model->backbone_convs[2].kernel_size = 3;
    model->backbone_convs[2].stride = 2;
    model->backbone_convs[2].padding = 1;
    snprintf(name, sizeof(name), "model.3");
    if (load_conv_bn_layer(&model->backbone_convs[2].conv, &model->backbone_convs[2].bn, name,
                          model->backbone_convs[2].in_channels, model->backbone_convs[2].out_channels,
                          model->backbone_convs[2].kernel_size, model->backbone_convs[2].stride,
                          model->backbone_convs[2].padding, (weights_loader_int8_t*)model->weights_int8) != 0)
        return -1;
    model->backbone_convs[2].is_fused = 1;
    
    // Layer 4: C3(64→64, n=2) for YOLOv5n
    int32_t c3_1_c1 = get_actual_channels(256, width_multiple);  // 64 for YOLOv5n
    int32_t c3_1_c2 = get_actual_channels(256, width_multiple);  // 64 for YOLOv5n
    model->backbone_c3s[1].c1 = c3_1_c1;
    model->backbone_c3s[1].c2 = c3_1_c2;
    model->backbone_c3s[1].n = get_actual_repeats(6, depth_multiple);  // 2
    model->backbone_c3s[1].shortcut = 1;
    if (c3_init(&model->backbone_c3s[1].block, c3_1_c1, c3_1_c2, 2, 1) != 0) return -1;
    snprintf(name, sizeof(name), "model.4");
    if (c3_load_weights(&model->backbone_c3s[1].block, name, model->weights_int8) != 0) {
        c3_free(&model->backbone_c3s[1].block);
        return -1;
    }
    
    // Layer 5: Conv(64→128, 3×3, s=2) for YOLOv5n
    model->backbone_convs[3].in_channels = get_actual_channels(256, width_multiple);  // 64 for YOLOv5n
    model->backbone_convs[3].out_channels = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    model->backbone_convs[3].kernel_size = 3;
    model->backbone_convs[3].stride = 2;
    model->backbone_convs[3].padding = 1;
    snprintf(name, sizeof(name), "model.5");
    if (load_conv_bn_layer(&model->backbone_convs[3].conv, &model->backbone_convs[3].bn, name,
                          model->backbone_convs[3].in_channels, model->backbone_convs[3].out_channels,
                          model->backbone_convs[3].kernel_size, model->backbone_convs[3].stride,
                          model->backbone_convs[3].padding, (weights_loader_int8_t*)model->weights_int8) != 0)
        return -1;
    model->backbone_convs[3].is_fused = 1;
    
    // Layer 6: C3(128→128, n=3) for YOLOv5n
    int32_t c3_2_c1 = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    int32_t c3_2_c2 = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    model->backbone_c3s[2].c1 = c3_2_c1;
    model->backbone_c3s[2].c2 = c3_2_c2;
    model->backbone_c3s[2].n = get_actual_repeats(9, depth_multiple);  // 3
    model->backbone_c3s[2].shortcut = 1;
    if (c3_init(&model->backbone_c3s[2].block, c3_2_c1, c3_2_c2, 3, 1) != 0) return -1;
    snprintf(name, sizeof(name), "model.6");
    if (c3_load_weights(&model->backbone_c3s[2].block, name, model->weights_int8) != 0) {
        c3_free(&model->backbone_c3s[2].block);
        return -1;
    }
    
    // Layer 7: Conv(128→256, 3×3, s=2) for YOLOv5n
    model->backbone_convs[4].in_channels = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    model->backbone_convs[4].out_channels = get_actual_channels(1024, width_multiple);  // 256 for YOLOv5n
    model->backbone_convs[4].kernel_size = 3;
    model->backbone_convs[4].stride = 2;
    model->backbone_convs[4].padding = 1;
    snprintf(name, sizeof(name), "model.7");
    if (load_conv_bn_layer(&model->backbone_convs[4].conv, &model->backbone_convs[4].bn, name,
                          model->backbone_convs[4].in_channels, model->backbone_convs[4].out_channels,
                          model->backbone_convs[4].kernel_size, model->backbone_convs[4].stride,
                          model->backbone_convs[4].padding, (weights_loader_int8_t*)model->weights_int8) != 0)
        return -1;
    model->backbone_convs[4].is_fused = 1;
    
    // Layer 8: C3(256→256, n=1) for YOLOv5n
    int32_t c3_3_c1 = get_actual_channels(1024, width_multiple);  // 256 for YOLOv5n
    int32_t c3_3_c2 = get_actual_channels(1024, width_multiple);  // 256 for YOLOv5n
    model->backbone_c3s[3].c1 = c3_3_c1;
    model->backbone_c3s[3].c2 = c3_3_c2;
    model->backbone_c3s[3].n = get_actual_repeats(3, depth_multiple);  // 1
    model->backbone_c3s[3].shortcut = 1;
    if (c3_init(&model->backbone_c3s[3].block, c3_3_c1, c3_3_c2, 1, 1) != 0) return -1;
    snprintf(name, sizeof(name), "model.8");
    if (c3_load_weights(&model->backbone_c3s[3].block, name, model->weights_int8) != 0) {
        c3_free(&model->backbone_c3s[3].block);
        return -1;
    }
    
    // Layer 9: SPPF(256→256, k=5) for YOLOv5n
    int32_t sppf_c = get_actual_channels(1024, width_multiple);  // 256 for YOLOv5n
    if (sppf_init(&model->sppf, sppf_c, sppf_c, 5) != 0) return -1;
    snprintf(name, sizeof(name), "model.9");
    if (sppf_load_weights(&model->sppf, name, model->weights_int8) != 0) {
        sppf_free(&model->sppf);
        return -1;
    }
    
    // ========== Head ==========
    
    // Layer 10: Conv(256→128, 1×1) for YOLOv5n
    int32_t head_conv0_in = get_actual_channels(1024, width_multiple);  // 256 for YOLOv5n
    int32_t head_conv0_out = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    model->head_convs[0].in_channels = head_conv0_in;
    model->head_convs[0].out_channels = head_conv0_out;
    snprintf(name, sizeof(name), "model.10");
    if (load_conv_bn_layer(&model->head_convs[0].conv, &model->head_convs[0].bn, name,
                          model->head_convs[0].in_channels, model->head_convs[0].out_channels,
                          1, 1, 0, (weights_loader_int8_t*)model->weights_int8) != 0)
        return -1;
    model->head_convs[0].is_fused = 1;
    
    // Layer 13: C3(256→128, n=1, shortcut=False) for YOLOv5n
    // Input is from concat of layer 11 (head_conv0_out=128) and layer 6 (backbone_c3s[2].c2=128) = 256 channels
    int32_t head_c3_0_c1 = head_conv0_out + model->backbone_c3s[2].c2;  // 128 + 128 = 256 for YOLOv5n
    int32_t head_c3_0_c2 = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    model->head_c3s[0].c1 = head_c3_0_c1;
    model->head_c3s[0].c2 = head_c3_0_c2;
    model->head_c3s[0].n = get_actual_repeats(3, depth_multiple);  // 1
    model->head_c3s[0].shortcut = 0;
    if (c3_init(&model->head_c3s[0].block, head_c3_0_c1, head_c3_0_c2, 1, 0) != 0) return -1;
    snprintf(name, sizeof(name), "model.13");
    if (c3_load_weights(&model->head_c3s[0].block, name, model->weights_int8) != 0) {
        c3_free(&model->head_c3s[0].block);
        return -1;
    }
    
    // Layer 14: Conv(128→64, 1×1) for YOLOv5n
    int32_t head_conv1_in = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    int32_t head_conv1_out = get_actual_channels(256, width_multiple);  // 64 for YOLOv5n
    model->head_convs[1].in_channels = head_conv1_in;
    model->head_convs[1].out_channels = head_conv1_out;
    snprintf(name, sizeof(name), "model.14");
    if (load_conv_bn_layer(&model->head_convs[1].conv, &model->head_convs[1].bn, name,
                          model->head_convs[1].in_channels, model->head_convs[1].out_channels,
                          1, 1, 0, (weights_loader_int8_t*)model->weights_int8) != 0)
        return -1;
    model->head_convs[1].is_fused = 1;
    
    // Layer 17: C3(128→64, n=1, shortcut=False) for YOLOv5n
    // Input is from concat of layer 15 (head_conv1_out=64) and layer 4 (backbone_c3s[1].c2=64) = 128 channels
    int32_t head_c3_1_c1 = head_conv1_out + model->backbone_c3s[1].c2;  // 64 + 64 = 128 for YOLOv5n
    int32_t head_c3_1_c2 = get_actual_channels(256, width_multiple);  // 64 for YOLOv5n
    model->head_c3s[1].c1 = head_c3_1_c1;
    model->head_c3s[1].c2 = head_c3_1_c2;
    model->head_c3s[1].n = get_actual_repeats(3, depth_multiple);  // 1
    model->head_c3s[1].shortcut = 0;
    if (c3_init(&model->head_c3s[1].block, head_c3_1_c1, head_c3_1_c2, 1, 0) != 0) return -1;
    snprintf(name, sizeof(name), "model.17");
    if (c3_load_weights(&model->head_c3s[1].block, name, model->weights_int8) != 0) {
        c3_free(&model->head_c3s[1].block);
        return -1;
    }
    
    // Layer 18: Conv(64→64, 3×3, s=2) for YOLOv5n
    int32_t head_conv2_in = get_actual_channels(256, width_multiple);  // 64 for YOLOv5n
    int32_t head_conv2_out = get_actual_channels(256, width_multiple);  // 64 for YOLOv5n
    model->head_convs[2].in_channels = head_conv2_in;
    model->head_convs[2].out_channels = head_conv2_out;
    snprintf(name, sizeof(name), "model.18");
    if (load_conv_bn_layer(&model->head_convs[2].conv, &model->head_convs[2].bn, name,
                          model->head_convs[2].in_channels, model->head_convs[2].out_channels,
                          3, 2, 1, (weights_loader_int8_t*)model->weights_int8) != 0)
        return -1;
    model->head_convs[2].is_fused = 1;
    
    // Layer 20: C3(128→128, n=1, shortcut=False) for YOLOv5n
    // Input is from concat of layer 18 (head_conv2_out=64) and layer 14 (head_conv1_out=64) = 128 channels
    int32_t head_c3_2_c1 = head_conv2_out + head_conv1_out;  // 64 + 64 = 128 for YOLOv5n
    int32_t head_c3_2_c2 = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    model->head_c3s[2].c1 = head_c3_2_c1;
    model->head_c3s[2].c2 = head_c3_2_c2;
    model->head_c3s[2].n = get_actual_repeats(3, depth_multiple);  // 1
    model->head_c3s[2].shortcut = 0;
    if (c3_init(&model->head_c3s[2].block, head_c3_2_c1, head_c3_2_c2, 1, 0) != 0) return -1;
    snprintf(name, sizeof(name), "model.20");
    if (c3_load_weights(&model->head_c3s[2].block, name, model->weights_int8) != 0) {
        c3_free(&model->head_c3s[2].block);
        return -1;
    }
    
    // Layer 21: Conv(128→128, 3×3, s=2) for YOLOv5n
    int32_t head_conv3_in = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    int32_t head_conv3_out = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    model->head_convs[3].in_channels = head_conv3_in;
    model->head_convs[3].out_channels = head_conv3_out;
    snprintf(name, sizeof(name), "model.21");
    if (load_conv_bn_layer(&model->head_convs[3].conv, &model->head_convs[3].bn, name,
                          model->head_convs[3].in_channels, model->head_convs[3].out_channels,
                          3, 2, 1, (weights_loader_int8_t*)model->weights_int8) != 0)
        return -1;
    model->head_convs[3].is_fused = 1;
    
    // Layer 23: C3(256→256, n=1, shortcut=False) for YOLOv5n
    // Input is from concat of layer 21 (head_conv3_out=128) and layer 10 (head_conv0_out=128) = 256 channels
    int32_t head_c3_3_c1 = head_conv3_out + head_conv0_out;  // 128 + 128 = 256 for YOLOv5n
    int32_t head_c3_3_c2 = get_actual_channels(1024, width_multiple);  // 256 for YOLOv5n
    model->head_c3s[3].c1 = head_c3_3_c1;
    model->head_c3s[3].c2 = head_c3_3_c2;
    model->head_c3s[3].n = get_actual_repeats(3, depth_multiple);  // 1
    model->head_c3s[3].shortcut = 0;
    if (c3_init(&model->head_c3s[3].block, head_c3_3_c1, head_c3_3_c2, 1, 0) != 0) return -1;
    snprintf(name, sizeof(name), "model.23");
    if (c3_load_weights(&model->head_c3s[3].block, name, model->weights_int8) != 0) {
        c3_free(&model->head_c3s[3].block);
        return -1;
    }
    
    // Initialize Detect head conv layers (Layer 24)
    // P3: 64 -> 255 (for YOLOv5n)
    // P4: 128 -> 255 (for YOLOv5n)
    // P5: 256 -> 255 (for YOLOv5n)
    int32_t detect_in_channels[3] = {
        get_actual_channels(256, width_multiple),  // P3: 64 for YOLOv5n
        get_actual_channels(512, width_multiple),  // P4: 128 for YOLOv5n
        get_actual_channels(1024, width_multiple)   // P5: 256 for YOLOv5n
    };
    int32_t detect_out_channels = 3 * (model->num_classes + 5);  // 255
    
    for (int i = 0; i < 3; i++) {
        model->detect_convs[i].in_channels = detect_in_channels[i];
        model->detect_convs[i].out_channels = detect_out_channels;
        
        conv2d_params_t detect_params = {
            .out_channels = detect_out_channels,
            .kernel_size = 1,
            .stride = 1,
            .padding = 0,
            .groups = 1,
            .dilation = 1
        };
        
        if (conv2d_init(&model->detect_convs[i].conv, detect_in_channels[i], &detect_params) != 0) {
            // Free already initialized detect convs
            for (int j = 0; j < i; j++) {
                conv2d_free(&model->detect_convs[j].conv);
            }
            return -1;
        }
        
        // Load weights: model.24.m.0, model.24.m.1, model.24.m.2 (int8 + bias.bin)
        snprintf(name, sizeof(name), "model.24.m.%d.weight", i);
        weights_loader_int8_t* int8_loader = (weights_loader_int8_t*)model->weights_int8;
        const int8_t* qptr = NULL;
        float scale_w = 0.f;
        size_t numel = 0;
        const float* bias_ptr = NULL;
        size_t bias_count = 0;
        if (weights_loader_int8_get(int8_loader, name, &qptr, &scale_w, &numel) != 0 ||
            weights_loader_int8_get_bias(int8_loader, name, &bias_ptr, &bias_count) != 0 ||
            !bias_ptr || bias_count != (size_t)detect_out_channels) {
            fprintf(stderr, "Error: Detect head %s requires weight + bias in int8 package\n", name);
            for (int j = 0; j <= i; j++) conv2d_free(&model->detect_convs[j].conv);
            return -1;
        }
        if (conv2d_load_weights_int8(&model->detect_convs[i].conv, qptr, numel, scale_w, bias_ptr) != 0) {
            for (int j = 0; j <= i; j++) conv2d_free(&model->detect_convs[j].conv);
            return -1;
        }
    }
    
    printf("YOLOv5n model loaded successfully\n");
    return 0;
}
