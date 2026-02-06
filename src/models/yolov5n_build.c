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

// Helper: Load Conv+BN layer weights. int8_loader 있으면 int8 가중치 우선 사용 (float weight/minmax 없음).
// Returns: -1 on error, 0 if not fused, 1 if fused
int load_conv_bn_layer(conv2d_layer_t* conv, batchnorm2d_layer_t* bn,
                       weights_loader_t* loader, const char* prefix,
                       int32_t in_channels, int32_t out_channels,
                       int32_t kernel_size, int32_t stride, int32_t padding,
                       weights_loader_int8_t* int8_loader) {
    if (!conv || !bn || !loader || !prefix) return -1;
    char name[256];
    int32_t shape[4];
    int num_dims;
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
    float* fused_bias = NULL;
    snprintf(name, sizeof(name), "%s.conv.bias", prefix);
    fused_bias = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.conv.weight", prefix);
    if (int8_loader) {
        const int8_t* qptr = NULL;
        float scale_w = 0.f;
        size_t numel = 0;
        if (weights_loader_int8_get(int8_loader, name, &qptr, &scale_w, &numel) == 0) {
            if (conv2d_load_weights_int8(conv, qptr, numel, scale_w, fused_bias) == 0) {
                goto load_bn;
            }
        }
    }
    float* w = weights_loader_get(loader, name, shape, &num_dims);
    if (!w) {
        fprintf(stderr, "Error: Failed to load weight for %s\n", name);
        return -1;
    }
    conv2d_load_weights(conv, w, fused_bias);
load_bn:
    
    // Initialize BN layer
    batchnorm2d_params_t bn_params = {
        .num_features = out_channels,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(bn, out_channels, &bn_params) != 0) {
        conv2d_free(conv);
        return -1;
    }
    
    // Load BN weights (only if not fused - check if BN weights exist)
    snprintf(name, sizeof(name), "%s.bn.weight", prefix);
    float* bn_w = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.bn.bias", prefix);
    float* bn_b = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.bn.running_mean", prefix);
    float* bn_mean = weights_loader_get(loader, name, shape, &num_dims);
    snprintf(name, sizeof(name), "%s.bn.running_var", prefix);
    float* bn_var = weights_loader_get(loader, name, shape, &num_dims);
    
    // If fused_bias was loaded, Conv+BN is fused - set BN to identity (no-op)
    // Otherwise, load BN weights normally
    // Note: We need to return fused status to caller, but since this is a helper function,
    // we'll store it in a way that can be checked later. For now, we'll use a flag in the struct.
    // Actually, we can't modify the struct here. Let's return the fused status.
    // But the function signature doesn't allow that. Let's use a different approach:
    // Store fused status in a way that can be checked. Actually, we can check if bias exists.
    
    if (fused_bias) {
        // Model is fused - BN is already incorporated into Conv weight/bias
        // Set BN to identity: weight=1, bias=0, mean=0, var=1
        for (int i = 0; i < out_channels; i++) {
            bn->weight[i] = 1.0f;
            bn->bias[i] = 0.0f;
            bn->running_mean[i] = 0.0f;
            bn->running_var[i] = 1.0f;
        }
        return 1;  // Return 1 to indicate fused (we'll need to modify return type)
    } else if (bn_w && bn_b && bn_mean && bn_var) {
        // Not fused - load BN weights normally
        batchnorm2d_load_weights(bn, bn_w, bn_b, bn_mean, bn_var);
        return 0;  // Return 0 to indicate not fused
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
    
    // Load weights
    printf("Loading weights from: %s\n", weights_path);
    model->weights = weights_loader_create(weights_path);
    if (!model->weights) {
        fprintf(stderr, "Error: Failed to load weights from %s\n", weights_path);
        fprintf(stderr, "Check if file exists and is readable\n");
        free(model);
        return NULL;
    }
    printf("Weights loaded successfully (size: %zu bytes)\n", model->weights->size);
    char weights_dir[512];
    strncpy(weights_dir, weights_path, sizeof(weights_dir) - 1);
    weights_dir[sizeof(weights_dir) - 1] = '\0';
    char* last_slash = strrchr(weights_dir, '/');
    if (!last_slash) last_slash = strrchr(weights_dir, '\\');
    if (last_slash) *last_slash = '\0';
    else weights_dir[0] = '\0';
    model->weights_int8 = weights_loader_int8_create(weights_dir[0] ? weights_dir : ".");
    if (model->weights_int8)
        printf("INT8 weights loaded (weights_int8.bin + scales_int8.json)\n");
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
    
    if (model->weights)
        weights_loader_free(model->weights);
    if (model->weights_int8)
        weights_loader_int8_free((weights_loader_int8_t*)model->weights_int8);
    free(model);
}

int yolov5n_load_weights(yolov5n_model_t* model) {
    if (!model || !model->weights) return -1;
    
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
    int fused_status = load_conv_bn_layer(&model->backbone_convs[0].conv, &model->backbone_convs[0].bn,
                          model->weights, name,
                          model->backbone_convs[0].in_channels,
                          model->backbone_convs[0].out_channels,
                          model->backbone_convs[0].kernel_size,
                          model->backbone_convs[0].stride,
                          model->backbone_convs[0].padding,
                          model->weights_int8);
    if (fused_status < 0) {
        return -1;
    }
    model->backbone_convs[0].is_fused = (fused_status > 0) ? 1 : 0;
    
    // Layer 1: Conv(16→32, 3×3, s=2) for YOLOv5n
    model->backbone_convs[1].in_channels = get_actual_channels(64, width_multiple);  // 16 for YOLOv5n
    model->backbone_convs[1].out_channels = get_actual_channels(128, width_multiple);  // 32 for YOLOv5n
    model->backbone_convs[1].kernel_size = 3;
    model->backbone_convs[1].stride = 2;
    model->backbone_convs[1].padding = 1;
    snprintf(name, sizeof(name), "model.1");
    fused_status = load_conv_bn_layer(&model->backbone_convs[1].conv, &model->backbone_convs[1].bn,
                          model->weights, name,
                          model->backbone_convs[1].in_channels,
                          model->backbone_convs[1].out_channels,
                          model->backbone_convs[1].kernel_size,
                          model->backbone_convs[1].stride,
                          model->backbone_convs[1].padding,
                          model->weights_int8);
    if (fused_status < 0) {
        return -1;
    }
    model->backbone_convs[1].is_fused = (fused_status > 0) ? 1 : 0;
    
    // Layer 2: C3(32→32, n=1) for YOLOv5n
    int32_t c3_0_c1 = get_actual_channels(128, width_multiple);  // 32 for YOLOv5n
    int32_t c3_0_c2 = get_actual_channels(128, width_multiple);  // 32 for YOLOv5n
    model->backbone_c3s[0].c1 = c3_0_c1;
    model->backbone_c3s[0].c2 = c3_0_c2;
    model->backbone_c3s[0].n = get_actual_repeats(3, depth_multiple);  // 1
    model->backbone_c3s[0].shortcut = 1;
    if (c3_init(&model->backbone_c3s[0].block, c3_0_c1, c3_0_c2, 1, 1) != 0) return -1;
    snprintf(name, sizeof(name), "model.2");
    if (c3_load_weights(&model->backbone_c3s[0].block, model->weights, name, model->weights_int8) != 0) {
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
    fused_status = load_conv_bn_layer(&model->backbone_convs[2].conv, &model->backbone_convs[2].bn,
                          model->weights, name,
                          model->backbone_convs[2].in_channels,
                          model->backbone_convs[2].out_channels,
                          model->backbone_convs[2].kernel_size,
                          model->backbone_convs[2].stride,
                          model->backbone_convs[2].padding,
                          model->weights_int8);
    if (fused_status < 0) {
        return -1;
    }
    model->backbone_convs[2].is_fused = (fused_status > 0) ? 1 : 0;
    
    // Layer 4: C3(64→64, n=2) for YOLOv5n
    int32_t c3_1_c1 = get_actual_channels(256, width_multiple);  // 64 for YOLOv5n
    int32_t c3_1_c2 = get_actual_channels(256, width_multiple);  // 64 for YOLOv5n
    model->backbone_c3s[1].c1 = c3_1_c1;
    model->backbone_c3s[1].c2 = c3_1_c2;
    model->backbone_c3s[1].n = get_actual_repeats(6, depth_multiple);  // 2
    model->backbone_c3s[1].shortcut = 1;
    if (c3_init(&model->backbone_c3s[1].block, c3_1_c1, c3_1_c2, 2, 1) != 0) return -1;
    snprintf(name, sizeof(name), "model.4");
    if (c3_load_weights(&model->backbone_c3s[1].block, model->weights, name, model->weights_int8) != 0) {
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
    fused_status = load_conv_bn_layer(&model->backbone_convs[3].conv, &model->backbone_convs[3].bn,
                          model->weights, name,
                          model->backbone_convs[3].in_channels,
                          model->backbone_convs[3].out_channels,
                          model->backbone_convs[3].kernel_size,
                          model->backbone_convs[3].stride,
                          model->backbone_convs[3].padding,
                          model->weights_int8);
    if (fused_status < 0) {
        return -1;
    }
    model->backbone_convs[3].is_fused = (fused_status > 0) ? 1 : 0;
    
    // Layer 6: C3(128→128, n=3) for YOLOv5n
    int32_t c3_2_c1 = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    int32_t c3_2_c2 = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    model->backbone_c3s[2].c1 = c3_2_c1;
    model->backbone_c3s[2].c2 = c3_2_c2;
    model->backbone_c3s[2].n = get_actual_repeats(9, depth_multiple);  // 3
    model->backbone_c3s[2].shortcut = 1;
    if (c3_init(&model->backbone_c3s[2].block, c3_2_c1, c3_2_c2, 3, 1) != 0) return -1;
    snprintf(name, sizeof(name), "model.6");
    if (c3_load_weights(&model->backbone_c3s[2].block, model->weights, name, model->weights_int8) != 0) {
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
    fused_status = load_conv_bn_layer(&model->backbone_convs[4].conv, &model->backbone_convs[4].bn,
                          model->weights, name,
                          model->backbone_convs[4].in_channels,
                          model->backbone_convs[4].out_channels,
                          model->backbone_convs[4].kernel_size,
                          model->backbone_convs[4].stride,
                          model->backbone_convs[4].padding,
                          model->weights_int8);
    if (fused_status < 0) {
        return -1;
    }
    model->backbone_convs[4].is_fused = (fused_status > 0) ? 1 : 0;
    
    // Layer 8: C3(256→256, n=1) for YOLOv5n
    int32_t c3_3_c1 = get_actual_channels(1024, width_multiple);  // 256 for YOLOv5n
    int32_t c3_3_c2 = get_actual_channels(1024, width_multiple);  // 256 for YOLOv5n
    model->backbone_c3s[3].c1 = c3_3_c1;
    model->backbone_c3s[3].c2 = c3_3_c2;
    model->backbone_c3s[3].n = get_actual_repeats(3, depth_multiple);  // 1
    model->backbone_c3s[3].shortcut = 1;
    if (c3_init(&model->backbone_c3s[3].block, c3_3_c1, c3_3_c2, 1, 1) != 0) return -1;
    snprintf(name, sizeof(name), "model.8");
    if (c3_load_weights(&model->backbone_c3s[3].block, model->weights, name, model->weights_int8) != 0) {
        c3_free(&model->backbone_c3s[3].block);
        return -1;
    }
    
    // Layer 9: SPPF(256→256, k=5) for YOLOv5n
    int32_t sppf_c = get_actual_channels(1024, width_multiple);  // 256 for YOLOv5n
    if (sppf_init(&model->sppf, sppf_c, sppf_c, 5) != 0) return -1;
    snprintf(name, sizeof(name), "model.9");
    if (sppf_load_weights(&model->sppf, model->weights, name, model->weights_int8) != 0) {
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
    fused_status = load_conv_bn_layer(&model->head_convs[0].conv, &model->head_convs[0].bn,
                          model->weights, name,
                          model->head_convs[0].in_channels,
                          model->head_convs[0].out_channels,
                          1, 1, 0, model->weights_int8);
    if (fused_status < 0) {
        return -1;
    }
    model->head_convs[0].is_fused = (fused_status > 0) ? 1 : 0;
    
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
    if (c3_load_weights(&model->head_c3s[0].block, model->weights, name, model->weights_int8) != 0) {
        c3_free(&model->head_c3s[0].block);
        return -1;
    }
    
    // Layer 14: Conv(128→64, 1×1) for YOLOv5n
    int32_t head_conv1_in = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    int32_t head_conv1_out = get_actual_channels(256, width_multiple);  // 64 for YOLOv5n
    model->head_convs[1].in_channels = head_conv1_in;
    model->head_convs[1].out_channels = head_conv1_out;
    snprintf(name, sizeof(name), "model.14");
    fused_status = load_conv_bn_layer(&model->head_convs[1].conv, &model->head_convs[1].bn,
                          model->weights, name,
                          model->head_convs[1].in_channels,
                          model->head_convs[1].out_channels,
                          1, 1, 0, model->weights_int8);
    if (fused_status < 0) {
        return -1;
    }
    model->head_convs[1].is_fused = (fused_status > 0) ? 1 : 0;
    
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
    if (c3_load_weights(&model->head_c3s[1].block, model->weights, name, model->weights_int8) != 0) {
        c3_free(&model->head_c3s[1].block);
        return -1;
    }
    
    // Layer 18: Conv(64→64, 3×3, s=2) for YOLOv5n
    int32_t head_conv2_in = get_actual_channels(256, width_multiple);  // 64 for YOLOv5n
    int32_t head_conv2_out = get_actual_channels(256, width_multiple);  // 64 for YOLOv5n
    model->head_convs[2].in_channels = head_conv2_in;
    model->head_convs[2].out_channels = head_conv2_out;
    snprintf(name, sizeof(name), "model.18");
    fused_status = load_conv_bn_layer(&model->head_convs[2].conv, &model->head_convs[2].bn,
                          model->weights, name,
                          model->head_convs[2].in_channels,
                          model->head_convs[2].out_channels,
                          3, 2, 1, model->weights_int8);
    if (fused_status < 0) {
        return -1;
    }
    model->head_convs[2].is_fused = (fused_status > 0) ? 1 : 0;
    
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
    if (c3_load_weights(&model->head_c3s[2].block, model->weights, name, model->weights_int8) != 0) {
        c3_free(&model->head_c3s[2].block);
        return -1;
    }
    
    // Layer 21: Conv(128→128, 3×3, s=2) for YOLOv5n
    int32_t head_conv3_in = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    int32_t head_conv3_out = get_actual_channels(512, width_multiple);  // 128 for YOLOv5n
    model->head_convs[3].in_channels = head_conv3_in;
    model->head_convs[3].out_channels = head_conv3_out;
    snprintf(name, sizeof(name), "model.21");
    fused_status = load_conv_bn_layer(&model->head_convs[3].conv, &model->head_convs[3].bn,
                          model->weights, name,
                          model->head_convs[3].in_channels,
                          model->head_convs[3].out_channels,
                          3, 2, 1, model->weights_int8);
    if (fused_status < 0) {
        return -1;
    }
    model->head_convs[3].is_fused = (fused_status > 0) ? 1 : 0;
    
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
    if (c3_load_weights(&model->head_c3s[3].block, model->weights, name, model->weights_int8) != 0) {
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
        
        // Load weights: model.24.m.0, model.24.m.1, model.24.m.2
        snprintf(name, sizeof(name), "model.24.m.%d.weight", i);
        int32_t shape[4];
        int num_dims;
        float* w = weights_loader_get(model->weights, name, shape, &num_dims);
        if (!w) {
            fprintf(stderr, "Error: Failed to load detect head weight for %s\n", name);
            // Free already initialized detect convs
            for (int j = 0; j <= i; j++) {
                conv2d_free(&model->detect_convs[j].conv);
            }
            return -1;
        }
        
        // Detect head conv has bias (no BN)
        snprintf(name, sizeof(name), "model.24.m.%d.bias", i);
        float* bias = weights_loader_get(model->weights, name, shape, &num_dims);
        conv2d_load_weights(&model->detect_convs[i].conv, w, bias);
    }
    
    printf("YOLOv5n model loaded successfully\n");
    return 0;
}
