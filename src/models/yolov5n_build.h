#ifndef YOLOV5N_BUILD_H
#define YOLOV5N_BUILD_H

#include "../core/tensor.h"
#include "../ops/conv2d.h"
#include "../ops/batchnorm2d.h"
#include "../blocks/c3.h"
#include "../blocks/sppf.h"

/**
 * YOLOv5n model structure
 * 
 * Backbone (0-9):
 *   0, 1, 3, 5, 7: Conv + BN
 *   2, 4, 6, 8: C3 blocks
 *   9: SPPF
 * 
 * Head (10-23):
 *   10, 14, 18, 21: Conv + BN
 *   11, 15: Upsample (no weights)
 *   12, 16, 19, 22: Concat (no weights)
 *   13, 17, 20, 23: C3 blocks
 */
typedef struct {
    // Backbone Conv layers (0, 1, 3, 5, 7)
    struct {
        conv2d_layer_t conv;
        batchnorm2d_layer_t bn;
        int32_t is_fused;  // 1 if Conv+BN is fused, 0 otherwise
        int32_t in_channels;
        int32_t out_channels;
        int32_t kernel_size;
        int32_t stride;
        int32_t padding;
    } backbone_convs[5];
    
    // Backbone C3 blocks (2, 4, 6, 8)
    struct {
        c3_block_t block;
        int32_t c1, c2, n;
        int shortcut;
    } backbone_c3s[4];
    
    // SPPF (9)
    sppf_block_t sppf;
    
    // Head Conv layers (10, 14, 18, 21)
    struct {
        conv2d_layer_t conv;
        batchnorm2d_layer_t bn;
        int32_t is_fused;  // 1 if Conv+BN is fused, 0 otherwise
        int32_t in_channels;
        int32_t out_channels;
    } head_convs[4];
    
    // Head C3 blocks (13, 17, 20, 23)
    struct {
        c3_block_t block;
        int32_t c1, c2, n;
        int shortcut;
    } head_c3s[4];
    
    // Detect head Conv layers (24: P3, P4, P5)
    struct {
        conv2d_layer_t conv;
        int32_t in_channels;
        int32_t out_channels;
    } detect_convs[3];
    
    // Saved feature maps (for FPN connections and validation)
    // Indices: 0=layer0, 1=layer1, 2=layer2, 3=layer3, 4=layer4, 5=layer5, 6=layer6, 7=layer7, 8=layer9, 9=layer17, 10=layer20, 11=layer23
    tensor_t* saved_features[12];
    
    // INT8 가중치 (weights_int8.bin + scales_int8.json + bias.bin)
    void* weights_int8;

    // Model parameters
    float depth_multiple;
    float width_multiple;
    int32_t num_classes;
    int32_t input_size;
} yolov5n_model_t;

/**
 * Build YOLOv5n model
 * @param weights_path Path to weights.bin
 * @param model_meta_path Path to model_meta.json (optional, can be NULL)
 * @return Model instance or NULL on failure
 */
yolov5n_model_t* yolov5n_build(const char* weights_path, const char* model_meta_path);

/**
 * Free YOLOv5n model
 */
void yolov5n_free(yolov5n_model_t* model);

/**
 * Load weights into model
 */
int yolov5n_load_weights(yolov5n_model_t* model);

/**
 * Helper: Load Conv+BN layer from int8 package (weight + scale + bias.bin)
 */
int load_conv_bn_layer(conv2d_layer_t* conv, batchnorm2d_layer_t* bn,
                       const char* prefix,
                       int32_t in_channels, int32_t out_channels,
                       int32_t kernel_size, int32_t stride, int32_t padding,
                       void* int8_loader);  /* weights_loader_int8_t* */

#endif // YOLOV5N_BUILD_H
