#ifndef BOTTLENECK_H
#define BOTTLENECK_H

#include "../core/tensor.h"
#include "../ops/conv2d.h"
#include "../ops/batchnorm2d.h"
#include "../ops/activation.h"

/**
 * Bottleneck block (used in C3)
 * 
 * Structure:
 *   x -> conv1 -> bn1 -> silu -> conv2 -> bn2 -> silu -> (x + output) if shortcut
 */
typedef struct {
    conv2d_layer_t conv1;      // 1×1 conv
    batchnorm2d_layer_t bn1;
    int conv1_is_fused;        // Flag: conv1 Conv+BN is fused
    conv2d_layer_t conv2;      // 3×3 conv
    batchnorm2d_layer_t bn2;
    int conv2_is_fused;        // Flag: conv2 Conv+BN is fused
    int shortcut;              // Use shortcut connection
    int32_t c1, c2;            // Input/output channels
} bottleneck_t;

/**
 * Initialize bottleneck block
 */
int bottleneck_init(bottleneck_t* block, int32_t c1, int32_t c2, int shortcut);

/**
 * Free bottleneck block
 */
void bottleneck_free(bottleneck_t* block);

/**
 * Forward pass (int8 quant path)
 */
int bottleneck_forward(bottleneck_t* block, const tensor_t* input, tensor_t* output, tensor_t* workspace);

/**
 * Forward pass using float conv only (for validation reference). Requires block loaded with float weights.
 */
int bottleneck_forward_float(bottleneck_t* block, const tensor_t* input, tensor_t* output, tensor_t* workspace);

/**
 * Load weights from weights loader
 */
int bottleneck_load_weights(bottleneck_t* block, void* weights_loader, const char* prefix);

/**
 * Set debug output directory for intermediate tensor dumps
 */
void bottleneck_set_debug_dir(const char* dir);

#endif // BOTTLENECK_H
