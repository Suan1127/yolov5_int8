#ifndef C3_H
#define C3_H

#include "../core/tensor.h"
#include "../ops/conv2d.h"
#include "../ops/batchnorm2d.h"
#include "bottleneck.h"

/**
 * C3 block structure
 * 
 * C3(c1, c2, n=1, shortcut=True):
 *   c_ = int(c2 * 0.5)  # hidden channels
 *   cv1 = Conv(c1, c_, 1, 1)      # 1×1 conv
 *   cv2 = Conv(c1, c_, 1, 1)      # 1×1 conv (skip path)
 *   cv3 = Conv(2*c_, c2, 1, 1)    # 1×1 conv (output)
 *   m = Sequential(Bottleneck(c_, c_, shortcut, e=1.0) × n)
 *   
 *   forward:
 *     y1 = cv3(m(cv1(x)))  # main path
 *     y2 = cv2(x)           # skip path
 *     return cv3(concat([y1, y2]))
 */
typedef struct {
    conv2d_layer_t cv1;        // 1×1 conv: c1 -> c_
    batchnorm2d_layer_t cv1_bn;
    int cv1_is_fused;          // Flag: cv1 Conv+BN is fused
    conv2d_layer_t cv2;        // 1×1 conv: c1 -> c_ (skip path)
    batchnorm2d_layer_t cv2_bn;
    int cv2_is_fused;          // Flag: cv2 Conv+BN is fused
    conv2d_layer_t cv3;        // 1×1 conv: 2*c_ -> c2
    batchnorm2d_layer_t cv3_bn;
    int cv3_is_fused;          // Flag: cv3 Conv+BN is fused
    bottleneck_t* bottlenecks;  // Array of n bottlenecks
    int32_t n;                  // Number of bottlenecks
    int32_t c1, c2, c_;        // Input, output, hidden channels
    int shortcut;              // Use shortcut in bottlenecks
} c3_block_t;

/**
 * Initialize C3 block
 */
int c3_init(c3_block_t* block, int32_t c1, int32_t c2, int32_t n, int shortcut);

/**
 * Free C3 block
 */
void c3_free(c3_block_t* block);

/**
 * Forward pass (int8 quant path)
 */
int c3_forward(c3_block_t* block, const tensor_t* input, tensor_t* output, 
               tensor_t* workspace1, tensor_t* workspace2);

/**
 * Forward pass using float conv only (validation reference). Block must have float weights.
 */
int c3_forward_float(c3_block_t* block, const tensor_t* input, tensor_t* output,
                     tensor_t* workspace1, tensor_t* workspace2);

/**
 * Load weights. If int8_loader is non-NULL, loads int8 for all convs (for quant path).
 */
int c3_load_weights(c3_block_t* block, const char* prefix, void* int8_loader);

/**
 * Set debug output directory for intermediate tensor dumps
 */
void c3_set_debug_dir(const char* dir);

#endif // C3_H
