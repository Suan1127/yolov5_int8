#ifndef SPPF_H
#define SPPF_H

#include "../core/tensor.h"
#include "../ops/conv2d.h"
#include "../ops/batchnorm2d.h"
#include "../ops/pooling.h"

/**
 * SPPF (Spatial Pyramid Pooling Fast) block
 * 
 * SPPF(c1, c2, k=5):
 *   c_ = c1 // 2
 *   cv1 = Conv(c1, c_, 1, 1)
 *   cv2 = Conv(c_ * 4, c2, 1, 1)
 *   m = MaxPool2d(k, stride=1, padding=k//2)
 *   
 *   forward:
 *     x = cv1(x)
 *     y1 = x
 *     y2 = m(y1)
 *     y3 = m(y2)
 *     y4 = m(y3)
 *     return cv2(concat([y1, y2, y3, y4]))
 */
typedef struct {
    conv2d_layer_t cv1;        // 1×1 conv: c1 -> c_
    batchnorm2d_layer_t cv1_bn;
    int cv1_is_fused;          // Flag: cv1 Conv+BN is fused
    conv2d_layer_t cv2;        // 1×1 conv: 4*c_ -> c2
    batchnorm2d_layer_t cv2_bn;
    int cv2_is_fused;          // Flag: cv2 Conv+BN is fused
    maxpool2d_params_t pool_params;  // MaxPool parameters
    int32_t c1, c2, c_;       // Input, output, hidden channels
} sppf_block_t;

/**
 * Initialize SPPF block
 */
int sppf_init(sppf_block_t* block, int32_t c1, int32_t c2, int32_t k);

/**
 * Free SPPF block
 */
void sppf_free(sppf_block_t* block);

/**
 * Forward pass (int8 quant path)
 */
int sppf_forward(sppf_block_t* block, const tensor_t* input, tensor_t* output,
                 tensor_t* workspace1, tensor_t* workspace2, tensor_t* workspace3);

/**
 * Forward pass using float conv only (validation reference). Block must have float weights.
 */
int sppf_forward_float(sppf_block_t* block, const tensor_t* input, tensor_t* output,
                       tensor_t* workspace1, tensor_t* workspace2, tensor_t* workspace3);

/**
 * Load weights. If int8_loader is non-NULL, loads int8 for cv1 and cv2.
 */
int sppf_load_weights(sppf_block_t* block, const char* prefix, void* int8_loader);

/**
 * Set debug directory for intermediate outputs (for debugging)
 */
void sppf_set_debug_dir(const char* dir);

#endif // SPPF_H
