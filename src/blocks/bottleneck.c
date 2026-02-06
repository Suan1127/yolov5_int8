#include "bottleneck.h"
#include "../core/common.h"
#include "../core/tensor.h"
#include <stdlib.h>
#include "../core/weights_loader.h"
#include "../ops/activation.h"

int bottleneck_init(bottleneck_t* block, int32_t c1, int32_t c2, int shortcut) {
    if (!block) return -1;
    
    memset(block, 0, sizeof(bottleneck_t));
    block->conv1_is_fused = 0;
    block->conv2_is_fused = 0;
    block->c1 = c1;
    block->c2 = c2;
    block->shortcut = shortcut;
    
    // Conv1: 1×1, c1 -> c2
    conv2d_params_t conv1_params = {
        .out_channels = c2,
        .kernel_size = 1,
        .stride = 1,
        .padding = 0,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(&block->conv1, c1, &conv1_params) != 0) return -1;
    
    batchnorm2d_params_t bn1_params = {
        .num_features = c2,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(&block->bn1, c2, &bn1_params) != 0) {
        conv2d_free(&block->conv1);
        return -1;
    }
    
    // Conv2: 3×3, c2 -> c2
    conv2d_params_t conv2_params = {
        .out_channels = c2,
        .kernel_size = 3,
        .stride = 1,
        .padding = 1,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(&block->conv2, c2, &conv2_params) != 0) {
        batchnorm2d_free(&block->bn1);
        conv2d_free(&block->conv1);
        return -1;
    }
    
    batchnorm2d_params_t bn2_params = {
        .num_features = c2,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(&block->bn2, c2, &bn2_params) != 0) {
        conv2d_free(&block->conv2);
        batchnorm2d_free(&block->bn1);
        conv2d_free(&block->conv1);
        return -1;
    }
    
    return 0;
}

void bottleneck_free(bottleneck_t* block) {
    if (block) {
        conv2d_free(&block->conv1);
        batchnorm2d_free(&block->bn1);
        conv2d_free(&block->conv2);
        batchnorm2d_free(&block->bn2);
        memset(block, 0, sizeof(bottleneck_t));
    }
}

// Global debug directory for bottleneck intermediate dumps
static char g_bottleneck_debug_dir[512] = "";

void bottleneck_set_debug_dir(const char* dir) {
    if (dir) {
        strncpy(g_bottleneck_debug_dir, dir, sizeof(g_bottleneck_debug_dir) - 1);
        g_bottleneck_debug_dir[sizeof(g_bottleneck_debug_dir) - 1] = '\0';
        
        // Note: Directory should already exist, so we don't create it here
        // If directory doesn't exist, file operations will fail gracefully
    } else {
        g_bottleneck_debug_dir[0] = '\0';
    }
}

int bottleneck_forward(bottleneck_t* block, const tensor_t* input, tensor_t* output, tensor_t* workspace) {
    if (!block || !input || !output) return -1;
    
    // workspace is used for intermediate results
    int need_free_workspace = 0;
    if (!workspace) {
        // Check if input and output share the same memory
        // If they do, we need a separate workspace to avoid overwriting input data
        if (input->data == output->data) {
            // Allocate separate workspace
            workspace = tensor_create(input->n, block->c2, input->h, input->w);
            if (!workspace) {
                fprintf(stderr, "Error: bottleneck_forward: Failed to allocate workspace\n");
                return -1;
            }
            need_free_workspace = 1;
        } else {
            workspace = output;  // Use output as workspace if not provided and memory is safe
        }
    }
    
    // Memory relationship check removed - issue resolved
    
    // Conv1 -> BN1 -> SiLU (fused to reduce DDR read/write)
    if (conv2d_quant_bn_silu_forward(&block->conv1, block->conv1_is_fused ? NULL : &block->bn1,
            block->conv1_is_fused, input, workspace) != 0) {
        if (need_free_workspace && workspace) tensor_free(workspace);
        return -1;
    }
    
    // Conv2 -> BN2 -> SiLU (fused). Need separate temp: conv2 output cannot overwrite workspace (used as input).
    tensor_t* temp = tensor_create(input->n, block->c2, input->h, input->w);
    if (!temp) {
        if (need_free_workspace && workspace) tensor_free(workspace);
        return -1;
    }
    if (conv2d_quant_bn_silu_forward(&block->conv2, block->conv2_is_fused ? NULL : &block->bn2,
            block->conv2_is_fused, workspace, temp) != 0) {
        tensor_free(temp);
        if (need_free_workspace && workspace) tensor_free(workspace);
        return -1;
    }
    
    // Add shortcut if enabled
    if (block->shortcut) {
        // output = input + temp
        for (size_t i = 0; i < tensor_size(input); i++) {
            output->data[i] = input->data[i] + temp->data[i];
        }
    } else {
        // output = temp
        tensor_copy(output, temp);
    }
    
    // Free temp buffer (always allocated separately)
    tensor_free(temp);
    
    // Free workspace if we allocated it
    if (need_free_workspace && workspace) {
        tensor_free(workspace);
    }
    
    return 0;
}

/* Float path: conv2d_forward + (BN if !fused) + SiLU. Requires layer->weight. */
static int conv_bn_silu_float(conv2d_layer_t* conv, batchnorm2d_layer_t* bn, int fused,
                              const tensor_t* input, tensor_t* output) {
    if (conv2d_forward(conv, input, output) != 0) return -1;
    if (bn && !fused && batchnorm2d_forward(bn, output, output) != 0) return -1;
    activation_silu(output);
    return 0;
}

int bottleneck_forward_float(bottleneck_t* block, const tensor_t* input, tensor_t* output, tensor_t* workspace) {
    if (!block || !input || !output) return -1;
    if (!block->conv1.weight) return -1;  /* float path requires float weights */

    int need_free_workspace = 0;
    if (!workspace) {
        if (input->data == output->data) {
            workspace = tensor_create(input->n, block->c2, input->h, input->w);
            if (!workspace) return -1;
            need_free_workspace = 1;
        } else {
            workspace = output;
        }
    }

    if (conv_bn_silu_float(&block->conv1, &block->bn1, block->conv1_is_fused, input, workspace) != 0) {
        if (need_free_workspace && workspace) tensor_free(workspace);
        return -1;
    }

    tensor_t* temp = tensor_create(input->n, block->c2, input->h, input->w);
    if (!temp) {
        if (need_free_workspace && workspace) tensor_free(workspace);
        return -1;
    }
    if (conv_bn_silu_float(&block->conv2, &block->bn2, block->conv2_is_fused, workspace, temp) != 0) {
        tensor_free(temp);
        if (need_free_workspace && workspace) tensor_free(workspace);
        return -1;
    }

    if (block->shortcut) {
        for (size_t i = 0; i < tensor_size(input); i++)
            output->data[i] = input->data[i] + temp->data[i];
    } else {
        tensor_copy(output, temp);
    }
    tensor_free(temp);
    if (need_free_workspace && workspace) tensor_free(workspace);
    return 0;
}

int bottleneck_load_weights(bottleneck_t* block, void* weights_loader, const char* prefix) {
    if (!block || !weights_loader) return -1;
    
    weights_loader_t* loader = (weights_loader_t*)weights_loader;
    char name[256];
    int32_t shape[4];
    int num_dims;
    
    // Load conv1 weights
    snprintf(name, sizeof(name), "%s.cv1.conv.weight", prefix);
    float* w = weights_loader_get(loader, name, shape, &num_dims);
    if (!w) {
        fprintf(stderr, "Error: Failed to load weight for %s\n", name);
        return -1;
    }
    // Try to load fused bias
    snprintf(name, sizeof(name), "%s.cv1.conv.bias", prefix);
    float* fused_bias = weights_loader_get(loader, name, shape, &num_dims);
    conv2d_load_weights(&block->conv1, w, fused_bias);
    
    // Load BN weights or set to identity if fused
    if (fused_bias) {
        // Fused: set BN to identity
        block->conv1_is_fused = 1;
        for (int i = 0; i < block->c2; i++) {
            block->bn1.weight[i] = 1.0f;
            block->bn1.bias[i] = 0.0f;
            block->bn1.running_mean[i] = 0.0f;
            block->bn1.running_var[i] = 1.0f;
        }
    } else {
        // Not fused: load BN weights
        block->conv1_is_fused = 0;
        snprintf(name, sizeof(name), "%s.cv1.bn.weight", prefix);
        float* bn_w = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv1.bn.bias", prefix);
        float* bn_b = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv1.bn.running_mean", prefix);
        float* bn_mean = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv1.bn.running_var", prefix);
        float* bn_var = weights_loader_get(loader, name, shape, &num_dims);
        if (bn_w && bn_b && bn_mean && bn_var) {
            batchnorm2d_load_weights(&block->bn1, bn_w, bn_b, bn_mean, bn_var);
        }
    }
    
    // Load conv2 weights
    snprintf(name, sizeof(name), "%s.cv2.conv.weight", prefix);
    w = weights_loader_get(loader, name, shape, &num_dims);
    if (!w) {
        fprintf(stderr, "Error: Failed to load weight for %s\n", name);
        return -1;
    }
    // Try to load fused bias
    snprintf(name, sizeof(name), "%s.cv2.conv.bias", prefix);
    fused_bias = weights_loader_get(loader, name, shape, &num_dims);
    conv2d_load_weights(&block->conv2, w, fused_bias);
    
    // Load BN weights or set to identity if fused
    if (fused_bias) {
        // Fused: set BN to identity
        block->conv2_is_fused = 1;
        for (int i = 0; i < block->c2; i++) {
            block->bn2.weight[i] = 1.0f;
            block->bn2.bias[i] = 0.0f;
            block->bn2.running_mean[i] = 0.0f;
            block->bn2.running_var[i] = 1.0f;
        }
    } else {
        // Not fused: load BN weights
        block->conv2_is_fused = 0;
        snprintf(name, sizeof(name), "%s.cv2.bn.weight", prefix);
        float* bn_w = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv2.bn.bias", prefix);
        float* bn_b = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv2.bn.running_mean", prefix);
        float* bn_mean = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv2.bn.running_var", prefix);
        float* bn_var = weights_loader_get(loader, name, shape, &num_dims);
        if (bn_w && bn_b && bn_mean && bn_var) {
            batchnorm2d_load_weights(&block->bn2, bn_w, bn_b, bn_mean, bn_var);
        }
    }
    
    return 0;
}
