#include "sppf.h"
#include "../core/common.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../core/weights_loader_int8.h"
#include "../ops/activation.h"
#include "../ops/concat.h"
#include "../core/tensor.h"
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <windows.h>
#define access(path, mode) _access(path, mode)
#define F_OK 0
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

// Debug: Save intermediate outputs if output directory is set
static char g_sppf_debug_dir[512] = {0};
static int g_sppf_debug_enabled = 0;  // Only enable for Layer 9
void sppf_set_debug_dir(const char* dir) {
    if (dir) {
        snprintf(g_sppf_debug_dir, sizeof(g_sppf_debug_dir), "%s", dir);
        g_sppf_debug_enabled = 1;
    } else {
        g_sppf_debug_dir[0] = '\0';
        g_sppf_debug_enabled = 0;
    }
}

// Helper: Find or create debug directory by trying multiple paths
static const char* find_or_create_debug_dir(void) {
    static char found_path[512] = {0};
    static int initialized = 0;
    
    if (initialized && found_path[0] != '\0') {
        return found_path;
    }
    
    // Try paths from project root first (since program runs from build/Release/)
    const char* debug_paths[] = {
        "../../debug/c",      // From build/Release/ -> project root
        "../../../debug/c",    // From build/Release/ -> project root (alternative)
        "../debug/c",         // Fallback
        "debug/c"             // Last resort (current directory)
    };
    
    // Try to find existing directory (prioritize project root)
    for (int i = 0; i < 4; i++) {
        #ifdef _WIN32
        // Convert / to \ for Windows
        char win_path[512];
        snprintf(win_path, sizeof(win_path), "%s", debug_paths[i]);
        for (int j = 0; win_path[j]; j++) {
            if (win_path[j] == '/') win_path[j] = '\\';
        }
        DWORD attrs = GetFileAttributesA(win_path);
        if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY)) {
            snprintf(found_path, sizeof(found_path), "%s", debug_paths[i]);
            initialized = 1;
            return found_path;
        }
        #else
        if (access(debug_paths[i], F_OK) == 0) {
            snprintf(found_path, sizeof(found_path), "%s", debug_paths[i]);
            initialized = 1;
            return found_path;
        }
        #endif
    }
    
    // If not found, try to create in each path
    for (int i = 0; i < 4; i++) {
        // Extract parent directory
        char parent[512];
        snprintf(parent, sizeof(parent), "%s", debug_paths[i]);
        char* last_slash = strrchr(parent, '/');
        #ifdef _WIN32
        if (!last_slash) last_slash = strrchr(parent, '\\');
        #endif
        if (last_slash) {
            *last_slash = '\0';
        }
        
        // Try to create parent and then debug/c
        #ifdef _WIN32
        char win_path[512];
        snprintf(win_path, sizeof(win_path), "%s", debug_paths[i]);
        for (int j = 0; win_path[j]; j++) {
            if (win_path[j] == '/') win_path[j] = '\\';
        }
        if (parent[0] != '\0') {
            char win_parent[512];
            snprintf(win_parent, sizeof(win_parent), "%s", parent);
            for (int j = 0; win_parent[j]; j++) {
                if (win_parent[j] == '/') win_parent[j] = '\\';
            }
            _mkdir(win_parent);
        }
        _mkdir(win_path);
        // Test if creation was successful
        DWORD attrs = GetFileAttributesA(win_path);
        if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY)) {
            snprintf(found_path, sizeof(found_path), "%s", debug_paths[i]);
            initialized = 1;
            return found_path;
        }
        #else
        if (parent[0] != '\0') {
            mkdir(parent, 0755);
        }
        if (mkdir(debug_paths[i], 0755) == 0 || access(debug_paths[i], F_OK) == 0) {
            snprintf(found_path, sizeof(found_path), "%s", debug_paths[i]);
            initialized = 1;
            return found_path;
        }
        #endif
    }
    
    // Fallback to first path
    snprintf(found_path, sizeof(found_path), "debug/c");
    initialized = 1;
    return found_path;
}

// Helper: Build full path for debug file
static void build_debug_path(char* out_path, size_t out_size, const char* filename) {
    const char* debug_dir = find_or_create_debug_dir();
    #ifdef _WIN32
    // Convert / to \ for Windows
    char win_dir[512];
    snprintf(win_dir, sizeof(win_dir), "%s", debug_dir);
    for (int i = 0; win_dir[i]; i++) {
        if (win_dir[i] == '/') win_dir[i] = '\\';
    }
    snprintf(out_path, out_size, "%s\\%s", win_dir, filename);
    #else
    snprintf(out_path, out_size, "%s/%s", debug_dir, filename);
    #endif
}

int sppf_init(sppf_block_t* block, int32_t c1, int32_t c2, int32_t k) {
    if (!block) return -1;
    
    memset(block, 0, sizeof(sppf_block_t));
    block->c1 = c1;
    block->c2 = c2;
    block->c_ = c1 / 2;  // hidden channels = c1 // 2
    block->cv1_is_fused = 0;
    block->cv2_is_fused = 0;
    
    // cv1: 1×1 conv, c1 -> c_
    conv2d_params_t cv1_params = {
        .out_channels = block->c_,
        .kernel_size = 1,
        .stride = 1,
        .padding = 0,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(&block->cv1, c1, &cv1_params) != 0) return -1;
    
    batchnorm2d_params_t cv1_bn_params = {
        .num_features = block->c_,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(&block->cv1_bn, block->c_, &cv1_bn_params) != 0) {
        conv2d_free(&block->cv1);
        return -1;
    }
    
    // cv2: 1×1 conv, 4*c_ -> c2
    conv2d_params_t cv2_params = {
        .out_channels = c2,
        .kernel_size = 1,
        .stride = 1,
        .padding = 0,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(&block->cv2, 4 * block->c_, &cv2_params) != 0) {
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv1);
        return -1;
    }
    
    batchnorm2d_params_t cv2_bn_params = {
        .num_features = c2,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(&block->cv2_bn, c2, &cv2_bn_params) != 0) {
        conv2d_free(&block->cv2);
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv1);
        return -1;
    }
    
    // MaxPool parameters: k×k, stride=1, padding=k//2
    block->pool_params.kernel_size = k;
    block->pool_params.stride = 1;
    block->pool_params.padding = k / 2;
    
    return 0;
}

void sppf_free(sppf_block_t* block) {
    if (block) {
        conv2d_free(&block->cv1);
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv2);
        batchnorm2d_free(&block->cv2_bn);
        memset(block, 0, sizeof(sppf_block_t));
    }
}

int sppf_forward(sppf_block_t* block, const tensor_t* input, tensor_t* output,
                 tensor_t* workspace1, tensor_t* workspace2, tensor_t* workspace3) {
    if (!block || !input || !output) return -1;
    
    // Allocate workspaces if not provided
    int need_free_ws1 = 0, need_free_ws2 = 0, need_free_ws3 = 0;
    
    if (!workspace1) {
        workspace1 = tensor_create(input->n, block->c_, input->h, input->w);
        if (!workspace1) return -1;
        need_free_ws1 = 1;
    }
    
    if (!workspace2) {
        workspace2 = tensor_create(input->n, block->c_, input->h, input->w);
        if (!workspace2) {
            if (need_free_ws1) tensor_free(workspace1);
            return -1;
        }
        need_free_ws2 = 1;
    }
    
    if (!workspace3) {
        workspace3 = tensor_create(input->n, 4 * block->c_, input->h, input->w);
        if (!workspace3) {
            if (need_free_ws1) tensor_free(workspace1);
            if (need_free_ws2) tensor_free(workspace2);
            return -1;
        }
        need_free_ws3 = 1;
    }
    
    // cv1: c1 -> c_ (fused Conv+BN+SiLU to reduce DDR traffic)
    if (conv2d_quant_bn_silu_forward(&block->cv1, block->cv1_is_fused ? NULL : &block->cv1_bn,
            block->cv1_is_fused, input, workspace1) != 0) goto error;
    
    // Debug: Save cv1 output (only if debug enabled)
    if (g_sppf_debug_enabled) {
        char filepath[512];
        build_debug_path(filepath, sizeof(filepath), "sppf_cv1_output.bin");
        if (tensor_dump(workspace1, filepath) != 0) {
            fprintf(stderr, "Warning: Failed to save cv1 output to %s\n", filepath);
        }
    }
    
    // PyTorch SPPF: x = cv1(x), y1 = m(x), y2 = m(y1), y4 = m(y2), concat([x, y1, y2, y4])
    tensor_t* x = workspace1;  // cv1 output
    
    // Create temporary tensors for y1, y2, y4
    tensor_t* y1 = tensor_create(input->n, block->c_, input->h, input->w);
    tensor_t* y2 = tensor_create(input->n, block->c_, input->h, input->w);
    tensor_t* y4 = tensor_create(input->n, block->c_, input->h, input->w);
    
    if (!y1 || !y2 || !y4) {
        if (y1) tensor_free(y1);
        if (y2) tensor_free(y2);
        if (y4) tensor_free(y4);
        goto error;
    }
    
    // Copy x to keep it for concat
    tensor_t* x_copy = tensor_create(input->n, block->c_, input->h, input->w);
    if (!x_copy) {
        tensor_free(y1);
        tensor_free(y2);
        tensor_free(y4);
        goto error;
    }
    tensor_copy(x_copy, x);
    
    // y1 = m(x)
    if (maxpool2d_forward(&block->pool_params, x, y1) != 0) {
        tensor_free(x_copy);
        tensor_free(y1);
        tensor_free(y2);
        tensor_free(y4);
        goto error;
    }
    
    // Debug: Save y1 output
    if (g_sppf_debug_enabled) {
        char filepath[512];
        build_debug_path(filepath, sizeof(filepath), "sppf_y1_output.bin");
        if (tensor_dump(y1, filepath) != 0) {
            fprintf(stderr, "Warning: Failed to save y1 output to %s\n", filepath);
        }
    }
    
    // y2 = m(y1)
    if (maxpool2d_forward(&block->pool_params, y1, y2) != 0) {
        tensor_free(x_copy);
        tensor_free(y1);
        tensor_free(y2);
        tensor_free(y4);
        goto error;
    }
    
    // Debug: Save y2 output
    if (g_sppf_debug_enabled) {
        char filepath[512];
        build_debug_path(filepath, sizeof(filepath), "sppf_y2_output.bin");
        if (tensor_dump(y2, filepath) != 0) {
            fprintf(stderr, "Warning: Failed to save y2 output to %s\n", filepath);
        }
    }
    
    // y4 = m(y2)
    if (maxpool2d_forward(&block->pool_params, y2, y4) != 0) {
        tensor_free(x_copy);
        tensor_free(y1);
        tensor_free(y2);
        tensor_free(y4);
        goto error;
    }
    
    // Debug: Save y4 output
    if (g_sppf_debug_enabled) {
        char filepath[512];
        build_debug_path(filepath, sizeof(filepath), "sppf_y4_output.bin");
        if (tensor_dump(y4, filepath) != 0) {
            fprintf(stderr, "Warning: Failed to save y4 output to %s\n", filepath);
        }
    }
    
    // Concat: [x_copy, y1, y2, y4] -> workspace3
    const tensor_t* concat_inputs[4] = {x_copy, y1, y2, y4};
    if (concat_forward(concat_inputs, 4, workspace3) != 0) {
        tensor_free(x_copy);
        tensor_free(y1);
        tensor_free(y2);
        tensor_free(y4);
        goto error;
    }
    
    // Debug: Save concat output
    if (g_sppf_debug_enabled) {
        char filepath[512];
        build_debug_path(filepath, sizeof(filepath), "sppf_concat_output.bin");
        if (tensor_dump(workspace3, filepath) != 0) {
            fprintf(stderr, "Warning: Failed to save concat output to %s\n", filepath);
        }
    }
    
    // cv2: 4*c_ -> c2 (fused Conv+BN+SiLU to reduce DDR traffic)
    if (conv2d_quant_bn_silu_forward(&block->cv2, block->cv2_is_fused ? NULL : &block->cv2_bn,
            block->cv2_is_fused, workspace3, output) != 0) {
        tensor_free(x_copy);
        tensor_free(y1);
        tensor_free(y2);
        tensor_free(y4);
        goto error;
    }
    
    // Debug: Save cv2 output
    if (g_sppf_debug_enabled) {
        char filepath[512];
        build_debug_path(filepath, sizeof(filepath), "sppf_cv2_output.bin");
        if (tensor_dump(output, filepath) != 0) {
            fprintf(stderr, "Warning: Failed to save cv2 output to %s\n", filepath);
        }
    }
    
    tensor_free(x_copy);
    tensor_free(y1);
    tensor_free(y2);
    tensor_free(y4);
    
    if (need_free_ws1) tensor_free(workspace1);
    if (need_free_ws2) tensor_free(workspace2);
    if (need_free_ws3) tensor_free(workspace3);
    return 0;
    
error:
    if (need_free_ws1) tensor_free(workspace1);
    if (need_free_ws2) tensor_free(workspace2);
    if (need_free_ws3) tensor_free(workspace3);
    return -1;
}

static int sppf_conv_bn_silu_float(conv2d_layer_t* conv, batchnorm2d_layer_t* bn, int fused,
                                  const tensor_t* input, tensor_t* output) {
    if (conv2d_forward(conv, input, output) != 0) return -1;
    if (bn && !fused && batchnorm2d_forward(bn, output, output) != 0) return -1;
    activation_silu(output);
    return 0;
}

int sppf_forward_float(sppf_block_t* block, const tensor_t* input, tensor_t* output,
                      tensor_t* workspace1, tensor_t* workspace2, tensor_t* workspace3) {
    if (!block || !input || !output) return -1;
    if (!block->cv1.weight) return -1;

    int need_free_ws1 = 0, need_free_ws2 = 0, need_free_ws3 = 0;
    if (!workspace1) {
        workspace1 = tensor_create(input->n, block->c_, input->h, input->w);
        if (!workspace1) return -1;
        need_free_ws1 = 1;
    }
    if (!workspace2) {
        workspace2 = tensor_create(input->n, block->c_, input->h, input->w);
        if (!workspace2) {
            if (need_free_ws1) tensor_free(workspace1);
            return -1;
        }
        need_free_ws2 = 1;
    }
    if (!workspace3) {
        workspace3 = tensor_create(input->n, 4 * block->c_, input->h, input->w);
        if (!workspace3) {
            if (need_free_ws1) tensor_free(workspace1);
            if (need_free_ws2) tensor_free(workspace2);
            return -1;
        }
        need_free_ws3 = 1;
    }

    if (sppf_conv_bn_silu_float(&block->cv1, &block->cv1_bn, block->cv1_is_fused, input, workspace1) != 0) {
        if (need_free_ws1) tensor_free(workspace1);
        if (need_free_ws2) tensor_free(workspace2);
        if (need_free_ws3) tensor_free(workspace3);
        return -1;
    }

    tensor_t* x = workspace1;
    tensor_t* y1 = tensor_create(input->n, block->c_, input->h, input->w);
    tensor_t* y2 = tensor_create(input->n, block->c_, input->h, input->w);
    tensor_t* y4 = tensor_create(input->n, block->c_, input->h, input->w);
    tensor_t* x_copy = tensor_create(input->n, block->c_, input->h, input->w);
    if (!y1 || !y2 || !y4 || !x_copy) {
        if (y1) tensor_free(y1);
        if (y2) tensor_free(y2);
        if (y4) tensor_free(y4);
        if (x_copy) tensor_free(x_copy);
        if (need_free_ws1) tensor_free(workspace1);
        if (need_free_ws2) tensor_free(workspace2);
        if (need_free_ws3) tensor_free(workspace3);
        return -1;
    }
    tensor_copy(x_copy, x);
    if (maxpool2d_forward(&block->pool_params, x, y1) != 0 ||
        maxpool2d_forward(&block->pool_params, y1, y2) != 0 ||
        maxpool2d_forward(&block->pool_params, y2, y4) != 0) {
        tensor_free(x_copy);
        tensor_free(y1);
        tensor_free(y2);
        tensor_free(y4);
        if (need_free_ws1) tensor_free(workspace1);
        if (need_free_ws2) tensor_free(workspace2);
        if (need_free_ws3) tensor_free(workspace3);
        return -1;
    }
    const tensor_t* concat_inputs[4] = {x_copy, y1, y2, y4};
    if (concat_forward(concat_inputs, 4, workspace3) != 0) {
        tensor_free(x_copy);
        tensor_free(y1);
        tensor_free(y2);
        tensor_free(y4);
        if (need_free_ws1) tensor_free(workspace1);
        if (need_free_ws2) tensor_free(workspace2);
        if (need_free_ws3) tensor_free(workspace3);
        return -1;
    }
    tensor_free(x_copy);
    tensor_free(y1);
    tensor_free(y2);
    tensor_free(y4);

    if (sppf_conv_bn_silu_float(&block->cv2, &block->cv2_bn, block->cv2_is_fused, workspace3, output) != 0) {
        if (need_free_ws1) tensor_free(workspace1);
        if (need_free_ws2) tensor_free(workspace2);
        if (need_free_ws3) tensor_free(workspace3);
        return -1;
    }
    if (need_free_ws1) tensor_free(workspace1);
    if (need_free_ws2) tensor_free(workspace2);
    if (need_free_ws3) tensor_free(workspace3);
    return 0;
}

int sppf_load_weights(sppf_block_t* block, const char* prefix, void* int8_loader) {
    if (!block || !int8_loader) return -1;
    weights_loader_int8_t* int8 = (weights_loader_int8_t*)int8_loader;
    char name[256];
    const float* bias_ptr;
    size_t bias_count;
    const int8_t* qptr;
    float scale_w;
    size_t numel;
    snprintf(name, sizeof(name), "%s.cv1.conv.weight", prefix);
    if (weights_loader_int8_get(int8, name, &qptr, &scale_w, &numel) != 0 ||
        weights_loader_int8_get_bias(int8, name, &bias_ptr, &bias_count) != 0 ||
        !bias_ptr || bias_count != (size_t)block->c_) return -1;
    if (conv2d_load_weights_int8(&block->cv1, qptr, numel, scale_w, bias_ptr) != 0) return -1;
    block->cv1_is_fused = 1;
    for (int i = 0; i < block->c_; i++) {
        block->cv1_bn.weight[i] = 1.0f; block->cv1_bn.bias[i] = 0.0f;
        block->cv1_bn.running_mean[i] = 0.0f; block->cv1_bn.running_var[i] = 1.0f;
    }
    snprintf(name, sizeof(name), "%s.cv2.conv.weight", prefix);
    if (weights_loader_int8_get(int8, name, &qptr, &scale_w, &numel) != 0 ||
        weights_loader_int8_get_bias(int8, name, &bias_ptr, &bias_count) != 0 ||
        !bias_ptr || bias_count != (size_t)block->c2) return -1;
    if (conv2d_load_weights_int8(&block->cv2, qptr, numel, scale_w, bias_ptr) != 0) return -1;
    block->cv2_is_fused = 1;
    for (int i = 0; i < block->c2; i++) {
        block->cv2_bn.weight[i] = 1.0f; block->cv2_bn.bias[i] = 0.0f;
        block->cv2_bn.running_mean[i] = 0.0f; block->cv2_bn.running_var[i] = 1.0f;
    }
    return 0;
}
