#include "c3.h"
#include "../core/common.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../core/weights_loader.h"
#include "../core/weights_loader_int8.h"
#include "../ops/activation.h"
#include "../ops/concat.h"
#include "../core/tensor.h"
#include "bottleneck.h"
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
static char g_c3_debug_dir[512] = {0};
static int g_c3_debug_enabled = 0;  // Only enable for Layer 2
void c3_set_debug_dir(const char* dir) {
    if (dir) {
        snprintf(g_c3_debug_dir, sizeof(g_c3_debug_dir), "%s", dir);
        g_c3_debug_enabled = 1;
    } else {
        g_c3_debug_dir[0] = '\0';
        g_c3_debug_enabled = 0;
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

int c3_init(c3_block_t* block, int32_t c1, int32_t c2, int32_t n, int shortcut) {
    if (!block) return -1;
    
    memset(block, 0, sizeof(c3_block_t));
    block->c1 = c1;
    block->c2 = c2;
    block->c_ = c2 / 2;  // hidden channels = c2 * 0.5
    block->n = n;
    block->shortcut = shortcut;
    
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
    
    // cv2: 1×1 conv, c1 -> c_ (skip path)
    conv2d_params_t cv2_params = {
        .out_channels = block->c_,
        .kernel_size = 1,
        .stride = 1,
        .padding = 0,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(&block->cv2, c1, &cv2_params) != 0) {
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv1);
        return -1;
    }
    
    batchnorm2d_params_t cv2_bn_params = {
        .num_features = block->c_,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(&block->cv2_bn, block->c_, &cv2_bn_params) != 0) {
        conv2d_free(&block->cv2);
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv1);
        return -1;
    }
    
    // cv3: 1×1 conv, 2*c_ -> c2
    conv2d_params_t cv3_params = {
        .out_channels = c2,
        .kernel_size = 1,
        .stride = 1,
        .padding = 0,
        .groups = 1,
        .dilation = 1
    };
    if (conv2d_init(&block->cv3, 2 * block->c_, &cv3_params) != 0) {
        batchnorm2d_free(&block->cv2_bn);
        conv2d_free(&block->cv2);
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv1);
        return -1;
    }
    
    batchnorm2d_params_t cv3_bn_params = {
        .num_features = c2,
        .eps = 1e-5f,
        .momentum = 0.1f
    };
    if (batchnorm2d_init(&block->cv3_bn, c2, &cv3_bn_params) != 0) {
        conv2d_free(&block->cv3);
        batchnorm2d_free(&block->cv2_bn);
        conv2d_free(&block->cv2);
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv1);
        return -1;
    }
    
    // Initialize bottlenecks
    if (n > 0) {
        block->bottlenecks = (bottleneck_t*)calloc(n, sizeof(bottleneck_t));
        if (!block->bottlenecks) {
            batchnorm2d_free(&block->cv3_bn);
            conv2d_free(&block->cv3);
            batchnorm2d_free(&block->cv2_bn);
            conv2d_free(&block->cv2);
            batchnorm2d_free(&block->cv1_bn);
            conv2d_free(&block->cv1);
            return -1;
        }
        
        for (int i = 0; i < n; i++) {
            if (bottleneck_init(&block->bottlenecks[i], block->c_, block->c_, shortcut) != 0) {
                // Cleanup
                for (int j = 0; j < i; j++) {
                    bottleneck_free(&block->bottlenecks[j]);
                }
                free(block->bottlenecks);
                batchnorm2d_free(&block->cv3_bn);
                conv2d_free(&block->cv3);
                batchnorm2d_free(&block->cv2_bn);
                conv2d_free(&block->cv2);
                batchnorm2d_free(&block->cv1_bn);
                conv2d_free(&block->cv1);
                return -1;
            }
        }
    }
    
    return 0;
}

void c3_free(c3_block_t* block) {
    if (block) {
        conv2d_free(&block->cv1);
        batchnorm2d_free(&block->cv1_bn);
        conv2d_free(&block->cv2);
        batchnorm2d_free(&block->cv2_bn);
        conv2d_free(&block->cv3);
        batchnorm2d_free(&block->cv3_bn);
        
        if (block->bottlenecks) {
            for (int i = 0; i < block->n; i++) {
                bottleneck_free(&block->bottlenecks[i]);
            }
            free(block->bottlenecks);
        }
        
        memset(block, 0, sizeof(c3_block_t));
    }
}

int c3_forward(c3_block_t* block, const tensor_t* input, tensor_t* output, 
               tensor_t* workspace1, tensor_t* workspace2) {
    if (!block || !input || !output) {
        fprintf(stderr, "Error: c3_forward: NULL pointer\n");
        return -1;
    }
    
    // Debug directory will be found/created when needed
    
    // Allocate workspaces if not provided
    int need_free_ws1 = 0, need_free_ws2 = 0;
    
    if (!workspace1) {
        workspace1 = tensor_create(input->n, block->c_, input->h, input->w);
        if (!workspace1) {
            fprintf(stderr, "Error: c3_forward: Failed to allocate workspace1 (%d, %d, %d, %d)\n",
                    input->n, block->c_, input->h, input->w);
            return -1;
        }
        need_free_ws1 = 1;
    }
    
    if (!workspace2) {
        workspace2 = tensor_create(input->n, 2 * block->c_, input->h, input->w);
        if (!workspace2) {
            fprintf(stderr, "Error: c3_forward: Failed to allocate workspace2 (%d, %d, %d, %d)\n",
                    input->n, 2 * block->c_, input->h, input->w);
            if (need_free_ws1) tensor_free(workspace1);
            return -1;
        }
        need_free_ws2 = 1;
    }
    
    // Debug: Check memory relationship
    if (input->data == workspace1->data || input->data == workspace2->data || 
        input->data == output->data || workspace1->data == output->data) {
        fprintf(stderr, "Warning: c3_forward: Memory overlap detected!\n");
        fprintf(stderr, "  input->data=%p, output->data=%p\n", (void*)input->data, (void*)output->data);
        fprintf(stderr, "  workspace1->data=%p, workspace2->data=%p\n", (void*)workspace1->data, (void*)workspace2->data);
    }
    
    // Main path: cv1 -> bottlenecks -> (stored in workspace1)
    // Check if input and workspace1 share the same memory
    if (input->data == workspace1->data) {
        fprintf(stderr, "Error: c3_forward: cv1 input and workspace1 share the same memory!\n");
        fprintf(stderr, "  input->data=%p, workspace1->data=%p\n", (void*)input->data, (void*)workspace1->data);
        fprintf(stderr, "  This will cause input data to be overwritten. Need separate workspace.\n");
        goto error;
    }
    
    if (conv2d_quant_bn_silu_forward(&block->cv1, block->cv1_is_fused ? NULL : &block->cv1_bn,
            block->cv1_is_fused, input, workspace1) != 0) {
        fprintf(stderr, "Error: c3_forward: cv1 conv2d (fused) failed\n");
        fprintf(stderr, "  input: (%d, %d, %d, %d), workspace1: (%d, %d, %d, %d)\n",
                input->n, input->c, input->h, input->w,
                workspace1->n, workspace1->c, workspace1->h, workspace1->w);
        fprintf(stderr, "  cv1: in_channels=%d, out_channels=%d\n",
                block->cv1.in_channels, block->cv1.params.out_channels);
        goto error;
    }
    
    // Debug: Save cv1 output (only if debug enabled)
    if (g_c3_debug_enabled) {
        char filepath[512];
        build_debug_path(filepath, sizeof(filepath), "c3_cv1_output.bin");
        if (tensor_dump(workspace1, filepath) != 0) {
            fprintf(stderr, "Warning: Failed to save cv1 output to %s\n", filepath);
        }
    }
    
    // Pass through bottlenecks
    // Bottlenecks need temporary buffer (c_ channels)
    // For n > 1, we need separate buffer for intermediate bottlenecks
    tensor_t* bottleneck_temp = NULL;
    if (block->n > 1) {
        bottleneck_temp = tensor_create(input->n, block->c_, input->h, input->w);
        if (!bottleneck_temp) {
            fprintf(stderr, "Error: c3_forward: Failed to allocate bottleneck_temp\n");
            goto error;
        }
    }
    
    tensor_t* bottleneck_input = workspace1;
    for (int i = 0; i < block->n; i++) {
        // For last bottleneck, output goes to workspace1 (reuse)
        // For others, use bottleneck_temp
        tensor_t* bottleneck_output = (i == block->n - 1) ? workspace1 : bottleneck_temp;
        if (bottleneck_forward(&block->bottlenecks[i], bottleneck_input, bottleneck_output, NULL) != 0) {
            fprintf(stderr, "Error: c3_forward: bottleneck %d failed\n", i);
            if (bottleneck_temp) tensor_free(bottleneck_temp);
            goto error;
        }
        bottleneck_input = bottleneck_output;
    }
    
    if (bottleneck_temp) tensor_free(bottleneck_temp);
    
    // Debug: Save bottleneck output (only if debug enabled)
    if (g_c3_debug_enabled) {
        char filepath[512];
        build_debug_path(filepath, sizeof(filepath), "c3_bottleneck_output.bin");
        if (tensor_dump(workspace1, filepath) != 0) {
            fprintf(stderr, "Warning: Failed to save bottleneck output to %s\n", filepath);
        }
    }
    
    // Skip path: cv2
    // Need separate buffer for skip path output (c_ channels)
    tensor_t* skip_output = tensor_create(input->n, block->c_, input->h, input->w);
    if (!skip_output) {
        fprintf(stderr, "Error: c3_forward: Failed to allocate skip_output\n");
        goto error;
    }
    
    if (conv2d_quant_bn_silu_forward(&block->cv2, block->cv2_is_fused ? NULL : &block->cv2_bn,
            block->cv2_is_fused, input, skip_output) != 0) {
        fprintf(stderr, "Error: c3_forward: cv2 conv2d (fused) failed\n");
        tensor_free(skip_output);
        goto error;
    }
    
    // Debug: Save cv2 output (only if debug enabled)
    if (g_c3_debug_enabled) {
        char filepath[512];
        build_debug_path(filepath, sizeof(filepath), "c3_cv2_output.bin");
        if (tensor_dump(skip_output, filepath) != 0) {
            fprintf(stderr, "Warning: Failed to save cv2 output to %s\n", filepath);
        }
    }
    
    // Concat: [workspace1 (main path), skip_output] -> workspace2
    const tensor_t* concat_inputs[2] = {workspace1, skip_output};
    if (concat_forward(concat_inputs, 2, workspace2) != 0) {
        fprintf(stderr, "Error: c3_forward: concat failed\n");
        tensor_free(skip_output);
        goto error;
    }
    
    tensor_free(skip_output);  // Free skip_output after concat
    
    // Debug: Save concat output (only if debug enabled)
    if (g_c3_debug_enabled) {
        char filepath[512];
        build_debug_path(filepath, sizeof(filepath), "c3_concat_output.bin");
        if (tensor_dump(workspace2, filepath) != 0) {
            fprintf(stderr, "Warning: Failed to save concat output to %s\n", filepath);
        }
    }
    
    // cv3: 2*c_ -> c2
    // Verify tensor sizes before conv2d
    if (workspace2->c != 2 * block->c_) {
        fprintf(stderr, "Error: c3_forward: workspace2 channels mismatch. Expected %d, got %d\n",
                2 * block->c_, workspace2->c);
        goto error;
    }
    if (output->c != block->c2) {
        fprintf(stderr, "Error: c3_forward: output channels mismatch. Expected %d, got %d\n",
                block->c2, output->c);
        goto error;
    }
    if (block->cv3.in_channels != 2 * block->c_) {
        fprintf(stderr, "Error: c3_forward: cv3 in_channels mismatch. Expected %d, got %d\n",
                2 * block->c_, block->cv3.in_channels);
        goto error;
    }
    
    if (conv2d_quant_bn_silu_forward(&block->cv3, block->cv3_is_fused ? NULL : &block->cv3_bn,
            block->cv3_is_fused, workspace2, output) != 0) {
        fprintf(stderr, "Error: c3_forward: cv3 conv2d (fused) failed\n");
        fprintf(stderr, "  workspace2: (%d, %d, %d, %d)\n",
                workspace2->n, workspace2->c, workspace2->h, workspace2->w);
        fprintf(stderr, "  output: (%d, %d, %d, %d)\n",
                output->n, output->c, output->h, output->w);
        fprintf(stderr, "  cv3: in_channels=%d, out_channels=%d\n",
                block->cv3.in_channels, block->cv3.params.out_channels);
        goto error;
    }
    
    // Debug: Save final cv3 output (only if debug enabled)
    if (g_c3_debug_enabled) {
        char filepath[512];
        build_debug_path(filepath, sizeof(filepath), "c3_final_output.bin");
        if (tensor_dump(output, filepath) != 0) {
            fprintf(stderr, "Warning: Failed to save final cv3 output to %s\n", filepath);
        }
    }
    
    if (need_free_ws1) tensor_free(workspace1);
    if (need_free_ws2) tensor_free(workspace2);
    return 0;
    
error:
    if (need_free_ws1) tensor_free(workspace1);
    if (need_free_ws2) tensor_free(workspace2);
    // Note: bottleneck_temp and skip_output are freed in their respective sections
    return -1;
}

/* Float path: conv2d_forward + (BN if !fused) + SiLU. Requires layer->weight. */
static int c3_conv_bn_silu_float(conv2d_layer_t* conv, batchnorm2d_layer_t* bn, int fused,
                                 const tensor_t* input, tensor_t* output) {
    if (conv2d_forward(conv, input, output) != 0) return -1;
    if (bn && !fused && batchnorm2d_forward(bn, output, output) != 0) return -1;
    activation_silu(output);
    return 0;
}

int c3_forward_float(c3_block_t* block, const tensor_t* input, tensor_t* output,
                    tensor_t* workspace1, tensor_t* workspace2) {
    if (!block || !input || !output) return -1;
    if (!block->cv1.weight) return -1;  /* float path requires float weights */

    int need_free_ws1 = 0, need_free_ws2 = 0;
    if (!workspace1) {
        workspace1 = tensor_create(input->n, block->c_, input->h, input->w);
        if (!workspace1) return -1;
        need_free_ws1 = 1;
    }
    if (!workspace2) {
        workspace2 = tensor_create(input->n, 2 * block->c_, input->h, input->w);
        if (!workspace2) {
            if (need_free_ws1) tensor_free(workspace1);
            return -1;
        }
        need_free_ws2 = 1;
    }
    if (input->data == workspace1->data) {
        if (need_free_ws1) tensor_free(workspace1);
        if (need_free_ws2) tensor_free(workspace2);
        return -1;
    }

    if (c3_conv_bn_silu_float(&block->cv1, &block->cv1_bn, block->cv1_is_fused, input, workspace1) != 0) {
        if (need_free_ws1) tensor_free(workspace1);
        if (need_free_ws2) tensor_free(workspace2);
        return -1;
    }

    tensor_t* bottleneck_temp = NULL;
    if (block->n > 1) {
        bottleneck_temp = tensor_create(input->n, block->c_, input->h, input->w);
        if (!bottleneck_temp) {
            if (need_free_ws1) tensor_free(workspace1);
            if (need_free_ws2) tensor_free(workspace2);
            return -1;
        }
    }
    tensor_t* bottleneck_input = workspace1;
    for (int i = 0; i < block->n; i++) {
        tensor_t* bottleneck_output = (i == block->n - 1) ? workspace1 : bottleneck_temp;
        if (bottleneck_forward_float(&block->bottlenecks[i], bottleneck_input, bottleneck_output, NULL) != 0) {
            if (bottleneck_temp) tensor_free(bottleneck_temp);
            if (need_free_ws1) tensor_free(workspace1);
            if (need_free_ws2) tensor_free(workspace2);
            return -1;
        }
        bottleneck_input = bottleneck_output;
    }
    if (bottleneck_temp) tensor_free(bottleneck_temp);

    tensor_t* skip_output = tensor_create(input->n, block->c_, input->h, input->w);
    if (!skip_output) {
        if (need_free_ws1) tensor_free(workspace1);
        if (need_free_ws2) tensor_free(workspace2);
        return -1;
    }
    if (c3_conv_bn_silu_float(&block->cv2, &block->cv2_bn, block->cv2_is_fused, input, skip_output) != 0) {
        tensor_free(skip_output);
        if (need_free_ws1) tensor_free(workspace1);
        if (need_free_ws2) tensor_free(workspace2);
        return -1;
    }

    const tensor_t* concat_inputs[2] = {workspace1, skip_output};
    if (concat_forward(concat_inputs, 2, workspace2) != 0) {
        tensor_free(skip_output);
        if (need_free_ws1) tensor_free(workspace1);
        if (need_free_ws2) tensor_free(workspace2);
        return -1;
    }
    tensor_free(skip_output);

    if (c3_conv_bn_silu_float(&block->cv3, &block->cv3_bn, block->cv3_is_fused, workspace2, output) != 0) {
        if (need_free_ws1) tensor_free(workspace1);
        if (need_free_ws2) tensor_free(workspace2);
        return -1;
    }

    if (need_free_ws1) tensor_free(workspace1);
    if (need_free_ws2) tensor_free(workspace2);
    return 0;
}

int c3_load_weights(c3_block_t* block, void* weights_loader, const char* prefix, void* int8_loader) {
    if (!block || !weights_loader) return -1;
    
    weights_loader_t* loader = (weights_loader_t*)weights_loader;
    weights_loader_int8_t* int8 = (weights_loader_int8_t*)int8_loader;
    char name[256];
    int32_t shape[4];
    int num_dims;
    
    // Load cv1
    snprintf(name, sizeof(name), "%s.cv1.conv.weight", prefix);
    float* w = weights_loader_get(loader, name, shape, &num_dims);
    if (!w) {
        fprintf(stderr, "Error: Failed to load weight for %s\n", name);
        return -1;
    }
    // Try to load fused bias
    snprintf(name, sizeof(name), "%s.cv1.conv.bias", prefix);
    float* fused_bias = weights_loader_get(loader, name, shape, &num_dims);
    conv2d_load_weights(&block->cv1, w, fused_bias);
    
    // Load BN weights or set to identity if fused
    if (fused_bias) {
        // Fused: set BN to identity
        block->cv1_is_fused = 1;
        for (int i = 0; i < block->c_; i++) {
            block->cv1_bn.weight[i] = 1.0f;
            block->cv1_bn.bias[i] = 0.0f;
            block->cv1_bn.running_mean[i] = 0.0f;
            block->cv1_bn.running_var[i] = 1.0f;
        }
    } else {
        // Not fused: load BN weights
        block->cv1_is_fused = 0;
        snprintf(name, sizeof(name), "%s.cv1.bn.weight", prefix);
        float* bn_w = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv1.bn.bias", prefix);
        float* bn_b = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv1.bn.running_mean", prefix);
        float* bn_mean = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv1.bn.running_var", prefix);
        float* bn_var = weights_loader_get(loader, name, shape, &num_dims);
        if (bn_w && bn_b && bn_mean && bn_var) {
            batchnorm2d_load_weights(&block->cv1_bn, bn_w, bn_b, bn_mean, bn_var);
        }
    }
    if (int8) {
        snprintf(name, sizeof(name), "%s.cv1.conv.weight", prefix);
        const int8_t* qptr; float scale_w; size_t numel;
        if (weights_loader_int8_get(int8, name, &qptr, &scale_w, &numel) == 0)
            conv2d_load_weights_int8(&block->cv1, qptr, numel, scale_w, fused_bias);
    }
    
    // Load cv2
    snprintf(name, sizeof(name), "%s.cv2.conv.weight", prefix);
    w = weights_loader_get(loader, name, shape, &num_dims);
    if (!w) {
        fprintf(stderr, "Error: Failed to load weight for %s\n", name);
        return -1;
    }
    // Try to load fused bias
    snprintf(name, sizeof(name), "%s.cv2.conv.bias", prefix);
    fused_bias = weights_loader_get(loader, name, shape, &num_dims);
    
    conv2d_load_weights(&block->cv2, w, fused_bias);
    
    // Load BN weights or set to identity if fused
    if (fused_bias) {
        // Fused: set BN to identity
        block->cv2_is_fused = 1;
        for (int i = 0; i < block->c_; i++) {
            block->cv2_bn.weight[i] = 1.0f;
            block->cv2_bn.bias[i] = 0.0f;
            block->cv2_bn.running_mean[i] = 0.0f;
            block->cv2_bn.running_var[i] = 1.0f;
        }
    } else {
        // Not fused: load BN weights
        block->cv2_is_fused = 0;
        snprintf(name, sizeof(name), "%s.cv2.bn.weight", prefix);
        float* bn_w = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv2.bn.bias", prefix);
        float* bn_b = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv2.bn.running_mean", prefix);
        float* bn_mean = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv2.bn.running_var", prefix);
        float* bn_var = weights_loader_get(loader, name, shape, &num_dims);
        if (bn_w && bn_b && bn_mean && bn_var) {
            batchnorm2d_load_weights(&block->cv2_bn, bn_w, bn_b, bn_mean, bn_var);
        }
    }
    if (int8) {
        snprintf(name, sizeof(name), "%s.cv2.conv.weight", prefix);
        const int8_t* qptr; float scale_w; size_t numel;
        if (weights_loader_int8_get(int8, name, &qptr, &scale_w, &numel) == 0)
            conv2d_load_weights_int8(&block->cv2, qptr, numel, scale_w, fused_bias);
    }
    
    // Load cv3
    snprintf(name, sizeof(name), "%s.cv3.conv.weight", prefix);
    w = weights_loader_get(loader, name, shape, &num_dims);
    if (!w) {
        fprintf(stderr, "Error: Failed to load weight for %s\n", name);
        return -1;
    }
    
    // Try to load fused bias
    snprintf(name, sizeof(name), "%s.cv3.conv.bias", prefix);
    fused_bias = weights_loader_get(loader, name, shape, &num_dims);
    conv2d_load_weights(&block->cv3, w, fused_bias);
    
    // Load BN weights or set to identity if fused
    if (fused_bias) {
        // Fused: set BN to identity
        block->cv3_is_fused = 1;
        for (int i = 0; i < block->c2; i++) {
            block->cv3_bn.weight[i] = 1.0f;
            block->cv3_bn.bias[i] = 0.0f;
            block->cv3_bn.running_mean[i] = 0.0f;
            block->cv3_bn.running_var[i] = 1.0f;
        }
    } else {
        // Not fused: load BN weights
        block->cv3_is_fused = 0;
        snprintf(name, sizeof(name), "%s.cv3.bn.weight", prefix);
        float* bn_w = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv3.bn.bias", prefix);
        float* bn_b = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv3.bn.running_mean", prefix);
        float* bn_mean = weights_loader_get(loader, name, shape, &num_dims);
        snprintf(name, sizeof(name), "%s.cv3.bn.running_var", prefix);
        float* bn_var = weights_loader_get(loader, name, shape, &num_dims);
        if (bn_w && bn_b && bn_mean && bn_var) {
            batchnorm2d_load_weights(&block->cv3_bn, bn_w, bn_b, bn_mean, bn_var);
        }
    }
    if (int8) {
        snprintf(name, sizeof(name), "%s.cv3.conv.weight", prefix);
        const int8_t* qptr; float scale_w; size_t numel;
        if (weights_loader_int8_get(int8, name, &qptr, &scale_w, &numel) == 0)
            conv2d_load_weights_int8(&block->cv3, qptr, numel, scale_w, fused_bias);
    }
    
    // Load bottlenecks
    for (int i = 0; i < block->n; i++) {
        snprintf(name, sizeof(name), "%s.m.%d", prefix, i);
        bottleneck_load_weights(&block->bottlenecks[i], loader, name);
        if (int8) {
            float* fb;
            snprintf(name, sizeof(name), "%s.m.%d.cv1.conv.weight", prefix, i);
            const int8_t* qptr; float scale_w; size_t numel;
            if (weights_loader_int8_get(int8, name, &qptr, &scale_w, &numel) == 0) {
                snprintf(name, sizeof(name), "%s.m.%d.cv1.conv.bias", prefix, i);
                fb = weights_loader_get(loader, name, shape, &num_dims);
                conv2d_load_weights_int8(&block->bottlenecks[i].conv1, qptr, numel, scale_w, fb);
            }
            snprintf(name, sizeof(name), "%s.m.%d.cv2.conv.weight", prefix, i);
            if (weights_loader_int8_get(int8, name, &qptr, &scale_w, &numel) == 0) {
                snprintf(name, sizeof(name), "%s.m.%d.cv2.conv.bias", prefix, i);
                fb = weights_loader_get(loader, name, shape, &num_dims);
                conv2d_load_weights_int8(&block->bottlenecks[i].conv2, qptr, numel, scale_w, fb);
            }
        }
    }
    
    return 0;
}
