#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifdef _WIN32
#include <windows.h>
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif
#include "core/tensor.h"
#include "core/yolo_config.h"
#include "models/yolov5n_build.h"
#include "models/yolov5n_infer.h"
#include "postprocess/detect.h"
#include "postprocess/nms.h"

static void print_tensor_stats_p(const char* name, const tensor_t* t) {
    if (!t || !t->data) return;
    size_t count = tensor_size(t);
    const float* data = t->data;
    float min_val = data[0], max_val = data[0];
    double sum = 0.0;
    for (size_t i = 0; i < count; i++) {
        float val = data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    double mean = sum / (double)count;
    printf("  %s: shape=(%d,%d,%d,%d) min=%.4f max=%.4f mean=%.4f\n",
           name, t->n, t->c, t->h, t->w, min_val, max_val, mean);
}

/* 레이어별 통계 콜백 (UART/임베디드 검증용) */
static void layer_stats_callback(int layer_idx, const tensor_t* t) {
    char label[24];
    snprintf(label, sizeof(label), "Layer %d", layer_idx);
    print_tensor_stats_p(label, t);
}

// Helper function to save detections to a file
static void save_detections_file(const char* filepath, const char* img_name,
                                  detection_t* dets, int32_t count, int32_t img_size) {
    // Create directory if it doesn't exist
    char dir_path[512];
    strncpy(dir_path, filepath, sizeof(dir_path) - 1);
    dir_path[sizeof(dir_path) - 1] = '\0';
    char* last_slash = strrchr(dir_path, '/');
    if (!last_slash) last_slash = strrchr(dir_path, '\\');
    if (last_slash) {
        *last_slash = '\0';
        #ifdef _WIN32
        // Convert / to \ for Windows
        for (int j = 0; dir_path[j]; j++) {
            if (dir_path[j] == '/') dir_path[j] = '\\';
        }
        #endif
        mkdir(dir_path, 0755);
    }
    
    FILE* fp = fopen(filepath, "w");
    if (fp) {
        fprintf(fp, "Image: %s\n", img_name);
        fprintf(fp, "Total detections: %d\n", count);
        fprintf(fp, "Format: class_id confidence x y w h (normalized 0-1)\n");
        fprintf(fp, "Format: class_id confidence x_pixel y_pixel w_pixel h_pixel\n");
        fprintf(fp, "\n");
        
        for (int i = 0; i < count; i++) {
            detection_t* det = &dets[i];
            fprintf(fp, "%d %.4f %.4f %.4f %.4f %.4f\n",
                    det->cls_id, det->conf,
                    det->x, det->y, det->w, det->h);
            fprintf(fp, "%d %.4f %.1f %.1f %.1f %.1f\n",
                    det->cls_id, det->conf,
                    det->x * img_size, det->y * img_size,
                    det->w * img_size, det->h * img_size);
        }
        fclose(fp);
        printf("Results saved to: %s\n", filepath);
    } else {
        fprintf(stderr, "Warning: Failed to save results to: %s\n", filepath);
    }
}

void print_usage(const char* prog_name) {
    printf("Usage: %s <image_name> [weights.bin] [model_meta.json]\n", prog_name);
    printf("\n");
    printf("Arguments:\n");
    printf("  image_name        Image name (without extension, e.g., 'bus')\n");
    printf("                    Input tensor will be loaded from: data/inputs/<image_name>.bin\n");
    printf("  weights.bin       Model weights file (default: " YOLO_WEIGHTS_DEFAULT_DIR "/" YOLO_WEIGHTS_DEFAULT_FILE ")\n");
    printf("  model_meta.json   Model metadata (default: " YOLO_WEIGHTS_DEFAULT_DIR "/" YOLO_META_DEFAULT_FILE ")\n");
    printf("\n");
    printf("Example:\n");
    printf("  %s bus\n", prog_name);
    printf("  %s bus weights/weights.bin weights/model_meta.json\n", prog_name);
    printf("\n");
    printf("Note: Run 'python tools/preprocess.py' first to generate input tensors from images in data/images/\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char* image_name = argv[1];
    // Default paths (will try multiple variations if not provided)
    const char* weights_path_arg = (argc >= 3) ? argv[2] : NULL;
    const char* model_meta_path_arg = (argc >= 4) ? argv[3] : NULL;
    
    // Construct input tensor path: try multiple locations
    // Try data/yolov5n/inputs/ first (YOLOv5n), then data/inputs/ as fallback
    char input_tensor_paths[4][512];
    snprintf(input_tensor_paths[0], sizeof(input_tensor_paths[0]), "data/yolov5n/inputs/%s.bin", image_name);
    snprintf(input_tensor_paths[1], sizeof(input_tensor_paths[1]), "../data/yolov5n/inputs/%s.bin", image_name);
    snprintf(input_tensor_paths[2], sizeof(input_tensor_paths[2]), "../../data/yolov5n/inputs/%s.bin", image_name);
    snprintf(input_tensor_paths[3], sizeof(input_tensor_paths[3]), "data/inputs/%s.bin", image_name);  // Fallback
    
    printf("=== YOLOv5 C Inference ===\n\n");
    printf("Image name: %s\n", image_name);
    printf("\n");
    
    // Load input tensor
    printf("Loading input tensor...\n");
    
    // Try multiple path variations (YOLOv5n first, then fallback)
    char paths[4][512];
    snprintf(paths[0], sizeof(paths[0]), "data/yolov5n/inputs/%s.bin", image_name);
    snprintf(paths[1], sizeof(paths[1]), "../data/yolov5n/inputs/%s.bin", image_name);
    snprintf(paths[2], sizeof(paths[2]), "../../data/yolov5n/inputs/%s.bin", image_name);
    snprintf(paths[3], sizeof(paths[3]), "data/inputs/%s.bin", image_name);  // Fallback
    
    const char* found_path = NULL;
    for (int i = 0; i < 4; i++) {
        FILE* test_fp = fopen(paths[i], "rb");
        if (test_fp) {
            fclose(test_fp);
            found_path = paths[i];
            break;
        }
    }
    
    if (!found_path) {
        fprintf(stderr, "Error: Cannot find input tensor file\n");
        fprintf(stderr, "Tried:\n");
        for (int i = 0; i < 4; i++) {
            fprintf(stderr, "  - %s\n", paths[i]);
        }
        #ifdef _WIN32
        char cwd[MAX_PATH];
        if (GetCurrentDirectoryA(MAX_PATH, cwd)) {
            fprintf(stderr, "Current directory: %s\n", cwd);
        }
        #else
        char* cwd = getcwd(NULL, 0);
        if (cwd) {
            fprintf(stderr, "Current directory: %s\n", cwd);
            free(cwd);
        }
        #endif
        fprintf(stderr, "\nHint: Run 'python tools/preprocess.py' to generate input tensors\n");
        fprintf(stderr, "Hint: Make sure you're running from build/Release or project root\n");
        return 1;
    }
    
    printf("Found input tensor: %s\n", found_path);
    tensor_t* input = tensor_load(found_path);
    if (!input) {
        fprintf(stderr, "Error: Failed to load input tensor from %s\n", found_path);
        fprintf(stderr, "Hint: The file exists but may be corrupted or in wrong format\n");
        return 1;
    }
    
    printf("Input shape: (%d, %d, %d, %d)\n", input->n, input->c, input->h, input->w);
    
    // Verify input shape (allow any size, but must be NCHW format)
    if (input->n != 1 || input->c != 3) {
        fprintf(stderr, "Error: Invalid input shape. Expected (1, 3, H, W), got (%d, %d, %d, %d)\n",
                input->n, input->c, input->h, input->w);
        tensor_free(input);
        return 1;
    }
    
    // Store input dimensions for later use
    int32_t input_h = input->h;
    int32_t input_w = input->w;
    int32_t input_size = input_h > input_w ? input_h : input_w;  // Use max dimension
    printf("Detected input size: %dx%d (using %d as reference)\n", input_h, input_w, input_size);
    
    // Find weights file
    printf("Finding weights file...\n");
    char weights_paths[3][512];
    if (weights_path_arg) {
        snprintf(weights_paths[0], sizeof(weights_paths[0]), "%s", weights_path_arg);
        snprintf(weights_paths[1], sizeof(weights_paths[1]), "../%s", weights_path_arg);
        snprintf(weights_paths[2], sizeof(weights_paths[2]), "../../%s", weights_path_arg);
    } else {
        /* Default: fused weights (단일 Conv+BN+SiLU, 임베디드 메모리/연산 최소화) */
        snprintf(weights_paths[0], sizeof(weights_paths[0]), "%s/%s", YOLO_WEIGHTS_DEFAULT_DIR, YOLO_WEIGHTS_DEFAULT_FILE);
        snprintf(weights_paths[1], sizeof(weights_paths[1]), "../%s/%s", YOLO_WEIGHTS_DEFAULT_DIR, YOLO_WEIGHTS_DEFAULT_FILE);
        snprintf(weights_paths[2], sizeof(weights_paths[2]), "../../%s/%s", YOLO_WEIGHTS_DEFAULT_DIR, YOLO_WEIGHTS_DEFAULT_FILE);
    }
    
    const char* found_weights_path = NULL;
    for (int i = 0; i < 3; i++) {
        FILE* test_fp = fopen(weights_paths[i], "rb");
        if (test_fp) {
            fclose(test_fp);
            found_weights_path = weights_paths[i];
            break;
        }
    }
    
    if (!found_weights_path) {
        fprintf(stderr, "Error: Cannot find weights file\n");
        fprintf(stderr, "Tried:\n");
        for (int i = 0; i < 3; i++) {
            fprintf(stderr, "  - %s\n", weights_paths[i]);
        }
        tensor_free(input);
        return 1;
    }
    printf("Found weights: %s\n", found_weights_path);
    
    // Find model meta file
    printf("Finding model meta file...\n");
    char meta_paths[3][512];
    if (model_meta_path_arg) {
        snprintf(meta_paths[0], sizeof(meta_paths[0]), "%s", model_meta_path_arg);
        snprintf(meta_paths[1], sizeof(meta_paths[1]), "../%s", model_meta_path_arg);
        snprintf(meta_paths[2], sizeof(meta_paths[2]), "../../%s", model_meta_path_arg);
    } else {
        /* Default: fused 메타 (weights_fused.bin과 쌍) */
        snprintf(meta_paths[0], sizeof(meta_paths[0]), "%s/%s", YOLO_WEIGHTS_DEFAULT_DIR, YOLO_META_DEFAULT_FILE);
        snprintf(meta_paths[1], sizeof(meta_paths[1]), "../%s/%s", YOLO_WEIGHTS_DEFAULT_DIR, YOLO_META_DEFAULT_FILE);
        snprintf(meta_paths[2], sizeof(meta_paths[2]), "../../%s/%s", YOLO_WEIGHTS_DEFAULT_DIR, YOLO_META_DEFAULT_FILE);
    }
    
    const char* found_meta_path = NULL;
    for (int i = 0; i < 3; i++) {
        FILE* test_fp = fopen(meta_paths[i], "r");
        if (test_fp) {
            fclose(test_fp);
            found_meta_path = meta_paths[i];
            break;
        }
    }
    
    if (!found_meta_path) {
        printf("Warning: Model meta file not found, using defaults\n");
        printf("Tried:\n");
        for (int i = 0; i < 3; i++) {
            printf("  - %s\n", meta_paths[i]);
        }
    } else {
        printf("Found model meta: %s\n", found_meta_path);
    }
    printf("\n");
    
    // Build model
    printf("Building YOLOv5n model...\n");
    yolov5n_model_t* model = yolov5n_build(found_weights_path, found_meta_path);
    if (!model) {
        fprintf(stderr, "Error: Failed to build model\n");
        fprintf(stderr, "Check if weights file is valid: %s\n", found_weights_path);
        tensor_free(input);
        return 1;
    }
    printf("Model built successfully\n");
    
    // Allocate output tensors
    printf("\nAllocating output tensors...\n");
    tensor_t* outputs[3];
    outputs[0] = tensor_create(1, 64, 80, 80);   // P3 for YOLOv5n
    outputs[1] = tensor_create(1, 128, 40, 40);   // P4 for YOLOv5n
    outputs[2] = tensor_create(1, 256, 20, 20);   // P5 for YOLOv5n
    
    if (!outputs[0] || !outputs[1] || !outputs[2]) {
        fprintf(stderr, "Error: Failed to allocate output tensors\n");
        yolov5n_free(model);
        tensor_free(input);
        return 1;
    }
    
    // Set output directory for saving intermediate layer tensors (for validation)
    // Only save intermediate tensors to testdata/c/, not final detection results
    // Final detection results are saved to data/yolov5n/outputs/ in the detection section below
    const char* output_dir = NULL;
    
    // Try multiple paths for testdata_n/c directory (YOLOv5n)
    const char* test_paths[] = {
        "testdata_n/c",
        "../testdata_n/c",
        "../../testdata_n/c"
    };
    
    #ifdef _WIN32
    for (int i = 0; i < 3; i++) {
        char win_path[512];
        snprintf(win_path, sizeof(win_path), "%s", test_paths[i]);
        // Convert / to \ for Windows
        for (int j = 0; win_path[j]; j++) {
            if (win_path[j] == '/') win_path[j] = '\\';
        }
        DWORD attrs = GetFileAttributesA(win_path);
        if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY)) {
            output_dir = test_paths[i];
            printf("Found testdata_n/c directory: %s\n", output_dir);
            break;
        }
    }
    #else
    for (int i = 0; i < 3; i++) {
        if (access(test_paths[i], F_OK) == 0) {
            output_dir = test_paths[i];
            printf("Found testdata_n/c directory: %s\n", output_dir);
            break;
        }
    }
    #endif
    
    // Always try to set output directory (will create if doesn't exist)
    // Default to testdata_n/c for YOLOv5n
    if (!output_dir) {
        output_dir = "testdata_n/c";
        printf("Setting default output directory: %s (will be created if needed)\n", output_dir);
    }
    
    int set_dir_ret = yolov5n_set_output_dir(model, output_dir);
    if (set_dir_ret == 0) {
        printf("Intermediate layer output directory set to: %s\n", output_dir);
        printf("(Final detection results will be saved to data/yolov5n/outputs/)\n");
    } else {
        fprintf(stderr, "Warning: Failed to set output directory: %s\n", output_dir);
    }
    
    /* 레이어별 한 줄 출력(shape, min/max/mean, ms)은 yolov5n_forward 내부에서 출력 */
    /* yolov5n_set_layer_stats_callback(layer_stats_callback); */
    
    // Forward pass
    printf("\nRunning forward pass...\n");
    print_tensor_stats_p("input", input);
    fflush(stdout);
    
    int ret = yolov5n_forward(model, input, outputs);
    if (ret != 0) {
        fprintf(stderr, "\nError: Forward pass failed (return code: %d)\n", ret);
        fprintf(stderr, "Possible causes:\n");
        fprintf(stderr, "  - Memory allocation failure\n");
        fprintf(stderr, "  - Invalid layer configuration\n");
        fprintf(stderr, "  - Missing or corrupted weights\n");
        tensor_free(outputs[0]);
        tensor_free(outputs[1]);
        tensor_free(outputs[2]);
        yolov5n_free(model);
        tensor_free(input);
        return 1;
    }
    printf("Forward pass completed successfully\n");
    printf("P3 output: (%d, %d, %d, %d)\n", outputs[0]->n, outputs[0]->c, outputs[0]->h, outputs[0]->w);
    printf("P4 output: (%d, %d, %d, %d)\n", outputs[1]->n, outputs[1]->c, outputs[1]->h, outputs[1]->w);
    printf("P5 output: (%d, %d, %d, %d)\n", outputs[2]->n, outputs[2]->c, outputs[2]->h, outputs[2]->w);
    /* UART 검증용: platform_print_tensor_stats와 동일 형식 (임베디드 출력과 비교) */
    print_tensor_stats_p("P3", outputs[0]);
    print_tensor_stats_p("P4", outputs[1]);
    print_tensor_stats_p("P5", outputs[2]);

    // Get P3, P4, P5 features for Detect head
    tensor_t* p3_feature = yolov5n_get_saved_feature(model, 17);
    tensor_t* p4_feature = yolov5n_get_saved_feature(model, 20);
    tensor_t* p5_feature = yolov5n_get_saved_feature(model, 23);
    
    if (!p3_feature || !p4_feature || !p5_feature) {
        fprintf(stderr, "Error: Failed to get detect features\n");
        tensor_free(outputs[0]);
        tensor_free(outputs[1]);
        tensor_free(outputs[2]);
        yolov5n_free(model);
        tensor_free(input);
        return 1;
    }
    
    // Detect head
    printf("\nRunning Detect head...\n");
    detect_params_t detect_params;
    // Use actual input size for detection (use max dimension as reference)
    // input_size is already declared above
    detect_init_params(&detect_params, 80, input_size);
    printf("Using input size: %d for detection (from input %dx%d)\n", input_size, input_h, input_w);
    
    detect_output_t detect_output;
    ret = detect_forward(model, p3_feature, p4_feature, p5_feature, &detect_output, &detect_params);
    if (ret != 0) {
        fprintf(stderr, "Error: Detect forward failed\n");
        tensor_free(outputs[0]);
        tensor_free(outputs[1]);
        tensor_free(outputs[2]);
        yolov5n_free(model);
        tensor_free(input);
        return 1;
    }
    printf("Detect head completed\n");
    
    // Save detect head outputs to testdata_n/c (for comparison with Python)
    // Python saves as output_1_0.bin (P3), output_1_1.bin (P4), output_1_2.bin (P5)
    // because output[1] is the list of 3 tensors [P3, P4, P5]
    if (output_dir && strlen(output_dir) > 0) {
        char filepath[512];
        // Save P3 detect output as output_1_0.bin (to match Python)
        snprintf(filepath, sizeof(filepath), "%s/output_1_0.bin", output_dir);
        tensor_dump(detect_output.p3_output, filepath);
        
        // Save P4 detect output as output_1_1.bin (to match Python)
        snprintf(filepath, sizeof(filepath), "%s/output_1_1.bin", output_dir);
        tensor_dump(detect_output.p4_output, filepath);
        
        // Save P5 detect output as output_1_2.bin (to match Python)
        snprintf(filepath, sizeof(filepath), "%s/output_1_2.bin", output_dir);
        tensor_dump(detect_output.p5_output, filepath);
    }
    
    // Decode detections
    printf("\nDecoding detections...\n");
    detection_t* detections = NULL;
    int32_t num_detections = 0;
    float conf_threshold = 0.25f;
    
    ret = detect_decode(&detect_output, &detect_params, &detections, &num_detections, conf_threshold);
    if (ret != 0) {
        fprintf(stderr, "Error: Decode failed\n");
        detect_free_output(&detect_output);
        tensor_free(outputs[0]);
        tensor_free(outputs[1]);
        tensor_free(outputs[2]);
        yolov5n_free(model);
        tensor_free(input);
        return 1;
    }
    printf("Found %d detections (confidence > %.2f)\n", num_detections, conf_threshold);
    
    // Sort all detections by confidence (descending) before limiting
    // This matches Python: x = x[x[:, 4].argsort(descending=True)[:max_nms]]
    for (int i = 0; i < num_detections - 1; i++) {
        for (int j = i + 1; j < num_detections; j++) {
            if (detections[i].conf < detections[j].conf) {
                detection_t temp = detections[i];
                detections[i] = detections[j];
                detections[j] = temp;
            }
        }
    }
    
    // Limit to top 30000 detections before NMS (matching Python max_nms=30000)
    int32_t max_nms = 30000;
    if (num_detections > max_nms) {
        num_detections = max_nms;
        printf("Limited to top %d detections before NMS (matching Python max_nms)\n", max_nms);
    }
    
    // NMS
    printf("\nRunning NMS...\n");
    detection_t* nms_detections = NULL;
    int32_t nms_count = 0;
    float iou_threshold = 0.45f;
    int32_t max_detections = 300;  // Match Python default max_det=300
    
    ret = nms(detections, num_detections, &nms_detections, &nms_count, iou_threshold, max_detections);
    if (ret != 0) {
        fprintf(stderr, "Error: NMS failed\n");
        detect_free_detections(detections, num_detections);
        detect_free_output(&detect_output);
        tensor_free(outputs[0]);
        tensor_free(outputs[1]);
        tensor_free(outputs[2]);
        yolov5n_free(model);
        tensor_free(input);
        return 1;
    }
    printf("After NMS: %d detections\n", nms_count);
    
    // Print results (one compact block per detection)
    printf("\n=== Detection Results ===\n");
    printf("Total: %d\n\n", nms_count);
    int print_count = nms_count < 10 ? nms_count : 10;
    for (int i = 0; i < print_count; i++) {
        detection_t* det = &nms_detections[i];
        float px = det->x * input_size;
        float py = det->y * input_size;
        float pw = det->w * input_size;
        float ph = det->h * input_size;
        printf("[%d] class=%d conf=%.4f  bbox=(%.4f,%.4f,%.4f,%.4f)  px=(%.1f,%.1f,%.1f,%.1f)\n",
               i + 1, det->cls_id, det->conf, det->x, det->y, det->w, det->h, px, py, pw, ph);
    }
    if (nms_count > 0) {
        printf("\n");
    }
    
    // Save results to file
    printf("\nSaving results...\n");
    
    // Save to data/yolov5n/outputs/ (original location)
    char output_paths[3][512];
    snprintf(output_paths[0], sizeof(output_paths[0]), "data/yolov5n/outputs/%s_detections.txt", image_name);
    snprintf(output_paths[1], sizeof(output_paths[1]), "../data/yolov5n/outputs/%s_detections.txt", image_name);
    snprintf(output_paths[2], sizeof(output_paths[2]), "../../data/yolov5n/outputs/%s_detections.txt", image_name);
    
    const char* saved_path = NULL;
    for (int i = 0; i < 3; i++) {
        FILE* test_fp = fopen(output_paths[i], "w");
        if (test_fp) {
            fclose(test_fp);
            save_detections_file(output_paths[i], image_name, nms_detections, nms_count, input_size);
            saved_path = output_paths[i];
            break;
        }
    }
    
    if (!saved_path) {
        fprintf(stderr, "Warning: Failed to save results to data/yolov5n/outputs/\n");
        fprintf(stderr, "Tried paths:\n");
        for (int i = 0; i < 3; i++) {
            fprintf(stderr, "  - %s\n", output_paths[i]);
        }
    }
    
    // Also save to testdata_n/c/ (for comparison with Python)
    if (output_dir && strlen(output_dir) > 0) {
        char testdata_path[512];
        snprintf(testdata_path, sizeof(testdata_path), "%s/%s_detections.txt", output_dir, image_name);
        save_detections_file(testdata_path, image_name, nms_detections, nms_count, input_size);
    }
    
    // Cleanup
    detect_free_detections(nms_detections, nms_count);
    detect_free_detections(detections, num_detections);
    detect_free_output(&detect_output);
    tensor_free(outputs[0]);
    tensor_free(outputs[1]);
    tensor_free(outputs[2]);
    yolov5n_free(model);
    tensor_free(input);
    
    printf("\n=== Inference completed successfully ===\n");
    return 0;
}
