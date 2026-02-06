#include "detect.h"
#include "../ops/conv2d.h"
#include "../ops/batchnorm2d.h"
#include "../models/yolov5n_build.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void detect_init_params(detect_params_t* params, int32_t num_classes, int32_t input_size) {
    if (!params) return;
    
    params->num_classes = num_classes;
    params->num_anchors = 3;
    params->input_size = input_size;
    
    // Default anchors for YOLOv5n (from model_meta.json)
    // P3: [10, 13, 16, 30, 33, 23]
    params->anchors[0][0] = 10.0f;   // anchor 0: w, h
    params->anchors[0][1] = 13.0f;
    params->anchors[0][2] = 16.0f;   // anchor 1: w, h
    params->anchors[0][3] = 30.0f;
    params->anchors[0][4] = 33.0f;   // anchor 2: w, h
    params->anchors[0][5] = 23.0f;
    
    // P4: [30, 61, 62, 45, 59, 119]
    params->anchors[1][0] = 30.0f;
    params->anchors[1][1] = 61.0f;
    params->anchors[1][2] = 62.0f;
    params->anchors[1][3] = 45.0f;
    params->anchors[1][4] = 59.0f;
    params->anchors[1][5] = 119.0f;
    
    // P5: [116, 90, 156, 198, 373, 326]
    params->anchors[2][0] = 116.0f;
    params->anchors[2][1] = 90.0f;
    params->anchors[2][2] = 156.0f;
    params->anchors[2][3] = 198.0f;
    params->anchors[2][4] = 373.0f;
    params->anchors[2][5] = 326.0f;
}

int detect_forward(void* model_ptr, const tensor_t* p3_feature, const tensor_t* p4_feature, const tensor_t* p5_feature,
                   detect_output_t* output, const detect_params_t* params) {
    if (!model_ptr || !p3_feature || !p4_feature || !p5_feature || !output || !params) return -1;
    
    yolov5n_model_t* model = (yolov5n_model_t*)model_ptr;
    
    int32_t num_outputs = 3 * (params->num_classes + 5);  // 3 anchors * (4 bbox + 1 obj + 80 cls) = 255
    
    // Get actual feature map sizes
    int32_t p3_h = p3_feature->h;
    int32_t p3_w = p3_feature->w;
    int32_t p4_h = p4_feature->h;
    int32_t p4_w = p4_feature->w;
    int32_t p5_h = p5_feature->h;
    int32_t p5_w = p5_feature->w;
    
    // P3: (1, 64, H, W) -> (1, 255, H, W) for YOLOv5n
    output->p3_output = tensor_create(1, num_outputs, p3_h, p3_w);
    if (!output->p3_output) return -1;
    
    if (model->detect_convs[0].conv.q_weight && model->detect_convs[0].conv.scale_w > 0.f) {
        if (conv2d_quant_forward(&model->detect_convs[0].conv, p3_feature, output->p3_output) != 0) {
            fprintf(stderr, "Error: Detect head P3 conv (int8) forward failed\n");
            tensor_free(output->p3_output);
            return -1;
        }
    } else if (conv2d_forward(&model->detect_convs[0].conv, p3_feature, output->p3_output) != 0) {
        fprintf(stderr, "Error: Detect head P3 conv forward failed\n");
        tensor_free(output->p3_output);
        return -1;
    }
    
    // P4: (1, 128, H, W) -> (1, 255, H, W) for YOLOv5n
    output->p4_output = tensor_create(1, num_outputs, p4_h, p4_w);
    if (!output->p4_output) {
        tensor_free(output->p3_output);
        return -1;
    }
    
    if (model->detect_convs[1].conv.q_weight && model->detect_convs[1].conv.scale_w > 0.f) {
        if (conv2d_quant_forward(&model->detect_convs[1].conv, p4_feature, output->p4_output) != 0) {
            fprintf(stderr, "Error: Detect head P4 conv (int8) forward failed\n");
            tensor_free(output->p3_output);
            tensor_free(output->p4_output);
            return -1;
        }
    } else if (conv2d_forward(&model->detect_convs[1].conv, p4_feature, output->p4_output) != 0) {
        fprintf(stderr, "Error: Detect head P4 conv forward failed\n");
        tensor_free(output->p3_output);
        tensor_free(output->p4_output);
        return -1;
    }
    
    // P5: (1, 256, H, W) -> (1, 255, H, W) for YOLOv5n
    output->p5_output = tensor_create(1, num_outputs, p5_h, p5_w);
    if (!output->p5_output) {
        tensor_free(output->p3_output);
        tensor_free(output->p4_output);
        return -1;
    }
    
    if (model->detect_convs[2].conv.q_weight && model->detect_convs[2].conv.scale_w > 0.f) {
        if (conv2d_quant_forward(&model->detect_convs[2].conv, p5_feature, output->p5_output) != 0) {
            fprintf(stderr, "Error: Detect head P5 conv (int8) forward failed\n");
            tensor_free(output->p3_output);
            tensor_free(output->p4_output);
            tensor_free(output->p5_output);
            return -1;
        }
    } else if (conv2d_forward(&model->detect_convs[2].conv, p5_feature, output->p5_output) != 0) {
        fprintf(stderr, "Error: Detect head P5 conv forward failed\n");
        tensor_free(output->p3_output);
        tensor_free(output->p4_output);
        tensor_free(output->p5_output);
        return -1;
    }
    
    return 0;
}

int detect_decode(const detect_output_t* output, const detect_params_t* params,
                  detection_t** detections, int32_t* num_detections, float conf_threshold) {
    if (!output || !params || !detections || !num_detections) return -1;
    
    *detections = NULL;
    *num_detections = 0;
    
    // Allocate temporary array for all detections
    detection_t* temp_detections = (detection_t*)calloc(25200, sizeof(detection_t));  // Max: 19200 + 4800 + 1200
    if (!temp_detections) return -1;
    
    int32_t count = 0;
    
    // Decode each scale
    for (int scale = 0; scale < 3; scale++) {
        const tensor_t* feature = NULL;
        int32_t grid_h = 0, grid_w = 0;
        float stride = 0.0f;
        const float* anchors = NULL;
        
        switch (scale) {
            case 0:  // P3
                feature = output->p3_output;
                if (!feature) break;
                grid_h = feature->h;
                grid_w = feature->w;
                stride = (float)params->input_size / (float)grid_h;  // Calculate stride dynamically
                anchors = params->anchors[0];
                break;
            case 1:  // P4
                feature = output->p4_output;
                if (!feature) break;
                grid_h = feature->h;
                grid_w = feature->w;
                stride = (float)params->input_size / (float)grid_h;
                anchors = params->anchors[1];
                break;
            case 2:  // P5
                feature = output->p5_output;
                if (!feature) break;
                grid_h = feature->h;
                grid_w = feature->w;
                stride = (float)params->input_size / (float)grid_h;
                anchors = params->anchors[2];
                break;
        }
        
        if (!feature) continue;
        
        // Process each grid cell
        for (int32_t y = 0; y < grid_h; y++) {
            for (int32_t x = 0; x < grid_w; x++) {
                // Process each anchor
                for (int a = 0; a < 3; a++) {
                    // Get output for this anchor
                    // Output format: (batch, 255, grid_h, grid_w) where 255 = 3 * 85
                    // Layout: [anchor0_85, anchor1_85, anchor2_85]
                    // Each anchor has 85 values: [x, y, w, h, obj_conf, cls_0, ..., cls_79]
                    int32_t anchor_start = a * 85;  // Start channel for this anchor
                    int32_t spatial_idx = y * grid_w + x;  // Spatial position
                    int32_t base_idx = (anchor_start * grid_h * grid_w) + spatial_idx;
                    
                    // Get bbox (x, y, w, h) - these are raw logits
                    float bx = feature->data[base_idx];
                    float by = feature->data[base_idx + 1 * (grid_h * grid_w)];
                    float bw = feature->data[base_idx + 2 * (grid_h * grid_w)];
                    float bh = feature->data[base_idx + 3 * (grid_h * grid_w)];
                    
                    // Get object confidence (sigmoid)
                    float obj_conf = 1.0f / (1.0f + expf(-feature->data[base_idx + 4 * (grid_h * grid_w)]));
                    
                    // Get class confidences (sigmoid for each)
                    float max_cls_conf = 0.0f;
                    int32_t max_cls_id = 0;
                    for (int c = 0; c < params->num_classes; c++) {
                        float cls_logit = feature->data[base_idx + (5 + c) * (grid_h * grid_w)];
                        float cls_conf = 1.0f / (1.0f + expf(-cls_logit));
                        temp_detections[count].cls_conf[c] = cls_conf;
                        if (cls_conf > max_cls_conf) {
                            max_cls_conf = cls_conf;
                            max_cls_id = c;
                        }
                    }
                    
                    // Calculate final confidence
                    float conf = obj_conf * max_cls_conf;
                    
                    // Apply confidence threshold
                    if (conf < conf_threshold) continue;
                    
                    // Decode bbox coordinates (YOLOv5 new format)
                    // Python: xy = (xy * 2 + grid) * stride, wh = (wh * 2) ** 2 * anchor_grid
                    // where grid = (x, y) - 0.5, anchor_grid = (anchors / stride) * stride = anchors
                    // Note: In Python, anchors are normalized by stride first, then multiplied by stride for anchor_grid
                    // In C, anchors are stored in pixel units, so anchor_grid = anchors (no normalization needed)
                    float anchor_w = anchors[a * 2];
                    float anchor_h = anchors[a * 2 + 1];
                    
                    // Apply sigmoid to x, y, w, h
                    float tx = 1.0f / (1.0f + expf(-bx));
                    float ty = 1.0f / (1.0f + expf(-by));
                    float tw_sigmoid = 1.0f / (1.0f + expf(-bw));
                    float th_sigmoid = 1.0f / (1.0f + expf(-bh));
                    
                    // Decode: xy = (sigmoid(xy) * 2 + grid) * stride
                    // grid = (x, y) - 0.5 (YOLOv5 format)
                    float grid_x = (float)x - 0.5f;
                    float grid_y = (float)y - 0.5f;
                    float decoded_x = (tx * 2.0f + grid_x) * stride;
                    float decoded_y = (ty * 2.0f + grid_y) * stride;
                    
                    // Decode: wh = (sigmoid(wh) * 2) ** 2 * anchor_grid
                    // anchor_grid = anchors (in pixel units, matching Python's normalized anchors * stride)
                    // In Python: anchor_grid = (anchors / stride) * stride = anchors (pixel units)
                    float decoded_w = (tw_sigmoid * 2.0f) * (tw_sigmoid * 2.0f) * anchor_w;
                    float decoded_h = (th_sigmoid * 2.0f) * (th_sigmoid * 2.0f) * anchor_h;
                    
                    // Normalize to 0-1 range
                    temp_detections[count].x = decoded_x / params->input_size;
                    temp_detections[count].y = decoded_y / params->input_size;
                    temp_detections[count].w = decoded_w / params->input_size;
                    temp_detections[count].h = decoded_h / params->input_size;
                    
                    temp_detections[count].conf = conf;
                    temp_detections[count].cls_id = max_cls_id;
                    
                    count++;
                    if (count >= 25200) goto done;  // Safety limit
                }
            }
        }
    }
    
done:
    *num_detections = count;
    if (count > 0) {
        *detections = (detection_t*)realloc(temp_detections, count * sizeof(detection_t));
        if (!*detections) {
            free(temp_detections);
            return -1;
        }
    } else {
        free(temp_detections);
    }
    
    return 0;
}

void detect_free_output(detect_output_t* output) {
    if (output) {
        if (output->p3_output) tensor_free(output->p3_output);
        if (output->p4_output) tensor_free(output->p4_output);
        if (output->p5_output) tensor_free(output->p5_output);
        memset(output, 0, sizeof(detect_output_t));
    }
}

void detect_free_detections(detection_t* detections, int32_t num_detections) {
    if (detections) {
        free(detections);
    }
}
