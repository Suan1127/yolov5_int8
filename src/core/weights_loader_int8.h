#ifndef WEIGHTS_LOADER_INT8_H
#define WEIGHTS_LOADER_INT8_H

#include <stdint.h>
#include <stddef.h>

/**
 * INT8 conv weights loader (weights_int8.bin + scales_int8.json).
 * Export 시점에 int8로 저장해 두면 C에서 float weight / weight minmax 없이 사용.
 */
typedef struct weights_loader_int8_t weights_loader_int8_t;

/**
 * Create loader from weights_int8.bin and scales_int8.json (같은 디렉터리 기준).
 * weights_dir 에서 weights_int8.bin, scales_int8.json 찾음.
 * @return NULL if files not found (optional; float path fallback 가능)
 */
weights_loader_int8_t* weights_loader_int8_create(const char* weights_dir);

void weights_loader_int8_free(weights_loader_int8_t* loader);

/**
 * Get int8 weight pointer and scale_w for a key (e.g. "model.0.conv.weight").
 * @param out_ptr  output pointer to int8 data (into loader buffer; do not free)
 * @param out_scale_w  output scale_w (symmetric)
 * @param out_numel  output number of elements
 * @return 0 if found, -1 if not
 */
int weights_loader_int8_get(weights_loader_int8_t* loader, const char* name,
                            const int8_t** out_ptr, float* out_scale_w, size_t* out_numel);

#endif /* WEIGHTS_LOADER_INT8_H */
