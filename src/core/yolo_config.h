#ifndef YOLO_CONFIG_H
#define YOLO_CONFIG_H

/**
 * YOLO 런타임 기본 설정 (int8 전용, 임베디드 고려)
 *
 * - 기본값: int8 가중치 사용 (weights_int8.bin + scales_int8.json + bias.bin)
 * - 임베디드: CMake 또는 컴파일 시 아래 매크로를 오버라이드하여
 *   ROM/플래시 경로·파일명으로 변경 가능
 *
 * 예 (CMake):
 *   add_compile_definitions(
 *     YOLO_WEIGHTS_DEFAULT_DIR="/rom/weights"
 *     YOLO_WEIGHTS_DEFAULT_FILE="weights_int8.bin"
 *     YOLO_META_DEFAULT_FILE="model_meta_fused.json"
 *   )
 */

#ifndef YOLO_WEIGHTS_DEFAULT_DIR
#define YOLO_WEIGHTS_DEFAULT_DIR "weights/yolov5n"
#endif

#ifndef YOLO_WEIGHTS_DEFAULT_FILE
#define YOLO_WEIGHTS_DEFAULT_FILE "weights_int8.bin"
#endif

#ifndef YOLO_META_DEFAULT_FILE
#define YOLO_META_DEFAULT_FILE "model_meta_fused.json"
#endif

#endif /* YOLO_CONFIG_H */
