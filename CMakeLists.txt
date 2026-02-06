cmake_minimum_required(VERSION 3.12)
project(YOLOv5_C VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build options
option(BUILD_TESTS "Build test programs" ON)
option(BUILD_TOOLS "Build C tools" ON)
# Fused 전용: 기본값 사용. 임베디드에서는 아래 경로를 오버라이드하여 ROM/플래시 경로 지정
option(YOLO_EMBEDDED "Embedded build: override default weights/meta paths at compile time" OFF)
if(YOLO_EMBEDDED)
    # 예: add_compile_definitions(YOLO_WEIGHTS_DEFAULT_DIR="/rom" YOLO_WEIGHTS_DEFAULT_FILE="weights.bin" YOLO_META_DEFAULT_FILE="model_meta.json")
    # 프로젝트에서 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DYOLO_WEIGHTS_DEFAULT_DIR=...") 등으로 설정
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/core
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ops
    ${CMAKE_CURRENT_SOURCE_DIR}/src/blocks
    ${CMAKE_CURRENT_SOURCE_DIR}/src/models
    ${CMAKE_CURRENT_SOURCE_DIR}/src/postprocess
    ${CMAKE_CURRENT_SOURCE_DIR}/src/validation
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/jsmn
)

# Source files
set(CORE_SOURCES
    src/core/tensor.c
    src/core/memory.c
    src/core/weights_loader_int8.c
    third_party/jsmn/jsmn.c
)

set(OPS_SOURCES
    src/ops/conv2d.c
    src/ops/activation.c
    src/ops/batchnorm2d.c
    src/ops/pooling.c
    src/ops/upsample.c
    src/ops/concat.c
)
set(VALIDATION_SOURCES_LIB
    src/validation/quant_util.c
    src/validation/conv2d_int8.c
)

set(BLOCKS_SOURCES
    src/blocks/bottleneck.c
    src/blocks/c3.c
    src/blocks/sppf.c
)

set(MODELS_SOURCES
    src/models/yolov5n_graph.c
    src/models/yolov5n_build.c
    src/models/yolov5n_infer.c
)

set(POSTPROCESS_SOURCES
    src/postprocess/detect.c
    src/postprocess/nms.c
)

# Main library (quant_util, conv2d_int8 used by conv2d.c quant path)
add_library(yolov5_c STATIC
    ${CORE_SOURCES}
    ${OPS_SOURCES}
    ${BLOCKS_SOURCES}
    ${MODELS_SOURCES}
    ${POSTPROCESS_SOURCES}
    ${VALIDATION_SOURCES_LIB}
)

# Main executable
add_executable(yolov5_infer src/main.c)
target_link_libraries(yolov5_infer yolov5_c)
# Math library (only needed on Unix/Linux, not Windows)
if(NOT MSVC)
    target_link_libraries(yolov5_infer m)
endif()

# Layer0 quantization validation (calibration .bin required: python tools/export_calibration_inputs.py)
add_executable(layer0_validate src/validation/layer0_validate.c)
target_link_libraries(layer0_validate yolov5_c)
if(NOT MSVC)
    target_link_libraries(layer0_validate m)
endif()

# Detect head quantization validation (float vs int8 for 1x1 Conv)
add_executable(detect_head_validate src/validation/detect_head_validate.c)
target_link_libraries(detect_head_validate yolov5_c)
if(NOT MSVC)
    target_link_libraries(detect_head_validate m)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(/W4)
    # _CRT_SECURE_NO_WARNINGS is already defined in src/core/common.h
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    add_compile_options(-O2 -march=native)  # Optimization
endif()
