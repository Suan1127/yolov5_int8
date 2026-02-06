# YOLOv5 C 포팅 프로젝트

YOLOv5n 모델을 Python/PyTorch에서 순수 C로 포팅하여 임베디드/엣지 디바이스에서 실행 가능하도록 구현한 프로젝트입니다.

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [아키텍처 개요](#아키텍처-개요)
3. [프로젝트 구조](#프로젝트-구조)
4. [핵심 컴포넌트](#핵심-컴포넌트)
5. [전체 워크플로우](#전체-워크플로우)
6. [빌드 및 실행](#빌드-및-실행)
7. [검증 및 디버깅](#검증-및-디버깅)
8. [기술적 세부사항](#기술적-세부사항)
9. [문서 및 참고 자료](#문서-및-참고-자료)

---

## 프로젝트 개요

### 목적
- YOLOv5n 모델을 PyTorch에서 순수 C로 완전히 포팅
- PyTorch 구현과 수치적으로 동일한 결과 보장 (레이어별 검증 완료)
- 임베디드/엣지 디바이스에서 실행 가능한 경량 구현

### 주요 특징
- ✅ **완전한 YOLOv5n 구현**: Backbone (10 layers) + Head (14 layers) + Detect (1 layer) = 총 25개 레이어
- ✅ **동적 입력 크기 지원**: 640×640 외 다양한 입력 크기 처리
- ✅ **Cross-platform**: Windows/MSVC 및 Linux/GCC 지원
- ✅ **메모리 효율적**: Arena allocator 및 ping-pong 버퍼 사용
- ✅ **End-to-end 파이프라인**: 이미지 입력부터 검출 결과 출력까지
- ✅ **정확도 검증 완료**: PyTorch와 레이어별 비교 검증 (Layer 0-23)

---

## 아키텍처 개요

### 레이어드 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  (main.c: 이미지 입력 → 추론 → 검출 결과 출력)              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Layer                               │
│  - yolov5n_build.c: 모델 초기화 및 가중치 로드              │
│  - yolov5n_infer.c: Forward pass 파이프라인                 │
│  - yolov5n_graph.c: 레이어 그래프 정의                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Block Layer                               │
│  - c3.c: Cross-stage partial bottleneck                     │
│  - bottleneck.c: C3 내부 구성 요소                          │
│  - sppf.c: Spatial Pyramid Pooling Fast                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Operation Layer                           │
│  - conv2d.c: 2D Convolution                                 │
│  - batchnorm2d.c: Batch Normalization                       │
│  - activation.c: SiLU activation                             │
│  - pooling.c: MaxPool2D                                     │
│  - upsample.c: Nearest-neighbor upsampling                  │
│  - concat.c: Channel-wise concatenation                     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Core Layer                                │
│  - tensor.c: NCHW 텐서 관리                                 │
│  - memory.c: Arena allocator                                │
│  - weights_loader.c: 가중치 파일 로드                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Post-processing Layer                     │
│  - detect.c: Detect head decode                             │
│  - nms.c: Non-Maximum Suppression                           │
└─────────────────────────────────────────────────────────────┘
```

### 모델 구조 (25개 레이어)

**Backbone (Layers 0-9):**
- Layer 0: Conv(3→16, 6×6, s=2) → (1,16,320,320)
- Layer 1: Conv(16→32, 3×3, s=2) → (1,32,160,160)
- Layer 2: C3(32→32, n=1) → (1,32,160,160)
- Layer 3: Conv(32→64, 3×3, s=2) → (1,64,80,80)
- Layer 4: C3(64→64, n=2) → (1,64,80,80)
- Layer 5: Conv(64→128, 3×3, s=2) → (1,128,40,40)
- Layer 6: C3(128→128, n=3) → (1,128,40,40)
- Layer 7: Conv(128→256, 3×3, s=2) → (1,256,20,20)
- Layer 8: C3(256→256, n=1) → (1,256,20,20)
- Layer 9: SPPF(256→256) → (1,256,20,20)

**Head (Layers 10-23):**
- Layer 10: Conv(256→128, 1×1) → (1,128,20,20)
- Layer 11: Upsample(×2) → (1,128,40,40)
- Layer 12: Concat([Layer 6, Layer 11]) → (1,256,40,40)
- Layer 13: C3(256→128, n=1) → (1,128,40,40)
- Layer 14: Conv(128→64, 1×1) → (1,64,40,40)
- Layer 15: Upsample(×2) → (1,64,80,80)
- Layer 16: Concat([Layer 4, Layer 15]) → (1,128,80,80)
- Layer 17: C3(128→64, n=1) → (1,64,80,80) → **P3**
- Layer 18: Conv(64→64, 3×3, s=2) → (1,64,40,40)
- Layer 19: Concat([Layer 13, Layer 18]) → (1,128,40,40)
- Layer 20: C3(128→128, n=1) → (1,128,40,40) → **P4**
- Layer 21: Conv(128→128, 3×3, s=2) → (1,128,20,20)
- Layer 22: Concat([Layer 10, Layer 21]) → (1,256,20,20)
- Layer 23: C3(256→256, n=1) → (1,256,20,20) → **P5**

**Detect (Layer 24):**
- P3, P4, P5를 입력으로 받아 최종 검출 결과 출력

---

## 프로젝트 구조

```
YOLO_c/
├── src/                          # C 소스 코드
│   ├── main.c                    # 메인 진입점 (이미지 입력 → 추론 → 검출)
│   │
│   ├── core/                     # 핵심 인프라
│   │   ├── tensor.h/c           # NCHW 텐서 구조체 및 유틸리티
│   │   │                         # - tensor_create, tensor_free
│   │   │                         # - tensor_load, tensor_save
│   │   │                         # - tensor_dump (디버깅용)
│   │   ├── memory.h/c            # Arena allocator
│   │   │                         # - arena_create, arena_alloc
│   │   │                         # - 16-byte 정렬 (SIMD 최적화)
│   │   ├── weights_loader.h/c    # 가중치 파일 로더
│   │   │                         # - weights.bin 바이너리 로드
│   │   │                         # - weights_map.json 파싱
│   │   └── common.h              # 공통 매크로 (SNPRINTF 등)
│   │
│   ├── ops/                      # Primitive 연산
│   │   ├── conv2d.h/c           # 2D Convolution
│   │   │                         # - 1×1, 3×3, 6×6 커널 지원
│   │   │                         # - Padding, stride, dilation
│   │   │                         # - Fused BN 지원 (bias에 BN 파라미터 포함)
│   │   ├── batchnorm2d.h/c       # Batch Normalization
│   │   │                         # - 학습된 gamma, beta, mean, var 적용
│   │   │                         # - Fused 모드에서는 identity로 설정
│   │   ├── activation.h/c        # Activation 함수
│   │   │                         # - SiLU: x * sigmoid(x)
│   │   ├── pooling.h/c            # Pooling 연산
│   │   │                         # - MaxPool2D (SPPF용 5×5)
│   │   ├── upsample.h/c          # Upsampling
│   │   │                         # - Nearest-neighbor ×2
│   │   └── concat.h/c            # Concatenation
│   │                             # - Channel 차원 기준 결합
│   │
│   ├── blocks/                   # 복합 블록
│   │   ├── bottleneck.h/c        # Bottleneck 블록
│   │   │                         # - C3 내부 구성 요소
│   │   │                         # - Conv → BN → SiLU → Conv → BN → SiLU
│   │   ├── c3.h/c                # C3 블록 (Cross-stage partial bottleneck)
│   │   │                         # - cv1: Conv+BN+SiLU
│   │   │                         # - bottleneck: n개 반복
│   │   │                         # - cv2: Conv+BN+SiLU (skip path)
│   │   │                         # - concat: [cv1 path, cv2 path]
│   │   │                         # - cv3: Conv+BN+SiLU
│   │   │                         # - Fused BN 지원 (cv1, cv2, cv3)
│   │   └── sppf.h/c              # SPPF 블록
│   │                             # - cv1: Conv+BN+SiLU
│   │                             # - MaxPool 3회: y1=m(x), y2=m(y1), y4=m(y2)
│   │                             # - concat: [x, y1, y2, y4]
│   │                             # - cv2: Conv+BN+SiLU
│   │                             # - Fused BN 지원 (cv1, cv2)
│   │
│   ├── models/                   # 모델 레벨
│   │   ├── yolov5n_graph.h/c     # 모델 그래프 정의
│   │   │                         # - 25개 레이어 구조 정의
│   │   │                         # - 각 레이어의 입력/출력 크기
│   │   ├── yolov5n_build.h/c     # 모델 빌드
│   │   │                         # - 모든 레이어 초기화
│   │   │                         # - 가중치 로드 (Conv, BN, C3, SPPF)
│   │   │                         # - Fused BN 감지 및 처리
│   │   ├── yolov5n_infer.h/c     # Forward pass
│   │   │                         # - 레이어별 forward 호출
│   │   │                         # - 중간 텐서 저장 (디버깅/검증용)
│   │   │                         # - P3, P4, P5 feature map 추출
│   │   └── yolov5n_infer_utils.h # 유틸리티 매크로
│   │
│   └── postprocess/              # 후처리
│       ├── detect.h/c             # Detect head
│       │                         # - P3, P4, P5 → 검출 박스 decode
│       │                         # - Anchor 기반 좌표 변환
│       └── nms.h/c                # Non-Maximum Suppression
│                                 # - IoU 기반 중복 제거
│
├── tools/                        # Python 도구
│   ├── preprocess.py             # 이미지 전처리
│   │                             # - Letterbox resize
│   │                             # - Normalize [0,255] → [0.0,1.0]
│   │                             # - NCHW 변환 및 저장
│   ├── export_yolov5s.py         # 가중치 Export
│   │                             # - PyTorch .pt → weights.bin
│   │                             # - weights_map.json 생성
│   │                             # - model_meta.json 생성
│   ├── dump_golden.py            # PyTorch Golden 데이터 생성
│   │                             # - 모든 레이어 출력 저장
│   │                             # - testdata/python/layer_XXX.bin
│   ├── compare_tensors.py         # 텐서 비교
│   │                             # - PyTorch vs C 출력 비교
│   │                             # - Max diff, Mean diff, RMSE
│   ├── debug_layer2.py            # C3 블록 디버깅 (Layer 2)
│   ├── debug_layer9.py           # SPPF 블록 디버깅 (Layer 9)
│   ├── compare_c3_steps.py        # C3 단계별 비교
│   ├── compare_sppf_steps.py      # SPPF 단계별 비교
│   └── validate.py                # 통합 검증 스크립트
│
├── data/                         # 데이터 디렉토리
│   ├── images/                   # 원본 이미지
│   │   ├── bus.jpg
│   │   └── zidane.jpg
│   ├── yolov5n/                  # YOLOv5n 데이터
│   │   ├── inputs/               # 전처리된 텐서
│   │   │   ├── bus.bin           # NCHW 텐서 (바이너리)
│   │   │   └── bus_meta.txt      # 메타데이터
│   │   └── outputs/              # 검출 결과
│   │       └── bus_detections.txt
│   └── yolov5s/                  # YOLOv5s 데이터 (선택사항)
│       └── inputs/
│
├── testdata_n/                   # YOLOv5n 검증 데이터
│   ├── python/                   # PyTorch Golden 출력
│   │   ├── input.bin
│   │   ├── layer_000.bin ~ layer_023.bin
│   │   ├── output_1_0.bin        # Detect head P3 출력
│   │   ├── output_1_1.bin        # Detect head P4 출력
│   │   └── output_1_2.bin        # Detect head P5 출력
│   └── c/                        # C 구현 출력
│       ├── input.bin
│       ├── layer_000.bin ~ layer_023.bin
│       ├── output_1_0.bin
│       ├── output_1_1.bin
│       └── output_1_2.bin
│
├── debug/                        # 디버깅 중간 출력
│   ├── pytorch/                  # PyTorch 중간 텐서
│   │   ├── c3_cv1_output.bin
│   │   ├── c3_bottleneck_output.bin
│   │   ├── sppf_cv1_output.bin
│   │   └── ...
│   └── c/                        # C 중간 텐서
│       └── (동일한 파일명)
│
├── weights/                      # 모델 가중치
│   ├── yolov5n/                  # YOLOv5n 가중치
│   │   ├── weights_unfused.bin   # C용 바이너리 가중치 (unfused, 기본)
│   │   ├── weights_fused.bin     # C용 바이너리 가중치 (fused)
│   │   ├── weights_map_*.json    # 가중치 매핑
│   │   └── model_meta_*.json    # 모델 메타데이터
│   └── yolov5s/                  # YOLOv5s 가중치 (선택사항)
│       ├── weights_yolov5s.bin
│       ├── weights_map_yolov5s.json
│       └── model_meta_yolov5s.json
│
├── docs/                         # 문서
│   ├── MODULE_ARCHITECTURE.md    # 모듈 아키텍처 상세
│   └── TESTING_PROCEDURE.md      # 테스트 절차 가이드
│
├── tests/                        # 단위 테스트
│   ├── test_conv1x1.c            # Conv 1×1 테스트
│   └── test_integration.c        # 통합 테스트
│
├── third_party/                  # 서드파티
│   ├── yolov5/                   # YOLOv5 원본 (git submodule)
│   └── jsmn/                     # JSON 파서
│
├── CMakeLists.txt                # 빌드 설정
├── README.md                      # 이 파일
├── PROJECT_STATUS.md             # 프로젝트 상태
└── TESTING.md                    # 테스트 가이드
```

---

## 핵심 컴포넌트

### 1. 텐서 관리 (`src/core/tensor.c`)

**텐서 구조:**
```c
typedef struct {
    int32_t n, c, h, w;      // NCHW 레이아웃
    float* data;              // 실제 데이터 (16-byte 정렬)
    size_t capacity;         // 할당된 용량
} tensor_t;
```

**주요 기능:**
- `tensor_create(n, c, h, w)`: 텐서 생성
- `tensor_load(path)`: 바이너리 파일에서 로드
- `tensor_save(tensor, path)`: 바이너리 파일로 저장
- `tensor_dump(tensor, path)`: 디버깅용 덤프

**파일 형식:**
- 헤더: 4개 int32 (n, c, h, w)
- 데이터: n×c×h×w 개 float32

### 2. 메모리 관리 (`src/core/memory.c`)

**Arena Allocator:**
- 한 번에 큰 메모리 블록 할당
- 개별 텐서는 arena 내부에서 할당
- Forward pass 종료 시 전체 해제
- 16-byte 정렬 (SIMD 최적화 준비)

**사용 예:**
```c
arena_t* arena = arena_create(100 * 1024 * 1024);  // 100MB
tensor_t* t = tensor_create_with_arena(arena, 1, 3, 640, 640);
// ... 사용 ...
arena_free(arena);  // 모든 텐서 자동 해제
```

### 3. 가중치 로더 (`src/core/weights_loader.c`)

**가중치 파일 구조:**
- `weights.bin`: 모든 가중치를 하나의 바이너리 파일로 저장
- `weights_map.json`: 레이어별 오프셋 및 shape 정보

**예시:**
```json
{
  "model.0.conv.weight": {
    "offset": 256,
    "shape": [32, 3, 6, 6]
  },
  "model.0.bn.weight": {
    "offset": 128,
    "shape": [32]
  }
}
```

**Fused BN 감지:**
- `model.X.conv.bias`가 존재하면 → Fused BN
- BN을 identity로 설정 (gamma=1, beta=0, mean=0, var=1)
- Conv의 bias에 BN 파라미터가 이미 포함됨

### 4. C3 블록 (`src/blocks/c3.c`)

**구조:**
```
Input
  ├─→ cv1 (Conv+BN+SiLU) ──┐
  │                        │
  └─→ cv2 (Conv+BN+SiLU) ──┤
                          │
                    ┌─────▼─────┐
                    │  Concat   │
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │    cv3    │
                    │(Conv+BN+SiLU)
                    └─────┬─────┘
                          │
                       Output
```

**cv1 경로:**
- Conv → BN (또는 Fused BN 스킵) → SiLU
- Bottleneck n회 반복
- cv3: Conv → BN (또는 Fused BN 스킵) → SiLU

**cv2 경로 (skip):**
- Conv → BN (또는 Fused BN 스킵) → SiLU

**중요:** cv2에도 SiLU activation이 필요함 (PyTorch Conv 클래스는 기본적으로 SiLU 포함)

### 5. SPPF 블록 (`src/blocks/sppf.c`)

**구조:**
```
Input
  │
  ▼
cv1 (Conv+BN+SiLU) → x
  │
  ├─→ MaxPool → y1
  │     │
  │     ├─→ MaxPool → y2
  │     │     │
  │     │     └─→ MaxPool → y4
  │     │
  └─────┴─────────┐
                 ▼
            Concat([x, y1, y2, y4])
                 │
                 ▼
            cv2 (Conv+BN+SiLU)
                 │
                 ▼
              Output
```

**중요:** PyTorch는 `y1 = m(x)`, `y2 = m(y1)`, `y4 = m(y2)`로 3번만 MaxPool 호출 (y3 없음)

---

## 전체 워크플로우

### 1. 이미지 전처리

```bash
python tools/preprocess.py --image bus.jpg --output data/yolov5n/inputs/
```

**처리 과정:**
1. 이미지 로드 (BGR)
2. BGR → RGB 변환
3. Letterbox resize (비율 유지, 640×640으로 패딩)
4. 정규화: [0, 255] → [0.0, 1.0]
5. NCHW 변환: (H, W, C) → (1, 3, H, W)
6. 바이너리 저장: `data/yolov5n/inputs/bus.bin`

**출력:**
- `data/yolov5n/inputs/bus.bin`: NCHW 텐서
- `data/yolov5n/inputs/bus_meta.txt`: 원본 크기, 비율 등 메타데이터

### 2. 가중치 Export

```bash
python tools/export_yolov5s.py yolov5n.pt --output weights/yolov5n/
```

**처리 과정:**
1. PyTorch 모델 로드
2. 모든 레이어의 가중치 추출
3. `weights_fused.bin` / `weights_unfused.bin`: 바이너리 파일로 저장
4. `weights_map_fused.json` / `weights_map_unfused.json`: 레이어별 오프셋 및 shape 정보
5. `model_meta_*.json`: 모델 메타데이터 (입력 크기, 클래스 수 등)

**출력:**
- `weights/yolov5n/weights_unfused.bin`: 모든 가중치 (unfused, 기본)
- `weights/yolov5n/weights_fused.bin`: 모든 가중치 (fused)
- `weights/yolov5n/weights_map_*.json`: 가중치 매핑
- `weights/yolov5n/model_meta_*.json`: 모델 메타데이터

### 3. C 프로그램 실행

```bash
cd build/Release
yolov5s_infer.exe bus
```

**처리 과정:**
1. 입력 텐서 로드: `data/inputs/bus.bin`
2. 모델 빌드:
   - 가중치 로드: `weights/weights.bin`
   - 모든 레이어 초기화
   - Fused BN 감지 및 처리
3. Forward pass:
   - Backbone (Layers 0-9)
   - Head (Layers 10-23)
   - P3, P4, P5 feature map 추출
   - 중간 텐서 저장 (선택적): `testdata/c/layer_XXX.bin`
4. Detect head:
   - P3, P4, P5 → 검출 박스 decode
   - NMS 적용
5. 결과 저장: `data/outputs/bus_detections.txt`

**출력:**
- `data/outputs/bus_detections.txt`: 검출 결과
- `testdata/c/layer_XXX.bin`: 중간 텐서 (검증용)

### 4. 검증 (PyTorch와 비교)

```bash
# 1. PyTorch Golden 생성
python tools/dump_golden.py yolov5n.pt bus --output testdata_n/python

# 2. C 구현 실행
.\build\Release\yolov5_infer.exe bus

# 3. 비교
python tools/compare_tensors.py testdata_n/python testdata_n/c
```

**비교 결과:**
- 각 레이어별 Max diff, Mean diff, RMSE
- Tolerance: 0.0001 (기본값)
- Upsample 레이어 (11, 15)는 자동 SKIP
- 이미지 파일 (bus.bin 등)은 자동 SKIP

---

## 빌드 및 실행

### 빌드 요구사항

- **CMake**: 3.10 이상
- **C 컴파일러**: GCC, Clang, 또는 MSVC
- **Python**: 3.6 이상 (도구 사용 시)
- **Python 패키지**: `torch`, `torchvision`, `opencv-python`, `numpy`

### Fused 전용 (기본)

- **기본 동작**: 가중치/메타는 **fused** 방식만 사용합니다.
  - 기본 경로: `weights/yolov5n/weights_fused.bin`, `weights/yolov5n/model_meta_fused.json`
  - Conv+BN+SiLU가 한 번에 적용되어 메모리 접근·연산이 적어 **임베디드에 유리**합니다.
- **임베디드에서 경로 오버라이드**: `src/core/yolo_config.h`의 매크로를 컴파일 시 정의하면 됩니다.
  - `YOLO_WEIGHTS_DEFAULT_DIR`, `YOLO_WEIGHTS_DEFAULT_FILE`, `YOLO_META_DEFAULT_FILE`
  - 예: ROM에 `weights.bin`만 둘 경우  
    `-DYOLO_WEIGHTS_DEFAULT_DIR="/rom" -DYOLO_WEIGHTS_DEFAULT_FILE="weights.bin" -DYOLO_META_DEFAULT_FILE="model_meta.json"`

### Linux/macOS

```bash
# 빌드
mkdir build && cd build
cmake ..
make -j4

# 실행
./yolov5_infer bus
```

### Windows (Visual Studio)

```bash
# 빌드
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release

# 실행
cd Release
yolov5_infer.exe bus
```

### 실행 파일 위치

- Linux/macOS: `build/yolov5_infer`
- Windows: `build/Release/yolov5_infer.exe`

---

## 검증 및 디버깅

### 전체 레이어 비교

```bash
# 1. PyTorch Golden 생성
python tools/dump_golden.py weights/yolov5s.pt bus --output testdata/python

# 2. C 프로그램 실행 (중간 텐서 저장)
cd build/Release
yolov5s_infer.exe bus

# 3. 비교
cd ../..
python tools/compare_tensors.py testdata/python testdata/c
```

### 특정 블록 디버깅

**C3 블록 (Layer 2):**
```bash
# 1. PyTorch 중간 출력 생성
python tools/debug_layer2.py

# 2. C 프로그램 실행 (디버그 모드)
# src/models/yolov5s_infer.c에서 Layer 2 실행 전:
c3_set_debug_dir("debug/c");

# 3. 비교
python tools/compare_c3_steps.py
```

**SPPF 블록 (Layer 9):**
```bash
# 1. PyTorch 중간 출력 생성
python tools/debug_layer9.py

# 2. C 프로그램 실행 (디버그 모드)
# src/models/yolov5s_infer.c에서 Layer 9 실행 전:
sppf_set_debug_dir("debug/c");

# 3. 비교
python tools/compare_sppf_steps.py
```

### 디버깅 체크리스트

새로운 불일치 발견 시:

1. ✅ 전체 비교로 첫 번째 실패 레이어 확인
2. ✅ 해당 레이어의 구조 확인 (Conv? C3? SPPF?)
3. ✅ 중간 출력 생성 스크립트 작성
4. ✅ 단계별 비교
5. ✅ 근본 원인 파악 (Activation 누락? Fused BN? 로직 오류?)
6. ✅ 수정 및 검증

자세한 내용은 `docs/DEBUGGING_PROCESS.md` 참고.

---

## 기술적 세부사항

### 1. Fused Batch Normalization

**문제:**
- PyTorch는 학습 후 일부 Conv+BN을 fuse하여 성능 향상
- Fused된 경우: Conv의 bias에 BN 파라미터가 포함됨
- Fused되지 않은 경우: Conv의 bias는 None, BN은 별도 파라미터

**해결:**
- `weights_map.json`에서 `model.X.conv.bias` 존재 여부 확인
- 존재하면 → Fused BN:
  - BN을 identity로 설정 (gamma=1, beta=0, mean=0, var=1)
  - `batchnorm2d_forward` 스킵
- 존재하지 않으면 → Normal BN:
  - BN 파라미터 로드
  - `batchnorm2d_forward` 실행

**구현 위치:**
- `src/models/yolov5n_build.c`: 가중치 로드 시 감지
- `src/blocks/c3.c`: `cv1_is_fused`, `cv2_is_fused`, `cv3_is_fused` 플래그
- `src/blocks/sppf.c`: `cv1_is_fused`, `cv2_is_fused` 플래그

### 2. 메모리 관리 전략

**Arena Allocator:**
- Forward pass 시작 시 큰 메모리 블록 할당
- 모든 중간 텐서는 arena 내부에서 할당
- Forward pass 종료 시 arena 전체 해제
- 장점: 빠른 할당/해제, 메모리 단편화 최소화

**Ping-pong 버퍼:**
- 일부 연산에서 입력과 출력이 같은 텐서를 사용
- 예: `batchnorm2d_forward(input, input)` (in-place)

### 3. 텐서 레이아웃

**NCHW 형식:**
- N: Batch size (항상 1)
- C: Channels
- H: Height
- W: Width

**메모리 배치:**
```
[0,0,0,0] [0,0,0,1] ... [0,0,0,W-1]
[0,0,1,0] [0,0,1,1] ... [0,0,1,W-1]
...
[0,0,H-1,0] ... [0,0,H-1,W-1]
[0,1,0,0] ... (다음 채널)
```

### 4. 검증된 수정 사항

**Layer 2 (C3 블록):**
- 문제: cv2 경로에 SiLU activation 누락
- 해결: `activation_silu(skip_output)` 추가
- 검증: `compare_c3_steps.py`로 모든 단계 일치 확인

**Layer 9 (SPPF 블록):**
- 문제 1: MaxPool 로직 오류 (y3 불필요, concat 순서 잘못)
- 해결: `y1 = m(x)`, `y2 = m(y1)`, `y4 = m(y2)`, `concat([x, y1, y2, y4])`
- 문제 2: Fused BN 처리 누락
- 해결: `cv1_is_fused`, `cv2_is_fused` 플래그 추가
- 검증: `compare_sppf_steps.py`로 모든 단계 일치 확인

---

## 문서 및 참고 자료

### 핵심 문서

- **`docs/MODULE_ARCHITECTURE.md`**: 모듈 아키텍처 상세 설명 (각 모듈의 역할과 메커니즘)
- **`docs/TESTING_PROCEDURE.md`**: 테스트 절차 가이드 (단위 테스트, 통합 테스트, 정확도 검증)

### 프로젝트 상태

- **`PROJECT_STATUS.md`**: 완료된 작업 및 진행 상황
- **`TESTING.md`**: 테스트 가이드 (기존)

### 외부 참고

- **YOLOv5 원본**: `third_party/yolov5/` (git submodule)
- **PyTorch 구현**: `third_party/yolov5/models/common.py` 참고

---

## 빠른 시작

### 전체 워크플로우 (한 번에)

```bash
# 1. 이미지 전처리
python tools/preprocess.py --image bus.jpg --output data/yolov5n/inputs/

# 2. PyTorch Golden 생성
python tools/dump_golden.py yolov5n.pt bus --output testdata_n/python

# 3. C 프로그램 실행
.\build\Release\yolov5_infer.exe bus

# 4. 비교
python tools/compare_tensors.py testdata_n/python testdata_n/c
```

### 새로운 이미지로 테스트

```bash
# 1. 이미지 파일을 data/images/에 복사
cp new_image.jpg data/images/

# 2. 전처리
python tools/preprocess.py --image new_image.jpg --output data/yolov5n/inputs/

# 3. PyTorch Golden 생성
python tools/dump_golden.py yolov5n.pt new_image --output testdata_n/python

# 4. C 프로그램 실행
.\build\Release\yolov5_infer.exe new_image

# 5. 비교
python tools/compare_tensors.py testdata_n/python testdata_n/c
```

---

## 출력 형식

### 검출 결과 (`data/outputs/{image}_detections.txt`)

```
Total detections: N

Detection 1:
  Class ID: 0
  Confidence: 0.8523
  BBox: (0.1234, 0.5678, 0.2345, 0.3456)  # normalized [0-1]
  Pixel coords: x=79.0, y=363.4, w=150.1, h=221.2  # pixel coordinates

...

# 파일 끝에 요약 정보
class_id confidence x y w h (normalized)
class_id confidence x_pixel y_pixel w_pixel h_pixel
0 0.8523 0.1234 0.5678 0.2345 0.3456
0 0.8523 79.0 363.4 150.1 221.2
...
```

---

## 라이선스

이 프로젝트는 YOLOv5의 C 포팅 구현입니다. YOLOv5 원본 라이선스를 따릅니다.

---

## 기여 및 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 등록해주세요.

---

**마지막 업데이트**: 2025년 1월
