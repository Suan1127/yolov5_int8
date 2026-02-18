# YOLOv5 C 구현 모듈 아키텍처

이 문서는 `src/` 디렉토리의 각 모듈이 수행하는 역할과 메커니즘을 설명합니다.

## 목차

1. [개요](#개요)
2. [Core 모듈](#core-모듈)
3. [Ops 모듈](#ops-모듈)
4. [Blocks 모듈](#blocks-모듈)
5. [Models 모듈](#models-모듈)
6. [Postprocess 모듈](#postprocess-모듈)
7. [Main 진입점](#main-진입점)

---

## 개요

YOLOv5 C 구현은 다음과 같은 계층 구조로 구성되어 있습니다:

```
main.c (진입점)
  ↓
models/ (모델 구성 및 추론)
  ├── core/ (핵심 유틸리티)
  ├── ops/ (기본 연산)
  ├── blocks/ (복합 블록)
  └── postprocess/ (후처리)
```

---

## Core 모듈

핵심 유틸리티 기능을 제공하는 기반 모듈입니다.

### `tensor.h/c` - 텐서 관리

**역할**: 다차원 배열(텐서) 데이터 구조와 기본 연산을 제공합니다.

**메커니즘**:
- **데이터 구조**: NCHW 레이아웃으로 데이터 저장 (Batch, Channels, Height, Width)
- **메모리 관리**: 동적 할당/해제를 통한 텐서 생성 및 소멸
- **인덱싱**: `tensor_at()` 함수로 (n, c, h, w) 위치의 요소에 직접 접근
- **I/O**: 바이너리 형식으로 텐서를 파일에 저장/로드

**주요 함수**:
- `tensor_create()`: 지정된 차원으로 텐서 생성
- `tensor_free()`: 텐서 메모리 해제
- `tensor_load()`/`tensor_dump()`: 파일에서 로드/파일로 저장

### `memory.h/c` - 메모리 관리

**역할**: Arena 할당자를 통한 효율적인 메모리 관리

**메커니즘**:
- **Arena 할당자**: 연속된 메모리 블록에서 할당하여 단일 해제로 모든 메모리 정리
- **효율성**: 개별 할당/해제 오버헤드 감소
- **사용 시나리오**: 모델 추론 중 임시 텐서 할당에 유용

**주요 함수**:
- `arena_create()`: 지정된 용량의 arena 생성
- `arena_alloc()`: arena에서 메모리 할당
- `arena_destroy()`: arena와 모든 할당된 메모리 해제

### `weights_loader.h/c` - 가중치 로더

**역할**: 사전 학습된 모델 가중치를 바이너리 파일에서 로드

**메커니즘**:
- **바이너리 파일**: `weights.bin` 파일에서 모든 가중치를 한 번에 로드
- **메타데이터 매핑**: `weights_map.json`을 사용하여 레이어 이름으로 가중치 위치 찾기
- **지연 로딩**: 필요한 레이어의 가중치만 포인터로 접근 (실제 복사 없음)

**주요 함수**:
- `weights_loader_create()`: 가중치 파일 로드
- `weights_loader_get()`: 레이어 이름으로 가중치 포인터 반환

### `common.h` - 공통 유틸리티

**역할**: 플랫폼 호환성 및 공통 정의

**메커니즘**:
- Windows/Linux 호환성 처리
- 공통 매크로 정의

---

## Ops 모듈

신경망의 기본 연산(레이어)을 구현합니다.

### `conv2d.h/c` - 2D 합성곱

**역할**: 2D 합성곱 연산 수행

**메커니즘**:
- **합성곱 연산**: 커널을 입력 텐서 위에 슬라이딩하며 내적 계산
- **파라미터**: 출력 채널 수, 커널 크기, stride, padding, groups(깊이별 합성곱용)
- **가중치**: `[out_channels, in_channels, kernel_h, kernel_w]` 형태
- **편향**: 선택적 `[out_channels]` 형태

**주요 함수**:
- `conv2d_init()`: 합성곱 레이어 초기화
- `conv2d_load_weights_int8()`: int8 가중치 로드
- `conv2d_quant_forward()`: BN 없는 Conv (Detect head 등) int8 순전파
- `conv2d_quant_bn_silu_forward()`: Conv+BN+SiLU int8 순전파

### `batchnorm2d.h/c` - 배치 정규화

**역할**: 배치 정규화로 학습 안정성 향상

**메커니즘**:
- **추론 모드**: 학습 시 계산된 `running_mean`과 `running_var` 사용
- **정규화 공식**: `output = (input - mean) / sqrt(var + eps) * weight + bias`
- **파라미터**: `weight` (gamma), `bias` (beta), `running_mean`, `running_var`

**주요 함수**:
- `batchnorm2d_init()`: 배치 정규화 레이어 초기화
- `batchnorm2d_forward()`: 순전파 수행

### `activation.h/c` - 활성화 함수

**역할**: 비선형 활성화 함수 적용

**메커니즘**:
- **SiLU (Swish)**: `x * sigmoid(x)` - YOLOv5에서 주로 사용
- **In-place 연산**: 입력 텐서를 직접 수정

**주요 함수**:
- `activation_silu()`: SiLU 활성화 적용

### `pooling.h/c` - 풀링 연산

**역할**: 공간 차원 축소 및 특징 추출

**메커니즘**:
- **MaxPool2D**: 커널 영역 내 최대값 선택
- **파라미터**: 커널 크기, stride, padding

**주요 함수**:
- `maxpool2d_forward()`: MaxPool 순전파

### `upsample.h/c` - 업샘플링

**역할**: 공간 차원 확대 (주로 FPN에서 사용)

**메커니즘**:
- **Nearest Neighbor**: 가장 가까운 픽셀 값 복사
- **Bilinear**: 선형 보간 (현재 구현은 nearest만 지원)
- **스케일 팩터**: 2배 업샘플링이 일반적

**주요 함수**:
- `upsample_forward()`: 업샘플링 수행

### `concat.h/c` - 텐서 연결

**역할**: 여러 텐서를 채널 차원으로 연결 (FPN에서 사용)

**메커니즘**:
- **채널 연결**: `dim=1` (채널 차원)을 따라 텐서 결합
- **출력 채널**: 모든 입력 텐서의 채널 수 합

**주요 함수**:
- `concat_forward()`: 여러 텐서를 채널 차원으로 연결

---

## Blocks 모듈

여러 기본 연산을 조합한 복합 블록을 구현합니다.

### `bottleneck.h/c` - Bottleneck 블록

**역할**: ResNet 스타일의 잔차 연결을 가진 병목 블록

**메커니즘**:
```
입력 x
  ↓
Conv 1×1 → BN → SiLU
  ↓
Conv 3×3 → BN → SiLU
  ↓
[shortcut이면 x와 더하기]
  ↓
출력
```

**특징**:
- **1×1 Conv**: 채널 수 조정
- **3×3 Conv**: 공간 특징 추출
- **Shortcut**: 입력과 출력을 더하는 잔차 연결 (선택적)

**주요 함수**:
- `bottleneck_init()`: 블록 초기화
- `bottleneck_forward()`: 순전파 수행

### `c3.h/c` - C3 블록

**역할**: YOLOv5의 핵심 블록으로 병렬 경로와 병목 블록을 결합

**메커니즘**:
```
입력 x
  ├─→ Conv 1×1 (cv1) → Bottleneck × n → Conv 1×1 (cv3) ─┐
  │                                                      ↓
  └─→ Conv 1×1 (cv2) ───────────────────────────→ Concat → Conv 1×1 (cv3) → 출력
```

**특징**:
- **병렬 경로**: 메인 경로와 스킵 경로
- **Bottleneck 스택**: 메인 경로에 n개의 Bottleneck 블록
- **채널 분할**: 출력 채널의 절반을 hidden 채널로 사용

**주요 함수**:
- `c3_init()`: C3 블록 초기화
- `c3_forward()`: 순전파 수행

### `sppf.h/c` - SPPF 블록

**역할**: Spatial Pyramid Pooling Fast - 다양한 스케일의 특징 추출

**메커니즘**:
```
입력 x
  ↓
Conv 1×1 (cv1)
  ↓
  ├─→ y1 (원본)
  ├─→ MaxPool → y2
  ├─→ MaxPool → y3
  └─→ MaxPool → y4
  ↓
Concat [y1, y2, y3, y4]
  ↓
Conv 1×1 (cv2)
  ↓
출력
```

**특징**:
- **다중 스케일 풀링**: 동일한 MaxPool을 연속 적용하여 다양한 수용 영역 생성
- **효율성**: 원본 SPP보다 빠르면서 유사한 효과

**주요 함수**:
- `sppf_init()`: SPPF 블록 초기화
- `sppf_forward()`: 순전파 수행

---

## Models 모듈

YOLOv5 모델의 전체 구조와 추론 로직을 관리합니다.

### `yolov5n_graph.h/c` - 모델 그래프 정의

**역할**: YOLOv5n의 레이어 구조와 연결 관계를 정의

**메커니즘**:
- **레이어 타입**: CONV, C3, SPPF, UPSAMPLE, CONCAT, DETECT
- **연결 정보**: 각 레이어의 입력 레이어 인덱스 저장
- **저장 플래그**: FPN 연결을 위해 특정 레이어 출력 저장 여부

**구조**:
- **Backbone (0-9)**: 특징 추출 네트워크
- **Head (10-23)**: FPN (Feature Pyramid Network) 구조
- **Detect (24)**: 검출 헤드

**주요 함수**:
- `yolov5n_get_layer()`: 레이어 정의 조회
- `yolov5n_get_save_list()`: 저장할 레이어 목록 반환

### `yolov5n_build.h/c` - 모델 빌드

**역할**: 모델 구조 생성 및 가중치 로드

**메커니즘**:
- **구조 초기화**: 모든 레이어(Conv, C3, SPPF 등) 초기화
- **가중치 로드**: `weights_loader`를 사용하여 각 레이어에 가중치 할당
- **메타데이터**: `model_meta.json`에서 모델 파라미터 읽기 (선택적)

**모델 구조**:
```c
typedef struct {
    // Backbone: Conv 레이어 5개, C3 블록 4개, SPPF 1개
    // Head: Conv 레이어 4개, C3 블록 4개, Upsample 2개, Concat 3개
    // Detect: Conv 레이어 3개 (P3, P4, P5용)
    // 저장된 특징 맵 12개 (FPN 연결용)
} yolov5n_model_t;
```

**주요 함수**:
- `yolov5n_build()`: 모델 생성 및 가중치 로드
- `yolov5n_free()`: 모델 메모리 해제

### `yolov5n_infer.h/c` - 모델 추론

**역할**: 순전파(forward pass) 수행

**메커니즘**:
1. **Backbone 순전파**: 입력 이미지에서 특징 추출
2. **FPN 순전파**: 
   - 상향 경로: Upsample + Concat으로 고해상도 특징 생성
   - 하향 경로: Downsample + Concat으로 저해상도 특징 생성
3. **특징 저장**: P3, P4, P5 레이어 출력 저장 (Detect 헤드용)
4. **Detect 헤드**: 별도 모듈에서 처리

**데이터 흐름**:
```
입력 (1, 3, 640, 640)
  ↓
Backbone (0-9)
  ↓
Head (10-23)
  ├─→ P3 (1, 64, 80, 80)   [layer 17]
  ├─→ P4 (1, 128, 40, 40)  [layer 20]
  └─→ P5 (1, 256, 20, 20)  [layer 23]
```

**주요 함수**:
- `yolov5n_forward()`: 전체 모델 순전파
- `yolov5n_get_saved_feature()`: 저장된 특징 맵 조회
- `yolov5n_set_output_dir()`: 중간 레이어 출력 저장 디렉토리 설정

---

## Postprocess 모듈

모델 출력을 실제 검출 결과로 변환합니다.

### `detect.h/c` - 검출 헤드

**역할**: 특징 맵을 바운딩 박스와 클래스 확률로 변환

**메커니즘**:
1. **1×1 Conv 적용**: P3, P4, P5 특징 맵에 각각 1×1 합성곱 적용
   - 출력 형태: `(1, 3, H, W, 85)`
   - 85 = 4(bbox) + 1(obj_conf) + 80(class_conf)
2. **디코딩**: 앵커 기반 디코딩으로 실제 바운딩 박스 좌표 계산
   - 앵커 박스와 예측 오프셋을 결합
   - 그리드 좌표를 이미지 좌표로 변환
3. **필터링**: 신뢰도 임계값 이상의 검출만 유지

**앵커 기반 디코딩**:
```
예측: (dx, dy, dw, dh, obj_conf, class_conf[80])
앵커: (aw, ah)
실제 박스:
  x = (grid_x + sigmoid(dx)) * stride
  y = (grid_y + sigmoid(dy)) * stride
  w = anchor_w * exp(dw) * stride
  h = anchor_h * exp(dh) * stride
```

**주요 함수**:
- `detect_forward()`: Detect 헤드 순전파
- `detect_decode()`: 검출 결과 디코딩
- `detect_init_params()`: 앵커 및 파라미터 초기화

### `nms.h/c` - 비최대 억제

**역할**: 겹치는 검출 결과를 제거하여 최종 검출만 유지

**메커니즘**:
1. **신뢰도 정렬**: 검출 결과를 신뢰도 내림차순 정렬
2. **IoU 계산**: 모든 검출 쌍의 IoU (Intersection over Union) 계산
3. **억제**: 높은 신뢰도 검출과 IoU가 임계값 이상인 검출 제거
4. **제한**: 최대 검출 개수 제한

**IoU 계산**:
```
IoU = (교집합 영역) / (합집합 영역)
```

**주요 함수**:
- `nms()`: 비최대 억제 수행
- `calculate_iou()`: 두 바운딩 박스의 IoU 계산

---

## Main 진입점

### `main.c` - 메인 프로그램

**역할**: 전체 추론 파이프라인을 조율하는 진입점

**실행 흐름**:
1. **입력 로드**: 전처리된 이미지 텐서 로드 (`data/yolov5n/inputs/*.bin`)
2. **모델 빌드**: 가중치 파일과 메타데이터로 모델 생성
3. **순전파**: `yolov5n_forward()` 호출하여 P3, P4, P5 특징 생성
4. **검출 헤드**: `detect_forward()`로 검출 결과 생성
5. **디코딩**: `detect_decode()`로 바운딩 박스 변환
6. **NMS**: `nms()`로 중복 검출 제거
7. **결과 저장**: 검출 결과를 텍스트 파일로 저장

**주요 기능**:
- 경로 자동 탐색 (여러 상대 경로 시도)
- 중간 레이어 출력 저장 (디버깅/검증용)
- 검출 결과 파일 저장 (`data/yolov5n/outputs/*_detections.txt`)

---

## 모듈 간 의존성

```
main.c
  ├─→ models/yolov5n_infer.c
  │     ├─→ models/yolov5n_build.c
  │     │     ├─→ core/weights_loader.c
  │     │     ├─→ ops/conv2d.c
  │     │     ├─→ ops/batchnorm2d.c
  │     │     ├─→ blocks/c3.c
  │     │     └─→ blocks/sppf.c
  │     └─→ blocks/bottleneck.c
  ├─→ postprocess/detect.c
  │     └─→ ops/conv2d.c
  └─→ postprocess/nms.c
```

---

## 설계 원칙

이 구현은 다음 SOLID 원칙을 따릅니다:

1. **단일 책임 원칙 (SRP)**: 각 모듈은 하나의 명확한 역할만 수행
   - `tensor.c`: 텐서 데이터 구조만 관리
   - `conv2d.c`: 합성곱 연산만 수행
   - `c3.c`: C3 블록만 구현

2. **개방/폐쇄 원칙 (OCP)**: 확장에는 열려있고 수정에는 닫혀있음
   - 새로운 블록 타입 추가 시 기존 코드 수정 없이 확장 가능
   - 새로운 활성화 함수 추가 시 기존 레이어 수정 불필요

3. **의존성 역전 원칙 (DIP)**: 추상화에 의존
   - `tensor_t` 구조체를 통한 데이터 추상화
   - 함수 포인터나 인터페이스를 통한 연산 추상화 (향후 확장 가능)

---

## 참고사항

- 모든 텐서는 **NCHW 레이아웃**을 사용합니다
- 가중치는 **사전 학습된 PyTorch 모델에서 변환**된 바이너리 형식입니다
- 중간 레이어 출력은 **디버깅 및 검증** 목적으로 저장할 수 있습니다
- 메모리 관리는 **명시적 할당/해제**를 사용합니다 (가비지 컬렉션 없음)
