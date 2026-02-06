# YOLOv5 C 구현 테스트 절차

이 문서는 YOLOv5 C 구현의 전체 테스트 및 검증 절차를 상세히 설명합니다.

## 목차

1. [테스트 개요](#테스트-개요)
2. [사전 준비](#사전-준비)
3. [단위 테스트](#단위-테스트)
4. [통합 테스트](#통합-테스트)
5. [정확도 검증](#정확도-검증)
6. [최종 검출 결과 검증](#최종-검출-결과-검증)
7. [성능 테스트](#성능-테스트)
8. [문제 해결](#문제-해결)

---

## 테스트 개요

YOLOv5 C 구현의 테스트는 다음 단계로 구성됩니다:

```
1. 단위 테스트 (Unit Tests)
   └─> 개별 모듈/함수 검증

2. 통합 테스트 (Integration Tests)
   └─> 모델 빌드 및 순전파 검증

3. 정확도 검증 (Accuracy Validation)
   └─> Python 골든 참조와 텐서 비교

4. 최종 검출 결과 검증 (Detection Validation)
   └─> 바운딩 박스 및 NMS 결과 비교

5. 성능 테스트 (Performance Tests)
   └─> 추론 시간 및 메모리 사용량 측정
```

---

## 사전 준비

### 1.1 환경 설정

```bash
# 프로젝트 루트 디렉토리로 이동
cd YOLO_c

# Python 가상환경 활성화 (선택사항)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 또는
venv\Scripts\activate  # Windows

# 필요한 Python 패키지 설치
pip install torch torchvision numpy opencv-python
```

### 1.2 모델 가중치 준비

YOLOv5n 가중치 파일이 다음 위치에 있어야 합니다:

```
weights/yolov5n/
├── weights_fused.bin            # 바이너리 가중치 (fused, 기본 사용)
├── weights_map_fused.json       # 가중치 맵 (fused)
├── model_meta_fused.json        # 모델 메타데이터 (fused)
├── weights_unfused.bin          # (선택) unfused 가중치
├── weights_map_unfused.json     # (선택) unfused 맵
└── model_meta_*.json            # 기타 메타데이터
```

**가중치 파일이 없는 경우**:
```bash
# PyTorch 모델에서 가중치 추출
python tools/export_yolov5s.py yolov5n.pt --output weights/yolov5n/
```

### 1.3 테스트 이미지 준비

```bash
# 테스트 이미지가 data/images/ 디렉토리에 있어야 함
# 예: data/images/bus.jpg

# 이미지 전처리 (텐서로 변환)
python tools/preprocess.py --image bus.jpg --output data/yolov5n/inputs/
```

**출력**: `data/yolov5n/inputs/bus.bin` (NCHW 형식의 텐서)

### 1.4 프로젝트 빌드

```bash
# 빌드 디렉토리 생성
mkdir build
cd build

# CMake 설정
cmake ..

# 빌드
# Windows
cmake --build . --config Release

# Linux/macOS
make

cd ..
```

**빌드 결과**:
- Windows: `build/Release/yolov5_infer.exe`
- Linux/macOS: `build/yolov5_infer`

---

## 단위 테스트

.\yolov5_infer.exe bus

### 2.1 텐서 연산 테스트

```bash
# 텐서 생성 및 기본 연산 테스트
cd tests
# Windows
.\run_tests.bat

# Linux/macOS
./run_tests.sh
```

**테스트 항목**:
- 텐서 생성/해제
- 텐서 복사
- 텐서 I/O (저장/로드)
- 메모리 관리

### 2.2 합성곱 레이어 테스트

```bash
# 1×1 합성곱 레이어 테스트
cd build
# Windows
.\Release\test_conv1x1.exe

# Linux/macOS
./test_conv1x1
```

**검증 사항**:
- 합성곱 연산 정확도
- 가중치 로드
- 출력 shape 확인

### 2.3 통합 테스트

```bash
# 모델 빌드 및 순전파 테스트
cd build
# Windows
.\Release\test_integration.exe

# Linux/macOS
./test_integration
```

**검증 사항**:
- 모델 빌드 성공
- 가중치 로드 성공
- 순전파 실행 성공
- 출력 텐서 shape 확인

---

## 통합 테스트

### 3.1 전체 파이프라인 테스트

```bash
# C 구현 실행
# Windows
.\build\Release\yolov5_infer.exe bus

# Linux/macOS
./build/yolov5_infer bus
```

**실행 결과**:
- 입력 텐서 로드
- 모델 빌드
- 순전파 실행
- P3, P4, P5 특징 생성
- Detect 헤드 실행
- NMS 수행
- 검출 결과 저장

**출력 위치**:
- 중간 레이어: `testdata_n/c/layer_*.bin`
- Detect 헤드: `testdata_n/c/output_1_*.bin`
- 최종 검출: `data/yolov5n/outputs/bus_detections.txt`

### 3.2 출력 검증

```bash
# 출력 텐서 shape 확인
# P3: (1, 64, 80, 80)
# P4: (1, 128, 40, 40)
# P5: (1, 256, 20, 20)

# 검출 결과 확인
cat data/yolov5n/outputs/bus_detections.txt
```

**기대 결과**:
- 모든 레이어가 정상적으로 실행됨
- 출력 shape가 예상과 일치함
- 검출 결과가 정상적으로 저장됨

---

## 정확도 검증

### 4.1 Python 골든 참조 생성

Python PyTorch 모델을 사용하여 골든 참조 텐서를 생성합니다.

```bash
# 모든 레이어 출력 저장 (0-23)
python tools/dump_golden.py yolov5n.pt bus --output testdata_n/python

# 특정 레이어만 저장
python tools/dump_golden.py yolov5n.pt bus --output testdata_n/python --layers 0 1 2 3
```

**출력 파일**:
```
testdata_n/python/
├── input.bin              # 입력 텐서
├── layer_000.bin          # Layer 0 출력
├── layer_001.bin          # Layer 1 출력
├── ...
├── layer_023.bin          # Layer 23 출력
├── output_1_0.bin         # Detect head P3 출력
├── output_1_1.bin         # Detect head P4 출력
├── output_1_2.bin         # Detect head P5 출력
└── golden_meta.json       # 메타데이터
```

### 4.2 C 구현 실행

```bash
# C 구현 실행 (중간 레이어 저장)
.\build\Release\yolov5_infer.exe bus
```

**출력 파일**:
```
testdata_n/c/
├── input.bin              # 입력 텐서
├── layer_000.bin          # Layer 0 출력
├── layer_001.bin          # Layer 1 출력
├── ...
├── layer_023.bin          # Layer 23 출력
├── output_1_0.bin         # Detect head P3 출력
├── output_1_1.bin         # Detect head P4 출력
└── output_1_2.bin         # Detect head P5 출력
```

### 4.3 텐서 비교

```bash
# 전체 디렉토리 비교
python tools/compare_tensors.py testdata_n/python testdata_n/c

# 특정 레이어만 비교
python tools/compare_tensors.py testdata_n/python/layer_000.bin testdata_n/c/layer_000.bin

# Tolerance 설정 (기본값: 0.0001)
python tools/compare_tensors.py testdata_n/python testdata_n/c --tolerance 0.001
```

**비교 결과 예시**:
```
Comparing layer_000.bin...
  Shape: [  1  16 320 320]
  Max diff: 1.811981e-05
  Mean diff: 1.023055e-06
  RMSE: 1.591757e-06
  Within tolerance (0.0001): OK

Comparing layer_001.bin...
  Shape: [  1  32 160 160]
  Max diff: 2.345678e-05
  Mean diff: 1.234567e-06
  RMSE: 1.987654e-06
  Within tolerance (0.0001): OK

...

Summary:
  Total files compared: 24
  Passed: 24
  Failed: 0
```

**기대 결과**: 모든 레이어가 tolerance (기본값: 0.0001) 내에서 일치해야 합니다.

### 4.4 레이어별 검증 체크리스트

| 레이어 | 타입 | 입력 Shape | 출력 Shape | 상태 |
|--------|------|------------|------------|------|
| 0 | Conv | (1, 3, 640, 640) | (1, 16, 320, 320) | ✅ |
| 1 | Conv | (1, 16, 320, 320) | (1, 32, 160, 160) | ✅ |
| 2 | C3 | (1, 32, 160, 160) | (1, 32, 160, 160) | ✅ |
| 3 | Conv | (1, 32, 160, 160) | (1, 64, 80, 80) | ✅ |
| 4 | C3 | (1, 64, 80, 80) | (1, 64, 80, 80) | ✅ |
| 5 | Conv | (1, 64, 80, 80) | (1, 128, 40, 40) | ✅ |
| 6 | C3 | (1, 128, 40, 40) | (1, 128, 40, 40) | ✅ |
| 7 | Conv | (1, 128, 40, 40) | (1, 256, 20, 20) | ✅ |
| 8 | C3 | (1, 256, 20, 20) | (1, 256, 20, 20) | ✅ |
| 9 | SPPF | (1, 256, 20, 20) | (1, 256, 20, 20) | ✅ |
| 10 | Conv | (1, 256, 20, 20) | (1, 128, 20, 20) | ✅ |
| 11 | Upsample | (1, 128, 20, 20) | (1, 128, 40, 40) | ✅ |
| 12 | Concat | (1, 128, 40, 40), (1, 128, 40, 40) | (1, 256, 40, 40) | ✅ |
| 13 | C3 | (1, 256, 40, 40) | (1, 128, 40, 40) | ✅ |
| 14 | Conv | (1, 128, 40, 40) | (1, 64, 40, 40) | ✅ |
| 15 | Upsample | (1, 64, 40, 40) | (1, 64, 80, 80) | ✅ |
| 16 | Concat | (1, 64, 80, 80), (1, 64, 80, 80) | (1, 128, 80, 80) | ✅ |
| 17 | C3 | (1, 128, 80, 80) | (1, 64, 80, 80) | ✅ (P3) |
| 18 | Conv | (1, 64, 80, 80) | (1, 64, 40, 40) | ✅ |
| 19 | Concat | (1, 64, 40, 40), (1, 128, 40, 40) | (1, 128, 40, 40) | ✅ |
| 20 | C3 | (1, 128, 40, 40) | (1, 128, 40, 40) | ✅ (P4) |
| 21 | Conv | (1, 128, 40, 40) | (1, 128, 20, 20) | ✅ |
| 22 | Concat | (1, 128, 20, 20), (1, 128, 20, 20) | (1, 256, 20, 20) | ✅ |
| 23 | C3 | (1, 256, 20, 20) | (1, 256, 20, 20) | ✅ (P5) |

**참고**: Upsample 레이어(11, 15)는 가중치가 없으므로 비교에서 자동으로 제외됩니다.

---

## 최종 검출 결과 검증

### 5.1 Python 검출 결과 생성

```bash
# Python 모델로 검출 결과 생성
python tools/dump_golden.py yolov5n.pt bus --output testdata_n/python --save-detections
```

또는 직접 Python 스크립트로 실행:
```python
import torch
from yolov5 import YOLOv5

model = YOLOv5('yolov5n.pt')
results = model('data/images/bus.jpg')
results.save('testdata_n/python/bus_detections.txt')
```

### 5.2 C 검출 결과 확인

```bash
# C 구현 검출 결과 확인
cat data/yolov5n/outputs/bus_detections.txt
```

**출력 형식**:
```
Image: bus
Total detections: 3
Format: class_id confidence x y w h (normalized 0-1)
Format: class_id confidence x_pixel y_pixel w_pixel h_pixel

0 0.8231 0.3816 0.5816 0.1322 0.4333
0 0.8231 245.2 372.2 84.6 277.3

0 0.8158 0.2411 0.5933 0.1527 0.6846
0 0.8158 154.3 379.7 97.7 438.1

...
```

### 5.3 검출 결과 비교

```bash
# Python과 C 검출 결과 비교
python tools/compare_detections.py \
    testdata_n/python/bus_detections.txt \
    testdata_n/c/bus_detections.txt \
    --iou-threshold 0.5 \
    --conf-tolerance 0.01
```

**비교 항목**:
- 검출 개수
- 바운딩 박스 좌표 (IoU 기반)
- 신뢰도 점수
- 클래스 ID

**비교 결과 예시**:
```
Comparing detections...
  Python detections: 3
  C detections: 3
  
  Matched: 3
  Unmatched Python: 0
  Unmatched C: 0
  
  Average IoU: 0.92
  Average confidence diff: 0.0023
  
  Status: PASS
```

### 5.4 NMS 테스트

```bash
# NMS 로직 테스트
python tools/test_nms.py
```

**검증 사항**:
- IoU 계산 정확도
- NMS 알고리즘 동작
- Python torchvision NMS와의 일치성

---

## 성능 테스트

### 6.1 추론 시간 측정

```bash
# C 구현 추론 시간 측정
# Windows
powershell -Command "Measure-Command { .\build\Release\yolov5_infer.exe bus }"

# Linux/macOS
time ./build/yolov5_infer bus
```

또는 Python 스크립트로 측정:
```python
import time
import subprocess

start = time.time()
subprocess.run(['./build/yolov5_infer', 'bus'])
end = time.time()
print(f"Inference time: {end - start:.3f} seconds")
```

### 6.2 메모리 사용량 측정

```bash
# Windows (PowerShell)
Get-Process yolov5_infer | Select-Object WorkingSet

# Linux
/usr/bin/time -v ./build/yolov5_infer bus
```

### 6.3 벤치마크

```bash
# 여러 이미지에 대해 벤치마크 실행
for img in bus car person; do
    echo "Testing $img..."
    time ./build/yolov5_infer $img
done
```

**기대 성능** (참고):
- 추론 시간: ~100-200ms (CPU, 640×640 입력)
- 메모리 사용량: ~100-200MB

---

## 문제 해결

### 7.1 모델 빌드 실패

**증상**: `yolov5n_build()` 반환값이 NULL

**해결 방법**:
1. 가중치 파일 확인
   ```bash
   ls -lh weights/yolov5n/weights_unfused.bin
   ```
2. 가중치 맵 파일 확인
   ```bash
   cat weights/yolov5n/weights_map_yolov5n.json | head -20
   ```
3. 파일 경로 확인 (상대 경로 vs 절대 경로)

### 7.2 Forward Pass 실패

**증상**: `yolov5n_forward()` 반환값이 0이 아님

**해결 방법**:
1. 입력 텐서 shape 확인
   ```bash
   # 입력 텐서 shape 확인
   python -c "
   import numpy as np
   with open('data/yolov5n/inputs/bus.bin', 'rb') as f:
       dims = np.fromfile(f, dtype=np.int32, count=4)
       print(f'Input shape: {dims}')
   "
   ```
   기대값: `[1, 3, 640, 640]`

2. 메모리 부족 확인
   - 시스템 메모리 확인
   - 중간 텐서 저장 비활성화

3. 채널 수 불일치 확인
   - YOLOv5n: 16, 32, 64, 128, 256
   - YOLOv5s: 32, 64, 128, 256, 512

### 7.3 텐서 비교 실패

**증상**: `compare_tensors.py`에서 일부 레이어가 tolerance 초과

**해결 방법**:
1. 입력 텐서 일치 확인
   ```bash
   python tools/compare_tensors.py \
       testdata_n/python/input.bin \
       testdata_n/c/input.bin
   ```

2. 특정 레이어 상세 분석
   ```bash
   # 큰 차이가 있는 위치 찾기
   python tools/compare_tensors.py \
       testdata_n/python/layer_002.bin \
       testdata_n/c/layer_002.bin \
       --tolerance 0.0001 \
       --verbose
   ```

3. 레이어별 디버깅
   - C3 블록: `tools/debug_layer2.py`
   - SPPF 블록: `tools/debug_layer9.py`

### 7.4 검출 결과 불일치

**증상**: Python과 C 검출 결과가 크게 다름

**해결 방법**:
1. Detect 헤드 출력 비교
   ```bash
   python tools/compare_tensors.py \
       testdata_n/python/output_1_0.bin \
       testdata_n/c/output_1_0.bin
   ```

2. 앵커 설정 확인
   - `detect.c`의 앵커 값 확인
   - Python 모델의 앵커와 일치하는지 확인

3. 디코딩 로직 확인
   - 좌표 변환 공식 확인
   - 신뢰도 계산 확인

### 7.5 출력이 모두 0

**증상**: 모든 레이어 출력이 0

**해결 방법**:
1. 가중치 로드 확인
   ```bash
   # 가중치 파일 크기 확인
   ls -lh weights/yolov5n/weights_unfused.bin
   ```

2. 레이어 실행 순서 확인
   - `yolov5n_graph.c`의 레이어 순서 확인

3. Fused Batch Normalization 처리 확인
   - `conv2d.c`의 Fused BN 플래그 확인

---

## 자동화된 테스트 스크립트

### 8.1 전체 검증 파이프라인

```bash
# validate.py 사용
python tools/validate.py \
    --image bus \
    --model yolov5n.pt \
    --input-size 640 \
    --tolerance 0.0001
```

**실행 단계**:
1. 이미지 전처리 (필요시)
2. Python 골든 참조 생성
3. C 구현 실행
4. 텐서 비교
5. 결과 요약

### 8.2 CI/CD 통합

```yaml
# .github/workflows/test.yml 예시
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build
        run: |
          mkdir build && cd build
          cmake .. && make
      - name: Run Tests
        run: |
          python tools/validate.py --image bus --model yolov5n.pt
```

---

## 테스트 체크리스트

### 기본 기능 테스트
- [x] 모델 빌드 성공
- [x] 가중치 로드 성공
- [x] Forward pass 실행 성공
- [x] 출력 텐서 shape 확인
- [x] Saved features 확인

### 정확도 테스트
- [x] 골든 참조와 텐서 비교 (24/24 레이어 통과)
- [x] 레이어별 출력 검증
- [x] Detect head 출력 검증
- [x] 최종 검출 결과 비교

### 후처리 테스트
- [x] NMS 알고리즘 검증
- [x] 바운딩 박스 디코딩 검증
- [x] 검출 결과 파일 저장 검증

### 성능 테스트
- [ ] 인퍼런스 시간 측정
- [ ] 메모리 사용량 측정
- [ ] 배치 처리 성능 측정

---

## 참고 자료

- **모델 아키텍처**: `docs/MODULE_ARCHITECTURE.md`
- **프로젝트 상태**: `PROJECT_STATUS.md`
- **기존 테스트 가이드**: `TESTING.md`

---

## 부록: 테스트 데이터 구조

```
testdata_n/
├── python/              # Python 골든 참조
│   ├── input.bin
│   ├── layer_*.bin
│   ├── output_1_*.bin
│   └── golden_meta.json
└── c/                   # C 구현 출력
    ├── input.bin
    ├── layer_*.bin
    └── output_1_*.bin

data/
├── images/              # 원본 이미지
│   └── bus.jpg
├── yolov5n/
│   ├── inputs/          # 전처리된 입력 텐서
│   │   └── bus.bin
│   └── outputs/         # 최종 검출 결과
│       └── bus_detections.txt
└── ...

weights/
└── yolov5n/
    ├── weights_unfused.bin
    ├── weights_map_unfused.json
    ├── weights_fused.bin
    ├── weights_map_fused.json
    └── model_meta_*.json
```
