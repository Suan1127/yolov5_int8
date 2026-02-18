# YOLOv5n C 추론 — Conv 가속기 설계용 현황 정리

**목적**: Conv 전용 하드웨어 가속기 설계. 이 문서를 GPT 등에 넣어 현재 연산 흐름과 구조를 이해시키기 위한 요약이다.

---

## 1. 전체 추론 파이프라인

```
[입력 이미지 전처리] → [Backbone 0–9] → [Neck 10–23] → [Detect head 24] → [Decode + NMS] → [Detection 결과]
```

- **입력**: float32 NCHW, `(1, 3, H, W)` (예: 640×640).
- **Backbone**: Layer 0–9. 특징 추출. Conv → C3 → Conv → … → SPPF.
- **Neck**: Layer 10–23. FPN 형태. Conv, Upsample, Concat, C3 반복. 출력 P3, P4, P5 저장.
- **Detect head**: Layer 24. P3/P4/P5 각각 1×1 Conv → (1, 255, h, w) 형태의 detection map.
- **Decode**: confidence threshold 적용, bbox/class 디코딩.
- **NMS**: Non-Maximum Suppression 후 최종 detection 개수 제한.

**현재 가중치**: float 가중치 로더는 제거됨. **int8 전용**  
- `weights_int8.bin` + `scales_int8.json` + `bias.bin` 만 사용.

---

## 2. Conv가 쓰이는 위치 (가속 대상)

모든 Conv는 **int8 가중치(q_weight) + per-layer scale_w + fused bias** 로 동작한다.

### 2.1 단일 Conv 레이어 (Backbone / Head)

| Layer | 용도 | 연산 | in_c→out_c | k,s,p | 비고 |
|-------|------|------|------------|-------|------|
| 0 | Backbone | Conv+BN+SiLU | 3→16 | 6,2,2 | 첫 레이어 |
| 1 | Backbone | Conv+BN+SiLU | 16→32 | 3,2,1 | |
| 3 | Backbone | Conv+BN+SiLU | 32→64 | 3,2,1 | |
| 5 | Backbone | Conv+BN+SiLU | 64→128 | 3,2,1 | |
| 7 | Backbone | Conv+BN+SiLU | 128→256 | 3,2,1 | |
| 10 | Neck | Conv+BN+SiLU | 256→128 | 1,1,0 | 1×1 |
| 14 | Neck | Conv+BN+SiLU | 128→64 | 1,1,0 | 1×1 |
| 18 | Neck | Conv+BN+SiLU | 64→64 | 3,2,1 | |
| 21 | Neck | Conv+BN+SiLU | 128→128 | 3,2,1 | |
| 24.m.0/1/2 | Detect | Conv only (1×1) | 64/128/256→255 | 1,1,0 | BN 없음, bias만 |

- Backbone/Neck Conv: **conv2d_quant_bn_silu_forward**  
  - int8 Conv → dequant + fused bias → (선택) BN → **SiLU** (CPU에서 float).
- Detect head Conv: **conv2d_quant_forward**  
  - int8 Conv → dequant + bias → BN 없음.

### 2.2 블록 내부 Conv (C3, SPPF)

- **C3**: cv1, cv2, cv3 (각 Conv+BN+SiLU) + bottleneck n개 (각 bottleneck 내 conv1, conv2: Conv+BN+SiLU).  
  - 모두 **conv2d_quant_bn_silu_forward** 호출.
- **SPPF**: cv1, cv2 (각 Conv+BN+SiLU) + MaxPool 3회.  
  - Conv만 **conv2d_quant_bn_silu_forward**.

즉, **가속기 입장에서 “Conv 연산”은 항상 다음 두 형태 중 하나**다.

1. **Conv+BN fused + SiLU**  
   - int8 conv → int32 accumulate → dequant + bias(BN fused) → float → SiLU(CPU).
2. **Conv + bias only (Detect head)**  
   - int8 conv → int32 accumulate → dequant + bias → float.

---

## 3. Conv 연산의 소프트웨어 흐름 (가속기 인터페이스 설계 참고)

### 3.1 공통: int8 Conv까지

1. **입력**: float32 activation `(N, C_in, H, W)`.
2. **입력 양자화**:  
   - 현재는 **동적** min/max로 symmetric scale 계산 → `scale_in`.  
   - `q_in = round(float / scale_in)` → int8.  
   - (가속기 설계 시: calibration으로 고정 scale 쓰면 이 부분을 단순화 가능.)
3. **가중치**: 이미 int8. 레이어별 `scale_w` (float) 하나.
4. **연산**:  
   - `acc32[b,oc,oh,ow] = sum_over_ic,kh,kw( q_in[b,ic,ih,iw] * q_weight[oc,ic,kh,kw] )`  
   - 즉 **int8×int8 → int32 누적합**.  
   - 이 부분이 **Conv 가속기가 담당할 핵심**이다.

### 3.2 가중치/스케일 형식

- **q_weight**: int8, layout `[out_c, in_c, k, k]`.
- **scale_w**: float, 레이어당 1개 (symmetric weight quantization).
- **bias**: float32, `[out_c]`.  
  - Backbone/Neck: Conv+BN가 이미 fusion된 bias.  
  - Detect: Conv bias만.

로딩: `weights_loader_int8`가 `weights_int8.bin` + `scales_int8.json`에서 `q_weight`/`scale_w`를, `bias.bin`에서 bias를 읽어 레이어별로 넘긴다.

#### weights_int8.bin 형식 (한 레이어 블록 기준)

- **파일**: `weights_int8.bin` 하나에 여러 레이어가 **연속**으로 들어감. 레이어별 시작 오프셋/numel은 `scales_int8.json`의 `order`/`shapes`로 결정.
- **한 레이어 블록** (해당 레이어용 q_weight만 잘랐을 때):
  - **레이아웃**: `[out_c, in_c, k, k]`
  - **플랫 순서(메모리/파일)**: **oc → ic → kh → kw** (kw가 가장 안쪽).
  - **인덱스**: `idx = oc*(IC*K*K) + ic*(K*K) + kh*K + kw`
  - **타입**: int8, 1바이트씩(바이트 스트림).
- 한 oc에 대해 `IC*K*K` 바이트가 연속하고, 그 안은 ic → kh → kw 순 (kw 연속).

#### RTL이 가중치 스트림으로 기대해야 할 순서

- DMA/AXI로 한 레이어(또는 OC_TILE 구간) weight를 줄 때 **동일 순서**: **oc → ic → kh → kw** (kw 연속).
- **총 바이트**: `OC_TILE * IC * K * K`.
- RTL은 이 순서대로 받아 oc별로 BRAM에 채우거나 (ic, kh, kw) 순으로 MAC에 공급하면 C와 동일 인덱싱.

### 3.3 Dequant 및 이후 (CPU 또는 별도 유닛)

- **공식**:  
  `float_out = (float)acc32 * scale_in * scale_w + bias`  
  (per-channel bias 적용 시: `float_out[oc] = acc32[oc] * scale_in * scale_w + bias[oc]`.)
- **conv2d_quant_bn_silu_forward**  
  - 위 dequant+bias 후, BN이 fused가 아니면 별도 BN 연산, 그 다음 **SiLU**.
- **conv2d_quant_forward**  
  - dequant+bias까지만. Detect head용.

가속기 설계 시:
- **가속**: int8 입력 + int8 가중치 → int32 accumulator 출력.
- **호스트/다른 유닛**: scale_in, scale_w, bias로 float 복원 후, 필요 시 BN/SiLU.

---

## 4. 레이어별 Conv 호출 정리 (코드 경로)

| 구분 | 호출 함수 | 사용처 |
|------|-----------|--------|
| Backbone 0,1,3,5,7 | conv2d_quant_bn_silu_forward | yolov5n_infer.c |
| Head 10,14,18,21 | conv2d_quant_bn_silu_forward | yolov5n_infer.c |
| C3 내부 cv1/cv2/cv3 + bottleneck conv1/conv2 | conv2d_quant_bn_silu_forward | c3.c |
| SPPF cv1/cv2 | conv2d_quant_bn_silu_forward | sppf.c |
| Detect 24 (P3/P4/P5 각 1×1) | conv2d_quant_forward | detect.c |

- **conv2d_quant_bn_silu_forward**  
  - `conv2d_int8_forward`(int8 conv) → `dequant_acc32_to_float`(scale_in, scale_w, bias) → (선택) BN → SiLU.
- **conv2d_quant_forward**  
  - `conv2d_int8_forward` → `dequant_acc32_to_float` 만.

int8 Conv의 실제 루프는 `src/ops/conv2d_int8.c`의 **conv2d_int8_forward** (및 one_pixel 커널)에 정의되어 있으며, 가속기는 이 연산을 하드웨어로 대체하면 된다.

---

## 5. 텐서 레이아웃 및 크기 (YOLOv5n, 640×640 기준)

- **레이아웃**: NCHW.
- Backbone 통과 후 채널 수 (YOLOv5n): 16 → 32 → 32 → 64 → 64 → 128 → 128 → 256 → 256 (SPPF 출력 256 ch).
- Neck에서 P3/P4/P5 해상도:  
  - P3: 80×80, 64 ch → Detect 후 255 ch.  
  - P4: 40×40, 128 ch → 255 ch.  
  - P5: 20×20, 256 ch → 255 ch.

Conv 커널 종류: **6×6(s=2), 3×3(s=1 또는 2), 1×1**.  
가속기는 위 in/out 채널, stride, padding을 지원하면 된다.

---

## 6. 가속기 설계 시 참고 포인트

1. **연산 단위**: int8×int8 → int32 누적. 레이어당 scale_w 1개, bias는 float32 per channel.
2. **입력**: 현재는 float→int8 양자화를 CPU에서 동적 min/max로 수행. 가속기에서 int8 입력을 받을 경우, scale_in은 호스트가 넘기거나 고정 calibration 값 사용.
3. **출력**: int32 accumulator를 호스트로 반환하면, 기존 `dequant_acc32_to_float`와 동일 식으로 float 복원 후 BN/SiLU 처리 가능.
4. **블록**: C3/SPPF는 여러 Conv를 순차 호출하므로, 가속기 API가 “한 레이어 Conv 실행” 단위면 기존 소프트웨어 호출을 그대로 가속기 호출로 치환하기 쉽다.
5. **파일**:  
   - `src/ops/conv2d.c`: conv2d_quant_forward, conv2d_quant_bn_silu_forward (호스트: 차원/양자화/백엔드 호출/디양자화).  
   - `src/ops/conv_backend.c/h`: 통합 Conv API. conv_backend_run_int8(params, ...) 한 개; 백엔드만 SW/HW 갈아끼움.  
   - `src/ops/conv2d_int8.c`: int8 conv 루프 (SW 백엔드에서 사용).  
   - `src/ops/quant_util.c`: dequant_acc32_to_float, quant_scale_symmetric_from_minmax.

이 문서만으로 “어디서 Conv가 어떻게 쓰이고, 가중치/스케일/데이터 형식이 무엇인지”를 GPT에 전달해 Conv 가속기 설계 논의를 이어갈 수 있다.

---

## 7. 하나의 Conv IP + 백엔드 갈아끼우기 (권장 구조)

- **상황**: YOLO는 1×1, 3×3, 6×6이 섞여 나오므로, 커널마다 IP를 따로 만들면 IP 수 증가, BD/검증/유지보수 부담.
- **HW**: Conv 엔진 하나. 레지스터로 K, stride, padding만 바꿔 1×1/3×3/6×6 모두 처리. RTL 내부는 `if (K==1) 1×1 path; else if (K==3) 3×3 path` 등 분기.
- **C**: 겉 API는 `conv_backend_run_int8(params, ...)` 한 개. 백엔드를 SW(기존 conv2d_int8) / HW(가속기 호출)로 갈아끼움. 1×1_one_pixel, 3×3_one_pixel은 SW 백엔드 구현으로 유지.

---

## 8. 가속기 설계 확정 사항 (코드 기반, HW/DMA 계약)

CPU 후처리(dequant_acc32_to_float, BN, SiLU)와 맞추기 위해 아래 세 가지를 **코드와 동일하게** 확정한다.

### 8.1 출력 레이아웃 (DDR 주소 순서) — 확정

- **채택**: **NCHW, ow가 가장 안쪽(연속)**.  
  한 배치(b) 기준: `oc → oh → ow` (ow가 최하위 차원).
- **Linear 인덱스 (n=1 기준)**:
  ```text
  idx = (oc * OH + oh) * OW + ow
  ```
- **근거 코드**:
  - `conv2d_int8.c` 54–55행:  
    `out_idx = b*(OC*OH*OW) + oc*(OH*OW) + oh*OW + ow`
  - `quant_util.c` dequant_acc32_to_float:  
    `oc = (i / (out_h*out_w)) % out_c` 로 읽으므로, 동일하게 **oc major → oh → ow** 가정.
- **DMA**: 연속 쓰기는 ow 방향이므로 burst에 유리. 가속기는 이 순서로만 DDR에 쓴다.

### 8.2 라인버퍼 윈도우 정의 — 확정

- **채택**: **한 윈도우 = 한 출력 픽셀 후보 (oh, ow) 하나에 대한 K×K 입력**.
  - 즉, (oh, ow) 한 위치마다, 그 위치를 계산하는 데 필요한 **전체 IC에 대한 K×K**가 하나의 윈도우(또는 IC 타일 단위로 잘린 윈도우).
  - 여러 ow를 한 번에 묶는 “가로 스트립” 형태는 **초기 설계에서 사용하지 않음**.
- **근거 코드**:  
  `conv2d_int8_one_pixel()`이 **(b, oc, oh, ow) 한 점**에 대해 ic, kh, kw 루프로 한 개의 `sum`을 계산. 즉 SW는 “한 위치(oh, ow) = 한 출력 픽셀 후보” 단위.
- **효과**: 한 윈도우당 한 출력 픽셀 대응이라 디버깅·golden 비교가 단순함.  
  (나중에 ow 여러 개를 묶어서 throughput 올리는 것은 선택 확장.)

### 8.3 IC 타일링 누적 방식 — 확정

- **채택**:
  1. **acc는 레지스터에 유지**  
     - 출력 픽셀(또는 OC_TILE 개수)만큼의 int32 acc를 레지스터에 두고, IC 타일을 **순차**로 돌면서 누적.
  2. **IC 타일은 순차 누적**  
     - 예: IC=256, IC_TILE=8 → 한 (oh, ow)당 32번, IC 타일을 순서대로 읽어서 같은 acc에 누적.
  3. **OC 방향**  
     - 첫 설계: **OC_TILE=1** (한 번에 한 출력 채널)로 단순화 가능.  
     - 확장 시: OC_TILE만큼 병렬로 여러 oc에 대한 acc를 동시에 레지스터에 두고, 같은 입력/가중치 타일로 여러 oc를 갱신(자원/DSP 허용 범위에서).
- **근거 코드**:  
  `conv2d_int8_one_pixel()` 내부가 `for (ic...) { for (kh,kw...) sum += ... }` 로 **한 sum(acc)을 레지스터처럼 유지**하며 ic를 순회.  
  외부 루프는 `conv2d_int8_forward()`에서 `b → oc → oh → ow` 순서.  
  따라서 “acc 레지스터 + IC 순차 누적”이 C 시맨틱과 일치.
- **정리**:  
  - **IC**: 타일 단위 순차 누적, acc는 레지스터.  
  - **OC**: 1부터 시작, 필요 시 OC_TILE 병렬 확장.  
  이렇게 하면 HW가 C와 동일한 출력 레이아웃과 수치 시맨틱을 유지할 수 있다.

### 8.4 보수적 설정 (리소스 세이프 모드, Artix-7 등 소형 FPGA)

- **OC_TILE**: 1차 목표 **32** (P_LANES=4면 OC_GROUPS=8). 제어 단순, weight 용량 관리 가능.
- **k=6 유지 시 위험**:  
  한 run에서 IC 전체를 누적하려면 weight를 **IC 전체**에 대해 BRAM에 들고 있어야 함.  
  `bytes_per_bank = OC_GROUPS × IC × (k*k)` (int8 1바이트).  
  OC_GROUPS=8, IC=256(MAX_IC), k=6 → kk=36 → 8×256×36 = **73,728 bytes ≈ 72KB** (bank 1개).  
  bank 4개면 **288KB**. Artix-7 XC7A35T 블록 RAM은 총 약 **225KB** 수준이라, k=6 지원 시 BRAM에 안 들어감.
- **권장 보수적 설정**  
  - **런타임**: OC_TILE ≤ 32, **k ∈ {1, 3}만 지원** (6은 일단 제외).  
    YOLO 계열은 1×1/3×3이 대부분이고, 6×6은 Layer0 등 소수만 사용 → 구조 검증 우선 시 1/3만으로도 충분.  
  - **컴파일 타임** (BRAM/합성 절약):  
    - MAX_OC_TILE: 64 → **32**  
    - MAX_K, K_MAX: 6 → **3**  
    지원 범위를 줄이면 합성 시 최대치 기준 BRAM/LUT가 바로 줄어듦.

### 8.5 공간 타일 크기 및 acc_buf (Artix-7 35T 기준)

- **공간 타일** = 한 번에 처리하는 출력 픽셀(윈도우) 개수. acc_buf = 타일 수 × OC_TILE × 4(int32) 바이트.
- **추천 (1차 목표)**  
  - **8×8 (= 64 win)** → acc_buf = 64×32×4 = **8KB**.  
    구조 단순, 35T에서 여유 있음. 먼저 이걸로 파이프라인/정확도 검증.
  - 16×8 (128 win) → 16KB. 8×8 검증 후 BRAM 여유 있으면 선택.
  - 16×16 (256 win) → 32KB. 35T에서는 부담 크므로 비추.
- **OC_TILE 최대**: **32** 유지. 64는 acc_buf·weight staging·배선이 모두 커져 35T에서 위험.
- **정리**: 1차는 **공간 타일 8×8, OC_TILE 최대 32**로 고정해 두고, 합성/리소스 보고 16×8 등은 나중에 옵션으로 두는 게 안전함.

### 8.6 한 run 흐름 및 호스트 스트림 규칙

- **한 번 start(run) 시 내부 동작**  
  1. **acc_buf 전부 0으로 초기화**  
  2. **ic_slice = 0 .. (IC/IC_TILE - 1)** 반복 (예: IC=256, IC_TILE=8 → 32회):  
     - **(a) weight 로드**: 이번 ic_slice에 해당하는 weight — **OC_TILE=32 전체** (oc_group 8개분)를 w_axis로 받아 weight bank에 채움.  
     - **(b) activation 로드**: 이번 ic_slice에 해당하는 activation (IC_TILE=8 ch)을 s_axis로 받음 → linebuf_window가 **8×8 윈도우 64개** 생성 → tile_buf에 저장.  
     - **(c) oc_group 루프(8회)**: tile_buf를 재생(replay)하면서 MAC 수행, 결과를 **acc_buf에 += 누적**.  
  3. 마지막 ic_slice까지 끝나면 **acc_buf(최종 int32)**를 AXI-Stream으로 쭉 출력 → DMA가 **DDR에 write**.  
  4. CPU는 DDR에서 읽어서 **dequant** (scale_in×scale_w×acc32 + bias → float) 등 후처리 수행.  

- **DDR에는 partial이 아니라 최종 int32만 저장** (한 run당 한 공간 타일 8×8×OC_TILE 개의 int32).

- **호스트가 반드시 맞춰줘야 하는 스트림 규칙**  
  - **(1) activation 스트림 (s_axis)**  
    - **IC 슬라이스마다** 한 번씩 들어와야 함. IC=256이면 256/8=**32번**, 매번 **8채널** activation 타일을 **정해진 순서**로 전송.  
    - 각 슬라이스 입력은 linebuf_window가 **8×8 결과(64 window)**를 만들 수 있을 만큼의 범위/양이어야 함.  
    - **s_axis_tlast**는 타일 끝에서 1로 보내는 것이 제어상 가장 단순.  
  - **(2) weight 스트림 (w_axis)**  
    - **IC 슬라이스마다** 한 번씩 들어와야 함.  
    - 매 ic_slice마다 **OC_TILE=32 전체**(oc_group 8개)에 해당하는 weight가 스트림으로 들어와 weight bank에 채워져야 함.  
    - 순서: **oc → ic → kh → kw** (kw 연속). 기존 “스트림 순서” 계약과 동일.
