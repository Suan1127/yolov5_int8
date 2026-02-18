# 임베디드 프로젝트 동기화 가이드

PC(YOLO_c)와 임베디드에서 **input.bin부터 chk(체크섬)가 동일**하게 나오도록 하려면 **입력 텐서 크기(shape)** 와 **바이너리 포맷**을 반드시 맞춰야 합니다. 이 문서는 (1) 입력 크기/포맷 설정 방법, (2) PC 쪽에서 적용한 수정 사항을 임베디드에 반영하는 방법을 정리합니다.

**비교 방식**: chk 비교가 어려우면 **데이터 앞부분(head)만 hex로 출력**해 같은지 확인하면 됩니다. PC는 각 텐서마다 **앞 32바이트**를 `  <name> head(32B): 3D 4D 12 ...` 형식으로 출력합니다. 임베디드에서도 동일한 범위·형식으로 출력해 비교하세요.

---

## 1. 입력 크기(Shape) 설정 — chk가 PC와 같게 나오도록

### 1.1 왜 input부터 chk가 다르게 나오나?

- **input.bin**은 **shape에 따라 바이트 수와 내용이 완전히 달라집니다.**
- PC는 보통 **N=1, C=3, H=640, W=640** 으로 전처리한 뒤 `(1,3,640,640)` shape의 bin을 씁니다.
- 임베디드에서 **다른 shape**(예: 320×320, 416×416)로 할당하거나, **파일에서 shape를 읽지 않고** 고정 크기로만 쓰면, **같은 파일을 써도 메모리 레이아웃이 달라져 chk가 달라집니다.**
- 따라서 **입력 shape를 PC와 동일하게 맞추는 것**이 필수입니다.

### 1.2 input.bin 바이너리 포맷 (PC와 동일해야 함)

| 순서 | 내용 | 타입 | 크기 |
|------|------|------|------|
| 1 | 차원 4개 | `n, c, h, w` 각각 **int32_t** (리틀 엔디언) | 16 bytes |
| 2 | 픽셀 데이터 | **float32**, NCHW 순서 (batch, channel, height, width) | n×c×h×w × 4 bytes |

- **총 파일 크기** = `16 + n * c * h * w * 4` bytes  
- 예: (1, 3, 640, 640) → 16 + 1×3×640×640×4 = **4,915,216** bytes

### 1.3 임베디드에서 해야 할 것

1. **input.bin을 PC와 동일한 방식으로 생성**
   - PC 쪽과 **같은 이미지 전처리**를 쓰려면:
     - `tools/preprocess.py` 사용, **`--size 640`** (기본값)
     - 출력: `[1, 3, 640, 640]` NCHW, float32, [0, 1] 정규화
   - 생성된 `*.bin`과 `*_meta.txt`를 임베디드에서 그대로 쓰면 shape가 맞습니다.

2. **임베디드에서 input.bin을 로드할 때**
   - **파일 앞 16바이트에서 (n, c, h, w)를 읽어서** 그 shape으로 버퍼를 할당하고, 그 다음 `n*c*h*w`개의 float32를 읽습니다.
   - **고정 shape(예: 320×320)으로만 할당하지 마세요.** PC가 640×640으로 만든 bin이면 임베디드도 640×640으로 써야 chk가 같아집니다.
   - C 쪽 참고 코드 (PC 프로젝트 `src/core/tensor.c`의 `tensor_load`):
     - 먼저 `fread(dims, sizeof(int32_t), 4, fp)` 로 `n,c,h,w` 읽기
     - `tensor_create(dims[0], dims[1], dims[2], dims[3])` 로 생성
     - 그 다음 `fread(t->data, sizeof(float), n*c*h*w, fp)` 로 데이터 읽기

3. **한 가지 크기로 통일할 때 (권장)**
   - **입력 해상도**: **640×640** (YOLOv5 기본값)
   - Shape: **N=1, C=3, H=640, W=640**
   - 메타 파일 예 (`bus_meta.txt`):
     - `Tensor shape: [1, 3, 640, 640]`
     - `Image size: 640`
   - 이렇게 맞추면 **input.bin부터 모든 레이어의 shape와 chk**를 PC와 동일하게 맞출 수 있습니다.

### 1.4 Shape는 같은데 input chk만 다를 때 (예: PC 0x41A2EE62 vs Bare-metal 0xBA363FF1)

shape가 동일한데 **input chk만 다르면** 같은 바이트 배열이 아니라는 뜻입니다. 가능한 원인과 대응:

| 원인 | 설명 | 대응 |
|------|------|------|
| **엔디언** | input.bin은 **리틀 엔디언**으로 저장됨. CPU가 빅엔디언이면 4바이트 float를 읽을 때 바이트 순서가 뒤바뀌어 값이 달라짐 → chk 달라짐. | 리틀 엔디언으로 해석하도록 로더에서 바이트 스왑, 또는 리틀엔디언 CPU 사용. |
| **다른 input.bin** | 다른 이미지·다른 전처리(예: 다른 --size)로 만든 bin을 쓰면 당연히 chk 다름. | PC와 **같은** preprocess.py·같은 이미지·같은 --size로 생성한 bin을 그대로 복사해 사용. |
| **chk 계산 시점/버퍼** | chk를 입력 로드 **완료 전**에 계산하거나, 로드한 버퍼가 아닌 다른 버퍼(미초기화 등)에 대해 계산. | 입력을 DDR/메모리에 **완전히 로드한 뒤**, 그 버퍼의 `data` 포인터로 chk 계산. |
| **float 해석** | bin은 float32. 다른 포맷으로 읽거나, 헤더(16바이트)를 건너뛰지 않고 data부터 읽으면 안 됨. | 16바이트 헤더(n,c,h,w) 다음부터 `n*c*h*w*4` 바이트를 float32로 읽기. |

PC와 동일한 input.bin 파일을 리틀 엔디언으로 읽고, **같은 버퍼**에 대해 로드 완료 후 chk를 계산하면 input chk가 일치합니다.

### 1.4.1 PC에서 input chk를 계산하는 범위 (임베디드와 동일하게 맞출 것)

PC에서는 **input.bin 파일의 헤더(16바이트)는 제외**하고, **float 데이터만** 체크섬 대상으로 씁니다.

| 항목 | PC 동작 |
|------|--------|
| **대상 버퍼** | `input->data` (텐서의 float 배열 포인터. `tensor_load`가 헤더 다음부터 읽어 넣은 영역) |
| **시작** | `input->data`의 **첫 바이트** (즉 `(uint8_t*)input->data`) |
| **끝** | `input->data` 기준 **`count * sizeof(float)` 바이트** (끝 포함하지 않음: `[0, count*4)` 구간) |
| **count** | `n * c * h * w` = `input->n * input->c * input->h * input->w` = `tensor_size(input)` |

- **포함하지 않는 것**: 파일 앞 16바이트(헤더 `n,c,h,w`). chk에는 **참여하지 않음**.
- **포함하는 것**: 헤더 다음부터 파일에 있는 **float32 데이터 전체** = 메모리에서는 `input->data`가 가리키는 **`count * 4` 바이트**.

예: shape (1,3,640,640) → count = 1×3×640×640 = 1,228,800 → **4,915,200 바이트**를 `input->data`부터 연속으로 체크섬.

```c
// PC와 동일한 범위로 chk 계산 (main.c의 print_tensor_stats_p / checksum32_bytes와 동일)
size_t count = (size_t)input->n * input->c * input->h * input->w;
uint32_t chk = checksum32_bytes(input->data, count * sizeof(float));
```

### 1.5 요약 표

| 항목 | PC (YOLO_c) | 임베디드에서 맞춰야 할 값 |
|------|-------------|---------------------------|
| 입력 shape | (1, 3, 640, 640) | **파일에서 읽은 (n,c,h,w) 사용** 또는 1,3,640,640 고정 |
| Bin 포맷 | 4×int32 (n,c,h,w) + NCHW float32 | 동일 (**리틀 엔디언**, float32) |
| 입력 생성 | `python tools/preprocess.py --size 640` | 같은 스크립트/옵션 또는 동일 규격으로 생성 |
| input.bin 크기 (640×640) | 4,915,216 bytes | 동일해야 함 |

---

## 2. PC 쪽에서 적용한 수정 사항 (임베디드에 반영용)

아래는 PC(YOLO_c) 프로젝트에서 이미 적용한 변경입니다. 임베디드에서 **동일 로직/구조**로 맞추면 연산 결과와 chk를 일치시킬 수 있습니다.

### 2.1 Conv+BN+SiLU 융합 (Fused Op)

- **목적**: Conv 결과를 DDR에 썼다가 BN·SiLU를 위해 다시 읽는 왕복을 제거해, 메모리 트래픽을 줄입니다.
- **함수**: `conv2d_fused_bn_silu_forward(conv2d_layer_t *layer, batchnorm2d_layer_t *bn, input, output)`
  - `bn == NULL`이면 Conv+SiLU만 수행 (BN이 이미 bias에 융합된 경우).
- **적용 위치**:
  - **Backbone / Head**: 모든 Conv → BN → SiLU 구간을 이 한 번의 호출로 대체.
  - **C3 블록**: cv1, cv2, cv3 각각 Conv+BN+SiLU → `conv2d_fused_bn_silu_forward` 한 번씩.
  - **SPPF 블록**: cv1, cv2 동일.
  - **Bottleneck 블록**: conv1, conv2 동일.
- **호출**: Conv+BN+SiLU 구간은 `conv2d_quant_bn_silu_forward` 한 번으로 수행합니다. (int8 전용, float 경로 제거됨)

### 2.2 연산 단위 메모리 접근 최소화

- **Conv2d**: `tensor_at`/`tensor_at_const` 제거, 포인터 순차 접근 및 1×1/3×3 전용 경로.
- **Batchnorm2d**: 채널별 `inv_std` 1회 계산, 순차 스캔.
- **SiLU**: 인라인 시그모이드, 순차 포인터 접근.
- **Concat**: 채널 슬라이스 단위 `memcpy`.
- **Pooling / Upsample**: 인덱스 최소화, 블록 단위 쓰기.

(임베디드에서 연산 코드를 PC와 동일한 소스로 맞추면 자동으로 반영됩니다.)

### 2.3 Layer0 한 점 Golden (conv_raw / bn_out / silu_out)

PC에서 `yolov5_infer bus` 실행 시 **Layer 0** 직후에 **한 점(b=0, oc=0, oh=0, ow=0)** 에 대해 다음 세 값을 **hex(float 비트 패턴)** 로 출력합니다.

| 항목 | 의미 | 임베디드와 다르면 의심할 것 |
|------|------|-----------------------------|
| **conv_raw** | Conv 누적(바이어스 포함) | 입력/가중치/인덱싱/누적 순서 |
| **bn_out** | BN 적용 후 | BN 파라미터(γ, β, mean, var, eps) 또는 BN 수식 |
| **silu_out** | SiLU 적용 후 = Layer0 첫 float | SiLU(exp/sigmoid) 구현 |

- 출력 예: `=== Layer0 one-pixel golden (b=0,oc=0,oh=0,ow=0) ===` 다음에 `conv_raw`, `bn_out`, `silu_out` 각각 `0xXXXXXXXX` (리틀 엔디언 float 비트).
- 임베디드에서 **같은 점(0,0,0,0)** 으로 위 세 단계를 같은 방식으로 찍어 Golden과 비교하면, 어느 단계부터 달라지는지 바로 알 수 있습니다.

**상세 덤프 (같은 한 픽셀)**  
Golden에서 추가로 다음을 출력합니다. 임베디드에서 동일한 (b=0, oc=0, oh=0, ow=0)으로 같은 순서로 찍어 비교하면 됩니다.

| 항목 | 의미 |
|------|------|
| **bias[0]** | Layer0 Conv의 bias[0] (hex, float 비트 LE). |
| **acc_after_ic0** | bias + ic=0 채널만 누적 (hex). |
| **acc_after_ic1** | acc_after_ic0 + ic=1 채널 누적 (hex). |
| **acc_after_ic2** | acc_after_ic1 + ic=2 채널 누적 = conv_raw (hex). |
| **w[oc=0][ic=0] 6×6 first8** | 가중치 w[0][0] 6×6 커널의 앞 8개 float (row-major), 4바이트씩 LE hex. |
| **x[b=0][ic=0] (0,0)(0,1)(1,0)(1,1)** | 입력 채널 0의 (0,0), (0,1), (1,0), (1,1) 네 점 (LE hex). payload 첫 값은 E5 E4 E4 3E ≈ 0.45. |

### 2.4 문서/메모리 최적화

- **docs/MEMORY_OPTIMIZATION.md**: DDR/임베디드 메모리 최적화 정리(융합 op, 워크스페이스 재사용, BRAM/staging/tiling 아이디어). 참고용으로 임베디드 쪽에도 복사해 두면 좋습니다.

---

## 3. 체크섬(chk) 비교 시 유의사항

- chk는 **같은 바이트 배열**이면 동일하게 나옵니다.
- **Shape가 다르면** 요소 수가 달라서 바이트 수가 달라지고, 당연히 chk도 달라집니다.
- 따라서 **input부터 chk가 다르다면** 먼저 다음을 확인하세요:
  1. input.bin을 **같은 preprocess(같은 --size, 같은 이미지)** 로 생성했는지
  2. 임베디드에서 **파일에서 읽은 (n,c,h,w)** 로 버퍼를 쓰는지, 아니면 다른 고정 크기를 쓰는지
  3. **엔디언**과 **float32** 해석이 PC와 동일한지

위를 맞춘 뒤에는 **같은 input.bin**을 쓰면 **같은 shape·같은 바이트**가 되므로, input chk부터 PC와 동일하게 맞출 수 있습니다.
