# gadget_burn

Multi-GPU cuBLAS GEMM 기반 GPU Burn-in / 성능 측정 도구

본 프로젝트는 GPU-Burn (https://github.com/wilicc/gpu-burn) 에 영감을 받아서 만들었습니다. 

---

## 실행 예시

```
╔══════════════════════════════════════════════════════════════╗
║          gadget_burn  –  Multi-GPU GEMM Burn-in Tool         ║
╠══════════════════════════════════════════════════════════════╣
  연산 타입    : SGEMM (FP32)
  메모리 사용  : 100% VRAM (GPU 별 자동 계산)
  GPU 당 강도  : 1 스트림
  측정 시간    : 3600 초
  사용 GPU     : 6 개  [0,1,2,3,4,5]
╠══════════════════════════════════════════════════════════════╣
  [초기화] GPU 0: NVIDIA GeForce RTX 5090  TDP 600 W
           Cores  FP32=21760 / FP64=340 / Tensor=680  (tc_ops=256)
           행렬 크기 51968 x 51968,  스트림 1
           VRAM 할당: 30906 MiB
  [초기화] GPU 1: NVIDIA GeForce RTX 5090  TDP 600 W
           Cores  FP32=21760 / FP64=340 / Tensor=680  (tc_ops=256)
           행렬 크기 51968 x 51968,  스트림 1
           VRAM 할당: 30906 MiB
  ...
╠══════════════════════════════════════════════════════════════╣
  측정 시작
  [===                   ] 528/3600s  │  합계  519.834 TFLOPS  │  총  3594.9 W
   GPU    TFLOPS    Peak%(Rpeak)    전력(W)    TDP%   Util%    Clock   온도     VRAM
  ------  -------  ---------------  ---------  ------  ------  -------  -----  ---------
  GPU  0   86.795   82.8%( 105T)    598.9 W   99.8%   99.7%  2734MHz   66°C  30906 MiB
  GPU  1   85.698   81.8%( 105T)    599.0 W   99.8%   99.8%  2702MHz   66°C  30906 MiB
  GPU  2   85.489   81.6%( 105T)    599.1 W   99.8%   99.8%  2695MHz   66°C  30906 MiB
  GPU  3   85.523   81.6%( 105T)    599.0 W   99.8%   99.8%  2695MHz   67°C  30906 MiB
  GPU  4   86.376   82.4%( 105T)    599.0 W   99.8%   99.7%  2722MHz   67°C  30906 MiB
  GPU  5   85.931   82.0%( 105T)    599.9 W  100.0%   99.8%  2709MHz   68°C  30906 MiB
```

> RTX 5090 × 6 GPU, `./gadget_burn` 기본값 실행 (sgemm, 100% VRAM, 3600초)  
> Peak%의 기준 Rpeak는 현재 실측 SM 클럭으로 동적 계산되므로, GPU마다 클럭이 다르면 Rpeak 수치도 달라집니다.

---

## 목적

GPU 서버를 납품받거나 클러스터를 구성한 직후, 혹은 장시간 연속 운전 전에 **모든 GPU가 설계 성능을 유지하는지 검증**하고 싶을 때 사용합니다.

- 납품 수락 검사(Acceptance Test): 실측 클럭 기반 Peak%로 GPU 상태를 즉시 정량 확인
- 안정성 Burn-in: 수 시간 연속 가동해 열적·전기적 이상 여부 조기 발견
- 냉각 시스템 검증: 전력·온도 실시간 추적으로 액냉/공냉 성능 확인
- 성능 회귀 탐지: 드라이버 업그레이드, 파워캡 변경 후 성능 변화 정량 비교

가장 신경을 쓴 부분은, 모든 GPU의 적정 성능을 찾아 일일히 비교하는 것은 번거롭기 때문에, 실제 ALU 갯수와 실시간 클럭을 기반으로 원래 나왔어야 할 성능과 현재 실측된 성능을 비교합니다.

이를 통하여 GPU의 구조에 익숙하지 않으신 분들도, 현재 시스템의 GPU 동작의 정상 여부를 쉽게 판단할 수 있습니다. 

---

## 특징

### 부하 크기를 자유롭게 조절 (`-m`)

`-m` 옵션은 VRAM 사용 비율 또는 절대 행렬 크기를 지정합니다. 원하는 수준의 부하를 정밀하게 선택할 수 있습니다.

```bash
./gadget_burn -m 0.1%    # 아주 작은 GEMM → 낮은 GPU 부하 (프로파일링 용도)
./gadget_burn -m 10%     # 중간 크기 GEMM → 중간 부하
./gadget_burn -m 100%    # VRAM 가득 채운 GEMM → 최대 compute 부하 (기본값)
./gadget_burn -m 8192    # 절대 크기 M=N=K=8192 지정
```

부하 크기를 낮추면 메모리 대역폭 포화도는 줄어들고 compute 집약도는 높아집니다. 작은 행렬은 SM 점유율이 낮아지므로 `-i` 옵션과 함께 사용하면 원하는 강도를 유지할 수 있습니다.

### 단일 GEMM 한계를 초과하는 부하 (`-i`)

단일 대형 GEMM은 이론적으로 GPU를 포화시키지만, 실제로는 메모리 레이아웃·스케줄러·L2 캐시 충돌 등으로 인해 여유 SM이 생길 수 있습니다. `-i`로 동시 스트림 수를 늘리면 독립적인 GEMM 여러 개가 동시에 실행되어 남은 SM을 채웁니다.

```bash
./gadget_burn -i 1          # GEMM 1개 (기본)
./gadget_burn -i 4          # 동시에 4개 스트림 병렬 실행
./gadget_burn -m 30% -i 4   # 적은 메모리 × 많은 스트림 조합
```

TFLOPS 계산은 모든 스트림의 연산량을 합산하므로 `-i`를 늘려도 측정 결과가 정확하게 유지됩니다.

### Multi-GPU 동시 측정

시스템에 연결된 모든 GPU를 기본으로 사용하며, 각 GPU마다 독립 스레드로 GEMM과 모니터링을 병렬 수행합니다.

```bash
./gadget_burn              # 전체 GPU 자동 감지
./gadget_burn -g 0,2,4    # 특정 GPU만 선택
```

### 실측 클럭 기반 동적 Peak% 표시

하드코딩된 고정 Rpeak 값을 사용하지 않고, **NVML로 측정한 실제 SM 클럭**을 기반으로 그 시점의 이론 피크 성능을 동적으로 계산합니다.

```
FP32  Rpeak = FP32_cores   × 2      × clock_MHz × 1e-6  [TFLOPS]
FP64  Rpeak = FP64_cores   × 2      × clock_MHz × 1e-6  [TFLOPS]
FP16T Rpeak = Tensor_cores × 256    × clock_MHz × 1e-6  [TFLOPS]
```

고정값 방식에서는 Boost Clock이 스펙을 초과하면 Peak%가 100%를 넘어 비정상처럼 보이는 문제가 있었습니다. 동적 방식은 클럭 변동이 즉시 반영되어 "현재 클럭 대비 얼마나 효율적으로 동작하는가"를 정확히 나타냅니다.

| 색상 | 달성률 | 의미 |
|---|---|---|
| 🟢 초록 | ≥ 70% | 정상 범위 |
| 🟡 노랑 | 60~69% | 확인 권장 |
| 🔴 빨강 | < 60% | 이상 가능성 |

### 정확한 GPU별 Util 측정

`nvmlDeviceGetUtilizationRates()`는 드라이버 내부 공유 버퍼를 사용하기 때문에 멀티-GPU 환경에서 모든 GPU가 동일한 값을 반환하는 버그가 있습니다. gadget_burn은 `nvmlDeviceGetSamples()`의 GPU별 독립 링버퍼를 사용해 이 문제를 우회합니다.

### 지원 정밀도

| 옵션 | 연산 | API |
|---|---|---|
| `-p sgemm` | FP32 | `cublasSgemm` |
| `-p dgemm` | FP64 | `cublasDgemm` |
| `-p hgemm` | FP16 Tensor Core | `cublasGemmEx` + `CUBLAS_COMPUTE_16F` + `TENSOR_OP` |

---

## 빠른 시작

```bash
# 빌드
make

# 실행 (기본: 전체 GPU, sgemm, 100% VRAM, 3600초)
./gadget_burn

# GPU 목록 확인
./gadget_burn -l
```

---

## 컴파일 옵션

### 빌드 모드

```bash
make                # native (기본) — 현재 머신 GPU에 최적화된 SASS 생성
make mode=ptx       # PTX only — 빌드 머신에 GPU 없어도 컴파일 가능, 이식성 최고
make mode=fatbin    # Fat binary — sm_60~sm_90a 모든 아키텍처 SASS + PTX 포함
```

| 모드 | 특징 | 권장 상황 |
|---|---|---|
| `native` | 빌드 시점 GPU에 최적, 타 GPU 이식 불가 | 단일 환경 사용 |
| `ptx` | 런타임 JIT 컴파일, 첫 실행 수 초 지연 | 여러 GPU 아키텍처 배포 |
| `fatbin` | 빠른 첫 실행, 바이너리 크기 큼 | 배포 + 성능 모두 필요 |

### 추가 Make 타깃

```bash
make info        # 바이너리에 포함된 GPU 아키텍처 목록 출력 (cuobjdump 필요)
make dump_ptx    # PTX 텍스트 파일 추출 (gadget_burn.ptx)
make clean       # 빌드 파일 삭제
```

### 의존성

| 라이브러리 | 용도 |
|---|---|
| CUDA Runtime | GPU 메모리 관리, 스트림, 이벤트 |
| cuBLAS | GEMM 연산 |
| NVML | 전력·온도·Clock·Util 측정 |
| pthreads | GPU 별 독립 스레드 |

---

## 실행 파라미터

```
./gadget_burn [옵션]
```

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `-t <초>` | 총 측정 시간 | `3600` |
| `-i <강도>` | GPU 당 동시 GEMM 스트림 수 | `1` |
| `-p <타입>` | `sgemm` / `dgemm` / `hgemm` | `sgemm` |
| `-m <값>` | 메모리 사용량 (`80%` 또는 `8192`) | `100%` |
| `-g <목록>` | 사용할 GPU ID (쉼표 구분) | 전체 |
| `-l` | GPU 목록 출력 후 종료 | — |
| `-h` | 도움말 | — |

### 사용 예시

```bash
# 기본 실행 (전체 GPU, sgemm, 100% VRAM, 1시간)
./gadget_burn

# FP16 Tensor Core, VRAM 80%, 2시간
./gadget_burn -p hgemm -m 80% -t 7200

# GPU 0, 1번만, 스트림 4개, 30분
./gadget_burn -g 0,1 -i 4 -t 1800

# 낮은 부하 테스트 (작은 GEMM)
./gadget_burn -m 1%

# 절대 행렬 크기 지정, dgemm, 2분
./gadget_burn -m 4096 -p dgemm -t 120
```

---

## 측정 방법

### TFLOPS

GEMM 연산의 이론 FLOPs를 실제 소요 시간으로 나눠 계산합니다.

```
TFLOPS = (2 × M × N × K × intensity × 반복 횟수) / 경과 시간
```

`intensity`개 스트림이 동시에 실행되므로 모든 스트림의 연산량을 합산합니다. 각 iteration의 wall time은 **가장 오래 걸린 스트림의 이벤트 경과 시간** (`cudaEventElapsedTime`)을 사용합니다.

실시간 표시와 최종 결과는 서로 다른 기준을 사용합니다.

| 구분 | 기준 | 목적 |
|---|---|---|
| 실시간 TFLOPS | 슬라이딩 윈도우 (최근 10초) | 급격한 변화를 빠르게 반영 |
| 최종 TFLOPS | 전체 측정 구간 누적 평균 | 장기 안정성 대표값 |

슬라이딩 윈도우는 원형 버퍼(100 슬롯 × 100ms = 10초)로 구현되며, 가장 오래된 샘플이 새 샘플로 교체되면서 항상 최근 10초 구간의 평균을 유지합니다.

### 동적 Rpeak와 Peak%

Peak%를 계산하기 위한 이론 피크 성능은 하드코딩된 고정값이 아니라 **측정 시점의 실제 SM 클럭**으로 동적 계산합니다.

```
FP32  Rpeak = FP32_cores   × 2   ops/cycle × clock_MHz × 1e-6  [TFLOPS]
FP64  Rpeak = FP64_cores   × 2   ops/cycle × clock_MHz × 1e-6  [TFLOPS]
FP16T Rpeak = Tensor_cores × 256 ops/cycle × clock_MHz × 1e-6  [TFLOPS]
```

**`tc_ops = 256`** 은 Tensor Core 1개가 1 클럭 사이클에 처리하는 FP16 FMA 연산 횟수입니다. 하나의 Tensor Core는 `16×16=256`개의 FMA를 1 MMA(Matrix Multiply-Accumulate) instruction으로 처리하며, NVIDIA Ada(4th gen)와 Blackwell(5th gen) 소비자·전문가급 RTX 라인업에서 이 값은 공통으로 **256**입니다.

RTX 5090과 RTX 4090의 공식 스펙으로 검증하면 다음과 같습니다.

```
RTX 5090: 680 TC × 256 × 2407 MHz = 419.0 TFLOPS  ✓
RTX 4090: 512 TC × 256 × 2520 MHz = 330.3 TFLOPS  ✓
```

이 수치는 `CUBLAS_COMPUTE_16F` 기준(입력 FP16, 누산 FP16, dense, no sparsity)입니다. NVIDIA 스펙시트에는 동일 하드웨어에 대해 여러 수치가 병기되므로, 어떤 조건의 값인지 구분이 중요합니다.

| 정밀도 | 누산(acc) | sparsity | RTX 5090 |
|---|---|---|---|
| FP16 | FP32 | dense | 209.5 TFLOPS |
| **FP16** | **FP16** | **dense** | **419.0 TFLOPS** ← gadget_burn hgemm 기준 |
| FP16 | FP16 | sparse | 838.0 TFLOPS |
| FP8 | FP16 | sparse | 1676.0 TFLOPS |

세대 간 FP16 처리량 차이는 TC당 처리량 증가가 아니라 SM당 Tensor Core 수와 클럭의 차이에서 비롯됩니다. 예를 들어 RTX 4090(AD102)의 512 TC에서 RTX 5090(GB202)의 680 TC로 늘어난 것이 주된 요인입니다.

> FP64 코어: Ada/Blackwell 소비자급은 SM당 2개로 FP32 처리량의 1/64 수준입니다. H200 NVL은 Hopper 아키텍처 전용 고성능 Tensor Core를 탑재해 tc_ops가 다릅니다.


실시간 Peak%는 슬라이딩 윈도우 평균 클럭, 최종 결과 Peak%는 전체 구간 평균 클럭을 기준으로 합니다.

### 전력 (W)

`nvmlDeviceGetPowerUsage()`로 100ms마다 측정합니다. 반환값은 밀리와트(mW) 단위이며 와트(W)로 변환해 저장합니다. 실시간 표시는 최근 10초 슬라이딩 윈도우 평균, 최종 결과는 전체 구간 누적 평균입니다. TDP%는 `nvmlDeviceGetPowerManagementLimit()`로 조회한 TDP 대비 비율입니다.

### GPU 사용률 (Util%)

일반적으로 사용하는 `nvmlDeviceGetUtilizationRates()`는 드라이버 내부의 **전역 공유 샘플링 버퍼**를 읽기 때문에, 멀티-GPU 환경에서 여러 스레드가 동시에 호출하면 서로 다른 GPU임에도 동일한 값이 반환되는 문제가 있습니다.

gadget_burn은 `nvmlDeviceGetSamples(NVML_GPU_UTILIZATION_SAMPLES)`를 사용합니다. 이 API는 GPU handle별로 독립적인 링버퍼를 유지하며, 마지막으로 읽은 타임스탬프 이후에 쌓인 샘플만 반환하므로 GPU 간 교차 오염이 없습니다.

```
각 GPU 모니터링 스레드
  └─ nvmlDeviceGetSamples(handle, last_timestamp)
       └─ GPU별 독립 링버퍼에서 새 샘플만 추출
            └─ last_timestamp 갱신 → 다음 호출 시 중복 제외
```

실시간 표시는 최근 10초 슬라이딩 윈도우 평균, 최종 결과는 전체 구간 누적 평균입니다.

### SM Clock (MHz)

`nvmlDeviceGetClockInfo(NVML_CLOCK_SM)`으로 100ms마다 현재 SM 동작 주파수를 측정합니다. 실시간 표시는 최근 10초 슬라이딩 윈도우 평균, 최종 결과는 전체 구간 누적 평균과 종료 시점 순간값을 함께 표시합니다. 측정된 SM 클럭은 Peak% 계산에도 직접 사용됩니다.

### 온도 (°C)

`nvmlDeviceGetTemperature(NVML_TEMPERATURE_GPU)`로 100ms마다 GPU 다이 온도를 측정합니다. 온도는 항상 **현재값(실시간)**을 표시하며, 최종 결과에는 전체 구간 평균과 종료 시점 값을 함께 표시합니다.

| 온도 | 색상 | 의미 |
|---|---|---|
| < 65°C | 🟢 초록 | 정상 |
| 65~74°C | 🟠 주황 | 주의 |
| ≥ 75°C | 🔴 빨강 | 고온 경고 |

### 메모리 대역폭 추정 (TB/s)

실제 메모리 트래픽을 직접 측정하지 않고, GEMM 연산의 이론적 데이터 이동량으로 추정합니다.

```
추정 BW = bpe × 3 × M² × intensity × 반복 횟수 / 경과 시간
```

실제 L2 캐시 재사용 등을 반영하지 않으므로 참고용 상한 추정치입니다.

### 샘플링 구조 요약

```
100ms 폴링 (모니터링 스레드, GPU 당 1개)
  ├─ 전력      nvmlDeviceGetPowerUsage()
  ├─ Util%     nvmlDeviceGetSamples()         ← GPU별 독립 링버퍼
  ├─ SM Clock  nvmlDeviceGetClockInfo()        ← Peak% 계산에도 사용
  └─ 온도      nvmlDeviceGetTemperature()

GEMM 반복 (벤치 스레드, GPU 당 1개)
  └─ 소요 시간  cudaEventElapsedTime()         ← iteration별 측정

실시간 표시: 슬라이딩 윈도우 (원형 버퍼, 100슬롯 × 100ms = 최근 10초)
최종 결과:  전체 구간 누적 평균
```

---

## 출력 설명

### 실시간 출력

```
  [===                   ] 528/3600s  │  합계  519.834 TFLOPS  │  총  3594.9 W
   GPU    TFLOPS    Peak%(Rpeak)    전력(W)    TDP%   Util%    Clock   온도     VRAM
  ------  -------  ---------------  ---------  ------  ------  -------  -----  ---------
  GPU  0   86.795   82.8%( 105T)    598.9 W   99.8%   99.7%  2734MHz   66°C  30906 MiB
  GPU  5   85.931   82.0%( 105T)    599.9 W  100.0%   99.8%  2709MHz   68°C  30906 MiB
```

| 컬럼 | 측정 방법 | 표시 기준 |
|---|---|---|
| TFLOPS | `2×M×N×K×intensity×iters / time` | 슬라이딩 윈도우 (최근 10초) |
| Peak%(Rpeak) | `TFLOPS / (TC × 256 × clock) × 100` | 슬라이딩 윈도우 클럭 기준 |
| 전력(W) | `nvmlDeviceGetPowerUsage` | 슬라이딩 윈도우 (최근 10초) |
| TDP% | `전력 / TDP × 100` | 동일 |
| Util% | `nvmlDeviceGetSamples` (GPU별 독립 버퍼) | 슬라이딩 윈도우 (최근 10초) |
| Clock | `nvmlDeviceGetClockInfo(NVML_CLOCK_SM)` | 슬라이딩 윈도우 (최근 10초) |
| 온도 | `nvmlDeviceGetTemperature` | 현재값 (실시간) |
| VRAM | 초기화 시 `cudaMalloc` 합산 | 고정값 |

### 최종 결과

측정 종료 시 GPU별 상세 결과와 전체 합산을 출력합니다. TFLOPS, 전력, Util, Clock은 **전체 측정 구간 누적 평균**이며, Peak%는 전체 구간 평균 클럭 기반 동적 Rpeak와 비교합니다.

```
  GPU  0  NVIDIA GeForce RTX 5090
    ├ 행렬 크기    : 51968 x 51968,  스트림 1
    ├ 총 반복 횟수 : 1284 회
    ├ 유효 시간    : 3598.234 초
    ├ VRAM 사용    : 30906 MiB
    ├ 성능         : 86.4321 TFLOPS   82.4%  (104.8 TFLOPS @ 2710 MHz 기준)
    ├ 평균 전력    : 595.3 W  (TDP 대비 99.2%)
    ├ GPU 사용률   : 99.6%
    ├ SM Clock     : 2710 MHz 평균  (종료 시 2715 MHz)
    ├ 평균 온도    : 67.3°C  (종료 시 68°C)
    ├ 메모리 BW    : 1.247 TB/s (추정)
    └ 전력 효율    : 0.1451 TFLOPS/W
```

---

## 지원 GPU 목록

내장 코어 DB에 등록된 GPU입니다. 미등록 GPU도 전력 측정 및 성능 평가는 정상적으로 진행되나 Peak% 표시 없이 TFLOPS 절댓값만 출력합니다.

| GPU | FP32 코어 | FP64 코어 | Tensor 코어 | tc_ops |
|---|---|---|---|---|
| RTX PRO 6000 Blackwell (전 라인업) | 24,064 | 376 | 752 | 256 |
| GeForce RTX 5090 | 21,760 | 340 | 680 | 256 |
| GeForce RTX 5080 | 10,752 | 168 | 336 | 256 |
| GeForce RTX 4090 | 16,384 | 256 | 512 | 256 |
| GeForce RTX 4080 | 9,728 | 152 | 304 | 256 |
| RTX 6000 Ada | 18,176 | 284 | 568 | 256 |
| L40S | 18,176 | 284 | 568 | 256 |
| H200 NVL | 16,896 | 8,448 | 528 | 2048 |


## 라이선스

MIT License