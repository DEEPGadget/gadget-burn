# gadget_burn

**NVIDIA(cuBLAS) 및 AMD(rocBLAS)** 멀티-GPU GEMM 기반 GPU Burn-in / 성능·전력 측정 도구.

단일 소스를 두 벤더로 빌드하며([아키텍처](docs/architecture.md)), 실제 코어 수와 실시간 클럭으로 계산한 **동적 Peak%**로 각 GPU가 설계 성능을 내는지 한눈에 보여줍니다. [gpu-burn](https://github.com/wilicc/gpu-burn)에서 영감을 받았습니다.

## 목적

GPU 서버 납품 직후나 장시간 연속 운전 전에 **모든 GPU가 설계 성능을 유지하는지 검증**할 때 사용합니다.

- **납품 수락 검사**: 실측 클럭 기반 Peak%로 GPU 상태를 즉시 정량 확인
- **안정성 Burn-in**: 수 시간 연속 가동해 열적·전기적 이상 조기 발견
- **냉각 검증**: 전력·온도 실시간 추적으로 액냉/공냉 성능 확인
- **성능 회귀 탐지**: 드라이버 업그레이드·파워캡 변경 후 정량 비교

모든 GPU의 적정 성능을 일일이 찾아 비교하는 번거로움을 없애기 위해, 실제 ALU 수와 실시간 클럭으로 "원래 나왔어야 할 성능"과 실측 성능을 자동 비교합니다. GPU 구조에 익숙하지 않아도 정상 여부를 쉽게 판단할 수 있습니다.

## 빠른 시작

```bash
# 빌드 (백엔드 자동 선택)
make            # NVIDIA (기본)
make amd        # AMD (빌드 머신 GPU 자동 감지)

# 실행 (기본: 전체 GPU, 100% VRAM, 3600초)
./gadget_burn

# GPU 목록 확인
./gadget_burn -l
```

> NVIDIA는 CUDA Toolkit + cuBLAS + NVML, AMD는 ROCm + rocBLAS + amd_smi가 필요합니다.
> 빌드 모드·아키텍처 이식성(fatbin/generic)·공유 FS 주의사항은 **[빌드 가이드](docs/build.md)** 참고.

## 실행 예시

```
╔══════════════════════════════════════════════════════════════╗
║          gadget-burn  –  Multi-GPU GEMM Burn-in Tool         ║
╠══════════════════════════════════════════════════════════════╣
  연산 타입    : SGEMM_TF32 (FP32 storage, TF32 Tensor Core compute)
  메모리 사용  : 100% VRAM (GPU 별 자동 계산)
  GPU 당 강도  : 1 스트림
  측정 시간    : 3600 초
  사용 GPU     : 6 개  [0,1,2,3,4,5]
╠══════════════════════════════════════════════════════════════╣
  측정 시작
  [===                   ] 528/3600s  │  합계  519.834 TFLOPS  │  총  3594.9 W
   GPU    TFLOPS     Peak% (Rpeak)    전력(W)    TDP%   Util%    Clock   온도   Throt    VRAM
  ------  -------  ---------------  ---------  ------  ------  -------  -----  -----  ---------
  GPU  0   86.795   82.8%( 105T)    598.9 W   99.8%   99.7%  2734MHz   66°C    -    30906 MiB
  GPU  1   85.698   81.8%( 105T)    599.0 W   99.8%   99.8%  2702MHz   66°C    -    30906 MiB
  ...
```

> Peak%의 기준 Rpeak는 실측 클럭으로 동적 계산되므로, 클럭이 다르면 Rpeak도 달라집니다.

## 주요 특징

- **멀티벤더 단일 소스** — NVIDIA/AMD를 추상화 레이어 하나로 지원. → [아키텍처](docs/architecture.md)
- **실측 클럭 기반 동적 Peak%** — 고정 스펙값이 아니라 실시간 클럭 × 코어 수로 이론 피크를 계산. 🟢≥70% / 🟡60~69% / 🔴<60%. → [측정 방법](docs/measurement.md)
- **정확한 GPU별 측정** — 멀티-GPU에서 Util 교차오염을 우회(NVIDIA), device 핸들을 PCIe BDF로 매칭(AMD). → [측정 방법](docs/measurement.md)
- **메모리 압박 패턴** — 가용 VRAM을 채우는 multi-C ring buffer + random 데이터로 메모리 컨트롤러까지 풀로드 (gpu-burn 패턴, 항상 활성).
- **부하 자유 조절** — VRAM 비율/행렬 크기(`-m`, `-X`)와 동시 스트림 수(`-i`)로 부하를 정밀 조절.

## 연산 타입 (`-p`)

| 옵션 | 연산 |
|---|---|
| `sgemm` | FP32 (shader, 매트릭스 미사용) |
| `dgemm` | FP64 |
| `hgemm` | FP16 in / FP16 acc |
| `hgemm_mix` | FP16 in / FP32 acc (mixed precision) |
| `sgemm_tf32` | FP32 storage + TF32 compute |

**기본값은 백엔드별로 다릅니다**: NVIDIA=`sgemm_tf32`(gpu-burn -tc 호환), AMD=`hgemm_mix`(Matrix Core 네이티브 고속 경로). 정밀도별 Rpeak 계산과 NVIDIA Tensor Core / AMD RDNA·CDNA OPS 체계는 → **[정밀도와 Rpeak](docs/precision.md)**

## 실행 파라미터

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `-t <초>` | 총 측정 시간 | `3600` |
| `-i <강도>` | GPU 당 동시 GEMM 스트림 수 | `1` |
| `-p <타입>` | `sgemm`/`dgemm`/`hgemm`/`hgemm_mix`/`sgemm_tf32` | 백엔드별 |
| `-m <값>` | 메모리 사용량 (`80%` 또는 `8192`) | `100%` |
| `-g <목록>` | 사용할 GPU ID (쉼표 구분) | 전체 |
| `-X <크기>` | 행렬 크기 M override (`8192`/`16384`/`32768`) | `16384` |
| `-I <모드>` | A,B 데이터 초기화: `memset`/`rand` | `rand` |
| `-l` | GPU 목록 출력 후 종료 | — |
| `-h` | 도움말 | — |

```bash
# 정밀도 선택
./gadget_burn -p hgemm                 # FP16 in/acc (이론 피크 최대)
./gadget_burn -p sgemm                 # FP32 shader (TC 미사용)

# 행렬 크기 / VRAM 조절
./gadget_burn -X 8192                   # 작은 행렬 (gpu-burn 정확 모사)
./gadget_burn -m 50%                    # VRAM 50% 만 사용

# 데이터 entropy 비교 (TDP 영향 분리 측정)
./gadget_burn -I memset                 # denormal byte 0x01 (switching 최소)
./gadget_burn -I rand                   # xorshift PRNG (기본, switching 풀가동)

# 운영
./gadget_burn -g 0,1 -i 4 -t 1800      # GPU 0,1, 스트림 4개, 30분
```

## 문서

| 문서 | 내용 |
|---|---|
| [아키텍처](docs/architecture.md) | 멀티벤더 백엔드 구조, 추상화 레이어, 동시성 모델 |
| [빌드 가이드](docs/build.md) | 백엔드/모드 선택, AMD arch 이식성(fatbin/generic), 공유 FS 주의 |
| [정밀도와 Rpeak](docs/precision.md) | 연산 타입, NVIDIA Tensor Core / AMD RDNA·CDNA OPS 체계, 검증 |
| [측정 방법](docs/measurement.md) | TFLOPS·Peak%·전력·Util·클럭·온도·throttle 측정 상세, 출력 컬럼 |
| [지원 GPU 목록](docs/gpu-support.md) | 내장 코어 DB 전체 (NVIDIA + AMD), 미등록 GPU 추가 방법 |

## 라이선스

MIT License
