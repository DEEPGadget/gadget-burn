# 아키텍처 — 멀티벤더 백엔드 구조

gadget_burn은 단일 소스(`gadget_burn.cu`)를 **NVIDIA(CUDA/cuBLAS/NVML)** 와 **AMD(HIP/rocBLAS/amd_smi)** 양쪽으로 빌드합니다. 본체 로직(스레드 모델, 통계, 슬라이딩 윈도우, 출력, CLI)은 벤더 중립이며, 벤더 의존 부분만 얇은 추상화 레이어 뒤에 숨깁니다.

## 파일 구성

```
gadget_burn.cu        본체 (벤더 중립) — #include "gpu_backend.h"
gpu_backend.h         추상화 인터페이스 + 백엔드 선택 + 공용 타입/상수
backend_nvidia.h   ─┬ CUDA / cuBLAS / NVML 구현
core_table_nvidia.h ┘  NVIDIA GPU 코어 DB (GB_NVIDIA_CORE_ENTRIES)
backend_amd.h      ─┬ HIP / rocBLAS / amd_smi 구현
core_table_amd.h    ┘  AMD GPU 코어 DB (GB_AMD_CORE_ENTRIES)
Makefile              make nvidia / make amd 분기
```

빌드 시 `-DGB_BACKEND_NVIDIA` 또는 `-DGB_BACKEND_AMD` 매크로로 백엔드가 선택되면, `gpu_backend.h`가 해당 백엔드 헤더를 주입합니다. 두 백엔드는 서로 배타적으로 컴파일되므로 한 빌드에는 한 벤더의 구현·코어 테이블만 들어갑니다.

## 추상화 범위

추상화하는 것은 세 영역뿐이고, 나머지 ~90%의 코드는 공유됩니다.

| 영역 | NVIDIA | AMD | 추상화 방식 |
|---|---|---|---|
| **GPU Runtime** | CUDA (`cudaMalloc`, 스트림, 이벤트) | HIP (`hipMalloc` …) | 이름이 거의 동일 → backend_amd.h가 `cuda* → hip*` 매크로 별칭. 본체는 기존 `cuda*` 호출 유지 |
| **BLAS GEMM** | `cublasGemmEx` | `rocblas_gemm_ex` | 시그니처가 다름(D 출력 별도 인자 등) → `gb_gemm()` 단일 함수로 통일 |
| **모니터링** | NVML | amd_smi | 핸들 모델·단위가 다름 → `gb_mon_*` 인터페이스로 통일 (전력/온도/클럭/util/throttle) |

본체의 `monitor_thread`/`bench_thread`는 이 인터페이스만 호출하므로 벤더를 알 필요가 없습니다.

## 인터페이스 계약 (gpu_backend.h)

```c
/* BLAS — op=N/N, alpha=1, beta=0, lda=M ldb=K ldc=M 고정 */
int  gb_blas_create (gb_blas_handle_t *h, gb_stream_t stream);
void gb_blas_destroy(gb_blas_handle_t h);
int  gb_gemm(gb_blas_handle_t h, gb_prec_t prec, int M, int N, int K,
             const void *A, const void *B, void *C);

/* 모니터링 — 전력 단위는 mW 로 통일 (amd_smi 의 W 는 backend 에서 ×1000) */
int      gb_mon_init(void);
void     gb_mon_shutdown(void);
int      gb_mon_open(int dev_id, gb_mon_t *out);
unsigned gb_mon_power_mw (gb_mon_t *);
unsigned gb_mon_tdp_mw   (gb_mon_t *);
int      gb_mon_temp_c   (gb_mon_t *);   // 실패 -1
unsigned gb_mon_clock_mhz(gb_mon_t *);   // 실패 0
double   gb_mon_util_pct (gb_mon_t *);   // 실패 -1
unsigned gb_mon_throttle (gb_mon_t *);   // GB_THROTTLE_* 비트마스크
```

## 동시성 모델

GPU N개 → 스레드 2N개. 각 GPU마다 monitor와 bench가 독립 스레드로 병렬 동작하며, `GpuCtx`의 `volatile` 필드로 통신합니다(락 없는 단일 writer/single reader 패턴). 이 구조는 벤더 무관하게 공유됩니다.

```
GPU당 monitor_thread : 100ms 폴링 → gb_mon_* (전력/util/클럭/온도/throttle)
GPU당 bench_thread   : GEMM 무한 반복 → gb_gemm() + iteration 시간 측정
메인 루프            : 1초마다 실시간 표 출력
```

## 벤더별 주요 차이

추상화 뒤에서 벤더마다 다르게 처리하는 핵심 지점:

- **Util 교차오염**: NVIDIA `nvmlDeviceGetUtilizationRates()`는 멀티-GPU에서 모든 GPU가 같은 값을 반환하는 버그가 있어, GPU별 독립 링버퍼인 `nvmlDeviceGetSamples()`로 우회합니다. AMD `amdsmi_get_gpu_activity()`는 핸들별 즉시 현재값을 주므로 이 우회가 불필요 — AMD 구현이 오히려 단순합니다.
- **device 핸들 매핑**: amd_smi의 GPU enumeration 순서가 HIP device 인덱스와 일치한다는 보장이 없어, **PCIe BDF로 두 라이브러리 핸들을 매칭**합니다(안 하면 다른 GPU를 측정해 값이 엉킴).
- **TF32**: NVIDIA는 `CUBLAS_COMPUTE_32F_FAST_TF32`로 TF32 가속. AMD RDNA3은 TF32 경로가 없어 일반 f32로 폴백.
- **기본 정밀도**: NVIDIA=`sgemm_tf32`(gpu-burn -tc 호환), AMD=`hgemm_mix`(Matrix Core 네이티브 고속 경로). `gpu_backend.h`의 `GB_DEFAULT_PREC` 매크로로 분기.

자세한 정밀도·OPS 체계는 [precision.md](precision.md), 빌드·이식성은 [build.md](build.md)를 참고하세요.
