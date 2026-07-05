/**
 * gadget_burn.cu
 *
 * Multi-GPU cuBLAS GEMM 기반 GPU Burn-in / 성능 측정 도구
 *
 * 빌드:
 *   make              (native, 현재 머신 GPU 최적)
 *   make mode=ptx     (PTX only, 이식성 최고)
 *   make mode=fatbin  (sm_60~sm_90a 전체 SASS 포함)
 *
 * 실행:
 *   ./gadget_burn [옵션]
 *
 *   -t <초>       총 측정 시간                       (기본: 3600)
 *   -i <강도>     GPU 당 동시 GEMM 스트림 수         (기본: 1)
 *   -p <타입>     sgemm | dgemm | hgemm | hgemm_mix | sgemm_tf32  (기본: sgemm_tf32)
 *                 hgemm      → FP16 in / FP16 acc Tensor Core (피크 TFLOPS 최대)
 *                 hgemm_mix  → FP16 in / FP32 acc Tensor Core (mixed precision)
 *                 sgemm_tf32 → FP32 storage + TF32 Tensor Core (기본, gpu-burn -tc 호환)
 *   -m <값>       메모리 사용량
 *                   숫자   : M=N=K 행렬 크기         (예: -m 8192)
 *                   숫자%  : VRAM 대비 비율          (예: -m 80%)
 *                 (기본: 100%)
 *   -g <목록>     GPU ID 쉼표 구분                  (기본: 전체)
 *   -X <크기>     행렬 크기 M override (기본 32768, opt: 8192/16384)
 *   -I <모드>     데이터 초기화: memset (denormal) | rand (xorshift PRNG, 기본)
 *   -P <와트>     전력 캡(TDP) 설정 [W] (root 필요, 종료 시 원래 캡 복원)
 *   -o [경로]     측정 데이터를 CSV로 기록 (results_host_timestamp.csv, 그래프용)
 *   -S <plan>     TDP 스윕: W:T[,W:T]... 여러 (전력캡,시간) 조건 연속 실행 (root 필요)
 *                 예: -S 300:10m,400:10m,600:10m  (메모리·데이터타입은 전 구간 고정)
 *   -l            GPU 목록 출력 후 종료
 *   -h            도움말
 *
 *   기본 동작: C 행렬을 ring buffer로 다수 슬롯 할당해 가용 VRAM을 채움,
 *             매 iteration마다 다른 슬롯에 write → 메모리 트래픽 분산.
 *             gpu-burn 패턴의 핵심을 default로 채택.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <signal.h>
#include <ctype.h>
#include <sys/stat.h>

/* 벤더 추상화 레이어. 빌드 시 -DGB_BACKEND_NVIDIA 또는 -DGB_BACKEND_AMD 로
   백엔드를 선택하면, 이 헤더가 CUDA/cuBLAS/NVML 또는 HIP/rocBLAS/amd_smi
   구현(타입·함수·매크로)을 주입합니다. 본체는 벤더 SDK 헤더를 직접
   포함하지 않습니다. (GPU 커널의 __half 등 fp16 타입도 backend 헤더가 제공) */
#include "gpu_backend.h"

/* ─────────────────────────────────────────────────────────
   Burn-pressure compare kernels (-x 모드 전용)

   gpu-burn 영감: 다중 C 버퍼 ring을 read하면서 atomicAdd로 카운터 갱신.
   메모리 BW를 거의 풀로드로 끌어내고 atomic contention으로 L2/NoC
   추가 부담을 만들어, Tensor Core compute와 메모리 서브시스템이
   동시에 활성화되도록 합니다.
   ───────────────────────────────────────────────────────── */

/* ─────────────────────────────────────────────────────────
   행렬 random 초기화 커널 (burn 모드 전용)

   기본 cudaMemset(buf, 1, ...)은 모든 byte를 0x01로 채워서 FP32로는 ~2.4e-38
   (denormal) 값을 가집니다. 이런 데이터로 multiplier MAC을 돌리면 underflow→0
   flush 되어 silicon 내부 bit toggle activity가 거의 0 → dynamic power 손실.

   특히 Tensor Core는 wide multiplier array라 입력 bit pattern에 따라 power가
   크게 변동합니다. gpu-burn처럼 random 0-10 데이터를 사용하면 매 곱셈마다
   bit pattern이 크게 변해 MAC 회로 풀 스위칭 → TDP ↑.

   curand 라이브러리 의존성 회피를 위해 xorshift32 PRNG로 직접 채웁니다.
   ───────────────────────────────────────────────────────── */

__device__ inline unsigned int xorshift32(unsigned int x)
{
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

extern "C" __global__ void init_random_f32(float *buf, size_t n, unsigned seed)
{
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         tid < n; tid += stride) {
        unsigned int s = (unsigned int)(tid * 2654435761u) ^ seed;
        s = xorshift32(s);
        s = xorshift32(s);
        /* [0, 10.0) range — gpu-burn 패턴과 일치 */
        buf[tid] = (float)s * (10.0f / 4294967296.0f);
    }
}

extern "C" __global__ void init_random_f64(double *buf, size_t n, unsigned seed)
{
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         tid < n; tid += stride) {
        unsigned int s1 = (unsigned int)(tid * 2654435761u) ^ seed;
        s1 = xorshift32(s1);
        s1 = xorshift32(s1);
        unsigned int s2 = xorshift32(s1 ^ 0xDEADBEEFu);
        /* 53-bit precision for mantissa — 두 32비트 결합 */
        double hi = (double)s1 / 4294967296.0;
        double lo = (double)s2 / (4294967296.0 * 4294967296.0);
        buf[tid] = (hi + lo) * 10.0;
    }
}

extern "C" __global__ void init_random_f16(__half *buf, size_t n, unsigned seed)
{
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         tid < n; tid += stride) {
        unsigned int s = (unsigned int)(tid * 2654435761u) ^ seed;
        s = xorshift32(s);
        s = xorshift32(s);
        /* FP16 max ~65504. K=8192 accumulation 시 overflow 방지 위해 [0, 2.0) 사용 */
        float val = (float)s * (2.0f / 4294967296.0f);
        buf[tid] = __float2half(val);
    }
}

/* random init 호출 도우미: precision에 맞춰 커널 dispatch */

/* ─────────────────────────────────────────────────────────
   상수
   ───────────────────────────────────────────────────────── */

#define MAX_GPUS       64
#define MAX_INTENSITY  32
#define MEM_RESERVE_MB 512  /* GPU 메모리 예약 여유분 (MB) */

/* 슬라이딩 윈도우 크기: 100슬롯 × 100ms = 최근 10초 */
#define MON_WIN_SIZE   100

/* ANSI 색상 코드 */
#define CLR_RESET  "\033[0m"
#define CLR_RED    "\033[1;31m"
#define CLR_YELLOW "\033[1;33m"
#define CLR_GREEN  "\033[1;32m"
#define CLR_CYAN   "\033[1;36m"
#define CLR_WARN   "\033[0;33m"

/* ─────────────────────────────────────────────────────────
   오류 처리 매크로
   ─────────────────────────────────────────────────────────
   GPU runtime / BLAS 검사 매크로는 추상화 레이어(backend_*.h)가
   GPU_CHECK / GB_BLAS_CHECK 로 제공합니다. 본체는 기존 이름
   CUDA_CHECK / CUBLAS_CHECK 를 그대로 쓰기 위해 별칭만 둡니다.
   (모니터링은 gb_mon_* 로 추상화되어 NVML_CHECK 는 더 이상 불필요) */

#define CUDA_CHECK(call)    GPU_CHECK(call)
#define CUBLAS_CHECK(call)  GB_BLAS_CHECK(call)

/* ─────────────────────────────────────────────────────────
   GPU 코어 수 테이블 (동적 Rpeak 계산용)
   ─────────────────────────────────────────────────────────
   CoreEntry 구조체만 본체에 두고, 실제 GPU 엔트리는 백엔드별 헤더에서
   매크로로 주입됩니다:
     NVIDIA → core_table_nvidia.h (GB_NVIDIA_CORE_ENTRIES)
     AMD    → core_table_amd.h    (GB_AMD_CORE_ENTRIES)
   필드 의미와 출처는 각 헤더 주석 참고. AMD 는 Tensor Core 가 없어
   필드 의미를 재해석합니다 (cores_tensor=CU 수, tc_ops=WMMA ops/CU 등).

   동적 Rpeak 공식 (백엔드 무관):
     FP32  = cores_fp32   × 2 × clock_MHz × 1e-6  [TFLOPS]
     FP64  = cores_fp64   × 2 × clock_MHz × 1e-6  [TFLOPS]
     FP16T = cores_tensor × tc_ops × clock_MHz × 1e-6  [TFLOPS]

   고정값 대신 실측 Boost Clock에 비례하므로 클럭 하락/상승이
   즉시 Peak%에 반영됩니다. (100% 초과도 가능 → 의도된 동작)
   ───────────────────────────────────────────────────────── */

typedef struct {
    const char *substr;        /* cudaDeviceProp.name 부분 매칭 키 */
    int         cores_fp32;    /* CUDA FP32 코어 수 */
    int         cores_fp64;    /* CUDA FP64 코어 수 (0 = 미지원) */
    int         cores_tensor;  /* Tensor Core 수 */
    int         tc_ops;        /* TC 1개당 클럭당 ops, FP16 in / FP16 acc, dense */
    int         tc_ops_mix;    /* TC 1개당 클럭당 ops, FP16 in / FP32 acc, dense
                                  소비자급: tc_ops의 1/2 (HW 반속)
                                  Pro/서버급: tc_ops와 동일 (페널티 없음) */
    int         tc_ops_tf32;   /* TC 1개당 클럭당 ops, TF32 (FP32 in/out, TF32 compute), dense
                                  소비자급: tc_ops의 1/4 (이중 페널티)
                                  Pro급:    tc_ops의 1/2
                                  서버급:   tc_ops의 1/2 */
    int         fp32_matrix_ops; /* FP32 matrix 가속 ops/cycle (cores_tensor 단위당).
                                    AMD CDNA 는 Matrix Core 가 FP32 도 가속하므로
                                    sgemm 이 vector peak 를 초과 가능 → 이 값(>0)이면
                                    sgemm Rpeak 를 cores_tensor × 이 값 으로 계산.
                                    NVIDIA 와 RDNA(매트릭스 FP32 미지원)는 0 →
                                    기존 FP32 vector(cores_fp32 × 2) 기준 유지. */
} CoreEntry;

/* GPU 코어 DB. 엔트리는 백엔드별 헤더에서 매크로로 주입됩니다:
     NVIDIA → core_table_nvidia.h 의 GB_NVIDIA_CORE_ENTRIES
     AMD    → core_table_amd.h    의 GB_AMD_CORE_ENTRIES
   (각 헤더는 백엔드 헤더 backend_*.h 가 include). 두 테이블은 서로 다른
   백엔드 빌드에서만 컴파일되므로 GPU 이름 충돌이 없고, 한 빌드에 한 벤더의
   엔트리만 들어갑니다. 구체적인(긴) 이름을 먼저 배치해야 substring 매칭이
   올바르게 동작합니다. */
static const CoreEntry CORE_TABLE[] = {
#if defined(GB_BACKEND_NVIDIA)
    GB_NVIDIA_CORE_ENTRIES
#elif defined(GB_BACKEND_AMD)
    GB_AMD_CORE_ENTRIES
#endif
    /* sentinel */
    { NULL, 0, 0, 0, 0, 0, 0 }
};

/* GPU 이름으로 CoreEntry 검색 (대소문자 무시, substring 매칭) */
static const CoreEntry *core_find(const char *name)
{
    if (!name) return NULL;
    for (int i = 0; CORE_TABLE[i].substr; i++) {
        size_t nlen = strlen(CORE_TABLE[i].substr);
        for (const char *p = name; *p; p++)
            if (strncasecmp(p, CORE_TABLE[i].substr, nlen) == 0)
                return &CORE_TABLE[i];
    }
    return NULL;
}

/* ─────────────────────────────────────────────────────────
   색상 판별
   ───────────────────────────────────────────────────────── */

/* Peak%: ≥70% 초록 / 60~69% 노랑 / <60% 빨강 */
static const char *rpeak_color(double pct)
{
    if (pct >= 70.0) return CLR_GREEN;
    if (pct >= 60.0) return CLR_YELLOW;
    return CLR_RED;
}

/* edge(표면) 온도: ≥75°C 빨강 / 65~74°C 노랑 / <65°C 초록 */
static const char *temp_color_edge(double t)
{
    if (t >= 75.0) return CLR_RED;
    if (t >= 65.0) return CLR_YELLOW;
    return CLR_GREEN;
}

/* junction(hotspot) 온도: 다이 국소 최고점. burn-in 수락검사에서는 보수적으로
   80°C 부터 주의(노랑), 90°C 부터 위험(빨강)으로 경고한다.
   ≥90 빨강 / 80~89 노랑 / <80 초록 */
static const char *temp_color_junction(double t)
{
    if (t >= 90.0) return CLR_RED;
    if (t >= 80.0) return CLR_YELLOW;
    return CLR_GREEN;
}

/* ─────────────────────────────────────────────────────────
   유틸리티
   ───────────────────────────────────────────────────────── */

static inline double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ─────────────────────────────────────────────────────────
   타입 정의
   ───────────────────────────────────────────────────────── */

/* 정밀도 타입은 추상화 레이어의 gb_prec_t 를 그대로 사용합니다.
   본체 코드 호환을 위해 기존 이름(PrecType / PREC_*)을 별칭으로 둡니다. */
typedef gb_prec_t PrecType;
#define PREC_SGEMM       GB_PREC_SGEMM
#define PREC_DGEMM       GB_PREC_DGEMM
#define PREC_HGEMM       GB_PREC_HGEMM
#define PREC_HGEMM_MIX   GB_PREC_HGEMM_MIX
#define PREC_SGEMM_TF32  GB_PREC_SGEMM_TF32

/* random init dispatch helper (PrecType 정의 후) */
static void init_buffer_random(void *buf, size_t n_elem, PrecType prec,
                               unsigned seed, cudaStream_t stream)
{
    const int blocks  = 256;
    const int threads = 128;
    switch (prec) {
        case PREC_DGEMM:
            init_random_f64<<<blocks, threads, 0, stream>>>(
                (double *)buf, n_elem, seed);
            break;
        case PREC_HGEMM:
        case PREC_HGEMM_MIX:
            init_random_f16<<<blocks, threads, 0, stream>>>(
                (__half *)buf, n_elem, seed);
            break;
        default:
            init_random_f32<<<blocks, threads, 0, stream>>>(
                (float *)buf, n_elem, seed);
            break;
    }
}

typedef struct {
    int    is_percent;  /* 1 = VRAM 비율, 0 = 절대 행렬 크기 */
    double value;
} MemSpec;

/* -I 옵션: A, B 매트릭스 초기화 방식 */
typedef enum {
    INIT_MEMSET,  /* cudaMemset(1) — 모든 byte 0x01 (FP32는 ~2.4e-38 denormal) */
    INIT_RAND     /* xorshift32 PRNG로 random 데이터 (gpu-burn 패턴, 기본) */
} InitMode;

/* -S plan: 여러 (TDP, 시간) 조건을 한 실행에서 순차 진행 */
#define MAX_PHASES       64
#define PHASE_SETTLE_SEC 5   /* plan 전환 직후 평균에서 제외할 정착 구간(초) */

typedef struct {
    int tdp_w;        /* 전력 캡 [W]. (plan 항목은 항상 >0) */
    int duration_s;   /* 이 phase 지속 시간 [초] */
} Phase;

typedef struct {
    int      duration_sec;
    int      intensity;
    PrecType prec;
    MemSpec  mem_spec;
    int      gpu_ids[MAX_GPUS];
    int      num_gpus;
    int      mat_size_override; /* -X 또는 -m 절대크기: 행렬 크기 override (0이면 DEFAULT_MAT_SIZE) */
    InitMode init_mode;         /* -I: 초기화 방식 override (기본 AUTO) */
    int      tdp_cap_w;         /* -P: 전력 캡(TDP) 목표 [W]. 0 = 미설정 */
    int      csv_enable;        /* -o: CSV 기록 여부 */
    char     csv_path[512];     /* -o 인자: 출력 경로(파일/디렉터리). 빈 문자열=자동명 */
    Phase    plan[MAX_PHASES];  /* -S: TDP 스윕 plan */
    int      num_phases;        /* -S phase 개수. 0 = plan 미사용(단일 실행) */
} Config;

/* 기본 행렬 크기 (M=N=K).
   16384는 대부분의 GPU에서 메모리 트래픽과 SM 점유율을 충분히 확보하면서도
   초기화 시간과 VRAM 부담이 과하지 않은 균형점. 매우 큰 GPU(RTX 5090,
   PRO 6000 SE, H200 등)에서 더 강한 부하가 필요하면 -X 32768 로 키울 수 있고,
   작은 GPU나 gpu-burn 모사에는 -X 8192 를 사용. */
#define DEFAULT_MAT_SIZE 16384

/* 단일 GEMM 스트림 워커 */
typedef struct {
    int              device_id;
    gb_blas_handle_t handle;       /* cuBLAS / rocBLAS 핸들 (추상화) */
    cudaStream_t     stream;
    cudaEvent_t    ev_start, ev_stop;
    void          *dA, *dB;
    void          *dC_ring;       /* 단일 연속 할당, num_c × c_bytes */
    int            num_c_buffers; /* C ring 슬롯 수 (non-burn=1) */
    int            cur_c_idx;     /* 다음에 write할 C 슬롯 */
    int            M, N, K;
    PrecType       prec;
} GemmWorker;

/* GPU 당 컨텍스트 */
typedef struct {
    /* 설정 */
    int         device_id;      /* 물리(런타임) device id — cudaSetDevice/모니터링용 */
    int         logical_id;     /* 논리 인덱스 = BDF 오름차순(-g 입력·표시용) */
    int         gpu_index;
    char        name[96];       /* cudaDeviceProp.name 캐시 (CSV/메타용) */
    int         intensity;
    PrecType    prec;
    int         mat_size;
    size_t      vram_used_mb;

    /* 모니터링 핸들 (NVML / amd_smi 추상화) */
    gb_mon_t     mon;
    unsigned int tdp_mw;

    /* -P 전력 캡(TDP) 설정 상태 */
    unsigned int orig_cap_mw;   /* 변경 전 원래 전력 캡 (복원용) */
    int          cap_applied;   /* 1 = 이 GPU에 캡을 적용함 → 종료 시 복원 */

    /* 코어 정보 (동적 Rpeak 계산용) */
    CoreEntry   core_info;

    /* 모니터링 스레드 */
    pthread_t    mon_tid;
    volatile int mon_running;

    /* 전체 누적 (최종 결과용) */
    double       sum_power;
    double       sum_util;
    double       sum_temp_edge;   /* edge(표면) 온도 누적 */
    double       sum_temp_hot;    /* junction(hotspot) 온도 누적 (지원 GPU만) */
    double       sum_clock;
    long         mon_samples;
    long         util_samples;
    long         temp_hot_samples;   /* junction 유효 샘플 수 (N/A GPU 는 0 유지) */
    long         throttle_thermal_samples;     /* SW/HW thermal slowdown 감지 횟수 */
    long         throttle_powerbrake_samples;  /* HW power brake 감지 횟수 */

    /* phase 경계 baseline 스냅샷 (per-phase 통계 = 현재 − baseline). plan(-S)용.
       리셋 대신 스냅샷이라 모니터/벤치 스레드와의 race 없이 phase별 평균 산출. */
    double       base_sum_power, base_sum_util, base_sum_clock;
    double       base_sum_temp_edge, base_sum_temp_hot;
    long         base_mon_samples, base_util_samples, base_temp_hot_samples;
    long         base_throttle_thermal, base_throttle_power;
    long         base_total_iters;
    double       base_total_gpu_ms;

    /* 슬라이딩 윈도우 (실시간 표시용, 원형 버퍼) */
    double       win_power[MON_WIN_SIZE];
    double       win_util[MON_WIN_SIZE];
    double       win_clock[MON_WIN_SIZE];
    int          win_util_valid[MON_WIN_SIZE];
    int          win_head;
    int          win_count;

    volatile int          cur_temp_edge;   /* 현재 edge 온도(°C) */
    volatile int          cur_temp_hot;    /* 현재 junction 온도(°C), -1=N/A */
    volatile unsigned int cur_util;
    volatile unsigned int cur_clock;
    volatile unsigned int cur_throttle;  /* GB_THROTTLE_* 비트마스크 (벤더 중립) */

    /* 벤치 스레드 */
    GemmWorker  *workers;
    pthread_t    bench_tid;
    volatile int bench_running;
    long         total_iters;   /* 전체 누적 (최종 결과용) */
    double       total_gpu_ms;

    /* TFLOPS 슬라이딩 윈도우 */
    double       win_iter_ms[MON_WIN_SIZE];
    int          tflops_head;
    int          tflops_count;
} GpuCtx;

/* ─────────────────────────────────────────────────────────
   C ring 슬롯 수 계산

   A와 B는 worker(intensity)당 1개씩만 공유, C는 ring으로 가용 VRAM을 채움.
   slots = (target_vram - intensity*(A+B)) / c_bytes
   최소 4, 최대 256 슬롯으로 clamp.
   ───────────────────────────────────────────────────────── */
static int calc_c_buffers(int device_id, MemSpec spec,
                          PrecType prec, int intensity, int M)
{
    CUDA_CHECK(cudaSetDevice(device_id));
    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    size_t reserve = (size_t)MEM_RESERVE_MB * 1024 * 1024;
    size_t usable  = (free_b > reserve) ? free_b - reserve : free_b;
    size_t target  = spec.is_percent
                       ? (size_t)(usable * (spec.value / 100.0))
                       : usable;

    size_t bpe;
    switch (prec) {
        case PREC_DGEMM:     bpe = 8; break;
        case PREC_HGEMM:
        case PREC_HGEMM_MIX: bpe = 2; break;
        default:             bpe = 4; break;
    }

    size_t c_bytes  = bpe * (size_t)M * (size_t)M;
    size_t ab_bytes = 2 * c_bytes * intensity;
    if (target <= ab_bytes) return 4;

    size_t ring_budget = (target - ab_bytes) / intensity;
    int    n = (int)(ring_budget / c_bytes);
    if (n < 4)   n = 4;
    if (n > 256) n = 256;
    return n;
}

/* ─────────────────────────────────────────────────────────
   GemmWorker 관리
   ───────────────────────────────────────────────────────── */

static void worker_init(GemmWorker *w, int device_id,
                        int M, int N, int K, PrecType prec,
                        int num_c_buffers, int use_random_init)
{
    w->device_id = device_id;
    w->M = M; w->N = N; w->K = K;
    w->prec = prec;
    w->num_c_buffers = (num_c_buffers < 1) ? 1 : num_c_buffers;
    w->cur_c_idx = 0;

    CUDA_CHECK(cudaStreamCreate(&w->stream));

    /* BLAS 핸들 생성 + 스트림 바인딩 (+ NVIDIA는 math mode DEFAULT 고정).
       Modern API(cublasGemmEx/rocblas_gemm_ex)가 compute type을 명시
       지정하므로 핸들 math mode 는 무영향. 세부는 backend gb_blas_create. */
    if (gb_blas_create(&w->handle, w->stream) != 0) {
        fprintf(stderr, "\n[BLAS ERR] %s:%d  handle 생성 실패\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaEventCreate(&w->ev_start));
    CUDA_CHECK(cudaEventCreate(&w->ev_stop));

    size_t bpe;
    switch (prec) {
        case PREC_DGEMM:     bpe = sizeof(double); break;
        case PREC_HGEMM:
        case PREC_HGEMM_MIX: bpe = sizeof(__half);  break;
        default:             bpe = sizeof(float);   break;
    }

    size_t szA = (size_t)M * K * bpe;
    size_t szB = (size_t)K * N * bpe;
    size_t szC = (size_t)M * N * bpe;

    CUDA_CHECK(cudaMalloc(&w->dA, szA));
    CUDA_CHECK(cudaMalloc(&w->dB, szB));
    CUDA_CHECK(cudaMalloc(&w->dC_ring, szC * (size_t)w->num_c_buffers));
    CUDA_CHECK(cudaMemset(w->dC_ring, 0, szC * (size_t)w->num_c_buffers));

    /* A, B 초기화: use_random_init로 명시적 선택 (-I 옵션이 결정).
       - random: xorshift PRNG로 0~10 (FP32/FP64) 또는 0~2 (FP16) 채움
                 → bit toggle activity 확보, TC MAC silicon 풀가동
       - memset: byte 0x01 → FP32는 denormal(~2.4e-38)
                 → underflow flush, multiplier switching 최소
       두 방식의 TDP/TFLOPS 차이를 직접 비교 측정 가능. */
    if (use_random_init) {
        size_t n_a = (size_t)M * K;
        size_t n_b = (size_t)K * N;
        init_buffer_random(w->dA, n_a, prec, (unsigned)(device_id * 7919 + 1), w->stream);
        init_buffer_random(w->dB, n_b, prec, (unsigned)(device_id * 7919 + 2), w->stream);
    } else {
        CUDA_CHECK(cudaMemset(w->dA, 1, szA));
        CUDA_CHECK(cudaMemset(w->dB, 1, szB));
    }
}

static size_t worker_vram_bytes(const GemmWorker *w)
{
    size_t bpe;
    switch (w->prec) {
        case PREC_DGEMM:     bpe = 8; break;
        case PREC_HGEMM:
        case PREC_HGEMM_MIX: bpe = 2; break;
        default:             bpe = 4; break;
    }
    /* A + B + (C × num_c_buffers) */
    size_t one_mat = bpe * (size_t)w->M * (size_t)w->N;
    return one_mat * (2 + (size_t)w->num_c_buffers);
}

static void worker_free(GemmWorker *w)
{
    cudaFree(w->dA);
    cudaFree(w->dB);
    cudaFree(w->dC_ring);
    cudaEventDestroy(w->ev_start);
    cudaEventDestroy(w->ev_stop);
    gb_blas_destroy(w->handle);
    cudaStreamDestroy(w->stream);
}

/* ─────────────────────────────────────────────────────────
   GEMM 실행
   : sgemm/dgemm → 표준 cuBLAS API
   : hgemm       → GemmEx + COMPUTE_16F + TENSOR_OP (Tensor Core 강제)
   ───────────────────────────────────────────────────────── */

/* burn 모드에서 현재 슬롯의 C 포인터를 계산.
   non-burn (num_c_buffers=1)이면 dC_ring 그대로. */
static void *current_c_ptr(const GemmWorker *w)
{
    size_t bpe;
    switch (w->prec) {
        case PREC_DGEMM:     bpe = sizeof(double); break;
        case PREC_HGEMM:
        case PREC_HGEMM_MIX: bpe = sizeof(__half);  break;
        default:             bpe = sizeof(float);   break;
    }
    size_t one = bpe * (size_t)w->M * (size_t)w->N;
    return (char *)w->dC_ring + (size_t)w->cur_c_idx * one;
}

static void run_gemm(GemmWorker *w)
{
    const int M = w->M, N = w->N, K = w->K;
    void *dC = current_c_ptr(w);

    /* GEMM 실행은 추상화 레이어 gb_gemm() 로 통일.
       op=N/N, alpha=1, beta=0, lda=M ldb=K ldc=M 고정.
       정밀도별 data/compute type 매핑은 backend 가 담당:
         NVIDIA → cublasGemmEx (COMPUTE_32F / 64F / 16F / 32F_FAST_TF32 ...)
         AMD    → rocblas_gemm_ex (f32_r / f64_r / f16_r, TF32는 f32 폴백)
       TF32 가속 효과나 multi-C ring 메모리 부하는 호출 의미와 무관하게
       유지됩니다 (call API만 추상화, 부하 패턴은 본체 로직 그대로). */
    if (gb_gemm(w->handle, w->prec, M, N, K, w->dA, w->dB, dC) != 0) {
        fprintf(stderr, "\n[BLAS ERR] %s:%d  gb_gemm 실패\n", __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    /* ring 슬롯 advance */
    w->cur_c_idx = (w->cur_c_idx + 1) % w->num_c_buffers;
}

/* ─────────────────────────────────────────────────────────
   모니터링 스레드 (GPU 당 1개, 100ms 폴링)
   ─────────────────────────────────────────────────────────
   측정값은 추상화 레이어 gb_mon_* 로 읽습니다. 백엔드 차이:
     - NVIDIA: 전력 mW, util 은 GPU별 독립 링버퍼(GetSamples)로 멀티-GPU
               교차오염 우회. throttle 은 NVML reason 비트.
     - AMD:    전력 W→mW 변환, util 은 amd_smi 가 핸들별 즉시 현재값을
               주므로 교차오염 없음. throttle 은 gpu_metrics 상태.
   본체는 mW 를 받아 W 로 변환하고, throttle 은 GB_THROTTLE_* 비트로 받습니다.
   ───────────────────────────────────────────────────────── */

static void *monitor_thread(void *arg)
{
    GpuCtx *g = (GpuCtx *)arg;

    /* 첫 루프 전에 cur_* 필드를 초기화해 race condition 방지 */
    {
        double u = gb_mon_util_pct(&g->mon);
        g->cur_util  = (u >= 0.0) ? (unsigned int)u : 0;
        int te = -1, th = -1;
        gb_mon_temp2_c(&g->mon, &te, &th);
        if (te >= 0) g->cur_temp_edge = te;
        g->cur_temp_hot = th;   /* -1 이면 N/A */
        unsigned int c = gb_mon_clock_mhz(&g->mon);
        if (c > 0) g->cur_clock = c;
    }

    while (g->mon_running) {
        /* 전력 (mW → W) */
        double pw_w = 0.0;
        unsigned int pw_mw = gb_mon_power_mw(&g->mon);
        if (pw_mw > 0) {
            pw_w = pw_mw / 1000.0;
            g->sum_power += pw_w;
        }

        /* Util */
        double u = gb_mon_util_pct(&g->mon);
        if (u >= 0.0) {
            g->sum_util += u;
            g->cur_util  = (unsigned int)u;
            g->util_samples++;
        }

        /* 온도 (edge / junction 동시) */
        int te = -1, th = -1;
        if (gb_mon_temp2_c(&g->mon, &te, &th) == 0) {
            if (te >= 0) { g->sum_temp_edge += (double)te; g->cur_temp_edge = te; }
            if (th >= 0) { g->sum_temp_hot  += (double)th; g->cur_temp_hot  = th;
                           g->temp_hot_samples++; }
            else g->cur_temp_hot = -1;
        }

        /* SM/GFX Clock */
        double clk_d = 0.0;
        unsigned int clk = gb_mon_clock_mhz(&g->mon);
        if (clk > 0) {
            clk_d = (double)clk;
            g->sum_clock += clk_d;
            g->cur_clock  = clk;
        }

        /* Throttle (벤더 중립 GB_THROTTLE_* 비트마스크).
           SW Power Cap 등 의도된 상태는 backend 에서 제외하고 thermal /
           power brake 만 보고합니다. */
        unsigned int reasons = gb_mon_throttle(&g->mon);
        g->cur_throttle = reasons;
        if (reasons & GB_THROTTLE_THERMAL)
            g->throttle_thermal_samples++;
        if (reasons & GB_THROTTLE_POWER_BRAKE)
            g->throttle_powerbrake_samples++;

        g->mon_samples++;

        /* 슬라이딩 윈도우 갱신 (원형 버퍼) */
        int idx = g->win_head;
        g->win_power[idx]      = pw_w;
        g->win_clock[idx]      = clk_d;
        g->win_util[idx]       = (u >= 0.0) ? u : 0.0;
        g->win_util_valid[idx] = (u >= 0.0) ? 1 : 0;
        g->win_head  = (idx + 1) % MON_WIN_SIZE;
        if (g->win_count < MON_WIN_SIZE) g->win_count++;

        usleep(100 * 1000);
    }
    return NULL;
}

/* ─────────────────────────────────────────────────────────
   벤치 스레드 (GPU 당 1개)
   ───────────────────────────────────────────────────────── */

static void *bench_thread(void *arg)
{
    GpuCtx *g = (GpuCtx *)arg;
    CUDA_CHECK(cudaSetDevice(g->device_id));

    while (g->bench_running) {

        /* 모든 스트림에 GEMM 발행 */
        for (int s = 0; s < g->intensity; s++) {
            cudaEventRecord(g->workers[s].ev_start, g->workers[s].stream);
            run_gemm(&g->workers[s]);
            cudaEventRecord(g->workers[s].ev_stop,  g->workers[s].stream);
        }

        /* 모든 스트림 완료 대기 */
        for (int s = 0; s < g->intensity; s++)
            CUDA_CHECK(cudaStreamSynchronize(g->workers[s].stream));

        /* 가장 오래 걸린 스트림 시간을 이 iteration의 wall time으로 사용 */
        float max_ms = 0.f;
        for (int s = 0; s < g->intensity; s++) {
            float ms = 0.f;
            cudaEventElapsedTime(&ms, g->workers[s].ev_start, g->workers[s].ev_stop);
            if (ms > max_ms) max_ms = ms;
        }

        g->total_iters++;
        g->total_gpu_ms += max_ms;

        /* TFLOPS 슬라이딩 윈도우 갱신 */
        int idx = g->tflops_head;
        g->win_iter_ms[idx] = max_ms;
        g->tflops_head  = (idx + 1) % MON_WIN_SIZE;
        if (g->tflops_count < MON_WIN_SIZE) g->tflops_count++;
    }

    /* shader_burn_finite는 매 iter마다 자연 종료되므로 별도 stop 신호 불필요 */
    return NULL;
}

/* ─────────────────────────────────────────────────────────
   슬라이딩 윈도우 평균 (실시간 표시용)
   win_count가 0이면 cur_* 값으로 폴백
   ───────────────────────────────────────────────────────── */

static double win_avg_power(const GpuCtx *gc)
{
    if (gc->win_count == 0) return 0.0;
    double s = 0.0;
    for (int i = 0; i < gc->win_count; i++) s += gc->win_power[i];
    return s / gc->win_count;
}

static double win_avg_util(const GpuCtx *gc)
{
    if (gc->win_count == 0) return (double)gc->cur_util;
    double s = 0.0; int n = 0;
    for (int i = 0; i < gc->win_count; i++) {
        if (gc->win_util_valid[i]) { s += gc->win_util[i]; n++; }
    }
    return (n > 0) ? s / n : (double)gc->cur_util;
}

static double win_avg_clock(const GpuCtx *gc)
{
    if (gc->win_count == 0) return (double)gc->cur_clock;
    double s = 0.0;
    for (int i = 0; i < gc->win_count; i++) s += gc->win_clock[i];
    return s / gc->win_count;
}

/* ─────────────────────────────────────────────────────────
   TFLOPS 계산
   - calc_tflops_win : 슬라이딩 윈도우 기반 (실시간 표시용)
   - calc_tflops_all : 전체 누적 기반 (최종 결과용)
   ───────────────────────────────────────────────────────── */

static double calc_tflops_all(const GpuCtx *gc)
{
    if (gc->total_gpu_ms <= 0.0) return 0.0;
    double flops = 2.0 * gc->mat_size * (double)gc->mat_size
                   * gc->mat_size * gc->intensity;
    return flops * gc->total_iters / (gc->total_gpu_ms * 1e-3) * 1e-12;
}

static double calc_tflops_win(const GpuCtx *gc)
{
    if (gc->tflops_count == 0) return calc_tflops_all(gc);
    double sum_ms = 0.0;
    for (int i = 0; i < gc->tflops_count; i++)
        sum_ms += gc->win_iter_ms[i];
    if (sum_ms <= 0.0) return 0.0;
    double flops = 2.0 * gc->mat_size * (double)gc->mat_size
                   * gc->mat_size * gc->intensity;
    /* 윈도우 내 tflops_count 회 iteration이 sum_ms ms 걸린 것 */
    return flops * gc->tflops_count / (sum_ms * 1e-3) * 1e-12;
}

/* ─────────────────────────────────────────────────────────
   동적 Rpeak 계산
   현재 SM 클럭을 기반으로 이론 피크 성능을 산출합니다.
   고정 하드코딩 값과 달리 실측 클럭에 비례하므로
   Boost Clock 변동이 즉시 Peak%에 반영됩니다.

   FP32      : cores_fp32   × 2 ops/cycle       × clock_MHz × 1e-6  [TFLOPS]
   FP64      : cores_fp64   × 2 ops/cycle       × clock_MHz × 1e-6  [TFLOPS]
   FP16T     : cores_tensor × tc_ops/cycle      × clock_MHz × 1e-6  [TFLOPS]
   FP16T_MIX : cores_tensor × tc_ops_mix/cycle  × clock_MHz × 1e-6  [TFLOPS]
               소비자급: tc_ops_mix = tc_ops/2 (HW 반속)
               Pro/서버급: tc_ops_mix = tc_ops (페널티 없음)
   TF32      : cores_tensor × tc_ops_tf32/cycle × clock_MHz × 1e-6  [TFLOPS]
               소비자급: tc_ops_tf32 = tc_ops/4 (이중 페널티)
               Pro/서버급: tc_ops_tf32 = tc_ops/2
   ───────────────────────────────────────────────────────── */

static double calc_dynamic_rpeak(const GpuCtx *gc, PrecType prec, double clock_mhz)
{
    if (gc->core_info.cores_fp32 == 0 || clock_mhz <= 0.0)
        return 0.0;

    switch (prec) {
        case PREC_DGEMM:
            return gc->core_info.cores_fp64 * 2.0 * clock_mhz * 1e-6;
        case PREC_HGEMM:
            return gc->core_info.cores_tensor
                   * (double)gc->core_info.tc_ops * clock_mhz * 1e-6;
        case PREC_HGEMM_MIX:
            return gc->core_info.cores_tensor
                   * (double)gc->core_info.tc_ops_mix * clock_mhz * 1e-6;
        case PREC_SGEMM_TF32:
            /* TF32 미지원 GPU (Turing/Volta, tc_ops_tf32=0)에서는 cuBLAS가 자동으로
               FP32 SGEMM으로 fallback. 실제 workload는 FP32 shader이므로 Peak%도
               FP32 shader Rpeak 기준으로 보여줘야 정확. */
            if (gc->core_info.tc_ops_tf32 == 0)
                return gc->core_info.cores_fp32 * 2.0 * clock_mhz * 1e-6;
            return gc->core_info.cores_tensor
                   * (double)gc->core_info.tc_ops_tf32 * clock_mhz * 1e-6;
        case PREC_SGEMM:
        default:
            /* AMD CDNA 는 Matrix Core 가 FP32 도 가속하므로, BLAS(rocBLAS)가
               sgemm 을 FP32 matrix 로 올려 vector peak 를 초과할 수 있습니다.
               fp32_matrix_ops > 0 이면 matrix 기준 Rpeak 를 사용 (cores_tensor
               단위당 ops). NVIDIA/RDNA 는 0 이라 기존 FP32 vector 기준 유지. */
            if (gc->core_info.fp32_matrix_ops > 0)
                return gc->core_info.cores_tensor
                       * (double)gc->core_info.fp32_matrix_ops * clock_mhz * 1e-6;
            return gc->core_info.cores_fp32 * 2.0 * clock_mhz * 1e-6;
    }
}

/* ─────────────────────────────────────────────────────────
   실시간 진행 출력
   ───────────────────────────────────────────────────────── */

static void print_progress(int elapsed, int total,
                           int num_gpus, GpuCtx *gpus,
                           PrecType prec,
                           double total_tflops, double total_power_w)
{
    const int bar_w  = 22;
    const int filled = (total > 0) ? (int)((double)elapsed / total * bar_w) : 0;

    /* 출력 줄 수: 진행막대(1) + 헤더(1) + 구분선(1) + GPU행(N) = N+3 */
    printf("\033[%dA\r", num_gpus + 3);

    /* 진행 막대 */
    printf("  [");
    for (int i = 0; i < bar_w; i++) putchar(i < filled ? '=' : ' ');
    printf("] %3d/%3ds  │  합계 %s%8.3f TFLOPS%s  │  총 %s%7.1f W%s        \n",
           elapsed, total,
           CLR_GREEN, total_tflops, CLR_RESET,
           CLR_YELLOW, total_power_w, CLR_RESET);

    /* ── 컬럼 레이아웃 (터미널 표시 열 기준, 한글=2열)
       각 컬럼: 헤더 라벨 = 구분선 = 데이터 (표시 열 수 동일)
         GPU     :  6열  sep "------"          data "GPU  0"
         TFLOPS  :  7열  sep "-------"         data " 85.092"
         Peak%   : 13열  sep "-------------"   data " 82.8%( 105T)"
         전력     :  9열  sep "---------"       data "  580.5 W"
         TDP%    :  6열  sep "------"          data " 96.8%"
         Util%   :  6열  sep "------"          data " 98.8%"
         Clock   :  7열  sep "-------"         data "2635MHz"
         온도     :  5열  sep "-----"           data " 66°C"
         Throt   :  5열  sep "-----"           data "  -  " / "THERM" / "  PWR"
         VRAM    :  9열  sep "---------"       data "30906 MiB"   */

    printf("  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s\n",
           " GPU  ",                           /*  6열 */
           "TFLOPS ",                          /*  7열 */
           "  Peak% (Rpeak) ",                  /* 15열 */
           " \xec\xa0\x84\xeb\xa0\xa5(W) ",   /*  9열 " 전력(W) " */
           " TDP% ",                           /*  6열 */
           "Util% ",                           /*  6열 */
           " Clock ",                          /*  7열 */
           "\xec\x98\xa8\xeb\x8f\x84(e/j)",   /*  9열 "온도(e/j)" edge/junction */
           "Throt",                            /*  5열 */
           "  VRAM   ");                       /*  9열 */

    printf("  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s\n",
           "------", "-------", "---------------",
           "---------", "------", "------", "-------", "---------", "-----", "---------");

    /* GPU 행 */
    for (int g = 0; g < num_gpus; g++) {
        GpuCtx *gc = &gpus[g];

        /* 슬라이딩 윈도우 기반 실시간 값 */
        double tflops  = calc_tflops_win(gc);
        double pw      = win_avg_power(gc);
        double util    = win_avg_util(gc);
        double clk_avg = win_avg_clock(gc);
        unsigned int clk = (unsigned int)clk_avg;

        /* 현재 클럭 기반 동적 Rpeak */
        double rpeak = calc_dynamic_rpeak(gc, prec, clk_avg);

        /* Peak% + Rpeak 수치를 한 필드에 표시: " 82.8%( 105T)"
           미등록 GPU는 공백으로 채워 컬럼 너비 유지 */
        char peak_buf[18];
        if (rpeak > 0.0)
            snprintf(peak_buf, sizeof(peak_buf), "%5.1f%% (%5.0fT)",
                     tflops / rpeak * 100.0, rpeak);
        else
            snprintf(peak_buf, sizeof(peak_buf), "%13s", "");

        /* TDP%: "100.0%" = 6열 */
        char tdp_buf[8];
        if (gc->tdp_mw > 0)
            snprintf(tdp_buf, sizeof(tdp_buf), "%5.1f%%",
                     pw / (gc->tdp_mw / 1000.0) * 100.0);
        else
            snprintf(tdp_buf, sizeof(tdp_buf), "  N/A");

        /* 온도(edge/junction): 각각 색상, junction N/A(-1)면 '-'. "°"=UTF-8 2바이트
           /1열. 색 코드가 섞여 %Ns 정렬 불가 → 가시 폭 계산 후 수동 패딩(9열). */
        int e_t = gc->cur_temp_edge, h_t = gc->cur_temp_hot;
        char e_num[8], h_num[8], temp_buf[64], temp_pad[16] = "";
        snprintf(e_num, sizeof(e_num), "%d", e_t >= 0 ? e_t : 0);
        const char *ec = temp_color_edge((double)(e_t >= 0 ? e_t : 0));
        int vis;
        if (h_t >= 0) {
            snprintf(h_num, sizeof(h_num), "%d", h_t);
            const char *hc = temp_color_junction((double)h_t);
            snprintf(temp_buf, sizeof(temp_buf), "%s%s%s/%s%s%s\xc2\xb0\x43",
                     ec, e_num, CLR_RESET, hc, h_num, CLR_RESET);
            vis = (int)strlen(e_num) + 1 + (int)strlen(h_num) + 2;
        } else {
            snprintf(temp_buf, sizeof(temp_buf), "%s%s%s/-\xc2\xb0\x43",
                     ec, e_num, CLR_RESET);
            vis = (int)strlen(e_num) + 1 + 1 + 2;
        }
        { int pad = 9 - vis; if (pad < 0) pad = 0;
          for (int pi = 0; pi < pad && pi < 15; pi++) temp_pad[pi] = ' '; }

        const char *rc = (rpeak > 0.0) ? rpeak_color(tflops / rpeak * 100.0) : "";

        /* Throttle 표시 (5열): THERM (열) > PWR (전력 brake) > "  -  " (없음)
           우선순위는 thermal > power brake. SW Power Cap은 의도된 상태라 표시 안 함.
           벤더 중립 GB_THROTTLE_* 비트 기준 (backend 가 native → 공용 매핑). */
        const char *throt_str;
        const char *throt_color;
        unsigned int r = gc->cur_throttle;
        if (r & GB_THROTTLE_THERMAL) {
            throt_str   = "THERM";
            throt_color = CLR_RED;
        } else if (r & GB_THROTTLE_POWER_BRAKE) {
            throt_str   = "  PWR";
            throt_color = CLR_YELLOW;
        } else {
            throt_str   = "  -  ";
            throt_color = "";
        }

        printf("  GPU %2d  %7.3f  %s%13s%s  %7.1f W  %6s  %5.1f%%  %4uMHz  %s%s  %s%s%s  %5zu MiB\n",
               gc->logical_id,
               tflops,
               rc, peak_buf, CLR_RESET,
               pw,
               tdp_buf,
               util,
               clk,
               temp_buf, temp_pad,
               throt_color, throt_str, CLR_RESET,
               gc->vram_used_mb);
    }
    fflush(stdout);
}

/* ─────────────────────────────────────────────────────────
   per-phase 통계 (plan 모드 / 단일 실행 공용)
   ─────────────────────────────────────────────────────────
   누적값 − baseline 으로 한 phase 구간의 평균을 계산합니다. 단일 실행은
   baseline 이 0(시작) 이므로 전체 구간 평균과 동일합니다.
   ───────────────────────────────────────────────────────── */
typedef struct {
    double tflops, rpeak;
    double avg_pw, avg_util, avg_temp_edge, avg_temp_hot, avg_clk;
    int    has_hot;   /* junction 평균 유효 여부 (N/A GPU 는 0) */
    double membw, eff;
    double th_sec, pwr_sec, th_pct, pwr_pct;
    long   iters;
    double wall_s;
    long   mon_samples;
} PhaseStat;

static PhaseStat phase_stat(const GpuCtx *gc, PrecType prec, size_t bpe)
{
    PhaseStat s;
    memset(&s, 0, sizeof(s));

    long   d_mon  = gc->mon_samples  - gc->base_mon_samples;
    long   d_util = gc->util_samples - gc->base_util_samples;
    double d_pw   = gc->sum_power - gc->base_sum_power;
    double d_ut   = gc->sum_util  - gc->base_sum_util;
    double d_tpe  = gc->sum_temp_edge - gc->base_sum_temp_edge;
    double d_tph  = gc->sum_temp_hot  - gc->base_sum_temp_hot;
    long   d_hot  = gc->temp_hot_samples - gc->base_temp_hot_samples;
    double d_ck   = gc->sum_clock - gc->base_sum_clock;
    long   d_iter = gc->total_iters  - gc->base_total_iters;
    double d_ms   = gc->total_gpu_ms - gc->base_total_gpu_ms;
    long   d_thh  = gc->throttle_thermal_samples    - gc->base_throttle_thermal;
    long   d_thp  = gc->throttle_powerbrake_samples - gc->base_throttle_power;

    s.mon_samples = d_mon;
    s.iters  = d_iter;
    s.wall_s = d_ms * 1e-3;
    s.avg_pw   = (d_mon  > 0) ? d_pw / d_mon  : 0.0;
    s.avg_util = (d_util > 0) ? d_ut / d_util : 0.0;
    s.avg_temp_edge = (d_mon  > 0) ? d_tpe / d_mon  : 0.0;
    s.avg_temp_hot  = (d_hot  > 0) ? d_tph / d_hot  : 0.0;
    s.has_hot  = (d_hot > 0);
    s.avg_clk  = (d_mon  > 0) ? d_ck / d_mon  : 0.0;
    s.tflops   = (d_ms > 0)
        ? 2.0 * gc->mat_size * (double)gc->mat_size * gc->mat_size * gc->intensity
          * d_iter / (d_ms * 1e-3) * 1e-12
        : 0.0;
    s.rpeak = calc_dynamic_rpeak(gc, prec, s.avg_clk);
    s.membw = (s.wall_s > 0.0)
        ? bpe * 3.0 * gc->mat_size * (double)gc->mat_size * gc->intensity
          * d_iter / s.wall_s * 1e-12
        : 0.0;
    s.eff = (s.avg_pw > 0.1) ? s.tflops / s.avg_pw : 0.0;
    s.th_sec  = d_thh * 0.1;
    s.pwr_sec = d_thp * 0.1;
    s.th_pct  = (d_mon > 0) ? (double)d_thh / d_mon * 100.0 : 0.0;
    s.pwr_pct = (d_mon > 0) ? (double)d_thp / d_mon * 100.0 : 0.0;
    return s;
}

/* phase 시작(정착 후) baseline 스냅샷 */
static void snapshot_baseline(GpuCtx *gc)
{
    gc->base_sum_power = gc->sum_power;
    gc->base_sum_util  = gc->sum_util;
    gc->base_sum_temp_edge = gc->sum_temp_edge;
    gc->base_sum_temp_hot  = gc->sum_temp_hot;
    gc->base_temp_hot_samples = gc->temp_hot_samples;
    gc->base_sum_clock = gc->sum_clock;
    gc->base_mon_samples      = gc->mon_samples;
    gc->base_util_samples     = gc->util_samples;
    gc->base_throttle_thermal = gc->throttle_thermal_samples;
    gc->base_throttle_power   = gc->throttle_powerbrake_samples;
    gc->base_total_iters  = gc->total_iters;
    gc->base_total_gpu_ms = gc->total_gpu_ms;
}

/* ─────────────────────────────────────────────────────────
   CSV 로깅 (-o)
   ─────────────────────────────────────────────────────────
   테스트 구간 동안 per-GPU 지표를 1Hz long/tidy CSV 로 기록합니다.
   파일 상단에는 '#' 주석으로 실험 메타데이터(호스트·명령행·옵션·GPU 정보)를
   남겨 나중에 조건 파악이 쉽도록 합니다 (pandas read_csv(comment='#')·gnuplot
   가 '#' 줄 자동 무시). 본체 슬라이딩 윈도우 값만 쓰므로 백엔드 무관.
   ───────────────────────────────────────────────────────── */

static const char *prec_label(PrecType p)
{
    switch (p) {
    case PREC_DGEMM:      return "DGEMM (FP64)";
    case PREC_HGEMM:      return "HGEMM (FP16 in / FP16 acc)";
    case PREC_HGEMM_MIX:  return "HGEMM_MIX (FP16 in / FP32 acc)";
    case PREC_SGEMM_TF32: return "SGEMM_TF32 (FP32 storage / TF32 compute)";
    case PREC_SGEMM:
    default:              return "SGEMM (FP32)";
    }
}

/* -o 경로에서 최종 CSV 파일 경로 생성.
     비어있음            → 현재 디렉터리에 자동명
     기존 디렉터리 / 끝'/' → 그 안에 자동명
     그 외               → 파일명으로 그대로 사용
   자동명: results_<hostname>_<YYYYMMDD_HHMMSS>.csv */
static void csv_build_path(const Config *cfg, char *out, size_t outsz)
{
    char host[64] = "host";
    if (gethostname(host, sizeof(host)) != 0) strcpy(host, "host");
    host[sizeof(host) - 1] = '\0';
    for (char *p = host; *p; p++)
        if (!isalnum((unsigned char)*p) && *p != '.' && *p != '-' && *p != '_')
            *p = '_';

    time_t now = time(NULL);
    struct tm tmv;
    localtime_r(&now, &tmv);
    char ts[24];
    strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tmv);

    char autoname[160];
    snprintf(autoname, sizeof(autoname), "results_%s_%s.csv", host, ts);

    const char *path = cfg->csv_path;
    size_t plen = strlen(path);
    if (plen == 0) {
        snprintf(out, outsz, "%s", autoname);
        return;
    }
    struct stat st;
    int is_dir = (path[plen - 1] == '/')
                 || (stat(path, &st) == 0 && S_ISDIR(st.st_mode));
    if (is_dir) {
        const char *sep = (path[plen - 1] == '/') ? "" : "/";
        snprintf(out, outsz, "%s%s%s", path, sep, autoname);
    } else {
        snprintf(out, outsz, "%s", path);
    }
}

/* '#' 메타데이터 헤더 + CSV 컬럼 헤더 기록 */
static void csv_write_meta(FILE *f, const Config *cfg, const char *path,
                           const Phase *phases, int num_phases, int plan_mode,
                           GpuCtx *gpus, int num_gpus, int argc, char **argv)
{
    char host[64] = "unknown";
    if (gethostname(host, sizeof(host)) != 0) strcpy(host, "unknown");
    host[sizeof(host) - 1] = '\0';

    time_t now = time(NULL);
    struct tm tmv;
    localtime_r(&now, &tmv);
    char when[32];
    strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S", &tmv);

    fprintf(f, "# gadget_burn — GPU burn-in CSV log\n");
    fprintf(f, "# file        : %s\n", path);
    fprintf(f, "# generated   : %s\n", when);
    fprintf(f, "# host        : %s\n", host);
    fprintf(f, "# command     : ");
    for (int i = 0; i < argc; i++) fprintf(f, "%s%s", i ? " " : "", argv[i]);
    fprintf(f, "\n");
    fprintf(f, "# backend     : %s (%s / %s)\n",
            GB_BACKEND_NAME, GB_BLAS_NAME, GB_MON_NAME);
    fprintf(f, "# precision   : %s\n", prec_label(cfg->prec));
    if (plan_mode) {
        int total = 0;
        for (int i = 0; i < num_phases; i++) total += phases[i].duration_s;
        fprintf(f, "# duration_s  : %d (total of %d phases)\n", total, num_phases);
    } else {
        fprintf(f, "# duration_s  : %d\n", cfg->duration_sec);
    }
    fprintf(f, "# intensity   : %d stream(s)/GPU\n", cfg->intensity);
    if (num_gpus > 0)
        fprintf(f, "# matrix_size : %d (M=N=K)\n", gpus[0].mat_size);
    if (cfg->mem_spec.is_percent)
        fprintf(f, "# memory      : %.4g%% VRAM\n", cfg->mem_spec.value);
    else
        fprintf(f, "# memory      : %d x %d matrix\n",
                (int)cfg->mem_spec.value, (int)cfg->mem_spec.value);
    fprintf(f, "# init_mode   : %s\n",
            (cfg->init_mode == INIT_MEMSET) ? "memset" : "rand");
    if (plan_mode) {
        fprintf(f, "# plan        : %d phases\n", num_phases);
        for (int i = 0; i < num_phases; i++)
            fprintf(f, "# phase%-2d     : tdp=%dW duration=%ds\n",
                    i, phases[i].tdp_w, phases[i].duration_s);
    } else if (cfg->tdp_cap_w > 0) {
        fprintf(f, "# tdp_cap_w   : %d (requested)\n", cfg->tdp_cap_w);
    } else {
        fprintf(f, "# tdp_cap_w   : none\n");
    }
    fprintf(f, "# interval_s  : 1\n");
    fprintf(f, "# num_gpus    : %d\n", num_gpus);

    for (int g = 0; g < num_gpus; g++) {
        GpuCtx *gc = &gpus[g];
        char bdf[32] = "?";
        cudaDeviceGetPCIBusId(bdf, (int)sizeof(bdf), gc->device_id);
        fprintf(f, "# gpu%-2d      : id=%d \"%s\" bdf=%s",
                g, gc->logical_id, gc->name, bdf);
        if (gc->tdp_mw > 0) fprintf(f, " tdp=%uW", gc->tdp_mw / 1000);
        if (gc->core_info.cores_fp32 > 0)
            fprintf(f, " fp32=%d fp64=%d tensor=%d",
                    gc->core_info.cores_fp32, gc->core_info.cores_fp64,
                    gc->core_info.cores_tensor);
        fprintf(f, "\n");
    }

    fprintf(f, "timestamp,elapsed_sec,phase_idx,phase_tdp_w,phase_elapsed_sec,"
               "gpu_id,gpu_name,tflops,peak_pct,rpeak_tflops,"
               "power_w,tdp_pct,util_pct,clock_mhz,temp_edge_c,temp_junction_c,"
               "throttle,vram_mib\n");
}

/* 한 GPU의 현재(슬라이딩 윈도우) 값을 한 행으로 기록.
   elapsed=전체 경과초, phase_elapsed=현재 phase 내 경과초. */
static void csv_write_row(FILE *f, const char *ts, int elapsed,
                          int phase_idx, int phase_tdp_w, int phase_elapsed,
                          const GpuCtx *gc, PrecType prec)
{
    double tflops  = calc_tflops_win(gc);
    double pw      = win_avg_power(gc);
    double util    = win_avg_util(gc);
    double clk_avg = win_avg_clock(gc);
    unsigned clk   = (unsigned)clk_avg;
    double rpeak   = calc_dynamic_rpeak(gc, prec, clk_avg);

    /* rpeak/ tdp 가 0(미등록 GPU/캡 미지원)이면 빈 칸으로 → 그래프에서 NaN 처리 */
    char peak_s[24], rpeak_s[24], tdp_s[24];
    if (rpeak > 0.0) {
        snprintf(peak_s,  sizeof(peak_s),  "%.1f", tflops / rpeak * 100.0);
        snprintf(rpeak_s, sizeof(rpeak_s), "%.0f", rpeak);
    } else { peak_s[0] = '\0'; rpeak_s[0] = '\0'; }
    if (gc->tdp_mw > 0)
        snprintf(tdp_s, sizeof(tdp_s), "%.1f", pw / (gc->tdp_mw / 1000.0) * 100.0);
    else tdp_s[0] = '\0';

    const char *throt = "none";
    if (gc->cur_throttle & GB_THROTTLE_THERMAL)          throt = "thermal";
    else if (gc->cur_throttle & GB_THROTTLE_POWER_BRAKE) throt = "power";

    char ptdp_s[16];
    if (phase_tdp_w > 0) snprintf(ptdp_s, sizeof(ptdp_s), "%d", phase_tdp_w);
    else ptdp_s[0] = '\0';

    /* junction N/A(-1)면 CSV 는 빈 칸 → pandas/gnuplot 에서 NaN 처리 */
    char hot_s[12];
    if (gc->cur_temp_hot >= 0) snprintf(hot_s, sizeof(hot_s), "%d", gc->cur_temp_hot);
    else hot_s[0] = '\0';

    fprintf(f, "%s,%d,%d,%s,%d,%d,\"%s\",%.3f,%s,%s,%.1f,%s,%.1f,%u,%d,%s,%s,%zu\n",
            ts, elapsed, phase_idx, ptdp_s, phase_elapsed,
            gc->logical_id, gc->name,
            tflops, peak_s, rpeak_s, pw, tdp_s, util, clk,
            gc->cur_temp_edge, hot_s, throt, gc->vram_used_mb);
}

/* 파일 끝에 '#' 주석으로 phase별 per-GPU 평균 요약 기록 (phase 종료 시마다) */
static void csv_write_phase_summary(FILE *f, int ph, const Phase *P,
                                    const GpuCtx *gpus, const PhaseStat *ps,
                                    int num_gpus)
{
    if (P->tdp_w > 0)
        fprintf(f, "# --- summary phase%d (tdp=%dW, %ds) ---\n",
                ph, P->tdp_w, P->duration_s);
    else
        fprintf(f, "# --- summary phase%d (tdp=none, %ds) ---\n",
                ph, P->duration_s);
    for (int g = 0; g < num_gpus; g++) {
        const PhaseStat *s = &ps[g];
        char hot_avg[16];
        if (s->has_hot) snprintf(hot_avg, sizeof(hot_avg), "%.1f", s->avg_temp_hot);
        else hot_avg[0] = '\0';
        fprintf(f, "# gpu%-2d      : avg_tflops=%.3f avg_power_w=%.1f avg_util=%.1f "
                   "avg_clock_mhz=%.0f avg_temp_edge_c=%.1f avg_temp_junction_c=%s "
                   "throttle_thermal_s=%.1f throttle_power_s=%.1f total_iters=%ld\n",
                g, s->tflops, s->avg_pw, s->avg_util, s->avg_clk,
                s->avg_temp_edge, hot_avg, s->th_sec, s->pwr_sec, s->iters);
    }
}

/* phase 결과 박스 (콘솔). 단일 실행은 "최종 측정 결과" 박스, plan 은
   phase 구분 헤더로 출력 (②a). */
static void print_phase_results(int ph, int num_phases, int plan_mode,
                                const Phase *P, const GpuCtx *gpus,
                                const PhaseStat *ps, int num_gpus)
{
    if (plan_mode) {
        printf("════════════════════════════════════════════════════════════════\n");
        printf("  ▶ Phase %d/%d 결과  (목표 %dW, %d초)\n",
               ph + 1, num_phases, P->tdp_w, P->duration_s);
        printf("════════════════════════════════════════════════════════════════\n");
    } else {
        printf("╔══════════════════════════════════════════════════════════════╗\n");
        printf("║                        최종 측정 결과                        ║\n");
        printf("╠══════════════════════════════════════════════════════════════╣\n");
    }

    double grand_tflops = 0.0, grand_power = 0.0, grand_util = 0.0, grand_membw = 0.0;
    for (int g = 0; g < num_gpus; g++) {
        const GpuCtx    *gc = &gpus[g];
        const PhaseStat *s  = &ps[g];

        printf("  GPU %2d  %s\n", gc->logical_id, gc->name);
        printf("    ├ 행렬 크기    : %d x %d,  스트림 %d\n",
               gc->mat_size, gc->mat_size, gc->intensity);
        printf("    ├ 총 반복 횟수 : %ld 회\n", s->iters);
        printf("    ├ 유효 시간    : %.3f 초\n", s->wall_s);
        printf("    ├ VRAM 사용    : %zu MiB\n", gc->vram_used_mb);

        printf("    ├ 성능         : %s%.4f TFLOPS%s", CLR_GREEN, s->tflops, CLR_RESET);
        if (s->rpeak > 0.0) {
            double pct = s->tflops / s->rpeak * 100.0;
            printf("  %s%5.1f%%%s  (%.1f TFLOPS @ %.0f MHz 기준)",
                   rpeak_color(pct), pct, CLR_RESET, s->rpeak, s->avg_clk);
        }
        printf("\n");

        printf("    ├ 평균 전력    : %s%.1f W%s", CLR_YELLOW, s->avg_pw, CLR_RESET);
        if (gc->tdp_mw > 0)
            printf("  (TDP 대비 %.1f%%)", s->avg_pw / (gc->tdp_mw / 1000.0) * 100.0);
        printf("\n");

        printf("    ├ GPU 사용률   : %s%.1f %%%s\n", CLR_CYAN, s->avg_util, CLR_RESET);
        printf("    ├ SM Clock     : %.0f MHz 평균  (종료 시 %u MHz)\n",
               s->avg_clk, gc->cur_clock);
        if (s->has_hot)
            printf("    ├ 평균 온도    : edge %s%.1f°C%s / junction %s%.1f°C%s"
                   "  (종료 시 %d / %d°C)\n",
                   temp_color_edge(s->avg_temp_edge), s->avg_temp_edge, CLR_RESET,
                   temp_color_junction(s->avg_temp_hot), s->avg_temp_hot, CLR_RESET,
                   gc->cur_temp_edge, gc->cur_temp_hot);
        else
            printf("    ├ 평균 온도    : edge %s%.1f°C%s / junction N/A"
                   "  (종료 시 %d°C)\n",
                   temp_color_edge(s->avg_temp_edge), s->avg_temp_edge, CLR_RESET,
                   gc->cur_temp_edge);

        if (s->mon_samples > 0) {
            const char *th_col  = (s->th_sec  > 0.0) ? CLR_RED    : "";
            const char *pwr_col = (s->pwr_sec > 0.0) ? CLR_YELLOW : "";
            printf("    ├ Throttle    : %s열 %.1f초 (%.1f%%)%s, "
                   "%s전력 %.1f초 (%.1f%%)%s\n",
                   th_col,  s->th_sec,  s->th_pct,  (s->th_sec  > 0.0) ? CLR_RESET : "",
                   pwr_col, s->pwr_sec, s->pwr_pct, (s->pwr_sec > 0.0) ? CLR_RESET : "");
        }

        printf("    ├ 메모리 BW    : %.3f TB/s (추정)\n", s->membw);
        printf("    └ 전력 효율    : %.4f TFLOPS/W\n", s->eff);

        if (g < num_gpus - 1)
            printf("  ──────────────────────────────────────────────────────────────\n");

        grand_tflops += s->tflops;
        grand_power  += s->avg_pw;
        grand_util   += s->avg_util;
        grand_membw  += s->membw;
    }

    if (num_gpus > 1) {
        printf("  ══════════════════════════════════════════════════════════════\n");
        printf("  전체 합산 (%d GPU)\n", num_gpus);
        printf("    ★ 총 성능        : %s%.4f TFLOPS%s\n", CLR_GREEN,  grand_tflops, CLR_RESET);
        printf("    ★ 총 전력        : %s%.1f W%s\n",      CLR_YELLOW, grand_power,  CLR_RESET);
        printf("    ★ 평균 GPU 사용률: %s%.1f %%%s\n",     CLR_CYAN, grand_util / num_gpus, CLR_RESET);
        printf("    ★ 총 메모리 BW   : %.3f TB/s (추정)\n", grand_membw);
        if (grand_power > 0.1)
            printf("    ★ 시스템 전력효율: %.4f TFLOPS/W\n", grand_tflops / grand_power);
    }

    if (plan_mode)
        printf("════════════════════════════════════════════════════════════════\n\n");
    else
        printf("╚══════════════════════════════════════════════════════════════╝\n\n");
}

/* ─────────────────────────────────────────────────────────
   옵션 파싱 헬퍼
   ───────────────────────────────────────────────────────── */

static MemSpec parse_mem_spec(const char *s)
{
    MemSpec ms = { .is_percent = 1, .value = 100.0 };
    if (!s) return ms;

    size_t len = strlen(s);
    if (len > 0 && s[len - 1] == '%') {
        ms.is_percent = 1;
        ms.value      = atof(s);
        if (ms.value <= 0 || ms.value > 100) {
            fprintf(stderr, "오류: -m 비율은 0%%~100%% 사이여야 합니다.\n");
            exit(1);
        }
    } else {
        ms.is_percent = 0;
        ms.value      = (double)atoi(s);
        if (ms.value < 128) {
            fprintf(stderr, "오류: -m 절대 크기는 128 이상이어야 합니다.\n");
            exit(1);
        }
    }
    return ms;
}

static int parse_gpu_list(const char *str, int *ids, int max_n)
{
    int   n   = 0;
    char *buf = strdup(str);
    char *tok = strtok(buf, ",");
    while (tok && n < max_n) {
        ids[n++] = atoi(tok);
        tok = strtok(NULL, ",");
    }
    free(buf);
    return n;
}

/* -S plan 의 시간 토큰: 정수 + 선택적 s/m/h (기본 s). 실패 시 -1 */
static int parse_duration_token(const char *t)
{
    char *end = NULL;
    long  v   = strtol(t, &end, 10);
    if (end == t || v <= 0) return -1;
    if (*end == '\0' || *end == 's' || *end == 'S') return (int)v;
    if (*end == 'm' || *end == 'M') return (int)(v * 60);
    if (*end == 'h' || *end == 'H') return (int)(v * 3600);
    return -1;
}

/* -S plan 파싱: "W:T[,W:T]..." (W=와트, T=시간 s/m/h).
   성공 시 phase 개수, 형식 오류 시 -1. */
static int parse_plan(const char *s, Phase *plan, int max_n)
{
    char *buf = strdup(s);
    if (!buf) return -1;
    int   n    = 0;
    char *save = NULL;
    for (char *tok = strtok_r(buf, ",", &save);
         tok && n < max_n;
         tok = strtok_r(NULL, ",", &save)) {
        char *colon = strchr(tok, ':');
        if (!colon) { free(buf); return -1; }
        *colon = '\0';
        int w = atoi(tok);
        int d = parse_duration_token(colon + 1);
        if (w <= 0 || d <= 0) { free(buf); return -1; }
        plan[n].tdp_w      = w;
        plan[n].duration_s = d;
        n++;
    }
    free(buf);
    return n;
}

/* ─────────────────────────────────────────────────────────
   도움말 / GPU 목록
   ───────────────────────────────────────────────────────── */

static void print_usage(const char *prog)
{
    printf("\n사용법: %s [옵션]\n\n", prog);
    printf("  -t <초>      총 측정 시간                          (기본: 3600)\n");
    printf("  -i <강도>    GPU 당 동시 GEMM 스트림 수            (기본: 1)\n");
    printf("  -p <타입>    sgemm | dgemm | hgemm | hgemm_mix | sgemm_tf32   (기본: %s)\n",
           GB_DEFAULT_PREC_NAME);
    printf("               sgemm      → FP32 (cublasGemmEx, COMPUTE_32F)\n");
    printf("               dgemm      → FP64 (cublasGemmEx, COMPUTE_64F)\n");
    printf("               hgemm      → FP16 in / FP16 acc Tensor Core\n");
    printf("                            이론 피크 TFLOPS가 가장 높음\n");
#if defined(GB_BACKEND_AMD)
    printf("               hgemm_mix  → FP16 in / FP32 acc Matrix Core (mixed precision) [기본]\n");
    printf("                            AMD Matrix Core(WMMA/MFMA)의 네이티브 고속 경로,\n");
    printf("                            실측 TFLOPS·TDP 최대. (hgemm은 CDNA에서 저속 폴백)\n");
    printf("               sgemm_tf32 → FP32 storage + TF32 compute (RDNA3는 FP32 폴백)\n");
    printf("                            메모리 대역폭 부하 + compute 부하 동시 발생.\n");
#else
    printf("               hgemm_mix  → FP16 in / FP32 acc Tensor Core (mixed precision)\n");
    printf("                            소비자급은 hgemm의 1/2 처리량,\n");
    printf("                            Pro/서버급(A100/H100/H200)은 hgemm과 동일.\n");
    printf("               sgemm_tf32 → FP32 storage + TF32 Tensor Core compute (기본)\n");
    printf("                            CUBLAS_COMPUTE_32F_FAST_TF32, Ampere+ 필요.\n");
    printf("                            메모리 대역폭 부하 + TC 부하 동시 발생,\n");
    printf("                            gpu-burn -tc 와 동일 의도.\n");
#endif
    printf("  -m <값>      메모리 사용량\n");
    printf("                 숫자   : M=N=K 행렬 크기            (예: -m 8192)\n");
    printf("                 숫자%%  : VRAM 대비 비율             (예: -m 80%%)\n");
    printf("               (기본: -m 100%%)\n");
    printf("  -g <목록>    사용할 GPU ID (쉼표 구분)             (기본: 전체)\n");
    printf("  -X <크기>    행렬 크기 M override (기본 %d)\n", DEFAULT_MAT_SIZE);
    printf("               권장값: 8192 (gpu-burn 정확 모사),\n");
    printf("                       16384 (기본, 균형), 32768 (큰 GPU 최대 부하)\n");
    printf("               우선순위: -X > -m 절대크기 > 기본 %d\n", DEFAULT_MAT_SIZE);
    printf("  -I <모드>    A,B 데이터 초기화 방식: memset | rand   (기본: rand)\n");
    printf("                 memset: cudaMemset(1) — denormal FP32 (TC switching 최소)\n");
    printf("                 rand  : xorshift PRNG로 random 데이터 (TC switching 풀가동)\n");
    printf("               비교 측정용 — 데이터 entropy가 TDP에 미치는 영향 분리 확인\n");
    printf("  -P <와트>    전력 캡(TDP) 설정 [W] — GPU가 이 전력 안에서 운전 (root 필요)\n");
    printf("                 GPU 펌웨어가 클럭을 자동 조절. throttle에 PWR 표시는 정상.\n");
    printf("                 설정 가능 범위 밖이면 자동 클램프. 종료 시 원래 캡 복원.\n");
    printf("  -o [경로]    측정 데이터를 CSV로 기록 (그래프용, 1Hz long/tidy 형식)\n");
    printf("                 인자 없으면 ./results_<host>_<시각>.csv 자동 생성,\n");
    printf("                 디렉터리를 주면 그 안에, 파일명을 주면 그 이름으로 저장.\n");
    printf("                 파일 상단 #주석에 실험 옵션·GPU 정보 기록(pandas/gnuplot 호환).\n");
    printf("  -S <plan>    TDP 스윕 plan: 여러 (전력캡,시간) 조건을 연속 실행 (root 필요)\n");
    printf("                 형식 W:T[,W:T]... (W=와트, T=시간 s/m/h, 기본 초)\n");
    printf("                 예: -S 300:10m,400:10m,600:10m  (300W→400W→600W 각 10분)\n");
    printf("                 메모리(-m/-X)·데이터타입(-p)은 전 구간 고정. -P/-t 는 무시.\n");
    printf("                 CSV(-o) 사용 시 phase_idx/phase_tdp_w 컬럼으로 자동 구분.\n");
    printf("  -l           GPU 목록 출력 후 종료\n");
    printf("  -h           도움말\n\n");
    printf("예시:\n");
    printf("  %s                           # 기본: %s + rand, M=%d ring\n",
           prog, GB_DEFAULT_PREC_NAME, DEFAULT_MAT_SIZE);
    printf("  %s -p hgemm_mix              # FP16 in / FP32 acc Tensor Core 측정\n", prog);
    printf("  %s -p hgemm                  # FP16 in / FP16 acc TC (이론 피크 TFLOPS 최대)\n", prog);
    printf("  %s -p sgemm                  # FP32 SGEMM (TC 미사용)\n", prog);
    printf("  %s -p sgemm_tf32 -I memset   # rand vs memset 비교 측정\n", prog);
    printf("  %s -X 8192                   # 작은 행렬 (gpu-burn 정확 모사)\n", prog);
    printf("  %s -m 50%%                   # VRAM 50%% 만 사용 (ring 슬롯 ↓)\n", prog);
    printf("  %s -g 0,1 -i 4 -t 120      # GPU 0,1, 스트림 4개, 2분\n", prog);
    printf("  sudo %s -P 250 -t 600       # 전력 캡 250W로 10분 (root 필요)\n", prog);
    printf("  %s -o -t 600              # CSV 기록하며 10분 측정 (자동 파일명)\n", prog);
    printf("  %s -o logs/ -p hgemm       # logs/ 폴더에 CSV 저장하며 측정\n", prog);
    printf("  sudo %s -S 300:10m,400:10m,600:10m -o   # TDP 스윕 + CSV 기록\n\n", prog);
}

/* ─────────────────────────────────────────────────────────
   디바이스 열거 순서 정렬 (BDF 오름차순 = nvidia-smi/amd-smi 와 동일)
   ─────────────────────────────────────────────────────────
   HIP 는 디바이스를 PCI-BDF 오름차순으로 열거한다는 보장이 없다(예: RDNA4
   2장 시스템에서 역순으로 관측). 그러면 도구의 index 와 amd-smi/rocm-smi 의
   index 가 어긋나 -g 선택·라벨이 엉뚱한 카드를 가리킨다. NVIDIA 는
   gb_init_device_order 가 CUDA_DEVICE_ORDER=PCI_BUS_ID 를 걸어 이미 BDF 순이라
   이 매핑이 항등(identity)이 된다.
     g_dev_order[논리 인덱스] = 물리(런타임) device id
   본체는 사용자 표시·-g 입력을 "논리 인덱스"로, cudaSetDevice/모니터링은
   "물리 device id"로 사용한다. (모니터링은 BDF 매칭이라 물리 id로 정상 동작) */
static int g_dev_order[MAX_GPUS];
static int g_dev_order_n = 0;

static unsigned long long bdf_key(int dev)
{
    char s[32] = "";
    if (cudaDeviceGetPCIBusId(s, (int)sizeof(s), dev) != cudaSuccess)
        return ~0ULL;   /* 조회 실패는 맨 뒤로 */
    unsigned dom = 0, bus = 0, d = 0, f = 0;
    sscanf(s, "%x:%x:%x.%x", &dom, &bus, &d, &f);
    return ((unsigned long long)dom << 24) | ((unsigned long long)bus << 16)
           | ((unsigned long long)d << 8) | (unsigned long long)f;
}

/* 전 디바이스를 BDF 오름차순으로 정렬해 논리→물리 매핑을 구성. main 최상단,
   첫 CUDA 호출 시점(=NVIDIA 는 env 적용 이후)에 1회 호출. */
static void build_device_order(void)
{
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    if (n > MAX_GPUS) n = MAX_GPUS;
    unsigned long long key[MAX_GPUS];
    for (int i = 0; i < n; i++) { g_dev_order[i] = i; key[i] = bdf_key(i); }
    for (int i = 1; i < n; i++) {          /* 삽입정렬 (n 작음) */
        int cur = g_dev_order[i];
        unsigned long long k = key[cur];
        int j = i - 1;
        while (j >= 0 && key[g_dev_order[j]] > k) {
            g_dev_order[j + 1] = g_dev_order[j];
            j--;
        }
        g_dev_order[j + 1] = cur;
    }
    g_dev_order_n = n;
}

/* 논리 인덱스 → 물리 device id (미구성/범위 밖이면 항등) */
static int phys_of(int logical)
{
    if (logical < 0 || logical >= g_dev_order_n) return logical;
    return g_dev_order[logical];
}

static void print_gpu_list(void)
{
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    gb_mon_init();

    printf("\n시스템 GPU 목록 (%d 개)  ※ 인덱스는 PCI 버스 순서 = nvidia-smi/amd-smi 와 동일\n", n);
    printf("──────────────────────────────────────────────────────────────────────\n");
    printf("  ID  이름                          VRAM        TDP    PCI(BDF)\n");
    printf("──────────────────────────────────────────────────────────────────────\n");

    for (int L = 0; L < n; L++) {
        int phys = phys_of(L);           /* 논리 L → 물리 device id */
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, phys));
        unsigned int tdp_mw = 0;
        gb_mon_t mon;
        if (gb_mon_open(phys, &mon) == 0)
            tdp_mw = gb_mon_tdp_mw(&mon);
        char bdf[32] = "?";
        cudaDeviceGetPCIBusId(bdf, (int)sizeof(bdf), phys);
        printf("  [%d]  %-28s  %5zu MiB", L, p.name,
               p.totalGlobalMem / (1024 * 1024));
        if (tdp_mw > 0) printf("  %4u W", tdp_mw / 1000);
        else            printf("        ");   /* TDP 자리 맞춤 (8칸) */
        printf("   %s\n", bdf);
    }

    printf("──────────────────────────────────────────────────────────────────────\n\n");
    gb_mon_shutdown();
}

/* ─────────────────────────────────────────────────────────
   전력 캡(TDP) 복원
   ─────────────────────────────────────────────────────────
   -P 로 변경한 전력 캡은 프로세스가 종료돼도 GPU에 그대로 남으므로,
   반드시 원래 값으로 되돌려야 합니다. 정상 종료(main 끝/atexit)와
   Ctrl-C(SIGINT)/SIGTERM 모두에서 복원합니다.
   ───────────────────────────────────────────────────────── */

static GpuCtx *g_cap_gpus     = NULL;   /* 복원 대상 GpuCtx 배열 */
static int     g_cap_num_gpus = 0;

/* 적용된 캡을 모두 원래 값으로 복원 (cap_applied 가드로 idempotent) */
static void restore_power_caps(void)
{
    if (!g_cap_gpus) return;
    for (int g = 0; g < g_cap_num_gpus; g++) {
        GpuCtx *gc = &g_cap_gpus[g];
        if (gc->cap_applied) {
            gb_mon_set_power_cap_mw(&gc->mon, gc->orig_cap_mw);
            gc->cap_applied = 0;
        }
    }
}

/* Ctrl-C/SIGTERM: 캡 복원 후 기본 동작으로 재발생시켜 종료 */
static void on_terminate_signal(int signo)
{
    restore_power_caps();
    signal(signo, SIG_DFL);
    raise(signo);
}

/* GPU 에 전력 캡 적용 (범위 클램프 포함). 성공 0, 실패 -1.
   성공 시 cap_applied=1, tdp_mw=적용값(TDP% 기준). ind 는 메시지 들여쓰기.
   단일 -P(init) 와 plan(-S) phase 전환에서 공용. */
static int apply_power_cap(GpuCtx *gc, int watts, const char *ind)
{
    if (!gc->mon.valid || gc->orig_cap_mw == 0) {
        fprintf(stderr, "\n오류: GPU %d 전력 캡을 읽을 수 없어 적용 불가\n", gc->logical_id);
        return -1;
    }
    unsigned req_mw = (unsigned)watts * 1000u;
    unsigned cmin = 0, cmax = 0;
    if (gb_mon_power_cap_range_mw(&gc->mon, &cmin, &cmax) == 0 && cmax > 0) {
        if (req_mw < cmin) {
            printf("%s%s[TDP] GPU %d 요청 %dW < 최소 %uW → %uW로 클램프%s\n",
                   ind, CLR_WARN, gc->logical_id, watts, cmin / 1000, cmin / 1000, CLR_RESET);
            req_mw = cmin;
        } else if (req_mw > cmax) {
            printf("%s%s[TDP] GPU %d 요청 %dW > 최대 %uW → %uW로 클램프%s\n",
                   ind, CLR_WARN, gc->logical_id, watts, cmax / 1000, cmax / 1000, CLR_RESET);
            req_mw = cmax;
        }
    }
    if (gb_mon_set_power_cap_mw(&gc->mon, req_mw) != 0) {
        fprintf(stderr, "\n오류: GPU %d 전력 캡 설정 실패 — "
                "root 권한이 필요하거나 미지원 GPU입니다.\n", gc->logical_id);
        return -1;
    }
    gc->cap_applied = 1;
    gc->tdp_mw      = req_mw;
    printf("%s%s[TDP] GPU %d 전력 캡 적용: %uW (원래 %uW)%s\n",
           ind, CLR_CYAN, gc->logical_id, req_mw / 1000, gc->orig_cap_mw / 1000, CLR_RESET);
    return 0;
}

/* ─────────────────────────────────────────────────────────
   main
   ───────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    /* GPU 열거 순서를 nvidia-smi/NVML(PCI 버스 순서)과 일치시킨다.
       반드시 첫 CUDA 런타임 호출 이전에 호출. (AMD 는 no-op) */
    gb_init_device_order();

    /* 기본값 설정 */
    Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.duration_sec        = 3600;
    cfg.intensity           = 1;
    cfg.prec                = GB_DEFAULT_PREC;  /* 백엔드별 (NVIDIA=sgemm_tf32, AMD=hgemm_mix) */
    cfg.mem_spec.is_percent = 1;
    cfg.mem_spec.value      = 100.0;
    cfg.init_mode           = INIT_RAND;
    cfg.tdp_cap_w           = 0;        /* -P 미지정 */

    int list_only = 0;
    int opt;
    while ((opt = getopt(argc, argv, "t:i:p:m:g:X:I:P:o::S:lh")) != -1) {
        switch (opt) {
        case 't': cfg.duration_sec      = atoi(optarg);                  break;
        case 'i': cfg.intensity         = atoi(optarg);                  break;
        case 'm': cfg.mem_spec          = parse_mem_spec(optarg);        break;
        case 'X': cfg.mat_size_override = atoi(optarg);                  break;
        case 'I':
            if      (!strcmp(optarg, "memset")) cfg.init_mode = INIT_MEMSET;
            else if (!strcmp(optarg, "rand"))   cfg.init_mode = INIT_RAND;
            else {
                fprintf(stderr, "오류: -I는 memset 또는 rand만 허용\n");
                return 1;
            }
            break;
        case 'p':
            if      (!strcmp(optarg, "dgemm"))      cfg.prec = PREC_DGEMM;
            else if (!strcmp(optarg, "hgemm_mix"))  cfg.prec = PREC_HGEMM_MIX;
            else if (!strcmp(optarg, "hgemm"))      cfg.prec = PREC_HGEMM;
            else if (!strcmp(optarg, "sgemm_tf32")) cfg.prec = PREC_SGEMM_TF32;
            else if (!strcmp(optarg, "sgemm"))      cfg.prec = PREC_SGEMM;
            else                                    cfg.prec = GB_DEFAULT_PREC;
            break;
        case 'g':
            cfg.num_gpus = parse_gpu_list(optarg, cfg.gpu_ids, MAX_GPUS);
            break;
        case 'P': cfg.tdp_cap_w          = atoi(optarg);                 break;
        case 'o':
            /* -o (인자 없음) → 자동명. -oPATH 또는 -o PATH → 경로 지정.
               getopt o:: 는 공백 분리 인자를 안 받으므로 다음 argv를 직접 확인. */
            cfg.csv_enable = 1;
            if (optarg)
                snprintf(cfg.csv_path, sizeof(cfg.csv_path), "%s", optarg);
            else if (optind < argc && argv[optind] && argv[optind][0] != '-') {
                snprintf(cfg.csv_path, sizeof(cfg.csv_path), "%s", argv[optind]);
                optind++;
            }
            break;
        case 'S': {
            int n = parse_plan(optarg, cfg.plan, MAX_PHASES);
            if (n <= 0) {
                fprintf(stderr, "오류: -S 형식 오류. 예: -S 300:10m,400:10m,600:10m\n"
                                "      (W:T 쌍을 쉼표로, 시간은 s/m/h 접미사, 기본 초)\n");
                return 1;
            }
            cfg.num_phases = n;
            break;
        }
        case 'l': list_only = 1;                                         break;
        case 'h': print_usage(argv[0]); return 0;
        default:  print_usage(argv[0]); return 1;
        }
    }

    /* 디바이스 열거 순서(논리 인덱스)를 BDF 오름차순으로 구성. 첫 CUDA 호출
       시점이며, NVIDIA 는 위 gb_init_device_order 의 env 가 이미 적용된 뒤다.
       print_gpu_list / -g 선택 모두 이 매핑을 사용한다. */
    build_device_order();

    /* list_only 는 print_gpu_list 가 자체적으로 모니터링 라이브러리를
       init/shutdown 하므로 여기서 따로 init 하지 않습니다. */
    if (list_only) { print_gpu_list(); return 0; }

    if (gb_mon_init() != 0)
        fprintf(stderr, "[%s WARN] 모니터링 라이브러리 init 실패 — "
                        "측정값이 0으로 표시될 수 있습니다.\n", GB_MON_NAME);

    /* GPU 목록 결정 */
    int sys_gpu_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&sys_gpu_count));
    if (sys_gpu_count == 0) {
        fprintf(stderr, "오류: CUDA GPU를 찾을 수 없습니다.\n");
        return 1;
    }
    if (cfg.num_gpus == 0) {
        cfg.num_gpus = sys_gpu_count;
        for (int i = 0; i < sys_gpu_count; i++) cfg.gpu_ids[i] = i;
    }

    /* 입력값 검증 */
    for (int g = 0; g < cfg.num_gpus; g++) {
        if (cfg.gpu_ids[g] < 0 || cfg.gpu_ids[g] >= sys_gpu_count) {
            fprintf(stderr, "오류: 잘못된 GPU ID %d (유효 범위: 0~%d)\n",
                    cfg.gpu_ids[g], sys_gpu_count - 1);
            return 1;
        }
    }
    if (cfg.intensity < 1 || cfg.intensity > MAX_INTENSITY) {
        fprintf(stderr, "오류: -i 범위는 1~%d 입니다.\n", MAX_INTENSITY);
        return 1;
    }
    if (cfg.tdp_cap_w < 0) {
        fprintf(stderr, "오류: -P 값은 양의 정수(W)여야 합니다.\n");
        return 1;
    }
    /* plan(-S)이 있으면 -P/-t 는 plan 이 대체하므로 무시 */
    if (cfg.num_phases > 0 && cfg.tdp_cap_w > 0) {
        fprintf(stderr, "%s[참고] -S(plan) 사용 시 -P 는 무시됩니다 "
                        "(전력 캡은 plan 의 phase 값 사용).%s\n", CLR_WARN, CLR_RESET);
        cfg.tdp_cap_w = 0;
    }

    const char *prec_str[] = {
        "SGEMM (FP32)",
        "DGEMM (FP64)",
        "HGEMM (FP16 in / FP16 acc, Tensor Core)",
        "HGEMM_MIX (FP16 in / FP32 acc, Tensor Core, mixed precision)",
        "SGEMM_TF32 (FP32 storage, TF32 Tensor Core compute)"
    };

    /* ── 실행 설정 헤더 출력 ── */
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║          gadget-burn  –  Multi-GPU GEMM Burn-in Tool         ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("  연산 타입    : %s\n", prec_str[cfg.prec]);
    if (cfg.mem_spec.is_percent)
        printf("  메모리 사용  : %.4g%% VRAM (GPU 별 자동 계산)\n", cfg.mem_spec.value);
    else
        printf("  행렬 크기    : %d x %d  (M=N=K)\n",
               (int)cfg.mem_spec.value, (int)cfg.mem_spec.value);
    printf("  GPU 당 강도  : %d 스트림\n", cfg.intensity);
    if (cfg.num_phases > 0) {
        int total = 0;
        for (int i = 0; i < cfg.num_phases; i++) total += cfg.plan[i].duration_s;
        printf("  실행 plan    : %d phases, 총 %d초  (TDP 캡 변경 → root 권한 필요)\n",
               cfg.num_phases, total);
        for (int i = 0; i < cfg.num_phases; i++)
            printf("                 phase %d: %dW × %d초\n",
                   i + 1, cfg.plan[i].tdp_w, cfg.plan[i].duration_s);
    } else {
        printf("  측정 시간    : %d 초\n", cfg.duration_sec);
    }
    printf("  사용 GPU     : %d 개  [", cfg.num_gpus);
    for (int g = 0; g < cfg.num_gpus; g++)
        printf("%s%d", g ? "," : "", cfg.gpu_ids[g]);
    printf("]\n");
    printf("  데이터 초기화: %s\n",
           (cfg.init_mode == INIT_MEMSET) ? "memset (denormal byte 0x01)"
                                          : "rand (xorshift32 PRNG)");
    if (cfg.tdp_cap_w > 0)
        printf("  TDP 캡 목표  : %d W  (전력 캡 설정 모드, root 권한 필요)\n",
               cfg.tdp_cap_w);
    printf("╠══════════════════════════════════════════════════════════════╣\n");

    /* ── GPU 별 초기화 ── */
    GpuCtx *gpus = (GpuCtx *)calloc(cfg.num_gpus, sizeof(GpuCtx));

    /* 전력 캡(-P 또는 -S plan) 사용 시: 종료/중단 시 캡을 원복하도록 훅 등록.
       (캡은 프로세스 종료 후에도 GPU에 남으므로 반드시 복원해야 함) */
    if (cfg.tdp_cap_w > 0 || cfg.num_phases > 0) {
        g_cap_gpus     = gpus;
        g_cap_num_gpus = cfg.num_gpus;
        atexit(restore_power_caps);
        signal(SIGINT,  on_terminate_signal);
        signal(SIGTERM, on_terminate_signal);
    }

    for (int g = 0; g < cfg.num_gpus; g++) {
        GpuCtx *gc = &gpus[g];
        gc->logical_id = cfg.gpu_ids[g];              /* 사용자 표시·-g 입력 (BDF 순) */
        gc->device_id  = phys_of(cfg.gpu_ids[g]);     /* 실제 런타임 device id */
        gc->gpu_index = g;
        gc->intensity = cfg.intensity;
        gc->prec      = cfg.prec;

        CUDA_CHECK(cudaSetDevice(gc->device_id));

        /* 행렬 크기 결정. 우선순위: -X > -m 절대크기 > DEFAULT_MAT_SIZE.
           num_c_buffers는 -m 비율로 정해진 target VRAM에 맞춰 자동 계산. */
        int m_size = DEFAULT_MAT_SIZE;
        if (cfg.mat_size_override > 0)
            m_size = cfg.mat_size_override;
        else if (!cfg.mem_spec.is_percent)
            m_size = (int)cfg.mem_spec.value;
        gc->mat_size = m_size;
        int num_c_buffers = calc_c_buffers(gc->device_id, cfg.mem_spec,
                                           cfg.prec, cfg.intensity,
                                           gc->mat_size);

        if (gb_mon_open(gc->device_id, &gc->mon) == 0) {
            gc->tdp_mw      = gb_mon_tdp_mw(&gc->mon);
            gc->orig_cap_mw = gc->tdp_mw;   /* -P 복원용 원래 캡 */
        }

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gc->device_id));
        snprintf(gc->name, sizeof(gc->name), "%s", prop.name);  /* CSV/메타용 캐시 */

        /* 코어 정보 조회 및 캐싱 */
        const CoreEntry *ce = core_find(prop.name);
        if (ce)
            gc->core_info = *ce;
        else
            memset(&gc->core_info, 0, sizeof(CoreEntry));

        char bdf[32] = "?";
        cudaDeviceGetPCIBusId(bdf, (int)sizeof(bdf), gc->device_id);
        printf("  [초기화] GPU %d: %s  [%s]", gc->logical_id, prop.name, bdf);
        if (gc->tdp_mw > 0) printf("  TDP %u W", gc->tdp_mw / 1000);
        printf("\n");

        if (gc->core_info.cores_fp32 > 0)
            printf("           Cores  FP32=%d / FP64=%d / Tensor=%d  (tc_ops=%d, mix=%d, tf32=%d)\n",
                   gc->core_info.cores_fp32, gc->core_info.cores_fp64,
                   gc->core_info.cores_tensor,
                   gc->core_info.tc_ops, gc->core_info.tc_ops_mix,
                   gc->core_info.tc_ops_tf32);
        else
            printf("           %s[주의] Core DB 미등록 GPU — 동적 Peak%% 표시 불가%s\n",
                   CLR_WARN, CLR_RESET);

        /* TF32 미지원 GPU에서 sgemm_tf32 실행 시 BLAS fallback 안내
           (NVIDIA Turing/Volta, AMD RDNA3 등) */
        if (cfg.prec == PREC_SGEMM_TF32
            && gc->core_info.cores_fp32 > 0
            && gc->core_info.tc_ops_tf32 == 0)
            printf("           %s[참고] TF32 미지원 → %s가 FP32 SGEMM으로 자동 fallback "
                   "(Peak%% 기준도 FP32 shader)%s\n",
                   CLR_WARN, GB_BLAS_NAME, CLR_RESET);

        /* CDNA(fp32_matrix_ops>0)에서 hgemm(FP16 입력/FP16 누산) 안내.
           CDNA 의 Matrix Core(MFMA)는 FP16 입력→FP32 누산이 네이티브이고
           FP16 누산은 네이티브 미지원 → rocBLAS 가 저속 경로로 폴백해
           Peak% 가 낮게 나옵니다. FP16 워크로드는 hgemm_mix(FP32 누산)가
           CDNA 의 정식 고속 경로입니다. */
        if (cfg.prec == PREC_HGEMM
            && gc->core_info.fp32_matrix_ops > 0)
            printf("           %s[참고] CDNA Matrix Core는 FP16 누산 미지원 → hgemm은 "
                   "저속 폴백. FP16 부하는 hgemm_mix(FP32 누산) 권장%s\n",
                   CLR_WARN, CLR_RESET);

        /* -P (단일 캡): plan(-S) 모드가 아니면 여기서 적용.
           plan 모드는 phase 별로 캡이 바뀌므로 phase 루프에서 적용한다.
           orig_cap_mw 는 종료/중단 시 restore_power_caps() 가 복원. */
        if (cfg.tdp_cap_w > 0 && cfg.num_phases == 0) {
            if (apply_power_cap(gc, cfg.tdp_cap_w, "           ") != 0) {
                restore_power_caps();
                return 1;
            }
        }

        printf("           행렬 크기 %d x %d,  스트림 %d,  %sC ring %d 슬롯%s\n",
               gc->mat_size, gc->mat_size, gc->intensity,
               CLR_CYAN, num_c_buffers, CLR_RESET);

        int use_random_init = (cfg.init_mode == INIT_RAND) ? 1 : 0;

        /* 워커 할당 및 초기화 */
        gc->workers = (GemmWorker *)calloc(cfg.intensity, sizeof(GemmWorker));
        size_t total_vram = 0;
        for (int s = 0; s < cfg.intensity; s++) {
            worker_init(&gc->workers[s], gc->device_id,
                        gc->mat_size, gc->mat_size, gc->mat_size, gc->prec,
                        num_c_buffers, use_random_init);
            total_vram += worker_vram_bytes(&gc->workers[s]);
        }
        gc->vram_used_mb = total_vram / (1024 * 1024);
        printf("           VRAM 할당: %zu MiB\n", gc->vram_used_mb);

        /* 워밍업 */
        for (int s = 0; s < cfg.intensity; s++) run_gemm(&gc->workers[s]);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* ── 실행 phase 목록 구성 ──
       -S 가 있으면 그대로, 없으면 단일 phase (tdp=-P 값(없으면 0), duration=-t). */
    Phase  phases[MAX_PHASES];
    int    num_phases;
    int    plan_mode = (cfg.num_phases > 0);
    if (plan_mode) {
        num_phases = cfg.num_phases;
        memcpy(phases, cfg.plan, sizeof(Phase) * num_phases);
    } else {
        num_phases = 1;
        phases[0].tdp_w      = cfg.tdp_cap_w;   /* 0 이면 캡 미설정 */
        phases[0].duration_s = cfg.duration_sec;
    }

    /* bytes/element (메모리 BW 추정 등에 사용) */
    size_t bpe;
    switch (cfg.prec) {
        case PREC_DGEMM:     bpe = 8; break;
        case PREC_HGEMM:
        case PREC_HGEMM_MIX: bpe = 2; break;
        default:             bpe = 4; break;
    }

    /* CSV 로깅 파일 오픈 (-o). GPU 초기화 후라 이름·BDF·TDP캡이 모두 확정됨. */
    FILE *csv = NULL;
    char  csv_path[640] = "";
    if (cfg.csv_enable) {
        csv_build_path(&cfg, csv_path, sizeof(csv_path));
        csv = fopen(csv_path, "w");
        if (!csv)
            fprintf(stderr, "  %s[CSV 경고] '%s' 열기 실패 — CSV 기록 비활성화%s\n",
                    CLR_WARN, csv_path, CLR_RESET);
        else {
            csv_write_meta(csv, &cfg, csv_path, phases, num_phases, plan_mode,
                           gpus, cfg.num_gpus, argc, argv);
            fflush(csv);
            printf("  CSV 기록     : %s\n", csv_path);
        }
    }

    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("  측정 시작\n");
    fflush(stdout);

    /* ── 스레드 시작 (한 번만; phase 가 바뀌어도 GEMM 부하는 끊지 않음) ──
       monitor_thread를 먼저 시작하고 150ms 대기(첫 샘플 확보)한 뒤
       bench_thread를 시작해 cur_util 초기값 레이스 컨디션을 방지 */
    double overall_start = now_sec();
    for (int g = 0; g < cfg.num_gpus; g++) {
        gpus[g].mon_running = 1;
        pthread_create(&gpus[g].mon_tid, NULL, monitor_thread, &gpus[g]);
    }
    usleep(150 * 1000);
    for (int g = 0; g < cfg.num_gpus; g++) {
        gpus[g].bench_running = 1;
        pthread_create(&gpus[g].bench_tid, NULL, bench_thread, &gpus[g]);
    }

    /* ── phase 루프: 각 phase = (전력 캡, 시간) 한 조건을 연속 수행 ──
       단일 실행은 phase 1개로 동일 경로를 탄다. */
    int fatal = 0;
    for (int ph = 0; ph < num_phases && !fatal; ph++) {
        Phase *P = &phases[ph];

        /* plan 모드: 이 phase 의 전력 캡 적용 (단일 모드는 init 에서 이미 적용) */
        if (plan_mode && P->tdp_w > 0) {
            printf("\n");
            for (int g = 0; g < cfg.num_gpus; g++)
                if (apply_power_cap(&gpus[g], P->tdp_w, "  ") != 0) { fatal = 1; break; }
            if (fatal) break;
        }

        if (plan_mode)
            printf("\n── Phase %d/%d : 목표 %dW, %d초 ──\n",
                   ph + 1, num_phases, P->tdp_w, P->duration_s);

        /* 전환 직후 정착(settle) 구간은 phase 평균에서 제외 (③b).
           CSV 시계열에는 settle 구간도 전부 기록된다. */
        int settle_eff = (plan_mode && P->duration_s > 2 * PHASE_SETTLE_SEC)
                         ? PHASE_SETTLE_SEC : 0;
        int baselined = 0;

        /* print_progress 가 덮어쓸 줄 미리 확보: 막대+헤더+구분선+GPU행 = N+3 */
        for (int i = 0; i < cfg.num_gpus + 3; i++) printf("\n");
        fflush(stdout);

        double p_start = now_sec();
        double p_end   = p_start + P->duration_s;
        int    last_pe = -1;
        while (now_sec() < p_end) {
            double tnow = now_sec();
            int pe = (int)(tnow - p_start);        /* phase-local 경과 */
            int ge = (int)(tnow - overall_start);  /* 전체 경과 */

            if (!baselined && pe >= settle_eff) {
                for (int g = 0; g < cfg.num_gpus; g++) snapshot_baseline(&gpus[g]);
                baselined = 1;
            }

            if (pe != last_pe) {
                last_pe = pe;
                double total_tflops = 0.0, total_power = 0.0;
                for (int g = 0; g < cfg.num_gpus; g++) {
                    total_tflops += calc_tflops_win(&gpus[g]);
                    total_power  += win_avg_power(&gpus[g]);
                }
                print_progress(pe, P->duration_s, cfg.num_gpus, gpus, cfg.prec,
                               total_tflops, total_power);

                /* CSV 행 기록 (1Hz). 매초 flush 로 중단/크래시에도 직전까지 보존. */
                if (csv) {
                    time_t wc = time(NULL);
                    struct tm tv;
                    localtime_r(&wc, &tv);
                    char wcs[24];
                    strftime(wcs, sizeof(wcs), "%Y-%m-%d %H:%M:%S", &tv);
                    for (int g = 0; g < cfg.num_gpus; g++)
                        csv_write_row(csv, wcs, ge, ph, P->tdp_w, pe,
                                      &gpus[g], cfg.prec);
                    fflush(csv);
                }
            }
            usleep(200 * 1000);
        }
        if (!baselined)   /* 매우 짧은 phase 안전망 */
            for (int g = 0; g < cfg.num_gpus; g++) snapshot_baseline(&gpus[g]);

        printf("\n\n");

        /* phase 종료 시점에 통계 캡처(스레드는 계속 도므로 즉시) 후 출력 */
        PhaseStat ps[MAX_GPUS];
        for (int g = 0; g < cfg.num_gpus; g++)
            ps[g] = phase_stat(&gpus[g], cfg.prec, bpe);
        print_phase_results(ph, num_phases, plan_mode, P, gpus, ps, cfg.num_gpus);
        if (csv)
            csv_write_phase_summary(csv, ph, P, gpus, ps, cfg.num_gpus);
    }

    /* ── 스레드 종료 ── */
    for (int g = 0; g < cfg.num_gpus; g++) {
        gpus[g].bench_running = 0;
        pthread_join(gpus[g].bench_tid, NULL);
    }
    for (int g = 0; g < cfg.num_gpus; g++) {
        gpus[g].mon_running = 0;
        pthread_join(gpus[g].mon_tid, NULL);
    }

    /* CSV: phase 요약은 phase 마다 기록되었으므로 여기선 닫기만 한다.
       Ctrl-C 중단 시엔 매초 flush 된 시계열 + 완료된 phase 요약까지 보존됨. */
    if (csv) {
        fclose(csv);
        printf("  CSV 저장 완료: %s\n\n", csv_path);
    }

    /* ── 정리 ── */
    for (int g = 0; g < cfg.num_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(gpus[g].device_id));
        for (int s = 0; s < cfg.intensity; s++)
            worker_free(&gpus[g].workers[s]);
        free(gpus[g].workers);

    }
    restore_power_caps();   /* -P/-S 로 변경한 전력 캡 원복 (mon 핸들 유효할 때) */
    g_cap_gpus = NULL;      /* atexit/시그널 핸들러가 해제된 메모리를 보지 않도록 */
    free(gpus);
    gb_mon_shutdown();
    return fatal ? 1 : 0;   /* plan phase 캡 적용 실패 시 비정상 종료 코드 */
}
