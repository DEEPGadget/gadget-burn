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

/* 온도: ≥75°C 빨강 / 65~74°C 노랑 / <65°C 초록 */
static const char *temp_color(double t)
{
    if (t >= 75.0) return CLR_RED;
    if (t >= 65.0) return CLR_YELLOW;
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
    int         device_id;
    int         gpu_index;
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
    double       sum_temp;
    double       sum_clock;
    long         mon_samples;
    long         util_samples;
    long         throttle_thermal_samples;     /* SW/HW thermal slowdown 감지 횟수 */
    long         throttle_powerbrake_samples;  /* HW power brake 감지 횟수 */

    /* 슬라이딩 윈도우 (실시간 표시용, 원형 버퍼) */
    double       win_power[MON_WIN_SIZE];
    double       win_util[MON_WIN_SIZE];
    double       win_clock[MON_WIN_SIZE];
    int          win_util_valid[MON_WIN_SIZE];
    int          win_head;
    int          win_count;

    volatile unsigned int cur_temp;
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
        int t = gb_mon_temp_c(&g->mon);
        if (t >= 0) g->cur_temp = (unsigned int)t;
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

        /* 온도 */
        int temp = gb_mon_temp_c(&g->mon);
        if (temp >= 0) {
            g->sum_temp += (double)temp;
            g->cur_temp  = (unsigned int)temp;
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
           "\xec\x98\xa8\xeb\x8f\x84 ",        /*  5열 "온도 " */
           "Throt",                            /*  5열 */
           "  VRAM   ");                       /*  9열 */

    printf("  %s  %s  %s  %s  %s  %s  %s  %s  %s  %s\n",
           "------", "-------", "---------------",
           "---------", "------", "------", "-------", "-----", "-----", "---------");

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

        /* 온도: "°"는 UTF-8 2바이트, 표시 1열 → snprintf 후 %5s로 5열 확보 */
        char temp_buf[16];
        snprintf(temp_buf, sizeof(temp_buf), "%u\xc2\xb0\x43", gc->cur_temp);

        const char *rc = (rpeak > 0.0) ? rpeak_color(tflops / rpeak * 100.0) : "";
        const char *tc = temp_color((double)gc->cur_temp);

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

        printf("  GPU %2d  %7.3f  %s%13s%s  %7.1f W  %6s  %5.1f%%  %4uMHz  %s%5s%s  %s%s%s  %5zu MiB\n",
               gc->device_id,
               tflops,
               rc, peak_buf, CLR_RESET,
               pw,
               tdp_buf,
               util,
               clk,
               tc, temp_buf, CLR_RESET,
               throt_color, throt_str, CLR_RESET,
               gc->vram_used_mb);
    }
    fflush(stdout);
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
    printf("  sudo %s -P 250 -t 600       # 전력 캡 250W로 10분 (root 필요)\n\n", prog);
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

    for (int i = 0; i < n; i++) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, i));
        unsigned int tdp_mw = 0;
        gb_mon_t mon;
        if (gb_mon_open(i, &mon) == 0)
            tdp_mw = gb_mon_tdp_mw(&mon);
        char bdf[32] = "?";
        cudaDeviceGetPCIBusId(bdf, (int)sizeof(bdf), i);
        printf("  [%d]  %-28s  %5zu MiB", i, p.name,
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
    while ((opt = getopt(argc, argv, "t:i:p:m:g:X:I:P:lh")) != -1) {
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
        case 'l': list_only = 1;                                         break;
        case 'h': print_usage(argv[0]); return 0;
        default:  print_usage(argv[0]); return 1;
        }
    }

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
    printf("  측정 시간    : %d 초\n", cfg.duration_sec);
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

    /* -P 전력 캡 사용 시: 종료/중단 시 캡을 원복하도록 훅 등록.
       (캡은 프로세스 종료 후에도 GPU에 남으므로 반드시 복원해야 함) */
    if (cfg.tdp_cap_w > 0) {
        g_cap_gpus     = gpus;
        g_cap_num_gpus = cfg.num_gpus;
        atexit(restore_power_caps);
        signal(SIGINT,  on_terminate_signal);
        signal(SIGTERM, on_terminate_signal);
    }

    for (int g = 0; g < cfg.num_gpus; g++) {
        GpuCtx *gc = &gpus[g];
        gc->device_id = cfg.gpu_ids[g];
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

        /* 코어 정보 조회 및 캐싱 */
        const CoreEntry *ce = core_find(prop.name);
        if (ce)
            gc->core_info = *ce;
        else
            memset(&gc->core_info, 0, sizeof(CoreEntry));

        char bdf[32] = "?";
        cudaDeviceGetPCIBusId(bdf, (int)sizeof(bdf), gc->device_id);
        printf("  [초기화] GPU %d: %s  [%s]", gc->device_id, prop.name, bdf);
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

        /* -P: 전력 캡(TDP) 설정. root 권한 필요. orig_cap_mw 에 원래 값이
           저장돼 있으며 종료/중단 시 restore_power_caps() 가 복원한다. */
        if (cfg.tdp_cap_w > 0) {
            if (!gc->mon.valid || gc->orig_cap_mw == 0) {
                fprintf(stderr, "\n오류: GPU %d 전력 캡을 읽을 수 없어 -P 적용 불가\n",
                        gc->device_id);
                restore_power_caps();
                return 1;
            }
            unsigned req_mw = (unsigned)cfg.tdp_cap_w * 1000u;
            unsigned cmin = 0, cmax = 0;
            if (gb_mon_power_cap_range_mw(&gc->mon, &cmin, &cmax) == 0 && cmax > 0) {
                if (req_mw < cmin) {
                    printf("           %s[TDP] 요청 %dW < 최소 %uW → %uW로 클램프%s\n",
                           CLR_WARN, cfg.tdp_cap_w, cmin / 1000, cmin / 1000, CLR_RESET);
                    req_mw = cmin;
                } else if (req_mw > cmax) {
                    printf("           %s[TDP] 요청 %dW > 최대 %uW → %uW로 클램프%s\n",
                           CLR_WARN, cfg.tdp_cap_w, cmax / 1000, cmax / 1000, CLR_RESET);
                    req_mw = cmax;
                }
            }
            if (gb_mon_set_power_cap_mw(&gc->mon, req_mw) != 0) {
                fprintf(stderr, "\n오류: GPU %d 전력 캡 설정 실패 — "
                        "root 권한이 필요하거나 미지원 GPU입니다.\n", gc->device_id);
                restore_power_caps();
                return 1;
            }
            gc->cap_applied = 1;
            gc->tdp_mw      = req_mw;   /* TDP% 기준을 설정한 캡으로 */
            printf("           %s[TDP] 전력 캡 적용: %uW (원래 %uW)%s\n",
                   CLR_CYAN, req_mw / 1000, gc->orig_cap_mw / 1000, CLR_RESET);
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

    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("  측정 시작\n");
    fflush(stdout);

    double t_start = now_sec();
    double t_end   = t_start + cfg.duration_sec;

    /* ── 스레드 시작 ──
       monitor_thread를 먼저 시작하고 150ms 대기(첫 샘플 확보)한 뒤
       bench_thread를 시작해 cur_util 초기값 레이스 컨디션을 방지 */
    for (int g = 0; g < cfg.num_gpus; g++) {
        gpus[g].mon_running = 1;
        pthread_create(&gpus[g].mon_tid, NULL, monitor_thread, &gpus[g]);
    }
    usleep(150 * 1000);
    for (int g = 0; g < cfg.num_gpus; g++) {
        gpus[g].bench_running = 1;
        pthread_create(&gpus[g].bench_tid, NULL, bench_thread, &gpus[g]);
    }

    /* print_progress가 덮어쓸 줄 미리 확보: 진행막대+헤더+구분선+GPU행 = N+3 */
    for (int i = 0; i < cfg.num_gpus + 3; i++) printf("\n");
    fflush(stdout);

    /* ── 메인 루프: 1초마다 진행 상황 출력 ── */
    int last_elapsed = -1;
    while (now_sec() < t_end) {
        int elapsed = (int)(now_sec() - t_start);
        if (elapsed != last_elapsed) {
            last_elapsed = elapsed;

            double total_tflops = 0.0, total_power = 0.0;
            for (int g = 0; g < cfg.num_gpus; g++) {
                total_tflops += calc_tflops_win(&gpus[g]);
                total_power  += win_avg_power(&gpus[g]);
            }

            print_progress(elapsed, cfg.duration_sec,
                           cfg.num_gpus, gpus, cfg.prec,
                           total_tflops, total_power);
        }
        usleep(200 * 1000);
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
    printf("\n\n");

    /* ── 최종 결과 출력 ── */
    size_t bpe;
    switch (cfg.prec) {
        case PREC_DGEMM:     bpe = 8; break;
        case PREC_HGEMM:
        case PREC_HGEMM_MIX: bpe = 2; break;
        default:             bpe = 4; break;
    }

    double grand_tflops = 0.0, grand_power = 0.0;
    double grand_util   = 0.0, grand_membw = 0.0;

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                        최종 측정 결과                        ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");

    for (int g = 0; g < cfg.num_gpus; g++) {
        GpuCtx *gc = &gpus[g];

        double wall_sec  = gc->total_gpu_ms * 1e-3;
        double tflops    = calc_tflops_all(gc);
        double avg_pw    = (gc->mon_samples > 0)  ? gc->sum_power / gc->mon_samples  : 0.0;
        double avg_util  = (gc->util_samples > 0) ? gc->sum_util  / gc->util_samples : 0.0;
        double avg_temp  = (gc->mon_samples > 0)  ? gc->sum_temp  / gc->mon_samples  : 0.0;
        double avg_clock = (gc->mon_samples > 0)  ? gc->sum_clock / gc->mon_samples  : 0.0;
        double membw     = (wall_sec > 0)
            ? bpe * 3.0 * gc->mat_size * (double)gc->mat_size * gc->intensity
              * gc->total_iters / wall_sec * 1e-12
            : 0.0;
        double eff = (avg_pw > 0.1) ? tflops / avg_pw : 0.0;

        /* 최종 결과: 전체 구간 평균 클럭 기반 동적 Rpeak */
        double rpeak = calc_dynamic_rpeak(gc, cfg.prec, avg_clock);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, gc->device_id);

        printf("  GPU %2d  %s\n", gc->device_id, prop.name);
        printf("    ├ 행렬 크기    : %d x %d,  스트림 %d\n",
               gc->mat_size, gc->mat_size, gc->intensity);
        printf("    ├ 총 반복 횟수 : %ld 회\n", gc->total_iters);
        printf("    ├ 유효 시간    : %.3f 초\n", wall_sec);
        printf("    ├ VRAM 사용    : %zu MiB\n", gc->vram_used_mb);

        printf("    ├ 성능         : %s%.4f TFLOPS%s", CLR_GREEN, tflops, CLR_RESET);
        if (rpeak > 0.0) {
            double pct = tflops / rpeak * 100.0;
            printf("  %s%5.1f%%%s  (%.1f TFLOPS @ %.0f MHz 기준)",
                   rpeak_color(pct), pct, CLR_RESET, rpeak, avg_clock);
        }
        printf("\n");

        printf("    ├ 평균 전력    : %s%.1f W%s", CLR_YELLOW, avg_pw, CLR_RESET);
        if (gc->tdp_mw > 0)
            printf("  (TDP 대비 %.1f%%)", avg_pw / (gc->tdp_mw / 1000.0) * 100.0);
        printf("\n");

        printf("    ├ GPU 사용률   : %s%.1f %%%s\n", CLR_CYAN, avg_util, CLR_RESET);
        printf("    ├ SM Clock     : %.0f MHz 평균  (종료 시 %u MHz)\n",
               avg_clock, gc->cur_clock);
        printf("    ├ 평균 온도    : %s%.1f°C%s  (종료 시 %u°C)\n",
               temp_color(avg_temp), avg_temp, CLR_RESET, gc->cur_temp);

        /* Throttle 누적: 100ms 폴링 기준 → samples × 0.1초 */
        if (gc->mon_samples > 0) {
            double th_sec  = gc->throttle_thermal_samples * 0.1;
            double pwr_sec = gc->throttle_powerbrake_samples * 0.1;
            double th_pct  = (double)gc->throttle_thermal_samples
                             / gc->mon_samples * 100.0;
            double pwr_pct = (double)gc->throttle_powerbrake_samples
                             / gc->mon_samples * 100.0;
            const char *th_col  = (th_sec  > 0.0) ? CLR_RED    : "";
            const char *pwr_col = (pwr_sec > 0.0) ? CLR_YELLOW : "";
            printf("    ├ Throttle    : %s열 %.1f초 (%.1f%%)%s, "
                   "%s전력 %.1f초 (%.1f%%)%s\n",
                   th_col,  th_sec,  th_pct,  (th_sec  > 0.0) ? CLR_RESET : "",
                   pwr_col, pwr_sec, pwr_pct, (pwr_sec > 0.0) ? CLR_RESET : "");
        }

        printf("    ├ 메모리 BW    : %.3f TB/s (추정)\n", membw);
        printf("    └ 전력 효율    : %.4f TFLOPS/W\n", eff);

        if (g < cfg.num_gpus - 1)
            printf("  ──────────────────────────────────────────────────────────────\n");

        grand_tflops += tflops;
        grand_power  += avg_pw;
        grand_util   += avg_util;
        grand_membw  += membw;
    }

    if (cfg.num_gpus > 1) {
        printf("  ══════════════════════════════════════════════════════════════\n");
        printf("  전체 합산 (%d GPU)\n", cfg.num_gpus);
        printf("    ★ 총 성능        : %s%.4f TFLOPS%s\n",
               CLR_GREEN,  grand_tflops, CLR_RESET);
        printf("    ★ 총 전력        : %s%.1f W%s\n",
               CLR_YELLOW, grand_power,  CLR_RESET);
        printf("    ★ 평균 GPU 사용률: %s%.1f %%%s\n",
               CLR_CYAN, grand_util / cfg.num_gpus, CLR_RESET);
        printf("    ★ 총 메모리 BW   : %.3f TB/s (추정)\n", grand_membw);
        if (grand_power > 0.1)
            printf("    ★ 시스템 전력효율: %.4f TFLOPS/W\n",
                   grand_tflops / grand_power);
    }

    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* ── 정리 ── */
    for (int g = 0; g < cfg.num_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(gpus[g].device_id));
        for (int s = 0; s < cfg.intensity; s++)
            worker_free(&gpus[g].workers[s]);
        free(gpus[g].workers);

    }
    restore_power_caps();   /* -P 로 변경한 전력 캡 원복 (mon 핸들 유효할 때) */
    g_cap_gpus = NULL;      /* atexit/시그널 핸들러가 해제된 메모리를 보지 않도록 */
    free(gpus);
    gb_mon_shutdown();
    return 0;
}
