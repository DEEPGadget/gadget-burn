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
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <nvml.h>

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
   ───────────────────────────────────────────────────────── */

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t _e = (call);                                       \
        if (_e != cudaSuccess) {                                       \
            fprintf(stderr, "\n[CUDA ERR] %s:%d  %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(_e));       \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                             \
    do {                                                               \
        cublasStatus_t _s = (call);                                    \
        if (_s != CUBLAS_STATUS_SUCCESS) {                             \
            fprintf(stderr, "\n[cuBLAS ERR] %s:%d  code=%d\n",        \
                    __FILE__, __LINE__, (int)_s);                      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

#define NVML_CHECK(call)                                               \
    do {                                                               \
        nvmlReturn_t _r = (call);                                      \
        if (_r != NVML_SUCCESS)                                        \
            fprintf(stderr, "[NVML WARN] %s\n", nvmlErrorString(_r)); \
    } while (0)

/* ─────────────────────────────────────────────────────────
   GPU 코어 수 테이블 (동적 Rpeak 계산용)
   ─────────────────────────────────────────────────────────
   출처: NVIDIA 공식 아키텍처 화이트페이퍼 / datasheet
   cores_fp32   : CUDA FP32 코어 수
   cores_fp64   : CUDA FP64 코어 수 (0 = 미지원)
                  소비자급 Ada/Blackwell = FP32 코어의 1/64 (2개/SM)
                  Hopper = SM당 64개 전용 FP64 코어
                  L40S   = FP64 코어 없음 → 0
   cores_tensor : Tensor Core 수
   tc_ops       : Tensor Core 1개당 클럭당 FP16 ops (acc=FP16 dense 기준)
                  Blackwell 5th gen: ~1024
                  Ada 4th gen:        ~512
                  Hopper 4th gen:    ~2048

   동적 Rpeak 공식:
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
} CoreEntry;

/* 구체적인 이름(긴 것)을 먼저 배치해야 substring 매칭이 올바르게 동작 */
static const CoreEntry CORE_TABLE[] = {
    /* Blackwell RTX PRO — GB202 full (188 SM)
       FP32=128/SM×188=24064, FP64=2/SM=376
       whitepaper 명시대로 FP32 누산 페널티 없음 → tc_ops_mix = tc_ops = 256
       TF32 dense = tc_ops/2 = 128 (Pro 풀스피드)                          */
    { "RTX PRO 6000 Blackwell Server Edition", 24064, 376, 752,  256,  256, 128 },
    { "RTX PRO 6000 Blackwell Max-Q",          24064, 376, 752,  256,  256, 128 },
    { "RTX PRO 6000 Blackwell",                24064, 376, 752,  256,  256, 128 },
    /* Blackwell GeForce — GB202 (170 SM), GB203 (84 SM)
       FP64=2/SM (소비자급). HW 강제 반속 → tc_ops_mix = tc_ops/2 = 128
       TF32 dense = tc_ops/4 = 64 (이중 페널티)                            */
    { "GeForce RTX 5090",                      21760, 340, 680,  256,  128,  64 },
    { "GeForce RTX 5080",                      10752, 168, 336,  256,  128,  64 },
    /* Ada GeForce — AD102 (128 SM), AD103 (76 SM)
       FP64=2/SM (소비자급). HW 강제 반속, TF32 이중 페널티                */
    { "GeForce RTX 4090",                      16384, 256, 512,  256,  128,  64 },
    { "GeForce RTX 4080",                       9728, 152, 304,  256,  128,  64 },
    /* Ada Professional / Data Center — AD102 full (142 SM)
       whitepaper 명시대로 풀스피드, TF32 = tc_ops/2 = 128                 */
    { "RTX 6000 Ada",                          18176, 284, 568,  256,  256, 128 },
    { "L40S",                                  18176, 284, 568,  256,  256, 128 },
    /* Hopper — GH100 (132 SM for H200 NVL)
       FP64=64/SM=8448 (전용 코어). FP16 dense = 989 TFLOPS @ 1830 MHz
       TF32 dense = 494 TFLOPS = tc_ops/2 = 512                            */
    { "H200 NVL",                              16896, 8448, 528, 1024, 1024,  512 },
    { "H200",                                  16896, 8448, 528, 1024, 1024,  512 },
    /* Blackwell DC — B200 (160 SM × 2-die fused, FP32=128/SM=20480)
       FP64=64/SM=10240. FP16 dense = 2250 TFLOPS @ ~1717 MHz
       NVIDIA가 세대마다 tc_ops 2배: A100=512 → H200=1024 → B200=2048
       TF32 dense = 1125 TFLOPS = tc_ops/2 = 1024                          */
    { "B200",                                  20480, 10240, 640, 2048, 2048, 1024 },
    /* Ampere DC — GA100, FP32 누산 페널티 없음
       FP16 dense = 312 TFLOPS, TF32 dense = 156 TFLOPS @ 1410 MHz → 256   */
    { "A100",                                   6912, 3456, 432,  512,  512,  256 },
    /* Ampere Pro — RTX A-series, GA102/GA104. Pro 풀스피드 (FP32 acc 페널티 없음).
       FP64=2/SM (1/64 of FP32, 소비자급 비율). ops: 256/256/128             */
    { "RTX A6000",                             10752,  168, 336,  256,  256,  128 },
    { "RTX A5000",                              8192,  128, 256,  256,  256,  128 },
    { "RTX A4000",                              6144,   96, 192,  256,  256,  128 },
    /* Ampere GeForce — GA102 (84 SM), GA104 (48 SM), GA106 (28 SM)
       FP64=2/SM (소비자급). 소비자 페널티 적용: ops 256/128/64
       Ti 변형은 더 구체적이므로 일반 변형보다 먼저 배치 (substring matching) */
    { "GeForce RTX 3090 Ti",                   10752,  168, 336,  256,  128,   64 },
    { "GeForce RTX 3090",                      10496,  164, 328,  256,  128,   64 },
    { "GeForce RTX 3080 Ti",                   10240,  160, 320,  256,  128,   64 },
    { "GeForce RTX 3080",                       8704,  136, 272,  256,  128,   64 },
    { "GeForce RTX 3070 Ti",                    6144,   96, 192,  256,  128,   64 },
    { "GeForce RTX 3070",                       5888,   92, 184,  256,  128,   64 },
    { "GeForce RTX 3060 Ti",                    4864,   76, 152,  256,  128,   64 },
    { "GeForce RTX 3060",                       3584,   56, 112,  256,  128,   64 },
    { "GeForce RTX 3050",                       2560,   40,  80,  256,  128,   64 },
    /* Turing Pro (Quadro RTX) — TU102 full / TU104
       2nd gen TC, TF32 미지원 (tc_ops_tf32 = 0).
       RTX 8000/6000은 compute 동일 (VRAM만 다름)                            */
    { "Quadro RTX 8000",                        4608,  144, 576,  128,  128,    0 },
    { "Quadro RTX 6000",                        4608,  144, 576,  128,  128,    0 },
    { "Quadro RTX 5000",                        3072,   96, 384,  128,  128,    0 },
    /* Turing GeForce/Titan — TU102 cut (68 SM) / TU102 full (72 SM) / TU104 (46 SM)
       2nd gen TC, TF32 미지원. FP32 acc 페널티 없음 (소비자 페널티는 Ampere부터). */
    { "GeForce RTX 2080 Ti",                    4352,  136, 544,  128,  128,    0 },
    { "Titan RTX",                              4608,  144, 576,  128,  128,    0 },
    { "GeForce RTX 2080",                       2944,   92, 368,  128,  128,    0 },
    /* Volta DC — GV100 (80 SM), 1st gen TC, FP16 input only, TF32 미지원.
       FP64=32/SM=2560 (1:2 of FP32, DC 풀 FP64). "Tesla V100"이 "V100"보다 먼저. */
    { "Tesla V100",                             5120, 2560, 640,  128,  128,    0 },
    { "V100",                                   5120, 2560, 640,  128,  128,    0 },
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

typedef enum {
    PREC_SGEMM,
    PREC_DGEMM,
    PREC_HGEMM,
    PREC_HGEMM_MIX,
    PREC_SGEMM_TF32   /* FP32 storage + TF32 Tensor Core compute */
} PrecType;

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
} Config;

/* 기본 행렬 크기 (M=N=K).
   32768은 큰 GPU(RTX 5090, PRO 6000 SE, H200 등)에서 메모리 트래픽과 SM 점유율
   모두 충분히 확보되어 max TDP 도달이 가장 안정적. -X 옵션으로 override 가능
   (예: -X 16384, -X 8192). */
#define DEFAULT_MAT_SIZE 32768

/* 단일 GEMM 스트림 워커 */
typedef struct {
    int            device_id;
    cublasHandle_t handle;
    cudaStream_t   stream;
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

    /* NVML */
    nvmlDevice_t nvml_dev;
    unsigned int tdp_mw;

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
    volatile unsigned long long cur_throttle_reasons;  /* NVML throttle bitmask */

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
    CUBLAS_CHECK(cublasCreate(&w->handle));
    CUBLAS_CHECK(cublasSetStream(w->handle, w->stream));

    /* Modern API: cublasGemmEx에 compute type을 명시적으로 지정하므로
       handle의 math mode는 cublasGemmEx에 무영향. DEFAULT로 고정. */
    CUBLAS_CHECK(cublasSetMathMode(w->handle, CUBLAS_DEFAULT_MATH));

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
    cublasDestroy(w->handle);
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

    /* 모든 GEMM을 cublasGemmEx로 통일 (modern API).
       compute type 및 data type을 명시적으로 지정 → cuBLAS heuristic이
       deprecated cublasSetMathMode에 의존하지 않음. */
    switch (w->prec) {
    case PREC_SGEMM: {
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmEx(
            w->handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha, w->dA, CUDA_R_32F, M,
                    w->dB, CUDA_R_32F, K,
            &beta,  dC, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F,        /* 순수 FP32, TC 미사용 */
            CUBLAS_GEMM_DEFAULT));
        break;
    }
    case PREC_DGEMM: {
        const double alpha = 1.0, beta = 0.0;
        CUBLAS_CHECK(cublasGemmEx(
            w->handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha, w->dA, CUDA_R_64F, M,
                    w->dB, CUDA_R_64F, K,
            &beta,  dC, CUDA_R_64F, M,
            CUBLAS_COMPUTE_64F,        /* FP64 */
            CUBLAS_GEMM_DEFAULT));
        break;
    }
    case PREC_HGEMM: {
        const __half alpha = __float2half(1.f);
        const __half beta  = __float2half(0.f);
        CUBLAS_CHECK(cublasGemmEx(
            w->handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha, w->dA, CUDA_R_16F, M,
                    w->dB, CUDA_R_16F, K,
            &beta,  dC, CUDA_R_16F, M,
            CUBLAS_COMPUTE_16F,        /* FP16 in/acc, Tensor Core */
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        break;
    }
    case PREC_HGEMM_MIX: {
        /* FP16 입력/출력 storage + FP32 누산 (mixed precision)
           이론 TFLOPS는 PREC_HGEMM의 1/2 (소비자/Pro급) 또는 동일 (서버급)이지만
           실측 TDP가 가장 높아 burn-in 기본 정밀도로 사용됩니다. */
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmEx(
            w->handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha, w->dA, CUDA_R_16F, M,
                    w->dB, CUDA_R_16F, K,
            &beta,  dC, CUDA_R_16F, M,
            CUBLAS_COMPUTE_32F,        /* FP16 in, FP32 acc — TC mixed precision */
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        break;
    }
    case PREC_SGEMM_TF32: {
        /* FP32 storage + TF32 Tensor Core compute.
           CUBLAS_COMPUTE_32F_FAST_TF32 → 입력 FP32 mantissa를 TF32(19-bit)로
           자르고 TC MMA로 가속, 누산은 FP32. Ampere(sm_80) 이상 필요.
           burn 모드의 TDP 효과는 cuBLAS call 자체가 아니라 multi-C ring +
           compare 커널 + random init에서 나옴 (call API는 통일). */
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasGemmEx(
            w->handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha, w->dA, CUDA_R_32F, M,
                    w->dB, CUDA_R_32F, K,
            &beta,  dC, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        break;
    }
    }

    /* ring 슬롯 advance */
    w->cur_c_idx = (w->cur_c_idx + 1) % w->num_c_buffers;
}

/* ─────────────────────────────────────────────────────────
   NVML Util 샘플링
   ─────────────────────────────────────────────────────────
   nvmlDeviceGetUtilizationRates()는 드라이버 공유 버퍼를 사용해
   멀티-GPU 환경에서 서로 다른 GPU임에도 같은 값이 반환되는 버그가
   있습니다. nvmlDeviceGetSamples()는 GPU handle 별 독립 링버퍼를
   유지하므로 각 GPU의 Util을 정확히 측정할 수 있습니다.
   ───────────────────────────────────────────────────────── */

static double read_util_samples(nvmlDevice_t dev, unsigned long long *last_ts)
{
    unsigned int count = 0;
    nvmlValueType_t vtype;

    nvmlReturn_t r = nvmlDeviceGetSamples(
        dev, NVML_GPU_UTILIZATION_SAMPLES, *last_ts, &vtype, &count, NULL);
    if (r != NVML_SUCCESS || count == 0)
        return -1.0;

    nvmlSample_t *buf = (nvmlSample_t *)malloc(count * sizeof(nvmlSample_t));
    if (!buf) return -1.0;

    r = nvmlDeviceGetSamples(
        dev, NVML_GPU_UTILIZATION_SAMPLES, *last_ts, &vtype, &count, buf);

    double sum = 0.0;
    unsigned int valid = 0;
    if (r == NVML_SUCCESS) {
        for (unsigned int i = 0; i < count; i++) {
            sum += buf[i].sampleValue.uiVal;
            if (buf[i].timeStamp > *last_ts)
                *last_ts = buf[i].timeStamp;
            valid++;
        }
    }
    free(buf);
    return (valid > 0) ? sum / valid : -1.0;
}

/* ─────────────────────────────────────────────────────────
   모니터링 스레드 (GPU 당 1개, 100ms 폴링)
   ───────────────────────────────────────────────────────── */

static void *monitor_thread(void *arg)
{
    GpuCtx *g = (GpuCtx *)arg;
    unsigned long long util_last_ts = 0;

    /* 첫 루프 전에 cur_* 필드를 초기화해 race condition 방지 */
    {
        unsigned int tmp = 0;
        double u = read_util_samples(g->nvml_dev, &util_last_ts);
        g->cur_util  = (u >= 0.0) ? (unsigned int)u : 0;
        if (nvmlDeviceGetTemperature(g->nvml_dev, NVML_TEMPERATURE_GPU, &tmp) == NVML_SUCCESS)
            g->cur_temp = tmp;
        if (nvmlDeviceGetClockInfo(g->nvml_dev, NVML_CLOCK_SM, &tmp) == NVML_SUCCESS)
            g->cur_clock = tmp;
    }

    while (g->mon_running) {
        unsigned int pw = 0, temp = 0, clk = 0;

        /* 전력 */
        double pw_w = 0.0;
        if (nvmlDeviceGetPowerUsage(g->nvml_dev, &pw) == NVML_SUCCESS) {
            pw_w = pw / 1000.0;
            g->sum_power += pw_w;
        }

        /* Util */
        double u = read_util_samples(g->nvml_dev, &util_last_ts);
        if (u >= 0.0) {
            g->sum_util += u;
            g->cur_util  = (unsigned int)u;
            g->util_samples++;
        }

        /* 온도 */
        if (nvmlDeviceGetTemperature(g->nvml_dev, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
            g->sum_temp += (double)temp;
            g->cur_temp  = temp;
        }

        /* SM Clock */
        double clk_d = 0.0;
        if (nvmlDeviceGetClockInfo(g->nvml_dev, NVML_CLOCK_SM, &clk) == NVML_SUCCESS) {
            clk_d = (double)clk;
            g->sum_clock += clk_d;
            g->cur_clock  = clk;
        }

        /* Throttle reasons (CUDA 12.5+ 에서 EventReasons로 rename, 구버전 함수도 유효).
           SW Power Cap은 burn 모드에서 의도된 동작이라 별도 추적 안 함.
           HW Slowdown 우산 flag도 thermal/power brake 둘 중 하나가 같이 set되면 자동 카운트됨. */
        unsigned long long reasons = 0;
        if (nvmlDeviceGetCurrentClocksThrottleReasons(g->nvml_dev, &reasons) == NVML_SUCCESS) {
            g->cur_throttle_reasons = reasons;
            if (reasons & (nvmlClocksThrottleReasonSwThermalSlowdown
                         | nvmlClocksThrottleReasonHwThermalSlowdown))
                g->throttle_thermal_samples++;
            if (reasons & nvmlClocksThrottleReasonHwPowerBrakeSlowdown)
                g->throttle_powerbrake_samples++;
        }

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
           우선순위는 thermal > power brake. SW Power Cap은 의도된 상태라 표시 안 함. */
        const char *throt_str;
        const char *throt_color;
        unsigned long long r = gc->cur_throttle_reasons;
        if (r & (nvmlClocksThrottleReasonSwThermalSlowdown
               | nvmlClocksThrottleReasonHwThermalSlowdown)) {
            throt_str   = "THERM";
            throt_color = CLR_RED;
        } else if (r & nvmlClocksThrottleReasonHwPowerBrakeSlowdown) {
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
    printf("  -p <타입>    sgemm | dgemm | hgemm | hgemm_mix | sgemm_tf32   (기본: sgemm_tf32)\n");
    printf("               sgemm      → FP32 (cublasGemmEx, COMPUTE_32F)\n");
    printf("               dgemm      → FP64 (cublasGemmEx, COMPUTE_64F)\n");
    printf("               hgemm      → FP16 in / FP16 acc Tensor Core\n");
    printf("                            이론 피크 TFLOPS가 가장 높음\n");
    printf("               hgemm_mix  → FP16 in / FP32 acc Tensor Core (mixed precision)\n");
    printf("                            소비자급은 hgemm의 1/2 처리량,\n");
    printf("                            Pro/서버급(A100/H100/H200)은 hgemm과 동일.\n");
    printf("               sgemm_tf32 → FP32 storage + TF32 Tensor Core compute (기본)\n");
    printf("                            CUBLAS_COMPUTE_32F_FAST_TF32, Ampere+ 필요.\n");
    printf("                            메모리 대역폭 부하 + TC 부하 동시 발생,\n");
    printf("                            gpu-burn -tc 와 동일 의도.\n");
    printf("  -m <값>      메모리 사용량\n");
    printf("                 숫자   : M=N=K 행렬 크기            (예: -m 8192)\n");
    printf("                 숫자%%  : VRAM 대비 비율             (예: -m 80%%)\n");
    printf("               (기본: -m 100%%)\n");
    printf("  -g <목록>    사용할 GPU ID (쉼표 구분)             (기본: 전체)\n");
    printf("  -X <크기>    행렬 크기 M override (기본 %d)\n", DEFAULT_MAT_SIZE);
    printf("               권장값: 8192 (gpu-burn 정확 모사)\n");
    printf("                       16384, 32768 (큰 GPU에서 더 강한 메모리 부하)\n");
    printf("               우선순위: -X > -m 절대크기 > 기본 %d\n", DEFAULT_MAT_SIZE);
    printf("  -I <모드>    A,B 데이터 초기화 방식: memset | rand   (기본: rand)\n");
    printf("                 memset: cudaMemset(1) — denormal FP32 (TC switching 최소)\n");
    printf("                 rand  : xorshift PRNG로 random 데이터 (TC switching 풀가동)\n");
    printf("               비교 측정용 — 데이터 entropy가 TDP에 미치는 영향 분리 확인\n");
    printf("  -l           GPU 목록 출력 후 종료\n");
    printf("  -h           도움말\n\n");
    printf("예시:\n");
    printf("  %s                           # 기본: sgemm_tf32 + rand, M=%d ring\n", prog, DEFAULT_MAT_SIZE);
    printf("  %s -p hgemm_mix              # FP16 in / FP32 acc Tensor Core 측정\n", prog);
    printf("  %s -p hgemm                  # FP16 in / FP16 acc TC (이론 피크 TFLOPS 최대)\n", prog);
    printf("  %s -p sgemm                  # FP32 SGEMM (TC 미사용)\n", prog);
    printf("  %s -p sgemm_tf32 -I memset   # rand vs memset 비교 측정\n", prog);
    printf("  %s -X 8192                   # 작은 행렬 (gpu-burn 정확 모사)\n", prog);
    printf("  %s -m 50%%                   # VRAM 50%% 만 사용 (ring 슬롯 ↓)\n", prog);
    printf("  %s -g 0,1 -i 4 -t 120      # GPU 0,1, 스트림 4개, 2분\n\n", prog);
}

static void print_gpu_list(void)
{
    int n = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n));
    NVML_CHECK(nvmlInit());

    printf("\n시스템 GPU 목록 (%d 개)\n", n);
    printf("──────────────────────────────────────────────────────\n");
    printf("  ID  이름                          VRAM        TDP\n");
    printf("──────────────────────────────────────────────────────\n");

    for (int i = 0; i < n; i++) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, i));
        nvmlDevice_t dev;
        unsigned int tdp_mw = 0;
        if (nvmlDeviceGetHandleByIndex(i, &dev) == NVML_SUCCESS)
            nvmlDeviceGetPowerManagementLimit(dev, &tdp_mw);
        printf("  [%d]  %-28s  %5zu MiB", i, p.name,
               p.totalGlobalMem / (1024 * 1024));
        if (tdp_mw > 0) printf("  %4u W", tdp_mw / 1000);
        printf("\n");
    }

    printf("──────────────────────────────────────────────────────\n\n");
    nvmlShutdown();
}

/* ─────────────────────────────────────────────────────────
   main
   ───────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    /* 기본값 설정 */
    Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.duration_sec        = 3600;
    cfg.intensity           = 1;
    cfg.prec                = PREC_SGEMM_TF32;
    cfg.mem_spec.is_percent = 1;
    cfg.mem_spec.value      = 100.0;
    cfg.init_mode           = INIT_RAND;

    int list_only = 0;
    int opt;
    while ((opt = getopt(argc, argv, "t:i:p:m:g:X:I:lh")) != -1) {
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
            else                                    cfg.prec = PREC_SGEMM_TF32;
            break;
        case 'g':
            cfg.num_gpus = parse_gpu_list(optarg, cfg.gpu_ids, MAX_GPUS);
            break;
        case 'l': list_only = 1;                                         break;
        case 'h': print_usage(argv[0]); return 0;
        default:  print_usage(argv[0]); return 1;
        }
    }

    NVML_CHECK(nvmlInit());
    if (list_only) { print_gpu_list(); nvmlShutdown(); return 0; }

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
    printf("╠══════════════════════════════════════════════════════════════╣\n");

    /* ── GPU 별 초기화 ── */
    GpuCtx *gpus = (GpuCtx *)calloc(cfg.num_gpus, sizeof(GpuCtx));

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

        NVML_CHECK(nvmlDeviceGetHandleByIndex(gc->device_id, &gc->nvml_dev));
        nvmlDeviceGetPowerManagementLimit(gc->nvml_dev, &gc->tdp_mw);

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gc->device_id));

        /* 코어 정보 조회 및 캐싱 */
        const CoreEntry *ce = core_find(prop.name);
        if (ce)
            gc->core_info = *ce;
        else
            memset(&gc->core_info, 0, sizeof(CoreEntry));

        printf("  [초기화] GPU %d: %s", gc->device_id, prop.name);
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

        /* TF32 미지원 GPU에서 sgemm_tf32 실행 시 cuBLAS fallback 안내 */
        if (cfg.prec == PREC_SGEMM_TF32
            && gc->core_info.cores_fp32 > 0
            && gc->core_info.tc_ops_tf32 == 0)
            printf("           %s[참고] TF32 미지원 → cuBLAS가 FP32 SGEMM으로 자동 fallback "
                   "(Peak%% 기준도 FP32 shader)%s\n",
                   CLR_WARN, CLR_RESET);

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
    free(gpus);
    nvmlShutdown();
    return 0;
}
