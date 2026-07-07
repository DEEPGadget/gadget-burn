/**
 * backend_amd.h  —  HIP / rocBLAS / amd_smi 백엔드 구현
 * ─────────────────────────────────────────────────────────────
 * gpu_backend.h 가 정의한 계약을 AMD ROCm SDK로 구현합니다.
 *
 *   Runtime  : HIP        (cuda* 호출을 hip* 로 매크로 별칭)
 *   BLAS     : rocBLAS    (rocblas_gemm_ex, 직접 호출)
 *   모니터링 : amd_smi    (NVML 대체)
 *
 * 본체(gadget_burn.cu)는 cuda* 이름을 그대로 사용하고, 이 헤더가 그것을
 * hip* 로 치환합니다. 따라서 본체 소스는 벤더 중립으로 유지됩니다.
 * ───────────────────────────────────────────────────────────── */

#ifndef BACKEND_AMD_H
#define BACKEND_AMD_H

#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
/* rocblas_gemm_ex_get_solutions (autotune 용 solution 열거) 는 beta 기능 */
#define ROCBLAS_BETA_FEATURES
#include <rocblas/rocblas.h>
#include <rocblas/internal/rocblas-beta.h>
#include <hipblaslt/hipblaslt.h>
#include <amd_smi/amdsmi.h>

/* AMD GPU 코어 DB (동적 Rpeak 계산용 GB_AMD_CORE_ENTRIES 매크로 제공).
   본체 CORE_TABLE 안에서 #ifdef GB_BACKEND_AMD 로 펼쳐집니다. */
#include "core_table_amd.h"

/* ─────────────────────────────────────────────────────────
   [1] CUDA Runtime → HIP 별칭
   ─────────────────────────────────────────────────────────
   본체가 사용하는 cuda* 심볼 전체를 hip* 로 매핑합니다.
   (grep 으로 추출한 본체 사용 심볼 기준, 빠짐없이 1:1)
   ───────────────────────────────────────────────────────── */

/* 타입 */
#define cudaError_t              hipError_t
#define cudaStream_t             hipStream_t
#define cudaEvent_t              hipEvent_t
#define cudaDeviceProp           hipDeviceProp_t

/* 상수 */
#define cudaSuccess              hipSuccess

/* 디바이스 / 메모리 */
#define cudaSetDevice            hipSetDevice
#define cudaGetDeviceCount       hipGetDeviceCount
#define cudaGetDeviceProperties  hipGetDeviceProperties
#define cudaDeviceSynchronize    hipDeviceSynchronize
#define cudaMalloc               hipMalloc
#define cudaFree                 hipFree
#define cudaMemset               hipMemset
#define cudaMemGetInfo           hipMemGetInfo
#define cudaGetErrorString       hipGetErrorString
#define cudaDeviceGetPCIBusId    hipDeviceGetPCIBusId

/* 스트림 */
#define cudaStreamCreate         hipStreamCreate
#define cudaStreamDestroy        hipStreamDestroy
#define cudaStreamSynchronize    hipStreamSynchronize

/* 이벤트 */
#define cudaEventCreate          hipEventCreate
#define cudaEventDestroy         hipEventDestroy
#define cudaEventRecord          hipEventRecord
#define cudaEventElapsedTime     hipEventElapsedTime

/* ─────────────────────────────────────────────────────────
   GPU half 타입
   HIP 도 __half / __float2half 를 native 제공
   ───────────────────────────────────────────────────────── */
typedef __half gb_half;
static inline gb_half gb_float2half(float f) { return __float2half(f); }

/* GPU bfloat16 타입 (BF16 in / FP32 acc 경로용). HIP 는 __hip_bfloat16 제공. */
typedef __hip_bfloat16 gb_bfloat16;

/* GPU fp8 타입 (OCP e4m3). RDNA4(gfx1201)/CDNA 의 fp8 GEMM 입력용. */
typedef __hip_fp8_e4m3 gb_fp8;

typedef hipStream_t gb_stream_t;

/* 디바이스 열거 순서 정렬: HIP 는 기본적으로 PCI 버스 순서로 열거되고
   (amd-smi 와 동일), 모니터링은 gb_mon_open 이 BDF 로 직접 매칭하므로
   추가 정렬 강제가 불필요. no-op 으로 둔다 (NVIDIA 와 인터페이스 통일). */
static inline void gb_init_device_order(void) { (void)0; }

/* ─────────────────────────────────────────────────────────
   오류 처리 매크로
   ───────────────────────────────────────────────────────── */
#define GPU_CHECK(call)                                                \
    do {                                                               \
        hipError_t _e = (call);                                        \
        if (_e != hipSuccess) {                                        \
            fprintf(stderr, "\n[GPU ERR] %s:%d  %s\n",                 \
                    __FILE__, __LINE__, hipGetErrorString(_e));        \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

#define GB_BLAS_CHECK(call)                                            \
    do {                                                               \
        rocblas_status _s = (call);                                    \
        if (_s != rocblas_status_success) {                           \
            fprintf(stderr, "\n[BLAS ERR] %s:%d  code=%d\n",           \
                    __FILE__, __LINE__, (int)_s);                      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/* ─────────────────────────────────────────────────────────
   [2] BLAS GEMM (rocBLAS)
   ─────────────────────────────────────────────────────────
   rocblas_gemm_ex 는 cublasGemmEx 와 달리:
     - 출력 D 가 별도 인자 (in-place 로 C=D, ldc=ldd 사용)
     - compute_type 도 rocblas_datatype 재사용
     - 끝에 solution_index(0), flags(none) 추가
   TF32: rocBLAS 에는 cuBLAS 의 COMPUTE_32F_FAST_TF32 같은 단일 호출
   경로가 없고, XF32 가속은 CDNA(MI200+) 의 xDL 에서만 동작합니다.
   RDNA3(gfx1100) 등에서는 일반 f32_r 로 폴백합니다 (정확성 유지, TC 미가속).
   ───────────────────────────────────────────────────────── */
/* fp8(e4m3) GEMM 은 rocblas_gemm_ex 경로에 없고 hipBLASLt 가 담당하므로,
   핸들에 rocBLAS 핸들 + hipBLASLt 핸들 + workspace + fp8 실행계획 캐시를 함께
   보관한다. 본체는 gb_blas_handle_t 를 opaque 하게만 사용(포인터). */
typedef struct {
    rocblas_handle     blas;
    hipblasLtHandle_t  lt;
    void              *ws;
    size_t             ws_sz;
    hipStream_t        stream;
    int                rb_sol;   /* autotune 이 고른 rocBLAS solution_index (0=기본) */
    /* fp8 실행계획: 첫 fp8 호출 시 lazy 구성 (worker 당 M,N,K,prec 고정 가정) */
    int                              lt_ready;
    hipblasLtMatmulDesc_t            lt_desc;
    hipblasLtMatrixLayout_t          lt_lA, lt_lB, lt_lC, lt_lD;
    hipblasLtMatmulHeuristicResult_t lt_heur;
} gb_blas_ctx_t;
typedef gb_blas_ctx_t* gb_blas_handle_t;

static inline int gb_blas_create(gb_blas_handle_t *h, gb_stream_t stream)
{
    gb_blas_ctx_t *c = (gb_blas_ctx_t *)calloc(1, sizeof(gb_blas_ctx_t));
    if (!c) return -1;
    c->stream = stream;
    if (rocblas_create_handle(&c->blas) != rocblas_status_success) { free(c); return -1; }
    rocblas_set_stream(c->blas, stream);
    /* hipBLASLt 핸들 + workspace (fp8 전용, 다른 정밀도는 미사용) */
    if (hipblasLtCreate(&c->lt) != HIPBLAS_STATUS_SUCCESS) {
        rocblas_destroy_handle(c->blas); free(c); return -1;
    }
    c->ws_sz = 128ull * 1024 * 1024;   /* 128MB Lt workspace */
    if (hipMalloc(&c->ws, c->ws_sz) != hipSuccess) { c->ws = NULL; c->ws_sz = 0; }
    *h = c;
    return 0;
}

static inline void gb_blas_destroy(gb_blas_handle_t h)
{
    if (!h) return;
    if (h->lt_ready) {
        hipblasLtMatrixLayoutDestroy(h->lt_lA);
        hipblasLtMatrixLayoutDestroy(h->lt_lB);
        hipblasLtMatrixLayoutDestroy(h->lt_lC);
        hipblasLtMatrixLayoutDestroy(h->lt_lD);
        hipblasLtMatmulDescDestroy(h->lt_desc);
    }
    if (h->ws) hipFree(h->ws);
    hipblasLtDestroy(h->lt);
    rocblas_destroy_handle(h->blas);
    free(h);
}

/* fp8(e4m3) GEMM via hipBLASLt. e4m3 in / FP32 누산 / bf16 out(최종 downcast).
   compute=FP32 (gfx1201 에서 유효한 유일 경로; fp16/bf16 누산은 무효 커널로 확인됨).
   첫 호출 시 desc/layout/algo 를 캐시. 지원 algo 가 없으면 -1(→ 본체가 종료). */
static inline int gb_gemm_fp8(gb_blas_handle_t h, gb_prec_t prec,
                              int M, int N, int K,
                              const void *A, const void *B, void *C)
{
    (void)prec;
    if (!h->lt_ready) {
        hipDataType outT = HIP_R_16BF;
        if (hipblasLtMatmulDescCreate(&h->lt_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
            != HIPBLAS_STATUS_SUCCESS) return -1;
        hipblasLtMatrixLayoutCreate(&h->lt_lA, HIP_R_8F_E4M3, M, K, M);
        hipblasLtMatrixLayoutCreate(&h->lt_lB, HIP_R_8F_E4M3, K, N, K);
        hipblasLtMatrixLayoutCreate(&h->lt_lC, outT, M, N, M);
        hipblasLtMatrixLayoutCreate(&h->lt_lD, outT, M, N, M);
        hipblasLtMatmulPreference_t pref;
        if (hipblasLtMatmulPreferenceCreate(&pref) != HIPBLAS_STATUS_SUCCESS) return -1;
        hipblasLtMatmulPreferenceSetAttribute(pref,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &h->ws_sz, sizeof(h->ws_sz));
        int found = 0;
        hipblasStatus_t hs = hipblasLtMatmulAlgoGetHeuristic(h->lt, h->lt_desc,
            h->lt_lA, h->lt_lB, h->lt_lC, h->lt_lD, pref, 1, &h->lt_heur, &found);
        hipblasLtMatmulPreferenceDestroy(pref);
        if (hs != HIPBLAS_STATUS_SUCCESS || found == 0) return -1;
        h->lt_ready = 1;
    }
    const float alpha = 1.f, beta = 0.f;
    hipblasStatus_t st = hipblasLtMatmul(h->lt, h->lt_desc, &alpha,
        A, h->lt_lA, B, h->lt_lB, &beta,
        C, h->lt_lC, C, h->lt_lD, &h->lt_heur.algo, h->ws, h->ws_sz, h->stream);
    return (st == HIPBLAS_STATUS_SUCCESS) ? 0 : -1;
}

/* rocBLAS GEMM 1회 실행 (solution_index 지정 가능).
   sol==0 → rocBLAS 기본 선택(algo_standard), sol>0 → 지정 solution. */
static inline rocblas_status rb_gemm_call(rocblas_handle rb, gb_prec_t prec,
                                          int M, int N, int K,
                                          const void *A, const void *B, void *C,
                                          int sol)
{
    const rocblas_operation opN = rocblas_operation_none;
    const rocblas_gemm_algo algo = (sol == 0) ? rocblas_gemm_algo_standard
                                              : rocblas_gemm_algo_solution_index;
    const uint32_t flags = rocblas_gemm_flags_none;
    rocblas_status st;

    switch (prec) {
    case GB_PREC_DGEMM: {
        const double alpha = 1.0, beta = 0.0;
        st = rocblas_gemm_ex(rb, opN, opN, M, N, K, &alpha,
                A, rocblas_datatype_f64_r, M,
                B, rocblas_datatype_f64_r, K, &beta,
                C, rocblas_datatype_f64_r, M,
                C, rocblas_datatype_f64_r, M,
                rocblas_datatype_f64_r, algo, sol, flags);
        break;
    }
    case GB_PREC_HGEMM: {
        /* FP16 in / FP16 acc */
        const rocblas_half alpha = (rocblas_half){0x3C00};  /* 1.0 in fp16 */
        const rocblas_half beta  = (rocblas_half){0x0000};  /* 0.0 in fp16 */
        st = rocblas_gemm_ex(rb, opN, opN, M, N, K, &alpha,
                A, rocblas_datatype_f16_r, M,
                B, rocblas_datatype_f16_r, K, &beta,
                C, rocblas_datatype_f16_r, M,
                C, rocblas_datatype_f16_r, M,
                rocblas_datatype_f16_r, algo, sol, flags);
        break;
    }
    case GB_PREC_HGEMM_MIX: {
        /* FP16 in / FP32 acc — alpha,beta 는 compute_type(FP32) 기준 */
        const float alpha = 1.f, beta = 0.f;
        st = rocblas_gemm_ex(rb, opN, opN, M, N, K, &alpha,
                A, rocblas_datatype_f16_r, M,
                B, rocblas_datatype_f16_r, K, &beta,
                C, rocblas_datatype_f16_r, M,
                C, rocblas_datatype_f16_r, M,
                rocblas_datatype_f32_r, algo, sol, flags);
        break;
    }
    case GB_PREC_BF16: {
        /* BF16 in / FP32 acc — alpha,beta 는 compute_type(FP32) 기준 */
        const float alpha = 1.f, beta = 0.f;
        st = rocblas_gemm_ex(rb, opN, opN, M, N, K, &alpha,
                A, rocblas_datatype_bf16_r, M,
                B, rocblas_datatype_bf16_r, K, &beta,
                C, rocblas_datatype_bf16_r, M,
                C, rocblas_datatype_bf16_r, M,
                rocblas_datatype_f32_r, algo, sol, flags);
        break;
    }
    case GB_PREC_SGEMM:
    case GB_PREC_SGEMM_TF32:   /* RDNA3: TF32 가속 경로 없음 → f32 폴백 */
    default: {
        const float alpha = 1.f, beta = 0.f;
        st = rocblas_gemm_ex(rb, opN, opN, M, N, K, &alpha,
                A, rocblas_datatype_f32_r, M,
                B, rocblas_datatype_f32_r, K, &beta,
                C, rocblas_datatype_f32_r, M,
                C, rocblas_datatype_f32_r, M,
                rocblas_datatype_f32_r, algo, sol, flags);
        break;
    }
    }
    return st;
}

static inline int gb_gemm(gb_blas_handle_t h, gb_prec_t prec,
                          int M, int N, int K,
                          const void *A, const void *B, void *C)
{
    /* fp8 계열은 hipBLASLt 경로 (rocblas_gemm_ex 미지원) */
    if (prec == GB_PREC_FP8)
        return gb_gemm_fp8(h, prec, M, N, K, A, B, C);
    /* autotune 이 고른 solution(rb_sol) 사용 (미튜닝 시 0=기본) */
    return (rb_gemm_call(h->blas, prec, M, N, K, A, B, C, h->rb_sol)
            == rocblas_status_success) ? 0 : -1;
}

#ifndef GB_AT_ITERS
#define GB_AT_ITERS   4      /* 유망 후보 정밀 측정 반복 */
#define GB_AT_MAXCAND 96     /* solution 후보 상한 */
#endif

/* 후보 1개 실측 헬퍼: 먼저 1회(스크린)로 대략 속도를 재고, 현재 best 의 절반
   이상으로 유망하면 iters 회로 정밀 재측정. 느린 후보는 1회 비용만 든다
   (gfx1201 처럼 나쁜 solution 이 매우 느린 경우 autotune 시간 폭증 방지). */
#define GB_AT_TIME_CANDIDATE(CALL_EXPR, flop, stream, e0, e1, best, out_tf)   \
    do {                                                                      \
        (out_tf) = -1.0;                                                      \
        if ((CALL_EXPR) != 0) break;              /* 첫 호출 실패 → 무효 */    \
        hipStreamSynchronize(stream);                                         \
        hipEventRecord(e0, stream); (void)(CALL_EXPR);                        \
        hipEventRecord(e1, stream); hipEventSynchronize(e1);                  \
        float _ms1 = 0; hipEventElapsedTime(&_ms1, e0, e1);                   \
        double _tf1 = (_ms1 > 0) ? (flop) / (_ms1 * 1e-3) * 1e-12 : 0;        \
        (out_tf) = _tf1;                                                      \
        if (_tf1 <= (best) * 0.5) break;          /* 명백히 느림 → 스킵 */    \
        hipStreamSynchronize(stream); hipEventRecord(e0, stream);             \
        for (int _k = 0; _k < GB_AT_ITERS; _k++) (void)(CALL_EXPR);           \
        hipEventRecord(e1, stream); hipEventSynchronize(e1);                  \
        float _ms = 0; hipEventElapsedTime(&_ms, e0, e1);                     \
        if (_ms > 0) (out_tf) = (flop) / ((_ms / GB_AT_ITERS) * 1e-3) * 1e-12;\
    } while (0)

static inline void rb_prec_types(gb_prec_t p, rocblas_datatype *io, rocblas_datatype *comp)
{
    switch (p) {
        case GB_PREC_DGEMM:     *io=rocblas_datatype_f64_r;  *comp=rocblas_datatype_f64_r; break;
        case GB_PREC_HGEMM:     *io=rocblas_datatype_f16_r;  *comp=rocblas_datatype_f16_r; break;
        case GB_PREC_HGEMM_MIX: *io=rocblas_datatype_f16_r;  *comp=rocblas_datatype_f32_r; break;
        case GB_PREC_BF16:      *io=rocblas_datatype_bf16_r; *comp=rocblas_datatype_f32_r; break;
        default:                *io=rocblas_datatype_f32_r;  *comp=rocblas_datatype_f32_r; break;
    }
}

/* 실측 기반 최적 solution/algo 선택 → 핸들에 캐시. worker 버퍼(A,B,C)/스트림 사용.
   반환: 캐시된 최고 TFLOPS(참고용). 이후 gb_gemm 이 이 solution/algo 를 사용. */
static inline double gb_gemm_autotune(gb_blas_handle_t h, gb_prec_t prec,
                                      int M, int N, int K,
                                      const void *A, const void *B, void *C)
{
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    const double flop = 2.0 * M * (double)N * K;
    double best_tf = 0.0;

    if (prec == GB_PREC_FP8) {
        hipDataType outT = HIP_R_16BF;
        if (hipblasLtMatmulDescCreate(&h->lt_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
            == HIPBLAS_STATUS_SUCCESS) {
            hipblasLtMatrixLayoutCreate(&h->lt_lA, HIP_R_8F_E4M3, M, K, M);
            hipblasLtMatrixLayoutCreate(&h->lt_lB, HIP_R_8F_E4M3, K, N, K);
            hipblasLtMatrixLayoutCreate(&h->lt_lC, outT, M, N, M);
            hipblasLtMatrixLayoutCreate(&h->lt_lD, outT, M, N, M);
            hipblasLtMatmulPreference_t pref; hipblasLtMatmulPreferenceCreate(&pref);
            hipblasLtMatmulPreferenceSetAttribute(pref,
                HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &h->ws_sz, sizeof(h->ws_sz));
            const int REQ = 64; hipblasLtMatmulHeuristicResult_t cand[64]; int found = 0;
            hipblasLtMatmulAlgoGetHeuristic(h->lt, h->lt_desc,
                h->lt_lA, h->lt_lB, h->lt_lC, h->lt_lD, pref, REQ, cand, &found);
            hipblasLtMatmulPreferenceDestroy(pref);
            const float alpha = 1.f, beta = 0.f; int best_i = -1;
            for (int i = 0; i < found; i++) {
                double tf;
                GB_AT_TIME_CANDIDATE(
                    (hipblasLtMatmul(h->lt, h->lt_desc, &alpha, A,h->lt_lA, B,h->lt_lB,
                         &beta, C,h->lt_lC, C,h->lt_lD, &cand[i].algo, h->ws, h->ws_sz,
                         h->stream) == HIPBLAS_STATUS_SUCCESS ? 0 : -1),
                    flop, h->stream, e0, e1, best_tf, tf);
                if (tf > best_tf) { best_tf = tf; best_i = i; }
            }
            if (best_i >= 0) { h->lt_heur = cand[best_i]; h->lt_ready = 1; }
        }
    } else {
        rocblas_datatype io, comp; rb_prec_types(prec, &io, &comp);
        double d1=1,d0=0; float f1=1,f0=0; rocblas_half hh1={0x3C00}, hh0={0x0000};
        const void *pa=&f1, *pb=&f0;
        if (comp == rocblas_datatype_f64_r)      { pa=&d1; pb=&d0; }
        else if (comp == rocblas_datatype_f16_r) { pa=&hh1; pb=&hh0; }

        int best_sol = 0;
        /* 기본(solution_index=0) 을 baseline 으로 측정 */
        { double tf0;
          GB_AT_TIME_CANDIDATE(
              (rb_gemm_call(h->blas, prec, M,N,K, A,B,C, 0)==rocblas_status_success?0:-1),
              flop, h->stream, e0, e1, best_tf, tf0);
          if (tf0 > best_tf) best_tf = tf0; }
        /* 전체 solution 열거 (beta API) 후 각각 실측 */
        rocblas_int cnt = 0;
        rocblas_gemm_ex_get_solutions(h->blas, rocblas_operation_none, rocblas_operation_none,
            M,N,K, pa, A,io,M, B,io,K, pb, C,io,M, C,io,M, comp,
            rocblas_gemm_algo_solution_index, rocblas_gemm_flags_none, NULL, &cnt);
        if (cnt > 0) {
            if (cnt > GB_AT_MAXCAND) cnt = GB_AT_MAXCAND;
            rocblas_int *sols = (rocblas_int*)malloc((size_t)cnt * sizeof(rocblas_int));
            rocblas_int got = cnt;
            if (sols && rocblas_gemm_ex_get_solutions(h->blas, rocblas_operation_none,
                    rocblas_operation_none, M,N,K, pa, A,io,M, B,io,K, pb, C,io,M, C,io,M,
                    comp, rocblas_gemm_algo_solution_index, rocblas_gemm_flags_none, sols, &got)
                == rocblas_status_success) {
                for (int i = 0; i < got; i++) {
                    if (sols[i] == 0) continue;
                    double tf;
                    GB_AT_TIME_CANDIDATE(
                        (rb_gemm_call(h->blas, prec, M,N,K, A,B,C, sols[i])==rocblas_status_success?0:-1),
                        flop, h->stream, e0, e1, best_tf, tf);
                    if (tf > best_tf) { best_tf = tf; best_sol = sols[i]; }
                }
            }
            free(sols);
        }
        h->rb_sol = best_sol;
    }
    hipEventDestroy(e0); hipEventDestroy(e1);
    return best_tf;
}

/* ─────────────────────────────────────────────────────────
   [3] 모니터링 (amd_smi)
   ─────────────────────────────────────────────────────────
   NVML 과 달리 amd_smi 는 socket → processor handle 2단계 모델입니다.
   gb_mon_init() 에서 전체 GPU processor handle 을 평탄화해 배열에 캐시하고,
   gb_mon_open(dev_id) 은 그 배열의 dev_id 번째를 가져옵니다.
   (hipGetDeviceProperties 의 device index 순서와 amd_smi 의 enumeration
    순서가 일반적으로 일치. PCIe BDF 정렬 동일.)

   단위/모델 차이:
     - 전력: amdsmi_get_power_info().average_socket_power 는 W → ×1000 (mW 통일)
     - util: amdsmi_get_gpu_activity().gfx_activity (%) — 멀티-GPU 교차오염
             버그가 없어 NVML 의 GetSamples 우회가 불필요 (직접 현재값)
     - power cap: amdsmi_get_power_cap_info().power_cap 는 µW(baremetal)
                  → /1000 (mW 통일)
     - throttle: gpu_metrics.throttle_status 비트 의미가 NVIDIA와 달라
                 보수적으로 매핑 (자세한 비트 정의는 추후 정밀화 TODO).
   ───────────────────────────────────────────────────────── */

#define GB_AMD_MAX_PROC 64

typedef struct {
    amdsmi_processor_handle handle;
    int                     valid;
} gb_mon_t;

/* 전역 processor handle 캐시 (init 시 1회 채움) */
static amdsmi_processor_handle g_gb_amd_procs[GB_AMD_MAX_PROC];
static int                     g_gb_amd_proc_count = 0;

static inline int gb_mon_init(void)
{
    if (amdsmi_init(AMDSMI_INIT_AMD_GPUS) != AMDSMI_STATUS_SUCCESS)
        return -1;

    uint32_t sock_count = 0;
    if (amdsmi_get_socket_handles(&sock_count, NULL) != AMDSMI_STATUS_SUCCESS)
        return -1;
    if (sock_count == 0) return -1;

    amdsmi_socket_handle *socks =
        (amdsmi_socket_handle *)malloc(sock_count * sizeof(*socks));
    if (!socks) return -1;
    if (amdsmi_get_socket_handles(&sock_count, socks) != AMDSMI_STATUS_SUCCESS) {
        free(socks);
        return -1;
    }

    g_gb_amd_proc_count = 0;
    for (uint32_t s = 0; s < sock_count && g_gb_amd_proc_count < GB_AMD_MAX_PROC; s++) {
        uint32_t pc = 0;
        if (amdsmi_get_processor_handles(socks[s], &pc, NULL) != AMDSMI_STATUS_SUCCESS)
            continue;
        if (pc == 0) continue;

        amdsmi_processor_handle *procs =
            (amdsmi_processor_handle *)malloc(pc * sizeof(*procs));
        if (!procs) continue;
        if (amdsmi_get_processor_handles(socks[s], &pc, procs) == AMDSMI_STATUS_SUCCESS) {
            for (uint32_t p = 0; p < pc && g_gb_amd_proc_count < GB_AMD_MAX_PROC; p++) {
                processor_type_t ptype;
                if (amdsmi_get_processor_type(procs[p], &ptype) == AMDSMI_STATUS_SUCCESS
                    && ptype == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
                    g_gb_amd_procs[g_gb_amd_proc_count++] = procs[p];
                }
            }
        }
        free(procs);
    }
    free(socks);
    return (g_gb_amd_proc_count > 0) ? 0 : -1;
}

static inline void gb_mon_shutdown(void) { amdsmi_shut_down(); }

/* HIP device 인덱스 ↔ amd_smi processor handle 매핑.
   ───────────────────────────────────────────────────────────
   amd_smi 의 GPU enumeration 순서는 HIP 의 device 인덱스 순서와
   일치한다는 보장이 없습니다 (서로 다른 라이브러리·정렬 기준).
   순서가 어긋나면 burn 은 HIP device 0 에서 도는데 모니터링은 다른
   물리 GPU(idle)를 읽어 전력·클럭·util 이 전부 엉뚱하게 나옵니다.
   따라서 PCIe BDF(domain:bus:device)로 두 라이브러리의 핸들을
   정확히 매칭합니다. 매칭 실패 시에만 인덱스 순서로 폴백합니다. */
static inline int gb_mon_open(int dev_id, gb_mon_t *out)
{
    out->valid = 0;
    if (dev_id < 0 || dev_id >= g_gb_amd_proc_count) return -1;

    /* HIP device 의 PCI 좌표 조회 */
    hipDeviceProp_t prop;
    int matched = -1;
    if (hipGetDeviceProperties(&prop, dev_id) == hipSuccess) {
        for (int i = 0; i < g_gb_amd_proc_count; i++) {
            amdsmi_bdf_t bdf;
            if (amdsmi_get_gpu_device_bdf(g_gb_amd_procs[i], &bdf) != AMDSMI_STATUS_SUCCESS)
                continue;
            if ((int)bdf.bus_number    == prop.pciBusID &&
                (int)bdf.device_number == prop.pciDeviceID &&
                (int)bdf.domain_number == prop.pciDomainID) {
                matched = i;
                break;
            }
        }
    }

    out->handle = (matched >= 0) ? g_gb_amd_procs[matched]
                                 : g_gb_amd_procs[dev_id];  /* 폴백 */
    out->valid  = 1;
    return 0;
}

static inline unsigned gb_mon_power_mw(gb_mon_t *m)
{
    amdsmi_power_info_t info;
    if (amdsmi_get_power_info(m->handle, &info) != AMDSMI_STATUS_SUCCESS)
        return 0;
    /* Navi(RDNA) 계열은 average_socket_power(W) 사용. 일부 칩은 0 이면
       current_socket_power 로 폴백. mW 로 통일. */
    unsigned w = info.average_socket_power;
    if (w == 0 || w == 0xFFFF) w = info.current_socket_power;
    return w * 1000u;
}

static inline unsigned gb_mon_tdp_mw(gb_mon_t *m)
{
    amdsmi_power_cap_info_t cap;
    if (amdsmi_get_power_cap_info(m->handle, 0, &cap) != AMDSMI_STATUS_SUCCESS)
        return 0;
    /* baremetal: µW 단위 → mW 로 변환 */
    return (unsigned)(cap.power_cap / 1000ull);
}

/* 전력 캡(TDP) 설정. amd_smi 는 µW 단위(baremetal) → mW×1000.
   sensor index 0 (1차 소켓 센서). root 권한 필요. */
static inline int gb_mon_set_power_cap_mw(gb_mon_t *m, unsigned mw)
{
    uint64_t cap_uw = (uint64_t)mw * 1000ull;
    return (amdsmi_set_power_cap(m->handle, 0, cap_uw) == AMDSMI_STATUS_SUCCESS) ? 0 : -1;
}

/* 설정 가능한 캡 범위 [mW]. power_cap_info 의 min/max(µW) → mW. */
static inline int gb_mon_power_cap_range_mw(gb_mon_t *m,
                                            unsigned *min_mw, unsigned *max_mw)
{
    amdsmi_power_cap_info_t cap;
    if (amdsmi_get_power_cap_info(m->handle, 0, &cap) != AMDSMI_STATUS_SUCCESS)
        return -1;
    if (min_mw) *min_mw = (unsigned)(cap.min_power_cap / 1000ull);
    if (max_mw) *max_mw = (unsigned)(cap.max_power_cap / 1000ull);
    return 0;
}

/* edge(표면)·junction(hotspot) 온도를 동시 조회.
   RDNA/CDNA 모두 두 센서를 노출한다. edge 는 카드 표면(만졌을 때 체감)에,
   hotspot 은 다이 국소 최고점에 해당해 부하 시 edge 보다 30~50°C 높다. */
static inline int gb_mon_temp2_c(gb_mon_t *m, int *edge_c, int *hot_c)
{
    int64_t t = 0;
    int e = -1, h = -1;
    if (amdsmi_get_temp_metric(m->handle, AMDSMI_TEMPERATURE_TYPE_EDGE,
                               AMDSMI_TEMP_CURRENT, &t) == AMDSMI_STATUS_SUCCESS)
        e = (int)t;
    if (amdsmi_get_temp_metric(m->handle, AMDSMI_TEMPERATURE_TYPE_HOTSPOT,
                               AMDSMI_TEMP_CURRENT, &t) == AMDSMI_STATUS_SUCCESS)
        h = (int)t;
    if (e < 0 && h >= 0) e = h;   /* edge 미지원 시 hotspot 으로 폴백 */
    if (edge_c) *edge_c = e;
    if (hot_c)  *hot_c  = h;
    return (e >= 0) ? 0 : -1;
}

static inline unsigned gb_mon_clock_mhz(gb_mon_t *m)
{
    amdsmi_clk_info_t info;
    if (amdsmi_get_clock_info(m->handle, AMDSMI_CLK_TYPE_GFX, &info)
        != AMDSMI_STATUS_SUCCESS)
        return 0;
    return info.clk;   /* MHz */
}

/* amd_smi 는 GPU handle 별 즉시 현재값을 반환하므로 NVML 같은
   멀티-GPU 교차오염이 없습니다. 링버퍼/타임스탬프 추적 불필요. */
static inline double gb_mon_util_pct(gb_mon_t *m)
{
    amdsmi_engine_usage_t use;
    if (amdsmi_get_gpu_activity(m->handle, &use) != AMDSMI_STATUS_SUCCESS)
        return -1.0;
    return (double)use.gfx_activity;
}

static inline unsigned gb_mon_throttle(gb_mon_t *m)
{
    amdsmi_gpu_metrics_t gm;
    if (amdsmi_get_gpu_metrics_info(m->handle, &gm) != AMDSMI_STATUS_SUCCESS)
        return GB_THROTTLE_NONE;
    /* throttle_status != 0 이면 어떤 형태든 throttling 중.
       AMD 의 indep_throttle_status 비트 정의는 ASIC 별로 달라
       thermal/power 세분류는 보수적으로 둡니다 (TODO: gfx1100 비트맵 정밀화).
       현재는 0이 아니면 POWER_BRAKE 로 표시 (대다수 burn 상황은 전력 한계). */
    if (gm.throttle_status != 0)
        return GB_THROTTLE_POWER_BRAKE;
    return GB_THROTTLE_NONE;
}

#endif /* BACKEND_AMD_H */
