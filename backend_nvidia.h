/**
 * backend_nvidia.h  —  CUDA / cuBLAS / NVML 백엔드 구현
 * ─────────────────────────────────────────────────────────────
 * gpu_backend.h 가 정의한 계약을 NVIDIA SDK로 구현합니다.
 * 본체(gadget_burn.cu)의 기존 cuda / cublas / nvml 호출은 대부분
 * 그대로 유지되며, BLAS GEMM 과 모니터링만 gb_* 래퍼로 통일합니다.
 * ───────────────────────────────────────────────────────────── */

#ifndef BACKEND_NVIDIA_H
#define BACKEND_NVIDIA_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <nvml.h>

/* NVIDIA GPU 코어 DB (동적 Rpeak 계산용 GB_NVIDIA_CORE_ENTRIES 매크로 제공).
   본체 CORE_TABLE 안에서 #if defined(GB_BACKEND_NVIDIA) 로 펼쳐집니다. */
#include "core_table_nvidia.h"

/* ─────────────────────────────────────────────────────────
   GPU half 타입 (본체에서 __half 사용) — CUDA는 native 제공
   ───────────────────────────────────────────────────────── */
typedef __half gb_half;
static inline gb_half gb_float2half(float f) { return __float2half(f); }

/* GPU bfloat16 타입 (BF16 in / FP32 acc 경로용). CUDA 는 __nv_bfloat16 제공. */
typedef __nv_bfloat16 gb_bfloat16;

/* GPU fp8 타입 (OCP e4m3). Ada/Hopper+ 의 fp8 GEMM 입력용. */
typedef __nv_fp8_e4m3 gb_fp8;

/* ─────────────────────────────────────────────────────────
   Runtime 타입 별칭 (본체가 gb_stream_t 등으로 참조)
   ───────────────────────────────────────────────────────── */
typedef cudaStream_t gb_stream_t;

/* ─────────────────────────────────────────────────────────
   디바이스 열거 순서 정렬
   ─────────────────────────────────────────────────────────
   CUDA 런타임 기본값(CUDA_DEVICE_ORDER 미설정)은 FASTEST_FIRST 라
   성능 휴리스틱으로 디바이스를 재정렬합니다. 반면 nvidia-smi/NVML 은
   PCI 버스 순서로 열거하므로, 서로 다른 GPU 가 섞인 시스템에서는
   CUDA index 와 nvidia-smi index 가 어긋납니다 (-g 선택과 모니터링이
   다른 카드를 가리킴). PCI_BUS_ID 로 강제해 둘을 일치시킵니다.
   반드시 첫 CUDA 런타임 호출 이전에 호출해야 적용됩니다. */
static inline void gb_init_device_order(void)
{
    setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);
}

/* ─────────────────────────────────────────────────────────
   오류 처리 매크로 (본체에서 사용)
   ───────────────────────────────────────────────────────── */
#define GPU_CHECK(call)                                                \
    do {                                                               \
        cudaError_t _e = (call);                                       \
        if (_e != cudaSuccess) {                                       \
            fprintf(stderr, "\n[GPU ERR] %s:%d  %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_e));       \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

#define GB_BLAS_CHECK(call)                                            \
    do {                                                               \
        cublasStatus_t _s = (call);                                    \
        if (_s != CUBLAS_STATUS_SUCCESS) {                             \
            fprintf(stderr, "\n[BLAS ERR] %s:%d  code=%d\n",           \
                    __FILE__, __LINE__, (int)_s);                      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/* ─────────────────────────────────────────────────────────
   BLAS GEMM
   ───────────────────────────────────────────────────────── */
/* fp8(e4m3) GEMM 은 cublasGemmEx 에 없고 cuBLASLt 가 담당하므로, 핸들에
   cuBLAS 핸들 + cuBLASLt 핸들 + workspace + fp8 실행계획 캐시를 함께 보관한다.
   본체는 gb_blas_handle_t 를 opaque 하게만 사용(포인터). */
typedef struct {
    cublasHandle_t     blas;
    cublasLtHandle_t   lt;
    void              *ws;
    size_t             ws_sz;
    cudaStream_t       stream;
    int                        lt_ready;   /* fp8 계획 lazy 구성 여부 */
    cublasLtMatmulDesc_t       lt_desc;
    cublasLtMatrixLayout_t     lt_lA, lt_lB, lt_lC, lt_lD;
    cublasLtMatmulHeuristicResult_t lt_heur;
} gb_blas_ctx_t;
typedef gb_blas_ctx_t* gb_blas_handle_t;

static inline int gb_blas_create(gb_blas_handle_t *h, gb_stream_t stream)
{
    gb_blas_ctx_t *c = (gb_blas_ctx_t *)calloc(1, sizeof(gb_blas_ctx_t));
    if (!c) return -1;
    c->stream = stream;
    if (cublasCreate(&c->blas) != CUBLAS_STATUS_SUCCESS) { free(c); return -1; }
    cublasSetStream(c->blas, stream);
    cublasSetMathMode(c->blas, CUBLAS_DEFAULT_MATH);
    if (cublasLtCreate(&c->lt) != CUBLAS_STATUS_SUCCESS) { cublasDestroy(c->blas); free(c); return -1; }
    c->ws_sz = 128ull * 1024 * 1024;
    if (cudaMalloc(&c->ws, c->ws_sz) != cudaSuccess) { c->ws = NULL; c->ws_sz = 0; }
    *h = c;
    return 0;
}

static inline void gb_blas_destroy(gb_blas_handle_t h)
{
    if (!h) return;
    if (h->lt_ready) {
        cublasLtMatrixLayoutDestroy(h->lt_lA);
        cublasLtMatrixLayoutDestroy(h->lt_lB);
        cublasLtMatrixLayoutDestroy(h->lt_lC);
        cublasLtMatrixLayoutDestroy(h->lt_lD);
        cublasLtMatmulDescDestroy(h->lt_desc);
    }
    if (h->ws) cudaFree(h->ws);
    cublasLtDestroy(h->lt);
    cublasDestroy(h->blas);
    free(h);
}

/* fp8(e4m3) GEMM via cuBLASLt.
     GB_PREC_FP8     : e4m3 in / e4m3 out
     GB_PREC_FP8_MIX : e4m3 in / bf16 out
   compute=FP32. cuBLASLt fp8 는 "TN"(opA=T, opB=N) 레이아웃만 지원하므로
   A 를 transpose 로 지정한다(정사각 M=N=K 라 버퍼 크기는 동일; burn 목적상
   결과 정확성은 무의미). 첫 호출 시 desc/layout/algo 를 캐시.
   ※ NVIDIA fp8 HW 미보유로 실기 미검증 — 실패 시 본체가 명시 종료. */
static inline int gb_gemm_fp8(gb_blas_handle_t h, gb_prec_t prec,
                              int M, int N, int K,
                              const void *A, const void *B, void *C)
{
    if (!h->lt_ready) {
        cudaDataType_t outT = (prec == GB_PREC_FP8) ? CUDA_R_8F_E4M3 : CUDA_R_16BF;
        if (cublasLtMatmulDescCreate(&h->lt_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F)
            != CUBLAS_STATUS_SUCCESS) return -1;
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(h->lt_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(h->lt_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        /* TN: A=[K,M] ld=K, B=[K,N] ld=K, C/D=[M,N] ld=M */
        cublasLtMatrixLayoutCreate(&h->lt_lA, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&h->lt_lB, CUDA_R_8F_E4M3, K, N, K);
        cublasLtMatrixLayoutCreate(&h->lt_lC, outT, M, N, M);
        cublasLtMatrixLayoutCreate(&h->lt_lD, outT, M, N, M);
        cublasLtMatmulPreference_t pref;
        if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) return -1;
        cublasLtMatmulPreferenceSetAttribute(pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &h->ws_sz, sizeof(h->ws_sz));
        int found = 0;
        cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(h->lt, h->lt_desc,
            h->lt_lA, h->lt_lB, h->lt_lC, h->lt_lD, pref, 1, &h->lt_heur, &found);
        cublasLtMatmulPreferenceDestroy(pref);
        if (st != CUBLAS_STATUS_SUCCESS || found == 0) return -1;
        h->lt_ready = 1;
    }
    const float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasLtMatmul(h->lt, h->lt_desc, &alpha,
        A, h->lt_lA, B, h->lt_lB, &beta,
        C, h->lt_lC, C, h->lt_lD, &h->lt_heur.algo, h->ws, h->ws_sz, h->stream);
    return (st == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

/* op=N/N, alpha=1, beta=0, lda=M ldb=K ldc=M 고정.
   모든 정밀도를 modern cublasGemmEx 로 통일 (data/compute type 명시). */
static inline int gb_gemm(gb_blas_handle_t h, gb_prec_t prec,
                          int M, int N, int K,
                          const void *A, const void *B, void *C)
{
    /* fp8 계열은 cuBLASLt 경로 (cublasGemmEx 미지원) */
    if (prec == GB_PREC_FP8 || prec == GB_PREC_FP8_MIX)
        return gb_gemm_fp8(h, prec, M, N, K, A, B, C);

    cublasHandle_t hb = h->blas;
    cublasStatus_t st;
    switch (prec) {
    case GB_PREC_SGEMM: {
        const float alpha = 1.f, beta = 0.f;
        st = cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                          &alpha, A, CUDA_R_32F, M,
                                  B, CUDA_R_32F, K,
                          &beta,  C, CUDA_R_32F, M,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        break;
    }
    case GB_PREC_DGEMM: {
        const double alpha = 1.0, beta = 0.0;
        st = cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                          &alpha, A, CUDA_R_64F, M,
                                  B, CUDA_R_64F, K,
                          &beta,  C, CUDA_R_64F, M,
                          CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
        break;
    }
    case GB_PREC_HGEMM: {
        const __half alpha = __float2half(1.f), beta = __float2half(0.f);
        st = cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                          &alpha, A, CUDA_R_16F, M,
                                  B, CUDA_R_16F, K,
                          &beta,  C, CUDA_R_16F, M,
                          CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        break;
    }
    case GB_PREC_HGEMM_MIX: {
        const float alpha = 1.f, beta = 0.f;   /* FP16 in, FP32 acc */
        st = cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                          &alpha, A, CUDA_R_16F, M,
                                  B, CUDA_R_16F, K,
                          &beta,  C, CUDA_R_16F, M,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        break;
    }
    case GB_PREC_BF16: {
        const float alpha = 1.f, beta = 0.f;   /* BF16 in, FP32 acc */
        st = cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                          &alpha, A, CUDA_R_16BF, M,
                                  B, CUDA_R_16BF, K,
                          &beta,  C, CUDA_R_16BF, M,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        break;
    }
    case GB_PREC_SGEMM_TF32:
    default: {
        const float alpha = 1.f, beta = 0.f;
        st = cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                          &alpha, A, CUDA_R_32F, M,
                                  B, CUDA_R_32F, K,
                          &beta,  C, CUDA_R_32F, M,
                          CUBLAS_COMPUTE_32F_FAST_TF32,
                          CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        break;
    }
    }
    return (st == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

#ifndef GB_AT_ITERS
#define GB_AT_ITERS   4
#endif

/* 후보 1회 스크린 후 유망하면 iters 정밀 측정 (느린 후보 시간 폭증 방지). */
#define GB_AT_TIME_CANDIDATE(CALL_EXPR, flop, stream, e0, e1, best, out_tf)   \
    do {                                                                      \
        (out_tf) = -1.0;                                                      \
        if ((CALL_EXPR) != 0) break;                                          \
        cudaStreamSynchronize(stream);                                        \
        cudaEventRecord(e0, stream); (void)(CALL_EXPR);                       \
        cudaEventRecord(e1, stream); cudaEventSynchronize(e1);                \
        float _ms1 = 0; cudaEventElapsedTime(&_ms1, e0, e1);                  \
        double _tf1 = (_ms1 > 0) ? (flop) / (_ms1 * 1e-3) * 1e-12 : 0;        \
        (out_tf) = _tf1;                                                      \
        if (_tf1 <= (best) * 0.5) break;                                      \
        cudaStreamSynchronize(stream); cudaEventRecord(e0, stream);           \
        for (int _k = 0; _k < GB_AT_ITERS; _k++) (void)(CALL_EXPR);           \
        cudaEventRecord(e1, stream); cudaEventSynchronize(e1);                \
        float _ms = 0; cudaEventElapsedTime(&_ms, e0, e1);                    \
        if (_ms > 0) (out_tf) = (flop) / ((_ms / GB_AT_ITERS) * 1e-3) * 1e-12;\
    } while (0)

/* 실측 기반 최적 algo 선택 → 핸들에 캐시.
   NVIDIA 는 fp8(cuBLASLt)만 autotune 실효 — cublasGemmEx 클래식 경로는 cuBLAS 가
   내부 자동선택하며 cublasGemmAlgo_t 힌트는 최신 GPU 에서 대부분 무시되므로 no-op.
   반환: 튜닝된 최고 TFLOPS(참고용). fp8 미해당/미튜닝이면 0. */
static inline double gb_gemm_autotune(gb_blas_handle_t h, gb_prec_t prec,
                                      int M, int N, int K,
                                      const void *A, const void *B, void *C)
{
    if (prec != GB_PREC_FP8 && prec != GB_PREC_FP8_MIX)
        return 0.0;   /* 클래식 경로: cuBLAS 기본 선택 사용 */

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    const double flop = 2.0 * M * (double)N * K;
    double best_tf = 0.0;

    cudaDataType_t outT = (prec == GB_PREC_FP8) ? CUDA_R_8F_E4M3 : CUDA_R_16BF;
    if (cublasLtMatmulDescCreate(&h->lt_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F)
        == CUBLAS_STATUS_SUCCESS) {
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(h->lt_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(h->lt_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtMatrixLayoutCreate(&h->lt_lA, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&h->lt_lB, CUDA_R_8F_E4M3, K, N, K);
        cublasLtMatrixLayoutCreate(&h->lt_lC, outT, M, N, M);
        cublasLtMatrixLayoutCreate(&h->lt_lD, outT, M, N, M);
        cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &h->ws_sz, sizeof(h->ws_sz));
        const int REQ = 64; cublasLtMatmulHeuristicResult_t cand[64]; int found = 0;
        cublasLtMatmulAlgoGetHeuristic(h->lt, h->lt_desc,
            h->lt_lA, h->lt_lB, h->lt_lC, h->lt_lD, pref, REQ, cand, &found);
        cublasLtMatmulPreferenceDestroy(pref);
        const float alpha = 1.f, beta = 0.f; int best_i = -1;
        for (int i = 0; i < found; i++) {
            double tf;
            GB_AT_TIME_CANDIDATE(
                (cublasLtMatmul(h->lt, h->lt_desc, &alpha, A,h->lt_lA, B,h->lt_lB,
                     &beta, C,h->lt_lC, C,h->lt_lD, &cand[i].algo, h->ws, h->ws_sz,
                     h->stream) == CUBLAS_STATUS_SUCCESS ? 0 : -1),
                flop, h->stream, e0, e1, best_tf, tf);
            if (tf > best_tf) { best_tf = tf; best_i = i; }
        }
        if (best_i >= 0) { h->lt_heur = cand[best_i]; h->lt_ready = 1; }
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return best_tf;
}

/* ─────────────────────────────────────────────────────────
   모니터링 (NVML)
   ─────────────────────────────────────────────────────────
   gb_mon_t 는 NVML device handle + util-samples 의 last timestamp 를
   함께 보관합니다 (멀티-GPU util 교차오염 우회용 GPU별 링버퍼 상태).
   ───────────────────────────────────────────────────────── */
typedef struct {
    nvmlDevice_t       dev;
    unsigned long long util_last_ts;  /* GetSamples 진행 위치 */
    int                valid;
} gb_mon_t;

static inline int  gb_mon_init(void)      { return (nvmlInit() == NVML_SUCCESS) ? 0 : -1; }
static inline void gb_mon_shutdown(void)  { nvmlShutdown(); }

static inline int gb_mon_open(int dev_id, gb_mon_t *out)
{
    out->util_last_ts = 0;
    out->valid = 0;
    if (nvmlDeviceGetHandleByIndex(dev_id, &out->dev) != NVML_SUCCESS)
        return -1;
    out->valid = 1;
    return 0;
}

static inline unsigned gb_mon_power_mw(gb_mon_t *m)
{
    unsigned int pw = 0;  /* NVML 은 이미 mW */
    if (nvmlDeviceGetPowerUsage(m->dev, &pw) != NVML_SUCCESS) return 0;
    return pw;
}

static inline unsigned gb_mon_tdp_mw(gb_mon_t *m)
{
    unsigned int lim = 0;  /* NVML power management limit, mW */
    if (nvmlDeviceGetPowerManagementLimit(m->dev, &lim) != NVML_SUCCESS) return 0;
    return lim;
}

/* 전력 캡(TDP) 설정. NVML 은 mW 단위. root 권한 필요(아니면 NO_PERMISSION). */
static inline int gb_mon_set_power_cap_mw(gb_mon_t *m, unsigned mw)
{
    return (nvmlDeviceSetPowerManagementLimit(m->dev, mw) == NVML_SUCCESS) ? 0 : -1;
}

/* 설정 가능한 캡 범위 [mW]. 본체가 요청값을 이 범위로 클램프. */
static inline int gb_mon_power_cap_range_mw(gb_mon_t *m,
                                            unsigned *min_mw, unsigned *max_mw)
{
    unsigned int lo = 0, hi = 0;
    if (nvmlDeviceGetPowerManagementLimitConstraints(m->dev, &lo, &hi) != NVML_SUCCESS)
        return -1;
    if (min_mw) *min_mw = lo;
    if (max_mw) *max_mw = hi;
    return 0;
}

/* edge(GPU 표준 온도)·junction(hotspot) 동시 조회.
   NVML 은 NVML_TEMPERATURE_GPU(=nvidia-smi 가 보여주는 GPU 온도)를 edge 로
   제공한다. hotspot(junction)은 대부분(특히 소비자) GPU 에서 NVML 로 노출되지
   않으므로 N/A(-1)로 둔다. (본체는 -1 이면 junction 을 표시하지 않는다.) */
static inline int gb_mon_temp2_c(gb_mon_t *m, int *edge_c, int *hot_c)
{
    unsigned int t = 0;
    int e = -1;
    if (nvmlDeviceGetTemperature(m->dev, NVML_TEMPERATURE_GPU, &t) == NVML_SUCCESS)
        e = (int)t;
    if (edge_c) *edge_c = e;
    if (hot_c)  *hot_c  = -1;   /* NVML junction 미노출 → N/A */
    return (e >= 0) ? 0 : -1;
}

static inline unsigned gb_mon_clock_mhz(gb_mon_t *m)
{
    unsigned int c = 0;
    if (nvmlDeviceGetClockInfo(m->dev, NVML_CLOCK_SM, &c) != NVML_SUCCESS) return 0;
    return c;
}

/* nvmlDeviceGetUtilizationRates() 는 멀티-GPU 에서 드라이버 공유 버퍼로
   인해 모든 GPU 가 같은 값을 반환하는 버그가 있어, GPU 별 독립 링버퍼인
   nvmlDeviceGetSamples() 를 사용합니다. */
static inline double gb_mon_util_pct(gb_mon_t *m)
{
    unsigned int count = 0;
    nvmlValueType_t vtype;
    nvmlReturn_t r = nvmlDeviceGetSamples(
        m->dev, NVML_GPU_UTILIZATION_SAMPLES, m->util_last_ts,
        &vtype, &count, NULL);
    if (r != NVML_SUCCESS || count == 0) return -1.0;

    nvmlSample_t *buf = (nvmlSample_t *)malloc(count * sizeof(nvmlSample_t));
    if (!buf) return -1.0;

    r = nvmlDeviceGetSamples(
        m->dev, NVML_GPU_UTILIZATION_SAMPLES, m->util_last_ts,
        &vtype, &count, buf);

    double sum = 0.0; unsigned int valid = 0;
    if (r == NVML_SUCCESS) {
        for (unsigned int i = 0; i < count; i++) {
            sum += buf[i].sampleValue.uiVal;
            if (buf[i].timeStamp > m->util_last_ts)
                m->util_last_ts = buf[i].timeStamp;
            valid++;
        }
    }
    free(buf);
    return (valid > 0) ? sum / valid : -1.0;
}

static inline unsigned gb_mon_throttle(gb_mon_t *m)
{
    unsigned long long reasons = 0;
    if (nvmlDeviceGetCurrentClocksThrottleReasons(m->dev, &reasons) != NVML_SUCCESS)
        return GB_THROTTLE_NONE;
    unsigned out = GB_THROTTLE_NONE;
    if (reasons & (nvmlClocksThrottleReasonSwThermalSlowdown
                 | nvmlClocksThrottleReasonHwThermalSlowdown))
        out |= GB_THROTTLE_THERMAL;
    if (reasons & nvmlClocksThrottleReasonHwPowerBrakeSlowdown)
        out |= GB_THROTTLE_POWER_BRAKE;
    return out;
}

#endif /* BACKEND_NVIDIA_H */
