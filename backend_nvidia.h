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
#include <cuda_fp16.h>
#include <nvml.h>

/* NVIDIA GPU 코어 DB (동적 Rpeak 계산용 GB_NVIDIA_CORE_ENTRIES 매크로 제공).
   본체 CORE_TABLE 안에서 #if defined(GB_BACKEND_NVIDIA) 로 펼쳐집니다. */
#include "core_table_nvidia.h"

/* ─────────────────────────────────────────────────────────
   GPU half 타입 (본체에서 __half 사용) — CUDA는 native 제공
   ───────────────────────────────────────────────────────── */
typedef __half gb_half;
static inline gb_half gb_float2half(float f) { return __float2half(f); }

/* ─────────────────────────────────────────────────────────
   Runtime 타입 별칭 (본체가 gb_stream_t 등으로 참조)
   ───────────────────────────────────────────────────────── */
typedef cudaStream_t gb_stream_t;

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
typedef cublasHandle_t gb_blas_handle_t;

static inline int gb_blas_create(gb_blas_handle_t *h, gb_stream_t stream)
{
    if (cublasCreate(h) != CUBLAS_STATUS_SUCCESS) return -1;
    cublasSetStream(*h, stream);
    /* Modern API: cublasGemmEx 가 compute type 을 명시 지정하므로
       handle math mode 는 무영향. DEFAULT 고정. */
    cublasSetMathMode(*h, CUBLAS_DEFAULT_MATH);
    return 0;
}

static inline void gb_blas_destroy(gb_blas_handle_t h)
{
    cublasDestroy(h);
}

/* op=N/N, alpha=1, beta=0, lda=M ldb=K ldc=M 고정.
   모든 정밀도를 modern cublasGemmEx 로 통일 (data/compute type 명시). */
static inline int gb_gemm(gb_blas_handle_t h, gb_prec_t prec,
                          int M, int N, int K,
                          const void *A, const void *B, void *C)
{
    cublasStatus_t st;
    switch (prec) {
    case GB_PREC_SGEMM: {
        const float alpha = 1.f, beta = 0.f;
        st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                          &alpha, A, CUDA_R_32F, M,
                                  B, CUDA_R_32F, K,
                          &beta,  C, CUDA_R_32F, M,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        break;
    }
    case GB_PREC_DGEMM: {
        const double alpha = 1.0, beta = 0.0;
        st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                          &alpha, A, CUDA_R_64F, M,
                                  B, CUDA_R_64F, K,
                          &beta,  C, CUDA_R_64F, M,
                          CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
        break;
    }
    case GB_PREC_HGEMM: {
        const __half alpha = __float2half(1.f), beta = __float2half(0.f);
        st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                          &alpha, A, CUDA_R_16F, M,
                                  B, CUDA_R_16F, K,
                          &beta,  C, CUDA_R_16F, M,
                          CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        break;
    }
    case GB_PREC_HGEMM_MIX: {
        const float alpha = 1.f, beta = 0.f;   /* FP16 in, FP32 acc */
        st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                          &alpha, A, CUDA_R_16F, M,
                                  B, CUDA_R_16F, K,
                          &beta,  C, CUDA_R_16F, M,
                          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        break;
    }
    case GB_PREC_SGEMM_TF32:
    default: {
        const float alpha = 1.f, beta = 0.f;
        st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
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

static inline int gb_mon_temp_c(gb_mon_t *m)
{
    unsigned int t = 0;
    if (nvmlDeviceGetTemperature(m->dev, NVML_TEMPERATURE_GPU, &t) != NVML_SUCCESS)
        return -1;
    return (int)t;
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
