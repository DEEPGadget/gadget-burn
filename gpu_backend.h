/**
 * gpu_backend.h
 *
 * 벤더 추상화 레이어 (NVIDIA CUDA / AMD ROCm 공통 인터페이스)
 * ─────────────────────────────────────────────────────────────
 * 본체(gadget_burn.cu)는 이 헤더가 제공하는 인터페이스만 사용하고,
 * 실제 구현은 빌드 시점에 선택된 백엔드 헤더로 분기됩니다.
 *
 *   GB_BACKEND_NVIDIA  → backend_nvidia.h  (CUDA / cuBLAS / NVML)
 *   GB_BACKEND_AMD     → backend_amd.h     (HIP  / rocBLAS / amd_smi)
 *
 * 빌드:
 *   make nvidia   (-DGB_BACKEND_NVIDIA, 기본)
 *   make amd      (-DGB_BACKEND_AMD)
 *
 * 추상화 범위
 * ─────────────────────────────────────────────────────────────
 * 1) GPU Runtime  : 메모리/스트림/이벤트/디바이스 질의
 *                   → CUDA와 HIP의 이름이 거의 동일하므로 각 backend
 *                     헤더에서 매크로 별칭(cuda* → hip*)으로 처리.
 *                     본체는 기존 cuda* 호출을 그대로 유지.
 * 2) BLAS GEMM    : cublasGemmEx / rocblas_gemm_ex 의 시그니처가 다르므로
 *                   gb_blas_* 핸들 타입 + gb_gemm() 단일 함수로 통일.
 * 3) 모니터링     : NVML / amd_smi 의 핸들 모델·단위가 다르므로
 *                   gb_mon_* 인터페이스로 통일 (전력/온도/클럭/util/throttle).
 *
 * 본 헤더 자체는 백엔드 중립이며, 어떤 vendor SDK 헤더도 직접 포함하지
 * 않습니다. 구체 타입(gb_blas_handle_t 등)과 enum 값은 backend_*.h 가
 * 정의합니다.
 * ───────────────────────────────────────────────────────────── */

#ifndef GPU_BACKEND_H
#define GPU_BACKEND_H

#include <stddef.h>

/* ─────────────────────────────────────────────────────────
   백엔드 선택 (정확히 하나만 정의되어야 함)
   ───────────────────────────────────────────────────────── */
#if defined(GB_BACKEND_NVIDIA) && defined(GB_BACKEND_AMD)
#  error "GB_BACKEND_NVIDIA 와 GB_BACKEND_AMD 를 동시에 정의할 수 없습니다."
#endif
#if !defined(GB_BACKEND_NVIDIA) && !defined(GB_BACKEND_AMD)
#  warning "백엔드가 지정되지 않아 GB_BACKEND_NVIDIA 로 기본 설정합니다."
#  define GB_BACKEND_NVIDIA
#endif

/* ─────────────────────────────────────────────────────────
   백엔드 식별 문자열 (출력/로그용)
   ───────────────────────────────────────────────────────── */
#if defined(GB_BACKEND_NVIDIA)
#  define GB_BACKEND_NAME   "NVIDIA CUDA"
#  define GB_BLAS_NAME      "cuBLAS"
#  define GB_MON_NAME       "NVML"
#elif defined(GB_BACKEND_AMD)
#  define GB_BACKEND_NAME   "AMD ROCm"
#  define GB_BLAS_NAME      "rocBLAS"
#  define GB_MON_NAME       "amd_smi"
#endif

/* ─────────────────────────────────────────────────────────
   정밀도 타입 (본체와 backend가 공유)
   본체에 PrecType이 이미 있으나, backend 헤더가 본체보다 먼저
   포함될 수 있으므로 공용 정의를 여기에 둡니다.
   ───────────────────────────────────────────────────────── */
typedef enum {
    GB_PREC_SGEMM,      /* FP32 (shader core, TC/Matrix 미사용)        */
    GB_PREC_DGEMM,      /* FP64                                         */
    GB_PREC_HGEMM,      /* FP16 in / FP16 acc (Tensor/Matrix Core)      */
    GB_PREC_HGEMM_MIX,  /* FP16 in / FP32 acc (mixed precision)         */
    GB_PREC_SGEMM_TF32, /* FP32 storage + TF32/XF32 compute
                           (RDNA3 등 미지원 HW에서는 FP32로 폴백)        */
    GB_PREC_BF16,       /* BF16 in / FP32 acc (Tensor/Matrix Core).
                           FP16 과 동일 처리율(Rpeak=tc_ops_mix 기준).        */
    GB_PREC_FP8         /* FP8(e4m3) in / FP32 누산 (hipBLASLt/cuBLASLt). 출력은
                           bf16(단순 최종 downcast, 성능·연산 의미 없음).
                           지원 HW 에서 FP16 의 2배 처리율(Rpeak=tc_ops_fp8).
                           플래그명 fp8_afp32 (= fp8 accumulate fp32). fp16/bf16
                           누산은 gfx1201 등에서 무효(무연산)라 FP32 만 제공.
                           ※ enum 은 항상 끝에 추가 — prec_str[] 서수 인덱싱. */
} gb_prec_t;

/* 백엔드별 기본 정밀도 (-p 미지정 시).
   - NVIDIA: sgemm_tf32 (FP32 storage + TF32 TC, gpu-burn -tc 호환). FP32
             메모리 부하 + TC compute 부하를 동시에 거는 균형 워크로드.
   - AMD:    hgemm_mix (FP16 in / FP32 acc). CDNA/RDNA 의 Matrix Core 가
             FP16/FP32-acc 를 네이티브 고속 경로로 처리해 실측 TFLOPS·TDP
             가 가장 높음 (FP16/FP16-acc 인 hgemm 은 CDNA 에서 저속 폴백). */
#if defined(GB_BACKEND_AMD)
#  define GB_DEFAULT_PREC       GB_PREC_HGEMM_MIX
#  define GB_DEFAULT_PREC_NAME  "hgemm_mix"
#else
#  define GB_DEFAULT_PREC       GB_PREC_SGEMM_TF32
#  define GB_DEFAULT_PREC_NAME  "sgemm_tf32"
#endif

/* ─────────────────────────────────────────────────────────
   Throttle 사유 비트마스크 (벤더 중립)
   각 backend가 자기 native 비트를 이 공용 비트로 매핑합니다.
   본체는 이 GB_THROTTLE_* 만 참조합니다.
   ───────────────────────────────────────────────────────── */
#define GB_THROTTLE_NONE        0x0u
#define GB_THROTTLE_THERMAL     0x1u   /* SW/HW thermal slowdown */
#define GB_THROTTLE_POWER_BRAKE 0x2u   /* HW power brake         */

/* ─────────────────────────────────────────────────────────
   BLAS / 모니터링 핸들 타입과 함수 시그니처는 backend 헤더가 정의.
   여기서는 "본체가 호출하는 함수 목록(계약)"만 문서화합니다.

   [디바이스 순서]
     void gb_init_device_order(void);
            // GPU 열거 순서를 모니터링 도구(nvidia-smi/amd-smi)와 일치시킨다.
            // 반드시 첫 GPU 런타임 호출(cudaGetDeviceCount 등) 이전에 부른다.
            // NVIDIA: CUDA_DEVICE_ORDER=PCI_BUS_ID 강제. 기본 FASTEST_FIRST 는
            //         혼합 GPU에서 CUDA index 가 nvidia-smi/NVML 과 어긋나
            //         -g 선택과 모니터링 대상이 서로 다른 카드를 가리킴.
            // AMD: no-op (HIP 는 이미 PCI 순서, gb_mon_open 이 BDF 로 매칭).

   [BLAS]
     int    gb_blas_create (gb_blas_handle_t *h, gb_stream_t stream);
     void   gb_blas_destroy(gb_blas_handle_t h);
     int    gb_gemm(gb_blas_handle_t h, gb_prec_t prec,
                    int M, int N, int K,
                    const void *A, const void *B, void *C);
            // 반환 0 = 성공. A,B,C 는 column-major, lda=M, ldb=K, ldc=M,
            // alpha=1 beta=0, op=N/N 고정 (본체 run_gemm 의미와 일치).
     double gb_gemm_autotune(gb_blas_handle_t h, gb_prec_t prec,
                    int M, int N, int K,
                    const void *A, const void *B, void *C);
            // 실측 기반 최적 solution/algo 를 골라 핸들에 캐시(이후 gb_gemm 이 사용).
            // worker 초기화 시 1회 호출(-A 로 비활성). 반환: 최고 TFLOPS(참고, 미튜닝 0).
            // AMD: rocBLAS get_solutions 전 타입 + hipBLASLt(fp8). NVIDIA: cuBLASLt(fp8)만.

   [모니터링]
     int      gb_mon_init(void);                 // 라이브러리 init (1회)
     void     gb_mon_shutdown(void);
     int      gb_mon_open(int dev_id, gb_mon_t *out);  // 0=성공
     unsigned gb_mon_power_mw   (gb_mon_t);       // 전력 (mW 로 통일)
     unsigned gb_mon_tdp_mw     (gb_mon_t);       // power cap / TDP (mW)
     int      gb_mon_temp2_c    (gb_mon_t, int *edge_c, int *hot_c);
            // edge(표면)/junction(hotspot) 온도(°C)를 동시 조회. 0=성공(edge 유효).
            // *hot_c = -1 이면 해당 GPU 가 junction 센서를 노출하지 않음(N/A).
            // AMD: EDGE / HOTSPOT. NVIDIA: NVML_TEMPERATURE_GPU / (대개 N/A).
     unsigned gb_mon_clock_mhz  (gb_mon_t);       // SM/GFX clock (MHz), 실패 0
     double   gb_mon_util_pct   (gb_mon_t);       // GPU 사용률 (%), 실패 -1
     unsigned gb_mon_throttle   (gb_mon_t);       // GB_THROTTLE_* 비트마스크

   [전력 캡 설정 — -P 옵션 전용, root 권한 필요]
     int gb_mon_set_power_cap_mw(gb_mon_t*, unsigned mw);
            // 전력 캡(TDP)을 mw [mW] 로 설정. 0=성공, -1=실패(권한 없음/미지원).
            // NVIDIA: nvmlDeviceSetPowerManagementLimit, AMD: amdsmi_set_power_cap.
            // 원래 캡은 본체가 gb_mon_tdp_mw() 로 미리 읽어 두었다가 종료 시 복원.
     int gb_mon_power_cap_range_mw(gb_mon_t*, unsigned *min_mw, unsigned *max_mw);
            // 설정 가능한 캡 범위 [mW]. 0=성공. 본체가 요청값을 이 범위로 클램프.

   전력 단위는 NVML(mW)에 맞춰 mW 로 통일합니다. amd_smi(W)는 backend
   에서 ×1000 변환합니다. 본체는 mW 를 받아 /1000.0 으로 W 변환합니다.
   ───────────────────────────────────────────────────────── */

/* 실제 타입·함수 구현 주입 */
#if defined(GB_BACKEND_NVIDIA)
#  include "backend_nvidia.h"
#elif defined(GB_BACKEND_AMD)
#  include "backend_amd.h"
#endif

#endif /* GPU_BACKEND_H */
