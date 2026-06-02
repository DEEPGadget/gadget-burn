# 정밀도와 Rpeak 계산

gadget_burn은 정밀도(`-p`)에 따라 다른 GEMM을 실행하고, 각 GPU의 코어 수·클럭으로 이론 피크(Rpeak)를 계산해 Peak%를 냅니다. NVIDIA와 AMD는 매트릭스 가속기 구조가 완전히 달라, 같은 `CoreEntry` 구조체의 필드를 벤더별로 다르게 해석합니다.

## 지원 정밀도 (`-p`)

| 옵션 | 연산 | NVIDIA (cublasGemmEx) | AMD (rocblas_gemm_ex) |
|---|---|---|---|
| `sgemm` | FP32 (shader, 매트릭스 미사용) | `CUDA_R_32F` + `COMPUTE_32F` | `f32_r` + `f32_r` |
| `dgemm` | FP64 | `CUDA_R_64F` + `COMPUTE_64F` | `f64_r` + `f64_r` |
| `hgemm` | FP16 in / FP16 acc | `CUDA_R_16F` + `COMPUTE_16F` + TENSOR_OP | `f16_r` + `f16_r` |
| `hgemm_mix` | FP16 in / FP32 acc | `CUDA_R_16F` + `COMPUTE_32F` + TENSOR_OP | `f16_r` + `f32_r` |
| `sgemm_tf32` | FP32 storage + TF32 compute | `COMPUTE_32F_FAST_TF32` + TENSOR_OP | (TF32 미지원 → f32 폴백) |

모든 GEMM은 modern API(`cublasGemmEx` / `rocblas_gemm_ex`)로 통일되어 compute type과 data type이 명시 지정됩니다.

### 기본 정밀도 (`-p` 미지정)

`gpu_backend.h`의 `GB_DEFAULT_PREC` 매크로로 백엔드별 분기:

- **NVIDIA → `sgemm_tf32`** : gpu-burn `-tc`와 동일 의도. A·B·C를 FP32 storage 그대로 두고 Tensor Core가 입력 mantissa를 TF32(19-bit)로 잘라 가속, FP32로 누산. FP32 메모리 부하 + TC compute 부하 + FP32 누산 회로가 동시에 활성화되어 가장 균형 잡힌 burn 워크로드.
- **AMD → `hgemm_mix`** : AMD Matrix Core(WMMA/MFMA)의 FP16 입력/FP32 누산이 **네이티브 고속 경로**라 실측 TFLOPS·TDP가 가장 높음. (`hgemm`인 FP16/FP16-acc는 CDNA에서 MFMA 미지원이라 저속 폴백 — 아래 참고.)

## Rpeak 공식 (벤더 공통)

```
FP32   Rpeak = cores_fp32   × 2      × clock_MHz × 1e-6  [TFLOPS]
FP64   Rpeak = cores_fp64   × 2      × clock_MHz × 1e-6  [TFLOPS]
FP16   Rpeak = cores_tensor × tc_ops × clock_MHz × 1e-6  [TFLOPS]
```

공식은 동일하고, 필드 값의 의미만 벤더별로 다릅니다.

---

## NVIDIA — Tensor Core OPS 체계

`tc_ops`는 Tensor Core 1개가 1 클럭에 처리하는 **FP16 입력 / FP16 누산 dense** FMA 횟수입니다. 하나의 TC는 16×16=256 FMA를 1 MMA로 처리하며, Ada(4th gen)·Blackwell(5th gen) 소비자·Pro RTX 라인업은 공통 **256**, Hopper(H200)는 **1024**, A100은 **512**입니다.

세 가지 matrix 처리량(`hgemm` 기준 대비):

| GPU 등급 | hgemm (FP16/16) | hgemm_mix (FP16/32) | sgemm_tf32 |
|---|---|---|---|
| 서버 DC (B200, Blackwell) | 1× (2048) | 1× (페널티 없음) | 1/2× (1024) |
| 서버 DC (H100/H200, Hopper) | 1× (1024) | 1× (페널티 없음) | 1/2× (512) |
| 서버 DC (A100, Ampere) | 1× (512) | 1× (페널티 없음) | 1/2× (256) |
| Pro/DC (RTX PRO 6000, RTX 6000 Ada, L40S) | 1× (256) | 1× (페널티 없음) | 1/2× (128) |
| 소비자 GeForce (RTX 30/40/50) | 1× (256) | **1/2× (HW 강제 반속, 128)** | **1/4× (이중 페널티, 64)** |

- 소비자 GeForce: HW에서 FP32 누산을 반속 강제 → `tc_ops_mix = tc_ops/2`. TF32는 이중 페널티 → `tc_ops_tf32 = tc_ops/4`.
- Pro/서버급: FP32 누산 페널티 없음 → `tc_ops_mix = tc_ops`, `tc_ops_tf32 = tc_ops/2`.
- NVIDIA는 DC 세대마다 base `tc_ops`를 2배씩 늘려왔습니다: A100(512) → H200(1024) → B200(2048). Pro·소비자는 Blackwell까지 256 유지.

### 공식 스펙 검증

```
RTX 5090     (hgemm     ): 680 TC × 256  × 2407 MHz = 419.0 TFLOPS  ✓
RTX 5090     (hgemm_mix ): 680 TC × 128  × 2407 MHz = 209.5 TFLOPS  ✓ (반속)
RTX 5090     (sgemm_tf32): 680 TC ×  64  × 2407 MHz = 104.8 TFLOPS  ✓ (이중 페널티)
RTX PRO 6000 (hgemm_mix ): 752 TC × 256  × 2617 MHz = 504.0 TFLOPS  ✓ (Pro 풀스피드)
A100         (sgemm_tf32): 432 TC × 256  × 1410 MHz = 156.0 TFLOPS  ✓
H200 NVL     (hgemm     ): 528 TC × 1024 × 1830 MHz = 989.5 TFLOPS  ✓
B200         (hgemm     ): 640 TC × 2048 × 1717 MHz = 2250  TFLOPS  ✓
```

> FP64: Ada/Blackwell 소비자급은 SM당 2개로 FP32의 1/64. Hopper는 SM당 64개 전용 FP64 코어.

---

## AMD — RDNA / CDNA OPS 체계

AMD는 Tensor Core 개념이 없어 필드 의미를 재해석합니다:

- `cores_tensor` = **Compute Unit(CU) 수** (CDNA/RDNA4의 물리 matrix core는 CU×4지만, CU 단위로 환산해 `tc_ops`에 CU당 총 OPS를 넣음)
- `tc_ops` = **CU 1개가 1클럭에 처리하는 matrix FP16 FLOP**
- `cores_fp32` = dual-issue 반영 유효 코어 수 (RDNA3+/CDNA3+는 표기 SP의 2배 = CU×128, 그 외 CU×64)
- `cores_fp64` = FP32 대비 비율로 환산 (세대마다 다름)

### 공통 원칙

- **소비자 RDNA1~3은 Matrix Core가 없음** (WMMA를 vector ALU로 실행). 코어 DB에 `tc_ops=0`으로 등록되어, `hgemm`도 vector Rpeak 기준으로 폴백 표시.
- **RDNA3(7000)부터 dual issue 도입** → 표기 FP32 ALU의 2배로 계산. CU당 표기 64 ALU지만 유효 128, FMA로 FP32 = 256 OPS/CU/clk.
- **AMD Matrix Core는 NVIDIA와 달리 FP32 matrix 연산도 지원** (CDNA). 이 때문에 CDNA에서 `sgemm`이 vector peak를 초과할 수 있어, 코어 DB에 `fp32_matrix_ops` 필드로 matrix 기준 Rpeak를 적용합니다.

### 아키텍처별 matrix OPS / CU / clk

| 아키텍처 | 대표 | FP16/BF16 | TF32 | INT8/FP8 | FP32 matrix | FP64 matrix |
|---|---|---|---|---|---|---|
| RDNA1/2 | RX 5000/6000 | — (matrix 없음) | — | — | — | — |
| RDNA3 | RX 7000 | 512 (WMMA) | 미지원 | 512 | — | — |
| RDNA4 | RX 9000 | 1024 (MC 4개×256) | 미지원 | 2048 | — | — |
| CDNA1 | MI100 | 256·BF16 128 | 미지원 | 256 | 64 | 미지원 |
| CDNA2 | MI200 | 256 | 미지원 | 256 | 64 | 64 |
| CDNA3 | MI300 | 512 | 256 | 1024 | 64 | 64 |
| CDNA4 | MI355 | 1024 | 미지원(BF16 에뮬) | 2048 | 64 | 32 |

> 표는 **물리 matrix core 1개당** OPS. 코어 DB의 `tc_ops`는 CU당 총합(=4 core × core당 OPS)으로 저장합니다. 예: CDNA1 MI100 `tc_ops = 4 × 256 = 1024`.

dual issue로 vector는 표기 코어의 2배: RDNA3+/CDNA3+. FP64 비율은 RDNA1/2=1:16, RDNA3=1:64, RDNA4=1:32.

### 공식 스펙 검증

```
RX 7900 XTX FP16 :  96 CU × 512  × 2498 MHz = 122.8 TFLOPS  ✓ (공식 ~123)
RX 7900 XTX FP32 : 12288  × 2 × 2498 MHz     =  61.4 TFLOPS  ✓
RX 7900 XTX FP64 : 192    × 2 × 2498 MHz      =   0.96 TFLOPS ✓ (1:64)
MI100       FP16 : 120 CU × 1024 × 1502 MHz = 184.6 TFLOPS  ✓
MI100       FP32 : 7680   × 2 × 1502 MHz      =  23.1 TFLOPS ✓ (vector)
MI100       FP64 : 3840   × 2 × 1502 MHz      =  11.5 TFLOPS ✓
MI300X      FP16 : 304 CU × 2048 × 2100 MHz = 1307.4 TFLOPS ✓
```

### CDNA의 특이 사항 (실측 검증)

MI100에서 정밀도별로 다음이 관찰됩니다:

- **`hgemm` (FP16/FP16-acc) 가 느림** : CDNA의 MFMA 명령은 FP16 입력 → **FP32 누산이 네이티브**이고 FP16 누산은 미지원 → rocBLAS가 저속 경로로 폴백 (~1/4 성능). 그래서 AMD 기본값은 `hgemm_mix`이고, `hgemm` 선택 시 초기화에 안내가 출력됩니다.
- **`sgemm` 이 FP32 matrix 로 가속** : AMD Matrix Core는 FP32 matrix를 지원하므로 rocBLAS가 sgemm을 matrix(MI100 ~46 TFLOPS)로 올립니다. 코어 DB의 `fp32_matrix_ops`(CDNA=256)로 sgemm Rpeak를 matrix 기준으로 계산해 Peak%가 올바르게 나옵니다.
- **TF32 미지원** : RDNA3·CDNA1/2/4는 TF32 경로가 없어 `sgemm_tf32`가 f32로 폴백하며, Peak%도 FP32 기준으로 표시됩니다.
