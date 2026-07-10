# CLAUDE.md

이 파일은 이 저장소에서 작업하는 Claude Code(및 다른 AI 에이전트)를 위한 안내서입니다.

## 프로젝트 개요

**gadget_burn** — Multi-GPU GEMM 기반 GPU burn-in / 성능·전력 측정 도구.
**단일 소스를 NVIDIA(cuBLAS/NVML)와 AMD(rocBLAS/amd_smi) 두 벤더로 빌드**하는 것이
핵심 특징입니다. 실제 코어 수와 실측 클럭으로 계산한 **동적 Peak%**로 각 GPU가
설계 성능을 내는지 보여줍니다. ([gpu-burn](https://github.com/wilicc/gpu-burn) 영감)

용도: GPU 서버 납품 수락 검사, 안정성 burn-in, 냉각 검증, 성능 회귀 탐지.

## 빌드 & 실행

```bash
# 빌드
make                          # NVIDIA (기본, CUDA/cuBLAS/NVML, native SASS)
make amd                      # AMD (HIP/rocBLAS/amd_smi, 빌드 머신 GPU 자동 감지)

# NVIDIA 빌드 모드 (mode=)
make mode=ptx                 # PTX only — 이식성 최고, 첫 실행 JIT
make mode=fatbin              # sm_60~sm_90a SASS + PTX 전부 포함

# AMD 빌드 아키텍처 (amdarch=)
make amd amdarch=gfx908       # CDNA1(MI100) / gfx90a MI200 / gfx942 MI300 / gfx1100 RDNA3 ...
make amd amdarch="gfx908 gfx1100"   # fat binary (다중 arch)

make clean
make info                     # 바이너리에 포함된 GPU 코드 목록(NVIDIA cuobjdump)

# 실행
./gadget_burn                 # 전체 GPU, 100% VRAM, 3600초, 백엔드별 기본 정밀도
./gadget_burn -l              # GPU 목록 출력 후 종료
./gadget_burn -h              # 도움말
```

주요 옵션: `-A`(GEMM autotune 끄기; 기본 켜짐) `-t`(초) `-i`(동시 스트림 수) `-p`(정밀도) `-m`(VRAM% 또는 행렬크기)
`-g`(GPU ID 목록) `-X`(행렬 크기 M override) `-I`(memset|rand 초기화)
`-P`(전력 캡/TDP 설정 [W], root 필요·종료 시 복원)
`-o`(CSV 기록 [경로], 1Hz long/tidy + `#` 메타 헤더, pandas/gnuplot 호환)
`-S`(TDP 스윕 plan `W:T,...`, 여러 조건 연속 실행; 메모리·데이터타입 고정, root 필요).

## 아키텍처

본체(`gadget_burn.cu`)는 **벤더 중립**입니다. 벤더 SDK 헤더를 직접 포함하지 않고,
빌드 시 `-DGB_BACKEND_NVIDIA` 또는 `-DGB_BACKEND_AMD`로 선택된 backend 헤더가
타입·함수·매크로를 주입합니다. 본체는 추상화 인터페이스(`gb_*`)와 `cuda*` 호출만 사용.

| 파일 | 역할 |
|---|---|
| `gadget_burn.cu` | **본체(벤더 중립)** — main, 워커/모니터/벤치 스레드, Rpeak·TFLOPS 계산, 출력 |
| `gpu_backend.h` | **추상화 계약** — 백엔드 선택, `gb_prec_t`, throttle 비트, 함수 시그니처 문서 |
| `backend_nvidia.h` | CUDA/cuBLAS/NVML 구현 |
| `backend_amd.h` | HIP/rocBLAS/amd_smi 구현 (`cuda*`→`hip*` 매크로 별칭) |
| `core_table_nvidia.h` / `core_table_amd.h` | GPU 코어 DB — 동적 Rpeak 계산용 사양 테이블 |
| `Makefile` | `backend=` / `mode=` / `amdarch=` 분기 |
| `docs/` | architecture·build·precision·measurement·gpu-support |

핵심 메커니즘:
- **추상화 레이어**: GEMM은 `gb_gemm()` 단일 함수, 모니터링은 `gb_mon_*`로 통일.
  전력 단위는 mW로 통일(AMD는 W→mW ×1000), throttle은 `GB_THROTTLE_*` 공용 비트.
- **동시성 모델**: GPU당 (모니터 스레드 100ms 폴링 + 벤치 스레드). 벤치는 `intensity`개
  스트림에 GEMM 발행→동기화→CUDA event로 시간 측정. 모니터를 150ms 먼저 띄워 race 방지.
- **동적 Rpeak**: 고정 스펙이 아니라 **실측 클럭 × 코어수 × ops/cycle**. `CORE_TABLE`
  substring 매칭. 정밀도별 공식 분기(`calc_dynamic_rpeak`). 100% 초과 가능(의도된 동작).
- **메모리 압박**: C 행렬 ring buffer로 가용 VRAM을 채우고 xorshift PRNG로 random
  초기화(bit toggle 확보 → TDP↑). gpu-burn 패턴이 기본 동작.

## 코드 작업 시 핵심 규칙

- **본체의 벤더 중립성을 유지**한다. `gadget_burn.cu`에서 `cuda*`/`gb_*` 외의 벤더 SDK
  심볼(cublas*, hip*, rocblas*, nvml* 등)을 직접 쓰지 않는다. 새 벤더 동작은 backend 헤더로.
- **새 정밀도 추가**: `gb_prec_t`(gpu_backend.h, **enum 은 끝에 추가** — `prec_str[]` 가
  서수 인덱싱) → 두 backend의 `gb_gemm()` → 본체 `calc_dynamic_rpeak`/옵션 파싱/`prec_str`/
  `prec_label`/`init_buffer_random`/`prec_in_bpe`·`prec_out_bpe`(입출력 요소 크기가 다르면)
  전부 갱신. **fp8 은 rocblas_gemm_ex/cublasGemmEx 에 없어 hipBLASLt/cuBLASLt(Lt) 경로**를
  타며, `gb_blas_handle_t` 가 Lt 핸들·workspace·실행계획을 함께 보관하는 struct 포인터다
  (fp8 입력 1B / mix 출력 2B 처럼 입출력 bpe 가 다를 수 있으므로 헬퍼로 분리).
- **GPU 추가**: 해당 `core_table_*.h`에만 엔트리 추가. **구체적(긴) 이름을 먼저 배치**
  (substring 선매칭). NVIDIA/AMD 필드 의미가 다르므로 각 헤더 상단 주석을 따른다
  (특히 AMD는 `cores_tensor`=CU 수, `tc_ops`=CU당 matrix OPS, `fp32_matrix_ops` 재해석).

## 함정 & 주의사항

- **AMD arch 불일치 → 런타임 SIGSEGV**. AMD는 PTX 같은 가상 ISA가 없어 잘못된 arch로
  빌드해도 빌드는 성공하고 커널 launch 때 죽는다. **공유 FS(NFS)에서 특히 위험** — 실행
  머신의 arch로 다시 빌드하거나 fat binary 사용.
- **AMD: HIP device ↔ amd_smi 핸들은 PCIe BDF로 매칭**한다(`gb_mon_open`). enumeration
  순서가 다를 수 있어, 안 맞추면 모니터링이 엉뚱한 GPU를 읽어 측정값이 전부 어긋난다.
- **NVIDIA: NVML util 교차오염** — `nvmlDeviceGetUtilizationRates`는 멀티-GPU에서 같은 값을
  반환하는 버그가 있어 GPU별 독립 링버퍼 `GetSamples`로 우회.
- **GPU 인덱스 정렬**: 런타임 기본 열거 순서는 nvidia-smi/amd-smi(PCI-BDF 순서)와 어긋날 수
  있어(NVIDIA=FASTEST_FIRST, HIP=BDF 오름차순 미보장 — RDNA4 2장에서 역순 관측) `-g` 선택·
  라벨이 다른 카드를 가리킨다. 두 단계로 정렬한다: ① `gb_init_device_order()`가 NVIDIA에서
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`를 강제(첫 CUDA 호출 전 필수, AMD는 no-op), ②
  `build_device_order()`가 전 디바이스를 BDF로 정렬해 **논리 인덱스↔물리 device id** 매핑을
  만든다. 본체는 표시·`-g`에 `logical_id`(BDF 순), cudaSetDevice/모니터링에 `device_id`(물리)를
  사용. 모니터링은 BDF 매칭이라 물리 id로 정상. (NVIDIA는 이미 BDF 순 → 매핑이 항등)
- **백엔드별 기본 정밀도가 다름**: NVIDIA=`sgemm_tf32`(gpu-burn -tc 호환),
  AMD=`hgemm_mix`(Matrix Core 네이티브 고속 경로). TF32 미지원 HW는 FP32로 폴백.
- **AMD throttle 비트는 보수적 매핑**(현재 0이 아니면 POWER_BRAKE). gfx별 정밀화는 TODO.
- **autotune(기본 켜짐, -A 로 끔)**: BLAS 기본 커널 선택이 gfx1201 에서 매우 나쁠 수 있어
  (sgemm 16384: 기본 0.9 → 튜닝 15 TFLOPS, 17×), worker 초기화 시 후보 solution/algo 를
  실측해 최적을 캐시(`gb_gemm_autotune`, 핸들의 `rb_sol`/`lt_heur` 에 저장). rocBLAS 는
  `rocblas_gemm_ex_get_solutions`(beta API, `ROCBLAS_BETA_FEATURES`), Lt 는 heuristic 리스트.
  느린 후보로 초기화가 수십 초~2분 걸릴 수 있음(early-abort 스크린으로 제한). NVIDIA 클래식
  경로(cublasGemmEx)는 no-op(cuBLAS 기본이 이미 양호).
- **fp8(-p fp8_afp32, 별칭 fp8)은 Lt 경로**(hipBLASLt/cuBLASLt). 누산 기준 단일 옵션 —
  출력 타입은 최종 downcast 라 의미 없어 구분 안 함(항상 bf16 out). 주의: ① gfx1201 은
  `COMPUTE_32F`(fp32 누산)만 유효, `COMPUTE_16F`(fp16 누산)는 **무효 커널**(연산 안 함, C=0,
  물리 불가 TFLOPS)이고 bf16 누산은 API 없음 → FP32 누산만 제공. ② 소비자 RDNA4 hipBLASLt
  fp8 커널이 미성숙해 실측이 이론(2×fp16)에 못 미침(autotune 으로 상당 회복). ③ cuBLASLt fp8
  은 TN(opA=T) 강제·NVIDIA 실기 미검증. 미지원 HW(`tc_ops_fp8=0`)는 시작 시 명시 종료.

## 검증 환경

- **NVIDIA**: server2(CUDA)에서 검증.
- **AMD**: 로컬 hipcc + docker로 빌드/검증. MI100 실벤치는 server1.
  (CDNA는 FP32 matrix 가속으로 sgemm Peak%가 100%를 넘을 수 있음 — 정상)

## 참고 문서 (`docs/`)

| 문서 | 내용 |
|---|---|
| `docs/architecture.md` | 멀티벤더 백엔드 구조, 추상화 레이어, 동시성 모델 |
| `docs/build.md` | 백엔드/모드 선택, AMD arch 이식성(fatbin/generic), 공유 FS 주의 |
| `docs/precision.md` | 연산 타입, NVIDIA Tensor Core / AMD RDNA·CDNA OPS 체계, 검증 |
| `docs/measurement.md` | TFLOPS·Peak%·전력·Util·클럭·온도·throttle 측정 상세, 출력 컬럼 |
| `docs/gpu-support.md` | 내장 코어 DB 전체, 미등록 GPU 추가 방법 |
| `docs/RDNA4.md` | 소비자 RDNA4(gfx120x) 특성·주의(인덱스/온도/autotune/fp8), 최종 벤치 |
| `docs/amd-benchmark.md` | AMD 3종(MI100/RX 7900 XTX/RX 9070 XT) 정밀도×행렬크기 스윕 결과·분석 |

## 작업 규약

- **코드 생성·수정 전 반드시 사용자에게 구현 방식을 브리핑하고, 의견 조율 후 최종
  컨펌을 받은 뒤** 코드를 작성한다.
- 모든 주석·문서·출력 문자열은 기존 코드와 동일하게 **한국어**로 작성한다.
