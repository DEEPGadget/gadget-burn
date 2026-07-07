# RDNA4 (gfx120x) 지원 및 특성 정리

> **기준일: 2026-07-07.** 이 문서의 실측값·라이브러리 동작·수정 내용은 모두 이 날짜
> 기준입니다. ROCm/hipBLASLt 버전업으로 특히 **fp8 성능·커널 동작은 바뀔 수 있으니**,
> 재검증 시 날짜와 버전을 갱신하세요.

RDNA4(소비자 Radeon RX 9000 시리즈, `gfx1200`/`gfx1201`)에서 gadget_burn 을 운용하며
확인한 특성과, 그에 따른 도구 동작·주의사항을 한곳에 정리합니다. CDNA(Instinct) 와
다른 "소비자 RDNA" 고유 지점이 핵심입니다.

## 검증 환경

| 항목 | 값 |
|---|---|
| GPU | **AMD Radeon RX 9070 XT × 2** (Navi 48, `gfx1201`) |
| ROCm | **7.2.4** (rocBLAS `.so.5.2.70204`, hipBLASLt 1.2.2) |
| CU / 클럭 | 64 CU, boost ~2.46 GHz(부하·정밀도별 2.4~3.2 GHz) |
| VRAM / TDP | 16 GB, 전력캡 317 W(최대 340 W) |
| BDF | GPU0 `0000:43:00.0`, GPU1 `0000:c3:00.0` |

빌드: `make amd amdarch=gfx1201` (fp8 는 hipBLASLt·cuBLASLt 링크가 Makefile 에 포함됨).

---

## 1. GPU 인덱스 순서 (HIP ≠ amd-smi)

**HIP 의 디바이스 열거 순서가 PCI-BDF 오름차순이 아닙니다.** 이 머신에서 HIP 는
`c3`(→device0), `43`(→device1) 순으로 열거하는데, amd-smi/rocm-smi 는 반대(`43`→0,
`c3`→1)입니다. 방치하면 `-g` 선택·라벨이 amd-smi 와 어긋납니다.

- 도구는 `build_device_order()` 로 전 디바이스를 **BDF 오름차순 정렬**해 논리 인덱스를
  만듭니다. 표시·`-g` 는 논리 인덱스(=amd-smi 와 동일), 내부 실행/모니터링은 물리
  device id. `./gadget_burn -l` 이 이제 amd-smi 순서와 일치합니다.
- 모니터링은 `gb_mon_open` 이 PCIe BDF 로 매칭하므로 물리 id 로도 올바른 카드를 읽습니다.

---

## 2. 온도 — edge(표면) vs junction(hotspot)

부하 시 **junction 90°C인데 edge(표면·체감)는 40°C**입니다. 처음엔 amd-smi 오류로
의심했으나 **amd-smi 와 rocm-smi 의 junction 값이 완전히 일치**(센서 정상). 헷갈림의 원인은
"어느 센서를 보느냐"였습니다.

| 센서 | 부하 중(c3) | 의미 |
|---|---|---|
| **edge** | ~40°C | 카드 표면/체감. rocm-smi·nvidia-smi 기본 화면이 보여주는 값 |
| **junction(hotspot)** | ~90°C | 다이 국소 최고점. 부하 시 edge 보다 30~50°C 높은 게 정상 |
| memory | ~50°C | |

- rocm-smi 를 그냥 실행하면 헤드라인 Temp 컬럼이 **Edge(40°C)** 라 "정상"으로 오인하기
  쉽지만, `rocm-smi --showtemp` 의 junction 은 90°C 로 amd-smi 와 같습니다.
- 도구는 **edge/junction 을 함께 표시**(`41/89°C`)합니다. junction 색상은 보수적으로
  **≥90 빨강 / 80~89 노랑 / <80 초록** (throttle 한계 ~110°C 대비 여유). 90°C junction 은
  안전 범위입니다.

### 슬롯/카드 냉각 비대칭 (실측)

같은 부하에서 두 카드의 junction 이 크게 다릅니다(센서가 아니라 실제 냉각 차이):

| | GPU0 (43) | GPU1 (c3) |
|---|---|---|
| junction | ~72°C | ~91°C |
| edge | ~34°C | ~41°C |
| **junction−edge 델타** | ~37°C | **~50°C** |
| GFX 클럭(동일 부하) | ~3116 MHz | ~2516 MHz |

동일 전력(~316 W)인데 c3 가 junction 이 높고 클럭이 낮습니다 → **c3 위치/카드의 다이-쿨러
접촉(TIM/마운트/에어플로)이 열등**. 델타가 큰(≥50°C) 카드는 수락검사에서 주의 대상입니다.
슬롯 vs 카드 확정은 두 카드를 **물리적으로 맞바꿔** 재측정(90°C가 카드를 따라가면 카드
문제, 슬롯에 남으면 에어플로 문제).

---

## 3. 성능 — 라이브러리 기본 커널이 나쁘다 → autotune 필수

RDNA4 최대 발견: **BLAS 라이브러리의 기본 커널 선택이 gfx1201 에서 매우 나쁠 수 있음.**
이게 "SGEMM < 1 TFLOPS" 의 진짜 원인이었습니다(라이브러리 한계가 아님).

- rocBLAS `solution_index=0`(기본)이 sgemm 16384 에서 0.9 TFLOPS. 하지만
  `rocblas_gemm_ex_get_solutions` 로 전체 solution 을 실측하면 **16.5 TFLOPS 짜리가 존재**.
- hipBLASLt fp8 도 heuristic `algo[0]` 이 나빠 8192 에서 16 TFLOPS(최적 68).

도구는 worker 초기화 시 **autotune**(기본 켜짐, `-A` 로 끔)으로 후보 solution/algo 를 실측해
최적을 캐시합니다. 효과(RX 9070 XT):

| `-p` | autotune OFF(`-A`) | autotune ON | 이득 |
|---|---|---|---|
| `sgemm` @16384 | 0.9 TF | **15.2 TF** | 17× |
| `fp8_afp32` @8192 | 16 TF | **64 TF** | 4× |
| `hgemm_mix` / `bf16` | 이미 양호 | 동등 | ~0% |

- autotune 초기화는 느린 후보 때문에 **수십 초~2분** 걸릴 수 있음(early-abort 스크린으로
  제한). 급하면 `-A`.
- `rocblas_gemm_ex_get_solutions` 는 beta API(`ROCBLAS_BETA_FEATURES`).

---

## 4. fp8 (`-p fp8_afp32`)

- **hipBLASLt 필요** (rocBLAS/cublasGemmEx 에 fp8 경로 없음). gfx1201 용 B8B8 커널 존재.
- **누산은 FP32 만 유효.** 실측 검증(A=B=0.5, K=512 → C=128 기대):
  `COMPUTE_32F` → C=128.00 정확, `COMPUTE_16F`(fp16 누산) → **C=0.00 무효**(연산 안 함,
  TFLOPS 는 물리 불가값). bf16 누산은 API 자체가 없음. → 출력 타입으로 옵션을 나눌 이유가
  없어 **단일 `fp8_afp32`**(e4m3 in / FP32 acc / bf16 out, 별칭 `fp8`)로 통합.
- **소비자 RDNA4 fp8 커널 미성숙**: 이론상 fp8 = 2×fp16(Rpeak `tc_ops_fp8`=2048)이나 실측은
  autotune 후에도 ~77 TFLOPS(Peak% ~20%)로, 오히려 **bf16(140)보다 느립니다.** 향후 ROCm
  버전업으로 개선될 여지가 큰 항목(재검증 필요).

---

## 5. 최종 벤치 (2× RX 9070 XT, `-X 8192`, autotune ON, 각 8초)

| `-p` | GPU0 | GPU1 | Peak% | 2-GPU 합계 |
|---|---|---|---|---|
| **bf16** | 140.5 | 139.6 | 85% | **280.1 TF** |
| hgemm (f16/f16) | 131.0 | 129.5 | 81% | 260.5 TF |
| hgemm_mix (f16/f32) | 131.3 | 128.9 | 80% | 260.2 TF |
| fp8_afp32 | 77.2 | 76.6 | 20% | 153.8 TF |
| sgemm (f32) | 15.3 | 15.2 | 34% | 30.5 TF |
| sgemm_tf32 | 15.3 | 15.2 | 34% | 30.5 TF (RDNA는 f32 폴백) |
| dgemm (f64) | 0.81 | 0.80 | 49% | 1.6 TF |

- 처리량 순위: **bf16 > hgemm ≈ hgemm_mix > fp8 > sgemm > dgemm.**
- 전력은 dgemm(~255 W) 제외 전 정밀도 ~316 W(TDP ~100%) 포화.
- dgemm 절대값은 낮지만(소비자 FP64 게이팅) 이론 대비 Peak% 49% 는 정상.

---

## 6. 코어 테이블 (RDNA4 항목, `core_table_amd.h`)

`cores_fp32=CU×128`(dual issue), `cores_fp64=CU×4`(1:32), `tc_ops(FP16)=1024`,
`tc_ops_fp8=2×FP16=2048`, TF32 미지원(0).

| 모델 | cores_fp32 | cores_tensor(CU) | tc_ops | tc_ops_fp8 |
|---|---|---|---|---|
| RX 9070 XT / AI PRO R9700 | 8192 | 64 | 1024 | 2048 |
| RX 9070 | 7168 | 56 | 1024 | 2048 |
| RX 9060 XT | 4096 | 32 | 1024 | 2048 |

> `tc_ops=1024`(FP16)는 AMD 공식 확정값. `tc_ops_fp8=2048`은 "fp8=2×fp16" 일반 규칙 기반
> 이론치이며, 실측(hipBLASLt)이 크게 못 미쳐 Peak% 가 낮게 나오는 것은 커널 미성숙 때문
> (정직 반영). 미등록/미지원(`tc_ops_fp8=0`) HW 에서 `-p fp8_afp32` 는 시작 시 명시 종료.

---

## 7. 운용 권장

- **번들 수락검사 기본**: `-p bf16`(최고 처리량) 또는 기본 `hgemm_mix`. autotune 켠 채로.
- **SGEMM/FP32 검사**: autotune 필수(끄면 0.9 TF 로 무의미). 초기화 시간이 아깝지 않은
  긴 burn 에 적합.
- **온도 판정**: junction 90°C 근처는 정상. 두 카드 **junction 격차·junction−edge 델타**가
  큰 개체를 냉각 불량 신호로 본다.
- **fp8**: 현재 소비자 RDNA4 에선 성능 이점 없음(bf16 이 더 빠름). ROCm 버전업 후 재측정
  가치 있음.

## 관련 문서

- [precision.md](precision.md) — 정밀도별 Rpeak·fp8 Lt 경로·autotune 상세
- [measurement.md](measurement.md) — edge/junction 온도·출력 컬럼
- [gpu-support.md](gpu-support.md) — 코어 DB 전체
