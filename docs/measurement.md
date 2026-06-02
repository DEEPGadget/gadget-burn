# 측정 방법

모든 측정값은 GPU당 모니터링 스레드가 100ms마다 폴링합니다. 실시간 표시는 **최근 10초 슬라이딩 윈도우 평균**, 최종 결과는 **전체 측정 구간 누적 평균**을 사용합니다. NVIDIA는 NVML, AMD는 amd_smi로 같은 값을 읽으며(전력 단위는 mW로 통일), 본체는 벤더를 구분하지 않습니다.

## TFLOPS

GEMM 연산의 이론 FLOPs를 실제 소요 시간으로 나눠 계산합니다.

```
TFLOPS = (2 × M × N × K × intensity × 반복 횟수) / 경과 시간
```

`intensity`개 스트림이 동시 실행되므로 모든 스트림의 연산량을 합산합니다. 각 iteration의 wall time은 **가장 오래 걸린 스트림의 이벤트 경과 시간**(`cudaEventElapsedTime` / `hipEventElapsedTime`)을 사용합니다.

| 구분 | 기준 | 목적 |
|---|---|---|
| 실시간 TFLOPS | 슬라이딩 윈도우 (최근 10초) | 급격한 변화를 빠르게 반영 |
| 최종 TFLOPS | 전체 측정 구간 누적 평균 | 장기 안정성 대표값 |

슬라이딩 윈도우는 원형 버퍼(100 슬롯 × 100ms = 10초)로 구현되며, 가장 오래된 샘플이 새 샘플로 교체되면서 항상 최근 10초 구간 평균을 유지합니다.

## 동적 Rpeak와 Peak%

Peak%의 기준이 되는 이론 피크 성능은 하드코딩된 고정값이 아니라 **측정 시점의 실제 SM/GFX 클럭**으로 동적 계산합니다.

```
FP32   Rpeak = cores_fp32   × 2      × clock_MHz × 1e-6  [TFLOPS]
FP64   Rpeak = cores_fp64   × 2      × clock_MHz × 1e-6  [TFLOPS]
FP16   Rpeak = cores_tensor × tc_ops × clock_MHz × 1e-6  [TFLOPS]
```

고정값 방식에서는 Boost Clock이 스펙을 초과하면 Peak%가 100%를 넘어 비정상처럼 보이는 문제가 있었습니다. 동적 방식은 클럭 변동이 즉시 반영되어 "현재 클럭 대비 얼마나 효율적으로 동작하는가"를 정확히 나타냅니다(100% 초과도 가능 — 의도된 동작).

실시간 Peak%는 슬라이딩 윈도우 평균 클럭, 최종 결과 Peak%는 전체 구간 평균 클럭을 기준으로 합니다.

| 색상 | 달성률 | 의미 |
|---|---|---|
| 🟢 초록 | ≥ 70% | 정상 범위 |
| 🟡 노랑 | 60~69% | 확인 권장 |
| 🔴 빨강 | < 60% | 이상 가능성 |

정밀도별 `tc_ops` 의미와 NVIDIA/AMD의 코어 수 환산은 [precision.md](precision.md)를 참고하세요.

## 전력 (W)

100ms마다 측정하며 TDP%는 power management limit 대비 비율입니다.

| 벤더 | 전력 API | TDP API |
|---|---|---|
| NVIDIA | `nvmlDeviceGetPowerUsage()` (mW) | `nvmlDeviceGetPowerManagementLimit()` |
| AMD | `amdsmi_get_power_info().average_socket_power` (W) | `amdsmi_get_power_cap_info()` |

본체는 mW 단위로 받아 W로 변환합니다. 실시간은 최근 10초 슬라이딩 윈도우 평균, 최종은 전체 구간 누적 평균입니다.

## GPU 사용률 (Util%)

| 벤더 | API | 비고 |
|---|---|---|
| NVIDIA | `nvmlDeviceGetSamples(NVML_GPU_UTILIZATION_SAMPLES)` | GPU별 독립 링버퍼로 교차오염 우회 |
| AMD | `amdsmi_get_gpu_activity().gfx_activity` | 핸들별 즉시 현재값, 교차오염 없음 |

NVIDIA의 `nvmlDeviceGetUtilizationRates()`는 드라이버 내부 전역 공유 버퍼를 읽기 때문에, 멀티-GPU 환경에서 서로 다른 GPU임에도 동일한 값이 반환되는 문제가 있습니다. gadget_burn은 GPU handle별 독립 링버퍼인 `nvmlDeviceGetSamples()`를 사용해 이를 우회합니다. AMD amd_smi는 핸들별로 현재값을 반환하므로 이 우회가 불필요합니다.

```
각 GPU 모니터링 스레드
  └─ (NVIDIA) GetSamples(handle, last_timestamp) → 새 샘플만 추출 → 타임스탬프 갱신
     (AMD)    get_gpu_activity(handle)           → 현재값 즉시 반환
```

## SM / GFX Clock (MHz)

`nvmlDeviceGetClockInfo(NVML_CLOCK_SM)` / `amdsmi_get_clock_info(AMDSMI_CLK_TYPE_GFX)`로 현재 동작 주파수를 측정합니다. 이 클럭은 Peak% 계산에도 직접 사용됩니다. 실시간은 최근 10초 평균, 최종은 전체 구간 평균과 종료 시점 순간값을 함께 표시합니다.

## 온도 (°C)

`nvmlDeviceGetTemperature(NVML_TEMPERATURE_GPU)` / `amdsmi_get_temp_metric(HOTSPOT, CURRENT)`로 측정합니다. 온도는 항상 **현재값(실시간)**을 표시하며, 최종 결과에는 전체 구간 평균과 종료 시점 값을 함께 표시합니다.

| 온도 | 색상 | 의미 |
|---|---|---|
| < 65°C | 🟢 초록 | 정상 |
| 65~74°C | 🟠 주황 | 주의 |
| ≥ 75°C | 🔴 빨강 | 고온 경고 |

## Throttle

벤더 native 스로틀 비트를 공용 `GB_THROTTLE_*`(THERMAL / POWER_BRAKE)로 매핑해 표시합니다. SW Power Cap 같은 burn 중 의도된 상태는 표시하지 않습니다.

| 벤더 | 소스 |
|---|---|
| NVIDIA | `nvmlDeviceGetCurrentClocksThrottleReasons()` |
| AMD | `amdsmi_get_gpu_metrics_info().throttle_status` |

## 메모리 대역폭 추정 (TB/s)

실제 메모리 트래픽을 직접 측정하지 않고, GEMM 연산의 이론적 데이터 이동량으로 추정합니다.

```
추정 BW = bpe × 3 × M² × intensity × 반복 횟수 / 경과 시간
```

실제 L2 캐시 재사용 등을 반영하지 않으므로 참고용 상한 추정치입니다.

## 샘플링 구조 요약

```
100ms 폴링 (모니터링 스레드, GPU 당 1개)
  ├─ 전력      gb_mon_power_mw()
  ├─ Util%     gb_mon_util_pct()    ← GPU별 독립 (NVIDIA 링버퍼 / AMD 직접)
  ├─ Clock     gb_mon_clock_mhz()   ← Peak% 계산에도 사용
  ├─ 온도      gb_mon_temp_c()
  └─ Throttle  gb_mon_throttle()

GEMM 반복 (벤치 스레드, GPU 당 1개)
  └─ 소요 시간  cudaEventElapsedTime / hipEventElapsedTime  ← iteration별 측정

실시간 표시: 슬라이딩 윈도우 (원형 버퍼, 100슬롯 × 100ms = 최근 10초)
최종 결과:  전체 구간 누적 평균
```

## 출력 컬럼

### 실시간 출력

| 컬럼 | 측정 방법 | 표시 기준 |
|---|---|---|
| TFLOPS | `2×M×N×K×intensity×iters / time` | 슬라이딩 윈도우 (최근 10초) |
| Peak%(Rpeak) | `TFLOPS / Rpeak × 100` | 슬라이딩 윈도우 클럭 기준 |
| 전력(W) | `gb_mon_power_mw` | 슬라이딩 윈도우 (최근 10초) |
| TDP% | `전력 / TDP × 100` | 동일 |
| Util% | `gb_mon_util_pct` (GPU별 독립) | 슬라이딩 윈도우 (최근 10초) |
| Clock | `gb_mon_clock_mhz` | 슬라이딩 윈도우 (최근 10초) |
| 온도 | `gb_mon_temp_c` | 현재값 (실시간) |
| Throt | `gb_mon_throttle` | 현재값 |
| VRAM | 초기화 시 할당 합산 | 고정값 |

### 최종 결과 예시

```
  GPU  0  NVIDIA GeForce RTX 5090
    ├ 행렬 크기    : 16384 x 16384,  스트림 1
    ├ 총 반복 횟수 : 1284 회
    ├ 유효 시간    : 3598.234 초
    ├ VRAM 사용    : 30906 MiB
    ├ 성능         : 86.4321 TFLOPS   82.4%  (104.8 TFLOPS @ 2710 MHz 기준)
    ├ 평균 전력    : 595.3 W  (TDP 대비 99.2%)
    ├ GPU 사용률   : 99.6%
    ├ SM Clock     : 2710 MHz 평균  (종료 시 2715 MHz)
    ├ 평균 온도    : 67.3°C  (종료 시 68°C)
    ├ Throttle    : 열 0.0초 (0.0%), 전력 0.0초 (0.0%)
    ├ 메모리 BW    : 1.247 TB/s (추정)
    └ 전력 효율    : 0.1451 TFLOPS/W
```
