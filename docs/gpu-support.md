# 지원 GPU 목록

내장 코어 DB에 등록된 GPU입니다. **미등록 GPU도 전력·온도·클럭·Util 측정과 TFLOPS 산출은 정상 동작**하며, Peak%(Rpeak 대비)만 표시되지 않습니다.

필드 의미와 OPS 체계는 [precision.md](precision.md)를 참고하세요. NVIDIA는 `cores_tensor`=Tensor Core 수·`tc_ops`=TC당 OPS, AMD는 `cores_tensor`=CU 수·`tc_ops`=CU당 총 matrix OPS로 해석합니다.

---

## NVIDIA (`core_table_nvidia.h`)

| 세대 | GPU | FP32 | FP64 | Tensor | tc_ops | _mix | _tf32 |
|---|---|---|---|---|---|---|---|
| Blackwell DC | B200 | 20,480 | 10,240 | 640 | 2048 | 2048 | 1024 |
| Hopper DC | H200 / H200 NVL | 16,896 | 8,448 | 528 | 1024 | 1024 | 512 |
| Ampere DC | A100 | 6,912 | 3,456 | 432 | 512 | 512 | 256 |
| Volta DC | Tesla V100 | 5,120 | 2,560 | 640 | 128 | 128 | 0 |
| Volta Titan | TITAN V | 5,120 | 2,560 | 640 | 128 | 128 | 0 |
| Blackwell Pro | RTX PRO 6000 Blackwell (전 라인업) | 24,064 | 376 | 752 | 256 | 256 | 128 |
| Ada Pro | RTX 6000 Ada / L40S | 18,176 | 284 | 568 | 256 | 256 | 128 |
| Ampere Pro | RTX A6000 | 10,752 | 168 | 336 | 256 | 256 | 128 |
| Ampere Pro | RTX A5000 | 8,192 | 128 | 256 | 256 | 256 | 128 |
| Ampere Pro | RTX A4000 | 6,144 | 96 | 192 | 256 | 256 | 128 |
| Turing Pro | Quadro RTX 8000 / 6000 | 4,608 | 144 | 576 | 128 | 128 | 0 |
| Turing Pro | Quadro RTX 5000 | 3,072 | 96 | 384 | 128 | 128 | 0 |
| Blackwell GeForce | RTX 5090 | 21,760 | 340 | 680 | 256 | 128 | 64 |
| Blackwell GeForce | RTX 5080 | 10,752 | 168 | 336 | 256 | 128 | 64 |
| Ada GeForce | RTX 4090 | 16,384 | 256 | 512 | 256 | 128 | 64 |
| Ada GeForce | RTX 4080 | 9,728 | 152 | 304 | 256 | 128 | 64 |
| Ampere GeForce | RTX 3090 Ti | 10,752 | 168 | 336 | 256 | 128 | 64 |
| Ampere GeForce | RTX 3090 | 10,496 | 164 | 328 | 256 | 128 | 64 |
| Ampere GeForce | RTX 3080 Ti | 10,240 | 160 | 320 | 256 | 128 | 64 |
| Ampere GeForce | RTX 3080 | 8,704 | 136 | 272 | 256 | 128 | 64 |
| Ampere GeForce | RTX 3070 Ti | 6,144 | 96 | 192 | 256 | 128 | 64 |
| Ampere GeForce | RTX 3070 | 5,888 | 92 | 184 | 256 | 128 | 64 |
| Ampere GeForce | RTX 3060 Ti | 4,864 | 76 | 152 | 256 | 128 | 64 |
| Ampere GeForce | RTX 3060 | 3,584 | 56 | 112 | 256 | 128 | 64 |
| Ampere GeForce | RTX 3050 | 2,560 | 40 | 80 | 256 | 128 | 64 |
| Turing GeForce | RTX 2080 Ti | 4,352 | 136 | 544 | 128 | 128 | 0 |
| Turing GeForce | Titan RTX | 4,608 | 144 | 576 | 128 | 128 | 0 |
| Turing GeForce | RTX 2080 | 2,944 | 92 | 368 | 128 | 128 | 0 |

- `tc_ops`: FP16 in / FP16 acc dense (`-p hgemm`) 시 TC당 클럭당 ops
- `tc_ops_mix`: FP16 in / FP32 acc dense (`-p hgemm_mix`)
- `tc_ops_tf32`: TF32 dense (`-p sgemm_tf32`)
- Pro·서버급: `_mix = tc_ops`, `_tf32 = tc_ops/2` (단일 페널티)
- 소비자 GeForce: `_mix = tc_ops/2`, `_tf32 = tc_ops/4` (이중 페널티)

---

## AMD (`core_table_amd.h`)

AMD는 `cores_tensor`=CU 수, `tc_ops`=CU당 총 matrix OPS(= 4 matrix core × core당 OPS)입니다. `cores_fp32`는 dual-issue 반영 유효 코어 수.

### CDNA (Instinct MI) — Matrix Core가 FP32/FP64도 가속

| 세대 | GPU | gfx | CU | FP32 | FP64 | tc_ops(FP16) | tf32 |
|---|---|---|---|---|---|---|---|
| CDNA1 | MI100 | gfx908 | 120 | 7,680 | 3,840 | 1024 | 0 |
| CDNA2 | MI250X (per-GCD) | gfx90a | 110 | 7,040 | 7,040 | 1024 | 0 |
| CDNA2 | MI250 / MI210 (per-GCD) | gfx90a | 104 | 6,656 | 6,656 | 1024 | 0 |
| CDNA3 | MI300X / MI325X | gfx942 | 304 | 38,912 | 19,456 | 2048 | 1024 |
| CDNA3 | MI300A | gfx942 | 228 | 29,184 | 14,592 | 2048 | 1024 |
| CDNA4 | MI350X / MI355X | gfx950 | 256 | 32,768 | 16,384 | 4096 | 0 |

> CDNA는 `fp32_matrix_ops=256`이 설정되어 sgemm Peak%를 FP32 matrix 기준으로 계산합니다.
> MI250/MI250X는 2-die(GCD)라 amd_smi가 GCD를 별도 device로 노출 → per-GCD CU로 등록.

### RDNA3 (RX 7000 / PRO W7000) — WMMA, Matrix Core 없음

| GPU | gfx | CU | FP32 | FP64 | tc_ops(FP16) |
|---|---|---|---|---|---|
| RX 7900 XTX / PRO W7900 | gfx1100 | 96 | 12,288 | 192 | 512 |
| RX 7900 XT | gfx1100 | 84 | 10,752 | 168 | 512 |
| RX 7900 GRE | gfx1100 | 80 | 10,240 | 160 | 512 |
| PRO W7800 | gfx1100 | 70 | 8,960 | 140 | 512 |
| RX 7800 XT | gfx1101 | 60 | 7,680 | 120 | 512 |
| RX 7700 XT | gfx1101 | 54 | 6,912 | 108 | 512 |
| PRO W7700 | gfx1101 | 48 | 6,144 | 96 | 512 |
| RX 7600 / 7600 XT | gfx1102 | 32 | 4,096 | 64 | 512 |

### RDNA4 (RX 9000 / AI PRO) — Matrix Core (CU당 4개)

| GPU | gfx | CU | FP32 | FP64 | tc_ops(FP16) |
|---|---|---|---|---|---|
| RX 9070 XT / AI PRO R9700 | gfx1201 | 64 | 8,192 | 256 | 1024 |
| RX 9070 | gfx1201 | 56 | 7,168 | 224 | 1024 |
| RX 9060 XT | gfx1200 | 32 | 4,096 | 128 | 1024 |

### RDNA1/2 (RX 5000 / 6000) — Matrix 가속 없음

`tc_ops=0`이라 hgemm은 vector Rpeak 기준으로 폴백 표시됩니다.

| GPU | gfx | CU | FP32 | FP64 |
|---|---|---|---|---|
| RX 6950 XT / 6900 XT / PRO W6900X | gfx1030 | 80 | 5,120 | 320 |
| RX 6800 XT | gfx1030 | 72 | 4,608 | 288 |
| RX 6800 / PRO W6800 | gfx1030 | 60 | 3,840 | 240 |
| RX 6700 XT | gfx1031 | 40 | 2,560 | 160 |
| RX 6600 XT | gfx1032 | 32 | 2,048 | 128 |
| RX 6600 | gfx1032 | 28 | 1,792 | 112 |
| RX 5700 XT | gfx1010 | 40 | 2,560 | 160 |
| RX 5700 / 5600 XT / PRO W5700 | gfx1010 | 36 | 2,304 | 144 |
| RX 5500 XT | gfx1012 | 22 | 1,408 | 88 |

> RDNA FP64 비율: RDNA1/2 = 1:16, RDNA3 = 1:64, RDNA4 = 1:32.

---

## 미등록 GPU 추가

`core_table_nvidia.h` / `core_table_amd.h`의 매크로에 한 줄 추가하면 됩니다. 필드 순서:

```c
{ "이름 substring", cores_fp32, cores_fp64, cores_tensor, tc_ops, tc_ops_mix, tc_ops_tf32 [, fp32_matrix_ops] }
```

이름은 `cudaDeviceProp.name` / `hipDeviceProp_t.name` substring으로 매칭하며, 구체적인(긴) 이름을 먼저 배치해야 합니다(예: "Tesla V100"을 "V100"보다 먼저). AMD CDNA만 8번째 `fp32_matrix_ops`를 채우고, 나머지는 생략(C 후행 0초기화)합니다.
