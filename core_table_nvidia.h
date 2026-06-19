/**
 * core_table_nvidia.h  —  NVIDIA GPU 코어 DB (동적 Rpeak 계산용)
 * ─────────────────────────────────────────────────────────────
 * 본체의 CoreEntry 구조체로 NVIDIA GPU 의 코어/Tensor Core 사양을
 * 정의합니다. 동적 Rpeak 공식:
 *   FP32  = cores_fp32   × 2      × clock_MHz × 1e-6  [TFLOPS]
 *   FP64  = cores_fp64   × 2      × clock_MHz × 1e-6  [TFLOPS]
 *   FP16  = cores_tensor × tc_ops × clock_MHz × 1e-6  [TFLOPS]
 *
 * 필드 의미:
 *   cores_fp32    : CUDA FP32 코어 수
 *   cores_fp64    : CUDA FP64 코어 수 (0 = 미지원)
 *   cores_tensor  : Tensor Core 수
 *   tc_ops        : TC 1개당 클럭당 ops, FP16 in / FP16 acc, dense
 *   tc_ops_mix    : FP16 in / FP32 acc — 소비자급 tc_ops/2(HW 반속),
 *                   Pro/서버급 tc_ops(페널티 없음)
 *   tc_ops_tf32   : TF32 — 소비자급 tc_ops/4(이중 페널티),
 *                   Pro/서버급 tc_ops/2. Turing/Volta(미지원)=0
 *   fp32_matrix_ops : 0 (NVIDIA Tensor Core 는 FP32 matrix 미지원).
 *                     후행 멤버이므로 C 후행 0초기화로 생략 가능.
 *
 * 매칭은 cudaDeviceProp.name substring (대소문자 무시).
 * 구체적인(긴) 이름을 먼저 배치해야 substring 매칭이 올바르게 동작.
 *
 * 출처: NVIDIA 공식 아키텍처 화이트페이퍼 / datasheet.
 * 고정값 대신 실측 Boost Clock 에 비례하므로 클럭 변동이 즉시
 * Peak% 에 반영됩니다 (100% 초과도 가능 → 의도된 동작).
 * ───────────────────────────────────────────────────────────── */

#ifndef CORE_TABLE_NVIDIA_H
#define CORE_TABLE_NVIDIA_H

/* CORE_TABLE 배열 안에서 펼쳐지는 NVIDIA 엔트리.
   필드: { substr, cores_fp32, cores_fp64, cores_tensor,
           tc_ops, tc_ops_mix, tc_ops_tf32 [, fp32_matrix_ops=0] } */

#define GB_NVIDIA_CORE_ENTRIES                                                  \
    /* Blackwell RTX PRO — GB202 full (188 SM)                                 \
       FP32=128/SM×188=24064, FP64=2/SM=376                                    \
       whitepaper 명시대로 FP32 누산 페널티 없음 → tc_ops_mix = tc_ops = 256   \
       TF32 dense = tc_ops/2 = 128 (Pro 풀스피드) */                           \
    { "RTX PRO 6000 Blackwell Server Edition", 24064, 376, 752,  256,  256, 128 }, \
    { "RTX PRO 6000 Blackwell Max-Q",          24064, 376, 752,  256,  256, 128 }, \
    { "RTX PRO 6000 Blackwell",                24064, 376, 752,  256,  256, 128 }, \
    /* Blackwell GeForce — GB202 (170 SM), GB203 (84 SM)                       \
       FP64=2/SM (소비자급). HW 강제 반속 → tc_ops_mix = tc_ops/2 = 128        \
       TF32 dense = tc_ops/4 = 64 (이중 페널티) */                            \
    { "GeForce RTX 5090",                      21760, 340, 680,  256,  128,  64 }, \
    { "GeForce RTX 5080",                      10752, 168, 336,  256,  128,  64 }, \
    /* Ada GeForce — AD102 (128 SM), AD103 (76 SM)                             \
       FP64=2/SM (소비자급). HW 강제 반속, TF32 이중 페널티 */                 \
    { "GeForce RTX 4090",                      16384, 256, 512,  256,  128,  64 }, \
    { "GeForce RTX 4080",                       9728, 152, 304,  256,  128,  64 }, \
    /* Ada Professional / Data Center — AD102 full (142 SM)                    \
       whitepaper 명시대로 풀스피드, TF32 = tc_ops/2 = 128 */                  \
    { "RTX 6000 Ada",                          18176, 284, 568,  256,  256, 128 }, \
    { "L40S",                                  18176, 284, 568,  256,  256, 128 }, \
    /* Hopper — GH100 (132 SM for H200 NVL)                                    \
       FP64=64/SM=8448 (전용 코어). FP16 dense = 989 TFLOPS @ 1830 MHz         \
       TF32 dense = 494 TFLOPS = tc_ops/2 = 512 */                            \
    { "H200 NVL",                              16896, 8448, 528, 1024, 1024,  512 }, \
    { "H200",                                  16896, 8448, 528, 1024, 1024,  512 }, \
    /* Blackwell DC — B200 (160 SM × 2-die fused, FP32=128/SM=20480)           \
       FP64=64/SM=10240. FP16 dense = 2250 TFLOPS @ ~1717 MHz                  \
       NVIDIA가 세대마다 tc_ops 2배: A100=512 → H200=1024 → B200=2048          \
       TF32 dense = 1125 TFLOPS = tc_ops/2 = 1024 */                          \
    { "B200",                                  20480, 10240, 640, 2048, 2048, 1024 }, \
    /* Ampere DC — GA100, FP32 누산 페널티 없음                                \
       FP16 dense = 312 TFLOPS, TF32 dense = 156 TFLOPS @ 1410 MHz → 256 */    \
    { "A100",                                   6912, 3456, 432,  512,  512,  256 }, \
    /* Ampere Pro — RTX A-series, GA102/GA104. Pro 풀스피드 (FP32 acc 페널티 없음). \
       FP64=2/SM (1/64 of FP32, 소비자급 비율). ops: 256/256/128 */            \
    { "RTX A6000",                             10752,  168, 336,  256,  256,  128 }, \
    { "RTX A5000",                              8192,  128, 256,  256,  256,  128 }, \
    { "RTX A4000",                              6144,   96, 192,  256,  256,  128 }, \
    /* Ampere GeForce — GA102 (84 SM), GA104 (48 SM), GA106 (28 SM)            \
       FP64=2/SM (소비자급). 소비자 페널티 적용: ops 256/128/64                \
       Ti 변형은 더 구체적이므로 일반 변형보다 먼저 배치 (substring matching) */ \
    { "GeForce RTX 3090 Ti",                   10752,  168, 336,  256,  128,   64 }, \
    { "GeForce RTX 3090",                      10496,  164, 328,  256,  128,   64 }, \
    { "GeForce RTX 3080 Ti",                   10240,  160, 320,  256,  128,   64 }, \
    { "GeForce RTX 3080",                       8704,  136, 272,  256,  128,   64 }, \
    { "GeForce RTX 3070 Ti",                    6144,   96, 192,  256,  128,   64 }, \
    { "GeForce RTX 3070",                       5888,   92, 184,  256,  128,   64 }, \
    { "GeForce RTX 3060 Ti",                    4864,   76, 152,  256,  128,   64 }, \
    { "GeForce RTX 3060",                       3584,   56, 112,  256,  128,   64 }, \
    { "GeForce RTX 3050",                       2560,   40,  80,  256,  128,   64 }, \
    /* Turing Pro (Quadro RTX) — TU102 full / TU104                            \
       2nd gen TC, TF32 미지원 (tc_ops_tf32 = 0).                              \
       RTX 8000/6000은 compute 동일 (VRAM만 다름) */                          \
    { "Quadro RTX 8000",                        4608,  144, 576,  128,  128,    0 }, \
    { "Quadro RTX 6000",                        4608,  144, 576,  128,  128,    0 }, \
    { "Quadro RTX 5000",                        3072,   96, 384,  128,  128,    0 }, \
    /* Turing GeForce/Titan — TU102 cut (68 SM) / TU102 full (72 SM) / TU104 (46 SM) \
       2nd gen TC, TF32 미지원. FP32 acc 페널티 없음 (소비자 페널티는 Ampere부터). */ \
    { "GeForce RTX 2080 Ti",                    4352,  136, 544,  128,  128,    0 }, \
    { "Titan RTX",                              4608,  144, 576,  128,  128,    0 }, \
    { "GeForce RTX 2080",                       2944,   92, 368,  128,  128,    0 }, \
    /* Volta Titan — GV100 (80 SM), V100 과 코어 스펙 동일. 차이는 VRAM/버스폭뿐  \
       (12GB HBM2 3072-bit) 이라 CoreEntry 값은 V100 과 같다. substring "TITAN V" \
       가 "NVIDIA TITAN V" 와 CEO Edition(32GB) 까지 매칭, "Titan RTX" 와 무충돌. */ \
    { "TITAN V",                                5120, 2560, 640,  128,  128,    0 }, \
    /* Volta DC — GV100 (80 SM), 1st gen TC, FP16 input only, TF32 미지원.     \
       FP64=32/SM=2560 (1:2 of FP32, DC 풀 FP64). "Tesla V100"이 "V100"보다 먼저. */ \
    { "Tesla V100",                             5120, 2560, 640,  128,  128,    0 }, \
    { "V100",                                   5120, 2560, 640,  128,  128,    0 },

#endif /* CORE_TABLE_NVIDIA_H */
