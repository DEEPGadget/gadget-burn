/**
 * core_table_amd.h  —  AMD GPU 코어 DB (동적 Rpeak 계산용)
 * ─────────────────────────────────────────────────────────────
 * 본체의 CoreEntry 구조체를 그대로 재사용하되, AMD(RDNA/CDNA)에서는
 * 필드 의미를 재해석합니다 (NVIDIA Tensor Core 개념이 없으므로).
 *
 *   cores_fp32    : FP32 처리량 환산 "코어 수".
 *                   - RDNA3+/CDNA3+ 는 dual issue → CU × 128 (표기 SP의 2배)
 *                   - RDNA1/2, CDNA1/2 는 dual issue 없음 → CU × 64
 *                   Rpeak_FP32 = cores_fp32 × 2(FMA) × clk = CU × (128|256) × clk
 *   cores_fp64    : FP64 vector 처리량 환산. FP32 대비 비율로 결정.
 *                   - RDNA1/2 = 1:16 → CU × 4
 *                   - RDNA3   = 1:64 → CU × 2
 *                   - RDNA4   = 1:32 → CU × 4   (dual issue라 base가 2배)
 *                   - CDNA    는 vector FP64 가 강함 (데이터시트 값으로 역산)
 *   cores_tensor  : "Matrix/WMMA 가속 단위 수" = Compute Unit(CU) 수로 통일.
 *                   (CDNA/RDNA4 의 물리 matrix core 는 CU×4 지만, 여기서는
 *                    CU 단위로 환산해 tc_ops 에 CU당 총 OPS 를 넣는다.)
 *   tc_ops        : CU 1개가 1클럭에 처리하는 FP16/FP16-acc matrix FLOP.
 *   tc_ops_mix    : FP16/FP32-acc. AMD 는 FP32 누산 페널티 없음 → tc_ops 와 동일.
 *   tc_ops_tf32   : TF32 matrix FLOP/CU/clk. 미지원이면 0
 *                   (본체 calc_dynamic_rpeak 가 0 이면 FP32 shader Rpeak 로
 *                    폴백 → sgemm_tf32 가 f32 로 폴백되는 동작과 정확히 일치).
 *
 * matrix 가속이 없는 GPU (RDNA1/2)는 tc_ops=tc_ops_mix=tc_ops_tf32=0.
 * 이 경우 hgemm 등은 vector 경로로 실행되며 Peak% 도 vector 기준으로 폴백.
 *
 * ─────────────────────────────────────────────────────────────
 * [아키텍처별 matrix OPS/CU/clk — 사용자 권위 사양, 공식 TFLOPS 로 검증됨]
 *
 *   RDNA1/2   : matrix 없음 (0/0/0)
 *   RDNA3     : WMMA, FP16=512, mix=512, TF32 미지원(0)
 *   RDNA4     : Matrix Core CU당 4개, core당 256 → CU당 1024, TF32 미지원(0)
 *   CDNA1     : matrix core 4/CU. core당 FP16 256 → CU 1024; TF32/FP64m 미지원
 *   CDNA2     : core당 FP16 256 → CU 1024; TF32 미지원
 *   CDNA3     : core당 FP16 512 → CU 2048; TF32 256 → CU 1024
 *   CDNA4     : core당 FP16 1024 → CU 4096; TF32 미지원(BF16 에뮬)
 *
 * 매칭은 hipDeviceProp_t.name substring. 긴(구체적) 이름을 먼저 배치.
 *
 * [검증 — 모두 AMD 공식 TFLOPS 와 일치]
 *   RX 7900 XTX FP16 : 96 CU × 512  × 2498 MHz = 122.8 TFLOPS  ✓ (공식 ~123)
 *   RX 7900 XTX FP32 : 12288 × 2 × 2498 MHz     =  61.4 TFLOPS  ✓
 *   RX 7900 XTX FP64 : 192   × 2 × 2498 MHz      =  0.96 TFLOPS ✓ (1:64)
 *   MI100        FP16: 120 CU × 1024 × 1502 MHz = 184.6 TFLOPS  ✓
 *   MI100        FP32: 7680  × 2 × 1502 MHz      =  23.1 TFLOPS ✓
 *   MI100        FP64: 3840  × 2 × 1502 MHz      =  11.5 TFLOPS ✓
 *   MI300X       FP16: 304 CU × 2048 × 2100 MHz = 1307.4 TFLOPS ✓
 * ───────────────────────────────────────────────────────────── */

#ifndef CORE_TABLE_AMD_H
#define CORE_TABLE_AMD_H

/* CORE_TABLE 배열 안에 #ifdef GB_BACKEND_AMD 로 펼쳐지는 AMD 엔트리.
   필드: { substr, cores_fp32, cores_fp64, cores_tensor(=CU),
           tc_ops, tc_ops_mix, tc_ops_tf32 } */

#define GB_AMD_CORE_ENTRIES                                                    \
    /* ══ CDNA (Instinct MI) — 데이터센터, Matrix Core 가 FP32/FP64 도 지원 ══ \
       cores_tensor=CU, tc_ops=CU당 총 matrix OPS(=4 core × core당 OPS).      \
       구체적 모델명을 먼저 배치 (substring). */                              \
    /* ── CDNA1 (gfx908) — MI100, 120 CU, dual issue 없음 ──                  \
       FP16 1024(=4×256), BF16(mix)는 절반이지만 도구 mix=FP16/FP32-acc 라    \
       페널티 없는 1024 유지. FP64 vector 11.5TF → cores_fp64=CU×32.          \
       8번째 fp32_matrix_ops=256(=4 core×64): CDNA 는 FP32 도 matrix 가속 →   \
       sgemm Rpeak 를 matrix 기준으로 (120×256×1502=46TF, 공식 FP32 matrix). */\
    { "Instinct MI100",   7680, 3840, 120, 1024, 1024,    0, 256 },           \
    { "MI100",            7680, 3840, 120, 1024, 1024,    0, 256 },           \
    /* ── CDNA2 (gfx90a) — MI210(1 GCD 104), MI250/250X(2 GCD; amd_smi 는      \
       GCD 단위로 노출하므로 per-GCD CU 로 등록). dual issue 없음.            \
       FP16 1024, TF32 미지원. FP64 vector 가 강함 → cores_fp64=CU×64.        \
       FP32 matrix ops=256. */                                                \
    { "Instinct MI250X",  7040, 7040, 110, 1024, 1024,    0, 256 },           \
    { "Instinct MI250",   6656, 6656, 104, 1024, 1024,    0, 256 },           \
    { "Instinct MI210",   6656, 6656, 104, 1024, 1024,    0, 256 },           \
    { "MI250X",           7040, 7040, 110, 1024, 1024,    0, 256 },           \
    { "MI250",            6656, 6656, 104, 1024, 1024,    0, 256 },           \
    { "MI210",            6656, 6656, 104, 1024, 1024,    0, 256 },           \
    /* ── CDNA3 (gfx942) — MI300X/325X(304 CU), MI300A(228 CU). dual issue.   \
       FP16 2048(=4×512), TF32 1024(=4×256). FP64 vector cores_fp64=CU×64.    \
       FP32 matrix ops=256(=4 core×64). */                                    \
    { "Instinct MI325X", 38912, 19456, 304, 2048, 2048, 1024, 256, 4096 },    \
    { "Instinct MI300X", 38912, 19456, 304, 2048, 2048, 1024, 256, 4096 },    \
    { "Instinct MI300A", 29184, 14592, 228, 2048, 2048, 1024, 256, 4096 },    \
    { "MI325X",          38912, 19456, 304, 2048, 2048, 1024, 256, 4096 },    \
    { "MI300X",          38912, 19456, 304, 2048, 2048, 1024, 256, 4096 },    \
    { "MI300A",          29184, 14592, 228, 2048, 2048, 1024, 256, 4096 },    \
    /* ── CDNA4 (gfx950) — MI350X/355X, 256 CU. dual issue.                   \
       FP16 4096(=4×1024), TF32 미지원(BF16 에뮬). FP64 가 CDNA3 대비 절반    \
       (FP64 matrix=vector 동일 rate) → cores_fp64=CU×64 유지(vector 기준).   \
       FP32 matrix ops=256. */                                                \
    { "Instinct MI355X", 32768, 16384, 256, 4096, 4096,    0, 256, 8192 },    \
    { "Instinct MI350X", 32768, 16384, 256, 4096, 4096,    0, 256, 8192 },    \
    { "MI355X",          32768, 16384, 256, 4096, 4096,    0, 256, 8192 },    \
    { "MI350X",          32768, 16384, 256, 4096, 4096,    0, 256, 8192 },    \
    /* ══ RDNA3 (gfx110x) — Navi 31/32/33, WMMA, dual issue ══               \
       cores_fp32=CU×128, cores_fp64=CU×2 (1:64), FP16=512, TF32 미지원. */   \
    { "Radeon PRO W7900",        12288, 192, 96, 512, 512, 0 },               \
    { "Radeon PRO W7800",         8960, 140, 70, 512, 512, 0 },               \
    { "Radeon PRO W7700",         6144,  96, 48, 512, 512, 0 },               \
    { "Radeon RX 7900 XTX",      12288, 192, 96, 512, 512, 0 },               \
    { "Radeon RX 7900 XT",       10752, 168, 84, 512, 512, 0 },               \
    { "Radeon RX 7900 GRE",      10240, 160, 80, 512, 512, 0 },               \
    { "Radeon RX 7800 XT",        7680, 120, 60, 512, 512, 0 },               \
    { "Radeon RX 7700 XT",        6912, 108, 54, 512, 512, 0 },               \
    { "Radeon RX 7600 XT",        4096,  64, 32, 512, 512, 0 },               \
    { "Radeon RX 7600",           4096,  64, 32, 512, 512, 0 },               \
    /* ══ RDNA4 (gfx120x) — Navi 44/48, Matrix Core(CU당 1024), dual issue ══ \
       cores_fp32=CU×128, cores_fp64=CU×4 (1:32), FP16=1024, TF32 미지원.     \
       fp8(e4m3)=2×FP16=2048 (8번째 fp32_matrix_ops=0, 9번째 tc_ops_fp8=2048). \
       ※ hipBLASLt gfx1201 fp8 커널은 아직 미성숙(실측 << 이론) — docs 참고. */\
    { "Radeon AI PRO R9700",      8192, 256, 64, 1024, 1024, 0, 0, 2048 },    \
    { "Radeon RX 9070 XT",        8192, 256, 64, 1024, 1024, 0, 0, 2048 },    \
    { "Radeon RX 9070",           7168, 224, 56, 1024, 1024, 0, 0, 2048 },    \
    { "Radeon RX 9060 XT",        4096, 128, 32, 1024, 1024, 0, 0, 2048 },    \
    /* ══ RDNA2 (gfx103x) — Navi 21/22/23, Matrix 없음, dual issue 없음 ══    \
       cores_fp32=CU×64, cores_fp64=CU×4 (1:16), matrix=0/0/0. */             \
    { "Radeon PRO W6900X",        5120, 320, 80, 0, 0, 0 },                   \
    { "Radeon PRO W6800",         3840, 240, 60, 0, 0, 0 },                   \
    { "Radeon RX 6950 XT",        5120, 320, 80, 0, 0, 0 },                   \
    { "Radeon RX 6900 XT",        5120, 320, 80, 0, 0, 0 },                   \
    { "Radeon RX 6800 XT",        4608, 288, 72, 0, 0, 0 },                   \
    { "Radeon RX 6800",           3840, 240, 60, 0, 0, 0 },                   \
    { "Radeon RX 6700 XT",        2560, 160, 40, 0, 0, 0 },                   \
    { "Radeon RX 6600 XT",        2048, 128, 32, 0, 0, 0 },                   \
    { "Radeon RX 6600",           1792, 112, 28, 0, 0, 0 },                   \
    /* ══ RDNA1 (gfx101x) — Navi 10/14, Matrix 없음, dual issue 없음 ══       \
       cores_fp32=CU×64, cores_fp64=CU×4 (1:16), matrix=0/0/0. */             \
    { "Radeon PRO W5700",         2304, 144, 36, 0, 0, 0 },                   \
    { "Radeon RX 5700 XT",        2560, 160, 40, 0, 0, 0 },                   \
    { "Radeon RX 5700",           2304, 144, 36, 0, 0, 0 },                   \
    { "Radeon RX 5600 XT",        2304, 144, 36, 0, 0, 0 },                   \
    { "Radeon RX 5500 XT",        1408,  88, 22, 0, 0, 0 },

#endif /* CORE_TABLE_AMD_H */
