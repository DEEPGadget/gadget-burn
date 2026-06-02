# 빌드와 이식성

## 백엔드 선택

```bash
make            # = make nvidia (기본)
make nvidia     # NVIDIA: nvcc + cuBLAS + NVML
make amd        # AMD: hipcc + rocBLAS + amd_smi (빌드 머신 GPU 자동 감지)
```

| 백엔드 | 컴파일러 | 라이브러리 | 매크로 |
|---|---|---|---|
| NVIDIA | `nvcc` | `-lcublas -lnvidia-ml` | `-DGB_BACKEND_NVIDIA` |
| AMD | `hipcc` | `-lrocblas -lamd_smi` | `-DGB_BACKEND_AMD` |

## 의존성

| 백엔드 | Runtime | BLAS | 모니터링 |
|---|---|---|---|
| NVIDIA | CUDA Runtime | cuBLAS | NVML (libnvidia-ml) |
| AMD | HIP Runtime | rocBLAS | amd_smi (libamd_smi) |

공통으로 pthreads(GPU별 독립 스레드)를 사용합니다.

## NVIDIA 빌드 모드

```bash
make nvidia                  # native (기본) — 현재 머신 GPU 최적 SASS
make nvidia mode=ptx         # PTX only — 빌드 머신 GPU 불필요, 이식성 최고
make nvidia mode=fatbin      # Fat binary — sm_60~sm_90a SASS + PTX 전부 포함
```

| 모드 | 특징 | 권장 |
|---|---|---|
| `native` | 빌드 시점 GPU에 최적, 타 GPU 이식 불가 | 단일 환경 |
| `ptx` | PTX(가상 ISA)만 임베드 → 런타임 드라이버가 JIT. 첫 실행 수 초 지연, 이식성 최고 | 여러 아키텍처 배포 |
| `fatbin` | 미리 컴파일된 SASS 포함 → 빠른 첫 실행, 바이너리 큼 | 배포 + 성능 |

## AMD 빌드 아키텍처

기본값은 `amdgpu-arch`로 **빌드 머신 GPU를 자동 감지**합니다(감지 실패 시 gfx1100 폴백).

```bash
make amd                       # 빌드 머신 GPU 자동 감지
make amd amdarch=gfx908        # CDNA1 (MI100)
make amd amdarch=gfx90a        # CDNA2 (MI200)
make amd amdarch=gfx942        # CDNA3 (MI300)
make amd amdarch=gfx1100       # RDNA3 (RX 7900 / W7900)
make amd amdarch=gfx1201       # RDNA4 (RX 9070)
```

### ⚠ AMD 바이너리는 빌드한 arch에서만 실행됩니다

NVIDIA PTX 같은 "전 세대 만능 가상 ISA"가 AMD에는 **없습니다.** AMD GPU ISA(gfx908 등)는 실제 머신 코드에 가까워서, **잘못된 arch로 빌드해도 빌드는 성공하고 GPU 커널 launch 시점에 SIGSEGV로 죽습니다** (예: gfx1100 바이너리를 MI100=gfx908에서 실행).

특히 **공유 파일시스템(NFS 등)에서 여러 GPU 머신이 같은 `./gadget_burn`을 볼 때**, 한 머신의 빌드가 다른 머신용 바이너리를 덮어쓰면 그쪽에서 실행 시 죽습니다. 기본값을 자동 감지로 둔 이유가 이것이며, 실행 머신에서 다시 빌드하는 것이 가장 안전합니다.

바이너리에 든 arch 확인:
```bash
/opt/rocm/llvm/bin/llvm-objdump --offloading gadget_burn | grep gfx
```

### AMD 이식성 옵션

PTX에 정확히 대응하는 단일 가상 ISA는 없지만, 두 가지 우회책이 있습니다:

**1) Fat binary** — 여러 arch를 한 바이너리에 (가장 확실, 공유 FS 안전):
```bash
make amd amdarch="gfx908 gfx90a gfx942 gfx1100 gfx1200"
```
공백으로 여러 arch를 주면 모두 포함됩니다. 어느 머신에서나 실행되지만 바이너리가 커지고, **빌드 시점에 대상 GPU를 미리 알아야** 합니다(미래 GPU 불가).

**2) Generic targets** (ROCm 6.3+, code-object v6) — 패밀리 단위 이식:
```bash
make amd amdarch=gfx9-generic    # gfx9 패밀리 (MI100/MI200/MI300/MI350)
```
`gfx9-generic` / `gfx10-1-generic` / `gfx10-3-generic` / `gfx11-generic` / `gfx12-generic`이 있으며 **패밀리 내부만** 이식됩니다(gfx9 ≠ gfx11). Makefile이 generic 감지 시 `-mcode-object-version=6`을 자동 추가합니다. 서로 다른 패밀리를 동시에 지원하려면 결국 fat binary가 필요합니다.

| | NVIDIA PTX | AMD Fat binary | AMD Generic |
|---|---|---|---|
| 단일 코드로 만능 | ✅ 전 세대 | ❌ 명시한 것만 | △ 패밀리 내만 |
| 미래 GPU | ✅ JIT | ❌ | △ 같은 패밀리면 |
| 바이너리 크기 | 작음 | 큼 | 중간 |

## 추가 Make 타깃

```bash
make info        # 바이너리에 포함된 GPU 아키텍처 목록 (NVIDIA: cuobjdump)
make dump_ptx    # PTX 텍스트 추출 (NVIDIA, gadget_burn.ptx)
make clean       # 빌드 파일 삭제
```
