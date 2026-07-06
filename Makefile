## gadget_burn — Makefile
## ═══════════════════════════════════════════════════════════════
##
## [백엔드 선택]  backend=nvidia (기본) | amd
##
##  make                  → NVIDIA (CUDA/cuBLAS/NVML), native SASS
##  make nvidia           → 위와 동일
##  make amd              → AMD (HIP/rocBLAS/amd_smi), gfx 자동/지정
##
## [NVIDIA 빌드 모드]  mode=native (기본) | ptx | fatbin
##
##  make                  → native, 현재 머신 GPU 최적 SASS
##  make mode=ptx         → PTX only : 빌드 머신 GPU 불필요, 이식성 최고
##  make mode=fatbin      → Fat binary : sm_60~sm_90a SASS + PTX 전부 포함
##
## [AMD 빌드 아키텍처]  amdarch=<자동 감지> (기본)
##
##  make amd                       → 빌드 머신 GPU 자동 감지 (amdgpu-arch)
##  make amd amdarch=gfx908        → CDNA1 (MI100)
##  make amd amdarch=gfx90a        → CDNA2 (MI200)
##  make amd amdarch=gfx942        → CDNA3 (MI300)
##  make amd amdarch=gfx1100       → RDNA3 (RX 7900 / W7900)
##  make amd amdarch=gfx1201       → RDNA4 (RX 9070)
##  make amd amdarch=native        → hipcc 내장 자동 감지
##
##  ┌─ ⚠ 중요: AMD 바이너리는 빌드한 gfx 아키텍처에서만 실행됩니다 ─────┐
##  │ NVIDIA 와 달리 AMD GPU ISA 는 가상 ISA(PTX 상당)가 없어, 잘못된    │
##  │ arch 로 빌드해도 빌드는 성공하고 GPU 커널 launch 시점에 SIGSEGV   │
##  │ 로 죽습니다 (예: gfx1100 바이너리를 MI100=gfx908 에서 실행).        │
##  │ 특히 공유 파일시스템(NFS 등)에서 여러 GPU 머신이 같은 ./gadget_burn │
##  │ 을 보면, 한 머신의 빌드가 다른 머신용 바이너리를 덮어쓰므로 반드시  │
##  │ 실행 머신의 arch 로 다시 빌드하세요 (실행 전 make amd amdarch=…).   │
##  └────────────────────────────────────────────────────────────────────┘
##
## [AMD 이식성 옵션 — NVIDIA PTX 에 정확히 대응하는 단일 가상 ISA 는 없음]
##
##  AMD 에는 PTX 같은 "전 세대 만능 가상 ISA + 런타임 JIT" 이 없고,
##  두 가지 우회책이 있습니다 (필요 시 amdarch 에 직접 지정):
##   1) Fat binary : 여러 arch 동시 포함. 가장 확실, 공유 FS 안전.
##        make amd amdarch="gfx908 gfx90a gfx942 gfx1100 gfx1200"
##        (Makefile 의 --offload-arch 는 공백 분리 다중 지정 가능)
##   2) Generic    : gfx9-generic 등 패밀리 단위 (ROCm 6.3+, code-object v6).
##        패밀리 내부(예 gfx9 = MI100/MI200/MI300/MI350)만 이식.
##        make amd amdarch=gfx9-generic   (서로 다른 패밀리는 fat binary 필요)
##
## [PTX vs Fat binary] (NVIDIA 전용)
##
##  native   : 빌드 시점 GPU에 최적화된 SASS → 즉시 실행, 타 GPU 이식 불가
##  PTX only : 런타임 드라이버가 JIT 컴파일 → 첫 실행 느리지만 이식성 최고
##  Fat bin  : 미리 컴파일된 SASS 포함 → 빠른 첫 실행, 바이너리 크기 증가
##
## ═══════════════════════════════════════════════════════════════

TARGET   = gadget_burn
SRC      = gadget_burn.cu

## 기본 백엔드: nvidia
backend ?= nvidia

## ══════════════════════════════════════════════════════════════
##  NVIDIA 백엔드 (CUDA / cuBLAS / NVML)
## ══════════════════════════════════════════════════════════════
ifeq ($(backend),nvidia)

NVCC    ?= nvcc
CC       = $(NVCC)
LIBS     = -lcublas -lcublasLt -lnvidia-ml
CFLAGS   = -DGB_BACKEND_NVIDIA -O3 -Xcompiler -pthread,-O3

## NVIDIA 빌드 모드: native (기본)
mode    ?= native

## ── Native: 빌드 시점 GPU SM 자동 감지 ──
NATIVE_FLAGS = -arch=native

## ── PTX only: compute_60 가상 아키텍처로 컴파일, SASS 없이 PTX만 임베드 ──
PTX_FLAGS = -gencode arch=compute_60,code=compute_60

## ── Fat binary: sm_60~sm_90a SASS + forward-compat PTX ──
FATBIN_FLAGS = \
    -gencode arch=compute_60,code=sm_60  \
    -gencode arch=compute_61,code=sm_61  \
    -gencode arch=compute_62,code=sm_62  \
    -gencode arch=compute_70,code=sm_70  \
    -gencode arch=compute_72,code=sm_72  \
    -gencode arch=compute_75,code=sm_75  \
    -gencode arch=compute_80,code=sm_80  \
    -gencode arch=compute_86,code=sm_86  \
    -gencode arch=compute_87,code=sm_87  \
    -gencode arch=compute_89,code=sm_89  \
    -gencode arch=compute_90,code=sm_90  \
    -gencode arch=compute_90,code=sm_90a \
    -gencode arch=compute_90,code=compute_90

ifeq ($(mode),ptx)
    ARCHFLAGS = $(PTX_FLAGS)
    MODE_DESC = NVIDIA / PTX only (compute_60 → 런타임 JIT)
else ifeq ($(mode),fatbin)
    ARCHFLAGS = $(FATBIN_FLAGS)
    MODE_DESC = NVIDIA / Fat binary (sm_60 ~ sm_90a SASS + PTX)
else
    ARCHFLAGS = $(NATIVE_FLAGS)
    MODE_DESC = NVIDIA / Native (현재 GPU 자동 감지)
endif

VERSION_CMD = $(NVCC) --version | grep release | awk '{print $$6}'

## ══════════════════════════════════════════════════════════════
##  AMD 백엔드 (HIP / rocBLAS / amd_smi)
## ══════════════════════════════════════════════════════════════
else ifeq ($(backend),amd)

HIPCC   ?= hipcc
CC       = $(HIPCC)
ROCM    ?= /opt/rocm
LIBS     = -L$(ROCM)/lib -lrocblas -lhipblaslt -lamd_smi
## hipcc(clang)는 -Xcompiler 문법을 받지 않음 → -pthread 직접 전달
CFLAGS   = -DGB_BACKEND_AMD -O3 -pthread -I. -I$(ROCM)/include

## AMD 빌드 아키텍처.
## 기본값은 빌드 머신에 설치된 GPU 를 amdgpu-arch 로 자동 감지합니다
## (공유 FS 에서 arch 불일치로 인한 런타임 SIGSEGV 방지). GPU 가 없거나
## 감지 실패 시 gfx1100 로 폴백. 사용자가 amdarch= 를 명시하면 그게 우선.
##
##   make amd                       → 빌드 머신 GPU 자동 감지
##   make amd amdarch=gfx908        → 명시 지정 (예: MI100)
##   make amd amdarch="gfx908 gfx1100"  → fat binary (여러 arch 동시)
AMD_DETECTED := $(shell { amdgpu-arch 2>/dev/null || $(ROCM)/llvm/bin/amdgpu-arch 2>/dev/null; } | sort -u | head -1)
amdarch ?= $(if $(AMD_DETECTED),$(AMD_DETECTED),gfx1100)
ifeq ($(amdarch),native)
    ARCHFLAGS =
    MODE_DESC = AMD / native (빌드 머신 GPU 자동 감지)
else
    ## 각 arch 에 --offload-arch= 를 붙여 다중(=fat binary) 지원.
    ## generic 타깃(gfx9-generic 등)은 code-object v6 필요.
    ARCHFLAGS = $(foreach a,$(amdarch),--offload-arch=$(a))
    ifneq ($(findstring generic,$(amdarch)),)
        ARCHFLAGS += -mcode-object-version=6
    endif
    MODE_DESC = AMD / $(amdarch)
endif

VERSION_CMD = $(HIPCC) --version 2>/dev/null | grep -i "HIP version" | head -1

else
$(error 알 수 없는 backend '$(backend)' — nvidia 또는 amd 를 사용하세요)
endif

## ══════════════════════════════════════════════════════════════
##  공통 빌드 규칙
## ══════════════════════════════════════════════════════════════

.PHONY: all clean info nvidia amd

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "══════════════════════════════════════════════"
	@echo " 빌드 대상  : $(MODE_DESC)"
	@echo " 컴파일러   : $(shell $(VERSION_CMD))"
	@echo "══════════════════════════════════════════════"
	$(CC) $(CFLAGS) $(ARCHFLAGS) -o $@ $< $(LIBS)
	@echo ""
	@echo "✔ 빌드 완료 → ./$(TARGET)   (backend=$(backend))"
	@echo "  빠른 시작:  ./$(TARGET)"
	@echo ""

## 백엔드 단축 명령
nvidia: ; $(MAKE) backend=nvidia
amd:    ; $(MAKE) backend=amd

## PTX 텍스트 덤프 (NVIDIA 디버그/검증용)
dump_ptx: $(SRC)
	$(NVCC) -DGB_BACKEND_NVIDIA -O3 -gencode arch=compute_60,code=compute_60 -ptx -o $(TARGET).ptx $<
	@echo "PTX 덤프 완료: $(TARGET).ptx"

## 바이너리에 포함된 아키텍처 목록 확인
info: $(TARGET)
	@echo "── 포함된 GPU 코드 목록 ──"
	cuobjdump -lelf $(TARGET) 2>/dev/null || \
	    echo "(cuobjdump 없음 – NVIDIA 빌드에서만 사용 가능)"

clean:
	rm -f $(TARGET) $(TARGET).ptx
