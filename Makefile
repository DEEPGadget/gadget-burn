## gadget_burn — Makefile
## ═══════════════════════════════════════════════════════════════
##
## [빌드 모드]
##
##  make                  → 기본 (native, 현재 머신 GPU 최적 SASS)
##  make mode=native      → 위와 동일
##  make mode=ptx         → PTX only : 빌드 머신 GPU 불필요, 이식성 최고
##  make mode=fatbin      → Fat binary : sm_60~sm_90a SASS + PTX 전부 포함
##
## [PTX vs Fat binary]
##
##  native   : 빌드 시점 GPU에 최적화된 SASS → 즉시 실행, 타 GPU 이식 불가
##  PTX only : 런타임 드라이버가 JIT 컴파일 → 첫 실행 느리지만 이식성 최고
##  Fat bin  : 미리 컴파일된 SASS 포함 → 빠른 첫 실행, 바이너리 크기 증가
##
## [지원 아키텍처 - sm_60 이후 전체]
##
##   sm_60  Pascal    (P100)          sm_75  Turing    (RTX 2080)
##   sm_61  Pascal    (GTX 1080)      sm_80  Ampere    (A100, A30)
##   sm_62  Parker    (Tegra TX2)     sm_86  Ampere    (RTX 3090, A6000)
##   sm_70  Volta     (V100)          sm_89  Ada       (RTX 4090, L40)
##   sm_72  Xavier    (Jetson AGX)    sm_90  Hopper    (H100 SXM)
##
## ═══════════════════════════════════════════════════════════════

NVCC    ?= nvcc
TARGET   = gadget_burn
SRC      = gadget_burn.cu
LIBS     = -lcublas -lnvidia-ml
CFLAGS   = -O3 -Xcompiler -pthread,-O3

## 기본 모드: native
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
    GENCODE   = $(PTX_FLAGS)
    MODE_DESC = PTX only (compute_60 → 런타임 JIT)
else ifeq ($(mode),fatbin)
    GENCODE   = $(FATBIN_FLAGS)
    MODE_DESC = Fat binary (sm_60 ~ sm_90a SASS + PTX)
else
    GENCODE   = $(NATIVE_FLAGS)
    MODE_DESC = Native (현재 GPU 자동 감지)
endif

## ── 빌드 규칙 ──

.PHONY: all clean info ptx fatbin native

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "══════════════════════════════════════════════"
	@echo " 빌드 모드  : $(MODE_DESC)"
	@echo " 컴파일러   : $(shell $(NVCC) --version | grep release | awk '{print $$6}')"
	@echo "══════════════════════════════════════════════"
	$(NVCC) $(CFLAGS) $(GENCODE) -o $@ $< $(LIBS)
	@echo ""
	@echo "✔ 빌드 완료 → ./$(TARGET)"
	@echo "  빠른 시작:  ./$(TARGET)"
	@echo ""

## 모드 단축 명령
ptx:    ; $(MAKE) mode=ptx
fatbin: ; $(MAKE) mode=fatbin
native: ; $(MAKE) mode=native

## PTX 텍스트 덤프 (디버그/검증용)
dump_ptx: $(SRC)
	$(NVCC) $(CFLAGS) $(PTX_FLAGS) -ptx -o $(TARGET).ptx $<
	@echo "PTX 덤프 완료: $(TARGET).ptx"

## 바이너리에 포함된 아키텍처 목록 확인
info: $(TARGET)
	@echo "── 포함된 GPU 코드 목록 ──"
	cuobjdump -lelf $(TARGET) 2>/dev/null || \
	    echo "(cuobjdump 없음 – CUDA toolkit bin 경로 확인)"

clean:
	rm -f $(TARGET) $(TARGET).ptx