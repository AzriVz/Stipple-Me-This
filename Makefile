# Stipple-Me-This
# Build:  make            (membutuhkan CUDA toolkit + gcc + OpenMP)
#         make bench-clean  (build tanpa auto-vektorisasi utk benchmark SIMD yg adil)
#
# CUDA_HOME default mengarah ke toolkit lokal tanpa sudo di ~/cuda-13.3.
# Ubah CUDA_HOME bila memakai instalasi sistem (mis. /usr/local/cuda).

CUDA_HOME ?= $(HOME)/cuda-13.3/usr/local/cuda-13.3
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CXX       ?= g++

CPPFLAGS  := -Ithird_party -Isrc -std=c++17
WARN      := -Wall -Wextra
QT_CFLAGS := $(shell pkg-config --cflags Qt6Widgets Qt6Concurrent)
QT_LIBS   := $(shell pkg-config --libs Qt6Widgets Qt6Concurrent)

# baseline scalar: tanpa auto-vektorisasi agar SIMD diukur secara adil
SCALAR_FLAGS := -O2 -fno-tree-vectorize -fopenmp
# implementasi SIMD eksplisit
SIMD_FLAGS   := -O3 -mavx2 -mfma -fopenmp
# TU umum
GENERAL_FLAGS := -O3 -fopenmp
NVCC_FLAGS  := -O3 -std=c++17 -arch=sm_75 -Ithird_party -Isrc -Xcompiler -O3

CUDA_LIB   := $(CUDA_HOME)/lib64
LDFLAGS    := -fopenmp -L$(CUDA_LIB) -Wl,-rpath,$(CUDA_LIB) -lcudart -lm

OBJS := build/main.o build/gui.o build/image.o build/stipple.o \
        build/serial.o build/openmp.o build/simd.o build/cuda.o

all: stipple

build:
	mkdir -p build

build/main.o: src/main.cpp src/stipple.hpp src/image.hpp src/gui.hpp | build
	$(CXX) $(CPPFLAGS) $(GENERAL_FLAGS) $(WARN) -c $< -o $@

build/gui.o: src/gui.cpp src/gui.hpp src/stipple.hpp src/image.hpp | build
	$(CXX) $(CPPFLAGS) $(QT_CFLAGS) $(GENERAL_FLAGS) $(WARN) -fPIC -c $< -o $@

build/image.o: src/image.cpp src/image.hpp src/stipple.hpp src/gif_writer.hpp third_party/stb_image.h third_party/stb_image_write.h | build
	$(CXX) $(CPPFLAGS) $(GENERAL_FLAGS) $(WARN) -c $< -o $@

build/stipple.o: src/stipple.cpp src/stipple.hpp | build
	$(CXX) $(CPPFLAGS) $(GENERAL_FLAGS) $(WARN) -c $< -o $@

build/serial.o: src/serial.cpp src/stipple.hpp | build
	$(CXX) $(CPPFLAGS) $(SCALAR_FLAGS) $(WARN) -c $< -o $@

build/openmp.o: src/openmp.cpp src/stipple.hpp | build
	$(CXX) $(CPPFLAGS) $(SCALAR_FLAGS) $(WARN) -c $< -o $@

build/simd.o: src/simd.cpp src/stipple.hpp | build
	$(CXX) $(CPPFLAGS) $(SIMD_FLAGS) $(WARN) -c $< -o $@

build/cuda.o: src/cuda_stippler.cu src/stipple.hpp | build
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

stipple: $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) $(QT_LIBS) -o $@

clean:
	rm -rf build stipple

.PHONY: all clean
