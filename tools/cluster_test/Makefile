# Cluster 特性探测工具 Makefile
# 自动检测 GPU 架构，支持跨平台编译

# 自动检测 nvcc 路径
NVCC ?= $(shell which nvcc 2>/dev/null)
ifeq ($(NVCC),)
  $(error 未找到 nvcc，请设置 NVCC 环境变量或确保 nvcc 在 PATH 中)
endif

# 自动检测 GPU 计算能力 (例: 12.0 -> 120, 8.9 -> 89)
GPU_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(GPU_CC),)
  $(warning 无法检测 GPU 架构，使用默认值 sm_80)
  GPU_CC := 80
endif
ARCH ?= sm_$(GPU_CC)

FLAGS = -O3 -arch=$(ARCH) -lineinfo
INCLUDE = -I../../common

TARGET = cluster_test
SRC = cluster_test.cu

all: $(TARGET)

$(TARGET): $(SRC) ../../common/benchmark.h
	@echo "编译 $(TARGET) (arch=$(ARCH))..."
	$(NVCC) $(FLAGS) $(INCLUDE) $(SRC) -o $(TARGET)
	@echo "编译完成！"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o

.PHONY: all run clean
