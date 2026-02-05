# 内存带宽测试 Makefile

NVCC = /usr/local/cuda-12.1/bin/nvcc
ARCH = sm_80
FLAGS = -O3 -arch=$(ARCH) --use_fast_math -lineinfo
INCLUDE = -I../../../common

TARGET = bandwidth_test
SRC = bandwidth_test.cu

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "正在编译 $(TARGET)..."
	$(NVCC) $(FLAGS) $(INCLUDE) $(SRC) -o $(TARGET)
	@echo "编译完成！"

run: $(TARGET)
	@echo "运行测试..."
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o *.ptx

.PHONY: all run clean
