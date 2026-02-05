# NVIDIA GPU 性能基准测试套件

精简、清晰、可扩展的GPU性能测试框架。

## 快速开始

```bash
cd benchmarks/memory/bandwidth
make run
```

## 项目结构

```
nv_benchmark_clean/
├── benchmarks/memory/bandwidth/  # 内存带宽测试
├── common/benchmark.h            # 公共组件（117行）
├── docs/                         # 文档
└── README.md                     # 本文件
```

## 核心特性

### 1. 公共组件抽象
- `CUDA_CHECK()` - 统一错误检查
- `DeviceInfo` - 设备信息获取
- `Timer` - 高精度计时
- `calculate_bandwidth()` - 带宽计算

### 2. 模块化设计
每个测试独立，易于添加新测试。

### 3. 中文文档
便于理解和使用。

## 测试结果 (A800-80GB)

| 配置 | 带宽 | 效率 |
|------|------|------|
| float4 | **1723 GB/s** | 84.5% |
| float (8x SMs) | 1236 GB/s | 60.6% |
| float (1x SMs) | 225 GB/s | 11.0% |

## 关键发现

1. **向量化至关重要**：float4 提升 2.25倍
2. **并发度很重要**：8x SM blocks 最优
3. **L2缓存感知**：< 40MB 性能提升 2.3倍

## 添加新测试

```bash
# 1. 创建目录
mkdir -p benchmarks/category/new_test

# 2. 编写代码（使用 common/benchmark.h）
#include ../../../common/benchmark.h

# 3. 添加Makefile和README
```

## 公共组件示例

```c
#include common/benchmark.h

int main() {
    // 获取设备信息
    DeviceInfo info;
    get_device_info(&info);
    print_device_info(&info);
    
    // 计时
    Timer timer;
    timer_init(&timer);
    timer_start(&timer);
    // ... kernel ...
    float ms = timer_stop(&timer);
    
    // 计算带宽
    double bw = calculate_bandwidth(bytes, ms);
    
    timer_destroy(&timer);
}
```

## 许可证

MIT License

---

**代码简洁 | 架构清晰 | 易于扩展**
