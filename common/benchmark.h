/**
 * @file benchmark.h
 * @brief 基准测试公共组件
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA错误检查宏
#define CUDA_CHECK(call)  do {   cudaError_t err = call;   if (err != cudaSuccess) { fprintf(stderr, "CUDA错误 %s:%d: %s\n", __FILE__, __LINE__,cudaGetErrorString(err));   exit(EXIT_FAILURE); } } while(0)

// 设备信息结构
typedef struct {
    char name[256];
    int sm_count;
    int compute_major;
    int compute_minor;
    size_t total_memory;
    size_t l2_cache_size;
    int memory_bus_width;
    int memory_clock_rate;
    double theoretical_bandwidth;
} DeviceInfo;

/**
 * @brief 获取GPU设备信息
 */
inline void get_device_info(DeviceInfo* info) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    snprintf(info->name, sizeof(info->name), "%s", prop.name);
    info->sm_count = prop.multiProcessorCount;
    info->compute_major = prop.major;
    info->compute_minor = prop.minor;
    info->total_memory = prop.totalGlobalMem;
    info->l2_cache_size = prop.l2CacheSize;
    info->memory_bus_width = prop.memoryBusWidth;
    info->memory_clock_rate = prop.memoryClockRate;
    
    // 计算理论带宽 (GB/s)
    info->theoretical_bandwidth = 2.0 * prop.memoryClockRate * 
                                  (prop.memoryBusWidth / 8.0) / 1e6;
}

/**
 * @brief 打印设备信息
 */
inline void print_device_info(const DeviceInfo* info) {
    printf("\n========== GPU 设备信息 ==========\n");
    printf("设备名称: %s\n", info->name);
    printf("计算能力: %d.%d\n", info->compute_major, info->compute_minor);
    printf("SM数量: %d\n", info->sm_count);
    printf("全局内存: %.2f GB\n", info->total_memory / (1024.0*1024.0*1024.0));
    printf("L2缓存: %.2f MB\n", info->l2_cache_size / (1024.0*1024.0));
    printf("内存时钟: %.2f GHz\n", info->memory_clock_rate / 1e6);
    printf("内存总线宽度: %d bits\n", info->memory_bus_width);
    printf("理论带宽: %.2f GB/s\n", info->theoretical_bandwidth);
    printf("==================================\n\n");
}

/**
 * @brief 计时器结构
 */
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} Timer;

/**
 * @brief 初始化计时器
 */
inline void timer_init(Timer* timer) {
    CUDA_CHECK(cudaEventCreate(&timer->start));
    CUDA_CHECK(cudaEventCreate(&timer->stop));
}

/**
 * @brief 开始计时
 */
inline void timer_start(Timer* timer) {
    CUDA_CHECK(cudaEventRecord(timer->start));
}

/**
 * @brief 停止计时并返回经过的时间（毫秒）
 */
inline float timer_stop(Timer* timer) {
    float ms;
    CUDA_CHECK(cudaEventRecord(timer->stop));
    CUDA_CHECK(cudaEventSynchronize(timer->stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, timer->start, timer->stop));
    return ms;
}

/**
 * @brief 销毁计时器
 */
inline void timer_destroy(Timer* timer) {
    CUDA_CHECK(cudaEventDestroy(timer->start));
    CUDA_CHECK(cudaEventDestroy(timer->stop));
}

/**
 * @brief 计算带宽 (GB/s)
 * @param bytes 传输的字节数
 * @param ms 时间（毫秒）
 */
inline double calculate_bandwidth(size_t bytes, float ms) {
    return (bytes / (ms / 1000.0)) / 1e9;
}

#endif // BENCHMARK_H
