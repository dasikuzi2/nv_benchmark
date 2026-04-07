/**
 * @file gpu_info.cu
 * @brief GPU 设备信息全量查询工具
 *
 * 可移植：通过 CUDART_VERSION 宏自动适配不同 CUDA 版本，
 * 低版本 CUDA 编译时跳过不可用属性。
 */

#include "benchmark.h"
#include <cstring>

// ============================================================
//  辅助打印
// ============================================================

static const char* get_arch_name(int major, int minor) {
    if (major == 12) return "Blackwell";
    if (major == 10) return "Blackwell";
    if (major == 9 && minor == 0) return "Hopper";
    if (major == 8 && minor == 9) return "Ada Lovelace";
    if (major == 8 && minor == 6) return "Ampere";
    if (major == 8 && minor == 0) return "Ampere";
    if (major == 7 && minor == 5) return "Turing";
    if (major == 7 && minor == 0) return "Volta";
    if (major == 6) return "Pascal";
    if (major == 5) return "Maxwell";
    return "Unknown";
}

static void section(const char* title) {
    printf("\n========================================\n");
    printf("  %s\n", title);
    printf("========================================\n");
}

static void kv(const char* name, const char* value) {
    printf("  %-45s : %s\n", name, value);
}

static void kv_int(const char* name, int value) {
    printf("  %-45s : %d\n", name, value);
}

static void kv_size(const char* name, size_t bytes) {
    if (bytes >= (1ULL << 30))
        printf("  %-45s : %zu bytes (%.2f GB)\n", name, bytes, (double)bytes / (1ULL << 30));
    else if (bytes >= (1ULL << 20))
        printf("  %-45s : %zu bytes (%.2f MB)\n", name, bytes, (double)bytes / (1ULL << 20));
    else if (bytes >= (1ULL << 10))
        printf("  %-45s : %zu bytes (%.2f KB)\n", name, bytes, (double)bytes / (1ULL << 10));
    else
        printf("  %-45s : %zu bytes\n", name, bytes);
}

static void kv_bool(const char* name, int value) {
    kv(name, value ? "Yes" : "No");
}

// ============================================================
//  单 GPU 详细信息
// ============================================================

static void print_gpu_detail(int dev) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("\n################################################################\n");
    printf("  Device %d: %s\n", dev, prop.name);
    printf("################################################################\n");

    // --- 基本信息 ---
    section("基本信息 (Basic Info)");
    kv("GPU Name", prop.name);
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d.%d (%s, sm_%d%d)",
                 prop.major, prop.minor, get_arch_name(prop.major, prop.minor),
                 prop.major, prop.minor);
        kv("Compute Capability", buf);
    }
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "%02x:%02x.0 (domain %d, bus %d, device %d)",
                 prop.pciBusID, prop.pciDeviceID, prop.pciDomainID,
                 prop.pciBusID, prop.pciDeviceID);
        kv("PCI Bus ID", buf);
    }
    kv_bool("Multi-GPU Board", prop.isMultiGpuBoard);

    // --- SM 与核心 ---
    section("SM 与计算核心 (SM & Compute Cores)");
    kv_int("SM Count", prop.multiProcessorCount);
    kv_int("Max Threads per SM", prop.maxThreadsPerMultiProcessor);
    kv_int("Max Threads per Block", prop.maxThreadsPerBlock);
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "[%d, %d, %d]",
                 prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        kv("Max Thread Dimensions (block)", buf);
    }
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "[%d, %d, %d]",
                 prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        kv("Max Grid Dimensions", buf);
    }
    kv_int("Warp Size", prop.warpSize);
    kv_int("Max Blocks per SM", prop.maxBlocksPerMultiProcessor);
    kv_int("Max Registers per Block", prop.regsPerBlock);
    kv_int("Max Registers per SM", prop.regsPerMultiprocessor);

    // --- 时钟频率 ---
    section("时钟频率 (Clock Rates)");
    {
        int clockRate = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev));
        char buf[64];
        snprintf(buf, sizeof(buf), "%d MHz (%.2f GHz)", clockRate / 1000, clockRate / 1e6);
        kv("GPU Core Clock Rate", buf);
    }
    {
        int memClockRate = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, dev));
        char buf[64];
        snprintf(buf, sizeof(buf), "%d MHz (%.2f GHz)", memClockRate / 1000, memClockRate / 1e6);
        kv("Memory Clock Rate", buf);
    }

    // --- 全局显存 ---
    section("全局显存 (Global Memory)");
    kv_size("Total Global Memory", prop.totalGlobalMem);
    {
        char buf[64];
        snprintf(buf, sizeof(buf), "%d-bit", prop.memoryBusWidth);
        kv("Memory Bus Width", buf);
    }
    {
        int memClk = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&memClk, cudaDevAttrMemoryClockRate, dev));
        double bw = 2.0 * (memClk * 1000.0) * (prop.memoryBusWidth / 8.0) / 1e9;
        char buf[128];
        snprintf(buf, sizeof(buf), "%.2f GB/s (theoretical peak)", bw);
        kv("Memory Bandwidth", buf);
    }
    kv_bool("ECC Enabled", prop.ECCEnabled);
    kv_bool("Unified Addressing", prop.unifiedAddressing);
    kv_bool("Managed Memory", prop.managedMemory);
    kv_bool("Pageable Memory Access", prop.pageableMemoryAccess);

    // --- L2 Cache ---
    section("L2 缓存 (L2 Cache)");
    kv_size("L2 Cache Size", prop.l2CacheSize);
    if (prop.multiProcessorCount > 0) {
        int perSM = prop.l2CacheSize / prop.multiProcessorCount;
        char buf[128];
        snprintf(buf, sizeof(buf), "%d bytes (%.2f KB)", perSM, perSM / 1024.0);
        kv("L2 Cache per SM (avg)", buf);
    }
    kv_int("Persisting L2 Cache Max Size", prop.persistingL2CacheMaxSize);

    // --- Shared Memory ---
    section("共享内存 (Shared Memory)");
    kv_size("Shared Memory per Block", prop.sharedMemPerBlock);
    kv_size("Shared Memory per SM", prop.sharedMemPerMultiprocessor);
    {
        int maxOptin = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
        char buf[128];
        snprintf(buf, sizeof(buf), "%d bytes (%.2f KB)", maxOptin, maxOptin / 1024.0);
        kv("Max Shared Memory per Block (Optin)", buf);
    }
#if CUDART_VERSION >= 12000
    {
        int reserved = 0;
        cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, dev);
        char buf[128];
        snprintf(buf, sizeof(buf), "%d bytes", reserved);
        kv("Reserved Shared Memory per Block", buf);
    }
#endif

    // --- 常量内存与纹理 ---
    section("常量内存与纹理 (Constant Memory & Texture)");
    kv_size("Total Constant Memory", prop.totalConstMem);
    kv_size("Max Texture 1D Size", prop.maxTexture1D);
    kv_size("Texture Alignment", prop.textureAlignment);

    // --- 内存访问特性 ---
    section("内存访问特性 (Memory Access)");
    kv_bool("Global L1 Cache Supported", prop.globalL1CacheSupported);
    kv_bool("Local L1 Cache Supported", prop.localL1CacheSupported);
    kv_bool("Can Map Host Memory", prop.canMapHostMemory);

    // --- 异步与并发 ---
    section("异步与并发 (Async & Concurrency)");
    kv_int("Async Engine Count (DMA)", prop.asyncEngineCount);
    kv_bool("Concurrent Kernels", prop.concurrentKernels);
    kv_bool("Concurrent Managed Access", prop.concurrentManagedAccess);
    kv_bool("Cooperative Launch", prop.cooperativeLaunch);

    // --- 计算模式 ---
    section("计算模式 (Compute Mode)");
    {
        const char* modes[] = {"Default", "Exclusive", "Prohibited", "Exclusive Process"};
        int mode = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&mode, cudaDevAttrComputeMode, dev));
        kv("Compute Mode", (mode >= 0 && mode <= 3) ? modes[mode] : "Unknown");
    }
    {
        int timeout = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&timeout, cudaDevAttrKernelExecTimeout, dev));
        kv_bool("Kernel Execution Timeout", timeout);
    }
    kv_bool("Integrated GPU", prop.integrated);

    // --- 高级特性 ---
    section("高级特性 (Advanced Features)");
#if CUDART_VERSION >= 12000
    {
        int val = 0;
        cudaDeviceGetAttribute(&val, cudaDevAttrClusterLaunch, dev);
        kv_bool("Cluster Launch Support", val);
    }
    {
        int val = 0;
        cudaDeviceGetAttribute(&val, cudaDevAttrHostNativeAtomicSupported, dev);
        kv_bool("Host Native Atomic Supported", val);
    }
#endif
#if CUDART_VERSION >= 11060
    {
        int val = 0;
        cudaDeviceGetAttribute(&val, cudaDevAttrDeferredMappingCudaArraySupported, dev);
        kv_bool("Deferred Mapping CUDA Array", val);
    }
#endif
#if CUDART_VERSION >= 11020
    {
        int val = 0;
        cudaDeviceGetAttribute(&val, cudaDevAttrMemoryPoolsSupported, dev);
        kv_bool("Memory Pools Supported", val);
    }
#endif
    {
        int val = 0;
        cudaDeviceGetAttribute(&val, cudaDevAttrGPUDirectRDMASupported, dev);
        kv_bool("GPUDirect RDMA Supported", val);
    }
    {
        int ratio = 0;
        cudaDeviceGetAttribute(&ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, dev);
        char buf[64];
        snprintf(buf, sizeof(buf), "%d:1", ratio);
        kv("Single/Double Precision Perf Ratio", buf);
    }
}

// ============================================================
//  P2P 与 Runtime 信息
// ============================================================

static void print_p2p_info(int deviceCount) {
    if (deviceCount <= 1) return;
    section("P2P 连接 (Peer-to-Peer)");
    for (int i = 0; i < deviceCount; i++) {
        for (int j = 0; j < deviceCount; j++) {
            if (i == j) continue;
            int canAccess = 0;
            cudaDeviceCanAccessPeer(&canAccess, i, j);
            char buf[128];
            snprintf(buf, sizeof(buf), "Device %d -> Device %d P2P Access", i, j);
            kv_bool(buf, canAccess);
        }
    }
}

static void print_runtime_info() {
    section("CUDA Runtime & Driver 版本");
    int runtimeVer = 0, driverVer = 0;
    cudaRuntimeGetVersion(&runtimeVer);
    cudaDriverGetVersion(&driverVer);
    char buf[64];
    snprintf(buf, sizeof(buf), "%d.%d", runtimeVer / 1000, (runtimeVer % 1000) / 10);
    kv("CUDA Runtime Version", buf);
    snprintf(buf, sizeof(buf), "%d.%d", driverVer / 1000, (driverVer % 1000) / 10);
    kv("CUDA Driver Version", buf);
}

// ============================================================
//  main
// ============================================================

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("Detected %d CUDA device(s)\n", deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        print_gpu_detail(dev);

        // 如果所有 GPU 型号相同，后续跳过
        if (dev == 0 && deviceCount > 1) {
            cudaDeviceProp p0, p1;
            CUDA_CHECK(cudaGetDeviceProperties(&p0, 0));
            bool allSame = true;
            for (int i = 1; i < deviceCount; i++) {
                CUDA_CHECK(cudaGetDeviceProperties(&p1, i));
                if (strcmp(p1.name, p0.name) != 0) { allSame = false; break; }
            }
            if (allSame) {
                printf("\n[所有 %d 个 GPU 型号相同，跳过重复输出]\n", deviceCount);
                break;
            }
        }
    }

    print_p2p_info(deviceCount);
    print_runtime_info();

    printf("\n");
    return 0;
}
