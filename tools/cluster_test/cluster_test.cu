/**
 * @file cluster_test.cu
 * @brief Cluster 特性探测工具
 *
 * 可移植：
 *   - CUDA >= 12.0 且 GPU 支持 cluster launch: 完整测试
 *   - CUDA >= 12.0 但 GPU 不支持 cluster: 打印提示后退出
 *   - CUDA < 12.0: 编译期跳过 cluster 相关代码，运行时提示不支持
 */

#include "benchmark.h"

#if CUDART_VERSION >= 12000
#define HAS_CLUSTER_API 1
#else
#define HAS_CLUSTER_API 0
#endif

// ============================================================
//  Kernel（所有架构可编译）
// ============================================================

__global__ void cluster_kernel() {
    __shared__ char shmem[49152];  // 48KB — 强制单 block/SM
    if (threadIdx.x == 0) shmem[0] = 1;
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("  [Kernel] Hello from block 0, thread 0 (shmem[0]=%d)\n",
               (int)shmem[0]);
    }
}

constexpr int BLOCK_SIZE = 1024;
constexpr int DYN_SMEM   = 48 * 1024;

// ============================================================
//  Cluster 探测（仅 CUDA >= 12）
// ============================================================

#if HAS_CLUSTER_API

static void probe_max_cluster_size(int dev) {
    printf("\n[3] Probing Maximum Cluster Size via cudaOccupancyMaxPotentialClusterSize...\n");
    int maxClusterSize = 0;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim  = dim3(1, 1, 1);
    cfg.blockDim = dim3(BLOCK_SIZE, 1, 1);
    cfg.dynamicSmemBytes = DYN_SMEM;
    cfg.numAttrs = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialClusterSize(
        &maxClusterSize, (void*)cluster_kernel, &cfg));
    printf("    Max Potential Cluster Size: %d blocks\n", maxClusterSize);
}

static int probe_cluster_sizes() {
    printf("\n[4] Probing cluster sizes by trial launch...\n");
    int maxWorking = 0;
    for (int cs = 1; cs <= 32; cs *= 2) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim  = dim3(cs, 1, 1);
        cfg.blockDim = dim3(BLOCK_SIZE, 1, 1);
        cfg.dynamicSmemBytes = DYN_SMEM;
        cfg.stream   = 0;

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {(unsigned)cs, 1, 1};
        cfg.attrs    = attrs;
        cfg.numAttrs = 1;

        int maxActive = 0;
        cudaError_t err = cudaOccupancyMaxActiveClusters(
            &maxActive, (void*)cluster_kernel, &cfg);
        if (err == cudaSuccess && maxActive > 0) {
            printf("    Cluster size %2d: OK  (max active clusters: %d)\n",
                   cs, maxActive);
            maxWorking = cs;
        } else {
            printf("    Cluster size %2d: FAIL (%s, active=%d)\n",
                   cs, cudaGetErrorString(err), maxActive);
            break;
        }
    }
    printf("\n    => Max Working Cluster Size: %d blocks\n", maxWorking);
    return maxWorking;
}

static void probe_scheduling_policies() {
    printf("\n[5] Testing Cluster Scheduling Policies...\n");
    const char* names[] = {"DEFAULT", "SPREAD", "LOAD_BAL"};
    cudaClusterSchedulingPolicy policies[] = {
        cudaClusterSchedulingPolicyDefault,
        cudaClusterSchedulingPolicySpread,
        cudaClusterSchedulingPolicyLoadBalancing
    };
    for (int i = 0; i < 3; i++) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim  = dim3(2, 1, 1);
        cfg.blockDim = dim3(BLOCK_SIZE, 1, 1);
        cfg.dynamicSmemBytes = DYN_SMEM;
        cfg.stream   = 0;

        cudaLaunchAttribute attrs[2];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {2, 1, 1};
        attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
        attrs[1].val.clusterSchedulingPolicyPreference = policies[i];
        cfg.attrs    = attrs;
        cfg.numAttrs = 2;

        cudaError_t err  = cudaLaunchKernelEx(&cfg, cluster_kernel);
        cudaError_t sync = cudaDeviceSynchronize();
        if (err == cudaSuccess && sync == cudaSuccess)
            printf("    Policy %-8s : SUPPORTED\n", names[i]);
        else
            printf("    Policy %-8s : FAILED (%s / %s)\n", names[i],
                   cudaGetErrorString(err), cudaGetErrorString(sync));
    }
}

static void launch_cluster_kernel(int maxWorking) {
    int cs = (maxWorking >= 2) ? 2 : 1;
    printf("\n[6] Launching kernel with cluster size %d...\n", cs);
    if (maxWorking >= 2) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim  = dim3(2, 1, 1);
        cfg.blockDim = dim3(BLOCK_SIZE, 1, 1);
        cfg.dynamicSmemBytes = DYN_SMEM;
        cfg.stream   = 0;

        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeClusterDimension;
        attrs[0].val.clusterDim = {2, 1, 1};
        cfg.attrs    = attrs;
        cfg.numAttrs = 1;

        CUDA_CHECK(cudaLaunchKernelEx(&cfg, cluster_kernel));
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("    Cluster kernel launch: SUCCESS\n");
    } else {
        printf("    Skipped (cluster size < 2)\n");
    }
}

static void cross_gpu_comparison(int deviceCount) {
    printf("\n=== Cross-GPU Comparison (1 block/SM config) ===\n");
    printf("%-6s %-25s %-8s  CS=1  CS=2  CS=4  CS=8\n",
           "GPU#", "Name", "SM Count");
    printf("------  -------------------------  --------  ----  ----  ----  ----\n");

    for (int dev = 0; dev < deviceCount; dev++) {
        CUDA_CHECK(cudaSetDevice(dev));
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        int clusterSupport = 0;
        cudaDeviceGetAttribute(&clusterSupport, cudaDevAttrClusterLaunch, dev);
        if (!clusterSupport) {
            printf("  %d     %-25s %3d       [cluster not supported]\n",
                   dev, prop.name, prop.multiProcessorCount);
            continue;
        }

        int results[4] = {};
        int sizes[] = {1, 2, 4, 8};
        for (int i = 0; i < 4; i++) {
            cudaLaunchConfig_t cfg = {};
            cfg.gridDim  = dim3(sizes[i], 1, 1);
            cfg.blockDim = dim3(BLOCK_SIZE, 1, 1);
            cfg.dynamicSmemBytes = DYN_SMEM;
            cfg.stream   = 0;

            cudaLaunchAttribute attrs[1];
            attrs[0].id = cudaLaunchAttributeClusterDimension;
            attrs[0].val.clusterDim = {(unsigned)sizes[i], 1, 1};
            cfg.attrs    = attrs;
            cfg.numAttrs = 1;

            cudaOccupancyMaxActiveClusters(
                &results[i], (void*)cluster_kernel, &cfg);
        }
        printf("  %d     %-25s %3d       %3d   %3d   %3d   %3d\n",
               dev, prop.name, prop.multiProcessorCount,
               results[0], results[1], results[2], results[3]);
    }
}

static void run_cluster_probe(int dev) {
    printf("\n========================================\n");
    printf("  Cluster Feature Probe (dev %d)\n", dev);
    printf("========================================\n\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("[1] Device: %s\n", prop.name);
    printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("    SM Count: %d\n", prop.multiProcessorCount);

    int clusterSupport = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clusterSupport, cudaDevAttrClusterLaunch, dev));
    printf("\n[2] Cluster Launch Supported: %s\n",
           clusterSupport ? "YES" : "NO");
    if (!clusterSupport) {
        printf("    Cluster not supported on this device. Stopping.\n");
        return;
    }

    probe_max_cluster_size(dev);
    int maxWorking = probe_cluster_sizes();
    probe_scheduling_policies();
    launch_cluster_kernel(maxWorking);

    printf("\n========================================\n");
    printf("  SUMMARY\n");
    printf("========================================\n");
    printf("  Cluster Support:            YES\n");
    printf("  Max Verified Cluster Size:  %d blocks\n", maxWorking);
    printf("========================================\n\n");
}

#endif // HAS_CLUSTER_API

// ============================================================
//  main
// ============================================================

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("Found %d GPU(s)\n", deviceCount);

#if !HAS_CLUSTER_API
    printf("\n[!] CUDA Runtime %d 不支持 Cluster API (需要 >= 12000)\n",
           CUDART_VERSION);
    printf("    请升级 CUDA Toolkit 后重新编译。\n");
    return 0;
#else
    // 详细探测第一个设备
    CUDA_CHECK(cudaSetDevice(0));
    run_cluster_probe(0);

    // 多 GPU 对比表
    if (deviceCount > 0)
        cross_gpu_comparison(deviceCount);

    printf("\n");
    return 0;
#endif
}
