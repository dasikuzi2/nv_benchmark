/**
 * @file bandwidth_test.cu
 * @brief HBM内存带宽测试
 * 
 * 测试项目:
 * 1. Block配置对带宽的影响
 * 2. 数据大小与L2缓存效应
 * 3. 向量化加载 (float vs float4)
 */

#include "../../../common/benchmark.h"

// ==================== Kernel实现 ====================

/**
 * @brief 标量加载测试 (float)
 */
__global__ void load_float_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    for (size_t i = idx; i < n; i += stride) {
        sum += input[i];
    }
    
    // 写入一个值（最小化写带宽影响）
    if (idx < gridDim.x * blockDim.x) {
        output[idx] = sum;
    }
}

/**
 * @brief 向量化加载测试 (float4)
 */
__global__ void load_float4_kernel(const float4* __restrict__ input,
                                    float4* __restrict__ output,
                                    size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (size_t i = idx; i < n; i += stride) {
        float4 val = input[i];
        sum.x += val.x;
        sum.y += val.y;
        sum.z += val.z;
        sum.w += val.w;
    }
    
    if (idx < gridDim.x * blockDim.x) {
        output[idx] = sum;
    }
}

// ==================== 测试函数 ====================

/**
 * @brief 测试不同Block配置
 */
void test_block_config(const DeviceInfo* info, float* d_input, float* d_output, 
                       size_t n, int iterations) {
    printf("\n========== 测试1: 标量测试float ==========\n\n");
    
    int block_size[] = {128, 256, 512, 1024};
    int configs[] = {1, 2, 4};
    
    Timer timer;
    timer_init(&timer);
    
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++){
            int num_blocks = info->sm_count * configs[i];
            int threadperblock = block_size[j];
            // Warmup
            load_float_kernel<<<num_blocks, threadperblock>>>(d_input, d_output, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // 计时
            timer_start(&timer);
            for (int j = 0; j < iterations; j++) {
                load_float_kernel<<<num_blocks, threadperblock>>>(d_input, d_output, n);
            }
            float ms = timer_stop(&timer);
            
            double bw = calculate_bandwidth(n * sizeof(float) * iterations, ms);
            double efficiency = (bw / info->theoretical_bandwidth) * 100.0;
            
            printf("Blocks=%4d (%dx SMs), threadperblock=%4d: %7.2f GB/s (效率 %.1f%%)\n",
                num_blocks, configs[i], block_size[j], bw, efficiency);
        }
    }
    
    timer_destroy(&timer);
}

/**
 * @brief 测试L2缓存效应
 */
void test_l2_cache(const DeviceInfo* info, float* d_input, float* d_output,
                   int iterations) {
    printf("\n========== 测试2: L2缓存效应 ==========\n\n");
    printf("L2缓存大小: %.2f MB\n\n", info->l2_cache_size / (1024.0*1024.0));
    
    size_t l2_size = info->l2_cache_size;
    size_t sizes[] = {
        l2_size / 4,      // 0.25x L2
        l2_size / 2,      // 0.5x L2
        l2_size,          // 1.0x L2
        l2_size * 2,      // 2.0x L2
        l2_size * 4,      // 4.0x L2
        256ULL * 1024 * 1024  // 1GB
    };
    
    const int num_blocks = info->sm_count * 4;
    const int block_size = 256;
    
    Timer timer;
    timer_init(&timer);
    
    for (int i = 0; i < 6; i++) {
        size_t n = sizes[i] / sizeof(float);
        double ratio = (double)sizes[i] / l2_size;
        
        // Warmup
        load_float_kernel<<<num_blocks, block_size>>>(d_input, d_output, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 计时
        timer_start(&timer);
        for (int j = 0; j < iterations; j++) {
            load_float_kernel<<<num_blocks, block_size>>>(d_input, d_output, n);
        }
        float ms = timer_stop(&timer);
        
        double bw = calculate_bandwidth(sizes[i] * iterations, ms);
        
        printf("数据大小: %6.1f MB (%.2fx L2): %7.2f GB/s\n",
               sizes[i] / (1024.0*1024.0), ratio, bw);
    }
    
    timer_destroy(&timer);
}

/**
 * @brief 测试向量化
 */
void test_vectorization(const DeviceInfo* info, float* d_input, float* d_output,
                        size_t n, int iterations) {
    printf("\n========== 测试3: 向量测试float4 ==========\n\n");
    
    int configs[] = {1, 2, 4};
    int block_size[] = {128, 256, 512, 1024};
    
    Timer timer;
    timer_init(&timer);

    float4* d_input4 = reinterpret_cast<float4*>(d_input);
    float4* d_output4 = reinterpret_cast<float4*>(d_output);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            
            int num_blocks = info->sm_count * configs[i];
            int threadperblock = block_size[j];
            // Warmup
            load_float4_kernel<<<num_blocks, threadperblock>>>(d_input4, d_output4, n / 4);
            CUDA_CHECK(cudaDeviceSynchronize());

            timer_start(&timer);
            for (int j = 0; j < iterations; j++) {
                load_float4_kernel<<<num_blocks, threadperblock>>>(d_input4, d_output4, n / 4);
            }
            float ms = timer_stop(&timer);

            double bw = calculate_bandwidth(n * sizeof(float) * iterations, ms);
            double efficiency = (bw / info->theoretical_bandwidth) * 100.0;

            printf("Blocks=%4d (%dx SMs), threadperblock=%4d: %7.2f GB/s (效率 %.1f%%)\n",
                num_blocks, configs[i], block_size[j], bw, efficiency);
        }
    }
    
    timer_destroy(&timer);
}

// ==================== 主函数 ====================

int main() {
    // 获取设备信息
    DeviceInfo info;
    get_device_info(&info);
    print_device_info(&info);
    
    // 分配内存
    const size_t N = 256 * 1024 * 1024;
    const size_t bytes = N * sizeof(float); // 1GB
    const int iterations = 100;
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemset(d_input, 0x42, bytes));
    
    printf("数据大小: %.2f GB\n", bytes / (1024.0*1024.0*1024.0));
    printf("迭代次数: %d\n", iterations);
    
    // 运行测试
    test_block_config(&info, d_input, d_output, N, iterations);
    // test_l2_cache(&info, d_input, d_output, iterations);
    test_vectorization(&info, d_input, d_output, N, iterations);
    
    // 清理
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    printf("\n========== 测试完成 ==========\n\n");
    
    return 0;
}
