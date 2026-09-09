/**
 * @file n4.cu
 * @brief Compares CPU and GPU performance for squaring ten million values.
 *
 * The program initializes a vector of floating-point values and squares each
 * element twice: first with the C++ standard library on the CPU, then with
 * Thrust on the GPU. It measures only the transformation time for each
 * implementation, synchronizing the CUDA device before stopping the GPU
 * timer so that kernel execution is included. Finally, it prints five GPU
 * results and reports both execution times and the calculated GPU speedup.
 *
 * The initial GPU vector population and memory transfers are intentionally
 * excluded from the GPU benchmark, allowing the comparison to focus on the
 * square operation itself.
 */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>

// Functor for GPU transformation
struct Square {
    __host__ __device__ float operator()(float x) const { return x * x; }
};

int main() {
    // 10 million elements to clearly see GPU vs CPU performance difference
    const size_t N = 10'000'000;

    std::cout << "Benchmarking 10 Million Elements (Square Operation)...\n\n";

    // ==========================================
    // 1. CPU EXECUTION
    // ==========================================
    std::vector<float> cpu_data(N);
    for (size_t i = 0; i < N; ++i) {
        cpu_data[i] = static_cast<float>(i);
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();

    std::transform(cpu_data.begin(), cpu_data.end(), cpu_data.begin(), 
                   [](float x) { return x * x; });

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = end_cpu - start_cpu;

    // ==========================================
    // 2. GPU EXECUTION (Thrust)
    // ==========================================
    thrust::device_vector<float> d_vec(N);
    for (size_t i = 0; i < N; ++i) {
        d_vec[i] = static_cast<float>(i);
    }

    // Wait for initial host-to-device transfers to finish before timing
    cudaDeviceSynchronize();

    auto start_gpu = std::chrono::high_resolution_clock::now();

    // Perform operation on GPU
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), Square());

    // Wait for GPU kernel execution to finish
    cudaDeviceSynchronize();

    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = end_gpu - start_gpu;

    // ==========================================
    // 3. PRINT 5 OUTPUT SAMPLES
    // ==========================================
    std::cout << "--- 5 Sample Outputs from GPU ---" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "d_vec[" << i << "] = " << d_vec[i] << std::endl;
    }

    // ==========================================
    // 4. PRINT BENCHMARK RESULTS
    // ==========================================
    std::cout << "\n--- Execution Time Comparison ---" << std::endl;
    std::cout << "CPU Time: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << "GPU Time: " << gpu_duration.count() << " ms" << std::endl;
    std::cout << "Speedup:  " << cpu_duration.count() / gpu_duration.count() << "x faster on GPU" << std::endl;

    return 0;
}
