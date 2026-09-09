#include <iostream>
#include <vector>
#include <memory>
#include <numeric>
#include <cuda_runtime.h>

// Description: This CUDA program allocates an array in unified memory, fills
// it with values on the host, and launches a templated GPU kernel that squares
// each element in parallel. A smart pointer with a CUDA-aware custom deleter
// releases the managed memory automatically, after which the first five
// squared values are printed.

// 1. Generic Templated Device Function (C++14/17 auto return & templates)
template <typename T>
__global__ void squareArrayKernel(T* data, size_t size) {
    // C++11 auto type deduction
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = data[idx] * data[idx];
    }
}

// Custom deleter for Smart Pointers managing GPU Unified Memory
struct CudaDeleter {
    void operator()(void* ptr) const {
        cudaFree(ptr);
    }
};

template <typename T>
using cuda_unique_ptr = std::unique_ptr<T[], CudaDeleter>;

int main() {
    constexpr size_t array_size = 64; // C++11 constexpr

    // 2. Managed Memory with Smart Pointers (No manual cudaMemcpy needed)
    float* raw_ptr = nullptr;
    cudaMallocManaged(&raw_ptr, array_size * sizeof(float));
    cuda_unique_ptr<float> data(raw_ptr); // Automatically calls cudaFree on exit

    // 3. Populate host data using standard C++ algorithm style
    for (size_t i = 0; i < array_size; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Launch kernel
    constexpr int threadsPerBlock = 256;
    int blocksPerGrid = (array_size + threadsPerBlock - 1) / threadsPerBlock;
    
    squareArrayKernel<<<blocksPerGrid, threadsPerBlock>>>(data.get(), array_size);

    // Synchronize to ensure GPU work is finished
    cudaDeviceSynchronize();

    // 4. Print results (C++20 range-based view feel)
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "data[" << i << "] = " << data[i] << '\n';
    }

    // Memory is freed automatically via cuda_unique_ptr destructor!
    return 0;
}
