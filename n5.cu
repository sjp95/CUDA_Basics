#include <iostream>
#include <cuda_runtime.h>
#include "Eigen/Dense"

/**
 * Demonstrates interoperability between Eigen on the host and raw CUDA on
 * the device. An Eigen vector containing the values 0 through 63 is copied to
 * GPU memory, squared by a CUDA kernel, and copied back into another Eigen
 * vector. The first five results are printed before releasing device memory.
 */

// CUDA Kernel using Eigen::Map for element-wise operations
__global__ void squareKernel(const float* d_in, float* d_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_out[idx] = d_in[idx] * d_in[idx];
    }
}

int main() {
    const int N = 64;
    const size_t bytes = N * sizeof(float);

    // 1. CPU Eigen Vector initialization
    Eigen::VectorXf h_in = Eigen::VectorXf::LinSpaced(N, 0.0f, 63.0f);
    Eigen::VectorXf h_out(N);

    // 2. Allocate Device Memory
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // 3. Copy Eigen vector data to Device
    cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice);

    // 4. Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // 5. Copy Device results back to CPU Eigen Vector
    cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    // Print 5 outputs
    std::cout << "--- Without Thrust (Raw CUDA + Eigen) ---" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "h_out[" << i << "] = " << h_out[i] << std::endl;
    }

    // Free GPU memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
