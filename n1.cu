/**
 * Demonstrates basic CUDA memory management and kernel execution.
 *
 * The program initializes an array of 64 floating-point values on the host,
 * copies it to device memory, and launches one CUDA block with 64 threads.
 * Each thread squares one input value and stores the result in the output
 * array. The results are copied back to the host, the first five values are
 * printed, and the allocated device memory is released.
 */
#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>

using namespace std;

// 1. CUDA Kernel function
__global__ void squareArray(float *d_out, float *d_in) {
    int idx = threadIdx.x;
    d_out[idx] = d_in[idx] * d_in[idx];
}

int main(int argc, char **argv) {
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // Declare and initialize CPU arrays
    float h_in[ARRAY_SIZE];
    float h_out[ARRAY_SIZE];

    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = float(i);
    }

    // 2. Allocate GPU memory
    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, ARRAY_BYTES);
    cudaMalloc((void**)&d_out, ARRAY_BYTES);

    // 3. Copy host data to GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    // 4. Launch kernel with 1 block of 64 threads
    squareArray<<<1, ARRAY_SIZE>>>(d_out, d_in);

    // 5. Copy result back to host
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // Verify output
    for (int i = 0; i < 5; i++) {
        cout << "h_out[" << i << "] = " << h_out[i] << endl;
    }

    // 6. Free GPU memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
