# CUDA Basics with C++, cuBLAS, and Thrust

This README introduces the basics of CUDA programming in C++, including GPU kernels, memory management, vector addition, matrix operations, cuBLAS, and Thrust.

## 1. What is CUDA?

CUDA is NVIDIA's platform for running parallel code on a GPU. A CUDA program commonly contains:

- **Host code**: C++ code that runs on the CPU.
- **Device code**: Kernels that run on the GPU.
- **Threads, blocks, and grids**: The hierarchy used to organize GPU work.

CUDA source files usually use the `.cu` extension and can be compiled with `nvcc`.

```bash
nvcc program.cu -o program
./program
```

## 2. A Basic CUDA Kernel

The `__global__` qualifier defines a kernel callable from the CPU and executed on the GPU.

```cpp
#include <cstdio>

__global__ void helloFromGPU() {
	printf("Hello from GPU thread %d\n", threadIdx.x);
}

int main() {
	helloFromGPU<<<1, 4>>>();
	cudaDeviceSynchronize();
	return 0;
}
```

`<<<1, 4>>>` launches one block containing four threads. `cudaDeviceSynchronize()` waits for the GPU to finish.

## 3. GPU Memory

Typical memory operations are:

```cpp
float* deviceData = nullptr;
cudaMalloc(&deviceData, 100 * sizeof(float));
cudaMemcpy(deviceData, hostData, 100 * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(hostData, deviceData, 100 * sizeof(float), cudaMemcpyDeviceToHost);
cudaFree(deviceData);
```

Always check CUDA errors in production code:

```cpp
#define CUDA_CHECK(call) do {                                      \
	cudaError_t error = (call);                                   \
	if (error != cudaSuccess) {                                   \
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error)); \
		return 1;                                                  \
	}                                                              \
} while (0)
```

## 4. Vector Addition

Each GPU thread can add one pair of numbers:

```cpp
__global__ void addVectors(const float* a, const float* b, float* c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) c[i] = a[i] + b[i];
}

// Launch with:
int threads = 256;
int blocks = (n + threads - 1) / threads;
addVectors<<<blocks, threads>>>(d_a, d_b, d_c, n);
```

The bounds check prevents threads beyond the array size from accessing invalid memory.

## 5. Matrix Addition

For matrices stored in row-major order, flatten the row and column into one index:

```cpp
__global__ void addMatrices(const float* A, const float* B, float* C,
							int rows, int columns) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int size = rows * columns;
	if (index < size) C[index] = A[index] + B[index];
}
```

For a matrix element, the row-major index is `row * columns + column`.

## 6. Matrix Multiplication

A simple kernel computes one output element per thread:

```cpp
__global__ void matrixMultiply(const float* A, const float* B, float* C,
							   int M, int N, int K) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < N) {
		float sum = 0.0f;
		for (int k = 0; k < K; ++k)
			sum += A[row * K + k] * B[k * N + col];
		C[row * N + col] = sum;
	}
}

dim3 threads2D(16, 16);
dim3 blocks2D((N + 15) / 16, (M + 15) / 16);
matrixMultiply<<<blocks2D, threads2D>>>(d_A, d_B, d_C, M, N, K);
```

This basic version is useful for learning. Optimized multiplication normally uses shared memory or a library such as cuBLAS.

## 7. cuBLAS

cuBLAS is NVIDIA's optimized BLAS library for vector and matrix operations.

```cpp
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

const float alpha = 1.0f;
const float beta = 0.0f;

// Column-major: C = alpha * A * B + beta * C
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			M, N, K, &alpha,
			d_A, M, d_B, K, &beta, d_C, M);

cublasDestroy(handle);
```

Important cuBLAS functions include:

- `cublasSaxpy`: `y = alpha * x + y`
- `cublasSdot`: vector dot product
- `cublasSnrm2`: vector norm
- `cublasSgemv`: matrix-vector multiplication
- `cublasSgemm`: matrix-matrix multiplication

cuBLAS uses column-major matrices by default, unlike typical C++ row-major arrays. Account for this when passing dimensions and leading dimensions.

## 8. Thrust

Thrust provides high-level parallel algorithms and containers.

```cpp
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

thrust::device_vector<float> a{1, 2, 3};
thrust::device_vector<float> b{4, 5, 6};
thrust::device_vector<float> c(3);

thrust::transform(a.begin(), a.end(), b.begin(), c.begin(),
				  thrust::plus<float>());
```

Useful Thrust algorithms include `sort`, `reduce`, `transform`, `copy`, `fill`, and `transform_reduce`.

## 9. Good Practices

- Synchronize only when necessary; synchronization can reduce performance.
- Use pinned host memory for faster transfers when appropriate.
- Minimize CPU-to-GPU data transfers.
- Choose enough threads to keep the GPU occupied.
- Use shared memory for data reused by threads in the same block.
- Profile with NVIDIA Nsight Systems or Nsight Compute.
- Check kernel launch and library errors.
- Prefer cuBLAS and other CUDA libraries for optimized operations.

## 10. Practice Exercises

1. Write a kernel to multiply every vector element by a scalar.
2. Implement vector subtraction and element-wise multiplication.
3. Add two matrices using 2D thread blocks.
4. Compare your matrix multiplication kernel with `cublasSgemm`.
5. Use Thrust to sort a device vector and calculate its sum.

## Requirements

- NVIDIA GPU with a compatible driver
- CUDA Toolkit
- C++ compiler
- Optional: cuBLAS and Thrust (included with the CUDA Toolkit)