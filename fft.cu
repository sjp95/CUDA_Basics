// CUDA FFT example using NVIDIA cuFFT.
// Build: nvcc -O2 -o fft_cuda fft_cuda.cu -lcufft

#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
	do {                                                                       \
		cudaError_t error = (call);                                            \
		if (error != cudaSuccess) {                                            \
			std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
						 __FILE__, __LINE__, cudaGetErrorString(error));       \
			std::exit(EXIT_FAILURE);                                           \
		}                                                                       \
	} while (0)

#define CUFFT_CHECK(call)                                                      \
	do {                                                                       \
		cufftResult error = (call);                                             \
		if (error != CUFFT_SUCCESS) {                                          \
			std::fprintf(stderr, "cuFFT error at %s:%d: %d\n",                 \
						 __FILE__, __LINE__, static_cast<int>(error));         \
			std::exit(EXIT_FAILURE);                                           \
		}                                                                       \
	} while (0)

int main() {
	constexpr int N = 1024;
	constexpr float pi = 3.14159265358979323846f;

	std::vector<cufftComplex> host(N);
	for (int i = 0; i < N; ++i) {
		// Example signal: two sine waves.
		float t = static_cast<float>(i) / N;
		host[i].x = std::sin(2.0f * pi * 32.0f * t) +
					0.5f * std::sin(2.0f * pi * 96.0f * t);
		host[i].y = 0.0f;
	}

	cufftComplex* device = nullptr;
	CUDA_CHECK(cudaMalloc(&device, N * sizeof(cufftComplex)));
	CUDA_CHECK(cudaMemcpy(device, host.data(), N * sizeof(cufftComplex),
						  cudaMemcpyHostToDevice));

	cufftHandle plan;
	CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
	CUFFT_CHECK(cufftExecC2C(plan, device, device, CUFFT_FORWARD));
	CUDA_CHECK(cudaDeviceSynchronize());

	CUDA_CHECK(cudaMemcpy(host.data(), device, N * sizeof(cufftComplex),
						  cudaMemcpyDeviceToHost));

	for (int k = 0; k < 10; ++k) {
		float magnitude = std::hypot(host[k].x, host[k].y);
		std::printf("bin %d: real=% .5f imag=% .5f magnitude=% .5f\n",
					k, host[k].x, host[k].y, magnitude);
	}

	CUFFT_CHECK(cufftDestroy(plan));
	CUDA_CHECK(cudaFree(device));
	return 0;
}
