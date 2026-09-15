#include "cufft_wrapper.h"
#include "fft_cuda.cuh"
#include <iostream>

#if defined(__CUDACC__) && defined(ENABLE_CUFFT)
#include <cuda_runtime.h>
#include <cufft.h>
#endif

namespace cuda_fft {

#if defined(__CUDACC__) && defined(ENABLE_CUFFT)

void cufft_1d_wrapper(const std::vector<Complex>& input, std::vector<Complex>& output, bool inverse) {
    int n = static_cast<int>(input.size());
    if (n == 0) return;

    if (!is_cuda_available()) {
        fft_1d_cpu(input, output, inverse);
        return;
    }

    cufftDoubleComplex *d_data = nullptr;
    size_t bytes = n * sizeof(cufftDoubleComplex);

    cudaMalloc(&d_data, bytes);
    std::vector<cufftDoubleComplex> h_data(n);
    for (int i = 0; i < n; ++i) {
        h_data[i].x = input[i].real();
        h_data[i].y = input[i].imag();
    }

    cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, n, CUFFT_Z2Z, 1);

    int direction = inverse ? CUFFT_INVERSE : CUFFT_FORWARD;
    cufftExecZ2Z(plan, d_data, d_data, direction);

    cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    output.resize(n);
    for (int i = 0; i < n; ++i) {
        double scale = inverse ? (1.0 / n) : 1.0;
        output[i] = Complex(h_data[i].x * scale, h_data[i].y * scale);
    }

    cufftDestroy(plan);
    cudaFree(d_data);
}

void cufft_2d_wrapper(const std::vector<Complex>& input, std::vector<Complex>& output, int width, int height, bool inverse) {
    int n = width * height;
    if (n == 0) return;

    if (!is_cuda_available()) {
        fft_2d_cpu(input, output, width, height, inverse);
        return;
    }

    cufftDoubleComplex *d_data = nullptr;
    size_t bytes = n * sizeof(cufftDoubleComplex);

    cudaMalloc(&d_data, bytes);
    std::vector<cufftDoubleComplex> h_data(n);
    for (int i = 0; i < n; ++i) {
        h_data[i].x = input[i].real();
        h_data[i].y = input[i].imag();
    }

    cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan2d(&plan, height, width, CUFFT_Z2Z);

    int direction = inverse ? CUFFT_INVERSE : CUFFT_FORWARD;
    cufftExecZ2Z(plan, d_data, d_data, direction);

    cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost);

    output.resize(n);
    for (int i = 0; i < n; ++i) {
        double scale = inverse ? (1.0 / n) : 1.0;
        output[i] = Complex(h_data[i].x * scale, h_data[i].y * scale);
    }

    cufftDestroy(plan);
    cudaFree(d_data);
}

#else

void cufft_1d_wrapper(const std::vector<Complex>& input, std::vector<Complex>& output, bool inverse) {
    fft_1d_cpu(input, output, inverse);
}

void cufft_2d_wrapper(const std::vector<Complex>& input, std::vector<Complex>& output, int width, int height, bool inverse) {
    fft_2d_cpu(input, output, width, height, inverse);
}

#endif

} // namespace cuda_fft
