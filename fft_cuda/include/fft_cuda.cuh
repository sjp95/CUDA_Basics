#ifndef FFT_CUDA_CUH
#define FFT_CUDA_CUH

#include "fft_cpu.h"
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuComplex.h>
#endif

namespace cuda_fft {

// CUDA availability check
bool is_cuda_available();

// 1D Custom CUDA FFT
void fft_1d_cuda(const std::vector<Complex>& input, std::vector<Complex>& output, bool inverse = false);

// 2D Custom CUDA FFT
void fft_2d_cuda(const std::vector<Complex>& input, std::vector<Complex>& output, int width, int height, bool inverse = false);

} // namespace cuda_fft

#endif // FFT_CUDA_CUH
