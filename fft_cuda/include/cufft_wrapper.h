#ifndef CUFFT_WRAPPER_H
#define CUFFT_WRAPPER_H

#include "fft_cpu.h"
#include <vector>

namespace cuda_fft {

// cuFFT 1D wrapper (falls back to CPU if cuFFT/CUDA unavailable)
void cufft_1d_wrapper(const std::vector<Complex>& input, std::vector<Complex>& output, bool inverse = false);

// cuFFT 2D wrapper
void cufft_2d_wrapper(const std::vector<Complex>& input, std::vector<Complex>& output, int width, int height, bool inverse = false);

} // namespace cuda_fft

#endif // CUFFT_WRAPPER_H
