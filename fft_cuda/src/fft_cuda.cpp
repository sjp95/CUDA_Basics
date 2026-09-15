#include "fft_cuda.cuh"

namespace cuda_fft {

#ifndef __CUDACC__

bool is_cuda_available() {
    return false;
}

void fft_1d_cuda(const std::vector<Complex>& input, std::vector<Complex>& output, bool inverse) {
    fft_1d_cpu(input, output, inverse);
}

void fft_2d_cuda(const std::vector<Complex>& input, std::vector<Complex>& output, int width, int height, bool inverse) {
    fft_2d_cpu(input, output, width, height, inverse);
}

#endif

} // namespace cuda_fft
