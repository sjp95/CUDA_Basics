#include "cufft_wrapper.h"

namespace cuda_fft {

#ifndef __CUDACC__

void cufft_1d_wrapper(const std::vector<Complex>& input, std::vector<Complex>& output, bool inverse) {
    fft_1d_cpu(input, output, inverse);
}

void cufft_2d_wrapper(const std::vector<Complex>& input, std::vector<Complex>& output, int width, int height, bool inverse) {
    fft_2d_cpu(input, output, width, height, inverse);
}

#endif

} // namespace cuda_fft
