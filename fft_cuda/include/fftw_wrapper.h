#ifndef FFTW_WRAPPER_H
#define FFTW_WRAPPER_H

#include <complex>
#include <vector>
#include <cstddef>

namespace cuda_fft {

using Complex = std::complex<double>;

// FFTW3 Wrapper 1D In-Place Transform
void fftw_1d_wrapper(Complex* data, size_t n, bool inverse = false);

// FFTW3 Wrapper 2D In-Place Transform
void fftw_2d_wrapper(Complex* data, int width, int height, bool inverse = false);

} // namespace cuda_fft

#endif // FFTW_WRAPPER_H
