#ifndef FFT_CPU_H
#ifndef FFT_TYPES_H
#define FFT_TYPES_H

#include <complex>
#include <vector>
#include <cmath>
#include <cstddef>

namespace cuda_fft {

using Complex = std::complex<double>;

// 1D CPU FFT (Cooley-Tukey Radix-2 / Stockham)
void fft_1d_cpu(const std::vector<Complex>& input, std::vector<Complex>& output, bool inverse = false);

// 2D CPU FFT (Row-column decomposition)
void fft_2d_cpu(const std::vector<Complex>& input, std::vector<Complex>& output, int width, int height, bool inverse = false);

// FFT shift (shift zero-frequency component to center of spectrum)
void fftshift_2d(std::vector<Complex>& data, int width, int height);

} // namespace cuda_fft

#endif // FFT_TYPES_H
#endif // FFT_CPU_H
