#ifndef FFT_CPU_CACHE_OPTIMIZED_H
#define FFT_CPU_CACHE_OPTIMIZED_H

#include <complex>
#include <vector>
#include <cstddef>
#include <cstdlib>

namespace cuda_fft {

using Complex = std::complex<double>;

// Cache-aligned aligned memory allocator/deallocator for 64-byte boundaries
Complex* allocate_aligned_complex(size_t count);
void free_aligned_complex(Complex* ptr);

// In-place 1D Cache-Tiled Cooley-Tukey FFT / Stockham decomposition
void fft_1d_cache_optimized(Complex* data, size_t n, bool inverse = false);

// In-place 2D Cache-Tiled FFT
void fft_2d_cache_optimized(Complex* data, int width, int height, bool inverse = false);

} // namespace cuda_fft

#endif // FFT_CPU_CACHE_OPTIMIZED_H
