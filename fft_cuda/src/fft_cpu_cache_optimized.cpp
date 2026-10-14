#include "fft_cpu_cache_optimized.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstdlib>

#if defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
#include <stdlib.h>
#endif

namespace cuda_fft {

Complex* allocate_aligned_complex(size_t count) {
    size_t bytes = count * sizeof(Complex);
    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(bytes, 64);
#elif defined(__STDC_LIB_EXT1__) || defined(__posix_memalign__) || defined(_POSIX_C_SOURCE)
    if (posix_memalign(&ptr, 64, bytes) != 0) {
        ptr = nullptr;
    }
#else
    ptr = std::aligned_alloc(64, (bytes + 63) & ~63ULL);
#endif
    if (!ptr) {
        ptr = std::malloc(bytes);
    }
    return static_cast<Complex*>(ptr);
}

void free_aligned_complex(Complex* ptr) {
    if (!ptr) return;
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

static size_t bit_reverse(size_t x, int log2n) {
    size_t n = 0;
    for (int i = 0; i < log2n; ++i) {
        n = (n << 1) | (x & 1);
        x >>= 1;
    }
    return n;
}

void fft_1d_cache_optimized(Complex* data, size_t n, bool inverse) {
    if (n == 0 || !data) return;
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("FFT size must be a power of 2");
    }

    int log2n = 0;
    while ((1ULL << log2n) < n) log2n++;

    // In-place bit-reversal permutation using aligned register access
    for (size_t i = 0; i < n; ++i) {
        size_t rev = bit_reverse(i, log2n);
        if (i < rev) {
            std::swap(data[i], data[rev]);
        }
    }

    const double pi = 3.14159265358979323846;
    double angle_sign = inverse ? 1.0 : -1.0;

    // Cache blocking factor (tile size fit for L1/L2 cache lines)
    constexpr size_t BLOCK_SIZE = 64;

    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = angle_sign * 2.0 * pi / len;
        Complex wlen(std::cos(ang), std::sin(ang));
        size_t half_len = len >> 1;

        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (size_t j = 0; j < half_len; j += BLOCK_SIZE) {
                size_t j_end = std::min(j + BLOCK_SIZE, half_len);
                #pragma omp simd
                for (size_t k = j; k < j_end; ++k) {
                    Complex u = data[i + k];
                    Complex v = data[i + k + half_len] * w;
                    data[i + k] = u + v;
                    data[i + k + half_len] = u - v;
                    w *= wlen;
                }
            }
        }
    }

    if (inverse) {
        double scale = 1.0 / static_cast<double>(n);
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            data[i] *= scale;
        }
    }
}

void fft_2d_cache_optimized(Complex* data, int width, int height, bool inverse) {
    if (!data || width <= 0 || height <= 0) return;

    // Row-wise transforms
    for (int r = 0; r < height; ++r) {
        fft_1d_cache_optimized(data + r * width, width, inverse);
    }

    // Cache-tiled matrix transposition for column locality
    constexpr int TILE_DIM = 32;
    Complex* temp = allocate_aligned_complex(width * height);

    for (int r0 = 0; r0 < height; r0 += TILE_DIM) {
        for (int c0 = 0; c0 < width; c0 += TILE_DIM) {
            int r_end = std::min(r0 + TILE_DIM, height);
            int c_end = std::min(c0 + TILE_DIM, width);
            for (int r = r0; r < r_end; ++r) {
                for (int c = c0; c < c_end; ++c) {
                    temp[c * height + r] = data[r * width + c];
                }
            }
        }
    }

    // Column-wise transforms (now row-contiguous in transposed temp)
    for (int c = 0; c < width; ++c) {
        fft_1d_cache_optimized(temp + c * height, height, inverse);
    }

    // Transpose back into data
    for (int c0 = 0; c0 < width; c0 += TILE_DIM) {
        for (int r0 = 0; r0 < height; r0 += TILE_DIM) {
            int c_end = std::min(c0 + TILE_DIM, width);
            int r_end = std::min(r0 + TILE_DIM, height);
            for (int c = c0; c < c_end; ++c) {
                for (int r = r0; r < r_end; ++r) {
                    data[r * width + c] = temp[c * height + r];
                }
            }
        }
    }

    free_aligned_complex(temp);
}

} // namespace cuda_fft
