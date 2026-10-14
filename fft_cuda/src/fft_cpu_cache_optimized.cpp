#include "fft_cpu_cache_optimized.h"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <omp.h>

namespace cuda_fft {

static bool is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static size_t bit_reverse_idx(size_t x, int log2n) {
    size_t n = 0;
    for (int i = 0; i < log2n; ++i) {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

FFTCacheOptimizedPlan::FFTCacheOptimizedPlan(size_t n, bool inverse)
    : n_(n), inverse_(inverse), log2n_(0) {
    if (n_ == 0 || !is_power_of_two(n_)) {
        throw std::invalid_argument("FFT size must be a positive power of 2");
    }

    while ((1ULL << log2n_) < n_) {
        log2n_++;
    }

    twiddles_.resize(n_ / 2);
    const double pi = 3.141592653589793238462643383279502884;
    double angle_sign = inverse_ ? 1.0 : -1.0;

    #pragma omp parallel for schedule(static) if(n_ >= 4096)
    for (size_t k = 0; k < n_ / 2; ++k) {
        double theta = angle_sign * 2.0 * pi * static_cast<double>(k) / static_cast<double>(n_);
        twiddles_[k] = Complex(std::cos(theta), std::sin(theta));
    }

    bit_rev_table_.resize(n_);
    #pragma omp parallel for schedule(static) if(n_ >= 4096)
    for (size_t i = 0; i < n_; ++i) {
        bit_rev_table_[i] = bit_reverse_idx(i, log2n_);
    }
}

void FFTCacheOptimizedPlan::execute_bit_reversal(Complex* __restrict__ data) const {
    // Parallel O(N) bit-reversal reordering using pre-calculated index lookup
    #pragma omp parallel for schedule(static) if(n_ >= 4096)
    for (size_t i = 0; i < n_; ++i) {
        size_t rev = bit_rev_table_[i];
        if (rev > i) {
            std::swap(data[i], data[rev]);
        }
    }
}

void FFTCacheOptimizedPlan::execute(Complex* __restrict__ data) const {
    if (n_ <= 1) return;

    // Step 1: Bit-reversal permutation
    execute_bit_reversal(data);

    // Step 2: Parallelized Radix-2 Butterfly Operations
    // Total butterflies per stage is exactly N/2.
    const size_t half_n = n_ >> 1;

    for (size_t len = 2; len <= n_; len <<= 1) {
        size_t half_len = len >> 1;
        size_t twiddle_step = n_ / len;

        // Flatten butterfly index space across all subgroups to maximize OpenMP thread utilization
        #pragma omp parallel for schedule(static) if(n_ >= 2048)
        for (size_t idx = 0; idx < half_n; ++idx) {
            size_t g = idx / half_len;
            size_t j = idx % half_len;
            size_t i = g * len;

            Complex twiddle = twiddles_[j * twiddle_step];
            Complex u = data[i + j];
            Complex v = data[i + j + half_len] * twiddle;

            data[i + j]            = u + v;
            data[i + j + half_len] = u - v;
        }
    }

    // Step 3: Scaling for Inverse FFT
    if (inverse_) {
        double scale = 1.0 / static_cast<double>(n_);
        #pragma omp parallel for schedule(static) if(n_ >= 4096)
        for (size_t i = 0; i < n_; ++i) {
            data[i] *= scale;
        }
    }
}

void fft_1d_cpu_cache_optimized(Complex* __restrict__ data, size_t n, bool inverse) {
    FFTCacheOptimizedPlan plan(n, inverse);
    plan.execute(data);
}

void fft_1d_cpu_cache_optimized(std::vector<Complex>& data, bool inverse) {
    if (data.empty()) return;
    fft_1d_cpu_cache_optimized(data.data(), data.size(), inverse);
}

void fft_2d_cpu_cache_optimized(std::vector<Complex>& data, int width, int height, bool inverse) {
    if (width <= 0 || height <= 0 || data.size() != static_cast<size_t>(width * height)) {
        throw std::invalid_argument("Invalid 2D matrix dimensions");
    }

    size_t w = static_cast<size_t>(width);
    size_t h = static_cast<size_t>(height);

    FFTCacheOptimizedPlan row_plan(w, inverse);
    FFTCacheOptimizedPlan col_plan(h, inverse);

    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < h; ++r) {
        row_plan.execute(data.data() + r * w);
    }

    std::vector<Complex> transposed(w * h);
    const size_t tile = 64;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t r = 0; r < h; r += tile) {
        for (size_t c = 0; c < w; c += tile) {
            size_t r_end = std::min(r + tile, h);
            size_t c_end = std::min(c + tile, w);
            for (size_t i = r; i < r_end; ++i) {
                for (size_t j = c; j < c_end; ++j) {
                    transposed[j * h + i] = data[i * w + j];
                }
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t c = 0; c < w; ++c) {
        col_plan.execute(transposed.data() + c * h);
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t c = 0; c < w; c += tile) {
        for (size_t r = 0; r < h; r += tile) {
            size_t c_end = std::min(c + tile, w);
            size_t r_end = std::min(r + tile, h);
            for (size_t j = c; j < c_end; ++j) {
                for (size_t i = r; i < r_end; ++i) {
                    data[i * w + j] = transposed[j * h + i];
                }
            }
        }
    }
}

} // namespace cuda_fft
