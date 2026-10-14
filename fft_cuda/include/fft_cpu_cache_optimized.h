#ifndef FFT_CPU_CACHE_OPTIMIZED_H
#define FFT_CPU_CACHE_OPTIMIZED_H

#include <complex>
#include <vector>
#include <cstddef>

namespace cuda_fft {

using Complex = std::complex<double>;

/**
 * @brief FFTCacheOptimizedPlan precomputes twiddle factors and bit-reversal lookup tables
 * to enable cache-blocked, allocation-free execution of 1D Cooley-Tukey Radix-2 FFTs.
 */
class FFTCacheOptimizedPlan {
public:
    /**
     * @brief Construct a new FFTCacheOptimizedPlan object
     * @param n Size of the 1D FFT (must be a power of 2)
     * @param inverse True for inverse FFT, false for forward FFT
     */
    FFTCacheOptimizedPlan(size_t n, bool inverse = false);

    /**
     * @brief Executes in-place FFT on 64-byte aligned data array
     * @param data Pointer to array of N std::complex<double> elements (aligned to 64 bytes)
     */
    void execute(Complex* __restrict__ data) const;

    /**
     * @brief Size of the FFT plan
     */
    size_t size() const { return n_; }

    /**
     * @brief Check if inverse transform
     */
    bool is_inverse() const { return inverse_; }

private:
    size_t n_;
    bool inverse_;
    int log2n_;

    // Pre-calculated twiddle factors aligned to cache lines
    alignas(64) std::vector<Complex> twiddles_;

    // Pre-calculated bit reversal index table for zero-overhead bit reversal
    std::vector<size_t> bit_rev_table_;

    // Helper for blocked bit-reversal reordering
    void execute_bit_reversal(Complex* __restrict__ data) const;
};

/**
 * @brief Cache-optimized 1D CPU FFT operating in-place on raw pointer.
 * @param data Raw pointer to complex double array (preferably 64-byte aligned).
 * @param n Size of the FFT input array (must be a power of 2).
 * @param inverse Perform inverse FFT if true.
 */
void fft_1d_cpu_cache_optimized(Complex* __restrict__ data, size_t n, bool inverse = false);

/**
 * @brief Cache-optimized 1D CPU FFT operating in-place on std::vector<Complex>.
 * @param data Vector of std::complex<double> elements.
 * @param inverse Perform inverse FFT if true.
 */
void fft_1d_cpu_cache_optimized(std::vector<Complex>& data, bool inverse = false);

/**
 * @brief Cache-optimized 2D CPU FFT operating in-place on std::vector<Complex>.
 * @param data Vector of std::complex<double> elements (size width * height).
 * @param width Width of 2D matrix (power of 2).
 * @param height Height of 2D matrix (power of 2).
 * @param inverse Perform inverse FFT if true.
 */
void fft_2d_cpu_cache_optimized(std::vector<Complex>& data, int width, int height, bool inverse = false);

} // namespace cuda_fft

#endif // FFT_CPU_CACHE_OPTIMIZED_H
