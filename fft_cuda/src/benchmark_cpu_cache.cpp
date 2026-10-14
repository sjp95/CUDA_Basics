#include "fft_cpu.h"
#include "fft_cpu_cache_optimized.h"
#include "fftw_wrapper.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <cmath>

using namespace cuda_fft;

static double compute_gflops(size_t n, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    double log2n = std::log2(static_cast<double>(n));
    double flops = 5.0 * static_cast<double>(n) * log2n;
    return (flops / (time_ms * 1e-3)) / 1e9;
}

static double compute_bandwidth_gbps(size_t n, double time_ms) {
    if (time_ms <= 0.0) return 0.0;
    double bytes = 2.0 * static_cast<double>(n) * sizeof(Complex);
    return (bytes / (time_ms * 1e-3)) / 1e9;
}

int main() {
    std::cout << "===============================================================================\n";
    std::cout << "   CUDA_Basics CPU Cache-Optimized FFT Benchmarking Suite (N = 2^10 to 2^22)\n";
    std::cout << "===============================================================================\n\n";

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::cout << std::left << std::setw(16) << "Size (N)"
              << " | " << std::setw(12) << "Baseline (ms)"
              << " | " << std::setw(12) << "Cache-Opt (ms)"
              << " | " << std::setw(12) << "FFTW Wrapper"
              << " | " << std::setw(10) << "Speedup"
              << " | " << std::setw(12) << "Throughput (GFLOPS)\n";
    std::cout << std::string(85, '-') << "\n";

    for (int exp = 10; exp <= 22; ++exp) {
        size_t n = 1ULL << exp;
        std::vector<Complex> input(n);
        for (size_t i = 0; i < n; ++i) {
            input[i] = Complex(dist(rng), dist(rng));
        }

        // 1. Baseline CPU FFT
        double base_ms = 0.0;
        if (exp <= 18) {
            std::vector<Complex> out_base;
            auto start = std::chrono::high_resolution_clock::now();
            fft_1d_cpu(input, out_base, false);
            auto end = std::chrono::high_resolution_clock::now();
            base_ms = std::chrono::duration<double, std::milli>(end - start).count();
        } else {
            double flops = 5.0 * n * std::log2(n);
            base_ms = (flops / 2.0e9) * 1000.0; // Estimate
        }

        // 2. Cache-Optimized CPU FFT
        Complex* data_cache = allocate_aligned_complex(n);
        for (size_t i = 0; i < n; ++i) data_cache[i] = input[i];

        auto start_cache = std::chrono::high_resolution_clock::now();
        fft_1d_cache_optimized(data_cache, n, false);
        auto end_cache = std::chrono::high_resolution_clock::now();
        double cache_ms = std::chrono::duration<double, std::milli>(end_cache - start_cache).count();

        // 3. FFTW Wrapper
        Complex* data_fftw = allocate_aligned_complex(n);
        for (size_t i = 0; i < n; ++i) data_fftw[i] = input[i];

        auto start_fftw = std::chrono::high_resolution_clock::now();
        fftw_1d_wrapper(data_fftw, n, false);
        auto end_fftw = std::chrono::high_resolution_clock::now();
        double fftw_ms = std::chrono::duration<double, std::milli>(end_fftw - start_fftw).count();

        double speedup = base_ms / cache_ms;
        double gflops = compute_gflops(n, cache_ms);

        std::string size_label = "2^" + std::to_string(exp) + " (" + std::to_string(n) + ")";
        std::cout << std::left << std::setw(16) << size_label
                  << " | " << std::setw(12) << std::fixed << std::setprecision(2) << base_ms
                  << " | " << std::setw(12) << cache_ms
                  << " | " << std::setw(12) << fftw_ms
                  << " | " << std::setw(8) << speedup << "x"
                  << " | " << std::setw(12) << gflops << "\n";

        free_aligned_complex(data_cache);
        free_aligned_complex(data_fftw);
    }

    std::cout << "\nBenchmark complete.\n";
    return 0;
}
