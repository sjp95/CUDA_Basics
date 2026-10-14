#include "fft_cpu.h"
#include "fft_cpu_cache_optimized.h"
#include "fftw_wrapper.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <cmath>
#include <algorithm>

using namespace cuda_fft;

struct CacheBenchmarkResult {
    int exponent;               // Power of 2 (10 to 22)
    size_t size_n;              // N = 2^exponent

    double baseline_time_us;    // Naive CPU FFT time (microseconds)
    double baseline_gflops;     // Naive CPU FFT GFLOPS
    double baseline_gbps;       // Naive CPU FFT memory bandwidth (GB/s)

    double cache_opt_time_us;   // Cache-optimized FFT time (microseconds)
    double cache_opt_gflops;    // Cache-optimized FFT GFLOPS
    double cache_opt_gbps;      // Cache-optimized FFT memory bandwidth (GB/s)

    double fftw_time_us;        // FFTW3 wrapper time (microseconds)
    double fftw_gflops;         // FFTW3 GFLOPS
    double fftw_gbps;           // FFTW3 memory bandwidth (GB/s)

    double speedup_vs_baseline; // Speedup of cache-optimized over naive
    double speedup_vs_fftw;     // Ratio of FFTW3 vs cache-optimized
};

static double compute_gflops(size_t n, double time_us) {
    if (time_us <= 0.0) return 0.0;
    double log2n = std::log2(static_cast<double>(n));
    double flops = 5.0 * static_cast<double>(n) * log2n;
    double time_s = time_us * 1e-6;
    return (flops / time_s) / 1e9;
}

static double compute_gbps(size_t n, double time_us) {
    if (time_us <= 0.0) return 0.0;
    // Each element is read and written in-place: 2 * N * sizeof(Complex)
    double bytes = 2.0 * static_cast<double>(n) * sizeof(Complex);
    double time_s = time_us * 1e-6;
    return (bytes / time_s) / 1e9;
}

int main(int argc, char** argv) {
    std::cout << "========================================================================================\n";
    std::cout << "          CUDA_Basics: CPU Cache-Optimized FFT vs Naive FFT vs FFTW3 Benchmark Tool\n";
    std::cout << "========================================================================================\n\n";

    std::vector<CacheBenchmarkResult> results;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    const int MIN_EXP = 10; // N = 2^10 = 1,024
    const int MAX_EXP = 22; // N = 2^22 = 4,194,304

    std::cout << std::left
              << std::setw(8)  << "2^N"
              << std::setw(12) << "Size (N)"
              << std::setw(15) << "Naive (us)"
              << std::setw(15) << "Cache-Opt(us)"
              << std::setw(15) << "FFTW3 (us)"
              << std::setw(14) << "Opt GFLOPS"
              << std::setw(14) << "FFTW GFLOPS"
              << std::setw(12) << "Speedup(x)"
              << "\n";
    std::cout << std::string(105, '-') << "\n";

    for (int exp = MIN_EXP; exp <= MAX_EXP; ++exp) {
        size_t n = 1ULL << exp;

        // Generate input data
        std::vector<Complex> input_orig(n);
        for (size_t i = 0; i < n; ++i) {
            input_orig[i] = Complex(dist(rng), dist(rng));
        }

        // 1. Real execution measurement for Naive Baseline FFT (fft_cpu.cpp) across all sizes
        int num_runs_naive = (exp <= 14) ? 5 : ((exp <= 18) ? 2 : 1);
        std::vector<Complex> in_vec = input_orig;
        std::vector<Complex> out_vec;

        // Warmup
        fft_1d_cpu(in_vec, out_vec, false);

        auto start_naive = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < num_runs_naive; ++r) {
            fft_1d_cpu(in_vec, out_vec, false);
        }
        auto end_naive = std::chrono::high_resolution_clock::now();
        double baseline_time_us = std::chrono::duration<double, std::micro>(end_naive - start_naive).count() / num_runs_naive;

        // 2. Cache-Optimized FFT real execution
        int num_runs_opt = (exp <= 16) ? 10 : ((exp <= 20) ? 3 : 1);
        FFTCacheOptimizedPlan cache_plan(n, false);

        std::vector<Complex> opt_buffer = input_orig;
        // Warmup
        cache_plan.execute(opt_buffer.data());

        auto start_opt = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < num_runs_opt; ++r) {
            opt_buffer = input_orig;
            cache_plan.execute(opt_buffer.data());
        }
        auto end_opt = std::chrono::high_resolution_clock::now();
        double cache_opt_time_us = std::chrono::duration<double, std::micro>(end_opt - start_opt).count() / num_runs_opt;

        // 3. FFTW3 Production Baseline real execution
        FFTW3PlanWrapper fftw_plan(n, false, FFTW_ESTIMATE);

        Complex* fftw_buf = allocate_aligned_complex(n);
        std::copy(input_orig.begin(), input_orig.end(), fftw_buf);
        // Warmup
        fftw_plan.execute(fftw_buf);

        auto start_fftw = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < num_runs_opt; ++r) {
            std::copy(input_orig.begin(), input_orig.end(), fftw_buf);
            fftw_plan.execute(fftw_buf);
        }
        auto end_fftw = std::chrono::high_resolution_clock::now();
        double fftw_time_us = std::chrono::duration<double, std::micro>(end_fftw - start_fftw).count() / num_runs_opt;

        free_aligned_complex(fftw_buf);

        // Metrics calculation
        CacheBenchmarkResult res;
        res.exponent = exp;
        res.size_n = n;

        res.baseline_time_us = baseline_time_us;
        res.baseline_gflops  = compute_gflops(n, baseline_time_us);
        res.baseline_gbps    = compute_gbps(n, baseline_time_us);

        res.cache_opt_time_us = cache_opt_time_us;
        res.cache_opt_gflops  = compute_gflops(n, cache_opt_time_us);
        res.cache_opt_gbps    = compute_gbps(n, cache_opt_time_us);

        res.fftw_time_us = fftw_time_us;
        res.fftw_gflops  = compute_gflops(n, fftw_time_us);
        res.fftw_gbps    = compute_gbps(n, fftw_time_us);

        res.speedup_vs_baseline = baseline_time_us / cache_opt_time_us;
        res.speedup_vs_fftw     = cache_opt_time_us / fftw_time_us;

        results.push_back(res);

        std::cout << std::left
                  << "2^" << std::setw(6) << exp
                  << std::setw(12) << n
                  << std::fixed << std::setprecision(1)
                  << std::setw(15) << baseline_time_us
                  << std::setw(15) << cache_opt_time_us
                  << std::setw(15) << fftw_time_us
                  << std::setprecision(2)
                  << std::setw(14) << res.cache_opt_gflops
                  << std::setw(14) << res.fftw_gflops
                  << std::setw(12) << res.speedup_vs_baseline << "x\n";
    }

    std::cout << std::string(105, '-') << "\n";

    // Export benchmark results to JSON file
    std::ofstream json_file("benchmark_cpu_cache_results.json");
    if (json_file.is_open()) {
        json_file << "[\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            json_file << "  {\n"
                      << "    \"exponent\": " << r.exponent << ",\n"
                      << "    \"size_n\": " << r.size_n << ",\n"
                      << "    \"baseline_time_us\": " << r.baseline_time_us << ",\n"
                      << "    \"baseline_gflops\": " << r.baseline_gflops << ",\n"
                      << "    \"baseline_gbps\": " << r.baseline_gbps << ",\n"
                      << "    \"cache_opt_time_us\": " << r.cache_opt_time_us << ",\n"
                      << "    \"cache_opt_gflops\": " << r.cache_opt_gflops << ",\n"
                      << "    \"cache_opt_gbps\": " << r.cache_opt_gbps << ",\n"
                      << "    \"fftw_time_us\": " << r.fftw_time_us << ",\n"
                      << "    \"fftw_gflops\": " << r.fftw_gflops << ",\n"
                      << "    \"fftw_gbps\": " << r.fftw_gbps << ",\n"
                      << "    \"speedup_vs_baseline\": " << r.speedup_vs_baseline << ",\n"
                      << "    \"speedup_vs_fftw\": " << r.speedup_vs_fftw << "\n"
                      << "  }" << (i + 1 < results.size() ? "," : "") << "\n";
        }
        json_file << "]\n";
        std::cout << "\n[Info] Benchmark results exported successfully to benchmark_cpu_cache_results.json\n";
    }

    return 0;
}
