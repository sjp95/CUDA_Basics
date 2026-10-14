#include "fft_cpu.h"
#include "fft_cpu_cache_optimized.h"
#include "fftw_wrapper.h"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cassert>

using namespace cuda_fft;

void test_1d_cache_optimized() {
    size_t n = 1024;
    std::vector<Complex> data_naive(n);
    std::vector<Complex> data_opt(n);
    std::vector<Complex> data_fftw(n);

    for (size_t i = 0; i < n; ++i) {
        double val = std::sin(2.0 * 3.141592653589793 * i / 128.0);
        data_naive[i] = Complex(val, 0.0);
        data_opt[i]   = Complex(val, 0.0);
        data_fftw[i]  = Complex(val, 0.0);
    }

    std::vector<Complex> out_naive;
    fft_1d_cpu(data_naive, out_naive, false);
    fft_1d_cpu_cache_optimized(data_opt, false);
    fft_1d_fftw(data_fftw, false);

    for (size_t i = 0; i < n; ++i) {
        double err_opt = std::abs(out_naive[i] - data_opt[i]);
        double err_fftw = std::abs(out_naive[i] - data_fftw[i]);
        if (err_opt > 1e-6 || err_fftw > 1e-6) {
            std::cerr << "Mismatch at index " << i << ": naive=" << out_naive[i]
                      << ", opt=" << data_opt[i] << ", fftw=" << data_fftw[i] << "\n";
            assert(false);
        }
    }
    std::cout << "[PASS] 1D Cache-Optimized & FFTW3 match Naive Baseline!\n";
}

int main() {
    test_1d_cache_optimized();
    return 0;
}
