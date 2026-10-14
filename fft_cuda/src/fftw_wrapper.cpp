#include "fftw_wrapper.h"
#include "fft_cpu_cache_optimized.h"

#ifdef ENABLE_FFTW
#include <fftw3.h>
#endif

namespace cuda_fft {

#ifdef ENABLE_FFTW

void fftw_1d_wrapper(Complex* data, size_t n, bool inverse) {
    if (!data || n == 0) return;

    fftw_complex* ptr = reinterpret_cast<fftw_complex*>(data);
    int sign = inverse ? FFTW_BACKWARD : FFTW_FORWARD;

    fftw_plan plan = fftw_plan_dft_1d(static_cast<int>(n), ptr, ptr, sign, FFTW_ESTIMATE);
    if (plan) {
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }

    if (inverse) {
        double scale = 1.0 / static_cast<double>(n);
        for (size_t i = 0; i < n; ++i) {
            data[i] *= scale;
        }
    }
}

void fftw_2d_wrapper(Complex* data, int width, int height, bool inverse) {
    if (!data || width <= 0 || height <= 0) return;

    fftw_complex* ptr = reinterpret_cast<fftw_complex*>(data);
    int sign = inverse ? FFTW_BACKWARD : FFTW_FORWARD;

    fftw_plan plan = fftw_plan_dft_2d(height, width, ptr, ptr, sign, FFTW_ESTIMATE);
    if (plan) {
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }

    if (inverse) {
        double scale = 1.0 / static_cast<double>(width * height);
        for (int i = 0; i < width * height; ++i) {
            data[i] *= scale;
        }
    }
}

#else

void fftw_1d_wrapper(Complex* data, size_t n, bool inverse) {
    fft_1d_cache_optimized(data, n, inverse);
}

void fftw_2d_wrapper(Complex* data, int width, int height, bool inverse) {
    fft_2d_cache_optimized(data, width, height, inverse);
}

#endif

} // namespace cuda_fft
