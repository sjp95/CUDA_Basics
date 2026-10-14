#include "fftw_wrapper.h"
#include <stdexcept>
#include <cstring>
#include <omp.h>

namespace cuda_fft {

Complex* allocate_aligned_complex(size_t n) {
    void* ptr = fftw_malloc(sizeof(fftw_complex) * n);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return reinterpret_cast<Complex*>(ptr);
}

void free_aligned_complex(Complex* ptr) {
    if (ptr) {
        fftw_free(reinterpret_cast<fftw_complex*>(ptr));
    }
}

FFTW3PlanWrapper::FFTW3PlanWrapper(size_t n, bool inverse, unsigned flags)
    : n_(n), inverse_(inverse), plan_(nullptr), sample_buffer_(nullptr) {
    if (n_ == 0) {
        throw std::invalid_argument("FFT size must be greater than zero");
    }

    // Allocate temporary aligned buffer to let FFTW measure optimal kernel plans
    sample_buffer_ = reinterpret_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * n_));
    if (!sample_buffer_) {
        throw std::bad_alloc();
    }

    int direction = inverse_ ? FFTW_BACKWARD : FFTW_FORWARD;

    // Create 1D in-place DFT plan with vendor SIMD & HW cache optimizations
    plan_ = fftw_plan_dft_1d(
        static_cast<int>(n_),
        sample_buffer_,
        sample_buffer_,
        direction,
        flags
    );

    if (!plan_) {
        fftw_free(sample_buffer_);
        throw std::runtime_error("Failed to create FFTW3 plan");
    }
}

FFTW3PlanWrapper::~FFTW3PlanWrapper() {
    if (plan_) {
        fftw_destroy_plan(plan_);
        plan_ = nullptr;
    }
    if (sample_buffer_) {
        fftw_free(sample_buffer_);
        sample_buffer_ = nullptr;
    }
}

void FFTW3PlanWrapper::execute(Complex* data) const {
    fftw_complex* fftw_data = reinterpret_cast<fftw_complex*>(data);
    fftw_execute_dft(plan_, fftw_data, fftw_data);

    // Apply 1/N scaling for inverse FFT
    if (inverse_) {
        double scale = 1.0 / static_cast<double>(n_);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_; ++i) {
            data[i] *= scale;
        }
    }
}

void FFTW3PlanWrapper::execute(fftw_complex* data) const {
    fftw_execute_dft(plan_, data, data);

    if (inverse_) {
        double scale = 1.0 / static_cast<double>(n_);
        Complex* comp_ptr = reinterpret_cast<Complex*>(data);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_; ++i) {
            comp_ptr[i] *= scale;
        }
    }
}

void fft_1d_fftw(Complex* data, size_t n, bool inverse, unsigned planner_flags) {
    FFTW3PlanWrapper wrapper(n, inverse, planner_flags);
    wrapper.execute(data);
}

void fft_1d_fftw(std::vector<Complex>& data, bool inverse, unsigned planner_flags) {
    if (data.empty()) return;
    fft_1d_fftw(data.data(), data.size(), inverse, planner_flags);
}

void fft_2d_fftw(std::vector<Complex>& data, int width, int height, bool inverse, unsigned planner_flags) {
    if (width <= 0 || height <= 0 || data.size() != static_cast<size_t>(width * height)) {
        throw std::invalid_argument("Invalid 2D matrix dimensions for FFTW");
    }

    size_t n = data.size();
    fftw_complex* fftw_data = reinterpret_cast<fftw_complex*>(data.data());

    int direction = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
    fftw_plan plan = fftw_plan_dft_2d(
        height, width,
        fftw_data, fftw_data,
        direction,
        planner_flags
    );

    if (!plan) {
        throw std::runtime_error("Failed to create 2D FFTW plan");
    }

    fftw_execute(plan);
    fftw_destroy_plan(plan);

    if (inverse) {
        double scale = 1.0 / static_cast<double>(n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            data[i] *= scale;
        }
    }
}

} // namespace cuda_fft
