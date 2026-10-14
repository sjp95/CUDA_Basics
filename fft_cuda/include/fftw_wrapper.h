#ifndef FFTW_WRAPPER_H
#define FFTW_WRAPPER_H

#include <complex>
#include <vector>
#include <cstddef>
#include <fftw3.h>

namespace cuda_fft {

using Complex = std::complex<double>;

/**
 * @brief FFTW3 Plan Wrapper class managing FFTW plan lifecycle, aligned buffer allocation,
 * and execution with FFTW_MEASURE / FFTW_PATIENT planner flags.
 */
class FFTW3PlanWrapper {
public:
    /**
     * @brief Construct an FFTW3 1D Plan Wrapper
     * @param n FFT length (number of elements)
     * @param inverse False for forward transform, true for inverse
     * @param flags FFTW planner flags (default: FFTW_MEASURE)
     */
    FFTW3PlanWrapper(size_t n, bool inverse = false, unsigned flags = FFTW_MEASURE);

    ~FFTW3PlanWrapper();

    // Disable copy constructors to prevent double freeing of FFTW plans
    FFTW3PlanWrapper(const FFTW3PlanWrapper&) = delete;
    FFTW3PlanWrapper& operator=(const FFTW3PlanWrapper&) = delete;

    /**
     * @brief Execute in-place 1D FFT on 64-byte aligned FFTW buffer or std::complex<double>* data
     * @param data Raw pointer to complex data array
     */
    void execute(Complex* data) const;

    /**
     * @brief Execute in-place 1D FFT on fftw_complex raw pointer
     * @param data Raw pointer to fftw_complex array
     */
    void execute(fftw_complex* data) const;

    size_t size() const { return n_; }

private:
    size_t n_;
    bool inverse_;
    fftw_plan plan_;
    fftw_complex* sample_buffer_; // Buffer used during planning
};

/**
 * @brief Allocate 64-byte cache-aligned memory buffer for std::complex<double> using fftw_malloc
 * @param n Number of complex double elements to allocate
 * @return Complex* Pointer to 64-byte aligned complex buffer
 */
Complex* allocate_aligned_complex(size_t n);

/**
 * @brief Free 64-byte aligned buffer allocated by allocate_aligned_complex
 * @param ptr Pointer to complex buffer
 */
void free_aligned_complex(Complex* ptr);

/**
 * @brief High-performance FFTW3 1D FFT execution routine
 * @param data Complex vector input/output (transformed in-place)
 * @param inverse True for inverse FFT, false for forward FFT
 * @param planner_flags FFTW planner flags (e.g., FFTW_MEASURE or FFTW_PATIENT)
 */
void fft_1d_fftw(std::vector<Complex>& data, bool inverse = false, unsigned planner_flags = FFTW_MEASURE);

/**
 * @brief High-performance FFTW3 1D FFT execution routine on raw complex pointer
 * @param data Complex raw array pointer (transformed in-place)
 * @param n Size of the 1D transform
 * @param inverse True for inverse FFT, false for forward FFT
 * @param planner_flags FFTW planner flags (e.g., FFTW_MEASURE or FFTW_PATIENT)
 */
void fft_1d_fftw(Complex* data, size_t n, bool inverse = false, unsigned planner_flags = FFTW_MEASURE);

/**
 * @brief High-performance FFTW3 2D FFT execution routine
 * @param data Complex vector input/output (transformed in-place)
 * @param width Matrix width
 * @param height Matrix height
 * @param inverse True for inverse FFT, false for forward FFT
 * @param planner_flags FFTW planner flags
 */
void fft_2d_fftw(std::vector<Complex>& data, int width, int height, bool inverse = false, unsigned planner_flags = FFTW_MEASURE);

} // namespace cuda_fft

#endif // FFTW_WRAPPER_H
