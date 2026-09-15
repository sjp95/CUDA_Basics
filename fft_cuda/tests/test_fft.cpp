#include "fft_cpu.h"
#include "fft_cuda.cuh"
#include "cufft_wrapper.h"
#include "image_processing.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

using namespace cuda_fft;

void test_1d_fft_cpu() {
    std::cout << "[TEST] Running 1D CPU FFT test..." << std::endl;
    std::vector<Complex> input = {
        {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0},
        {5.0, 0.0}, {6.0, 0.0}, {7.0, 0.0}, {8.0, 0.0}
    };
    std::vector<Complex> freq, reconstructed;

    fft_1d_cpu(input, freq, false);
    fft_1d_cpu(freq, reconstructed, true);

    double max_err = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        max_err = std::max(max_err, std::abs(input[i] - reconstructed[i]));
    }
    std::cout << "  1D CPU Roundtrip Max Error: " << max_err << std::endl;
    assert(max_err < 1e-4);
}

void test_1d_fft_cuda() {
    std::cout << "[TEST] Running 1D Custom CUDA FFT test..." << std::endl;
    std::vector<Complex> input = {
        {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0},
        {5.0, 0.0}, {6.0, 0.0}, {7.0, 0.0}, {8.0, 0.0}
    };
    std::vector<Complex> freq_cpu, freq_cuda;

    fft_1d_cpu(input, freq_cpu, false);
    fft_1d_cuda(input, freq_cuda, false);

    double max_err = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        max_err = std::max(max_err, std::abs(freq_cpu[i] - freq_cuda[i]));
    }
    std::cout << "  1D CUDA vs CPU FFT Max Error: " << max_err << std::endl;
    assert(max_err < 1e-4);
}

void test_2d_fft() {
    std::cout << "[TEST] Running 2D FFT test..." << std::endl;
    int w = 16, h = 16;
    std::vector<Complex> input(w * h);
    for (int i = 0; i < w * h; ++i) {
        input[i] = Complex(i % 7, (i * 3) % 11);
    }

    std::vector<Complex> freq, reconstructed;
    fft_2d_cpu(input, freq, w, h, false);
    fft_2d_cpu(freq, reconstructed, w, h, true);

    double max_err = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        max_err = std::max(max_err, std::abs(input[i] - reconstructed[i]));
    }
    std::cout << "  2D CPU Roundtrip Max Error: " << max_err << std::endl;
    assert(max_err < 1e-4);
}

void test_image_filtering() {
    std::cout << "[TEST] Running Image Filtering test..." << std::endl;
    ImagePPM img = generate_synthetic_image(64, 64, true);
    ImagePPM filtered = apply_frequency_filter(img, FilterType::LOW_PASS_GAUSSIAN, 15.0);

    assert(filtered.width == 64 && filtered.height == 64);
    assert(filtered.pixels.size() == 64 * 64 * 3);
    std::cout << "  Image filtering executed successfully." << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "      Executing CUDA_Basics Unit Tests   " << std::endl;
    std::cout << "========================================" << std::endl;

    test_1d_fft_cpu();
    test_1d_fft_cuda();
    test_2d_fft();
    test_image_filtering();

    std::cout << "\nALL UNIT TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
