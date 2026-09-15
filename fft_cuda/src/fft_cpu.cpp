#include "fft_cpu.h"
#include <algorithm>
#include <stdexcept>
#include <numbers>

namespace cuda_fft {

static bool is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static size_t bit_reverse(size_t x, int log2n) {
    size_t n = 0;
    for (int i = 0; i < log2n; ++i) {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

void fft_1d_cpu(const std::vector<Complex>& input, std::vector<Complex>& output, bool inverse) {
    size_t n = input.size();
    if (n == 0) return;
    if (!is_power_of_two(n)) {
        throw std::invalid_argument("FFT size must be a power of 2");
    }

    int log2n = 0;
    while ((1ULL << log2n) < n) {
        log2n++;
    }

    output.resize(n);
    for (size_t i = 0; i < n; ++i) {
        output[bit_reverse(i, log2n)] = input[i];
    }

    const double pi = 3.14159265358979323846;
    double angle_sign = inverse ? 1.0 : -1.0;

    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = angle_sign * 2.0 * pi / len;
        Complex wlen(std::cos(ang), std::sin(ang));
        size_t half_len = len >> 1;

        for (size_t i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (size_t j = 0; j < half_len; ++j) {
                Complex u = output[i + j];
                Complex v = output[i + j + half_len] * w;
                output[i + j] = u + v;
                output[i + j + half_len] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        for (size_t i = 0; i < n; ++i) {
            output[i] /= static_cast<double>(n);
        }
    }
}

void fft_2d_cpu(const std::vector<Complex>& input, std::vector<Complex>& output, int width, int height, bool inverse) {
    if (width <= 0 || height <= 0 || input.size() != static_cast<size_t>(width * height)) {
        throw std::invalid_argument("Invalid 2D dimensions or input size mismatch");
    }

    std::vector<Complex> intermediate(width * height);
    output.resize(width * height);

    // Row-wise 1D FFT
    for (int r = 0; r < height; ++r) {
        std::vector<Complex> row_in(width);
        std::vector<Complex> row_out(width);
        for (int c = 0; c < width; ++c) {
            row_in[c] = input[r * width + c];
        }
        fft_1d_cpu(row_in, row_out, inverse);
        for (int c = 0; c < width; ++c) {
            intermediate[r * width + c] = row_out[c];
        }
    }

    // Column-wise 1D FFT
    for (int c = 0; c < width; ++c) {
        std::vector<Complex> col_in(height);
        std::vector<Complex> col_out(height);
        for (int r = 0; r < height; ++r) {
            col_in[r] = intermediate[r * width + c];
        }
        fft_1d_cpu(col_in, col_out, inverse);
        for (int r = 0; r < height; ++r) {
            output[r * width + c] = col_out[r];
        }
    }
}

void fftshift_2d(std::vector<Complex>& data, int width, int height) {
    int half_w = width / 2;
    int half_h = height / 2;

    std::vector<Complex> temp = data;
    for (int r = 0; r < height; ++r) {
        int new_r = (r + half_h) % height;
        for (int c = 0; c < width; ++c) {
            int new_c = (c + half_w) % width;
            data[new_r * width + new_c] = temp[r * width + c];
        }
    }
}

} // namespace cuda_fft
