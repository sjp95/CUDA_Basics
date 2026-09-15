#include "image_processing.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>

namespace cuda_fft {

bool load_ppm(const std::string& filepath, ImagePPM& image) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    std::string format;
    file >> format;
    if (format != "P3" && format != "P6" && format != "P2" && format != "P5") {
        return false;
    }

    bool is_binary = (format == "P6" || format == "P5");
    bool is_rgb = (format == "P3" || format == "P6");

    int w, h, max_val;
    file >> w >> h >> max_val;

    image.width = w;
    image.height = h;
    image.channels = is_rgb ? 3 : 1;
    image.pixels.resize(w * h * image.channels);

    file.get(); // skip single whitespace/newline

    if (!is_binary) {
        for (size_t i = 0; i < image.pixels.size(); ++i) {
            double val;
            file >> val;
            image.pixels[i] = val;
        }
    } else {
        std::vector<unsigned char> buf(image.pixels.size());
        file.read(reinterpret_cast<char*>(buf.data()), buf.size());
        for (size_t i = 0; i < buf.size(); ++i) {
            image.pixels[i] = static_cast<double>(buf[i]);
        }
    }
    return true;
}

bool save_ppm(const std::string& filepath, const ImagePPM& image) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    bool is_rgb = (image.channels == 3);
    file << (is_rgb ? "P6" : "P5") << "\n";
    file << image.width << " " << image.height << "\n255\n";

    std::vector<unsigned char> buf(image.pixels.size());
    for (size_t i = 0; i < image.pixels.size(); ++i) {
        double p = std::clamp(image.pixels[i], 0.0, 255.0);
        buf[i] = static_cast<unsigned char>(std::round(p));
    }
    file.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    return true;
}

ImagePPM generate_synthetic_image(int width, int height, bool add_noise) {
    ImagePPM img;
    img.width = width;
    img.height = height;
    img.channels = 3;
    img.pixels.resize(width * height * 3);

    std::mt19937 rng(42);
    std::normal_distribution<double> gaussian_noise(0.0, 25.0);

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            int idx = (r * width + c) * 3;
            // Draw gradient and geometric pattern
            double red = (static_cast<double>(r) / height) * 255.0;
            double green = (static_cast<double>(c) / width) * 255.0;
            double blue = (std::sin(r * 0.05) * std::cos(c * 0.05) * 0.5 + 0.5) * 255.0;

            // Add sharp circle feature
            double dx = c - width / 2.0;
            double dy = r - height / 2.0;
            if (std::sqrt(dx * dx + dy * dy) < width / 6.0) {
                red = 220.0;
                green = 80.0;
                blue = 40.0;
            }

            if (add_noise) {
                // Add Gaussian noise
                red += gaussian_noise(rng);
                green += gaussian_noise(rng);
                blue += gaussian_noise(rng);

                // Add periodic pattern noise
                double periodic = 30.0 * std::sin(2.0 * 3.1415926535 * c / 8.0);
                red += periodic;
                green += periodic;
                blue += periodic;
            }

            img.pixels[idx + 0] = std::clamp(red, 0.0, 255.0);
            img.pixels[idx + 1] = std::clamp(green, 0.0, 255.0);
            img.pixels[idx + 2] = std::clamp(blue, 0.0, 255.0);
        }
    }
    return img;
}

ImagePPM apply_frequency_filter(const ImagePPM& input, FilterType filter_type, double cutoff_frequency, int butterworth_order, double threshold_ratio) {
    ImagePPM output = input;
    int w = input.width;
    int h = input.height;
    int channels = input.channels;

    int center_r = h / 2;
    int center_c = w / 2;

    for (int ch = 0; ch < channels; ++ch) {
        std::vector<Complex> channel_data(w * h);
        for (int i = 0; i < w * h; ++i) {
            channel_data[i] = Complex(input.pixels[i * channels + ch], 0.0);
        }

        std::vector<Complex> freq_data;
        fft_2d_cpu(channel_data, freq_data, w, h, false);
        fftshift_2d(freq_data, w, h);

        // Calculate maximum magnitude for adaptive thresholding
        double max_mag = 0.0;
        if (filter_type == FilterType::ADAPTIVE_THRESHOLD) {
            for (int i = 0; i < w * h; ++i) {
                max_mag = std::max(max_mag, std::abs(freq_data[i]));
            }
        }

        // Apply transfer function H(u, v)
        for (int r = 0; r < h; ++r) {
            for (int c = 0; c < w; ++c) {
                double du = r - center_r;
                double dv = c - center_c;
                double dist = std::sqrt(du * du + dv * dv);
                int idx = r * w + c;

                double h_val = 1.0;
                switch (filter_type) {
                    case FilterType::LOW_PASS_IDEAL:
                        h_val = (dist <= cutoff_frequency) ? 1.0 : 0.0;
                        break;
                    case FilterType::LOW_PASS_GAUSSIAN:
                        h_val = std::exp(-(dist * dist) / (2.0 * cutoff_frequency * cutoff_frequency));
                        break;
                    case FilterType::LOW_PASS_BUTTERWORTH:
                        h_val = 1.0 / (1.0 + std::pow(dist / cutoff_frequency, 2 * butterworth_order));
                        break;
                    case FilterType::HIGH_PASS_GAUSSIAN:
                        h_val = 1.0 - std::exp(-(dist * dist) / (2.0 * cutoff_frequency * cutoff_frequency));
                        break;
                    case FilterType::ADAPTIVE_THRESHOLD:
                        if (dist > cutoff_frequency && std::abs(freq_data[idx]) > threshold_ratio * max_mag) {
                            h_val = 0.0; // Filter out high frequency spikes (periodic noise)
                        } else {
                            h_val = 1.0;
                        }
                        break;
                }
                freq_data[idx] *= h_val;
            }
        }

        // Inverse shift and inverse 2D FFT
        fftshift_2d(freq_data, w, h);
        std::vector<Complex> spatial_data;
        fft_2d_cpu(freq_data, spatial_data, w, h, true);

        for (int i = 0; i < w * h; ++i) {
            output.pixels[i * channels + ch] = std::clamp(spatial_data[i].real(), 0.0, 255.0);
        }
    }
    return output;
}

ImagePPM adjust_colors_frequency(const ImagePPM& input, double red_scale, double green_scale, double blue_scale, double contrast_factor) {
    ImagePPM output = input;
    if (input.channels != 3) return output;

    int w = input.width;
    int h = input.height;
    double scales[3] = {red_scale, green_scale, blue_scale};

    for (int ch = 0; ch < 3; ++ch) {
        std::vector<Complex> channel_data(w * h);
        for (int i = 0; i < w * h; ++i) {
            channel_data[i] = Complex(input.pixels[i * 3 + ch], 0.0);
        }

        std::vector<Complex> freq_data;
        fft_2d_cpu(channel_data, freq_data, w, h, false);

        // Adjust DC component (brightness / color balance) and AC components (contrast)
        for (size_t i = 0; i < freq_data.size(); ++i) {
            if (i == 0) {
                freq_data[i] *= scales[ch];
            } else {
                freq_data[i] *= (scales[ch] * contrast_factor);
            }
        }

        std::vector<Complex> spatial_data;
        fft_2d_cpu(freq_data, spatial_data, w, h, true);

        for (int i = 0; i < w * h; ++i) {
            output.pixels[i * 3 + ch] = std::clamp(spatial_data[i].real(), 0.0, 255.0);
        }
    }
    return output;
}

} // namespace cuda_fft
