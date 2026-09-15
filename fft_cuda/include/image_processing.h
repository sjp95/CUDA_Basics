#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include "fft_cpu.h"
#include <vector>
#include <string>

namespace cuda_fft {

enum class FilterType {
    LOW_PASS_IDEAL,
    LOW_PASS_GAUSSIAN,
    LOW_PASS_BUTTERWORTH,
    HIGH_PASS_GAUSSIAN,
    ADAPTIVE_THRESHOLD
};

struct ImagePPM {
    int width = 0;
    int height = 0;
    int channels = 3; // 1 for grayscale, 3 for RGB
    std::vector<double> pixels; // values in [0, 255]
};

// Image I/O (PPM/PGM ASCII and binary format)
bool load_ppm(const std::string& filepath, ImagePPM& image);
bool save_ppm(const std::string& filepath, const ImagePPM& image);

// Synthetic image generator (for noise removal testing and demo)
ImagePPM generate_synthetic_image(int width, int height, bool add_noise = true);

// Frequency domain filtering
ImagePPM apply_frequency_filter(const ImagePPM& input, FilterType filter_type, double cutoff_frequency, int butterworth_order = 2, double threshold_ratio = 0.1);

// Color Adjustment in Frequency Domain
ImagePPM adjust_colors_frequency(const ImagePPM& input, double red_scale, double green_scale, double blue_scale, double contrast_factor);

} // namespace cuda_fft

#endif // IMAGE_PROCESSING_H
