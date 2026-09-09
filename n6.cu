// This example demonstrates interoperability between Eigen and Thrust:
// Eigen stores vectors in host memory, while Thrust manages GPU memory and
// launches a parallel transform to square each element.
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include "Eigen/Dense"

struct SquareFunctor {
    __host__ __device__ float operator()(float x) const { return x * x; }
};

int main() {
    const int N = 64;

    // 1. CPU Eigen Vector initialization
    Eigen::VectorXf h_in = Eigen::VectorXf::LinSpaced(N, 0.0f, 63.0f);
    Eigen::VectorXf h_out(N);

    // 2. Copy Eigen vector data into Thrust device vector
    thrust::device_vector<float> d_vec(h_in.data(), h_in.data() + N);

    // 3. GPU execution via Thrust transform
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), SquareFunctor());

    // 4. Copy Thrust GPU results back to CPU Eigen vector
    thrust::copy(d_vec.begin(), d_vec.end(), h_out.data());

    // Print 5 outputs
    std::cout << "--- With Thrust + Eigen ---" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "h_out[" << i << "] = " << h_out[i] << std::endl;
    }

    return 0;
}
