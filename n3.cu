#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <iostream>

// Description: Allocates 64 floating-point values on the GPU, squares them
// with a parallel transform, and prints one result on the host.
// Package: Thrust, a CUDA C++ library providing GPU containers and algorithms.
// Advantage: Thrust simplifies GPU programming by handling device memory and
// parallel execution without requiring manual CUDA kernels or launch syntax.

struct Square {
    __host__ __device__ float operator()(float x) const { return x * x; }
};

int main() {
    // Automatically allocates GPU memory and initializes data
    thrust::device_vector<float> d_vec(64);
    for (size_t i = 0; i < d_vec.size(); ++i) d_vec[i] = i;

    // Runs on GPU automatically without writing kernel launch syntax <<<>>>
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec.begin(), Square());

    // Automatically copies to CPU when printing
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "d_vec[" << i << "] = " << d_vec[i] << std::endl;
    }
    //std::cout << "d_vec[3] = " << d_vec[3] << std::endl; // Output: 9
    return 0;
}
