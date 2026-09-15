#include "fft_cuda.cuh"
#include <iostream>
#include <cmath>

namespace cuda_fft {

#ifdef __CUDACC__

#define CUDA_CHECK_FFT(call)                                                 \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " code=" << err << " \"" << cudaGetErrorString(err) \
                      << "\"" << std::endl;                                 \
        }                                                                   \
    } while (0)

// Helper: Bit reversal on GPU
__device__ inline unsigned int bit_reverse_gpu(unsigned int x, int log2n) {
    unsigned int n = 0;
    for (int i = 0; i < log2n; ++i) {
        n = (n << 1) | (x & 1);
        x >>= 1;
    }
    return n;
}

// Warp shuffle butterfly exchange for sub-warp FFT stages (warp size = 32)
__device__ inline cuDoubleComplex warp_butterfly_shuffle(cuDoubleComplex val, int mask, bool inverse) {
    double real = __shfl_xor_sync(0xffffffff, val.x, mask);
    double imag = __shfl_xor_sync(0xffffffff, val.y, mask);
    return make_cuDoubleComplex(real, imag);
}

// Optimized 1D FFT kernel using Shared Memory and Warp Shuffle primitives
__global__ void fft_1d_shared_warp_kernel(const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int n, int log2n, bool inverse) {
    extern __shared__ cuDoubleComplex s_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        unsigned int rev = bit_reverse_gpu(gid, log2n);
        s_data[tid] = d_in[rev];
    } else {
        s_data[tid] = make_cuDoubleComplex(0.0, 0.0);
    }
    __syncthreads();

    // In-block shared memory Cooley-Tukey butterfly stage passes
    for (int len = 2; len <= blockDim.x && len <= n; len <<= 1) {
        int half_len = len >> 1;
        int mask = half_len;

        if (len <= 32) {
            // Use warp shuffle primitives for intra-warp stages
            cuDoubleComplex partner = warp_butterfly_shuffle(s_data[tid], mask, inverse);
            bool is_lower = (tid & mask) == 0;
            int pos_in_sub = tid % half_len;

            double angle_sign = inverse ? 1.0 : -1.0;
            const double pi = 3.14159265358979323846;
            double angle = angle_sign * 2.0 * pi * pos_in_sub / len;
            cuDoubleComplex w = make_cuDoubleComplex(cos(angle), sin(angle));

            if (is_lower) {
                s_data[tid] = cuCadd(s_data[tid], cuCmul(partner, w));
            } else {
                cuDoubleComplex diff = cuCsub(partner, cuCmul(s_data[tid], w));
                s_data[tid] = diff;
            }
            __syncthreads();
        } else {
            // Shared memory butterfly pass for block-level stages
            if ((tid % len) < half_len) {
                int pos_in_sub = tid % half_len;
                double angle_sign = inverse ? 1.0 : -1.0;
                const double pi = 3.14159265358979323846;
                double angle = angle_sign * 2.0 * pi * pos_in_sub / len;

                cuDoubleComplex w = make_cuDoubleComplex(cos(angle), sin(angle));
                cuDoubleComplex u = s_data[tid];
                cuDoubleComplex v = s_data[tid + half_len];
                cuDoubleComplex vw = cuCmul(v, w);

                s_data[tid] = cuCadd(u, vw);
                s_data[tid + half_len] = cuCsub(u, vw);
            }
            __syncthreads();
        }
    }

    if (gid < n) {
        d_out[gid] = s_data[tid];
    }
}

__global__ void fft_1d_stage_kernel(cuDoubleComplex* d_data, int n, int stage_len, bool inverse) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_len = stage_len / 2;

    if (tid >= n / 2) return;

    int sub_idx = tid / half_len;
    int pos_in_sub = tid % half_len;

    int idx0 = sub_idx * stage_len + pos_in_sub;
    int idx1 = idx0 + half_len;

    double angle_sign = inverse ? 1.0 : -1.0;
    const double pi = 3.14159265358979323846;
    double angle = angle_sign * 2.0 * pi * pos_in_sub / stage_len;

    cuDoubleComplex w = make_cuDoubleComplex(cos(angle), sin(angle));
    cuDoubleComplex u = d_data[idx0];
    cuDoubleComplex v = d_data[idx1];
    cuDoubleComplex vw = cuCmul(v, w);

    d_data[idx0] = cuCadd(u, vw);
    d_data[idx1] = cuCsub(u, vw);
}

__global__ void fft_scale_kernel(cuDoubleComplex* d_data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        d_data[tid].x /= n;
        d_data[tid].y /= n;
    }
}

bool is_cuda_available() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}

void fft_1d_cuda(const std::vector<Complex>& input, std::vector<Complex>& output, bool inverse) {
    int n = static_cast<int>(input.size());
    if (n == 0) return;

    if (!is_cuda_available()) {
        fft_1d_cpu(input, output, inverse);
        return;
    }

    int log2n = 0;
    while ((1 << log2n) < n) log2n++;

    size_t bytes = n * sizeof(cuDoubleComplex);
    cuDoubleComplex *d_in = nullptr, *d_out = nullptr;

    CUDA_CHECK_FFT(cudaMalloc(&d_in, bytes));
    CUDA_CHECK_FFT(cudaMalloc(&d_out, bytes));

    std::vector<cuDoubleComplex> h_in(n);
    for (int i = 0; i < n; ++i) {
        h_in[i] = make_cuDoubleComplex(input[i].real(), input[i].imag());
    }

    CUDA_CHECK_FFT(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    int threads = std::min(256, n);
    int blocks = (n + threads - 1) / threads;
    size_t shared_mem_bytes = threads * sizeof(cuDoubleComplex);

    fft_1d_shared_warp_kernel<<<blocks, threads, shared_mem_bytes>>>(d_in, d_out, n, log2n, inverse);
    CUDA_CHECK_FFT(cudaDeviceSynchronize());

    for (int len = threads * 2; len <= n; len <<= 1) {
        int half_blocks = ((n / 2) + threads - 1) / threads;
        fft_1d_stage_kernel<<<half_blocks, threads>>>(d_out, n, len, inverse);
        CUDA_CHECK_FFT(cudaDeviceSynchronize());
    }

    if (inverse) {
        fft_scale_kernel<<<blocks, threads>>>(d_out, n);
        CUDA_CHECK_FFT(cudaDeviceSynchronize());
    }

    std::vector<cuDoubleComplex> h_out(n);
    CUDA_CHECK_FFT(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    output.resize(n);
    for (int i = 0; i < n; ++i) {
        output[i] = Complex(cuCreal(h_out[i]), cuCimag(h_out[i]));
    }

    CUDA_CHECK_FFT(cudaFree(d_in));
    CUDA_CHECK_FFT(cudaFree(d_out));
}

void fft_2d_cuda(const std::vector<Complex>& input, std::vector<Complex>& output, int width, int height, bool inverse) {
    if (!is_cuda_available()) {
        fft_2d_cpu(input, output, width, height, inverse);
        return;
    }

    // 2D FFT using 1D row/col transforms
    std::vector<Complex> intermediate(width * height);
    output.resize(width * height);

    for (int r = 0; r < height; ++r) {
        std::vector<Complex> row_in(width);
        std::vector<Complex> row_out(width);
        for (int c = 0; c < width; ++c) row_in[c] = input[r * width + c];
        fft_1d_cuda(row_in, row_out, inverse);
        for (int c = 0; c < width; ++c) intermediate[r * width + c] = row_out[c];
    }

    for (int c = 0; c < width; ++c) {
        std::vector<Complex> col_in(height);
        std::vector<Complex> col_out(height);
        for (int r = 0; r < height; ++r) col_in[r] = intermediate[r * width + c];
        fft_1d_cuda(col_in, col_out, inverse);
        for (int r = 0; r < height; ++r) output[r * width + c] = col_out[r];
    }
}

#else

bool is_cuda_available() {
    return false;
}

void fft_1d_cuda(const std::vector<Complex>& input, std::vector<Complex>& output, bool inverse) {
    fft_1d_cpu(input, output, inverse);
}

void fft_2d_cuda(const std::vector<Complex>& input, std::vector<Complex>& output, int width, int height, bool inverse) {
    fft_2d_cpu(input, output, width, height, inverse);
}

#endif

} // namespace cuda_fft
