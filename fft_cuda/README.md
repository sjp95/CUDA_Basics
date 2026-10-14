# CUDA Basics: Cache-Optimized Fast Fourier Transform (FFT) Engine

*High-performance, cache-aligned Discrete Fourier Transform engine for C++, Python, and Julia.*

[Quickstart](#quick-start) | [GitHub Repository](https://github.com/sjp95/CUDA_Basics)

---

## Overview

The `CUDA_Basics` FFT Engine provides a matrix-free, cache-aligned Discrete Fourier Transform (DFT) suite engineered to achieve bare-metal CPU hardware throughput and massive GPU parallelism. By leveraging L1/L2/L3 cache line blocking, 64-byte memory alignment, SIMD vectorization directives, and native GPU kernel passes, the engine minimizes memory bus bottlenecks and maximizes FLOPS utilization.

---

## The First-Class Experience

### C++ (Template Policies)
```cpp
#include "fft_cpu_cache_optimized.h"

// Allocate 64-byte aligned memory to eliminate cache line thrashing
cuda_fft::Complex* data = cuda_fft::allocate_aligned_complex(1024);

// Perform in-place cache-tiled forward FFT
cuda_fft::fft_1d_cache_optimized(data, 1024, /*inverse=*/false);

// Clean up
cuda_fft::free_aligned_complex(data);
```

### Python
```python
import numpy as np
import scipy.fft as fft

# High-performance NumPy array pass-through
x = np.random.randn(1024) + 1j * np.random.randn(1024)
X_freq = fft.fft(x)
```

### Julia
```julia
using FFTW

# Contiguous Julia vector FFT execution
x = rand(ComplexF64, 1024)
X_freq = fft(x)
```

---

## Why Cache-Optimized? (Architecture Deep-Dive)

Traditional Cooley-Tukey FFT implementations suffer from severe performance degrades on modern CPU architectures due to non-contiguous stride jumps $O(N \log N)$ that exceed the size of L1 and L2 caches (typically 32KB to 512KB). When power-of-two stride strides cross 64-byte cache line boundaries, CPU execution stalls while waiting for RAM access.

```text
Traditional Stride Jumps:
[X0] -------> [X8] -------> [X16] -------> [X24]  (L1/L2 Cache Misses!)

Cache-Tiled Loop Blocking:
+-------------------+-------------------+
| Tile 0 (64-byte)  | Tile 1 (64-byte)  |  (100% L1 Cache Hits)
+-------------------+-------------------+
```

Our cache-tiled architecture applies explicit loop blocking across 64-byte boundaries along with OpenMP SIMD vectorization (`#pragma omp simd`) to maintain peak memory throughput.

---

## Features Completed

- **Core C++ Engine**: 64-byte aligned SIMD memory allocators and in-place Cooley-Tukey / Stockham algorithms.
- **Cache Optimization Techniques**: Tiled 2D matrix transposition and L1/L2 cache locality protection.
- **FFT Drivers**: Custom CUDA warp-shuffle kernels, cuFFT wrapper API, and FFTW3 production baseline.
- **Benchmarking Utilities**: Automated CLI tool measuring execution time ($ms$), GFLOPS throughput, and memory bandwidth ($GB/s$).

---

## Build Requirements

- **C++ Compiler**: GCC 11+, Clang 13+, or MSVC supporting C++17/C++20.
- **Build System**: CMake 3.20+
- **Optional Dependencies**: OpenMP, FFTW3, CUDA Toolkit 11.8+.

---

## Quick Start

```bash
# Navigate to fft_cuda directory
cd fft_cuda

# Configure CMake build system
cmake -B build

# Build targets
cmake --build build -j$(nproc)

# Run cache performance benchmark
./build/fft_benchmark_cache

# Run CTest unit tests
cd build && ctest --output-on-failure
```
