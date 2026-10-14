#!/usr/bin/env bash
set -e

echo "=== Git Automation & Build Script for sjp95/CUDA_Basics FFT Suite ==="

# 1. Create required directory sub-paths
echo "[1/5] Ensuring directory structure exists..."
mkdir -p fft_cuda/include
mkdir -p fft_cuda/src
mkdir -p fft_cuda/tests

# 2. Build and run tests
echo "[2/5] Configuring and building with CMake..."
cmake -B fft_cuda/build -S fft_cuda
cmake --build fft_cuda/build

echo "[3/5] Running test suite..."
ctest --test-dir fft_cuda/build --output-on-failure

# 3. Stage changes
echo "[4/5] Staging changes in git..."
git add fft_cuda/include/fft_cpu_cache_optimized.h \
        fft_cuda/src/fft_cpu_cache_optimized.cpp \
        fft_cuda/include/fftw_wrapper.h \
        fft_cuda/src/fftw_wrapper.cpp \
        fft_cuda/src/benchmark_cpu_cache.cpp \
        fft_cuda/tests/test_cache_fft.cpp \
        fft_cuda/CMakeLists.txt \
        automate_git_fft.sh

# 4. Commit and push (with status check)
echo "[5/5] Committing changes..."
if git diff-index --quiet HEAD --; then
    echo "No changes to commit."
else
    git commit -m "feat(fft_cuda): add cache-tiled Cooley-Tukey FFT and FFTW3 benchmark suite"
    echo "Changes committed successfully."
fi

# Note: Pushing to remote requires valid credentials/remote setup.
if git remote | grep -q "origin"; then
    echo "Pushing to origin main..."
    git push origin main || echo "Git push skipped or remote rejected."
fi

echo "=== Git Automation Script Execution Complete ==="
