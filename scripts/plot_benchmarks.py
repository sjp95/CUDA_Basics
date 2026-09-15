#!/usr/bin/env python3
"""
Plots performance execution benchmarks and speedup metrics comparing CPU vs Custom CUDA vs cuFFT.
Compatible with NumPy 2.x and standard Pandas/Seaborn dependencies.
"""

import json
import os
import sys

# Suppress warnings and handle C-extension fallback for numexpr/bottleneck under NumPy 2.x
os.environ["NUMEXPR_MAX_THREADS"] = "8"
try:
    import pandas as pd
    pd.set_option("compute.use_bottleneck", False)
    pd.set_option("compute.use_numexpr", False)
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Use modern Python builtin types for NumPy 2.x compatibility
FLOAT_TYPE = float
INT_TYPE = int

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

def calc_range(arr):
    """Calculates peak-to-peak amplitude safely across NumPy 1.x and 2.x."""
    a = np.asarray(arr, dtype=FLOAT_TYPE)
    return np.max(a) - np.min(a)

def plot_benchmarks():
    json_path = "fft_cuda/build/benchmark_results.json"
    if not os.path.exists(json_path):
        json_path = "benchmark_results.json"

    if not os.path.exists(json_path):
        print("benchmark_results.json not found, skipping plot generation.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    data_1d = [d for d in data if d["mode"] == "1D"]
    data_2d = [d for d in data if d["mode"] == "2D"]

    os.makedirs("docs/images", exist_ok=True)

    # Figure 1: 1D Execution Time and Speedup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sizes_1d = [INT_TYPE(d["size_n"]) for d in data_1d]
    labels_1d = [str(d["size_label"]).split()[0] for d in data_1d]
    cpu_1d = [FLOAT_TYPE(d["cpu_ms"]) for d in data_1d]
    cuda_1d = [FLOAT_TYPE(d["custom_cuda_ms"]) for d in data_1d]
    cufft_1d = [FLOAT_TYPE(d["cufft_ms"]) for d in data_1d]
    speedup_1d = [FLOAT_TYPE(d["speedup_cufft"]) for d in data_1d]

    ax1.plot(labels_1d, cpu_1d, 'o-', label="CPU (Cooley-Tukey)", color="#d62728", linewidth=2)
    ax1.plot(labels_1d, cuda_1d, 's-', label="Custom CUDA FFT", color="#1f77b4", linewidth=2)
    ax1.plot(labels_1d, cufft_1d, '^--', label="cuFFT Baseline", color="#2ca02c", linewidth=2)
    ax1.set_yscale('log')
    ax1.set_xlabel("1D Signal Array Size N ($2^{8}$ to $2^{24}$)")
    ax1.set_ylabel("Execution Time (ms, log scale)")
    ax1.set_title("1D FFT Execution Time Comparison")
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()

    ax2.plot(labels_1d, speedup_1d, 'D-', color="#9467bd", linewidth=2.5, label="cuFFT vs CPU Speedup")
    ax2.plot(labels_1d, [FLOAT_TYPE(d["speedup_custom"]) for d in data_1d], 'o--', color="#1f77b4", linewidth=2, label="Custom CUDA vs CPU Speedup")
    ax2.set_xlabel("1D Signal Array Size N ($2^{8}$ to $2^{24}$)")
    ax2.set_ylabel("Speedup Factor ($T_{CPU} / T_{GPU}$)")
    ax2.set_title("1D FFT GPU Acceleration Factor")
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()

    plt.suptitle("1D C++/CUDA FFT Performance Benchmarks", fontsize=14, fontweight='bold')
    plt.savefig("docs/images/benchmark_1d_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved docs/images/benchmark_1d_performance.png")

    # Figure 2: 2D Execution Time and Speedup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    labels_2d = [str(d["size_label"]) for d in data_2d]
    cpu_2d = [FLOAT_TYPE(d["cpu_ms"]) for d in data_2d]
    cuda_2d = [FLOAT_TYPE(d["custom_cuda_ms"]) for d in data_2d]
    cufft_2d = [FLOAT_TYPE(d["cufft_ms"]) for d in data_2d]
    speedup_2d = [FLOAT_TYPE(d["speedup_cufft"]) for d in data_2d]

    ax1.plot(labels_2d, cpu_2d, 'o-', label="CPU 2D FFT", color="#d62728", linewidth=2)
    ax1.plot(labels_2d, cuda_2d, 's-', label="Custom CUDA 2D FFT", color="#1f77b4", linewidth=2)
    ax1.plot(labels_2d, cufft_2d, '^--', label="cuFFT 2D Baseline", color="#2ca02c", linewidth=2)
    ax1.set_yscale('log')
    ax1.set_xlabel("2D Image Resolution ($N \\times N$)")
    ax1.set_ylabel("Execution Time (ms, log scale)")
    ax1.set_title("2D Image FFT Execution Time Comparison")
    ax1.legend()

    ax2.bar(labels_2d, speedup_2d, color="#2ca02c", alpha=0.8, label="cuFFT Speedup")
    ax2.set_xlabel("2D Image Resolution ($N \\times N$)")
    ax2.set_ylabel("Speedup Factor ($T_{CPU} / T_{GPU}$)")
    ax2.set_title("2D FFT GPU Speedup Multiplier")
    ax2.legend()

    plt.suptitle("2D Image FFT Performance Benchmarks", fontsize=14, fontweight='bold')
    plt.savefig("docs/images/benchmark_2d_performance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved docs/images/benchmark_2d_performance.png")

if __name__ == "__main__":
    plot_benchmarks()
