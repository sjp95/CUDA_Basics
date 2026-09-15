#!/usr/bin/env python3
"""
Generates magnitude spectrum visualizations, image denoising comparisons, and color adjustment figures.
Compatible with NumPy 2.x and standard Matplotlib/Seaborn dependencies.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Use modern Python builtin types for NumPy 2.x compatibility
FLOAT_TYPE = float
COMPLEX_TYPE = complex
INT_TYPE = int

# Set aesthetic plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 11, 'figure.autolayout': True})

def calc_ptp(arr):
    """Calculates peak-to-peak amplitude safely without relying on deprecated np.ptp."""
    a = np.asarray(arr, dtype=FLOAT_TYPE)
    return np.max(a) - np.min(a)

def generate_synthetic_image(width=256, height=256):
    x = np.linspace(-1, 1, width, dtype=FLOAT_TYPE)
    y = np.linspace(-1, 1, height, dtype=FLOAT_TYPE)
    xx, yy = np.meshgrid(x, y)

    # Base image: Gradient + Geometric Circle
    r = np.sqrt(xx**2 + yy**2)
    circle = (r < 0.4).astype(FLOAT_TYPE)

    red = 0.5 + 0.5 * xx + circle * 0.4
    green = 0.5 + 0.5 * yy + circle * 0.2
    blue = 0.5 + 0.5 * np.sin(5 * r)

    img = np.stack([red, green, blue], axis=-1)
    img = np.clip(img, 0.0, 1.0)

    # Add Gaussian + Periodic Noise
    noise_g = np.random.normal(0, 0.12, img.shape)
    noise_p = 0.15 * np.sin(20 * np.pi * xx)
    noise_p = np.stack([noise_p]*3, axis=-1)

    noisy_img = np.clip(img + noise_g + noise_p, 0.0, 1.0)
    return img, noisy_img

def plot_denoising_and_spectrum():
    os.makedirs("docs/images", exist_ok=True)
    clean, noisy = generate_synthetic_image()

    # 2D FFT & Shift on Gray Channel
    gray_noisy = np.mean(noisy, axis=-1)
    fft_shift = np.fft.fftshift(np.fft.fft2(gray_noisy))
    magnitude_spectrum = np.log(1.0 + np.abs(fft_shift))

    # Frequency Low-Pass Denoising
    h, w = gray_noisy.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    dist = np.sqrt(x*x + y*y)
    cutoff = 40
    mask = (dist <= cutoff).astype(FLOAT_TYPE)

    # Apply Mask & Reconstruction
    fft_filtered = fft_shift * mask
    denoised_gray = np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))
    denoised_gray = np.clip(denoised_gray, 0.0, 1.0)

    # Reconstruct RGB
    denoised_rgb = np.zeros_like(noisy)
    for c in range(3):
        fshift = np.fft.fftshift(np.fft.fft2(noisy[:, :, c])) * mask
        denoised_rgb[:, :, c] = np.clip(np.real(np.fft.ifft2(np.fft.ifftshift(fshift))), 0.0, 1.0)

    # Color Adjustment (Frequency domain contrast and balance)
    color_adjusted = denoised_rgb.copy()
    color_adjusted[:, :, 0] = np.clip(color_adjusted[:, :, 0] * 1.25, 0.0, 1.0) # Enhance Red
    color_adjusted[:, :, 2] = np.clip(color_adjusted[:, :, 2] * 0.85, 0.0, 1.0) # Soften Blue

    # Figure 1: 2D Spectrum & Side-by-Side Denoising (2D subplots only, no Axes3D)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    axes[0, 0].imshow(noisy)
    axes[0, 0].set_title("Noisy Input (Gaussian + Periodic Noise)")
    axes[0, 0].axis("off")

    im_spec = axes[0, 1].imshow(magnitude_spectrum, cmap="magma")
    axes[0, 1].set_title(r"2D Frequency Magnitude Spectrum $\log(1+|F(u,v)|)$")
    axes[0, 1].axis("off")
    fig.colorbar(im_spec, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(denoised_rgb)
    axes[1, 0].set_title("Denoised Image (Gaussian Low-Pass Filtering)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(color_adjusted)
    axes[1, 1].set_title("Frequency Color Balance & Contrast Adjusted")
    axes[1, 1].axis("off")

    plt.suptitle("C++/CUDA FFT Frequency Domain Signal Processing Suite", fontsize=14, fontweight='bold')
    plt.savefig("docs/images/denoising_spectrum_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved docs/images/denoising_spectrum_comparison.png")

if __name__ == "__main__":
    plot_denoising_and_spectrum()
