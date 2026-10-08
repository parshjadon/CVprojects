"""
CSc 8830 - Computer Vision
Module 3 Assignment

Image blurring in:
1. Spatial domain using convolution
2. Frequency domain using Fourier transforms

The purpose is to experimentally verify the convolution theorem:
    f * h  <->  F H
"""

import cv2
import numpy as np


def gaussian_kernel(size=15, sigma=3.0):
    """Create a normalized 2D Gaussian kernel."""

    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    k = cv2.getGaussianKernel(size, sigma)
    kernel = k @ k.T

    return kernel / kernel.sum()


def spatial_blur(image, kernel):
    """
    Blur using spatial-domain convolution.

    BORDER_CONSTANT is used so that the boundary condition matches
    the zero-padding used in the Fourier implementation.
    """

    return cv2.filter2D(
        image.astype(np.float64),
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_CONSTANT
    )


def fourier_blur(image, kernel):
    """
    Perform the same linear convolution using the Fourier domain.

    The image and kernel are zero-padded so FFT multiplication
    represents linear convolution instead of circular convolution.
    """

    image = image.astype(np.float64)

    h, w = image.shape
    kh, kw = kernel.shape

    # Size needed for full linear convolution
    fft_h = h + kh - 1
    fft_w = w + kw - 1

    # Fourier transform of image
    F = np.fft.fft2(image, s=(fft_h, fft_w))

    # Fourier transform of kernel
    H = np.fft.fft2(kernel, s=(fft_h, fft_w))

    # Convolution theorem: multiplication in frequency domain
    G = F * H

    # Return to spatial domain
    full_result = np.real(np.fft.ifft2(G))

    # Crop the full convolution to the same size as the input image
    pad_y = kh // 2
    pad_x = kw // 2

    result = full_result[
        pad_y:pad_y + h,
        pad_x:pad_x + w
    ]

    return result


def compare_results(spatial, fourier):
    """Calculate numerical differences between both implementations."""

    difference = np.abs(spatial - fourier)

    mae = np.mean(difference)
    mse = np.mean((spatial - fourier) ** 2)
    max_error = np.max(difference)

    return difference, mae, mse, max_error