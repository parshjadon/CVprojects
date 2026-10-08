"""
CSc 8830 - Computer Vision
Module 3 Assignment

Web application demonstrating that image convolution in the
spatial domain is equivalent to multiplication in the Fourier domain.

Run:
    streamlit run app.py
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from blur import (
    gaussian_kernel,
    spatial_blur,
    fourier_blur,
    compare_results
)


st.set_page_config(
    page_title="Spatial vs Fourier Image Blurring",
    page_icon="🖼️",
    layout="wide"
)

st.title("Image Blurring: Spatial vs Fourier Domain")

st.write(
    "This application applies the same Gaussian blur in two ways: "
    "spatial-domain convolution and Fourier-domain multiplication."
)

st.latex(r"g(x,y)=f(x,y)*h(x,y)")

st.latex(r"G(u,v)=F(u,v)H(u,v)")


# -------------------------------------------------
# Image upload
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("Upload an image to start the experiment.")
    st.stop()


image = Image.open(uploaded_file).convert("L")
image = np.array(image, dtype=np.float64)


# -------------------------------------------------
# Settings
# -------------------------------------------------

st.sidebar.header("Blur Settings")

kernel_size = st.sidebar.slider(
    "Kernel Size",
    min_value=3,
    max_value=51,
    value=15,
    step=2
)

sigma = st.sidebar.slider(
    "Gaussian Sigma",
    min_value=0.5,
    max_value=10.0,
    value=3.0,
    step=0.5
)


kernel = gaussian_kernel(kernel_size, sigma)


# -------------------------------------------------
# Perform filtering
# -------------------------------------------------

spatial_result = spatial_blur(image, kernel)
fourier_result = fourier_blur(image, kernel)

difference, mae, mse, max_error = compare_results(
    spatial_result,
    fourier_result
)


# -------------------------------------------------
# Display results
# -------------------------------------------------

st.subheader("Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.image(
        image.astype(np.uint8),
        caption="Original Image",
        use_container_width=True
    )

with col2:
    st.image(
        np.clip(spatial_result, 0, 255).astype(np.uint8),
        caption="Spatial-Domain Blur",
        use_container_width=True
    )

with col3:
    st.image(
        np.clip(fourier_result, 0, 255).astype(np.uint8),
        caption="Fourier-Domain Blur",
        use_container_width=True
    )


# -------------------------------------------------
# Numerical validation
# -------------------------------------------------

st.subheader("Numerical Validation")

m1, m2, m3 = st.columns(3)

m1.metric("Mean Absolute Error", f"{mae:.3e}")
m2.metric("Mean Squared Error", f"{mse:.3e}")
m3.metric("Maximum Difference", f"{max_error:.3e}")


st.write(
    "If the implementation is correct, these values should be extremely "
    "small and mainly caused by floating-point numerical precision."
)


# -------------------------------------------------
# Difference image
# -------------------------------------------------

st.subheader("Difference Between the Two Results")

if max_error > 0:
    difference_display = difference / max_error * 255
else:
    difference_display = np.zeros_like(difference)

st.image(
    difference_display.astype(np.uint8),
    caption="Amplified Absolute Difference",
    use_container_width=True
)


# -------------------------------------------------
# Kernel
# -------------------------------------------------

st.subheader("Gaussian Filter")

st.write(
    f"Kernel size: {kernel_size} × {kernel_size}, Sigma: {sigma}"
)

st.dataframe(kernel)


# -------------------------------------------------
# Fourier spectrum
# -------------------------------------------------

st.subheader("Fourier Spectrum")

F = np.fft.fftshift(np.fft.fft2(image))
spectrum = np.log1p(np.abs(F))

spectrum = (
    (spectrum - spectrum.min())
    / (spectrum.max() - spectrum.min())
    * 255
)

st.image(
    spectrum.astype(np.uint8),
    caption="Magnitude Spectrum of Original Image",
    use_container_width=True
)