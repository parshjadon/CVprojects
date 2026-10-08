"""
CSc 8830 Module 4 Streamlit app.

Run locally:
    streamlit run streamlit_app.py

Deploy on Streamlit Community Cloud by selecting this file as the app entry
point and using requirements.txt for dependencies.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation import segment_and_save  # noqa: E402


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def bgra_to_rgba(image):
    return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)


def save_uploaded_file(uploaded_file, suffix_hint: str) -> Path:
    suffix = Path(uploaded_file.name).suffix or suffix_hint
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.getbuffer())
    temp.close()
    return Path(temp.name)


def show_saved_output_pngs(output_dir: Path) -> None:
    st.subheader("Saved Output PNG Files")

    png_files = sorted(output_dir.glob("*.png"))
    if not png_files:
        st.write("Run segmentation to generate output PNG files.")
        return

    for row_start in range(0, len(png_files), 3):
        cols = st.columns(3)
        for col, png_file in zip(cols, png_files[row_start : row_start + 3]):
            with col:
                st.image(str(png_file), caption=png_file.name, use_container_width=True)


st.set_page_config(
    page_title="CSc 8830 Human Boundary Extraction",
    layout="wide",
)

st.title("Human Boundary Extraction")
st.caption("Classical OpenCV segmentation for RGB and thermal images, with optional SAM2 mask comparison.")

with st.sidebar:
    st.header("Input")
    mode = st.radio(
        "Image type",
        options=["rgb", "thermal"],
        format_func=lambda value: "RGB camera image" if value == "rgb" else "Thermal image",
    )
    image_file = st.file_uploader("Upload input image", type=["png", "jpg", "jpeg", "bmp", "webp"])
    sam2_file = st.file_uploader("Optional SAM2 mask", type=["png", "jpg", "jpeg", "bmp", "webp"])
    run_button = st.button("Run segmentation", type="primary", use_container_width=True)

st.info(
    "The OpenCV solution uses thresholding, morphology, contours, color-space transforms, "
    "and GrabCut only. SAM2 is used only as an external reference mask for comparison."
)

output_dir = ROOT / "outputs" / "streamlit_results"

if run_button:
    if image_file is None:
        st.error("Please upload an input image first.")
    else:
        with st.spinner("Running OpenCV segmentation..."):
            input_path = save_uploaded_file(image_file, ".png")
            sam2_path = save_uploaded_file(sam2_file, ".png") if sam2_file is not None else None
            result = segment_and_save(input_path, mode, output_dir, sam2_path)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Overlay")
            st.image(bgr_to_rgb(result.overlay), use_container_width=True)
        with col2:
            st.subheader("Mask")
            st.image(result.mask, clamp=True, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Boundary")
            st.image(result.boundary, clamp=True, use_container_width=True)
        with col4:
            st.subheader("Transparent Cutout")
            st.image(bgra_to_rgba(result.cutout), use_container_width=True)

        st.subheader("SAM2 Comparison")
        if result.metrics:
            metric_cols = st.columns(4)
            metric_cols[0].metric("IoU", f"{result.metrics['iou']:.4f}")
            metric_cols[1].metric("Dice", f"{result.metrics['dice']:.4f}")
            metric_cols[2].metric("OpenCV pixels", f"{result.metrics['predicted_pixels']:,}")
            metric_cols[3].metric("SAM2 pixels", f"{result.metrics['reference_pixels']:,}")
            st.json(result.metrics)
        else:
            st.write("No SAM2 mask uploaded. Upload a binary SAM2 mask to calculate IoU and Dice.")
else:
    st.write("Upload an image in the sidebar, choose RGB or thermal, then run segmentation.")

st.divider()
show_saved_output_pngs(output_dir)
