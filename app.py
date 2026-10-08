from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Computer Vision Projects",
    layout="wide",
)

st.title("Computer Vision Projects")
st.caption("One Streamlit deployment for all submitted modules.")

st.write(
    "Use the pages in the sidebar to open each module from this single deployed app."
)

modules = [
    {
        "title": "Module 2: Camera Calibration and Object Measurement",
        "page": "pages/1_Module_2_Camera_Measurement.py",
        "description": "Calibrate a camera, draw an object box, and estimate real-world dimensions.",
    },
    {
        "title": "Module 3: Spatial vs Fourier Image Blurring",
        "page": "pages/2_Module_3_Spatial_vs_Fourier_Blur.py",
        "description": "Compare Gaussian blur in the spatial and Fourier domains.",
    },
    {
        "title": "Module 4: Human Boundary Extraction",
        "page": "pages/3_Module_4_Human_Boundary_Extraction.py",
        "description": "Run classical OpenCV segmentation on RGB or thermal images.",
    },
    {
        "title": "Assignment 6: Optical Flow and Structure from Motion",
        "page": "pages/4_Assignment_6_Optical_Flow_and_SfM.py",
        "description": "Review generated optical flow, tracking, and SfM results.",
    },
]

for module in modules:
    with st.container(border=True):
        st.subheader(module["title"])
        st.write(module["description"])
        st.page_link(module["page"], label="Open", icon=":material/open_in_new:")

