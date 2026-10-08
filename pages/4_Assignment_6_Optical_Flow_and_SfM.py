from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "CSc8830_Assignment6"
OUTPUTS = MODULE_ROOT / "outputs"
README = MODULE_ROOT / "README.md"

st.title("Assignment 6: Optical Flow and Structure from Motion")

if README.exists():
    st.markdown(README.read_text(encoding="utf-8"))
else:
    st.info("README.md was not found for this assignment.")

st.divider()
st.subheader("Generated Outputs")

if not OUTPUTS.exists():
    st.info("No outputs folder was found.")
    st.stop()

image_files = sorted(
    [
        path
        for path in OUTPUTS.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
)
video_files = sorted(
    [path for path in OUTPUTS.iterdir() if path.suffix.lower() in {".mp4", ".mov"}]
)
text_files = sorted(
    [path for path in OUTPUTS.iterdir() if path.suffix.lower() in {".txt", ".csv"}]
)

if image_files:
    st.subheader("Images")
    for start in range(0, len(image_files), 3):
        cols = st.columns(3)
        for col, image_file in zip(cols, image_files[start : start + 3]):
            with col:
                st.image(str(image_file), caption=image_file.name, use_container_width=True)

if video_files:
    st.subheader("Videos")
    for video_file in video_files:
        st.caption(video_file.name)
        st.video(str(video_file))

if text_files:
    st.subheader("Text and CSV Results")
    for text_file in text_files:
        with st.expander(text_file.name):
            st.code(text_file.read_text(encoding="utf-8", errors="replace"))

if not image_files and not video_files and not text_files:
    st.info("No generated output files were found.")

