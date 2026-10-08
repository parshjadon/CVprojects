# CSc 8830 Module 4: Human Boundary Extraction

This project extracts human boundaries from RGB and thermal images using classical OpenCV techniques only. No machine learning or deep learning model is used for the implemented RGB or thermal segmentation pipelines. SAM2 is used only as an optional external reference mask for comparison.

## Project Structure

```text
cv-module-4/
+-- streamlit_app.py
+-- requirements.txt
+-- README.md
+-- docs/
|   +-- report.md
+-- outputs/
|   +-- .gitkeep
+-- src/
    +-- segmentation.py
    +-- rgb_human_boundary.py
    +-- thermal_human_boundary.py
```

## Setup

Install Python 3.10 or newer. Then create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Streamlit Web App

Run the web demonstration:

```bash
python -m streamlit run streamlit_app.py
```

The app lets you upload:

- an RGB or thermal image,
- an optional SAM2 binary mask for comparison.

It displays:

- segmentation overlay,
- binary mask,
- boundary image,
- transparent cutout,
- IoU and Dice metrics when a SAM2 mask is provided.

## Command Line Usage

Use your own image file paths.

RGB image:

```bash
python src/rgb_human_boundary.py --image path/to/rgb_image.jpg
```

RGB image with SAM2 comparison:

```bash
python src/rgb_human_boundary.py --image path/to/rgb_image.jpg --sam2-mask path/to/sam2_mask.png
```

Thermal image:

```bash
python src/thermal_human_boundary.py --image path/to/thermal_image.jpg
```

Thermal image with SAM2 comparison:

```bash
python src/thermal_human_boundary.py --image path/to/thermal_image.jpg --sam2-mask path/to/sam2_mask.png
```

## Output Files

Script outputs are saved in `outputs/`:

- `*_mask.png`: binary human mask
- `*_boundary.png`: extracted boundary/contour
- `*_overlay.png`: mask and boundary overlay on the original image
- `*_cutout.png`: transparent-background human cutout
- `*_metrics.json`: IoU and Dice comparison values when a SAM2 mask is supplied

## SAM2 Mask

The SAM2 mask should be a binary image for the same input image:

- white pixels represent the human,
- black pixels represent the background.

This project does not run SAM2 directly. Generate the SAM2 mask using an external SAM2 demo or tool, then upload it in the Streamlit app or pass it with `--sam2-mask`.

## Deployment

For Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Create a new Streamlit app.
3. Select this repo.
4. Set the main file path to `streamlit_app.py`.
5. Streamlit installs dependencies from `requirements.txt`.

