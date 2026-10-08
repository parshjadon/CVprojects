# Camera Calibration and Object Measurement Project

This project is a simple computer vision application. The goal was to take a normal camera image, correct lens distortion, and estimate the real-world size of an object from that image using camera calibration and a basic pinhole-camera model.

In simple terms, the app first learns how the camera sees the world by checking a chessboard pattern in several images. Then it uses that calibration to measure the size of an object in a new image by drawing a rectangle around it and entering the camera-to-object distance.

---

## Assignment summary

The assignment focused on building a camera calibration pipeline and using it for measurement estimation. The main idea was:

- Use multiple checkerboard images to calculate the camera matrix and distortion coefficients
- Remove distortion from the image
- Let the user draw a box around an object
- Estimate the object’s width and height in real-world units using the camera distance and pixel size
- Save measurements and compare estimated values with actual values when available

This gives a practical, easy-to-understand way to turn a 2D image into a rough real-world measurement tool.

---

## Project approach

I approached the task in two main steps:

1. Calibration step
   - The script in `calibrate.py` looks through all images in the `calibration_images` folder
   - It detects checkerboard corners using OpenCV
   - It computes the camera matrix and distortion coefficients
   - It saves the result in `calibration.npz` and writes a summary report in `calibration_report.txt`

2. Measurement step
   - The app in `app.py` loads the saved calibration file
   - The user uploads an image and draws a rectangle around the object
   - The app converts the selected pixel width and height into real-world dimensions using the camera distance and focal length
   - Results are stored in `measurements.csv`

This is a lightweight and realistic solution for a camera measurement project without needing a full 3D reconstruction pipeline.

---

## Project structure

- `calibrate.py` — camera calibration script
- `app.py` — Streamlit app for object measurement
- `calibration_images/` — checkerboard calibration images
- `calibration.npz` — saved camera calibration parameters
- `calibration_report.txt` — summary of calibration results
- `measurements.csv` — saved measurement records
- `requirements.txt` — Python dependencies

---

## Requirements

- Python 3.9+
- A webcam or a phone camera for taking calibration images
- A printed checkerboard pattern

---

## How to set up the project

Open a terminal in the project folder and run:

```bash
python -m venv .venv
```

Then activate the virtual environment:

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If you get an import error for the drawing canvas package, install it as well:

```bash
pip install streamlit-drawable-canvas
```

---

## How to prepare calibration images

1. Print a checkerboard pattern with known square size
2. Put the pattern in front of the camera in different positions and angles
3. Take several clear images (15 to 25 is ideal)
4. Save them in the `calibration_images` folder
5. Make sure the images are in a common format such as `.jpg`, `.png`, or `.bmp`

The more varied the checkerboard positions, the better the calibration quality.

---

## Run the calibration

From the project root, run:

```bash
python calibrate.py
```

This will:

- detect checkerboard corners
- compute calibration values
- save `calibration.npz`
- generate `calibration_report.txt`

If the calibration fails, check that:

- the checkerboard images are clear
- the pattern is not too blurry or distorted
- there are enough valid images in `calibration_images`

---

## Run the app

Start the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal in your browser.

---

## How to use the app

1. Upload an image of the object you want to measure
2. Draw a rectangle around the object
3. Enter the camera-to-object distance in meters
4. Enter the actual width and height if you know them
5. Click the button to calculate the estimated dimensions
6. The app saves the result to `measurements.csv`

The app also shows error statistics such as MAE, RMSE, and bias when actual dimensions are provided.

---

## Output files

- `calibration.npz` — stores the camera matrix and distortion coefficients
- `calibration_report.txt` — human-readable calibration summary
- `measurements.csv` — stores all object measurements and validation data

---

## Notes

This project is a practical example of computer vision-based measurement. It is not perfect, but it is a solid starting point for camera calibration and simple object dimension estimation.

The approach works best when:

- the object is facing the camera directly
- the object is roughly flat
- the distance is measured accurately
- the calibration images are clear and well distributed

---

## Final thought

This project was a good way to connect theory with practice. By calibrating the camera first and then using the result to estimate object size, we turn raw images into more useful measurement data in a simple and understandable workflow.
