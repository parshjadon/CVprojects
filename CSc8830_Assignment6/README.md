# CSc 8830 - Computer Vision
## Assignment 6: Optical Flow and Structure from Motion

This project contains my implementation for Assignment 6.

The assignment has two parts:

1. Optical Flow and Motion Tracking
2. Structure from Motion using four views of a planar object

The project is implemented in Python using OpenCV.

---

## Project Structure

```text
CSc8830_Assignment6/
|
|-- optical_flow/
|   |-- video1.mp4
|   |-- video2.mp4
|   |-- optical_flow.py
|   |-- frame_displacement.py
|   `-- find_best_consecutive_frames.py
|
|-- sfm/
|   |-- view1.jpeg
|   |-- view2.jpeg
|   |-- view3.jpeg
|   |-- view4.jpeg
|   `-- sfm_part2.py
|
|-- outputs/
|   `-- generated results
|
|-- requirements.txt
`-- README.md
```

---

## Part 1 - Optical Flow

For Part 1, I used two videos containing motion.

- Video 1: Person moving
- Video 2: Toy car moving

Lucas-Kanade optical flow was used to track feature points between consecutive frames.

The optical flow constraint equation is:

```text
Ixu + Iyv + It = 0
```

### Run Optical Flow

Move to the optical flow folder:

```bash
cd optical_flow
```

For Video 1:

```bash
python optical_flow.py video1.mp4 ../outputs/flow_video1.mp4 video1
```

For Video 2:

```bash
python optical_flow.py video2.mp4 ../outputs/flow_video2.mp4 video2
```

### Find Consecutive Frames

```bash
python find_best_consecutive_frames.py video1.mp4 750 850
```

Example results:

**Video 1**

```text
Frames: 760 -> 761
u = 0.900 pixels
v = -0.103 pixels
Displacement = 0.905 pixels
```

**Video 2**

```text
Frames: 845 -> 846
u = 1.111 pixels
v = -0.040 pixels
Displacement = 1.112 pixels
```

---

## Part 2 - Structure from Motion

For Part 2, I used an A4 sheet as a planar object.

A4 dimensions:

```text
Width  = 21.0 cm
Height = 29.7 cm
```

Eight points were marked around the sheet and four photos were taken from different camera positions.

The points were arranged as:

```text
1 -------- 2 -------- 3
|                     |
8                     4
|                     |
7 -------- 6 -------- 5
```

### Run Part 2

Move to the SfM folder:

```bash
cd sfm
```

Run:

```bash
python sfm_part2.py
```

For every image, select the 8 points in the same order.

The program calculates:

- Homography
- Camera intrinsic parameters
- Rotation and translation
- Camera positions
- Reprojection error
- Rectified images

---

## Part 2 Results

Estimated camera positions:

```text
View 1: (11.58, -4.39, -31.52) cm
View 2: (10.38, 33.15, -33.68) cm
View 3: (25.94, 18.58, -32.35) cm
View 4: (-6.42, 18.22, -30.53) cm
```

Reprojection errors:

```text
View 1: 6.98 pixels
View 2: 4.28 pixels
View 3: 5.67 pixels
View 4: 6.89 pixels
```

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install opencv-python numpy matplotlib
```

---

## Requirements

```text
opencv-python
numpy
matplotlib
```

---

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Lucas-Kanade Optical Flow
- Homography
- Camera Calibration

---

## References

- CSc 8830 Computer Vision lecture materials
- Lucas and Kanade, 1981
- Hartley and Zisserman, Multiple View Geometry in Computer Vision
- OpenCV Documentation

---

## Author

Parsh Jadon  
CSc 8830 - Computer Vision
