# CSc 8830 – Computer Vision

## Module 3 Assignment: Image Blurring

### Overview

This project was created for the **CSc 8830 Computer Vision – Module 3 Assignment**.

The goal of this project is to implement image blurring using two approaches:

* Spatial-domain convolution
* Fourier-domain filtering

The project demonstrates the **Convolution Theorem**, which states that convolution in the spatial domain is equivalent to multiplication in the frequency domain.

### Implementation

A Gaussian filter is used to blur an uploaded image.

The spatial-domain method performs:

**g(x, y) = f(x, y) * h(x, y)**

The Fourier-domain method performs:

**G(u, v) = F(u, v)H(u, v)**

followed by the inverse Fourier transform:

**g(x, y) = F⁻¹{G(u, v)}**

Both results are compared using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Maximum Difference

### Technologies Used

* Python
* OpenCV
* NumPy
* Streamlit
* Pillow

### Project Files

```text
cv-module3-assignment/
│
├── app.py
├── blur.py
├── requirements.txt
├── README.md
└── test_images/
```

`app.py` contains the Streamlit web application.

`blur.py` contains the spatial and Fourier-domain filtering functions.

`requirements.txt` contains the required Python packages.

### How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Upload an image and select the Gaussian **kernel size** and **sigma** from the sidebar.

The application will display the original image, spatial-domain blur, Fourier-domain blur, difference image, numerical error values, and Fourier spectrum.

### Experimental Result

For a **15 × 15 Gaussian kernel with sigma = 3.0**, the experiment produced:

* MAE: `6.263e-14`
* MSE: `6.323e-27`
* Maximum Difference: `4.832e-13`

The extremely small differences show that the spatial and Fourier-domain results are practically identical and support the Convolution Theorem.

### Author

**Parsh Jadon**

CSc 8830 – Computer Vision
Module 3 Assignment
