
import cv2
import numpy as np
from pathlib import Path

# Checkerboard settings: 9 x 6 INNER corners, 20 mm per square
CHECKERBOARD = (9, 6)
SQUARE_SIZE_MM = 20.0

IMAGE_DIR = Path("calibration_images")
OUTPUT_FILE = Path("calibration.npz")
REPORT_FILE = Path("calibration_report.txt")

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Could not find '{IMAGE_DIR}'. Create it and put your checkerboard "
            "photos inside."
        )

    image_paths = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in '{IMAGE_DIR}'."
        )

    # 3D checkerboard corner locations in millimeters.
    # The board is assumed to lie on the Z = 0 plane.
    object_template = np.zeros(
        (CHECKERBOARD[0] * CHECKERBOARD[1], 3),
        dtype=np.float32
    )
    object_template[:, :2] = (
        np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]]
        .T.reshape(-1, 2)
        * SQUARE_SIZE_MM
    )

    object_points = []
    image_points = []
    image_size = None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print(
                f"Skipping {image_path}: image size {current_size} "
                f"does not match {image_size}."
            )
            continue

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD,
            flags=(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
        )

        if found:
            refined_corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria
            )

            object_points.append(object_template.copy())
            image_points.append(refined_corners)
            print(f"Detected: {image_path}")
        else:
            print(f"Not detected: {image_path}")

    if len(object_points) < 5:
        raise RuntimeError(
            f"Only {len(object_points)} usable checkerboard images were found. "
            "Use at least 5; 15–25 varied photos are recommended."
        )

    print(f"\nCalibrating with {len(object_points)} images...")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None
    )

    # Calculate mean reprojection error across all detected corners.
    per_image_errors = []

    for i in range(len(object_points)):
        projected_points, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )

        # Both arrays become (number_of_corners, 2).
        observed = image_points[i].reshape(-1, 2)
        projected = projected_points.reshape(-1, 2)

        # Average Euclidean pixel error for this image.
        error = np.linalg.norm(observed - projected, axis=1).mean()
        per_image_errors.append(float(error))

    mean_reprojection_error = float(np.mean(per_image_errors))

    np.savez(
        OUTPUT_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=image_size[0],
        image_height=image_size[1],
        checkerboard_columns=CHECKERBOARD[0],
        checkerboard_rows=CHECKERBOARD[1],
        square_size_mm=SQUARE_SIZE_MM,
        rms_error=float(rms),
        mean_reprojection_error=mean_reprojection_error
    )

    report = f"""CAMERA CALIBRATION REPORT

Checkerboard inner corners: {CHECKERBOARD[0]} x {CHECKERBOARD[1]}
Checkerboard square size: {SQUARE_SIZE_MM:.2f} mm
Images used: {len(object_points)}
Image resolution: {image_size[0]} x {image_size[1]}

RMS calibration error: {float(rms):.6f} pixels
Mean reprojection error: {mean_reprojection_error:.6f} pixels

Camera matrix:
{camera_matrix}

Distortion coefficients:
{dist_coeffs}

Per-image mean reprojection errors (pixels):
{per_image_errors}
"""

    REPORT_FILE.write_text(report, encoding="utf-8")

    print("\nCalibration complete.")
    print(f"RMS calibration error: {float(rms):.6f} pixels")
    print(f"Mean reprojection error: {mean_reprojection_error:.6f} pixels")
    print(f"Saved calibration data to: {OUTPUT_FILE}")
    print(f"Saved report to: {REPORT_FILE}")


if __name__ == "__main__":
    main()