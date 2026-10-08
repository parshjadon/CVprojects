"""
CSc 8830 - Computer Vision
Assignment 6 - Part 2

Planar Structure from Motion / Multi-view Reconstruction

Object:
    A4 sheet = 21.0 cm x 29.7 cm

Eight known points are placed around the boundary.

Click order:

    1 -------- 2 -------- 3
    |                     |
    |                     |
    8                     4
    |                     |
    |                     |
    7 -------- 6 -------- 5

Controls:
    Left Click = select point
    R          = reset current image
    Enter      = confirm after 8 points
    Esc        = quit

Run:
    python sfm_part2.py
"""

import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt


# =========================================================
# SETTINGS
# =========================================================

OUTPUT_FOLDER = "../outputs"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


IMAGE_FILES = [
    "view1.jpeg",
    "view2.jpeg",
    "view3.jpeg",
    "view4.jpeg"
]


# A4 dimensions in centimeters
WIDTH_CM = 21.0
HEIGHT_CM = 29.7


# ---------------------------------------------------------
# Known world coordinates
# ---------------------------------------------------------

object_points = np.array([

    # 1 - top left
    [0.0, 0.0, 0.0],

    # 2 - top middle
    [10.5, 0.0, 0.0],

    # 3 - top right
    [21.0, 0.0, 0.0],

    # 4 - right middle
    [21.0, 14.85, 0.0],

    # 5 - bottom right
    [21.0, 29.7, 0.0],

    # 6 - bottom middle
    [10.5, 29.7, 0.0],

    # 7 - bottom left
    [0.0, 29.7, 0.0],

    # 8 - left middle
    [0.0, 14.85, 0.0]

], dtype=np.float32)


# Only X and Y are needed for homography
plane_points = object_points[:, :2]


# =========================================================
# POINT SELECTION FUNCTION
# =========================================================

def select_points(image, view_number):

    original_height, original_width = image.shape[:2]

    # Resize only for display if image is very large.
    max_width = 1100
    max_height = 750

    scale = min(
        max_width / original_width,
        max_height / original_height,
        1.0
    )

    display_width = int(
        original_width * scale
    )

    display_height = int(
        original_height * scale
    )

    display_base = cv2.resize(
        image,
        (display_width, display_height)
    )

    selected_points = []


    def mouse_callback(
        event,
        x,
        y,
        flags,
        param
    ):

        if event == cv2.EVENT_LBUTTONDOWN:

            if len(selected_points) < 8:

                # Convert displayed coordinates
                # back to original image coordinates.
                original_x = x / scale
                original_y = y / scale

                selected_points.append(
                    [
                        original_x,
                        original_y
                    ]
                )


    window_name = (
        f"View {view_number} - Select 8 points"
    )

    cv2.namedWindow(window_name)

    cv2.setMouseCallback(
        window_name,
        mouse_callback
    )


    while True:

        display = display_base.copy()


        # ---------------------------------------------
        # Draw selected points
        # ---------------------------------------------

        for i, point in enumerate(
            selected_points
        ):

            px = int(
                point[0] * scale
            )

            py = int(
                point[1] * scale
            )


            cv2.circle(
                display,
                (px, py),
                7,
                (0, 0, 255),
                -1
            )


            cv2.putText(
                display,
                str(i + 1),
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


        # ---------------------------------------------
        # Instructions
        # ---------------------------------------------

        cv2.rectangle(
            display,
            (0, 0),
            (display_width, 85),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            display,
            (
                f"View {view_number}: "
                "Click markers 1 -> 8 clockwise"
            ),
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            display,
            (
                f"Selected: "
                f"{len(selected_points)}/8"
                " | R = Reset"
                " | ENTER = Confirm"
            ),
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2
        )


        cv2.imshow(
            window_name,
            display
        )


        key = cv2.waitKey(20) & 0xFF


        # Reset
        if key == ord("r"):

            selected_points.clear()


        # ESC
        elif key == 27:

            cv2.destroyAllWindows()

            print(
                "Selection cancelled."
            )

            sys.exit()


        # Enter
        elif key == 13:

            if len(selected_points) == 8:

                break

            else:

                print(
                    "Please select all 8 points first."
                )


    cv2.destroyWindow(
        window_name
    )


    return np.array(
        selected_points,
        dtype=np.float32
    )


# =========================================================
# LOAD IMAGES AND COLLECT POINTS
# =========================================================

images = []

all_image_points = []

image_size = None


for i, filename in enumerate(
    IMAGE_FILES,
    start=1
):

    image = cv2.imread(
        filename
    )


    if image is None:

        print(
            f"Could not open {filename}"
        )

        sys.exit()


    height, width = image.shape[:2]


    if image_size is None:

        image_size = (
            width,
            height
        )


    # Camera calibration expects
    # all images to have same resolution.
    if (
        width != image_size[0]
        or height != image_size[1]
    ):

        print(
            "\nERROR:"
        )

        print(
            "All four photos should have "
            "the same resolution."
        )

        print(
            filename,
            "has resolution",
            width,
            "x",
            height
        )

        sys.exit()


    print(
        f"\nSelecting points for {filename}"
    )

    print(
        "Click in this order:"
    )

    print(
        "1 TL, 2 Top Middle, 3 TR,"
    )

    print(
        "4 Right Middle, 5 BR,"
    )

    print(
        "6 Bottom Middle, 7 BL,"
    )

    print(
        "8 Left Middle"
    )


    points = select_points(
        image,
        i
    )


    images.append(
        image
    )

    all_image_points.append(
        points
    )


    # ---------------------------------------------
    # Save clicked point visualization
    # ---------------------------------------------

    clicked_image = image.copy()


    for j, point in enumerate(
        points
    ):

        x = int(
            round(point[0])
        )

        y = int(
            round(point[1])
        )


        cv2.circle(
            clicked_image,
            (x, y),
            12,
            (0, 0, 255),
            -1
        )


        cv2.putText(
            clicked_image,
            str(j + 1),
            (x + 15, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )


    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            f"view{i}_clicked.jpg"
        ),
        clicked_image
    )


# =========================================================
# CALCULATE HOMOGRAPHIES
# =========================================================

homographies = []


print(
    "\n===================================="
)

print(
    "HOMOGRAPHY RESULTS"
)

print(
    "===================================="
)


for i, image_points in enumerate(
    all_image_points,
    start=1
):

    H, _ = cv2.findHomography(
        plane_points,
        image_points,
        method=0
    )


    homographies.append(
        H
    )


    print(
        f"\nHomography H{i}:"
    )

    print(H)


# =========================================================
# CAMERA CALIBRATION
# =========================================================

# We have the same physical object
# in all four views.
calibration_object_points = [

    object_points.copy()
    for _ in IMAGE_FILES

]


calibration_image_points = [

    points.reshape(-1, 1, 2)
    for points in all_image_points

]


# Initial guess for camera matrix
initial_camera_matrix = (
    cv2.initCameraMatrix2D(
        calibration_object_points,
        calibration_image_points,
        image_size,
        0
    )
)


flags = (
    cv2.CALIB_USE_INTRINSIC_GUESS
    |
    cv2.CALIB_ZERO_TANGENT_DIST
    |
    cv2.CALIB_FIX_K3
)


rms_calibration, camera_matrix, distortion, \
rvecs, tvecs = cv2.calibrateCamera(

    calibration_object_points,
    calibration_image_points,
    image_size,
    initial_camera_matrix,
    None,
    flags=flags
)


print(
    "\n===================================="
)

print(
    "CAMERA PARAMETERS"
)

print(
    "===================================="
)


print(
    "\nCalibration RMS error:"
)

print(
    rms_calibration
)


print(
    "\nCamera intrinsic matrix K:"
)

print(
    camera_matrix
)


print(
    "\nDistortion coefficients:"
)

print(
    distortion.ravel()
)


# =========================================================
# EXTRINSIC PARAMETERS AND CAMERA POSITIONS
# =========================================================

camera_centers = []

reprojection_errors = []


results_text = []


results_text.append(
    "CSc 8830 - Assignment 6 Part 2\n"
)

results_text.append(
    "Planar Multi-view Reconstruction\n\n"
)


results_text.append(
    "A4 physical size:\n"
)

results_text.append(
    "Width = 21.0 cm\n"
)

results_text.append(
    "Height = 29.7 cm\n\n"
)


results_text.append(
    "Camera intrinsic matrix K:\n"
)

results_text.append(
    str(camera_matrix)
)

results_text.append(
    "\n\n"
)


results_text.append(
    "Distortion coefficients:\n"
)

results_text.append(
    str(distortion.ravel())
)

results_text.append(
    "\n\n"
)


for i in range(4):

    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(
        rvecs[i]
    )


    t = tvecs[i]


    # Camera center in world coordinates
    #
    # C = -R^T t
    camera_center = (
        -R.T @ t
    )


    camera_centers.append(
        camera_center.flatten()
    )


    # ---------------------------------------------
    # Reproject known object points
    # ---------------------------------------------

    projected_points, _ = (
        cv2.projectPoints(

            object_points,

            rvecs[i],

            tvecs[i],

            camera_matrix,

            distortion
        )
    )


    projected_points = (
        projected_points.reshape(-1, 2)
    )


    observed_points = (
        all_image_points[i]
    )


    errors = np.linalg.norm(
        projected_points
        -
        observed_points,
        axis=1
    )


    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )


    reprojection_errors.append(
        rmse
    )


    print(
        "\n--------------------------------"
    )

    print(
        f"VIEW {i + 1}"
    )

    print(
        "--------------------------------"
    )


    print(
        "\nRotation matrix R:"
    )

    print(R)


    print(
        "\nTranslation vector t:"
    )

    print(
        t.flatten()
    )


    print(
        "\nCamera center C:"
    )

    print(
        camera_center.flatten()
    )


    print(
        "\nReprojection RMSE:"
    )

    print(
        f"{rmse:.4f} pixels"
    )


    # ---------------------------------------------
    # Save text results
    # ---------------------------------------------

    results_text.append(
        f"VIEW {i + 1}\n"
    )


    results_text.append(
        "Rotation Matrix R:\n"
    )

    results_text.append(
        str(R)
    )

    results_text.append(
        "\n\nTranslation t:\n"
    )

    results_text.append(
        str(t.flatten())
    )

    results_text.append(
        "\n\nCamera Center C:\n"
    )

    results_text.append(
        str(
            camera_center.flatten()
        )
    )

    results_text.append(
        (
            f"\n\nReprojection RMSE: "
            f"{rmse:.4f} pixels\n\n"
        )
    )


    # ---------------------------------------------
    # Draw observed vs reprojected points
    # ---------------------------------------------

    result_image = (
        images[i].copy()
    )


    for observed, projected in zip(
        observed_points,
        projected_points
    ):

        ox = int(
            round(observed[0])
        )

        oy = int(
            round(observed[1])
        )


        px = int(
            round(projected[0])
        )

        py = int(
            round(projected[1])
        )


        # Observed = Green
        cv2.circle(
            result_image,
            (ox, oy),
            11,
            (0, 255, 0),
            3
        )


        # Reprojected = Red
        cv2.circle(
            result_image,
            (px, py),
            6,
            (0, 0, 255),
            -1
        )


        cv2.line(
            result_image,
            (ox, oy),
            (px, py),
            (255, 0, 0),
            2
        )


    cv2.putText(
        result_image,
        (
            "Green = observed, "
            "Red = reconstructed"
        ),
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3
    )


    cv2.putText(
        result_image,
        (
            f"RMSE = "
            f"{rmse:.2f} px"
        ),
        (30, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3
    )


    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            f"view{i + 1}_reprojection.jpg"
        ),
        result_image
    )


# =========================================================
# SAVE TOP-DOWN RECTIFIED VIEWS
# =========================================================

# 25 pixels for every centimeter.
pixels_per_cm = 25


rectified_width = int(
    WIDTH_CM
    *
    pixels_per_cm
)

rectified_height = int(
    HEIGHT_CM
    *
    pixels_per_cm
)


destination_points = np.array([

    [0, 0],

    [rectified_width / 2, 0],

    [rectified_width - 1, 0],

    [
        rectified_width - 1,
        rectified_height / 2
    ],

    [
        rectified_width - 1,
        rectified_height - 1
    ],

    [
        rectified_width / 2,
        rectified_height - 1
    ],

    [
        0,
        rectified_height - 1
    ],

    [
        0,
        rectified_height / 2
    ]

], dtype=np.float32)


for i in range(4):

    # Image -> normalized A4 rectangle
    H_rect, _ = cv2.findHomography(
        all_image_points[i],
        destination_points,
        method=0
    )


    rectified = cv2.warpPerspective(
        images[i],
        H_rect,
        (
            rectified_width,
            rectified_height
        )
    )


    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            f"view{i + 1}_rectified.jpg"
        ),
        rectified
    )


# =========================================================
# CAMERA POSITION PLOT
# =========================================================

camera_centers = np.array(
    camera_centers
)


fig = plt.figure(
    figsize=(8, 7)
)


ax = fig.add_subplot(
    111,
    projection="3d"
)


# Sheet corners
sheet_x = [
    0,
    WIDTH_CM,
    WIDTH_CM,
    0,
    0
]

sheet_y = [
    0,
    0,
    HEIGHT_CM,
    HEIGHT_CM,
    0
]

sheet_z = [
    0,
    0,
    0,
    0,
    0
]


ax.plot(
    sheet_x,
    sheet_y,
    sheet_z,
    marker="o",
    label="A4 sheet boundary"
)


# Camera positions
ax.scatter(
    camera_centers[:, 0],
    camera_centers[:, 1],
    camera_centers[:, 2],
    s=80,
    label="Camera positions"
)


for i, center in enumerate(
    camera_centers
):

    ax.text(
        center[0],
        center[1],
        center[2],
        f"View {i + 1}"
    )


ax.set_xlabel(
    "X (cm)"
)

ax.set_ylabel(
    "Y (cm)"
)

ax.set_zlabel(
    "Z (cm)"
)

ax.set_title(
    "Estimated Camera Positions Around A4 Sheet"
)

ax.legend()


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "sfm_camera_positions.png"
    ),
    dpi=200
)


plt.close()


# =========================================================
# SAVE FINAL RESULT TEXT FILE
# =========================================================

results_text.append(
    "SUMMARY OF REPROJECTION ERRORS\n"
)


for i, error in enumerate(
    reprojection_errors
):

    results_text.append(
        (
            f"View {i + 1}: "
            f"{error:.4f} pixels\n"
        )
    )


result_file = os.path.join(
    OUTPUT_FOLDER,
    "sfm_results.txt"
)


with open(
    result_file,
    "w"
) as file:

    file.writelines(
        results_text
    )


# =========================================================
# DONE
# =========================================================

print(
    "\n===================================="
)

print(
    "PART 2 PROCESSING COMPLETE"
)

print(
    "===================================="
)


print(
    "\nCheck the outputs folder for:"
)

print(
    "view1_clicked.jpg ... view4_clicked.jpg"
)

print(
    "view1_reprojection.jpg ... view4_reprojection.jpg"
)

print(
    "view1_rectified.jpg ... view4_rectified.jpg"
)

print(
    "sfm_camera_positions.png"
)

print(
    "sfm_results.txt"
)