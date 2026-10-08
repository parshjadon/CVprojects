"""
CSc 8830 - Computer Vision
Assignment 6

Track feature displacement between two selected frames.

Example:
    python frame_displacement.py video1.mp4 750 850
"""

import cv2
import numpy as np
import sys
import os


if len(sys.argv) != 4:
    print("Usage:")
    print("python frame_displacement.py video.mp4 start_frame end_frame")
    sys.exit()


video_path = sys.argv[1]
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])


if end_frame <= start_frame:
    print("Error: end_frame must be greater than start_frame.")
    sys.exit()


output_folder = "../outputs"
os.makedirs(output_folder, exist_ok=True)


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()


fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print("\nVIDEO INFORMATION")
print("-----------------------------")
print("FPS:", fps)
print("Total frames:", total_frames)
print("Start frame:", start_frame)
print("End frame:", end_frame)


if end_frame >= total_frames:
    print("Error: End frame is outside the video.")
    sys.exit()


# ---------------------------------------------------------
# READ START FRAME
# ---------------------------------------------------------

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    start_frame
)

ret, start_image = cap.read()

if not ret:
    print("Could not read start frame.")
    sys.exit()


start_gray = cv2.cvtColor(
    start_image,
    cv2.COLOR_BGR2GRAY
)


# ---------------------------------------------------------
# DETECT FEATURES AT FRAME 750
# ---------------------------------------------------------

feature_params = dict(
    maxCorners=200,
    qualityLevel=0.2,
    minDistance=7,
    blockSize=7
)


points = cv2.goodFeaturesToTrack(
    start_gray,
    mask=None,
    **feature_params
)


if points is None:
    print("No features detected.")
    sys.exit()


print(
    "Features detected at start:",
    len(points)
)


original_points = points.copy()


# ---------------------------------------------------------
# LUCAS-KANADE SETTINGS
# ---------------------------------------------------------

lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS |
        cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)


previous_gray = start_gray.copy()

current_points = points.copy()


# ---------------------------------------------------------
# TRACK FRAME-BY-FRAME FROM 750 TO 850
# ---------------------------------------------------------

current_frame_number = start_frame


while current_frame_number < end_frame:

    ret, frame = cap.read()

    if not ret:
        print("Could not read next frame.")
        break


    current_gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        current_points,
        None,
        **lk_params
    )


    if next_points is None:
        print("Tracking failed.")
        break


    status = status.flatten()


    # Keep only successfully tracked features
    current_points = next_points[
        status == 1
    ].reshape(-1, 1, 2)


    original_points = original_points[
        status == 1
    ].reshape(-1, 1, 2)


    previous_gray = current_gray.copy()

    current_frame_number += 1


cap.release()


# ---------------------------------------------------------
# CALCULATE TOTAL DISPLACEMENT
# ---------------------------------------------------------

start_points = original_points.reshape(-1, 2)

end_points = current_points.reshape(-1, 2)


if len(start_points) == 0:
    print("No points survived tracking.")
    sys.exit()


difference = (
    end_points - start_points
)


displacements = np.linalg.norm(
    difference,
    axis=1
)


# Find point with largest displacement
best_index = np.argmax(displacements)


x1, y1 = start_points[best_index]
x2, y2 = end_points[best_index]


u = x2 - x1
v = y2 - y1


total_displacement = np.sqrt(
    u ** 2 +
    v ** 2
)


frame_difference = (
    end_frame - start_frame
)


delta_t = (
    frame_difference / fps
)


velocity_x = (
    u / delta_t
)

velocity_y = (
    v / delta_t
)


velocity_magnitude = np.sqrt(
    velocity_x ** 2 +
    velocity_y ** 2
)


# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------

print("\n================================")
print("FRAME DISPLACEMENT RESULT")
print("================================")

print(
    f"Start frame: {start_frame}"
)

print(
    f"End frame: {end_frame}"
)

print(
    f"Frame difference: "
    f"{frame_difference}"
)


print(
    f"\nStart pixel:"
    f" ({x1:.3f}, {y1:.3f})"
)

print(
    f"End pixel:"
    f" ({x2:.3f}, {y2:.3f})"
)


print(
    f"\nHorizontal displacement u:"
    f" {u:.3f} pixels"
)

print(
    f"Vertical displacement v:"
    f" {v:.3f} pixels"
)

print(
    f"Total displacement:"
    f" {total_displacement:.3f} pixels"
)


print(
    f"\nElapsed time:"
    f" {delta_t:.3f} seconds"
)

print(
    f"Velocity X:"
    f" {velocity_x:.3f} pixels/sec"
)

print(
    f"Velocity Y:"
    f" {velocity_y:.3f} pixels/sec"
)

print(
    f"Velocity magnitude:"
    f" {velocity_magnitude:.3f} pixels/sec"
)


# ---------------------------------------------------------
# SAVE VISUALIZATION
# ---------------------------------------------------------

cap = cv2.VideoCapture(video_path)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    end_frame
)

ret, end_image = cap.read()

cap.release()


if ret:

    evidence = end_image.copy()


    start_pt = (
        int(round(x1)),
        int(round(y1))
    )

    end_pt = (
        int(round(x2)),
        int(round(y2))
    )


    # Start point
    cv2.circle(
        evidence,
        start_pt,
        8,
        (0, 255, 0),
        -1
    )


    # End point
    cv2.circle(
        evidence,
        end_pt,
        8,
        (0, 0, 255),
        -1
    )


    # Displacement arrow
    cv2.arrowedLine(
        evidence,
        start_pt,
        end_pt,
        (255, 0, 0),
        3,
        tipLength=0.15
    )


    cv2.putText(
        evidence,
        f"Frame {start_frame} -> {end_frame}",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    cv2.putText(
        evidence,
        (
            f"Start: "
            f"({x1:.1f}, {y1:.1f})"
        ),
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    cv2.putText(
        evidence,
        (
            f"End: "
            f"({x2:.1f}, {y2:.1f})"
        ),
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    cv2.putText(
        evidence,
        (
            f"u={u:.1f}, "
            f"v={v:.1f}, "
            f"d={total_displacement:.1f}px"
        ),
        (25, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    output_path = os.path.join(
        output_folder,
        f"displacement_{start_frame}_{end_frame}.jpg"
    )


    cv2.imwrite(
        output_path,
        evidence
    )


    print(
        "\nEvidence image saved to:",
        output_path
    )