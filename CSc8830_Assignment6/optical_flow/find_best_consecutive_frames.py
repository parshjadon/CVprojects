"""
CSc 8830 - Assignment 6

Searches a selected frame range and finds the pair of
consecutive frames with the strongest valid optical flow.

Example:
    python find_best_consecutive_frames.py video1.mp4 750 850
"""

import cv2
import numpy as np
import sys
import os


if len(sys.argv) != 4:
    print("Usage:")
    print(
        "python find_best_consecutive_frames.py "
        "video.mp4 start_frame end_frame"
    )
    sys.exit()


video_path = sys.argv[1]
start_frame = int(sys.argv[2])
end_frame = int(sys.argv[3])

video_name = os.path.splitext(
    os.path.basename(video_path)
)[0]

output_folder = "../outputs"
os.makedirs(output_folder, exist_ok=True)


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Could not open video.")
    sys.exit()


fps = cap.get(cv2.CAP_PROP_FPS)


feature_params = dict(
    maxCorners=200,
    qualityLevel=0.2,
    minDistance=7,
    blockSize=7
)


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


best_displacement = 0

best_data = None

best_frame1 = None
best_frame2 = None


for frame_number in range(
    start_frame,
    end_frame
):

    # Read frame N
    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        frame_number
    )

    ret1, frame1 = cap.read()

    if not ret1:
        continue


    # Read frame N+1
    ret2, frame2 = cap.read()

    if not ret2:
        continue


    gray1 = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )


    points1 = cv2.goodFeaturesToTrack(
        gray1,
        mask=None,
        **feature_params
    )


    if points1 is None:
        continue


    points2, status, error = (
        cv2.calcOpticalFlowPyrLK(
            gray1,
            gray2,
            points1,
            None,
            **lk_params
        )
    )


    if points2 is None:
        continue


    status = status.flatten()


    good_old = points1[
        status == 1
    ].reshape(-1, 2)

    good_new = points2[
        status == 1
    ].reshape(-1, 2)


    if len(good_old) == 0:
        continue


    differences = (
        good_new - good_old
    )

    displacements = np.linalg.norm(
        differences,
        axis=1
    )


    # Ignore extremely small or suspiciously huge motion
    valid = np.where(
        (displacements >= 0.05)
        &
        (displacements <= 50.0)
    )[0]


    if len(valid) == 0:
        continue


    best_index = valid[
        np.argmax(
            displacements[valid]
        )
    ]


    current_displacement = (
        displacements[best_index]
    )


    if (
        current_displacement
        > best_displacement
    ):

        old_pt = good_old[
            best_index
        ]

        new_pt = good_new[
            best_index
        ]

        x1, y1 = old_pt
        x2, y2 = new_pt

        u = x2 - x1
        v = y2 - y1

        best_displacement = (
            current_displacement
        )

        best_data = {
            "frame1": frame_number,
            "frame2": frame_number + 1,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "u": float(u),
            "v": float(v)
        }

        best_frame1 = frame1.copy()
        best_frame2 = frame2.copy()


cap.release()


if best_data is None:
    print("No suitable consecutive frames found.")
    sys.exit()


print("\n================================")
print("BEST CONSECUTIVE FRAME PAIR")
print("================================")

print(
    "Frame 1:",
    best_data["frame1"]
)

print(
    "Frame 2:",
    best_data["frame2"]
)

print(
    f"\nStart pixel:"
    f" ({best_data['x1']:.3f},"
    f" {best_data['y1']:.3f})"
)

print(
    f"End pixel:"
    f" ({best_data['x2']:.3f},"
    f" {best_data['y2']:.3f})"
)

print(
    f"\nu = "
    f"{best_data['u']:.3f} pixels"
)

print(
    f"v = "
    f"{best_data['v']:.3f} pixels"
)

print(
    f"Displacement = "
    f"{best_displacement:.3f} pixels"
)


delta_t = 1 / fps

vx = (
    best_data["u"]
    / delta_t
)

vy = (
    best_data["v"]
    / delta_t
)

velocity = np.sqrt(
    vx ** 2 +
    vy ** 2
)


print(
    f"\nDelta t = "
    f"{delta_t:.6f} seconds"
)

print(
    f"Velocity X = "
    f"{vx:.3f} pixels/sec"
)

print(
    f"Velocity Y = "
    f"{vy:.3f} pixels/sec"
)

print(
    f"Velocity magnitude = "
    f"{velocity:.3f} pixels/sec"
)


# ----------------------------------------
# SAVE ORIGINAL FRAMES
# ----------------------------------------

cv2.imwrite(
    os.path.join(
        output_folder,
        f"{video_name}_best_frame1.jpg"
    ),
    best_frame1
)

cv2.imwrite(
    os.path.join(
        output_folder,
        f"{video_name}_best_frame2.jpg"
    ),
    best_frame2
)


# ----------------------------------------
# CREATE EVIDENCE IMAGE
# ----------------------------------------

evidence = best_frame2.copy()

start_point = (
    int(round(best_data["x1"])),
    int(round(best_data["y1"]))
)

end_point = (
    int(round(best_data["x2"])),
    int(round(best_data["y2"]))
)


cv2.circle(
    evidence,
    start_point,
    8,
    (0, 255, 0),
    -1
)

cv2.circle(
    evidence,
    end_point,
    8,
    (0, 0, 255),
    -1
)

cv2.arrowedLine(
    evidence,
    start_point,
    end_point,
    (255, 0, 0),
    3,
    tipLength=0.3
)


cv2.putText(
    evidence,
    (
        f"Frame "
        f"{best_data['frame1']} -> "
        f"{best_data['frame2']}"
    ),
    (25, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)


cv2.putText(
    evidence,
    (
        f"u={best_data['u']:.2f}, "
        f"v={best_data['v']:.2f}"
    ),
    (25, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 0),
    2
)


cv2.putText(
    evidence,
    (
        f"Displacement="
        f"{best_displacement:.2f}px"
    ),
    (25, 105),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)


cv2.imwrite(
    os.path.join(
        output_folder,
        f"{video_name}_best_consecutive_tracking.jpg"
    ),
    evidence
)


print(
    "\nSaved:"
)

print(
    f"../outputs/{video_name}_best_frame1.jpg"
)

print(
    f"../outputs/{video_name}_best_frame2.jpg"
)

print(
    f"../outputs/{video_name}_best_consecutive_tracking.jpg"
)