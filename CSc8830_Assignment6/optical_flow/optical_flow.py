"""
CSc 8830 - Computer Vision
Assignment 6
Optical Flow using Pyramidal Lucas-Kanade

Purpose:
    1. Computes optical flow using Lucas-Kanade.
    2. Visualizes tracked feature points and motion vectors.
    3. Automatically finds a pair of consecutive frames
       with meaningful motion.
    4. Saves actual pixel coordinates for theoretical validation.
    5. Saves clean frame images and a tracking evidence image.

Usage:
    python optical_flow.py video1.mp4 ../outputs/flow_video1.mp4 video1

Example:
    python optical_flow.py video2.mp4 ../outputs/flow_video2.mp4 video2

Requirements:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import sys
import os
import csv


# ---------------------------------------------------------
# COMMAND LINE ARGUMENTS
# ---------------------------------------------------------

if len(sys.argv) != 4:
    print("Usage:")
    print(
        "python optical_flow.py "
        "input_video output_video video_name"
    )
    sys.exit()


input_video = sys.argv[1]
output_video = sys.argv[2]
video_name = sys.argv[3]


# ---------------------------------------------------------
# CREATE OUTPUT DIRECTORY
# ---------------------------------------------------------

output_folder = "../outputs"

os.makedirs(
    output_folder,
    exist_ok=True
)


# ---------------------------------------------------------
# OPEN VIDEO
# ---------------------------------------------------------

cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print("\n--------------------------------")
print("VIDEO INFORMATION")
print("--------------------------------")

print("Video:", input_video)
print("FPS:", fps)
print("Resolution:", width, "x", height)
print("Total frames:", total_frames)

if fps > 0:
    duration = total_frames / fps
    print(f"Duration: {duration:.2f} seconds")


# ---------------------------------------------------------
# OUTPUT VIDEO WRITER
# ---------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

video_writer = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)


# ---------------------------------------------------------
# SHI-TOMASI FEATURE DETECTION PARAMETERS
# ---------------------------------------------------------

feature_params = dict(

    # Maximum number of corners/features
    maxCorners=200,

    # Minimum quality of detected corner
    qualityLevel=0.2,

    # Minimum distance between features
    minDistance=7,

    # Neighborhood size
    blockSize=7
)


# ---------------------------------------------------------
# LUCAS-KANADE PARAMETERS
# ---------------------------------------------------------

lk_params = dict(

    # Size of search window
    winSize=(21, 21),

    # Number of pyramid levels
    maxLevel=3,

    # Termination criteria
    criteria=(
        cv2.TERM_CRITERIA_EPS
        |
        cv2.TERM_CRITERIA_COUNT,
        30,
        0.01
    )
)


# ---------------------------------------------------------
# MOTION VALIDATION SETTINGS
# ---------------------------------------------------------

# Minimum number of pixels a feature must move
# before being considered useful evidence.
minimum_displacement = 2.0

# Avoid automatically selecting extremely large
# displacement because this may be caused by
# tracking failure.
maximum_displacement = 30.0

# Skip the first few frames.
# This lets tracking stabilize before saving evidence.
minimum_frame_number = 30

validation_saved = False


# ---------------------------------------------------------
# READ FIRST FRAME
# ---------------------------------------------------------

ret, first_frame = cap.read()

if not ret:
    print("Error: Could not read first frame.")
    sys.exit()


# Store a clean copy
old_clean_frame = first_frame.copy()


# Convert to grayscale
old_gray = cv2.cvtColor(
    old_clean_frame,
    cv2.COLOR_BGR2GRAY
)


# ---------------------------------------------------------
# DETECT INITIAL FEATURES
# ---------------------------------------------------------

p0 = cv2.goodFeaturesToTrack(
    old_gray,
    mask=None,
    **feature_params
)


if p0 is None:
    print(
        "Error: No feature points were detected "
        "in the first frame."
    )
    sys.exit()


print(
    "Initial feature points detected:",
    len(p0)
)


# ---------------------------------------------------------
# MASK USED FOR DRAWING TRACKING TRAILS
# ---------------------------------------------------------

tracking_mask = np.zeros_like(
    first_frame
)


frame_number = 0


# ---------------------------------------------------------
# PROCESS VIDEO
# ---------------------------------------------------------

while True:

    ret, current_frame = cap.read()

    if not ret:
        break


    # Keep a clean copy before drawing anything
    current_clean_frame = current_frame.copy()


    # Convert current frame to grayscale
    current_gray = cv2.cvtColor(
        current_clean_frame,
        cv2.COLOR_BGR2GRAY
    )


    # -----------------------------------------------------
    # CALCULATE OPTICAL FLOW
    # -----------------------------------------------------

    p1, status, error = cv2.calcOpticalFlowPyrLK(

        old_gray,
        current_gray,
        p0,
        None,

        **lk_params
    )


    # If tracking fails, redetect points
    if p1 is None or status is None:

        print(
            "Tracking lost. "
            "Redetecting features..."
        )

        p0 = cv2.goodFeaturesToTrack(
            old_gray,
            mask=None,
            **feature_params
        )

        if p0 is None:
            break

        old_clean_frame = (
            current_clean_frame.copy()
        )

        old_gray = current_gray.copy()

        frame_number += 1

        continue


    # -----------------------------------------------------
    # KEEP ONLY SUCCESSFULLY TRACKED FEATURES
    # -----------------------------------------------------

    good_new = p1[
        status.flatten() == 1
    ]

    good_old = p0[
        status.flatten() == 1
    ]


    # -----------------------------------------------------
    # AUTOMATICALLY FIND USEFUL TRACKING EVIDENCE
    # -----------------------------------------------------

    if (
        not validation_saved
        and frame_number >= minimum_frame_number
        and len(good_new) > 0
    ):

        # Calculate displacement for every feature
        differences = (
            good_new - good_old
        )

        displacements = np.linalg.norm(
            differences,
            axis=1
        )


        # Find valid features within desired range
        valid_indices = np.where(

            (
                displacements
                >= minimum_displacement
            )
            &
            (
                displacements
                <= maximum_displacement
            )

        )[0]


        if len(valid_indices) > 0:

            # Select valid feature with
            # largest displacement
            best_index = valid_indices[
                np.argmax(
                    displacements[
                        valid_indices
                    ]
                )
            ]


            old_pt = good_old[best_index].reshape(-1)
            new_pt = good_new[best_index].reshape(-1)

            x1 = float(old_pt[0])
            y1 = float(old_pt[1])

            x2 = float(new_pt[0])
            y2 = float(new_pt[1])


            # Optical flow displacement components
            u = x2 - x1
            v = y2 - y1


            displacement = np.sqrt(
                u ** 2
                +
                v ** 2
            )


            # Time between consecutive frames
            if fps > 0:
                delta_t = 1.0 / fps
            else:
                delta_t = 1.0


            # Velocity in pixels / second
            velocity_x = (
                u / delta_t
            )

            velocity_y = (
                v / delta_t
            )


            velocity_magnitude = np.sqrt(
                velocity_x ** 2
                +
                velocity_y ** 2
            )


            # ---------------------------------------------
            # PRINT RESULTS
            # ---------------------------------------------

            print(
                "\n================================"
            )

            print(
                "TRACKING VALIDATION FOUND"
            )

            print(
                "================================"
            )

            print(
                "Video:",
                video_name
            )

            print(
                "Frame 1:",
                frame_number
            )

            print(
                "Frame 2:",
                frame_number + 1
            )

            print(
                f"\nStart location:"
                f" ({x1:.3f}, {y1:.3f})"
            )

            print(
                f"End location:"
                f" ({x2:.3f}, {y2:.3f})"
            )

            print(
                f"\nu = {u:.3f} pixels"
            )

            print(
                f"v = {v:.3f} pixels"
            )

            print(
                f"Displacement = "
                f"{displacement:.3f} pixels"
            )

            print(
                f"\nDelta t = "
                f"{delta_t:.6f} seconds"
            )

            print(
                f"Velocity X = "
                f"{velocity_x:.3f} pixels/sec"
            )

            print(
                f"Velocity Y = "
                f"{velocity_y:.3f} pixels/sec"
            )

            print(
                f"Velocity magnitude = "
                f"{velocity_magnitude:.3f} "
                f"pixels/sec"
            )

            print(
                "================================\n"
            )


            # ---------------------------------------------
            # SAVE CSV DATA
            # ---------------------------------------------

            csv_path = os.path.join(
                output_folder,
                "tracking_results.csv"
            )


            file_exists = os.path.isfile(
                csv_path
            )


            with open(
                csv_path,
                "a",
                newline=""
            ) as csv_file:

                csv_writer = csv.writer(
                    csv_file
                )


                if not file_exists:

                    csv_writer.writerow([
                        "Video",
                        "Frame 1",
                        "Frame 2",
                        "FPS",
                        "Delta Time",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                        "u",
                        "v",
                        "Displacement",
                        "Velocity X",
                        "Velocity Y",
                        "Velocity Magnitude"
                    ])


                csv_writer.writerow([
                    video_name,
                    frame_number,
                    frame_number + 1,
                    fps,
                    delta_t,
                    x1,
                    y1,
                    x2,
                    y2,
                    u,
                    v,
                    displacement,
                    velocity_x,
                    velocity_y,
                    velocity_magnitude
                ])


            # ---------------------------------------------
            # SAVE CLEAN CONSECUTIVE FRAMES
            # ---------------------------------------------

            frame1_path = os.path.join(
                output_folder,
                f"{video_name}_frame1.jpg"
            )

            frame2_path = os.path.join(
                output_folder,
                f"{video_name}_frame2.jpg"
            )


            cv2.imwrite(
                frame1_path,
                old_clean_frame
            )

            cv2.imwrite(
                frame2_path,
                current_clean_frame
            )


            # ---------------------------------------------
            # CREATE TRACKING EVIDENCE IMAGE
            # ---------------------------------------------

            evidence = (
                current_clean_frame.copy()
            )


            point1 = (
                int(round(x1)),
                int(round(y1))
            )

            point2 = (
                int(round(x2)),
                int(round(y2))
            )


            # Starting point - green
            cv2.circle(
                evidence,
                point1,
                9,
                (0, 255, 0),
                -1
            )


            # Ending point - red
            cv2.circle(
                evidence,
                point2,
                9,
                (0, 0, 255),
                -1
            )


            # Motion arrow - blue
            cv2.arrowedLine(
                evidence,
                point1,
                point2,
                (255, 0, 0),
                3,
                tipLength=0.4
            )


            # Add text
            cv2.putText(
                evidence,
                (
                    f"Frame "
                    f"{frame_number} -> "
                    f"{frame_number + 1}"
                ),
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


            cv2.putText(
                evidence,
                (
                    f"Start: "
                    f"({x1:.2f}, {y1:.2f})"
                ),
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            cv2.putText(
                evidence,
                (
                    f"End: "
                    f"({x2:.2f}, {y2:.2f})"
                ),
                (30, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


            cv2.putText(
                evidence,
                (
                    f"u = {u:.2f}, "
                    f"v = {v:.2f}"
                ),
                (30, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )


            cv2.putText(
                evidence,
                (
                    f"Displacement = "
                    f"{displacement:.2f} px"
                ),
                (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


            evidence_path = os.path.join(
                output_folder,
                (
                    f"{video_name}_"
                    f"tracking_evidence.jpg"
                )
            )


            cv2.imwrite(
                evidence_path,
                evidence
            )


            validation_saved = True


            print(
                "Saved validation images:"
            )

            print(frame1_path)
            print(frame2_path)
            print(evidence_path)


    # -----------------------------------------------------
    # DRAW ALL OPTICAL FLOW VECTORS
    # -----------------------------------------------------

    display_frame = (
        current_clean_frame.copy()
    )


    for new, old in zip(
        good_new,
        good_old
    ):

        a, b = new.ravel()
        c, d = old.ravel()


        a_int = int(round(a))
        b_int = int(round(b))

        c_int = int(round(c))
        d_int = int(round(d))


        # Motion trail
        tracking_mask = cv2.line(
            tracking_mask,
            (c_int, d_int),
            (a_int, b_int),
            (0, 255, 0),
            2
        )


        # Current point
        cv2.circle(
            display_frame,
            (a_int, b_int),
            4,
            (0, 0, 255),
            -1
        )


        # Motion vector
        cv2.arrowedLine(
            display_frame,
            (c_int, d_int),
            (a_int, b_int),
            (255, 0, 0),
            1,
            tipLength=0.3
        )


    # Combine image and tracking trails
    output_frame = cv2.add(
        display_frame,
        tracking_mask
    )


    # Show current frame number
    cv2.putText(
        output_frame,
        f"Frame: {frame_number}",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # Write output video
    video_writer.write(
        output_frame
    )


    # Show video while processing
    cv2.imshow(
        "Lucas-Kanade Optical Flow",
        output_frame
    )


    # ESC key quits
    if cv2.waitKey(1) & 0xFF == 27:
        break


    # -----------------------------------------------------
    # PREPARE FOR NEXT FRAME
    # -----------------------------------------------------

    old_clean_frame = (
        current_clean_frame.copy()
    )

    old_gray = (
        current_gray.copy()
    )


    if len(good_new) > 0:

        p0 = good_new.reshape(
            -1,
            1,
            2
        )

    else:

        p0 = cv2.goodFeaturesToTrack(
            old_gray,
            mask=None,
            **feature_params
        )


    frame_number += 1


    # -----------------------------------------------------
    # PERIODICALLY REDETECT FEATURES
    # -----------------------------------------------------

    if (
        p0 is None
        or len(p0) < 25
        or frame_number % 60 == 0
    ):

        new_features = (
            cv2.goodFeaturesToTrack(
                old_gray,
                mask=None,
                **feature_params
            )
        )


        if new_features is not None:

            p0 = new_features


        # Reset old trails
        tracking_mask = np.zeros_like(
            current_clean_frame
        )


# ---------------------------------------------------------
# RELEASE RESOURCES
# ---------------------------------------------------------

cap.release()

video_writer.release()

cv2.destroyAllWindows()


# ---------------------------------------------------------
# FINAL STATUS
# ---------------------------------------------------------

print("\n--------------------------------")
print("PROCESSING COMPLETE")
print("--------------------------------")

print(
    "Optical flow video saved to:"
)

print(output_video)


if validation_saved:

    print(
        "\nTracking validation "
        "was successfully saved."
    )

    print(
        "Check the outputs folder."
    )

else:

    print(
        "\nWARNING:"
    )

    print(
        "No suitable motion was found "
        "for automatic validation."
    )

    print(
        "Try lowering "
        "minimum_displacement."
    )

    print(
        "Current value:",
        minimum_displacement
    )