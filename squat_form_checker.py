
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from pathlib import Path

# Shared geometry functions used by multiple exercise trackers.
from core.geometry import (
    calculate_angle as calculate_angle_3d,
    world_point,
    screen_point,
    leg_confidence,
)


# ==========================================
# SMART MOTION COACH - SQUAT FORM CHECKER
# ==========================================

# Locate the model relative to this script instead of the terminal directory.
MODEL_PATH = str(
    Path(__file__).resolve().parent / "pose_landmarker_lite.task"
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# Body landmark connections used to draw the skeleton.
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31),
    (28, 30), (30, 32)
]


# Experimental thresholds for identifying squat positions.
SQUAT_DOWN_ANGLE = 100
SQUAT_UP_ANGLE = 160
TORSO_LEAN_LIMIT = 45
MIN_VISIBILITY = 0.5


# ==========================================
# TORSO LEAN CALCULATION
# ==========================================

def calculate_torso_lean(landmarks, shoulder_index, hip_index):
    """Estimate how far the torso leans away from vertical."""

    shoulder = np.array(
        world_point(landmarks, shoulder_index)
    )

    hip = np.array(
        world_point(landmarks, hip_index)
    )

    # The vector from the hip to the shoulder describes torso direction.
    torso = shoulder - hip

    # Horizontal movement includes both side-to-side and forward/backward lean.
    horizontal = math.sqrt(
        torso[0] ** 2 + torso[2] ** 2
    )

    vertical = abs(torso[1])

    # atan2 gives the lean angle relative to the vertical direction.
    return math.degrees(
        math.atan2(horizontal, vertical)
    )


# ==========================================
# TRACKING VARIABLES
# ==========================================

squat_count = 0
squat_stage = "up"

feedback = "Move into camera view"
previous_feedback = ""
feedback_until = 0

tracking_leg = "None"


# ==========================================
# MEDIAPIPE SETUP
# ==========================================

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    raise SystemExit

start_time = time.monotonic()
last_timestamp_ms = -1


# ==========================================
# MAIN CAMERA LOOP
# ==========================================

try:
    with PoseLandmarker.create_from_options(options) as landmarker:

        while True:

            success, frame = camera.read()

            if not success:
                print("Failed to capture frame.")
                break

            # Mirror the camera so movement feels more natural.
            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            # MediaPipe expects an RGB image.
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Video timestamps must increase with every processed frame.
            timestamp_ms = max(
                int(
                    (time.monotonic() - start_time) * 1000
                ),
                last_timestamp_ms + 1
            )

            last_timestamp_ms = timestamp_ms

            # Detect body landmarks in the current camera frame.
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            feedback = "Move into camera view"
            feedback_color = (0, 255, 255)

            left_angle = None
            right_angle = None
            torso_lean = None
            tracking_leg = "None"

            if (
                result.pose_landmarks
                and result.pose_world_landmarks
            ):

                screen_landmarks = result.pose_landmarks[0]
                world_landmarks = result.pose_world_landmarks[0]

                # ==================================
                # DRAW BODY SKELETON
                # ==================================

                for start, end in POSE_CONNECTIONS:

                    # Skip connections if either joint is not visible enough.
                    if min(
                        screen_landmarks[start].visibility,
                        screen_landmarks[end].visibility
                    ) < MIN_VISIBILITY:
                        continue

                    p1 = screen_point(
                        screen_landmarks,
                        start,
                        width,
                        height
                    )

                    p2 = screen_point(
                        screen_landmarks,
                        end,
                        width,
                        height
                    )

                    cv2.line(
                        frame,
                        p1,
                        p2,
                        (255, 0, 0),
                        3
                    )

                # Draw a green circle for each visible landmark.
                for landmark in screen_landmarks:

                    if landmark.visibility < MIN_VISIBILITY:
                        continue

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # ==================================
                # CALCULATE 3D KNEE ANGLES
                # ==================================

                # Hip, knee, and ankle landmarks for each leg.
                left_indices = [23, 25, 27]
                right_indices = [24, 26, 28]

                left_confidence = leg_confidence(
                    screen_landmarks,
                    left_indices
                )

                right_confidence = leg_confidence(
                    screen_landmarks,
                    right_indices
                )

                if left_confidence >= MIN_VISIBILITY:

                    left_angle = calculate_angle_3d(
                        world_point(world_landmarks, 23),
                        world_point(world_landmarks, 25),
                        world_point(world_landmarks, 27)
                    )

                if right_confidence >= MIN_VISIBILITY:

                    right_angle = calculate_angle_3d(
                        world_point(world_landmarks, 24),
                        world_point(world_landmarks, 26),
                        world_point(world_landmarks, 28)
                    )

                # ==================================
                # SELECT RELIABLE LEG TRACKING
                # ==================================

                knee_angle = None

                # Average both knee angles when both legs are visible.
                if (
                    left_angle is not None
                    and right_angle is not None
                ):

                    knee_angle = (
                        left_angle + right_angle
                    ) / 2

                    tracking_leg = "Both Legs"

                elif left_angle is not None:

                    knee_angle = left_angle
                    tracking_leg = "Left Leg"

                elif right_angle is not None:

                    knee_angle = right_angle
                    tracking_leg = "Right Leg"

                # ==================================
                # CALCULATE TORSO LEAN
                # ==================================

                # Use the left side when visible, otherwise try the right.
                if leg_confidence(
                    screen_landmarks,
                    [11, 23]
                ) >= MIN_VISIBILITY:

                    torso_lean = calculate_torso_lean(
                        world_landmarks,
                        11,
                        23
                    )

                elif leg_confidence(
                    screen_landmarks,
                    [12, 24]
                ) >= MIN_VISIBILITY:

                    torso_lean = calculate_torso_lean(
                        world_landmarks,
                        12,
                        24
                    )

                # ==================================
                # SQUAT COUNTING
                # ==================================

                if knee_angle is not None:

                    # A squat starts when the knee bends past the down threshold.
                    if knee_angle < SQUAT_DOWN_ANGLE:
                        squat_stage = "down"

                    # Count one rep only after returning to standing.
                    if (
                        knee_angle > SQUAT_UP_ANGLE
                        and squat_stage == "down"
                    ):

                        squat_stage = "up"
                        squat_count += 1

                        feedback_until = (
                            time.monotonic() + 1.5
                        )

                        print(
                            "Squat completed:",
                            squat_count
                        )

                    # ==============================
                    # EXERCISE FEEDBACK
                    # ==============================

                    if (
                        torso_lean is not None
                        and torso_lean > TORSO_LEAN_LIMIT
                        and knee_angle < SQUAT_UP_ANGLE
                    ):

                        feedback = "Check torso lean"
                        feedback_color = (0, 0, 255)

                    elif time.monotonic() < feedback_until:

                        feedback = "Rep completed!"
                        feedback_color = (0, 255, 0)

                    elif knee_angle < SQUAT_DOWN_ANGLE:

                        feedback = "Target depth reached!"
                        feedback_color = (0, 255, 0)

                    elif knee_angle <= SQUAT_UP_ANGLE:

                        if squat_stage == "down":
                            feedback = "Stand back up!"
                        else:
                            feedback = "Bend your knees more"

                        feedback_color = (0, 255, 255)

                    else:

                        feedback = "Standing position"
                        feedback_color = (255, 255, 255)

                else:

                    feedback = "Cannot track knees"
                    feedback_color = (0, 0, 255)

            # ==================================
            # DISPLAY INFORMATION
            # ==================================

            def draw_text(text, y, color=(255, 255, 255)):
                """Draw a line of information on the camera frame."""

                cv2.putText(
                    frame,
                    text,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )

            draw_text(
                f"Squats: {squat_count}",
                40,
                (0, 255, 0)
            )

            draw_text(
                f"Stage: {squat_stage}",
                75
            )

            draw_text(
                f"Tracking: {tracking_leg}",
                110
            )

            draw_text(
                "Left Knee: " +
                (
                    f"{left_angle:.0f} deg"
                    if left_angle is not None
                    else "Not visible"
                ),
                145
            )

            draw_text(
                "Right Knee: " +
                (
                    f"{right_angle:.0f} deg"
                    if right_angle is not None
                    else "Not visible"
                ),
                180
            )

            draw_text(
                "Torso Lean: " +
                (
                    f"{torso_lean:.0f} deg"
                    if torso_lean is not None
                    else "Not visible"
                ),
                215
            )

            # ==================================
            # FEEDBACK BANNER
            # ==================================

            cv2.rectangle(
                frame,
                (0, height - 85),
                (width, height),
                (30, 30, 30),
                -1
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.85

            text_width = cv2.getTextSize(
                feedback,
                font,
                font_scale,
                2
            )[0][0]

            # Reduce font size if the feedback is too wide.
            if text_width > width - 30:
                font_scale *= (
                    (width - 30) / text_width
                )

            cv2.putText(
                frame,
                feedback,
                (15, height - 30),
                font,
                font_scale,
                feedback_color,
                2
            )

            # Print feedback only when the message changes.
            if feedback != previous_feedback:

                print(
                    "Exercise Feedback:",
                    feedback
                )

                previous_feedback = feedback

            # Display the webcam feed.
            cv2.imshow(
                "Smart Motion Coach - 3D Squat Tracking",
                frame
            )

            # Press Q to exit the camera.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    # Release the webcam even if an error occurs.
    camera.release()
    cv2.destroyAllWindows()

print("Total squats completed:", squat_count)
