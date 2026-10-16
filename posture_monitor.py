
import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ==========================================
# SMART MOTION COACH - POSTURE MONITOR
# ==========================================

MODEL_PATH = "pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Experimental thresholds
SHOULDER_TILT_LIMIT = 8
TORSO_LEAN_LIMIT = 15
MIN_VISIBILITY = 0.5

# How long poor alignment must persist
# before a warning appears
WARNING_DELAY = 1.5

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def world_point(landmarks, index):
    landmark = landmarks[index]

    return np.array(
        [landmark.x, landmark.y, landmark.z],
        dtype=float
    )


def screen_point(landmarks, index, width, height):
    landmark = landmarks[index]

    return (
        int(landmark.x * width),
        int(landmark.y * height)
    )


def visible(landmarks, indices):
    return all(
        landmarks[i].visibility >= MIN_VISIBILITY
        for i in indices
    )


def calculate_shoulder_tilt(landmarks):
    """
    Estimate shoulder height difference using
    MediaPipe world coordinates.
    """

    left = world_point(landmarks, 11)
    right = world_point(landmarks, 12)

    difference = right - left

    horizontal_distance = math.sqrt(
        difference[0] ** 2 +
        difference[2] ** 2
    )

    if horizontal_distance < 1e-8:
        return None

    return math.degrees(
        math.atan2(
            abs(difference[1]),
            horizontal_distance
        )
    )


def calculate_torso_lean(landmarks):
    """
    Estimate torso lean relative to vertical
    using the shoulder and hip midpoints.
    """

    left_shoulder = world_point(landmarks, 11)
    right_shoulder = world_point(landmarks, 12)

    left_hip = world_point(landmarks, 23)
    right_hip = world_point(landmarks, 24)

    shoulder_midpoint = (
        left_shoulder + right_shoulder
    ) / 2

    hip_midpoint = (
        left_hip + right_hip
    ) / 2

    torso_vector = shoulder_midpoint - hip_midpoint

    horizontal_distance = math.sqrt(
        torso_vector[0] ** 2 +
        torso_vector[2] ** 2
    )

    vertical_distance = abs(torso_vector[1])

    if np.linalg.norm(torso_vector) < 1e-8:
        return None

    return math.degrees(
        math.atan2(
            horizontal_distance,
            vertical_distance
        )
    )


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

previous_feedback = ""
warning_start = None
active_warning = None

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

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = max(
                int((time.monotonic() - start_time) * 1000),
                last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            shoulder_tilt = None
            torso_lean = None

            feedback = "Move into camera view"
            feedback_color = (0, 255, 255)

            detected_warning = None

            if (
                result.pose_landmarks
                and result.pose_world_landmarks
            ):
                screen_landmarks = result.pose_landmarks[0]
                world_landmarks = result.pose_world_landmarks[0]

                # Draw skeleton connections
                for start, end in POSE_CONNECTIONS:

                    if visible(
                        screen_landmarks,
                        [start, end]
                    ):
                        cv2.line(
                            frame,
                            screen_point(
                                screen_landmarks,
                                start,
                                width,
                                height
                            ),
                            screen_point(
                                screen_landmarks,
                                end,
                                width,
                                height
                            ),
                            (255, 0, 0),
                            3
                        )

                # Draw visible landmarks
                for landmark in screen_landmarks:

                    if landmark.visibility >= MIN_VISIBILITY:
                        cv2.circle(
                            frame,
                            (
                                int(landmark.x * width),
                                int(landmark.y * height)
                            ),
                            5,
                            (0, 255, 0),
                            -1
                        )

                # Need shoulders and hips
                if visible(
                    screen_landmarks,
                    [11, 12, 23, 24]
                ):

                    shoulder_tilt = calculate_shoulder_tilt(
                        world_landmarks
                    )

                    torso_lean = calculate_torso_lean(
                        world_landmarks
                    )

                    if (
                        shoulder_tilt is not None
                        and torso_lean is not None
                    ):

                        if shoulder_tilt > SHOULDER_TILT_LIMIT:
                            detected_warning = (
                                "Check shoulder alignment"
                            )

                        elif torso_lean > TORSO_LEAN_LIMIT:
                            detected_warning = (
                                "Check upper-body lean"
                            )

                        else:
                            feedback = "Posture looks aligned"
                            feedback_color = (0, 255, 0)

                    else:
                        feedback = "Unable to measure posture"

                else:
                    feedback = "Show shoulders and hips"

            # ==================================
            # WARNING TIMER
            # ==================================

            current_time = time.monotonic()

            if detected_warning is None:
                warning_start = None
                active_warning = None

            else:
                if detected_warning != active_warning:
                    active_warning = detected_warning
                    warning_start = current_time

                if (
                    warning_start is not None
                    and current_time - warning_start
                    >= WARNING_DELAY
                ):
                    feedback = detected_warning
                    feedback_color = (0, 0, 255)

                else:
                    feedback = "Checking posture..."
                    feedback_color = (0, 255, 255)

            # ==================================
            # DISPLAY INFORMATION
            # ==================================

            cv2.putText(
                frame,
                "SMART MOTION COACH - POSTURE",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            shoulder_text = (
                f"{shoulder_tilt:.1f} deg"
                if shoulder_tilt is not None
                else "Not available"
            )

            torso_text = (
                f"{torso_lean:.1f} deg"
                if torso_lean is not None
                else "Not available"
            )

            cv2.putText(
                frame,
                f"Shoulder Tilt: {shoulder_text}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Torso Lean: {torso_text}",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # Feedback banner
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

            if text_width > width - 30:
                font_scale *= (
                    width - 30
                ) / text_width

            cv2.putText(
                frame,
                feedback,
                (15, height - 30),
                font,
                font_scale,
                feedback_color,
                2
            )

            # Print feedback changes to terminal
            if feedback != previous_feedback:
                print("Posture Feedback:", feedback)
                previous_feedback = feedback

            cv2.imshow(
                "Smart Motion Coach - Posture Monitor",
                frame
            )

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Posture monitoring stopped.")
