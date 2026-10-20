
import cv2
import mediapipe as mp
import time
from pathlib import Path

# Shared functions for calculating angles and accessing body landmarks.
from core.geometry import (
    calculate_angle,
    world_point,
    screen_point,
    visible,
)


# ==========================================
# SMART MOTION COACH - BICEP CURL TRACKER
# ==========================================

# Locate the MediaPipe model relative to this script.
MODEL_PATH = str(
    Path(__file__).resolve().parent / "pose_landmarker_lite.task"
)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# Body connections used to draw the skeleton.
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]


# Experimental elbow-angle thresholds.
CURL_UP_ANGLE = 60
CURL_DOWN_ANGLE = 150
MIN_VISIBILITY = 0.5


# ==========================================
# TRACKING VARIABLES
# ==========================================

rep_count = 0
curl_stage = "down"
previous_feedback = ""
feedback_until = 0


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

            # Mirror the camera to make movement easier to follow.
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            # Convert OpenCV's BGR frame into RGB for MediaPipe.
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Ensure timestamps increase for each video frame.
            timestamp_ms = max(
                int((time.monotonic() - start_time) * 1000),
                last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            # Estimate the user's pose from the webcam frame.
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            feedback = "Move into camera view"
            feedback_color = (0, 255, 255)

            left_angle = None
            right_angle = None

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

                    # Skip joints that are not sufficiently visible.
                    if visible(screen_landmarks, [start, end]):
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

                # Mark visible body joints with green circles.
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

                # ==================================
                # CALCULATE ELBOW ANGLES
                # ==================================

                # Left elbow: Shoulder -> Elbow -> Wrist
                if visible(screen_landmarks, [11, 13, 15]):
                    left_angle = calculate_angle(
                        world_point(world_landmarks, 11),
                        world_point(world_landmarks, 13),
                        world_point(world_landmarks, 15)
                    )

                # Right elbow: Shoulder -> Elbow -> Wrist
                if visible(screen_landmarks, [12, 14, 16]):
                    right_angle = calculate_angle(
                        world_point(world_landmarks, 12),
                        world_point(world_landmarks, 14),
                        world_point(world_landmarks, 16)
                    )

                # ==================================
                # BICEP CURL COUNTING
                # ==================================

                # Both arms must be visible to track a full repetition.
                if (
                    left_angle is not None
                    and right_angle is not None
                ):
                    both_curled = (
                        left_angle < CURL_UP_ANGLE
                        and right_angle < CURL_UP_ANGLE
                    )

                    both_extended = (
                        left_angle > CURL_DOWN_ANGLE
                        and right_angle > CURL_DOWN_ANGLE
                    )

                    # Detect when both elbows reach the curled position.
                    if both_curled and curl_stage == "down":
                        curl_stage = "up"

                    # Count a rep when both arms return to full extension.
                    elif both_extended and curl_stage == "up":
                        curl_stage = "down"
                        rep_count += 1
                        feedback_until = time.monotonic() + 1.5

                        print("Bicep curl completed:", rep_count)

                    # ==================================
                    # EXERCISE FEEDBACK
                    # ==================================

                    if time.monotonic() < feedback_until:
                        feedback = "Rep completed!"
                        feedback_color = (0, 255, 0)

                    elif both_curled:
                        feedback = "Target reached! Lower arms"
                        feedback_color = (0, 255, 0)

                    elif curl_stage == "up":
                        feedback = "Lower your arms"
                        feedback_color = (0, 255, 255)

                    elif both_extended:
                        feedback = "Curl both arms upward"
                        feedback_color = (255, 255, 255)

                    else:
                        feedback = "Curl your arms higher"
                        feedback_color = (0, 255, 255)

                else:
                    feedback = "Show both arms to camera"

            # ==================================
            # DISPLAY INFORMATION
            # ==================================

            cv2.putText(
                frame,
                f"Bicep Curls: {rep_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Stage: {curl_stage}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            left_text = (
                f"{left_angle:.0f} deg"
                if left_angle is not None
                else "Not visible"
            )

            right_text = (
                f"{right_angle:.0f} deg"
                if right_angle is not None
                else "Not visible"
            )

            cv2.putText(
                frame,
                f"Left Elbow: {left_text}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Right Elbow: {right_text}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
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

            # Shrink feedback text if it is wider than the camera window.
            if text_width > width - 30:
                font_scale *= (width - 30) / text_width

            cv2.putText(
                frame,
                feedback,
                (15, height - 30),
                font,
                font_scale,
                feedback_color,
                2
            )

            # Avoid repeating identical feedback in the terminal.
            if feedback != previous_feedback:
                print("Exercise Feedback:", feedback)
                previous_feedback = feedback

            # Display the camera window.
            cv2.imshow(
                "Smart Motion Coach - Bicep Curl Tracker",
                frame
            )

            # Press Q to close the tracker.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    # Always release the webcam when the program exits.
    camera.release()
    cv2.destroyAllWindows()

print("Total bicep curls completed:", rep_count)
