
import cv2
import mediapipe as mp
import numpy as np
import time

# ==========================================
# SMART MOTION COACH - ARM RAISE TRACKER
# ==========================================

MODEL_PATH = "pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Skeleton connections
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28)
]

# Experimental angle thresholds
ARMS_DOWN_ANGLE = 25
ARMS_UP_ANGLE = 80
MIN_VISIBILITY = 0.5

# ==========================================
# ANGLE CALCULATION
# ==========================================

def calculate_angle(a, b, c):
    """Calculate angle at point b using 3D coordinates."""

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return None

    cosine_angle = np.dot(ba, bc) / denominator

    return float(
        np.degrees(
            np.arccos(
                np.clip(cosine_angle, -1.0, 1.0)
            )
        )
    )


def world_point(landmarks, index):
    landmark = landmarks[index]
    return [landmark.x, landmark.y, landmark.z]


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


# ==========================================
# TRACKING VARIABLES
# ==========================================

rep_count = 0
arm_stage = "down"
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

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
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

                # Draw skeleton
                for start, end in POSE_CONNECTIONS:
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

                # Draw visible body landmarks
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

                # Left shoulder angle:
                # Elbow -> Shoulder -> Hip
                if visible(screen_landmarks, [13, 11, 23]):
                    left_angle = calculate_angle(
                        world_point(world_landmarks, 13),
                        world_point(world_landmarks, 11),
                        world_point(world_landmarks, 23)
                    )

                # Right shoulder angle:
                # Elbow -> Shoulder -> Hip
                if visible(screen_landmarks, [14, 12, 24]):
                    right_angle = calculate_angle(
                        world_point(world_landmarks, 14),
                        world_point(world_landmarks, 12),
                        world_point(world_landmarks, 24)
                    )

                # Both arms must be visible
                if (
                    left_angle is not None
                    and right_angle is not None
                ):
                    both_down = (
                        left_angle < ARMS_DOWN_ANGLE
                        and right_angle < ARMS_DOWN_ANGLE
                    )

                    both_up = (
                        left_angle > ARMS_UP_ANGLE
                        and right_angle > ARMS_UP_ANGLE
                    )

                    # Start a repetition when both arms
                    # reach the raised position
                    if both_up and arm_stage == "down":
                        arm_stage = "up"
                        feedback = "Target reached!"
                        feedback_color = (0, 255, 0)

                    # Count a repetition when both arms
                    # return to the lowered position
                    elif both_down and arm_stage == "up":
                        arm_stage = "down"
                        rep_count += 1
                        feedback_until = time.monotonic() + 1.5

                        print("Arm raise completed:", rep_count)

                    # Live feedback
                    if time.monotonic() < feedback_until:
                        feedback = "Rep completed!"
                        feedback_color = (0, 255, 0)

                    elif both_up:
                        feedback = "Target reached! Lower arms"
                        feedback_color = (0, 255, 0)

                    elif arm_stage == "up":
                        feedback = "Lower both arms"
                        feedback_color = (0, 255, 255)

                    elif both_down:
                        feedback = "Raise both arms sideways"
                        feedback_color = (255, 255, 255)

                    else:
                        feedback = "Raise your arms higher"
                        feedback_color = (0, 255, 255)

                else:
                    feedback = "Show both shoulders and arms"

            # ==================================
            # DISPLAY INFORMATION
            # ==================================

            cv2.putText(
                frame,
                f"Arm Raises: {rep_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Stage: {arm_stage}",
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
                f"Left Shoulder: {left_text}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Right Shoulder: {right_text}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
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
                feedback, font, font_scale, 2
            )[0][0]

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

            # Print feedback when it changes
            if feedback != previous_feedback:
                print("Exercise Feedback:", feedback)
                previous_feedback = feedback

            # Show webcam
            cv2.imshow(
                "Smart Motion Coach - Arm Raise Tracker",
                frame
            )

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Total arm raises completed:", rep_count)
