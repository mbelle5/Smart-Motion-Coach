
import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe pose detection model
model_path = "pose_landmarker_lite.task"

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
    (24, 26), (26, 28),
    (27, 29), (29, 31),
    (28, 30), (30, 32)
]

# Function to calculate the angle between 3 points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle


# Configure MediaPipe
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
    raise SystemExit

start_time = time.monotonic()
last_timestamp_ms = -1

try:
    with PoseLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

            if result.pose_landmarks:
                height, width = frame.shape[:2]
                landmarks = result.pose_landmarks[0]

                # Draw skeleton connections
                for start, end in POSE_CONNECTIONS:
                    p1 = landmarks[start]
                    p2 = landmarks[end]

                    cv2.line(
                        frame,
                        (int(p1.x * width), int(p1.y * height)),
                        (int(p2.x * width), int(p2.y * height)),
                        (255, 0, 0),
                        3
                    )

                # Draw body landmarks
                for landmark in landmarks:
                    cv2.circle(
                        frame,
                        (int(landmark.x * width),
                         int(landmark.y * height)),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Left arm landmarks
                shoulder = [
                    landmarks[11].x,
                    landmarks[11].y
                ]

                elbow = [
                    landmarks[13].x,
                    landmarks[13].y
                ]

                wrist = [
                    landmarks[15].x,
                    landmarks[15].y
                ]

                # Calculate elbow angle
                angle = calculate_angle(
                    shoulder,
                    elbow,
                    wrist
                )

                # Display angle near elbow
                elbow_position = (
                    int(elbow[0] * width),
                    int(elbow[1] * height)
                )

                cv2.putText(
                    frame,
                    f"{int(angle)} deg",
                    elbow_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

                # Display angle at top of screen
                cv2.putText(
                    frame,
                    f"Left Elbow Angle: {int(angle)}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            cv2.imshow(
                "Smart Motion Coach - Joint Angles",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Joint angle tracking stopped.")
