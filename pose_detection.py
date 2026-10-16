
import cv2
import mediapipe as mp
import time

# Load MediaPipe pose detection model
model_path = "pose_landmarker_lite.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Connections between body landmarks
POSE_CONNECTIONS = [
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (11, 23), (12, 24),  # Torso
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (27, 29), (29, 31),  # Left foot
    (28, 30), (30, 32)   # Right foot
]

# Configure MediaPipe
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

# Open MacBook camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit

start_time = time.monotonic()
last_timestamp_ms = -1

try:
    with PoseLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to capture frame.")
                break

            # Flip camera image
            frame = cv2.flip(frame, 1)

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Create increasing timestamps
            timestamp_ms = max(
                int((time.monotonic() - start_time) * 1000),
                last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            # Detect body landmarks
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.pose_landmarks:
                height, width = frame.shape[:2]
                landmarks = result.pose_landmarks[0]

                # Draw lines connecting body joints
                for start, end in POSE_CONNECTIONS:
                    point1 = landmarks[start]
                    point2 = landmarks[end]

                    x1 = int(point1.x * width)
                    y1 = int(point1.y * height)
                    x2 = int(point2.x * width)
                    y2 = int(point2.y * height)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 0),
                        3
                    )

                # Draw green dots on body joints
                for landmark in landmarks:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

            # Display video
            cv2.imshow(
                "Smart Motion Coach - Skeleton Tracking",
                frame
            )

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Skeleton tracking stopped.")
