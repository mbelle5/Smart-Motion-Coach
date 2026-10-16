
import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe model
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

# Calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return 0

    cosine_angle = np.dot(ba, bc) / denominator

    angle = np.degrees(
        np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    )

    return angle


# Squat tracking variables
squat_count = 0
squat_stage = "up"

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
                print("Failed to capture frame.")
                break

            # Mirror camera
            frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]

            # Convert to RGB
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Video timestamp
            timestamp_ms = max(
                int((time.monotonic() - start_time) * 1000),
                last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            # Detect pose
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

                # Draw skeleton
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

                # Draw landmarks
                for landmark in landmarks:
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

                # Left leg landmarks
                hip = [
                    landmarks[23].x,
                    landmarks[23].y
                ]

                knee = [
                    landmarks[25].x,
                    landmarks[25].y
                ]

                ankle = [
                    landmarks[27].x,
                    landmarks[27].y
                ]

                # Only calculate when leg is visible
                leg_visible = all(
                    landmarks[i].visibility > 0.6
                    for i in [23, 25, 27]
                )

                if leg_visible:
                    knee_angle = calculate_angle(
                        hip,
                        knee,
                        ankle
                    )

                    # Detect squat going down
                    if knee_angle < 100:
                        squat_stage = "down"

                    # Detect standing back up
                    if knee_angle > 160 and squat_stage == "down":
                        squat_stage = "up"
                        squat_count += 1
                        print("Squat completed:", squat_count)

                    # Display knee angle
                    cv2.putText(
                        frame,
                        f"Knee Angle: {int(knee_angle)}",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                    # Display angle near knee
                    cv2.putText(
                        frame,
                        f"{int(knee_angle)} deg",
                        (
                            int(knee[0] * width),
                            int(knee[1] * height)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2
                    )

            # Display squat count
            cv2.putText(
                frame,
                f"Squats: {squat_count}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

            # Display stage
            cv2.putText(
                frame,
                f"Stage: {squat_stage}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            # Show webcam
            cv2.imshow(
                "Smart Motion Coach - Squat Detector",
                frame
            )

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Total squats completed:", squat_count)
