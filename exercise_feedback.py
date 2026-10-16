
import cv2
import mediapipe as mp
import numpy as np
import time

# Load MediaPipe model
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

# Calculate joint angle
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator == 0:
        return None

    cosine_angle = np.dot(ba, bc) / denominator

    angle = np.degrees(
        np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    )

    return angle


# Squat tracking variables
squat_count = 0
squat_stage = "up"

# Feedback tracking variables
feedback = "Stand straight to begin"
previous_feedback = ""
feedback_until = 0

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

            # Convert camera frame to RGB
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Generate video timestamp
            timestamp_ms = max(
                int((time.monotonic() - start_time) * 1000),
                last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            # Detect body pose
            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # Default feedback
            feedback = "Move into camera view"
            feedback_color = (0, 255, 255)

            if result.pose_landmarks:
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

                # Draw green body landmarks
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

                # Left leg landmark indices
                hip_index = 23
                knee_index = 25
                ankle_index = 27

                # Check if full leg is visible
                leg_visible = all(
                    landmarks[i].visibility > 0.6
                    for i in [
                        hip_index,
                        knee_index,
                        ankle_index
                    ]
                )

                if leg_visible:
                    hip = [
                        landmarks[hip_index].x,
                        landmarks[hip_index].y
                    ]

                    knee = [
                        landmarks[knee_index].x,
                        landmarks[knee_index].y
                    ]

                    ankle = [
                        landmarks[ankle_index].x,
                        landmarks[ankle_index].y
                    ]

                    # Calculate knee angle
                    knee_angle = calculate_angle(
                        hip,
                        knee,
                        ankle
                    )

                    if knee_angle is not None:

                        # Detect lowered squat position
                        if knee_angle < 100:
                            squat_stage = "down"

                        # Detect completed squat
                        if (
                            knee_angle > 160
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

                        # Give real-time exercise feedback
                        if time.monotonic() < feedback_until:
                            feedback = "Rep completed!"
                            feedback_color = (0, 255, 0)

                        elif knee_angle < 100:
                            feedback = "Target depth reached!"
                            feedback_color = (0, 255, 0)

                        elif knee_angle <= 160:
                            if squat_stage == "down":
                                feedback = "Stand back up!"
                            else:
                                feedback = "Bend your knees more"

                            feedback_color = (0, 255, 255)

                        else:
                            feedback = "Standing position"
                            feedback_color = (255, 255, 255)

                        # Display knee angle
                        cv2.putText(
                            frame,
                            f"Knee Angle: {int(knee_angle)}",
                            (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
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
                            0.6,
                            (0, 255, 255),
                            2
                        )

                else:
                    feedback = "Show your full leg"
                    feedback_color = (0, 255, 255)

            # Display squat counter
            cv2.putText(
                frame,
                f"Squats: {squat_count}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )

            # Display squat stage
            cv2.putText(
                frame,
                f"Stage: {squat_stage}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Display feedback banner
            banner_height = 85

            cv2.rectangle(
                frame,
                (0, height - banner_height),
                (width, height),
                (30, 30, 30),
                -1
            )

            # Automatically scale feedback text to fit
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9

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

            # NEW: Print feedback to terminal
            # Only print when the message changes
            if feedback != previous_feedback:
                print("Exercise Feedback:", feedback)
                previous_feedback = feedback

            # Display webcam window
            cv2.imshow(
                "Smart Motion Coach - Exercise Feedback",
                frame
            )

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Total squats completed:", squat_count)
