
"""Shared MediaPipe setup and skeleton drawing functions."""

from pathlib import Path

import cv2
import mediapipe as mp

from core.geometry import screen_point, visible


# ==========================================
# MEDIAPIPE MODEL CONFIGURATION
# ==========================================

# Locate the model in the main project folder.
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "pose_landmarker_lite.task"
)

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
    (28, 30), (30, 32),
]


# ==========================================
# POSE DETECTOR SETUP
# ==========================================

def create_pose_landmarker():
    """Create a MediaPipe pose detector for webcam video."""

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
    )

    return mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    )


# ==========================================
# SKELETON DRAWING
# ==========================================

def draw_skeleton(frame, landmarks, minimum_visibility=0.5):
    """
    Draw visible body landmarks and their connections.

    frame: OpenCV camera frame.
    landmarks: MediaPipe pose landmarks.
    minimum_visibility: Minimum confidence required to draw a joint.
    """

    height, width = frame.shape[:2]

    # Only connect joints that are sufficiently visible.
    for start, end in POSE_CONNECTIONS:

        if visible(
            landmarks,
            (start, end),
            minimum_visibility
        ):
            cv2.line(
                frame,
                screen_point(landmarks, start, width, height),
                screen_point(landmarks, end, width, height),
                (255, 0, 0),
                3,
            )

    # Draw a green circle at each visible joint.
    for point in landmarks:

        if point.visibility >= minimum_visibility:
            cv2.circle(
                frame,
                (
                    int(point.x * width),
                    int(point.y * height)
                ),
                5,
                (0, 255, 0),
                -1,
            )
