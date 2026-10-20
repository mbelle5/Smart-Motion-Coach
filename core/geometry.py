
"""Geometry functions shared by the exercise trackers.

MediaPipe estimates joint positions in 3D space.
The exercise angle thresholds are experimental.
"""

import numpy as np

MIN_VISIBILITY = 0.5


def calculate_angle(a, b, c):
    """Calculate the angle at point b using three coordinates.

    Returns the angle in degrees, or None if the points
    do not form a valid angle.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    # Create two vectors that meet at the middle joint.
    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:
        return None

    # The dot product gives the cosine of the angle.
    # Clipping prevents rounding errors in arccos.
    cosine = np.clip(
        np.dot(ba, bc) / denominator, -1.0, 1.0
    )

    return float(np.degrees(np.arccos(cosine)))


def world_point(landmarks, index):
    """Get the estimated 3D coordinates of a body landmark."""
    point = landmarks[index]
    return [point.x, point.y, point.z]


def screen_point(landmarks, index, width, height):
    """Convert normalized coordinates to camera pixel positions."""
    point = landmarks[index]

    return (
        int(point.x * width),
        int(point.y * height),
    )


def visible(landmarks, indices, minimum=MIN_VISIBILITY):
    """Check that every required body landmark is visible."""
    return all(
        landmarks[i].visibility >= minimum
        for i in indices
    )


def leg_confidence(landmarks, indices):
    """Return the lowest visibility score among selected leg joints."""
    return min(
        landmarks[i].visibility
        for i in indices
    )
