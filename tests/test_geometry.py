
"""Unit tests for shared geometry functions."""

import unittest

from core.geometry import (
    calculate_angle,
    visible,
    leg_confidence,
)


class DummyLandmark:
    """Simple test landmark with a visibility score."""

    def __init__(self, visibility):
        self.visibility = visibility


class GeometryTests(unittest.TestCase):

    def test_right_angle(self):
        # Perpendicular vectors should form a 90-degree angle.
        angle = calculate_angle(
            [1, 0, 0],
            [0, 0, 0],
            [0, 1, 0],
        )
        self.assertAlmostEqual(angle, 90)

    def test_straight_angle(self):
        # Opposite vectors should form a 180-degree angle.
        angle = calculate_angle(
            [-1, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        )
        self.assertAlmostEqual(angle, 180)

    def test_degenerate_points(self):
        # Identical joint positions cannot form a valid angle.
        angle = calculate_angle(
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
        )
        self.assertIsNone(angle)

    def test_visibility(self):
        points = [
            DummyLandmark(0.8),
            DummyLandmark(0.4),
        ]

        self.assertTrue(visible(points, [0]))
        self.assertFalse(visible(points, [0, 1]))
        self.assertEqual(
            leg_confidence(points, [0, 1]),
            0.4,
        )


if __name__ == "__main__":
    unittest.main()
