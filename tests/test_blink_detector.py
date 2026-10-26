"""Tests for blink detector logic and side-view robustness."""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

import config
from blink_detector import _dist, _eye_openness, BlinkDetector, LEFT_EYE_IDX, RIGHT_EYE_IDX


class _MockPoint:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def _build_eye_landmarks(gap, width=3.0):
    # 6 points: [corner1, top1, top2, corner2, bottom1, bottom2]
    # width = dist((0,0), (width, 0)) = width
    # lid_gap = ((gap/2 - -gap/2) + (gap/2 - -gap/2)) / 2 = gap
    # ear = gap / width
    return [
        _MockPoint(0.0, 0.0),
        _MockPoint(1.0, gap / 2.0),
        _MockPoint(2.0, gap / 2.0),
        _MockPoint(width, 0.0),
        _MockPoint(2.0, -gap / 2.0),
        _MockPoint(1.0, -gap / 2.0),
    ]


class BlinkDetectorTests(unittest.TestCase):
    def test_dist(self):
        p1 = _MockPoint(0.0, 0.0)
        p2 = _MockPoint(3.0, 4.0)
        self.assertAlmostEqual(_dist(p1, p2), 5.0)

    def test_eye_openness_euclidean(self):
        landmarks = _build_eye_landmarks(gap=2.0, width=3.0)
        ear = _eye_openness(landmarks, [0, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(ear, 2.0 / 3.0)

    @patch("blink_detector.mp_vision.FaceLandmarker.create_from_options")
    def test_side_view_blink_detected_with_min_ear(self, mock_create):
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker
        detector = BlinkDetector()

        # Build full landmark map for 500 points
        def make_face_landmarks(left_gap, right_gap):
            points = [_MockPoint(0, 0) for _ in range(500)]
            left_eye = _build_eye_landmarks(left_gap)
            right_eye = _build_eye_landmarks(right_gap)
            for idx, p in zip(LEFT_EYE_IDX, left_eye):
                points[idx] = p
            for idx, p in zip(RIGHT_EYE_IDX, right_eye):
                points[idx] = p
            return [points]

        # 15 frames of open eyes (gap=1.0 -> ear=1.0/3.0=0.333)
        open_result = MagicMock()
        open_result.face_landmarks = make_face_landmarks(left_gap=1.0, right_gap=1.0)
        mock_landmarker.detect.return_value = open_result

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(15):
            _, blinked, _, debug = detector.process_frame(dummy_frame)
            self.assertFalse(blinked)
            self.assertTrue(debug["is_open"])

        self.assertIsNotNone(detector._baseline)
        self.assertAlmostEqual(detector._baseline, 1.0 / 3.0, places=3)

        # Side-view blink: user looks sideways.
        # Near eye (Right) closes completely: right_gap = 0.15 -> ear = 0.05
        # Far eye (Left) remains distorted/high due to perspective foreshortening: left_gap = 1.0 -> ear = 0.333
        blink_result = MagicMock()
        blink_result.face_landmarks = make_face_landmarks(left_gap=1.0, right_gap=0.15)
        mock_landmarker.detect.return_value = blink_result

        _, blinked, _, debug = detector.process_frame(dummy_frame)
        self.assertFalse(blinked)
        # min(0.333, 0.05) = 0.05. Ratio = 0.05 / 0.333 = 0.15 < 0.60
        self.assertFalse(debug["is_open"])

        # Reopen eyes
        mock_landmarker.detect.return_value = open_result
        _, blinked, _, debug = detector.process_frame(dummy_frame)
        self.assertTrue(blinked, f"expected blink, got {blinked}")
        self.assertTrue(debug["is_open"], f"expected is_open True, got {debug['is_open']}")
        self.assertEqual(detector.blink_count, 1, f"expected blink count 1, got {detector.blink_count}")

    @patch("blink_detector.mp_vision.FaceLandmarker.create_from_options")
    def test_consecutive_closed_frames_threshold(self, mock_create):
        mock_landmarker = MagicMock()
        mock_create.return_value = mock_landmarker
        detector = BlinkDetector()

        def make_face_landmarks(left_gap, right_gap):
            points = [_MockPoint(0, 0) for _ in range(500)]
            left_eye = _build_eye_landmarks(left_gap)
            right_eye = _build_eye_landmarks(right_gap)
            for idx, p in zip(LEFT_EYE_IDX, left_eye):
                points[idx] = p
            for idx, p in zip(RIGHT_EYE_IDX, right_eye):
                points[idx] = p
            return [points]

        open_result = MagicMock()
        open_result.face_landmarks = make_face_landmarks(left_gap=1.0, right_gap=1.0)
        blink_result = MagicMock()
        blink_result.face_landmarks = make_face_landmarks(left_gap=0.1, right_gap=0.1)

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_landmarker.detect.return_value = open_result
        for _ in range(15):
            detector.process_frame(dummy_frame)

        original_consec = config.BLINK_CONSECUTIVE_FRAMES
        try:
            config.BLINK_CONSECUTIVE_FRAMES = 2

            # 1 closed frame then reopen: should NOT trigger blink because threshold is 2
            mock_landmarker.detect.return_value = blink_result
            _, blinked, _, _ = detector.process_frame(dummy_frame)
            self.assertFalse(blinked, f"expected no blink during closure, got {blinked}")

            mock_landmarker.detect.return_value = open_result
            _, blinked, _, _ = detector.process_frame(dummy_frame)
            self.assertFalse(blinked, f"expected no blink for 1 closed frame when threshold is 2, got {blinked}")
            self.assertEqual(detector.blink_count, 0, f"expected 0 blinks, got {detector.blink_count}")

            # 2 consecutive closed frames then reopen: SHOULD trigger blink
            mock_landmarker.detect.return_value = blink_result
            detector.process_frame(dummy_frame)
            detector.process_frame(dummy_frame)

            mock_landmarker.detect.return_value = open_result
            _, blinked, _, _ = detector.process_frame(dummy_frame)
            self.assertTrue(blinked, f"expected blink for 2 closed frames, got {blinked}")
            self.assertEqual(detector.blink_count, 1, f"expected 1 blink, got {detector.blink_count}")
        finally:
            config.BLINK_CONSECUTIVE_FRAMES = original_consec

    def test_no_face_frame_interval_config(self):
        self.assertTrue(hasattr(config, "PROCESS_EVERY_N_FRAMES_NO_FACE"))
        self.assertEqual(config.PROCESS_EVERY_N_FRAMES_NO_FACE, 30, f"expected 30, got {config.PROCESS_EVERY_N_FRAMES_NO_FACE}")
        self.assertIn("PROCESS_EVERY_N_FRAMES_NO_FACE", config._PERSIST_KEYS)

    def test_dynamic_interval_logic(self):
        # Verify interval switching based on face detection flag
        face_detected = True
        interval = max(
            1,
            config.PROCESS_EVERY_N_FRAMES
            if face_detected
            else getattr(config, "PROCESS_EVERY_N_FRAMES_NO_FACE", 30),
        )
        self.assertEqual(interval, config.PROCESS_EVERY_N_FRAMES, f"expected {config.PROCESS_EVERY_N_FRAMES}, got {interval}")

        face_detected = False
        interval = max(
            1,
            config.PROCESS_EVERY_N_FRAMES
            if face_detected
            else getattr(config, "PROCESS_EVERY_N_FRAMES_NO_FACE", 30),
        )
        self.assertEqual(interval, 30, f"expected 30, got {interval}")


if __name__ == "__main__":
    unittest.main()
