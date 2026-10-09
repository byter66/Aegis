"""Smoke tests for the landmark inference interface."""

import unittest
from pathlib import Path

from src.inference import predict_landmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAMPLE_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_dataset"
    / "images"
    / "120734"
    / "0f80cebb3c5d9d2c.jpg"
)


class TestLandmarkInference(unittest.TestCase):

    def test_prediction_output_contract(self):
        """A valid image should return the required output fields."""
        result = predict_landmark(SAMPLE_IMAGE)

        self.assertEqual(
            set(result.keys()),
            {
                "landmark_detected",
                "predicted_landmark",
                "confidence",
            },
        )
        self.assertIsInstance(result["landmark_detected"], bool)
        self.assertIsInstance(result["predicted_landmark"], str)
        self.assertIsInstance(result["confidence"], float)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

    def test_high_threshold_marks_prediction_not_detected(self):
        """A high threshold should mark the sample as not detected."""
        result = predict_landmark(
            SAMPLE_IMAGE,
            confidence_threshold=0.99,
        )

        self.assertFalse(result["landmark_detected"])
        self.assertTrue(result["predicted_landmark"])
        self.assertGreaterEqual(result["confidence"], 0.0)

    def test_missing_image_raises_error(self):
        """A nonexistent image should raise FileNotFoundError."""
        missing_image = PROJECT_ROOT / "missing_test_image.jpg"

        with self.assertRaises(FileNotFoundError):
            predict_landmark(missing_image)

    def test_threshold_below_zero_raises_error(self):
        """A negative confidence threshold should raise ValueError."""
        with self.assertRaisesRegex(ValueError, "confidence_threshold"):
            predict_landmark(
                SAMPLE_IMAGE,
                confidence_threshold=-0.1,
            )

    def test_threshold_above_one_raises_error(self):
        """A confidence threshold above one should raise ValueError."""
        with self.assertRaisesRegex(ValueError, "confidence_threshold"):
            predict_landmark(
                SAMPLE_IMAGE,
                confidence_threshold=1.1,
            )


if __name__ == "__main__":
    unittest.main()