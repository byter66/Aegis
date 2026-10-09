"""Tests for the landmark inference interface."""

import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image
import torch

# Ensure the landmark-recognition package root is on sys.path for test portability.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import CHECKPOINT_PATH, predict_landmark

SAMPLE_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_dataset"
    / "images"
    / "120734"
    / "0f80cebb3c5d9d2c.jpg"
)


class _DeterministicModelStub(torch.nn.Module):
    """Deterministic model stub returning fixed logits for unit testing."""

    def forward(self, input_tensor):
        return torch.tensor([[5.0, 1.0]])


class TestLandmarkInferenceUnit(unittest.TestCase):
    """Unit tests using lightweight fixtures and model stubs without requiring trained weights."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_image_path = Path(cls.temp_dir.name) / "test_fixture.jpg"
        image = Image.new("RGB", (32, 32), color="blue")
        image.save(cls.temp_image_path)

        cls.model = _DeterministicModelStub()
        cls.class_mapping = {"120734": 0, "101399": 1}

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_prediction_output_contract(self):
        """A valid image should return the required output fields with correct types."""
        result = predict_landmark(
            self.temp_image_path,
            model=self.model,
            class_mapping=self.class_mapping,
        )

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
        self.assertTrue(result["landmark_detected"])
        self.assertEqual(result["predicted_landmark"], "120734")

    def test_high_threshold_marks_prediction_not_detected(self):
        """A threshold higher than model confidence should mark landmark as not detected."""
        result = predict_landmark(
            self.temp_image_path,
            model=self.model,
            class_mapping=self.class_mapping,
            confidence_threshold=0.999,
        )

        self.assertFalse(result["landmark_detected"])
        self.assertEqual(result["predicted_landmark"], "120734")
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
                self.temp_image_path,
                confidence_threshold=-0.1,
            )

    def test_threshold_above_one_raises_error(self):
        """A confidence threshold above one should raise ValueError."""
        with self.assertRaisesRegex(ValueError, "confidence_threshold"):
            predict_landmark(
                self.temp_image_path,
                confidence_threshold=1.1,
            )

    def test_boundary_thresholds_accepted(self):
        """Boundary thresholds 0.0 and 1.0 should be accepted without raising ValueError."""
        result_zero = predict_landmark(
            self.temp_image_path,
            model=self.model,
            class_mapping=self.class_mapping,
            confidence_threshold=0.0,
        )
        self.assertTrue(result_zero["landmark_detected"])

        result_one = predict_landmark(
            self.temp_image_path,
            model=self.model,
            class_mapping=self.class_mapping,
            confidence_threshold=1.0,
        )
        self.assertFalse(result_one["landmark_detected"])


class TestLandmarkInferenceSmoke(unittest.TestCase):
    """Real-model smoke tests requiring the trained checkpoint and sample image."""

    def test_real_model_prediction_smoke(self):
        """Smoke test executing inference with the trained EfficientNet-B0 model."""
        if not (SAMPLE_IMAGE.exists() and CHECKPOINT_PATH.exists()):
            self.skipTest("Requires local model checkpoint and sample dataset image")

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


if __name__ == "__main__":
    unittest.main()
