import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from fusion_service import AegisRiskFusion


class TestAegisRiskFusionService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fusion = AegisRiskFusion()

    def test_nested_clean_image(self):
        """A clean image with nested payload must be Low Risk and safe to post."""
        payload = {
            "ocr_ner": {"status": "OK", "findings": [], "error": None},
            "landmark": {"status": "OK", "detected": False, "confidence": 0.0}
        }
        res = self.fusion.evaluate_image(payload)
        self.assertEqual(res["risk_level"], "Low")
        self.assertEqual(res["color"], "green")
        self.assertTrue(res["safe_to_post"])

    def test_nested_pii_with_phone_and_name(self):
        """Phone and person_name must be classified as High risk and preserve UI findings."""
        payload = {
            "ocr_ner": {
                "status": "OK",
                "findings": [
                    {"type": "phone", "confidence": 0.95, "redacted_text": "555-***-0199", "bbox": [10, 20, 100, 40]},
                    {"type": "person_name", "confidence": 0.90, "redacted_text": "J*** D***", "bbox": [10, 50, 100, 70]}
                ]
            },
            "landmark": {"status": "OK", "detected": False, "confidence": 0.0}
        }
        res = self.fusion.evaluate_image(payload)
        self.assertEqual(res["risk_level"], "High")
        self.assertEqual(res["color"], "red")
        self.assertFalse(res["safe_to_post"])
        # Verify findings are preserved for UI
        self.assertEqual(len(res["findings"]), 2)
        self.assertEqual(res["findings"][0]["bbox"], [10, 20, 100, 40])
        # Verify explainable reason includes name and phone
        reasons_text = " ".join(res["reasons"])
        self.assertIn("Phone number", reasons_text)
        self.assertIn("personal name", reasons_text)

    def test_deduplication_of_repeated_pii(self):
        """Multiple instances of the same PII type must not double-count categories."""
        payload = {
            "ocr_ner": {
                "status": "OK",
                "findings": [
                    {"type": "phone", "confidence": 0.95, "redacted_text": "555-***-0199"},
                    {"type": "phone", "confidence": 0.92, "redacted_text": "555-***-0200"}
                ]
            }
        }
        res = self.fusion.evaluate_image(payload)
        self.assertIn(res["risk_level"], ["Medium", "High"])

    def test_detector_error_returns_unknown_risk(self):
        """An ERROR status from upstream must return Unknown risk and safe_to_post False."""
        payload = {
            "ocr_ner": {
                "status": "ERROR",
                "findings": [],
                "error": {"code": "IMAGE_CORRUPT", "message": "Corrupted image file."}
            }
        }
        res = self.fusion.evaluate_image(payload)
        self.assertEqual(res["risk_level"], "Unknown")
        self.assertIsNone(res["risk_score"])
        self.assertEqual(res["status"], "DETECTOR_ERROR")
        self.assertFalse(res["safe_to_post"])
        self.assertIn("Corrupted image file", res["reasons"][0])

    def test_backward_compatibility_flat_dict(self):
        """Legacy flat dictionary inputs must continue to work without errors."""
        flat = {
            "ocr_pii_flag": True,
            "pii_types": ["phone"],
            "landmark_flag": False,
            "landmark_confidence": 0.0
        }
        res = self.fusion.evaluate_image(flat)
        self.assertIn(res["risk_level"], ["Medium", "High"])


if __name__ == "__main__":
    unittest.main()
