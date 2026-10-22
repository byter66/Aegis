import unittest
import sys
import os
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from fusion_service import AegisRiskFusion, SCENE_MEDIUM_THRESHOLD


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

    def test_low_model_score_does_not_allow_posting_with_detected_pii(self):
        """Outstanding findings block posting even if the model predicts Low."""
        payload = {
            "ocr_ner": {
                "status": "OK",
                "findings": [
                    {"type": "email", "confidence": 0.1, "redacted_text": "[EMAIL]"}
                ],
            }
        }
        with patch.object(self.fusion.model, "predict", return_value=[0]), patch.object(
            self.fusion.model, "predict_proba", return_value=[[1.0, 0.0, 0.0]]
        ):
            result = self.fusion.evaluate_image(payload)

        self.assertEqual(result["risk_level"], "Low")
        self.assertFalse(result["safe_to_post"])

    def test_partial_ocr_analysis_never_allows_posting(self):
        payload = {"ocr_ner": {"status": "PARTIAL", "findings": []}}
        with patch.object(self.fusion.model, "predict", return_value=[0]), patch.object(
            self.fusion.model, "predict_proba", return_value=[[1.0, 0.0, 0.0]]
        ):
            result = self.fusion.evaluate_image(payload)

        self.assertEqual(result["risk_level"], "Low")
        self.assertFalse(result["safe_to_post"])

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

    def test_landmark_positive_detection(self):
        """A high-confidence landmark detection must be included in explainable reasons."""
        payload = {
            "ocr_ner": {"status": "OK", "findings": [], "error": None},
            "landmark": {
                "status": "OK",
                "detected": True,
                "name": "Taj Mahal",
                "confidence": 0.88,
                "error": None
            }
        }
        res = self.fusion.evaluate_image(payload)
        reasons_text = " ".join(res["reasons"])
        self.assertIn("Taj Mahal", reasons_text)
        self.assertIn(res["risk_level"], ["Medium", "High"])

    # -------------------------------------------------------------------------
    # NEW TESTS FOR SCENE RULE, REASON CONSISTENCY, AND RETRAINING PREPARATION
    # -------------------------------------------------------------------------

    def test_scene_high_revealing_raises_low_to_medium(self):
        """scene 0.99, no PII, no landmark -> 'Medium', score 35-59, no reason mentions 'Low'/'Medium' incorrectly."""
        payload = {
            "ocr_ner": {"status": "OK", "findings": [], "error": None},
            "landmark": {"status": "OK", "detected": False, "confidence": 0.0},
            "scene": {"score": 0.99}
        }
        res = self.fusion.evaluate_image(payload)
        self.assertEqual(res["risk_level"], "Medium")
        self.assertEqual(res["color"], "yellow")
        self.assertFalse(res["safe_to_post"])
        self.assertIsNotNone(res["risk_score"])
        self.assertGreaterEqual(res["risk_score"], 35)
        self.assertLessEqual(res["risk_score"], 59)

        reasons_text = " ".join(res["reasons"])
        # Check that scene reason is present
        self.assertIn("The background (signs, building fronts, views) could reveal where this photo was taken.", reasons_text)
        # Check that no reason incorrectly says 'Low'
        self.assertNotIn("Low", reasons_text)
        # Check that the Medium reassurance is appropriately present since final risk is Medium
        self.assertIn("overall privacy risk remains Medium", reasons_text)

    def test_scene_high_with_pii_remains_high(self):
        """scene 0.99 + phone + address -> still 'High'."""
        payload = {
            "ocr_ner": {
                "status": "OK",
                "findings": [
                    {"type": "phone", "confidence": 0.95, "redacted_text": "555-***-0199"},
                    {"type": "address", "confidence": 0.90, "redacted_text": "123 Main St"}
                ]
            },
            "landmark": {"status": "OK", "detected": False, "confidence": 0.0},
            "scene": {"score": 0.99}
        }
        res = self.fusion.evaluate_image(payload)
        self.assertEqual(res["risk_level"], "High")
        self.assertEqual(res["color"], "red")
        self.assertFalse(res["safe_to_post"])
        self.assertGreaterEqual(res["risk_score"], 60)

        reasons_text = " ".join(res["reasons"])
        # Direct PII must be in reasons
        self.assertIn("Residential address", reasons_text)
        self.assertIn("Phone number", reasons_text)
        # Must never contradict High by saying 'overall privacy risk remains Medium'
        self.assertNotIn("overall privacy risk remains Medium", reasons_text)

    def test_scene_low_revealing_stays_low(self):
        """scene 0.30, no PII, no landmark -> stays 'Low'."""
        payload = {
            "ocr_ner": {"status": "OK", "findings": [], "error": None},
            "landmark": {"status": "OK", "detected": False, "confidence": 0.0},
            "scene": {"score": 0.30}
        }
        res = self.fusion.evaluate_image(payload)
        self.assertEqual(res["risk_level"], "Low")
        self.assertEqual(res["color"], "green")
        self.assertTrue(res["safe_to_post"])
        self.assertLess(res["risk_score"], 30)

        reasons_text = " ".join(res["reasons"])
        # No contradiction: must NOT mention Medium
        self.assertNotIn("overall privacy risk remains Medium", reasons_text)
        self.assertNotIn("Medium", reasons_text)
        self.assertIn("No sensitive personal identifiers or distinct locations found.", reasons_text)

    def test_payload_without_scene_key_works_as_before(self):
        """payload without 'scene' key -> works as before (score treated as 0.0)."""
        payload = {
            "ocr_ner": {"status": "OK", "findings": [], "error": None},
            "landmark": {"status": "OK", "detected": False, "confidence": 0.0}
        }
        res = self.fusion.evaluate_image(payload)
        self.assertEqual(res["risk_level"], "Low")
        self.assertEqual(res["color"], "green")
        self.assertTrue(res["safe_to_post"])
        reasons_text = " ".join(res["reasons"])
        self.assertNotIn("overall privacy risk remains Medium", reasons_text)
        self.assertIn("No sensitive personal identifiers or distinct locations found.", reasons_text)

    def test_scene_medium_threshold_constant(self):
        """Verify SCENE_MEDIUM_THRESHOLD constant is exposed and equals 0.60."""
        self.assertEqual(SCENE_MEDIUM_THRESHOLD, 0.60)


if __name__ == "__main__":
    unittest.main()
