import unittest
import sys
import os

# Add the src folder to Python path so it can import baseline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from baseline import rule_based_risk


class TestBaselineRisk(unittest.TestCase):

    def test_clean_image(self):
        """A safe image with no PII and no landmark should be Low risk (0 points)."""
        sample = {
            "ocr_pii_flag": False,
            "pii_types": [],
            "landmark_flag": False,
            "landmark_confidence": 0.0
        }
        score, label = rule_based_risk(sample)
        self.assertEqual(score, 0)
        self.assertEqual(label, "low")

    def test_landmark_only(self):
        """A landmark with high confidence should produce a medium/low score."""
        sample = {
            "ocr_pii_flag": False,
            "pii_types": [],
            "landmark_flag": True,
            "landmark_confidence": 0.9
        }
        score, label = rule_based_risk(sample)
        self.assertEqual(score, 27)
        self.assertEqual(label, "low")

    def test_high_risk_address_and_phone(self):
        """An address and phone together must trigger High risk."""
        sample = {
            "ocr_pii_flag": True,
            "pii_types": ["address", "phone"],
            "landmark_flag": True,
            "landmark_confidence": 0.8
        }
        score, label = rule_based_risk(sample)
        self.assertGreaterEqual(score, 60)
        self.assertEqual(label, "high")

    def test_score_cap_at_100(self):
        """Score should never exceed 100 even with many leaks."""
        sample = {
            "ocr_pii_flag": True,
            "pii_types": ["address", "phone", "email", "name"],
            "landmark_flag": True,
            "landmark_confidence": 1.0
        }
        score, label = rule_based_risk(sample)
        self.assertEqual(score, 100)
        self.assertEqual(label, "high")


if __name__ == "__main__":
    unittest.main()
