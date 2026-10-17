import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from fusion_service import AegisRiskFusion


class TestAegisRiskFusionService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load model once for all tests
        cls.fusion = AegisRiskFusion()

    def test_clean_image_is_safe_to_post(self):
        """A clean image must be safe to post."""
        clean = {"ocr_pii_flag": False, "pii_types": [], "landmark_flag": False}
        res = self.fusion.evaluate_image(clean)
        self.assertEqual(res["risk_level"], "Low")
        self.assertEqual(res["color"], "green")
        self.assertTrue(res["safe_to_post"])
        self.assertLessEqual(res["risk_score"], 20)

    def test_critical_leak_is_not_safe(self):
        """A phone number and address must be High risk and unsafe to post."""
        leak = {"ocr_pii_flag": True, "pii_types": ["address", "phone"]}
        res = self.fusion.evaluate_image(leak)
        self.assertEqual(res["risk_level"], "High")
        self.assertEqual(res["color"], "red")
        self.assertFalse(res["safe_to_post"])
        self.assertGreaterEqual(res["risk_score"], 70)

    def test_missing_keys_fallback(self):
        """Passing an empty dictionary must not crash (defensive coding)."""
        empty = {}
        res = self.fusion.evaluate_image(empty)
        self.assertIn(res["risk_level"], ["Low", "Medium", "High"])
        self.assertIsInstance(res["risk_score"], int)


if __name__ == "__main__":
    unittest.main()
