import os, sys, tempfile, unittest
from unittest import mock
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import landmark

def _fake(flag, conf, name, src="cnn", scene=0.5):
    return {"landmark_flag": flag, "landmark_confidence": conf, "landmark_name": name,
            "landmark_source": src, "location_revealing_scene": scene}

def _payload(fake):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.jpg")
        Image.new("RGB", (64, 64), "white").save(p)
        with mock.patch.object(landmark, "landmark_module", return_value=fake):
            return landmark.landmark_payload(p)

class LandmarkPayloadTests(unittest.TestCase):
    def test_missing_image_returns_error_and_never_raises(self):
        out = landmark.landmark_payload("/nonexistent.jpg")
        self.assertEqual(out["landmark"]["status"], "ERROR")
        self.assertFalse(out["landmark"]["detected"])
        self.assertIsNone(out["scene"]["score"])

    def test_payload_structure(self):
        out = _payload(_fake(True, 0.88, "Category:Lotus_Temple"))
        self.assertIn("landmark", out); self.assertIn("scene", out)
        for k in ["status", "detected", "name", "confidence", "error", "source", "raw_confidence"]:
            self.assertIn(k, out["landmark"])
        self.assertEqual(out["landmark"]["name"], "Lotus Temple")

    def test_confidence_below_floor_is_not_detected(self):
        out = _payload(_fake(True, 0.35, "Lotus_Temple"))
        self.assertFalse(out["landmark"]["detected"])
        self.assertEqual(out["landmark"]["confidence"], 0.0)
        self.assertIsNone(out["landmark"]["name"])
        self.assertEqual(out["landmark"]["raw_confidence"], 0.35)

    def test_source_is_reported(self):
        self.assertEqual(_payload(_fake(True, 0.9, "India_Gate", "cnn"))["landmark"]["source"], "cnn")
        self.assertEqual(_payload(_fake(True, 0.9, "India Gate", "clip"))["landmark"]["source"], "clip")

    def test_missing_dependency_gives_structured_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.jpg"); Image.new("RGB", (8, 8)).save(p)
            with mock.patch.object(landmark, "landmark_module", side_effect=ImportError("no tf")):
                out = landmark.landmark_payload(p)
        self.assertEqual(out["landmark"]["error"]["code"], "DEPENDENCY_MISSING")

if __name__ == "__main__":
    unittest.main()