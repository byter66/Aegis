"""Production Risk Fusion Service for AEGIS.
Takes detection outputs from OCR, NER, and Landmark modules and returns
the final color-coded risk assessment and explainable warnings.
"""
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "fusion_model.joblib")


class AegisRiskFusion:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Train it first!")
        self.model = joblib.load(model_path)
        self.classes = ["Low", "Medium", "High"]
        self.feature_columns = [
            "ocr_pii_flag", "num_pii_types", "has_address", 
            "has_phone", "landmark_flag", "landmark_confidence"
        ]

    def evaluate_image(self, detections: dict) -> dict:
        # 1. Safe extraction with default fallbacks
        ocr_flag = bool(detections.get("ocr_pii_flag", False))
        pii_types = detections.get("pii_types", []) or []
        landmark_flag = bool(detections.get("landmark_flag", False))
        landmark_conf = float(detections.get("landmark_confidence", 0.0) or 0.0)

        # Ignore landmarks with negligible confidence (< 0.40)
        if landmark_conf < 0.40:
            landmark_flag = False
            landmark_conf = 0.0

        has_address = int("address" in pii_types)
        has_phone = int("phone" in pii_types)
        num_pii = len(pii_types)

        # 2. Build feature DataFrame with exact column names (removes sklearn warning)
        features = pd.DataFrame([{
            "ocr_pii_flag": int(ocr_flag),
            "num_pii_types": num_pii,
            "has_address": has_address,
            "has_phone": has_phone,
            "landmark_flag": int(landmark_flag),
            "landmark_confidence": landmark_conf
        }], columns=self.feature_columns)

        # 3. Predict class and probabilities
        pred_class_idx = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        risk_level = self.classes[pred_class_idx]
        continuous_score = int(round((probabilities[1] * 50) + (probabilities[2] * 100)))

        # 4. Generate explainable reasons
        reasons = []
        if has_address:
            reasons.append("Residential address or street name detected in image.")
        if has_phone:
            reasons.append("Phone number or contact information detected.")
        if any(t in ["email", "id_number"] for t in pii_types):
            reasons.append("Personal account or identification numbers detected.")
        if landmark_flag and landmark_conf >= 0.60:
            reasons.append(f"Identifiable geographic landmark recognized ({int(landmark_conf * 100)}% match).")
        if not reasons:
            reasons.append("No sensitive personal identifiers or distinct locations found.")

        color_map = {"Low": "green", "Medium": "yellow", "High": "red"}
        safe_to_post = risk_level == "Low"

        return {
            "risk_level": risk_level,
            "risk_score": continuous_score,
            "color": color_map[risk_level],
            "safe_to_post": safe_to_post,
            "reasons": reasons
        }


if __name__ == "__main__":
    fusion = AegisRiskFusion()

    test_scenarios = {
        "Scenario 1: Clean Beach Selfie": {
            "ocr_pii_flag": False,
            "pii_types": [],
            "landmark_flag": False,
            "landmark_confidence": 0.0
        },
        "Scenario 2: Tourist at Eiffel Tower": {
            "ocr_pii_flag": False,
            "pii_types": [],
            "landmark_flag": True,
            "landmark_confidence": 0.88
        },
        "Scenario 3: Delivery Box with Address & Phone": {
            "ocr_pii_flag": True,
            "pii_types": ["address", "phone"],
            "landmark_flag": False,
            "landmark_confidence": 0.0
        }
    }

    print("=" * 65)
    print("      AEGIS RISK FUSION: LIVE INFERENCE DEMO")
    print("=" * 65)
    for name, data in test_scenarios.items():
        res = fusion.evaluate_image(data)
        print(f"\n--- {name} ---")
        print(f"Risk Level   : {res['risk_level']} [{res['color'].upper()}]")
        print(f"Risk Score   : {res['risk_score']}%")
        print(f"Safe to Post : {res['safe_to_post']}")
        print("Reasons      :")
        for r in res['reasons']:
            print(f" - {r}")
