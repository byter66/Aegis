"""Production Risk Fusion Service for AEGIS.
Consumes structured detection outputs from OCR/NER and Landmark modules,
aggregates PII, handles detector errors, and produces an explainable privacy risk assessment.
"""
import joblib
import pandas as pd
import numpy as np
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

    def evaluate_image(self, payload: dict) -> dict:
        """Evaluates an image payload and returns the final risk assessment.

        Accepts both nested format:
        {
            "ocr_ner": {"status": "OK", "findings": [...], "error": None},
            "landmark": {"status": "OK", "detected": True, "name": "...", "confidence": 0.85}
        }
        and legacy flat format for backward compatibility.
        """
        # 1. Check for Detector Errors (Safe Fallback)
        ocr_ner_data = payload.get("ocr_ner", {})
        landmark_data = payload.get("landmark", {})

        if ocr_ner_data.get("status") == "ERROR" or landmark_data.get("status") == "ERROR":
            err_msg = None
            if ocr_ner_data.get("error"):
                err_msg = ocr_ner_data["error"].get("message")
            elif landmark_data.get("error"):
                err_msg = landmark_data["error"].get("message")

            return {
                "risk_level": "Unknown",
                "risk_score": None,
                "status": "DETECTOR_ERROR",
                "color": "gray",
                "safe_to_post": False,
                "reasons": [err_msg or "Upstream detector error occurred; cannot verify privacy safety."],
                "findings": []
            }

        # 2. Extract OCR / NER findings (Supports nested and flat)
        findings = []
        if "ocr_ner" in payload and isinstance(payload["ocr_ner"], dict):
            findings = payload["ocr_ner"].get("findings", []) or []
            ocr_flag = len(findings) > 0
        else:
            # Fallback for flat dictionary
            ocr_flag = bool(payload.get("ocr_pii_flag", False))
            flat_types = payload.get("pii_types", []) or []
            findings = [{"type": t, "confidence": 1.0} for t in flat_types]

        # 3. Aggregate unique PII categories without double-counting
        # Filter findings with confidence >= 0.40
        valid_findings = [f for f in findings if float(f.get("confidence", 1.0) or 1.0) >= 0.40]
        unique_pii_types = set(f.get("type", "").lower() for f in valid_findings if f.get("type"))

        has_address = int("address" in unique_pii_types)
        has_phone = int("phone" in unique_pii_types)
        has_email = int("email" in unique_pii_types)
        has_name = int("person_name" in unique_pii_types or "name" in unique_pii_types)
        num_pii = len(unique_pii_types)

        # 4. Extract Landmark findings (Supports nested and flat)
        if "landmark" in payload and isinstance(payload["landmark"], dict):
            landmark_flag = bool(payload["landmark"].get("detected", False))
            landmark_conf = float(payload["landmark"].get("confidence", 0.0) or 0.0)
            landmark_name = payload["landmark"].get("name", "Unknown Landmark")
        else:
            landmark_flag = bool(payload.get("landmark_flag", False))
            landmark_conf = float(payload.get("landmark_confidence", 0.0) or 0.0)
            landmark_name = payload.get("landmark_name", "Unknown Landmark")

        # Ignore landmarks with negligible confidence (< 0.40)
        if landmark_conf < 0.40:
            landmark_flag = False
            landmark_conf = 0.0

        # 5. Build feature DataFrame for ML model
        features = pd.DataFrame([{
            "ocr_pii_flag": int(len(unique_pii_types) > 0),
            "num_pii_types": num_pii,
            "has_address": has_address,
            "has_phone": has_phone,
            "landmark_flag": int(landmark_flag),
            "landmark_confidence": landmark_conf
        }], columns=self.feature_columns)

        # 6. Predict class and probabilities
        pred_class_idx = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        risk_level = self.classes[pred_class_idx]
        continuous_score = int(round((probabilities[1] * 50) + (probabilities[2] * 100)))

        # 7. Generate explainable reasons for the user
        reasons = []
        if has_address:
            reasons.append("Residential address or street name detected in image.")
        if has_phone:
            reasons.append("Phone number or direct contact information detected.")
        if has_email:
            reasons.append("Personal email address detected.")
        if has_name:
            reasons.append("Identifiable personal name detected in text.")
        if landmark_flag and landmark_conf >= 0.50:
            reasons.append(f"Identifiable landmark recognized: {landmark_name} ({int(landmark_conf * 100)}% confidence).")
        if not reasons:
            reasons.append("No sensitive personal identifiers or distinct locations found.")

        color_map = {"Low": "green", "Medium": "yellow", "High": "red"}
        safe_to_post = risk_level == "Low"

        # 8. Return complete payload including preserved UI provenance
        return {
            "risk_level": risk_level,
            "risk_score": continuous_score,
            "color": color_map[risk_level],
            "safe_to_post": safe_to_post,
            "reasons": reasons,
            "findings": valid_findings  # Preserved for UI bounding boxes and redacted text
        }
