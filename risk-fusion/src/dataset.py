"""Dataset generator for AEGIS Risk Fusion module.
Generates realistic labeled training samples covering safe, moderate, and risky images.
"""
import os
import random
import pandas as pd


def generate_synthetic_dataset(num_samples: int = 250) -> pd.DataFrame:
    """Creates a balanced dataset with ground-truth privacy risk labels:

    0 = Low Risk (safe selfies, scenic views, food photos)
    1 = Medium Risk (tourist spot, receipt without personal details, revealing scenes)
    2 = High Risk (exposed home address, phone numbers, ID cards)
    """
    records = []
    random.seed(42)  # For reproducible results

    for i in range(num_samples):
        # Pick target label evenly to ensure balanced classes (0=Low, 1=Medium, 2=High)
        target_label = random.choice([0, 1, 2])

        if target_label == 0:
            # Safe photo (Low risk, label 0):
            # selfies, food, plain scenery, indoor rooms -> scene_score uniform 0.00-0.55
            # Borderline rows (~10% of safe): scene_score 0.55-0.60 -> still Low (0)
            is_borderline = (random.random() < 0.10)
            if is_borderline:
                scene_score = round(random.uniform(0.55, 0.60), 2)
            else:
                scene_score = round(random.uniform(0.00, 0.55), 2)

            has_landmark = random.choice([True, False])
            landmark_conf = round(random.uniform(0.10, 0.45), 2) if has_landmark else 0.0
            ocr_flag = False
            pii_types = []
            label = 0

        elif target_label == 1:
            # Medium risk (label 1): tourist spot landmark OR revealing scene
            scenario = random.choice(["medium_landmark", "revealing_scene"])
            if scenario == "medium_landmark":
                # Tourist spot landmark: clear landmark, scene_score uniform 0.30-0.95
                has_landmark = True
                landmark_conf = round(random.uniform(0.65, 0.95), 2)
                ocr_flag = random.choice([True, False])
                pii_types = ["general_text"] if ocr_flag else []
                scene_score = round(random.uniform(0.30, 0.95), 2)
                label = 1
            else:
                # NEW scenario "revealing_scene": no PII, no confirmed landmark
                # (landmark_flag 0 or low-confidence under 0.45), but background reveals location
                # scene_score uniform 0.60-1.00 -> Medium (1)
                has_landmark = random.choice([True, False])
                landmark_conf = round(random.uniform(0.10, 0.44), 2) if has_landmark else 0.0
                ocr_flag = False
                pii_types = []
                scene_score = round(random.uniform(0.60, 1.00), 2)
                label = 1

        else:
            # Dangerous photo (High risk, label 2):
            # Sensitive personal identifiers leaked (address, phone, email, id_number)
            # Personal details make it High regardless of the background (scene_score uniform 0.00-1.00)
            has_landmark = random.choice([True, False])
            landmark_conf = round(random.uniform(0.50, 0.95), 2) if has_landmark else 0.0
            ocr_flag = True
            possible_pii = ["address", "phone", "email", "id_number"]
            pii_types = random.sample(possible_pii, k=random.randint(1, 3))
            scene_score = round(random.uniform(0.00, 1.00), 2)
            label = 2

        # Extract numerical features for Machine Learning
        # Column order: sample_id, ocr_pii_flag, num_pii_types, has_address, has_phone,
        #               landmark_flag, landmark_confidence, scene_score, risk_label
        records.append({
            "sample_id": f"img_{i+1:03d}",
            "ocr_pii_flag": int(ocr_flag),
            "num_pii_types": len(pii_types),
            "has_address": int("address" in pii_types),
            "has_phone": int("phone" in pii_types),
            "landmark_flag": int(has_landmark),
            "landmark_confidence": landmark_conf,
            "scene_score": scene_score,
            "risk_label": label  # 0: Low, 1: Medium, 2: High
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset(250)
    print("Dataset generated successfully!")
    print("\nClass distribution (samples per risk category):")
    print(df["risk_label"].value_counts().rename({0: "Low (0)", 1: "Medium (1)", 2: "High (2)"}))
    print("\nFirst 5 samples of feature matrix:")
    print(df.head())

    # Save to risk_dataset.csv
    csv_path = "risk-fusion/risk_dataset.csv"
    if not os.path.exists("risk-fusion") and os.path.exists("src"):
        csv_path = "risk_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to {csv_path}")
