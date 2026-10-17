"""Dataset generator for AEGIS Risk Fusion module.
Generates realistic labeled training samples covering safe, moderate, and risky images.
"""
import random
import pandas as pd


def generate_synthetic_dataset(num_samples: int = 150) -> pd.DataFrame:
    """Creates a balanced dataset with ground-truth privacy risk labels:

    0 = Low Risk (safe selfies, scenic views, food photos)
    1 = Medium Risk (tourist spot, receipt without personal details)
    2 = High Risk (exposed home address, phone numbers, ID cards)
    """
    records = []
    random.seed(42)  # For reproducible results

    for i in range(num_samples):
        # Decide scenario type to ensure a balanced dataset
        scenario_type = random.choice(["safe", "medium", "danger"])

        if scenario_type == "safe":
            # Safe photo: No PII, maybe low-confidence landmark or none
            has_landmark = random.choice([True, False])
            landmark_conf = round(random.uniform(0.1, 0.45), 2) if has_landmark else 0.0
            ocr_flag = False
            pii_types = []
            label = 0  # Low Risk

        elif scenario_type == "medium":
            # Moderate risk: A clear tourist landmark alone, or minor non-sensitive text
            has_landmark = True
            landmark_conf = round(random.uniform(0.65, 0.95), 2)
            ocr_flag = random.choice([True, False])
            pii_types = ["general_text"] if ocr_flag else []
            label = 1  # Medium Risk

        else:
            # Dangerous photo: Sensitive personal identifiers leaked
            has_landmark = random.choice([True, False])
            landmark_conf = round(random.uniform(0.5, 0.95), 2) if has_landmark else 0.0
            ocr_flag = True
            # High risk text types
            possible_pii = ["address", "phone", "email", "id_number"]
            pii_types = random.sample(possible_pii, k=random.randint(1, 3))
            label = 2  # High Risk

        # Extract numerical features for Machine Learning
        records.append({
            "sample_id": f"img_{i+1:03d}",
            "ocr_pii_flag": int(ocr_flag),
            "num_pii_types": len(pii_types),
            "has_address": int("address" in pii_types),
            "has_phone": int("phone" in pii_types),
            "landmark_flag": int(has_landmark),
            "landmark_confidence": landmark_conf,
            "risk_label": label  # 0: Low, 1: Medium, 2: High
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset(150)
    print("Dataset generated successfully!")
    print("\nClass distribution (samples per risk category):")
    print(df["risk_label"].value_counts().rename({0: "Low (0)", 1: "Medium (1)", 2: "High (2)"}))
    print("\nFirst 5 samples of feature matrix:")
    print(df.head())
    
    # Save to a CSV file in risk-fusion folder
    df.to_csv("risk-fusion/risk_dataset.csv", index=False)
    print("\nSaved to risk-fusion/risk_dataset.csv")
