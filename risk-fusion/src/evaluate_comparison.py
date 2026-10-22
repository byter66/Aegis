"""Direct comparison between Rule-Based Baseline and Learned ML Fusion.
Generates the comparison metrics table for the project report.
"""
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from baseline import rule_based_risk


def run_comparison():
    # 1. Load the dataset
    dataset_path = "risk-fusion/risk_dataset.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "risk_dataset.csv"
    df = pd.read_csv(dataset_path)

    feature_cols = [
        "ocr_pii_flag", "num_pii_types", "has_address", 
        "has_phone", "landmark_flag", "landmark_confidence"
    ]
    if "scene_score" in df.columns:
        feature_cols.append("scene_score")

    X = df[feature_cols]
    y = df["risk_label"]

    # Use the exact same 20% test split (random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ---------------------------------------------------------
    # Evaluation 1: Rule-Based Baseline Predictions on Test Set
    # ---------------------------------------------------------
    baseline_numeric_preds = []
    for _, row in X_test.iterrows():
        # Reconstruct the dictionary format expected by baseline
        mock_input = {
            "ocr_pii_flag": bool(row["ocr_pii_flag"]),
            "pii_types": (["address"] if row["has_address"] else []) + 
                         (["phone"] if row["has_phone"] else []),
            "landmark_flag": bool(row["landmark_flag"]),
            "landmark_confidence": float(row["landmark_confidence"])
        }
        score, label_str = rule_based_risk(mock_input)
        # Map string label to integer: "low"->0, "medium"->1, "high"->2
        label_map = {"low": 0, "medium": 1, "high": 2}
        baseline_numeric_preds.append(label_map[label_str])

    # ---------------------------------------------------------
    # Evaluation 2: Learned ML Model Predictions on Test Set
    # ---------------------------------------------------------
    model_path = "risk-fusion/fusion_model.joblib"
    if not os.path.exists(model_path):
        model_path = "fusion_model.joblib"
    ml_model = joblib.load(model_path)
    if isinstance(ml_model, dict) and "model" in ml_model:
        ml_model = ml_model["model"]
    ml_preds = ml_model.predict(X_test)

    # Calculate metrics
    base_acc = accuracy_score(y_test, baseline_numeric_preds)
    base_f1 = f1_score(y_test, baseline_numeric_preds, average="weighted")
    base_prec = precision_score(y_test, baseline_numeric_preds, average="weighted", zero_division=0)
    base_rec = recall_score(y_test, baseline_numeric_preds, average="weighted")

    ml_acc = accuracy_score(y_test, ml_preds)
    ml_f1 = f1_score(y_test, ml_preds, average="weighted")
    ml_prec = precision_score(y_test, ml_preds, average="weighted", zero_division=0)
    ml_rec = recall_score(y_test, ml_preds, average="weighted")

    # ---------------------------------------------------------
    # Print the Final Comparison Table
    # ---------------------------------------------------------
    print("=" * 65)
    print("      AEGIS RISK FUSION: MODEL COMPARISON REPORT")
    print("=" * 65)
    print(f"{'Metric':<18} | {'Rule-Based Baseline':<20} | {'Learned ML Fusion':<18}")
    print("-" * 65)
    print(f"{'Accuracy':<18} | {base_acc * 100:>18.2f}% | {ml_acc * 100:>16.2f}%")
    print(f"{'Precision (Weighted)':<18} | {base_prec * 100:>18.2f}% | {ml_prec * 100:>16.2f}%")
    print(f"{'Recall (Weighted)':<18} | {base_rec * 100:>18.2f}% | {ml_rec * 100:>16.2f}%")
    print(f"{'F1-Score (Weighted)':<18} | {base_f1 * 100:>18.2f}% | {ml_f1 * 100:>16.2f}%")
    print("=" * 65)

    if ml_f1 >= base_f1:
        diff = (ml_f1 - base_f1) * 100
        print(f"\nConclusion: Learned ML Fusion shows a +{diff:.2f}% improvement in F1-score")
        print("over the baseline, validating the machine learning approach.")


if __name__ == "__main__":
    run_comparison()
