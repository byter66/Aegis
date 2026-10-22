"""Train and evaluate the AEGIS Learned Risk Fusion Classifier."""
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


def train_risk_models():
    # 1. Load the dataset
    dataset_path = "risk-fusion/risk_dataset.csv"
    if not os.path.exists(dataset_path):
        # Fallback to local path if executed from risk-fusion directory
        dataset_path = "risk_dataset.csv"
    df = pd.read_csv(dataset_path)

    # 2. Separate features (X) from the target label (y)
    # CHANGE 3: If risk_dataset.csv has a "scene_score" column, include it as a 7th feature;
    # otherwise train on the current 6.
    feature_columns = [
        "ocr_pii_flag",
        "num_pii_types",
        "has_address",
        "has_phone",
        "landmark_flag",
        "landmark_confidence"
    ]
    if "scene_score" in df.columns:
        feature_columns.append("scene_score")
        print("Detected 'scene_score' in dataset; training with 7 features.")
    else:
        print("Training with standard 6 features.")

    X = df[feature_columns]
    y = df["risk_label"]

    # 3. Train-Test Split (80% for training, 20% held out for fair testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print("=" * 60)

    # -------------------------------------------------------------
    # Model 1: Logistic Regression Fusion Classifier
    # -------------------------------------------------------------
    print("\n--- Training Model 1: Logistic Regression Fusion ---")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    lr_preds = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)

    print(f"Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
    print("\nClassification Report (Precision, Recall, F1-Score):")
    target_names = ["Low (0)", "Medium (1)", "High (2)"]
    print(classification_report(y_test, lr_preds, target_names=target_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_preds))

    # -------------------------------------------------------------
    # Model 2: Shallow MLP (Neural Network)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("\n--- Training Model 2: Shallow MLP Neural Network ---")
    mlp_model = MLPClassifier(hidden_layer_sizes=(8,), max_iter=1000, random_state=42)
    mlp_model.fit(X_train, y_train)

    mlp_preds = mlp_model.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_preds)

    print(f"Shallow MLP Accuracy: {mlp_acc * 100:.2f}%")
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, mlp_preds, target_names=target_names))

    # 4. Save the trained Logistic Regression model and feature names list to disk
    model_output_path = "risk-fusion/fusion_model.joblib"
    if not os.path.exists("risk-fusion") and os.path.exists("src"):
        model_output_path = "fusion_model.joblib"

    # Save model and a sidecar JSON features list next to fusion_model.joblib
    joblib.dump(lr_model, model_output_path)
    features_json_path = os.path.splitext(model_output_path)[0] + "_features.json"
    with open(features_json_path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    print(f"\nSaved trained model to: {model_output_path}")
    print(f"Saved feature list to: {features_json_path}")


if __name__ == "__main__":
    train_risk_models()
