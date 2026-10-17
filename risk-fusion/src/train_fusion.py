"""Train and evaluate the AEGIS Learned Risk Fusion Classifier."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib


def train_risk_models():
    # 1. Load the dataset you just created
    df = pd.read_csv("risk-fusion/risk_dataset.csv")

    # 2. Separate features (X) from the target label (y)
    feature_columns = [
        "ocr_pii_flag",
        "num_pii_types",
        "has_address",
        "has_phone",
        "landmark_flag",
        "landmark_confidence"
    ]
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

    # 4. Save the trained Logistic Regression model to disk for production use
    joblib.dump(lr_model, "risk-fusion/fusion_model.joblib")
    print("\nSaved trained model to: risk-fusion/fusion_model.joblib")


if __name__ == "__main__":
    train_risk_models()
