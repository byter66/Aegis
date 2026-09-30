"""Evaluation utilities for test accuracy and per-class diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path

import torch

try:
    from .dataset import create_dataloaders
    from .model import build_model
except ImportError:
    from dataset import create_dataloaders
    from model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "data" / "processed" / "evaluation"


def evaluate(checkpoint_path: Path) -> None:
    loaders, class_mapping = create_dataloaders()
    model = build_model(num_classes=len(class_mapping), pretrained=False, freeze_backbone=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    predictions: list[int] = []
    top5_predictions: list[list[int]] = []
    labels: list[int] = []
    with torch.no_grad():
        for images, batch_labels in loaders["test"]:
            logits = model(images)
            predictions.extend(logits.argmax(dim=1).tolist())
            top5_predictions.extend(logits.topk(5, dim=1).indices.tolist())
            labels.extend(batch_labels.tolist())

    class_indices = sorted(class_mapping.values())
    confusion = [[0 for _ in class_indices] for _ in class_indices]
    for actual, predicted in zip(labels, predictions):
        confusion[actual][predicted] += 1
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    with (EVALUATION_DIR / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *class_indices])
        writer.writerows([[index, *confusion[index]] for index in class_indices])

    with (EVALUATION_DIR / "classification_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_index", "landmark_id", "precision", "recall", "f1_score", "support"])
        for index in class_indices:
            support = sum(confusion[index])
            correct = confusion[index][index]
            predicted_total = sum(confusion[actual][index] for actual in class_indices)
            precision = correct / predicted_total if predicted_total else 0.0
            recall = correct / support if support else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            landmark_id = next(key for key, value in class_mapping.items() if value == index)
            writer.writerow([index, landmark_id, precision, recall, f1_score, support])
    accuracy = sum(actual == predicted for actual, predicted in zip(labels, predictions)) / len(labels)
    top5_accuracy = sum(
        actual in candidates for actual, candidates in zip(labels, top5_predictions)
    ) / len(labels)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test top-5 accuracy: {top5_accuracy:.4f}")
    print(f"Evaluation outputs saved to: {EVALUATION_DIR}")
