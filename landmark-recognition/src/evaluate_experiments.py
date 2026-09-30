"""Evaluate the baseline and regularized checkpoints on the untouched test split."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .dataset import LandmarkDataset, SPLITS_DIR, load_class_mapping
    from .model import build_model
except ImportError:
    from dataset import LandmarkDataset, SPLITS_DIR, load_class_mapping
    from model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = SPLITS_DIR / "test.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
TEST_IMAGE_COUNT = 142
EXPECTED_CLASSES = 50


def _validate_preflight() -> tuple[dict[str, int], dict[int, str]]:
    required = {
        "efficientnet_b0_best.pth",
        "efficientnet_b0_regularized_best.pth",
    }
    missing = [name for name in required if not (CHECKPOINT_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")
    if not TEST_CSV.is_file():
        raise FileNotFoundError(f"Missing test split: {TEST_CSV}")
    class_mapping, reverse_mapping = load_class_mapping()
    if len(class_mapping) != EXPECTED_CLASSES:
        raise ValueError(f"Expected {EXPECTED_CLASSES} classes, found {len(class_mapping)}")

    dataset = LandmarkDataset(TEST_CSV, class_mapping, training=False)
    if len(dataset) != TEST_IMAGE_COUNT:
        raise ValueError(f"Expected {TEST_IMAGE_COUNT} test rows, found {len(dataset)}")
    missing_images = [
        row["image_path"]
        for row in dataset.rows
        if not (PROJECT_ROOT / row["image_path"]).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"Missing test image(s): {missing_images[:5]}")
    if len({row["landmark_id"] for row in dataset.rows}) != EXPECTED_CLASSES:
        raise ValueError("Test split does not contain all 50 classes")
    return class_mapping, reverse_mapping


def _load_model(checkpoint_path: Path, class_count: int, regularized: bool) -> nn.Module:
    model = build_model(
        num_classes=class_count,
        pretrained=False,
        freeze_backbone=True,
        dropout=0.3 if regularized else None,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def _write_matrix(path: Path, matrix: list[list[int]], reverse_mapping: dict[int, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *[reverse_mapping[index] for index in range(len(matrix))]])
        for index, row in enumerate(matrix):
            writer.writerow([reverse_mapping[index], *row])


def _evaluate(
    model_name: str,
    checkpoint_path: Path,
    output_dir: Path,
    class_mapping: dict[str, int],
    reverse_mapping: dict[int, str],
    regularized: bool,
) -> dict[str, float | int | str]:
    dataset = LandmarkDataset(TEST_CSV, class_mapping, training=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = _load_model(checkpoint_path, len(class_mapping), regularized)
    criterion = nn.CrossEntropyLoss()
    class_count = len(class_mapping)
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    total_loss = 0.0
    total = 0
    top1 = 0
    top5 = 0

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            loss = criterion(logits, labels)
            predictions = logits.argmax(dim=1)
            top5_predictions = logits.topk(5, dim=1).indices
            total_loss += loss.item() * labels.size(0)
            top1 += int(predictions.eq(labels).sum().item())
            top5 += int(top5_predictions.eq(labels.unsqueeze(1)).any(dim=1).sum().item())
            total += labels.size(0)
            for actual, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[actual][predicted] += 1

    per_class_rows: list[dict[str, float | int | str]] = []
    report_rows: list[dict[str, float | int | str]] = []
    for index in range(class_count):
        correct = matrix[index][index]
        support = sum(matrix[index])
        predicted_total = sum(matrix[actual][index] for actual in range(class_count))
        accuracy = correct / support if support else 0.0
        precision = correct / predicted_total if predicted_total else 0.0
        recall = accuracy
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        landmark_id = reverse_mapping[index]
        per_class_rows.append(
            {
                "class_index": index,
                "landmark_id": landmark_id,
                "correct": correct,
                "incorrect": support - correct,
                "accuracy": accuracy,
                "support": support,
            }
        )
        report_rows.append(
            {
                "class_index": index,
                "landmark_id": landmark_id,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "support": support,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    metrics = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "test_loss": total_loss / total,
        "top1_accuracy": top1 / total,
        "top5_accuracy": top5 / total,
        "test_images": total,
        "num_classes": class_count,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class_rows[0]))
        writer.writeheader()
        writer.writerows(per_class_rows)
    with (output_dir / "classification_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    _write_matrix(output_dir / "confusion_matrix.csv", matrix, reverse_mapping)
    return metrics


def evaluate_experiments() -> None:
    class_mapping, reverse_mapping = _validate_preflight()
    print(f"Test split: {TEST_CSV}")
    print(f"Test records: {TEST_IMAGE_COUNT}")
    print(f"Classes: {len(class_mapping)}")
    print("Test images missing: 0")

    results = [
        _evaluate(
            "experiment1_frozen_baseline",
            CHECKPOINT_DIR / "efficientnet_b0_best.pth",
            EVALUATION_DIR / "experiment1",
            class_mapping,
            reverse_mapping,
            regularized=False,
        ),
        _evaluate(
            "experiment3_regularized_frozen",
            CHECKPOINT_DIR / "efficientnet_b0_regularized_best.pth",
            EVALUATION_DIR / "experiment3",
            class_mapping,
            reverse_mapping,
            regularized=True,
        ),
    ]
    comparison_path = EVALUATION_DIR / "model_comparison.csv"
    with comparison_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "test_loss", "top1_accuracy", "top5_accuracy", "test_images", "num_classes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: result[field] for field in fields} for result in results)

    for result in results:
        print(f"\n{result['model']}")
        print(f"Test Top-1: {float(result['top1_accuracy']):.2%}")
        print(f"Test Top-5: {float(result['top5_accuracy']):.2%}")
        print(f"Test Loss: {float(result['test_loss']):.4f}")

    top1_difference = float(results[0]["top1_accuracy"]) - float(results[1]["top1_accuracy"])
    top5_difference = float(results[0]["top5_accuracy"]) - float(results[1]["top5_accuracy"])
    print("\nComparison:")
    print(f"Higher Top-1 accuracy: {results[0]['model'] if top1_difference > 0 else results[1]['model'] if top1_difference < 0 else 'Tie'}")
    print(f"Higher Top-5 accuracy: {results[0]['model'] if top5_difference > 0 else results[1]['model'] if top5_difference < 0 else 'Tie'}")
    print(f"Absolute Top-1 difference: {abs(top1_difference):.2%}")
    print(f"Absolute Top-5 difference: {abs(top5_difference):.2%}")
    print(f"Comparison saved to: {comparison_path}")


def main() -> int:
    try:
        evaluate_experiments()
    except (FileExistsError, FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
