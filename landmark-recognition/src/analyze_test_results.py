"""Analyze existing Experiment 3 test-evaluation artifacts without model inference."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "evaluation" / "experiment3"
OUTPUT_PATH = INPUT_DIR / "test_analysis.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze() -> None:
    metrics_path = INPUT_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_class = _read_csv(INPUT_DIR / "per_class_metrics.csv")
    report = _read_csv(INPUT_DIR / "classification_report.csv")
    matrix_rows = _read_csv(INPUT_DIR / "confusion_matrix.csv")

    report_by_class = {row["landmark_id"]: row for row in report}
    class_ids = [row["landmark_id"] for row in per_class]
    matrix = {
        row["actual\\predicted"]: {
            landmark_id: int(row[landmark_id])
            for landmark_id in class_ids
        }
        for row in matrix_rows
    }

    ranked_best = sorted(per_class, key=lambda row: (-float(row["accuracy"]), row["landmark_id"]))
    ranked_worst = sorted(per_class, key=lambda row: (float(row["accuracy"]), row["landmark_id"]))
    zero_correct = [row["landmark_id"] for row in per_class if int(row["correct"]) == 0]
    confusion_pairs = sorted(
        (
            (count, actual, predicted)
            for actual, values in matrix.items()
            for predicted, count in values.items()
            if actual != predicted and count > 0
        ),
        reverse=True,
    )

    total_support = sum(int(row["support"]) for row in report)
    macro = {
        metric: sum(float(row[metric]) for row in report) / len(report)
        for metric in ("precision", "recall", "f1_score")
    }
    weighted = {
        metric: sum(float(row[metric]) * int(row["support"]) for row in report) / total_support
        for metric in ("precision", "recall", "f1_score")
    }
    error_by_class = Counter(
        {row["landmark_id"]: int(row["incorrect"]) for row in per_class}
    )
    total_errors = sum(error_by_class.values())
    top_error_count = sum(count for _, count in error_by_class.most_common(10))
    error_classes = sum(count > 0 for count in error_by_class.values())
    concentration = (
        f"{top_error_count}/{total_errors} errors ({top_error_count / total_errors:.2%}) "
        f"come from the top 10 classes; errors affect {error_classes}/{len(class_ids)} classes. "
        + ("Errors are concentrated in a small number of classes."
           if top_error_count / total_errors >= 0.60
           else "Errors are broadly distributed across classes.")
    )

    output_rows: list[dict[str, str | int | float]] = []
    for row in per_class:
        report_row = report_by_class[row["landmark_id"]]
        output_rows.append(
            {
                "record_type": "class",
                "landmark_id": row["landmark_id"],
                "accuracy": row["accuracy"],
                "correct": row["correct"],
                "incorrect": row["incorrect"],
                "support": row["support"],
                "precision": report_row["precision"],
                "recall": report_row["recall"],
                "f1_score": report_row["f1_score"],
                "actual_landmark_id": "",
                "predicted_landmark_id": "",
                "confusion_count": "",
            }
        )
    for count, actual, predicted in confusion_pairs:
        output_rows.append(
            {
                "record_type": "confusion_pair",
                "landmark_id": "",
                "accuracy": "",
                "correct": "",
                "incorrect": "",
                "support": "",
                "precision": "",
                "recall": "",
                "f1_score": "",
                "actual_landmark_id": actual,
                "predicted_landmark_id": predicted,
                "confusion_count": count,
            }
        )

    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "record_type", "landmark_id", "accuracy", "correct", "incorrect",
            "support", "precision", "recall", "f1_score",
            "actual_landmark_id", "predicted_landmark_id", "confusion_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print("Experiment 3 test results")
    print("=========================")
    print(f"Test loss: {float(metrics['test_loss']):.4f}")
    print(f"Top-1 accuracy: {float(metrics['top1_accuracy']):.2%}")
    print(f"Top-5 accuracy: {float(metrics['top5_accuracy']):.2%}")
    print(f"Test images: {metrics['test_images']}")
    print(f"Classes: {metrics['num_classes']}")

    print("\nTop 10 best-performing classes")
    print("landmark_id | accuracy | correct | incorrect")
    for row in ranked_best[:10]:
        print(f"{row['landmark_id']} | {float(row['accuracy']):.2%} | {row['correct']} | {row['incorrect']}")

    print("\nBottom 10 worst-performing classes")
    print("landmark_id | accuracy | correct | incorrect")
    for row in ranked_worst[:10]:
        print(f"{row['landmark_id']} | {float(row['accuracy']):.2%} | {row['correct']} | {row['incorrect']}")

    print(f"\nClasses with 0 correct predictions ({len(zero_correct)}): {', '.join(zero_correct) or 'None'}")
    print("\nMost frequent confusion pairs")
    print("actual -> predicted | count")
    for count, actual, predicted in confusion_pairs[:10]:
        print(f"{actual} -> {predicted} | {count}")

    print("\nPer-class precision, recall, F1")
    print("landmark_id | precision | recall | f1_score")
    for row in report:
        print(
            f"{row['landmark_id']} | {float(row['precision']):.2%} | "
            f"{float(row['recall']):.2%} | {float(row['f1_score']):.2%}"
        )
    print("\nMacro-average:")
    print("precision={precision:.4f}, recall={recall:.4f}, f1_score={f1_score:.4f}".format(**macro))
    print("Weighted-average:")
    print("precision={precision:.4f}, recall={recall:.4f}, f1_score={f1_score:.4f}".format(**weighted))
    print(f"\nError distribution: {concentration}")
    print(f"\nAnalysis saved to: {OUTPUT_PATH}")


def main() -> int:
    try:
        analyze()
    except (FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Analysis error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
