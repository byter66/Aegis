"""Read-only validation of the final landmark dataset."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "processed" / "final_dataset"
IMAGES_DIR = FINAL_DIR / "images"
LABELS_PATH = FINAL_DIR / "labels.csv"
SUMMARY_PATH = FINAL_DIR / "dataset_summary.csv"


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        return list(reader)


def validate() -> bool:
    labels = _read_csv(
        LABELS_PATH,
        {"image_id", "landmark_id", "filename", "url"},
    )
    summary = _read_csv(SUMMARY_PATH, {"landmark_id", "image_count"})
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Missing images directory: {IMAGES_DIR}")

    actual_paths = {
        path.relative_to(IMAGES_DIR)
        for path in IMAGES_DIR.rglob("*")
        if path.is_file()
    }
    expected_paths = {
        Path(row["landmark_id"]) / row["filename"] for row in labels
    }
    missing_images = sorted(expected_paths - actual_paths)
    extra_images = sorted(actual_paths - expected_paths)
    duplicate_paths = [
        path for path, count in Counter(
            Path(row["landmark_id"]) / row["filename"] for row in labels
        ).items() if count > 1
    ]
    duplicate_ids = [
        image_id for image_id, count in Counter(
            row["image_id"] for row in labels
        ).items() if count > 1
    ]
    missing_labels = [
        index + 2 for index, row in enumerate(labels)
        if not row["image_id"] or not row["landmark_id"] or not row["filename"]
    ]

    label_counts = Counter(row["landmark_id"] for row in labels)
    summary_counts = {
        row["landmark_id"]: int(row["image_count"]) for row in summary
    }
    mismatches = sorted(
        (landmark_id, label_counts.get(landmark_id, 0), summary_counts.get(landmark_id, 0))
        for landmark_id in set(label_counts) | set(summary_counts)
        if label_counts.get(landmark_id, 0) != summary_counts.get(landmark_id, 0)
    )
    counts = list(label_counts.values())
    class_range_ok = bool(counts) and min(counts) >= 15 and max(counts) <= 20
    checks_passed = (
        len(labels) == 955
        and len(label_counts) == 50
        and not missing_images
        and not extra_images
        and not duplicate_paths
        and not duplicate_ids
        and not missing_labels
        and not mismatches
        and class_range_ok
    )

    print("Final dataset validation report")
    print("=" * 32)
    print(f"Total images: {len(labels)}")
    print(f"Total classes: {len(label_counts)}")
    print(f"Minimum images/class: {min(counts) if counts else 0}")
    print(f"Maximum images/class: {max(counts) if counts else 0}")
    print(f"Median images/class: {statistics.median(counts) if counts else 0}")
    print(f"Missing images: {missing_images or 'None'}")
    print(f"Extra images: {extra_images or 'None'}")
    print(f"Duplicate records: {duplicate_paths or 'None'}")
    print(f"Duplicate image IDs: {duplicate_ids or 'None'}")
    print(f"Missing labels: {missing_labels or 'None'}")
    print(f"Class-count mismatches: {mismatches or 'None'}")
    print(f"Class range 15-20: {'PASS' if class_range_ok else 'FAIL'}")

    print("\nClass distribution")
    print("landmark_id | image_count")
    print("-------------------------")
    for landmark_id in sorted(label_counts, key=lambda value: (-label_counts[value], value)):
        print(f"{landmark_id} | {label_counts[landmark_id]}")

    if checks_passed:
        print("\nFINAL DATASET VALIDATION PASSED — READY FOR TRAIN/VALIDATION/TEST SPLIT.")
    else:
        print("\nFINAL DATASET VALIDATION FAILED.")
    return checks_passed


def main() -> int:
    try:
        return 0 if validate() else 1
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
