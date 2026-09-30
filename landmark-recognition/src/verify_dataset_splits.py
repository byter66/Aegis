"""Read-only verification of reproducible dataset split CSV files."""

from __future__ import annotations

import csv
import random
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "final_dataset" / "labels.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MASTER_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "final_dataset" / "images"
RANDOM_SEED = 42
SPLIT_NAMES = ("train", "val", "test")


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _expected_splits(labels: list[dict[str, str]]) -> dict[str, list[str]]:
    by_class: dict[str, list[str]] = {}
    for row in labels:
        by_class.setdefault(row["landmark_id"], []).append(
            f"data/processed/final_dataset/images/{row['landmark_id']}/{row['filename']}"
        )
    expected = {name: [] for name in SPLIT_NAMES}
    for landmark_id in sorted(by_class):
        paths = by_class[landmark_id]
        random.Random(f"{RANDOM_SEED}:{landmark_id}").shuffle(paths)
        count = len(paths)
        train = max(1, int(count * 0.70))
        val = max(1, int(count * 0.15))
        test = max(1, int(count * 0.15))
        while train + val + test > count:
            candidates = [(train, 0), (val, 1), (test, 2)]
            index = min(
                (item for item in candidates if item[0] > 1),
                key=lambda item: (item[0] - count * (0.70, 0.15, 0.15)[item[1]], item[1]),
            )[1]
            if index == 0:
                train -= 1
            elif index == 1:
                val -= 1
            else:
                test -= 1
        while train + val + test < count:
            remainders = (
                count * 0.70 - train,
                count * 0.15 - val,
                count * 0.15 - test,
            )
            index = max(range(3), key=remainders.__getitem__)
            if index == 0:
                train += 1
            elif index == 1:
                val += 1
            else:
                test += 1
        boundaries = (train, train + val)
        expected["train"].extend(paths[: boundaries[0]])
        expected["val"].extend(paths[boundaries[0] : boundaries[1]])
        expected["test"].extend(paths[boundaries[1] :])
    return expected


def verify() -> bool:
    labels = _read(LABELS_PATH)
    splits = {name: _read(SPLITS_DIR / f"{name}.csv") for name in SPLIT_NAMES}
    summary = _read(SPLITS_DIR / "split_summary.csv")
    expected_paths = {
        f"data/processed/final_dataset/images/{row['landmark_id']}/{row['filename']}"
        for row in labels
    }
    all_split_paths = [row["image_path"] for split in splits.values() for row in split]
    path_counts = Counter(all_split_paths)
    duplicate_paths = [path for path, count in path_counts.items() if count > 1]
    duplicate_within = {
        name: [path for path, count in Counter(
            row["image_path"] for row in split
        ).items() if count > 1]
        for name, split in splits.items()
    }
    missing = [
        path for path in all_split_paths
        if not (PROJECT_ROOT / path).is_file()
    ]
    classes = {
        name: {row["landmark_id"] for row in split}
        for name, split in splits.items()
    }
    summary_counts = {
        row["landmark_id"]: tuple(int(row[key]) for key in ("total", "train", "val", "test"))
        for row in summary
    }
    actual_counts = {
        landmark_id: (
            sum(row["landmark_id"] == landmark_id for row in labels),
            *(
                sum(row["landmark_id"] == landmark_id for row in splits[name])
                for name in SPLIT_NAMES
            ),
        )
        for landmark_id in {row["landmark_id"] for row in labels}
    }
    expected_splits = _expected_splits(labels)
    reproducible = all(
        [row["image_path"] for row in splits[name]] == expected_splits[name]
        for name in SPLIT_NAMES
    )
    mismatches = [
        landmark_id for landmark_id in actual_counts
        if summary_counts.get(landmark_id) != actual_counts[landmark_id]
    ]
    extra = sorted(set(all_split_paths) - expected_paths)
    passed = (
        len(labels) == 955
        and sum(len(split) for split in splits.values()) == 955
        and all(len(classes[name]) == 50 for name in SPLIT_NAMES)
        and not duplicate_paths
        and not any(duplicate_within.values())
        and not missing
        and not extra
        and len(set(all_split_paths)) == len(expected_paths)
        and not mismatches
        and reproducible
    )

    print("Dataset split validation report")
    print(f"Total labels: {len(labels)}")
    for name in SPLIT_NAMES:
        print(f"{name.title()}: {len(splits[name])} records, {len(classes[name])} classes")
    print(f"Combined split records: {sum(len(split) for split in splits.values())}")
    print(f"Missing referenced images: {missing or 'None'}")
    print(f"Unexpected extra images: {extra or 'None'}")
    print(f"Duplicate image paths across splits: {duplicate_paths or 'None'}")
    print(f"Duplicate paths within splits: {duplicate_within or 'None'}")
    print(f"Class-count mismatches: {mismatches or 'None'}")
    print(f"Reproducible with seed {RANDOM_SEED}: {'PASS' if reproducible else 'FAIL'}")
    print("\nlandmark_id | total | train | val | test")
    print("-----------------------------------------")
    for landmark_id in sorted(actual_counts):
        total, train, val, test = actual_counts[landmark_id]
        print(f"{landmark_id} | {total} | {train} | {val} | {test}")
    if passed:
        print("\nSPLIT VALIDATION PASSED — READY FOR MODEL TRAINING.")
    else:
        print("\nSPLIT VALIDATION FAILED.")
    return passed


def main() -> int:
    try:
        return 0 if verify() else 1
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Validation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
