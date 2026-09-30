"""Create reproducible stratified references to the master image dataset."""

from __future__ import annotations

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "final_dataset" / "labels.csv"
MASTER_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "final_dataset" / "images"
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
RANDOM_SEED = 42
SPLIT_NAMES = ("train", "val", "test")


def _allocation(count: int) -> tuple[int, int, int]:
    """Return the closest integer allocation to 70/15/15 with three nonempty sets."""
    quotas = [count * 0.70, count * 0.15, count * 0.15]
    values = [max(1, int(quota)) for quota in quotas]
    while sum(values) > count:
        candidates = [index for index, value in enumerate(values) if value > 1]
        index = min(candidates, key=lambda item: quotas[item] - values[item])
        values[index] -= 1
    while sum(values) < count:
        index = max(range(3), key=lambda item: quotas[item] - values[item])
        values[index] += 1
    return values[0], values[1], values[2]


def _read_labels() -> list[dict[str, str]]:
    with LABELS_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "landmark_id", "filename"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"labels.csv is missing columns: {sorted(missing)}")
    return rows


def create_splits() -> None:
    if SPLITS_DIR.exists():
        raise FileExistsError(f"Split directory already exists: {SPLITS_DIR}")
    if not LABELS_PATH.is_file() or not MASTER_IMAGES_DIR.is_dir():
        raise FileNotFoundError("Master labels or image directory is missing")

    rows = _read_labels()
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        image_path = MASTER_IMAGES_DIR / row["landmark_id"] / row["filename"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Referenced master image is missing: {image_path}")
        by_class[row["landmark_id"]].append(
            {
                "image_path": str(
                    image_path.relative_to(PROJECT_ROOT).as_posix()
                ),
                "landmark_id": row["landmark_id"],
            }
        )

    split_rows: dict[str, list[dict[str, str]]] = {name: [] for name in SPLIT_NAMES}
    for landmark_id in sorted(by_class):
        class_rows = list(by_class[landmark_id])
        random.Random(f"{RANDOM_SEED}:{landmark_id}").shuffle(class_rows)
        train_count, val_count, test_count = _allocation(len(class_rows))
        boundaries = (train_count, train_count + val_count, train_count + val_count + test_count)
        split_rows["train"].extend(class_rows[: boundaries[0]])
        split_rows["val"].extend(class_rows[boundaries[0] : boundaries[1]])
        split_rows["test"].extend(class_rows[boundaries[1] : boundaries[2]])

    SPLITS_DIR.mkdir(parents=True)
    for name, split in split_rows.items():
        with (SPLITS_DIR / f"{name}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_path", "landmark_id"])
            writer.writeheader()
            writer.writerows(split)

    with (SPLITS_DIR / "split_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["landmark_id", "total", "train", "val", "test"])
        for landmark_id in sorted(by_class):
            counts = [
                sum(row["landmark_id"] == landmark_id for row in split_rows[name])
                for name in SPLIT_NAMES
            ]
            writer.writerow([landmark_id, len(by_class[landmark_id]), *counts])

    print("Dataset split report")
    print(f"Total images: {len(rows)}")
    for name in SPLIT_NAMES:
        print(f"{name.title()}: {len(split_rows[name])}")
    print("Number of classes in each:")
    for name in SPLIT_NAMES:
        print(f"{name.title()}: {len({row['landmark_id'] for row in split_rows[name]})}")
    print("\nlandmark_id | total | train | val | test")
    print("-----------------------------------------")
    for landmark_id in sorted(by_class):
        counts = [
            sum(row["landmark_id"] == landmark_id for row in split_rows[name])
            for name in SPLIT_NAMES
        ]
        print(f"{landmark_id} | {len(by_class[landmark_id])} | {counts[0]} | {counts[1]} | {counts[2]}")


def main() -> int:
    try:
        create_splits()
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
