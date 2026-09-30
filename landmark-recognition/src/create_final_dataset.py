"""Create a new final landmark dataset from locally available images."""

from __future__ import annotations

import csv
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_samples"
    / "candidate_image_samples_20.csv"
)
SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_samples"
    / "wikimedia_api_samples"
    / "images"
)
FINAL_DIR = PROJECT_ROOT / "data" / "processed" / "final_dataset"
FINAL_IMAGES_DIR = FINAL_DIR / "images"
LABELS_PATH = FINAL_DIR / "labels.csv"
SUMMARY_PATH = FINAL_DIR / "dataset_summary.csv"
REQUIRED_COLUMNS = {"landmark_id", "image_id", "url"}


def _read_metadata() -> dict[str, dict[str, str]]:
    with METADATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Metadata is missing columns: {sorted(missing)}")

        metadata: dict[str, dict[str, str]] = {}
        for row in reader:
            image_id = (row["image_id"] or "").strip()
            landmark_id = (row["landmark_id"] or "").strip()
            if not image_id or not landmark_id:
                continue
            if image_id in metadata:
                raise ValueError(f"Duplicate image_id in metadata: {image_id}")
            metadata[image_id] = {
                "landmark_id": landmark_id,
                "url": row["url"] or "",
            }
    return metadata


def create_final_dataset() -> None:
    if FINAL_DIR.exists():
        raise FileExistsError(
            f"Final dataset directory already exists; refusing to overwrite: {FINAL_DIR}"
        )
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {METADATA_PATH}")
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Source image directory does not exist: {SOURCE_DIR}")

    metadata = _read_metadata()
    source_files = sorted(path for path in SOURCE_DIR.iterdir() if path.is_file())
    matched: list[tuple[Path, str, dict[str, str]]] = []
    for source_path in source_files:
        record = metadata.get(source_path.stem)
        if record is not None:
            matched.append((source_path, source_path.name, record))

    FINAL_IMAGES_DIR.mkdir(parents=True)
    labels: list[dict[str, str]] = []
    for source_path, filename, record in matched:
        destination_dir = FINAL_IMAGES_DIR / record["landmark_id"]
        destination_dir.mkdir(exist_ok=True)
        destination = destination_dir / filename
        shutil.copy2(source_path, destination)
        labels.append(
            {
                "image_id": source_path.stem,
                "landmark_id": record["landmark_id"],
                "filename": filename,
                "url": record["url"],
            }
        )

    with LABELS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "landmark_id", "filename", "url"],
        )
        writer.writeheader()
        writer.writerows(labels)

    counts = Counter(label["landmark_id"] for label in labels)
    summary_rows = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    with SUMMARY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["landmark_id", "image_count"])
        writer.writerows(summary_rows)

    expected_paths = {
        Path(label["landmark_id"]) / label["filename"] for label in labels
    }
    actual_paths = {
        path.relative_to(FINAL_IMAGES_DIR)
        for path in FINAL_IMAGES_DIR.rglob("*")
        if path.is_file()
    }
    if expected_paths != actual_paths:
        raise RuntimeError("Final image files do not match labels.csv")
    if len(labels) != len(expected_paths):
        raise RuntimeError("labels.csv contains duplicate final image paths")

    values = [count for _, count in summary_rows]
    print("Final dataset summary:")
    print(f"Total source images found: {len(source_files)}")
    print(f"Total images matched to candidate metadata: {len(matched)}")
    print(f"Total images missing from candidate metadata: {len(source_files) - len(matched)}")
    print(f"Total landmark IDs represented: {len(summary_rows)}")
    print(f"Minimum images per landmark: {min(values) if values else 0}")
    print(f"Maximum images per landmark: {max(values) if values else 0}")
    print(f"Median images per landmark: {statistics.median(values) if values else 0}")
    print(f"Total final images: {len(labels)}")
    print(f"Labels written to: {LABELS_PATH}")
    print(f"Summary written to: {SUMMARY_PATH}")
    print(f"Images written to: {FINAL_IMAGES_DIR}")


def main() -> int:
    try:
        create_final_dataset()
    except (FileExistsError, FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
