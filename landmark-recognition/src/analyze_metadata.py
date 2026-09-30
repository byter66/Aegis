"""Analyze Google Landmarks v2 training metadata.

This module intentionally reads only ``train_clean.csv``.  The source file
stores the image IDs for each landmark as whitespace-separated values in one
CSV field, so image counts are computed without loading the image dataset.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "train_clean.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_ID_PATTERN = re.compile(r"\S+")


def _find_column(columns: Iterable[str], keywords: tuple[str, ...]) -> str | None:
    """Return the first column whose normalized name contains a keyword."""
    normalized = [(column, column.strip().lower()) for column in columns]
    for keyword in keywords:
        for column, name in normalized:
            if keyword in name:
                return column
    return None


def _identify_columns(frame: pd.DataFrame) -> tuple[str, str]:
    """Identify landmark and image-list columns from the observed CSV schema."""
    columns = [str(column) for column in frame.columns]
    landmark_column = _find_column(columns, ("landmark", "class", "label"))
    image_column = _find_column(columns, ("image", "images", "photo", "id"))

    if landmark_column is None and columns:
        landmark_column = columns[0]
    if image_column is None:
        remaining = [column for column in columns if column != landmark_column]
        if remaining:
            image_column = remaining[0]

    if landmark_column is None or image_column is None:
        raise ValueError(
            "Could not identify both a landmark column and an image-ID column "
            f"from columns: {columns!r}"
        )
    if landmark_column == image_column:
        raise ValueError(
            "Landmark and image-ID columns resolved to the same column: "
            f"{landmark_column!r}"
        )
    return landmark_column, image_column


def _count_image_ids(value: object) -> int:
    """Count non-empty whitespace-delimited image IDs in one field."""
    if pd.isna(value):
        return 0
    return len(IMAGE_ID_PATTERN.findall(str(value)))


def analyze_metadata(input_path: Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Read metadata, print inspection and summaries, and return ranked counts."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {input_path}")

    try:
        frame = pd.read_csv(
            input_path,
            dtype="string",
            on_bad_lines="warn",
            low_memory=False,
        )
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"Could not read metadata file {input_path}: {exc}") from exc

    print(f"Input: {input_path}")
    print("\nColumn names:")
    print(list(frame.columns))
    print(f"\nNumber of rows: {len(frame):,}")
    print("\nFirst 5 rows:")
    print(frame.head(5).to_string(index=False))
    print("\nData types:")
    print(frame.dtypes.to_string())

    landmark_column, image_column = _identify_columns(frame)
    print(f"\nUsing landmark column: {landmark_column!r}")
    print(f"Using image-ID column: {image_column!r}")

    valid_landmarks = frame[landmark_column].notna() & (
        frame[landmark_column].astype("string").str.strip() != ""
    )
    missing_landmarks = int((~valid_landmarks).sum())
    if missing_landmarks:
        print(f"Warning: skipping {missing_landmarks:,} row(s) with missing landmarks.")

    working = frame.loc[valid_landmarks, [landmark_column, image_column]].copy()
    working[landmark_column] = working[landmark_column].astype("string").str.strip()
    working["image_count"] = working[image_column].map(_count_image_ids)
    missing_images = int((working["image_count"] == 0).sum())
    if missing_images:
        print(
            f"Warning: {missing_images:,} row(s) have no usable image IDs; "
            "they contribute zero associations."
        )

    ranked = (
        working.groupby(landmark_column, sort=False, dropna=False)["image_count"]
        .sum()
        .rename_axis("landmark_id")
        .reset_index()
    )
    ranked["image_count"] = ranked["image_count"].astype("int64")
    ranked = ranked.sort_values(
        by=["image_count", "landmark_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "rank", pd.RangeIndex(start=1, stop=len(ranked) + 1))
    ranked = ranked[["landmark_id", "image_count", "rank"]]

    counts = ranked["image_count"]
    print("\nSummary statistics:")
    print(f"Total number of landmark classes: {len(ranked):,}")
    print(f"Total image associations: {int(counts.sum()):,}")
    print(f"Maximum images per class: {int(counts.max()) if len(counts) else 0:,}")
    print(f"Median images per class: {counts.median() if len(counts) else 0:g}")
    for threshold in (50, 100, 150, 200):
        print(
            f"Number of classes with >= {threshold} images: "
            f"{int((counts >= threshold).sum()):,}"
        )

    print("\nTop 30 ranked landmark classes:")
    print(ranked.head(30).to_string(index=False))
    return ranked


def main() -> int:
    """Run metadata analysis from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to train_clean.csv (defaults to data/raw/train_clean.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for processed ranking outputs.",
    )
    args = parser.parse_args()

    try:
        ranked = analyze_metadata(args.input)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        complete_path = args.output_dir / "landmark_class_counts.csv"
        top_path = args.output_dir / "top_100_landmark_candidates.csv"
        ranked.to_csv(complete_path, index=False)
        ranked.head(100).to_csv(top_path, index=False)
        print(f"\nSaved complete ranking to: {complete_path}")
        print(f"Saved top 100 candidates to: {top_path}")
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
