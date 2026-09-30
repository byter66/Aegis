"""Extract up to 20 image URLs for each selected GLDv2 landmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "processed" / "candidate_landmarks.csv"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "raw" / "train.csv"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "candidate_samples"
    / "candidate_image_samples_20.csv"
)
CHUNK_SIZE = 100_000
MAX_SAMPLES_PER_LANDMARK = 20
INPUT_COLUMNS = ["id", "url", "landmark_id"]
OUTPUT_COLUMNS = ["landmark_id", "image_id", "url"]


def _read_candidate_ids(path: Path) -> list[str]:
    """Read candidate IDs in their existing rank order."""
    if not path.is_file():
        raise FileNotFoundError(f"Candidate file does not exist: {path}")

    candidates = pd.read_csv(path, usecols=["landmark_id"], dtype="string")
    ids = candidates["landmark_id"].dropna().str.strip()
    ids = ids[ids != ""].drop_duplicates().tolist()
    if not ids:
        raise ValueError(f"No candidate landmark IDs found in {path}.")
    return ids


def sample_extended_urls(
    candidates_path: Path = DEFAULT_CANDIDATES,
    train_path: Path = DEFAULT_TRAIN,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """Read train metadata in chunks and save up to 20 unique URLs per candidate."""
    candidate_ids = _read_candidate_ids(candidates_path)
    candidate_set = set(candidate_ids)
    rows_by_landmark: dict[str, list[dict[str, str]]] = {
        landmark_id: [] for landmark_id in candidate_ids
    }
    seen_image_ids: dict[str, set[str]] = {landmark_id: set() for landmark_id in candidate_ids}
    seen_urls: dict[str, set[str]] = {landmark_id: set() for landmark_id in candidate_ids}

    if not train_path.is_file():
        raise FileNotFoundError(f"Training metadata file does not exist: {train_path}")

    reader = pd.read_csv(
        train_path,
        usecols=INPUT_COLUMNS,
        dtype="string",
        chunksize=CHUNK_SIZE,
        on_bad_lines="warn",
        low_memory=False,
    )
    for chunk in reader:
        chunk["landmark_id"] = chunk["landmark_id"].str.strip()
        chunk["id"] = chunk["id"].str.strip()
        chunk["url"] = chunk["url"].str.strip()
        matching = chunk[
            chunk["landmark_id"].isin(candidate_set)
            & chunk["id"].notna()
            & (chunk["id"] != "")
            & chunk["url"].notna()
            & (chunk["url"] != "")
        ]

        for row in matching.itertuples(index=False):
            landmark_id = str(row.landmark_id)
            image_id = str(row.id)
            url = str(row.url)
            if len(rows_by_landmark[landmark_id]) >= MAX_SAMPLES_PER_LANDMARK:
                continue
            if image_id in seen_image_ids[landmark_id] or url in seen_urls[landmark_id]:
                continue
            rows_by_landmark[landmark_id].append(
                {"landmark_id": landmark_id, "image_id": image_id, "url": url}
            )
            seen_image_ids[landmark_id].add(image_id)
            seen_urls[landmark_id].add(url)

        if all(
            len(rows_by_landmark[landmark_id]) >= MAX_SAMPLES_PER_LANDMARK
            for landmark_id in candidate_ids
        ):
            break

    rows = [
        row
        for landmark_id in candidate_ids
        for row in rows_by_landmark[landmark_id]
    ]
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    _print_summary(candidate_ids, rows_by_landmark, output_path)
    return result


def _print_summary(
    candidate_ids: list[str],
    rows_by_landmark: dict[str, list[dict[str, str]]],
    output_path: Path,
) -> None:
    """Print collection counts in candidate rank order."""
    counts = [len(rows_by_landmark[landmark_id]) for landmark_id in candidate_ids]
    fewer_than_limit = [
        landmark_id
        for landmark_id in candidate_ids
        if len(rows_by_landmark[landmark_id]) < MAX_SAMPLES_PER_LANDMARK
    ]
    print(f"Number of candidate landmark IDs: {len(candidate_ids)}")
    print(f"Total rows collected: {sum(counts)}")
    print(
        "Number of unique landmark IDs represented: "
        f"{sum(count > 0 for count in counts)}"
    )
    print("\nURLs collected per landmark:")
    for landmark_id in candidate_ids:
        print(f"{landmark_id}: {len(rows_by_landmark[landmark_id])}")
    print(f"\nMinimum samples per landmark: {min(counts)}")
    print(f"Maximum samples per landmark: {max(counts)}")
    print("\nCandidates with fewer than 20 URLs:")
    print(", ".join(fewer_than_limit) if fewer_than_limit else "None")
    print(f"\nSaved extended samples to: {output_path}")


def main() -> int:
    """Run extended URL extraction from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        sample_extended_urls(args.candidates, args.train, args.output)
    except (FileNotFoundError, OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
