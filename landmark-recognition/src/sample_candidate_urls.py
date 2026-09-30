"""Collect a small URL sample for each selected GLDv2 landmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "processed" / "candidate_landmarks.csv"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "raw" / "train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "candidate_image_samples.csv"
SAMPLES_PER_LANDMARK = 5
CHUNK_SIZE = 100_000
OUTPUT_COLUMNS = ["landmark_id", "image_id", "url"]


def _read_candidate_ids(path: Path) -> list[str]:
    """Read and validate candidate landmark IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Candidate file does not exist: {path}")

    candidates = pd.read_csv(path, usecols=["landmark_id"], dtype="string")
    ids = candidates["landmark_id"].dropna().str.strip()
    ids = ids[ids != ""].drop_duplicates().tolist()
    if not ids:
        raise ValueError(f"No landmark IDs found in {path}.")
    return ids


def sample_candidate_urls(
    candidates_path: Path = DEFAULT_CANDIDATES,
    train_path: Path = DEFAULT_TRAIN,
    output_path: Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """Read train metadata in chunks and save up to five URLs per candidate."""
    candidate_ids = _read_candidate_ids(candidates_path)
    remaining = set(candidate_ids)
    samples: list[pd.DataFrame] = []

    if not train_path.is_file():
        raise FileNotFoundError(f"Training metadata file does not exist: {train_path}")

    for chunk in pd.read_csv(
        train_path,
        usecols=["id", "url", "landmark_id"],
        dtype="string",
        chunksize=CHUNK_SIZE,
    ):
        matching = chunk[chunk["landmark_id"].isin(remaining)].copy()
        matching = matching.dropna(subset=["url"])
        matching["url"] = matching["url"].str.strip()
        matching = matching[matching["url"] != ""]
        if matching.empty:
            continue

        matching = matching.rename(columns={"id": "image_id"})
        for landmark_id, group in matching.groupby("landmark_id", sort=False):
            if landmark_id not in remaining:
                continue
            samples.append(group[["landmark_id", "image_id", "url"]].head(
                SAMPLES_PER_LANDMARK
            ))
            if sum(
                len(sample[sample["landmark_id"] == landmark_id]) for sample in samples
            ) >= SAMPLES_PER_LANDMARK:
                remaining.remove(landmark_id)

        if not remaining:
            break

    if samples:
        result = pd.concat(samples, ignore_index=True)
        result = result.groupby("landmark_id", sort=False, group_keys=False).head(
            SAMPLES_PER_LANDMARK
        )
        result["landmark_id"] = pd.Categorical(
            result["landmark_id"], categories=candidate_ids, ordered=True
        )
        result = result.sort_values("landmark_id").reset_index(drop=True)
        result["landmark_id"] = result["landmark_id"].astype("string")
        result = result[OUTPUT_COLUMNS]
    else:
        result = pd.DataFrame(columns=OUTPUT_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("Samples found by candidate landmark ID:")
    counts = result["landmark_id"].value_counts() if not result.empty else pd.Series(
        dtype="int64"
    )
    for landmark_id in candidate_ids:
        print(f"{landmark_id}: {int(counts.get(landmark_id, 0))}")
    print(f"\nSaved samples to: {output_path}")
    return result


def main() -> int:
    """Run URL sampling from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        sample_candidate_urls(args.candidates, args.train, args.output)
    except (FileNotFoundError, OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
