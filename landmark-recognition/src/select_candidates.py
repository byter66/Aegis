"""Select the most frequent landmark classes for the GLDv2 subset."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "landmark_class_counts.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "candidate_landmarks.csv"
MIN_IMAGE_COUNT = 200
NUM_CANDIDATES = 50
OUTPUT_COLUMNS = ("landmark_id", "image_count", "rank")


def select_candidates(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
) -> list[dict[str, int | str]]:
    """Read ranked class counts and save the top eligible landmark IDs."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or not {"landmark_id", "image_count"} <= set(
            reader.fieldnames
        ):
            raise ValueError(
                "Input CSV must contain landmark_id and image_count columns."
            )

        eligible: list[dict[str, int | str]] = []
        for row in reader:
            landmark_id = (row.get("landmark_id") or "").strip()
            if not landmark_id:
                continue
            try:
                image_count = int(row["image_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid image_count for landmark {landmark_id!r}."
                ) from exc
            if image_count >= MIN_IMAGE_COUNT:
                eligible.append(
                    {"landmark_id": landmark_id, "image_count": image_count}
                )

    eligible.sort(key=lambda row: (-int(row["image_count"]), str(row["landmark_id"])))
    candidates = [
        {**row, "rank": rank}
        for rank, row in enumerate(eligible[:NUM_CANDIDATES], start=1)
    ]
    if len(candidates) < NUM_CANDIDATES:
        raise ValueError(
            f"Found only {len(candidates)} eligible candidates; "
            f"{NUM_CANDIDATES} are required."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(candidates)

    print(f"Selected {len(candidates)} candidates from {input_path}:")
    print("rank  landmark_id  image_count")
    print("----  -----------  -----------")
    for candidate in candidates:
        print(
            f"{candidate['rank']:>4}  {candidate['landmark_id']:>11}  "
            f"{candidate['image_count']:>11}"
        )
    print(f"\nSaved candidates to: {output_path}")
    return candidates


def main() -> int:
    """Run candidate selection from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        select_candidates(args.input, args.output)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
