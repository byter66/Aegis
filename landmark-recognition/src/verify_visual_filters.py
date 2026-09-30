"""Read-only verification of the visual-filter image moves."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_samples"
    / "wikimedia_api_samples"
    / "images"
)
REJECTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_samples"
    / "wikimedia_api_samples"
    / "rejected_images"
)

INTENDED_IDS = (
    "84985455ed896f",
    "9db58a779a603512",
    "3af6daa10dbbbab",
    "2073185de9464e00",
    "408990c1472293fa",
    "585c38dc1627bc27",
    "7487d9e5ea99d461",
    "a875ef4e30b331a0",
    "b49a6bd3bd1c2922",
    "0e4c8e91b7d9208e",
    "3644473156428951",
    "1e41af907db725a4",
    "9c913c52b9ad345d",
    "554640a3c2e1fe30",
    "fdb4b7a41ef23da6",
    "bc9a24fd5732f90c",
    "da7108e7fd0118e",
    "552dc939dbc8ccf1",
    "36363f8bd0e32461",
    "d7400f175bf78f02",
    "8ac4a3687a9fb3e2",
    "376a37e33ab60d76",
    "8ac986756d7047e7",
    "5b82eaa2707fd949",
    "c28ba974656d8879",
    "531f4442c745e08b",
    "5ed314bab8075930",
    "6897e17953306c3c",
    "6d9b1f06226cfd4",
    "e7f10882ce0289f7",
    "5152130c1eb6f072",
    "ad57a3f63d284e9",
    "53a8011a771b1793",
    "1b26b8b2767cbf31",
    "0c4b22c120e2e1b5",
    "7da2991893476ef3",
    "b1398bcc79d9f451",
)


def _stems(directory: Path) -> tuple[set[str], list[str]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    files = sorted(path for path in directory.iterdir() if path.is_file())
    return {path.stem for path in files}, [path.name for path in files]


def verify() -> None:
    source_stems, _ = _stems(SOURCE_DIR)
    rejected_stems, rejected_files = _stems(REJECTED_DIR)
    duplicate_stems = sorted(source_stems & rejected_stems)

    print("image_id | location")
    print("-" * 30)
    for image_id in INTENDED_IDS:
        if image_id in source_stems:
            location = "SOURCE"
        elif image_id in rejected_stems:
            location = "REJECTED"
        else:
            location = "MISSING"
        print(f"{image_id} | {location}")

    found_source = sum(image_id in source_stems for image_id in INTENDED_IDS)
    found_rejected = sum(image_id in rejected_stems for image_id in INTENDED_IDS)
    missing = len(INTENDED_IDS) - found_source - found_rejected

    print("\nFiles currently in rejected_images/:")
    for filename in rejected_files:
        print(filename)

    print("\nSummary:")
    print(f"SOURCE total: {sum(1 for _ in SOURCE_DIR.iterdir() if _.is_file())}")
    print(f"REJECTED total: {len(rejected_files)}")
    print(f"Intended IDs: {len(INTENDED_IDS)}")
    print(f"Found in SOURCE: {found_source}")
    print(f"Found in REJECTED: {found_rejected}")
    print(f"Missing: {missing}")
    print(f"Duplicates: {len(duplicate_stems)}")
    if duplicate_stems:
        print("Duplicate stems:")
        for stem in duplicate_stems:
            print(stem)


def main() -> int:
    try:
        verify()
    except (FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
