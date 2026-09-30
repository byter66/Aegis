"""Move explicitly rejected Wikimedia sample images to a separate directory."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = (
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

REJECTED_IMAGE_IDS = (
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


def apply_visual_filters() -> None:
    """Move exact filename-stem matches without deleting any files."""
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Source image directory does not exist: {IMAGE_DIR}")

    REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    files_by_stem: dict[str, list[Path]] = {}
    for path in IMAGE_DIR.iterdir():
        if path.is_file():
            files_by_stem.setdefault(path.stem, []).append(path)

    moved: list[str] = []
    missing: list[str] = []
    for image_id in REJECTED_IMAGE_IDS:
        matches = files_by_stem.get(image_id, [])
        if not matches:
            missing.append(image_id)
            continue

        for source in matches:
            destination = REJECTED_DIR / source.name
            try:
                shutil.move(str(source), str(destination))
                moved.append(source.name)
            except OSError as exc:
                print(f"Could not move {source.name}: {exc}", file=sys.stderr)

    print("Visual-filter summary:")
    print(f"Requested removals: {len(REJECTED_IMAGE_IDS)}")
    print(f"Successfully moved: {len(moved)}")
    print(f"Missing: {len(missing)}")
    print("\nMoved files:")
    for filename in moved:
        print(filename)
    print("\nMissing files:")
    for image_id in missing:
        print(image_id)


def main() -> int:
    try:
        apply_visual_filters()
    except (FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
