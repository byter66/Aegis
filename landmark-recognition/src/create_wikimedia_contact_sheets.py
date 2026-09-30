"""Create landmark-grouped contact sheets from locally downloaded images."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_samples"
    / "wikimedia_api_samples"
    / "download_log.csv"
)
DEFAULT_SHEET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_samples"
    / "wikimedia_api_samples"
    / "contact_sheets"
)
DEFAULT_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_samples"
    / "wikimedia_api_samples"
    / "wikimedia_file_metadata.csv"
)
THUMBNAIL_SIZE = (180, 140)
LABEL_HEIGHT = 42
COLUMNS = 5
BACKGROUND = "white"
TEXT_COLOR = "black"
METADATA_COLUMNS = [
    "landmark_id",
    "image_id",
    "commons_filename",
    "original_url",
    "api_thumbnail_url",
    "returned_mime",
    "local_path",
]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _read_downloaded_log(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Download log does not exist: {path}")
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    required = {
        "landmark_id",
        "image_id",
        "commons_filename",
        "original_url",
        "api_thumbnail_url",
        "returned_mime",
        "download_status",
        "local_path",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Download log is missing columns: {sorted(missing)}")
    return frame[frame["download_status"].isin(["downloaded", "reused"])].copy()


def _resolve_image_path(local_path: str) -> Path:
    path = Path(local_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _make_sheet(
    group: pd.DataFrame,
    output_path: Path,
) -> tuple[int, int]:
    """Create one sheet and return (valid images, skipped images)."""
    tile_width, tile_height = THUMBNAIL_SIZE
    rows = math.ceil(len(group) / COLUMNS)
    sheet = Image.new(
        "RGB",
        (COLUMNS * tile_width, rows * (tile_height + LABEL_HEIGHT)),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(sheet)
    label_font = _font(12)
    valid = 0
    skipped = 0

    for index, record in enumerate(group.itertuples(index=False)):
        image_path = _resolve_image_path(str(record.local_path))
        left = (index % COLUMNS) * tile_width
        top = (index // COLUMNS) * (tile_height + LABEL_HEIGHT)
        try:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
                image.thumbnail(THUMBNAIL_SIZE)
                x = left + (tile_width - image.width) // 2
                y = top + (tile_height - image.height) // 2
                sheet.paste(image, (x, y))
            valid += 1
        except (FileNotFoundError, OSError, UnidentifiedImageError):
            skipped += 1
            draw.rectangle(
                (left + 2, top + 2, left + tile_width - 2, top + tile_height - 2),
                outline="red",
                width=2,
            )
            draw.text((left + 8, top + 60), "unreadable", fill="red", font=label_font)

        label = f"{record.image_id}"
        draw.text((left + 4, top + tile_height + 3), label, fill=TEXT_COLOR, font=label_font)

    sheet.save(output_path, format="JPEG", quality=90)
    return valid, skipped


def create_contact_sheets(
    log_path: Path = DEFAULT_LOG,
    sheet_dir: Path = DEFAULT_SHEET_DIR,
    metadata_path: Path = DEFAULT_METADATA,
) -> None:
    """Extract recorded Wikimedia metadata and create one sheet per landmark."""
    downloaded = _read_downloaded_log(log_path)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded[METADATA_COLUMNS].to_csv(metadata_path, index=False)

    total_valid = 0
    total_skipped = 0
    sheet_count = 0
    for landmark_id, group in downloaded.groupby("landmark_id", sort=True):
        sheet_path = sheet_dir / f"landmark_{landmark_id}.jpg"
        valid, skipped = _make_sheet(group, sheet_path)
        total_valid += valid
        total_skipped += skipped
        sheet_count += 1
        print(
            f"landmark {landmark_id}: {valid} images, {skipped} skipped -> "
            f"{sheet_path}"
        )

    print("\nContact-sheet summary:")
    print(f"Downloaded log rows used: {len(downloaded)}")
    print(f"Landmark IDs represented: {downloaded['landmark_id'].nunique()}")
    print(f"Contact sheets created: {sheet_count}")
    print(f"Readable images placed: {total_valid}")
    print(f"Unreadable/missing images skipped: {total_skipped}")
    print(f"Metadata extracted to: {metadata_path}")
    print(f"Contact sheets saved to: {sheet_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--sheet-dir", type=Path, default=DEFAULT_SHEET_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()
    try:
        create_contact_sheets(args.log, args.sheet_dir, args.metadata)
    except (FileNotFoundError, OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
