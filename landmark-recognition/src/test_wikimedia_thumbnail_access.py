"""Experiment with sequential Wikimedia Commons thumbnail retrieval."""

from __future__ import annotations

import argparse
import socket
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import pandas as pd
from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_samples"
    / "candidate_image_samples_20.csv"
)
DEFAULT_IMAGE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "candidate_samples"
    / "wikimedia_thumbnail_test"
)
DEFAULT_LOG = (
    PROJECT_ROOT / "data" / "processed" / "candidate_samples"
    / "wikimedia_thumbnail_test_log.csv"
)
SAMPLES_PER_LANDMARK = 5
REQUEST_TIMEOUT_SECONDS = 15
DELAY_SECONDS = 2.0
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
)
LOG_COLUMNS = [
    "landmark_id",
    "image_id",
    "original_url",
    "thumbnail_url",
    "status",
    "http_status_code",
    "local_path",
    "error",
]
FORMAT_EXTENSIONS = {
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tif",
    "WEBP": ".webp",
}


def build_thumbnail_url(original_url: str, width: int = 320) -> str:
    """Build a Wikimedia Commons thumbnail URL from an original file URL."""
    parsed = urlsplit(original_url)
    if parsed.hostname != "upload.wikimedia.org":
        raise ValueError("URL is not hosted on upload.wikimedia.org.")

    path_parts = parsed.path.split("/")
    if len(path_parts) < 6 or path_parts[1:3] != ["wikipedia", "commons"]:
        raise ValueError("URL is not a Wikimedia Commons file path.")
    if path_parts[3] == "thumb":
        raise ValueError("URL is not a direct Wikimedia Commons file URL.")

    hash_one, hash_two = path_parts[3], path_parts[4]
    encoded_filename = "/".join(path_parts[5:])
    filename = unquote(encoded_filename)
    if not filename:
        raise ValueError("Wikimedia URL has no filename.")
    thumbnail_filename = f"{width}px-{filename}"
    thumbnail_path = (
        "/wikipedia/commons/thumb/"
        f"{hash_one}/{hash_two}/{quote(filename, safe='')}/"
        f"{quote(thumbnail_filename, safe='')}"
    )
    return urlunsplit(
        (parsed.scheme, parsed.netloc, thumbnail_path, parsed.query, parsed.fragment)
    )


def _read_samples(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    frame = pd.read_csv(path, dtype="string")
    required = {"landmark_id", "image_id", "url"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Input CSV must contain {sorted(required)}.")
    frame = frame[["landmark_id", "image_id", "url"]].copy()
    frame["landmark_id"] = frame["landmark_id"].str.strip()
    frame["image_id"] = frame["image_id"].str.strip()
    frame["url"] = frame["url"].str.strip()
    selected = frame.groupby("landmark_id", sort=False, dropna=False).head(
        SAMPLES_PER_LANDMARK
    )
    if len(selected) != 250 or selected["landmark_id"].nunique() != 50:
        raise ValueError("Expected exactly 250 rows across 50 landmark IDs.")
    if selected.groupby("landmark_id").size().min() != SAMPLES_PER_LANDMARK:
        raise ValueError("Every landmark must have five selected URLs.")
    return selected.reset_index(drop=True)


def _request_thumbnail(url: str) -> tuple[bytes, int]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read(), response.status


def _validate_image(data: bytes) -> str:
    with Image.open(BytesIO(data)) as image:
        image.verify()
        image_format = image.format or ""
    with Image.open(BytesIO(data)) as image:
        image.load()
    return image_format


def _error_text(exc: Exception) -> str:
    if isinstance(exc, HTTPError):
        return f"http_error: {exc.code} {exc.reason}"
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "connection_timeout"
    if isinstance(exc, URLError):
        return f"network_error: {exc.reason}"
    return f"request_error: {exc}"


def test_thumbnail_access(
    input_path: Path = DEFAULT_INPUT,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    log_path: Path = DEFAULT_LOG,
) -> pd.DataFrame:
    """Request and validate the first five thumbnails for each landmark."""
    samples = _read_samples(input_path)
    image_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | int]] = []

    print(f"Input rows: {len(samples)}")
    print(f"Unique landmark IDs: {samples['landmark_id'].nunique()}")
    print("URLs per landmark: 5")
    print("Worker count: 1")
    print("Request strategy: sequential + conservative delay")

    for row in samples.itertuples(index=False):
        landmark_id = str(row.landmark_id)
        image_id = str(row.image_id)
        original_url = str(row.url)
        record: dict[str, str | int] = {
            "landmark_id": landmark_id,
            "image_id": image_id,
            "original_url": original_url,
            "thumbnail_url": "",
            "status": "failed",
            "http_status_code": "",
            "local_path": "",
            "error": "",
        }
        try:
            thumbnail_url = build_thumbnail_url(original_url)
            record["thumbnail_url"] = thumbnail_url
            data, status_code = _request_thumbnail(thumbnail_url)
            record["http_status_code"] = status_code
            if not data:
                record["error"] = "empty_response"
            else:
                image_format = _validate_image(data)
                extension = FORMAT_EXTENSIONS.get(image_format, ".img")
                output_path = image_dir / f"{image_id}{extension}"
                output_path.write_bytes(data)
                record["status"] = "downloaded"
                record["local_path"] = str(output_path.relative_to(PROJECT_ROOT))
        except HTTPError as exc:
            record["http_status_code"] = exc.code
            record["error"] = _error_text(exc)
        except (UnidentifiedImageError, SyntaxError, ValueError) as exc:
            record["status"] = "invalid_image"
            record["error"] = f"invalid_image: {exc}"
        except (TimeoutError, socket.timeout, URLError, OSError) as exc:
            record["error"] = _error_text(exc)
        records.append(record)
        time.sleep(DELAY_SECONDS)

    result = pd.DataFrame(records, columns=LOG_COLUMNS)
    result.to_csv(log_path, index=False)
    return result


def _print_summary(result: pd.DataFrame) -> None:
    downloaded = result["status"].eq("downloaded")
    invalid = result["status"].eq("invalid_image")
    print("\nThumbnail access summary:")
    print(f"Total URLs tested: {len(result)}")
    print(f"Successful downloads: {int(downloaded.sum())}")
    print(f"Failed downloads: {int((~downloaded & ~invalid).sum())}")
    print(f"Invalid images: {int(invalid.sum())}")
    print(f"Success rate: {downloaded.mean():.2%}")
    print("\nHTTP status-code distribution:")
    print(result["http_status_code"].replace("", "<blank>").value_counts().to_string())
    print("\nSuccessful images per landmark:")
    counts = result.loc[downloaded, "landmark_id"].value_counts()
    for landmark_id in result["landmark_id"].drop_duplicates():
        print(f"{landmark_id}: {int(counts.get(landmark_id, 0))}")
    print(f"\nHTTP 429 responses occurred: {int((result['http_status_code'] == 429).sum()) > 0}")
    print(f"429 count: {int((result['http_status_code'] == 429).sum())}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    try:
        result = test_thumbnail_access(args.input, args.image_dir, args.log)
        _print_summary(result)
        print(f"\nSaved thumbnail test log to: {args.log}")
    except (FileNotFoundError, OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
