"""Test Wikimedia Commons API metadata and thumbnail access for 10 samples."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlencode, urlsplit
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
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_samples"
    / "wikimedia_api_test_images"
)
DEFAULT_LOG = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_samples"
    / "wikimedia_api_test_log.csv"
)
API_URL = "https://commons.wikimedia.org/w/api.php"
REQUEST_TIMEOUT_SECONDS = 15
DELAY_SECONDS = 2.0
SAMPLES_PER_LANDMARK = 2
LANDMARKS_TO_TEST = 5
USER_AGENT = (
    "Aegis-GLDv2-academic-project/1.0 "
    "(contact: project@example.invalid)"
)
LOG_COLUMNS = [
    "landmark_id",
    "image_id",
    "original_url",
    "api_status",
    "api_error",
    "returned_mime",
    "thumbnail_url",
    "download_status",
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


def _extract_filename(original_url: str) -> str:
    """Extract and decode the file name from a Commons original-file path."""
    parsed = urlsplit(original_url)
    if parsed.hostname != "upload.wikimedia.org":
        raise ValueError("URL is not hosted on upload.wikimedia.org.")
    path_parts = parsed.path.split("/")
    if len(path_parts) < 6 or path_parts[1:3] != ["wikipedia", "commons"]:
        raise ValueError("URL is not a Wikimedia Commons file path.")
    filename = unquote("/".join(path_parts[5:]))
    if not filename or "/" in filename:
        raise ValueError("Could not extract a single Wikimedia filename.")
    return filename


class RequestPacer:
    """Ensure all API and thumbnail requests are separated by a fixed delay."""

    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.last_request = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_request
        if self.last_request and elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self.last_request = time.monotonic()


def _retry_after_seconds(exc: HTTPError) -> float:
    value = exc.headers.get("Retry-After")
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0


def _api_request(
    filename: str,
    pacer: RequestPacer,
) -> tuple[dict[str, object] | None, int | str, str]:
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "iiurlwidth": "320",
        "titles": f"File:{filename}",
    }
    request = Request(
        f"{API_URL}?{urlencode(params)}",
        headers={"User-Agent": USER_AGENT},
    )
    pacer.wait()
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.load(response), response.status, ""
    except HTTPError as exc:
        retry_after = _retry_after_seconds(exc)
        if retry_after:
            time.sleep(retry_after)
        return None, exc.code, f"http_error: {exc.code} {exc.reason}"
    except (TimeoutError, socket.timeout, URLError, OSError, ValueError) as exc:
        return None, "", f"api_request_error: {exc}"


def _download_thumbnail(
    thumbnail_url: str,
    pacer: RequestPacer,
) -> tuple[bytes | None, int | str, str]:
    request = Request(thumbnail_url, headers={"User-Agent": USER_AGENT})
    pacer.wait()
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.read(), response.status, ""
    except HTTPError as exc:
        retry_after = _retry_after_seconds(exc)
        if retry_after:
            time.sleep(retry_after)
        return None, exc.code, f"http_error: {exc.code} {exc.reason}"
    except (TimeoutError, socket.timeout, URLError, OSError) as exc:
        return None, "", f"download_request_error: {exc}"


def _image_format(data: bytes) -> str:
    with Image.open(BytesIO(data)) as image:
        image.verify()
        image_format = image.format or ""
    with Image.open(BytesIO(data)) as image:
        image.load()
    return image_format


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
    landmark_ids = frame["landmark_id"].drop_duplicates().head(LANDMARKS_TO_TEST)
    selected = frame[frame["landmark_id"].isin(landmark_ids)].groupby(
        "landmark_id", sort=False, group_keys=False
    ).head(SAMPLES_PER_LANDMARK)
    if len(landmark_ids) != LANDMARKS_TO_TEST or len(selected) != 10:
        raise ValueError("Expected 10 rows from the first 5 landmark IDs.")
    return selected.reset_index(drop=True)


def test_api_access(
    input_path: Path = DEFAULT_INPUT,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    log_path: Path = DEFAULT_LOG,
) -> pd.DataFrame:
    """Query the Commons API and validate its returned thumbnail URLs."""
    samples = _read_samples(input_path)
    image_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pacer = RequestPacer(DELAY_SECONDS)
    records: list[dict[str, str | int]] = []

    for row in samples.itertuples(index=False):
        record: dict[str, str | int] = {
            "landmark_id": str(row.landmark_id),
            "image_id": str(row.image_id),
            "original_url": str(row.url),
            "api_status": "failed",
            "api_error": "",
            "returned_mime": "",
            "thumbnail_url": "",
            "download_status": "not_attempted",
            "http_status_code": "",
            "local_path": "",
            "error": "",
        }
        try:
            filename = _extract_filename(str(row.url))
        except ValueError as exc:
            record["api_error"] = str(exc)
            records.append(record)
            continue

        payload, api_code, api_error = _api_request(filename, pacer)
        record["http_status_code"] = api_code
        record["api_error"] = api_error
        if payload is None:
            records.append(record)
            continue

        pages = payload.get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        image_info = page.get("imageinfo", [])
        if not image_info:
            record["api_error"] = "API returned no imageinfo."
            records.append(record)
            continue
        info = image_info[0]
        record["api_status"] = "success"
        record["returned_mime"] = str(info.get("mime", ""))
        record["thumbnail_url"] = str(info.get("thumburl", ""))
        if not record["thumbnail_url"]:
            record["api_error"] = "API response contained no thumbnail URL."
            records.append(record)
            continue

        record["download_status"] = "failed"
        data, download_code, download_error = _download_thumbnail(
            str(record["thumbnail_url"]), pacer
        )
        record["http_status_code"] = download_code
        record["error"] = download_error
        if data is None:
            records.append(record)
            continue
        try:
            image_format = _image_format(data)
            extension = FORMAT_EXTENSIONS.get(image_format, ".img")
            output_path = image_dir / f"{record['image_id']}{extension}"
            output_path.write_bytes(data)
            record["download_status"] = "downloaded"
            record["local_path"] = str(output_path.relative_to(PROJECT_ROOT))
            record["error"] = ""
        except (UnidentifiedImageError, SyntaxError, ValueError, OSError) as exc:
            record["download_status"] = "invalid_image"
            record["error"] = f"invalid_image: {exc}"
        records.append(record)

    result = pd.DataFrame(records, columns=LOG_COLUMNS)
    result.to_csv(log_path, index=False)
    return result


def _print_summary(result: pd.DataFrame) -> None:
    api_success = result["api_status"].eq("success")
    api_failure = ~api_success
    attempted = result["download_status"].ne("not_attempted")
    downloaded = result["download_status"].eq("downloaded")
    invalid = result["download_status"].eq("invalid_image")
    print(f"Total API requests: {len(result)}")
    print(f"API metadata successes: {int(api_success.sum())}")
    print(f"API metadata failures: {int(api_failure.sum())}")
    print(f"Thumbnail downloads attempted: {int(attempted.sum())}")
    print(f"Successful downloads: {int(downloaded.sum())}")
    print(f"Failed downloads: {int((attempted & ~downloaded & ~invalid).sum())}")
    print(f"Invalid images: {int(invalid.sum())}")
    print("\nHTTP status-code distribution:")
    print(result["http_status_code"].replace("", "<blank>").value_counts().to_string())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    try:
        result = test_api_access(args.input, args.image_dir, args.log)
        _print_summary(result)
        print(f"\nSaved API test log to: {args.log}")
    except (FileNotFoundError, OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
