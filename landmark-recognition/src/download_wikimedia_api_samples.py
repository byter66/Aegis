"""Download GLDv2 samples through the Wikimedia Commons MediaWiki API."""

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
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "candidate_samples"
    / "wikimedia_api_samples"
)
DEFAULT_IMAGE_DIR = DEFAULT_OUTPUT_DIR / "images"
DEFAULT_LOG = DEFAULT_OUTPUT_DIR / "download_log.csv"
API_URL = "https://commons.wikimedia.org/w/api.php"
EXPECTED_ROWS = 1_000
EXPECTED_LANDMARKS = 50
EXPECTED_ROWS_PER_LANDMARK = 20
REQUEST_TIMEOUT_SECONDS = 15
DELAY_SECONDS = 2.0
USER_AGENT = (
    "Aegis-GLDv2-academic-project/1.0 "
    "(contact: project@example.invalid)"
)
LOG_COLUMNS = [
    "landmark_id",
    "image_id",
    "original_url",
    "commons_filename",
    "api_status",
    "api_http_status",
    "api_thumbnail_url",
    "returned_mime",
    "thumbnail_http_status",
    "download_status",
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


class RequestPacer:
    """Enforce a minimum delay between all API and thumbnail requests."""

    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.last_request = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self.last_request
        if self.last_request and elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self.last_request = time.monotonic()

    def wait_retry_after(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


def _retry_after_seconds(exc: HTTPError) -> float:
    value = exc.headers.get("Retry-After")
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0


def _extract_filename(original_url: str) -> str:
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
        pacer.wait_retry_after(_retry_after_seconds(exc))
        return None, exc.code, f"http_error: {exc.code} {exc.reason}"
    except (TimeoutError, socket.timeout, URLError, OSError, ValueError) as exc:
        return None, "", f"api_network_error: {exc}"


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
        pacer.wait_retry_after(_retry_after_seconds(exc))
        return None, exc.code, f"http_error: {exc.code} {exc.reason}"
    except (TimeoutError, socket.timeout, URLError, OSError) as exc:
        return None, "", f"thumbnail_network_error: {exc}"


def _read_input(path: Path) -> pd.DataFrame:
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
    return frame


def _preflight(frame: pd.DataFrame) -> None:
    counts = frame.groupby("landmark_id").size()
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} input rows, found {len(frame)}.")
    if frame["landmark_id"].nunique() != EXPECTED_LANDMARKS:
        raise ValueError("Expected exactly 50 unique landmark IDs.")
    if len(counts) != EXPECTED_LANDMARKS or not counts.eq(
        EXPECTED_ROWS_PER_LANDMARK
    ).all():
        raise ValueError("Expected exactly 20 rows per landmark.")


def _existing_image(image_dir: Path, image_id: str) -> Path | None:
    matches = sorted(image_dir.glob(f"{image_id}.*"))
    return matches[0] if matches else None


def _validate_image(data: bytes) -> str:
    with Image.open(BytesIO(data)) as image:
        image.verify()
        image_format = image.format or ""
    with Image.open(BytesIO(data)) as image:
        image.load()
    return image_format


def _api_info(
    payload: dict[str, object],
) -> tuple[dict[str, object] | None, str]:
    query = payload.get("query")
    if not isinstance(query, dict):
        return None, "API response has no query object."
    pages = query.get("pages")
    if not isinstance(pages, dict) or not pages:
        return None, "API response has no pages."
    page = next(iter(pages.values()))
    if not isinstance(page, dict):
        return None, "API page result is malformed."
    imageinfo = page.get("imageinfo")
    if not isinstance(imageinfo, list) or not imageinfo:
        return None, "API result has no imageinfo."
    info = imageinfo[0]
    if not isinstance(info, dict):
        return None, "API imageinfo result is malformed."
    return info, ""


def download_samples(
    input_path: Path = DEFAULT_INPUT,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    log_path: Path = DEFAULT_LOG,
) -> pd.DataFrame:
    """Download one API-provided thumbnail for every input row sequentially."""
    frame = _read_input(input_path)
    _preflight(frame)
    image_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pacer = RequestPacer(DELAY_SECONDS)
    records: list[dict[str, str | int]] = []

    for row in frame.itertuples(index=False):
        record: dict[str, str | int] = {
            "landmark_id": str(row.landmark_id),
            "image_id": str(row.image_id),
            "original_url": str(row.url),
            "commons_filename": "",
            "api_status": "failed",
            "api_http_status": "",
            "api_thumbnail_url": "",
            "returned_mime": "",
            "thumbnail_http_status": "",
            "download_status": "api_failed",
            "local_path": "",
            "error": "",
        }
        try:
            filename = _extract_filename(str(row.url))
            record["commons_filename"] = filename
        except ValueError as exc:
            record["error"] = str(exc)
            records.append(record)
            continue

        payload, api_status, api_error = _api_request(filename, pacer)
        record["api_http_status"] = api_status
        if payload is None:
            record["download_status"] = "network_error" if not api_status else "api_failed"
            record["error"] = api_error
            records.append(record)
            continue

        info, info_error = _api_info(payload)
        if info is None:
            record["download_status"] = "missing_api_result"
            record["error"] = info_error
            records.append(record)
            continue

        record["api_status"] = "success"
        record["returned_mime"] = str(info.get("mime", ""))
        thumbnail_url = str(info.get("thumburl", ""))
        record["api_thumbnail_url"] = thumbnail_url
        if not thumbnail_url:
            record["download_status"] = "missing_thumbnail"
            record["error"] = "API returned no thumbnail URL."
            records.append(record)
            continue
        if not record["returned_mime"].startswith("image/"):
            record["download_status"] = "invalid_image"
            record["error"] = f"Non-image MIME type: {record['returned_mime']}"
            records.append(record)
            continue

        existing_path = _existing_image(image_dir, str(row.image_id))
        if existing_path is not None:
            record["download_status"] = "reused"
            record["local_path"] = str(existing_path.relative_to(PROJECT_ROOT))
            records.append(record)
            continue

        data, thumbnail_status, thumbnail_error = _download_thumbnail(
            thumbnail_url, pacer
        )
        record["thumbnail_http_status"] = thumbnail_status
        if data is None:
            record["download_status"] = (
                "network_error" if not thumbnail_status else "thumbnail_failed"
            )
            record["error"] = thumbnail_error
            records.append(record)
            continue
        try:
            image_format = _validate_image(data)
            extension = FORMAT_EXTENSIONS.get(image_format, ".img")
            output_path = image_dir / f"{row.image_id}{extension}"
            output_path.write_bytes(data)
            record["download_status"] = "downloaded"
            record["local_path"] = str(output_path.relative_to(PROJECT_ROOT))
        except (UnidentifiedImageError, SyntaxError, ValueError, OSError) as exc:
            record["download_status"] = "invalid_image"
            record["error"] = f"invalid_image: {exc}"
        records.append(record)

    result = pd.DataFrame(records, columns=LOG_COLUMNS)
    result.to_csv(log_path, index=False)
    return result


def _print_summary(result: pd.DataFrame) -> None:
    successful = result["download_status"].isin(["downloaded", "reused"])
    invalid = result["download_status"].eq("invalid_image")
    failed = ~successful & ~invalid
    print(f"Total input rows: {len(result)}")
    print(f"Unique landmark IDs: {result['landmark_id'].nunique()}")
    print(f"API requests attempted: {len(result)}")
    print(f"API metadata successes: {int(result['api_status'].eq('success').sum())}")
    print(f"API metadata failures: {int((result['api_status'] != 'success').sum())}")
    print(
        "Thumbnail downloads attempted: "
        f"{int(result['download_status'].isin(['downloaded', 'thumbnail_failed', 'invalid_image', 'network_error']).sum())}"
    )
    print(f"Successful downloads: {int(successful.sum())}")
    print(f"Failed downloads: {int(failed.sum())}")
    print(f"Invalid images: {int(invalid.sum())}")
    print(f"Success rate: {successful.mean():.2%}")
    print("\nSuccessful images per landmark:")
    counts = result.loc[successful].groupby("landmark_id").size()
    for landmark_id in result["landmark_id"].drop_duplicates():
        print(f"{landmark_id}: {int(counts.get(landmark_id, 0))}")
    for threshold in (5, 10, 15, 20):
        print(
            f"Candidates with at least {threshold} successful images: "
            f"{int((counts >= threshold).sum())}"
        )

    summary = (
        result.assign(_successful=successful)
        .groupby("landmark_id", sort=False)
        .agg(tested=("image_id", "size"), downloaded=("_successful", "sum"))
        .reset_index()
    )
    summary["failed"] = summary["tested"] - summary["downloaded"]
    summary["success_rate"] = summary["downloaded"] / summary["tested"]
    summary = summary.sort_values(
        ["downloaded", "success_rate", "landmark_id"],
        ascending=[False, False, True],
    )
    print("\nCandidate summary:")
    print(summary.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    args = parser.parse_args()
    try:
        frame = _read_input(args.input)
        _preflight(frame)
        print(f"Preflight input: {args.input}")
        print(f"Input rows: {len(frame)}")
        print(f"Unique landmark IDs: {frame['landmark_id'].nunique()}")
        print("Rows per landmark: 20")
        result = download_samples(args.input, args.image_dir, args.log)
        _print_summary(result)
        print(f"\nSaved images to: {args.image_dir}")
        print(f"Saved download log to: {args.log}")
    except (FileNotFoundError, OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
