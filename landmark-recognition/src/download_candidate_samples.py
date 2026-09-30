"""Test candidate image URL availability without touching the source metadata."""

from __future__ import annotations

import argparse
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "candidate_samples"
DEFAULT_IMAGE_DIR = DEFAULT_OUTPUT_DIR / "availability_test_images"
DEFAULT_LOG = DEFAULT_OUTPUT_DIR / "availability_test_log.csv"
DEFAULT_SUMMARY = DEFAULT_OUTPUT_DIR / "candidate_availability_summary.csv"
MAX_URLS_PER_LANDMARK = 20
MAX_WORKERS = 8
REQUEST_TIMEOUT_SECONDS = 10
MAX_RETRIES = 1
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
)
LOG_COLUMNS = [
    "landmark_id",
    "image_id",
    "url",
    "status",
    "http_status_code",
    "local_path",
    "error",
]
SUMMARY_COLUMNS = [
    "landmark_id",
    "urls_tested",
    "downloaded",
    "failed",
    "invalid_images",
    "success_rate",
]
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FORMAT_EXTENSIONS = {
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tif",
    "WEBP": ".webp",
}


def _extension_from_url(url: str) -> str:
    suffix = Path(unquote(urlsplit(url).path)).suffix.lower()
    return suffix if suffix in IMAGE_EXTENSIONS else ""


def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return "connection_timeout"
    if isinstance(exc, HTTPError):
        return f"http_error: {exc.code} {exc.reason}"
    if isinstance(exc, URLError):
        return f"network_error: {exc.reason}"
    return f"network_error: {exc}"


def _test_url(row: tuple[str, str, str]) -> dict[str, str | int]:
    """Fetch and validate one URL, returning bytes only for valid images."""
    landmark_id, image_id, url = row
    record: dict[str, str | int] = {
        "landmark_id": landmark_id,
        "image_id": image_id,
        "url": url,
        "status": "failed",
        "http_status_code": "",
        "local_path": "",
        "error": "",
        "_image_bytes": b"",
        "_image_format": "",
    }
    if not url:
        record["error"] = "missing_url"
        return record
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        record["error"] = "invalid_url"
        return record

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                record["http_status_code"] = response.status
                data = response.read()
            if not data:
                record["error"] = "empty_response"
                return record
            with Image.open(BytesIO(data)) as image:
                image.verify()
                image_format = image.format or ""
            with Image.open(BytesIO(data)) as image:
                image.load()
            record["status"] = "downloaded"
            record["_image_bytes"] = data
            record["_image_format"] = image_format
            return record
        except HTTPError as exc:
            record["http_status_code"] = exc.code
            last_error = exc
        except (TimeoutError, socket.timeout, URLError, OSError) as exc:
            last_error = exc
        except (UnidentifiedImageError, SyntaxError, ValueError) as exc:
            record["status"] = "invalid_image"
            record["error"] = f"invalid_image: {exc}"
            return record
        if attempt < MAX_RETRIES:
            continue

    record["error"] = _failure_reason(last_error or OSError("request failed"))
    return record


def _read_input(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    frame = pd.read_csv(path, dtype="string")
    required = {"landmark_id", "image_id", "url"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Input CSV must contain {sorted(required)}; found {list(frame.columns)}."
        )
    frame = frame[["landmark_id", "image_id", "url"]].copy()
    frame["_landmark_order"] = pd.factorize(frame["landmark_id"])[0]
    frame["_candidate_row"] = frame.groupby("landmark_id", dropna=False).cumcount()
    return frame.loc[
        frame["_candidate_row"] < MAX_URLS_PER_LANDMARK,
        ["landmark_id", "image_id", "url"],
    ]


def _save_valid_image(record: dict[str, str | int], image_dir: Path) -> None:
    image_id = str(record["image_id"])
    image_format = str(record["_image_format"])
    extension = _extension_from_url(str(record["url"])) or FORMAT_EXTENSIONS.get(
        image_format, ".img"
    )
    output_path = image_dir / f"{image_id}{extension}"
    output_path.write_bytes(record["_image_bytes"])
    try:
        record["local_path"] = str(output_path.relative_to(PROJECT_ROOT))
    except ValueError:
        record["local_path"] = str(output_path)
    record.pop("_image_bytes", None)
    record.pop("_image_format", None)


def _write_summary(records: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows = []
    for landmark_id, group in records.groupby("landmark_id", sort=False, dropna=False):
        downloaded = int(group["status"].eq("downloaded").sum())
        invalid = int(group["status"].eq("invalid_image").sum())
        tested = len(group)
        rows.append(
            {
                "landmark_id": landmark_id,
                "urls_tested": tested,
                "downloaded": downloaded,
                "failed": tested - downloaded - invalid,
                "invalid_images": invalid,
                "success_rate": downloaded / tested if tested else 0.0,
            }
        )
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary = summary.sort_values(
        ["success_rate", "downloaded"], ascending=[False, False], kind="mergesort"
    )
    summary.to_csv(path, index=False)
    return summary


def test_availability(
    input_path: Path = DEFAULT_INPUT,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    log_path: Path = DEFAULT_LOG,
    summary_path: Path = DEFAULT_SUMMARY,
) -> pd.DataFrame:
    """Test up to 20 URLs per candidate concurrently and save detailed results."""
    samples = _read_input(input_path)
    image_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        (
            "" if pd.isna(row.landmark_id) else str(row.landmark_id),
            "" if pd.isna(row.image_id) else str(row.image_id),
            "" if pd.isna(row.url) else str(row.url),
        )
        for row in samples.itertuples(index=False)
    ]
    duplicate_ids = samples["image_id"].notna() & samples["image_id"].duplicated(
        keep="first"
    )
    records: list[dict[str, str | int]] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for index, row in enumerate(rows):
            if duplicate_ids.iloc[index]:
                records.append(
                    {
                        "landmark_id": row[0],
                        "image_id": row[1],
                        "url": row[2],
                        "status": "failed",
                        "http_status_code": "",
                        "local_path": "",
                        "error": "duplicate_image_id",
                    }
                )
            else:
                futures[executor.submit(_test_url, row)] = index
        completed: dict[int, dict[str, str | int]] = {}
        for future in as_completed(futures):
            completed[futures[future]] = future.result()
        records.extend(completed.values())

    records.sort(key=lambda record: (str(record["landmark_id"]), str(record["image_id"])))
    for record in records:
        if record["status"] == "downloaded":
            try:
                _save_valid_image(record, image_dir)
            except OSError as exc:
                record["status"] = "failed"
                record["error"] = f"local_write_error: {exc}"
                record["local_path"] = ""
                record.pop("_image_bytes", None)
                record.pop("_image_format", None)
    result = pd.DataFrame(records, columns=LOG_COLUMNS)
    result.to_csv(log_path, index=False)
    summary = _write_summary(result, summary_path)
    _print_summary(result, summary)
    print(f"\nSaved detailed log to: {log_path}")
    print(f"Saved availability summary to: {summary_path}")
    return result


def _print_summary(result: pd.DataFrame, summary: pd.DataFrame) -> None:
    downloaded = result["status"].eq("downloaded")
    invalid = result["status"].eq("invalid_image")
    print("\nCandidate availability:")
    print("landmark_id | tested | downloaded | failed | invalid | success_rate")
    print("-" * 72)
    for row in summary.itertuples(index=False):
        tested = int(row.urls_tested)
        downloaded_count = int(row.downloaded)
        invalid_count = int(row.invalid_images)
        failed = tested - downloaded_count - invalid_count
        print(
            f"{row.landmark_id} | {tested} | {downloaded_count} | "
            f"{failed} | {invalid_count} | {row.success_rate:.1%}"
        )
    print(f"\nTotal URLs tested: {len(result)}")
    print(f"Total successful downloads: {int(downloaded.sum())}")
    print(f"Total failures: {int((~downloaded & ~invalid).sum())}")
    print(f"Total invalid images: {int(invalid.sum())}")
    print(
        "Candidates with at least 50% success: "
        f"{int((summary['success_rate'] >= 0.5).sum())}"
    )
    print(
        "Candidates with at least 70% success: "
        f"{int((summary['success_rate'] >= 0.7).sum())}"
    )
    print("\nBest 10 candidates:")
    print(summary.head(10).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    try:
        test_availability(args.input, args.image_dir, args.log, args.summary)
    except (FileNotFoundError, OSError, pd.errors.ParserError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
