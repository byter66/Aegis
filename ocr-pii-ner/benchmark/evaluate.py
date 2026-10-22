"""Run the labelled synthetic OCR benchmark and persist sanitized aggregates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from typing import Any

from PIL import Image

BENCHMARK_ROOT = Path(__file__).resolve().parent
MODULE_ROOT = BENCHMARK_ROOT.parent
sys.path.insert(0, str(MODULE_ROOT / "src"))

from aegis_ocr import EasyOCREngine, SpacyNERAdapter, ValuePolicy, analyze_image
from common import (
    CATEGORIES,
    EMAIL_MISMATCH_CATEGORIES,
    IOU_THRESHOLD,
    ImageAnnotation,
    Prediction,
    _maximum_matching,
    classify_email_mismatch,
    load_annotations,
    normalize_value,
    score_results,
    serializer_contract_check,
)


class _CountingOCR:
    def __init__(self, engine: EasyOCREngine) -> None:
        self.engine = engine
        self.last_result: Any = None

    def extract(self, image_path: str | Path) -> Any:
        self.last_result = None
        self.last_result = self.engine.extract(image_path)
        return self.last_result


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _versions(nlp: Any | None = None) -> dict[str, object]:
    versions: dict[str, object] = {
        "python": sys.version.split()[0],
        "easyocr": _installed_version("easyocr"),
        "spacy": _installed_version("spacy"),
        "pillow": _installed_version("Pillow"),
        "numpy": _installed_version("numpy"),
        "torch": _installed_version("torch"),
        "torchvision": _installed_version("torchvision"),
        "opencv_python": _installed_version("opencv-python")
        or _installed_version("opencv-python-headless"),
    }
    if nlp is not None:
        versions["spacy_model"] = {
            "name": nlp.meta.get("name"),
            "version": nlp.meta.get("version"),
        }
    return versions


def _prediction_from_finding(finding: Any) -> Prediction:
    if finding.image_region is None:
        box = None
    else:
        points = finding.image_region.polygon
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        box = (min(xs), min(ys), max(xs), max(ys))
    return Prediction(
        category=finding.category.value,
        value=finding.value or "",
        bbox_xyxy=box,
    )


def _error_metadata(result: Any) -> list[dict[str, str]]:
    return [
        {"code": error.code, "stage": error.stage}
        for error in result.errors
    ]


def _email_mismatch_category_counts_for_image(
    annotation: ImageAnnotation,
    run: dict[str, object],
) -> dict[str, int]:
    """Count one image's localized email mismatches without retaining values."""
    counts = {category: 0 for category in EMAIL_MISMATCH_CATEGORIES}
    if run.get("status") not in ("ok", "partial"):
        return counts

    expected = tuple(
        entity for entity in annotation.entities if entity.category == "email"
    )
    all_predictions = tuple(
        item if isinstance(item, Prediction) else Prediction(**item)
        for item in run.get("predictions", ())
    )
    predictions = tuple(
        prediction for prediction in all_predictions if prediction.category == "email"
    )
    localized_pairs = _maximum_matching(
        expected,
        predictions,
        threshold=IOU_THRESHOLD,
        require_value=False,
    )
    strict_pairs = _maximum_matching(
        expected,
        predictions,
        threshold=IOU_THRESHOLD,
        require_value=True,
    )
    strict_prediction_indices = {prediction_index for prediction_index, _ in strict_pairs}

    for prediction_index, expected_index in localized_pairs:
        prediction = predictions[prediction_index]
        truth = expected[expected_index]
        if prediction_index in strict_prediction_indices:
            continue
        if normalize_value("email", prediction.value) == truth.normalized_value:
            continue
        category = classify_email_mismatch(truth.expected_value, prediction.value)
        counts[category] += 1
    return counts


def _email_mismatch_category_counts_by_image(
    annotations: tuple[ImageAnnotation, ...],
    runs: dict[str, dict[str, object]],
) -> dict[str, dict[str, int]]:
    return {
        annotation.image_id: _email_mismatch_category_counts_for_image(
            annotation, runs.get(annotation.image_id, {})
        )
        for annotation in annotations
    }


def _aggregate_email_mismatch_category_counts(
    by_image: dict[str, dict[str, int]],
) -> dict[str, int]:
    return {
        category: sum(counts[category] for counts in by_image.values())
        for category in EMAIL_MISMATCH_CATEGORIES
    }


def _email_mismatch_category_counts(
    annotations: tuple[ImageAnnotation, ...],
    runs: dict[str, dict[str, object]],
) -> dict[str, int]:
    """Aggregate per-image email mismatch counts without retaining values."""
    by_image = _email_mismatch_category_counts_by_image(annotations, runs)
    return _aggregate_email_mismatch_category_counts(by_image)


def _add_email_mismatch_category_summaries(
    metrics: dict[str, object],
    annotations: tuple[ImageAnnotation, ...],
    runs: dict[str, dict[str, object]],
) -> None:
    by_image = _email_mismatch_category_counts_by_image(annotations, runs)
    metrics["email_mismatch_categories"] = _aggregate_email_mismatch_category_counts(
        by_image
    )
    for image_summary in metrics["per_image"]:
        image_summary["email_mismatch_categories"] = by_image[
            image_summary["image_id"]
        ]


def _prepare_runtime(
    tile_size: int,
    tile_overlap: int,
) -> tuple[_CountingOCR, SpacyNERAdapter, dict[str, float], dict[str, object]]:
    import easyocr
    import spacy

    init_times: dict[str, float] = {}
    started = time.perf_counter()
    reader = easyocr.Reader(["en"], gpu=False, download_enabled=False)
    init_times["easyocr_reader_ms"] = round((time.perf_counter() - started) * 1000, 3)
    engine = EasyOCREngine(
        reader=reader,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
    )

    started = time.perf_counter()
    nlp = spacy.load("en_core_web_sm")
    init_times["spacy_model_ms"] = round((time.perf_counter() - started) * 1000, 3)
    ner = SpacyNERAdapter(nlp=nlp)
    return _CountingOCR(engine), ner, init_times, _versions(nlp)


def _load_split(split: str) -> tuple[ImageAnnotation, ...]:
    return load_annotations(
        BENCHMARK_ROOT / "data" / "annotations" / f"{split}.jsonl",
        expected_split=split,
    )


def _validate_image_files(annotations: tuple[ImageAnnotation, ...]) -> None:
    data_root = BENCHMARK_ROOT / "data"
    for annotation in annotations:
        path = data_root / annotation.image_file
        if not path.is_file() or not path.resolve().is_relative_to(data_root.resolve()):
            raise ValueError("benchmark image path is missing or escapes the data directory")
        with Image.open(path) as image:
            if image.size != (annotation.width, annotation.height):
                raise ValueError("benchmark image size does not match annotation")
            image.verify()


def evaluate_split(
    annotations: tuple[ImageAnnotation, ...],
    ocr: _CountingOCR,
    ner: SpacyNERAdapter,
    *,
    phone_mode: str = "strict",
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    runs: dict[str, dict[str, object]] = {}
    for annotation in annotations:
        image_path = BENCHMARK_ROOT / "data" / annotation.image_file
        started = time.perf_counter()
        try:
            result = analyze_image(
                image_path,
                ocr=ocr,
                ner=ner,
                value_policy=ValuePolicy.RAW,
            )
            ocr_result = ocr.last_result
            segments = getattr(ocr_result, "segments", ())
            predictions = tuple(
                _prediction_from_finding(finding)
                for finding in result.findings
            )
            status = result.status.value
            errors = _error_metadata(result)
        except Exception:
            # A benchmark adapter failure is recorded without exception text,
            # which could contain image paths or detected values.
            predictions = ()
            segments = ()
            status = "error"
            errors = [{"code": "benchmark_image_failed", "stage": "evaluation"}]
        elapsed_ms = (time.perf_counter() - started) * 1000
        runs[annotation.image_id] = {
            "status": status,
            "elapsed_ms": elapsed_ms,
            "segment_count": len(segments),
            "predictions": predictions,
            "errors": errors,
        }
    metrics = score_results(annotations, runs, phone_mode=phone_mode)
    _add_email_mismatch_category_summaries(metrics, annotations, runs)
    return runs, metrics


def _sanitize_failed_initialization(
    annotations: tuple[ImageAnnotation, ...],
    exception: BaseException,
    *,
    phone_mode: str = "strict",
) -> dict[str, object]:
    error_type = type(exception).__name__
    runs = {
        annotation.image_id: {
            "status": "error",
            "elapsed_ms": 0.0,
            "segment_count": 0,
            "predictions": (),
            "errors": [{"code": "benchmark_initialization_failed", "stage": "initialization"}],
        }
        for annotation in annotations
    }
    metrics = score_results(annotations, runs, phone_mode=phone_mode)
    _add_email_mismatch_category_summaries(metrics, annotations, runs)
    return {
        "run_status": "initialization_error",
        "initialization_error_type": error_type,
        "initialization_error_code": "benchmark_initialization_failed",
        "metrics": metrics,
    }


def run(
    *,
    split: str,
    code_revision: str,
    tile_size: int,
    tile_overlap: int,
    phone_mode: str = "strict",
) -> dict[str, object]:
    annotations = _load_split(split)
    _validate_image_files(annotations)
    if not serializer_contract_check():
        raise RuntimeError("Risk Fusion serializer contract check failed")

    base: dict[str, object] = {
        "benchmark_version": "aegis-ocr-pii-benchmark-v1",
        "split": split,
        "code_revision": code_revision,
        "phone_mode": phone_mode,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_count": len(annotations),
        "tile_configuration": {
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "canvas_size": None,
        },
        "serializer_contract_check": "passed",
    }
    initialization_started = time.perf_counter()
    try:
        ocr, ner, cold_initialization_ms, versions = _prepare_runtime(
            tile_size, tile_overlap
        )
    except Exception as exception:
        base.update(_sanitize_failed_initialization(annotations, exception, phone_mode=phone_mode))
        base["cold_initialization_ms"] = {
            "initialization_failed_after_ms": round(
                (time.perf_counter() - initialization_started) * 1000, 3
            )
        }
        base["versions"] = _versions()
        return base

    runs, metrics = evaluate_split(annotations, ocr, ner, phone_mode=phone_mode)
    base.update(
        {
            "run_status": "completed",
            "cold_initialization_ms": cold_initialization_ms,
            "versions": versions,
            "metrics": metrics,
        }
    )
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "heldout"), default="development")
    parser.add_argument("--code-revision", required=True, help="recorded source revision; supplied explicitly without running Git")
    parser.add_argument("--tile-size", type=int, default=1600)
    parser.add_argument("--tile-overlap", type=int, default=128)
    parser.add_argument(
        "--phone-mode",
        choices=("strict", "india_national"),
        default="strict",
        help="phone number normalization mode: strict (default) or india_national",
    )
    parser.add_argument("--output", type=Path, help="optional output JSON path inside benchmark/results")
    arguments = parser.parse_args()
    if arguments.tile_size <= 0 or arguments.tile_overlap < 0 or arguments.tile_overlap >= arguments.tile_size:
        parser.error("tile size must be positive and overlap must be in [0, tile_size)")
    report = run(
        split=arguments.split,
        code_revision=arguments.code_revision,
        tile_size=arguments.tile_size,
        tile_overlap=arguments.tile_overlap,
        phone_mode=arguments.phone_mode,
    )
    results_directory = BENCHMARK_ROOT / "results"
    results_directory.mkdir(exist_ok=True)
    output = arguments.output or results_directory / (
        f"{arguments.split}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    resolved_output = output.resolve()
    if not resolved_output.is_relative_to(results_directory.resolve()):
        parser.error("output must be inside benchmark/results")
    resolved_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report["run_status"] != "completed":
        print(f"benchmark initialization failed ({report['initialization_error_type']}); sanitized report written")
        return 2
    metrics = report["metrics"]
    strict = metrics["strict"]
    print(
        "benchmark completed; "
        f"images={metrics['image_count']} "
        f"strict_f1={strict['micro_overall']['f1']:.4f} "
        f"partial={metrics['status_counts']['partial']} "
        f"errors={metrics['status_counts']['error']}"
    )
    display_path = resolved_output.relative_to(BENCHMARK_ROOT.resolve())
    print(f"sanitized aggregate report: {display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
