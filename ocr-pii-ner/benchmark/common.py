"""Annotation validation and scoring helpers for the synthetic OCR benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import string
import unicodedata
from pathlib import Path
from typing import Sequence


ANNOTATION_VERSION = "aegis-ocr-pii-benchmark-v1"
CATEGORIES = ("person_name", "email", "phone")
IOU_THRESHOLD = 0.5
EMAIL_MISMATCH_CATEGORIES = (
    "exact_match",
    "case_or_outer_whitespace_only",
    "missing_at_symbol",
    "missing_dot",
    "internal_whitespace_difference",
    "punctuation_difference",
    "character_difference",
    "truncation_like",
    "multiple_character_edits",
    "other_difference",
)

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class EntityAnnotation:
    category: str
    expected_value: str
    normalized_value: str
    bbox_xyxy: Box
    fragment_boxes_xyxy: tuple[Box, ...] = ()


@dataclass(frozen=True)
class ImageAnnotation:
    image_id: str
    split: str
    image_file: str
    width: int
    height: int
    template_id: str
    entities: tuple[EntityAnnotation, ...]


@dataclass(frozen=True)
class Prediction:
    category: str
    value: str
    bbox_xyxy: Box | None


def normalize_value(category: str, value: str) -> str:
    """Normalize only formatting that should not change synthetic identity."""

    normalized = unicodedata.normalize("NFKC", value).strip()
    if category == "person_name":
        return " ".join(normalized.casefold().split())
    if category == "email":
        return normalized.casefold()
    if category == "phone":
        digits = "".join(character for character in normalized if "0" <= character <= "9")
        return ("+" if normalized.startswith("+") else "") + digits
    raise ValueError("unknown finding category")


def classify_email_mismatch(expected: str, predicted: str) -> str:
    """Return a fixed, privacy-safe category describing an email difference.

    Categories describe string differences, not their cause. This is a
    conservative diagnostic, not an email validator. It never returns either
    input or any part of either input.
    """

    if not isinstance(expected, str) or not isinstance(predicted, str):
        return "other_difference"
    if expected == predicted:
        return "exact_match"

    expected_normalized = normalize_value("email", expected)
    predicted_normalized = normalize_value("email", predicted)
    if expected_normalized == predicted_normalized:
        if expected.strip().casefold() == predicted.strip().casefold():
            return "case_or_outer_whitespace_only"
        return "other_difference"

    if (
        expected_normalized.count("@") == 1
        and predicted_normalized.count("@") == 0
        and expected_normalized.replace("@", "", 1) == predicted_normalized
    ):
        return "missing_at_symbol"

    expected_parts = expected_normalized.split("@")
    predicted_parts = predicted_normalized.split("@")
    if (
        len(expected_parts) == len(predicted_parts) == 2
        and expected_parts[0] == predicted_parts[0]
        and expected_parts[1].count(".") == predicted_parts[1].count(".") + 1
        and any(
            expected_parts[1][:index] + expected_parts[1][index + 1 :] == predicted_parts[1]
            for index, char in enumerate(expected_parts[1])
            if char == "."
        )
    ):
        return "missing_dot"

    if (
        any(char.isspace() for char in expected_normalized + predicted_normalized)
        and "".join(expected_normalized.split()) == "".join(predicted_normalized.split())
    ):
        return "internal_whitespace_difference"

    def is_punctuation(char: str) -> bool:
        return char in string.punctuation or unicodedata.category(char).startswith("P")

    if len(expected_normalized) == len(predicted_normalized):
        differing = [
            (left, right)
            for left, right in zip(expected_normalized, predicted_normalized)
            if left != right
        ]
        if len(differing) == 1 and all(is_punctuation(char) for char in differing[0]):
            return "punctuation_difference"
    elif abs(len(expected_normalized) - len(predicted_normalized)) == 1:
        longer, shorter = sorted(
            (expected_normalized, predicted_normalized), key=len, reverse=True
        )
        if any(
            is_punctuation(char)
            and longer[:index] + longer[index + 1 :] == shorter
            for index, char in enumerate(longer)
        ):
            return "punctuation_difference"

    differing = [
        (left, right)
        for left, right in zip(expected_normalized, predicted_normalized)
        if left != right
    ]
    if (
        len(expected_normalized) == len(predicted_normalized)
        and len(differing) == 1
        and all(char.isalnum() for char in differing[0])
    ):
        return "character_difference"

    if (
        len(predicted_normalized) < len(expected_normalized)
        and expected_normalized.startswith(predicted_normalized)
    ):
        return "truncation_like"

    def alphanumeric_runs(value: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        runs: list[str] = []
        separators: list[str] = []
        current: list[str] = []
        for character in value:
            if character.isalnum():
                current.append(character)
            else:
                runs.append("".join(current))
                separators.append(character)
                current = []
        runs.append("".join(current))
        return tuple(runs), tuple(separators)

    def edit_distance(first: str, second: str) -> int:
        previous = list(range(len(second) + 1))
        for first_index, first_character in enumerate(first, start=1):
            current = [first_index]
            for second_index, second_character in enumerate(second, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[second_index] + 1,
                        previous[second_index - 1]
                        + (first_character != second_character),
                    )
                )
            previous = current
        return previous[-1]

    expected_runs, expected_separators = alphanumeric_runs(expected_normalized)
    predicted_runs, predicted_separators = alphanumeric_runs(predicted_normalized)
    differing_runs = [
        (left, right)
        for left, right in zip(expected_runs, predicted_runs)
        if left != right
    ]
    if (
        expected_separators == predicted_separators
        and len(expected_runs) == len(predicted_runs)
        and len(differing_runs) == 1
        and edit_distance(*differing_runs[0]) >= 2
    ):
        return "multiple_character_edits"

    return "other_difference"


def box_iou(first: Box, second: Box) -> float:
    """Return axis-aligned intersection over union; degenerate boxes score zero."""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _valid_box(value: object, width: int, height: int) -> Box:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("box must contain four coordinates")
    coordinates: list[float] = []
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise ValueError("box coordinates must be numeric")
        number = float(coordinate)
        if not math.isfinite(number):
            raise ValueError("box coordinates must be finite")
        coordinates.append(number)
    x1, y1, x2, y2 = coordinates
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("box is degenerate or outside the image")
    return x1, y1, x2, y2


def load_annotations(
    path: Path,
    *,
    expected_split: str | None = None,
) -> tuple[ImageAnnotation, ...]:
    """Load versioned JSONL annotations and validate values, paths, and boxes."""

    records: list[ImageAnnotation] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict) or raw.get("annotation_version") != ANNOTATION_VERSION:
                raise ValueError
            image_id = raw["image_id"]
            split = raw["split"]
            image_file = raw["image_file"]
            dimensions = raw["image_size"]
            width, height = dimensions["width"], dimensions["height"]
            template_id = raw["template_id"]
            entity_rows = raw["entities"]
            if (
                not isinstance(image_id, str)
                or not image_id
                or image_id in seen_ids
                or split not in ("development", "heldout")
                or (expected_split is not None and split != expected_split)
                or not isinstance(image_file, str)
                or Path(image_file).is_absolute()
                or ".." in Path(image_file).parts
                or not isinstance(width, int)
                or isinstance(width, bool)
                or width <= 0
                or not isinstance(height, int)
                or isinstance(height, bool)
                or height <= 0
                or not isinstance(template_id, str)
                or not template_id
                or not isinstance(entity_rows, list)
            ):
                raise ValueError
            entities: list[EntityAnnotation] = []
            for entity in entity_rows:
                if not isinstance(entity, dict):
                    raise ValueError
                category = entity["category"]
                value = entity["expected_value"]
                normalized_value = entity["expected_normalized_value"]
                if (
                    category not in CATEGORIES
                    or not isinstance(value, str)
                    or not value
                    or not isinstance(normalized_value, str)
                    or normalize_value(category, value) != normalized_value
                ):
                    raise ValueError
                box = _valid_box(entity["bbox_xyxy"], width, height)
                fragments_raw = entity.get("fragment_boxes_xyxy", [])
                if not isinstance(fragments_raw, list):
                    raise ValueError
                fragments = tuple(_valid_box(item, width, height) for item in fragments_raw)
                if any(
                    box[0] > fragment[0]
                    or box[1] > fragment[1]
                    or box[2] < fragment[2]
                    or box[3] < fragment[3]
                    for fragment in fragments
                ):
                    raise ValueError
                entities.append(
                    EntityAnnotation(category, value, normalized_value, box, fragments)
                )
            records.append(
                ImageAnnotation(
                    image_id,
                    split,
                    image_file,
                    width,
                    height,
                    template_id,
                    tuple(entities),
                )
            )
            seen_ids.add(image_id)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise ValueError(f"invalid annotation record at line {line_number}") from None
    return tuple(records)


def _maximum_matching(
    ground_truth: Sequence[EntityAnnotation],
    predictions: Sequence[Prediction],
    *,
    threshold: float,
    require_value: bool,
) -> tuple[tuple[int, int], ...]:
    """Deterministic maximum-cardinality bipartite matching (prediction, GT)."""

    edges: list[list[int]] = []
    for prediction in predictions:
        candidates: list[int] = []
        for index, expected in enumerate(ground_truth):
            if prediction.category != expected.category or prediction.bbox_xyxy is None:
                continue
            if box_iou(prediction.bbox_xyxy, expected.bbox_xyxy) < threshold:
                continue
            if require_value and normalize_value(
                prediction.category, prediction.value
            ) != expected.normalized_value:
                continue
            candidates.append(index)
        edges.append(candidates)

    matched_gt: dict[int, int] = {}

    def assign(prediction_index: int, visited: set[int]) -> bool:
        for gt_index in edges[prediction_index]:
            if gt_index in visited:
                continue
            visited.add(gt_index)
            previous = matched_gt.get(gt_index)
            if previous is None or assign(previous, visited):
                matched_gt[gt_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(predictions)):
        assign(prediction_index, set())
    return tuple(sorted((prediction_index, gt_index) for gt_index, prediction_index in matched_gt.items()))


def _metric_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def score_results(
    annotations: Sequence[ImageAnnotation],
    image_runs: dict[str, dict[str, object]],
    *,
    iou_threshold: float = IOU_THRESHOLD,
) -> dict[str, object]:
    """Score strict entities and localization while retaining sanitized run data."""

    categories: dict[str, dict[str, int]] = {
        category: {"tp": 0, "fp": 0, "fn": 0} for category in CATEGORIES
    }
    localization: dict[str, dict[str, int]] = {
        category: {"tp": 0, "fp": 0, "fn": 0} for category in CATEGORIES
    }
    strict_total = {"tp": 0, "fp": 0, "fn": 0}
    localization_total = {"tp": 0, "fp": 0, "fn": 0}
    localized_value_exact = 0
    localized_total = 0
    transcription_mismatches = 0
    image_summaries: list[dict[str, object]] = []
    negative_images = 0
    false_positive_negative_images = 0
    status_counts = {"ok": 0, "partial": 0, "error": 0}

    for annotation in annotations:
        run = image_runs.get(annotation.image_id, {})
        status = str(run.get("status", "error"))
        if status not in status_counts:
            status = "error"
        status_counts[status] += 1
        raw_predictions = run.get("predictions", ()) if status != "error" else ()
        predictions = tuple(
            item if isinstance(item, Prediction) else Prediction(**item)
            for item in raw_predictions
        )
        for prediction in predictions:
            if prediction.category not in CATEGORIES:
                continue
        image_tp = 0
        image_fp = 0
        for category in CATEGORIES:
            expected = tuple(entity for entity in annotation.entities if entity.category == category)
            predicted = tuple(prediction for prediction in predictions if prediction.category == category)
            strict_pairs_indices = _maximum_matching(
                expected, predicted, threshold=iou_threshold, require_value=True
            )
            localized_pairs_indices = _maximum_matching(
                expected, predicted, threshold=iou_threshold, require_value=False
            )
            tp = len(strict_pairs_indices)
            fp = len(predicted) - tp
            fn = len(expected) - tp
            categories[category]["tp"] += tp
            categories[category]["fp"] += fp
            categories[category]["fn"] += fn
            strict_total["tp"] += tp
            strict_total["fp"] += fp
            strict_total["fn"] += fn
            image_tp += tp
            image_fp += fp
            loc_tp = len(localized_pairs_indices)
            localization[category]["tp"] += loc_tp
            localization[category]["fp"] += len(predicted) - loc_tp
            localization[category]["fn"] += len(expected) - loc_tp
            localization_total["tp"] += loc_tp
            localization_total["fp"] += len(predicted) - loc_tp
            localization_total["fn"] += len(expected) - loc_tp
            for pred_index, gt_index in localized_pairs_indices:
                localized_total += 1
                if normalize_value(category, predicted[pred_index].value) == expected[gt_index].normalized_value:
                    localized_value_exact += 1
                else:
                    transcription_mismatches += 1

        if not annotation.entities:
            negative_images += 1
            if predictions:
                false_positive_negative_images += 1
        codes = run.get("errors", ())
        safe_codes = [
            {"code": str(item.get("code", "unknown")), "stage": str(item.get("stage", "unknown"))}
            for item in codes
            if isinstance(item, dict)
        ]
        category_counts = {category: 0 for category in CATEGORIES}
        for prediction in predictions:
            if prediction.category in category_counts:
                category_counts[prediction.category] += 1
        image_summaries.append(
            {
                "image_id": annotation.image_id,
                "split": annotation.split,
                "image_size": {"width": annotation.width, "height": annotation.height},
                "status": status,
                "elapsed_ms": round(float(run.get("elapsed_ms", 0.0)), 3),
                "segment_count": int(run.get("segment_count", 0)),
                "finding_count": len(predictions),
                "finding_counts_by_category": category_counts,
                "errors": safe_codes,
                "strict_tp": image_tp,
                "strict_fp": image_fp,
                "strict_fn": len(annotation.entities) - image_tp,
            }
        )

    strict_by_category = {
        category: _metric_counts(**counts) for category, counts in categories.items()
    }
    localization_by_category = {
        category: _metric_counts(**counts) for category, counts in localization.items()
    }
    return {
        "image_count": len(annotations),
        "strict": {
            "per_category": strict_by_category,
            "micro_overall": _metric_counts(**strict_total),
            "macro_f1": sum(item["f1"] for item in strict_by_category.values()) / len(CATEGORIES),
        },
        "localization_only": {
            "per_category": localization_by_category,
            "micro_overall": _metric_counts(**localization_total),
        },
        "value_exactness_on_localized_matches": (
            localized_value_exact / localized_total if localized_total else 0.0
        ),
        "localized_match_count": localized_total,
        "transcription_mismatch_count": transcription_mismatches,
        "negative_image_false_positive_rate": (
            false_positive_negative_images / negative_images if negative_images else 0.0
        ),
        "negative_image_count": negative_images,
        "negative_images_with_findings": false_positive_negative_images,
        "status_counts": status_counts,
        "partial_rate": status_counts["partial"] / len(annotations) if annotations else 0.0,
        "error_rate": status_counts["error"] / len(annotations) if annotations else 0.0,
        "per_image": image_summaries,
    }


def serializer_contract_check() -> bool:
    """Ensure serialized redacted data never falls back to an available raw value."""

    from aegis_ocr.risk_fusion import to_risk_fusion_ocr_payload
    from aegis_ocr.schema import (
        Finding,
        FindingCategory,
        PipelineResult,
        PipelineStatus,
        TextSpan,
    )

    finding = Finding(
        category=FindingCategory.PHONE,
        detector="benchmark_contract_check",
        span=TextSpan(0, 3),
        value="synthetic-raw-sentinel",
        redacted_value=None,
    )
    payload = to_risk_fusion_ocr_payload(
        PipelineResult("1.0", PipelineStatus.OK, findings=(finding,))
    )
    serialized = json.dumps(payload, sort_keys=True)
    return (
        payload["findings"][0]["redacted_text"] is None
        and "synthetic-raw-sentinel" not in serialized
    )
