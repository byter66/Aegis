"""OCR, structured PII, and person-name analysis pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .ner import SpacyNERAdapter
from .ocr import EasyOCREngine, OCRResult, OCRSegment
from .pii import detect_addresses, detect_emails, detect_phones
from .schema import (
    Finding,
    FindingCategory,
    ImageRegion,
    PipelineError,
    PipelineResult,
    PipelineStatus,
    TextSpan,
    ValuePolicy,
)


_SCHEMA_VERSION = "1.0"

_SAFE_COMPONENT_ERRORS: dict[tuple[str, str], str] = {
    ("invalid_path", "input_validation"): "A non-empty image path is required.",
    ("unsupported_format", "input_validation"): "Only JPEG and PNG image files are supported.",
    ("image_not_found", "input_validation"): "The image file does not exist or is not a regular file.",
    ("image_unreadable", "image_validation"): "The image file is unreadable or corrupt.",
    ("unsupported_format", "image_validation"): "The image contents do not match the supported file extension.",
    ("ocr_failed", "ocr"): "The OCR engine could not process this image.",
    ("ocr_partial", "ocr"): "OCR returned a partial result.",
    ("ner_model_unavailable", "ner_model"): "The English NER model is unavailable.",
    ("ner_failed", "ner"): "The NER model could not process this text.",
    ("address_detector_failed", "address_detector"): (
        "The address detector could not process this OCR segment."
    ),
}


class _OCREngine(Protocol):
    def extract(self, image_path: str | Path) -> OCRResult: ...


class _NEREngine(Protocol):
    def extract(
        self,
        text: str,
        *,
        value_policy: ValuePolicy,
    ) -> PipelineResult: ...


EmailDetector = Callable[..., object]
PhoneDetector = Callable[..., object]
AddressDetector = Callable[..., object]

_MAX_PHONE_GROUP_SEGMENTS = 3
_PHONE_LINE_OVERLAP_RATIO = 0.5
_PHONE_LINE_CENTER_TOLERANCE = 0.5
_PHONE_MAX_HORIZONTAL_GAP_HEIGHTS = 1.5
_PHONE_DUPLICATE_BOX_IOU = 0.5

_PhoneTextPart = tuple[int, int, int]
_PhoneBounds = tuple[float, float, float, float]
_PhoneOccurrence = tuple[str, tuple[_PhoneTextPart, ...], _PhoneBounds]


def analyze_image(
    image_path: str | Path,
    *,
    ocr: _OCREngine | None = None,
    email_detector: EmailDetector = detect_emails,
    phone_detector: PhoneDetector = detect_phones,
    address_detector: AddressDetector = detect_addresses,
    ner: _NEREngine | None = None,
    value_policy: ValuePolicy = ValuePolicy.REDACTED,
) -> PipelineResult:
    """Run OCR, PII candidate detectors, and person-name NER per segment.

    Email and person-name detection stays segment-local. Phone detection also
    checks bounded groups of spatially adjacent same-line OCR segments. A
    grouped phone finding has no single source-segment index or OCR confidence;
    its span is relative to the temporary joined text and its image region
    covers the contributing source segments. OCR confidence is never copied to
    the detector score field.
    """

    if not isinstance(value_policy, ValuePolicy):
        raise TypeError("value_policy must be a ValuePolicy")

    ocr_engine = EasyOCREngine() if ocr is None else ocr
    ner_engine = SpacyNERAdapter() if ner is None else ner
    try:
        ocr_result = ocr_engine.extract(image_path)
    except Exception:
        return _result(
            PipelineStatus.ERROR,
            errors=(_error("ocr_failed", "ocr", "The OCR engine could not process this image."),),
        )

    if not isinstance(ocr_result, OCRResult):
        return _result(
            PipelineStatus.ERROR,
            errors=(_error("ocr_failed", "ocr", "The OCR engine returned an invalid result."),),
        )
    if any(not isinstance(segment, OCRSegment) for segment in ocr_result.segments):
        return _result(
            PipelineStatus.ERROR,
            errors=(_error("ocr_failed", "ocr", "The OCR engine returned an invalid result."),),
        )

    if ocr_result.status is PipelineStatus.ERROR:
        return _result(
            PipelineStatus.ERROR,
            errors=_normalize_errors(
                ocr_result.errors,
                fallback_code="ocr_failed",
                fallback_stage="ocr",
                fallback_message="The OCR engine could not process this image.",
            ),
        )
    if ocr_result.status not in (PipelineStatus.OK, PipelineStatus.PARTIAL):
        return _result(
            PipelineStatus.ERROR,
            errors=(_error("ocr_failed", "ocr", "The OCR engine returned an invalid result."),),
        )

    pipeline_errors: list[PipelineError] = []
    upstream_partial = ocr_result.status is PipelineStatus.PARTIAL
    if upstream_partial:
        pipeline_errors.extend(
            _normalize_errors(
                ocr_result.errors,
                fallback_code="ocr_partial",
                fallback_stage="ocr",
                fallback_message="OCR returned a partial result.",
            )
        )
    if not ocr_result.segments:
        return _result(
            PipelineStatus.PARTIAL if upstream_partial else PipelineStatus.OK,
            errors=tuple(pipeline_errors),
        )

    findings: list[Finding] = []
    phone_occurrences: list[_PhoneOccurrence] = []
    usable_calls = 0
    failed_calls = 0

    for segment_index, segment in enumerate(ocr_result.segments):
        for detector, expected_category, detector_name, failure_code in (
            (email_detector, FindingCategory.EMAIL, "email", "email_detector_failed"),
            (phone_detector, FindingCategory.PHONE, "phone", "phone_detector_failed"),
            (
                address_detector,
                FindingCategory.ADDRESS,
                "address",
                "address_detector_failed",
            ),
        ):
            try:
                output = detector(segment.text, value_policy=value_policy)
                local_findings = _validate_findings(
                    output,
                    expected_category=expected_category,
                    text_length=len(segment.text),
                )
                for finding in local_findings:
                    attached = _attach_segment(
                        finding, segment_index, segment, value_policy
                    )
                    if expected_category is FindingCategory.PHONE:
                        occurrence = (
                            _phone_digits(
                                segment.text[finding.span.start : finding.span.end]
                            ),
                            ((segment_index, finding.span.start, finding.span.end),),
                            _segment_bounds(segment),
                        )
                        if any(
                            _same_phone_occurrence(occurrence, existing)
                            for existing in phone_occurrences
                        ):
                            continue
                        phone_occurrences.append(occurrence)
                    findings.append(attached)
                usable_calls += 1
            except Exception:
                failed_calls += 1
                pipeline_errors.append(
                    _error(
                        failure_code,
                        f"{detector_name}_detector",
                        f"The {detector_name} detector could not process this OCR segment.",
                    )
                )

        try:
            ner_result = ner_engine.extract(segment.text, value_policy=value_policy)
            if not isinstance(ner_result, PipelineResult):
                raise TypeError("NER result has an invalid type")
            if ner_result.status is PipelineStatus.ERROR:
                if ner_result.findings:
                    raise ValueError("failed NER result cannot contain findings")
                failed_calls += 1
                pipeline_errors.extend(
                    _normalize_errors(
                        ner_result.errors,
                        fallback_code="ner_failed",
                        fallback_stage="ner",
                        fallback_message="The NER detector could not process this OCR segment.",
                    )
                )
                continue
            if ner_result.status not in (PipelineStatus.OK, PipelineStatus.PARTIAL):
                raise ValueError("NER result has an invalid status")
            if ner_result.status is PipelineStatus.OK and ner_result.errors:
                raise ValueError("successful NER result cannot contain errors")

            local_findings = _validate_findings(
                ner_result.findings,
                expected_category=FindingCategory.PERSON_NAME,
                text_length=len(segment.text),
            )
            findings.extend(
                _attach_segment(finding, segment_index, segment, value_policy)
                for finding in local_findings
            )
            if ner_result.status is PipelineStatus.PARTIAL:
                failed_calls += 1
                if local_findings:
                    # Partial NER output is usable when it contains validated findings.
                    usable_calls += 1
                pipeline_errors.extend(
                    _normalize_errors(
                        ner_result.errors,
                        fallback_code="ner_failed",
                        fallback_stage="ner",
                        fallback_message="The NER detector returned a partial result.",
                    )
                )
            else:
                usable_calls += 1
        except Exception:
            failed_calls += 1
            pipeline_errors.append(
                _error(
                    "ner_failed",
                    "ner",
                    "The NER detector could not process this OCR segment.",
                )
            )

    for group_indices in _phone_segment_groups(ocr_result.segments):
        group_text, text_parts = _join_phone_segments(
            group_indices, ocr_result.segments
        )
        if "+" not in group_text or sum(
            character in "0123456789" for character in group_text
        ) < 8:
            continue
        try:
            output = phone_detector(group_text, value_policy=value_policy)
            grouped_findings = _validate_findings(
                output,
                expected_category=FindingCategory.PHONE,
                text_length=len(group_text),
            )
            usable_calls += 1
        except Exception:
            failed_calls += 1
            pipeline_errors.append(
                _error(
                    "phone_detector_failed",
                    "phone_detector",
                    "The phone detector could not process OCR text.",
                )
            )
            continue

        for finding in grouped_findings:
            locations = _phone_locations_for_span(finding.span, text_parts)
            if not locations:
                continue
            digits = _phone_digits(
                group_text[finding.span.start : finding.span.end]
            )
            contributing_indices = tuple(dict.fromkeys(part[0] for part in locations))
            contributing_segments = tuple(
                ocr_result.segments[index] for index in contributing_indices
            )
            region = _combined_region(contributing_segments)
            occurrence = (
                digits,
                locations,
                _segment_bounds_from_region(region),
            )
            if any(
                _same_phone_occurrence(occurrence, existing)
                for existing in phone_occurrences
            ):
                continue
            findings.append(
                replace(
                    finding,
                    value=(
                        finding.value
                        if value_policy in (ValuePolicy.RAW, ValuePolicy.BOTH)
                        else None
                    ),
                    redacted_value=(
                        finding.redacted_value
                        if value_policy in (ValuePolicy.REDACTED, ValuePolicy.BOTH)
                        else None
                    ),
                    source_segment_index=None,
                    image_region=region,
                    ocr_confidence=None,
                )
            )
            phone_occurrences.append(occurrence)

    findings.sort(
        key=lambda item: (
            item.source_segment_index is None,
            item.source_segment_index if item.source_segment_index is not None else 0,
            item.span.start,
            item.span.end,
            item.category.value,
        )
    )

    if upstream_partial or failed_calls:
        status = PipelineStatus.PARTIAL if usable_calls else PipelineStatus.ERROR
    else:
        status = PipelineStatus.OK

    if status is PipelineStatus.ERROR:
        findings.clear()
    return _result(status, findings=tuple(findings), errors=tuple(_dedupe_errors(pipeline_errors)))


def _validate_findings(
    output: object,
    *,
    expected_category: FindingCategory,
    text_length: int,
) -> tuple[Finding, ...]:
    if not isinstance(output, (list, tuple)):
        raise TypeError("detector output must be a list or tuple of findings")
    validated: list[Finding] = []
    seen: set[tuple[FindingCategory, TextSpan]] = set()
    for finding in output:
        if not isinstance(finding, Finding):
            raise TypeError("detector output contains a non-finding value")
        if finding.category is not expected_category:
            raise ValueError("detector output contains an unexpected category")
        span = finding.span
        if not isinstance(span, TextSpan) or not 0 <= span.start < span.end <= text_length:
            raise ValueError("detector output contains an invalid text span")
        if finding.source_segment_index is not None and (
            isinstance(finding.source_segment_index, bool)
            or not isinstance(finding.source_segment_index, int)
            or finding.source_segment_index < 0
        ):
            raise ValueError("detector output contains an invalid source segment index")
        key = (finding.category, span)
        if key in seen:
            raise ValueError("detector output contains a duplicate finding")
        seen.add(key)
        validated.append(finding)
    return tuple(validated)


def _attach_segment(
    finding: Finding,
    index: int,
    segment: OCRSegment,
    value_policy: ValuePolicy,
) -> Finding:
    return replace(
        finding,
        value=(
            finding.value
            if value_policy in (ValuePolicy.RAW, ValuePolicy.BOTH)
            else None
        ),
        redacted_value=(
            finding.redacted_value
            if value_policy in (ValuePolicy.REDACTED, ValuePolicy.BOTH)
            else None
        ),
        source_segment_index=index,
        image_region=segment.image_region,
        ocr_confidence=segment.ocr_confidence,
    )


def _phone_segment_groups(
    segments: tuple[OCRSegment, ...],
) -> tuple[tuple[int, ...], ...]:
    """Return bounded consecutive windows of nearby segments on one visual line."""

    bounds = tuple(_segment_bounds(segment) for segment in segments)
    largest_height = max(
        (bottom - top for _, top, _, bottom in bounds), default=0.0
    )
    ordered_indices = sorted(range(len(segments)), key=lambda i: (bounds[i][0], i))
    groups: list[tuple[int, ...]] = []
    for start in range(len(ordered_indices)):
        group = [ordered_indices[start]]
        search_from = start + 1
        while len(group) < _MAX_PHONE_GROUP_SEGMENTS:
            current_index = group[-1]
            current = segments[current_index]
            current_bounds = bounds[current_index]
            current_height = current_bounds[3] - current_bounds[1]
            if current_height <= 0:
                break
            next_position = None
            for position in range(search_from, len(ordered_indices)):
                candidate_index = ordered_indices[position]
                candidate_bounds = bounds[candidate_index]
                horizontal_gap = candidate_bounds[0] - current_bounds[2]
                if horizontal_gap > _PHONE_MAX_HORIZONTAL_GAP_HEIGHTS * largest_height:
                    break
                if not _same_visual_line(current, segments[candidate_index]):
                    continue
                if horizontal_gap < 0:
                    # An overlapping box on this line makes text order ambiguous.
                    break
                candidate_height = candidate_bounds[3] - candidate_bounds[1]
                if horizontal_gap > _PHONE_MAX_HORIZONTAL_GAP_HEIGHTS * max(
                    current_height, candidate_height
                ):
                    break
                next_position = position
                break
            if next_position is None:
                break
            group.append(ordered_indices[next_position])
            groups.append(tuple(group))
            search_from = next_position + 1
    return tuple(groups)


def _segment_bounds(segment: OCRSegment) -> tuple[float, float, float, float]:
    polygon = segment.image_region.polygon
    return (
        min(point[0] for point in polygon),
        min(point[1] for point in polygon),
        max(point[0] for point in polygon),
        max(point[1] for point in polygon),
    )


def _same_visual_line(first: OCRSegment, second: OCRSegment) -> bool:
    _, first_top, _, first_bottom = _segment_bounds(first)
    _, second_top, _, second_bottom = _segment_bounds(second)
    first_height = first_bottom - first_top
    second_height = second_bottom - second_top
    if first_height <= 0 or second_height <= 0:
        return False

    min_height = min(first_height, second_height)
    max_height = max(first_height, second_height)
    vertical_overlap = max(
        0.0, min(first_bottom, second_bottom) - max(first_top, second_top)
    )
    center_difference = abs(
        (first_top + first_bottom) / 2 - (second_top + second_bottom) / 2
    )
    return (
        vertical_overlap / min_height >= _PHONE_LINE_OVERLAP_RATIO
        and center_difference <= _PHONE_LINE_CENTER_TOLERANCE * max_height
    )


def _join_phone_segments(
    indices: tuple[int, ...], segments: tuple[OCRSegment, ...]
) -> tuple[str, tuple[_PhoneTextPart, ...]]:
    pieces: list[str] = []
    text_parts: list[_PhoneTextPart] = []
    offset = 0
    for position, index in enumerate(indices):
        text = segments[index].text
        pieces.append(text)
        text_parts.append((offset, offset + len(text), index))
        offset += len(text)
        if position + 1 < len(indices):
            pieces.append(" ")
            offset += 1
    return "".join(pieces), tuple(text_parts)


def _phone_locations_for_span(
    span: TextSpan, text_parts: tuple[_PhoneTextPart, ...]
) -> tuple[_PhoneTextPart, ...]:
    locations: list[_PhoneTextPart] = []
    for group_start, group_end, segment_index in text_parts:
        start = max(span.start, group_start)
        end = min(span.end, group_end)
        if start < end:
            locations.append((segment_index, start - group_start, end - group_start))
    return tuple(locations)


def _phone_digits(text: str) -> str:
    return "".join(character for character in text if character.isdigit())


def _same_phone_occurrence(
    first: _PhoneOccurrence, second: _PhoneOccurrence
) -> bool:
    first_digits, first_locations, first_bounds = first
    second_digits, second_locations, second_bounds = second
    if first_digits != second_digits:
        return False

    shared_segment = any(
        first_index == second_index
        for first_index, _, _ in first_locations
        for second_index, _, _ in second_locations
    )
    if shared_segment:
        return any(
            first_index == second_index
            and max(first_start, second_start) < min(first_end, second_end)
            for first_index, first_start, first_end in first_locations
            for second_index, second_start, second_end in second_locations
        )

    return _bounds_iou(first_bounds, second_bounds) >= _PHONE_DUPLICATE_BOX_IOU


def _segment_bounds_from_region(region: ImageRegion) -> _PhoneBounds:
    xs = [point[0] for point in region.polygon]
    ys = [point[1] for point in region.polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _bounds_iou(first: _PhoneBounds, second: _PhoneBounds) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _combined_region(segments: tuple[OCRSegment, ...]) -> ImageRegion:
    bounds = [_segment_bounds(segment) for segment in segments]
    left = min(item[0] for item in bounds)
    top = min(item[1] for item in bounds)
    right = max(item[2] for item in bounds)
    bottom = max(item[3] for item in bounds)
    return ImageRegion(
        ((left, top), (right, top), (right, bottom), (left, bottom))
    )


def _normalize_errors(
    errors: object,
    *,
    fallback_code: str,
    fallback_stage: str,
    fallback_message: str,
) -> tuple[PipelineError, ...]:
    if not isinstance(errors, (list, tuple)) or not errors:
        return (_error(fallback_code, fallback_stage, fallback_message),)
    normalized: list[PipelineError] = []
    for error in errors:
        if not isinstance(error, PipelineError):
            normalized.append(_error(fallback_code, fallback_stage, fallback_message))
            continue
        safe_message = _SAFE_COMPONENT_ERRORS.get((error.code, error.stage))
        if safe_message is None:
            normalized.append(_error(fallback_code, fallback_stage, fallback_message))
        else:
            # Reconstruct from a fixed message so injected components cannot leak text.
            normalized.append(PipelineError(error.code, safe_message, error.stage))
    return tuple(_dedupe_errors(normalized))


def _dedupe_errors(errors: list[PipelineError]) -> list[PipelineError]:
    unique: list[PipelineError] = []
    seen: set[tuple[str, str, str]] = set()
    for error in errors:
        key = (error.code, error.stage, error.message)
        if key not in seen:
            seen.add(key)
            unique.append(error)
    return unique


def _error(code: str, stage: str, message: str) -> PipelineError:
    return PipelineError(code=code, stage=stage, message=message)


def _result(
    status: PipelineStatus,
    *,
    findings: tuple[Finding, ...] = (),
    errors: tuple[PipelineError, ...] = (),
) -> PipelineResult:
    return PipelineResult(
        schema_version=_SCHEMA_VERSION,
        status=status,
        findings=findings,
        errors=errors,
    )
