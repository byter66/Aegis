"""Segment-local OCR, structured PII, and person-name analysis pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from .ner import SpacyNERAdapter
from .ocr import EasyOCREngine, OCRResult, OCRSegment
from .pii import detect_emails, detect_phones
from .schema import (
    Finding,
    FindingCategory,
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
    ("ner_model_unavailable", "ner_model"): "The English NER model is unavailable.",
    ("ner_failed", "ner"): "The NER model could not process this text.",
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


def analyze_image(
    image_path: str | Path,
    *,
    ocr: _OCREngine | None = None,
    email_detector: EmailDetector = detect_emails,
    phone_detector: PhoneDetector = detect_phones,
    ner: _NEREngine | None = None,
    value_policy: ValuePolicy = ValuePolicy.REDACTED,
) -> PipelineResult:
    """Run OCR and independent detectors on each returned OCR segment.

    Finding spans remain local to their source segment. Each finding carries
    that segment's index, image polygon, and OCR confidence. OCR confidence is
    never copied to the detector score field.
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
    usable_calls = 0
    failed_calls = 0

    for segment_index, segment in enumerate(ocr_result.segments):
        for detector, expected_category, detector_name, failure_code in (
            (email_detector, FindingCategory.EMAIL, "email", "email_detector_failed"),
            (phone_detector, FindingCategory.PHONE, "phone", "phone_detector_failed"),
        ):
            try:
                output = detector(segment.text, value_policy=value_policy)
                local_findings = _validate_findings(
                    output,
                    expected_category=expected_category,
                    text_length=len(segment.text),
                )
                findings.extend(
                    _attach_segment(finding, segment_index, segment, value_policy)
                    for finding in local_findings
                )
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

    findings.sort(
        key=lambda item: (
            item.source_segment_index if item.source_segment_index is not None else -1,
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
