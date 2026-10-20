"""Serialize OCR/NER pipeline results for the Risk Fusion input contract."""

from __future__ import annotations

from typing import TypedDict

from .schema import PipelineResult, PipelineStatus


_SAFE_ERROR_CODES = frozenset(
    {
        "invalid_path",
        "unsupported_format",
        "image_not_found",
        "image_unreadable",
        "ocr_failed",
        "ocr_partial",
        "email_detector_failed",
        "phone_detector_failed",
        "address_detector_failed",
        "ner_model_unavailable",
        "ner_failed",
    }
)

_ERROR_MESSAGES = {
    PipelineStatus.PARTIAL: "OCR/PII analysis completed with one or more errors.",
    PipelineStatus.ERROR: "OCR/PII analysis failed; privacy could not be assessed.",
}


class RiskFusionError(TypedDict):
    """Safe error structure consumed by Risk Fusion."""

    code: str
    message: str


class RiskFusionFinding(TypedDict, total=False):
    """Finding dictionary accepted and preserved by Risk Fusion."""

    type: str
    confidence: float
    redacted_text: str | None
    bbox: list[float]
    source_segment_index: int
    span: dict[str, int]


class RiskFusionOCRPayload(TypedDict):
    """Nested OCR/NER payload to place under the combined payload's key."""

    status: str
    findings: list[RiskFusionFinding]
    error: RiskFusionError | None


def to_risk_fusion_ocr_payload(result: PipelineResult) -> RiskFusionOCRPayload:
    """Convert a typed pipeline result to Risk Fusion's nested ``ocr_ner`` value.

    Raw finding values and component error messages are intentionally omitted.
    OCR/NER statuses use the uppercase enum names expected by Risk Fusion.
    Bounding boxes are derived from source-image pixel polygons when present.
    """

    if not isinstance(result, PipelineResult):
        raise TypeError("result must be a PipelineResult")
    if result.status is PipelineStatus.OK and result.errors:
        raise ValueError("an OK pipeline result cannot contain errors")

    findings: list[RiskFusionFinding] = []
    for finding in result.findings:
        serialized: RiskFusionFinding = {
            "type": finding.category.value,
            # Deterministic detectors have no calibrated score; this fixed
            # value is the downstream convention agreed with the owner.
            "confidence": 1.0,
            "redacted_text": finding.redacted_value,
            "span": {"start": finding.span.start, "end": finding.span.end},
        }
        if finding.source_segment_index is not None:
            serialized["source_segment_index"] = finding.source_segment_index
        if finding.image_region is not None:
            xs = [point[0] for point in finding.image_region.polygon]
            ys = [point[1] for point in finding.image_region.polygon]
            serialized["bbox"] = [min(xs), min(ys), max(xs), max(ys)]
        findings.append(serialized)

    error = _serialize_error(result)
    return {
        "status": result.status.name,
        "findings": findings,
        "error": error,
    }


def _serialize_error(result: PipelineResult) -> RiskFusionError | None:
    if result.status is PipelineStatus.OK:
        return None
    if result.status is PipelineStatus.PARTIAL and not result.errors:
        return None

    code = "ocr_ner_error"
    if result.errors and result.errors[0].code in _SAFE_ERROR_CODES:
        code = result.errors[0].code
    return {
        "code": code,
        "message": _ERROR_MESSAGES[result.status],
    }
