from __future__ import annotations

import pytest

from aegis_ocr import to_risk_fusion_ocr_payload
from aegis_ocr.schema import (
    Finding,
    FindingCategory,
    ImageRegion,
    PipelineError,
    PipelineResult,
    PipelineStatus,
    TextSpan,
)


_VERSION = "1.0"


@pytest.mark.parametrize(
    ("category", "expected_type", "raw_value", "redacted_value"),
    [
        (FindingCategory.EMAIL, "email", "person@example.invalid", "[EMAIL]"),
        (FindingCategory.PHONE, "phone", "+12025550123", "[PHONE]"),
        (FindingCategory.PERSON_NAME, "person_name", "Ada Example", "[PERSON_NAME]"),
        (FindingCategory.ADDRESS, "address", "42 Main Street", "[ADDRESS]"),
    ],
)
def test_serializes_supported_categories_with_fixed_confidence_and_redaction(
    category: FindingCategory,
    expected_type: str,
    raw_value: str,
    redacted_value: str,
) -> None:
    result = PipelineResult(
        _VERSION,
        PipelineStatus.OK,
        findings=(
            Finding(
                category,
                "test_detector",
                TextSpan(2, 8),
                value=raw_value,
                redacted_value=redacted_value,
            ),
        ),
    )

    payload = to_risk_fusion_ocr_payload(result)

    assert payload == {
        "status": "OK",
        "findings": [
            {
                "type": expected_type,
                "confidence": 1.0,
                "redacted_text": redacted_value,
                "span": {"start": 2, "end": 8},
            }
        ],
        "error": None,
    }
    assert raw_value not in repr(payload)


def test_bbox_is_xyxy_extrema_and_provenance_is_preserved() -> None:
    finding = Finding(
        FindingCategory.EMAIL,
        "email_regex",
        TextSpan(4, 15),
        redacted_value="[EMAIL]",
        image_region=ImageRegion(((20, 30), (50, 28), (48, 44), (18, 42))),
        ocr_confidence=0.93,
        source_segment_index=2,
    )

    payload = to_risk_fusion_ocr_payload(
        PipelineResult(_VERSION, PipelineStatus.OK, findings=(finding,))
    )

    assert payload["findings"][0] == {
        "type": "email",
        "confidence": 1.0,
        "redacted_text": "[EMAIL]",
        "bbox": [18, 28, 50, 44],
        "source_segment_index": 2,
        "span": {"start": 4, "end": 15},
    }


def test_missing_image_region_omits_bbox() -> None:
    finding = Finding(
        FindingCategory.PHONE,
        "phone_regex",
        TextSpan(0, 8),
        redacted_value="[PHONE]",
    )

    payload = to_risk_fusion_ocr_payload(
        PipelineResult(_VERSION, PipelineStatus.OK, findings=(finding,))
    )

    assert "bbox" not in payload["findings"][0]


def test_missing_redacted_value_does_not_expose_raw_value() -> None:
    raw_value = "private.person@example.com"
    finding = Finding(
        FindingCategory.EMAIL,
        "email_regex",
        TextSpan(0, len(raw_value)),
        value=raw_value,
        redacted_value=None,
    )

    payload = to_risk_fusion_ocr_payload(
        PipelineResult(_VERSION, PipelineStatus.OK, findings=(finding,))
    )

    assert payload["findings"][0]["redacted_text"] is None
    assert raw_value not in repr(payload)


def test_empty_findings_use_risk_fusion_nested_ocr_shape() -> None:
    ocr_ner = to_risk_fusion_ocr_payload(
        PipelineResult(_VERSION, PipelineStatus.OK)
    )
    combined_payload = {
        "ocr_ner": ocr_ner,
        "landmark": {"status": "OK", "detected": False, "confidence": 0.0},
    }

    assert combined_payload == {
        "ocr_ner": {"status": "OK", "findings": [], "error": None},
        "landmark": {"status": "OK", "detected": False, "confidence": 0.0},
    }


@pytest.mark.parametrize(
    ("status", "expected_status"),
    [
        (PipelineStatus.OK, "OK"),
        (PipelineStatus.PARTIAL, "PARTIAL"),
        (PipelineStatus.ERROR, "ERROR"),
    ],
)
def test_status_is_serialized_using_uppercase_risk_fusion_value(
    status: PipelineStatus,
    expected_status: str,
) -> None:
    payload = to_risk_fusion_ocr_payload(PipelineResult(_VERSION, status))

    assert payload["status"] == expected_status


def test_partial_result_serializes_safe_structured_error_and_findings() -> None:
    finding = Finding(
        FindingCategory.PERSON_NAME,
        "spacy_ner",
        TextSpan(0, 3),
        redacted_value="[PERSON_NAME]",
    )
    result = PipelineResult(
        _VERSION,
        PipelineStatus.PARTIAL,
        findings=(finding,),
        errors=(PipelineError("ner_failed", "safe internal error", "ner"),),
    )

    payload = to_risk_fusion_ocr_payload(result)

    assert payload["status"] == "PARTIAL"
    assert payload["findings"][0]["type"] == "person_name"
    assert payload["error"] == {
        "code": "ner_failed",
        "message": "OCR/PII analysis completed with one or more errors.",
    }


def test_error_result_maps_to_fail_safe_risk_fusion_error_shape() -> None:
    result = PipelineResult(
        _VERSION,
        PipelineStatus.ERROR,
        errors=(
            PipelineError(
                "ocr_failed",
                "sensitive OCR text and exception details must not pass through",
                "ocr",
            ),
        ),
    )

    payload = to_risk_fusion_ocr_payload(result)

    assert payload == {
        "status": "ERROR",
        "findings": [],
        "error": {
            "code": "ocr_failed",
            "message": "OCR/PII analysis failed; privacy could not be assessed.",
        },
    }
    assert "sensitive OCR text" not in repr(payload)


def test_unknown_error_code_and_message_are_not_exposed() -> None:
    result = PipelineResult(
        _VERSION,
        PipelineStatus.ERROR,
        errors=(PipelineError("private-value", "private-value", "private-value"),),
    )

    payload = to_risk_fusion_ocr_payload(result)

    assert payload["error"] == {
        "code": "ocr_ner_error",
        "message": "OCR/PII analysis failed; privacy could not be assessed.",
    }
    assert "private-value" not in repr(payload)


def test_partial_status_without_errors_has_no_fabricated_error() -> None:
    payload = to_risk_fusion_ocr_payload(
        PipelineResult(_VERSION, PipelineStatus.PARTIAL)
    )

    assert payload["status"] == "PARTIAL"
    assert payload["error"] is None


def test_ok_status_with_errors_is_rejected_as_inconsistent() -> None:
    result = PipelineResult(
        _VERSION,
        PipelineStatus.OK,
        errors=(PipelineError("ocr_failed", "message", "ocr"),),
    )

    with pytest.raises(ValueError, match="OK pipeline result cannot contain errors"):
        to_risk_fusion_ocr_payload(result)


def test_accepts_only_pipeline_result() -> None:
    with pytest.raises(TypeError, match="result must be a PipelineResult"):
        to_risk_fusion_ocr_payload({"status": "OK"})  # type: ignore[arg-type]
