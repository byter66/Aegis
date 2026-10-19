from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aegis_ocr.ner import SpacyNERAdapter
from aegis_ocr.ocr import OCRResult, OCRSegment
from aegis_ocr.pipeline import analyze_image
from aegis_ocr.risk_fusion import to_risk_fusion_ocr_payload
from aegis_ocr.schema import (
    Finding,
    FindingCategory,
    ImageRegion,
    PipelineError,
    PipelineResult,
    PipelineStatus,
    TextSpan,
    ValuePolicy,
    apply_value_policy,
)


def _segment(text: str, confidence: float = 0.91, x: float = 0.0) -> OCRSegment:
    return OCRSegment(
        text=text,
        image_region=ImageRegion(((x, 0.0), (x + 100.0, 0.0), (x + 100.0, 20.0), (x, 20.0))),
        ocr_confidence=confidence,
    )


def _positioned_segment(
    text: str,
    *,
    left: float,
    top: float,
    right: float,
    bottom: float,
    confidence: float = 0.91,
) -> OCRSegment:
    return OCRSegment(
        text=text,
        image_region=ImageRegion(
            ((left, top), (right, top), (right, bottom), (left, bottom))
        ),
        ocr_confidence=confidence,
    )


class FakeOCR:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[str] = []

    def extract(self, image_path: str) -> object:
        self.calls.append(image_path)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeNER:
    def __init__(self, spans: dict[str, list[tuple[int, int]]] | None = None) -> None:
        self.spans = spans or {}
        self.calls: list[tuple[str, ValuePolicy]] = []
        self.error: Exception | None = None
        self.result: object | None = None

    def extract(self, text: str, *, value_policy: ValuePolicy) -> object:
        self.calls.append((text, value_policy))
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        findings = []
        for start, end in self.spans.get(text, []):
            value, redacted = apply_value_policy(
                text[start:end], "[PERSON_NAME]", value_policy
            )
            findings.append(
                Finding(
                    category=FindingCategory.PERSON_NAME,
                    detector="spacy_ner",
                    span=TextSpan(start, end),
                    value=value,
                    redacted_value=redacted,
                )
            )
        return PipelineResult("1.0", PipelineStatus.OK, findings=tuple(findings))


def _result(*segments: OCRSegment) -> OCRResult:
    return OCRResult(PipelineStatus.OK, segments=tuple(segments))


def test_combines_all_finding_types_with_segment_provenance() -> None:
    text = "Ada fake@example.invalid +1 202-555-0100"
    region = _segment(text, confidence=0.83, x=12).image_region
    segment = _segment(text, confidence=0.83, x=12)
    ner = FakeNER({text: [(0, 3)]})

    result = analyze_image("synthetic.png", ocr=FakeOCR(_result(segment)), ner=ner)

    assert result.status is PipelineStatus.OK
    assert [finding.category for finding in result.findings] == [
        FindingCategory.PERSON_NAME,
        FindingCategory.EMAIL,
        FindingCategory.PHONE,
    ]
    for finding in result.findings:
        assert finding.source_segment_index == 0
        assert finding.image_region == region
        assert finding.ocr_confidence == 0.83
        assert finding.detector_score is None
        assert finding.detector_score_semantics is None
    assert [(item.span.start, item.span.end) for item in result.findings] == [
        (0, 3),
        (4, 24),
        (25, len(text)),
    ]


def test_two_adjacent_same_line_segments_detect_split_phone_with_union_region() -> None:
    first = _positioned_segment("+1 555", left=10, top=20, right=70, bottom=40)
    second = _positioned_segment("234 5678", left=74, top=21, right=150, bottom=41)

    result = analyze_image(
        "synthetic.png", ocr=FakeOCR(_result(first, second)), ner=FakeNER()
    )

    (phone,) = [item for item in result.findings if item.category is FindingCategory.PHONE]
    assert phone.value is None
    assert phone.redacted_value == "[PHONE]"
    assert phone.source_segment_index is None
    assert phone.ocr_confidence is None
    assert phone.image_region == ImageRegion(
        ((10.0, 20.0), (150.0, 20.0), (150.0, 41.0), (10.0, 41.0))
    )
    assert phone.span == TextSpan(0, len("+1 555 234 5678"))


def test_three_adjacent_same_line_segments_detect_split_phone() -> None:
    segments = _result(
        _positioned_segment("+1", left=0, top=10, right=24, bottom=30),
        _positioned_segment("555", left=27, top=10, right=63, bottom=30),
        _positioned_segment("234 5678", left=66, top=10, right=145, bottom=30),
    )

    result = analyze_image("synthetic.png", ocr=FakeOCR(segments), ner=FakeNER())

    phones = [item for item in result.findings if item.category is FindingCategory.PHONE]
    assert len(phones) == 1
    assert phones[0].source_segment_index is None
    assert phones[0].image_region == ImageRegion(
        ((0.0, 10.0), (145.0, 10.0), (145.0, 30.0), (0.0, 30.0))
    )


def test_adjacent_unrelated_text_is_not_joined_into_phone_candidate() -> None:
    segments = _result(
        _positioned_segment("+1 555", left=0, top=10, right=60, bottom=30),
        _positioned_segment("extension", left=63, top=10, right=140, bottom=30),
        _positioned_segment("234 5678", left=143, top=10, right=220, bottom=30),
    )

    result = analyze_image("synthetic.png", ocr=FakeOCR(segments), ner=FakeNER())

    assert not any(item.category is FindingCategory.PHONE for item in result.findings)


def test_segments_on_different_lines_are_not_joined_for_phone_detection() -> None:
    segments = _result(
        _positioned_segment("+1 555", left=0, top=0, right=60, bottom=20),
        _positioned_segment("234 5678", left=63, top=70, right=140, bottom=90),
    )

    result = analyze_image("synthetic.png", ocr=FakeOCR(segments), ner=FakeNER())

    assert not any(item.category is FindingCategory.PHONE for item in result.findings)


def test_distant_segments_are_not_joined_for_phone_detection() -> None:
    segments = _result(
        _positioned_segment("+1 555", left=0, top=0, right=60, bottom=20),
        _positioned_segment("234 5678", left=500, top=0, right=580, bottom=20),
    )

    result = analyze_image("synthetic.png", ocr=FakeOCR(segments), ner=FakeNER())

    assert not any(item.category is FindingCategory.PHONE for item in result.findings)


def test_single_segment_phone_is_not_duplicated_by_grouped_detection() -> None:
    segments = _result(
        _positioned_segment("+1 202-555-0100", left=0, top=0, right=160, bottom=20),
        _positioned_segment("extension", left=165, top=0, right=240, bottom=20),
    )

    result = analyze_image("synthetic.png", ocr=FakeOCR(segments), ner=FakeNER())

    phones = [item for item in result.findings if item.category is FindingCategory.PHONE]
    assert len(phones) == 1
    assert phones[0].source_segment_index == 0
    assert phones[0].ocr_confidence == 0.91


def test_overlapping_ocr_representations_of_one_phone_emit_one_finding() -> None:
    segments = _result(
        _positioned_segment(
            "+1 202-555-0100", left=20, top=30, right=180, bottom=54
        ),
        _positioned_segment(
            "+1 (202) 555 0100", left=23, top=31, right=183, bottom=55
        ),
    )

    result = analyze_image("synthetic.png", ocr=FakeOCR(segments), ner=FakeNER())

    phones = [item for item in result.findings if item.category is FindingCategory.PHONE]
    assert len(phones) == 1
    assert phones[0].category is FindingCategory.PHONE
    assert phones[0].redacted_value == "[PHONE]"
    assert phones[0].image_region == segments.segments[0].image_region


def test_identical_phones_at_separate_locations_remain_separate_findings() -> None:
    segments = _result(
        _positioned_segment(
            "+1 202-555-0100", left=20, top=30, right=180, bottom=54
        ),
        _positioned_segment(
            "+1 202-555-0100", left=420, top=230, right=580, bottom=254
        ),
    )

    result = analyze_image("synthetic.png", ocr=FakeOCR(segments), ner=FakeNER())

    phones = [item for item in result.findings if item.category is FindingCategory.PHONE]
    assert len(phones) == 2
    assert [item.category for item in phones] == [
        FindingCategory.PHONE,
        FindingCategory.PHONE,
    ]
    assert [item.redacted_value for item in phones] == ["[PHONE]", "[PHONE]"]
    assert [item.image_region for item in phones] == [
        segments.segments[0].image_region,
        segments.segments[1].image_region,
    ]


@pytest.mark.parametrize(
    ("policy", "expected_has_value", "expected_redacted"),
    [
        (ValuePolicy.RAW, True, None),
        (ValuePolicy.REDACTED, False, "[PHONE]"),
        (ValuePolicy.BOTH, True, "[PHONE]"),
        (ValuePolicy.NEITHER, False, None),
    ],
)
def test_grouped_phone_obeys_value_policy_and_serializes_existing_contract(
    policy: ValuePolicy,
    expected_has_value: bool,
    expected_redacted: str | None,
) -> None:
    segments = _result(
        _positioned_segment("+1 555", left=10, top=20, right=70, bottom=40),
        _positioned_segment("234 5678", left=74, top=20, right=150, bottom=40),
    )

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(segments),
        ner=FakeNER(),
        value_policy=policy,
    )

    (phone,) = [item for item in result.findings if item.category is FindingCategory.PHONE]
    assert (phone.value is not None) == expected_has_value
    assert phone.redacted_value == expected_redacted
    payload = to_risk_fusion_ocr_payload(result)
    assert payload["status"] == "OK"
    serialized = next(item for item in payload["findings"] if item["type"] == "phone")
    serialized_redacted = serialized["redacted_text"]
    assert (serialized_redacted is not None) == (expected_redacted is not None)
    if expected_redacted is not None:
        assert serialized_redacted == "[PHONE]"
    assert serialized["bbox"] == [10.0, 20.0, 150.0, 40.0]
    assert "source_segment_index" not in serialized
    assert serialized["span"] == {"start": 0, "end": len("+1 555 234 5678")}


def test_grouped_phone_detector_failure_returns_only_safe_error() -> None:
    segments = _result(
        _positioned_segment("+1 555", left=0, top=0, right=60, bottom=20),
        _positioned_segment("234 5678", left=63, top=0, right=140, bottom=20),
    )

    def fail_grouped(text: str, *, value_policy: ValuePolicy) -> list[Finding]:
        if " " in text:
            raise RuntimeError("private synthetic value")
        from aegis_ocr.pii import detect_phones

        return detect_phones(text, value_policy=value_policy)

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(segments),
        phone_detector=fail_grouped,
        ner=FakeNER(),
    )

    assert result.status is PipelineStatus.PARTIAL
    assert result.findings == ()
    assert result.errors[0].code == "phone_detector_failed"
    assert "private synthetic value" not in result.errors[0].message


def test_repeated_same_local_entity_across_segments_is_preserved() -> None:
    first_text = second_text = "Ada"
    ner = FakeNER({"Ada": [(0, 3)]})

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(_result(_segment(first_text, x=0), _segment(second_text, x=200))),
        ner=ner,
    )

    assert result.status is PipelineStatus.OK
    assert [(item.source_segment_index, item.span) for item in result.findings] == [
        (0, TextSpan(0, 3)),
        (1, TextSpan(0, 3)),
    ]
    assert result.findings[0].image_region != result.findings[1].image_region


def test_findings_follow_segment_then_span_end_and_category_order() -> None:
    text = "+1 202-555-0100 fake@example.invalid Ada"
    ner = FakeNER({text: [(len(text) - 3, len(text))]})
    second = "fake2@example.invalid"

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(_result(_segment(text), _segment(second, x=200))),
        ner=ner,
    )

    assert [(item.source_segment_index, item.span.start, item.category) for item in result.findings] == [
        (0, 0, FindingCategory.PHONE),
        (0, 16, FindingCategory.EMAIL),
        (0, len(text) - 3, FindingCategory.PERSON_NAME),
        (1, 0, FindingCategory.EMAIL),
    ]


def test_empty_ocr_result_succeeds_without_detector_calls() -> None:
    email = Mock()
    phone = Mock()
    ner = FakeNER()

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(_result()),
        email_detector=email,
        phone_detector=phone,
        ner=ner,
    )

    assert result.status is PipelineStatus.OK
    assert result.findings == ()
    assert result.errors == ()
    email.assert_not_called()
    phone.assert_not_called()
    assert ner.calls == []


def test_ocr_failure_returns_error_and_safe_ocr_error() -> None:
    error = PipelineError("image_not_found", "path is /private/file.png", "input_validation")
    ocr_result = OCRResult(PipelineStatus.ERROR, errors=(error,))

    result = analyze_image("secret/path.png", ocr=FakeOCR(ocr_result))

    assert result.status is PipelineStatus.ERROR
    assert result.findings == ()
    assert result.errors == (
        PipelineError(
            "image_not_found",
            "The image file does not exist or is not a regular file.",
            "input_validation",
        ),
    )
    assert "/private" not in result.errors[0].message


def test_ocr_exception_returns_generic_safe_error() -> None:
    secret = "secret/path and text"

    result = analyze_image("synthetic.png", ocr=FakeOCR(RuntimeError(secret)))

    assert result.status is PipelineStatus.ERROR
    assert result.errors[0].code == "ocr_failed"
    assert secret not in result.errors[0].message


def test_partial_ocr_with_segments_preserves_results_and_safe_error() -> None:
    text = "Ada"
    ocr_error = PipelineError("ocr_failed", "private text leaked", "ocr")
    ocr_result = OCRResult(PipelineStatus.PARTIAL, (_segment(text),), (ocr_error,))

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(ocr_result),
        ner=FakeNER({text: [(0, 3)]}),
    )

    assert result.status is PipelineStatus.PARTIAL
    assert [item.category for item in result.findings] == [FindingCategory.PERSON_NAME]
    assert result.errors == (
        PipelineError("ocr_failed", "The OCR engine could not process this image.", "ocr"),
    )


def test_partial_ocr_tile_error_keeps_specific_safe_error_code() -> None:
    segment = _segment("synthetic")
    ocr_result = OCRResult(
        PipelineStatus.PARTIAL,
        (segment,),
        (PipelineError("ocr_partial", "private detail", "ocr"),),
    )

    result = analyze_image("synthetic.png", ocr=FakeOCR(ocr_result), ner=FakeNER())

    assert result.status is PipelineStatus.PARTIAL
    assert result.errors == (
        PipelineError("ocr_partial", "OCR returned a partial result.", "ocr"),
    )


def test_partial_ocr_with_segments_and_no_errors_adds_safe_error() -> None:
    text = "Ada"
    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(OCRResult(PipelineStatus.PARTIAL, (_segment(text),))),
        ner=FakeNER({text: [(0, 3)]}),
    )

    assert result.status is PipelineStatus.PARTIAL
    assert [item.category for item in result.findings] == [FindingCategory.PERSON_NAME]
    assert result.errors == (
        PipelineError("ocr_partial", "OCR returned a partial result.", "ocr"),
    )


def test_partial_ocr_without_segments_returns_partial_and_safe_error() -> None:
    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(OCRResult(PipelineStatus.PARTIAL)),
    )

    assert result.status is PipelineStatus.PARTIAL
    assert result.findings == ()
    assert result.errors == (
        PipelineError("ocr_partial", "OCR returned a partial result.", "ocr"),
    )


def test_failures_on_one_segment_and_success_on_another_returns_partial() -> None:
    first_text = "first segment"
    second_text = "fake@example.invalid"

    def email_detector(text: str, *, value_policy: ValuePolicy) -> list[Finding]:
        if text == first_text:
            raise RuntimeError(f"sensitive failure: {text}")
        from aegis_ocr.pii import detect_emails

        return detect_emails(text, value_policy=value_policy)

    def phone_detector(text: str, *, value_policy: ValuePolicy) -> list[Finding]:
        if text == first_text:
            raise RuntimeError(f"sensitive failure: {text}")
        return []

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(_result(_segment(first_text), _segment(second_text, x=200))),
        email_detector=email_detector,
        phone_detector=phone_detector,
        ner=FakeNER(),
    )

    assert result.status is PipelineStatus.PARTIAL
    assert len(result.findings) == 1
    assert result.findings[0].category is FindingCategory.EMAIL
    assert result.findings[0].source_segment_index == 1
    assert {error.code for error in result.errors} == {
        "email_detector_failed",
        "phone_detector_failed",
    }
    assert all(first_text not in error.message for error in result.errors)


def test_partial_ner_findings_remain_usable_when_email_and_phone_fail() -> None:
    text = "Ada"
    person = Finding(FindingCategory.PERSON_NAME, "spacy_ner", TextSpan(0, 3))
    ner = FakeNER()
    ner.result = PipelineResult(
        "1.0",
        PipelineStatus.PARTIAL,
        findings=(person,),
        errors=(PipelineError("ner_failed", "sensitive detail", "ner"),),
    )
    failure = Mock(side_effect=RuntimeError(f"private value: {text}"))

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(_result(_segment(text))),
        email_detector=failure,
        phone_detector=failure,
        ner=ner,
    )

    assert result.status is PipelineStatus.PARTIAL
    assert result.findings[0].category is FindingCategory.PERSON_NAME
    assert result.errors == (
        PipelineError("email_detector_failed", "The email detector could not process this OCR segment.", "email_detector"),
        PipelineError("phone_detector_failed", "The phone detector could not process this OCR segment.", "phone_detector"),
        PipelineError("ner_failed", "The NER model could not process this text.", "ner"),
    )
    assert all("sensitive detail" not in error.message for error in result.errors)


def test_partial_ner_without_errors_keeps_findings_and_adds_safe_error() -> None:
    text = "Ada"
    person = Finding(FindingCategory.PERSON_NAME, "spacy_ner", TextSpan(0, 3))
    ner = FakeNER()
    ner.result = PipelineResult("1.0", PipelineStatus.PARTIAL, findings=(person,))

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(_result(_segment(text))),
        ner=ner,
    )

    assert result.status is PipelineStatus.PARTIAL
    assert result.findings[0].category is FindingCategory.PERSON_NAME
    assert result.errors == (
        PipelineError("ner_failed", "The NER detector returned a partial result.", "ner"),
    )


def test_malformed_ner_result_is_safe_and_does_not_discard_other_results() -> None:
    text = "Ada fake@example.invalid"
    ner = FakeNER()
    ner.result = object()

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(_result(_segment(text))),
        ner=ner,
    )

    assert result.status is PipelineStatus.PARTIAL
    assert [item.category for item in result.findings] == [FindingCategory.EMAIL]
    assert result.errors == (
        PipelineError("ner_failed", "The NER detector could not process this OCR segment.", "ner"),
    )


def test_single_detector_failure_makes_result_partial_and_keeps_other_findings() -> None:
    text = "fake@example.invalid Ada"
    email_failure = Mock(side_effect=RuntimeError(f"private {text}"))
    ner = FakeNER({text: [(21, 24)]})

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(_result(_segment(text))),
        email_detector=email_failure,
        ner=ner,
    )

    assert result.status is PipelineStatus.PARTIAL
    assert [item.category for item in result.findings] == [FindingCategory.PERSON_NAME]
    assert result.errors[0].code == "email_detector_failed"
    assert "fake@example.invalid" not in result.errors[0].message


def test_all_detector_failures_return_error_without_exception_details() -> None:
    text = "fictional sensitive text"
    failure = lambda value, *, value_policy: (_ for _ in ()).throw(RuntimeError(value))
    ner = FakeNER()
    ner.error = RuntimeError(text)

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(_result(_segment(text))),
        email_detector=failure,
        phone_detector=failure,
        ner=ner,
    )

    assert result.status is PipelineStatus.ERROR
    assert result.findings == ()
    assert {error.code for error in result.errors} == {
        "email_detector_failed",
        "phone_detector_failed",
        "ner_failed",
    }
    assert all(text not in error.message for error in result.errors)


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ValuePolicy.RAW, ("fake@example.invalid", "Ada", None, None)),
        (ValuePolicy.REDACTED, (None, None, "[EMAIL]", "[PERSON_NAME]")),
        (ValuePolicy.BOTH, ("fake@example.invalid", "Ada", "[EMAIL]", "[PERSON_NAME]")),
        (ValuePolicy.NEITHER, (None, None, None, None)),
    ],
)
def test_policy_applies_to_email_and_ner_and_is_forwarded_to_detectors(
    policy: ValuePolicy,
    expected: tuple[str | None, str | None, str | None, str | None],
) -> None:
    text = "Ada fake@example.invalid"
    email_calls: list[ValuePolicy] = []
    phone_calls: list[ValuePolicy] = []

    def email_detector(value: str, *, value_policy: ValuePolicy) -> list[Finding]:
        email_calls.append(value_policy)
        from aegis_ocr.pii import detect_emails

        return detect_emails(value, value_policy=value_policy)

    def phone_detector(value: str, *, value_policy: ValuePolicy) -> list[Finding]:
        phone_calls.append(value_policy)
        from aegis_ocr.pii import detect_phones

        return detect_phones(value, value_policy=value_policy)

    ner = FakeNER({text: [(0, 3)]})
    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(_result(_segment(text))),
        email_detector=email_detector,
        phone_detector=phone_detector,
        ner=ner,
        value_policy=policy,
    )

    email_finding = next(item for item in result.findings if item.category is FindingCategory.EMAIL)
    person_finding = next(item for item in result.findings if item.category is FindingCategory.PERSON_NAME)
    assert (
        email_finding.value,
        person_finding.value,
        email_finding.redacted_value,
        person_finding.redacted_value,
    ) == expected
    assert email_calls == [policy]
    assert phone_calls == [policy]
    assert ner.calls == [(text, policy)]


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ValuePolicy.RAW, ("email raw", "phone raw", "person raw", None, None, None)),
        (ValuePolicy.REDACTED, (None, None, None, "[EMAIL]", "[PHONE]", "[PERSON_NAME]")),
        (ValuePolicy.BOTH, ("email raw", "phone raw", "person raw", "[EMAIL]", "[PHONE]", "[PERSON_NAME]")),
        (ValuePolicy.NEITHER, (None, None, None, None, None, None)),
    ],
)
def test_pipeline_enforces_policy_on_custom_detector_and_ner_findings(
    policy: ValuePolicy,
    expected: tuple[str | None, ...],
) -> None:
    text = "Ada fake@example.invalid +1 202-555-0100"
    email_finding = Finding(
        FindingCategory.EMAIL,
        "custom_email",
        TextSpan(4, 24),
        value="email raw",
        redacted_value="[EMAIL]",
    )
    phone_finding = Finding(
        FindingCategory.PHONE,
        "custom_phone",
        TextSpan(25, len(text)),
        value="phone raw",
        redacted_value="[PHONE]",
    )
    person_finding = Finding(
        FindingCategory.PERSON_NAME,
        "custom_ner",
        TextSpan(0, 3),
        value="person raw",
        redacted_value="[PERSON_NAME]",
    )
    ner = FakeNER()
    ner.result = PipelineResult("1.0", PipelineStatus.OK, findings=(person_finding,))

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(_result(_segment(text))),
        email_detector=lambda _text, **_kwargs: [email_finding],
        phone_detector=lambda _text, **_kwargs: [phone_finding],
        ner=ner,
        value_policy=policy,
    )

    assert result.status is PipelineStatus.OK
    by_category = {finding.category: finding for finding in result.findings}
    assert (
        by_category[FindingCategory.EMAIL].value,
        by_category[FindingCategory.PHONE].value,
        by_category[FindingCategory.PERSON_NAME].value,
        by_category[FindingCategory.EMAIL].redacted_value,
        by_category[FindingCategory.PHONE].redacted_value,
        by_category[FindingCategory.PERSON_NAME].redacted_value,
    ) == expected
    assert all(item.source_segment_index == 0 for item in result.findings)
    assert all(item.image_region == _segment(text).image_region for item in result.findings)


@pytest.mark.parametrize(
    "invalid_output",
    [
        "not a findings collection",
        [object()],
        [Finding(FindingCategory.PHONE, "bad", TextSpan(0, 3))],
        [Finding(FindingCategory.EMAIL, "email_regex", TextSpan(0, 99))],
        [
            Finding(FindingCategory.EMAIL, "email_regex", TextSpan(0, 3)),
            Finding(FindingCategory.EMAIL, "email_regex", TextSpan(0, 3)),
        ],
    ],
)
def test_invalid_detector_output_is_a_safe_detector_failure(invalid_output: object) -> None:
    text = "Ada"
    ner = FakeNER({text: [(0, 3)]})

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(_result(_segment(text))),
        email_detector=Mock(return_value=invalid_output),
        ner=ner,
    )

    assert result.status is PipelineStatus.PARTIAL
    assert [finding.category for finding in result.findings] == [FindingCategory.PERSON_NAME]
    assert result.errors[0].code == "email_detector_failed"


def test_default_spacy_adapter_accepts_pipeline_policy_override() -> None:
    text = "Ada"
    nlp = Mock(return_value=SimpleNamespace(ents=(SimpleNamespace(label_="PERSON", start_char=0, end_char=3),)))
    ner = SpacyNERAdapter(nlp=nlp)

    result = analyze_image(
        "synthetic.png",
        ocr=FakeOCR(_result(_segment(text))),
        ner=ner,
        value_policy=ValuePolicy.RAW,
    )

    assert result.findings[0].value == "Ada"
    assert result.findings[0].redacted_value is None
