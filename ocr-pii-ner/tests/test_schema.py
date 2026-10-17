from __future__ import annotations

import pytest

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


def test_valid_statuses_and_structured_error() -> None:
    for status in PipelineStatus:
        result = PipelineResult("1.0", status)
        assert result.status is status

    error = PipelineError(code="ocr_failed", message="Unable to process image", stage="ocr")
    result = PipelineResult("1.0", PipelineStatus.ERROR, errors=(error,))
    assert result.errors == (error,)


def test_span_uses_non_empty_end_exclusive_offsets() -> None:
    text = "prefix fake@example.invalid suffix"
    span = TextSpan(7, 27)
    assert text[span.start:span.end] == "fake@example.invalid"
    with pytest.raises(ValueError):
        TextSpan(5, 5)


def test_optional_image_region_and_separate_scores() -> None:
    finding = Finding(
        category=FindingCategory.EMAIL,
        detector="email_regex",
        span=TextSpan(0, 4),
        image_region=ImageRegion(((0.0, 0.0), (5.0, 0.0), (5.0, 3.0))),
        ocr_confidence=0.8,
    )
    assert finding.ocr_confidence == 0.8
    assert finding.detector_score is None


def test_detector_score_requires_explicit_semantics() -> None:
    with pytest.raises(ValueError):
        Finding(
            category=FindingCategory.EMAIL,
            detector="email_regex",
            span=TextSpan(0, 4),
            detector_score=0.75,
        )


def test_result_rejects_duplicate_category_and_span() -> None:
    finding = Finding(FindingCategory.EMAIL, "email_regex", TextSpan(0, 4))
    with pytest.raises(ValueError, match="duplicate finding"):
        PipelineResult("1.0", PipelineStatus.OK, findings=(finding, finding))


@pytest.mark.parametrize("index", [-1, True, False, 1.5, "1"])
def test_source_segment_index_rejects_invalid_values(index: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Finding(
            FindingCategory.EMAIL,
            "email_regex",
            TextSpan(0, 4),
            source_segment_index=index,  # type: ignore[arg-type]
        )


def test_duplicate_findings_in_distinct_segments_are_allowed() -> None:
    first = Finding(
        FindingCategory.PERSON_NAME,
        "spacy_ner",
        TextSpan(0, 3),
        source_segment_index=0,
    )
    second = Finding(
        FindingCategory.PERSON_NAME,
        "spacy_ner",
        TextSpan(0, 3),
        source_segment_index=1,
    )

    result = PipelineResult("1.0", PipelineStatus.OK, findings=(first, second))

    assert result.findings == (first, second)


def test_duplicate_findings_in_same_segment_are_rejected() -> None:
    first = Finding(
        FindingCategory.PERSON_NAME,
        "spacy_ner",
        TextSpan(0, 3),
        source_segment_index=2,
    )
    duplicate = Finding(
        FindingCategory.PERSON_NAME,
        "spacy_ner",
        TextSpan(0, 3),
        source_segment_index=2,
    )

    with pytest.raises(ValueError, match="duplicate finding"):
        PipelineResult("1.0", PipelineStatus.OK, findings=(first, duplicate))


def test_image_region_copies_outer_and_nested_mutable_lists() -> None:
    point_a = [1.0, 2.0]
    point_b = [5.0, 2.0]
    point_c = [5.0, 6.0]
    polygon = [point_a, point_b, point_c]

    region = ImageRegion(polygon)  # type: ignore[arg-type]

    point_a[0] = 99.0
    polygon.append([0.0, 0.0])
    assert region.polygon == ((1.0, 2.0), (5.0, 2.0), (5.0, 6.0))
    assert isinstance(region.polygon, tuple)
    assert all(isinstance(point, tuple) for point in region.polygon)


@pytest.mark.parametrize(
    "polygon",
    [
        None,
        "not a polygon",
        [object(), [1.0, 2.0], [3.0, 4.0]],
        [[1.0], [3.0, 4.0], [5.0, 6.0]],
    ],
)
def test_image_region_rejects_malformed_collection_or_point_inputs(polygon: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        ImageRegion(polygon)  # type: ignore[arg-type]


@pytest.mark.parametrize("coordinate", [True, False, "1.0", object()])
def test_image_region_rejects_boolean_and_non_numeric_coordinates(
    coordinate: object,
) -> None:
    polygon = ((coordinate, 0.0), (1.0, 0.0), (1.0, 1.0))

    with pytest.raises(TypeError, match="coordinates must be real numbers"):
        ImageRegion(polygon)  # type: ignore[arg-type]


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), float("-inf")])
def test_image_region_rejects_non_finite_coordinates(coordinate: float) -> None:
    polygon = ((coordinate, 0.0), (1.0, 0.0), (1.0, 1.0))

    with pytest.raises(ValueError, match="finite and non-negative"):
        ImageRegion(polygon)


def test_pipeline_result_copies_lists_before_duplicate_validation() -> None:
    finding = Finding(FindingCategory.EMAIL, "email_regex", TextSpan(0, 4))
    error = PipelineError("detector_failed", "safe failure", "email_detector")
    findings = [finding]
    errors = [error]

    result = PipelineResult(
        "1.0",
        PipelineStatus.PARTIAL,
        findings=findings,  # type: ignore[arg-type]
        errors=errors,  # type: ignore[arg-type]
    )
    findings.append(finding)
    errors.clear()

    assert result.findings == (finding,)
    assert result.errors == (error,)
    with pytest.raises(ValueError, match="duplicate finding"):
        PipelineResult("1.0", PipelineStatus.OK, findings=[finding, finding])  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["findings", "errors"])
def test_pipeline_result_rejects_unsupported_collection_types(field: str) -> None:
    kwargs = {field: None}
    with pytest.raises(TypeError, match=f"{field} must be a list or tuple"):
        PipelineResult("1.0", PipelineStatus.OK, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ValuePolicy.RAW, ("fake@example.invalid", None)),
        (ValuePolicy.REDACTED, (None, "[EMAIL]")),
        (ValuePolicy.BOTH, ("fake@example.invalid", "[EMAIL]")),
        (ValuePolicy.NEITHER, (None, None)),
    ],
)
def test_value_policy(policy: ValuePolicy, expected: tuple[str | None, str | None]) -> None:
    assert apply_value_policy("fake@example.invalid", "[EMAIL]", policy) == expected
