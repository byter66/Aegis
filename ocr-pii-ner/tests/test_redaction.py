from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from aegis_ocr.redaction import RedactionError, redact_image
from aegis_ocr.schema import (
    Finding,
    FindingCategory,
    ImageRegion,
    PipelineResult,
    PipelineStatus,
    TextSpan,
)


def _finding(
    polygon: tuple[tuple[float, float], ...] | None,
    *,
    category: FindingCategory = FindingCategory.EMAIL,
) -> Finding:
    return Finding(
        category=category,
        detector="synthetic_test",
        span=TextSpan(0, 1),
        image_region=ImageRegion(polygon) if polygon is not None else None,
    )


def _result(*findings: Finding, status: PipelineStatus = PipelineStatus.OK) -> PipelineResult:
    return PipelineResult(schema_version="1.0", status=status, findings=findings)


def test_redacts_multiple_regions_and_preserves_one_image_for_preview_and_export() -> None:
    source = Image.new("RGB", (100, 80), "white")
    result = _result(
        _finding(((10, 10), (30, 10), (30, 20), (10, 20))),
        _finding(
            ((70, 50), (95, 50), (95, 65), (70, 65)),
            category=FindingCategory.ADDRESS,
        ),
    )

    redacted = redact_image(source, result)
    preview_pixel = redacted.getpixel((15, 15))

    assert preview_pixel == (0, 0, 0)
    assert redacted.getpixel((80, 55)) == (0, 0, 0)
    assert redacted.getpixel((50, 35)) == (255, 255, 255)
    assert source.getpixel((15, 15)) == (255, 255, 255)

    exported = BytesIO()
    redacted.save(exported, format="PNG")
    exported.seek(0)
    with Image.open(exported) as reloaded:
        assert reloaded.convert("RGB").tobytes() == redacted.tobytes()


def test_out_of_bounds_region_is_clipped_to_image_edge() -> None:
    source = Image.new("RGB", (20, 20), "white")
    result = _result(_finding(((18, 18), (25, 18), (25, 25), (18, 25))))

    redacted = redact_image(source, result)

    assert redacted.getpixel((19, 19)) == (0, 0, 0)
    assert redacted.getpixel((10, 10)) == (255, 255, 255)


def test_region_entirely_outside_image_fails_closed() -> None:
    result = _result(_finding(((30, 30), (40, 30), (40, 40), (30, 40))))

    with pytest.raises(RedactionError, match="outside the image bounds"):
        redact_image(Image.new("RGB", (20, 20), "white"), result)


@pytest.mark.parametrize(
    "polygon",
    [
        ((-1, 1), (3, 1), (3, 3)),
        ((float("nan"), 1), (3, 1), (3, 3)),
        ((float("inf"), 1), (3, 1), (3, 3)),
    ],
)
def test_invalid_negative_or_non_finite_region_is_rejected(
    polygon: tuple[tuple[float, float], ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _finding(polygon)


def test_missing_region_fails_closed() -> None:
    with pytest.raises(RedactionError, match="no reliable image region"):
        redact_image(Image.new("RGB", (20, 20), "white"), _result(_finding(None)))


@pytest.mark.parametrize("status", [PipelineStatus.PARTIAL, PipelineStatus.ERROR])
def test_incomplete_analysis_fails_closed(status: PipelineStatus) -> None:
    with pytest.raises(RedactionError, match="Incomplete analysis"):
        redact_image(Image.new("RGB", (20, 20), "white"), _result(status=status))

