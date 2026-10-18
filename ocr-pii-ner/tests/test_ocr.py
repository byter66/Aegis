from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image
import pytest

from aegis_ocr.ocr import EasyOCREngine, OCRResult, OCRSegment
from aegis_ocr.schema import (
    Finding,
    FindingCategory,
    ImageRegion,
    PipelineError,
    PipelineStatus,
    TextSpan,
)


class StubReader:
    def __init__(self, results: list[object] | None = None, error: Exception | None = None) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[tuple[str, int, bool]] = []
        self.options: list[dict[str, object]] = []

    def readtext(
        self, image: str, *, detail: int, paragraph: bool, **kwargs: object
    ) -> list[object]:
        self.calls.append((image, detail, paragraph))
        self.options.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results


def _write_png(path: Path) -> Path:
    Image.new("RGB", (24, 12), color="white").save(path, format="PNG")
    return path


def _box(left: int, top: int, right: int, bottom: int) -> list[tuple[int, int]]:
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def test_invalid_and_missing_paths_return_safe_errors(tmp_path: Path) -> None:
    engine = EasyOCREngine(reader=StubReader())

    invalid = engine.extract("")
    invalid_type = engine.extract(None)  # type: ignore[arg-type]
    missing = engine.extract(tmp_path / "missing.png")

    assert invalid.status is PipelineStatus.ERROR
    assert invalid.errors[0].code == "invalid_path"
    assert invalid_type.status is PipelineStatus.ERROR
    assert invalid_type.errors[0].code == "invalid_path"
    assert missing.status is PipelineStatus.ERROR
    assert missing.errors[0].code == "image_not_found"
    assert str(tmp_path) not in missing.errors[0].message


def test_unsupported_extension_is_rejected_before_ocr(tmp_path: Path) -> None:
    gif = tmp_path / "image.gif"
    gif.write_bytes(b"not an image")
    reader = StubReader()

    result = EasyOCREngine(reader=reader).extract(gif)

    assert result.status is PipelineStatus.ERROR
    assert result.errors[0].code == "unsupported_format"
    assert reader.calls == []


def test_corrupt_image_returns_unreadable_error(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"definitely not a PNG")

    result = EasyOCREngine(reader=StubReader()).extract(corrupt)

    assert result.status is PipelineStatus.ERROR
    assert result.errors[0].code == "image_unreadable"


def test_empty_ocr_results_are_success_not_failure(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "empty.png")

    result = EasyOCREngine(reader=StubReader()).extract(image)

    assert result.status is PipelineStatus.OK
    assert result.segments == ()
    assert result.errors == ()


@pytest.mark.parametrize("canvas_size", [0, -1, True, 1.5, "1280"])
def test_canvas_size_must_be_a_positive_integer_or_none(canvas_size: object) -> None:
    with pytest.raises(ValueError, match="canvas_size must be a positive integer or None"):
        EasyOCREngine(canvas_size=canvas_size)  # type: ignore[arg-type]


def test_default_inference_sizing_preserves_existing_readtext_arguments(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "default-sizing.png")
    reader = StubReader()

    result = EasyOCREngine(reader=reader).extract(image)

    assert result.status is PipelineStatus.OK
    assert reader.calls == [(str(image), 1, False)]
    assert reader.options == [{}]


def test_configured_canvas_size_is_forwarded_to_readtext(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "reduced-canvas.png")
    reader = StubReader()

    result = EasyOCREngine(reader=reader, canvas_size=1280).extract(image)

    assert result.status is PipelineStatus.OK
    assert reader.calls == [(str(image), 1, False)]
    assert reader.options == [{"canvas_size": 1280}]


def test_multiple_segments_preserve_engine_order_scores_and_polygons(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "segments.png")
    recognized = [
        (_box(3, 4, 18, 10), "first synthetic", 0.82),
        (_box(2, 14, 20, 20), "second synthetic", 0.67),
    ]
    reader = StubReader(recognized)

    result = EasyOCREngine(reader=reader).extract(image)

    assert result.status is PipelineStatus.OK
    assert [segment.text for segment in result.segments] == ["first synthetic", "second synthetic"]
    assert [segment.ocr_confidence for segment in result.segments] == [0.82, 0.67]
    assert result.segments[0].image_region.polygon == tuple((float(x), float(y)) for x, y in _box(3, 4, 18, 10))
    assert reader.calls == [(str(image), 1, False)]


def test_engine_failure_returns_safe_error_without_exception_text(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "failure.png")
    reader = StubReader(error=RuntimeError("sensitive extracted text must not leak"))

    result = EasyOCREngine(reader=reader).extract(image)

    assert result.status is PipelineStatus.ERROR
    assert result.errors[0].code == "ocr_failed"
    assert "sensitive extracted text" not in result.errors[0].message


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_ocr_coordinates_return_safe_error(tmp_path: Path, coordinate: float) -> None:
    image = _write_png(tmp_path / "invalid-coordinate.png")
    malformed = [([(coordinate, 0), (10, 0), (10, 5), (0, 5)], "synthetic", 0.8)]

    result = EasyOCREngine(reader=StubReader(malformed)).extract(image)

    assert result.status is PipelineStatus.ERROR
    assert result.segments == ()
    assert result.errors[0].code == "ocr_failed"
    assert "nan" not in result.errors[0].message.lower()
    assert "inf" not in result.errors[0].message.lower()


@pytest.mark.parametrize(
    "confidence",
    ["not-a-number", "0.8", None, True, float("nan"), float("inf"), float("-inf"), -0.01, 1.01],
)
def test_malformed_ocr_confidence_returns_safe_error(tmp_path: Path, confidence: object) -> None:
    image = _write_png(tmp_path / "invalid-confidence.png")
    malformed = [(_box(0, 0, 10, 5), "synthetic", confidence)]

    result = EasyOCREngine(reader=StubReader(malformed)).extract(image)

    assert result.status is PipelineStatus.ERROR
    assert result.segments == ()
    assert result.errors[0].code == "ocr_failed"
    assert "not-a-number" not in result.errors[0].message


def test_default_easyocr_reader_construction_is_mocked_and_reused(tmp_path: Path) -> None:
    image_a = _write_png(tmp_path / "default-a.png")
    image_b = _write_png(tmp_path / "default-b.png")
    reader = StubReader()
    reader_constructor = Mock(return_value=reader)

    with patch(
        "aegis_ocr.ocr.importlib.import_module",
        return_value=SimpleNamespace(Reader=reader_constructor),
    ) as import_module:
        engine = EasyOCREngine()
        first = engine.extract(image_a)
        second = engine.extract(image_b)

    assert first.status is second.status is PipelineStatus.OK
    import_module.assert_called_once_with("easyocr")
    reader_constructor.assert_called_once_with(["en"], gpu=False)
    assert len(reader.calls) == 2


def test_reader_is_initialized_once_and_reused(tmp_path: Path) -> None:
    image_a = _write_png(tmp_path / "a.png")
    image_b = _write_png(tmp_path / "b.png")
    reader = StubReader()
    calls: list[tuple[list[str], bool]] = []

    def factory(languages: list[str], *, gpu: bool) -> StubReader:
        calls.append((languages, gpu))
        return reader

    engine = EasyOCREngine(reader_factory=factory)
    first = engine.extract(image_a)
    second = engine.extract(image_b)

    assert first.status is second.status is PipelineStatus.OK
    assert calls == [(["en"], False)]
    assert len(reader.calls) == 2


def test_ocr_score_is_not_copied_into_detector_score(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "score.png")
    result = EasyOCREngine(reader=StubReader([(_box(0, 0, 10, 5), "fake@example.invalid", 0.93)])).extract(image)
    segment = result.segments[0]
    finding = Finding(
        category=FindingCategory.EMAIL,
        detector="email_regex",
        span=TextSpan(0, len(segment.text)),
        ocr_confidence=segment.ocr_confidence,
    )

    assert finding.ocr_confidence == 0.93
    assert finding.detector_score is None


def test_ocr_result_copies_mutable_segment_and_error_lists() -> None:
    segment = OCRSegment(
        "synthetic",
        ImageRegion(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0))),
        0.8,
    )
    error = PipelineError("ocr_failed", "safe failure", "ocr")
    segments = [segment]
    errors = [error]

    result = OCRResult(
        PipelineStatus.PARTIAL,
        segments=segments,  # type: ignore[arg-type]
        errors=errors,  # type: ignore[arg-type]
    )
    segments.clear()
    errors.clear()

    assert result.segments == (segment,)
    assert result.errors == (error,)


@pytest.mark.parametrize("field", ["segments", "errors"])
def test_ocr_result_rejects_unsupported_collection_types(field: str) -> None:
    kwargs = {field: None}
    with pytest.raises(TypeError, match=f"{field} must be a list or tuple"):
        OCRResult(PipelineStatus.OK, **kwargs)  # type: ignore[arg-type]
