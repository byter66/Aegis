from __future__ import annotations

from pathlib import Path
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw
import pytest

from aegis_ocr.ocr import EasyOCREngine, OCRResult, OCRSegment, _tile_starts
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
        self.calls: list[tuple[object, int, bool]] = []
        self.options: list[dict[str, object]] = []

    def readtext(
        self, image: object, *, detail: int, paragraph: bool, **kwargs: object
    ) -> list[object]:
        self.calls.append((image, detail, paragraph))
        self.options.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.results


def _write_png(path: Path) -> Path:
    Image.new("RGB", (24, 12), color="white").save(path, format="PNG")
    return path


class SequencedReader:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls: list[object] = []

    def readtext(
        self, image: object, *, detail: int, paragraph: bool, **kwargs: object
    ) -> list[object]:
        self.calls.append(image)
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]


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


@pytest.mark.parametrize(
    "configuration",
    [
        {"tile_size": True},
        {"tile_size": 0},
        {"tile_size": -1},
        {"tile_size": 1.5},
        {"tile_size": "10"},
        {"tile_size": 10, "tile_overlap": True},
        {"tile_size": 10, "tile_overlap": -1},
        {"tile_size": 10, "tile_overlap": 1.5},
        {"tile_size": 10, "tile_overlap": 10},
        {"tile_overlap": 1},
    ],
)
def test_invalid_tiling_configuration_is_rejected(configuration: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        EasyOCREngine(**configuration)  # type: ignore[arg-type]


def test_tiling_uses_bounded_regions_and_covers_every_source_pixel(tmp_path: Path) -> None:
    image_path = tmp_path / "coordinate-grid.png"
    source = Image.new("RGB", (23, 17))
    source.putdata([(x, y, 0) for y in range(17) for x in range(23)])
    source.save(image_path)
    reader = StubReader()
    factory_calls: list[tuple[list[str], bool]] = []

    def factory(languages: list[str], *, gpu: bool) -> StubReader:
        factory_calls.append((languages, gpu))
        return reader

    engine = EasyOCREngine(reader_factory=factory, tile_size=10, tile_overlap=3)
    assert factory_calls == []

    result = engine.extract(image_path)

    assert result.status is PipelineStatus.OK
    assert factory_calls == [(["en"], False)]
    assert len(reader.calls) == 6
    covered: set[tuple[int, int]] = set()
    for tile, _, _ in reader.calls:
        assert not isinstance(tile, str)
        height, width = tile.shape[:2]
        assert width <= 10
        assert height <= 10
        covered.update(
            (int(pixel[2]), int(pixel[1]))
            for row in tile
            for pixel in row
        )
    assert covered == {(x, y) for y in range(17) for x in range(23)}


@pytest.mark.parametrize("length", [800, 1600, 1601, 1844, 3072, 4096, 6144, 8192])
@pytest.mark.parametrize("overlap", [0, 128])
def test_tile_intervals_cover_axis_without_excessive_overlap(
    length: int, overlap: int
) -> None:
    tile_size = 1600

    starts = _tile_starts(length, tile_size, overlap)
    intervals = [(start, min(start + tile_size, length)) for start in starts]

    assert starts[0] == 0
    assert len(starts) == len(set(starts))
    assert all(start < end <= length for start, end in intervals)
    assert intervals[-1][1] == length
    for (start, end), (next_start, next_end) in zip(intervals, intervals[1:]):
        assert start < next_start
        assert next_start <= end  # no uncovered pixel gap
        assert end - next_start <= overlap
        assert next_end > end  # each tile contributes new pixels


def test_tiled_detections_map_coordinates_deduplicate_overlap_and_keep_edge_text(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "large-synthetic-labeled.png"
    source = Image.new("RGB", (32, 20), "white")
    draw = ImageDraw.Draw(source)
    draw.text((1, 1), "First", fill="black")
    draw.text((13, 5), "Overlap", fill="black")
    draw.text((25, 14), "E", fill="black")
    source.save(image_path)

    outcomes: list[object] = [[] for _ in range(6)]
    outcomes[0] = [
        (_box(1, 1, 3, 3), "first", 0.8),
        (_box(13, 5, 15, 7), "repeat", 0.8),
    ]
    # The second x tile starts at 12: one duplicate and one separate instance.
    outcomes[1] = [
        (_box(1, 5, 3, 7), "repeat", 0.9),
        (_box(7, 5, 9, 7), "repeat", 0.7),
    ]
    # The final tile covers the right and bottom image edges.
    outcomes[5] = [(_box(6, 4, 8, 8), "edge", 0.6)]
    reader = SequencedReader(outcomes)

    result = EasyOCREngine(
        reader=reader, tile_size=16, tile_overlap=4
    ).extract(image_path)

    assert result.status is PipelineStatus.OK
    assert [segment.text for segment in result.segments] == [
        "first",
        "repeat",
        "repeat",
        "edge",
    ]
    polygons = [segment.image_region.polygon for segment in result.segments]
    assert polygons[0] == tuple((float(x), float(y)) for x, y in _box(1, 1, 3, 3))
    # The duplicate keeps the higher-confidence overlap result at original x=13.
    assert polygons[1] == tuple((float(x), float(y)) for x, y in _box(13, 5, 15, 7))
    # The same text at a separate location is not treated as a duplicate.
    assert polygons[2] == tuple((float(x), float(y)) for x, y in _box(19, 5, 21, 7))
    # Final edge tile reaches the source image's right edge without exceeding it.
    assert polygons[3] == tuple((float(x), float(y)) for x, y in _box(30, 16, 32, 20))
    assert all(0 <= x <= 32 and 0 <= y <= 20 for polygon in polygons for x, y in polygon)
    assert len(reader.calls) == 6


def test_tiled_partial_failure_preserves_successful_segments_and_safe_error(
    tmp_path: Path,
) -> None:
    image = _write_png(tmp_path / "partial-tiles.png")
    outcomes: list[object] = [
        RuntimeError("private recognized text"),
        [(_box(1, 1, 4, 4), "synthetic", 0.8)],
        [],
        [],
        [],
        [],
    ]
    reader = SequencedReader(outcomes)

    result = EasyOCREngine(reader=reader, tile_size=10, tile_overlap=2).extract(image)

    assert result.status is PipelineStatus.PARTIAL
    assert [segment.text for segment in result.segments] == ["synthetic"]
    assert result.errors[0].code == "ocr_partial"
    assert "private recognized text" not in result.errors[0].message


def test_tile_timing_is_opt_in_and_reports_only_safe_stage_metadata(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    image = _write_png(tmp_path / "diagnostic.png")
    reader = StubReader([(_box(1, 1, 4, 4), "SENSITIVE_SYNTHETIC_SENTINEL", 0.8)])

    with caplog.at_level(logging.INFO, logger="aegis_ocr.ocr"):
        ordinary = EasyOCREngine(reader=reader, tile_size=10, tile_overlap=2).extract(image)
    assert ordinary.status is PipelineStatus.OK
    assert len(ordinary.segments) == 6
    assert "ocr_diagnostic" not in caplog.text

    caplog.clear()
    reader = StubReader([(_box(1, 1, 4, 4), "SENSITIVE_SYNTHETIC_SENTINEL", 0.8)])
    with caplog.at_level(logging.INFO, logger="aegis_ocr.ocr"):
        diagnostic = EasyOCREngine(
            reader=reader,
            tile_size=10,
            tile_overlap=2,
            diagnostic_timing=True,
        ).extract(image)

    assert diagnostic == ordinary
    messages = caplog.text
    for event in (
        "image_validation",
        "reader_reuse",
        "image_decode",
        "tile_plan",
        "tile_preprocessing",
        "tile_inference_started",
        "tile_inference",
        "tile_result_processing",
        "tile_complete",
        "tile_deduplication",
    ):
        assert f"event={event}" in messages
    assert "tile_count=6" in messages
    assert "tile_index=0" in messages
    assert "tile_width=10" in messages
    assert "status=success" in messages
    assert "SENSITIVE_SYNTHETIC_SENTINEL" not in messages
    assert str(image) not in messages


def test_tile_timing_failure_does_not_change_safe_partial_result(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    image = _write_png(tmp_path / "diagnostic-partial.png")
    reader = SequencedReader(
        [RuntimeError("SENSITIVE_SYNTHETIC_SENTINEL")] + [[] for _ in range(5)]
    )

    with caplog.at_level(logging.INFO, logger="aegis_ocr.ocr"):
        result = EasyOCREngine(
            reader=reader,
            tile_size=10,
            tile_overlap=2,
            diagnostic_timing=True,
        ).extract(image)

    assert result.status is PipelineStatus.PARTIAL
    assert result.segments == ()
    assert result.errors[0].code == "ocr_partial"
    assert "event=tile_inference status=failure" in caplog.text
    assert "event=tile_complete status=failure" in caplog.text
    assert "SENSITIVE_SYNTHETIC_SENTINEL" not in caplog.text


def test_timing_distinguishes_reader_initialization_from_reuse(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    image = _write_png(tmp_path / "reader-timing.png")
    reader = StubReader()
    factory = Mock(return_value=reader)
    engine = EasyOCREngine(
        reader_factory=factory,
        tile_size=10,
        tile_overlap=2,
        diagnostic_timing=True,
    )

    with caplog.at_level(logging.INFO, logger="aegis_ocr.ocr"):
        engine.extract(image)
        engine.extract(image)

    assert caplog.text.count("event=reader_initialization ") == 1
    assert caplog.text.count("event=reader_reuse ") == 1
    factory.assert_called_once_with(["en"], gpu=False)


def test_all_tiled_inference_failures_return_safe_ocr_error(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "all-tiles-fail.png")
    reader = SequencedReader([RuntimeError("sensitive text")] * 6)

    result = EasyOCREngine(reader=reader, tile_size=10, tile_overlap=2).extract(image)

    assert result.status is PipelineStatus.ERROR
    assert result.segments == ()
    assert result.errors[0].code == "ocr_failed"
    assert "sensitive text" not in result.errors[0].message


def test_successful_tiled_empty_scan_is_ok_without_findings(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "empty-tiles.png")
    reader = SequencedReader([[], [], [], [], [], []])

    result = EasyOCREngine(reader=reader, tile_size=10, tile_overlap=2).extract(image)

    assert result.status is PipelineStatus.OK
    assert result.segments == ()
    assert result.errors == ()


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
