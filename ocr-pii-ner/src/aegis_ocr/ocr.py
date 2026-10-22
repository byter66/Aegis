"""EasyOCR extraction for single JPEG and PNG images."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import math
from numbers import Real
from pathlib import Path
import time
from typing import Protocol, TypeAlias, cast

import numpy as np
from PIL import Image, UnidentifiedImageError

from .schema import ImageRegion, PipelineError, PipelineStatus


OCRPoint: TypeAlias = tuple[float, float]
_TileBounds: TypeAlias = tuple[int, int, int, int]
_TILE_DUPLICATE_IOU = 0.5
_LOGGER = logging.getLogger(__name__)


class _Reader(Protocol):
    def readtext(
        self,
        image: object,
        *,
        detail: int,
        paragraph: bool,
        **kwargs: object,
    ) -> list[object]: ...


class _ReaderFactory(Protocol):
    def __call__(self, languages: list[str], *, gpu: bool) -> _Reader: ...


@dataclass(frozen=True, slots=True)
class OCRSegment:
    """One recognized text segment, its OCR score, and source-image polygon."""

    text: str
    image_region: ImageRegion
    ocr_confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("OCR segment text must be a non-empty string")
        if not isinstance(self.image_region, ImageRegion):
            raise TypeError("image_region must be an ImageRegion")
        if (
            isinstance(self.ocr_confidence, bool)
            or not isinstance(self.ocr_confidence, Real)
            or not math.isfinite(self.ocr_confidence)
            or not 0.0 <= self.ocr_confidence <= 1.0
        ):
            raise ValueError("ocr_confidence must be finite and between 0.0 and 1.0")
        object.__setattr__(self, "ocr_confidence", float(self.ocr_confidence))


@dataclass(frozen=True, slots=True)
class OCRResult:
    """Result for one image; OCR engine failures are represented as errors."""

    status: PipelineStatus
    segments: tuple[OCRSegment, ...] = ()
    errors: tuple[PipelineError, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.segments, (list, tuple)):
            raise TypeError("segments must be a list or tuple")
        if not isinstance(self.errors, (list, tuple)):
            raise TypeError("errors must be a list or tuple")
        object.__setattr__(self, "segments", tuple(self.segments))
        object.__setattr__(self, "errors", tuple(self.errors))
        if not isinstance(self.status, PipelineStatus):
            raise TypeError("status must be a PipelineStatus")
        if any(not isinstance(segment, OCRSegment) for segment in self.segments):
            raise TypeError("segments must contain only OCRSegment instances")
        if any(not isinstance(error, PipelineError) for error in self.errors):
            raise TypeError("errors must contain only PipelineError instances")
        if self.status is PipelineStatus.OK and self.errors:
            raise ValueError("successful OCR results cannot contain errors")
        if self.status is PipelineStatus.ERROR and not self.errors:
            raise ValueError("failed OCR results must contain at least one error")


class EasyOCREngine:
    """Reusable, lazily initialized EasyOCR reader configured for English/CPU.

    Pass a ``reader`` or ``reader_factory`` to substitute a stub in tests. The
    default reader is loaded once per engine instance; model downloads follow
    EasyOCR's own cache behavior and are never attempted for invalid images.

    This MVP class is not thread-safe: lazy initialization and calls to the
    underlying reader are not synchronized. Use one engine per worker or
    synchronize calls externally.

    Set ``diagnostic_timing=True`` to emit privacy-safe INFO logs for image
    validation/decoding, reader initialization or reuse, tile preprocessing,
    inference, result conversion, and deduplication. Logs contain timing and geometry only;
    configure the ``aegis_ocr.ocr`` logger at INFO to collect them.
    """

    def __init__(
        self,
        *,
        reader: _Reader | None = None,
        reader_factory: _ReaderFactory | None = None,
        canvas_size: int | None = None,
        tile_size: int | None = None,
        tile_overlap: int = 0,
        diagnostic_timing: bool = False,
    ) -> None:
        if not isinstance(diagnostic_timing, bool):
            raise ValueError("diagnostic_timing must be a boolean")
        if canvas_size is not None and (
            isinstance(canvas_size, bool) or not isinstance(canvas_size, int) or canvas_size <= 0
        ):
            raise ValueError("canvas_size must be a positive integer or None")
        if tile_size is not None and (
            isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size <= 0
        ):
            raise ValueError("tile_size must be a positive integer or None")
        if isinstance(tile_overlap, bool) or not isinstance(tile_overlap, int):
            raise ValueError("tile_overlap must be a non-negative integer")
        if tile_overlap < 0:
            raise ValueError("tile_overlap must be a non-negative integer")
        if tile_size is None and tile_overlap != 0:
            raise ValueError("tile_overlap must be 0 when tile_size is not configured")
        if tile_size is not None and tile_overlap >= tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size")
        self._reader = reader
        self._reader_factory = reader_factory
        self._canvas_size = canvas_size
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._diagnostic_timing = diagnostic_timing

    def extract(self, image_path: str | Path) -> OCRResult:
        """Extract printed English text from one readable JPEG or PNG image."""

        path, path_error = _validate_path(image_path)
        if path_error is not None:
            return _error_result(path_error)
        assert path is not None

        started = time.perf_counter() if self._diagnostic_timing else 0.0
        image_error = _validate_image(path)
        if self._diagnostic_timing:
            _emit_timing(
                "image_validation",
                started,
                status="failure" if image_error else "success",
            )
        if image_error is not None:
            return _error_result(image_error)

        try:
            reader_was_uninitialized = self._reader is None
            reader_started = time.perf_counter() if self._diagnostic_timing else 0.0
            reader: _Reader | None = None
            try:
                reader = self._get_reader()
            finally:
                if self._diagnostic_timing:
                    _emit_timing(
                        "reader_initialization"
                        if reader_was_uninitialized
                        else "reader_reuse",
                        reader_started,
                        status="success" if reader is not None else "failure",
                    )
            assert reader is not None
            inference_options = (
                {"canvas_size": self._canvas_size}
                if self._canvas_size is not None
                else {}
            )
            if self._tile_size is not None:
                return self._extract_tiled(path, reader, inference_options)
            raw_results = reader.readtext(
                str(path), detail=1, paragraph=False, **inference_options
            )
            segments = tuple(_to_segment(item) for item in raw_results)
        except Exception:
            # Exception messages may include paths or extracted text; keep them out of results.
            return _error_result(
                PipelineError(
                    code="ocr_failed",
                    message="The OCR engine could not process this image.",
                    stage="ocr",
                )
            )
        return OCRResult(status=PipelineStatus.OK, segments=segments)

    def _extract_tiled(
        self,
        path: Path,
        reader: _Reader,
        inference_options: dict[str, object],
    ) -> OCRResult:
        assert self._tile_size is not None
        detections: list[tuple[OCRSegment, int, _TileBounds]] = []
        tile_count = 0
        failed_tiles = 0

        decode_started = time.perf_counter() if self._diagnostic_timing else 0.0
        decode_complete = False
        try:
            with Image.open(path) as image:
                image.load()
                decode_complete = True
                if self._diagnostic_timing:
                    _emit_timing("image_decode", decode_started, status="success")
                image_width, image_height = image.size
                plan_started = time.perf_counter() if self._diagnostic_timing else 0.0
                x_starts = _tile_starts(image_width, self._tile_size, self._tile_overlap)
                y_starts = _tile_starts(image_height, self._tile_size, self._tile_overlap)
                if self._diagnostic_timing:
                    _emit_timing(
                        "tile_plan",
                        plan_started,
                        status="success",
                        image_width=image_width,
                        image_height=image_height,
                        tile_size=self._tile_size,
                        tile_overlap=self._tile_overlap,
                        tile_count=len(x_starts) * len(y_starts),
                    )

                for top in y_starts:
                    for left in x_starts:
                        right = min(left + self._tile_size, image_width)
                        bottom = min(top + self._tile_size, image_height)
                        bounds = (left, top, right, bottom)
                        tile_index = tile_count
                        tile_count += 1
                        tile_started = time.perf_counter() if self._diagnostic_timing else 0.0
                        try:
                            tile_segments = _read_tile(
                                image,
                                reader,
                                bounds,
                                image_size=(image_width, image_height),
                                inference_options=inference_options,
                                diagnostic_timing=self._diagnostic_timing,
                                tile_index=tile_index,
                            )
                        except Exception:
                            # Keep exception messages and recognized text out of errors.
                            failed_tiles += 1
                            if self._diagnostic_timing:
                                _emit_timing(
                                    "tile_complete",
                                    tile_started,
                                    status="failure",
                                    tile_index=tile_index,
                                    tile_width=right - left,
                                    tile_height=bottom - top,
                                    segment_count=0,
                                )
                            continue
                        if self._diagnostic_timing:
                            _emit_timing(
                                "tile_complete",
                                tile_started,
                                status="success",
                                tile_index=tile_index,
                                tile_width=right - left,
                                tile_height=bottom - top,
                                segment_count=len(tile_segments),
                            )
                        detections.extend(
                            (segment, tile_index, bounds) for segment in tile_segments
                        )
        except Exception:
            if self._diagnostic_timing and not decode_complete:
                _emit_timing("image_decode", decode_started, status="failure")
            raise

        dedup_started = time.perf_counter() if self._diagnostic_timing else 0.0
        deduplicated = _deduplicate_tile_segments(detections)
        if self._diagnostic_timing:
            _emit_timing(
                "tile_deduplication",
                dedup_started,
                status="success",
                input_segments=len(detections),
                output_segments=len(deduplicated),
            )

        if failed_tiles == tile_count:
            return _error_result(
                PipelineError(
                    code="ocr_failed",
                    message="The OCR engine could not process this image.",
                    stage="ocr",
                )
            )
        if failed_tiles:
            return OCRResult(
                status=PipelineStatus.PARTIAL,
                segments=deduplicated,
                errors=(
                    PipelineError(
                        code="ocr_partial",
                        message="OCR returned a partial result.",
                        stage="ocr",
                    ),
                ),
            )
        return OCRResult(
            status=PipelineStatus.OK,
            segments=deduplicated,
        )

    def _get_reader(self) -> _Reader:
        if self._reader is None:
            factory = self._reader_factory
            if factory is None:
                factory = importlib.import_module("easyocr").Reader
            if not callable(factory):
                raise TypeError("reader_factory must be callable")
            self._reader = factory(["en"], gpu=False)
        return self._reader


def _validate_path(image_path: str | Path) -> tuple[Path | None, PipelineError | None]:
    if not isinstance(image_path, (str, Path)) or not str(image_path).strip():
        return None, PipelineError(
            code="invalid_path",
            message="A non-empty image path is required.",
            stage="input_validation",
        )
    try:
        path = Path(image_path)
    except (TypeError, ValueError, OSError):
        return None, PipelineError(
            code="invalid_path",
            message="The image path is invalid.",
            stage="input_validation",
        )
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return None, PipelineError(
            code="unsupported_format",
            message="Only JPEG and PNG image files are supported.",
            stage="input_validation",
        )
    try:
        is_file = path.is_file()
    except (OSError, ValueError):
        return None, PipelineError(
            code="invalid_path",
            message="The image path could not be accessed.",
            stage="input_validation",
        )
    if not is_file:
        return None, PipelineError(
            code="image_not_found",
            message="The image file does not exist or is not a regular file.",
            stage="input_validation",
        )
    return path, None


def _validate_image(path: Path) -> PipelineError | None:
    try:
        with Image.open(path) as image:
            actual_format = image.format
            image.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError):
        return PipelineError(
            code="image_unreadable",
            message="The image file is unreadable or corrupt.",
            stage="image_validation",
        )
    expected_format = "PNG" if path.suffix.lower() == ".png" else "JPEG"
    if actual_format != expected_format:
        return PipelineError(
            code="unsupported_format",
            message="The image contents do not match the supported file extension.",
            stage="image_validation",
        )
    return None


def _to_segment(raw: object) -> OCRSegment:
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        raise ValueError("OCR result must contain a bounding polygon, text, and confidence")
    raw_polygon, raw_text, raw_confidence = raw
    if not isinstance(raw_polygon, (tuple, list)) or len(raw_polygon) < 3:
        raise ValueError("OCR result has an invalid bounding polygon")
    polygon: list[OCRPoint] = []
    for point in raw_polygon:
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError("OCR result has an invalid polygon point")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("OCR polygon coordinates must be finite")
        x, y = max(0.0, x), max(0.0, y)
        polygon.append((x, y))
    if not isinstance(raw_text, str):
        raise ValueError("OCR result text must be a string")
    return OCRSegment(
        text=raw_text,
        image_region=ImageRegion(tuple(polygon)),
        ocr_confidence=cast(float, raw_confidence),
    )


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = [0]
    while starts[-1] + tile_size < length:
        starts.append(starts[-1] + stride)
    return starts


def _read_tile(
    image: Image.Image,
    reader: _Reader,
    bounds: _TileBounds,
    *,
    image_size: tuple[int, int],
    inference_options: dict[str, object],
    diagnostic_timing: bool = False,
    tile_index: int = 0,
) -> tuple[OCRSegment, ...]:
    left, top, right, bottom = bounds
    tile_width, tile_height = right - left, bottom - top
    preprocess_started = time.perf_counter() if diagnostic_timing else 0.0
    preprocess_succeeded = False
    try:
        tile = image.crop(bounds)
        if tile.mode != "RGB":
            tile = tile.convert("RGB")
        # EasyOCR treats three-channel ndarray input as BGR. Only the bounded tile
        # is copied into that layout; the full source image is never converted.
        tile_bgr = np.ascontiguousarray(np.asarray(tile)[:, :, ::-1])
        preprocess_succeeded = True
    finally:
        if diagnostic_timing:
            _emit_timing(
                "tile_preprocessing",
                preprocess_started,
                status="success" if preprocess_succeeded else "failure",
                tile_index=tile_index,
                tile_width=tile_width,
                tile_height=tile_height,
            )

    inference_started = time.perf_counter() if diagnostic_timing else 0.0
    if diagnostic_timing:
        _emit_timing(
            "tile_inference_started",
            inference_started,
            status="started",
            tile_index=tile_index,
            tile_width=tile_width,
            tile_height=tile_height,
        )
    inference_succeeded = False
    try:
        raw_results = reader.readtext(
            tile_bgr, detail=1, paragraph=False, **inference_options
        )
        inference_succeeded = True
    finally:
        if diagnostic_timing:
            _emit_timing(
                "tile_inference",
                inference_started,
                status="success" if inference_succeeded else "failure",
                tile_index=tile_index,
                tile_width=tile_width,
                tile_height=tile_height,
            )

    processing_started = time.perf_counter() if diagnostic_timing else 0.0
    processing_succeeded = False
    try:
        segments: list[OCRSegment] = []
        image_width, image_height = image_size
        for item in raw_results:
            segment = _to_segment(item)
            translated: list[OCRPoint] = []
            for x, y in segment.image_region.polygon:
                # Reject impossible local coordinates rather than clamping or
                # fabricating a point at an image edge.
                if not 0.0 <= x <= tile_width or not 0.0 <= y <= tile_height:
                    raise ValueError("OCR result coordinates exceed the image tile")
                source_x, source_y = x + left, y + top
                if source_x > image_width or source_y > image_height:
                    raise ValueError("OCR result coordinates exceed the source image")
                translated.append((source_x, source_y))
            segments.append(
                OCRSegment(
                    text=segment.text,
                    image_region=ImageRegion(tuple(translated)),
                    ocr_confidence=segment.ocr_confidence,
                )
            )
        processing_succeeded = True
        return tuple(segments)
    finally:
        if diagnostic_timing:
            _emit_timing(
                "tile_result_processing",
                processing_started,
                status="success" if processing_succeeded else "failure",
                tile_index=tile_index,
                tile_width=tile_width,
                tile_height=tile_height,
                segment_count=len(segments) if processing_succeeded else 0,
            )


def _deduplicate_tile_segments(
    detections: list[tuple[OCRSegment, int, _TileBounds]],
) -> tuple[OCRSegment, ...]:
    """Remove only strong same-text duplicates from overlapping source tiles."""

    kept: list[tuple[OCRSegment, int, _TileBounds]] = []
    for candidate in detections:
        segment, tile_index, tile_bounds = candidate
        duplicate_index = next(
            (
                index
                for index, (existing, existing_tile, existing_bounds) in enumerate(kept)
                if tile_index != existing_tile
                and _tiles_overlap(tile_bounds, existing_bounds)
                and _normalized_text(segment.text) == _normalized_text(existing.text)
                and _box_iou(segment.image_region.polygon, existing.image_region.polygon)
                >= _TILE_DUPLICATE_IOU
            ),
            None,
        )
        if duplicate_index is None:
            kept.append(candidate)
        elif segment.ocr_confidence > kept[duplicate_index][0].ocr_confidence:
            kept[duplicate_index] = candidate
    return tuple(segment for segment, _, _ in kept)


def _normalized_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def _tiles_overlap(first: _TileBounds, second: _TileBounds) -> bool:
    return (
        max(first[0], second[0]) < min(first[2], second[2])
        and max(first[1], second[1]) < min(first[3], second[3])
    )


def _box_iou(first: tuple[OCRPoint, ...], second: tuple[OCRPoint, ...]) -> float:
    first_left = min(point[0] for point in first)
    first_top = min(point[1] for point in first)
    first_right = max(point[0] for point in first)
    first_bottom = max(point[1] for point in first)
    second_left = min(point[0] for point in second)
    second_top = min(point[1] for point in second)
    second_right = max(point[0] for point in second)
    second_bottom = max(point[1] for point in second)
    intersection_width = max(
        0.0, min(first_right, second_right) - max(first_left, second_left)
    )
    intersection_height = max(
        0.0, min(first_bottom, second_bottom) - max(first_top, second_top)
    )
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_right - first_left) * max(
        0.0, first_bottom - first_top
    )
    second_area = max(0.0, second_right - second_left) * max(
        0.0, second_bottom - second_top
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _emit_timing(event: str, started: float, *, status: str, **fields: int) -> None:
    """Emit only fixed event names, numeric metadata, and a fixed status token."""
    try:
        details = " ".join(f"{name}={value}" for name, value in fields.items())
        suffix = f" {details}" if details else ""
        elapsed_ms = (time.perf_counter() - started) * 1000
        _LOGGER.info(
            "ocr_diagnostic event=%s status=%s elapsed_ms=%.3f%s",
            event,
            status,
            elapsed_ms,
            suffix,
        )
    except Exception:
        # Diagnostics must never alter OCR results or error handling.
        pass


def _error_result(error: PipelineError) -> OCRResult:
    return OCRResult(status=PipelineStatus.ERROR, errors=(error,))
