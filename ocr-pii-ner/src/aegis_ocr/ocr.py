"""EasyOCR extraction for single JPEG and PNG images."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from numbers import Real
from pathlib import Path
from typing import Protocol, TypeAlias, cast

from PIL import Image, UnidentifiedImageError

from .schema import ImageRegion, PipelineError, PipelineStatus


OCRPoint: TypeAlias = tuple[float, float]


class _Reader(Protocol):
    def readtext(self, image: str, *, detail: int, paragraph: bool) -> list[object]: ...


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
    """

    def __init__(
        self,
        *,
        reader: _Reader | None = None,
        reader_factory: _ReaderFactory | None = None,
    ) -> None:
        self._reader = reader
        self._reader_factory = reader_factory

    def extract(self, image_path: str | Path) -> OCRResult:
        """Extract printed English text from one readable JPEG or PNG image."""

        path, path_error = _validate_path(image_path)
        if path_error is not None:
            return _error_result(path_error)
        assert path is not None

        image_error = _validate_image(path)
        if image_error is not None:
            return _error_result(image_error)

        try:
            reader = self._get_reader()
            raw_results = reader.readtext(str(path), detail=1, paragraph=False)
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
        polygon.append((x, y))
    if not isinstance(raw_text, str):
        raise ValueError("OCR result text must be a string")
    return OCRSegment(
        text=raw_text,
        image_region=ImageRegion(tuple(polygon)),
        ocr_confidence=cast(float, raw_confidence),
    )


def _error_result(error: PipelineError) -> OCRResult:
    return OCRResult(status=PipelineStatus.ERROR, errors=(error,))
