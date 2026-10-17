"""Typed, validated data structures for Aegis PII findings."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import TypeVar


_CollectionItem = TypeVar("_CollectionItem")


class FindingCategory(str, Enum):
    """Supported finding categories; PERSON_NAME is provisional for Risk Fusion."""

    EMAIL = "email"
    PHONE = "phone"
    PERSON_NAME = "person_name"


class PipelineStatus(str, Enum):
    """Overall completion state of a pipeline run."""

    OK = "ok"
    PARTIAL = "partial"
    ERROR = "error"


class ValuePolicy(str, Enum):
    """Controls which representation(s) a detector places in a finding."""

    RAW = "raw"
    REDACTED = "redacted"
    BOTH = "both"
    NEITHER = "neither"


@dataclass(frozen=True, slots=True)
class TextSpan:
    """Zero-based Python string character offsets, with an end-exclusive end."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or isinstance(self.end, bool):
            raise TypeError("span offsets must be integers")
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("span offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("span must be non-empty and satisfy 0 <= start < end")


@dataclass(frozen=True, slots=True)
class ImageRegion:
    """Polygon in source-image pixel coordinates, represented as (x, y) points."""

    polygon: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.polygon, (list, tuple)):
            raise TypeError("image region polygon must be a list or tuple of points")
        normalized_points: list[tuple[float, float]] = []
        for point in self.polygon:
            if not isinstance(point, (list, tuple)):
                raise TypeError("each image-region point must be a list or tuple")
            if len(point) != 2:
                raise ValueError("each image-region point must contain x and y")
            normalized_points.append(tuple(point))
        if len(normalized_points) < 3:
            raise ValueError("image region polygon must contain at least three points")
        for point in normalized_points:
            if any(isinstance(coord, bool) or not isinstance(coord, Real) for coord in point):
                raise TypeError("image-region coordinates must be real numbers")
            if any(not math.isfinite(coord) or coord < 0 for coord in point):
                raise ValueError("image-region coordinates must be finite and non-negative")
        object.__setattr__(self, "polygon", tuple(normalized_points))


@dataclass(frozen=True, slots=True)
class Finding:
    """A detected entity with separate OCR and entity-detector score fields."""

    category: FindingCategory
    detector: str
    span: TextSpan
    value: str | None = None
    redacted_value: str | None = None
    image_region: ImageRegion | None = None
    ocr_confidence: float | None = None
    detector_score: float | None = None
    detector_score_semantics: str | None = None
    source_segment_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.category, FindingCategory):
            raise TypeError("category must be a FindingCategory")
        if not isinstance(self.detector, str) or not self.detector.strip():
            raise ValueError("detector must be a non-empty string")
        if not isinstance(self.span, TextSpan):
            raise TypeError("span must be a TextSpan")
        if self.value is not None and not isinstance(self.value, str):
            raise TypeError("value must be a string or None")
        if self.redacted_value is not None and not isinstance(self.redacted_value, str):
            raise TypeError("redacted_value must be a string or None")
        if self.image_region is not None and not isinstance(self.image_region, ImageRegion):
            raise TypeError("image_region must be an ImageRegion or None")
        _validate_score("ocr_confidence", self.ocr_confidence)
        _validate_score("detector_score", self.detector_score)
        if (self.detector_score is None) != (self.detector_score_semantics is None):
            raise ValueError("detector_score and detector_score_semantics must be supplied together")
        if self.detector_score_semantics is not None and not self.detector_score_semantics.strip():
            raise ValueError("detector_score_semantics must not be empty")
        if self.source_segment_index is not None:
            if isinstance(self.source_segment_index, bool) or not isinstance(self.source_segment_index, int):
                raise TypeError("source_segment_index must be a non-negative integer or None")
            if self.source_segment_index < 0:
                raise ValueError("source_segment_index must be non-negative")


@dataclass(frozen=True, slots=True)
class PipelineError:
    """Safe, structured error metadata; messages must not include detected values."""

    code: str
    message: str
    stage: str

    def __post_init__(self) -> None:
        for name, value in (("code", self.code), ("message", self.message), ("stage", self.stage)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Versioned findings and errors for one input/pipeline run."""

    schema_version: str
    status: PipelineStatus
    findings: tuple[Finding, ...] = ()
    errors: tuple[PipelineError, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "findings", _immutable_collection("findings", self.findings))
        object.__setattr__(self, "errors", _immutable_collection("errors", self.errors))
        if not isinstance(self.schema_version, str) or not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string")
        if not isinstance(self.status, PipelineStatus):
            raise TypeError("status must be a PipelineStatus")
        if any(not isinstance(finding, Finding) for finding in self.findings):
            raise TypeError("findings must contain only Finding instances")
        if any(not isinstance(error, PipelineError) for error in self.errors):
            raise TypeError("errors must contain only PipelineError instances")
        seen: set[tuple[FindingCategory, TextSpan, int | None]] = set()
        for finding in self.findings:
            key = (finding.category, finding.span, finding.source_segment_index)
            if key in seen:
                raise ValueError("duplicate finding for category, text span, and source segment")
            seen.add(key)


def _immutable_collection(
    name: str,
    value: list[_CollectionItem] | tuple[_CollectionItem, ...],
) -> tuple[_CollectionItem, ...]:
    """Normalize supported collection inputs without retaining caller lists."""

    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    return tuple(value)


def apply_value_policy(
    value: str,
    redacted_value: str,
    policy: ValuePolicy,
) -> tuple[str | None, str | None]:
    """Return raw/redacted fields according to policy without logging either."""

    if not isinstance(policy, ValuePolicy):
        raise TypeError("policy must be a ValuePolicy")
    return (
        value if policy in (ValuePolicy.RAW, ValuePolicy.BOTH) else None,
        redacted_value if policy in (ValuePolicy.REDACTED, ValuePolicy.BOTH) else None,
    )


def _validate_score(name: str, score: float | None) -> None:
    if score is None:
        return
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be finite and between 0.0 and 1.0")
