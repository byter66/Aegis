"""Fail-closed image redaction for OCR/PII findings.

The pipeline maps each finding to its OCR segment polygon (or, for grouped
phone findings, to the union of contributing segments). Redaction therefore
covers the complete segment region conservatively rather than attempting to
infer character-level glyph boxes from text offsets.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from .schema import PipelineResult, PipelineStatus


class RedactionError(ValueError):
    """Raised when a result cannot be safely converted into a redacted image."""


def redact_image(image: Image.Image, result: PipelineResult) -> Image.Image:
    """Return a redacted RGB copy, or fail closed when analysis is incomplete.

    Every finding's complete axis-aligned OCR region is covered with an opaque
    fill and a small safety margin. The caller must use this returned image for
    both review and export. This function does not establish that an image with
    no findings is free of undetected PII.
    """

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a Pillow image")
    if not isinstance(result, PipelineResult):
        raise TypeError("result must be a PipelineResult")
    if result.status is not PipelineStatus.OK or result.errors:
        raise RedactionError("Incomplete analysis cannot produce a postable redacted image.")

    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    for finding in result.findings:
        region = finding.image_region
        if region is None:
            raise RedactionError("A finding has no reliable image region.")

        points = region.polygon
        if any(
            not math.isfinite(float(coord))
            for point in points
            for coord in point
        ):
            raise RedactionError("A finding has invalid image coordinates.")

        left = min(float(point[0]) for point in points)
        top = min(float(point[1]) for point in points)
        right = max(float(point[0]) for point in points)
        bottom = max(float(point[1]) for point in points)
        if right <= left or bottom <= top:
            raise RedactionError("A finding has an empty image region.")

        # The source polygons are pixel coordinates. Expand around the segment
        # box to cover antialiasing and tight OCR polygons, then clip safely.
        margin = max(2.0, (bottom - top) * 0.08)
        padded_left, padded_top = left - margin, top - margin
        padded_right, padded_bottom = right + margin, bottom + margin
        if (
            padded_right < 0
            or padded_bottom < 0
            or padded_left >= width
            or padded_top >= height
        ):
            raise RedactionError("A finding is outside the image bounds.")
        x0 = max(0, min(width - 1, math.floor(padded_left)))
        y0 = max(0, min(height - 1, math.floor(padded_top)))
        x1 = max(0, min(width - 1, math.ceil(padded_right)))
        y1 = max(0, min(height - 1, math.ceil(padded_bottom)))
        if x1 < x0 or y1 < y0:
            raise RedactionError("A finding is outside the image bounds.")
        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))

    return output
