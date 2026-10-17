"""Aegis OCR + PII/NER schema and initial structured PII detectors."""

from .pii import detect_emails, detect_phones
from .ocr import EasyOCREngine, OCRResult, OCRSegment
from .ner import SpacyNERAdapter
from .pipeline import analyze_image
from .risk_fusion import to_risk_fusion_ocr_payload
from .schema import (
    Finding,
    FindingCategory,
    ImageRegion,
    PipelineError,
    PipelineResult,
    PipelineStatus,
    TextSpan,
    ValuePolicy,
)

__all__ = [
    "Finding",
    "FindingCategory",
    "EasyOCREngine",
    "ImageRegion",
    "PipelineError",
    "PipelineResult",
    "PipelineStatus",
    "SpacyNERAdapter",
    "analyze_image",
    "to_risk_fusion_ocr_payload",
    "OCRResult",
    "OCRSegment",
    "TextSpan",
    "ValuePolicy",
    "detect_emails",
    "detect_phones",
]
