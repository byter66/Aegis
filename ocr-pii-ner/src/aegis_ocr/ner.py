"""Pretrained spaCy English person-name entity extraction."""

from __future__ import annotations

from collections.abc import Callable
import importlib
from typing import Protocol

from .schema import (
    Finding,
    FindingCategory,
    PipelineError,
    PipelineResult,
    PipelineStatus,
    TextSpan,
    ValuePolicy,
    apply_value_policy,
)


_MODEL_NAME = "en_core_web_sm"
_SCHEMA_VERSION = "1.0"
_REDACTION = "[PERSON_NAME]"


class _Entity(Protocol):
    label_: str
    start_char: int
    end_char: int


class _Document(Protocol):
    @property
    def ents(self) -> tuple[_Entity, ...]: ...


class _NLP(Protocol):
    def __call__(self, text: str) -> _Document: ...


class SpacyNERAdapter:
    """Extract English PERSON entities into provisional person-name findings.

    The default value policy is the same as the existing PII detectors:
    redact matched values. Pass an NLP callable to test without spaCy or its
    model. The default model is loaded on the first non-empty input and reused.
    This adapter does not attach OCR confidence or estimate NER confidence.
    Calls are not synchronized; use one adapter per worker or synchronize
    access externally.
    """

    def __init__(
        self,
        *,
        nlp: Callable[[str], _Document] | None = None,
        value_policy: ValuePolicy = ValuePolicy.REDACTED,
    ) -> None:
        if not isinstance(value_policy, ValuePolicy):
            raise TypeError("value_policy must be a ValuePolicy")
        if nlp is not None and not callable(nlp):
            raise TypeError("nlp must be callable or None")
        self._nlp = nlp
        self._value_policy = value_policy

    def extract(
        self,
        text: str,
        *,
        value_policy: ValuePolicy | None = None,
    ) -> PipelineResult:
        """Return findings for PERSON entities in text, with safe errors."""

        if not isinstance(text, str):
            return _error_result(
                "invalid_text",
                "NER input must be a string.",
                stage="input_validation",
            )
        if not text.strip():
            return PipelineResult(_SCHEMA_VERSION, PipelineStatus.OK)
        effective_policy = self._value_policy if value_policy is None else value_policy
        if not isinstance(effective_policy, ValuePolicy):
            return _error_result(
                "invalid_value_policy",
                "A valid value policy is required.",
                stage="input_validation",
            )

        try:
            nlp = self._get_nlp()
        except Exception:
            # Loader exceptions may contain local paths or other environment details.
            return _error_result(
                "ner_model_unavailable",
                "The English NER model is unavailable.",
                stage="ner_model",
            )

        try:
            document = nlp(text)
            findings: list[Finding] = []
            seen: set[tuple[int, int]] = set()
            for entity in document.ents:
                if entity.label_ != "PERSON":
                    continue
                start, end = entity.start_char, entity.end_char
                if (
                    isinstance(start, bool)
                    or isinstance(end, bool)
                    or not isinstance(start, int)
                    or not isinstance(end, int)
                    or not 0 <= start < end <= len(text)
                ):
                    raise ValueError("NER entity offsets are invalid")
                key = (start, end)
                if key in seen:
                    continue
                seen.add(key)
                value, redacted = apply_value_policy(
                    text[start:end], _REDACTION, effective_policy
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
            return PipelineResult(
                _SCHEMA_VERSION,
                PipelineStatus.OK,
                findings=tuple(findings),
            )
        except Exception:
            # Do not expose exception messages or the input text in result errors.
            return _error_result(
                "ner_failed",
                "The NER model could not process this text.",
                stage="ner",
            )

    def _get_nlp(self) -> Callable[[str], _Document]:
        if self._nlp is None:
            spacy = importlib.import_module("spacy")
            self._nlp = spacy.load(_MODEL_NAME)
        return self._nlp


def _error_result(code: str, message: str, *, stage: str) -> PipelineResult:
    return PipelineResult(
        _SCHEMA_VERSION,
        PipelineStatus.ERROR,
        errors=(PipelineError(code=code, message=message, stage=stage),),
    )
