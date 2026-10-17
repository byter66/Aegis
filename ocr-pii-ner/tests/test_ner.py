from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from aegis_ocr.ner import SpacyNERAdapter
from aegis_ocr.schema import FindingCategory, PipelineStatus, ValuePolicy


def _entity(label: str, start: int, end: int) -> SimpleNamespace:
    return SimpleNamespace(label_=label, start_char=start, end_char=end)


def test_person_entities_keep_exact_offsets_and_default_redaction() -> None:
    text = "Contact Ada Lovelace at the office."
    start, end = text.index("Ada"), text.index(" at")
    nlp = Mock(return_value=SimpleNamespace(ents=(_entity("PERSON", start, end),)))

    result = SpacyNERAdapter(nlp=nlp).extract(text)

    assert result.status is PipelineStatus.OK
    (finding,) = result.findings
    assert finding.category is FindingCategory.PERSON_NAME
    assert finding.detector == "spacy_ner"
    assert (finding.span.start, finding.span.end) == (start, end)
    assert text[finding.span.start:finding.span.end] == "Ada Lovelace"
    assert finding.value is None
    assert finding.redacted_value == "[PERSON_NAME]"


def test_unsupported_labels_are_ignored() -> None:
    text = "Ada works at Acme in London."
    nlp = Mock(
        return_value=SimpleNamespace(
            ents=(
                _entity("ORG", text.index("Acme"), text.index("Acme") + 4),
                _entity("GPE", text.index("London"), text.index("London") + 6),
                _entity("PERSON", 0, 3),
                _entity("DATE", 0, 3),
            )
        )
    )

    result = SpacyNERAdapter(nlp=nlp).extract(text)

    assert [(finding.category, text[finding.span.start:finding.span.end]) for finding in result.findings] == [
        (FindingCategory.PERSON_NAME, "Ada")
    ]


@pytest.mark.parametrize("text", ["", " \t\n "])
def test_empty_or_whitespace_input_succeeds_without_loading_model(text: str) -> None:
    adapter = SpacyNERAdapter()

    with patch("aegis_ocr.ner.importlib.import_module") as importer:
        result = adapter.extract(text)

    assert result.status is PipelineStatus.OK
    assert result.findings == ()
    assert result.errors == ()
    importer.assert_not_called()


def test_missing_model_returns_safe_structured_error() -> None:
    spacy = SimpleNamespace(load=Mock(side_effect=OSError("C:/private/model path")))
    with patch("aegis_ocr.ner.importlib.import_module", return_value=spacy):
        result = SpacyNERAdapter().extract("Fictional Person")

    assert result.status is PipelineStatus.ERROR
    assert result.findings == ()
    assert result.errors[0].code == "ner_model_unavailable"
    assert result.errors[0].stage == "ner_model"
    assert "private" not in result.errors[0].message


def test_missing_spacy_package_returns_safe_structured_error() -> None:
    with patch(
        "aegis_ocr.ner.importlib.import_module",
        side_effect=ModuleNotFoundError("spacy is unavailable at C:/private"),
    ):
        result = SpacyNERAdapter().extract("Fictional Person")

    assert result.status is PipelineStatus.ERROR
    assert result.findings == ()
    assert result.errors[0].code == "ner_model_unavailable"
    assert result.errors[0].stage == "ner_model"
    assert "private" not in result.errors[0].message


def test_inference_error_does_not_expose_input_or_exception() -> None:
    sensitive = "Fictional Person"
    nlp = Mock(side_effect=RuntimeError(f"failed on {sensitive}"))

    result = SpacyNERAdapter(nlp=nlp).extract(sensitive)

    assert result.status is PipelineStatus.ERROR
    assert result.errors[0].code == "ner_failed"
    assert result.errors[0].stage == "ner"
    assert sensitive not in result.errors[0].message


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (True, 3),
        (0, False),
        ("0", 3),
        (0, 3.0),
        (-1, 3),
        (3, 3),
        (0, 4),
    ],
)
def test_malformed_person_offsets_return_safe_error(start: object, end: object) -> None:
    sensitive = "Ada"
    nlp = Mock(return_value=SimpleNamespace(ents=(_entity("PERSON", start, end),)))

    result = SpacyNERAdapter(nlp=nlp).extract(sensitive)

    assert result.status is PipelineStatus.ERROR
    assert result.findings == ()
    assert result.errors[0].code == "ner_failed"
    assert result.errors[0].stage == "ner"
    assert sensitive not in result.errors[0].message


def test_findings_do_not_set_either_confidence_field() -> None:
    nlp = Mock(return_value=SimpleNamespace(ents=(_entity("PERSON", 0, 3),)))

    (finding,) = SpacyNERAdapter(nlp=nlp).extract("Ada").findings

    assert finding.detector_score is None
    assert finding.detector_score_semantics is None
    assert finding.ocr_confidence is None


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ValuePolicy.RAW, ("Ada", None)),
        (ValuePolicy.REDACTED, (None, "[PERSON_NAME]")),
        (ValuePolicy.BOTH, ("Ada", "[PERSON_NAME]")),
        (ValuePolicy.NEITHER, (None, None)),
    ],
)
def test_existing_value_policy_is_respected(
    policy: ValuePolicy, expected: tuple[str | None, str | None]
) -> None:
    nlp = Mock(return_value=SimpleNamespace(ents=(_entity("PERSON", 0, 3),)))

    (finding,) = SpacyNERAdapter(nlp=nlp, value_policy=policy).extract("Ada").findings

    assert (finding.value, finding.redacted_value) == expected


def test_default_model_pipeline_is_loaded_once_and_reused() -> None:
    text = "Ada"
    nlp = Mock(return_value=SimpleNamespace(ents=(_entity("PERSON", 0, 3),)))
    load = Mock(return_value=nlp)
    spacy = SimpleNamespace(load=load)

    with patch("aegis_ocr.ner.importlib.import_module", return_value=spacy) as importer:
        adapter = SpacyNERAdapter()
        first = adapter.extract(text)
        second = adapter.extract(text)

    assert first == second
    importer.assert_called_once_with("spacy")
    load.assert_called_once_with("en_core_web_sm")
    assert nlp.call_count == 2
