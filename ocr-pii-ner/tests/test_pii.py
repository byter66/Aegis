from __future__ import annotations

from aegis_ocr.pii import detect_emails, detect_phones
from aegis_ocr.schema import FindingCategory, ValuePolicy


def test_email_positive_and_exact_span() -> None:
    text = "Contact fake.person+demo@example.invalid today."
    (finding,) = detect_emails(text)
    assert text[finding.span.start:finding.span.end] == "fake.person+demo@example.invalid"
    assert finding.category is FindingCategory.EMAIL
    assert finding.value is None
    assert finding.redacted_value == "[EMAIL]"


def test_email_rejects_common_malformed_candidates() -> None:
    for text in (
        "missing-at.example.invalid",
        "@example.invalid",
        "fake..person@example.invalid",
        "fake@-example.invalid",
        "fake@example",
    ):
        assert detect_emails(text) == []


def test_multiple_emails_are_ordered_deterministic_and_unique() -> None:
    text = "fake1@example.invalid, fake2@example.invalid"
    first = detect_emails(text)
    second = detect_emails(text)
    assert first == second
    assert [text[item.span.start:item.span.end] for item in first] == [
        "fake1@example.invalid",
        "fake2@example.invalid",
    ]
    assert len({(item.category, item.span) for item in first}) == len(first)


def test_phone_accepts_explicit_international_candidate_only() -> None:
    text = "Call the fictional number +1 202-555-0100."
    (finding,) = detect_phones(text)
    assert text[finding.span.start:finding.span.end] == "+1 202-555-0100"
    assert finding.category is FindingCategory.PHONE
    assert finding.detector == "phone_international_candidate"


def test_phone_rejects_unqualified_arbitrary_or_implausible_digits() -> None:
    for text in (
        "555-0100",
        "2025550100",
        "+1 111-111-1111",
        "+12 123",
        "version 123456789",
    ):
        assert detect_phones(text) == []


def test_phone_multiple_candidates_have_stable_spans() -> None:
    text = "+1 202-555-0100; +44 20 7946 0958"
    first = detect_phones(text)
    assert first == detect_phones(text)
    assert [text[item.span.start:item.span.end] for item in first] == [
        "+1 202-555-0100",
        "+44 20 7946 0958",
    ]
    assert len({(item.category, item.span) for item in first}) == len(first)


def test_empty_and_no_findings() -> None:
    assert detect_emails("") == []
    assert detect_phones("") == []
    assert detect_emails("nothing identifiable") == []
    assert detect_phones("nothing identifiable") == []


def test_raw_value_is_opt_in_and_detector_does_not_log(capsys) -> None:
    sensitive = "fake.person@example.invalid"
    default_finding = detect_emails(sensitive)[0]
    assert default_finding.value is None
    assert default_finding.redacted_value == "[EMAIL]"

    raw_finding = detect_emails(sensitive, value_policy=ValuePolicy.BOTH)[0]
    assert raw_finding.value == sensitive
    assert raw_finding.redacted_value == "[EMAIL]"
    captured = capsys.readouterr()
    assert sensitive not in captured.out
    assert sensitive not in captured.err
