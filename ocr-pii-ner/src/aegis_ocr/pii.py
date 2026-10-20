"""Conservative structured PII candidate detectors for plain text."""

from __future__ import annotations

import re

from .schema import (
    Finding,
    FindingCategory,
    TextSpan,
    ValuePolicy,
    apply_value_policy,
)


_EMAIL_CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
    r"@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
    r"(?![A-Za-z0-9-])"
)
_EMAIL_LOCAL_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.!#$%&'*+/=?^_`{|}~-")

_PHONE_CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9+])"
    r"\+(?:[0-9][ .()-]*){7,14}[0-9]"
    r"(?![A-Za-z0-9])"
)

_STREET_ADDRESS = re.compile(
    r"(?<![A-Za-z0-9])\d{1,6}\s+"
    r"(?:[A-Za-z0-9][A-Za-z0-9.'-]*\s+){1,4}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|"
    r"Court|Ct|Circle|Cir|Way|Terrace|Ter|Place|Pl|Highway|Hwy)\b"
    r"(?:\s+(?:Apt|Apartment|Unit|Suite|Ste|#)\s*[A-Za-z0-9-]+)?",
    re.IGNORECASE,
)
_LABELLED_STREET_ADDRESS = re.compile(
    r"\b(?:street\s+address|mailing\s+address|address)\s*:\s*"
    r"(?P<candidate>\d{1,6}\s+"
    r"(?:[A-Za-z0-9][A-Za-z0-9.'-]*\s+){1,4}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|"
    r"Court|Ct|Circle|Cir|Way|Terrace|Ter|Place|Pl|Highway|Hwy)\b"
    r"(?:\s+(?:Apt|Apartment|Unit|Suite|Ste|#)\s*[A-Za-z0-9-]+)?)",
    re.IGNORECASE,
)
_LABELLED_LOCALITY = re.compile(
    r"\b(?:city|locality|town)\s*:\s*"
    r"(?P<candidate>[A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,3})",
    re.IGNORECASE,
)
_LABELLED_POSTAL_CODE = re.compile(
    r"\b(?:ZIP(?:\s+code)?|postal(?:\s+code)?|PIN(?:\s+code)?)\s*[:#-]\s*"
    r"(?P<candidate>(?:\d{5}(?:-\d{4})?|\d{6}))\b",
    re.IGNORECASE,
)


def detect_emails(
    text: str,
    *,
    value_policy: ValuePolicy = ValuePolicy.REDACTED,
) -> list[Finding]:
    """Find common unquoted email forms with basic, non-RFC-complete checks."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    findings: list[Finding] = []
    seen: set[tuple[int, int]] = set()
    for match in _EMAIL_CANDIDATE.finditer(text):
        start, end = match.span()
        candidate = match.group(0)
        local, domain = candidate.rsplit("@", 1)
        if (
            len(candidate) > 254
            or len(local) > 64
            or local.startswith(".")
            or local.endswith(".")
            or ".." in local
            or any(char not in _EMAIL_LOCAL_CHARS for char in local)
            or any(not label or label.startswith("-") or label.endswith("-") for label in domain.split("."))
        ):
            continue
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        value, redacted = apply_value_policy(candidate, "[EMAIL]", value_policy)
        findings.append(
            Finding(
                category=FindingCategory.EMAIL,
                detector="email_regex",
                span=TextSpan(start, end),
                value=value,
                redacted_value=redacted,
            )
        )
    return findings


def detect_phones(
    text: str,
    *,
    value_policy: ValuePolicy = ValuePolicy.REDACTED,
) -> list[Finding]:
    """Find explicit +prefixed international-format candidates, not validated numbers.

    Requires 8–15 digits and rejects candidates containing no variation in digits.
    No default region or numbering-plan validity is assumed.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    findings: list[Finding] = []
    seen: set[tuple[int, int]] = set()
    for match in _PHONE_CANDIDATE.finditer(text):
        candidate = match.group(0)
        digits = "".join(char for char in candidate if char.isdigit())
        if not 8 <= len(digits) <= 15 or len(set(digits)) < 2:
            continue
        start, end = match.span()
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        value, redacted = apply_value_policy(candidate, "[PHONE]", value_policy)
        findings.append(
            Finding(
                category=FindingCategory.PHONE,
                detector="phone_international_candidate",
                span=TextSpan(start, end),
                value=value,
                redacted_value=redacted,
            )
        )
    return findings


def detect_addresses(
    text: str,
    *,
    value_policy: ValuePolicy = ValuePolicy.REDACTED,
) -> list[Finding]:
    """Find conservative address candidates, not validated postal addresses.

    Candidates are numbered street addresses with a recognized street suffix,
    explicitly labelled localities, or explicitly labelled postal/PIN codes.
    Unlabelled numbers and place names are intentionally not sufficient.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(value_policy, ValuePolicy):
        raise TypeError("value_policy must be a ValuePolicy")

    candidates: list[tuple[int, int]] = []
    for pattern in (
        _STREET_ADDRESS,
        _LABELLED_STREET_ADDRESS,
        _LABELLED_LOCALITY,
        _LABELLED_POSTAL_CODE,
    ):
        for match in pattern.finditer(text):
            if "candidate" in match.re.groupindex:
                start, end = match.span("candidate")
            else:
                start, end = match.span()
            if start < end:
                candidates.append((start, end))

    # A labelled street match and the unlabelled street pattern can identify
    # the same text. Coalesce overlaps so one candidate produces one finding.
    merged: list[tuple[int, int]] = []
    for start, end in sorted(candidates):
        if merged and start < merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))

    findings: list[Finding] = []
    for start, end in merged:
        candidate = text[start:end]
        value, redacted = apply_value_policy(candidate, "[ADDRESS]", value_policy)
        findings.append(
            Finding(
                category=FindingCategory.ADDRESS,
                detector="address_candidate_regex",
                span=TextSpan(start, end),
                value=value,
                redacted_value=redacted,
            )
        )
    return findings
