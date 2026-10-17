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
