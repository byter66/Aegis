# Aegis OCR + PII/NER module

This standalone Python package provides a findings schema, email and phone detectors, a single-image EasyOCR adapter, an English person-name NER adapter, and an OCR-to-findings pipeline. Risk Fusion integration is not implemented.

## Setup

From the repository root, enter this module and install it with its configured `dev` extra (which includes pytest):

```powershell
Set-Location .\ocr-pii-ner
python -m pip install -e ".[dev]"
```

The package declares `easyocr==1.7.2` and `spacy==3.8.16`. The pretrained English spaCy model is a separate requirement; installing the package does not install that model. Install it in the same environment and verify that it loads:

```powershell
python -m pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print(nlp.meta['version'])"
```

EasyOCR may fetch its pretrained OCR weights when the reader is first initialized; package installation does not guarantee that weights can be downloaded or are available locally. **Real EasyOCR inference has not yet been verified: the EasyOCR model cache was empty and the required model weights were unavailable locally.** Mocked OCR tests do not prove that real image recognition works.

## Public API example

`analyze_image(image_path, *, ocr=None, email_detector=detect_emails, phone_detector=detect_phones, ner=None, value_policy=ValuePolicy.REDACTED)` accepts one local JPEG (`.jpg`/`.jpeg`) or PNG (`.png`) image path. The defaults construct an EasyOCR engine and spaCy NER adapter; OCR and NER can also be injected. This example prompts for the path of an existing supported image and inspects the returned status and findings without assuming any will be found:

```python
from pathlib import Path

from aegis_ocr import ValuePolicy, analyze_image

image_path = Path(input("Path to an existing JPEG or PNG image: ").strip())
result = analyze_image(image_path, value_policy=ValuePolicy.REDACTED)

print(f"Status: {result.status.value}")
for finding in result.findings:
    print(
        finding.category.value,
        finding.detector,
        (finding.span.start, finding.span.end),
        finding.source_segment_index,
    )
    # Do not print raw or redacted values in logs or shared output.

if result.errors:
    for error in result.errors:
        print(error.code, error.stage, error.message)

# To opt in to raw values, pass value_policy=ValuePolicy.RAW.
# Findings then carry raw text in `value` and None in `redacted_value`.
# REDACTED keeps only `redacted_value`; BOTH keeps both; NEITHER keeps neither.
```

The default `REDACTED` policy exposes detector-provided redacted representations in `finding.redacted_value` and clears `finding.value`. The other policies control those same two fields. Do not select `RAW` or `BOTH` unless the caller is prepared to protect the raw values. The email and phone detectors use `[EMAIL]` and `[PHONE]` as their redaction markers; person-name findings use the representation provided by the NER adapter. Findings are returned as typed objects; this package does not define a JSON serialization or Risk Fusion payload.

## Pipeline behavior

`analyze_image` runs email, phone, and person-name detection independently on each OCR segment. It does not join segment text. A span uses zero-based Python string character offsets with an end-exclusive end, relative to its source segment. Each finding carries `source_segment_index`, the segment's image polygon, and the OCR engine's score in `ocr_confidence`. Image polygons use source-image pixel coordinates. OCR confidence is kept separate from `detector_score`; the current detectors do not provide calibrated detector scores.

The pipeline uses these statuses:

- `ok`: OCR completed successfully and all detector calls completed without reported failures. A successful image with no recognized segments, or with no findings, can still be `ok`.
- `partial`: OCR reported a partial result, or detector calls reported failures while usable downstream results remain. Safe structured errors accompany reported failures.
- `error`: OCR failed, or downstream detector calls failed with no usable detector results remaining. The pipeline returns no findings for this downstream all-failed case.

Errors use structured codes, stages, and generic messages; raw OCR text, detected values, and exception details should not appear in them. These statuses describe this module's processing outcome only and do not define downstream Risk Fusion interpretation. An entity split between OCR segments may be missed. The `person_name` finding category is provisional pending Risk Fusion agreement; the downstream contract is not implemented.

## Findings and categories

The supported categories are `email`, `phone`, and provisional `person_name`. The schema carries a detector name, optional raw and redacted values, a text span, optional image region, optional `ocr_confidence`, and optional `detector_score` with required semantics when supplied. Regex matches do not receive invented confidence scores. `PipelineResult` contains a schema version, status, findings, and structured errors; it rejects duplicate findings with the same category, span, and source-segment index.

The schema and detectors use typed dataclasses and standard-library validation. The package supports Python 3.10 and later; the repository environment has Python 3.14.4. `EasyOCREngine` uses EasyOCR's English reader in CPU mode, initializes it lazily, and reuses it. It is not thread-safe in this MVP; use one engine per worker or synchronize access externally.

## Detector limitations

- Email detection covers common unquoted ASCII email forms with basic syntax checks. It is not a complete implementation of email address standards and does not verify mailbox existence.
- Phone detection accepts only explicit international-format candidates with a `+` prefix and 8–15 digits overall. It does not validate country numbering plans or assigned numbers, and it does not guess local/national formats.
- spaCy NER uses the pretrained `en_core_web_sm` pipeline and emits person-name entities only. The model must be installed separately as shown above.

## Tests

From `ocr-pii-ner/`, the following PowerShell commands use the repository virtual environment, disable bytecode and pytest cache writes, and give pytest a fresh external temporary directory. The GUID makes the chosen path unique; the guard stops if it already exists. This does not reuse, clear, or write to the module's `.pytest-tmp` directory.

```powershell
$python = Resolve-Path ..\.venv\Scripts\python.exe
$testBase = Join-Path $env:TEMP ("aegis-ocr-pii-tests-" + [guid]::NewGuid().ToString("N"))
if (Test-Path $testBase) { throw "Temporary path already exists: $testBase" }
$env:PYTHONDONTWRITEBYTECODE = "1"
& $python -B -m pytest -q -p no:cacheprovider --basetemp $testBase
if ($LASTEXITCODE -ne 0) { throw "pytest failed with exit code $LASTEXITCODE" }
```

The OCR tests use mocks and do not download models or perform real OCR inference. This command documents test execution; it does not claim that the suite has been run.
