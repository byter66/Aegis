# Aegis OCR/PII labelled benchmark

This directory contains a fixed synthetic development snapshot for the current `analyze_image` interface and `person_name`, `email`, and `phone` findings. It does not change the production pipeline or Risk Fusion schema. The benchmark uses `ValuePolicy.RAW` only in process memory to compare predictions with labels. Persisted reports contain aggregates, timings, counts, versions, safe error codes, and the caller-supplied source revision; they never contain prediction values or complete finding objects. A startup contract check verifies that Risk Fusion redacted serialization does not fall back to an available raw value.

## Fixed development snapshot

The committed snapshot contains 28 synthetic PNG images under `data/images/development/` and their labels in `data/annotations/development.jsonl`:

| Name only | Email only | Phone only | Mixed (all three) | Negative | Total |
|---:|---:|---:|---:|---:|---:|
| 6 | 6 | 6 | 4 | 6 | 28 |

All identities and contact details in this snapshot are synthetic. The images vary text placement, font size, contrast, mild background texture, and dimensions; some phone text is split across nearby regions. Negative images contain ordinary numeric/reference-like text but no labelled target PII. A split phone entity has one annotation and a union primary box; `fragment_boxes_xyxy` records the rendered pieces. Coordinates use `[x_min, y_min, x_max, y_max]` form and are checked against image dimensions.

The snapshot is ready for development evaluation as committed. Regenerating images or validating the complete dataset requires additional local-only benchmark tooling, a manifest, and data that are intentionally not included in this commit; those workflows are not available from this snapshot alone.

## Scoring

Names are normalized using Unicode NFKC, case folding, trimming, and internal whitespace collapse. Emails use NFKC, trimming, and case folding. Phones use NFKC and retain ASCII digits plus a leading `+`, ignoring formatting punctuation. A prediction is a strict true positive only when category and normalized value match and its axis-aligned polygon bounds have IoU at least 0.5 with the labelled primary box. Predictions and labels are matched one-to-one using deterministic maximum-cardinality bipartite matching. A duplicate prediction can therefore add a false positive but cannot inflate true positives.

Strict per-category precision, recall, and F1 are reported, along with micro-averaged overall counts/metrics and macro F1 over the three categories. Localization-only scores use the same category and IoU rule without requiring value equality. Value exactness reports the fraction of localized category/box matches whose normalized values are exact; transcription errors are separately counted and are a strict false positive plus false negative. Unmatched expected entities are false negatives. Unmatched predictions are false positives. An image with no expected entities contributes to the image-level negative false-positive rate if it has one or more findings. Zero-denominator metrics are reported as 0.

`PARTIAL` findings remain eligible for scoring, and partial/error rates are reported separately. `ERROR` runs contribute no predictions; all labels for those images remain false negatives. Per-image output includes status, elapsed time, dimensions, segment/finding counts, category counts, and sanitized error codes/stages. Cold EasyOCR and spaCy initialization is timed separately from per-image elapsed time. The report records the supplied code revision, tile configuration, Python/library versions, and spaCy model version when available. EasyOCR is initialized with downloads disabled, so required model weights must already exist locally.

## Running the evaluator

Run from the repository root with the module's existing virtual environment. Pass a source revision string explicitly; the script does not invoke Git:

```powershell
& .\.venv\Scripts\python.exe -B ocr-pii-ner\benchmark\evaluate.py --code-revision REPLACE_WITH_COMMIT_HASH --split development --tile-size 1600 --tile-overlap 128
```

Replace `REPLACE_WITH_COMMIT_HASH` with the exact source revision being evaluated. A sanitized timestamped JSON report is written under `ocr-pii-ner/benchmark/results/`. The evaluator processes each selected image once and reuses one OCR reader and one spaCy adapter. It does not store raw predictions, values, text, or exception messages. The local-only manifest is not required to evaluate this fixed development snapshot.

The benchmark unit tests and full module tests can be run with bytecode/cache disabled and a fresh pytest temporary directory, for example:

```powershell
& .\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp $env:TEMP\aegis-benchmark-test ocr-pii-ner\benchmark\tests
& .\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider --basetemp $env:TEMP\aegis-ocr-test ocr-pii-ner\tests
```

Choose fresh writable basetemp paths for each run and remove only those paths afterward.

## Limits and next validation

The 28-image synthetic development snapshot is a manageable engineering pilot, not evidence of real-world accuracy. Clean digital fonts do not capture camera blur, perspective, glare, compression, handwriting, diverse document layouts, language variation, unusual typography, or OCR hallucinations. The controlled phone splits test one segmentation pattern only. After this pilot, validate with a separate, appropriately consented real-image set that is manually labelled, access-controlled, and reviewed for privacy. Report category metrics and failure cases without retaining or publishing raw PII. Do not infer that zero findings means an image is safe to post.

## Opt-in OCR stage timings

For a focused diagnosis, construct `EasyOCREngine(..., diagnostic_timing=True)` and enable INFO logging for `aegis_ocr.ocr`. The engine emits image validation/decode and reader initialization or reuse durations, tile-plan dimensions/count, and per-tile preprocessing, EasyOCR inference, result-processing, and total durations with tile index/dimensions and success/failure. Deduplication duration and segment counts are also reported. Events contain no image paths, recognized text, PII, exception messages, or image data. The option is disabled by default; it measures stages but does not add a timeout or interrupt native OCR inference.
