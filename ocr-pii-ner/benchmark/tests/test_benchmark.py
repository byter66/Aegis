from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image
import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = BENCHMARK_ROOT.parent
sys.path.insert(0, str(BENCHMARK_ROOT))
sys.path.insert(0, str(MODULE_ROOT / "src"))

import common
import evaluate
from evaluate import _CountingOCR
from aegis_ocr.schema import FindingCategory, ImageRegion, PipelineStatus


def _entity(category="phone", value="+1 202-555-0100", box=(0, 0, 100, 40)):
    return common.EntityAnnotation(
        category=category,
        expected_value=value,
        normalized_value=common.normalize_value(category, value),
        bbox_xyxy=box,
    )


def _annotation(image_id="one", entities=(), split="development"):
    return common.ImageAnnotation(
        image_id=image_id,
        split=split,
        image_file=f"images/{split}/{image_id}.png",
        width=200,
        height=100,
        template_id=f"{split}-layout-01",
        entities=tuple(entities),
    )


def _prediction(value="+1 202 555 0100", box=(0, 0, 100, 40), category="phone"):
    return common.Prediction(category, value, box)


def _run(predictions=(), status="ok"):
    return {"one": {"status": status, "predictions": tuple(predictions)}}


def test_committed_development_snapshot_composition_and_files():
    annotations = common.load_annotations(
        BENCHMARK_ROOT / "data/annotations/development.jsonl",
        expected_split="development",
    )
    assert len(annotations) == 28
    category_counts = {
        category: sum(
            entity.category == category
            for record in annotations
            for entity in record.entities
        )
        for category in common.CATEGORIES
    }
    assert category_counts == {"person_name": 10, "email": 10, "phone": 10, "address": 0}
    assert sum(not record.entities for record in annotations) == 6

    for record in annotations:
        image_path = BENCHMARK_ROOT / "data" / record.image_file
        assert image_path.is_file()
        with Image.open(image_path) as image:
            assert image.size == (record.width, record.height)


def test_annotation_validation_checks_bounds_and_fragments(tmp_path):
    path = tmp_path / "annotations.jsonl"
    record = {
        "annotation_version": common.ANNOTATION_VERSION,
        "image_id": "synthetic-1",
        "split": "development",
        "image_file": "images/development/synthetic-1.png",
        "image_size": {"width": 100, "height": 60},
        "template_id": "development-layout-01",
        "entities": [{
            "category": "phone", "expected_value": "+1 202-555-0100",
            "expected_normalized_value": "+12025550100", "bbox_xyxy": [10, 10, 90, 40],
            "fragment_boxes_xyxy": [[10, 10, 40, 40], [45, 10, 90, 40]],
        }],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert common.load_annotations(path)[0].entities[0].fragment_boxes_xyxy
    record["entities"][0]["bbox_xyxy"] = [10, 10, 101, 40]
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        common.load_annotations(path)


@pytest.mark.parametrize(
    ("category", "raw", "normalized"),
    [
        ("person_name", "  Avery   Sample ", "avery sample"),
        ("email", "Synthetic-Dev-01@Example.Com", "synthetic-dev-01@example.com"),
        ("phone", "+1 (202) 555-0100", "+12025550100"),
        ("address", "  42 Main   Street, Apt 4B  ", "42 main street, apt 4b"),
    ],
)
def test_category_normalization(category, raw, normalized):
    assert common.normalize_value(category, raw) == normalized


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("Alex@Example.com", "alex@example.com"),
        ("  alex@example.com  ", "alex@example.com"),
    ],
)
def test_email_normalization_handles_case_and_outer_whitespace(raw, normalized):
    assert common.normalize_value("email", raw) == normalized


@pytest.mark.parametrize(
    "raw",
    ["alex @example.com", "alex@example,com"],
)
def test_email_normalization_preserves_internal_whitespace_and_punctuation(raw):
    assert common.normalize_value("email", raw) != common.normalize_value(
        "email", "alex@example.com"
    )


@pytest.mark.parametrize(
    ("expected", "predicted", "category"),
    [
        ("alex@example.com", "alex@example.com", "exact_match"),
        (" Alex@Example.com ", "alex@example.com", "case_or_outer_whitespace_only"),
        ("alex@example.com", "alexexample.com", "missing_at_symbol"),
        ("alex@example.com", "alex@examplecom", "missing_dot"),
        ("alex@example.com", "alex @example.com", "internal_whitespace_difference"),
        ("alex+tag@example.com", "alex-tag@example.com", "punctuation_difference"),
        ("alex@example.com", "alix@example.com", "character_difference"),
        ("alex@example.com", "alex@example.co", "truncation_like"),
        ("alex@example.com", "arix@example.com", "multiple_character_edits"),
        ("alex@example.com", "alexexamplecom", "other_difference"),
        ("alex@example.com", "avery@sample.org", "other_difference"),
        ("alex@example.com", "avery@sample.com", "other_difference"),
        ("alex+tag@example.com", "arix-tag@example.com", "other_difference"),
        ("", "alex@example.com", "other_difference"),
        ("not-an-email", "not-an-email", "exact_match"),
    ],
)
def test_email_mismatch_classifier_returns_only_safe_categories(
    expected, predicted, category
):
    result = common.classify_email_mismatch(expected, predicted)

    assert result == category
    assert common.classify_email_mismatch(expected, predicted) == result
    assert result in {
        "exact_match",
        "case_or_outer_whitespace_only",
        "missing_at_symbol",
        "missing_dot",
        "internal_whitespace_difference",
        "punctuation_difference",
        "character_difference",
        "truncation_like",
        "multiple_character_edits",
        "other_difference",
    }
    for value in (expected, predicted):
        if value:
            assert value not in result
            assert result not in value


def test_existing_specific_category_precedes_truncation_like():
    assert common.classify_email_mismatch("a@b.", "a@b") == "missing_dot"


def test_empty_email_mismatch_summary_initializes_every_category_to_zero():
    assert evaluate._email_mismatch_category_counts((), {}) == {
        category: 0 for category in common.EMAIL_MISMATCH_CATEGORIES
    }


def test_initialization_error_report_has_zero_filled_email_mismatch_summaries():
    annotations = (
        _annotation(
            "initialization-email",
            (_entity(category="email", value="synthetic@example.com"),),
        ),
        _annotation("initialization-negative"),
    )

    report = evaluate._sanitize_failed_initialization(
        annotations, RuntimeError("synthetic initialization failure")
    )

    assert report["run_status"] == "initialization_error"
    metrics = report["metrics"]
    zero_counts = {category: 0 for category in common.EMAIL_MISMATCH_CATEGORIES}
    assert metrics["email_mismatch_categories"] == zero_counts
    assert tuple(metrics["email_mismatch_categories"]) == common.EMAIL_MISMATCH_CATEGORIES
    assert all(type(count) is int for count in metrics["email_mismatch_categories"].values())
    per_image = {
        row["image_id"]: row["email_mismatch_categories"]
        for row in metrics["per_image"]
    }
    assert per_image == {
        "initialization-email": zero_counts,
        "initialization-negative": zero_counts,
    }
    assert {
        category: sum(counts[category] for counts in per_image.values())
        for category in common.EMAIL_MISMATCH_CATEGORIES
    } == metrics["email_mismatch_categories"]


def test_iou_handles_overlap_and_degenerate_boxes():
    assert common.box_iou((0, 0, 100, 100), (50, 0, 150, 100)) == pytest.approx(1 / 3)
    assert common.box_iou((0, 0, 0, 0), (0, 0, 5, 5)) == 0


def test_one_to_one_matching_finds_maximum_cardinality():
    ground_truth = (
        _entity(box=(0, 0, 100, 100)),
        _entity(box=(40, 0, 140, 100)),
    )
    predictions = (
        _prediction(box=(20, 0, 120, 100)),
        _prediction(box=(0, 0, 100, 100)),
    )
    metrics = common.score_results((_annotation(entities=ground_truth),), _run(predictions))
    assert metrics["strict"]["per_category"]["phone"]["tp"] == 2
    assert metrics["strict"]["per_category"]["phone"]["fp"] == 0


def test_duplicate_predictions_cannot_inflate_true_positives():
    metrics = common.score_results(
        (_annotation(entities=(_entity(),)),),
        _run((_prediction(), _prediction(box=(2, 0, 102, 40)))),
    )
    assert metrics["strict"]["per_category"]["phone"] == {
        "tp": 1, "fp": 1, "fn": 0, "precision": 0.5, "recall": 1.0,
        "f1": pytest.approx(2 / 3),
    }


def test_missing_detection_is_false_negative():
    metrics = common.score_results((_annotation(entities=(_entity(),)),), _run())
    assert metrics["strict"]["per_category"]["phone"]["fn"] == 1
    assert metrics["strict"]["per_category"]["phone"]["recall"] == 0


def test_transcription_error_is_strict_fp_fn_but_localized_match():
    metrics = common.score_results(
        (_annotation(entities=(_entity(),)),),
        _run((_prediction(value="+1 202 555 0199"),)),
    )
    strict = metrics["strict"]["per_category"]["phone"]
    localized = metrics["localization_only"]["per_category"]["phone"]
    assert (strict["tp"], strict["fp"], strict["fn"]) == (0, 1, 1)
    assert localized["tp"] == 1
    assert metrics["value_exactness_on_localized_matches"] == 0
    assert metrics["transcription_mismatch_count"] == 1


def test_evaluator_reports_only_localized_email_mismatch_categories(
    monkeypatch,
):
    wrong_email = _annotation(
        "wrong-email",
        (_entity(category="email", value="alex@example.com"),),
    )
    exact_email = _annotation(
        "exact-email",
        (_entity(category="email", value="case@example.com"),),
    )
    wrong_phone = _annotation(
        "wrong-phone",
        (_entity(category="phone", value="+1 202-555-0100"),),
    )
    truncated_email = _annotation(
        "truncated-email",
        (_entity(category="email", value="alex@example.com"),),
    )
    multiple_edits_email = _annotation(
        "multiple-edits-email",
        (_entity(category="email", value="alex@example.com"),),
    )

    def finding(category, value, box):
        left, top, right, bottom = box
        return SimpleNamespace(
            category=category,
            value=value,
            image_region=ImageRegion(
                ((left, top), (right, top), (right, bottom), (left, bottom))
            ),
        )

    findings_by_image = {
        "wrong-email": (
            finding(FindingCategory.EMAIL, "alix@example.com", (0, 0, 100, 40)),
            finding(FindingCategory.EMAIL, "other@example.org", (140, 50, 190, 90)),
        ),
        "exact-email": (
            finding(FindingCategory.EMAIL, " Case@Example.com ", (0, 0, 100, 40)),
        ),
        "wrong-phone": (
            finding(FindingCategory.PHONE, "+1 202-555-0199", (0, 0, 100, 40)),
        ),
        "truncated-email": (
            finding(FindingCategory.EMAIL, "alex@example.co", (0, 0, 100, 40)),
        ),
        "multiple-edits-email": (
            finding(FindingCategory.EMAIL, "arix@example.com", (0, 0, 100, 40)),
        ),
    }

    def fake_analyze(image_path, *, ocr, **_kwargs):
        ocr.last_result = SimpleNamespace(segments=())
        return SimpleNamespace(
            status=PipelineStatus.OK,
            findings=findings_by_image[Path(image_path).stem],
            errors=(),
        )

    class UnusedEngine:
        pass

    monkeypatch.setattr(evaluate, "analyze_image", fake_analyze)
    annotations = (
        wrong_email,
        exact_email,
        wrong_phone,
        truncated_email,
        multiple_edits_email,
    )
    _, metrics = evaluate.evaluate_split(
        annotations,
        _CountingOCR(UnusedEngine()),
        None,
    )

    assert metrics["email_mismatch_categories"] == {
        "exact_match": 0,
        "case_or_outer_whitespace_only": 0,
        "missing_at_symbol": 0,
        "missing_dot": 0,
        "internal_whitespace_difference": 0,
        "punctuation_difference": 0,
        "character_difference": 1,
        "truncation_like": 1,
        "multiple_character_edits": 1,
        "other_difference": 0,
    }
    assert tuple(metrics["email_mismatch_categories"]) == common.EMAIL_MISMATCH_CATEGORIES
    assert all(
        type(count) is int
        for count in metrics["email_mismatch_categories"].values()
    )
    per_image = {
        row["image_id"]: row["email_mismatch_categories"]
        for row in metrics["per_image"]
    }
    assert per_image["wrong-email"]["character_difference"] == 1
    assert per_image["truncated-email"]["truncation_like"] == 1
    assert per_image["multiple-edits-email"]["multiple_character_edits"] == 1
    assert per_image["exact-email"] == {
        category: 0 for category in common.EMAIL_MISMATCH_CATEGORIES
    }
    assert per_image["wrong-phone"] == {
        category: 0 for category in common.EMAIL_MISMATCH_CATEGORIES
    }
    assert all(
        tuple(counts) == common.EMAIL_MISMATCH_CATEGORIES
        and all(type(count) is int for count in counts.values())
        for counts in per_image.values()
    )
    assert {
        category: sum(counts[category] for counts in per_image.values())
        for category in common.EMAIL_MISMATCH_CATEGORIES
    } == metrics["email_mismatch_categories"]
    serialized_report_metrics = json.dumps(metrics, sort_keys=True)
    for private_value in (
        "alex@example.com",
        "alix@example.com",
        "other@example.org",
        "case@example.com",
        " Case@Example.com ",
        "alex@example.co",
        "arix@example.com",
    ):
        assert private_value not in serialized_report_metrics


def test_email_value_mismatch_is_localized_but_not_a_strict_match():
    annotation = _annotation(
        entities=(_entity(category="email", value="alex@example.com"),)
    )
    prediction = _prediction(
        category="email",
        value="alix@example.com",
        box=(5, 2, 105, 42),
    )

    metrics = common.score_results((annotation,), _run((prediction,)))

    strict = metrics["strict"]["per_category"]["email"]
    localized = metrics["localization_only"]["per_category"]["email"]
    assert (strict["tp"], strict["fp"], strict["fn"]) == (0, 1, 1)
    assert localized["tp"] == 1
    assert localized["fp"] == localized["fn"] == 0
    assert metrics["value_exactness_on_localized_matches"] == 0
    assert metrics["transcription_mismatch_count"] == 1


def test_partial_result_counts_only_localized_email_mismatch_once(monkeypatch):
    annotation = _annotation(
        "partial-email",
        (_entity(category="email", value="alex@example.com"),),
    )

    def finding(category, value, box):
        left, top, right, bottom = box
        return SimpleNamespace(
            category=category,
            value=value,
            image_region=ImageRegion(
                ((left, top), (right, top), (right, bottom), (left, bottom))
            ),
        )

    findings = (
        finding(FindingCategory.EMAIL, "alix@example.com", (0, 0, 100, 40)),
        finding(FindingCategory.EMAIL, "unmatched@example.org", (140, 50, 190, 90)),
        finding(FindingCategory.PHONE, "+1 202-555-0100", (0, 50, 90, 90)),
        finding(FindingCategory.PERSON_NAME, "Avery Sample", (110, 0, 190, 40)),
    )

    def fake_analyze(_image_path, *, ocr, **_kwargs):
        ocr.last_result = SimpleNamespace(segments=())
        return SimpleNamespace(
            status=PipelineStatus.PARTIAL,
            findings=findings,
            errors=(),
        )

    class UnusedEngine:
        pass

    monkeypatch.setattr(evaluate, "analyze_image", fake_analyze)
    _, metrics = evaluate.evaluate_split(
        (annotation,), _CountingOCR(UnusedEngine()), None
    )

    expected_counts = {
        category: 0 for category in common.EMAIL_MISMATCH_CATEGORIES
    }
    expected_counts["character_difference"] = 1
    assert metrics["status_counts"]["partial"] == 1
    assert metrics["email_mismatch_categories"] == expected_counts
    per_image = metrics["per_image"]
    assert len(per_image) == 1
    assert per_image[0]["status"] == "partial"
    assert per_image[0]["email_mismatch_categories"] == expected_counts
    assert sum(metrics["email_mismatch_categories"].values()) == 1
    assert {
        category: sum(row["email_mismatch_categories"][category] for row in per_image)
        for category in common.EMAIL_MISMATCH_CATEGORIES
    } == metrics["email_mismatch_categories"]


def test_complete_report_serialization_excludes_raw_email_and_ocr_text(monkeypatch):
    expected_email = "expected-private-sentinel@example.test"
    predicted_email = "predicted-private-sentinel@example.test"
    ocr_text = "OCR-RAW-TEXT-PRIVACY-SENTINEL"
    annotation = _annotation(
        "report-privacy",
        (_entity(category="email", value=expected_email),),
    )
    left, top, right, bottom = (0, 0, 100, 40)
    finding = SimpleNamespace(
        category=FindingCategory.EMAIL,
        value=predicted_email,
        image_region=ImageRegion(
            ((left, top), (right, top), (right, bottom), (left, bottom))
        ),
    )

    class UnusedEngine:
        pass

    ocr = _CountingOCR(UnusedEngine())

    def fake_prepare_runtime(tile_size, tile_overlap):
        return ocr, object(), {"easyocr_reader_ms": 1.0}, {"python": "test"}

    def fake_analyze(_image_path, *, ocr, **_kwargs):
        ocr.last_result = SimpleNamespace(
            segments=(SimpleNamespace(text=ocr_text),)
        )
        return SimpleNamespace(
            status=PipelineStatus.OK,
            findings=(finding,),
            errors=(),
        )

    monkeypatch.setattr(evaluate, "_load_split", lambda _split: (annotation,))
    monkeypatch.setattr(evaluate, "_validate_image_files", lambda _annotations: None)
    monkeypatch.setattr(evaluate, "serializer_contract_check", lambda: True)
    monkeypatch.setattr(evaluate, "_prepare_runtime", fake_prepare_runtime)
    monkeypatch.setattr(evaluate, "analyze_image", fake_analyze)

    report = evaluate.run(
        split="development",
        code_revision="synthetic-test-revision",
        tile_size=1600,
        tile_overlap=128,
    )

    serialized_report = json.dumps(report, sort_keys=True)
    categories = report["metrics"]["email_mismatch_categories"]
    assert report["run_status"] == "completed"
    assert set(categories) == set(common.EMAIL_MISMATCH_CATEGORIES)
    assert all(type(count) is int for count in categories.values())
    assert sum(categories.values()) == 1
    assert "email_mismatch_categories" in serialized_report
    for private_value in (expected_email, predicted_email, ocr_text):
        assert private_value not in serialized_report


def test_negative_image_false_positive_rate():
    annotation = _annotation(entities=())
    metrics = common.score_results((annotation,), _run((_prediction(),)))
    assert metrics["negative_image_false_positive_rate"] == 1
    assert metrics["strict"]["micro_overall"]["fp"] == 1


def test_partial_predictions_are_scored_and_error_predictions_are_discarded():
    partial = _annotation("one", (_entity(),), "development")
    failed = _annotation("two", (_entity(category="email", value="a@example.com", box=(0, 0, 100, 40)),), "development")
    runs = {
        "one": {"status": "partial", "predictions": (_prediction(),), "errors": [{"code": "ocr_partial", "stage": "ocr"}]},
        "two": {"status": "error", "predictions": (_prediction(category="email", value="a@example.com"),), "errors": [{"code": "ocr_failed", "stage": "ocr"}]},
    }
    metrics = common.score_results((partial, failed), runs)
    assert metrics["status_counts"] == {"ok": 0, "partial": 1, "error": 1}
    assert metrics["partial_rate"] == metrics["error_rate"] == 0.5
    assert metrics["strict"]["per_category"]["phone"]["tp"] == 1
    assert metrics["strict"]["per_category"]["email"]["fn"] == 1
    assert metrics["per_image"][1]["finding_count"] == 0


def test_serializer_redaction_contract():
    assert common.serializer_contract_check()


def test_counting_ocr_does_not_reuse_stale_segments_after_failure():
    class Engine:
        def extract(self, image_path):
            raise RuntimeError("synthetic failure")

    adapter = _CountingOCR(Engine())
    adapter.last_result = object()
    with pytest.raises(RuntimeError):
        adapter.extract("synthetic.png")
    assert adapter.last_result is None


@pytest.mark.parametrize("output_kind", ["relative", "absolute"])
def test_main_reports_relative_and_absolute_output_paths(
    monkeypatch, tmp_path, capsys, output_kind
):
    benchmark_root = tmp_path / "benchmark"
    results_directory = benchmark_root / "results"
    results_directory.mkdir(parents=True)
    monkeypatch.setattr(evaluate, "BENCHMARK_ROOT", benchmark_root)
    monkeypatch.chdir(tmp_path)
    output_path = results_directory / "test-report.json"
    argument_path = (
        Path("benchmark") / "results" / output_path.name
        if output_kind == "relative"
        else output_path
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--code-revision", "synthetic-test-revision",
            "--split", "development",
            "--output", str(argument_path),
        ],
    )
    monkeypatch.setattr(
        evaluate,
        "run",
        lambda **kwargs: {
            "run_status": "completed",
            "metrics": {
                "image_count": 0,
                "strict": {"micro_overall": {"f1": 1.0}},
                "status_counts": {"partial": 0, "error": 0},
            },
        },
    )

    assert evaluate.main() == 0
    assert output_path.is_file()
    output = capsys.readouterr().out
    expected_display = str(Path("results") / output_path.name)
    assert f"sanitized aggregate report: {expected_display}" in output


def test_address_scoring_strict_and_localization():
    gt = _entity(category="address", value="42 Main Street", box=(10, 10, 100, 40))
    annotation = _annotation(entities=(gt,))

    # Exact match
    pred_exact = _prediction(category="address", value="42 Main Street", box=(10, 10, 100, 40))
    metrics_exact = common.score_results((annotation,), _run((pred_exact,)))
    strict = metrics_exact["strict"]["per_category"]["address"]
    localized = metrics_exact["localization_only"]["per_category"]["address"]
    assert strict == {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert localized["tp"] == 1

    # Transcription mismatch: same box, different address text
    pred_diff_text = _prediction(category="address", value="99 Other Road", box=(10, 10, 100, 40))
    metrics_diff = common.score_results((annotation,), _run((pred_diff_text,)))
    assert metrics_diff["strict"]["per_category"]["address"] == {
        "tp": 0, "fp": 1, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0
    }
    assert metrics_diff["localization_only"]["per_category"]["address"]["tp"] == 1

    # Degenerate/non-overlapping box: IoU < 0.5
    pred_bad_box = _prediction(category="address", value="42 Main Street", box=(150, 50, 200, 90))
    metrics_box = common.score_results((annotation,), _run((pred_bad_box,)))
    assert metrics_box["strict"]["per_category"]["address"]["tp"] == 0
    assert metrics_box["localization_only"]["per_category"]["address"]["tp"] == 0


def test_macro_f1_with_zero_label_categories():
    # Only email and phone are labelled; person_name and address have zero labels.
    annotation = _annotation(
        entities=(
            _entity(category="email", value="alex@example.com", box=(0, 0, 100, 40)),
            _entity(category="phone", value="+1 202-555-0100", box=(0, 50, 100, 90)),
        )
    )
    predictions = (
        _prediction(category="email", value="alex@example.com", box=(0, 0, 100, 40)),
        _prediction(category="phone", value="+1 202-555-0100", box=(0, 50, 100, 90)),
    )
    metrics = common.score_results((annotation,), _run(predictions))
    # Both email and phone have F1 = 1.0. Macro F1 must be (1.0 + 1.0) / 2 = 1.0, not 0.5!
    assert metrics["strict"]["macro_f1"] == 1.0
    assert metrics["strict"]["per_category"]["address"]["f1"] is None
    assert metrics["strict"]["per_category"]["address"]["precision"] is None
    assert metrics["strict"]["per_category"]["address"]["recall"] is None
    assert metrics["strict"]["per_category"]["person_name"]["f1"] is None
    assert metrics["strict"]["per_category"]["person_name"]["precision"] is None
    assert metrics["strict"]["per_category"]["person_name"]["recall"] is None

    # Negative dataset with zero labels across all categories must return 0.0
    negative_annotation = _annotation(entities=())
    metrics_neg = common.score_results((negative_annotation,), _run())
    assert metrics_neg["strict"]["macro_f1"] == 0.0


def test_zero_label_category_reporting_with_and_without_false_positives():
    # Case 1: Category with zero labels and zero false positives -> unscored (None)
    annotation = _annotation(
        entities=(
            _entity(category="email", value="alex@example.com", box=(0, 0, 100, 40)),
        )
    )
    prediction_clean = (
        _prediction(category="email", value="alex@example.com", box=(0, 0, 100, 40)),
    )
    metrics_clean = common.score_results((annotation,), _run(prediction_clean))
    address_clean = metrics_clean["strict"]["per_category"]["address"]
    assert address_clean["tp"] == 0
    assert address_clean["fp"] == 0
    assert address_clean["fn"] == 0
    assert address_clean["precision"] is None
    assert address_clean["recall"] is None
    assert address_clean["f1"] is None
    # Verify JSON serialization represents undefined as null
    serialized_clean = json.dumps(metrics_clean)
    assert '"precision": null' in serialized_clean
    assert '"f1": null' in serialized_clean

    # Case 2: Category with zero labels but 1 false positive -> precision=0.0, f1=0.0 (penalty preserved)
    prediction_with_fp = (
        _prediction(category="email", value="alex@example.com", box=(0, 0, 100, 40)),
        _prediction(category="address", value="42 Main Street", box=(10, 10, 80, 40)),
    )
    metrics_fp = common.score_results((annotation,), _run(prediction_with_fp))
    address_fp = metrics_fp["strict"]["per_category"]["address"]
    assert address_fp["tp"] == 0
    assert address_fp["fp"] == 1
    assert address_fp["fn"] == 0
    assert address_fp["precision"] == 0.0
    assert address_fp["recall"] == 0.0
    assert address_fp["f1"] == 0.0
    assert metrics_fp["strict"]["micro_overall"]["fp"] == 1


def test_phone_normalization_strict_matches_existing():
    # Strict mode retains existing behavior
    assert common.normalize_value("phone", "+1 (202) 555-0100", phone_mode="strict") == "+12025550100"
    assert common.normalize_value("phone", "+91 98765 43210", phone_mode="strict") == "+919876543210"
    assert common.normalize_value("phone", "91 98765 43210", phone_mode="strict") == "919876543210"
    assert common.normalize_value("phone", "098765 43210", phone_mode="strict") == "09876543210"
    assert common.normalize_value("phone", "98765 43210", phone_mode="strict") == "9876543210"

    # Default call without phone_mode must match strict
    assert common.normalize_value("phone", "098765 43210") == "09876543210"


def test_phone_normalization_india_national_equivalent_forms():
    expected_canonical = "+919876543210"
    equivalent_inputs = [
        "+91 98765 43210",
        "91 98765 43210",
        "098765 43210",
        "98765 43210",
        "+91 (0) 98765 43210",
    ]
    for raw in equivalent_inputs:
        assert common.normalize_value("phone", raw, phone_mode="india_national") == expected_canonical


def test_phone_normalization_unsupported_and_ambiguous():
    # Non-Indian 10-digit number not starting with 6, 7, 8, or 9
    assert common.normalize_value("phone", "202 555 0100", phone_mode="india_national") == "2025550100"
    # Invalid length (9 digits)
    assert common.normalize_value("phone", "987654321", phone_mode="india_national") == "987654321"
    # Invalid length (11 digits starting with 9)
    assert common.normalize_value("phone", "98765432100", phone_mode="india_national") == "98765432100"
    # Short string
    assert common.normalize_value("phone", "12345", phone_mode="india_national") == "12345"
    # International numbers from other countries
    assert common.normalize_value("phone", "+1 202 555 0100", phone_mode="india_national") == "+12025550100"
    assert common.normalize_value("phone", "+44 20 7946 0919", phone_mode="india_national") == "+442079460919"

    # Unknown mode raises ValueError
    with pytest.raises(ValueError, match="unknown phone normalization mode"):
        common.normalize_value("phone", "9876543210", phone_mode="unsupported")


@pytest.mark.parametrize(
    "pred_value",
    [
        "+91 98765 43210",
        "91 98765 43210",
        "098765 43210",
        "98765 43210",
        "+91 (0) 98765 43210",
    ],
)
def test_phone_scoring_india_national_mode(pred_value):
    gt = _entity(category="phone", value="+91 98765 43210", box=(0, 0, 100, 40))
    pred = _prediction(category="phone", value=pred_value, box=(0, 0, 100, 40))
    annotation = _annotation(entities=(gt,))

    metrics_india = common.score_results((annotation,), _run((pred,)), phone_mode="india_national")
    assert metrics_india["strict"]["per_category"]["phone"]["tp"] == 1
    assert metrics_india["strict"]["per_category"]["phone"]["fp"] == 0
    assert metrics_india["strict"]["per_category"]["phone"]["fn"] == 0


def test_phone_scoring_strict_mode_mismatch():
    gt = _entity(category="phone", value="+91 98765 43210", box=(0, 0, 100, 40))
    pred = _prediction(category="phone", value="098765 43210", box=(0, 0, 100, 40))
    annotation = _annotation(entities=(gt,))

    metrics_strict = common.score_results((annotation,), _run((pred,)), phone_mode="strict")
    assert metrics_strict["strict"]["per_category"]["phone"]["tp"] == 0
    assert metrics_strict["strict"]["per_category"]["phone"]["fp"] == 1
    assert metrics_strict["strict"]["per_category"]["phone"]["fn"] == 1


def test_annotation_loading_optional_template_and_condition(tmp_path):
    # Record without template_id
    path = tmp_path / "no_template.jsonl"
    record1 = {
        "annotation_version": common.ANNOTATION_VERSION,
        "image_id": "test-01",
        "split": "development",
        "image_file": "images/development/test-01.png",
        "image_size": {"width": 100, "height": 60},
        "entities": [],
    }
    path.write_text(json.dumps(record1) + "\n", encoding="utf-8")
    loaded1 = common.load_annotations(path)[0]
    assert loaded1.template_id is None
    assert loaded1.condition is None

    # Record with generation.condition
    path2 = tmp_path / "with_condition.jsonl"
    record2 = {
        "annotation_version": common.ANNOTATION_VERSION,
        "image_id": "test-02",
        "split": "development",
        "image_file": "images/development/test-02.png",
        "image_size": {"width": 100, "height": 60},
        "template_id": "custom-template-01",
        "generation": {
            "condition": {
                "resolution": [1024, 768],
                "blur": 0.5,
                "jpeg_quality": 85,
                "rotation": 0,
            }
        },
        "entities": [],
    }
    path2.write_text(json.dumps(record2) + "\n", encoding="utf-8")
    loaded2 = common.load_annotations(path2)[0]
    assert loaded2.template_id == "custom-template-01"
    assert loaded2.condition == {
        "resolution": [1024, 768],
        "blur": 0.5,
        "jpeg_quality": 85,
        "rotation": 0,
    }


def test_grouped_reporting_by_template_and_condition():
    ann1 = common.ImageAnnotation(
        image_id="img-1",
        split="development",
        image_file="images/development/img-1.png",
        width=100,
        height=60,
        template_id="layout-A",
        entities=(_entity(category="phone", value="+1 202-555-0100"),),
        condition={"resolution": [1024, 768], "blur": 0.0, "jpeg_quality": 95, "rotation": 0},
    )
    ann2 = common.ImageAnnotation(
        image_id="img-2",
        split="development",
        image_file="images/development/img-2.png",
        width=100,
        height=60,
        template_id="layout-B",
        entities=(_entity(category="phone", value="+1 202-555-0100"),),
        condition={"resolution": [1280, 960], "blur": 1.0, "jpeg_quality": 80, "rotation": 90},
    )
    runs = {
        "img-1": {"status": "ok", "predictions": (_prediction(value="+1 202 555 0100"),)},
        "img-2": {"status": "ok", "predictions": ()},
    }

    metrics = common.score_results((ann1, ann2), runs)

    # by_template_id check
    assert "layout-A" in metrics["by_template_id"]
    assert "layout-B" in metrics["by_template_id"]
    assert metrics["by_template_id"]["layout-A"]["strict"]["micro_overall"]["tp"] == 1
    assert metrics["by_template_id"]["layout-B"]["strict"]["micro_overall"]["fn"] == 1

    # by_condition dimensions check
    by_cond = metrics["by_condition"]
    assert "0.0" in by_cond["blur"]
    assert "1.0" in by_cond["blur"]
    assert by_cond["blur"]["0.0"]["strict"]["micro_overall"]["tp"] == 1
    assert by_cond["blur"]["1.0"]["strict"]["micro_overall"]["fn"] == 1
    assert "1024x768" in by_cond["resolution"]
    assert "1280x960" in by_cond["resolution"]
    assert "95" in by_cond["jpeg_quality"]
    assert "80" in by_cond["jpeg_quality"]
    assert "0" in by_cond["rotation"]
    assert "90" in by_cond["rotation"]

    # by_object composite check
    assert len(by_cond["by_object"]) == 2

    # Graceful handling when condition is absent
    ann_empty = _annotation("img-empty", entities=())
    metrics_empty = common.score_results((ann_empty,), {"img-empty": {"status": "ok", "predictions": ()}})
    assert metrics_empty["by_condition"]["blur"] == {}
    assert metrics_empty["by_condition"]["by_object"] == {}


def test_cli_phone_mode_option_and_reporting(monkeypatch, tmp_path):
    captured_report = {}

    def fake_run(**kwargs):
        captured_report.update(kwargs)
        return {
            "run_status": "completed",
            "phone_mode": kwargs.get("phone_mode", "strict"),
            "metrics": {
                "image_count": 0,
                "strict": {"micro_overall": {"f1": 1.0}},
                "status_counts": {"partial": 0, "error": 0},
            },
        }

    monkeypatch.setattr(evaluate, "run", fake_run)
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(evaluate, "BENCHMARK_ROOT", tmp_path)
    output_path = results_dir / "test-cli-report.json"

    # Default CLI call must have phone_mode="strict"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--code-revision", "test-rev",
            "--split", "development",
            "--output", str(output_path),
        ],
    )
    assert evaluate.main() == 0
    assert captured_report["phone_mode"] == "strict"

    # Explicit --phone-mode india_national
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--code-revision", "test-rev",
            "--split", "development",
            "--phone-mode", "india_national",
            "--output", str(output_path),
        ],
    )
    assert evaluate.main() == 0
    assert captured_report["phone_mode"] == "india_national"
