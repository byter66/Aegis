from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pytest

from aegis_ocr.ner import SpacyNERAdapter
from aegis_ocr.ocr import EasyOCREngine
from aegis_ocr.schema import FindingCategory, PipelineStatus


pytestmark = pytest.mark.live


@pytest.fixture(autouse=True)
def require_live_test_opt_in() -> None:
    if os.environ.get("AEGIS_RUN_LIVE_TESTS") != "1":
        pytest.skip("set AEGIS_RUN_LIVE_TESTS=1 to run local-model smoke tests")


def _font(size: int) -> ImageFont.FreeTypeFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    pytest.fail("live OCR smoke test requires Arial or DejaVu Sans installed locally")


def test_live_easyocr_reads_synthetic_text_without_model_downloads(tmp_path: Path) -> None:
    if importlib.util.find_spec("easyocr") is None:
        pytest.fail("EasyOCR is unavailable; live test was explicitly enabled")

    module_root = Path(
        os.environ.get("EASYOCR_MODULE_PATH", str(Path.home() / ".EasyOCR"))
    )
    model_root = module_root / "model"
    required_weights = ("craft_mlt_25k.pth", "english_g2.pth")
    if not all((model_root / name).is_file() for name in required_weights):
        pytest.fail("required local EasyOCR weights are unavailable; downloads are disabled")

    image = Image.new("RGB", (760, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.text((28, 48), "SYNTHETIC TEST", fill="black", font=_font(58))
    image_path = tmp_path / "synthetic-ocr-smoke.png"
    image.save(image_path)

    import easyocr

    def offline_reader_factory(languages: list[str], *, gpu: bool) -> object:
        return easyocr.Reader(languages, gpu=gpu, download_enabled=False)

    result = EasyOCREngine(reader_factory=offline_reader_factory).extract(image_path)
    assert result.status is PipelineStatus.OK, "EasyOCR live smoke inference did not complete"
    recognized = any(
        "".join(character.casefold() for character in segment.text if character.isalnum())
        == "synthetictest"
        for segment in result.segments
    )
    assert recognized, "EasyOCR did not recognize the synthetic text within the configured tolerance"


def test_live_spacy_ner_loads_local_model_for_synthetic_name() -> None:
    if importlib.util.find_spec("spacy") is None:
        pytest.fail("spaCy is unavailable; live test was explicitly enabled")
    import spacy

    if not spacy.util.is_package("en_core_web_sm"):
        pytest.fail("en_core_web_sm is unavailable; live tests do not download models")

    text = "Peter Parker met a colleague."
    result = SpacyNERAdapter().extract(text)
    assert result.status is PipelineStatus.OK, "spaCy live smoke extraction did not complete"
    assert any(
        finding.category is FindingCategory.PERSON_NAME
        and text[finding.span.start:finding.span.end].casefold() == "peter parker"
        for finding in result.findings
    ), "spaCy did not identify the synthetic person-name example"
