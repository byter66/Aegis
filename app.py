"""
Aegis — Streamlit frontend
==========================
"Check before you post": a privacy check for photos before they go online.

Place this file in the ROOT of the Aegis repo (next to README.md), then:

    py -3.11 -m pip install -r requirements-ui.txt
    py -3.11 -m streamlit run app.py

Flow (three screens, kept in st.session_state["step"]):
    compose  -> upload a photo + caption, click "Check before posting"
    review   -> "what a stranger could learn": profile of leaks, hide toggles,
                live re-scored risk (the real fusion model is re-run without
                the hidden findings)
    ready    -> the safer post preview + download

Needs the team's modules installed (landmark_module, ocr-pii-ner, risk-fusion).
If a module fails to load, the app says so instead of crashing.
"""

from __future__ import annotations

import html
import io
import os
import sys
import tempfile

import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

# ---------------------------------------------------------------------------
# Make the team's modules importable (three separate folders in the repo)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ("landmark_module", os.path.join("ocr-pii-ner", "src"), os.path.join("risk-fusion", "src")):
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.append(path)

# Per finding type: label, box colour, chip background, chip text, and how a stranger would read it
TYPES = {
    "person_name": dict(label="Name", colour="#B692F6", chip_bg="#2E1F4D", chip_fg="#D6BBFB",
                        row="Name", text="Visible in the image text"),
    "phone": dict(label="Phone number", colour="#F97066", chip_bg="#4A1D1A", chip_fg="#FDA29B",
                  row="How to reach you", text="A phone number is readable"),
    "email": dict(label="Email address", colour="#FDB022", chip_bg="#4A3510", chip_fg="#FEDF89",
                  row="How to reach you", text="An email address is readable"),
    "address": dict(label="Address", colour="#53B1FD", chip_bg="#102A56", chip_fg="#B2DDFF",
                    row="Street", text="An address is visible"),
}
ROW_ORDER = ["person_name", "landmark", "phone", "email", "address", "scene"]
LEVEL = {
    "High": dict(fg="#F97066", bg="#2A1215", border="#5C2126", word="High"),
    "Medium": dict(fg="#FEC84B", bg="#2A2110", border="#5C4813", word="Medium"),
    "Low": dict(fg="#47CD89", bg="#0F2A1D", border="#1D5C3C", word="Low"),
    "Unknown": dict(fg="#D0D5DD", bg="#1F2329", border="#3A3F47", word="Unknown"),
}
SCENE_THRESHOLD = 0.5  # scene score at which we call the background "revealing"
MAX_SIDE = 1600        # photos are shrunk to this many pixels on the longest side before analysis


def tinfo(t: str) -> dict:
    return TYPES.get(t, dict(label=t.replace("_", " ").title(), colour="#F97066", chip_bg="#4A1D1A",
                             chip_fg="#FDA29B", row=t.replace("_", " ").title(), text="Detected in the image text"))


def esc(s) -> str:
    return html.escape(str(s))


# ---------------------------------------------------------------------------
# Real models — loaded once, kept in memory
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading risk-fusion model...")
def load_fusion():
    from fusion_service import AegisRiskFusion
    return AegisRiskFusion()


@st.cache_resource(show_spinner="Loading landmark models (first time can take a minute)...")
def load_landmark_fn():
    from landmark import landmark_payload
    return landmark_payload


class _EdgeSafeReader:
    """Wraps EasyOCR so one odd result can't fail the whole text scan.

    On slanted text near an edge, EasyOCR can return a box corner at a negative
    coordinate (e.g. x = -32). The OCR module rejects negative coordinates, and
    one such box made the WHOLE scan fail ("ocr_failed"). Clamping corners to 0
    keeps every other result. Empty-text boxes are dropped for the same reason.
    Remove this once both fixes are in ocr.py itself.
    """

    def __init__(self, reader):
        self._reader = reader

    def readtext(self, *args, **kwargs):
        fixed = []
        for item in self._reader.readtext(*args, **kwargs):
            if isinstance(item, (list, tuple)) and len(item) == 3:
                box, text, conf = item
                # EasyOCR sometimes returns a box with empty text (confidence 0).
                # The OCR module rejects empty text, which also failed the whole scan,
                # so drop those boxes: there is nothing in them to check anyway.
                if not isinstance(text, str) or not text.strip():
                    continue
                try:
                    box = [[max(0.0, float(p[0])), max(0.0, float(p[1]))] for p in box]
                except (TypeError, ValueError, IndexError):
                    pass
                item = (box, text, conf)
            fixed.append(item)
        return fixed

    def __getattr__(self, name):
        return getattr(self._reader, name)


def _edge_safe_reader_factory(langs, gpu=False):
    import easyocr
    return _EdgeSafeReader(easyocr.Reader(langs, gpu=gpu))


@st.cache_resource(show_spinner="Loading OCR engine...")
def load_ocr():
    from aegis_ocr import EasyOCREngine, SpacyNERAdapter
    return EasyOCREngine(reader_factory=_edge_safe_reader_factory), SpacyNERAdapter()


# ---------------------------------------------------------------------------
# Fallback scoring
# ---------------------------------------------------------------------------
def fallback_fusion(payload: dict) -> dict:
    """Rule-based stand-in, used only if the trained fusion model can't load."""
    ocr, lm = payload["ocr_ner"], payload["landmark"]
    if ocr.get("status") == "ERROR" or lm.get("status") == "ERROR":
        return {"risk_level": "Unknown", "risk_score": None, "safe_to_post": False,
                "reasons": ["Upstream detector error occurred; cannot verify privacy safety."], "findings": []}
    types = {f["type"] for f in ocr.get("findings", [])}
    score = min(100, 30 * len(types) + (40 if lm.get("detected") else 0))
    level = "High" if score >= 60 else "Medium" if score >= 30 else "Low"
    return {"risk_level": level, "risk_score": score, "safe_to_post": level == "Low",
            "reasons": [], "findings": ocr.get("findings", [])}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def prepare_image(data: bytes, filename: str) -> tuple[Image.Image, str]:
    """Fix phone-camera rotation and save a clean temp copy (modules need a file path).
    Re-saving also drops the photo's EXIF metadata from the copy we analyse."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    is_png = filename.lower().endswith(".png")
    img = img.convert("RGBA" if is_png and img.mode in ("RGBA", "LA", "P") else "RGB")
    # Big phone photos (4000+ px) make OCR very slow on a laptop CPU. Shrinking the
    # longest side to MAX_SIDE keeps text readable and cuts the time a lot.
    # Boxes stay correct because the app shows and blurs this same resized image.
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    suffix = ".png" if is_png else ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        img.save(f, format="PNG" if is_png else "JPEG", quality=95)
        return img, f.name


def _err(code: str, msg: str) -> dict:
    return {"code": code, "message": msg}


def run_landmark(path: str) -> dict:
    try:
        return load_landmark_fn()(path)
    except Exception as e:
        return {"landmark": {"status": "ERROR", "detected": False, "name": None, "confidence": 0.0,
                             "error": _err("MODULE_UNAVAILABLE", f"Landmark module unavailable ({type(e).__name__})."),
                             "source": None, "raw_confidence": 0.0}, "scene": {"score": None}}


def run_ocr(path: str) -> dict:
    try:
        from aegis_ocr import analyze_image, to_risk_fusion_ocr_payload
        ocr_engine, ner_engine = load_ocr()
        return to_risk_fusion_ocr_payload(analyze_image(path, ocr=ocr_engine, ner=ner_engine))
    except Exception as e:
        return {"status": "ERROR", "findings": [],
                "error": _err("MODULE_UNAVAILABLE", f"Text scan module unavailable ({type(e).__name__}).")}


def fuse(landmark_out: dict, ocr_out: dict) -> tuple[dict, bool]:
    """Run risk fusion. Returns (result, used_trained_model)."""
    payload = {"ocr_ner": ocr_out, "landmark": landmark_out["landmark"],
               "scene": landmark_out.get("scene") or {"score": None}}
    try:
        model = load_fusion()
    except Exception:
        return fallback_fusion(payload), False
    return model.evaluate_image(payload), True


@st.cache_data(show_spinner=False, max_entries=20)
def analyse(data: bytes, filename: str) -> dict:
    """Whole pipeline, cached per image so clicking toggles never re-runs the models."""
    img, path = prepare_image(data, filename)
    try:
        lm = run_landmark(path)
        ocr = run_ocr(path)
    finally:
        os.remove(path)  # never keep users' photos on disk
    fused, trained = fuse(lm, ocr)
    return {"image": img, "landmark": lm, "ocr": ocr, "fusion": fused, "trained": trained}


def rescore(res: dict, hidden: set[str]) -> dict:
    """Re-run fusion as if the hidden finding types had been blurred out."""
    ocr = dict(res["ocr"])
    ocr["findings"] = [f for f in res["ocr"].get("findings", []) if f.get("type") not in hidden]
    return fuse(res["landmark"], ocr)[0]


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _box(bbox, img: Image.Image, pad: int = 0):
    x0, y0, x1, y1 = bbox
    return (max(0, int(x0) - pad), max(0, int(y0) - pad),
            min(img.width, int(round(x1)) + pad), min(img.height, int(round(y1)) + pad))


def blur_regions(img: Image.Image, findings: list[dict]) -> Image.Image:
    out = img.copy()
    radius = max(8, int(max(out.width, out.height) / 60))
    for f in findings:
        if "bbox" not in f:
            continue
        b = _box(f["bbox"], out, pad=max(4, radius // 2))
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        region = out.crop(b)
        small = region.resize((max(1, region.width // 12), max(1, region.height // 12)), Image.BILINEAR)
        out.paste(small.resize(region.size, Image.NEAREST).filter(ImageFilter.GaussianBlur(radius)), b[:2])
    return out


def draw_boxes(img: Image.Image, findings: list[dict]) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    scale = max(out.width, out.height) / 1000
    line = max(2, int(4 * scale))
    font = _font(max(12, int(18 * scale)))
    for f in findings:
        if "bbox" not in f:
            continue
        x0, y0, x1, y1 = _box(f["bbox"], out)
        info = tinfo(f.get("type", ""))
        d.rectangle([x0, y0, x1, y1], outline=info["colour"], width=line)
        tb = d.textbbox((0, 0), info["label"], font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = max(3, int(4 * scale))
        top = y0 - th - 2 * pad if y0 - th - 2 * pad >= 0 else y1
        d.rectangle([x0, top, x0 + tw + 2 * pad, top + th + 2 * pad], fill=info["colour"])
        d.text((x0 + pad - tb[0], top + pad - tb[1]), info["label"], fill="#0E1117", font=font)
    return out


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Building the "stranger profile"
# ---------------------------------------------------------------------------
def profile_rows(res: dict) -> list[dict]:
    """Turn module outputs into plain-language rows: what someone could infer."""
    rows = []
    lm = res["landmark"]["landmark"]
    scene = (res["landmark"].get("scene") or {}).get("score")
    seen = set()
    for f in res["fusion"].get("findings") or res["ocr"].get("findings", []):
        t = f.get("type", "")
        if t in seen:
            continue
        seen.add(t)
        info = tinfo(t)
        rows.append(dict(key=t, label=info["row"], text=info["text"], chip=f.get("redacted_text") or "[HIDDEN]",
                         chip_bg=info["chip_bg"], chip_fg=info["chip_fg"], mono=True))
    if lm.get("status") == "OK" and lm.get("detected"):
        src = {"cnn": "CNN", "clip": "CLIP"}.get(lm.get("source"), "model")
        rows.append(dict(key="landmark", label="Where you were", text=f"Near {lm.get('name') or 'a known landmark'}",
                         chip=f"{lm.get('confidence', 0) * 100:.0f}% · {src}", chip_bg="#1F2937", chip_fg="#E5E7EB", mono=False))
    if scene is not None and scene >= SCENE_THRESHOLD:
        rows.append(dict(key="scene", label="Surroundings", text="The background gives away the area",
                         chip=f"{scene * 100:.0f}% revealing", chip_bg="#1F2937", chip_fg="#E5E7EB", mono=False))
    rows.sort(key=lambda r: ROW_ORDER.index(r["key"]) if r["key"] in ROW_ORDER else 99)
    return rows


def errors_of(res: dict) -> list[str]:
    msgs = []
    lm, ocr = res["landmark"]["landmark"], res["ocr"]
    if lm.get("status") == "ERROR":
        msgs.append((lm.get("error") or {}).get("message", "The landmark check failed."))
    if ocr.get("status") == "ERROR":
        msgs.append((ocr.get("error") or {}).get("message", "The text scan failed."))
    elif ocr.get("status") == "PARTIAL":
        msgs.append("Part of the text scan failed, so some details may have been missed.")
    return msgs


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@500&display=swap');
html, body, .stApp, .stMarkdown, button, input, label, textarea {font-family: Inter, system-ui, sans-serif;}
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px;}
header[data-testid="stHeader"] {background: transparent;}
h1, h2, h3, .ag-display {font-family: 'Space Grotesk', Inter, sans-serif !important; letter-spacing: -0.01em;}
button[kind="primary"], [data-testid="stBaseButton-primary"], [data-testid="stDownloadButton"] button[kind="primary"] {color: #0E1117 !important; font-weight: 600;}
.stButton button, .stDownloadButton button {min-height: 46px; border-radius: 12px; font-weight: 600;}
[data-testid="stImage"] img {border-radius: 14px;}
[data-testid="stHeaderActionElements"], .stMarkdown h1 a, .stMarkdown h2 a {display: none !important;}

.ag-top {display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap; margin-bottom: 1.4rem;}
.ag-brand {display:flex; align-items:center; gap:10px; font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:22px;}
.ag-steps {display:flex; gap:6px; align-items:center; flex-wrap:wrap; font-size:13px; color:#9CA3AF;}
.ag-step {padding:6px 12px; border-radius:999px;}
.ag-step.done {background:#1E2235; color:#A5B4FC; font-weight:600;}
.ag-step.now {background:#818CF8; color:#0E1117; font-weight:700;}
.ag-sep {color:#4B5563;}

.ag-card {background:#161B22; border:1px solid #262C36; border-radius:16px; padding:24px;}
.ag-eyebrow {font-family:'JetBrains Mono',monospace; font-size:12px; letter-spacing:.08em; color:#9CA3AF;}
.ag-h1 {font-family:'Space Grotesk',sans-serif; font-size:40px; line-height:1.1; font-weight:700; margin:0 0 1.2rem 0; max-width: 900px;}
.ag-muted {color:#9CA3AF; font-size:14px; line-height:1.5;}

.ag-row {display:grid; grid-template-columns: 118px minmax(0,1fr) auto; gap:14px; align-items:center; padding:15px 0; border-bottom:1px solid #262C36;}
.ag-row:last-child {border-bottom:0;}
.ag-row .k {color:#9CA3AF; font-size:14px;}
.ag-row .v {font-size:16px; font-weight:600; min-width:0;}
.ag-row.hidden .v {color:#6B7280; text-decoration: line-through;}
.ag-chip {font-size:13px; padding:4px 10px; border-radius:6px; white-space:nowrap;}
.ag-chip.mono {font-family:'JetBrains Mono',monospace;}
.ag-chip.ok {background:#0F2A1D; color:#75E0A7;}

.ag-risk {border-radius:16px; padding:22px 24px; border:1px solid;}
.ag-risk .lbl {font-size:13px; font-weight:600;}
.ag-risk .num {font-family:'Space Grotesk',sans-serif; font-size:60px; font-weight:700; line-height:1;}
.ag-risk .word {font-size:18px; margin-left:8px;}
.ag-after {display:flex; justify-content:space-between; align-items:center; padding:14px 0 4px 0; border-top:1px solid #262C36; margin-top:8px;}

.ag-post {background:#161B22; border:1px solid #262C36; border-radius:16px; overflow:hidden;}
.ag-post .head {display:flex; gap:12px; align-items:center; padding:14px 18px;}
.ag-avatar {width:40px; height:40px; border-radius:50%; background:#312E81; color:#C7D2FE; display:flex; align-items:center; justify-content:center; font-weight:700;}
.ag-feature {display:flex; gap:14px; align-items:flex-start; padding:14px 0; border-bottom:1px solid #262C36;}
.ag-feature:last-child {border-bottom:0;}
.ag-dot {width:10px; height:10px; border-radius:3px; margin-top:6px; flex-shrink:0;}
.ag-banner {background:#2A2110; border:1px solid #5C4813; color:#FEDF89; border-radius:12px; padding:12px 16px; font-size:14px; margin-bottom:1rem;}
</style>
"""

SHIELD = ('<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#A5B4FC" stroke-width="2" '
          'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l8 3v6c0 4.5-3.4 8.3-8 9-4.6-.7-8-4.5-8-9V6l8-3z"></path>'
          '<path d="M9 12l2 2 4-4"></path></svg>')


def top_bar(step: str):
    order = [("compose", "1 Add photo"), ("review", "2 Review"), ("ready", "3 Post")]
    idx = [k for k, _ in order].index(step)
    parts = []
    for i, (k, label) in enumerate(order):
        cls = "now" if i == idx else "done" if i < idx else ""
        parts.append(f'<span class="ag-step {cls}">{label}</span>')
    steps = '<span class="ag-sep">—</span>'.join(parts)
    st.markdown(f'<div class="ag-top"><div class="ag-brand">{SHIELD}Aegis</div>'
                f'<nav class="ag-steps" aria-label="Steps">{steps}</nav></div>', unsafe_allow_html=True)


def risk_card(level: str, score, label: str = "Risk right now"):
    s = LEVEL.get(level, LEVEL["Unknown"])
    num = f'{score}' if score is not None else "?"
    st.markdown(
        f'<div class="ag-risk" style="background:{s["bg"]}; border-color:{s["border"]};">'
        f'<div class="lbl" style="color:{s["fg"]}">{esc(label)}</div>'
        f'<div style="display:flex; align-items:baseline; margin-top:6px;"><span class="num" style="color:{s["fg"]}">{num}</span>'
        f'<span class="word" style="color:{s["fg"]}">{s["word"]}</span></div></div>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------
def go(step: str):
    st.session_state.step = step


def reset():
    for k in list(st.session_state.keys()):
        if k in ("photo", "photo_name", "caption", "hidden_types", "after") or str(k).startswith("hide_"):
            st.session_state.pop(k, None)
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
    go("compose")


def screen_compose():
    top_bar("compose")
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown('<div class="ag-post"><div class="head"><div class="ag-avatar">A</div>'
                    '<div><div style="font-weight:600">New post</div><div class="ag-muted">Draft</div></div></div></div>',
                    unsafe_allow_html=True)
        up = st.file_uploader("Photo", type=["jpg", "jpeg", "png"], key=f"up_{st.session_state.get('uploader_key', 0)}")
        if up is not None:
            st.image(up, width="stretch")
        caption = st.text_input("Caption", value=st.session_state.get("caption", ""), placeholder="Write a caption...")
    with right:
        st.markdown('<p class="ag-eyebrow">BEFORE YOU POST</p>'
                    '<h1 class="ag-h1">Would you hand this photo to a stranger?</h1>'
                    '<p class="ag-muted" style="font-size:16px; margin-bottom:1.2rem">Aegis checks what your photo quietly gives away, '
                    'then helps you hide it before it goes online.</p>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ag-card">'
            '<div class="ag-feature"><span class="ag-dot" style="background:#A5B4FC"></span><div><div style="font-weight:600">Where you were</div>'
            '<div class="ag-muted">Recognises landmarks with a fine-tuned CNN and CLIP, and scores how much the background gives away.</div></div></div>'
            '<div class="ag-feature"><span class="ag-dot" style="background:#F97066"></span><div><div style="font-weight:600">Personal details in the picture</div>'
            '<div class="ag-muted">Reads text in the photo and finds names, phone numbers, emails and addresses.</div></div></div>'
            '<div class="ag-feature"><span class="ag-dot" style="background:#FEC84B"></span><div><div style="font-weight:600">One risk score</div>'
            '<div class="ag-muted">A trained model combines everything into Low, Medium or High.</div></div></div>'
            '</div>', unsafe_allow_html=True)
        st.write("")
        if st.button("Check before posting", type="primary", width="stretch", disabled=up is None):
            st.session_state.photo = up.getvalue()
            st.session_state.photo_name = up.name
            st.session_state.caption = caption
            st.session_state.hidden_types = None  # set to defaults on the review screen
            go("review")
            st.rerun()
        st.caption("Your photo is analysed in memory and deleted right after.")


def screen_review():
    top_bar("review")
    with st.spinner("Checking your photo..."):
        try:
            res = analyse(st.session_state.photo, st.session_state.photo_name)
        except Exception as e:
            st.error(f"Couldn't read this image ({type(e).__name__}). Try another JPG or PNG.")
            st.button("Choose another photo", on_click=reset)
            return

    fusion = res["fusion"]
    findings = [f for f in (fusion.get("findings") or res["ocr"].get("findings", [])) if "bbox" in f]
    types_present = list(dict.fromkeys(f["type"] for f in findings))
    if st.session_state.get("hidden_types") is None:          # first visit: hide everything by default
        st.session_state.hidden_types = set(types_present)
    rows = profile_rows(res)
    errs = errors_of(res)

    for e in errs:
        st.markdown(f'<div class="ag-banner">{esc(e)} Aegis can\'t confirm this photo is safe.</div>', unsafe_allow_html=True)

    n = len(rows)
    if errs and not rows:
        headline = "We couldn't fully check this photo."
    elif n == 0:
        headline = "A stranger wouldn't learn much from this photo."
    else:
        headline = f"Here's what a stranger could learn from this photo: {n} thing{'s' if n != 1 else ''}."
    st.markdown(f'<h1 class="ag-h1">{esc(headline)}</h1>', unsafe_allow_html=True)

    c_img, c_prof, c_side = st.columns([0.8, 1.65, 0.95], gap="medium")

    with c_side:
        risk_card(fusion.get("risk_level", "Unknown"), fusion.get("risk_score"))
        st.write("")
        with st.container(border=True):
            st.markdown("**Hide before posting**")
            hidden = set()
            if types_present:
                for t in types_present:
                    on = st.checkbox(f"Blur the {tinfo(t)['label'].lower()}", value=t in st.session_state.hidden_types, key=f"hide_{t}")
                    if on:
                        hidden.add(t)
            else:
                st.markdown('<span class="ag-muted">No text details to blur.</span>', unsafe_allow_html=True)
            st.session_state.hidden_types = hidden
            lm = res["landmark"]["landmark"]
            if lm.get("detected"):
                st.markdown(f'<p class="ag-muted" style="margin-top:6px">{esc(lm.get("name") or "The landmark")} can\'t be blurred away. '
                            'Crop it out, or post after you\'ve left.</p>', unsafe_allow_html=True)
            after = rescore(res, hidden) if hidden else fusion
            a_level = after.get("risk_level", "Unknown")
            a_style = LEVEL.get(a_level, LEVEL["Unknown"])
            a_score = after.get("risk_score")
            st.markdown(f'<div class="ag-after"><span style="color:#D1D5DB; font-size:14px">After these fixes</span>'
                        f'<span style="font-weight:700; color:{a_style["fg"]}">{a_score if a_score is not None else "?"} · {a_style["word"]}</span></div>',
                        unsafe_allow_html=True)
        st.session_state.after = {"level": a_level, "score": a_score}
        st.button("Continue with safer photo", type="primary", width="stretch", on_click=go, args=("ready",))
        st.button("Choose another photo", width="stretch", on_click=reset)

    with c_img:
        shown_hidden = [f for f in findings if f["type"] in hidden]
        shown_visible = [f for f in findings if f["type"] not in hidden]
        preview = draw_boxes(blur_regions(res["image"], shown_hidden), shown_visible)
        st.image(preview, width="stretch")
        st.markdown(f'<span class="ag-muted">{esc(st.session_state.photo_name)}</span>', unsafe_allow_html=True)

    with c_prof:
        body = []
        for r in rows:
            cls = "ag-row hidden" if r["key"] in hidden else "ag-row"
            chip_cls = "ag-chip mono" if r["mono"] else "ag-chip"
            chip = (f'<span class="ag-chip ok">Blurred</span>' if r["key"] in hidden
                    else f'<span class="{chip_cls}" style="background:{r["chip_bg"]}; color:{r["chip_fg"]}">{esc(r["chip"])}</span>')
            body.append(f'<div class="{cls}"><span class="k">{esc(r["label"])}</span><span class="v">{esc(r["text"])}</span>{chip}</div>')
        if not body:
            body.append('<div class="ag-row"><span class="k">Result</span><span class="v">No names, contact details or recognisable places found</span>'
                        '<span class="ag-chip ok">Clear</span></div>')
        st.markdown(f'<div class="ag-card"><p class="ag-eyebrow" style="margin:0 0 6px 0">PROFILE BUILT FROM YOUR PHOTO</p>{"".join(body)}</div>',
                    unsafe_allow_html=True)
        reasons = fusion.get("reasons") or []
        if reasons:
            with st.expander("Why the model scored it this way"):
                for reason in reasons:
                    st.markdown(f"- {reason}")
        with st.expander("Raw module outputs (for debugging)"):
            st.json({"landmark": res["landmark"], "ocr_ner": res["ocr"], "fusion": fusion,
                     "trained_fusion_model": res["trained"]})


def screen_ready():
    top_bar("ready")
    res = analyse(st.session_state.photo, st.session_state.photo_name)
    hidden = st.session_state.get("hidden_types") or set()
    findings = [f for f in (res["fusion"].get("findings") or res["ocr"].get("findings", [])) if "bbox" in f]
    safe = blur_regions(res["image"], [f for f in findings if f["type"] in hidden])
    after = st.session_state.get("after") or {"level": res["fusion"].get("risk_level"), "score": res["fusion"].get("risk_score")}
    caption = st.session_state.get("caption") or ""

    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown('<div class="ag-post"><div class="head"><div class="ag-avatar">A</div>'
                    '<div><div style="font-weight:600">New post</div><div class="ag-muted">Ready to share</div></div></div></div>',
                    unsafe_allow_html=True)
        st.image(safe, width="stretch")
        if caption:
            st.markdown(f'<p style="font-size:15px; margin-top:.4rem">{esc(caption)}</p>', unsafe_allow_html=True)
    with right:
        st.markdown('<p class="ag-eyebrow">YOUR SAFER VERSION</p>'
                    '<h1 class="ag-h1">Ready when you are.</h1>', unsafe_allow_html=True)
        risk_card(after["level"] or "Unknown", after["score"], label="Risk after your fixes")
        st.write("")
        n = len(hidden)
        msg = (f"{n} detail{'s' if n != 1 else ''} blurred. " if n else "Nothing was blurred. ")
        if res["landmark"]["landmark"].get("detected"):
            msg += "The landmark is still visible, so people could still tell where you were."
        st.markdown(f'<p class="ag-muted" style="font-size:15px">{esc(msg)}</p>', unsafe_allow_html=True)
        base = os.path.splitext(st.session_state.photo_name)[0]
        st.download_button("Download safer photo", png_bytes(safe), file_name=f"{base}_aegis_safe.png",
                           mime="image/png", type="primary", width="stretch")
        st.caption("Aegis doesn't post for you. Download the photo and upload it to your app.")
        c1, c2 = st.columns(2)
        c1.button("Back to review", width="stretch", on_click=go, args=("review",))
        c2.button("Check another photo", width="stretch", on_click=reset)


def main():
    st.set_page_config(page_title="Aegis: check before you post", page_icon="🛡️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.session_state.setdefault("step", "compose")
    step = st.session_state.step
    if step != "compose" and "photo" not in st.session_state:
        step = "compose"
    if step == "compose":
        screen_compose()
    elif step == "review":
        screen_review()
    else:
        screen_ready()


main()
