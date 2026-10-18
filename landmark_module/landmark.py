"""AEGIS Landmark Detection Module (ResNet50 CNN + CLIP Zero-Shot Ensembling)."""

import os
import json
import numpy as np
import torch
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from transformers import CLIPModel, CLIPProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = 0.70

# Lazy-loaded model singletons (prevents hanging/crashing on import)
_CNN_MODEL = None
_CLASS_NAMES = None
_CLIP_MODEL = None
_CLIP_PROC = None
_DEVICE = None


def _get_cnn():
    global _CNN_MODEL, _CLASS_NAMES
    if _CNN_MODEL is None:
        model_path = os.path.join(HERE, "models", "landmark_model.keras")
        names_path = os.path.join(HERE, "models", "class_names.json")
        _CNN_MODEL = load_model(model_path)
        with open(names_path, "r", encoding="utf-8") as f:
            _CLASS_NAMES = json.load(f)
    return _CNN_MODEL, _CLASS_NAMES


def _get_clip():
    global _CLIP_MODEL, _CLIP_PROC, _DEVICE
    if _CLIP_MODEL is None:
        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_DEVICE).eval()
        _CLIP_PROC = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _CLIP_MODEL, _CLIP_PROC, _DEVICE


# Known places and negative control prompts
places = [
    "Taj Mahal", "Red Fort", "India Gate", "Gateway of India", "Hawa Mahal",
    "Charminar", "Qutb Minar", "Victoria Memorial", "Mysore Palace",
    "Chinese fishing nets in Fort Kochi", "Fort Kochi", "Vagamon",
    "IIIT Kottayam", "Pala", "Saudi Arabia", "Melbourne", "Qatar",
    "Kochi International Airport", "Kottayam Railway Station"
]

negatives = [
    "food on a plate", "a selfie", "a room interior", "an ordinary street",
    "a beach", "a forest", "the sky", "a close-up of an object"
]

# Accurate generalized prompts (avoids claiming Melbourne/Qatar are in India)
place_prompts = [f"a photo of {p}, a landmark or recognizable location" for p in places]
neg_prompts = [f"a photo of {n}" for n in negatives]

revealing = [
    "an outdoor view of a distinctive building",
    "a street with shop signs and name boards",
    "a house exterior with a visible gate or name plate",
    "the view from a window showing the surroundings",
    "a building with a name board",
    "a campus entrance"
]

neutral = [
    "food on a plate", "a close-up of an object", "a plain sky", "a selfie",
    "an indoor room", "an ordinary street with no signs",
    "a mountain landscape", "a garden or plants",
    "water with nothing else visible", "a close-up of a person"
]


def clip_scores(img_path, prompts):
    clip, proc, device = _get_clip()
    with Image.open(img_path) as raw_img:
        rgb_img = raw_img.convert("RGB")
        inputs = proc(text=prompts, images=rgb_img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        return clip(**inputs).logits_per_image.softmax(dim=1)[0].cpu()


def scene_score(img_path):
    prompts = [f"a photo of {p}" for p in revealing + neutral]
    probs = clip_scores(img_path, prompts)
    return round(float(probs[:len(revealing)].sum()), 3)


def clip_landmark(img_path, threshold=0.5):
    probs = clip_scores(img_path, place_prompts + neg_prompts)
    i = int(probs.argmax())
    is_place = (i < len(places)) and (float(probs[i]) >= threshold)

    # BUG FIX: If top prediction is a negative prompt (e.g. food), return 0.0 confidence for landmark
    return {
        "landmark_flag": is_place,
        "landmark_confidence": round(float(probs[i]), 3) if is_place else 0.0,
        "landmark_name": places[i] if is_place else None
    }


def landmark_flag(img_path):
    model, class_names = _get_cnn()
    img = image.load_img(img_path, target_size=(224, 224))
    arr = preprocess_input(np.expand_dims(image.img_to_array(img), 0))
    p = model.predict(arr, verbose=0)[0]
    conf = float(p.max())
    return {
        "landmark_flag": conf >= THRESHOLD,
        "landmark_confidence": round(conf, 3),
        "landmark_name": class_names[int(p.argmax())] if conf >= THRESHOLD else None,
    }


def landmark_module(img_path):
    c = landmark_flag(img_path)
    z = clip_landmark(img_path)

    # BUG FIX: Only pick from detectors that actually flagged a positive landmark
    positive_candidates = [d for d in (c, z) if d["landmark_flag"]]

    if positive_candidates:
        best = max(positive_candidates, key=lambda d: d["landmark_confidence"])
        flag = True
    else:
        best = c if c["landmark_confidence"] >= z["landmark_confidence"] else z
        flag = False

    return {
        "landmark_flag": flag,
        "landmark_confidence": best["landmark_confidence"] if flag else 0.0,
        "landmark_name": best["landmark_name"] if flag else None,
        "landmark_source": "cnn" if best is c else "clip",
        "location_revealing_scene": scene_score(img_path)
    }


def _clean_name(name):
    if not name:
        return None
    name = str(name).replace("Category:", "").replace("_", " ").strip()
    return name or None


def landmark_payload(img_path):
    """Returns {"landmark": {...}, "scene": {...}} for Risk Fusion. Never raises."""
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        lm = landmark_module(img_path)
        raw = float(lm["landmark_confidence"])
        scene = round(float(lm["location_revealing_scene"]), 3)
        detected = bool(lm["landmark_flag"]) and raw >= 0.40

        landmark = {
            "status": "OK",
            "detected": detected,
            "name": _clean_name(lm["landmark_name"]) if detected else None,
            "confidence": round(raw, 3) if detected else 0.0,
            "error": None,
            "source": lm["landmark_source"] if detected else None,
            "raw_confidence": round(raw, 3),
        }
        return {"landmark": landmark, "scene": {"score": scene}}

    except Exception as e:
        return {
            "landmark": {
                "status": "ERROR",
                "detected": False,
                "name": None,
                "confidence": 0.0,
                "error": {
                    "code": "INFERENCE_FAILED",
                    "message": f"Failed to process image: {type(e).__name__}"
                },
                "source": None,
                "raw_confidence": 0.0,
            },
            "scene": {"score": None},
        }
