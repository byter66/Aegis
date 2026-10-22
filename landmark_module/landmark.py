"""AEGIS Landmark Detection Module (ResNet50 CNN + CLIP Zero-Shot Ensembling)."""

import os
import json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = 0.70
_CNN = None
_CLIP = None

places = [
    "Taj Mahal", "Red Fort", "India Gate", "Gateway of India", "Hawa Mahal",
    "Charminar", "Qutb Minar", "Victoria Memorial", "Mysore Palace",
    "Chinese fishing nets in Fort Kochi", "Fort Kochi", "Vagamon",
    "IIIT Kottayam", "Pala","Burj Khalifa", "Burj Al Arab", "Eiffel Tower", "Big Ben", "Tower Bridge",
    "Statue of Liberty", "Sydney Opera House", "Colosseum", "Golden Gate Bridge",
    "Petronas Towers", "Marina Bay Sands", "Kochi International Airport", "Kottayam Railway Station"
]

negatives = [
    "food on a plate", "a selfie", "a room interior", "an ordinary street",
    "a beach", "a forest", "the sky", "a close-up of an object"
]

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

VERIFY = True
NO_VERIFY = set()
VERIFY_NEG = [
    "a church", "a temple", "a mosque", "an ordinary building",
    "a house", "an ordinary street", "a landscape", "a bridge"
]


def _get_cnn():
    global _CNN
    if _CNN is None:
        from tensorflow.keras.models import load_model
        m = load_model(os.path.join(HERE, "models", "landmark_model.keras"))
        with open(os.path.join(HERE, "models", "class_names.json"), "r", encoding="utf-8") as f:
            names = json.load(f)
        _CNN = (m, names)
    return _CNN


def _get_clip():
    global _CLIP
    if _CLIP is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
        p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP = (m, p, dev)
    return _CLIP


def clip_scores(img_path, prompts):
    import torch
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


def clip_verify(img_path, name):
    label = name.replace("_", " ")
    probs = clip_scores(img_path, [f"a photo of {label}"] + [f"a photo of {n}" for n in VERIFY_NEG])
    return int(probs.argmax()) == 0


def clip_landmark(img_path, threshold=0.5):
    probs = clip_scores(img_path, place_prompts + neg_prompts)
    i = int(probs.argmax())
    is_place = (i < len(places)) and (float(probs[i]) >= threshold)
    return {
        "landmark_flag": is_place,
        "landmark_confidence": round(float(probs[i]), 3) if is_place else 0.0,
        "landmark_name": places[i] if is_place else None
    }


def landmark_flag(img_path):
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input

    model, class_names = _get_cnn()
    img = image.load_img(img_path, target_size=(224, 224))
    arr = preprocess_input(np.expand_dims(image.img_to_array(img), 0))
    p = model.predict(arr, verbose=0)[0]

    none_i = class_names.index("none") if "none" in class_names else None
    top = int(p.argmax())
    q = p.copy()
    if none_i is not None:
        q[none_i] = 0.0
    j = int(q.argmax())
    conf = float(q[j])
    name = class_names[j]

    ok = conf >= THRESHOLD and top != none_i
    if ok and VERIFY and name not in NO_VERIFY:
        ok = clip_verify(img_path, name)

    return {
        "landmark_flag": ok,
        "landmark_confidence": round(conf, 3),
        "landmark_name": name if ok else None
    }


def landmark_module(img_path):
    c = landmark_flag(img_path)
    z = clip_landmark(img_path)

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


def _error_payload(code, message):
    return {
        "landmark": {
            "status": "ERROR",
            "detected": False,
            "name": None,
            "confidence": 0.0,
            "error": {"code": code, "message": message},
            "source": None,
            "raw_confidence": 0.0
        },
        "scene": {"score": None}
    }


def landmark_payload(img_path):
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
            "raw_confidence": round(raw, 3)
        }
        return {"landmark": landmark, "scene": {"score": scene}}

    except ImportError:
        return _error_payload(
            "DEPENDENCY_MISSING",
            "Required deep learning runtime (TensorFlow/PyTorch/Transformers) is not available."
        )
    except Exception as e:
        return _error_payload("INFERENCE_FAILED", f"Failed to process image: {type(e).__name__}")
