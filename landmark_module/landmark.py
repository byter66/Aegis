import os, json
import numpy as np
import torch
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from transformers import CLIPModel, CLIPProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(HERE, "models", "landmark_model.keras"))
class_names = json.load(open(os.path.join(HERE, "models", "class_names.json")))
THRESHOLD = 0.70
CNN_OK = True

device = "cuda" if torch.cuda.is_available() else "cpu"
clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

places = ["Taj Mahal", "Red Fort", "India Gate", "Gateway of India", "Hawa Mahal", "Charminar", "Qutb Minar", "Victoria Memorial", "Mysore Palace", "Chinese fishing nets in Fort Kochi", "Fort Kochi", "Vagamon", "IIIT Kottayam", "Pala", "Saudi Arabia", "Melbourne", "Qatar", "Kochi International Airport", "Kottayam Railway Station"]

negatives = ["food on a plate", "a selfie", "a room interior", "an ordinary street","a beach", "a forest", "the sky", "a close-up of an object"]

place_prompts = [f"a photo of {p}, a famous landmark in India" for p in places]
neg_prompts = [f"a photo of {n}" for n in negatives]

revealing = ["an outdoor view of a distinctive building",
             "a street with shop signs and name boards",
             "a house exterior with a visible gate or name plate",
             "the view from a window showing the surroundings",
             "a building with a name board",
             "a campus entrance"]
neutral = ["food on a plate", "a close-up of an object", "a plain sky", "a selfie",
           "an indoor room", "an ordinary street with no signs",
           "a mountain landscape", "a garden or plants",
           "water with nothing else visible", "a close-up of a person"]

def scene_score(img_path):
    prompts = [f"a photo of {p}" for p in revealing + neutral]
    probs = clip_scores(img_path, prompts)
    return round(float(probs[:len(revealing)].sum()), 3)


def clip_scores(img_path, prompts):
    inputs = proc(text=prompts, images=Image.open(img_path).convert("RGB"),
                  return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        return clip(**inputs).logits_per_image.softmax(dim=1)[0].cpu()

def clip_landmark(img_path, threshold=0.5):
    probs = clip_scores(img_path, place_prompts + neg_prompts)
    i = int(probs.argmax())
    is_place = i < len(places) and float(probs[i]) >= threshold
    return {"landmark_flag": is_place,
            "landmark_confidence": round(float(probs[i]), 3),
            "landmark_name": places[i] if is_place else None}

def landmark_flag(img_path):
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
    best = c if c["landmark_confidence"] >= z["landmark_confidence"] else z
    flag = c["landmark_flag"] or z["landmark_flag"]
    return {"landmark_flag": flag,
            "landmark_confidence": best["landmark_confidence"],
            "landmark_name": best["landmark_name"] if flag else None,
            "landmark_source": "cnn" if best is c else "clip",
            "location_revealing_scene": scene_score(img_path)}

def _clean_name(name):
    if not name:
        return None
    name = str(name).replace("Category:", "").replace("_", " ").strip()
    return name or None

def landmark_payload(img_path):
    """Returns {"landmark": {...}, "scene": {...}} for Risk Fusion. Never raises."""
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(img_path)

        lm = landmark_module(img_path)              # CNN + CLIP merge
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
                "status": "ERROR", "detected": False, "name": None,
                "confidence": 0.0,
                "error": {"code": "INFERENCE_FAILED",
                          "message": f"Failed to process image: {type(e).__name__}"},
                "source": None, "raw_confidence": 0.0,
            },
            "scene": {"score": None},
        }

