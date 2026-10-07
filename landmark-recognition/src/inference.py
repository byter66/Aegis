"""Landmark recognition inference interface."""

from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_b0_regularized_best.pth"
)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model(checkpoint_path=CHECKPOINT_PATH):
    """Load the trained regularized EfficientNet-B0 model."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    class_mapping = checkpoint["class_mapping"]

    # class_mapping was created during training.
    num_classes = len(class_mapping)

    model = models.efficientnet_b0(weights=None)

    # Same classifier configuration used for Experiment 3.
    model.classifier[0] = torch.nn.Dropout(p=0.3)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    return model, class_mapping


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_landmark(
    image_path,
    model=None,
    class_mapping=None,
    confidence_threshold=0.50,
):
    """Predict the landmark present in an image.

    Parameters
    ----------
    image_path : str or Path
        Path to the input image.

    model : torch.nn.Module, optional
        Loaded EfficientNet-B0 model.

    class_mapping : dict, optional
        Mapping between class indices and landmark IDs.

    confidence_threshold : float
        Minimum confidence required to mark a landmark as detected.

    Returns
    -------
    dict
        Structured result for the Aegis risk-fusion module.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Load model and class mapping if not already supplied.
    if model is None or class_mapping is None:
        model, class_mapping = load_model()

    # Open image.
    image = Image.open(image_path).convert("RGB")

    # Apply the same deterministic preprocessing used during evaluation.
    input_tensor = TRANSFORM(image).unsqueeze(0)

    # Run inference.
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1,
    )

    confidence = float(confidence.item())
    predicted_index = int(predicted_index.item())

    # Convert model index back to the original landmark ID.
    if isinstance(class_mapping, dict):
        # Training mapping is expected to be:
        # landmark_id -> class_index
        index_to_class = {
            int(index): str(class_name)
            for class_name, index in class_mapping.items()
        }

        predicted_landmark = index_to_class[predicted_index]

    else:
        predicted_landmark = str(
            class_mapping[predicted_index]
        )

    # Determine whether the confidence is high enough.
    landmark_detected = confidence >= confidence_threshold

    return {
        "landmark_detected": landmark_detected,
        "predicted_landmark": predicted_landmark,
        "confidence": round(confidence, 4),
    }


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run landmark recognition on an image."
    )

    parser.add_argument(
        "image",
        help="Path to the image to analyze.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Confidence threshold for landmark detection.",
    )

    args = parser.parse_args()

    result = predict_landmark(
        args.image,
        confidence_threshold=args.threshold,
    )

    print(result)