"""Evaluate landmark detection on positive and negative validation images."""

from pathlib import Path

from src.inference import load_model, predict_landmark


PROJECT_ROOT = Path(__file__).resolve().parent

POSITIVE_DIR = PROJECT_ROOT / "tests" / "validation_landmarks"
NEGATIVE_DIR = PROJECT_ROOT / "tests" / "validation_non_landmarks"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
THRESHOLDS = [0.30, 0.50, 0.70]


def collect_images(folder, contains_landmark):
    """Collect validation images and their expected labels."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Validation folder not found: {folder}")

    images = [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return [(path, contains_landmark) for path in images]


def main():
    samples = (
        collect_images(POSITIVE_DIR, True)
        + collect_images(NEGATIVE_DIR, False)
    )

    if not samples:
        raise ValueError("No validation images found in the two folders.")

    model, class_mapping = load_model()
    results = []

    for image_path, actual_landmark in samples:
        try:
            result = predict_landmark(
                image_path,
                model=model,
                class_mapping=class_mapping,
                confidence_threshold=0.0,
            )

            results.append(
                {
                    "file": image_path.name,
                    "actual": actual_landmark,
                    "predicted_landmark": result["predicted_landmark"],
                    "confidence": result["confidence"],
                }
            )
        except Exception as error:
            print(f"Could not process {image_path}: {error}")

    if not results:
        raise RuntimeError("None of the validation images could be processed.")

    print(f"\nImages evaluated: {len(results)}")
    print("\nIndividual predictions (threshold = 0.50):")

    for result in results:
        detected = result["confidence"] >= 0.50
        actual = "LANDMARK" if result["actual"] else "NON-LANDMARK"

        print(
            f"{result['file']} | "
            f"actual={actual} | "
            f"predicted={result['predicted_landmark']} | "
            f"confidence={result['confidence']:.4f} | "
            f"detected={detected}"
        )

    print("\nThreshold comparison:")

    for threshold in THRESHOLDS:
        tp = fp = tn = fn = 0

        for result in results:
            actual = result["actual"]
            detected = result["confidence"] >= threshold

            if actual and detected:
                tp += 1
            elif actual and not detected:
                fn += 1
            elif not actual and detected:
                fp += 1
            else:
                tn += 1

        print(f"\nThreshold: {threshold:.2f}")
        print(f"True positives:  {tp}")
        print(f"False negatives: {fn}")
        print(f"False positives: {fp}")
        print(f"True negatives:  {tn}")


if __name__ == "__main__":
    main()