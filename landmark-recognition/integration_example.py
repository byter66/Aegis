"""Example usage of the Aegis Landmark Recognition module."""

from src.inference import predict_landmark


def analyze_image(image_path):
    """Run landmark recognition for an input image."""

    result = predict_landmark(image_path)

    return {
        "landmark_detected": result["landmark_detected"],
        "predicted_landmark": result["predicted_landmark"],
        "confidence": result["confidence"],
    }


if __name__ == "__main__":
    image_path = "data/processed/final_dataset/images/120734/0f80cebb3c5d9d2c.jpg"

    result = analyze_image(image_path)

    print("Landmark Recognition Result")
    print("---------------------------")
    print(f"Landmark detected : {result['landmark_detected']}")
    print(f"Predicted landmark: {result['predicted_landmark']}")
    print(f"Confidence        : {result['confidence']}")