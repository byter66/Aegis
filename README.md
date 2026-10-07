# Aegis
AI-Powered Pre-Upload Image Risk Detection &amp; Privacy Leak Prevention
## Landmark Recognition Inference

The trained EfficientNet-B0 model can be used to identify recognizable
landmarks in an input image.

### Usage

```python
from src.inference import predict_landmark

result = predict_landmark("path/to/image.jpg")

print(result)
Output
The inference module returns a structured dictionary:
{
  "landmark_detected": true,
  "predicted_landmark": "120734",
  "confidence": 0.8051
}

Output fields
- landmark_detected: Boolean indicating whether the prediction confidence meets the configured detection threshold.
- predicted_landmark: Predicted Google Landmark ID.
- confidence: Softmax confidence of the predicted landmark.
Confidence threshold
The default detection threshold is 0.50 and can be configured:
result = predict_landmark(    "path/to/image.jpg",    confidence_threshold=0.50)


The threshold is configurable so that the Risk Fusion module can adjust
the operating point if required.