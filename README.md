# Aegis
AI-Powered Pre-Upload Image Risk Detection &amp; Privacy Leak Prevention

## Landmark Recognition Inference
The trained EfficientNet-B0 model identifies potential landmarks in an input image and returns a structured result for the Aegis Risk Fusion module.

### Usage

```python
from src.inference import predict_landmark

result = predict_landmark("path/to/image.jpg")

print(result)
```

### Example Output

```json
{
  "landmark_detected": true,
  "predicted_landmark": "120734",
  "confidence": 0.8051
}
```

### Output Fields

- **`landmark_detected`**: Boolean indicating whether the prediction confidence meets the configured detection threshold.
- **`predicted_landmark`**: Predicted Google Landmark ID.
- **`confidence`**: Softmax confidence of the predicted landmark.

### Confidence Threshold

The default detection threshold is `0.50`. You can configure it when calling the inference function:

```python
result = predict_landmark(
    "path/to/image.jpg",
    confidence_threshold=0.50
)
```

The threshold is configurable so the Risk Fusion module can adjust the operating point if required.

### Model Requirements

The trained model checkpoint must be available at:

`landmark-recognition/models/checkpoints/efficientnet_b0_regularized_best.pth`

The checkpoint is not included in Git because model weights are excluded by `.gitignore`. Anyone running inference must obtain the checkpoint separately.