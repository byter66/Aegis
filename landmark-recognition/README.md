# Landmark Recognition

The Landmark Recognition module is part of **Aegis — AI-Powered Pre-Upload Image Risk Detection & Privacy Leak Prevention**.

It identifies potential landmarks in an image and provides a predicted landmark ID and confidence score to the Risk Fusion module.

## Model Architecture

- **Model:** EfficientNet-B0
- **Approach:** Transfer learning
- **Number of classes:** 50
- **Training strategy:** Frozen-backbone baseline with regularized training
- **Framework:** PyTorch and Torchvision

## Dataset

The model was developed using a selected subset of Google Landmarks v2 metadata and landmark images collected for the project.

The final dataset contains approximately 955 images across 50 landmark classes.

The dataset was divided into training, validation, and test sets using a reproducible split with random seed 42.

Dataset files and trained model checkpoints are excluded from Git and must be obtained separately.

## Model Evaluation

The regularized EfficientNet-B0 model achieved the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Test images | 142 |
| Landmark classes | 50 |
| Top-1 accuracy | 45.07% |
| Top-5 accuracy | 72.54% |

These results represent the current experimental baseline. Predictions should be treated as uncertain, particularly for unfamiliar landmarks.

Detailed results are available in `evaluation/experiment3/`, including the classification report, confusion matrix, per-class metrics, and test analysis.

## Inference

The inference interface is implemented in `src/inference.py`.

From the `landmark-recognition` directory, run:

```python
from src.inference import predict_landmark

result = predict_landmark("path/to/image.jpg")
print(result)
```

### Example output

```json
{
  "landmark_detected": true,
  "predicted_landmark": "120734",
  "confidence": 0.8051
}
```

The example values illustrate the output format; actual predictions depend on the input image.

### Output fields

- `landmark_detected`: Indicates whether the prediction meets the configured confidence threshold.
- `predicted_landmark`: Predicted Google Landmark ID.
- `confidence`: Softmax confidence score for the predicted class.

The default confidence threshold is `0.50`. It can be configured when calling the inference function and is an initial threshold, not a calibrated optimum.

## Model Checkpoint

Inference requires the trained checkpoint:

`models/checkpoints/efficientnet_b0_regularized_best.pth`

The checkpoint is not included in Git. Obtain it from the project team and place it at the path above before running inference.

## Running the Integration Example

From the `landmark-recognition` directory:

```bash
python integration_example.py
```

Ensure the required checkpoint and a valid sample image are available before running the example.

## Limitations

- The model recognizes only the 50 classes represented in its training dataset.
- The predicted ID is not itself a human-readable landmark name or geographic coordinate.
- A high confidence score does not guarantee that a prediction is correct.
- Landmark predictions should be combined with other privacy signals by the Aegis Risk Fusion module.

## Project Integration

The module provides the following fields to the Risk Fusion component:

- `landmark_detected`
- `predicted_landmark`
- `confidence`

These outputs can be combined with OCR/PII detection and other privacy indicators to support the overall image-risk assessment.