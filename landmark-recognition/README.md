# Landmark Recognition

This module will recognize selected landmark classes in images for the
privacy-risk detection project.

## Planned components

- Dataset preparation for a manageable Google Landmarks v2 subset
- Image preprocessing compatible with pretrained ResNet50
- Transfer-learning model training and evaluation
- Saved-model loading and single-image inference

The implementation is intentionally scaffolded only. No dataset is included,
and training has not been implemented yet.

## Layout

```text
data/raw/        Source images or dataset material (not committed)
data/processed/  Optional processed data (not committed)
data/splits/     Future train, validation, and test split metadata
src/             Landmark-recognition source modules
checkpoints/     Saved model files (not committed)
notebooks/       Exploratory notebooks
```

## Expected workflow

1. Prepare a small, documented landmark subset.
2. Create reproducible train, validation, and test splits.
3. Implement preprocessing and the ResNet50 transfer-learning model.
4. Train and evaluate the model.
5. Load a checkpoint for inference and return a structured prediction.
