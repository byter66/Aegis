"""EfficientNet-B0 transfer-learning model for the 50 landmark classes."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


NUM_CLASSES = 50


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout: float | None = None,
) -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    input_features = model.classifier[1].in_features
    if dropout is not None:
        model.classifier[0] = nn.Dropout(p=dropout)
    model.classifier[1] = nn.Linear(input_features, num_classes)
    return model
