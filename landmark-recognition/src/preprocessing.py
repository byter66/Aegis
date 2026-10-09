"""Shared image preprocessing configuration for landmark recognition."""

from torchvision import transforms

try:
    from .dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    from dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


EVALUATION_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def get_inference_transform():
    """Return deterministic preprocessing for validation and inference."""
    return EVALUATION_TRANSFORM