"""Datasets, deterministic class mapping, and transforms for landmark training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
MAPPING_PATH = SPLITS_DIR / "class_mapping.json"
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_class_mapping(split_paths: tuple[Path, ...] | None = None) -> tuple[dict[str, int], dict[int, str]]:
    paths = split_paths or tuple(SPLITS_DIR / f"{name}.csv" for name in ("train", "val", "test"))
    landmark_ids: set[str] = set()
    for path in paths:
        import csv

        with path.open("r", encoding="utf-8", newline="") as handle:
            landmark_ids.update(row["landmark_id"] for row in csv.DictReader(handle))
    ordered_ids = sorted(landmark_ids, key=lambda value: int(value))
    landmark_to_index = {landmark_id: index for index, landmark_id in enumerate(ordered_ids)}
    index_to_landmark = {index: landmark_id for landmark_id, index in landmark_to_index.items()}
    mapping = {"landmark_to_index": landmark_to_index, "index_to_landmark": index_to_landmark}
    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MAPPING_PATH.exists():
        existing = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
        existing_normalized = {
            "landmark_to_index": {
                str(key): int(value)
                for key, value in existing["landmark_to_index"].items()
            },
            "index_to_landmark": {
                int(key): str(value)
                for key, value in existing["index_to_landmark"].items()
            },
        }
        if existing_normalized != mapping:
            raise ValueError(f"Existing class mapping disagrees with split labels: {MAPPING_PATH}")
    else:
        MAPPING_PATH.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return landmark_to_index, index_to_landmark


def load_class_mapping() -> tuple[dict[str, int], dict[int, str]]:
    if not MAPPING_PATH.is_file():
        return build_class_mapping()
    data = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    landmark_to_index = {str(key): int(value) for key, value in data["landmark_to_index"].items()}
    index_to_landmark = {int(key): str(value) for key, value in data["index_to_landmark"].items()}
    return landmark_to_index, index_to_landmark


def get_transforms(training: bool, stronger: bool = False) -> Callable[[Image.Image], Tensor]:
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.70 if stronger else 0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(12 if stronger else 10),
                transforms.ColorJitter(
                    brightness=0.25 if stronger else 0.2,
                    contrast=0.25 if stronger else 0.2,
                    saturation=0.25 if stronger else 0.2,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class LandmarkDataset(Dataset[tuple[Tensor, int]]):
    def __init__(
        self,
        csv_path: Path,
        class_mapping: dict[str, int],
        training: bool = False,
        stronger_augmentation: bool = False,
    ) -> None:
        import csv

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        self.class_mapping = class_mapping
        self.transform = get_transforms(training, stronger=stronger_augmentation)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.rows[index]
        image = Image.open(PROJECT_ROOT / row["image_path"]).convert("RGB")
        return self.transform(image), self.class_mapping[row["landmark_id"]]


def create_dataloaders(batch_size: int = 32, seed: int = 42) -> tuple[dict[str, DataLoader[tuple[Tensor, int]]], dict[str, int]]:
    class_mapping, _ = build_class_mapping()
    generator = torch.Generator().manual_seed(seed)
    datasets = {
        "train": LandmarkDataset(SPLITS_DIR / "train.csv", class_mapping, training=True),
        "val": LandmarkDataset(SPLITS_DIR / "val.csv", class_mapping),
        "test": LandmarkDataset(SPLITS_DIR / "test.csv", class_mapping),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, generator=generator),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }
    return loaders, class_mapping


def create_train_val_loaders(
    batch_size: int = 32,
    seed: int = 42,
    stronger_augmentation: bool = False,
) -> tuple[dict[str, DataLoader[tuple[Tensor, int]]], dict[str, int]]:
    class_mapping, _ = load_class_mapping()
    generator = torch.Generator().manual_seed(seed)
    datasets = {
        "train": LandmarkDataset(
            SPLITS_DIR / "train.csv",
            class_mapping,
            training=True,
            stronger_augmentation=stronger_augmentation,
        ),
        "val": LandmarkDataset(SPLITS_DIR / "val.csv", class_mapping),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, generator=generator),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
    }
    return loaders, class_mapping


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
