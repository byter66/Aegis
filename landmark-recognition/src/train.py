"""Training loop for the landmark classifier; not run automatically on import."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from torch import nn

try:
    from .dataset import create_dataloaders, create_train_val_loaders, seed_everything
    from .model import build_model
except ImportError:
    from dataset import create_dataloaders, create_train_val_loaders, seed_everything
    from model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "training_history.csv"
FINETUNED_HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "training_history_finetuned.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
BASELINE_CHECKPOINT = CHECKPOINT_DIR / "efficientnet_b0_best.pth"
FINETUNED_CHECKPOINT = CHECKPOINT_DIR / "efficientnet_b0_finetuned_best.pth"
REGULARIZED_HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "training_history_regularized.csv"
REGULARIZED_CHECKPOINT = CHECKPOINT_DIR / "efficientnet_b0_regularized_best.pth"


def _accuracy(logits: torch.Tensor, labels: torch.Tensor, top_k: int = 1) -> int:
    predictions = logits.topk(top_k, dim=1).indices
    return int(predictions.eq(labels.unsqueeze(1)).any(dim=1).sum().item())


def _checkpoint_path() -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    base = CHECKPOINT_DIR / "efficientnet_b0_best.pth"
    if not base.exists():
        return base
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return CHECKPOINT_DIR / f"efficientnet_b0_best_{stamp}.pth"


def _set_batch_norm_eval(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


def fine_tune(
    epochs: int = 15,
    batch_size: int = 16,
    patience: int = 3,
    seed: int = 42,
) -> tuple[Path, list[dict[str, float | int]]]:
    """Fine-tune the final three EfficientNet feature blocks and classifier."""
    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Baseline checkpoint does not exist: {BASELINE_CHECKPOINT}")
    if FINETUNED_CHECKPOINT.exists():
        raise FileExistsError(f"Refusing to overwrite fine-tuned checkpoint: {FINETUNED_CHECKPOINT}")
    if FINETUNED_HISTORY_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite fine-tuned history: {FINETUNED_HISTORY_PATH}")

    seed_everything(seed)
    loaders, class_mapping = create_dataloaders(batch_size=batch_size, seed=seed)
    model = build_model(num_classes=len(class_mapping), pretrained=False, freeze_backbone=True)
    baseline = torch.load(BASELINE_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(baseline["model_state"])

    for block in model.features[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    unfrozen_layers = ["features[6]", "features[7]", "features[8]", "classifier"]

    frozen_parameters = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Frozen parameters: {frozen_parameters}")
    print(f"Trainable parameters: {trainable_parameters}")
    print(f"Unfrozen layers: {', '.join(unfrozen_layers)}")
    print("Backbone learning rate: 1e-05")
    print("Classification head learning rate: 1e-04")
    print(f"Batch size: {batch_size}")
    print(f"Maximum epochs: {epochs}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    backbone_parameters = [
        parameter for block in model.features[-3:] for parameter in block.parameters()
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 1e-5},
            {"params": list(model.classifier.parameters()), "lr": 1e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    best_accuracy = -1.0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        model.apply(_set_batch_norm_eval)
        train_loss = train_correct = train_total = 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += _accuracy(logits, labels)
            train_total += labels.size(0)

        model.eval()
        val_loss = val_correct = val_top5 = val_total = 0
        with torch.no_grad():
            for images, labels in loaders["val"]:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * labels.size(0)
                val_correct += _accuracy(logits, labels)
                val_top5 += _accuracy(logits, labels, top_k=5)
                val_total += labels.size(0)
        val_accuracy = val_correct / val_total
        scheduler.step(val_accuracy)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "val_loss": val_loss / val_total,
            "val_accuracy": val_accuracy,
            "val_top5_accuracy": val_top5 / val_total,
            "learning_rate": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs}: train_acc={row['train_accuracy']:.4f}, "
            f"val_acc={row['val_accuracy']:.4f}, val_top5={row['val_top5_accuracy']:.4f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_mapping": class_mapping,
                    "epoch": epoch,
                    "unfrozen_layers": unfrozen_layers,
                },
                FINETUNED_CHECKPOINT,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    FINETUNED_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FINETUNED_HISTORY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return FINETUNED_CHECKPOINT, history


def regularized_train(
    epochs: int = 15,
    batch_size: int = 16,
    patience: int = 4,
    seed: int = 42,
) -> tuple[Path, list[dict[str, float | int]]]:
    """Train a dropout and label-smoothed frozen-backbone baseline."""
    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Baseline checkpoint does not exist: {BASELINE_CHECKPOINT}")
    if REGULARIZED_CHECKPOINT.exists():
        raise FileExistsError(f"Refusing to overwrite regularized checkpoint: {REGULARIZED_CHECKPOINT}")
    if REGULARIZED_HISTORY_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite regularized history: {REGULARIZED_HISTORY_PATH}")

    seed_everything(seed)
    loaders, class_mapping = create_train_val_loaders(
        batch_size=batch_size,
        seed=seed,
        stronger_augmentation=True,
    )
    model = build_model(
        num_classes=len(class_mapping),
        pretrained=False,
        freeze_backbone=True,
        dropout=0.3,
    )
    baseline = torch.load(BASELINE_CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(baseline["model_state"])
    frozen_parameters = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Frozen parameters: {frozen_parameters}")
    print(f"Trainable parameters: {trainable_parameters}")
    print("Loss: CrossEntropyLoss(label_smoothing=0.1)")
    print("Dropout: 0.3 before final classifier")
    print("Augmentation: RandomResizedCrop(scale=0.70-1.0), HorizontalFlip, Rotation=12 degrees, ColorJitter=0.25")
    print("Learning rate: 1e-3")
    print("Weight decay: 1e-4")
    print(f"Batch size: {batch_size}")
    print(f"Maximum epochs: {epochs}")
    print("Test set used during training: False")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    best_accuracy = -1.0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_correct = train_total = 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += _accuracy(logits, labels)
            train_total += labels.size(0)

        model.eval()
        val_loss = val_correct = val_top5 = val_total = 0
        with torch.no_grad():
            for images, labels in loaders["val"]:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * labels.size(0)
                val_correct += _accuracy(logits, labels)
                val_top5 += _accuracy(logits, labels, top_k=5)
                val_total += labels.size(0)
        val_accuracy = val_correct / val_total
        scheduler.step(val_accuracy)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "val_loss": val_loss / val_total,
            "val_accuracy": val_accuracy,
            "val_top5_accuracy": val_top5 / val_total,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs}: train_acc={row['train_accuracy']:.4f}, "
            f"val_acc={row['val_accuracy']:.4f}, val_top5={row['val_top5_accuracy']:.4f}"
        )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_mapping": class_mapping,
                    "epoch": epoch,
                    "dropout": 0.3,
                    "label_smoothing": 0.1,
                },
                REGULARIZED_CHECKPOINT,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    REGULARIZED_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGULARIZED_HISTORY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return REGULARIZED_CHECKPOINT, history


def train(epochs: int = 10, batch_size: int = 32, patience: int = 3, seed: int = 42) -> Path:
    seed_everything(seed)
    loaders, class_mapping = create_dataloaders(batch_size=batch_size, seed=seed)
    model = build_model(num_classes=len(class_mapping), pretrained=True, freeze_backbone=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    best_accuracy = -1.0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    checkpoint_path = _checkpoint_path()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = train_correct = train_total = 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += _accuracy(logits, labels)
            train_total += labels.size(0)

        model.eval()
        val_loss = val_correct = val_top5 = val_total = 0
        with torch.no_grad():
            for images, labels in loaders["val"]:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * labels.size(0)
                val_correct += _accuracy(logits, labels)
                val_top5 += _accuracy(logits, labels, top_k=5)
                val_total += labels.size(0)
        val_accuracy = val_correct / val_total
        scheduler.step(val_accuracy)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / train_total,
            "train_accuracy": train_correct / train_total,
            "val_loss": val_loss / val_total,
            "val_accuracy": val_accuracy,
            "val_top5_accuracy": val_top5 / val_total,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            stale_epochs = 0
            torch.save({"model_state": model.state_dict(), "class_mapping": class_mapping, "epoch": epoch}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if HISTORY_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite training history: {HISTORY_PATH}")
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return checkpoint_path
