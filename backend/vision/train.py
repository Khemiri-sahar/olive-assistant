"""
vision/train.py - Train MobileNetV3 on the 3-class olive leaf disease dataset.

Dataset layout (dataset/ already downloaded):
    dataset/
        train/
            Healthy/
            aculus_olearius/
            olive_peacock_spot/
        test/
            Healthy/
            aculus_olearius/
            olive_peacock_spot/

Usage:
    cd /home/tarek-gritli/Desktop/olive-assistant
    source venv/bin/activate
    python -m backend.vision.train --data_dir ./dataset --output ./models/olive_cnn.pth --epochs 20 --batch 32
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.datasets as datasets
import numpy as np
from sklearn.metrics import classification_report

from vision.model import build_model, TRAIN_TRANSFORM, INFERENCE_TRANSFORM, NUM_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sampler(dataset) -> WeightedRandomSampler:
    class_counts = np.bincount([s[1] for s in dataset.samples])
    weights = 1.0 / class_counts
    sample_weights = [weights[s[1]] for s in dataset.samples]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if (batch_idx + 1) % 20 == 0:
            logger.info(
                f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                f"loss={running_loss/(batch_idx+1):.4f} acc={correct/total:.3f}"
            )
    return running_loss / len(loader), correct / total


@torch.inference_mode()
def score_on_test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    return running_loss / len(loader), acc, all_preds, all_labels


def run_training(
    data_dir: Path,
    output: Path,
    epochs: int = 20,
    batch: int = 32,
    lr: float = 1e-4,
    freeze_backbone: bool = False,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=TRAIN_TRANSFORM)
    test_ds  = datasets.ImageFolder(str(data_dir / "test"),  transform=INFERENCE_TRANSFORM)

    logger.info(f"Train: {len(train_ds)} | Test: {len(test_ds)}")
    logger.info(f"Class mapping: {train_ds.class_to_idx}")

    assert len(train_ds.classes) == NUM_CLASSES, (
        f"Dataset has {len(train_ds.classes)} classes but model expects {NUM_CLASSES}. "
        f"Found: {train_ds.classes}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch, sampler=get_sampler(train_ds), num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=4)

    model = build_model(num_classes=NUM_CLASSES, pretrained=True)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen - training head only")

    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc, preds, labels = score_on_test(model, test_loader, criterion, device)
        scheduler.step()
        logger.info(
            f"Epoch {epoch:02d}/{epochs} | train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} | lr={scheduler.get_last_lr()[0]:.2e} | "
            f"time={time.time()-t0:.1f}s"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_to_idx": train_ds.class_to_idx,
            }, output)
            logger.info(f"  Best model saved (val_acc={val_acc:.3f})")

    logger.info("\nFinal Classification Report:")
    class_names = [train_ds.classes[i] for i in range(len(train_ds.classes))]
    print(classification_report(labels, preds, target_names=class_names))
    logger.info(f"Best val acc: {best_val_acc:.4f} | Saved to: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output",   type=Path, default=Path("models/olive_cnn.pth"))
    parser.add_argument("--epochs",   type=int,  default=20)
    parser.add_argument("--batch",    type=int,  default=32)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--freeze_backbone", action="store_true")
    args = parser.parse_args()
    run_training(
        data_dir=args.data_dir,
        output=args.output,
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        freeze_backbone=args.freeze_backbone,
    )


if __name__ == "__main__":
    main()
