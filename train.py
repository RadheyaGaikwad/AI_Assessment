
"""
Phase 2: Custom PyTorch Training and Tracking.

Trains a CNN for image-level classification and single-box regression.
For images containing multiple annotations, the largest box is selected.

Expected dataset contract:
    COCOHumanPoseDataset returns (image_tensor, target_dict).
    target_dict contains:
        boxes: Nx4 COCO boxes (x, y, width, height)
        labels: N class labels
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import COCOHumanPoseDataset, collate_fn


# -------------------- Configuration --------------------

SEED = 42
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 0.001
BOX_LOSS_WEIGHT = 5.0
IMAGE_SIZE = 640

ANNOTATIONS_PATH = Path("data/annotations/annotations.json")
IMAGE_DIR = Path("data/raw_images")
METRICS_PATH = Path("data/metrics.csv")
CHECKPOINT_PATH = Path("data/model_checkpoint.pth")


def set_seed(seed=SEED):
    """Set random seeds for more reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prefer reproducibility over speed where supported.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------- Model --------------------

class MultiTaskCustomPoseCNN(nn.Module):
    """CNN with classification and bounding-box regression heads."""

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.shared_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.classifier_head = nn.Linear(256, num_classes)

        # Predict normalized center coordinates and dimensions:
        # cx, cy, width, height — each in [0, 1].
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(self, images):
        features = self.features(images)
        features = self.shared_fc(features)

        class_logits = self.classifier_head(features)
        predicted_boxes = self.bbox_head(features)

        return class_logits, predicted_boxes


# -------------------- Target preparation --------------------

def prepare_batch_targets(targets_list, device):
    """
    Select the largest box per image and convert COCO xywh boxes
    into normalized center-format boxes: cx, cy, width, height.

    This is a single-object target strategy, not multi-object detection.
    """
    labels_batch = []
    boxes_batch = []

    for target in targets_list:
        boxes = torch.as_tensor(
            target["boxes"], dtype=torch.float32
        ).reshape(-1, 4)

        labels = torch.as_tensor(
            target["labels"], dtype=torch.long
        ).reshape(-1)

        if boxes.shape[0] == 0 or labels.shape[0] == 0:
            raise ValueError(
                "An image has no annotated boxes or labels. "
                "This training setup requires at least one object per image."
            )

        if boxes.shape[0] != labels.shape[0]:
            raise ValueError(
                "The number of boxes does not match the number of labels."
            )

        # Choose the largest object by box area.
        areas = boxes[:, 2] * boxes[:, 3]
        selected_idx = torch.argmax(areas)

        x, y, width, height = boxes[selected_idx]
        label = labels[selected_idx]

        # Convert COCO xywh to center-format coordinates.
        cx = x + width / 2.0
        cy = y + height / 2.0

        normalized_box = torch.stack([
            cx / IMAGE_SIZE,
            cy / IMAGE_SIZE,
            width / IMAGE_SIZE,
            height / IMAGE_SIZE,
        ])

        labels_batch.append(label)
        boxes_batch.append(normalized_box)

    labels_batch = torch.stack(labels_batch).to(device)
    boxes_batch = torch.stack(boxes_batch).to(device)

    return labels_batch, boxes_batch


# -------------------- Training --------------------

def train_model():
    set_seed()

    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Annotations file not found: {ANNOTATIONS_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    print("[INFO] Loading dataset...")

    dataset = COCOHumanPoseDataset(
        json_path=str(ANNOTATIONS_PATH),
        img_dir=str(IMAGE_DIR),
    )

    if len(dataset) < 2:
        raise ValueError("At least two dataset samples are required.")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    split_generator = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    loader_generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        generator=loader_generator,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Training images: {len(train_set)}")
    print(f"[INFO] Validation images: {len(val_set)}")

    model = MultiTaskCustomPoseCNN().to(device)

    classification_loss = nn.CrossEntropyLoss()
    box_loss_function = nn.SmoothL1Loss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.1,
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    with METRICS_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "train_cls_loss",
            "train_box_loss",
            "val_cls_loss",
            "val_box_loss",
            "learning_rate",
        ])

        for epoch in range(EPOCHS):
            # Training phase
            model.train()

            train_total = 0.0
            train_cls_total = 0.0
            train_box_total = 0.0
            train_batches = 0

            for images, targets_list in train_loader:
                images = images.to(device, dtype=torch.float32)

                labels, target_boxes = prepare_batch_targets(
                    targets_list, device
                )

                optimizer.zero_grad(set_to_none=True)

                class_logits, predicted_boxes = model(images)

                cls_loss = classification_loss(class_logits, labels)
                box_loss = box_loss_function(
                    predicted_boxes, target_boxes
                )

                total_loss = cls_loss + BOX_LOSS_WEIGHT * box_loss

                if not torch.isfinite(total_loss):
                    raise RuntimeError(
                        f"Non-finite training loss at epoch {epoch + 1}."
                    )

                # Backpropagation and parameter update.
                total_loss.backward()
                optimizer.step()

                train_total += total_loss.item()
                train_cls_total += cls_loss.item()
                train_box_total += box_loss.item()
                train_batches += 1

            avg_train_loss = train_total / max(1, train_batches)
            avg_train_cls = train_cls_total / max(1, train_batches)
            avg_train_box = train_box_total / max(1, train_batches)

            # Validation phase
            model.eval()

            val_total = 0.0
            val_cls_total = 0.0
            val_box_total = 0.0
            val_batches = 0

            with torch.no_grad():
                for images, targets_list in val_loader:
                    images = images.to(device, dtype=torch.float32)

                    labels, target_boxes = prepare_batch_targets(
                        targets_list, device
                    )

                    class_logits, predicted_boxes = model(images)

                    cls_loss = classification_loss(class_logits, labels)
                    box_loss = box_loss_function(
                        predicted_boxes, target_boxes
                    )

                    total_loss = (
                        cls_loss + BOX_LOSS_WEIGHT * box_loss
                    )

                    val_total += total_loss.item()
                    val_cls_total += cls_loss.item()
                    val_box_total += box_loss.item()
                    val_batches += 1

            avg_val_loss = val_total / max(1, val_batches)
            avg_val_cls = val_cls_total / max(1, val_batches)
            avg_val_box = val_box_total / max(1, val_batches)

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"train={avg_train_loss:.4f} | "
                f"val={avg_val_loss:.4f} | "
                f"cls={avg_val_cls:.4f} | "
                f"box={avg_val_box:.4f}"
            )

            writer.writerow([
                epoch + 1,
                avg_train_loss,
                avg_val_loss,
                avg_train_cls,
                avg_train_box,
                avg_val_cls,
                avg_val_box,
                current_lr,
            ])
            metrics_file.flush()

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "seed": SEED,
                        "num_classes": 2,
                        "image_size": IMAGE_SIZE,
                    },
                    CHECKPOINT_PATH,
                )

                print(
                    f"[INFO] Best checkpoint saved: {CHECKPOINT_PATH}"
                )

            scheduler.step()

    print("[INFO] Training complete.")
    print(f"[INFO] Metrics: {METRICS_PATH}")
    print(f"[INFO] Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train_model()
