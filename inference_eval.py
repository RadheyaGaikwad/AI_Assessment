
"""
Phase 2: Inference and Evaluation.

Evaluates the custom CNN using the same deterministic split as train.py.
Reports classification accuracy and normalized bounding-box errors.
Saves a visualization of one prediction.
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset import COCOHumanPoseDataset, collate_fn
from train import (
    IMAGE_SIZE,
    SEED,
    MultiTaskCustomPoseCNN,
    prepare_batch_targets,
)


ANNOTATIONS_PATH = Path("data/annotations/annotations.json")
IMAGE_DIR = Path("data/raw_images")
CHECKPOINT_PATH = Path("data/model_checkpoint.pth")
OUTPUT_DIR = Path("data/outputs")


def evaluate_and_predict():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Evaluation device: {device}")

    if not ANNOTATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Annotations file not found: {ANNOTATIONS_PATH}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Run train.py first."
        )

    dataset = COCOHumanPoseDataset(
        json_path=str(ANNOTATIONS_PATH),
        img_dir=str(IMAGE_DIR),
    )

    if len(dataset) < 2:
        raise ValueError("At least two dataset samples are required.")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size

    # Must match the split created in train.py.
    split_generator = torch.Generator().manual_seed(SEED)
    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model = MultiTaskCustomPoseCNN(
        num_classes=checkpoint.get("num_classes", 2)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"[INFO] Loaded checkpoint from epoch "
        f"{checkpoint.get('epoch', 'unknown')}"
    )
    print(f"[INFO] Validation images: {len(val_dataset)}")

    correct = 0
    total = 0
    box_errors = []

    with torch.no_grad():
        for images, targets_list in val_loader:
            images = images.to(device, dtype=torch.float32)
            labels, target_boxes = prepare_batch_targets(
                targets_list, device
            )

            class_logits, predicted_boxes = model(images)
            predicted_labels = class_logits.argmax(dim=1)

            correct += (predicted_labels == labels).sum().item()
            total += labels.size(0)

            # MAE for normalized center-format box coordinates.
            errors = torch.abs(predicted_boxes - target_boxes)
            box_errors.extend(errors.cpu().numpy().tolist())

    accuracy = 100.0 * correct / total if total else 0.0
    mean_box_mae = (
        float(np.mean(box_errors)) if box_errors else 0.0
    )

    print(f"\n[RESULT] Classification accuracy: {accuracy:.2f}%")
    print(f"[RESULT] Correct predictions: {correct}/{total}")
    print(
        "[RESULT] Normalized bounding-box coordinate MAE: "
        f"{mean_box_mae:.4f}"
    )
    print(
        "[NOTE] Box MAE is not IoU or mAP. Lower is better, "
        "but it does not alone establish detection quality."
    )

    # Visualize one validation sample.
    image_tensor, target = val_dataset[0]
    input_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        class_logits, predicted_boxes = model(input_tensor)
        predicted_class = int(class_logits.argmax(dim=1).item())
        cx, cy, width, height = (
            predicted_boxes[0].cpu().numpy() * IMAGE_SIZE
        )

    # Convert predicted center format to corner coordinates.
    x1 = int(round(cx - width / 2))
    y1 = int(round(cy - height / 2))
    x2 = int(round(cx + width / 2))
    y2 = int(round(cy + height / 2))

    # Recover image for visualization.
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image_height, image_width = image_bgr.shape[:2]
    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    cv2.rectangle(
        image_bgr,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        image_bgr,
        f"Predicted class: {predicted_class}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sample_prediction.jpg"

    if not cv2.imwrite(str(output_path), image_bgr):
        raise RuntimeError(f"Failed to save prediction image: {output_path}")

    print(f"[INFO] Prediction visualization saved: {output_path}")


if __name__ == "__main__":
    evaluate_and_predict()
