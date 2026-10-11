
"""
Final Verification Script
Project: Augle AI Assessment

Checks:
1. COCO annotation JSON structure
2. Dataset loading and image preprocessing
3. Image tensor shape and values
4. Bounding-box and label validity
5. Dataset consistency and summary statistics
"""

import json
from collections import Counter
from pathlib import Path

import torch

from dataset import COCOHumanPoseDataset


ANNOTATION_PATH = Path("data/annotations/annotations.json")
IMAGE_DIR = Path("data/raw_images")
IMAGE_SIZE = (640, 640)


def check_annotation_json():
    """Validate the main COCO JSON structure."""
    print("\n" + "=" * 60)
    print("1. ANNOTATION JSON CHECK")
    print("=" * 60)

    if not ANNOTATION_PATH.is_file():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_PATH}"
        )

    with ANNOTATION_PATH.open("r", encoding="utf-8") as file:
        coco = json.load(file)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    image_ids = {item.get("id") for item in images}
    category_ids = {item.get("id") for item in categories}

    missing_image_refs = 0
    missing_category_refs = 0
    invalid_boxes = 0

    for ann in annotations:
        if ann.get("image_id") not in image_ids:
            missing_image_refs += 1

        if category_ids and ann.get("category_id") not in category_ids:
            missing_category_refs += 1

        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            invalid_boxes += 1
            continue

        try:
            x, y, w, h = map(float, bbox)
            if not all(torch.isfinite(torch.tensor(v)).item()
                       for v in (x, y, w, h)):
                invalid_boxes += 1
            elif w <= 0 or h <= 0:
                invalid_boxes += 1
        except (TypeError, ValueError):
            invalid_boxes += 1

    print(f"Images in JSON:             {len(images)}")
    print(f"Annotations in JSON:        {len(annotations)}")
    print(f"Categories in JSON:          {len(categories)}")
    print(f"Missing image references:   {missing_image_refs}")
    print(f"Missing category references: {missing_category_refs}")
    print(f"Invalid bounding boxes:     {invalid_boxes}")

    if not categories:
        print(
            "[WARNING] No categories list found; "
            "the dataset may infer labels from annotations."
        )

    return coco


def check_image_files(coco):
    """Check whether annotation-referenced image files exist."""
    print("\n" + "=" * 60)
    print("2. IMAGE FILE CHECK")
    print("=" * 60)

    missing_files = []
    existing_files = 0

    for info in coco.get("images", []):
        filename = info.get("file_name")
        if not filename:
            missing_files.append("<missing file_name>")
            continue

        image_path = IMAGE_DIR / filename

        if image_path.is_file():
            existing_files += 1
        else:
            missing_files.append(str(image_path))

    print(f"Existing image files: {existing_files}")
    print(f"Missing image files:  {len(missing_files)}")

    for path in missing_files[:10]:
        print(f"  MISSING: {path}")

    if len(missing_files) > 10:
        print(f"  ... and {len(missing_files) - 10} more")

    return len(missing_files) == 0


def check_dataset():
    """Load samples through the actual PyTorch dataset class."""
    print("\n" + "=" * 60)
    print("3. PYTORCH DATASET CHECK")
    print("=" * 60)

    dataset = COCOHumanPoseDataset(
        json_path=str(ANNOTATION_PATH),
        img_dir=str(IMAGE_DIR),
        image_size=IMAGE_SIZE,
    )

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("Dataset contains no images.")

    class_counts = Counter()
    empty_target_images = 0
    invalid_target_boxes = 0
    checked_images = 0

    # Check every sample so errors are not hidden in later batches.
    for index in range(len(dataset)):
        image, target = dataset[index]
        checked_images += 1

        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Sample {index}: image is not a tensor.")

        if image.shape != (3, IMAGE_SIZE[1], IMAGE_SIZE[0]):
            raise ValueError(
                f"Sample {index}: unexpected image shape {tuple(image.shape)}"
            )

        if not torch.isfinite(image).all():
            raise ValueError(f"Sample {index}: image has NaN or infinity.")

        if image.min().item() < 0 or image.max().item() > 1:
            raise ValueError(
                f"Sample {index}: image values are outside [0, 1]."
            )

        boxes = target.get("boxes")
        labels = target.get("labels")

        if boxes is None or labels is None:
            raise ValueError(
                f"Sample {index}: target must contain boxes and labels."
            )

        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(
                f"Sample {index}: boxes must have shape (N, 4)."
            )

        if labels.ndim != 1 or len(labels) != len(boxes):
            raise ValueError(
                f"Sample {index}: labels and boxes have inconsistent lengths."
            )

        if len(boxes) == 0:
            empty_target_images += 1

        if not torch.isfinite(boxes).all():
            raise ValueError(
                f"Sample {index}: bounding boxes contain NaN or infinity."
            )

        # The dataset returns scaled COCO xywh: x, y, width, height.
        for x, y, width, height in boxes.tolist():
            if width <= 0 or height <= 0:
                invalid_target_boxes += 1

            if (
                x < 0
                or y < 0
                or x + width > IMAGE_SIZE[0] + 1e-4
                or y + height > IMAGE_SIZE[1] + 1e-4
            ):
                invalid_target_boxes += 1

        for label in labels.tolist():
            class_counts[label] += 1

        if index == 0:
            print(f"First image tensor shape: {tuple(image.shape)}")
            print(f"First image pixel range: "
                  f"{image.min().item():.3f} to {image.max().item():.3f}")
            print(f"First image boxes shape: {tuple(boxes.shape)}")
            print(f"First image labels shape: {tuple(labels.shape)}")

    print(f"Samples checked:             {checked_images}")
    print(f"Images with no boxes:        {empty_target_images}")
    print(f"Invalid target box checks:   {invalid_target_boxes}")
    print(f"Label counts (mapped IDs):   {dict(class_counts)}")

    return {
        "dataset_size": len(dataset),
        "checked_images": checked_images,
        "empty_target_images": empty_target_images,
        "invalid_target_boxes": invalid_target_boxes,
    }


def check_training_artifacts():
    """Report whether training and evaluation outputs exist."""
    print("\n" + "=" * 60)
    print("4. TRAINING ARTIFACT CHECK")
    print("=" * 60)

    artifacts = [
        Path("data/metrics.csv"),
        Path("data/model_checkpoint.pth"),
        Path("data/outputs/sample_prediction.jpg"),
    ]

    for path in artifacts:
        status = "FOUND" if path.is_file() else "NOT FOUND"
        print(f"[{status}] {path}")


def main():
    print("=" * 60)
    print("AUGLE AI ASSESSMENT - FINAL VERIFICATION")
    print("=" * 60)

    try:
        coco = check_annotation_json()
        images_ok = check_image_files(coco)
        dataset_result = check_dataset()
        check_training_artifacts()

        print("\n" + "=" * 60)
        print("5. FINAL SUMMARY")
        print("=" * 60)

        checks_passed = (
            images_ok
            and dataset_result["checked_images"] == dataset_result["dataset_size"]
            and dataset_result["empty_target_images"] == 0
            and dataset_result["invalid_target_boxes"] == 0
        )

        if checks_passed:
            print("[PASS] Dataset and image checks completed.")
        else:
            print("[WARNING] One or more dataset checks need attention.")

        print("Verification completed.")

    except Exception as exc:
        print("\n[FAIL] Verification stopped because of an error.")
        print(f"Reason: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
