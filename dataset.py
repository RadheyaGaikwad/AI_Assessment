
"""
Phase 2: COCO dataset loading and image geometry validation.

Images are resized to a fixed size. COCO xywh boxes are scaled
using the same independent x/y scale factors as the images.

The dataset returns all valid instances per image.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class COCOHumanPoseDataset(Dataset):
    def __init__(
        self,
        json_path="data/annotations/annotations.json",
        img_dir="data/raw_images",
        image_size=(640, 640),
    ):
        self.img_dir = Path(img_dir)
        self.image_size = tuple(image_size)

        if len(self.image_size) != 2 or min(self.image_size) <= 0:
            raise ValueError("image_size must be (width, height), both positive.")

        json_path = Path(json_path)
        if not json_path.is_file():
            raise FileNotFoundError(
                f"COCO annotation file not found: {json_path}"
            )

        with json_path.open("r", encoding="utf-8") as file:
            self.coco = json.load(file)

        self.images = self.coco.get("images", [])
        annotations = self.coco.get("annotations", [])
        categories = self.coco.get("categories", [])

        if not self.images:
            raise ValueError("COCO JSON contains no images.")

        # Verify unique image IDs.
        image_ids = [item.get("id") for item in self.images]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("Duplicate image IDs found in COCO JSON.")

        self.image_info_by_id = {
            item["id"]: item for item in self.images
        }

        # Map arbitrary COCO category IDs to contiguous class indices.
        category_ids = sorted(
            category["id"] for category in categories
        )
        if not category_ids:
            category_ids = sorted({
                int(annotation.get("category_id", 1))
                for annotation in annotations
            })

        if not category_ids:
            raise ValueError("No categories or annotated category IDs found.")

        self.category_to_label = {
            category_id: index
            for index, category_id in enumerate(category_ids)
        }

        self.label_to_category = {
            index: category_id
            for category_id, index in self.category_to_label.items()
        }

        # Group annotations by image ID.
        self.annotations_by_image = {}
        self.invalid_annotation_count = 0

        for annotation in annotations:
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")

            if image_id not in self.image_info_by_id:
                self.invalid_annotation_count += 1
                continue

            if category_id not in self.category_to_label:
                self.invalid_annotation_count += 1
                continue

            self.annotations_by_image.setdefault(
                image_id, []
            ).append(annotation)

        self.clipped_box_count = 0
        self.skipped_box_count = 0
        self.empty_target_count = 0

        print(f"Loaded {len(self.images)} images.")
        print(f"Loaded {len(annotations)} annotations.")
        print(f"Categories: {len(self.category_to_label)}")
        print(
            "Invalid annotations with unknown image/category: "
            f"{self.invalid_annotation_count}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        image_id = info["id"]
        image_path = self.img_dir / info["file_name"]

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(
                f"Image missing or unreadable: {image_path}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]

        # Compare actual image dimensions with COCO metadata.
        declared_width = info.get("width")
        declared_height = info.get("height")

        if (
            declared_width is not None
            and declared_height is not None
            and (
                int(declared_width) != original_width
                or int(declared_height) != original_height
            )
        ):
            raise ValueError(
                f"Image dimension mismatch for {info['file_name']}: "
                f"JSON={declared_width}x{declared_height}, "
                f"actual={original_width}x{original_height}"
            )

        target_width, target_height = self.image_size

        scale_x = target_width / original_width
        scale_y = target_height / original_height

        image = cv2.resize(
            image,
            (target_width, target_height),
            interpolation=cv2.INTER_LINEAR,
        )

        boxes = []
        labels = []

        for annotation in self.annotations_by_image.get(image_id, []):
            bbox = annotation.get("bbox")

            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                self.skipped_box_count += 1
                continue

            try:
                x, y, width, height = map(float, bbox)
            except (TypeError, ValueError):
                self.skipped_box_count += 1
                continue

            if not np.isfinite([x, y, width, height]).all():
                self.skipped_box_count += 1
                continue

            if width <= 0 or height <= 0:
                self.skipped_box_count += 1
                continue

            # Original COCO box corners.
            original_x2 = x + width
            original_y2 = y + height

            # Clip in original image coordinates first.
            clipped_x1 = max(0.0, min(x, float(original_width)))
            clipped_y1 = max(0.0, min(y, float(original_height)))
            clipped_x2 = max(
                0.0, min(original_x2, float(original_width))
            )
            clipped_y2 = max(
                0.0, min(original_y2, float(original_height))
            )

            if (
                clipped_x1 != x
                or clipped_y1 != y
                or clipped_x2 != original_x2
                or clipped_y2 != original_y2
            ):
                self.clipped_box_count += 1

            if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
                self.skipped_box_count += 1
                continue

            # Scale corners using the same transform as the image.
            x1 = clipped_x1 * scale_x
            y1 = clipped_y1 * scale_y
            x2 = clipped_x2 * scale_x
            y2 = clipped_y2 * scale_y

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            labels.append(
                self.category_to_label[annotation["category_id"]]
            )

        if not boxes:
            self.empty_target_count += 1

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image)
        ).permute(2, 0, 1).float() / 255.0

        target = {
            "boxes": torch.tensor(
                boxes, dtype=torch.float32
            ).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor(image_id, dtype=torch.long),
        }

        return image_tensor, target


def collate_fn(batch):
    """Stack images and retain variable-length targets."""
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
