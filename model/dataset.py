"""Dataset and transform helpers for wheat disease training."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

from model.constants import (
    ALL_CLASS_NAMES,
    CLASS_NAMES,
    DISEASE_CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    STAGE1_CLASS_NAMES,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_train_transform() -> A.Compose:
    """Training-only augmentation pipeline for real-world robustness."""
    return A.Compose(
        [
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            A.RandomBrightnessContrast(
                brightness_limit=0.20,
                contrast_limit=0.20,
                p=0.50,
            ),
            A.HorizontalFlip(p=0.50),
            A.Rotate(limit=20, p=0.40),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.20),
            A.Blur(blur_limit=3, p=0.10),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_val_transform() -> A.Compose:
    """Validation transform that mirrors inference preprocessing."""
    return A.Compose(
        [
            A.Resize(INPUT_SIZE, INPUT_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class WheatDiseaseDataset(Dataset):
    """Custom dataset matching the expected data/train/<class> layout."""

    def __init__(
        self,
        root_dir: str | Path,
        class_names: Sequence[str] = CLASS_NAMES,
        transform: A.Compose | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.class_names = list(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.transform = transform
        self.samples = self._collect_samples()

        if not self.samples:
            raise ValueError(f"No image files found under {self.root_dir}")

    def _collect_samples(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root_dir}")

        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing class directory: {class_dir}")

            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((image_path, self.class_to_idx[class_name]))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        if self.transform is None:
            raise ValueError("A transform must be provided for WheatDiseaseDataset.")

        transformed = self.transform(image=image_np)
        tensor = transformed["image"]
        return tensor, label


class LegacyFlatWheatDiseaseDataset(Dataset):
    """Build stage-specific labels from the old flat train/val/test folder layout."""

    def __init__(
        self,
        root_dir: str | Path,
        stage: str,
        transform: A.Compose | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.stage = stage
        self.transform = transform

        if self.stage == "stage1":
            self.class_names = list(STAGE1_CLASS_NAMES)
        elif self.stage == "stage2":
            self.class_names = list(DISEASE_CLASS_NAMES)
        else:
            raise ValueError(f"Unsupported stage: {stage}")

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples = self._collect_samples()

        if not self.samples:
            raise ValueError(f"No image files found under {self.root_dir} for {self.stage}")

    def _map_class_name(self, class_name: str) -> str | None:
        if self.stage == "stage1":
            return "healthy" if class_name == "healthy" else "diseased"
        if class_name == "healthy":
            return None
        return class_name if class_name in self.class_to_idx else None

    def _collect_samples(self) -> list[tuple[Path, int]]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root_dir}")

        samples: list[tuple[Path, int]] = []
        available_dirs = {
            path.name.lower(): path
            for path in self.root_dir.iterdir()
            if path.is_dir()
        }

        missing = [name for name in ALL_CLASS_NAMES if name not in available_dirs]
        if missing:
            raise FileNotFoundError(
                f"Missing class directories in legacy flat dataset {self.root_dir}: {missing}"
            )

        for class_name, class_dir in sorted(available_dirs.items()):
            mapped_name = self._map_class_name(class_name)
            if mapped_name is None:
                continue

            for image_path in sorted(class_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append((image_path, self.class_to_idx[mapped_name]))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        if self.transform is None:
            raise ValueError("A transform must be provided for LegacyFlatWheatDiseaseDataset.")

        transformed = self.transform(image=image_np)
        tensor = transformed["image"]
        return tensor, label
