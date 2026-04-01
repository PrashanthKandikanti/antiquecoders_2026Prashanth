"""Shared constants used by hierarchical training and inference."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
DATA_DIR = TRAINING_DIR / "data" / "processed"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"
ARTIFACTS_DIR = TRAINING_DIR / "artifacts"

STAGE1_DATA_DIR = DATA_DIR / "stage1"
STAGE2_DATA_DIR = DATA_DIR / "stage2"
LEGACY_TRAIN_DIR = DATA_DIR / "train"
LEGACY_VAL_DIR = DATA_DIR / "val"
LEGACY_TEST_DIR = DATA_DIR / "test"

STAGE1_CLASS_NAMES = ["healthy", "diseased"]
DISEASE_CLASS_NAMES = ["rust", "blight", "mildew", "spot"]
ALL_CLASS_NAMES = ["healthy", *DISEASE_CLASS_NAMES]

# Backward-compatible alias for the complete label space.
CLASS_NAMES = ALL_CLASS_NAMES

DISPLAY_NAMES = {
    "healthy": "Healthy",
    "diseased": "Diseased",
    "rust": "Rust",
    "blight": "Leaf Blight",
    "mildew": "Powdery Mildew",
    "spot": "Spot Blotch",
}

INPUT_SIZE = 224
DEFAULT_TOP_K = 3

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STAGE1_HEALTHY_THRESHOLD = 0.80
STAGE2_CONFIDENCE_THRESHOLD = 0.65

MIN_BLUR_SCORE = 80.0
MIN_BRIGHTNESS_SCORE = 40.0
MAX_BRIGHTNESS_SCORE = 215.0

STAGE1_CHECKPOINT = CHECKPOINT_DIR / "stage1_best.pth"
STAGE2_CHECKPOINT = CHECKPOINT_DIR / "stage2_best.pth"
STAGE1_METRICS_PATH = ARTIFACTS_DIR / "stage1_training_metrics.json"
STAGE2_METRICS_PATH = ARTIFACTS_DIR / "stage2_training_metrics.json"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
