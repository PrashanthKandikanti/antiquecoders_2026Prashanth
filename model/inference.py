"""Inference helpers for upload gating and hierarchical wheat leaf diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model.constants import (
    DEFAULT_TOP_K,
    DISEASE_CLASS_NAMES,
    DISPLAY_NAMES,
    GATE_CHECKPOINT,
    GATE_CLASS_NAMES,
    GATE_REJECTION_THRESHOLD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    MAX_BRIGHTNESS_SCORE,
    MIN_BLUR_SCORE,
    MIN_BRIGHTNESS_SCORE,
    STAGE1_CHECKPOINT,
    STAGE1_CLASS_NAMES,
    STAGE1_HEALTHY_THRESHOLD,
    STAGE2_CHECKPOINT,
    STAGE2_CONFIDENCE_THRESHOLD,
)
from model.network import build_efficientnet_b0


class ModelNotReadyError(RuntimeError):
    """Raised when hierarchical inference weights are missing."""


def build_inference_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class HierarchicalPredictor:
    """Gate + hierarchical predictor used by the user upload flow."""

    def __init__(
        self,
        gate_path: Path = GATE_CHECKPOINT,
        stage1_path: Path = STAGE1_CHECKPOINT,
        stage2_path: Path = STAGE2_CHECKPOINT,
        device: str | None = None,
        gate_rejection_threshold: float = GATE_REJECTION_THRESHOLD,
        healthy_threshold: float = STAGE1_HEALTHY_THRESHOLD,
        disease_threshold: float = STAGE2_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.gate_path = Path(gate_path)
        self.stage1_path = Path(stage1_path)
        self.stage2_path = Path(stage2_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gate_rejection_threshold = gate_rejection_threshold
        self.healthy_threshold = healthy_threshold
        self.disease_threshold = disease_threshold
        self.transform = build_inference_transform()
        self._gate_bundle: tuple[torch.nn.Module, list[str]] | None = None
        self._stage1_bundle: tuple[torch.nn.Module, list[str]] | None = None
        self._stage2_bundle: tuple[torch.nn.Module, list[str]] | None = None

    def _load_bundle(
        self,
        checkpoint_path: Path,
        fallback_class_names: list[str],
    ) -> tuple[torch.nn.Module, list[str]]:
        if not checkpoint_path.exists():
            raise ModelNotReadyError(
                f"Missing checkpoint: {checkpoint_path}. "
                "Train the upload gate and wheat diagnosis models before using image upload."
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        class_names = list(checkpoint.get("class_names") or fallback_class_names)

        model = build_efficientnet_b0(num_classes=len(class_names), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model, class_names

    def _get_gate_bundle(self) -> tuple[torch.nn.Module, list[str]]:
        if self._gate_bundle is None:
            self._gate_bundle = self._load_bundle(self.gate_path, GATE_CLASS_NAMES)
        return self._gate_bundle

    def _get_stage1_bundle(self) -> tuple[torch.nn.Module, list[str]]:
        if self._stage1_bundle is None:
            self._stage1_bundle = self._load_bundle(self.stage1_path, STAGE1_CLASS_NAMES)
        return self._stage1_bundle

    def _get_stage2_bundle(self) -> tuple[torch.nn.Module, list[str]]:
        if self._stage2_bundle is None:
            self._stage2_bundle = self._load_bundle(self.stage2_path, DISEASE_CLASS_NAMES)
        return self._stage2_bundle

    def assess_image_quality(self, image: Image.Image) -> dict[str, Any]:
        image_array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_score = float(gray.mean())
        width, height = image.size

        issues: list[str] = []
        if blur_score < MIN_BLUR_SCORE:
            issues.append("blurry")
        if brightness_score < MIN_BRIGHTNESS_SCORE:
            issues.append("too dark")
        elif brightness_score > MAX_BRIGHTNESS_SCORE:
            issues.append("too bright")
        if min(width, height) < INPUT_SIZE:
            issues.append("low resolution")

        return {
            "ok": not issues,
            "issues": issues,
            "blur_score": round(blur_score, 2),
            "brightness_score": round(brightness_score, 2),
            "resolution": {"width": width, "height": height},
        }

    def _prepare_tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    @staticmethod
    def _top_predictions(
        probs: np.ndarray,
        class_names: list[str],
        topk: int,
    ) -> list[dict[str, Any]]:
        top_indices = probs.argsort()[::-1][:topk]
        return [
            {
                "label": DISPLAY_NAMES.get(class_names[index], class_names[index].title()),
                "code": class_names[index],
                "score": float(probs[index]),
            }
            for index in top_indices
        ]

    def _predict_probabilities(
        self,
        model: torch.nn.Module,
        image_tensor: torch.Tensor,
    ) -> np.ndarray:
        with torch.inference_mode():
            logits = model(image_tensor)
            return F.softmax(logits, dim=1).cpu().numpy()[0]

    def predict_image(self, image: Image.Image, topk: int = DEFAULT_TOP_K) -> dict[str, Any]:
        quality = self.assess_image_quality(image)
        if not quality["ok"]:
            return {
                "status": "reupload",
                "message": "Please upload a clearer image.",
                "reason": "The image is " + ", ".join(quality["issues"]) + ".",
                "quality": quality,
                "suggestions": [
                    "Take the photo in daylight.",
                    "Keep the leaf in sharp focus.",
                    "Move closer so the leaf fills most of the frame.",
                ],
            }

        image_tensor = self._prepare_tensor(image)

        gate_model, gate_class_names = self._get_gate_bundle()
        gate_probs = self._predict_probabilities(gate_model, image_tensor)
        gate_top = self._top_predictions(gate_probs, gate_class_names, topk=min(topk, len(gate_class_names)))
        gate_best = gate_top[0]

        if gate_best["code"] == "non_plant" and gate_best["score"] >= self.gate_rejection_threshold:
            return {
                "status": "invalid_subject",
                "message": "Please upload only plant leaf images.",
                "reason": "The uploaded image does not appear to be a plant leaf.",
                "quality": quality,
                "gate": {
                    "top_predictions": gate_top,
                },
            }

        if gate_best["code"] == "other_plant" and gate_best["score"] >= self.gate_rejection_threshold:
            return {
                "status": "unsupported_crop",
                "message": "We are working on support for leaves other than wheat.",
                "reason": "The uploaded image looks like a plant leaf, but not a wheat leaf.",
                "quality": quality,
                "gate": {
                    "top_predictions": gate_top,
                },
            }

        stage1_model, stage1_class_names = self._get_stage1_bundle()
        stage1_probs = self._predict_probabilities(stage1_model, image_tensor)
        stage1_top = self._top_predictions(stage1_probs, stage1_class_names, topk=2)

        healthy_score = next((item["score"] for item in stage1_top if item["code"] == "healthy"), 0.0)
        diseased_score = next((item["score"] for item in stage1_top if item["code"] == "diseased"), 0.0)

        if healthy_score >= self.healthy_threshold:
            return {
                "status": "ok",
                "decision": "healthy",
                "disease": DISPLAY_NAMES["healthy"],
                "disease_code": "healthy",
                "confidence": healthy_score,
                "quality": quality,
                "gate": {
                    "top_predictions": gate_top,
                },
                "stage1": {
                    "healthy_score": healthy_score,
                    "diseased_score": diseased_score,
                    "top_predictions": stage1_top,
                },
                "top_predictions": stage1_top,
            }

        stage2_model, stage2_class_names = self._get_stage2_bundle()
        stage2_probs = self._predict_probabilities(stage2_model, image_tensor)
        stage2_top = self._top_predictions(stage2_probs, stage2_class_names, topk=topk)
        best_prediction = stage2_top[0]

        status = "ok" if best_prediction["score"] >= self.disease_threshold else "uncertain"
        return {
            "status": status,
            "decision": "diseased",
            "disease": best_prediction["label"],
            "disease_code": best_prediction["code"],
            "confidence": best_prediction["score"],
            "quality": quality,
            "gate": {
                "top_predictions": gate_top,
            },
            "stage1": {
                "healthy_score": healthy_score,
                "diseased_score": diseased_score,
                "top_predictions": stage1_top,
            },
            "stage2": {
                "top_predictions": stage2_top,
            },
            "top_predictions": stage2_top,
            "message": (
                "Prediction confidence is low. Please upload a clearer image."
                if status == "uncertain"
                else "Prediction completed."
            ),
        }

    def predict_file(self, image_path: str | Path, topk: int = DEFAULT_TOP_K) -> dict[str, Any]:
        with Image.open(image_path) as image:
            return self.predict_image(image.convert("RGB"), topk=topk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run upload gating and hierarchical inference for one image.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--json", action="store_true", help="Print the full response as JSON.")
    args = parser.parse_args()

    predictor = HierarchicalPredictor()
    result = predictor.predict_file(args.image, topk=args.topk)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"Status: {result['status']}")
    print(f"Disease: {result.get('disease', 'Unknown')}")
    print(f"Confidence: {result.get('confidence', 0.0):.4f}")
    for prediction in result.get("top_predictions", []):
        print(f" - {prediction['label']}: {prediction['score']:.4f}")


if __name__ == "__main__":
    main()
