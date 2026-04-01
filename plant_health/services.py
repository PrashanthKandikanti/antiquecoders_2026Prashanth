"""Services for turning uploaded images into farmer-friendly diagnosis responses."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from PIL import Image, UnidentifiedImageError

from model.inference import HierarchicalPredictor, ModelNotReadyError
from plant_health.knowledge import get_disease_guidance


@lru_cache(maxsize=1)
def get_predictor() -> HierarchicalPredictor:
    return HierarchicalPredictor()


def _confidence_percent(score: float | None) -> str:
    if score is None:
        return "Unknown"
    return f"{round(float(score) * 100)}%"


def enrich_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    disease_code = prediction.get("disease_code")
    guidance = get_disease_guidance(disease_code)

    if prediction["status"] in {"ok", "uncertain"}:
        prediction = {
            **prediction,
            "confidence_percent": _confidence_percent(prediction.get("confidence")),
            "explanation": guidance.get(
                "explanation",
                "The model completed the analysis, but there is no guidance text for this label yet.",
            ),
            "treatment": guidance.get("treatment", []),
            "organic_treatment": guidance.get("organic_treatment", []),
            "chemical_treatment": guidance.get("chemical_treatment", []),
            "prevention": guidance.get("prevention", []),
            "monitoring": guidance.get("monitoring", ""),
            "safety_note": guidance.get("safety_note", ""),
        }

    return prediction


def diagnose_uploaded_image(uploaded_file) -> dict[str, Any]:
    try:
        with Image.open(uploaded_file) as image:
            prediction = get_predictor().predict_image(image.convert("RGB"))
    except UnidentifiedImageError:
        return {
            "status": "error",
            "message": "The uploaded file is not a readable image.",
        }
    except ModelNotReadyError as exc:
        return {
            "status": "model_not_ready",
            "message": "The upload gate or wheat diagnosis models are not trained yet.",
            "reason": str(exc),
        }

    return enrich_prediction(prediction)


def format_prediction_for_chat(prediction: dict[str, Any]) -> str:
    status = prediction.get("status")

    if status == "model_not_ready":
        return (
            "The upload pipeline is ready, but the model weights are missing. "
            "Train the `gate`, `stage1`, and `stage2` models first, then upload the image again."
        )

    if status == "error":
        return prediction.get("message", "The image could not be processed.")

    if status == "unsupported_crop":
        return (
            "We are working on support for leaves other than wheat. "
            "Please upload a wheat leaf image for now."
        )

    if status == "invalid_subject":
        return "Please upload only plant leaf images. Non-plant images are not supported."

    if status == "reupload":
        reason = prediction.get("reason", "Image quality is too low.")
        suggestions = prediction.get("suggestions", [])
        suggestion_text = " ".join(suggestions[:2])
        return f"Please upload a clearer image. {reason} {suggestion_text}".strip()

    top_predictions = prediction.get("top_predictions", [])
    alternatives = ", ".join(
        f"{item['label']} ({_confidence_percent(item['score'])})"
        for item in top_predictions[1:3]
    )

    message = (
        f"Diagnosis: {prediction.get('disease', 'Unknown')} "
        f"with confidence {prediction.get('confidence_percent', 'Unknown')}. "
        f"{prediction.get('explanation', '')}"
    )

    if status == "uncertain":
        message += " The model is not fully confident, so please retake the image in daylight if possible."

    if alternatives:
        message += f" Other likely options: {alternatives}."

    organic = prediction.get("organic_treatment", [])
    chemical = prediction.get("chemical_treatment", [])
    if organic:
        message += f" Organic option: {organic[0]}."
    if chemical:
        message += f" Chemical option: {chemical[0]}."

    return message.strip()
