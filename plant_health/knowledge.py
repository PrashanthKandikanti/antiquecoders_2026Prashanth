"""Rule-based disease and pesticide guidance shown after model prediction."""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parent / "data"
PESTICIDES_CSV_PATH = DATA_DIR / "wheat_disease_pesticides.csv"


BASE_DISEASE_GUIDANCE: dict[str, dict[str, Any]] = {
    "healthy": {
        "explanation": "The uploaded leaf does not show strong signs of the target wheat diseases.",
        "treatment": ["No immediate treatment is needed."],
        "prevention": [
            "Keep monitoring the field weekly.",
            "Maintain balanced irrigation and nutrition.",
        ],
    },
    "rust": {
        "explanation": "Rust usually appears as orange-brown pustules on the leaf surface.",
        "treatment": [
            "Remove heavily affected leaves where practical.",
            "Use a locally recommended fungicide only if needed.",
        ],
        "prevention": [
            "Monitor nearby leaves for spread.",
            "Confirm final product choice with local agriculture guidance.",
        ],
    },
    "blight": {
        "explanation": "Leaf blight often causes elongated brown lesions and dry patches.",
        "treatment": [
            "Remove badly damaged plant material when possible.",
            "Use region-approved disease management practices.",
        ],
        "prevention": [
            "Avoid prolonged leaf wetness.",
            "Improve airflow around the crop canopy.",
        ],
    },
    "mildew": {
        "explanation": "Powdery mildew looks like white powder-like growth on the leaf.",
        "treatment": [
            "Inspect nearby leaves for early spread.",
            "Use locally approved control only when necessary.",
        ],
        "prevention": [
            "Reduce dense canopy conditions.",
            "Keep regular field scouting in place.",
        ],
    },
    "spot": {
        "explanation": "Spot blotch often appears as dark brown lesions that expand across the leaf.",
        "treatment": [
            "Remove severely affected material where practical.",
            "Follow local agronomy advice for disease management.",
        ],
        "prevention": [
            "Use clean seed and resistant varieties where possible.",
            "Avoid carrying infected residue into the next cycle.",
        ],
    },
}


def _split_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


@lru_cache(maxsize=1)
def load_pesticide_guidance() -> dict[str, dict[str, Any]]:
    guidance: dict[str, dict[str, Any]] = {}

    if not PESTICIDES_CSV_PATH.exists():
        return guidance

    with PESTICIDES_CSV_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            disease_code = (row.get("disease_code") or "").strip().lower()
            if not disease_code:
                continue

            organic_treatment = _split_csv_list(row.get("organic_controls", ""))
            chemical_treatment = _split_csv_list(row.get("chemical_controls", ""))
            monitoring = (row.get("monitoring_and_timing") or "").strip()
            safety_note = (row.get("safety_note") or "").strip()

            combined_treatment = (
                [f"Organic: {item}" for item in organic_treatment]
                + [f"Chemical: {item}" for item in chemical_treatment]
            )

            guidance[disease_code] = {
                "disease_name": (row.get("disease_name") or disease_code.title()).strip(),
                "organic_treatment": organic_treatment,
                "chemical_treatment": chemical_treatment,
                "treatment": combined_treatment,
                "monitoring": monitoring,
                "safety_note": safety_note,
            }

    return guidance


def get_disease_guidance(disease_code: str | None) -> dict[str, Any]:
    if not disease_code:
        return {}

    disease_code = disease_code.lower()
    base = dict(BASE_DISEASE_GUIDANCE.get(disease_code, {}))
    pesticide = load_pesticide_guidance().get(disease_code, {})

    merged = {**base, **pesticide}
    if "treatment" not in merged:
        merged["treatment"] = base.get("treatment", [])

    return merged


DISEASE_GUIDANCE = BASE_DISEASE_GUIDANCE
