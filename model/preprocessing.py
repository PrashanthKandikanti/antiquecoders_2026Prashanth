"""Preprocessing helpers for the gate and wheat disease training datasets."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from model.constants import (
    ARTIFACTS_DIR,
    DATA_DIR,
    DISEASE_CLASS_NAMES,
    GATE_CLASS_NAMES,
    GATE_DATA_DIR,
    PREPROCESS_SUMMARY_PATH,
    RAW_DATA_DIR,
    RAW_GATE_DIR,
    RAW_WHEAT_DATA_DIR,
    STAGE1_CLASS_NAMES,
    STAGE1_DATA_DIR,
    STAGE2_DATA_DIR,
)
from model.dataset import IMAGE_EXTENSIONS

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


def _collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _output_has_images(output_dir: Path) -> bool:
    if not output_dir.exists():
        return False
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in output_dir.rglob("*")
    )


def _resolve_source_dir(
    preferred_dir: Path,
    fallback_dirs: Sequence[Path],
    required_classes: Sequence[str],
) -> Path:
    for candidate in (preferred_dir, *fallback_dirs):
        if candidate.exists() and all((candidate / class_name).exists() for class_name in required_classes):
            return candidate

    searched = ", ".join(str(path) for path in (preferred_dir, *fallback_dirs))
    raise FileNotFoundError(
        f"Could not find a raw dataset folder with classes {list(required_classes)}. "
        f"Searched: {searched}"
    )


def _split_counts(total: int, ratios: SplitRatios) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 1, 0
    if total == 3:
        return 1, 1, 1

    val_count = max(1, int(round(total * ratios.val)))
    test_count = max(1, int(round(total * ratios.test)))
    train_count = total - val_count - test_count

    while train_count < 1 and (val_count > 1 or test_count > 1):
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        train_count = total - val_count - test_count

    if train_count < 1:
        train_count = 1
        overflow = val_count + test_count + train_count - total
        if overflow > 0 and test_count >= overflow:
            test_count -= overflow
        elif overflow > 0:
            val_count = max(0, val_count - overflow)

    return train_count, val_count, test_count


def _split_paths(paths: list[Path], seed: int, ratios: SplitRatios) -> dict[str, list[Path]]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    train_count, val_count, test_count = _split_counts(len(shuffled), ratios)

    train_paths = shuffled[:train_count]
    val_paths = shuffled[train_count : train_count + val_count]
    test_paths = shuffled[train_count + val_count : train_count + val_count + test_count]
    return {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
    }


def _safe_reset_output_dir(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    processed_root = DATA_DIR.resolve()
    if processed_root not in output_dir.parents and output_dir != processed_root:
        raise ValueError(f"Refusing to delete data outside {processed_root}: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def _copy_split_files(
    source_dir: Path,
    output_dir: Path,
    class_names: Sequence[str],
    mapper: Callable[[str], str | None],
    seed: int,
    ratios: SplitRatios,
) -> dict[str, dict[str, int]]:
    counts = {split: {} for split in SPLITS}

    for split in SPLITS:
        for class_name in class_names:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        counts["train"].setdefault(class_name, 0)
        counts["val"].setdefault(class_name, 0)
        counts["test"].setdefault(class_name, 0)

    for source_class_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        mapped_name = mapper(source_class_dir.name.lower())
        if mapped_name is None:
            continue

        image_paths = _collect_images(source_class_dir)
        split_map = _split_paths(image_paths, seed=seed, ratios=ratios)

        for split, split_paths in split_map.items():
            target_dir = output_dir / split / mapped_name
            for image_path in split_paths:
                destination = target_dir / image_path.name
                if destination.exists():
                    destination = target_dir / f"{image_path.stem}_{abs(hash(str(image_path))) % 100000}{image_path.suffix.lower()}"
                shutil.copy2(image_path, destination)
                counts[split][mapped_name] += 1

    return counts


def prepare_gate_dataset(
    raw_data_dir: Path = RAW_DATA_DIR,
    output_dir: Path = GATE_DATA_DIR,
    *,
    seed: int = 42,
    force: bool = False,
    ratios: SplitRatios = SplitRatios(),
) -> dict[str, object]:
    try:
        gate_source_dir = _resolve_source_dir(
            raw_data_dir / "gate",
            [RAW_GATE_DIR],
            GATE_CLASS_NAMES,
        )
    except FileNotFoundError as exc:
        if _output_has_images(output_dir):
            return {
                "dataset": "gate",
                "status": "skipped",
                "output_dir": str(output_dir),
                "reason": "Processed files already exist and raw gate data was not found.",
            }
        return {
            "dataset": "gate",
            "status": "missing_raw",
            "output_dir": str(output_dir),
            "reason": str(exc),
        }

    if _output_has_images(output_dir) and not force:
        return {
            "dataset": "gate",
            "status": "skipped",
            "source_dir": str(gate_source_dir),
            "output_dir": str(output_dir),
            "reason": "Processed files already exist. Use --force-prepare to rebuild them.",
        }

    if force:
        _safe_reset_output_dir(output_dir)

    counts = _copy_split_files(
        source_dir=gate_source_dir,
        output_dir=output_dir,
        class_names=GATE_CLASS_NAMES,
        mapper=lambda class_name: class_name if class_name in GATE_CLASS_NAMES else None,
        seed=seed,
        ratios=ratios,
    )

    return {
        "dataset": "gate",
        "status": "prepared",
        "source_dir": str(gate_source_dir),
        "output_dir": str(output_dir),
        "counts": counts,
    }


def prepare_wheat_hierarchical_datasets(
    raw_data_dir: Path = RAW_DATA_DIR,
    *,
    seed: int = 42,
    force: bool = False,
    ratios: SplitRatios = SplitRatios(),
) -> list[dict[str, object]]:
    outputs = []
    stage_targets = [
        (
            "stage1",
            STAGE1_DATA_DIR,
            STAGE1_CLASS_NAMES,
            lambda class_name: "healthy" if class_name == "healthy" else "diseased",
        ),
        (
            "stage2",
            STAGE2_DATA_DIR,
            DISEASE_CLASS_NAMES,
            lambda class_name: class_name if class_name in DISEASE_CLASS_NAMES else None,
        ),
    ]

    try:
        wheat_source_dir = _resolve_source_dir(
            raw_data_dir / "wheat_disease",
            [raw_data_dir / "wheat", RAW_WHEAT_DATA_DIR],
            ["healthy", *DISEASE_CLASS_NAMES],
        )
    except FileNotFoundError as exc:
        for dataset_name, output_dir, _, _ in stage_targets:
            if _output_has_images(output_dir):
                outputs.append(
                    {
                        "dataset": dataset_name,
                        "status": "skipped",
                        "output_dir": str(output_dir),
                        "reason": "Processed files already exist and raw wheat data was not found.",
                    }
                )
            else:
                outputs.append(
                    {
                        "dataset": dataset_name,
                        "status": "missing_raw",
                        "output_dir": str(output_dir),
                        "reason": str(exc),
                    }
                )
        return outputs

    for dataset_name, output_dir, class_names, mapper in stage_targets:
        if _output_has_images(output_dir) and not force:
            outputs.append(
                {
                    "dataset": dataset_name,
                    "status": "skipped",
                    "source_dir": str(wheat_source_dir),
                    "output_dir": str(output_dir),
                    "reason": "Processed files already exist. Use --force-prepare to rebuild them.",
                }
            )
            continue

        if force:
            _safe_reset_output_dir(output_dir)

        counts = _copy_split_files(
            source_dir=wheat_source_dir,
            output_dir=output_dir,
            class_names=class_names,
            mapper=mapper,
            seed=seed,
            ratios=ratios,
        )
        outputs.append(
            {
                "dataset": dataset_name,
                "status": "prepared",
                "source_dir": str(wheat_source_dir),
                "output_dir": str(output_dir),
                "counts": counts,
            }
        )

    return outputs


def prepare_processed_datasets(
    raw_data_dir: Path = RAW_DATA_DIR,
    *,
    seed: int = 42,
    force: bool = False,
    ratios: SplitRatios = SplitRatios(),
    summary_path: Path = PREPROCESS_SUMMARY_PATH,
) -> dict[str, object]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "raw_data_dir": str(raw_data_dir),
        "seed": seed,
        "ratios": {
            "train": ratios.train,
            "val": ratios.val,
            "test": ratios.test,
        },
        "datasets": [
            prepare_gate_dataset(raw_data_dir=raw_data_dir, seed=seed, force=force, ratios=ratios),
            *prepare_wheat_hierarchical_datasets(
                raw_data_dir=raw_data_dir,
                seed=seed,
                force=force,
                ratios=ratios,
            ),
        ],
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary
