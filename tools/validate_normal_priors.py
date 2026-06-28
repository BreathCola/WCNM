#!/usr/bin/env python3
"""Strictly validate a complete RT-GS monocular-normal prior set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class PriorValidationError(RuntimeError):
    pass


def _duplicates(values: Sequence[str]) -> list[str]:
    return sorted(key for key, count in Counter(values).items() if count > 1)


def validate_scene(
    scene: Path,
    images_directory: str = "images",
    priors_directory: str = "normal_priors",
    manifest_name: str = "manifest.json",
    unit_atol: float = 5e-4,
    expected_data_type: str = "indoor",
    expected_resolution: int = 768,
    expected_steps: int = 10,
) -> dict[str, Any]:
    scene = Path(scene).expanduser().resolve()
    image_root = scene / images_directory
    prior_root = scene / priors_directory
    manifest_path = prior_root / manifest_name
    if not image_root.is_dir():
        raise PriorValidationError(f"image directory does not exist: {image_root}")
    if not prior_root.is_dir():
        raise PriorValidationError(f"prior directory does not exist: {prior_root}")
    if not manifest_path.is_file():
        raise PriorValidationError(f"manifest does not exist: {manifest_path}")

    images = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    priors = sorted(prior_root.glob("*.npy"))
    image_stems = [path.stem for path in images]
    expected_names = {f"{stem}.npy" for stem in image_stems}
    actual_names = {path.name for path in priors}
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("files", [])
    if not isinstance(entries, list):
        raise PriorValidationError("manifest files must be a list")
    output_names = [entry.get("output_file") for entry in entries if isinstance(entry, dict)]
    source_names = [entry.get("source_image") for entry in entries if isinstance(entry, dict)]
    entry_by_output = {
        entry.get("output_file"): entry for entry in entries if isinstance(entry, dict)
    }

    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    duplicate_image_stems = _duplicates(image_stems)
    duplicate_manifest_outputs = _duplicates(output_names)
    duplicate_manifest_sources = _duplicates(source_names)
    manifest_missing = sorted(expected_names - set(output_names))
    manifest_unexpected = sorted(set(output_names) - expected_names)
    temporary_files = sorted(path.name for path in prior_root.glob(".*.tmp"))
    unexpected_files = sorted(
        path.name for path in prior_root.iterdir()
        if path.is_file() and path.name != manifest_name and path.suffix != ".npy"
    )
    errors: list[str] = []
    max_unit_error = 0.0
    min_component = float("inf")
    max_component = float("-inf")

    for image in images:
        prior = prior_root / f"{image.stem}.npy"
        if not prior.is_file():
            continue
        try:
            values = np.load(prior, allow_pickle=False)
            with Image.open(image) as opened:
                expected_shape = (opened.height, opened.width, 3)
            if values.shape != expected_shape:
                errors.append(f"{prior.name}: shape={values.shape}, expected={expected_shape}")
            if values.dtype != np.float32:
                errors.append(f"{prior.name}: dtype={values.dtype}, expected=float32")
            if not np.isfinite(values).all():
                errors.append(f"{prior.name}: non-finite values")
                continue
            lengths = np.linalg.norm(values, axis=-1)
            if (lengths <= 1e-6).any():
                errors.append(f"{prior.name}: near-zero vectors")
            unit_error = float(np.max(np.abs(lengths - 1.0)))
            max_unit_error = max(max_unit_error, unit_error)
            if unit_error > unit_atol:
                errors.append(f"{prior.name}: max unit error={unit_error}")
            min_component = min(min_component, float(values.min()))
            max_component = max(max_component, float(values.max()))
            if float(np.max(np.abs(values))) > 1.0 + unit_atol:
                errors.append(f"{prior.name}: component outside [-1,1]")

            entry = entry_by_output.get(prior.name)
            if entry is None:
                continue
            expected_source = image.relative_to(scene).as_posix()
            if entry.get("source_image") != expected_source:
                errors.append(f"{prior.name}: manifest source={entry.get('source_image')}")
            if entry.get("shape") != list(expected_shape):
                errors.append(f"{prior.name}: manifest shape={entry.get('shape')}")
            if entry.get("dtype") != "float32":
                errors.append(f"{prior.name}: manifest dtype={entry.get('dtype')}")
            if entry.get("layout") != "HWC" or entry.get("space") != "camera":
                errors.append(f"{prior.name}: manifest layout/space mismatch")
            parameters = entry.get("generation_parameters", {})
            expected_parameters = {
                "model": "StableNormal",
                "data_type": expected_data_type,
                "resolution": expected_resolution,
                "steps": expected_steps,
                "yoso_version": "yoso-normal-v0-3",
                "diffusion_version": "stable-normal-v0-1",
            }
            for key, expected in expected_parameters.items():
                if parameters.get(key) != expected:
                    errors.append(
                        f"{prior.name}: generation_parameters[{key}]="
                        f"{parameters.get(key)!r}, expected={expected!r}"
                    )
        except Exception as error:
            errors.append(f"{prior.name}: {type(error).__name__}: {error}")

    errors.extend(f"missing prior: {name}" for name in missing)
    errors.extend(f"unexpected prior: {name}" for name in unexpected)
    errors.extend(f"duplicate image stem: {name}" for name in duplicate_image_stems)
    errors.extend(f"duplicate manifest output: {name}" for name in duplicate_manifest_outputs)
    errors.extend(f"duplicate manifest source: {name}" for name in duplicate_manifest_sources)
    errors.extend(f"manifest missing: {name}" for name in manifest_missing)
    errors.extend(f"manifest unexpected: {name}" for name in manifest_unexpected)
    errors.extend(f"temporary file: {name}" for name in temporary_files)
    errors.extend(f"unexpected file: {name}" for name in unexpected_files)
    if len(images) != len(priors):
        errors.append(f"count mismatch: images={len(images)} priors={len(priors)}")
    if len(entries) != len(images):
        errors.append(f"manifest count mismatch: entries={len(entries)} images={len(images)}")

    report = {
        "images": len(images),
        "priors": len(priors),
        "manifest_entries": len(entries),
        "missing": missing,
        "unexpected": unexpected,
        "duplicate_image_stems": duplicate_image_stems,
        "duplicate_manifest_outputs": duplicate_manifest_outputs,
        "duplicate_manifest_sources": duplicate_manifest_sources,
        "temporary_files": temporary_files,
        "unexpected_files": unexpected_files,
        "max_unit_error": max_unit_error,
        "component_range": [min_component, max_component],
        "errors": errors,
    }
    if errors:
        raise PriorValidationError("\n".join(errors))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--images", default="images")
    parser.add_argument("--priors", default="normal_priors")
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--unit-atol", type=float, default=5e-4)
    parser.add_argument("--expected-data-type", default="indoor")
    parser.add_argument("--expected-resolution", type=int, default=768)
    parser.add_argument("--expected-steps", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_scene(
            args.scene,
            images_directory=args.images,
            priors_directory=args.priors,
            manifest_name=args.manifest,
            unit_atol=args.unit_atol,
            expected_data_type=args.expected_data_type,
            expected_resolution=args.expected_resolution,
            expected_steps=args.expected_steps,
        )
    except (OSError, ValueError, PriorValidationError) as error:
        print(f"STRICT_VALIDATION=FAIL\n{error}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    print("STRICT_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
