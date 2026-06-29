#!/usr/bin/env python3
"""Strict validation for axis-unconfirmed DiffusionRenderer normal candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

try:
    from tools.adapt_diffusion_renderer_normals import sha256_file
except ModuleNotFoundError:  # Direct execution: python tools/validate_....py
    from adapt_diffusion_renderer_normals import sha256_file


UNIT_ATOL = 5e-4


class CandidateValidationError(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__("\n".join(report["errors"]))


def _duplicates(values: Sequence[str]) -> list[str]:
    return sorted(key for key, count in Counter(values).items() if count > 1)


def validate_candidates(
    scene: Path,
    raw_root: Path,
    candidate_root: Path,
    expected_count: int = 111,
    unit_atol: float = UNIT_ATOL,
) -> dict[str, Any]:
    scene = Path(scene).expanduser().resolve()
    raw_root = Path(raw_root).expanduser().resolve()
    candidate_root = Path(candidate_root).expanduser().resolve()
    image_root = scene / "images"
    normal_root = candidate_root / "raw_identity" / "normal"
    manifest_path = candidate_root / "manifests" / "adapter_manifest.json"
    raw_manifest_path = raw_root / "manifest.json"
    errors: list[str] = []

    for path, label in (
        (image_root, "image directory"),
        (normal_root, "candidate normal directory"),
    ):
        if not path.is_dir():
            errors.append(f"missing {label}: {path}")
    for path, label in (
        (manifest_path, "adapter manifest"),
        (raw_manifest_path, "raw manifest"),
    ):
        if not path.is_file():
            errors.append(f"missing {label}: {path}")
    if errors:
        raise CandidateValidationError({"errors": errors})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    raw_records = raw_manifest.get("frames_and_padding")
    if not isinstance(entries, list):
        errors.append("adapter manifest files must be a list")
        entries = []
    if not isinstance(raw_records, list):
        errors.append("raw manifest frames_and_padding must be a list")
        raw_records = []

    images = sorted(image_root.glob("*.jpg"))
    priors = sorted(normal_root.glob("*.npy"))
    expected_names = [f"{index:06d}.npy" for index in range(expected_count)]
    image_names = [path.name for path in images]
    prior_names = [path.name for path in priors]
    entry_outputs = [entry.get("output_file") for entry in entries if isinstance(entry, dict)]
    entry_sources = [entry.get("source_image") for entry in entries if isinstance(entry, dict)]
    entry_raw = [entry.get("raw_normal_file") for entry in entries if isinstance(entry, dict)]

    if image_names != [f"{index:06d}.jpg" for index in range(expected_count)]:
        errors.append("source image set is missing, unexpected, or out of order")
    if prior_names != expected_names:
        errors.append("candidate prior set is missing, unexpected, or out of order")
    if len(entries) != expected_count:
        errors.append(f"manifest entries={len(entries)}, expected={expected_count}")
    if any(entry.get("is_padding") for entry in entries if isinstance(entry, dict)):
        errors.append("adapter manifest contains a padding entry")
    if manifest.get("axis_mapping") is not None:
        errors.append("raw_identity manifest must not select an axis mapping")
    if manifest.get("axis_selection_status") != "awaiting_human_audit":
        errors.append("axis selection status is not awaiting_human_audit")
    if manifest.get("raw_manifest_padding_frame_count") != raw_manifest.get(
        "padding_frame_count"
    ):
        errors.append("raw padding provenance mismatch")
    if sha256_file(raw_manifest_path) != manifest.get("raw_manifest_sha256"):
        errors.append("raw manifest SHA-256 mismatch")

    errors.extend(f"duplicate output: {name}" for name in _duplicates(entry_outputs))
    errors.extend(f"duplicate source: {name}" for name in _duplicates(entry_sources))
    errors.extend(f"duplicate raw normal: {name}" for name in _duplicates(entry_raw))

    raw_real = {
        record.get("rtgs_real_frame_index"): record
        for record in raw_records
        if isinstance(record, dict) and not record.get("is_padding")
    }
    raw_padding = [
        record for record in raw_records
        if isinstance(record, dict) and record.get("is_padding")
    ]
    if len(raw_real) != expected_count:
        errors.append(f"raw real records={len(raw_real)}, expected={expected_count}")
    if len(raw_padding) != int(raw_manifest.get("padding_frame_count", -1)):
        errors.append("raw padding record count mismatch")
    if any(record.get("rtgs_input_file") is not None for record in raw_padding):
        errors.append("raw padding is mapped to a real RT-GS input")

    max_unit_error = 0.0
    component_min = float("inf")
    component_max = float("-inf")
    output_hashes: list[str] = []
    raw_hashes: list[str] = []
    for index in range(min(expected_count, len(entries))):
        entry = entries[index]
        stem = f"{index:06d}"
        if not isinstance(entry, dict):
            errors.append(f"entry {index} is not an object")
            continue
        source = image_root / f"{stem}.jpg"
        prior = normal_root / f"{stem}.npy"
        raw_record = raw_real.get(index)
        if raw_record is None:
            errors.append(f"{stem}: missing raw real record")
            continue
        raw_relative = raw_record.get("raw_prior_files", {}).get("normal")
        raw_path = raw_root / str(raw_relative)
        if entry.get("frame_index") != index:
            errors.append(f"{stem}: manifest frame_index mismatch")
        if entry.get("source_image") != f"images/{stem}.jpg":
            errors.append(f"{stem}: manifest source mismatch")
        if entry.get("output_file") != f"raw_identity/normal/{stem}.npy":
            errors.append(f"{stem}: manifest output mismatch")
        if entry.get("raw_normal_file") != raw_relative:
            errors.append(f"{stem}: raw normal trace mismatch")
        if entry.get("axis_transform_applied") is not None:
            errors.append(f"{stem}: unexpected axis transform")
        if entry.get("axis_selection_status") != "unselected":
            errors.append(f"{stem}: unexpected axis selection status")
        if entry.get("space") != "unconfirmed_diffusion_renderer_camera_candidate":
            errors.append(f"{stem}: candidate space label mismatch")
        if entry.get("source_to_raw_mapping") != raw_record.get("source_to_prior_mapping"):
            errors.append(f"{stem}: resize mapping mismatch")
        if not source.is_file() or not prior.is_file() or not raw_path.is_file():
            errors.append(f"{stem}: missing source, candidate, or raw PNG")
            continue
        if sha256_file(source) != entry.get("source_sha256"):
            errors.append(f"{stem}: source SHA-256 mismatch")
        raw_sha = sha256_file(raw_path)
        raw_hashes.append(raw_sha)
        if raw_sha != entry.get("raw_normal_sha256"):
            errors.append(f"{stem}: raw normal SHA-256 mismatch")
        output_sha = sha256_file(prior)
        output_hashes.append(output_sha)
        if output_sha != entry.get("output_sha256"):
            errors.append(f"{stem}: output SHA-256 mismatch")

        with Image.open(source) as opened:
            expected_shape = (opened.height, opened.width, 3)
        try:
            values = np.load(prior, allow_pickle=False, mmap_mode="r")
            if values.shape != expected_shape:
                errors.append(f"{stem}: shape={values.shape}, expected={expected_shape}")
            if values.dtype != np.float32:
                errors.append(f"{stem}: dtype={values.dtype}, expected=float32")
            if not np.isfinite(values).all():
                errors.append(f"{stem}: non-finite values")
                continue
            lengths = np.linalg.norm(values, axis=-1)
            if (lengths <= 1e-6).any():
                errors.append(f"{stem}: near-zero vectors")
            unit_error = float(np.max(np.abs(lengths - 1.0)))
            max_unit_error = max(max_unit_error, unit_error)
            if unit_error > unit_atol:
                errors.append(f"{stem}: max unit error={unit_error}")
            component_min = min(component_min, float(values.min()))
            component_max = max(component_max, float(values.max()))
            if entry.get("shape") != list(expected_shape):
                errors.append(f"{stem}: manifest shape mismatch")
            if entry.get("dtype") != "float32" or entry.get("layout") != "HWC":
                errors.append(f"{stem}: manifest dtype/layout mismatch")
        except (OSError, ValueError) as error:
            errors.append(f"{stem}: {type(error).__name__}: {error}")

    duplicate_output_sha256 = _duplicates(output_hashes)
    duplicate_raw_sha256 = _duplicates(raw_hashes)
    if duplicate_output_sha256:
        errors.append(f"duplicate candidate file contents: {duplicate_output_sha256}")
    if duplicate_raw_sha256:
        errors.append(f"duplicate real raw normal contents: {duplicate_raw_sha256}")

    unexpected_files = sorted(
        path.name for path in normal_root.iterdir()
        if path.is_file() and path.suffix != ".npy"
    )
    temporary_files = sorted(path.name for path in candidate_root.rglob(".*.tmp"))
    errors.extend(f"unexpected candidate file: {name}" for name in unexpected_files)
    errors.extend(f"temporary file: {name}" for name in temporary_files)

    report = {
        "validation": "FAIL" if errors else "PASS",
        "images": len(images),
        "priors": len(priors),
        "manifest_entries": len(entries),
        "raw_real_records": len(raw_real),
        "raw_padding_records_excluded": len(raw_padding),
        "layout": "HWC",
        "dtype": "float32",
        "expected_shape": [2152, 3827, 3] if expected_count == 111 else None,
        "axis_selection_status": manifest.get("axis_selection_status"),
        "max_unit_error": max_unit_error,
        "component_range": [component_min, component_max],
        "duplicate_output_names": _duplicates(entry_outputs),
        "duplicate_source_names": _duplicates(entry_sources),
        "duplicate_raw_names": _duplicates(entry_raw),
        "duplicate_candidate_sha256": duplicate_output_sha256,
        "duplicate_real_raw_sha256": duplicate_raw_sha256,
        "temporary_files": temporary_files,
        "unexpected_files": unexpected_files,
        "errors": errors,
    }
    if errors:
        raise CandidateValidationError(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=111)
    parser.add_argument("--unit-atol", type=float, default=UNIT_ATOL)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--summary", type=Path)
    return parser


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# DiffusionRenderer normal candidate validation",
        "",
        f"Result: **{report['validation']}**",
        "",
        f"- images / priors / manifest entries: {report['images']} / {report['priors']} / {report['manifest_entries']}",
        f"- raw padding slots excluded: {report['raw_padding_records_excluded']}",
        f"- contract: {report['layout']} {report['dtype']}, shape {report['expected_shape']}",
        f"- component range: {report['component_range']}",
        f"- maximum unit-length error: {report['max_unit_error']}",
        f"- axis status: {report['axis_selection_status']}",
        f"- errors: {len(report['errors'])}",
        "",
        "No padding frame is present in the 111 real-frame candidate mapping. No axis mapping is selected.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_candidates(
            args.scene,
            args.raw_root,
            args.candidates,
            args.expected_count,
            args.unit_atol,
        )
    except CandidateValidationError as error:
        report = error.report
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        print("DIFFRENDER_NORMAL_VALIDATION=FAIL")
        return 1
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.summary:
        _write_summary(args.summary, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print("DIFFRENDER_NORMAL_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
