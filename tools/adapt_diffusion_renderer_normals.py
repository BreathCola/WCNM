#!/usr/bin/env python3
"""Adapt raw DiffusionRenderer RGB normals into an axis-unconfirmed RT-GS candidate.

This tool deliberately applies no semantic axis mapping.  It only preserves the
raw component order while decoding, resizing each component with bilinear
interpolation, and restoring unit length.  The output directory must remain
separate from the accepted StableNormal ``normal_priors`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image


NORMAL_EPS = 1e-6
UNIT_ATOL = 5e-4
SCHEMA_VERSION = 1


class AdapterError(RuntimeError):
    """Raised when raw provenance or decoded normals violate the contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_raw_normal(raw_rgb: np.ndarray) -> np.ndarray:
    """Decode uint8 RGB as signed float32 components without normalizing."""
    values = np.asarray(raw_rgb)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise AdapterError(f"raw normal must have HWC shape [H,W,3], got {values.shape}")
    if values.dtype != np.uint8:
        raise AdapterError(f"raw normal dtype must be uint8, got {values.dtype}")
    decoded = values.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    if not np.isfinite(decoded).all():
        raise AdapterError("decoded raw normal contains NaN or Inf")
    return np.ascontiguousarray(decoded, dtype=np.float32)


def resize_and_unit_normalize(
    decoded: np.ndarray,
    target_hw: tuple[int, int],
    eps: float = NORMAL_EPS,
) -> np.ndarray:
    """Bilinearly resize three components, then normalize every output vector."""
    if decoded.ndim != 3 or decoded.shape[-1] != 3:
        raise AdapterError(f"decoded normal must have HWC shape [H,W,3], got {decoded.shape}")
    if decoded.dtype != np.float32:
        raise AdapterError(f"decoded normal dtype must be float32, got {decoded.dtype}")
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    if target_h <= 0 or target_w <= 0:
        raise AdapterError(f"invalid target size: {target_hw}")

    # OpenCV applies INTER_LINEAR independently to every HWC channel.
    resized = cv2.resize(decoded, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized = np.asarray(resized, dtype=np.float32)
    if resized.ndim != 3 or resized.shape != (target_h, target_w, 3):
        raise AdapterError(f"unexpected resized shape: {resized.shape}")
    if not np.isfinite(resized).all():
        raise AdapterError("resized normal contains NaN or Inf")

    lengths = np.linalg.norm(resized, axis=-1, keepdims=True)
    near_zero = lengths <= np.float32(eps)
    if near_zero.any():
        raise AdapterError(
            f"resized normal contains {int(near_zero.sum())} near-zero vectors; "
            "no invalid-pixel rule is authorized"
        )
    normalized = resized / lengths
    return np.ascontiguousarray(normalized, dtype=np.float32)


def adapt_raw_rgb(raw_rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    return resize_and_unit_normalize(decode_raw_normal(raw_rgb), target_hw)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("rb") as handle:
            reread = np.load(handle, allow_pickle=False)
        if reread.dtype != np.float32 or reread.shape != values.shape:
            raise AdapterError(f"atomic reread mismatch for {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_raw_manifest(raw_root: Path) -> dict[str, Any]:
    manifest_path = raw_root / "manifest.json"
    if not manifest_path.is_file():
        raise AdapterError(f"raw manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = manifest.get("frames_and_padding")
    if not isinstance(records, list):
        raise AdapterError("raw manifest frames_and_padding must be a list")
    return manifest


def collect_real_records(
    scene: Path,
    raw_root: Path,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Validate raw provenance and return exactly the ordered real-frame records."""
    scene = scene.resolve()
    raw_root = raw_root.resolve()
    manifest = _load_raw_manifest(raw_root)
    records = manifest["frames_and_padding"]
    real = [record for record in records if not record.get("is_padding")]
    padding = [record for record in records if record.get("is_padding")]
    if len(real) != expected_count:
        raise AdapterError(f"expected {expected_count} real records, got {len(real)}")
    if manifest.get("real_frame_count") != expected_count:
        raise AdapterError("raw manifest real_frame_count mismatch")
    if len(padding) != int(manifest.get("padding_frame_count", -1)):
        raise AdapterError("raw manifest padding count mismatch")

    expected_indices = list(range(expected_count))
    actual_indices = [record.get("rtgs_real_frame_index") for record in real]
    if actual_indices != expected_indices:
        raise AdapterError("real records are missing or out of order")
    if any(record.get("rtgs_input_file") is not None for record in padding):
        raise AdapterError("padding record is mapped to an RT-GS real input")

    raw_names: list[str] = []
    for index, record in enumerate(real):
        expected_source = f"data/TiHuBird/images/{index:06d}.jpg"
        source_text = record.get("rtgs_input_file")
        if source_text != expected_source:
            raise AdapterError(
                f"frame {index}: source={source_text!r}, expected={expected_source!r}"
            )
        source = scene / "images" / f"{index:06d}.jpg"
        if not source.is_file():
            raise AdapterError(f"source image does not exist: {source}")
        if sha256_file(source) != record.get("input_sha256"):
            raise AdapterError(f"source SHA-256 mismatch: {source}")
        raw_relative = record.get("raw_prior_files", {}).get("normal")
        if not isinstance(raw_relative, str):
            raise AdapterError(f"frame {index}: missing raw normal path")
        raw_path = raw_root / raw_relative
        if not raw_path.is_file():
            raise AdapterError(f"raw normal does not exist: {raw_path}")
        raw_names.append(raw_path.name)
    if len(raw_names) != len(set(raw_names)):
        raise AdapterError("duplicate raw normal paths among real records")
    return real


def adapt_scene(
    scene: Path,
    raw_root: Path,
    output_root: Path,
    expected_count: int = 111,
) -> dict[str, Any]:
    scene = Path(scene).expanduser().resolve()
    raw_root = Path(raw_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if output_root == (scene / "normal_priors").resolve():
        raise AdapterError("refusing to overwrite the StableNormal normal_priors directory")

    normal_root = output_root / "raw_identity" / "normal"
    manifest_root = output_root / "manifests"
    axis_root = output_root / "axis_audit"
    validation_root = output_root / "validation"
    if output_root.exists() and any(output_root.iterdir()):
        raise AdapterError(f"candidate output must be new or empty: {output_root}")
    normal_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    axis_root.mkdir(parents=True, exist_ok=True)
    validation_root.mkdir(parents=True, exist_ok=True)

    raw_manifest = _load_raw_manifest(raw_root)
    records = collect_real_records(scene, raw_root, expected_count)
    output_entries: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        stem = f"{index:06d}"
        source = scene / "images" / f"{stem}.jpg"
        with Image.open(source) as opened:
            target_hw = (opened.height, opened.width)
        raw_relative = record["raw_prior_files"]["normal"]
        raw_path = raw_root / raw_relative
        with Image.open(raw_path) as opened:
            raw_rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
        adapted = adapt_raw_rgb(raw_rgb, target_hw)
        output_path = normal_root / f"{stem}.npy"
        _atomic_save_npy(output_path, adapted)
        unit_error = float(np.max(np.abs(np.linalg.norm(adapted, axis=-1) - 1.0)))
        if unit_error > UNIT_ATOL:
            raise AdapterError(f"{stem}: unit error {unit_error} exceeds {UNIT_ATOL}")
        output_entries.append(
            {
                "frame_index": index,
                "source_image": f"images/{stem}.jpg",
                "source_sha256": record["input_sha256"],
                "raw_normal_file": raw_relative,
                "raw_normal_sha256": sha256_file(raw_path),
                "raw_slot": record["raw_slot"],
                "chunk_index": record["chunk_index"],
                "frame_index_within_chunk": record["frame_index"],
                "is_padding": False,
                "output_file": f"raw_identity/normal/{stem}.npy",
                "output_sha256": sha256_file(output_path),
                "shape": list(adapted.shape),
                "dtype": "float32",
                "layout": "HWC",
                "space": "unconfirmed_diffusion_renderer_camera_candidate",
                "axis_transform_applied": None,
                "axis_selection_status": "unselected",
                "max_unit_error": unit_error,
                "source_to_raw_mapping": record["source_to_prior_mapping"],
            }
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "DiffusionRenderer raw-identity normal candidate",
        "warning": (
            "Component order is preserved only as a technical raw identity; "
            "no RT-GS camera-axis mapping has been selected or validated."
        ),
        "scene": str(scene),
        "raw_root": str(raw_root),
        "raw_manifest_sha256": sha256_file(raw_root / "manifest.json"),
        "raw_manifest_real_frame_count": raw_manifest.get("real_frame_count"),
        "raw_manifest_padding_frame_count": raw_manifest.get("padding_frame_count"),
        "decode": "raw_rgb_uint8 / 127.5 - 1.0",
        "resize": "cv2.INTER_LINEAR independently over HWC float32 components",
        "postprocess": "per-pixel L2 normalize after resize",
        "black_pixel_rule": "no special invalid interpretation",
        "axis_mapping": None,
        "axis_selection_status": "awaiting_human_audit",
        "files": output_entries,
    }
    manifest_path = manifest_root / "adapter_manifest.json"
    _atomic_write_json(manifest_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=111)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = adapt_scene(args.scene, args.raw_root, args.output, args.expected_count)
    except (OSError, ValueError, AdapterError) as error:
        print(f"DIFFRENDER_NORMAL_ADAPTER=FAIL\n{error}")
        return 1
    print(
        json.dumps(
            {
                "files": len(payload["files"]),
                "axis_selection_status": payload["axis_selection_status"],
                "output": str(Path(args.output).resolve()),
            },
            indent=2,
        )
    )
    print("DIFFRENDER_NORMAL_ADAPTER=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
