"""Auditable packaging for external DiffusionRenderer inverse-rendering output.

This module deliberately contains no model code.  It validates an already
generated chunk tree, maps padded video slots back to an exact scene image set,
and writes the provenance contract consumed by the glass-mask proposal tools.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION


RAW_SCHEMA = "rtgs_diffusion_renderer_raw_v1"
VALIDATION_SCHEMA = "rtgs_diffusion_renderer_raw_validation_v1"
PASSES = ("basecolor", "normal", "depth", "diffuse_albedo")
ALL_KINDS = ("rgb",) + PASSES
NORMAL_DECODE = "rgb_uint8 / 127.5 - 1.0"
DEPTH_CONTRACT = (
    "DiffusionRenderer RGB depth visualization; per-view relative prior only; "
    "never metric and never compared across views without explicit calibration"
)


class DiffusionRendererIdentityError(RuntimeError):
    """The external code, model, config, or generated tree is incomplete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_files(files: Mapping[str, Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        record = files[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise DiffusionRendererIdentityError(f"missing/nonempty identity file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
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


def ordered_scene_images(scene: Path, images: str = "images") -> list[Path]:
    image_root = Path(scene).expanduser().resolve() / images
    if not image_root.is_dir():
        raise DiffusionRendererIdentityError(f"scene image directory is missing: {image_root}")
    paths = sorted(
        path for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise DiffusionRendererIdentityError(f"scene image directory is empty: {image_root}")
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise DiffusionRendererIdentityError("scene images contain duplicate stems")
    return paths


def resized_size(source_size: tuple[int, int], target_hw: tuple[int, int]) -> tuple[int, int]:
    """Mirror DiffusionRenderer's resize_upscale_without_padding dimensions."""
    width, height = source_size
    target_height, target_width = target_hw
    scale = max(target_width / width, target_height / height)
    new_width = max(64, math.ceil(int(width * scale) / 64) * 64)
    new_height = max(64, math.ceil(int(height * scale) / 64) * 64)
    return new_width, new_height


def _read_rgb(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    try:
        with Image.open(path) as opened:
            if opened.mode != "RGB" or opened.size != expected_size:
                raise DiffusionRendererIdentityError(
                    f"unexpected image contract {path}: mode={opened.mode} size={opened.size}; "
                    f"expected RGB {expected_size}"
                )
            values = np.asarray(opened, dtype=np.uint8)
    except OSError as error:
        raise DiffusionRendererIdentityError(f"cannot decode generated image {path}: {error}") from error
    return values


def validate_and_manifest(
    *,
    scene: Path,
    images: str,
    raw_root: Path,
    group_name: str,
    target_hw: tuple[int, int],
    frames_per_chunk: int,
    generation_identity: Mapping[str, Any],
    effective_config: Mapping[str, Any],
    repository_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate ordered upstream slots and return manifest plus diagnostics.

    DiffusionRenderer saves its resized conditioning RGB.  Replaying that resize
    under another Pillow release is not guaranteed bit-exact, so this records a
    tight cross-version interpolation bound instead of inventing pixel identity.
    """
    source_paths = ordered_scene_images(scene, images)
    source_sizes: set[tuple[int, int]] = set()
    source_modes: set[str] = set()
    for path in source_paths:
        with Image.open(path) as opened:
            source_sizes.add(opened.size)
            source_modes.add(opened.mode)
    if len(source_sizes) != 1 or source_modes != {"RGB"}:
        raise DiffusionRendererIdentityError(
            f"source images must share one RGB size, got modes={source_modes} sizes={source_sizes}"
        )
    source_size = next(iter(source_sizes))
    native_size = resized_size(source_size, target_hw)
    raw_root = Path(raw_root).expanduser().resolve()
    generated_root = raw_root / group_name
    if not generated_root.is_dir():
        raise DiffusionRendererIdentityError(f"generated group directory is missing: {generated_root}")
    if frames_per_chunk <= 0:
        raise DiffusionRendererIdentityError("frames_per_chunk must be positive")
    chunk_count = math.ceil(len(source_paths) / frames_per_chunk)
    # The splitter emits a short final chunk, then the inference loop repeats its
    # last source frame to the fixed model length before saving all model slots.
    total_slots = chunk_count * frames_per_chunk
    records: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    output_hashes: dict[str, dict[str, str]] = {}
    pixel_mapping_global_max = 0
    pixel_mapping_global_sum = 0
    pixel_mapping_global_values = 0
    for raw_slot in range(total_slots):
        chunk_index, frame_index = divmod(raw_slot, frames_per_chunk)
        real_index = raw_slot if raw_slot < len(source_paths) else len(source_paths) - 1
        source = source_paths[real_index]
        is_padding = raw_slot >= len(source_paths)
        prefix = f"{chunk_index:04d}.{frame_index:04d}"
        artifacts: dict[str, str] = {}
        hashes: dict[str, str] = {}
        for kind in ALL_KINDS:
            relative = Path(group_name) / f"{prefix}.{kind}.png"
            path = raw_root / relative
            values = _read_rgb(path, native_size)
            artifacts[kind] = relative.as_posix()
            hashes[kind] = sha256_file(path)
            if kind == "rgb":
                with Image.open(source) as opened:
                    replay = np.asarray(
                        opened.convert("RGB").resize(native_size, Image.Resampling.BILINEAR),
                        dtype=np.uint8,
                    )
                absolute = np.abs(values.astype(np.int16) - replay.astype(np.int16))
                difference = int(absolute.max())
                pixel_mapping_global_max = max(pixel_mapping_global_max, difference)
                pixel_mapping_global_sum += int(absolute.sum())
                pixel_mapping_global_values += int(absolute.size)
                if difference > 8 or float(absolute.mean()) > 0.25:
                    raise DiffusionRendererIdentityError(
                        "generated RGB is not aligned with the source replay for slot "
                        f"{raw_slot}: max_abs={difference} mean_abs={float(absolute.mean()):.6f}"
                    )
        record = {
            "raw_slot": raw_slot,
            "chunk_index": chunk_index,
            "frame_index": frame_index,
            "is_padding": is_padding,
            "padding_source_stem": source.stem if is_padding else None,
            "rtgs_real_frame_index": None if is_padding else real_index,
            "rtgs_input_file": None if is_padding else (
                Path("data") / Path(scene).name / images / source.name
            ).as_posix(),
            "input_sha256": sha256_file(source),
            "diffusion_renderer_rgb_file": artifacts["rgb"],
            "raw_prior_files": {kind: artifacts[kind] for kind in PASSES},
            "output_sha256": hashes,
            "source_to_prior_mapping": {
                "operation": "PIL.Image.BILINEAR resize without crop or pad",
                "source_size": list(source_size),
                "native_size": list(native_size),
                "validation_pillow_version": PILLOW_VERSION,
                "cross_version_max_abs_bound": 8,
                "cross_version_mean_abs_bound": 0.25,
            },
        }
        records.append(record)
        if not is_padding:
            source_hashes[source.stem] = record["input_sha256"]
            output_hashes[source.stem] = hashes
    expected_pngs = total_slots * len(ALL_KINDS)
    actual_pngs = sorted(generated_root.glob("*.png"))
    if len(actual_pngs) != expected_pngs:
        raise DiffusionRendererIdentityError(
            f"generated PNG count mismatch: {len(actual_pngs)} != {expected_pngs}"
        )
    success_root = raw_root / "TMP_SUCCESS_SIGNAL"
    success_files = sorted(path for path in success_root.iterdir() if path.is_file()) \
        if success_root.is_dir() else []
    if len(success_files) != chunk_count:
        raise DiffusionRendererIdentityError(
            f"success signal count mismatch: {len(success_files)} != {chunk_count}"
        )
    manifest = {
        "schema": RAW_SCHEMA,
        "artifact_role": "raw_diffusion_renderer_inverse_priors",
        "human_status": "machine_generated_audited",
        "scene": Path(scene).expanduser().resolve().name,
        "images_directory": images,
        "ordered_stems": [path.stem for path in source_paths],
        "real_frame_count": len(source_paths),
        "padding_frame_count": total_slots - len(source_paths),
        "frames_per_chunk": frames_per_chunk,
        "chunk_count": chunk_count,
        "model_passes": list(PASSES),
        "artifact_contracts": {
            kind: {
                "mode": "RGB", "dtype": "uint8",
                "shape_hwc": [native_size[1], native_size[0], 3],
                "representation": (
                    NORMAL_DECODE if kind == "normal" else
                    DEPTH_CONTRACT if kind == "depth" else
                    "DiffusionRenderer PNG output in its native encoded display domain"
                ),
            }
            for kind in ALL_KINDS
        },
        "generation_identity": dict(generation_identity),
        "effective_config": dict(effective_config),
        "source_to_prior_mapping": {
            "source_resolution": {"width": source_size[0], "height": source_size[1]},
            "native_prior_resolution": {"width": native_size[0], "height": native_size[1]},
            "resize_filter": "PIL.Image.BILINEAR",
            "crop": None,
            "pad": None,
        },
        "normal_coordinate_convention": {
            "stored": "RGB uint8",
            "decode": NORMAL_DECODE,
            "space": "DiffusionRenderer output convention",
            "axis_mapping": "unconfirmed; a later scene-specific axis audit is mandatory",
        },
        "depth_representation": DEPTH_CONTRACT,
        "source_rgb_sha256": source_hashes,
        "real_output_sha256": output_hashes,
        "frames_and_padding": records,
    }
    validation = {
        "schema": VALIDATION_SCHEMA,
        "validation": "PASS",
        "real_frames": len(source_paths),
        "padding_frames": total_slots - len(source_paths),
        "chunks": chunk_count,
        "ordered_stems_match": True,
        "source_modes": ["RGB"],
        "source_resolution": list(source_size),
        "native_prior_resolution": list(native_size),
        "output_mode": "RGB",
        "output_dtype": "uint8",
        "outputs_per_slot": len(ALL_KINDS),
        "pixel_mapping_global_max_abs_difference": pixel_mapping_global_max,
        "pixel_mapping_global_mean_abs_difference": (
            pixel_mapping_global_sum / pixel_mapping_global_values
            if pixel_mapping_global_values else 0.0
        ),
        "pixel_mapping_validation_pillow_version": PILLOW_VERSION,
        "pixel_mapping_interpretation": (
            "DiffusionRenderer-saved RGB is authoritative; source replay is bounded because "
            "Pillow bilinear interpolation can differ slightly across releases"
        ),
        "normal_coordinate_convention": manifest["normal_coordinate_convention"],
        "depth_representation": DEPTH_CONTRACT,
        "metric_depth": False,
    }
    return manifest, validation
