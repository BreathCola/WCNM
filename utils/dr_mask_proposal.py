"""Fail-closed DiffusionRenderer audit and glass-mask proposal utilities.

The generated masks are review proposals only.  This module deliberately has
no dependency on the Stage B training mask loader.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw


SCHEMA_VERSION = 1
METHOD_VERSION = "dr-glass-proposal-v1"
NORMAL_EPS = 1e-2
RGB_REPLAY_MAX_ABS_DIFFERENCE = 2
REQUIRED_ARTIFACTS = ("normal", "depth")
OPTIONAL_ARTIFACTS = ("basecolor", "diffuse_albedo")
ALL_RAW_KINDS = ("rgb",) + REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS
PROPOSAL_FILES = (
    "rgb.png",
    "dr_normal.png",
    "dr_depth.png",
    "normal_boundary.png",
    "depth_boundary.png",
    "proposal_soft.png",
    "proposal_hard_preview.png",
    "proposal_overlay.png",
    "uncertainty.png",
    "proposal_metadata.json",
)
RISK_NAMES = (
    "bird_may_be_included",
    "background_may_be_included",
    "glass_edge_may_be_missing",
    "reflection_may_be_misclassified",
    "low_confidence_region",
)


class ProposalAuditError(RuntimeError):
    """Raised when source/DR provenance is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _atomic_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="PNG")
        with Image.open(temporary) as reread:
            reread.load()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_rgb(path: Path, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    try:
        with Image.open(path) as opened:
            if opened.mode != "RGB":
                raise ProposalAuditError(f"expected RGB image, got mode={opened.mode}: {path}")
            if expected_size is not None and opened.size != expected_size:
                raise ProposalAuditError(
                    f"image size mismatch: {path}: {opened.size} != {expected_size}"
                )
            values = np.asarray(opened, dtype=np.uint8)
    except (OSError, ValueError) as error:
        if isinstance(error, ProposalAuditError):
            raise
        raise ProposalAuditError(f"cannot parse image {path}: {error}") from error
    if values.ndim != 3 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise ProposalAuditError(f"invalid RGB array: {path}: {values.shape} {values.dtype}")
    return values


def decode_normal(raw_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw_rgb)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[-1] != 3:
        raise ProposalAuditError("DR normal must be uint8 HWC RGB")
    decoded = values.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    lengths = np.linalg.norm(decoded, axis=-1)
    valid = np.isfinite(decoded).all(axis=-1) & (lengths > NORMAL_EPS)
    unit = np.zeros_like(decoded, dtype=np.float32)
    unit[valid] = decoded[valid] / lengths[valid, None]
    return unit, valid


def _record_raw_path(raw_root: Path, record: dict[str, Any], kind: str) -> Path:
    if kind == "rgb":
        relative = record.get("diffusion_renderer_rgb_file")
    else:
        relative = record.get("raw_prior_files", {}).get(kind)
    if not isinstance(relative, str) or not relative:
        raise ProposalAuditError(
            f"raw slot {record.get('raw_slot')}: missing {kind} path in manifest"
        )
    path = raw_root / relative
    if not path.is_file():
        raise ProposalAuditError(f"missing {kind} artifact: {path}")
    return path


def _duplicates(values: Iterable[Any]) -> list[Any]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def audit_dr_artifacts(
    scene: Path,
    raw_root: Path,
    expected_count: int = 111,
) -> dict[str, Any]:
    """Validate exact RGB-to-real-slot mapping and every real DR artifact."""
    scene = Path(scene).expanduser().resolve()
    raw_root = Path(raw_root).expanduser().resolve()
    manifest_path = raw_root / "manifest.json"
    validation_path = raw_root / "validation_summary.json"
    if not manifest_path.is_file():
        raise ProposalAuditError(f"missing DR manifest: {manifest_path}")
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProposalAuditError(f"cannot parse DR manifest: {error}") from error
    if not validation_path.is_file():
        raise ProposalAuditError(f"missing DR generation validation: {validation_path}")
    try:
        raw_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProposalAuditError(f"cannot parse DR generation validation: {error}") from error
    if raw_validation.get("validation") != "PASS":
        raise ProposalAuditError("DR generation validation did not pass")
    if raw_validation.get("pixel_mapping_global_max_abs_difference") != 0:
        raise ProposalAuditError("DR generation-time RGB alignment was not exact")
    if raw_validation.get("real_frames") != expected_count:
        raise ProposalAuditError("DR validation real-frame count mismatch")
    records = raw_manifest.get("frames_and_padding")
    if not isinstance(records, list):
        raise ProposalAuditError("DR manifest frames_and_padding must be a list")
    real = [record for record in records if isinstance(record, dict) and not record.get("is_padding")]
    padding = [record for record in records if isinstance(record, dict) and record.get("is_padding")]
    if len(real) != expected_count or raw_manifest.get("real_frame_count") != expected_count:
        raise ProposalAuditError(
            f"expected {expected_count} real records, got {len(real)}"
        )
    expected_padding = int(raw_manifest.get("padding_frame_count", -1))
    if len(padding) != expected_padding:
        raise ProposalAuditError(
            f"padding record mismatch: manifest={expected_padding} records={len(padding)}"
        )
    if any(record.get("rtgs_input_file") is not None for record in padding):
        raise ProposalAuditError("padding record is mapped to a real RT-GS image")
    indices = [record.get("rtgs_real_frame_index") for record in real]
    if indices != list(range(expected_count)):
        raise ProposalAuditError("real DR records are missing, duplicated, or out of order")

    image_root = scene / "images"
    image_paths = sorted(path for path in image_root.iterdir() if path.is_file())
    expected_names = [f"{index:06d}.jpg" for index in range(expected_count)]
    if [path.name for path in image_paths] != expected_names:
        raise ProposalAuditError("RGB source set is missing, unexpected, or out of order")

    mapping = raw_manifest.get("source_to_prior_mapping")
    if not isinstance(mapping, dict):
        raise ProposalAuditError("DR manifest lacks source_to_prior_mapping")
    source_resolution = mapping.get("source_resolution")
    native_resolution = mapping.get("native_prior_resolution")
    if not isinstance(source_resolution, dict) or not isinstance(native_resolution, dict):
        raise ProposalAuditError("DR manifest lacks source/native resolution")
    source_size = (int(source_resolution["width"]), int(source_resolution["height"]))
    native_size = (int(native_resolution["width"]), int(native_resolution["height"]))
    if source_size[0] <= 0 or source_size[1] <= 0 or native_size[0] <= 0 or native_size[1] <= 0:
        raise ProposalAuditError("invalid source/native resolution in DR manifest")
    if mapping.get("crop") is not None or mapping.get("pad") is not None:
        raise ProposalAuditError("this audit supports only the recorded no-crop/no-pad mapping")
    if mapping.get("resize_filter") != "PIL.Image.BILINEAR":
        raise ProposalAuditError("unexpected DR RGB resize filter")

    entries: list[dict[str, Any]] = []
    raw_paths_seen: list[str] = []
    artifact_summaries = {
        kind: {
            "count": 0,
            "global_min": 255,
            "global_max": 0,
            "valid_pixel_count": 0,
            "pixel_count": 0,
        }
        for kind in ALL_RAW_KINDS
    }
    rgb_mapping_max_difference = 0
    for index, record in enumerate(real):
        stem = f"{index:06d}"
        expected_source_text = f"data/TiHuBird/images/{stem}.jpg"
        if record.get("rtgs_input_file") != expected_source_text:
            raise ProposalAuditError(
                f"{stem}: source mapping mismatch: {record.get('rtgs_input_file')!r}"
            )
        source = image_root / f"{stem}.jpg"
        source_sha = sha256_file(source)
        if source_sha != record.get("input_sha256"):
            raise ProposalAuditError(f"{stem}: source SHA-256 mismatch")
        source_rgb = _read_rgb(source, source_size)
        artifact_entry: dict[str, Any] = {}
        artifact_values: dict[str, np.ndarray] = {}
        for kind in ALL_RAW_KINDS:
            path = _record_raw_path(raw_root, record, kind)
            values = _read_rgb(path, native_size)
            artifact_values[kind] = values
            raw_paths_seen.append(str(path.resolve()))
            summary = artifact_summaries[kind]
            summary["count"] += 1
            summary["global_min"] = min(summary["global_min"], int(values.min()))
            summary["global_max"] = max(summary["global_max"], int(values.max()))
            summary["pixel_count"] += int(values.shape[0] * values.shape[1])
            summary["valid_pixel_count"] += int(values.shape[0] * values.shape[1])
            artifact_entry[kind] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "mode": "RGB",
                "dtype": "uint8",
                "size": [native_size[0], native_size[1]],
                "channel_min": [int(v) for v in values.reshape(-1, 3).min(axis=0)],
                "channel_max": [int(v) for v in values.reshape(-1, 3).max(axis=0)],
            }

        resized_source = np.asarray(
            Image.fromarray(source_rgb).resize(native_size, Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        difference = int(
            np.abs(resized_source.astype(np.int16) - artifact_values["rgb"].astype(np.int16)).max()
        )
        rgb_mapping_max_difference = max(rgb_mapping_max_difference, difference)
        if difference > RGB_REPLAY_MAX_ABS_DIFFERENCE:
            raise ProposalAuditError(
                f"{stem}: DR RGB/source mapping replay differs by {difference} "
                f"(allowed cross-Pillow replay tolerance={RGB_REPLAY_MAX_ABS_DIFFERENCE})"
            )

        normal_unit, normal_valid = decode_normal(artifact_values["normal"])
        invalid_normal_count = int((~normal_valid).sum())
        if invalid_normal_count:
            raise ProposalAuditError(
                f"{stem}: DR normal has {invalid_normal_count} non-finite/near-zero pixels"
            )
        normal_lengths = np.linalg.norm(normal_unit[normal_valid], axis=-1)
        artifact_entry["normal"].update(
            {
                "decode": "rgb_uint8 / 127.5 - 1.0, then unit normalize for continuity only",
                "axis_convention": "unconfirmed DiffusionRenderer component order",
                "invalid_encoding": "no black sentinel; non-finite or decoded norm <= 0.01 fails",
                "valid_pixel_ratio": float(normal_valid.mean()),
                "unit_length_min": float(normal_lengths.min()),
                "unit_length_max": float(normal_lengths.max()),
            }
        )
        depth = artifact_values["depth"]
        depth_channels = depth.astype(np.float32) / np.float32(255.0)
        depth_scalar = depth_channels.mean(axis=-1)
        artifact_entry["depth"].update(
            {
                "semantic": "per-frame relative RGB depth visualization; no COLMAP scale",
                "invalid_encoding": "no invalid sentinel is documented; black is retained as data",
                "valid_pixel_ratio": 1.0,
                "all_black_pixel_ratio": float(np.all(depth == 0, axis=-1).mean()),
                "mean_channel_spread": float(
                    (depth_channels.max(axis=-1) - depth_channels.min(axis=-1)).mean()
                ),
                "scalar_luminance_min": float(depth_scalar.min()),
                "scalar_luminance_max": float(depth_scalar.max()),
            }
        )
        entries.append(
            {
                "stem": stem,
                "frame_index": index,
                "raw_slot": int(record["raw_slot"]),
                "chunk_index": int(record["chunk_index"]),
                "frame_index_within_chunk": int(record["frame_index"]),
                "is_padding": False,
                "rgb": {
                    "path": str(source.resolve()),
                    "sha256": source_sha,
                    "mode": "RGB",
                    "dtype": "uint8",
                    "size": [source_size[0], source_size[1]],
                },
                "artifacts": artifact_entry,
                "source_to_prior_mapping": record.get("source_to_prior_mapping"),
                "rgb_mapping_max_abs_difference": difference,
            }
        )

    duplicates = _duplicates(raw_paths_seen)
    if duplicates:
        raise ProposalAuditError(f"duplicate real-frame artifact paths: {duplicates}")
    expected_raw_path_count = expected_count * len(ALL_RAW_KINDS)
    if len(raw_paths_seen) != expected_raw_path_count:
        raise ProposalAuditError(
            f"real artifact path count={len(raw_paths_seen)}, expected={expected_raw_path_count}"
        )
    for summary in artifact_summaries.values():
        summary["valid_pixel_ratio"] = (
            summary.pop("valid_pixel_count") / summary["pixel_count"]
        )
    padding_slots = []
    for record in padding:
        paths = {
            kind: str(_record_raw_path(raw_root, record, kind).resolve())
            for kind in ALL_RAW_KINDS
        }
        padding_slots.append(
            {
                "raw_slot": int(record["raw_slot"]),
                "chunk_index": int(record["chunk_index"]),
                "frame_index_within_chunk": int(record["frame_index"]),
                "is_padding": True,
                "rtgs_input_file": None,
                "paths": paths,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "DiffusionRenderer glass-mask proposal input audit",
        "status": "PASS",
        "scene": str(scene),
        "raw_root": str(raw_root),
        "raw_manifest": str(manifest_path.resolve()),
        "raw_manifest_sha256": sha256_file(manifest_path),
        "generation_validation": str(validation_path.resolve()),
        "generation_validation_sha256": sha256_file(validation_path),
        "generation_validation_pixel_mapping_max_abs_difference": 0,
        "real_frame_count": len(entries),
        "padding_frame_count": len(padding_slots),
        "padding_policy": "excluded from every RGB/stem/proposal mapping",
        "source_resolution": [source_size[0], source_size[1]],
        "native_dr_resolution": [native_size[0], native_size[1]],
        "resolution_relation": (
            "DR artifacts intentionally remain in the manifest-declared native domain; "
            "proposal generation explicitly maps them to source resolution"
        ),
        "normal_contract": {
            "raw": "RGB uint8 [0,255]",
            "decode": "rgb / 127.5 - 1.0",
            "axis_convention": "unconfirmed; no semantic axis mapping is applied",
            "continuity_use": "unit-vector dot products only, invariant to signed axis permutations",
            "invalid_encoding": "no black sentinel; non-finite or decoded norm <= 0.01 fails closed",
        },
        "depth_contract": {
            "raw": "RGB uint8 [0,255]",
            "metric_status": "relative per-frame visualization only",
            "cross_view_comparison": False,
            "colmap_scale": False,
            "invalid_encoding": "none documented; black is not discarded",
            "proposal_use": "within-view robust luminance boundaries/continuity only",
        },
        "rgb_mapping_global_max_abs_difference": rgb_mapping_max_difference,
        "rgb_mapping_replay_tolerance": RGB_REPLAY_MAX_ABS_DIFFERENCE,
        "rgb_mapping_replay_note": (
            "Generation-time validation was exact. The current Pillow replay may differ by "
            "at most two uint8 levels; native proposal cues use the stored DR RGB artifact."
        ),
        "artifact_summaries": artifact_summaries,
        "padding_slots_excluded": padding_slots,
        "entries": entries,
    }


def _robust_unit(values: np.ndarray, low: float = 0.02, high: float = 0.98) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.quantile(values, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _gradient(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    gx = cv2.Scharr(values, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(values, cv2.CV_32F, 0, 1)
    return np.sqrt(gx * gx + gy * gy)


def boundary_maps(normal_rgb: np.ndarray, depth_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit, valid = decode_normal(normal_rgb)
    normal_grad = np.sqrt(sum(_gradient(unit[..., channel]) ** 2 for channel in range(3)))
    normal_grad[~valid] = normal_grad.max(initial=0.0)
    depth = depth_rgb.astype(np.float32).mean(axis=-1) / np.float32(255.0)
    return _robust_unit(normal_grad, 0.20, 0.985), _robust_unit(_gradient(depth), 0.20, 0.985)


def _morph(mask: np.ndarray, close_size: int = 11) -> np.ndarray:
    values = (np.asarray(mask) > 0).astype(np.uint8)
    # Keep the production-scale kernels unchanged, but avoid letting a fixed
    # 11--15 px kernel swallow tiny synthetic fixtures used by the contract
    # tests.  The cap is deliberately derived only from the audited image size.
    maximum = max(3, min(values.shape[:2]) // 6)
    if maximum % 2 == 0:
        maximum -= 1
    close_size = max(3, min(close_size, maximum))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    values = cv2.morphologyEx(values, cv2.MORPH_CLOSE, close_kernel)
    values = cv2.morphologyEx(values, cv2.MORPH_OPEN, open_kernel)
    return values


@dataclass
class Candidate:
    hull: np.ndarray
    mask: np.ndarray
    score: float
    source: str
    diagnostics: dict[str, float]


def _components(mask: np.ndarray, source: str) -> list[tuple[np.ndarray, str]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    result = []
    total = mask.size
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area / total < 0.018 or area / total > 0.48:
            continue
        component = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        if hull.shape[0] >= 3:
            result.append((hull, source))
    return result


def _hull_mask(shape: tuple[int, int], hull: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.reshape(-1, 2), 1)
    return mask


def propose_native_mask(
    rgb: np.ndarray,
    normal_rgb: np.ndarray,
    depth_rgb: np.ndarray,
    basecolor_rgb: np.ndarray | None,
) -> dict[str, Any]:
    """Combine DR geometry, RGB edges, connectivity, and optional basecolor."""
    height, width = depth_rgb.shape[:2]
    if rgb.shape[:2] != (height, width) or normal_rgb.shape[:2] != (height, width):
        raise ProposalAuditError("proposal inputs must share the audited native resolution")
    normal_boundary, depth_boundary = boundary_maps(normal_rgb, depth_rgb)
    normal_unit, normal_valid = decode_normal(normal_rgb)
    normal_smooth = np.clip(1.0 - cv2.GaussianBlur(normal_boundary, (0, 0), 1.5), 0, 1)
    normal_smooth[~normal_valid] = 0.0
    depth_gray = depth_rgb.astype(np.float32).mean(axis=-1) / np.float32(255.0)
    depth_unit = _robust_unit(depth_gray, 0.02, 0.98)
    depth_u8 = np.round(depth_unit * 255.0).astype(np.uint8)
    _, depth_dark = cv2.threshold(depth_u8, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    depth_dark = _morph(depth_dark, 13)

    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / np.float32(255.0)
    rgb_boundary = _robust_unit(_gradient(rgb_gray), 0.30, 0.99)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    rgb_highlight = (hsv[..., 1] / 255.0 < 0.28) & (hsv[..., 2] / 255.0 > 0.62)

    if basecolor_rgb is not None:
        if basecolor_rgb.shape[:2] != (height, width):
            raise ProposalAuditError("basecolor does not match audited native resolution")
        base_hsv = cv2.cvtColor(basecolor_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        base_score = (base_hsv[..., 2] / 255.0) * (1.0 - 0.72 * base_hsv[..., 1] / 255.0)
        threshold = max(0.42, float(np.quantile(base_score, 0.70)))
        base_bright = _morph(base_score >= threshold, 13)
    else:
        base_score = np.zeros((height, width), dtype=np.float32)
        base_bright = np.zeros((height, width), dtype=np.uint8)

    yy, xx = np.mgrid[:height, :width]
    center_prior = np.exp(
        -0.5 * (((xx / width - 0.5) / 0.34) ** 2 + ((yy / height - 0.52) / 0.40) ** 2)
    ).astype(np.float32)
    center_gate = center_prior > 0.22
    seed = depth_dark & center_gate & ((base_bright > 0) | (normal_smooth > 0.62))
    seed = _morph(seed, 15)
    consensus = _morph(((depth_dark > 0) & (base_bright > 0) & center_gate), 11)

    hulls: list[tuple[np.ndarray, str]] = []
    hulls.extend(_components(depth_dark & center_gate, "depth"))
    hulls.extend(_components(base_bright & center_gate, "basecolor"))
    hulls.extend(_components(seed, "geometry_consensus"))
    hulls.extend(_components(consensus, "depth_base_consensus"))
    if not hulls:
        raise ProposalAuditError("no connected glass proposal candidate survived")

    fused_boundary = np.maximum.reduce((normal_boundary, depth_boundary, rgb_boundary))
    candidates: list[Candidate] = []
    for hull, source in hulls:
        mask = _hull_mask((height, width), hull)
        area = float(mask.mean())
        if area <= 0:
            continue
        moments = cv2.moments(mask)
        cx = moments["m10"] / max(moments["m00"], 1e-6)
        cy = moments["m01"] / max(moments["m00"], 1e-6)
        centrality = float(center_prior[int(np.clip(cy, 0, height - 1)), int(np.clip(cx, 0, width - 1))])
        area_prior = math.exp(-0.5 * (math.log(max(area, 1e-5) / 0.14) / 0.75) ** 2)
        inside = mask.astype(bool)
        depth_support = float(depth_dark[inside].mean())
        base_support = float(base_score[inside].mean())
        smooth_support = float(normal_smooth[inside].mean())
        ring = cv2.dilate(mask, np.ones((7, 7), np.uint8)) - cv2.erode(
            mask, np.ones((7, 7), np.uint8)
        )
        boundary_support = float(fused_boundary[ring > 0].mean()) if np.any(ring) else 0.0
        touches = float(
            mask[0].mean() + mask[-1].mean() + mask[:, 0].mean() + mask[:, -1].mean()
        )
        score = (
            0.25 * depth_support
            + 0.18 * base_support
            + 0.13 * smooth_support
            + 0.18 * boundary_support
            + 0.16 * centrality
            + 0.10 * area_prior
            - 0.20 * min(touches, 1.0)
        )
        candidates.append(
            Candidate(
                hull=hull,
                mask=mask,
                score=float(score),
                source=source,
                diagnostics={
                    "area_ratio": area,
                    "depth_support": depth_support,
                    "basecolor_support": base_support,
                    "normal_smooth_support": smooth_support,
                    "boundary_support": boundary_support,
                    "centrality": centrality,
                    "area_prior": area_prior,
                    "border_touch_score": touches,
                },
            )
        )
    if not candidates:
        raise ProposalAuditError("all glass proposal candidates were invalid")
    best = max(candidates, key=lambda item: item.score)

    inside_distance = cv2.distanceTransform(best.mask, cv2.DIST_L2, 5)
    outside_distance = cv2.distanceTransform(1 - best.mask, cv2.DIST_L2, 5)
    signed_distance = inside_distance - outside_distance
    feather = 3.0
    soft = np.clip((signed_distance + feather) / (2.0 * feather), 0.0, 1.0)
    local_cues = np.stack(
        [depth_dark.astype(np.float32), base_score, normal_smooth, 1.0 - rgb_boundary], axis=-1
    )
    disagreement = local_cues.std(axis=-1)
    boundary_uncertainty = np.exp(-np.abs(signed_distance) / 5.0)
    uncertainty = np.clip(
        0.62 * boundary_uncertainty
        + 0.28 * disagreement * best.mask
        + 0.10 * (1.0 - np.clip(best.score, 0.0, 1.0)) * best.mask,
        0.0,
        1.0,
    )
    hard = soft >= 0.5
    contour, _ = cv2.findContours(hard.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_length = float(sum(cv2.arcLength(item, True) for item in contour))
    ys, xs = np.nonzero(hard)
    if xs.size == 0:
        raise ProposalAuditError("selected proposal is empty")
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    high_texture_inside = float((rgb_boundary[hard] > 0.65).mean())
    weak_cue_inside = float(
        ((depth_dark[hard] == 0) & (base_score[hard] < 0.45)).mean()
    )
    uncertain_fraction = float((uncertainty > 0.55).mean())
    risks = {
        "bird_may_be_included": {
            "flag": bool(high_texture_inside > 0.035),
            "reason": f"high RGB-edge fraction inside proposal={high_texture_inside:.4f}",
        },
        "background_may_be_included": {
            "flag": bool(weak_cue_inside > 0.12 or best.diagnostics["border_touch_score"] > 0.02),
            "reason": f"weak interior cue fraction={weak_cue_inside:.4f}",
        },
        "glass_edge_may_be_missing": {
            "flag": bool(best.diagnostics["boundary_support"] < 0.30),
            "reason": f"fused boundary support={best.diagnostics['boundary_support']:.4f}",
        },
        "reflection_may_be_misclassified": {
            "flag": bool((rgb_highlight & (uncertainty > 0.35)).mean() > 0.003),
            "reason": "RGB highlight regions overlap proposal uncertainty",
        },
        "low_confidence_region": {
            "flag": bool(uncertain_fraction > 0.025 or best.score < 0.58),
            "reason": f"uncertain image fraction={uncertain_fraction:.4f}, candidate score={best.score:.4f}",
        },
    }
    return {
        "soft": soft.astype(np.float32),
        "hard": hard,
        "uncertainty": uncertainty.astype(np.float32),
        "normal_boundary": normal_boundary,
        "depth_boundary": depth_boundary,
        "candidate": best,
        "bbox_native": bbox,
        "edge_length_native": edge_length,
        "confidence": np.clip(1.0 - uncertainty, 0.0, 1.0),
        "risks": risks,
        "cue_diagnostics": {
            **best.diagnostics,
            "candidate_score": best.score,
            "candidate_source": best.source,
            "high_rgb_texture_inside": high_texture_inside,
            "weak_cue_inside": weak_cue_inside,
            "uncertain_fraction": uncertain_fraction,
        },
    }


def _resize_scalar(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(np.asarray(values, dtype=np.float32), size, interpolation=cv2.INTER_LINEAR)


def _gray_image(values: np.ndarray) -> Image.Image:
    return Image.fromarray(np.round(np.clip(values, 0, 1) * 255).astype(np.uint8))


def _overlay(rgb: np.ndarray, soft: np.ndarray) -> Image.Image:
    base = rgb.astype(np.float32)
    color = np.zeros_like(base)
    color[..., 0] = 30
    color[..., 1] = 225
    color[..., 2] = 210
    alpha = np.clip(soft[..., None] * 0.48, 0, 0.48)
    result = np.round(base * (1.0 - alpha) + color * alpha).astype(np.uint8)
    return Image.fromarray(result)


def generate_view_proposal(
    audit_entry: dict[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    source_path = Path(audit_entry["rgb"]["path"])
    artifacts = audit_entry["artifacts"]
    source_rgb = _read_rgb(source_path, tuple(audit_entry["rgb"]["size"]))
    normal = _read_rgb(Path(artifacts["normal"]["path"]), tuple(artifacts["normal"]["size"]))
    depth = _read_rgb(Path(artifacts["depth"]["path"]), tuple(artifacts["depth"]["size"]))
    native_rgb = _read_rgb(Path(artifacts["rgb"]["path"]), tuple(artifacts["rgb"]["size"]))
    basecolor = (
        _read_rgb(Path(artifacts["basecolor"]["path"]), tuple(artifacts["basecolor"]["size"]))
        if "basecolor" in artifacts else None
    )
    native_size = (normal.shape[1], normal.shape[0])
    source_size = (source_rgb.shape[1], source_rgb.shape[0])
    proposal = propose_native_mask(native_rgb, normal, depth, basecolor)
    soft = _resize_scalar(proposal["soft"], source_size)
    uncertainty = _resize_scalar(proposal["uncertainty"], source_size)
    normal_boundary = _resize_scalar(proposal["normal_boundary"], source_size)
    depth_boundary = _resize_scalar(proposal["depth_boundary"], source_size)
    hard = soft >= 0.5
    normal_display = Image.fromarray(normal).resize(source_size, Image.Resampling.BILINEAR)
    depth_display = Image.fromarray(depth).resize(source_size, Image.Resampling.BILINEAR)

    output_directory = Path(output_directory)
    _atomic_image(output_directory / "rgb.png", Image.fromarray(source_rgb))
    _atomic_image(output_directory / "dr_normal.png", normal_display)
    _atomic_image(output_directory / "dr_depth.png", depth_display)
    _atomic_image(output_directory / "normal_boundary.png", _gray_image(normal_boundary))
    _atomic_image(output_directory / "depth_boundary.png", _gray_image(depth_boundary))
    _atomic_image(output_directory / "proposal_soft.png", _gray_image(soft))
    _atomic_image(
        output_directory / "proposal_hard_preview.png",
        Image.fromarray(hard.astype(np.uint8) * 255),
    )
    _atomic_image(output_directory / "proposal_overlay.png", _overlay(source_rgb, soft))
    _atomic_image(output_directory / "uncertainty.png", _gray_image(uncertainty))

    ys, xs = np.nonzero(hard)
    confidence_values = (1.0 - uncertainty)[hard]
    scale_x = source_size[0] / native_size[0]
    scale_y = source_size[1] / native_size[1]
    native_bbox = proposal["bbox_native"]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "automatic glass-mask review proposal",
        "training_role": None,
        "warning": "Not reviewed, not ground truth, and forbidden for L_spec.",
        "method_version": METHOD_VERSION,
        "stem": audit_entry["stem"],
        "source_size": [source_size[0], source_size[1]],
        "native_dr_size": [native_size[0], native_size[1]],
        "explicit_mapping": "manifest-verified bilinear source/DR pixel-center mapping",
        "area_ratio": float(hard.mean()),
        "soft_area_ratio": float(soft.mean()),
        "bbox_xyxy": [
            int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        ],
        "bbox_from_native_xyxy": [
            int(round(native_bbox[0] * scale_x)),
            int(round(native_bbox[1] * scale_y)),
            int(round(native_bbox[2] * scale_x)),
            int(round(native_bbox[3] * scale_y)),
        ],
        "edge_length_pixels_approx": float(
            proposal["edge_length_native"] * 0.5 * (scale_x + scale_y)
        ),
        "confidence": {
            "min": float(confidence_values.min()),
            "mean": float(confidence_values.mean()),
            "p05": float(np.quantile(confidence_values, 0.05)),
            "p50": float(np.quantile(confidence_values, 0.50)),
            "p95": float(np.quantile(confidence_values, 0.95)),
            "max": float(confidence_values.max()),
        },
        "uncertainty_fraction_gt_0_55": float((uncertainty > 0.55).mean()),
        "cue_diagnostics": proposal["cue_diagnostics"],
        "risks": proposal["risks"],
        "input_sha256": {
            "rgb": audit_entry["rgb"]["sha256"],
            "normal": artifacts["normal"]["sha256"],
            "depth": artifacts["depth"]["sha256"],
            "basecolor": artifacts.get("basecolor", {}).get("sha256"),
            "diffuse_albedo": artifacts.get("diffuse_albedo", {}).get("sha256"),
        },
        "files": list(PROPOSAL_FILES),
    }
    _atomic_json(output_directory / "proposal_metadata.json", metadata)
    return metadata


def _contact_panel(path: Path, size: tuple[int, int], mode: str = "RGB") -> Image.Image:
    with Image.open(path) as opened:
        if mode == "RGB":
            image = opened.convert("RGB")
        else:
            gray = opened.convert("L")
            image = Image.merge("RGB", (gray, gray, gray))
        return image.resize(size, Image.Resampling.LANCZOS)


def make_contact_sheet(proposal_root: Path, stems: Sequence[str], output: Path) -> None:
    panel_size = (420, 236)
    margin = 8
    title_height = 28
    labels = ("RGB", "DR normal", "DR depth", "proposal overlay", "uncertainty")
    names = ("rgb.png", "dr_normal.png", "dr_depth.png", "proposal_overlay.png", "uncertainty.png")
    width = margin + len(names) * (panel_size[0] + margin)
    height = margin + len(stems) * (panel_size[1] + title_height + margin)
    sheet = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    for row, stem in enumerate(stems):
        y = margin + row * (panel_size[1] + title_height + margin)
        directory = Path(proposal_root) / stem
        for column, (name, label) in enumerate(zip(names, labels)):
            x = margin + column * (panel_size[0] + margin)
            panel = _contact_panel(directory / name, panel_size, mode="L" if name == "uncertainty.png" else "RGB")
            sheet.paste(panel, (x, y + title_height))
            draw.text((x + 4, y + 5), f"{stem} {label}", fill=(235, 235, 235))
    _atomic_image(Path(output), sheet)


def review_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact": "glass-mask human review handoff contract",
        "proposal_soft": "automatic drafts; read-only evidence; never load for training",
        "review_queue": "future human working copies and per-frame review notes",
        "reviewed_soft": "future human-confirmed masks only; deliberately absent in this run",
        "reviewed_requirements": {
            "filename": "must match the RGB stem",
            "size": [3827, 2152],
            "encoding": "single-channel uint8: 0 non-glass, 255 glass, gray edge transition allowed",
            "outside_glass": 0,
            "semantic": (
                "camera-view projection covered by the glass enclosure; bird/background seen "
                "through glass are not separate mask targets"
            ),
        },
        "training_manifest_created": False,
        "lambda_spec_authorized": False,
    }


def generate_proposals(
    audit: dict[str, Any],
    output_root: Path,
    stems: Sequence[str],
) -> list[dict[str, Any]]:
    output_root = Path(output_root)
    proposal_root = output_root / "proposal_soft"
    by_stem = {entry["stem"]: entry for entry in audit["entries"]}
    if len(stems) != len(set(stems)):
        raise ProposalAuditError("proposal stem list contains duplicates")
    missing = sorted(set(stems) - set(by_stem))
    if missing:
        raise ProposalAuditError(f"proposal stems are absent from strict audit: {missing}")
    metadata = []
    for stem in stems:
        metadata.append(generate_view_proposal(by_stem[stem], proposal_root / stem))
    make_contact_sheet(proposal_root, stems, output_root / "contact_sheet_9views.png")
    _atomic_json(output_root / "review_contract.json", review_contract())
    _atomic_json(
        output_root / "proposal_index.json",
        {
            "schema_version": SCHEMA_VERSION,
            "artifact": "automatic glass-mask proposals",
            "training_role": None,
            "warning": "Nine-view human-review input only; not reviewed_soft and not L_spec supervision.",
            "stems": list(stems),
            "proposal_directories": [f"proposal_soft/{stem}" for stem in stems],
            "contact_sheet": "contact_sheet_9views.png",
            "reviewed_soft_created": False,
            "training_manifest_created": False,
        },
    )
    return metadata
