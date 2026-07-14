"""Stage D internal-object mask proposal/review and object-occupancy losses.

Proposal masks are human-review artifacts only.  Training accepts only the
formal reviewed manifest validated here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


PROPOSAL_ROLE = "stage_d_internal_object_mask_proposal"
REVIEWED_ROLE = "stage_d_internal_object_masks_reviewed"
SCHEMA_VERSION = 1
INTERNAL_OBJECT_SEMANTICS_VERSION = "tihubird_bird_and_internal_base_v3"
MASK_INTERPOLATION = "opencv.INTER_NEAREST"
HARD_THRESHOLD = 0.5
EXPECTED_STEMS = [f"{index:06d}" for index in range(111)]
MASK_ROLES = ("bird", "internal_base", "internal_object_union")
OPTIONAL_MASK_ROLES = ("internal_ignore",)
LEGACY_BASE_ALIAS = "base"
LEGACY_BIRD_SUPPORT_ALIAS = "bird_support"

BIRD_PROMPTS = (
    "the physical taxidermy bird specimen inside the glass display case",
    "the mounted bird specimen inside the enclosure",
    "the real bird exhibit inside the glass case",
    "the physical bird specimen",
)
YELLOW_BASE_BOARD_PROMPTS = (
    "the yellow rectangular display board at the bottom inside the glass case",
    "the yellow display floor inside the glass enclosure",
    "the yellow rectangular base board underneath the bird exhibit",
    "the yellow platform forming the bottom of the bird display",
)
WHITE_PLATFORM_PROMPTS = (
    "the small white rectangular platform directly underneath the bird",
    "the white display plinth immediately supporting the bird",
    "the small white platform below the mounted bird",
    "the rectangular white support platform under the bird",
)
CONNECTED_FIXTURE_PROMPTS = (
    "the small fixture directly connecting the bird feet to the display base",
    "the small mounting fixture touching the bird and the platform inside the glass case",
)


class InternalObjectMaskError(RuntimeError):
    """Raised when internal-object mask provenance or payload is invalid."""


DEFAULT_CANDIDATE_GUARDS = {
    "bird_max_glass_ratio": 0.45,
    "internal_base_min_glass_ratio": 0.03,
    "internal_base_max_glass_ratio": 0.65,
    "internal_object_union_max_glass_ratio": 0.75,
    "candidate_max_glass_iou": 0.75,
    "max_raw_outside_glass_ratio": 0.20,
    "max_glass_iou": 0.75,
    "base_max_bird_overlap_ratio": 0.35,
    "max_boundary_touch_count": 2,
    "min_inside_glass_ratio": 0.70,
    "internal_base_max_components": 5,
    "internal_object_union_max_components": 6,
    "fixture_max_aspect_ratio": 3.0,
    "fixture_min_distance_to_bird_px": 8.0,
    "fixture_min_distance_to_base_px": 6.0,
}


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    clean = dict(payload)
    clean.pop("manifest_payload_sha256", None)
    clean.pop("canonical_payload_sha256", None)
    clean.pop("payload_sha256", None)
    encoded = json.dumps(
        clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_image(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(values).save(temporary, format="PNG")
        with Image.open(temporary) as reread:
            reread.load()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_component_summary(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return {
            "connected_components": 0,
            "largest_component_area": 0,
            "largest_component_ratio": 0.0,
            "centroid_xy": None,
            "bbox_xyxy": None,
            "touches_image_boundary": False,
        }
    labels_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), 8
    )
    areas = stats[1:, cv2.CC_STAT_AREA] if labels_count > 1 else np.array([], dtype=np.int32)
    largest_index = int(np.argmax(areas)) + 1 if areas.size else 0
    bbox = _bbox(binary.astype(np.uint8))
    h, w = binary.shape
    touches = bool(
        binary[0, :].any() or binary[-1, :].any()
        or binary[:, 0].any() or binary[:, -1].any()
    )
    return {
        "connected_components": int(max(labels_count - 1, 0)),
        "largest_component_area": int(areas.max()) if areas.size else 0,
        "largest_component_ratio": float(areas.max() / max(binary.sum(), 1)) if areas.size else 0.0,
        "centroid_xy": (
            [float(centroids[largest_index][0]), float(centroids[largest_index][1])]
            if largest_index else None
        ),
        "bbox_xyxy": bbox,
        "touches_image_boundary": touches,
        "image_size": [int(w), int(h)],
    }


def candidate_metrics(
    mask: np.ndarray,
    glass_mask: np.ndarray,
    *,
    bird_mask: np.ndarray | None = None,
    yellow_base_board_mask: np.ndarray | None = None,
    white_platform_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    raw = np.asarray(mask, dtype=bool)
    glass = np.asarray(glass_mask, dtype=bool)
    if raw.shape != glass.shape:
        raise InternalObjectMaskError("candidate/glass shape mismatch")
    clipped = raw & glass
    union = raw | glass
    glass_area = int(glass.sum())
    raw_area = int(raw.sum())
    clipped_area = int(clipped.sum())
    outside = raw & ~glass
    component = mask_component_summary(raw)
    bbox = component["bbox_xyxy"]
    touch_glass_edges = {"top": False, "bottom": False, "left": False, "right": False}
    if clipped.any() and glass.any():
        ys, xs = np.where(clipped)
        gys, gxs = np.where(glass)
        touch_glass_edges = {
            "top": bool(ys.min() <= gys.min() + 2),
            "bottom": bool(ys.max() >= gys.max() - 2),
            "left": bool(xs.min() <= gxs.min() + 2),
            "right": bool(xs.max() >= gxs.max() - 2),
        }
    def _overlap_and_distance(other_mask: np.ndarray | None, name: str) -> tuple[int, float, float | None]:
        if other_mask is None:
            return 0, 0.0, None
        other = np.asarray(other_mask, dtype=bool)
        if other.shape != raw.shape:
            raise InternalObjectMaskError(f"candidate/{name} shape mismatch")
        overlap = int((raw & other).sum())
        if not raw.any() or not other.any():
            return overlap, float(overlap / max(raw_area, 1)), None
        raw_yx = np.column_stack(np.where(raw))
        other_yx = np.column_stack(np.where(other))
        step = max(1, int(max(len(raw_yx), len(other_yx)) / 2048))
        raw_sample = raw_yx[::step]
        other_sample = other_yx[::step]
        delta = raw_sample[:, None, :] - other_sample[None, :, :]
        dist = float(np.sqrt((delta * delta).sum(axis=-1)).min())
        return overlap, float(overlap / max(raw_area, 1)), dist

    bird_overlap, bird_overlap_ratio, bird_distance = _overlap_and_distance(bird_mask, "bird")
    yellow_overlap, yellow_overlap_ratio, yellow_distance = _overlap_and_distance(
        yellow_base_board_mask, "yellow_base_board"
    )
    white_overlap, white_overlap_ratio, white_distance = _overlap_and_distance(
        white_platform_mask, "white_platform"
    )
    aspect_ratio = None
    vertical_aspect_ratio = None
    lower_glass_fraction = None
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        vertical_aspect_ratio = float(height / width)
        if glass.any():
            gys = np.where(glass)[0]
            lower_start = gys.min() + 0.55 * max(1, (gys.max() - gys.min() + 1))
            lower_glass_fraction = float((np.where(clipped)[0] >= lower_start).sum() / max(clipped_area, 1))
    return {
        **component,
        "raw_area": raw_area,
        "clipped_area": clipped_area,
        "mask_image_ratio": float(raw_area / max(raw.size, 1)),
        "mask_glass_ratio": float(clipped_area / max(glass_area, 1)),
        "inside_glass_ratio": float(clipped_area / max(raw_area, 1)),
        "glass_iou": float((raw & glass).sum() / max(union.sum(), 1)),
        "raw_outside_glass_pixel_count": int(outside.sum()),
        "raw_outside_glass_ratio": float(outside.sum() / max(raw_area, 1)),
        "touches_glass_boundary": touch_glass_edges,
        "touches_glass_boundary_count": int(sum(touch_glass_edges.values())),
        "bird_overlap_pixels": bird_overlap,
        "bird_overlap_ratio": bird_overlap_ratio,
        "bird_distance_px": bird_distance,
        "yellow_base_board_overlap_pixels": yellow_overlap,
        "yellow_base_board_overlap_ratio": yellow_overlap_ratio,
        "yellow_base_board_distance_px": yellow_distance,
        "white_platform_overlap_pixels": white_overlap,
        "white_platform_overlap_ratio": white_overlap_ratio,
        "white_platform_distance_px": white_distance,
        "aspect_ratio": aspect_ratio,
        "vertical_aspect_ratio": vertical_aspect_ratio,
        "lower_glass_fraction": lower_glass_fraction,
        "forms_bird_to_base_connection": bool(
            bird_distance is not None and min(
                yellow_distance if yellow_distance is not None else 1e9,
                white_distance if white_distance is not None else 1e9,
            ) <= 6.0 and bird_distance <= 8.0
        ),
    }


def score_candidate(
    cls: str,
    confidence: float,
    metrics: dict[str, Any],
    guards: dict[str, float] | None = None,
) -> dict[str, Any]:
    guards = {**DEFAULT_CANDIDATE_GUARDS, **(guards or {})}
    positive = {
        "grounding_confidence": float(confidence),
        "inside_glass_ratio": float(metrics["inside_glass_ratio"]),
        "single_component": 1.0 if metrics["connected_components"] <= 1 else 0.0,
        "largest_component_ratio": float(metrics["largest_component_ratio"]),
    }
    penalties = {
        "outside_glass_ratio": float(metrics["raw_outside_glass_ratio"]),
        "glass_iou": max(0.0, float(metrics["glass_iou"]) - 0.50),
        "boundary_touch": 0.15 * float(metrics["touches_glass_boundary_count"]),
        "multi_component": 0.20 * max(0, int(metrics["connected_components"]) - 1),
    }
    if cls in ("yellow_base_board", "white_platform", "connected_fixture", "internal_base"):
        positive["base_area"] = max(0.0, 1.0 - float(metrics["mask_glass_ratio"]) / 0.65)
        penalties["bird_overlap"] = float(metrics.get("bird_overlap_ratio", 0.0))
        if cls == "yellow_base_board":
            positive["lower_glass_position"] = float(metrics.get("lower_glass_fraction") or 0.0)
        if cls == "white_platform":
            distance = metrics.get("bird_distance_px")
            positive["near_bird"] = 1.0 if distance is not None and distance <= 12.0 else 0.0
        if cls == "connected_fixture":
            positive["connection"] = 1.0 if metrics.get("forms_bird_to_base_connection") else 0.0
    else:
        positive["reasonable_bird_area"] = max(
            0.0, 1.0 - abs(float(metrics["mask_glass_ratio"]) - 0.20) / 0.30
        )
    score = sum(positive.values()) - sum(penalties.values())
    reject_reasons = []
    if metrics["raw_area"] <= 0:
        reject_reasons.append("empty_mask")
    if metrics["inside_glass_ratio"] < guards["min_inside_glass_ratio"]:
        reject_reasons.append("low_inside_glass_ratio")
    if metrics["raw_outside_glass_ratio"] > guards["max_raw_outside_glass_ratio"]:
        reject_reasons.append("raw_outside_glass_ratio_too_high")
    if metrics["glass_iou"] > guards["max_glass_iou"]:
        reject_reasons.append("candidate_too_similar_to_glass_hard")
    limit_key = "bird_max_glass_ratio" if cls == "bird" else "internal_base_max_glass_ratio"
    if metrics["mask_glass_ratio"] > guards[limit_key]:
        reject_reasons.append("mask_glass_ratio_too_large")
    if metrics["touches_glass_boundary_count"] > guards["max_boundary_touch_count"]:
        reject_reasons.append("touches_too_many_glass_boundaries")
    if cls == "bird" and metrics["connected_components"] > 1 and metrics["largest_component_ratio"] < 0.75:
        reject_reasons.append("multiple_distant_bird_components")
    if cls in ("yellow_base_board", "white_platform", "connected_fixture", "internal_base"):
        if metrics.get("bird_overlap_ratio", 0.0) > guards["base_max_bird_overlap_ratio"]:
            reject_reasons.append("base_overlaps_too_much_bird")
    if cls == "yellow_base_board" and (metrics.get("lower_glass_fraction") or 0.0) < 0.50:
        reject_reasons.append("yellow_base_not_in_lower_glass")
    if cls == "connected_fixture":
        bird_distance = metrics.get("bird_distance_px")
        base_distance = min(
            metrics.get("yellow_base_board_distance_px") if metrics.get("yellow_base_board_distance_px") is not None else 1e9,
            metrics.get("white_platform_distance_px") if metrics.get("white_platform_distance_px") is not None else 1e9,
        )
        if not metrics.get("forms_bird_to_base_connection"):
            reject_reasons.append("fixture_not_connected_bird_to_base")
        if max(metrics.get("vertical_aspect_ratio") or 0.0, metrics.get("aspect_ratio") or 0.0) > guards["fixture_max_aspect_ratio"] \
                and (bird_distance is None or bird_distance > guards["fixture_min_distance_to_bird_px"]) \
                and (base_distance > guards["fixture_min_distance_to_base_px"] or not metrics.get("forms_bird_to_base_connection")):
            reject_reasons.append("isolated_vertical_pole_or_rail")
    return {
        "candidate_score": float(score),
        "score_terms": {"positive": positive, "penalties": penalties},
        "reject_reasons": reject_reasons,
        "quality_status": "candidate_rejected" if reject_reasons else "candidate_viable",
        "guards": guards,
    }


def inspect_l_mask(path: Path | str, expected_size: tuple[int, int] | None = None) -> dict[str, Any]:
    path = Path(path)
    try:
        with Image.open(path) as image:
            mode = image.mode
            size = image.size
            values = np.asarray(image)
    except Exception as exc:
        raise InternalObjectMaskError(f"cannot parse internal-object mask: {path}") from exc
    if mode != "L" or values.dtype != np.uint8 or values.ndim != 2:
        raise InternalObjectMaskError(f"internal-object mask must be mode L uint8: {path}")
    if expected_size is not None and tuple(size) != tuple(expected_size):
        raise InternalObjectMaskError(
            f"internal-object mask size mismatch: {path}: {size} != {expected_size}"
        )
    binary = values >= 128
    area = int(binary.sum())
    pixel_count = int(binary.size)
    components = 0
    if area:
        components = int(cv2.connectedComponents(binary.astype(np.uint8), 8)[0] - 1)
    boundary = 0
    if area:
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(binary.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary = int((binary & ~eroded).sum())
    return {
        "size": [int(size[0]), int(size[1])],
        "mode": mode,
        "dtype": str(values.dtype),
        "min": int(values.min()),
        "max": int(values.max()),
        "area": area,
        "area_ratio": float(area / max(pixel_count, 1)),
        "bbox_xyxy": _bbox(values),
        "connected_components": components,
        "boundary_length_px": boundary,
    }


def _load_binary(path: Path | str, expected_size: tuple[int, int] | None = None) -> np.ndarray:
    facts = inspect_l_mask(path, expected_size)
    with Image.open(path) as image:
        values = np.asarray(image, dtype=np.uint8)
    if facts["min"] not in (0, 255) or facts["max"] not in (0, 255):
        raise InternalObjectMaskError(f"mask must be binary 0/255: {path}")
    return values >= 128


def clip_mask_to_glass(raw_mask: np.ndarray, glass_mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(raw_mask, dtype=bool)
    glass = np.asarray(glass_mask, dtype=bool)
    if raw.shape != glass.shape:
        raise InternalObjectMaskError(f"raw/glass mask shape mismatch: {raw.shape} != {glass.shape}")
    clipped = raw & glass
    removed = raw & ~glass
    raw_area = int(raw.sum())
    clipped_area = int(clipped.sum())
    removed_area = int(removed.sum())
    return clipped, {
        "raw_area": raw_area,
        "clipped_area": clipped_area,
        "removed_outside_glass_area": removed_area,
        "removed_fraction": float(removed_area / max(raw_area, 1)),
        "empty_after_clip": clipped_area == 0,
        "full_after_clip": clipped_area == clipped.size,
    }


def proposal_view_metadata(
    stem: str,
    rgb_path: Path,
    raw_bird_path: Path,
    raw_internal_base_path: Path,
    glass_mask_path: Path,
) -> dict[str, Any]:
    with Image.open(rgb_path) as rgb:
        if rgb.mode != "RGB":
            raise InternalObjectMaskError(f"RGB image must be mode RGB: {rgb_path}")
        size = rgb.size
    bird = _load_binary(raw_bird_path, size)
    internal_base = _load_binary(raw_internal_base_path, size)
    glass = _load_binary(glass_mask_path, size)
    raw_union = bird | internal_base
    clipped_bird, bird_clip = clip_mask_to_glass(bird, glass)
    clipped_base, base_clip = clip_mask_to_glass(internal_base, glass)
    clipped_union, union_clip = clip_mask_to_glass(raw_union, glass)
    return {
        "stem": stem,
        "rgb_path": str(rgb_path),
        "rgb_sha256": sha256_file(rgb_path),
        "raw": {
            "bird": inspect_l_mask(raw_bird_path, size),
            "internal_base": inspect_l_mask(raw_internal_base_path, size),
            "internal_object_union": {
                "area": int(raw_union.sum()),
                "area_ratio": float(raw_union.mean()),
                "bbox_xyxy": _bbox(raw_union.astype(np.uint8)),
            },
        },
        "glass": {
            "path": str(glass_mask_path),
            "sha256": sha256_file(glass_mask_path),
            "area": int(glass.sum()),
            "area_ratio": float(glass.mean()),
        },
        "clipping": {
            "bird": bird_clip,
            "internal_base": base_clip,
            "internal_object_union": union_clip,
        },
        "processed": {
            "bird_area": int(clipped_bird.sum()),
            "internal_base_area": int(clipped_base.sum()),
            "internal_object_union_area": int(clipped_union.sum()),
        },
        "review_flags": _review_flags(union_clip, clipped_union, glass),
    }


def _review_flags(union_clip: dict[str, Any], clipped_union: np.ndarray, glass: np.ndarray) -> list[str]:
    flags: list[str] = []
    if union_clip["empty_after_clip"]:
        flags.append("empty_after_glass_clip")
    if union_clip["full_after_clip"]:
        flags.append("full_frame_after_glass_clip")
    if union_clip["removed_fraction"] > 0.25:
        flags.append("large_prediction_outside_glass")
    glass_area = int(glass.sum())
    if glass_area and float(clipped_union.sum() / glass_area) > 0.90:
        flags.append("object_fills_nearly_all_glass")
    return flags


def write_proposal_manifest(
    output_root: Path | str,
    entries: list[dict[str, Any]],
    grounded_sam2: dict[str, Any],
    method: str,
) -> dict[str, Any]:
    output_root = Path(output_root)
    if [entry.get("stem") for entry in entries] != EXPECTED_STEMS:
        raise InternalObjectMaskError("proposal manifest requires ordered stems 000000--000110")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "role": PROPOSAL_ROLE,
        "method": method,
        "count": len(entries),
        "ordered_stems": EXPECTED_STEMS,
        "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
        "legacy_aliases": {
            "base": "legacy_not_auto_promoted_to_internal_base",
            "bird_support": "legacy_not_auto_promoted_to_internal_base",
        },
        "prompts": {
            "bird": list(BIRD_PROMPTS),
            "yellow_base_board": list(YELLOW_BASE_BOARD_PROMPTS),
            "white_platform": list(WHITE_PLATFORM_PROMPTS),
            "connected_fixture": list(CONNECTED_FIXTURE_PROMPTS),
        },
        "grounded_sam2": grounded_sam2,
        "human_status": "proposal_requires_review",
        "entries": entries,
    }
    payload["manifest_payload_sha256"] = canonical_payload_sha256(payload)
    _atomic_json(output_root / "manifest.json", payload)
    validation = validate_proposal_manifest(output_root / "manifest.json")
    _atomic_json(output_root / "validation.json", validation)
    review_path = output_root / "review_queue.csv"
    with review_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("stem", "flags", "removed_fraction", "union_area"))
        writer.writeheader()
        for entry in entries:
            writer.writerow({
                "stem": entry["stem"],
                "flags": ";".join(entry.get("review_flags", [])),
                "removed_fraction": entry.get("clipping", {}).get("union", {}).get("removed_fraction", 0.0),
                "union_area": entry.get("processed", {}).get("internal_object_union_area", 0),
            })
    return payload


def validate_proposal_manifest(manifest_path: Path | str) -> dict[str, Any]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("role") != PROPOSAL_ROLE:
        raise InternalObjectMaskError("not an internal-object proposal manifest")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise InternalObjectMaskError("unsupported internal-object proposal schema")
    if payload.get("ordered_stems") != EXPECTED_STEMS or payload.get("count") != 111:
        raise InternalObjectMaskError("proposal must contain exactly stems 000000--000110")
    if payload.get("manifest_payload_sha256") != canonical_payload_sha256(payload):
        raise InternalObjectMaskError("proposal payload hash mismatch")
    flags = {
        entry["stem"]: entry.get("review_flags", [])
        for entry in payload.get("entries", [])
        if entry.get("review_flags")
    }
    return {
        "validation": "REVIEW_REQUIRED" if flags else "PASS_REVIEW_STILL_REQUIRED",
        "role": PROPOSAL_ROLE,
        "count": 111,
        "manifest_file_sha256": sha256_file(path),
        "manifest_payload_sha256": payload["manifest_payload_sha256"],
        "flagged_view_count": len(flags),
        "flagged_views": flags,
    }


def validate_internal_object_mask_set(source_path, images_directory, manifest_path) -> dict[str, Any]:
    source = Path(source_path).resolve()
    image_root = (source / images_directory).resolve()
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = source / manifest
    manifest = manifest.resolve()
    if not manifest.is_file() or manifest.suffix.lower() != ".json":
        raise FileNotFoundError(f"formal internal-object mask manifest does not exist: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("role") == PROPOSAL_ROLE:
        raise ValueError("proposal internal-object masks are not formal training supervision")
    role = payload.get("artifact_role")
    if role != REVIEWED_ROLE or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("--internal_object_masks accepts only a formal reviewed internal-object release")
    if payload.get("role", REVIEWED_ROLE) != REVIEWED_ROLE:
        raise ValueError("formal internal-object role alias mismatch")
    if payload.get("human_status") != "accepted" or payload.get("accepted_count") != 111:
        raise ValueError("formal internal-object manifest must record accepted 111/111 masks")
    if payload.get("count") not in (None, 111):
        raise ValueError("formal internal-object count must be 111")
    if payload.get("internal_object_semantics_version") != INTERNAL_OBJECT_SEMANTICS_VERSION:
        raise ValueError(
            "formal internal-object manifest semantic version mismatch: "
            f"expected {INTERNAL_OBJECT_SEMANTICS_VERSION}"
        )
    if payload.get("mask_interpolation") != MASK_INTERPOLATION:
        raise ValueError(f"formal internal-object manifest must declare {MASK_INTERPOLATION}")
    if payload.get("ordered_stems") != EXPECTED_STEMS or payload.get("accepted_stems") != EXPECTED_STEMS:
        raise ValueError("formal internal-object manifest stems must be exactly 000000--000110")
    canonical = canonical_payload_sha256(payload)
    if payload.get("canonical_payload_sha256") != canonical:
        raise ValueError("formal internal-object canonical payload hash mismatch")
    if payload.get("manifest_payload_sha256") not in (None, canonical):
        raise ValueError("formal internal-object manifest payload hash mismatch")
    if not payload.get("source_proposal_payload_hash"):
        raise ValueError("formal internal-object manifest must record source proposal payload hash")
    if payload.get("legacy_aliases", {}).get("bird_support") == "internal_base":
        raise ValueError("legacy bird_support may not be auto-promoted to internal_base")
    images = {path.stem: path for path in sorted(image_root.iterdir()) if path.is_file()}
    if sorted(images) != EXPECTED_STEMS:
        raise ValueError("scene RGB stems must be exactly 000000--000110")
    entries = payload.get("entries")
    if not isinstance(entries, list) or [entry.get("stem") for entry in entries] != EXPECTED_STEMS:
        raise ValueError("formal internal-object entries are missing, duplicated, extra, or out of order")
    root = manifest.parent
    for role_name in (*MASK_ROLES, "glass_hard"):
        role_root = root / role_name
        if not role_root.is_dir():
            raise ValueError(f"formal internal-object missing role directory: {role_name}")
        if sorted(path.stem for path in role_root.glob("*.png")) != EXPECTED_STEMS:
            raise ValueError(f"formal internal-object {role_name} directory has missing or extra stems")
    ignore_root = root / "internal_ignore"
    if ignore_root.exists():
        referenced_ignore = sorted(
            entry["stem"] for entry in entries if entry.get("internal_ignore_mask_path") is not None
        )
        if sorted(path.stem for path in ignore_root.glob("*.png")) != referenced_ignore:
            raise ValueError("formal internal-object internal_ignore directory has missing or extra stems")
    hash_maps = {
        "rgb": payload.get("rgb_hashes"),
        "glass": payload.get("glass_hashes"),
        "glass_hard": payload.get("glass_hard_hashes"),
        "bird": payload.get("bird_hashes"),
        "internal_base": payload.get("internal_base_hashes"),
        "internal_object_union": payload.get("internal_object_union_hashes"),
    }
    for name, values in hash_maps.items():
        if not isinstance(values, dict) or sorted(values) != EXPECTED_STEMS:
            raise ValueError(f"formal internal-object {name} hash map must cover exactly 111 stems")
    runtime_entries: dict[str, Any] = {}
    aggregate = hashlib.sha256()
    for entry in entries:
        stem = entry["stem"]
        rgb = images[stem]
        if entry.get("rgb_path") != f"{images_directory}/{rgb.name}":
            raise ValueError(f"internal-object {stem} RGB path mismatch")
        rgb_sha = sha256_file(rgb)
        if entry.get("rgb_sha256") != rgb_sha or hash_maps["rgb"][stem] != rgb_sha:
            raise ValueError(f"internal-object {stem} RGB hash mismatch")
        with Image.open(rgb) as image:
            if image.mode != "RGB":
                raise ValueError(f"internal-object {stem} RGB mode mismatch")
            size = image.size
        glass_rel = entry.get("glass_mask_path")
        if glass_rel != f"specular_masks_reviewed_v1/{stem}.png":
            raise ValueError(f"internal-object {stem} glass path mismatch")
        glass_path = source / glass_rel
        glass_sha = sha256_file(glass_path)
        if entry.get("glass_mask_sha256") != glass_sha or hash_maps["glass"][stem] != glass_sha:
            raise ValueError(f"internal-object {stem} glass hash mismatch")
        glass_hard_rel = entry.get("glass_hard_mask_path")
        if glass_hard_rel != f"glass_hard/{stem}.png":
            raise ValueError(f"internal-object {stem} glass_hard path mismatch")
        glass_hard_path = root / glass_hard_rel
        glass_hard_sha = sha256_file(glass_hard_path)
        if entry.get("glass_hard_mask_sha256") != glass_hard_sha or hash_maps["glass_hard"][stem] != glass_hard_sha:
            raise ValueError(f"internal-object {stem} glass_hard hash mismatch")
        glass_binary = _load_binary(glass_hard_path, size)
        role_entries: dict[str, Any] = {}
        native_binary: dict[str, np.ndarray] = {}
        for role in MASK_ROLES:
            rel = entry.get(f"{role}_mask_path")
            if rel != f"{role}/{stem}.png":
                raise ValueError(f"internal-object {stem} {role} path mismatch")
            path = root / rel
            mask_sha = sha256_file(path)
            expected_sha = entry.get(f"{role}_mask_sha256")
            if expected_sha != mask_sha or hash_maps[role][stem] != mask_sha:
                raise ValueError(f"internal-object {stem} {role} hash mismatch")
            native_binary[role] = _load_binary(path, size)
            facts = inspect_l_mask(path, size)
            role_entries[role] = {"path": str(path), "sha256": mask_sha, "size": facts["size"]}
            aggregate.update(f"{stem} {role} {mask_sha}\n".encode("utf-8"))
        native_union = native_binary["bird"] | native_binary["internal_base"]
        if not np.array_equal(native_binary["internal_object_union"], native_union):
            raise ValueError(f"formal internal-object union is not bird|internal_base for camera {stem}")
        if (native_binary["bird"] & ~native_binary["internal_object_union"]).any():
            raise ValueError(f"formal internal-object bird is not subset of union for camera {stem}")
        if (native_binary["internal_base"] & ~native_binary["internal_object_union"]).any():
            raise ValueError(f"formal internal-object internal_base is not subset of union for camera {stem}")
        if (native_binary["internal_object_union"] & ~glass_binary).any():
            raise ValueError(f"formal internal-object union is not subset of glass_hard for camera {stem}")
        ignore_rel = entry.get("internal_ignore_mask_path")
        if ignore_rel is not None:
            if ignore_rel != f"internal_ignore/{stem}.png":
                raise ValueError(f"internal-object {stem} internal_ignore path mismatch")
            ignore_path = root / ignore_rel
            ignore_sha = sha256_file(ignore_path)
            if entry.get("internal_ignore_mask_sha256") != ignore_sha:
                raise ValueError(f"internal-object {stem} internal_ignore hash mismatch")
            facts = inspect_l_mask(ignore_path, size)
            role_entries["internal_ignore"] = {
                "path": str(ignore_path), "sha256": ignore_sha, "size": facts["size"],
            }
        runtime_entries[stem] = {"stem": stem, "rgb_sha256": entry["rgb_sha256"], **role_entries}
    if payload.get("aggregate_mask_sha256") != aggregate.hexdigest():
        raise ValueError("formal internal-object aggregate hash mismatch")
    return {
        "role": REVIEWED_ROLE,
        "artifact_role": REVIEWED_ROLE,
        "human_status": "accepted",
        "count": 111,
        "accepted_count": 111,
        "accepted_with_warning": payload.get("accepted_with_warning", []),
        "manifest_path": str(manifest),
        "manifest_file_sha256": sha256_file(manifest),
        "manifest_payload_sha256": payload.get("manifest_payload_sha256", payload["canonical_payload_sha256"]),
        "canonical_payload_sha256": payload["canonical_payload_sha256"],
        "source_proposal_payload_hash": payload["source_proposal_payload_hash"],
        "aggregate_sha256": aggregate.hexdigest(),
        "mask_interpolation": MASK_INTERPOLATION,
        "internal_object_semantics_version": INTERNAL_OBJECT_SEMANTICS_VERSION,
        "legacy_aliases": {
            "base": "legacy_not_auto_promoted_to_internal_base",
            "bird_support": "legacy_not_auto_promoted_to_internal_base",
        },
        "entries": runtime_entries,
        "provenance": payload.get("provenance", {}),
    }


def load_resized_internal_object_masks(
    validated_manifest: dict[str, Any],
    stem: str,
    original_size: tuple[int, int],
    output_size: tuple[int, int],
) -> dict[str, Any]:
    if validated_manifest is None or validated_manifest.get("role") != REVIEWED_ROLE:
        raise ValueError("internal-object loading requires a validated formal manifest")
    entry = validated_manifest["entries"].get(stem)
    if entry is None:
        raise ValueError(f"formal internal-object manifest has no mask for camera {stem}")
    loaded = {}
    native_binary = {}
    for role in MASK_ROLES:
        role_entry = entry[role]
        path = Path(role_entry["path"])
        if sha256_file(path) != role_entry["sha256"]:
            raise ValueError(f"internal-object mask changed after validation: {path}")
        inspect_l_mask(path, original_size)
        with Image.open(path) as image:
            raw_values = np.asarray(image, dtype=np.uint8)
        native_binary[role] = raw_values >= 128
        values = raw_values.astype(np.float32) / 255.0
        resized = cv2.resize(values, tuple(output_size), interpolation=cv2.INTER_NEAREST)
        resized = np.clip(resized, 0.0, 1.0).astype(np.float32, copy=False)
        if not np.isfinite(resized).all():
            raise ValueError(f"resized internal-object mask is non-finite: {path}")
        loaded[role] = torch.from_numpy(resized[None].copy())
        loaded[f"{role}_sha256"] = role_entry["sha256"]
    native_union = native_binary["bird"] | native_binary["internal_base"]
    if not np.array_equal(native_binary["internal_object_union"], native_union):
        raise ValueError(f"formal internal-object union is not bird|internal_base for camera {stem}")
    if "internal_ignore" in entry:
        role_entry = entry["internal_ignore"]
        path = Path(role_entry["path"])
        if sha256_file(path) != role_entry["sha256"]:
            raise ValueError(f"internal-object mask changed after validation: {path}")
        with Image.open(path) as image:
            raw_values = np.asarray(image, dtype=np.uint8)
        values = raw_values.astype(np.float32) / 255.0
        resized = cv2.resize(values, tuple(output_size), interpolation=cv2.INTER_NEAREST)
        loaded["internal_ignore"] = torch.from_numpy(np.clip(resized, 0.0, 1.0)[None].copy())
        loaded["internal_ignore_sha256"] = role_entry["sha256"]
    return loaded


def _as_hwc1(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 3 and value.shape[0] == 1:
        value = value.permute(1, 2, 0)
    elif value.ndim == 2:
        value = value[..., None]
    if value.ndim != 3 or value.shape[-1] != 1:
        raise ValueError(f"{name} must be HWC1, CHW1, or HW")
    return value.float()


def _morph(mask: torch.Tensor, pixels: int, op: str) -> torch.Tensor:
    if pixels <= 0:
        return mask
    chw = mask.permute(2, 0, 1).unsqueeze(0).float()
    kernel = int(2 * pixels + 1)
    if op == "dilate":
        out = F.max_pool2d(chw, kernel_size=kernel, stride=1, padding=pixels)
    elif op == "erode":
        out = 1.0 - F.max_pool2d(1.0 - chw, kernel_size=kernel, stride=1, padding=pixels)
    else:
        raise ValueError(f"unknown morphology op: {op}")
    return out.squeeze(0).permute(1, 2, 0)


def object_occupancy_domains(
    object_union: torch.Tensor,
    glass_hard: torch.Tensor,
    valid_two_hit: torch.Tensor,
    internal_ignore: torch.Tensor | None = None,
    erode_px: int = 3,
    dilate_px: int = 3,
    threshold: float = HARD_THRESHOLD,
) -> dict[str, torch.Tensor]:
    obj = (_as_hwc1(object_union, "object_union") >= threshold).float()
    glass = (_as_hwc1(glass_hard, "glass_hard") >= threshold).float()
    valid = (_as_hwc1(valid_two_hit, "valid_two_hit") >= threshold).float()
    if obj.shape != glass.shape or obj.shape != valid.shape:
        raise ValueError("object_union, glass_hard, and valid_two_hit shapes must match")
    explicit = torch.zeros_like(obj)
    if internal_ignore is not None:
        explicit = (_as_hwc1(internal_ignore, "internal_ignore") >= threshold).float()
        if explicit.shape != obj.shape:
            raise ValueError("internal_ignore shape must match object_union")
    eroded = _morph(obj, int(erode_px), "erode")
    dilated = _morph(obj, int(dilate_px), "dilate")
    base = glass * valid
    mpos = (eroded * base).clamp(0, 1)
    boundary = ((dilated - eroded).clamp(0, 1) * base).clamp(0, 1)
    mignore = ((boundary + explicit).clamp(0, 1) * base * (1.0 - mpos)).clamp(0, 1)
    mneg = ((1.0 - dilated) * base * (1.0 - explicit) * (1.0 - mpos) * (1.0 - mignore)).clamp(0, 1)
    overlap = (mpos * mignore + mpos * mneg + mignore * mneg).sum()
    if float(overlap) != 0.0:
        raise ValueError("internal-object occupancy domains must be mutually exclusive")
    return {
        "Mpos": mpos, "Mneg": mneg, "Mignore": mignore, "domain": base,
        "Mboundary": boundary, "Mexplicit_ignore": (explicit * base).clamp(0, 1),
    }


def object_occupancy_loss(
    inside_alpha: torch.Tensor,
    domains: dict[str, torch.Tensor],
    *,
    alpha_floor: float = 0.35,
    lambda_positive: float = 0.0,
    lambda_negative: float = 0.0,
) -> dict[str, torch.Tensor]:
    ain = _as_hwc1(inside_alpha, "inside_alpha")
    mpos = _as_hwc1(domains["Mpos"], "Mpos").to(device=ain.device, dtype=ain.dtype)
    mneg = _as_hwc1(domains["Mneg"], "Mneg").to(device=ain.device, dtype=ain.dtype)
    pos_denom = mpos.sum().clamp_min(1.0)
    neg_denom = mneg.sum().clamp_min(1.0)
    positive = (mpos * torch.relu(ain.new_tensor(float(alpha_floor)) - ain)).sum() / pos_denom
    negative = (mneg * ain).sum() / neg_denom
    total = float(lambda_positive) * positive + float(lambda_negative) * negative
    if not torch.isfinite(total):
        raise FloatingPointError("internal-object occupancy loss is NaN/Inf")
    return {"total": total, "positive": positive, "negative": negative}


def object_domain_metrics(
    package: dict[str, torch.Tensor],
    domains: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor] | None = None,
    *,
    alpha_floor: float = 0.35,
    erode_px: int = 3,
    dilate_px: int = 3,
) -> dict[str, float]:
    ain = _as_hwc1(package["inside_alpha"], "inside_alpha").detach().float()
    cin = package["inside_color"].detach().float()
    cout = package["outside_color"].detach().float()
    mpos = _as_hwc1(domains["Mpos"], "Mpos").to(ain.device) > 0.5
    mneg = _as_hwc1(domains["Mneg"], "Mneg").to(ain.device) > 0.5

    def masked_values(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(device=value.device)
        if value.shape[-1] != mask.shape[-1]:
            mask = mask.expand_as(value)
        return value[mask]

    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
        selected = masked_values(value, mask)
        return float(selected.mean()) if selected.numel() else 0.0

    def alpha_stats(name: str, mask: torch.Tensor) -> dict[str, float | int]:
        selected = masked_values(ain, mask)
        if not selected.numel():
            return {
                f"{name}_pixel_count": int(mask.sum()),
                f"{name}_ain_mean": 0.0,
                f"{name}_ain_median": 0.0,
                f"{name}_ain_p05": 0.0,
                f"{name}_ain_p50": 0.0,
                f"{name}_ain_p95": 0.0,
                f"{name}_ain_max": 0.0,
                f"{name}_ain_nonzero_ratio": 0.0,
                f"{name}_ain_above_alpha_floor_ratio": 0.0,
            }
        quantiles = torch.quantile(selected, selected.new_tensor([0.05, 0.50, 0.95]))
        return {
            f"{name}_pixel_count": int(mask.sum()),
            f"{name}_ain_mean": float(selected.mean()),
            f"{name}_ain_median": float(selected.median()),
            f"{name}_ain_p05": float(quantiles[0]),
            f"{name}_ain_p50": float(quantiles[1]),
            f"{name}_ain_p95": float(quantiles[2]),
            f"{name}_ain_max": float(selected.max()),
            f"{name}_ain_nonzero_ratio": float((selected > 0).float().mean()),
            f"{name}_ain_above_alpha_floor_ratio": float((selected >= float(alpha_floor)).float().mean()),
        }

    def color_stats(name: str, value: torch.Tensor, mask: torch.Tensor, prefix: str) -> dict[str, float]:
        selected = masked_values(value, mask)
        if not selected.numel():
            return {
                f"{name}_{prefix}_rgb_mean": 0.0,
                f"{name}_{prefix}_energy": 0.0,
                f"{name}_{prefix}_nonzero_ratio": 0.0,
            }
        return {
            f"{name}_{prefix}_rgb_mean": float(selected.mean()),
            f"{name}_{prefix}_energy": float(selected.abs().mean()),
            f"{name}_{prefix}_nonzero_ratio": float((selected.abs() > 0).float().mean()),
        }

    metrics = {
        "ain_mean_mpos": masked_mean(ain, mpos),
        "ain_mean_mneg": masked_mean(ain, mneg),
        "ain_positive_coverage_ge_0_35": masked_mean((ain >= 0.35).float(), mpos),
        "ain_negative_spill_ge_0_10": masked_mean((ain >= 0.10).float(), mneg),
        "cin_energy_mpos": masked_mean(cin.abs(), mpos),
        "cin_energy_mneg": masked_mean(cin.abs(), mneg),
        "cout_energy_mneg": masked_mean(cout.abs(), mneg),
    }
    if masks:
        glass = _as_hwc1(domains["domain"], "domain").to(ain.device) > 0.5
        ignore = _as_hwc1(domains["Mignore"], "Mignore").to(ain.device) > 0.5
        bird_raw = _as_hwc1(masks["bird"], "bird").to(ain.device)
        base_raw = _as_hwc1(masks["internal_base"], "internal_base").to(ain.device)
        union_raw = _as_hwc1(masks["internal_object_union"], "internal_object_union").to(ain.device)
        bird_pos = ((_morph((bird_raw >= HARD_THRESHOLD).float(), int(erode_px), "erode") > 0.5) & glass & ~ignore)
        base_pos = ((_morph((base_raw >= HARD_THRESHOLD).float(), int(erode_px), "erode") > 0.5) & glass & ~ignore)
        union_pos = ((_morph((union_raw >= HARD_THRESHOLD).float(), int(erode_px), "erode") > 0.5) & glass & ~ignore)
        union_dilated = _morph((union_raw >= HARD_THRESHOLD).float(), int(dilate_px), "dilate") > 0.5
        split_mneg = glass & ~union_dilated & ~ignore
        split_masks = {
            "bird": bird_pos,
            "internal_base": base_pos,
            "union": union_pos,
            "mneg": split_mneg,
        }
        metrics["bird_internal_base_overlap_count"] = int((bird_pos & base_pos).sum())
        metrics["split_metrics_source"] = "float_training_tensors"
        for name, mask in split_masks.items():
            metrics.update(alpha_stats(name, mask))
            metrics.update(color_stats(name, cin, mask, "cin"))
        metrics["mneg_cout_energy"] = color_stats("mneg", cout, split_mneg, "cout")["mneg_cout_energy"]
        metrics["mneg_cout_rgb_mean"] = color_stats("mneg", cout, split_mneg, "cout")["mneg_cout_rgb_mean"]
        metrics["mneg_cout_nonzero_ratio"] = color_stats("mneg", cout, split_mneg, "cout")["mneg_cout_nonzero_ratio"]
        outside_union = glass & ~union_pos
        metrics["outside_union_cout_energy"] = masked_mean(cout.abs(), outside_union)
        metrics["outside_union_cout_nonzero_ratio"] = masked_mean((cout.abs() > 0).float(), outside_union)
    return metrics
