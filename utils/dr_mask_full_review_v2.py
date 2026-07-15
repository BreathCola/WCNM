"""Full 111-view TiHuBird glass-mask v2 proposal packaging.

The generated masks are review proposals only. They are deliberately separated
from the formal reviewed-mask loader and never create a reviewed_v2 release.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from utils.dr_mask_high_risk_review import proposal_tree_digest
from utils.dr_mask_proposal import (
    ProposalAuditError,
    _atomic_image,
    _atomic_json,
    audit_dr_artifacts,
    boundary_maps,
    sha256_file,
)
from utils.dr_mask_review import _atomic_text, _clamped_box
from utils.specular_mask import validate_specular_mask_set


FULL_V2_SCHEMA_VERSION = 1
FULL_V2_METHOD_VERSION = "tihubird-glass-mask-full-review-v2-proposal"
DEFAULT_REPAIR_STEMS = ("000039", "000040", "000041")
DEFAULT_SOURCE_SIZE = (3827, 2152)
MASK_SEMANTICS = (
    "Mask is the full projected glass enclosure. Bird, yellow base board, and "
    "white platform pixels visible through the glass remain inside the glass "
    "mask. Exclude table, ceiling, dinosaur, external background, and large "
    "lights outside the glass. Internal objects are never treated as holes."
)
REVIEW_PAGE_SIZE = DEFAULT_SOURCE_SIZE
REVIEW_CROP_SIZE = (600, 337)


def tree_digest(root: Path) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise ProposalAuditError(f"missing tree for hashing: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return {"path": str(root), "file_count": len(files), "sha256": digest.hexdigest()}


def _expected_stems(count: int) -> list[str]:
    return [f"{index:06d}" for index in range(count)]


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProposalAuditError(f"cannot parse {label}: {path}: {error}") from error


def _read_image(path: Path, mode: str, size: tuple[int, int]) -> Image.Image:
    try:
        with Image.open(path) as opened:
            if opened.mode != mode or opened.size != size:
                raise ProposalAuditError(
                    f"image mismatch: {path}: mode={opened.mode}, size={opened.size}, "
                    f"expected mode={mode}, size={size}"
                )
            return opened.copy()
    except OSError as error:
        raise ProposalAuditError(f"cannot parse image: {path}: {error}") from error


def _bbox_from_bool(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _boundary_length(mask: np.ndarray) -> int:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return int(round(sum(cv2.arcLength(contour, True) for contour in contours)))


def _component_stats(mask: np.ndarray) -> dict[str, Any]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return {"component_count": 0, "largest_component_ratio": 0.0}
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
    return {
        "component_count": int(count - 1),
        "largest_component_ratio": float(areas.max() / max(areas.sum(), 1.0)),
    }


def _validate_mask_values(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = _read_image(path, "L", size)
    values = np.asarray(image, dtype=np.uint8)
    if values.ndim != 2 or values.dtype != np.uint8:
        raise ProposalAuditError(f"mask is not L uint8: {path}")
    if not np.isfinite(values.astype(np.float32)).all():
        raise ProposalAuditError(f"mask contains non-finite values: {path}")
    if int(values.min()) == int(values.max()):
        raise ProposalAuditError(f"mask is empty/full/constant: {path}")
    return values


def audit_v1_release_and_sources(
    scene: Path,
    reviewed_v1_manifest: Path,
    proposal_v1_root: Path,
    high_risk_root: Path,
    repair_v1_root: Path,
    *,
    expected_count: int = 111,
    source_size: tuple[int, int] = DEFAULT_SOURCE_SIZE,
    repair_stems: Sequence[str] = DEFAULT_REPAIR_STEMS,
    require_formal_loader: bool = True,
) -> dict[str, Any]:
    """Audit reviewed_v1 provenance and record the historical repair boundary."""
    scene = Path(scene).expanduser().resolve()
    reviewed_v1_manifest = Path(reviewed_v1_manifest).expanduser().resolve()
    reviewed_v1_root = reviewed_v1_manifest.parent
    proposal_v1_root = Path(proposal_v1_root).expanduser().resolve()
    high_risk_root = Path(high_risk_root).expanduser().resolve()
    repair_v1_root = Path(repair_v1_root).expanduser().resolve()
    expected = _expected_stems(expected_count)
    repair_stems = tuple(repair_stems)

    if require_formal_loader:
        validate_specular_mask_set(scene, "images", reviewed_v1_manifest)

    manifest = _load_json(reviewed_v1_manifest, "reviewed_v1 manifest")
    if manifest.get("count") != expected_count or manifest.get("ordered_stems") != expected:
        raise ProposalAuditError("reviewed_v1 manifest does not contain the exact expected stems")
    if manifest.get("human_status") != "accepted" or manifest.get("version") != "reviewed_v1":
        raise ProposalAuditError("reviewed_v1 manifest is not the accepted reviewed_v1 release")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ProposalAuditError("reviewed_v1 manifest must contain one entry per stem")
    if [entry.get("stem") for entry in entries] != expected:
        raise ProposalAuditError("reviewed_v1 entries are not exactly ordered")

    proposal_manifest = _load_json(
        proposal_v1_root / "review_manifest_111.json", "proposal_v1 review manifest"
    )
    if proposal_manifest.get("status") != "PASS":
        raise ProposalAuditError("proposal_v1 review manifest is not PASS")
    proposal_frames = {frame["stem"]: frame for frame in proposal_manifest.get("frames", [])}
    if sorted(proposal_frames) != expected:
        raise ProposalAuditError("proposal_v1 review manifest does not contain the exact stems")

    high_risk_manifest = _load_json(
        high_risk_root / "review_pack_manifest.json", "high-risk review manifest"
    )
    if high_risk_manifest.get("status") != "PASS":
        raise ProposalAuditError("high-risk review pack is not PASS")

    repair_report = _load_json(repair_v1_root / "repair_report.json", "repair report")
    if repair_report.get("status") != "PASS":
        raise ProposalAuditError("repair_v1 report is not PASS")
    if tuple(repair_report.get("repair_stems", [])) != repair_stems:
        raise ProposalAuditError("repair_v1 stems do not match the documented local repairs")
    for stem in repair_stems:
        operation = repair_report["repairs"][stem]["repair_operation"]
        if "subtractive" not in operation or "only" not in operation:
            raise ProposalAuditError(f"{stem}: old repair is not recorded as subtractive-only")

    source_counts = {"proposal_v1": 0, "repair_candidate_v1": 0}
    records: dict[str, Any] = {}
    for entry in entries:
        stem = entry["stem"]
        source = entry.get("source")
        if source not in source_counts:
            raise ProposalAuditError(f"{stem}: unexpected reviewed_v1 source {source!r}")
        source_counts[source] += 1
        mask_path = reviewed_v1_root / f"{stem}.png"
        values = _validate_mask_values(mask_path, source_size)
        if sha256_file(mask_path) != entry.get("mask_sha256"):
            raise ProposalAuditError(f"{stem}: reviewed_v1 mask hash mismatch")
        if source == "repair_candidate_v1":
            if stem not in repair_stems:
                raise ProposalAuditError(f"{stem}: unexpected local repair source in reviewed_v1")
            source_path = repair_v1_root / "repair_candidates" / f"{stem}.png"
            source_kind = "old_local_repair_subtractive_only"
        else:
            if stem in repair_stems:
                raise ProposalAuditError(f"{stem}: documented repair stem did not use repair source")
            source_path = proposal_v1_root / "proposal_soft" / stem / "proposal_soft.png"
            source_kind = "frozen_proposal_v1_inherited"
        if not source_path.is_file() or sha256_file(source_path) != entry.get("source_sha256"):
            raise ProposalAuditError(f"{stem}: reviewed_v1 source hash mismatch")
        records[stem] = {
            "stem": stem,
            "reviewed_v1_path": str(mask_path),
            "reviewed_v1_sha256": sha256_file(mask_path),
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "source": source,
            "source_kind": source_kind,
            "area_ratio": float((values >= 128).mean()),
            "bbox_xyxy": _bbox_from_bool(values >= 128),
            "proposal_v1_risk_score": float(proposal_frames[stem].get("risk_score", 0.0)),
            "proposal_v1_active_risks": proposal_frames[stem].get("active_risks", []),
        }
    if source_counts != {"proposal_v1": expected_count - len(repair_stems), "repair_candidate_v1": len(repair_stems)}:
        raise ProposalAuditError(f"reviewed_v1 source counts are wrong: {source_counts}")
    return {
        "status": "PASS",
        "reviewed_v1_manifest": str(reviewed_v1_manifest),
        "reviewed_v1_root": str(reviewed_v1_root),
        "source_counts": source_counts,
        "repair_stems": list(repair_stems),
        "old_repair_scope": {
            "only_local_repair_stems": list(repair_stems),
            "other_frames_inherit_frozen_proposal_v1_count": expected_count - len(repair_stems),
            "old_repair_was_subtractive_only": True,
            "old_repair_does_not_mean_all_111_were_refined": True,
        },
        "records": records,
        "proposal_v1_manifest_sha256": sha256_file(proposal_v1_root / "review_manifest_111.json"),
        "high_risk_review_manifest_sha256": sha256_file(
            high_risk_root / "review_pack_manifest.json"
        ),
        "repair_report_sha256": sha256_file(repair_v1_root / "repair_report.json"),
    }


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(np.argmax(areas) + 1)
    return labels == largest


def _fill_internal_holes(mask: np.ndarray) -> tuple[np.ndarray, int]:
    inverse = (~mask.astype(bool)).astype(np.uint8)
    flood = inverse.copy()
    height, width = flood.shape
    cv2.floodFill(flood, np.zeros((height + 2, width + 2), np.uint8), (0, 0), 2)
    holes = flood == 1
    filled = mask.astype(bool) | holes
    return filled, int(holes.sum())


def build_v2_candidate(
    reviewed_v1: np.ndarray,
    rgb: np.ndarray,
    dr_normal_rgb: np.ndarray,
    dr_depth_rgb: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a conservative, bidirectional full-review proposal from v1."""
    if reviewed_v1.dtype != np.uint8 or reviewed_v1.ndim != 2:
        raise ProposalAuditError("reviewed_v1 mask must be uint8 2D")
    height, width = reviewed_v1.shape
    hard = reviewed_v1 >= 128
    if not hard.any() or hard.all():
        raise ProposalAuditError("reviewed_v1 hard mask is empty or full")

    filled, filled_holes = _fill_internal_holes(hard)
    largest = _largest_component(filled)
    removed_components = filled & ~largest
    kernel_size = max(3, int(round(min(width, height) * 0.0025)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(largest.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
    opened = cv2.morphologyEx(closed.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
    if not opened.any() or opened.all():
        raise ProposalAuditError("v2 candidate morphology produced an empty/full mask")

    inside_distance = cv2.distanceTransform(opened.astype(np.uint8), cv2.DIST_L2, 3)
    outside_distance = cv2.distanceTransform((~opened).astype(np.uint8), cv2.DIST_L2, 3)
    signed = inside_distance - outside_distance
    feather = max(6.0, min(width, height) * 0.004)
    soft = np.clip(0.5 + signed / (2.0 * feather), 0.0, 1.0)

    normal_boundary, depth_boundary = boundary_maps(dr_normal_rgb, dr_depth_rgb)
    dr_boundary = np.maximum(normal_boundary, depth_boundary)
    dr_boundary_source = cv2.resize(
        dr_boundary.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )
    boundary_ring = np.abs(signed) <= (2.0 * feather)
    # Strong DR edges sharpen only the candidate's own boundary; they do not
    # import target colors or promote any frame automatically.
    sharpener = np.clip((dr_boundary_source - 0.25) / 0.35, 0.0, 1.0)
    soft[boundary_ring] = np.clip(
        soft[boundary_ring] + 0.10 * sharpener[boundary_ring] * np.sign(signed[boundary_ring]),
        0.0,
        1.0,
    )
    candidate = np.round(255.0 * soft).astype(np.uint8)
    candidate[opened & (inside_distance > feather)] = 255
    candidate[(~opened) & (outside_distance > feather)] = 0
    if candidate.min() != 0 or candidate.max() != 255:
        raise ProposalAuditError("v2 candidate must retain both 0 and 255")

    v1_hard = hard
    v2_hard = candidate >= 128
    added = ~v1_hard & v2_hard
    removed = v1_hard & ~v2_hard
    info = {
        "operation_policy": {
            "subtractive_only": False,
            "allows_added_pixels": True,
            "allows_removed_pixels": True,
            "allows_top_bottom_left_right_base_boundary_updates": True,
            "allows_soft_edge_updates": True,
        },
        "morphology_kernel_size": int(kernel_size),
        "soft_edge_feather_pixels": float(feather),
        "filled_internal_hole_pixels": int(filled_holes),
        "removed_detached_component_pixels": int(removed_components.sum()),
        "dr_boundary_ring_mean": float(dr_boundary_source[boundary_ring].mean()) if boundary_ring.any() else 0.0,
        "added_hard_pixel_count": int(added.sum()),
        "removed_hard_pixel_count": int(removed.sum()),
        "added_hard_ratio": float(added.mean()),
        "removed_hard_ratio": float(removed.mean()),
        "soft_changed_pixel_ratio": float((candidate != reviewed_v1).mean()),
    }
    return candidate, info


def _soft_overlay(rgb: Image.Image, mask: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    base = np.asarray(rgb, dtype=np.float32)
    values = np.asarray(mask, dtype=np.float32) / 255.0
    tint = np.zeros_like(base)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    alpha = values[..., None] * 0.45
    return Image.fromarray(np.round(base * (1.0 - alpha) + tint * alpha).astype(np.uint8))


def _change_overlay(
    rgb: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> Image.Image:
    base = np.asarray(rgb, dtype=np.float32)
    result = base.copy()
    result[mask] = 0.25 * result[mask] + 0.75 * np.asarray(color, dtype=np.float32)
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _boundary_difference(rgb: Image.Image, v1: np.ndarray, v2: np.ndarray) -> Image.Image:
    base = np.asarray(rgb, dtype=np.float32) * 0.35
    v1_hard = (v1 >= 128).astype(np.uint8)
    v2_hard = (v2 >= 128).astype(np.uint8)
    changed = v1_hard != v2_hard
    base[changed] = 0.25 * base[changed] + 0.75 * np.asarray([255, 130, 20], dtype=np.float32)
    result = np.clip(base, 0, 255).astype(np.uint8)
    for values, color in ((v1_hard, (255, 30, 210)), (v2_hard, (30, 235, 255))):
        contours, _ = cv2.findContours(values, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 5, cv2.LINE_AA)
    return Image.fromarray(result)


def _mask_panel(mask: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    values = np.asarray(mask, dtype=np.uint8)
    out = np.zeros((values.shape[0], values.shape[1], 3), dtype=np.uint8)
    out[..., 0] = (values.astype(np.uint16) * color[0] // 255).astype(np.uint8)
    out[..., 1] = (values.astype(np.uint16) * color[1] // 255).astype(np.uint8)
    out[..., 2] = (values.astype(np.uint16) * color[2] // 255).astype(np.uint8)
    return Image.fromarray(out)


def _boundary_crop(rgb: Image.Image, v1: np.ndarray, v2: np.ndarray, box: tuple[int, int, int, int]) -> Image.Image:
    return _boundary_difference(rgb.crop(box), v1[box[1] : box[3], box[0] : box[2]], v2[box[1] : box[3], box[0] : box[2]])


def _crop_boxes(
    bbox: list[int],
    source_size: tuple[int, int],
    rgb: np.ndarray,
    v2_hard: np.ndarray,
) -> dict[str, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    bbox_height = y2 - y1
    masked_rgb = rgb.copy()
    hsv = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    highlight = (hsv[..., 2] / 255.0) * (1.0 - 0.7 * hsv[..., 1] / 255.0)
    valid = cv2.dilate(v2_hard.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    highlight[~valid] = -1.0
    hy, hx = np.unravel_index(int(np.argmax(highlight)), highlight.shape)
    crop_size = (min(REVIEW_CROP_SIZE[0], source_size[0]), min(REVIEW_CROP_SIZE[1], source_size[1]))
    return {
        "top_crop": _clamped_box(center_x, y1, source_size, crop_size),
        "bottom_yellow_base_crop": _clamped_box(center_x, y2, source_size, crop_size),
        "left_crop": _clamped_box(x1, center_y, source_size, crop_size),
        "right_crop": _clamped_box(x2, center_y, source_size, crop_size),
        "lower_base_crop": _clamped_box(center_x, y1 + 0.78 * bbox_height, source_size, crop_size),
        "strongest_reflection_occlusion_crop": _clamped_box(float(hx), float(hy), source_size, crop_size),
    }


def _paste_labeled(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    label: str,
    box: tuple[int, int, int, int],
    *,
    direct: bool = False,
) -> None:
    x0, y0, x1, y1 = box
    label_h = 24
    panel_box = (x0, y0 + label_h, x1, y1)
    draw.text((x0 + 4, y0 + 5), label, fill=(246, 246, 246))
    panel_w = panel_box[2] - panel_box[0]
    panel_h = panel_box[3] - panel_box[1]
    if direct:
        if image.width != panel_w or image.height != panel_h:
            raise ProposalAuditError(f"direct panel size mismatch for {label}: {image.size}")
        panel = image.convert("RGB")
    else:
        panel = image.copy()
        panel.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
        framed = Image.new("RGB", (panel_w, panel_h), (5, 5, 5))
        framed.paste(panel.convert("RGB"), ((panel_w - panel.width) // 2, (panel_h - panel.height) // 2))
        panel = framed
    canvas.paste(panel, (panel_box[0], panel_box[1]))
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(60, 60, 60), width=1)


def _make_review_page(
    stem: str,
    rgb: Image.Image,
    reviewed_v1: Image.Image,
    candidate: Image.Image,
    metrics: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    size = rgb.size
    v1_values = np.asarray(reviewed_v1, dtype=np.uint8)
    v2_values = np.asarray(candidate, dtype=np.uint8)
    v1_hard = v1_values >= 128
    v2_hard = v2_values >= 128
    added = ~v1_hard & v2_hard
    removed = v1_hard & ~v2_hard
    rgb_values = np.asarray(rgb, dtype=np.uint8)
    bbox = metrics["v2_bbox_xyxy"]
    if bbox is None:
        raise ProposalAuditError(f"{stem}: v2 bbox unexpectedly empty")

    panels = [
        (rgb, "Original RGB"),
        (_mask_panel(reviewed_v1, (230, 230, 230)), "reviewed_v1 mask"),
        (_soft_overlay(rgb, reviewed_v1, (30, 225, 210)), "reviewed_v1 overlay"),
        (_mask_panel(candidate, (230, 230, 230)), "v2 candidate mask"),
        (_soft_overlay(rgb, candidate, (255, 210, 40)), "v2 candidate overlay"),
        (_change_overlay(rgb, added, (30, 255, 80)), "added pixels"),
        (_change_overlay(rgb, removed, (255, 55, 35)), "removed pixels"),
        (_boundary_difference(rgb, v1_values, v2_values), "boundary difference"),
    ]

    page = Image.new("RGB", REVIEW_PAGE_SIZE, (16, 16, 16))
    draw = ImageDraw.Draw(page)
    draw.text((18, 12), f"{stem} | full glass mask v2 proposal | review only", fill=(255, 255, 255))
    draw.text((18, 38), MASK_SEMANTICS, fill=(210, 225, 255))
    draw.text(
        (18, 66),
        f"area v1={metrics['v1_area_ratio']:.6f} v2={metrics['v2_area_ratio']:.6f} "
        f"added={metrics['added_hard_ratio']:.6f} removed={metrics['removed_hard_ratio']:.6f} "
        f"soft_changed={metrics['soft_changed_pixel_ratio']:.6f}",
        fill=(235, 235, 235),
    )
    draw.text(
        (18, 92),
        f"bbox v1={metrics['v1_bbox_xyxy']} v2={metrics['v2_bbox_xyxy']} "
        f"components={metrics['component_count']} largest={metrics['largest_component_ratio']:.6f} "
        f"boundary={metrics['boundary_length_px']} continuity_risk={metrics['boundary_continuity_risk']:.6f}",
        fill=(255, 205, 120),
    )
    draw.text(
        (18, 118),
        "Candidate allows add/remove/boundary/soft-edge changes; automatic risk only orders review, never promotion.",
        fill=(195, 235, 195),
    )

    margin = 10
    top = 150
    panel_w = 940
    panel_h = 480
    label_h = 24
    for index, (image, label) in enumerate(panels):
        col = index % 4
        row = index // 4
        x = margin + col * (panel_w + margin)
        y = top + row * (panel_h + label_h + margin)
        _paste_labeled(page, draw, image, label, (x, y, x + panel_w, y + label_h + panel_h))

    crop_boxes = _crop_boxes(bbox, size, rgb_values, v2_hard)
    crop_root = output_root / "review_crops" / stem
    crop_records = {}
    crop_top = top + 2 * (panel_h + label_h + margin) + 18
    crop_panel_size = (min(REVIEW_CROP_SIZE[0], size[0]), min(REVIEW_CROP_SIZE[1], size[1]))
    for index, (name, box) in enumerate(crop_boxes.items()):
        crop = _boundary_crop(rgb, v1_values, v2_values, box)
        if crop.size != crop_panel_size:
            raise ProposalAuditError(f"{stem}/{name}: crop size mismatch {crop.size}")
        crop_path = crop_root / f"{name}.png"
        _atomic_image(crop_path, crop)
        x = margin + index * (crop_panel_size[0] + margin)
        _paste_labeled(
            page,
            draw,
            crop,
            name.replace("_", " "),
            (x, crop_top, x + crop_panel_size[0], crop_top + label_h + crop_panel_size[1]),
            direct=True,
        )
        crop_records[name] = {
            "path": str(crop_path.resolve()),
            "sha256": sha256_file(crop_path),
            "box_xyxy_source": list(box),
            "source_pixel_size": list(crop_panel_size),
            "resampling": None,
        }

    page_path = output_root / "review_pages" / f"{stem}_review.png"
    _atomic_image(page_path, page)
    return {
        "path": str(page_path.resolve()),
        "sha256": sha256_file(page_path),
        "size": list(page.size),
        "crops": crop_records,
    }


def _make_contact_sheets(
    metrics: Sequence[dict[str, Any]],
    output_root: Path,
    *,
    risk_ranked: bool,
    per_page: int = 12,
) -> list[str]:
    ordered = sorted(metrics, key=lambda row: row["stem"])
    prefix = "chronological"
    if risk_ranked:
        ordered = sorted(
            metrics,
            key=lambda row: (
                -float(row["risk_score"]),
                -float(row["added_hard_ratio"] + row["removed_hard_ratio"]),
                row["stem"],
            ),
        )
        prefix = "risk_ranked"
    panel_size = (480, 270)
    label_h = 34
    margin = 8
    columns = 3
    rows = math.ceil(per_page / columns)
    paths: list[str] = []
    sheet_root = output_root / "contact_sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)
    for page_index, start in enumerate(range(0, len(ordered), per_page), 1):
        chunk = ordered[start : start + per_page]
        sheet = Image.new(
            "RGB",
            (margin + columns * (panel_size[0] + margin), margin + rows * (panel_size[1] + label_h + margin)),
            (18, 18, 18),
        )
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(chunk):
            page_path = Path(row["review_page_path"])
            image = _read_image(page_path, "RGB", tuple(row["review_page_size"]))
            thumb = image.crop((0, 150, min(image.width, 4 * 950), min(image.height, 150 + 2 * 520)))
            thumb.thumbnail(panel_size, Image.Resampling.LANCZOS)
            framed = Image.new("RGB", panel_size, (5, 5, 5))
            framed.paste(thumb, ((panel_size[0] - thumb.width) // 2, (panel_size[1] - thumb.height) // 2))
            col = index % columns
            r = index // columns
            x = margin + col * (panel_size[0] + margin)
            y = margin + r * (panel_size[1] + label_h + margin)
            sheet.paste(framed, (x, y + label_h))
            draw.text(
                (x + 4, y + 7),
                f"{row['stem']} risk={row['risk_score']:.3f} add={row['added_hard_ratio']:.4f} rem={row['removed_hard_ratio']:.4f}",
                fill=(240, 240, 240),
            )
        path = sheet_root / f"{prefix}_page_{page_index:02d}.png"
        _atomic_image(path, sheet)
        paths.append(str(path.resolve()))
    return paths


def _write_metrics_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    fields = [
        "stem",
        "candidate_path",
        "candidate_sha256",
        "v1_area_ratio",
        "v2_area_ratio",
        "added_hard_ratio",
        "removed_hard_ratio",
        "soft_changed_pixel_ratio",
        "risk_score",
        "boundary_continuity_risk",
        "component_count",
        "largest_component_ratio",
        "boundary_length_px",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    _atomic_text(path, stream.getvalue())


def _write_review_queue(rows: Sequence[dict[str, Any]], output_root: Path) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["risk_score"]),
            -float(row["boundary_continuity_risk"]),
            row["stem"],
        ),
    )
    queue = []
    for rank, row in enumerate(ordered, 1):
        item = {
            "rank": rank,
            "stem": row["stem"],
            "risk_score": row["risk_score"],
            "risk_reasons": row["risk_reasons"],
            "candidate_path": row["candidate_path"],
            "review_page_path": row["review_page_path"],
            "added_hard_ratio": row["added_hard_ratio"],
            "removed_hard_ratio": row["removed_hard_ratio"],
            "boundary_continuity_risk": row["boundary_continuity_risk"],
            "automatic_action": "review_queue_only_no_promotion",
        }
        queue.append(item)
    fields = [
        "rank",
        "stem",
        "risk_score",
        "risk_reasons",
        "candidate_path",
        "review_page_path",
        "added_hard_ratio",
        "removed_hard_ratio",
        "boundary_continuity_risk",
        "automatic_action",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for item in queue:
        row = dict(item)
        row["risk_reasons"] = ";".join(row["risk_reasons"])
        writer.writerow(row)
    _atomic_text(output_root / "review_queue.csv", stream.getvalue())
    return queue


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def generate_full_review_v2_proposal(
    *,
    scene: Path,
    raw_root: Path,
    reviewed_v1_manifest: Path,
    proposal_v1_root: Path,
    high_risk_root: Path,
    repair_v1_root: Path,
    output_root: Path,
    expected_count: int = 111,
    source_size: tuple[int, int] = DEFAULT_SOURCE_SIZE,
    require_formal_loader: bool = True,
    repair_stems: Sequence[str] = DEFAULT_REPAIR_STEMS,
    progress=None,
) -> dict[str, Any]:
    output_root = Path(output_root).expanduser().resolve()
    state_path = output_root / "run_state.json"
    if output_root.exists() and any(path for path in output_root.iterdir() if path != state_path):
        raise ProposalAuditError(f"v2 proposal output must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        state_path,
        {"artifact": "full glass-mask v2 proposal state", "status": "IN_PROGRESS", "training_role": None},
    )
    if (Path(scene) / "specular_masks_reviewed_v2").exists():
        raise ProposalAuditError("refusing to run with an existing reviewed_v2 data directory")

    audit = audit_dr_artifacts(scene, raw_root, expected_count=expected_count)
    if tuple(audit["source_resolution"]) != tuple(source_size):
        raise ProposalAuditError(
            f"source resolution mismatch: {audit['source_resolution']} != {list(source_size)}"
        )
    _atomic_json(output_root / "audit_manifest.json", audit)
    v1_audit = audit_v1_release_and_sources(
        scene,
        reviewed_v1_manifest,
        proposal_v1_root,
        high_risk_root,
        repair_v1_root,
        expected_count=expected_count,
        source_size=source_size,
        repair_stems=repair_stems,
        require_formal_loader=require_formal_loader,
    )
    _atomic_json(output_root / "v1_source_audit.json", v1_audit)

    source_trees_before = {
        "reviewed_v1": tree_digest(Path(reviewed_v1_manifest).parent),
        "proposal_v1": tree_digest(proposal_v1_root),
        "high_risk_review_pack_v1": tree_digest(high_risk_root),
        "repair_candidates_v1": tree_digest(repair_v1_root),
        "source_rgb_images": tree_digest(Path(scene).expanduser().resolve() / "images"),
        "dr_raw_root": tree_digest(raw_root),
    }
    proposal_soft_digest, proposal_soft_count = proposal_tree_digest(
        Path(proposal_v1_root).expanduser().resolve() / "proposal_soft"
    )
    source_trees_before["proposal_v1_soft_only"] = {
        "path": str((Path(proposal_v1_root).expanduser().resolve() / "proposal_soft")),
        "file_count": proposal_soft_count,
        "sha256": proposal_soft_digest,
    }

    entries = {entry["stem"]: entry for entry in audit["entries"]}
    rows: list[dict[str, Any]] = []
    candidates: dict[str, np.ndarray] = {}
    expected = _expected_stems(expected_count)
    for index, stem in enumerate(expected, 1):
        record = v1_audit["records"][stem]
        entry = entries[stem]
        rgb = _read_image(Path(entry["rgb"]["path"]), "RGB", source_size)
        v1 = _read_image(Path(record["reviewed_v1_path"]), "L", source_size)
        normal = np.asarray(_read_image(Path(entry["artifacts"]["normal"]["path"]), "RGB", tuple(audit["native_dr_resolution"])), dtype=np.uint8)
        depth = np.asarray(_read_image(Path(entry["artifacts"]["depth"]["path"]), "RGB", tuple(audit["native_dr_resolution"])), dtype=np.uint8)
        candidate_values, info = build_v2_candidate(
            np.asarray(v1, dtype=np.uint8), np.asarray(rgb, dtype=np.uint8), normal, depth
        )
        candidate = Image.fromarray(candidate_values)
        candidate_path = output_root / "candidate_masks" / f"{stem}.png"
        _atomic_image(candidate_path, candidate)
        reloaded = _validate_mask_values(candidate_path, source_size)
        if not np.array_equal(reloaded, candidate_values):
            raise ProposalAuditError(f"{stem}: candidate reload mismatch")
        candidates[stem] = candidate_values

        v1_values = np.asarray(v1, dtype=np.uint8)
        v1_hard = v1_values >= 128
        v2_hard = candidate_values >= 128
        component = _component_stats(v2_hard)
        bbox = _bbox_from_bool(v2_hard)
        row = {
            "stem": stem,
            "candidate_path": str(candidate_path.resolve()),
            "candidate_sha256": sha256_file(candidate_path),
            "candidate_mode": "L",
            "candidate_dtype": "uint8",
            "candidate_size": list(source_size),
            "rgb_path": entry["rgb"]["path"],
            "rgb_sha256": entry["rgb"]["sha256"],
            "reviewed_v1_path": record["reviewed_v1_path"],
            "reviewed_v1_sha256": record["reviewed_v1_sha256"],
            "reviewed_v1_source": record["source_kind"],
            "v1_area_ratio": float(v1_hard.mean()),
            "v2_area_ratio": float(v2_hard.mean()),
            "v1_bbox_xyxy": _bbox_from_bool(v1_hard),
            "v2_bbox_xyxy": bbox,
            "component_count": component["component_count"],
            "largest_component_ratio": component["largest_component_ratio"],
            "boundary_length_px": _boundary_length(v2_hard),
            "candidate_min": int(candidate_values.min()),
            "candidate_max": int(candidate_values.max()),
            **info,
        }
        rows.append(row)
        if progress is not None:
            progress(index, stem)

    area_values = {row["stem"]: row["v2_area_ratio"] for row in rows}
    boundary_values = {row["stem"]: row["boundary_length_px"] for row in rows}
    for row in rows:
        stem = row["stem"]
        idx = int(stem)
        neighbors = [f"{j:06d}" for j in (idx - 1, idx + 1) if 0 <= j < expected_count]
        if neighbors:
            area_delta = max(abs(row["v2_area_ratio"] - area_values[n]) for n in neighbors)
            boundary_delta = max(
                abs(row["boundary_length_px"] - boundary_values[n])
                / max(row["boundary_length_px"], boundary_values[n], 1)
                for n in neighbors
            )
        else:
            area_delta = 0.0
            boundary_delta = 0.0
        row["adjacent_area_delta_max"] = float(area_delta)
        row["adjacent_boundary_delta_max"] = float(boundary_delta)
        row["boundary_continuity_risk"] = float(area_delta + 0.25 * boundary_delta)
        row["risk_reasons"] = []
        if row["added_hard_ratio"] > 0.001:
            row["risk_reasons"].append("added_pixels")
        if row["removed_hard_ratio"] > 0.001:
            row["risk_reasons"].append("removed_pixels")
        if row["soft_changed_pixel_ratio"] > 0.01:
            row["risk_reasons"].append("soft_edge_changed")
        if row["boundary_continuity_risk"] > 0.01:
            row["risk_reasons"].append("adjacent_boundary_continuity")
        if row["component_count"] != 1:
            row["risk_reasons"].append("component_count")
        row["risk_score"] = float(
            10.0 * (row["added_hard_ratio"] + row["removed_hard_ratio"])
            + 3.0 * row["soft_changed_pixel_ratio"]
            + 8.0 * row["boundary_continuity_risk"]
            + (0.5 if row["component_count"] != 1 else 0.0)
        )

    # Rewrite pages once to include final continuity risk in their header.
    for row in rows:
        stem = row["stem"]
        entry = entries[stem]
        rgb = _read_image(Path(entry["rgb"]["path"]), "RGB", source_size)
        v1 = _read_image(Path(v1_audit["records"][stem]["reviewed_v1_path"]), "L", source_size)
        candidate = _read_image(Path(row["candidate_path"]), "L", source_size)
        review_page = _make_review_page(stem, rgb, v1, candidate, row, output_root)
        row["review_page_path"] = review_page["path"]
        row["review_page_sha256"] = review_page["sha256"]
        row["review_page_size"] = review_page["size"]
        row["review_crops"] = review_page["crops"]

    queue = _write_review_queue(rows, output_root)
    chronological_sheets = _make_contact_sheets(rows, output_root, risk_ranked=False)
    risk_sheets = _make_contact_sheets(rows, output_root, risk_ranked=True)
    _write_metrics_csv(rows, output_root / "per_frame_metrics.csv")
    _atomic_json(
        output_root / "per_frame_metrics.json",
        {"schema_version": FULL_V2_SCHEMA_VERSION, "count": len(rows), "frames": rows},
    )
    _atomic_json(
        output_root / "review_template.json",
        {
            "schema_version": FULL_V2_SCHEMA_VERSION,
            "artifact": "human review template for full glass-mask v2 proposal",
            "training_role": None,
            "allowed_status_values": ["accepted", "accepted_with_warning", "rejected", "manual_edit_required"],
            "required_per_stem_fields": {
                "stem": "000000",
                "status": None,
                "reviewer_notes": None,
                "approved_candidate_sha256": None,
            },
            "promotion_note": "A separate explicit promotion step is required; this proposal never creates reviewed_v2.",
        },
    )
    source_trees_after = {
        "reviewed_v1": tree_digest(Path(reviewed_v1_manifest).parent),
        "proposal_v1": tree_digest(proposal_v1_root),
        "high_risk_review_pack_v1": tree_digest(high_risk_root),
        "repair_candidates_v1": tree_digest(repair_v1_root),
        "source_rgb_images": tree_digest(Path(scene).expanduser().resolve() / "images"),
        "dr_raw_root": tree_digest(raw_root),
    }
    proposal_soft_digest_after, proposal_soft_count_after = proposal_tree_digest(
        Path(proposal_v1_root).expanduser().resolve() / "proposal_soft"
    )
    source_trees_after["proposal_v1_soft_only"] = {
        "path": str((Path(proposal_v1_root).expanduser().resolve() / "proposal_soft")),
        "file_count": proposal_soft_count_after,
        "sha256": proposal_soft_digest_after,
    }
    for key, before in source_trees_before.items():
        after = source_trees_after[key]
        if before != after:
            raise ProposalAuditError(f"source tree changed during v2 proposal generation: {key}")
    _atomic_json(
        output_root / "source_tree_hashes.json",
        {"before": source_trees_before, "after": source_trees_after, "unchanged": True},
    )

    candidate_files = sorted((output_root / "candidate_masks").glob("*.png"))
    review_pages = sorted((output_root / "review_pages").glob("*_review.png"))
    if [path.stem for path in candidate_files] != expected:
        raise ProposalAuditError("candidate mask stem set is incomplete or unexpected")
    if [path.name[:6] for path in review_pages] != expected:
        raise ProposalAuditError("review page stem set is incomplete or unexpected")
    if (Path(scene) / "specular_masks_reviewed_v2").exists():
        raise ProposalAuditError("generation created a forbidden reviewed_v2 directory")

    summary = {
        "schema_version": FULL_V2_SCHEMA_VERSION,
        "artifact": "TiHuBird full glass-mask v2 proposal summary",
        "status": "PASS",
        "verdict": "GLASS_MASK_V2_PROPOSAL_READY_FOR_USER_REVIEW",
        "training_role": None,
        "reviewed_v2_created": False,
        "formal_training_manifest_created": False,
        "training_run": False,
        "stage_c_or_d_artifacts_modified": False,
        "mask_semantics": MASK_SEMANTICS,
        "count": len(rows),
        "candidate_png_count": len(candidate_files),
        "review_page_count": len(review_pages),
        "ordered_stems": expected,
        "candidate_directory": str((output_root / "candidate_masks").resolve()),
        "review_pages_directory": str((output_root / "review_pages").resolve()),
        "review_crops_directory": str((output_root / "review_crops").resolve()),
        "chronological_contact_sheets": chronological_sheets,
        "risk_ranked_contact_sheets": risk_sheets,
        "review_queue_csv": str((output_root / "review_queue.csv").resolve()),
        "review_template_json": str((output_root / "review_template.json").resolve()),
        "per_frame_metrics_csv": str((output_root / "per_frame_metrics.csv").resolve()),
        "source_tree_hashes_json": str((output_root / "source_tree_hashes.json").resolve()),
        "source_tree_hashes_unchanged": True,
        "reviewed_v1_old_repair_scope": v1_audit["old_repair_scope"],
        "candidate_area_ratio": _summary([row["v2_area_ratio"] for row in rows]),
        "added_hard_ratio": _summary([row["added_hard_ratio"] for row in rows]),
        "removed_hard_ratio": _summary([row["removed_hard_ratio"] for row in rows]),
        "soft_changed_pixel_ratio": _summary([row["soft_changed_pixel_ratio"] for row in rows]),
        "risk_score": _summary([row["risk_score"] for row in rows]),
        "highest_risk": queue[0] if queue else None,
    }
    _atomic_json(output_root / "proposal_summary.json", summary)
    manifest = {
        "schema_version": FULL_V2_SCHEMA_VERSION,
        "artifact": "TiHuBird full glass-mask v2 proposal manifest",
        "status": "PASS",
        "method_version": FULL_V2_METHOD_VERSION,
        "training_role": None,
        "warning": "Review proposal only; not accepted supervision and not reviewed_v2.",
        "mask_semantics": MASK_SEMANTICS,
        "automatic_rules": {
            "fail_closed_exact_stems": True,
            "expected_count": expected_count,
            "expected_source_size": list(source_size),
            "candidate_mode": "L",
            "candidate_dtype": "uint8",
            "reject_empty_full_nan_or_wrong_size": True,
            "source_rgb_dr_v1_hash_mismatch_stops": True,
            "risk_is_review_queue_only": True,
            "automatic_promotion": False,
        },
        "code_identity": {
            "method_module": "utils.dr_mask_full_review_v2",
            "method_version": FULL_V2_METHOD_VERSION,
        },
        "source_audit": v1_audit,
        "source_tree_hashes": {"before": source_trees_before, "after": source_trees_after, "unchanged": True},
        "outputs": summary,
        "frames": rows,
    }
    _atomic_json(output_root / "proposal_manifest.json", manifest)
    _atomic_json(
        state_path,
        {
            "artifact": "full glass-mask v2 proposal state",
            "status": "PASS",
            "candidate_png_count": len(candidate_files),
            "review_page_count": len(review_pages),
            "training_role": None,
            "reviewed_v2_created": False,
        },
    )
    return manifest
