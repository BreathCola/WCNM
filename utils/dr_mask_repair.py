"""Localized top-boundary repair candidates for three failed DR proposals.

The source proposal tree is read-only.  Repairs are subtractive candidates,
not reviewed or final masks, and are generated only for 000039--000041.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from utils.dr_mask_high_risk_review import (
    _boundary_overlay,
    _read_image,
    _read_source_rgb,
    _shrink_only_panel,
    proposal_tree_digest,
)
from utils.dr_mask_proposal import (
    ProposalAuditError,
    _atomic_image,
    _atomic_json,
    boundary_maps,
    sha256_file,
)
from utils.dr_mask_review import _clamped_box


REPAIR_STEMS = ("000039", "000040", "000041")
REFERENCE_STEMS = ("000038", "000042")
CONTINUITY_STEMS = ("000037", "000038", "000039", "000040", "000041", "000042", "000043")
REPAIR_SCHEMA_VERSION = 1
FEATHER_PIXELS = 16.0


@dataclass(frozen=True)
class Line:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def slope(self) -> float:
        return (self.y2 - self.y1) / max(self.x2 - self.x1, 1e-8)

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    def y_at(self, x: np.ndarray | float) -> np.ndarray | float:
        return self.y1 + self.slope * (x - self.x1)

    def as_list(self) -> list[float]:
        return [float(self.x1), float(self.y1), float(self.x2), float(self.y2)]


def _largest_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(
        (np.asarray(mask, dtype=np.uint8) >= 128).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ProposalAuditError("proposal has no contour")
    return max(contours, key=cv2.contourArea)


def _hard_bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(np.asarray(mask) >= 128)
    if xs.size == 0:
        raise ProposalAuditError("proposal is empty")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def reference_top_line(mask: np.ndarray) -> Line:
    """Extract the long upper enclosure edge from an accepted neighbor mask."""
    contour = _largest_contour(mask)
    x, y, width, height = cv2.boundingRect(contour)
    polygon = cv2.approxPolyDP(
        contour, 0.012 * cv2.arcLength(contour, True), True
    ).reshape(-1, 2)
    candidates: list[Line] = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        if first[0] <= second[0]:
            line = Line(float(first[0]), float(first[1]), float(second[0]), float(second[1]))
        else:
            line = Line(float(second[0]), float(second[1]), float(first[0]), float(first[1]))
        mean_y = 0.5 * (line.y1 + line.y2)
        if (
            abs(line.slope) <= 0.15
            and line.length >= 0.50 * width
            and mean_y <= y + 0.28 * height
        ):
            candidates.append(line)
    if not candidates:
        raise ProposalAuditError("accepted neighbor has no stable top-boundary reference")
    return max(candidates, key=lambda item: item.length)


def _sample_line_cues(
    line: Line,
    rgb: np.ndarray,
    geometry_boundary: np.ndarray,
) -> tuple[float, float]:
    height, width = rgb.shape[:2]
    count = max(100, int(line.length / 3.0))
    xx = np.linspace(line.x1, line.x2, count)
    yy = np.linspace(line.y1, line.y2, count)
    xi = np.clip(np.rint(xx).astype(np.int32), 0, width - 1)
    yi = np.clip(np.rint(yy).astype(np.int32), 0, height - 1)
    pixels = rgb[yi, xi].astype(np.float32)
    green_excess = float(
        np.mean(pixels[:, 1] - 0.5 * (pixels[:, 0] + pixels[:, 2]))
    )
    native_height, native_width = geometry_boundary.shape
    gx = np.clip(np.rint(xx * native_width / width).astype(np.int32), 0, native_width - 1)
    gy = np.clip(np.rint(yy * native_height / height).astype(np.int32), 0, native_height - 1)
    samples = []
    for offset in range(-2, 3):
        samples.append(
            geometry_boundary[np.clip(gy + offset, 0, native_height - 1), gx]
        )
    geometry_support = float(np.max(np.stack(samples), axis=0).mean())
    return green_excess, geometry_support


def detect_repair_top_line(
    rgb: np.ndarray,
    normal_rgb: np.ndarray,
    depth_rgb: np.ndarray,
    original_soft: np.ndarray,
    references: Sequence[Line],
) -> tuple[Line, dict[str, Any]]:
    """Select a visible RGB top edge, using DR geometry and neighbor continuity."""
    if len(references) < 2:
        raise ProposalAuditError("top repair requires two accepted neighbor references")
    height, width = rgb.shape[:2]
    bbox = _hard_bbox(original_soft)
    x1, y1, x2, y2 = bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    normal_boundary, depth_boundary = boundary_maps(normal_rgb, depth_rgb)
    geometry_boundary = np.maximum(normal_boundary, depth_boundary)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    roi = np.zeros_like(edges)
    horizontal_margin = int(round(0.14 * bbox_width))
    roi_right = min(width, x2 + horizontal_margin)
    roi_left = max(0, x1 - horizontal_margin)
    roi_bottom = min(
        height,
        max(int(round(0.18 * height)), int(round(y1 + 0.35 * bbox_height))),
    )
    roi[:roi_bottom, roi_left:roi_right] = edges[:roi_bottom, roi_left:roi_right]
    hough = cv2.HoughLinesP(
        roi,
        1,
        np.pi / 180.0,
        threshold=max(10, int(round(0.075 * bbox_width))),
        minLineLength=max(20, int(round(0.35 * bbox_width))),
        maxLineGap=max(8, int(round(0.065 * bbox_width))),
    )
    if hough is None:
        raise ProposalAuditError("no RGB top-edge candidates detected")

    reference_slope = float(np.mean([line.slope for line in references]))
    center_x = 0.5 * (x1 + x2)
    reference_center_y = float(np.mean([line.y_at(center_x) for line in references]))
    candidates = []
    for raw in hough[:, 0]:
        ax, ay, bx, by = [float(value) for value in raw]
        if ax > bx:
            ax, ay, bx, by = bx, by, ax, ay
        if bx - ax <= 1:
            continue
        line = Line(ax, ay, bx, by)
        midpoint_ratio = 0.5 * (line.y1 + line.y2) / height
        if abs(line.slope) > 0.15 or not 0.04 <= midpoint_ratio <= 0.20:
            continue
        green_excess, geometry_support = _sample_line_cues(
            line, rgb, geometry_boundary
        )
        green_score = float(np.clip(green_excess / 25.0, 0.0, 1.0))
        span_score = float(min(line.length / max(bbox_width, 1), 1.0))
        candidate_center_y = float(line.y_at(center_x))
        continuity_score = float(
            math.exp(
                -0.5 * ((candidate_center_y - reference_center_y) / 100.0) ** 2
                -0.5 * ((line.slope - reference_slope) / 0.05) ** 2
            )
        )
        score = (
            0.45 * green_score
            + 0.15 * geometry_support
            + 0.15 * span_score
            + 0.25 * continuity_score
        )
        candidates.append(
            {
                "line": line,
                "score": float(score),
                "green_excess": green_excess,
                "green_score": green_score,
                "geometry_support": geometry_support,
                "span_ratio": span_score,
                "continuity_score": continuity_score,
                "center_y": candidate_center_y,
            }
        )
    if not candidates:
        raise ProposalAuditError("all RGB top-edge candidates failed repair constraints")
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[0]
    line = selected["line"]
    extended = Line(float(x1), float(line.y_at(x1)), float(x2 - 1), float(line.y_at(x2 - 1)))
    if min(extended.y1, extended.y2) < 0 or max(extended.y1, extended.y2) >= height:
        raise ProposalAuditError("selected top edge extrapolates outside the source image")
    evidence = {
        "selected_visible_segment_xyxy": line.as_list(),
        "extended_top_line_xyxy": extended.as_list(),
        "selected_score": selected["score"],
        "rgb_green_excess": selected["green_excess"],
        "dr_geometry_boundary_support": selected["geometry_support"],
        "visible_span_ratio": selected["span_ratio"],
        "neighbor_continuity_score": selected["continuity_score"],
        "reference_slope_mean": reference_slope,
        "reference_center_y_mean": reference_center_y,
        "candidate_count": len(candidates),
        "top_candidates": [
            {
                "segment_xyxy": item["line"].as_list(),
                "score": item["score"],
                "rgb_green_excess": item["green_excess"],
                "dr_geometry_boundary_support": item["geometry_support"],
                "visible_span_ratio": item["span_ratio"],
                "neighbor_continuity_score": item["continuity_score"],
            }
            for item in candidates[:5]
        ],
    }
    return extended, evidence


def apply_subtractive_top_repair(
    original_soft: np.ndarray,
    top_line: Line,
    feather_pixels: float = FEATHER_PIXELS,
) -> np.ndarray:
    values = np.asarray(original_soft, dtype=np.float32) / 255.0
    height, width = values.shape
    xx = np.arange(width, dtype=np.float32)[None, :]
    yy = np.arange(height, dtype=np.float32)[:, None]
    line_y = top_line.y_at(xx)
    gate = np.clip(
        (yy - line_y + feather_pixels) / (2.0 * feather_pixels), 0.0, 1.0
    )
    repaired = np.round(255.0 * values * gate).astype(np.uint8)
    if np.any(repaired > original_soft):
        raise ProposalAuditError("subtractive repair unexpectedly added proposal pixels")
    return repaired


def _soft_overlay(rgb: Image.Image, soft: Image.Image) -> Image.Image:
    base = np.asarray(rgb, dtype=np.float32)
    values = np.asarray(soft, dtype=np.float32) / 255.0
    color = np.zeros_like(base)
    color[..., 0] = 30
    color[..., 1] = 225
    color[..., 2] = 210
    alpha = values[..., None] * 0.48
    return Image.fromarray(
        np.round(base * (1.0 - alpha) + color * alpha).astype(np.uint8)
    )


def _boundary_difference(
    rgb: Image.Image,
    original_soft: Image.Image,
    repaired_soft: Image.Image,
) -> Image.Image:
    base = np.asarray(rgb, dtype=np.float32) * 0.30
    original = np.asarray(original_soft, dtype=np.uint8)
    repaired = np.asarray(repaired_soft, dtype=np.uint8)
    changed = np.abs(original.astype(np.int16) - repaired.astype(np.int16)) > 0
    base[changed] = 0.35 * base[changed] + 0.65 * np.array([255, 55, 30], dtype=np.float32)
    result = np.clip(base, 0, 255).astype(np.uint8)
    for values, color in ((original, (255, 35, 210)), (repaired, (30, 235, 255))):
        contours, _ = cv2.findContours(
            (values >= 128).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color, 7, cv2.LINE_AA)
    return Image.fromarray(result)


def _difference_stats(original: np.ndarray, repaired: np.ndarray) -> dict[str, Any]:
    original_hard = original >= 128
    repaired_hard = repaired >= 128
    changed = original != repaired
    changed_hard = original_hard != repaired_hard
    removed_hard = original_hard & ~repaired_hard
    added_hard = ~original_hard & repaired_hard
    changed_bbox = _hard_bbox(changed.astype(np.uint8) * 255) if np.any(changed) else None
    return {
        "v1_area_ratio": float(original_hard.mean()),
        "repair_area_ratio": float(repaired_hard.mean()),
        "v1_bbox_xyxy": _hard_bbox(original),
        "repair_bbox_xyxy": _hard_bbox(repaired),
        "changed_soft_pixel_ratio": float(changed.mean()),
        "changed_hard_pixel_ratio": float(changed_hard.mean()),
        "removed_hard_pixel_count": int(removed_hard.sum()),
        "added_hard_pixel_count": int(added_hard.sum()),
        "changed_region_bbox_xyxy": changed_bbox,
    }


def _labeled_direct_panel(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    label: str,
    x: int,
    y: int,
) -> None:
    draw.text((x + 4, y + 5), label, fill=(245, 245, 245))
    canvas.paste(image.convert("RGB"), (x, y + 30))


def _make_continuity_strip(
    proposal_root: Path,
    rgb_by_stem: Mapping[str, Image.Image],
    overlay_by_stem: Mapping[str, Image.Image],
    output: Path,
) -> Image.Image:
    panel_size = (1000, 562)
    header = 34
    margin = 8
    width = margin + len(CONTINUITY_STEMS) * (panel_size[0] + margin)
    height = margin + header + panel_size[1] + margin
    canvas = Image.new("RGB", (width, height), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    for index, stem in enumerate(CONTINUITY_STEMS):
        image = overlay_by_stem[stem]
        panel = _shrink_only_panel(image, panel_size)
        x = margin + index * (panel_size[0] + margin)
        canvas.paste(panel, (x, margin + header))
        role = "repair" if stem in REPAIR_STEMS else "v1 reference"
        draw.text((x + 4, margin + 7), f"{stem} | {role}", fill=(240, 240, 240))
    _atomic_image(Path(output), canvas)
    return canvas


def _make_review_page(
    stem: str,
    rgb: Image.Image,
    v1_overlay: Image.Image,
    repair_overlay: Image.Image,
    difference: Image.Image,
    repaired_soft: Image.Image,
    continuity_strip: Image.Image,
    stats: dict[str, Any],
    evidence: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    source_width, source_height = rgb.size
    margin = 16
    label_height = 30
    header_height = 132
    page_width = 2 * source_width + 3 * margin
    overview_row_height = source_height + label_height + margin
    crop_size = (900, 506)
    crop_row_height = crop_size[1] + label_height + margin
    continuity_panel = _shrink_only_panel(continuity_strip, (7000, 650))
    continuity_height = continuity_panel.height + label_height + margin
    page_height = (
        header_height + 2 * overview_row_height + crop_row_height + continuity_height + margin
    )
    canvas = Image.new("RGB", (page_width, page_height), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 12), f"{stem} | localized top-boundary repair candidate", fill=(255, 255, 255))
    draw.text(
        (18, 38),
        f"area v1={stats['v1_area_ratio']:.6f} repair={stats['repair_area_ratio']:.6f}  "
        f"changed_soft={stats['changed_soft_pixel_ratio']:.6f} "
        f"changed_hard={stats['changed_hard_pixel_ratio']:.6f}",
        fill=(225, 225, 225),
    )
    draw.text(
        (18, 64),
        f"bbox v1={stats['v1_bbox_xyxy']} repair={stats['repair_bbox_xyxy']}  "
        f"changed_bbox={stats['changed_region_bbox_xyxy']}",
        fill=(225, 225, 225),
    )
    draw.text(
        (18, 90),
        f"top line={evidence['extended_top_line_xyxy']}  score={evidence['selected_score']:.4f}  "
        f"visible_span={evidence['visible_span_ratio']:.4f}",
        fill=(255, 190, 90),
    )
    draw.text(
        (18, 112),
        "Candidate only; magenta=v1 boundary, cyan=repair boundary, red=changed region.",
        fill=(185, 210, 255),
    )
    panels = (
        (rgb, "Original RGB"),
        (v1_overlay, "v1 proposal overlay"),
        (repair_overlay, "repair candidate overlay"),
        (difference, "boundary difference"),
    )
    for index, (image, label) in enumerate(panels):
        column = index % 2
        row = index // 2
        x = margin + column * (source_width + margin)
        y = header_height + row * overview_row_height
        _labeled_direct_panel(canvas, draw, image, label, x, y)

    repair_boundary = _boundary_overlay(rgb, repaired_soft)
    bbox = stats["repair_bbox_xyxy"]
    x1, y1, x2, y2 = bbox
    crop_boxes = {
        "top_edge": _clamped_box(0.5 * (x1 + x2), y1, rgb.size, crop_size),
        "bottom_edge": _clamped_box(0.5 * (x1 + x2), y2, rgb.size, crop_size),
        "left_edge": _clamped_box(x1, 0.5 * (y1 + y2), rgb.size, crop_size),
        "right_edge": _clamped_box(x2, 0.5 * (y1 + y2), rgb.size, crop_size),
    }
    crop_root = output_root / "review_crops" / stem
    crop_records = {}
    crop_y = header_height + 2 * overview_row_height
    for index, (name, box) in enumerate(crop_boxes.items()):
        raw_crop = rgb.crop(box)
        boundary_crop = repair_boundary.crop(box)
        if raw_crop.size != crop_size or boundary_crop.size != crop_size:
            raise ProposalAuditError(f"{stem}/{name}: review crop is not 900x506")
        raw_path = crop_root / f"{name}_rgb.png"
        boundary_path = crop_root / f"{name}_repair_boundary.png"
        _atomic_image(raw_path, raw_crop)
        _atomic_image(boundary_path, boundary_crop)
        x = margin + index * (crop_size[0] + margin)
        _labeled_direct_panel(canvas, draw, boundary_crop, f"1:1 source crop | {name}", x, crop_y)
        crop_records[name] = {
            "box_xyxy_source": list(box),
            "resampling": None,
            "rgb_path": str(raw_path.resolve()),
            "rgb_sha256": sha256_file(raw_path),
            "repair_boundary_path": str(boundary_path.resolve()),
            "repair_boundary_sha256": sha256_file(boundary_path),
        }
    continuity_y = crop_y + crop_row_height
    draw.text((margin + 4, continuity_y + 5), "Cross-view contour continuity: 000037--000043", fill=(245, 245, 245))
    canvas.paste(continuity_panel, (margin, continuity_y + label_height))
    page_path = output_root / "review_pages" / f"{stem}_repair_review.png"
    _atomic_image(page_path, canvas)
    return {
        "path": str(page_path.resolve()),
        "sha256": sha256_file(page_path),
        "size": [page_width, page_height],
        "crops": crop_records,
    }


def _make_three_frame_comparison(
    rgb_by_stem: Mapping[str, Image.Image],
    v1_overlay_by_stem: Mapping[str, Image.Image],
    repair_overlay_by_stem: Mapping[str, Image.Image],
    difference_by_stem: Mapping[str, Image.Image],
    output: Path,
) -> None:
    panel_size = (960, 540)
    labels = ("RGB", "v1 overlay", "repair overlay", "boundary difference")
    margin = 8
    header = 34
    width = margin + 4 * (panel_size[0] + margin)
    height = margin + len(REPAIR_STEMS) * (panel_size[1] + header + margin)
    canvas = Image.new("RGB", (width, height), (15, 15, 15))
    draw = ImageDraw.Draw(canvas)
    for row, stem in enumerate(REPAIR_STEMS):
        images = (
            rgb_by_stem[stem], v1_overlay_by_stem[stem],
            repair_overlay_by_stem[stem], difference_by_stem[stem],
        )
        y = margin + row * (panel_size[1] + header + margin)
        for column, (image, label) in enumerate(zip(images, labels)):
            x = margin + column * (panel_size[0] + margin)
            canvas.paste(_shrink_only_panel(image, panel_size), (x, y + header))
            draw.text((x + 3, y + 6), f"{stem} | {label}", fill=(245, 245, 245))
    _atomic_image(Path(output), canvas)


def generate_repair_candidates(
    proposal_output_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    proposal_output_root = Path(proposal_output_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    report_source = proposal_output_root / "review_manifest_111.json"
    try:
        source_report = json.loads(report_source.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProposalAuditError(f"cannot parse source proposal report: {error}") from error
    if source_report.get("status") != "PASS":
        raise ProposalAuditError("source proposal report is not PASS")
    frames = {frame["stem"]: frame for frame in source_report["frames"]}
    required = set(CONTINUITY_STEMS)
    if not required.issubset(frames):
        raise ProposalAuditError("source proposal report lacks repair/reference frames")
    if output_root.exists() and any(
        path for path in output_root.iterdir() if path.name != "run_state.json"
    ):
        raise ProposalAuditError(f"repair output must be new or empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if (proposal_output_root / "reviewed_soft").exists():
        raise ProposalAuditError("source proposal unexpectedly contains reviewed_soft")

    digest_before, source_file_count = proposal_tree_digest(
        proposal_output_root / "proposal_soft"
    )
    source_size = tuple(source_report["method"]["source_domain"])
    rgb_by_stem = {}
    soft_by_stem = {}
    v1_overlay_by_stem = {}
    for stem in CONTINUITY_STEMS:
        frame = frames[stem]
        directory = proposal_output_root / "proposal_soft" / stem
        rgb_by_stem[stem] = _read_source_rgb(Path(frame["inputs"]["rgb"]["path"]), source_size)
        soft_by_stem[stem] = _read_image(directory / "proposal_soft.png", "L", source_size)
        v1_overlay_by_stem[stem] = _read_image(
            directory / "proposal_overlay.png", "RGB", source_size
        )
        if sha256_file(directory / "proposal_soft.png") != frame["outputs"]["proposal_soft"]["sha256"]:
            raise ProposalAuditError(f"{stem}: source proposal SHA-256 mismatch")

    references = [
        reference_top_line(np.asarray(soft_by_stem[stem], dtype=np.uint8))
        for stem in REFERENCE_STEMS
    ]
    repair_overlay_by_stem: dict[str, Image.Image] = {}
    difference_by_stem: dict[str, Image.Image] = {}
    repairs: dict[str, Image.Image] = {}
    records: dict[str, dict[str, Any]] = {}
    for stem in REPAIR_STEMS:
        frame = frames[stem]
        original = np.asarray(soft_by_stem[stem], dtype=np.uint8)
        normal_path = Path(frame["inputs"]["dr_normal"]["path"])
        depth_path = Path(frame["inputs"]["dr_depth"]["path"])
        with Image.open(normal_path) as opened:
            normal_rgb = np.asarray(opened, dtype=np.uint8)
        with Image.open(depth_path) as opened:
            depth_rgb = np.asarray(opened, dtype=np.uint8)
        rgb_values = np.asarray(rgb_by_stem[stem], dtype=np.uint8)
        top_line, evidence = detect_repair_top_line(
            rgb_values, normal_rgb, depth_rgb, original, references
        )
        repaired = apply_subtractive_top_repair(original, top_line)
        stats = _difference_stats(original, repaired)
        if stats["added_hard_pixel_count"] != 0:
            raise ProposalAuditError(f"{stem}: repair added hard-mask pixels")
        candidate = Image.fromarray(repaired)
        candidate_path = output_root / "repair_candidates" / f"{stem}.png"
        _atomic_image(candidate_path, candidate)
        repairs[stem] = candidate
        repair_overlay_by_stem[stem] = _soft_overlay(rgb_by_stem[stem], candidate)
        difference_by_stem[stem] = _boundary_difference(
            rgb_by_stem[stem], soft_by_stem[stem], candidate
        )
        records[stem] = {
            "stem": stem,
            "artifact": "localized automatic repair candidate",
            "training_role": None,
            "candidate_path": str(candidate_path.resolve()),
            "candidate_sha256": sha256_file(candidate_path),
            "source_v1_path": frame["outputs"]["proposal_soft"]["path"],
            "source_v1_sha256": frame["outputs"]["proposal_soft"]["sha256"],
            "repair_operation": "subtractive top-boundary clipping only",
            "copied_or_interpolated_neighbor_mask": False,
            "reference_stems": list(REFERENCE_STEMS),
            "feather_pixels": FEATHER_PIXELS,
            "evidence": evidence,
            "difference": stats,
            "manual_review": {
                "required": True,
                "reason": (
                    "the glass top edge is partially occluded/reflected; the visible RGB "
                    "segment is extrapolated across the hidden span"
                ),
                "hidden_top_fraction_estimate": float(1.0 - evidence["visible_span_ratio"]),
            },
        }

    continuity_overlay = dict(v1_overlay_by_stem)
    continuity_overlay.update(repair_overlay_by_stem)
    continuity_path = output_root / "continuity_comparison_000037_000043.png"
    continuity_strip = _make_continuity_strip(
        proposal_output_root, rgb_by_stem, continuity_overlay, continuity_path
    )
    for stem in REPAIR_STEMS:
        records[stem]["review_page"] = _make_review_page(
            stem,
            rgb_by_stem[stem],
            v1_overlay_by_stem[stem],
            repair_overlay_by_stem[stem],
            difference_by_stem[stem],
            repairs[stem],
            continuity_strip,
            records[stem]["difference"],
            records[stem]["evidence"],
            output_root,
        )
    comparison_path = output_root / "three_frame_repair_comparison.png"
    _make_three_frame_comparison(
        rgb_by_stem,
        v1_overlay_by_stem,
        repair_overlay_by_stem,
        difference_by_stem,
        comparison_path,
    )
    digest_after, final_source_file_count = proposal_tree_digest(
        proposal_output_root / "proposal_soft"
    )
    if digest_before != digest_after or source_file_count != final_source_file_count:
        raise ProposalAuditError("source proposal tree changed during repair generation")
    if (output_root / "reviewed_soft").exists():
        raise ProposalAuditError("repair generation must not create reviewed_soft")
    report = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "artifact": "three localized automatic glass-mask repair candidates",
        "status": "PASS",
        "training_role": None,
        "warning": "Repair candidates only; not reviewed, final, or L_spec supervision.",
        "source_proposal_root": str(proposal_output_root),
        "source_proposal_tree_file_count": source_file_count,
        "source_proposal_tree_sha256_before": digest_before,
        "source_proposal_tree_sha256_after": digest_after,
        "source_proposal_files_modified": False,
        "repair_stems": list(REPAIR_STEMS),
        "reference_stems": list(REFERENCE_STEMS),
        "continuity_stems": list(CONTINUITY_STEMS),
        "global_proposal_parameters_changed": False,
        "other_proposals_regenerated": 0,
        "three_frame_comparison": {
            "path": str(comparison_path.resolve()),
            "sha256": sha256_file(comparison_path),
        },
        "continuity_comparison": {
            "path": str(continuity_path.resolve()),
            "sha256": sha256_file(continuity_path),
        },
        "repairs": records,
        "reviewed_soft_created": False,
        "formal_training_manifest_created": False,
        "lambda_spec_authorized": False,
    }
    _atomic_json(output_root / "repair_report.json", report)
    return report
