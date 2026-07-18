"""Cue construction and review packaging for Tao DR geometry mask repair.

Nothing in this module implements a training loader or a formal geometry
release.  Old proposal pixels are accepted only by review-comparison helpers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from geometry.tao_dr_cuboid_bootstrap import (
    CUBOID_EDGES,
    CUBOID_FACES,
    PREDECLARED_THRESHOLDS,
    cuboid_vertices,
    project_vertices,
    projected_hull,
    rasterize_projected_cuboid,
)


PROPOSAL_SCHEMA = "rtgs_tao_dr_geometry_glass_mask_repair_proposal_v2"
METRICS_SCHEMA = "rtgs_tao_dr_geometry_mask_repair_metrics_v2"
REVIEW_TEMPLATE_SCHEMA = "rtgs_tao_dr_geometry_mask_repair_review_template_v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_image(path: Path, image: Image.Image) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        image.save(temporary, format="PNG", optimize=False)
        with Image.open(temporary) as opened:
            opened.verify()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _robust_unit(values: np.ndarray, low: float = 0.01, high: float = 0.99) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.quantile(finite, (low, high))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _color_gradient(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float32) / 255.0
    channels = []
    for index in range(3):
        gx = cv2.Scharr(value[..., index], cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(value[..., index], cv2.CV_32F, 0, 1)
        channels.append(gx * gx + gy * gy)
    return np.sqrt(np.sum(channels, axis=0))


def compute_native_cues(
    stored_rgb: np.ndarray,
    raw_normal: np.ndarray,
    raw_depth: np.ndarray,
    basecolor: np.ndarray,
    diffuse_albedo: np.ndarray,
) -> dict[str, Any]:
    arrays = [np.asarray(value) for value in (
        stored_rgb, raw_normal, raw_depth, basecolor, diffuse_albedo
    )]
    if any(value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3 for value in arrays):
        raise ValueError("all DR cues must be uint8 HxWx3")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("DR cue shapes do not match")
    normal = raw_normal.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(length <= 1e-6) or not np.isfinite(length).all():
        raise ValueError("DR normal cue has invalid vectors")
    normal /= length
    normal_energy = np.zeros(normal.shape[:2], dtype=np.float32)
    for index in range(3):
        gx = cv2.Scharr(normal[..., index], cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(normal[..., index], cv2.CV_32F, 0, 1)
        normal_energy += gx * gx + gy * gy
    normal_edge = _robust_unit(np.sqrt(normal_energy), 0.02, 0.995)
    depth_relative = raw_depth.astype(np.float32).mean(axis=-1) / np.float32(255.0)
    depth_gx = cv2.Scharr(depth_relative, cv2.CV_32F, 1, 0)
    depth_gy = cv2.Scharr(depth_relative, cv2.CV_32F, 0, 1)
    depth_edge = _robust_unit(np.sqrt(depth_gx * depth_gx + depth_gy * depth_gy), 0.02, 0.995)
    rgb_edge = _robust_unit(_color_gradient(stored_rgb), 0.02, 0.995)
    basecolor_edge = _robust_unit(_color_gradient(basecolor), 0.02, 0.995)
    diffuse_edge = _robust_unit(_color_gradient(diffuse_albedo), 0.02, 0.995)
    mean_normal = np.stack([
        cv2.GaussianBlur(normal[..., index], (0, 0), 2.0) for index in range(3)
    ], axis=-1)
    planar_consistency = np.clip(np.linalg.norm(mean_normal, axis=-1), 0.0, 1.0).astype(np.float32)
    rgb = stored_rgb.astype(np.float32) / 255.0
    base = basecolor.astype(np.float32) / 255.0
    diffuse = diffuse_albedo.astype(np.float32) / 255.0
    inverse_model_disagreement = np.mean(np.abs(rgb - base), axis=-1) \
        + 0.5 * np.mean(np.abs(base - diffuse), axis=-1)
    brightness = rgb.max(axis=-1)
    saturation = rgb.max(axis=-1) - rgb.min(axis=-1)
    highlight = np.clip((brightness - 0.72) / 0.28, 0.0, 1.0) * (1.0 - saturation)
    reflection_uncertainty = _robust_unit(inverse_model_disagreement + 0.35 * highlight, 0.05, 0.99)
    raw_fused = (
        0.30 * normal_edge + 0.25 * depth_edge + 0.15 * rgb_edge
        + 0.15 * basecolor_edge + 0.10 * diffuse_edge
        + 0.05 * (1.0 - planar_consistency)
    ) * (1.0 - 0.30 * reflection_uncertainty)
    fused = _robust_unit(raw_fused, 0.01, 0.995)
    # A small fixed blur represents localization uncertainty.  This remains a
    # continuous likelihood sampled by the 3-D objective, not a mask threshold.
    fused = cv2.GaussianBlur(fused, (0, 0), 1.5)
    return {
        "normal_unit_unmapped": normal,
        "relative_depth": depth_relative,
        "normal_discontinuity": normal_edge,
        "relative_depth_discontinuity": depth_edge,
        "rgb_edge": rgb_edge,
        "basecolor_edge": basecolor_edge,
        "diffuse_albedo_edge": diffuse_edge,
        "planar_normal_consistency": planar_consistency,
        "reflection_uncertainty": reflection_uncertainty,
        "fused_boundary": fused,
        "statistics": {
            "fused_mean": float(fused.mean()),
            "fused_p90": float(np.quantile(fused, 0.90)),
            "planar_mean": float(planar_consistency.mean()),
            "reflection_uncertainty_mean": float(reflection_uncertainty.mean()),
        },
    }


def mapped_world_normal(
    raw_normal: np.ndarray, mapping: np.ndarray, world_to_camera: np.ndarray
) -> np.ndarray:
    normal = raw_normal.astype(np.float64) / 127.5 - 1.0
    normal = normal @ np.asarray(mapping, dtype=np.float64).T
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    world = normal @ np.asarray(world_to_camera, dtype=np.float64)
    world /= np.linalg.norm(world, axis=-1, keepdims=True)
    return world.astype(np.float32)


def interior_likelihood(
    calibrated_depth: np.ndarray,
    planar_consistency: np.ndarray,
    intrinsics: np.ndarray,
    normal_unit_unmapped: np.ndarray | None = None,
) -> np.ndarray:
    depth = np.asarray(calibrated_depth, dtype=np.float32)
    planar = np.asarray(planar_consistency, dtype=np.float32)
    height, width = depth.shape
    valid = np.isfinite(depth) & (depth > 1e-6)
    depth_unit = np.ones_like(depth, dtype=np.float32)
    if np.any(valid):
        finite_scaled = _robust_unit(depth[valid], 0.02, 0.98)
        depth_unit[valid] = finite_scaled
    near = np.where(valid, 1.0 - depth_unit, 0.0)
    yy, xx = np.mgrid[:height, :width]
    cx = float(intrinsics[0, 2]); cy = float(intrinsics[1, 2])
    centrality = np.exp(
        -0.5 * ((xx - cx) / (0.34 * width)) ** 2
        -0.5 * ((yy - cy) / (0.40 * height)) ** 2
    ).astype(np.float32)
    if normal_unit_unmapped is None:
        normal_similarity = planar
    else:
        normal = np.asarray(normal_unit_unmapped, dtype=np.float32)
        if normal.shape != depth.shape + (3,):
            raise ValueError("normal/depth shape mismatch for interior likelihood")
        seed = centrality >= np.quantile(centrality, 0.92)
        seed_normal = np.median(normal[seed], axis=0)
        seed_length = float(np.linalg.norm(seed_normal))
        if seed_length <= 1e-6:
            raise ValueError("central DR normal seed is degenerate")
        seed_normal /= seed_length
        normal_similarity = np.abs(normal @ seed_normal)
    seed = centrality >= np.quantile(centrality, 0.92)
    seed_depth = float(np.median(depth[seed & valid])) if np.any(seed & valid) else float("nan")
    if np.isfinite(seed_depth):
        depth_scale = float(np.median(np.abs(depth[seed & valid] - seed_depth)))
        valid_depth = depth[valid]
        depth_scale = max(depth_scale, float(np.quantile(valid_depth, 0.75) - np.quantile(valid_depth, 0.25)) * 0.08, 1e-4)
        depth_similarity = np.zeros_like(depth, dtype=np.float32)
        depth_similarity[valid] = np.exp(-np.abs(depth[valid] - seed_depth) / (3.0 * depth_scale))
    else:
        depth_similarity = near
    # The central seed disambiguates the enclosure block from the equally-near
    # floor: a matching planar normal and calibrated-depth band are both needed.
    result = (
        0.40 * normal_similarity + 0.30 * depth_similarity
        + 0.20 * centrality + 0.10 * planar
    )
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def initialize_bounds_from_calibrated_views(
    views: Sequence[dict[str, Any]],
    sparse_points: np.ndarray,
    axes: np.ndarray,
    focus: np.ndarray,
    camera_radius: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a metric initializer; it is not a silhouette target or fallback."""
    dense_rows = []
    quantile = float(PREDECLARED_THRESHOLDS["initial_support_likelihood_quantile"])
    stride = int(PREDECLARED_THRESHOLDS["initial_dense_grid_stride"])
    radius_limit = float(camera_radius) * float(
        PREDECLARED_THRESHOLDS["initial_focus_radius_fraction"]
    )
    per_view = []
    for view in views:
        if float(view["depth_weight"]) <= 0:
            per_view.append({"stem": view["stem"], "selected": 0, "trusted": False})
            continue
        likelihood = np.asarray(view["interior_likelihood"])
        threshold = float(np.quantile(likelihood, quantile))
        yy, xx = np.mgrid[0:likelihood.shape[0]:stride, 0:likelihood.shape[1]:stride]
        pixel = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
        score = likelihood[yy, xx].reshape(-1)
        depth = np.asarray(view["calibrated_depth"])[yy, xx].reshape(-1)
        keep = (score >= threshold) & np.isfinite(depth) & (depth > 1e-6)
        pixel = pixel[keep]; depth = depth[keep]
        if pixel.size:
            xcam = (pixel[:, 0] - view["intrinsics"][0, 2]) / view["intrinsics"][0, 0] * depth
            ycam = (pixel[:, 1] - view["intrinsics"][1, 2]) / view["intrinsics"][1, 1] * depth
            camera_xyz = np.stack((xcam, ycam, depth), axis=1)
            world = camera_xyz @ view["world_to_camera"] + view["camera_center"][None]
            keep_focus = np.linalg.norm(world - focus[None], axis=1) <= radius_limit
            world = world[keep_focus]
            if world.size:
                dense_rows.append(world)
        per_view.append({
            "stem": view["stem"], "selected": int(0 if not pixel.size else world.shape[0]),
            "trusted": True, "likelihood_threshold": threshold,
        })
    if not dense_rows:
        raise RuntimeError("no calibrated DR support remained for cuboid initialization")
    dense = np.concatenate(dense_rows, axis=0)
    sparse = np.asarray(sparse_points, dtype=np.float64)
    sparse_focus = sparse[np.linalg.norm(sparse - focus[None], axis=1) <= radius_limit]
    if sparse_focus.shape[0] < 100:
        raise RuntimeError("Tao sparse geometry has insufficient robust focus support")
    dense_local = dense @ axes
    sparse_local = sparse_focus @ axes
    dense_lower, dense_upper = np.quantile(dense_local, (0.01, 0.99), axis=0)
    sparse_lower, sparse_upper = np.quantile(sparse_local, (0.02, 0.98), axis=0)
    lower = 0.80 * dense_lower + 0.20 * sparse_lower
    upper = 0.80 * dense_upper + 0.20 * sparse_upper
    extent = upper - lower
    expansion = np.maximum(
        extent * float(PREDECLARED_THRESHOLDS["initial_bounds_expansion_fraction"]),
        camera_radius * 0.005,
    )
    lower -= expansion; upper += expansion
    if np.any(upper <= lower) or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise RuntimeError("calibrated DR/sparse initializer produced invalid bounds")
    report = {
        "method": (
            "trusted per-view calibrated DR dense support blended 80/20 with robust "
            "focus-local Tao COLMAP sparse range"
        ),
        "old_proposal_used": False,
        "dense_support_point_count": int(dense.shape[0]),
        "sparse_focus_point_count": int(sparse_focus.shape[0]),
        "focus_radius_limit": radius_limit,
        "dense_local_quantile_lower": dense_lower.tolist(),
        "dense_local_quantile_upper": dense_upper.tolist(),
        "sparse_local_quantile_lower": sparse_lower.tolist(),
        "sparse_local_quantile_upper": sparse_upper.tolist(),
        "initial_lower": lower.tolist(), "initial_upper": upper.tolist(),
        "per_view": per_view,
    }
    return lower, upper, report


def soft_mask_from_projection(hard: np.ndarray, feather_pixels: float) -> np.ndarray:
    hard = np.asarray(hard, dtype=np.uint8)
    if hard.ndim != 2 or not np.any(hard) or np.all(hard):
        raise ValueError("projection hard mask must contain interior and exterior")
    inside = cv2.distanceTransform(hard, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - hard, cv2.DIST_L2, 5)
    signed = inside - outside
    soft = np.clip(0.5 + signed / (2.0 * float(feather_pixels)), 0.0, 1.0)
    return np.rint(soft * 255.0).astype(np.uint8)


def silhouette_boundary(hard: np.ndarray) -> np.ndarray:
    hard = np.asarray(hard, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    return (cv2.morphologyEx(hard, cv2.MORPH_GRADIENT, kernel) > 0).astype(np.uint8) * 255


def projected_geometry_overlay(
    rgb: np.ndarray,
    vertices: np.ndarray,
    view: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = np.asarray(rgb, dtype=np.uint8).copy()
    uv, depth = project_vertices(
        vertices, view["source_intrinsics"], view["world_to_camera"], view["camera_center"]
    )
    if np.any(depth <= 0) or not np.isfinite(uv).all():
        raise RuntimeError(f"cuboid is not projectable in {view['stem']}")
    for start, end in CUBOID_EDGES:
        cv2.line(
            result, tuple(np.rint(uv[start]).astype(int)), tuple(np.rint(uv[end]).astype(int)),
            (255, 40, 230), 4, cv2.LINE_AA,
        )
    for index, point in enumerate(uv):
        cv2.circle(result, tuple(np.rint(point).astype(int)), 7, (40, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(result, str(index), tuple(np.rint(point + 8).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 40), 2, cv2.LINE_AA)
    hull, _ = projected_hull(
        vertices, view["source_intrinsics"], view["world_to_camera"], view["camera_center"]
    )
    cv2.polylines(result, [np.rint(hull).astype(np.int32)], True, (40, 255, 80), 5, cv2.LINE_AA)
    return result, uv, hull


def tint_mask(rgb: np.ndarray, hard: np.ndarray, color=(30, 255, 90), alpha=0.38) -> np.ndarray:
    result = np.asarray(rgb, dtype=np.float32).copy()
    selection = np.asarray(hard, dtype=bool)
    result[selection] = (1.0 - alpha) * result[selection] + alpha * np.asarray(color)
    boundary = silhouette_boundary(selection.astype(np.uint8)) > 0
    result[boundary] = np.array([255, 30, 230])
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def old_new_difference(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    old = np.asarray(old, dtype=bool); new = np.asarray(new, dtype=bool)
    result = np.zeros(old.shape + (3,), dtype=np.uint8)
    result[old & new] = (55, 165, 70)
    result[new & ~old] = (40, 110, 255)
    result[old & ~new] = (255, 55, 55)
    return result


def face_visibility(vertices: np.ndarray, camera_center: np.ndarray) -> list[dict[str, Any]]:
    center = np.asarray(camera_center)
    rows = []
    for face_id, indices in enumerate(CUBOID_FACES):
        face = vertices[indices]
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        normal /= np.linalg.norm(normal)
        face_center = face.mean(axis=0)
        facing = float(np.dot(normal, center - face_center))
        rows.append({
            "face_id": face_id, "vertex_indices": indices.tolist(),
            "camera_facing": bool(facing > 0), "signed_facing": facing,
        })
    return rows


def aggregate_quality_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    area = values("new_area_ratio")
    edge = values("boundary_alignment")
    normal = values("normal_face_alignment")
    depth = values("depth_relative_residual")
    confidence = values("confidence")
    log_area_delta = np.abs(np.diff(np.log(np.maximum(area, 1e-8))))
    corner_delta = []
    for previous, current in zip(rows[:-1], rows[1:]):
        p = np.asarray(previous["projected_vertices"], dtype=np.float64)
        c = np.asarray(current["projected_vertices"], dtype=np.float64)
        diagonal = np.hypot(current["source_size"][0], current["source_size"][1])
        corner_delta.append(float(np.mean(np.linalg.norm(c - p, axis=1)) / diagonal))
    return {
        "area_ratio": _distribution(area),
        "boundary_alignment": _distribution(edge),
        "normal_face_alignment": _distribution(normal),
        "calibrated_depth_relative_residual": _distribution(depth),
        "confidence": _distribution(confidence),
        "adjacent_log_area_delta_p95": float(np.quantile(log_area_delta, 0.95)),
        "adjacent_corner_displacement_p95": float(np.quantile(corner_delta, 0.95)),
        "border_touching_view_count": int(sum(row["border_touching"] for row in rows)),
        "low_confidence_view_count": int(sum(row["low_confidence"] for row in rows)),
    }


def _distribution(value: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(value.min()), "mean": float(value.mean()),
        "median": float(np.median(value)), "p95": float(np.quantile(value, 0.95)),
        "maximum": float(value.max()),
    }


def confidence_and_risks(row: dict[str, Any]) -> tuple[float, list[str]]:
    edge = float(row["boundary_alignment"])
    normal = float(row["normal_face_alignment"])
    depth = float(row["depth_relative_residual"])
    depth_weight = float(row["depth_weight"])
    area = float(row["new_area_ratio"])
    confidence = (
        0.45 * np.clip(edge, 0, 1) + 0.25 * np.clip(normal, 0, 1)
        + 0.20 * depth_weight * np.clip(1.0 - depth, 0, 1)
        + 0.10 * (not row["border_touching"])
    )
    risks = []
    if edge < float(PREDECLARED_THRESHOLDS["minimum_view_boundary_alignment"]):
        risks.append("weak_dr_boundary_support")
    if depth_weight <= 0:
        risks.append("untrusted_depth_calibration")
    elif depth > float(PREDECLARED_THRESHOLDS["maximum_depth_relative_residual_median"]):
        risks.append("calibrated_depth_disagreement")
    if row["border_touching"]:
        risks.append("projected_silhouette_touches_border")
    if area < float(PREDECLARED_THRESHOLDS["minimum_projected_area_ratio"]):
        risks.append("projected_area_too_small")
    if area > float(PREDECLARED_THRESHOLDS["maximum_projected_area_ratio"]):
        risks.append("projected_area_background_or_near_full_risk")
    if abs(float(row["old_new_area_change"])) > 0.20:
        risks.append("large_old_new_area_change_review_only")
    return float(np.clip(confidence, 0.0, 1.0)), risks


def _panel(image: Image.Image, title: str, size=(352, 248)) -> Image.Image:
    canvas = Image.new("RGB", (size[0], size[1] + 28), "white")
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, 28 + (size[1] - fitted.height) // 2))
    ImageDraw.Draw(canvas).text((8, 7), title, fill="black")
    return canvas


def make_review_page(
    *,
    stem: str,
    source_rgb: np.ndarray,
    dr_normal: np.ndarray,
    dr_depth: np.ndarray,
    dr_basecolor: np.ndarray,
    fused_boundary: np.ndarray,
    geometry_overlay: np.ndarray,
    new_overlay: np.ndarray,
    old_overlay: np.ndarray,
    difference: np.ndarray,
    metric: dict[str, Any],
) -> Image.Image:
    fused_rgb = np.repeat(np.rint(np.clip(fused_boundary, 0, 1)[..., None] * 255).astype(np.uint8), 3, axis=2)
    lines = [
        f"{stem}  confidence={metric['confidence']:.3f}",
        f"area old/new={metric['old_area_ratio']:.4f}/{metric['new_area_ratio']:.4f}",
        f"boundary={metric['boundary_alignment']:.3f} normal={metric['normal_face_alignment']:.3f}",
        f"depth R2={metric['depth_calibration_r2']:.3f} residual={metric['depth_relative_residual']:.3f}",
        "risks=" + (", ".join(metric["risk_flags"]) if metric["risk_flags"] else "none"),
        "review-only; not training eligible; one fixed 3-D cuboid",
    ]
    text_panel = Image.new("RGB", (352, 248), "white")
    draw = ImageDraw.Draw(text_panel)
    for index, line in enumerate(lines):
        draw.text((10, 12 + 31 * index), line, fill="black")
    items = [
        _panel(Image.fromarray(source_rgb), "source RGB"),
        _panel(Image.fromarray(dr_normal), "DR normal"),
        _panel(Image.fromarray(dr_depth), "DR relative depth"),
        _panel(Image.fromarray(dr_basecolor), "DR basecolor"),
        _panel(Image.fromarray(fused_rgb), "fused boundary likelihood"),
        _panel(Image.fromarray(geometry_overlay), "projected cuboid vertices/edges"),
        _panel(Image.fromarray(new_overlay), "new projected mask overlay"),
        _panel(Image.fromarray(old_overlay), "old proposal overlay (comparison only)"),
        _panel(Image.fromarray(difference), "old/new difference: red old, blue new"),
        _panel(text_panel, "confidence / risk explanation"),
    ]
    width, height = items[0].size
    page = Image.new("RGB", (5 * width, 2 * height), "white")
    for index, item in enumerate(items):
        page.paste(item, ((index % 5) * width, (index // 5) * height))
    return page


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "stem", "new_area_ratio", "old_area_ratio", "old_new_area_change",
        "boundary_alignment", "normal_face_alignment", "depth_calibration_mode",
        "depth_calibration_r2", "depth_relative_residual", "depth_weight",
        "confidence", "low_confidence", "border_touching", "risk_flags",
        "review_page",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = dict(row); encoded["risk_flags"] = "|".join(row["risk_flags"])
            writer.writerow(encoded)


def write_review_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["rank", "stem", "confidence", "risk_flags", "new_area_ratio",
              "old_area_ratio", "boundary_alignment", "depth_calibration_r2", "review_page"]
    ranked = sorted(rows, key=lambda row: (row["confidence"], row["stem"]))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            encoded = dict(row); encoded["rank"] = rank
            encoded["risk_flags"] = "|".join(row["risk_flags"])
            writer.writerow(encoded)


def make_contact_sheets(
    output: Path,
    rows: list[dict[str, Any]],
    *,
    fixed_stems: Sequence[str],
) -> dict[str, list[str]]:
    output = Path(output)
    contact = output / "contact_sheets"; contact.mkdir(parents=True, exist_ok=True)
    orderings = {
        "chronological": sorted(rows, key=lambda row: row["stem"]),
        "risk_ranked": sorted(rows, key=lambda row: (row["confidence"], row["stem"])),
        "largest_area_change": sorted(rows, key=lambda row: (-abs(row["old_new_area_change"]), row["stem"])),
        "weakest_dr_support": sorted(rows, key=lambda row: (row["boundary_alignment"], row["stem"])),
        "weakest_depth_calibration": sorted(rows, key=lambda row: (row["depth_calibration_r2"], row["stem"])),
        "worst_boundary_alignment": sorted(rows, key=lambda row: (row["boundary_alignment"], row["stem"])),
        "fixed_representative_views": [row for stem in fixed_stems for row in rows if row["stem"] == stem],
    }
    manifests: dict[str, list[str]] = {}
    for name, ordered in orderings.items():
        manifests[name] = []
        for page_index, offset in enumerate(range(0, len(ordered), 12), 1):
            selection = ordered[offset:offset + 12]
            sheet = Image.new("RGB", (4 * 360, 3 * 265), "white")
            for index, row in enumerate(selection):
                with Image.open(output / "overlays" / f"{row['stem']}.png") as opened:
                    tile = _panel(opened.copy(), (
                        f"{row['stem']} conf={row['confidence']:.2f} "
                        f"edge={row['boundary_alignment']:.2f} area={row['new_area_ratio']:.3f}"
                    ), size=(352, 225))
                sheet.paste(tile, ((index % 4) * 360, (index // 4) * 265))
            relative = f"contact_sheets/{name}_{page_index:02d}.png"
            atomic_image(output / relative, sheet)
            manifests[name].append(relative)
    return manifests


def tree_manifest(root: Path, *, exclude: Iterable[str] = ()) -> tuple[list[dict[str, Any]], str]:
    exclude_set = set(exclude)
    rows = []
    digest = hashlib.sha256()
    for path in sorted(value for value in Path(root).rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in exclude_set:
            continue
        value = sha256_file(path)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": value})
        digest.update(relative.encode("utf-8")); digest.update(b"\0")
        digest.update(value.encode("ascii")); digest.update(b"\n")
    return rows, digest.hexdigest()
