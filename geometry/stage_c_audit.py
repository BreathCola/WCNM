"""CPU helpers for the Stage C Diffuse-geometry extraction audit."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class MaskVariants:
    soft: np.ndarray
    hard: np.ndarray
    eroded: np.ndarray


def make_mask_variants(
    soft: np.ndarray, hard_threshold: float = 0.5, erode_pixels: int = 2
) -> MaskVariants:
    values = np.asarray(soft, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("soft mask must be a finite HxW array")
    if not 0.0 < hard_threshold < 1.0:
        raise ValueError("hard_threshold must lie in (0, 1)")
    if erode_pixels < 0:
        raise ValueError("erode_pixels must be non-negative")
    values = np.clip(values, 0.0, 1.0)
    hard = values >= hard_threshold
    if erode_pixels:
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(
            hard.astype(np.uint8), kernel, iterations=int(erode_pixels)
        ).astype(bool)
    else:
        eroded = hard.copy()
    if hard.any() and not eroded.any():
        raise ValueError("mask erosion removed all hard-mask support")
    return MaskVariants(values, hard, eroded)


def _fraction(values: np.ndarray, support: np.ndarray) -> float:
    count = int(np.count_nonzero(support))
    return float(np.count_nonzero(values & support) / count) if count else 0.0


def audit_view(
    depth: np.ndarray,
    alpha: np.ndarray,
    normal: np.ndarray,
    masks: MaskVariants,
    alpha_threshold: float = 1e-4,
) -> dict:
    depth = np.asarray(depth, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    normal = np.asarray(normal, dtype=np.float32)
    if depth.shape != alpha.shape or depth.shape != masks.soft.shape:
        raise ValueError("depth, alpha, and masks must share HxW shape")
    if normal.shape != depth.shape + (3,):
        raise ValueError("normal must have shape HxWx3")

    depth_valid = np.isfinite(depth) & (depth > 0)
    alpha_valid = np.isfinite(alpha) & (alpha > alpha_threshold)
    normal_norm = np.linalg.norm(np.nan_to_num(normal), axis=-1)
    normal_valid = np.isfinite(normal).all(axis=-1) & (normal_norm > 0.9) & (normal_norm < 1.1)
    valid = depth_valid & alpha_valid & normal_valid

    hard_u8 = masks.hard.astype(np.uint8)
    boundary = cv2.morphologyEx(hard_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    components, component_count = ndimage.label(valid & masks.eroded)
    component_sizes = np.bincount(components.ravel())[1:]
    valid_eroded_count = int(np.count_nonzero(valid & masks.eroded))
    largest_component_fraction = (
        float(component_sizes.max() / valid_eroded_count)
        if component_sizes.size and valid_eroded_count
        else 0.0
    )
    selected_depth = depth[valid & masks.eroded]
    if selected_depth.size:
        quantiles = np.quantile(selected_depth, [0.01, 0.5, 0.99]).tolist()
    else:
        quantiles = [None, None, None]
    return {
        "mask_soft_mean": float(masks.soft.mean()),
        "mask_hard_fraction": float(masks.hard.mean()),
        "mask_eroded_fraction": float(masks.eroded.mean()),
        "valid_hard_fraction": _fraction(valid, masks.hard),
        "valid_eroded_fraction": _fraction(valid, masks.eroded),
        "depth_valid_eroded_fraction": _fraction(depth_valid, masks.eroded),
        "alpha_valid_eroded_fraction": _fraction(alpha_valid, masks.eroded),
        "normal_valid_eroded_fraction": _fraction(normal_valid, masks.eroded),
        "boundary_valid_fraction": _fraction(valid, boundary),
        "valid_eroded_component_count": int(component_count),
        "largest_valid_component_fraction": largest_component_fraction,
        "depth_p01": quantiles[0],
        "depth_p50": quantiles[1],
        "depth_p99": quantiles[2],
    }


def global_depth_scale(depth_samples: list[np.ndarray]) -> tuple[float, float]:
    selected = [np.asarray(values, dtype=np.float32).ravel() for values in depth_samples if values.size]
    if not selected:
        raise ValueError("no valid depth samples")
    values = np.concatenate(selected)
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        raise ValueError("no finite positive depth samples")
    lo, hi = np.quantile(values, [0.01, 0.99])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        raise ValueError("invalid global depth display scale")
    return float(lo), float(hi)


def voxel_consistency(
    points_by_view: list[np.ndarray], grid_resolution: int = 128
) -> dict:
    if grid_resolution < 16:
        raise ValueError("grid_resolution must be at least 16")
    nonempty = [
        np.asarray(points, dtype=np.float32).reshape(-1, 3)
        for points in points_by_view
        if np.asarray(points).size
    ]
    if not nonempty:
        raise ValueError("no points supplied for voxel consistency")
    all_points = np.concatenate(nonempty, axis=0)
    finite = np.isfinite(all_points).all(axis=1)
    all_points = all_points[finite]
    lo = np.quantile(all_points, 0.01, axis=0)
    hi = np.quantile(all_points, 0.99, axis=0)
    extent = np.maximum(hi - lo, 1e-6)
    voxel_size = float(extent.max() / grid_resolution)
    dims = np.maximum(np.ceil(extent / voxel_size).astype(np.int64) + 1, 2)
    support = np.zeros(tuple(int(v) for v in dims), dtype=np.uint16)
    total_in_bounds = 0
    for points in nonempty:
        index = np.floor((points - lo) / voxel_size).astype(np.int64)
        inside = np.all((index >= 0) & (index < dims), axis=1)
        index = np.unique(index[inside], axis=0)
        total_in_bounds += int(index.shape[0])
        support[index[:, 0], index[:, 1], index[:, 2]] += 1
    occupied = support > 0
    supported = support >= 2
    labels, count = ndimage.label(supported, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(labels.ravel())[1:]
    supported_count = int(np.count_nonzero(supported))
    return {
        "robust_bounds_min": lo.tolist(),
        "robust_bounds_max": hi.tolist(),
        "voxel_size": voxel_size,
        "grid_dims": dims.tolist(),
        "per_view_unique_voxel_sum": total_in_bounds,
        "occupied_voxels": int(np.count_nonzero(occupied)),
        "support_ge2_fraction": float(np.count_nonzero(supported) / max(np.count_nonzero(occupied), 1)),
        "support_ge3_fraction": float(np.count_nonzero(support >= 3) / max(np.count_nonzero(occupied), 1)),
        "support_ge5_fraction": float(np.count_nonzero(support >= 5) / max(np.count_nonzero(occupied), 1)),
        "supported_component_count": int(count),
        "largest_supported_component_fraction": (
            float(sizes.max() / supported_count) if sizes.size and supported_count else 0.0
        ),
    }


def classify_mesh_audit(per_view: list[dict], voxel: dict) -> tuple[str, list[str]]:
    if not per_view:
        return "STAGE_C_MESH_AUDIT_BLOCKED", ["no audited views"]
    checks = {
        "all 111 views audited": len(per_view) == 111,
        "median eroded valid coverage >= 0.95": np.median(
            [row["valid_eroded_fraction"] for row in per_view]
        ) >= 0.95,
        "minimum eroded valid coverage >= 0.75": min(
            row["valid_eroded_fraction"] for row in per_view
        ) >= 0.75,
        "median boundary valid coverage >= 0.90": np.median(
            [row["boundary_valid_fraction"] for row in per_view]
        ) >= 0.90,
        "median largest 2D component >= 0.95": np.median(
            [row["largest_valid_component_fraction"] for row in per_view]
        ) >= 0.95,
        "multi-view voxel support >=2 covers >=0.35": voxel["support_ge2_fraction"] >= 0.35,
        "largest supported 3D component >=0.70": voxel[
            "largest_supported_component_fraction"
        ] >= 0.70,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return (
        "STAGE_C_MESH_AUDIT_PASS" if not failures else "STAGE_C_MESH_AUDIT_BLOCKED",
        failures,
    )
