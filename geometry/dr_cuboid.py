"""DiffusionRenderer-guided cuboid reconstruction helpers for Stage C."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class DepthCalibration:
    mode: str
    slope: float
    intercept: float
    r2: float

    def apply(self, values: np.ndarray) -> np.ndarray:
        prediction = self.slope * np.asarray(values, dtype=np.float64) + self.intercept
        if self.mode == "inverse_depth":
            return 1.0 / np.maximum(prediction, 1e-6)
        return prediction


def decode_c03_normal(raw_rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_rgb)
    if values.ndim != 3 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise ValueError("raw DR normal must be uint8 HxWx3")
    normal = values.astype(np.float32) / np.float32(127.5) - np.float32(1.0)
    normal[..., 0] *= -1.0
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(length <= 1e-6):
        raise ValueError("raw DR normal contains a near-zero vector")
    return normal / length


def fit_depth_calibration(
    dr_depth: np.ndarray,
    metric_depth: np.ndarray,
    support: np.ndarray,
    maximum_samples: int = 20_000,
) -> DepthCalibration:
    x = np.asarray(dr_depth, dtype=np.float64)[support]
    z = np.asarray(metric_depth, dtype=np.float64)[support]
    valid = np.isfinite(x) & np.isfinite(z) & (x > 0.01) & (x < 0.99) & (z > 0)
    x, z = x[valid], z[valid]
    if x.size < 100:
        raise ValueError("insufficient opaque support for DR depth calibration")
    step = max(1, x.size // int(maximum_samples))
    x, z = x[::step], z[::step]
    lo, hi = np.quantile(z, (0.02, 0.98))
    keep = (z >= lo) & (z <= hi)
    x, z = x[keep], z[keep]
    design = np.stack((x, np.ones_like(x)), axis=1)
    best = None
    for mode, target in (("depth", z), ("inverse_depth", 1.0 / z)):
        weights = np.ones_like(target)
        coefficient = np.zeros(2)
        for _ in range(8):
            weighted = design * weights[:, None]
            coefficient = np.linalg.lstsq(weighted, target * weights, rcond=None)[0]
            residual = target - design @ coefficient
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
            weights = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), 1e-8))
        prediction = design @ coefficient
        denominator = np.sum((target - target.mean()) ** 2)
        r2 = float(1.0 - np.sum((target - prediction) ** 2) / max(denominator, 1e-12))
        candidate = DepthCalibration(mode, float(coefficient[0]), float(coefficient[1]), r2)
        if best is None or candidate.r2 > best.r2:
            best = candidate
    return best


def fit_orthogonal_axes(normals: np.ndarray, maximum_samples: int = 100_000) -> tuple[np.ndarray, dict]:
    values = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    values = values[np.isfinite(values).all(axis=1)]
    length = np.linalg.norm(values, axis=1)
    values = values[length > 1e-6] / length[length > 1e-6, None]
    if values.shape[0] < 100:
        raise ValueError("insufficient DR normals for orthogonal-axis fit")
    rng = np.random.default_rng(0)
    if values.shape[0] > maximum_samples:
        values = values[rng.choice(values.shape[0], maximum_samples, replace=False)]

    def objective(rotation_vector):
        axes = Rotation.from_rotvec(rotation_vector).as_matrix()
        alignment = np.max(np.abs(values @ axes), axis=1)
        return float(np.mean(1.0 - alignment * alignment))

    starts = [np.zeros(3)]
    for _ in range(8):
        vector = rng.normal(size=3)
        vector /= np.linalg.norm(vector)
        starts.append(vector * rng.uniform(0.25, np.pi))
    result = None
    for start in starts:
        candidate = minimize(
            objective, start, method="Powell",
            options={"maxiter": 150, "xtol": 1e-7, "ftol": 1e-9},
        )
        if result is None or candidate.fun < result.fun:
            result = candidate
    axes = Rotation.from_rotvec(result.x).as_matrix()
    alignment = np.max(np.abs(values @ axes), axis=1)
    return axes.astype(np.float64), {
        "objective": float(result.fun),
        "sample_count": int(values.shape[0]),
        "alignment_p10": float(np.quantile(alignment, 0.10)),
        "alignment_p50": float(np.quantile(alignment, 0.50)),
        "alignment_p90": float(np.quantile(alignment, 0.90)),
    }


def cuboid_vertices_faces(
    axes: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
        raise ValueError("invalid cuboid bounds")
    local = np.array([
        [lower[0], lower[1], lower[2]], [upper[0], lower[1], lower[2]],
        [upper[0], upper[1], lower[2]], [lower[0], upper[1], lower[2]],
        [lower[0], lower[1], upper[2]], [upper[0], lower[1], upper[2]],
        [upper[0], upper[1], upper[2]], [lower[0], upper[1], upper[2]],
    ], dtype=np.float64)
    vertices = local @ np.asarray(axes, dtype=np.float64).T
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=np.int32)
    return vertices.astype(np.float32), faces


def projected_mask(
    vertices: np.ndarray,
    intrinsics: np.ndarray,
    rotation: np.ndarray,
    camera_center: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    camera = (np.asarray(vertices) - camera_center[None]) @ rotation
    homogeneous = camera @ intrinsics.T
    if np.any(homogeneous[:, 2] <= 0):
        return np.zeros(shape, dtype=np.uint8)
    uv = homogeneous[:, :2] / homogeneous[:, 2:3]
    polygon = cv2.convexHull(np.rint(uv).astype(np.int32))
    result = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(result, polygon, 1)
    return result


def mask_metrics(predicted: np.ndarray, target: np.ndarray) -> dict:
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    intersection = int(np.count_nonzero(predicted & target))
    union = int(np.count_nonzero(predicted | target))
    return {
        "recall": float(intersection / max(np.count_nonzero(target), 1)),
        "precision": float(intersection / max(np.count_nonzero(predicted), 1)),
        "iou": float(intersection / max(union, 1)),
    }


def intersect_cuboid_near(
    axes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origin = np.asarray(origins, dtype=np.float64) @ np.asarray(axes, dtype=np.float64)
    direction = np.asarray(directions, dtype=np.float64) @ np.asarray(axes, dtype=np.float64)
    lower = np.asarray(lower)[None]
    upper = np.asarray(upper)[None]
    parallel = np.abs(direction) <= 1e-12
    parallel_outside = np.any(parallel & ((origin < lower) | (origin > upper)), axis=1)
    safe = np.where(parallel, 1.0, direction)
    first = (lower - origin) / safe
    second = (upper - origin) / safe
    slab_near = np.where(parallel, -np.inf, np.minimum(first, second))
    slab_far = np.where(parallel, np.inf, np.maximum(first, second))
    near = np.max(slab_near, axis=1)
    far = np.min(slab_far, axis=1)
    valid = (
        ~parallel_outside & np.isfinite(near) & np.isfinite(far)
        & (far > np.maximum(near, 0.0))
    )
    return near.astype(np.float64), valid


def optimize_cuboid_bounds(
    axes: np.ndarray,
    initial_lower: np.ndarray,
    initial_upper: np.ndarray,
    views: list[dict],
) -> tuple[np.ndarray, np.ndarray, dict]:
    center = (np.asarray(initial_lower) + np.asarray(initial_upper)) * 0.5
    half = (np.asarray(initial_upper) - np.asarray(initial_lower)) * 0.5

    def decode(parameters):
        fitted_center = center + parameters[:3] * half
        fitted_half = half * np.exp(parameters[3:])
        return fitted_center - fitted_half, fitted_center + fitted_half

    def evaluate(parameters):
        lower, upper = decode(parameters)
        vertices, _ = cuboid_vertices_faces(axes, lower, upper)
        rows = []
        for view in views:
            prediction = projected_mask(
                vertices, view["intrinsics"], view["rotation"],
                view["camera_center"], view["mask"].shape,
            )
            rows.append(mask_metrics(prediction, view["mask"]))
        return rows

    def objective(parameters):
        rows = evaluate(parameters)
        recall = np.mean([row["recall"] for row in rows])
        precision = np.mean([row["precision"] for row in rows])
        iou = np.mean([row["iou"] for row in rows])
        return float(1.0 - iou + 0.15 * (1.0 - recall) + 0.05 * (1.0 - precision))

    result = minimize(
        objective, np.zeros(6), method="Powell", bounds=[(-0.35, 0.35)] * 6,
        options={"maxiter": 30, "xtol": 1e-4, "ftol": 1e-7},
    )
    lower, upper = decode(result.x)
    return lower, upper, {
        "success": bool(result.success), "objective": float(result.fun),
        "parameters": result.x.tolist(), "iterations": int(result.nit),
        "function_evaluations": int(result.nfev), "per_view": evaluate(result.x),
    }
