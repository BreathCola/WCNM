"""Review-only Tao cuboid bootstrap from COLMAP and DiffusionRenderer cues.

This module is deliberately independent from the formal Tao geometry-release
builder.  It accepts no reviewed mask and no training state.  The only fitted
target is continuous, per-view DR boundary/interior/depth evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from geometry.tao_cuboid import signed_permutation_matrices


BOOTSTRAP_SCHEMA = "rtgs_tao_review_only_dr_cuboid_bootstrap_v2"
AXIS_SEARCH_SCHEMA = "rtgs_tao_dr_normal_axis_search_review_v2"
DEPTH_CALIBRATION_SCHEMA = "rtgs_tao_relative_depth_calibration_review_v2"
EXPECTED_DR_MANIFEST_SHA256 = (
    "a3036bc7662f96f6913367bb3dcfb81a3dd351e67820065d945c74630e6a85e9"
)
EXPECTED_COLMAP_SHA256 = {
    "cameras.bin": "9c36b0b29d3e781f827420e7b2e58df9fcb7f54a58095c32e17592d32370f3f0",
    "images.bin": "7efb0c17e4b5fe4699729b7bd882652d211bb2996946d0a5b11c367ab44ebc06",
    "points3D.bin": "d2940283fcf632cdd5df4ef51542f6fe7a885d04610e599e43c2d9bf4d7c4cfe",
}

# These gates are intentionally constants.  The executable writes them to its
# staging bootstrap_plan.json before it evaluates a cuboid, and never tunes
# them from the resulting proposal.
PREDECLARED_THRESHOLDS: dict[str, Any] = {
    "expected_view_count": 112,
    "expected_first_stem": "000000",
    "expected_last_stem": "000111",
    "expected_dr_manifest_sha256": EXPECTED_DR_MANIFEST_SHA256,
    "minimum_depth_projectable_points": 100,
    "minimum_depth_calibration_points": 80,
    "minimum_depth_calibration_r2": 0.15,
    "maximum_depth_relative_residual_median": 0.40,
    "minimum_trusted_depth_views": 56,
    "minimum_trusted_depth_view_fraction": 0.50,
    "minimum_normal_alignment_p50": 0.80,
    "maximum_axis_orthogonality_error": 1e-5,
    "minimum_extent_scene_fraction": 0.025,
    "maximum_extent_scene_fraction": 1.20,
    "minimum_projected_area_ratio": 0.03,
    "maximum_projected_area_ratio": 0.55,
    "maximum_border_touching_views": 16,
    "minimum_mean_boundary_alignment": 0.30,
    "minimum_view_boundary_alignment": 0.12,
    "minimum_mean_normal_face_alignment": 0.70,
    "maximum_mean_calibrated_depth_relative_residual": 0.40,
    "maximum_adjacent_log_area_delta_p95": 0.45,
    "maximum_adjacent_corner_displacement_p95": 0.35,
    "maximum_low_confidence_views": 28,
    "low_confidence_threshold": 0.42,
    "soft_mask_feather_pixels": 7.0,
    "erode_pixels": 5,
    "near_full_image_ratio": 0.80,
    "normal_samples_per_view": 192,
    "normal_planar_selection_minimum": 0.85,
    "optimization_max_iterations": 45,
    "optimization_center_bound_in_initial_half_extent": 0.75,
    "optimization_log_half_extent_bound": 0.65,
    "initial_support_likelihood_quantile": 0.80,
    "initial_dense_grid_stride": 10,
    "initial_focus_radius_fraction": 0.85,
    "initial_bounds_expansion_fraction": 0.12,
}


def validate_tao_stems(stems: Sequence[str]) -> list[str]:
    values = list(stems)
    expected = [f"{index:06d}" for index in range(112)]
    if values != expected:
        raise ValueError("Tao input stems must be exactly 000000--000111")
    return values


def validate_dr_manifest_identity(value: str) -> str:
    if value != EXPECTED_DR_MANIFEST_SHA256:
        raise RuntimeError("DR manifest SHA-256 differs from the predeclared Tao identity")
    return value


def validate_colmap_identity(values: dict[str, str]) -> dict[str, str]:
    normalized = {str(key): str(value) for key, value in values.items()}
    if normalized != EXPECTED_COLMAP_SHA256:
        raise RuntimeError("COLMAP SHA-256 identity differs from the audited Tao inputs")
    return normalized


@dataclass(frozen=True)
class RelativeDepthCalibration:
    mode: str
    slope: float
    intercept: float
    r2: float
    projectable_point_count: int
    valid_calibration_point_count: int
    metric_residual_median: float
    metric_residual_p90: float
    metric_relative_residual_median: float
    metric_relative_residual_p90: float
    trusted: bool
    trust_weight: float
    rejection_reasons: tuple[str, ...]

    def apply(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        linear = self.slope * values + self.intercept
        if self.mode == "inverse_depth":
            result = np.full_like(linear, np.nan, dtype=np.float64)
            valid = linear > 1e-8
            result[valid] = 1.0 / linear[valid]
            return result
        return linear

    def as_dict(self) -> dict[str, Any]:
        def finite(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None
        return {
            "calibration_mode": self.mode,
            "slope": self.slope,
            "intercept": self.intercept,
            "r2": finite(self.r2),
            "projectable_point_count": self.projectable_point_count,
            "valid_calibration_point_count": self.valid_calibration_point_count,
            "metric_residual_median": finite(self.metric_residual_median),
            "metric_residual_p90": finite(self.metric_residual_p90),
            "metric_relative_residual_median": finite(self.metric_relative_residual_median),
            "metric_relative_residual_p90": finite(self.metric_relative_residual_p90),
            "trusted": self.trusted,
            "trust_weight": self.trust_weight,
            "rejection_reasons": list(self.rejection_reasons),
            "depth_contract": "per-view relative-only; affine calibration is not shared",
        }


def _irls_affine(x: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    design = np.stack((x, np.ones_like(x)), axis=1)
    weights = np.ones_like(target)
    coefficient = np.zeros(2, dtype=np.float64)
    for _ in range(12):
        root_weight = np.sqrt(weights)
        coefficient = np.linalg.lstsq(
            design * root_weight[:, None], target * root_weight, rcond=None
        )[0]
        residual = target - design @ coefficient
        centered = residual - np.median(residual)
        scale = 1.4826 * np.median(np.abs(centered)) + 1e-8
        normalized = np.abs(centered) / (1.345 * scale)
        weights = np.where(normalized <= 1.0, 1.0, 1.0 / np.maximum(normalized, 1e-8))
    prediction = design @ coefficient
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(np.sum((target - prediction) ** 2)) / max(denominator, 1e-12)
    return coefficient, r2


def fit_relative_depth_calibration(
    relative_values: np.ndarray,
    metric_depth: np.ndarray,
    *,
    projectable_point_count: int | None = None,
    maximum_samples: int = 20_000,
    thresholds: dict[str, Any] = PREDECLARED_THRESHOLDS,
) -> RelativeDepthCalibration:
    """Fit one view's relative depth to COLMAP camera-z depth.

    Direct depth and inverse-depth affine models are compared.  No parameter is
    shared between views and raw values are never interpreted as metric.
    """
    x = np.asarray(relative_values, dtype=np.float64).reshape(-1)
    z = np.asarray(metric_depth, dtype=np.float64).reshape(-1)
    if x.shape != z.shape:
        raise ValueError("relative and metric depth arrays must match")
    projectable = int(x.size if projectable_point_count is None else projectable_point_count)
    valid = np.isfinite(x) & np.isfinite(z) & (z > 1e-6)
    x, z = x[valid], z[valid]
    if x.size:
        zlo, zhi = np.quantile(z, (0.01, 0.99))
        xlo, xhi = np.quantile(x, (0.005, 0.995))
        keep = (z >= zlo) & (z <= zhi) & (x >= xlo) & (x <= xhi)
        x, z = x[keep], z[keep]
    if x.size > maximum_samples:
        indices = np.linspace(0, x.size - 1, maximum_samples, dtype=np.int64)
        x, z = x[indices], z[indices]
    minimum = int(thresholds["minimum_depth_calibration_points"])
    if x.size < max(3, minimum):
        reasons = (f"valid calibration points {x.size} < {minimum}",)
        return RelativeDepthCalibration(
            "unavailable", 0.0, 0.0, float("-inf"), projectable, int(x.size),
            float("inf"), float("inf"), float("inf"), float("inf"),
            False, 0.0, reasons,
        )

    candidates = []
    for mode, target in (("depth", z), ("inverse_depth", 1.0 / z)):
        coefficient, r2 = _irls_affine(x, target)
        linear = coefficient[0] * x + coefficient[1]
        if mode == "inverse_depth":
            predicted = np.full_like(linear, np.nan)
            positive = linear > 1e-8
            predicted[positive] = 1.0 / linear[positive]
        else:
            predicted = linear
        finite = np.isfinite(predicted) & (predicted > 1e-8)
        if np.count_nonzero(finite) < minimum:
            continue
        residual = np.abs(predicted[finite] - z[finite])
        relative = residual / np.maximum(z[finite], 1e-8)
        candidates.append({
            "mode": mode,
            "coefficient": coefficient,
            "r2": float(r2),
            "residual_median": float(np.median(residual)),
            "residual_p90": float(np.quantile(residual, 0.90)),
            "relative_median": float(np.median(relative)),
            "relative_p90": float(np.quantile(relative, 0.90)),
        })
    if not candidates:
        return RelativeDepthCalibration(
            "unavailable", 0.0, 0.0, float("-inf"), projectable, int(x.size),
            float("inf"), float("inf"), float("inf"), float("inf"),
            False, 0.0, ("both affine modes produced invalid metric predictions",),
        )
    candidates.sort(key=lambda row: (-row["r2"], row["relative_median"], row["mode"]))
    best = candidates[0]
    reasons = []
    if projectable < int(thresholds["minimum_depth_projectable_points"]):
        reasons.append("insufficient projectable sparse observations")
    if best["r2"] < float(thresholds["minimum_depth_calibration_r2"]):
        reasons.append("R2 below the predeclared trust threshold")
    if best["relative_median"] > float(
        thresholds["maximum_depth_relative_residual_median"]
    ):
        reasons.append("relative metric residual exceeds the predeclared threshold")
    if abs(float(best["coefficient"][0])) <= 1e-8:
        reasons.append("near-zero calibration slope")
    trusted = not reasons
    r2_weight = np.clip(
        (best["r2"] - float(thresholds["minimum_depth_calibration_r2"])) / 0.60,
        0.0, 1.0,
    )
    residual_weight = np.clip(
        1.0 - best["relative_median"]
        / float(thresholds["maximum_depth_relative_residual_median"]),
        0.0, 1.0,
    )
    trust_weight = float(np.sqrt(r2_weight * residual_weight)) if trusted else 0.0
    return RelativeDepthCalibration(
        str(best["mode"]), float(best["coefficient"][0]),
        float(best["coefficient"][1]), float(best["r2"]), projectable,
        int(x.size), float(best["residual_median"]), float(best["residual_p90"]),
        float(best["relative_median"]), float(best["relative_p90"]), trusted,
        trust_weight, tuple(reasons),
    )


def decode_mapped_normal(raw_rgb: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_rgb)
    mapping = np.asarray(mapping, dtype=np.float64)
    if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[-1] != 3:
        raise ValueError("DR normal must be uint8 HxWx3")
    if mapping.shape != (3, 3) or not np.allclose(mapping @ mapping.T, np.eye(3), atol=1e-8):
        raise ValueError("normal mapping must be a signed permutation")
    value = raw.astype(np.float64) / 127.5 - 1.0
    value = value @ mapping.T
    length = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(~np.isfinite(length)) or np.any(length <= 1e-6):
        raise ValueError("DR normal contains invalid vectors")
    return value / length


def _fit_axes_fast(normals: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    values = values[np.isfinite(values).all(axis=1)]
    length = np.linalg.norm(values, axis=1)
    values = values[length > 1e-6] / length[length > 1e-6, None]
    if values.shape[0] < 100:
        raise ValueError("normal axis search has fewer than 100 valid samples")
    if values.shape[0] > 25_000:
        indices = np.linspace(0, values.shape[0] - 1, 25_000, dtype=np.int64)
        values = values[indices]
    starts = np.linspace(0, values.shape[0] - 1, 24, dtype=np.int64)
    best_axes = None
    best_loss = float("inf")
    for index in starts:
        axis0 = values[index]
        dots = np.abs(values @ axis0)
        axis1 = values[int(np.argmin(dots))]
        axis1 = axis1 - axis0 * np.dot(axis1, axis0)
        norm1 = np.linalg.norm(axis1)
        if norm1 <= 1e-6:
            continue
        axis1 /= norm1
        axis2 = np.cross(axis0, axis1)
        axes = np.stack((axis0, axis1, axis2), axis=1)
        alignment = np.max(np.abs(values @ axes), axis=1)
        loss = float(np.mean(1.0 - alignment * alignment))
        if loss < best_loss:
            best_loss, best_axes = loss, axes
    if best_axes is None:
        raise RuntimeError("normal axis search could not initialize an orthogonal triad")

    def objective(rotation_vector: np.ndarray) -> float:
        axes = Rotation.from_rotvec(rotation_vector).as_matrix()
        alignment = np.max(np.abs(values @ axes), axis=1)
        return float(np.mean(1.0 - alignment * alignment))

    start = Rotation.from_matrix(best_axes).as_rotvec()
    result = minimize(
        objective, start, method="Powell",
        options={"maxiter": 80, "xtol": 1e-6, "ftol": 1e-8},
    )
    axes = Rotation.from_rotvec(result.x).as_matrix()
    alignment = np.max(np.abs(values @ axes), axis=1)
    return axes, {
        "objective": float(result.fun),
        "sample_count": int(values.shape[0]),
        "alignment_p10": float(np.quantile(alignment, 0.10)),
        "alignment_p50": float(np.quantile(alignment, 0.50)),
        "alignment_p90": float(np.quantile(alignment, 0.90)),
    }


def search_normal_axis_convention(
    samples: Sequence[dict[str, Any]],
    *,
    maximum_samples_per_view: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Jointly score all 48 camera-normal signed permutations."""
    if not samples:
        raise ValueError("normal axis search has no views")
    limit = int(maximum_samples_per_view or PREDECLARED_THRESHOLDS["normal_samples_per_view"])
    prepared = []
    for view_index, sample in enumerate(samples):
        raw = np.asarray(sample["raw_normal"])
        rotation = np.asarray(sample["world_to_camera"], dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError("world_to_camera rotation must be 3x3")
        base = raw.astype(np.float64) / 127.5 - 1.0
        length = np.linalg.norm(base, axis=-1)
        selection = np.asarray(sample.get("selection", np.ones(length.shape, bool)), dtype=bool)
        indices = np.flatnonzero(selection & np.isfinite(length) & (length > 1e-6))
        if indices.size < 16:
            raise ValueError(f"view {view_index} has insufficient selected normals")
        if indices.size > limit:
            rng = np.random.default_rng(7100 + view_index)
            indices = np.sort(rng.choice(indices, limit, replace=False))
        prepared.append((base.reshape(-1, 3)[indices], rotation))

    rows = []
    fitted_axes = []
    for index, mapping in enumerate(signed_permutation_matrices()):
        world = []
        for base, rotation in prepared:
            camera_normal = base @ mapping.T
            camera_normal /= np.linalg.norm(camera_normal, axis=1, keepdims=True)
            # COLMAP uses X_cam = R * X_world.  Row normals therefore transform
            # camera -> world as n_cam @ R.
            world.append(camera_normal @ rotation)
        axes, fit = _fit_axes_fast(np.concatenate(world, axis=0))
        score = float(
            fit["alignment_p50"] + 0.25 * fit["alignment_p10"]
            + 0.05 * fit["alignment_p90"]
        )
        rows.append({
            "mapping_index": index,
            "mapping": np.asarray(mapping).tolist(),
            "score": score,
            **fit,
        })
        fitted_axes.append(axes)
    order = sorted(range(len(rows)), key=lambda idx: (-rows[idx]["score"], idx))
    selected = order[0]
    selected_score = rows[selected]["score"]
    equivalent = [
        row["mapping_index"] for row in rows
        if abs(row["score"] - selected_score) <= 1e-7
    ]
    report = {
        "schema": AXIS_SEARCH_SCHEMA,
        "candidate_count": len(rows),
        "candidates": rows,
        "ranked_mapping_indices": [rows[idx]["mapping_index"] for idx in order],
        "selected_mapping_index": int(selected),
        "selected_mapping": rows[selected]["mapping"],
        "selected_axes": fitted_axes[selected].tolist(),
        "selected_score": selected_score,
        "selected_fit": {
            key: rows[selected][key]
            for key in ("objective", "sample_count", "alignment_p10", "alignment_p50", "alignment_p90")
        },
        "equivalent_best_mapping_indices": equivalent,
        "selection_basis": (
            "highest joint world-space orthogonal-axis alignment; exact sign-symmetric "
            "ties use the lowest deterministic signed-permutation index"
        ),
        "semantic_axis_labels": None,
    }
    return np.asarray(rows[selected]["mapping"], dtype=np.float64), fitted_axes[selected], report


def camera_focus_point(camera_centers: np.ndarray, forward_directions: np.ndarray) -> dict[str, Any]:
    centers = np.asarray(camera_centers, dtype=np.float64)
    directions = np.asarray(forward_directions, dtype=np.float64)
    if centers.shape != directions.shape or centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("camera centers/directions must be matching Nx3")
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    identity = np.eye(3)
    matrices = identity[None] - directions[:, :, None] * directions[:, None, :]
    lhs = matrices.sum(axis=0)
    rhs = np.einsum("nij,nj->i", matrices, centers)
    focus = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    distance = np.linalg.norm(np.cross(focus[None] - centers, directions), axis=1)
    radius = np.linalg.norm(centers - focus[None], axis=1)
    return {
        "focus": focus,
        "line_residual_median": float(np.median(distance)),
        "line_residual_p90": float(np.quantile(distance, 0.90)),
        "camera_radius_median": float(np.median(radius)),
        "camera_radius_minimum": float(radius.min()),
        "camera_radius_maximum": float(radius.max()),
    }


CUBOID_EDGES = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
], dtype=np.int32)

CUBOID_FACES = np.array([
    [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
], dtype=np.int32)


def cuboid_vertices(axes: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if axes.shape != (3, 3) or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        raise ValueError("cuboid axes must be orthonormal columns")
    if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
        raise ValueError("cuboid extents must be positive")
    local = np.array([
        [lower[0], lower[1], lower[2]], [upper[0], lower[1], lower[2]],
        [upper[0], upper[1], lower[2]], [lower[0], upper[1], lower[2]],
        [lower[0], lower[1], upper[2]], [upper[0], lower[1], upper[2]],
        [upper[0], upper[1], upper[2]], [lower[0], upper[1], upper[2]],
    ], dtype=np.float64)
    result = local @ axes.T
    if not np.isfinite(result).all():
        raise ValueError("cuboid vertices are non-finite")
    return result


def project_vertices(
    vertices: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    camera_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    camera = (np.asarray(vertices) - np.asarray(camera_center)[None]) @ np.asarray(world_to_camera).T
    depth = camera[:, 2]
    homogeneous = camera @ np.asarray(intrinsics).T
    uv = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:3], 1e-12)
    return uv, depth


def projected_hull(
    vertices: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    camera_center: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    uv, depth = project_vertices(vertices, intrinsics, world_to_camera, camera_center)
    if np.any(~np.isfinite(uv)) or np.any(depth <= 1e-6):
        return np.empty((0, 2), np.float64), uv
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2).astype(np.float64)
    return hull, uv


def rasterize_projected_cuboid(
    vertices: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    camera_center: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hull, uv = projected_hull(vertices, intrinsics, world_to_camera, camera_center)
    mask = np.zeros(shape, dtype=np.uint8)
    if hull.shape[0] >= 3:
        cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 1)
    return mask, hull, uv


def _bilinear(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    points = np.asarray(points, dtype=np.float64)
    if not points.size:
        return np.empty((0,) + array.shape[2:], dtype=np.float64)
    height, width = array.shape[:2]
    x = np.clip(points[:, 0], 0, width - 1)
    y = np.clip(points[:, 1], 0, height - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1); y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0; wy = y - y0
    a = array[y0, x0].astype(np.float64)
    b = array[y0, x1].astype(np.float64)
    c = array[y1, x0].astype(np.float64)
    d = array[y1, x1].astype(np.float64)
    shape = (points.shape[0],) + (1,) * (array.ndim - 2)
    wx = wx.reshape(shape); wy = wy.reshape(shape)
    return (1 - wy) * ((1 - wx) * a + wx * b) + wy * ((1 - wx) * c + wx * d)


def _boundary_samples(hull: np.ndarray, samples_per_edge: int = 24) -> np.ndarray:
    rows = []
    for index in range(hull.shape[0]):
        start = hull[index]
        end = hull[(index + 1) % hull.shape[0]]
        t = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)[:, None]
        rows.append(start[None] * (1.0 - t) + end[None] * t)
    return np.concatenate(rows, axis=0) if rows else np.empty((0, 2))


def _interior_samples(hull: np.ndarray, count: int = 9) -> np.ndarray:
    if hull.shape[0] < 3:
        return np.empty((0, 2))
    lo = hull.min(axis=0); hi = hull.max(axis=0)
    xs = np.linspace(lo[0], hi[0], count + 2)[1:-1]
    ys = np.linspace(lo[1], hi[1], count + 2)[1:-1]
    points = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    keep = np.array([
        cv2.pointPolygonTest(hull.astype(np.float32), (float(x), float(y)), False) >= 0
        for x, y in points
    ])
    return points[keep]


def _cuboid_near_camera_z(
    points: np.ndarray,
    view: dict[str, Any],
    axes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not points.size:
        return np.empty(0), np.empty(0, dtype=bool)
    inverse = np.linalg.inv(view["intrinsics"])
    pixel = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    camera_direction = pixel @ inverse.T
    world_direction = camera_direction @ view["world_to_camera"]
    local_origin = np.asarray(view["camera_center"]) @ axes
    local_direction = world_direction @ axes
    safe = np.where(np.abs(local_direction) <= 1e-12, 1.0, local_direction)
    first = (lower[None] - local_origin[None]) / safe
    second = (upper[None] - local_origin[None]) / safe
    parallel = np.abs(local_direction) <= 1e-12
    slab_near = np.where(parallel, -np.inf, np.minimum(first, second))
    slab_far = np.where(parallel, np.inf, np.maximum(first, second))
    near = slab_near.max(axis=1)
    far = slab_far.min(axis=1)
    valid = np.isfinite(near) & np.isfinite(far) & (near > 0) & (far > near)
    # camera_direction has z=1, so its unnormalized ray parameter is camera-z.
    return near, valid


def evaluate_cuboid(
    axes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    views: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    vertices = cuboid_vertices(axes, lower, upper)
    rows = []
    for view in views:
        height, width = view["fused_boundary"].shape
        hull, uv = projected_hull(
            vertices, view["intrinsics"], view["world_to_camera"], view["camera_center"]
        )
        if hull.shape[0] < 3:
            rows.append({
                "stem": view["stem"], "projectable": False,
                "area_ratio": 0.0, "boundary_alignment": 0.0,
                "interior_support": 0.0, "normal_face_alignment": 0.0,
                "depth_relative_residual": 1.0, "depth_weight": float(view["depth_weight"]),
                "border_touching": True, "projected_vertices": uv.tolist(),
                "projected_hull": [],
            })
            continue
        boundary = _boundary_samples(hull)
        interior = _interior_samples(hull)
        edge_support = float(np.mean(_bilinear(view["boundary_alignment_likelihood"], boundary)))
        interior_support = float(np.mean(_bilinear(view["interior_likelihood"], interior)))
        world_normal = _bilinear(view["world_normal"], interior)
        normal_alignment = np.max(np.abs(world_normal @ axes), axis=1)
        normal_support = float(np.mean(normal_alignment))
        predicted_near, near_valid = _cuboid_near_camera_z(interior, view, axes, lower, upper)
        calibrated = _bilinear(view["calibrated_depth"], interior)
        depth_valid = near_valid & np.isfinite(calibrated) & (calibrated > 1e-6)
        if np.any(depth_valid):
            relative_residual = np.abs(
                predicted_near[depth_valid] - calibrated[depth_valid]
            ) / np.maximum(calibrated[depth_valid], 1e-6)
            depth_residual = float(np.median(np.clip(relative_residual, 0.0, 2.0)))
        else:
            depth_residual = 1.0
        area = float(abs(cv2.contourArea(hull.astype(np.float32))) / (width * height))
        border = bool(
            np.any(hull[:, 0] <= 2) or np.any(hull[:, 0] >= width - 3)
            or np.any(hull[:, 1] <= 2) or np.any(hull[:, 1] >= height - 3)
        )
        rows.append({
            "stem": view["stem"], "projectable": True,
            "area_ratio": area, "boundary_alignment": edge_support,
            "interior_support": interior_support,
            "normal_face_alignment": normal_support,
            "depth_relative_residual": depth_residual,
            "depth_weight": float(view["depth_weight"]),
            "border_touching": border,
            "projected_vertices": uv.tolist(), "projected_hull": hull.tolist(),
        })
    projectable = [row for row in rows if row["projectable"]]
    if not projectable:
        return {"per_view": rows, "objective": float("inf"), "components": {}}
    edge = np.array([row["boundary_alignment"] for row in projectable])
    interior = np.array([row["interior_support"] for row in projectable])
    normal = np.array([row["normal_face_alignment"] for row in projectable])
    area = np.array([row["area_ratio"] for row in projectable])
    depth = np.array([row["depth_relative_residual"] for row in projectable])
    depth_weight = np.array([row["depth_weight"] for row in projectable])
    if depth_weight.sum() > 0:
        depth_mean = float(np.sum(depth * depth_weight) / depth_weight.sum())
    else:
        depth_mean = 1.0
    minimum_area = float(PREDECLARED_THRESHOLDS["minimum_projected_area_ratio"])
    maximum_area = float(PREDECLARED_THRESHOLDS["maximum_projected_area_ratio"])
    area_penalty = float(np.mean(
        np.maximum(minimum_area - area, 0.0) / minimum_area
        + np.maximum(area - maximum_area, 0.0) / maximum_area
    ))
    border_penalty = float(np.mean([row["border_touching"] for row in projectable]))
    components = {
        "boundary_loss": float(1.0 - edge.mean()),
        "interior_loss": float(1.0 - interior.mean()),
        "normal_loss": float(1.0 - normal.mean()),
        "calibrated_depth_relative_residual": depth_mean,
        "area_penalty": area_penalty,
        "border_penalty": border_penalty,
        "unprojectable_fraction": float(1.0 - len(projectable) / len(rows)),
    }
    objective = (
        1.80 * components["boundary_loss"]
        + 0.45 * components["interior_loss"]
        + 0.25 * components["normal_loss"]
        + 0.35 * components["calibrated_depth_relative_residual"]
        + 2.00 * components["area_penalty"]
        + 0.80 * components["border_penalty"]
        + 5.00 * components["unprojectable_fraction"]
    )
    return {"per_view": rows, "objective": float(objective), "components": components}


def optimize_review_cuboid(
    axes: np.ndarray,
    initial_lower: np.ndarray,
    initial_upper: np.ndarray,
    views: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Optimize one fixed cuboid against DR cues only.

    There is intentionally no mask/proposal argument.  Callers cannot inject an
    old silhouette target into this objective.
    """
    initial_lower = np.asarray(initial_lower, dtype=np.float64)
    initial_upper = np.asarray(initial_upper, dtype=np.float64)
    initial_center = (initial_lower + initial_upper) * 0.5
    initial_half = (initial_upper - initial_lower) * 0.5
    if np.any(initial_half <= 0) or not np.isfinite(initial_half).all():
        raise ValueError("initial cuboid bounds are invalid")
    center_bound = float(
        PREDECLARED_THRESHOLDS["optimization_center_bound_in_initial_half_extent"]
    )
    log_bound = float(PREDECLARED_THRESHOLDS["optimization_log_half_extent_bound"])
    bounds = [(-center_bound, center_bound)] * 3 + [(-log_bound, log_bound)] * 3

    def decode(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = initial_center + parameters[:3] * initial_half
        half = initial_half * np.exp(parameters[3:])
        return center - half, center + half

    def objective(parameters: np.ndarray) -> float:
        lower, upper = decode(parameters)
        evaluation = evaluate_cuboid(axes, lower, upper, views)
        regularizer = 0.015 * float(np.mean(np.asarray(parameters) ** 2))
        return float(evaluation["objective"] + regularizer)

    starts = [
        np.zeros(6),
        np.array([0, 0, 0, 0.15, 0.15, 0.15]),
        np.array([0, 0, 0, -0.15, -0.15, -0.15]),
        np.array([0.10, -0.10, 0.05, 0, 0, 0]),
        np.array([-0.10, 0.10, -0.05, 0, 0, 0]),
    ]
    results = []
    for start_index, start in enumerate(starts):
        result = minimize(
            objective, start, method="Powell", bounds=bounds,
            options={
                "maxiter": int(PREDECLARED_THRESHOLDS["optimization_max_iterations"]),
                "xtol": 2e-3, "ftol": 2e-5,
            },
        )
        results.append((result, start_index))
    results.sort(key=lambda item: (float(item[0].fun), item[1]))
    best, selected_start = results[0]
    lower, upper = decode(best.x)
    final = evaluate_cuboid(axes, lower, upper, views)
    report = {
        "objective_inputs": (
            "DR fused boundary likelihood + DR planar/interior likelihood + calibrated "
            "per-view DR depth + COLMAP projection; no old proposal or reviewed mask"
        ),
        "old_proposal_used_in_objective": False,
        "fallback_available": False,
        "selected_start_index": selected_start,
        "success": bool(best.success),
        "message": str(best.message),
        "parameters": np.asarray(best.x).tolist(),
        "iterations": int(best.nit),
        "function_evaluations": int(best.nfev),
        "objective": float(best.fun),
        "final_components": final["components"],
        "starts": [
            {
                "start_index": index,
                "objective": float(result.fun),
                "success": bool(result.success),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
            }
            for result, index in results
        ],
    }
    return lower, upper, {"optimization": report, **final}


def geometry_parameter_sha256(axes: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> str:
    digest = hashlib.sha256()
    for label, value in (("axes", axes), ("lower", lower), ("upper", upper)):
        array = np.ascontiguousarray(value, dtype=np.float64)
        digest.update(label.encode("ascii")); digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()
