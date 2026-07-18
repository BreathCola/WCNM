"""Tao-owned cuboid fitting and analytic two-hit intersection helpers."""

from __future__ import annotations

import itertools
from typing import Any, Sequence

import numpy as np
import torch

from geometry.dr_cuboid import fit_orthogonal_axes


GEOMETRY_SCHEMA = "rtgs_tao_cuboid_geometry_v1"
RUNTIME_INTERSECTION_SCHEMA = "rtgs_tao_runtime_cuboid_intersection_v1"
FACE_NAMES = (
    "axis0_lower", "axis0_upper", "axis1_lower",
    "axis1_upper", "axis2_lower", "axis2_upper",
)


def decode_dr_normal(raw_rgb: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_rgb)
    matrix = np.asarray(mapping, dtype=np.float64)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("DR normal must be uint8 HxWx3")
    if matrix.shape != (3, 3) or not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-8):
        raise ValueError("normal mapping must be a signed permutation matrix")
    decoded = values.astype(np.float64) / 127.5 - 1.0
    decoded = decoded @ matrix.T
    length = np.linalg.norm(decoded, axis=-1, keepdims=True)
    if np.any(~np.isfinite(length)) or np.any(length <= 1e-6):
        raise ValueError("DR normal contains non-finite/near-zero vectors")
    return (decoded / length).astype(np.float64)


def signed_permutation_matrices() -> list[np.ndarray]:
    matrices = []
    for permutation in itertools.permutations(range(3)):
        base = np.eye(3)[list(permutation)]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrices.append(np.diag(signs) @ base)
    return matrices


def select_normal_axis_convention(
    samples: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    maximum_samples_per_view: int = 2_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Select a signed camera-normal mapping by multi-view world-axis coherence.

    ``samples`` contains ``(raw_normal_rgb, camera_to_world_rotation)`` pairs.
    No semantic axis name is assumed; the selected mapping and fit score are
    returned as provenance and must pass the caller's declared threshold.
    """
    if not samples:
        raise ValueError("normal convention search has no views")
    rng = np.random.default_rng(0)
    candidates = []
    for mapping_index, mapping in enumerate(signed_permutation_matrices()):
        world_values = []
        for raw, camera_to_world, selection_mask in samples:
            decoded = decode_dr_normal(raw, mapping)
            camera_normal = (
                decoded[np.asarray(selection_mask, dtype=bool)]
                if selection_mask is not None else decoded.reshape(-1, 3)
            )
            if camera_normal.shape[0] > maximum_samples_per_view:
                selected = rng.choice(
                    camera_normal.shape[0], maximum_samples_per_view, replace=False
                )
                camera_normal = camera_normal[selected]
            world_values.append(camera_normal @ np.asarray(camera_to_world).T)
        axes, report = fit_orthogonal_axes(np.concatenate(world_values, axis=0))
        candidates.append({
            "mapping_index": mapping_index,
            "mapping": mapping,
            "axes": axes,
            "report": report,
            "score": report["alignment_p50"] + 0.25 * report["alignment_p10"],
        })
    candidates.sort(key=lambda row: (-row["score"], row["mapping_index"]))
    best = candidates[0]
    second = candidates[1]
    report = {
        "schema": "rtgs_tao_dr_normal_axis_search_v1",
        "candidate_count": len(candidates),
        "selected_mapping_index": best["mapping_index"],
        "selected_mapping": best["mapping"].tolist(),
        "selected_fit": best["report"],
        "selected_score": float(best["score"]),
        "second_score": float(second["score"]),
        "score_margin": float(best["score"] - second["score"]),
        "semantic_axis_labels": None,
    }
    return best["mapping"], best["axes"], report


def _validate_inputs(axes, lower, upper, origins, directions):
    axes = np.asarray(axes, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if axes.shape != (3, 3) or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6):
        raise ValueError("cuboid axes must be orthonormal columns")
    if lower.shape != (3,) or upper.shape != (3,) or np.any(upper <= lower):
        raise ValueError("invalid cuboid bounds")
    if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("origins/directions must be matching Nx3 arrays")
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(~np.isfinite(origins)) or np.any(~np.isfinite(directions)) \
            or np.any(lengths <= 1e-12):
        raise ValueError("cuboid rays contain non-finite/zero directions")
    return axes, lower, upper, origins, directions / lengths[:, None]


def intersect_cuboid_numpy(
    axes: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    origins: np.ndarray,
    directions: np.ndarray,
) -> dict[str, np.ndarray]:
    axes, lower, upper, origins, directions = _validate_inputs(
        axes, lower, upper, origins, directions
    )
    local_origin = origins @ axes
    local_direction = directions @ axes
    parallel = np.abs(local_direction) <= 1e-12
    parallel_outside = np.any(
        parallel & ((local_origin < lower[None]) | (local_origin > upper[None])), axis=1
    )
    safe_direction = np.where(parallel, 1.0, local_direction)
    lower_t = (lower[None] - local_origin) / safe_direction
    upper_t = (upper[None] - local_origin) / safe_direction
    axis_near = np.where(parallel, -np.inf, np.minimum(lower_t, upper_t))
    axis_far = np.where(parallel, np.inf, np.maximum(lower_t, upper_t))
    near_axis = np.argmax(axis_near, axis=1)
    far_axis = np.argmin(axis_far, axis=1)
    t_near = axis_near[np.arange(origins.shape[0]), near_axis]
    t_far = axis_far[np.arange(origins.shape[0]), far_axis]
    valid = (
        ~parallel_outside & np.isfinite(t_near) & np.isfinite(t_far)
        & (t_near > 0.0) & (t_far > t_near)
    )
    near_from_lower = local_direction[np.arange(origins.shape[0]), near_axis] > 0
    far_from_lower = local_direction[np.arange(origins.shape[0]), far_axis] < 0
    front_face_id = 2 * near_axis + (~near_from_lower).astype(np.int64)
    back_face_id = 2 * far_axis + (~far_from_lower).astype(np.int64)
    front_normal_local = np.zeros_like(origins)
    back_normal_local = np.zeros_like(origins)
    front_normal_local[np.arange(origins.shape[0]), near_axis] = np.where(
        near_from_lower, -1.0, 1.0
    )
    back_normal_local[np.arange(origins.shape[0]), far_axis] = np.where(
        far_from_lower, -1.0, 1.0
    )
    front_normal = front_normal_local @ axes.T
    back_normal = back_normal_local @ axes.T
    front_position = origins + t_near[:, None] * directions
    back_position = origins + t_far[:, None] * directions
    invalid = ~valid
    for values in (t_near, t_far):
        values[invalid] = 0.0
    for values in (front_position, back_position, front_normal, back_normal):
        values[invalid] = 0.0
    front_face_id[invalid] = -1
    back_face_id[invalid] = -1
    return {
        "t_near": t_near.astype(np.float32),
        "t_far": t_far.astype(np.float32),
        "valid_two_hit": valid,
        "front_position": front_position.astype(np.float32),
        "back_position": back_position.astype(np.float32),
        "front_normal": front_normal.astype(np.float32),
        "back_normal": back_normal.astype(np.float32),
        "front_face_id": front_face_id.astype(np.int16),
        "back_face_id": back_face_id.astype(np.int16),
        "crossing_count": np.where(valid, 2, 0).astype(np.uint8),
    }


def intersect_cuboid_torch(
    axes: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    origins: torch.Tensor,
    directions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Differentiable analytic runtime intersection used by the Tao renderer."""
    if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[-1] != 3:
        raise ValueError("origins/directions must be matching Nx3 tensors")
    if axes.shape != (3, 3) or lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("invalid torch cuboid shapes")
    direction = torch.nn.functional.normalize(directions, dim=-1, eps=1e-12)
    local_origin = origins @ axes
    local_direction = direction @ axes
    parallel = local_direction.abs() <= 1e-12
    parallel_outside = (parallel & (
        (local_origin < lower[None]) | (local_origin > upper[None])
    )).any(dim=1)
    safe = torch.where(parallel, torch.ones_like(local_direction), local_direction)
    lower_t = (lower[None] - local_origin) / safe
    upper_t = (upper[None] - local_origin) / safe
    neg_inf = torch.full_like(lower_t, -torch.inf)
    pos_inf = torch.full_like(lower_t, torch.inf)
    axis_near = torch.where(parallel, neg_inf, torch.minimum(lower_t, upper_t))
    axis_far = torch.where(parallel, pos_inf, torch.maximum(lower_t, upper_t))
    t_near, near_axis = axis_near.max(dim=1)
    t_far, far_axis = axis_far.min(dim=1)
    valid = (~parallel_outside) & torch.isfinite(t_near) & torch.isfinite(t_far) \
        & (t_near > 0) & (t_far > t_near)
    index = torch.arange(origins.shape[0], device=origins.device)
    near_lower = local_direction[index, near_axis] > 0
    far_lower = local_direction[index, far_axis] < 0
    front_face_id = 2 * near_axis + (~near_lower).long()
    back_face_id = 2 * far_axis + (~far_lower).long()
    front_local = torch.zeros_like(origins)
    back_local = torch.zeros_like(origins)
    front_local[index, near_axis] = torch.where(
        near_lower, -torch.ones_like(t_near), torch.ones_like(t_near)
    )
    back_local[index, far_axis] = torch.where(
        far_lower, -torch.ones_like(t_far), torch.ones_like(t_far)
    )
    zeros = torch.zeros_like(t_near)
    t_near_safe = torch.where(valid, t_near, zeros)
    t_far_safe = torch.where(valid, t_far, zeros)
    valid3 = valid[:, None]
    return {
        "t_near": t_near_safe,
        "t_far": t_far_safe,
        "valid_two_hit": valid,
        "front_position": torch.where(
            valid3, origins + t_near_safe[:, None] * direction, torch.zeros_like(origins)
        ),
        "back_position": torch.where(
            valid3, origins + t_far_safe[:, None] * direction, torch.zeros_like(origins)
        ),
        "front_normal": torch.where(valid3, front_local @ axes.T, torch.zeros_like(origins)),
        "back_normal": torch.where(valid3, back_local @ axes.T, torch.zeros_like(origins)),
        "front_face_id": torch.where(valid, front_face_id, torch.full_like(front_face_id, -1)),
        "back_face_id": torch.where(valid, back_face_id, torch.full_like(back_face_id, -1)),
        "crossing_count": valid.to(torch.uint8) * 2,
    }


def reflect(direction: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    if direction.shape != normal.shape or direction.shape[-1] != 3:
        raise ValueError("reflection direction/normal shapes do not match")
    result = direction - 2.0 * (direction * normal).sum(dim=-1, keepdim=True) * normal
    return torch.nn.functional.normalize(result, dim=-1, eps=1e-12)
