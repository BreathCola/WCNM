"""Frozen cuboid-front path construction for transparent Stage D rays."""

from __future__ import annotations

import torch
import torch.nn.functional as F


PATH_SCHEMA = "rtgs_cuboid_front_path_v1"


def build_cuboid_front_path(
    camera_center: torch.Tensor,
    back_position: torch.Tensor,
    t_near: torch.Tensor,
    t_far: torch.Tensor,
    valid_two_hit: torch.Tensor,
    cuboid_space,
    *,
    plane_tolerance: float = 5e-4,
):
    """Build camera-ray front hits and entering-face normals without D geometry."""
    center = torch.as_tensor(camera_center)
    back = torch.as_tensor(back_position, device=center.device, dtype=center.dtype)
    near = torch.as_tensor(t_near, device=center.device, dtype=center.dtype)
    far = torch.as_tensor(t_far, device=center.device, dtype=center.dtype)
    valid = torch.as_tensor(valid_two_hit, device=center.device, dtype=torch.bool)
    if back.shape[-1] != 3 or near.shape != back.shape[:-1] \
            or far.shape != near.shape or valid.shape != near.shape:
        raise ValueError("cuboid-front cache fields have incompatible shapes")
    if center.shape != (3,):
        raise ValueError("camera center must have shape [3]")
    if not bool(valid.any()):
        raise ValueError("cuboid-front path requires at least one valid two-hit ray")
    selected_back = back[valid]
    selected_near = near[valid]
    selected_far = far[valid]
    if not torch.isfinite(center).all() or not torch.isfinite(selected_back).all() \
            or not torch.isfinite(selected_near).all() \
            or not torch.isfinite(selected_far).all() \
            or not torch.all(selected_near > 0) \
            or not torch.all(selected_far > selected_near):
        raise FloatingPointError("cuboid-front path has missing/non-finite/invalid frozen geometry")
    direction = F.normalize(selected_back - center.reshape(1, 3), dim=-1, eps=1e-8)
    back_residual = torch.abs(
        torch.linalg.vector_norm(selected_back - center.reshape(1, 3), dim=-1)
        - selected_far
    )
    if not torch.isfinite(back_residual).all() \
            or bool((back_residual > float(plane_tolerance)).any()):
        raise RuntimeError("frozen back position/t_far/camera direction are inconsistent")
    front = center.reshape(1, 3) + selected_near[:, None] * direction
    local = cuboid_space.world_to_local(front)
    lower = cuboid_space.lower.to(local)
    upper = cuboid_space.upper.to(local)
    plane_distance = torch.cat(((local - lower).abs(), (local - upper).abs()), dim=-1)
    residual, face_index = plane_distance.min(dim=-1)
    if not torch.isfinite(front).all() or not torch.isfinite(direction).all() \
            or not torch.isfinite(residual).all() \
            or bool((residual > float(plane_tolerance)).any()):
        raise RuntimeError("frozen t_near does not lie on a finite cuboid entering face")
    axes = cuboid_space.axes.to(front)
    normal = torch.empty_like(front)
    for axis in range(3):
        normal[face_index == axis] = -axes[:, axis]
        normal[face_index == axis + 3] = axes[:, axis]
    toward_camera = -direction
    normal = torch.where(
        (normal * toward_camera).sum(dim=-1, keepdim=True) < 0,
        -normal,
        normal,
    )
    normal = F.normalize(normal, dim=-1, eps=1e-8)
    if not torch.all((normal * toward_camera).sum(dim=-1) > 0):
        raise RuntimeError("cuboid entering-face normal is not camera-facing")
    return {
        "schema": PATH_SCHEMA,
        "valid": valid,
        "front_position_selected": front,
        "front_normal_selected": normal,
        "direction_selected": direction,
        "t_near_selected": selected_near[:, None],
        "t_far_selected": selected_far[:, None],
        "face_index_selected": face_index,
        "plane_residual_selected": residual[:, None],
        "back_distance_residual_selected": back_residual[:, None],
    }


def scatter_front_path(path, height: int, width: int, reference: torch.Tensor):
    """Scatter selected cuboid-front values into fail-closed HWC maps."""
    valid = path["valid"].reshape(-1)
    indices = valid.nonzero(as_tuple=False)[:, 0]

    def scatter(values, channels, fill=0.0):
        result = reference.new_full((height * width, channels), float(fill))
        result[indices] = values.to(reference)
        return result.reshape(height, width, channels)

    return {
        "front_position": scatter(path["front_position_selected"], 3),
        "front_normal": scatter(path["front_normal_selected"], 3),
        "frozen_camera_direction": scatter(path["direction_selected"], 3),
        "front_face_index": scatter(path["face_index_selected"][:, None], 1, -1.0),
        "front_plane_residual": scatter(path["plane_residual_selected"], 1),
        "frozen_back_distance_residual": scatter(
            path["back_distance_residual_selected"], 1
        ),
        "cuboid_front_valid": path["valid"][..., None].to(reference),
    }
