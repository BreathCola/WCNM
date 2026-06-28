"""Geometry helpers shared by the Stage A surfel renderer and tests."""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first quaternions ``[..., 4]`` to rotation matrices."""
    q = F.normalize(quaternion, dim=-1, eps=1e-12)
    w, x, y, z = q.unbind(dim=-1)
    matrix = torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    )
    return matrix.reshape(*q.shape[:-1], 3, 3)


def surfel_frame(
    rotation: torch.Tensor, scaling_2d: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tangent-u, tangent-v, and their normalized cross-product."""
    matrix = quaternion_to_rotation_matrix(rotation)
    tangent_u = matrix[..., :, 0]
    tangent_v = matrix[..., :, 1]
    if scaling_2d is not None:
        tangent_u = tangent_u * scaling_2d[..., 0:1]
        tangent_v = tangent_v * scaling_2d[..., 1:2]
    normal = F.normalize(torch.cross(tangent_u, tangent_v, dim=-1), dim=-1, eps=1e-12)
    return tangent_u, tangent_v, normal


def face_forward(normal: torch.Tensor, position: torch.Tensor, camera_center: torch.Tensor) -> torch.Tensor:
    """Orient normals toward the camera while preserving zero background normals."""
    view = camera_center.reshape(*([1] * (position.ndim - 1)), 3) - position
    view = F.normalize(view, dim=-1, eps=1e-12)
    flip = (normal * view).sum(dim=-1, keepdim=True) < 0
    oriented = torch.where(flip, -normal, normal)
    valid = torch.linalg.vector_norm(normal, dim=-1, keepdim=True) > 1e-8
    return torch.where(valid, F.normalize(oriented, dim=-1, eps=1e-12), torch.zeros_like(oriented))


def unproject_depth(camera, depth: torch.Tensor) -> torch.Tensor:
    """Unproject camera-z depth ``[1,H,W]`` to world positions ``[H,W,3]``."""
    if depth.ndim != 3 or depth.shape[0] != 1:
        raise ValueError(f"depth must have shape [1,H,W], got {tuple(depth.shape)}")
    device, dtype = depth.device, depth.dtype
    height, width = depth.shape[-2:]
    c2w = camera.world_view_transform.transpose(0, 1).inverse().to(device=device, dtype=dtype)
    ndc2pix = torch.tensor(
        [
            [width / 2, 0, 0, width / 2],
            [0, height / 2, 0, height / 2],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=dtype,
    ).transpose(0, 1)
    projection = c2w.transpose(0, 1) @ camera.full_proj_transform.to(device=device, dtype=dtype)
    intrinsics = (projection @ ndc2pix)[:3, :3].transpose(0, 1)
    grid_x, grid_y = torch.meshgrid(
        torch.arange(width, device=device, dtype=dtype),
        torch.arange(height, device=device, dtype=dtype),
        indexing="xy",
    )
    pixels = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1).reshape(-1, 3)
    rays = pixels @ intrinsics.inverse().transpose(0, 1) @ c2w[:3, :3].transpose(0, 1)
    origin = c2w[:3, 3]
    return (depth.reshape(-1, 1) * rays + origin).reshape(height, width, 3)


def camera_normals_to_world(camera, normal: torch.Tensor) -> torch.Tensor:
    """Transform an ``[H,W,3]`` camera-space normal prior to world space."""
    rotation = camera.world_view_transform[:3, :3].to(normal).transpose(0, 1)
    return F.normalize(normal @ rotation, dim=-1, eps=1e-12)


def alpha_composite(attributes: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Reference front-to-back compositing used to verify CUDA material maps.

    The first dimension is the depth-sorted surfel dimension. Remaining
    dimensions may describe pixels/batches, with the final attribute dimension
    broadcast-compatible with the scalar alpha channel.
    """
    if alpha.shape[-1] != 1:
        raise ValueError("alpha must have a scalar final dimension")
    if attributes.shape[0] != alpha.shape[0]:
        raise ValueError("attributes and alpha must have the same layer count")
    remaining_visibility = torch.cumprod(
        torch.cat((torch.ones_like(alpha[:1]), 1.0 - alpha[:-1]), dim=0), dim=0
    )
    weights = alpha * remaining_visibility
    return (attributes * weights).sum(dim=0)


def visualize_depth(depth: torch.Tensor, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Create a finite grayscale depth visualization in CHW layout."""
    values = depth.detach()
    valid = torch.isfinite(values) & (values > 0)
    if alpha is not None:
        valid = valid & (alpha.detach() > 1e-4)
    output = torch.zeros_like(values)
    if valid.any():
        selected = values[valid]
        lo = torch.quantile(selected, 0.02)
        hi = torch.quantile(selected, 0.98)
        normalized = (values - lo) / (hi - lo).clamp_min(1e-8)
        output = torch.where(valid, normalized.clamp(0, 1), output)
    if output.ndim == 3 and output.shape[-1] == 1:
        output = output.permute(2, 0, 1)
    return output.repeat(3, 1, 1)
