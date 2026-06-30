"""Small-scene PyTorch oracle; forbidden as a production training fallback."""

import torch
import torch.nn.functional as F

from utils.surfel_utils import quaternion_to_rotation_matrix


def _trace_chunk(model, origins, directions, cutoff_sigma, hit_threshold):
    directions = F.normalize(directions, dim=-1, eps=1e-8)
    xyz = model.get_xyz
    rotation = quaternion_to_rotation_matrix(model.get_rotation)
    tangent_u = rotation[:, :, 0]
    tangent_v = rotation[:, :, 1]
    normal = rotation[:, :, 2]
    scaling = model.get_scaling.clamp_min(1e-8)

    denominator = directions @ normal.transpose(0, 1)
    numerator = torch.einsum("mnj,nj->mn", xyz.unsqueeze(0) - origins.unsqueeze(1), normal)
    parallel = denominator.abs() <= 1e-8
    safe_denominator = torch.where(parallel, torch.ones_like(denominator), denominator)
    distance = numerator / safe_denominator
    point = origins[:, None, :] + distance[..., None] * directions[:, None, :]
    relative = point - xyz.unsqueeze(0)
    local_u = torch.einsum("mnj,nj->mn", relative, tangent_u) / scaling[:, 0].unsqueeze(0)
    local_v = torch.einsum("mnj,nj->mn", relative, tangent_v) / scaling[:, 1].unsqueeze(0)
    radius2 = local_u.square() + local_v.square()
    valid = (~parallel) & (distance > 0.0) & (radius2 <= float(cutoff_sigma) ** 2)
    opacity = model.get_opacity[:, 0].unsqueeze(0) * torch.exp(-0.5 * radius2)
    opacity = torch.where(valid, opacity.clamp(0.0, 1.0 - 1e-6), torch.zeros_like(opacity))

    sort_distance = torch.where(valid, distance, torch.full_like(distance, torch.inf))
    sorted_distance, order = torch.sort(sort_distance, dim=1)
    sorted_opacity = torch.gather(opacity, 1, order)
    colors = model.get_color.unsqueeze(0).expand(origins.shape[0], -1, -1)
    sorted_color = torch.gather(colors, 1, order[..., None].expand(-1, -1, 3))
    transmittance = torch.cumprod(
        torch.cat((torch.ones_like(sorted_opacity[:, :1]), 1.0 - sorted_opacity[:, :-1]), dim=1),
        dim=1,
    )
    weights = sorted_opacity * transmittance
    color = (weights[..., None] * sorted_color).sum(dim=1)
    alpha = weights.sum(dim=1, keepdim=True)
    finite_depth = torch.where(torch.isfinite(sorted_distance), sorted_distance, torch.zeros_like(sorted_distance))
    depth = (weights * finite_depth).sum(dim=1, keepdim=True) / alpha.clamp_min(1e-8)
    depth = torch.where(alpha > 0.0, depth, torch.zeros_like(depth))
    hit = alpha > float(hit_threshold)
    return color, alpha, depth, hit


def raytrace_bruteforce(
    model,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    cutoff_sigma: float = 3.0,
    hit_threshold: float = 1e-4,
    chunk_size: int = 1024,
):
    """Reference implementation for tests and finite-difference checks only."""
    if ray_origins.ndim != 2 or ray_origins.shape[-1] != 3:
        raise ValueError("ray_origins must have shape [M,3]")
    if ray_directions.shape != ray_origins.shape:
        raise ValueError("ray_directions must match ray_origins")
    if chunk_size <= 0 or cutoff_sigma <= 0:
        raise ValueError("chunk_size and cutoff_sigma must be positive")
    outputs = [[], [], [], []]
    for start in range(0, ray_origins.shape[0], chunk_size):
        chunk = _trace_chunk(
            model,
            ray_origins[start : start + chunk_size],
            ray_directions[start : start + chunk_size],
            cutoff_sigma,
            hit_threshold,
        )
        for target, value in zip(outputs, chunk):
            target.append(value)
    if ray_origins.shape[0] == 0:
        return (
            ray_origins.new_zeros((0, 3)),
            ray_origins.new_zeros((0, 1)),
            ray_origins.new_zeros((0, 1)),
            torch.zeros((0, 1), dtype=torch.bool, device=ray_origins.device),
        )
    return tuple(torch.cat(values, dim=0) for values in outputs)
