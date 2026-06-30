"""Differentiable exact ray/surfel intersection over CUDA-LBVH candidates."""

from dataclasses import dataclass

import torch

from utils.surfel_utils import quaternion_to_rotation_matrix


@dataclass
class RaytraceAux:
    contributing_indices: torch.Tensor
    contributing_weights: torch.Tensor


def trace_candidates(
    model,
    origins: torch.Tensor,
    directions: torch.Tensor,
    candidates: torch.Tensor,
    offsets: torch.Tensor,
    cutoff_sigma: float,
    hit_threshold: float,
    return_aux: bool = False,
):
    ray_count = origins.shape[0]
    counts = offsets[1:] - offsets[:-1]
    if ray_count == 0 or candidates.numel() == 0:
        empty = (
            origins.new_zeros((ray_count, 3)),
            origins.new_zeros((ray_count, 1)),
            origins.new_zeros((ray_count, 1)),
            torch.zeros((ray_count, 1), dtype=torch.bool, device=origins.device),
        )
        if return_aux:
            return empty, RaytraceAux(candidates.new_empty((0,)), origins.new_empty((0,)))
        return empty

    max_count = int(counts.max().item())
    padded = torch.full((ray_count, max_count), -1, dtype=torch.long, device=origins.device)
    rows = torch.repeat_interleave(torch.arange(ray_count, device=origins.device), counts)
    starts = torch.repeat_interleave(offsets[:-1], counts)
    slots = torch.arange(candidates.numel(), device=origins.device) - starts
    padded[rows, slots] = candidates
    candidate_valid = padded >= 0
    safe = padded.clamp_min(0)

    xyz = model.get_xyz[safe]
    rotation = quaternion_to_rotation_matrix(model.get_rotation)[safe]
    tangent_u = rotation[..., :, 0]
    tangent_v = rotation[..., :, 1]
    normal = rotation[..., :, 2]
    scaling = model.get_scaling[safe].clamp_min(1e-8)
    denominator = (directions[:, None, :] * normal).sum(dim=-1)
    numerator = ((xyz - origins[:, None, :]) * normal).sum(dim=-1)
    parallel = denominator.abs() <= 1e-8
    safe_denominator = torch.where(parallel, torch.ones_like(denominator), denominator)
    distance = numerator / safe_denominator
    point = origins[:, None, :] + distance[..., None] * directions[:, None, :]
    relative = point - xyz
    local_u = (relative * tangent_u).sum(dim=-1) / scaling[..., 0]
    local_v = (relative * tangent_v).sum(dim=-1) / scaling[..., 1]
    radius2 = local_u.square() + local_v.square()
    exact_valid = candidate_valid & (~parallel) & (distance > 0.0) & (radius2 <= float(cutoff_sigma) ** 2)
    opacity = model.get_opacity[safe, 0] * torch.exp(-0.5 * radius2)
    opacity = torch.where(exact_valid, opacity.clamp(0.0, 1.0 - 1e-6), torch.zeros_like(opacity))
    sort_distance = torch.where(exact_valid, distance, torch.full_like(distance, torch.inf))
    sorted_distance, order = torch.sort(sort_distance, dim=1)
    sorted_opacity = torch.gather(opacity, 1, order)
    sorted_indices = torch.gather(padded, 1, order)
    color = model.get_color[safe]
    sorted_color = torch.gather(color, 1, order[..., None].expand(-1, -1, 3))
    transmittance = torch.cumprod(
        torch.cat((torch.ones_like(sorted_opacity[:, :1]), 1.0 - sorted_opacity[:, :-1]), dim=1),
        dim=1,
    )
    weights = sorted_opacity * transmittance
    output_color = (weights[..., None] * sorted_color).sum(dim=1)
    output_alpha = weights.sum(dim=1, keepdim=True)
    finite_distance = torch.where(torch.isfinite(sorted_distance), sorted_distance, torch.zeros_like(sorted_distance))
    output_depth = (weights * finite_distance).sum(dim=1, keepdim=True) / output_alpha.clamp_min(1e-8)
    output_depth = torch.where(output_alpha > 0.0, output_depth, torch.zeros_like(output_depth))
    output_hit = output_alpha > float(hit_threshold)
    outputs = output_color, output_alpha, output_depth, output_hit
    if not return_aux:
        return outputs
    contributing = (weights > 0.0) & (sorted_indices >= 0)
    aux = RaytraceAux(sorted_indices[contributing], weights[contributing])
    return outputs, aux
