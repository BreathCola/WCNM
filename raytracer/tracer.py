"""Fail-closed production CUDA raytrace interface."""

import torch
import torch.nn.functional as F

from raytracer.acceleration_structure import CudaLBVH
from raytracer.differentiable_raytrace import RaytraceAux, trace_candidates


def raytrace(
    model,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    acceleration: CudaLBVH = None,
    chunk_size: int = 4096,
    cutoff_sigma: float = 3.0,
    hit_threshold: float = 1e-4,
    return_alpha: bool = True,
    return_depth: bool = True,
    return_hit_mask: bool = True,
    return_aux: bool = False,
):
    if not (return_alpha and return_depth and return_hit_mask):
        raise ValueError("Stage B raytrace currently requires alpha, depth, and hit-mask outputs")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not model.get_xyz.is_cuda or not ray_origins.is_cuda or not ray_directions.is_cuda:
        raise RuntimeError("production raytrace is CUDA-only and has no brute-force fallback")
    if ray_origins.ndim != 2 or ray_origins.shape[-1] != 3 or ray_directions.shape != ray_origins.shape:
        raise ValueError("ray origins/directions must have matching shape [M,3]")
    directions = F.normalize(ray_directions, dim=-1, eps=1e-8)
    if acceleration is None:
        acceleration = CudaLBVH(cutoff_sigma=cutoff_sigma)
    if abs(acceleration.cutoff_sigma - float(cutoff_sigma)) > 1e-12:
        raise ValueError("raytrace cutoff_sigma must match the LBVH cutoff")
    acceleration.ensure_current(model)
    output_chunks = [[], [], [], []]
    aux_indices, aux_weights = [], []
    for start in range(0, ray_origins.shape[0], chunk_size):
        origins_chunk = ray_origins[start : start + chunk_size]
        directions_chunk = directions[start : start + chunk_size]
        candidates, offsets = acceleration.candidates(origins_chunk, directions_chunk)
        result = trace_candidates(
            model,
            origins_chunk,
            directions_chunk,
            candidates,
            offsets,
            cutoff_sigma,
            hit_threshold,
            return_aux=return_aux,
        )
        if return_aux:
            outputs, aux = result
            aux_indices.append(aux.contributing_indices)
            aux_weights.append(aux.contributing_weights)
        else:
            outputs = result
        for target, value in zip(output_chunks, outputs):
            target.append(value)
    if ray_origins.shape[0] == 0:
        outputs = (
            ray_origins.new_zeros((0, 3)),
            ray_origins.new_zeros((0, 1)),
            ray_origins.new_zeros((0, 1)),
            torch.zeros((0, 1), dtype=torch.bool, device=ray_origins.device),
        )
    else:
        outputs = tuple(torch.cat(values, dim=0) for values in output_chunks)
    if not return_aux:
        return outputs
    aux = RaytraceAux(
        torch.cat(aux_indices) if aux_indices else torch.empty(0, dtype=torch.long, device=ray_origins.device),
        torch.cat(aux_weights) if aux_weights else ray_origins.new_empty((0,)),
    )
    return outputs, aux
