"""Fail-closed production CUDA raytrace interface."""

from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from raytracer.acceleration_structure import CudaLBVH
from raytracer.candidate_parameters import pack_reflection_parameters
from raytracer.differentiable_raytrace import RaytraceAux, trace_candidates


@dataclass
class RaytraceDiagnostics:
    candidate_counts: torch.Tensor
    exact_intersection_counts: torch.Tensor
    timing_ms: dict
    chunk_count: int
    reflection_surfel_count: int
    peak_memory_allocated_bytes: int
    peak_memory_delta_bytes: int
    bvh_rebuild_delta: int
    bvh_refit_delta: int


def _elapsed_cuda_ms(events):
    return float(sum(start.elapsed_time(end) for start, end in events))


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
    return_diagnostics: bool = False,
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

    device = ray_origins.device
    if return_diagnostics:
        torch.cuda.synchronize(device)
        diagnostic_wall_start = time.perf_counter()
        memory_at_start = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        stream = torch.cuda.current_stream(device)
        bvh_events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))]
        bvh_events[0][0].record(stream)
        rebuild_before = acceleration.rebuild_count
        refit_before = acceleration.refit_count
    acceleration.ensure_current(model)
    if return_diagnostics:
        bvh_events[0][1].record(stream)

    output_chunks = [[], [], [], []]
    aux_indices, aux_weights = [], []
    candidate_count_chunks, exact_count_chunks = [], []
    traversal_events, intersection_events = [], []
    candidate_parameter_table = pack_reflection_parameters(model)
    for start in range(0, ray_origins.shape[0], chunk_size):
        origins_chunk = ray_origins[start : start + chunk_size]
        directions_chunk = directions[start : start + chunk_size]
        if return_diagnostics:
            traversal_event = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            traversal_event[0].record(stream)
        candidates, offsets = acceleration.candidates(origins_chunk, directions_chunk)
        if return_diagnostics:
            traversal_event[1].record(stream)
            traversal_events.append(traversal_event)
            candidate_count_chunks.append((offsets[1:] - offsets[:-1]).detach())
            intersection_event = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            intersection_event[0].record(stream)
        result = trace_candidates(
            model,
            origins_chunk,
            directions_chunk,
            candidates,
            offsets,
            cutoff_sigma,
            hit_threshold,
            return_aux=return_aux,
            return_diagnostics=return_diagnostics,
            candidate_parameter_table=candidate_parameter_table,
        )
        if return_aux and return_diagnostics:
            outputs, aux, trace_diagnostics = result
            exact_count_chunks.append(trace_diagnostics.exact_intersection_counts)
        elif return_aux:
            outputs, aux = result
        elif return_diagnostics:
            outputs, trace_diagnostics = result
            exact_count_chunks.append(trace_diagnostics.exact_intersection_counts)
            aux = None
        else:
            outputs = result
            aux = None
        if return_diagnostics:
            intersection_event[1].record(stream)
            intersection_events.append(intersection_event)
        if return_aux:
            aux_indices.append(aux.contributing_indices)
            aux_weights.append(aux.contributing_weights)
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

    if return_aux:
        aux = RaytraceAux(
            torch.cat(aux_indices) if aux_indices else torch.empty(0, dtype=torch.long, device=device),
            torch.cat(aux_weights) if aux_weights else ray_origins.new_empty((0,)),
        )
    else:
        aux = None

    if return_diagnostics:
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device)
        diagnostics = RaytraceDiagnostics(
            candidate_counts=(
                torch.cat(candidate_count_chunks)
                if candidate_count_chunks else torch.zeros(0, dtype=torch.int64, device=device)
            ),
            exact_intersection_counts=(
                torch.cat(exact_count_chunks)
                if exact_count_chunks else torch.zeros(0, dtype=torch.int64, device=device)
            ),
            timing_ms={
                "bvh_sync_cuda": _elapsed_cuda_ms(bvh_events),
                "traversal_cuda": _elapsed_cuda_ms(traversal_events),
                "intersection_composite_cuda": _elapsed_cuda_ms(intersection_events),
                "raytrace_wall": float((time.perf_counter() - diagnostic_wall_start) * 1000.0),
            },
            chunk_count=len(traversal_events),
            reflection_surfel_count=int(model.get_xyz.shape[0]),
            peak_memory_allocated_bytes=int(peak_memory),
            peak_memory_delta_bytes=max(0, int(peak_memory - memory_at_start)),
            bvh_rebuild_delta=int(acceleration.rebuild_count - rebuild_before),
            bvh_refit_delta=int(acceleration.refit_count - refit_before),
        )
    else:
        diagnostics = None

    if return_aux and return_diagnostics:
        return outputs, aux, diagnostics
    if return_aux:
        return outputs, aux
    if return_diagnostics:
        return outputs, diagnostics
    return outputs
