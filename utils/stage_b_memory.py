"""Stage B adaptive memory policy and fail-closed headroom evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch


DEFAULT_POLICY = "default"
V4_POLICY = "adaptive_pressure_cache_and_ray_retry_v1"
V4_RETRY_IDENTITY = "oneshot_v4_memory_bounded_retry"


def validate_stage_b_memory_policy(
    policy: str,
    retry_identity: str,
    reference_peak_allocated_bytes: int,
    minimum_projected_headroom_bytes: int,
    pressure_release_free_bytes: int,
    base_chunk_size: int,
    minimum_chunk_size: int,
    operator_gate: bool,
) -> None:
    if policy not in (DEFAULT_POLICY, V4_POLICY):
        raise ValueError(f"unsupported Stage B allocator policy: {policy}")
    if policy == DEFAULT_POLICY:
        if (
            retry_identity
            or reference_peak_allocated_bytes
            or minimum_projected_headroom_bytes
            or pressure_release_free_bytes
            or minimum_chunk_size
        ):
            raise ValueError("default allocator policy forbids retry/headroom settings")
        return
    if not operator_gate:
        raise ValueError("allocator lifecycle retry policy is operator-only")
    if retry_identity != V4_RETRY_IDENTITY:
        raise ValueError(f"adaptive memory retry requires identity={V4_RETRY_IDENTITY}")
    if reference_peak_allocated_bytes < 0 or minimum_projected_headroom_bytes <= 0:
        raise ValueError("adaptive memory retry requires nonnegative reference peak and positive headroom")
    if pressure_release_free_bytes <= 0:
        raise ValueError("adaptive memory retry requires a positive pressure threshold")
    ray_retry_attempts(base_chunk_size, minimum_chunk_size)


def release_allocator_cache(
    policy: str,
    *,
    force: bool = False,
    device_free_bytes: int | None = None,
    pressure_free_bytes: int = 0,
) -> bool:
    """Release unused blocks only for a forced retry or measured pressure."""
    should_release = bool(force)
    if policy == V4_POLICY and not should_release:
        if device_free_bytes is None or pressure_free_bytes <= 0:
            raise ValueError("adaptive cache release requires free-memory pressure inputs")
        should_release = int(device_free_bytes) < int(pressure_free_bytes)
    if policy == V4_POLICY and should_release:
        torch.cuda.empty_cache()
        return True
    return False


def ray_retry_attempts(base_chunk_size: int, minimum_chunk_size: int):
    """Fast path, progressively smaller chunks, then checkpointed floor."""
    base = int(base_chunk_size)
    floor = int(minimum_chunk_size)
    if base <= 0 or floor <= 0 or floor > base:
        raise ValueError("ray retry chunk sizes must satisfy base >= minimum > 0")
    attempts = []
    chunk = base
    while True:
        attempts.append((chunk, False))
        if chunk == floor:
            break
        chunk = max(floor, chunk // 2)
    attempts.append((floor, True))
    return attempts


def evaluate_headroom(
    *,
    experiment: str,
    retry_identity: str,
    policy: str,
    global_iteration: int,
    reflection_local_iteration: int,
    current_allocated_bytes: int,
    current_reserved_bytes: int,
    device_free_bytes: int,
    device_total_bytes: int,
    current_step_peak_allocated_bytes: int,
    reference_peak_allocated_bytes: int,
    minimum_projected_headroom_bytes: int,
) -> Dict:
    required_peak = max(int(current_step_peak_allocated_bytes), int(reference_peak_allocated_bytes))
    allocator_capacity = int(current_allocated_bytes) + int(device_free_bytes)
    projected_headroom = allocator_capacity - required_peak
    passed = projected_headroom >= int(minimum_projected_headroom_bytes)
    return {
        "schema": "rtgs_stage_b_headroom_gate_v1",
        "experiment": str(experiment),
        "retry_identity": str(retry_identity),
        "allocator_policy": str(policy),
        "global_iteration": int(global_iteration),
        "reflection_local_iteration": int(reflection_local_iteration),
        "measurement_boundary": (
            "after completed optimizer/densification/checkpoint/telemetry step; "
            "ephemeral tensors deleted; pressure policy evaluated"
        ),
        "current_allocated_bytes": int(current_allocated_bytes),
        "current_reserved_bytes": int(current_reserved_bytes),
        "device_free_bytes": int(device_free_bytes),
        "device_total_bytes": int(device_total_bytes),
        "current_step_peak_allocated_bytes": int(current_step_peak_allocated_bytes),
        "reference_peak_allocated_bytes": int(reference_peak_allocated_bytes),
        "required_peak_allocated_bytes": required_peak,
        "allocator_capacity_bytes": allocator_capacity,
        "minimum_projected_headroom_bytes": int(minimum_projected_headroom_bytes),
        "projected_headroom_bytes": projected_headroom,
        "passed": bool(passed),
    }


def write_headroom_gate(path, record: Dict) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing existing headroom gate record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
