"""Allocator-only Stage B memory policy and fail-closed headroom evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch


DEFAULT_POLICY = "default"
V2_POLICY = "release_ephemeral_cache_each_step_v1"
V3_RETRY_IDENTITY = "oneshot_v3_allocator_lifecycle_retry"


def validate_stage_b_memory_policy(
    policy: str,
    retry_identity: str,
    reference_peak_allocated_bytes: int,
    minimum_projected_headroom_bytes: int,
    operator_gate: bool,
) -> None:
    if policy not in (DEFAULT_POLICY, V2_POLICY):
        raise ValueError(f"unsupported Stage B allocator policy: {policy}")
    if policy == DEFAULT_POLICY:
        if retry_identity or reference_peak_allocated_bytes or minimum_projected_headroom_bytes:
            raise ValueError("default allocator policy forbids retry/headroom settings")
        return
    if not operator_gate:
        raise ValueError("allocator lifecycle retry policy is operator-only")
    if retry_identity != V3_RETRY_IDENTITY:
        raise ValueError(f"allocator lifecycle retry requires identity={V3_RETRY_IDENTITY}")
    if reference_peak_allocated_bytes <= 0 or minimum_projected_headroom_bytes <= 0:
        raise ValueError("allocator lifecycle retry requires positive reference peak and headroom")


def release_allocator_cache(policy: str) -> None:
    """Release only unused cached blocks after caller-owned tensors are deleted."""
    if policy == V2_POLICY:
        torch.cuda.empty_cache()


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
            "ephemeral tensors deleted; torch.cuda.empty_cache completed"
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
