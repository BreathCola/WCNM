"""Opt-in, bounded Stage B training telemetry with a strict JSONL schema.

This module never invokes the renderer, ray tracer, backward, or CUDA
traversal.  Callers may only pass values and tensors already produced by the
normal training step.  Statistics over rendered ks are computed on CPU so the
telemetry mode does not add GPU reduction kernels.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch


SCHEMA_VERSION = 1
KS_STAT_KEYS = ("min", "mean", "p50", "p95", "p99")
REQUIRED_FIELDS = (
    "schema_version",
    "telemetry_step",
    "global_iteration",
    "reflection_local_iteration",
    "camera_stem",
    "phase_tag",
    "total_loss",
    "l_spec",
    "d_count",
    "r_count",
    "d_count_delta",
    "r_count_delta",
    "d_topology_event",
    "r_topology_version",
    "r_topology_event",
    "mask_support_fraction",
    "ks_inside",
    "ks_outside",
    "ks_state_definition",
    "candidate_p50",
    "candidate_p95",
    "candidate_p99",
    "exact_p50",
    "exact_p95",
    "exact_p99",
    "raytrace_forward_ms",
    "candidate_backward_ms",
    "whole_step_wall_ms",
    "whole_step_wall_definition",
    "cuda_memory_allocated_bytes",
    "cuda_memory_reserved_bytes",
    "cuda_max_memory_allocated_bytes",
    "cuda_max_memory_reserved_bytes",
    "cuda_peak_stats_reset_this_step",
    "cuda_peak_scope",
    "nonfinite_count",
    "nonfinite_scope",
    "unavailable_fields",
)


class TelemetrySchemaError(ValueError):
    pass


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_nullable_number(record: Dict, key: str, nonnegative: bool = False) -> None:
    value = record[key]
    if value is None:
        return
    if not _is_finite_number(value):
        raise TelemetrySchemaError(f"{key} must be a finite number or null")
    if nonnegative and value < 0:
        raise TelemetrySchemaError(f"{key} must be nonnegative")


def _validate_ks_stats(value, key: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != set(KS_STAT_KEYS):
        raise TelemetrySchemaError(f"{key} must be null or contain exactly {KS_STAT_KEYS}")
    for stat_key in KS_STAT_KEYS:
        stat = value[stat_key]
        if stat is not None and not _is_finite_number(stat):
            raise TelemetrySchemaError(f"{key}.{stat_key} must be finite or null")


def validate_telemetry_record(record: Dict) -> None:
    if not isinstance(record, dict):
        raise TelemetrySchemaError("telemetry record must be a dictionary")
    if set(record) != set(REQUIRED_FIELDS):
        missing = sorted(set(REQUIRED_FIELDS) - set(record))
        extra = sorted(set(record) - set(REQUIRED_FIELDS))
        raise TelemetrySchemaError(f"telemetry schema mismatch: missing={missing}, extra={extra}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise TelemetrySchemaError("unsupported telemetry schema_version")
    for key in (
        "telemetry_step", "global_iteration", "reflection_local_iteration",
        "d_count", "r_count", "d_count_delta", "r_count_delta",
        "r_topology_version", "nonfinite_count",
    ):
        if not isinstance(record[key], int) or isinstance(record[key], bool):
            raise TelemetrySchemaError(f"{key} must be an integer")
    if record["telemetry_step"] <= 0 or record["global_iteration"] < 0:
        raise TelemetrySchemaError("telemetry/global iterations must be positive/nonnegative")
    if record["reflection_local_iteration"] < 0 or record["d_count"] < 0 or record["r_count"] < 0:
        raise TelemetrySchemaError("local iteration and field counts must be nonnegative")
    if record["r_topology_version"] < 0 or record["nonfinite_count"] < 0:
        raise TelemetrySchemaError("topology version and nonfinite_count must be nonnegative")
    for key in ("d_topology_event", "r_topology_event", "cuda_peak_stats_reset_this_step"):
        if not isinstance(record[key], bool):
            raise TelemetrySchemaError(f"{key} must be boolean")
    for key in (
        "camera_stem", "phase_tag", "ks_state_definition",
        "whole_step_wall_definition", "cuda_peak_scope", "nonfinite_scope",
    ):
        if not isinstance(record[key], str) or not record[key]:
            raise TelemetrySchemaError(f"{key} must be a non-empty string")
    for key in (
        "total_loss", "l_spec", "mask_support_fraction",
        "candidate_p50", "candidate_p95", "candidate_p99",
        "exact_p50", "exact_p95", "exact_p99", "raytrace_forward_ms",
        "candidate_backward_ms", "whole_step_wall_ms",
    ):
        _validate_nullable_number(record, key, nonnegative=key != "total_loss")
    for key in (
        "cuda_memory_allocated_bytes", "cuda_memory_reserved_bytes",
        "cuda_max_memory_allocated_bytes", "cuda_max_memory_reserved_bytes",
    ):
        _validate_nullable_number(record, key, nonnegative=True)
    _validate_ks_stats(record["ks_inside"], "ks_inside")
    _validate_ks_stats(record["ks_outside"], "ks_outside")
    if not isinstance(record["unavailable_fields"], list) or any(
        not isinstance(value, str) or not value for value in record["unavailable_fields"]
    ):
        raise TelemetrySchemaError("unavailable_fields must be a list of non-empty strings")
    if any(value not in REQUIRED_FIELDS for value in record["unavailable_fields"]):
        raise TelemetrySchemaError("unavailable_fields contains an unknown schema field")
    unavailable = set(record["unavailable_fields"])
    nullable_availability_fields = (
        "l_spec", "mask_support_fraction", "ks_inside", "ks_outside",
        "candidate_p50", "candidate_p95", "candidate_p99",
        "exact_p50", "exact_p95", "exact_p99", "raytrace_forward_ms",
        "candidate_backward_ms", "cuda_memory_allocated_bytes",
        "cuda_memory_reserved_bytes", "cuda_max_memory_allocated_bytes",
        "cuda_max_memory_reserved_bytes",
    )
    for key in nullable_availability_fields:
        if (record[key] is None) != (key in unavailable):
            raise TelemetrySchemaError(f"{key} nullability must match unavailable_fields")
    json.dumps(record, sort_keys=True, allow_nan=False)


def validate_telemetry_options(path: str, max_steps: int, phase_tag: str) -> bool:
    if int(max_steps) < 0:
        raise ValueError("Stage B telemetry max_steps cannot be negative")
    enabled = bool(path)
    if enabled != (int(max_steps) > 0):
        raise ValueError("Stage B telemetry requires both a JSONL path and positive max_steps")
    if enabled and not str(phase_tag).strip():
        raise ValueError("Stage B telemetry requires a non-empty phase tag")
    if not enabled and str(phase_tag).strip():
        raise ValueError("Stage B telemetry phase tag requires telemetry to be enabled")
    return enabled


class StageBTelemetryWriter:
    def __init__(self, path: str, max_steps: int, phase_tag: str):
        validate_telemetry_options(path, max_steps, phase_tag)
        self.path = Path(path).expanduser().resolve()
        self.max_steps = int(max_steps)
        self.phase_tag = str(phase_tag)
        self.count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(f"refusing to append Stage B telemetry to non-empty file: {self.path}")

    def validate_planned_steps(self, planned_steps: int) -> None:
        if planned_steps <= 0:
            raise ValueError("Stage B telemetry requires at least one planned step")
        if planned_steps > self.max_steps:
            raise ValueError(
                f"planned Stage B telemetry steps {planned_steps} exceed max_steps {self.max_steps}"
            )

    def append(self, record: Dict) -> None:
        if self.count >= self.max_steps:
            raise RuntimeError("Stage B telemetry max_steps exceeded")
        validate_telemetry_record(record)
        if record["phase_tag"] != self.phase_tag:
            raise TelemetrySchemaError("record phase_tag does not match telemetry writer")
        if record["telemetry_step"] != self.count + 1:
            raise TelemetrySchemaError("telemetry_step must be contiguous and one-based")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.count += 1


def _cpu_finite_stats(values: torch.Tensor) -> Dict:
    values = values.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    finite = values[torch.isfinite(values)]
    if not finite.numel():
        return {key: None for key in KS_STAT_KEYS}
    quantiles = torch.quantile(finite, torch.tensor([0.5, 0.95, 0.99]))
    return {
        "min": float(finite.min().item()),
        "mean": float(finite.mean().item()),
        "p50": float(quantiles[0].item()),
        "p95": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
    }


def mask_ks_stats_from_existing_tensors(
    surface_ks: torch.Tensor,
    soft_mask: Optional[torch.Tensor],
    valid_surface: torch.Tensor,
) -> Dict:
    """Compute Phase-B ks statistics on CPU from the existing forward tensors."""
    if soft_mask is None:
        return {
            "mask_support_fraction": None,
            "ks_inside": None,
            "ks_outside": None,
            "nonfinite_count": 0,
        }
    ks = surface_ks.detach().to(device="cpu", dtype=torch.float32)
    mask = soft_mask.detach().to(device="cpu", dtype=torch.float32)
    valid = valid_surface.detach().to(device="cpu") > 0.5
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask.permute(1, 2, 0)
    if mask.shape != ks.shape or valid.shape != ks.shape:
        raise ValueError("telemetry ks, mask, and valid-surface tensors must share HWC shape")
    inside = valid & (mask > 0.0)
    outside = valid & (mask == 0.0)
    nonfinite = int((~torch.isfinite(ks[inside | outside])).sum().item())
    return {
        "mask_support_fraction": float((mask > 0.0).float().mean().item()),
        "ks_inside": _cpu_finite_stats(ks[inside]),
        "ks_outside": _cpu_finite_stats(ks[outside]),
        "nonfinite_count": nonfinite,
    }


def cuda_allocator_snapshot(reset_this_step: bool, peak_scope: str) -> Dict:
    if not peak_scope:
        raise ValueError("allocator peak scope must be explicit")
    if not torch.cuda.is_available():
        return {
            "cuda_memory_allocated_bytes": None,
            "cuda_memory_reserved_bytes": None,
            "cuda_max_memory_allocated_bytes": None,
            "cuda_max_memory_reserved_bytes": None,
            "cuda_peak_stats_reset_this_step": bool(reset_this_step),
            "cuda_peak_scope": f"unavailable_no_cuda; requested_scope={peak_scope}",
        }
    device = torch.cuda.current_device()
    return {
        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "cuda_peak_stats_reset_this_step": bool(reset_this_step),
        "cuda_peak_scope": peak_scope,
    }
