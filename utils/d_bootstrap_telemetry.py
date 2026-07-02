"""Bounded JSONL telemetry for an opt-in Diffuse-only bootstrap."""

from __future__ import annotations

import json
import math
from pathlib import Path


SCHEMA_NAME = "rtgs_d_bootstrap_telemetry"
SCHEMA_VERSION = 1
REQUIRED_FIELDS = (
    "schema_name",
    "schema_version",
    "telemetry_step",
    "global_iteration",
    "camera_stem",
    "phase_tag",
    "total_loss",
    "d_count",
    "d_count_delta",
    "d_topology_event",
    "whole_step_wall_ms",
    "whole_step_wall_definition",
    "cuda_memory_allocated_bytes",
    "cuda_memory_reserved_bytes",
    "cuda_max_memory_allocated_bytes",
    "cuda_max_memory_reserved_bytes",
    "cuda_peak_stats_reset_this_step",
    "cuda_peak_scope",
    "nonfinite_count",
    "unavailable_fields",
)


class DBootstrapTelemetryError(ValueError):
    pass


def validate_d_telemetry_options(path: str, max_steps: int, phase_tag: str) -> bool:
    if int(max_steps) < 0:
        raise ValueError("D bootstrap telemetry max_steps cannot be negative")
    enabled = bool(path)
    if enabled != (int(max_steps) > 0):
        raise ValueError("D bootstrap telemetry requires a path and positive max_steps")
    if enabled and not str(phase_tag).strip():
        raise ValueError("D bootstrap telemetry requires a phase tag")
    if not enabled and str(phase_tag).strip():
        raise ValueError("D bootstrap telemetry phase tag requires telemetry")
    return enabled


def validate_d_telemetry_record(record: dict) -> None:
    if set(record) != set(REQUIRED_FIELDS):
        missing = sorted(set(REQUIRED_FIELDS) - set(record))
        extra = sorted(set(record) - set(REQUIRED_FIELDS))
        raise DBootstrapTelemetryError(f"D telemetry schema mismatch: missing={missing}, extra={extra}")
    if record["schema_name"] != SCHEMA_NAME or record["schema_version"] != SCHEMA_VERSION:
        raise DBootstrapTelemetryError("unsupported D telemetry schema")
    for key in ("telemetry_step", "global_iteration", "d_count", "d_count_delta", "nonfinite_count"):
        if not isinstance(record[key], int) or isinstance(record[key], bool):
            raise DBootstrapTelemetryError(f"{key} must be an integer")
    if record["telemetry_step"] <= 0 or record["global_iteration"] <= 0 or record["d_count"] < 0:
        raise DBootstrapTelemetryError("D telemetry step/iteration/count is invalid")
    if record["nonfinite_count"] < 0 or not isinstance(record["d_topology_event"], bool):
        raise DBootstrapTelemetryError("D telemetry topology/nonfinite field is invalid")
    if not isinstance(record["cuda_peak_stats_reset_this_step"], bool):
        raise DBootstrapTelemetryError("cuda_peak_stats_reset_this_step must be boolean")
    for key in ("camera_stem", "phase_tag", "whole_step_wall_definition", "cuda_peak_scope"):
        if not isinstance(record[key], str) or not record[key]:
            raise DBootstrapTelemetryError(f"{key} must be a non-empty string")
    for key in (
        "total_loss", "whole_step_wall_ms", "cuda_memory_allocated_bytes",
        "cuda_memory_reserved_bytes", "cuda_max_memory_allocated_bytes",
        "cuda_max_memory_reserved_bytes",
    ):
        value = record[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise DBootstrapTelemetryError(f"{key} must be finite")
        if key != "total_loss" and value < 0:
            raise DBootstrapTelemetryError(f"{key} must be nonnegative")
    if record["unavailable_fields"] != []:
        raise DBootstrapTelemetryError("D bootstrap telemetry has no nullable requested fields")
    json.dumps(record, sort_keys=True, allow_nan=False)


class DBootstrapTelemetryWriter:
    def __init__(self, path: str, max_steps: int, phase_tag: str):
        validate_d_telemetry_options(path, max_steps, phase_tag)
        self.path = Path(path).expanduser().resolve()
        self.max_steps = int(max_steps)
        self.phase_tag = str(phase_tag)
        self.count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(f"refusing to append D telemetry to non-empty file: {self.path}")

    def validate_planned_steps(self, planned_steps: int) -> None:
        if planned_steps <= 0 or planned_steps > self.max_steps:
            raise ValueError(
                f"planned D telemetry steps {planned_steps} must be in [1, {self.max_steps}]"
            )

    def append(self, record: dict) -> None:
        if self.count >= self.max_steps:
            raise RuntimeError("D bootstrap telemetry max_steps exceeded")
        validate_d_telemetry_record(record)
        if record["phase_tag"] != self.phase_tag or record["telemetry_step"] != self.count + 1:
            raise DBootstrapTelemetryError("D telemetry phase/step sequence mismatch")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self.count += 1
