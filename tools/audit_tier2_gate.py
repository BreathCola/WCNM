#!/usr/bin/env python3
"""CPU-only, read-only gate packet generator for Tier 2 operator runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch


ANOMALY_RE = re.compile(
    r"traceback|out of memory|\boom\b|\bnan\b|\binf\b|killed|segmentation fault|runtimeerror",
    re.IGNORECASE,
)
STAGE_A_DEBUG = {
    "ground_truth.png", "diffuse_color.png", "diffuse_depth.png", "normal.png", "ks.png", "alpha.png",
}
STAGE_B_DEBUG = {
    "ground_truth.png", "final.png", "diffuse_color.png", "normal.png", "ks.png",
    "reflection_color.png", "reflection_alpha.png", "reflection_depth.png",
    "reflection_hit_mask.png", "reflection_contribution.png",
    "reflection_contribution_vis.png", "ray_candidate_count.png",
    "ray_exact_intersection_count.png", "reflection_metadata.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_scan(value, name="root", result=None):
    if result is None:
        result = {"tensor_count": 0, "element_count": 0, "nonfinite_count": 0, "paths": []}
    if torch.is_tensor(value):
        result["tensor_count"] += 1
        result["element_count"] += value.numel()
        if value.is_floating_point() or value.is_complex():
            count = int((~torch.isfinite(value)).sum().item())
            result["nonfinite_count"] += count
            if count:
                result["paths"].append({"path": name, "count": count})
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite_scan(child, f"{name}.{key}", result)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_scan(child, f"{name}[{index}]", result)
    return result


def _percentile(values, quantile):
    values = sorted(float(value) for value in values)
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _distribution(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return None
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": max(values),
    }


def _read_jsonl(path: Path):
    records = []
    malformed_trailing_line = False
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                malformed_trailing_line = True
                break
            raise
    return records, malformed_trailing_line


def _checkpoint_counts(checkpoint):
    if checkpoint["format"] == "rtgs_stage_a":
        state = checkpoint["model_state"]
        return int(state["xyz"].shape[0]), None, None
    diffuse, reflection = checkpoint["diffuse"], checkpoint["reflection"]
    return (
        int(diffuse["xyz"].shape[0]),
        int(reflection["xyz"].shape[0]),
        int(reflection["topology_version"]),
    )


def build_gate_packet(args):
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    telemetry_path = Path(args.telemetry).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    missing = [str(path) for path in (checkpoint_path, telemetry_path, run_dir, log_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required gate inputs: {missing}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("format") not in ("rtgs_stage_a", "rtgs_stage_b"):
        raise ValueError("gate checkpoint has an unsupported format")
    stage_b = checkpoint["format"] == "rtgs_stage_b"
    checkpoint_global = int(
        checkpoint["global_iteration"] if stage_b else checkpoint["iteration"]
    )
    checkpoint_local = int(checkpoint["reflection_iteration"]) if stage_b else None
    d_count, r_count, r_topology = _checkpoint_counts(checkpoint)

    all_records, malformed_tail = _read_jsonl(telemetry_path)
    records = [
        record for record in all_records
        if args.expected_global_start <= int(record["global_iteration"]) <= args.expected_global_end
    ]
    expected_globals = list(range(args.expected_global_start, args.expected_global_end + 1))
    observed_globals = [int(record["global_iteration"]) for record in records]
    continuity = observed_globals == expected_globals
    local_continuity = None
    if stage_b:
        observed_local = [int(record["reflection_local_iteration"]) for record in records]
        expected_local = list(range(args.expected_r_local_start, args.expected_r_local_end + 1))
        local_continuity = observed_local == expected_local

    debug_root = run_dir / "debug"
    debug_iterations = sorted(
        int(path.name.split("_")[-1]) for path in debug_root.glob("iteration_*") if path.is_dir()
    ) if debug_root.is_dir() else []
    endpoint_debug = debug_root / f"iteration_{args.expected_global_end:06d}"
    required_debug = set(STAGE_B_DEBUG if stage_b else STAGE_A_DEBUG)
    if stage_b and args.expected_r_local_end >= 101:
        required_debug |= {"transparent_mask.png", "overlay.png"}
    missing_debug = sorted(name for name in required_debug if not (endpoint_debug / name).is_file())

    ordinary = [record for record in records if int(record["global_iteration"]) not in debug_iterations]
    allocator_keys = (
        "cuda_memory_allocated_bytes", "cuda_memory_reserved_bytes",
        "cuda_max_memory_allocated_bytes", "cuda_max_memory_reserved_bytes",
    )
    allocator = {
        key: {"p50": _percentile([record[key] for record in ordinary], 0.5),
              "p95": _percentile([record[key] for record in ordinary], 0.95),
              "max": max((record[key] for record in ordinary), default=None)}
        for key in allocator_keys
    }
    d_events = [
        {"global": int(record["global_iteration"]), "delta": int(record["d_count_delta"]),
         "count": int(record["d_count"])}
        for record in records if record["d_topology_event"]
    ]
    r_events = [] if not stage_b else [
        {"global": int(record["global_iteration"]), "local": int(record["reflection_local_iteration"]),
         "delta": int(record["r_count_delta"]), "version": int(record["r_topology_version"])}
        for record in records if record["r_topology_event"]
    ]

    debug_summary = None
    metadata_path = endpoint_debug / "reflection_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw = metadata["raw_stats"]
        debug_summary = {
            "candidate": {key: raw["ray_candidate_count"][key] for key in ("p50", "p95", "p99", "max")},
            "exact": {key: raw["ray_exact_intersection_count"][key] for key in ("p50", "p95", "p99", "max")},
            "raytrace_timing_ms": metadata["raytrace"]["timing_ms"],
            "reflection_contribution": {
                key: raw["reflection_contribution"][key]
                for key in ("mean", "p50", "p95", "p99", "max", "nonzero_fraction", "nonfinite_count")
            },
            "raytrace_peak_allocated_bytes": metadata["raytrace"]["peak_memory_allocated_bytes"],
        }

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    anomaly_lines = [line.strip() for line in log_text.replace("\r", "\n").splitlines() if ANOMALY_RE.search(line)]
    log_complete = "Training complete." in log_text
    if not args.allow_running_log and not log_complete:
        missing.append("training log does not contain Training complete")
    if checkpoint_global != args.expected_global_end:
        missing.append(f"checkpoint global {checkpoint_global} != expected {args.expected_global_end}")
    if stage_b and checkpoint_local != args.expected_r_local_end:
        missing.append(f"checkpoint R-local {checkpoint_local} != expected {args.expected_r_local_end}")
    if missing_debug:
        missing.append(f"endpoint debug missing: {missing_debug}")
    if stage_b:
        ply_paths = [
            run_dir / "point_cloud" / branch / f"iteration_{args.expected_global_end}" / "point_cloud.ply"
            for branch in ("diffuse", "reflection")
        ]
    elif args.require_ply:
        ply_paths = [
            run_dir / "point_cloud" / f"iteration_{args.expected_global_end}" / "point_cloud.ply"
        ]
    else:
        ply_paths = []
    missing_ply = [str(path) for path in ply_paths if not path.is_file()]
    if missing_ply:
        missing.append(f"endpoint PLY missing: {missing_ply}")

    packet = {
        "packet_schema": "rtgs_tier2_gate_packet_v1",
        "read_only": True,
        "checkpoint": {
            "path": str(checkpoint_path), "sha256": _sha256(checkpoint_path),
            "format": checkpoint["format"], "version": checkpoint.get("checkpoint_version"),
            "global": checkpoint_global, "r_local": checkpoint_local,
            "d_count": d_count, "r_count": r_count, "r_topology_version": r_topology,
            "optimizer_step_completed": checkpoint.get("optimizer_step_completed"),
            "has_rng_state": "rng_state" in checkpoint,
            "has_runtime_state": "runtime_state" in checkpoint,
            "finite_scan": _finite_scan(checkpoint),
        },
        "telemetry": {
            "path": str(telemetry_path), "selected_rows": len(records),
            "global": [observed_globals[0], observed_globals[-1]] if observed_globals else None,
            "r_local": (
                [records[0]["reflection_local_iteration"], records[-1]["reflection_local_iteration"]]
                if stage_b and records else None
            ),
            "global_continuous": continuity, "r_local_continuous": local_continuity,
            "malformed_trailing_live_line": malformed_tail,
            "nonfinite_count_sum": sum(int(record["nonfinite_count"]) for record in records),
            "loss": _distribution([record["total_loss"] for record in records]),
            "l_spec": _distribution([record.get("l_spec") for record in records]),
            "mask_support": _distribution([record.get("mask_support_fraction") for record in records]),
            "ks_inside_mean": _distribution([
                record.get("ks_inside", {}).get("mean") if record.get("ks_inside") else None
                for record in records
            ]),
            "ks_outside_mean": _distribution([
                record.get("ks_outside", {}).get("mean") if record.get("ks_outside") else None
                for record in records
            ]),
            "ordinary_wall_ms": _distribution([record["whole_step_wall_ms"] for record in ordinary]),
            "allocator": allocator, "d_topology_events": d_events, "r_topology_events": r_events,
        },
        "debug": {
            "endpoint": str(endpoint_debug), "available_iterations": debug_iterations,
            "missing_endpoint_files": missing_debug, "reflection_metadata": debug_summary,
        },
        "ply": {"required": bool(ply_paths), "paths": [str(path) for path in ply_paths], "missing": missing_ply},
        "log": {
            "path": str(log_path), "training_complete": log_complete,
            "allow_running_log": bool(args.allow_running_log), "anomaly_hits": anomaly_lines[:20],
        },
        "missing_or_mismatched": missing,
    }
    packet["packet_ready_for_codex_review"] = bool(
        not missing and not anomaly_lines and continuity and local_continuity is not False
        and packet["checkpoint"]["finite_scan"]["nonfinite_count"] == 0
        and packet["telemetry"]["nonfinite_count_sum"] == 0
    )
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--expected-global-start", required=True, type=int)
    parser.add_argument("--expected-global-end", required=True, type=int)
    parser.add_argument("--expected-r-local-start", type=int)
    parser.add_argument("--expected-r-local-end", type=int)
    parser.add_argument("--allow-running-log", action="store_true")
    parser.add_argument("--require-ply", action="store_true")
    args = parser.parse_args()
    if (args.expected_r_local_start is None) != (args.expected_r_local_end is None):
        parser.error("R-local start/end must be provided together")
    packet = build_gate_packet(args)
    print(json.dumps(packet, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
