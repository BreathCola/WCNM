#!/usr/bin/env python3
"""CPU-only final audit for the minimum Stage D real-scene smoke."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_release import sha256_file, validate_geometry_release


REQUIRED_DEBUG = {
    "final.png", "diffuse_contribution.png", "reflection_contribution.png",
    "transmittance_contribution.png", "inside_color.png", "inside_alpha.png",
    "inside_depth.png", "outside_color.png", "outside_alpha.png",
    "outside_depth.png", "transmittance_color.png", "transmittance_alpha.png",
    "depth_violation.png", "near_depth.png", "far_depth.png", "two_hit_valid.png",
}


def scan_finite(value, path="checkpoint"):
    tensors = elements = 0
    if torch.is_tensor(value):
        tensors = 1
        elements = value.numel()
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite tensor at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            child_tensors, child_elements = scan_finite(child, f"{path}.{key}")
            tensors += child_tensors; elements += child_elements
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_tensors, child_elements = scan_finite(child, f"{path}[{index}]")
            tensors += child_tensors; elements += child_elements
    return tensors, elements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--geometry-manifest", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    args = parser.parse_args()
    output, checkpoint_path = args.output.resolve(), args.checkpoint.resolve()
    audit_output = args.audit_output.resolve()
    if audit_output.exists():
        raise FileExistsError(f"refusing to overwrite Stage D audit: {audit_output}")
    release = validate_geometry_release(args.geometry_manifest)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("format") != "rtgs_stage_d":
        raise ValueError("smoke checkpoint is not rtgs_stage_d")
    if checkpoint.get("optimizer_step_completed") is not True:
        raise ValueError("smoke checkpoint lacks its endpoint optimizer step")
    if checkpoint.get("geometry_release_id") != release["geometry_release_id"]:
        raise ValueError("smoke checkpoint release ID mismatch")
    if checkpoint.get("geometry_release_aggregate_sha256") != release["aggregate_sha256"]:
        raise ValueError("smoke checkpoint release aggregate mismatch")
    tensors, elements = scan_finite(checkpoint)
    states = {name: checkpoint[name] for name in ("diffuse", "reflection", "transmittance")}
    if states["transmittance"].get("model_type") != "transmittance_surfel":
        raise ValueError("checkpoint has no independent Transmittance namespace")
    if any(states[name].get("optimizer") is None for name in states):
        raise ValueError("D/R/T optimizer state is incomplete")
    storage = [states[name]["xyz"].untyped_storage().data_ptr() for name in states]
    if len(set(storage)) != 3:
        raise ValueError("D/R/T checkpoint tensors share storage")
    counts = {name: int(states[name]["xyz"].shape[0]) for name in states}
    iteration = int(checkpoint["global_iteration"])
    for branch in states:
        ply = output / "point_cloud" / branch / f"iteration_{iteration}" / "point_cloud.ply"
        if not ply.is_file() or len(PlyData.read(ply)["vertex"].data) != counts[branch]:
            raise ValueError(f"{branch} PLY is missing or count-mismatched")
    debug = output / "debug" / f"iteration_{iteration:06d}"
    missing_debug = sorted(name for name in REQUIRED_DEBUG if not (debug / name).is_file())
    if missing_debug:
        raise ValueError(f"Stage D debug outputs are missing: {missing_debug}")
    telemetry_path = output / "stage_d_telemetry.jsonl"
    rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    if not rows or rows[-1]["global_iteration"] != iteration:
        raise ValueError("Stage D telemetry does not reach the checkpoint")
    for row in rows:
        if row.get("nonfinite_count") != 0:
            raise ValueError("Stage D telemetry reports non-finite values")
        for section in ("loss",):
            if any(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and not math.isfinite(value)
                for value in row[section].values()
            ):
                raise ValueError("Stage D telemetry contains a non-finite scalar")
    summary = json.loads((output / "stage_d_smoke_summary.json").read_text(encoding="utf-8"))
    if not summary.get("independent_parameter_storage"):
        raise ValueError("runtime D/R/T independence check failed")
    result = {
        "verdict": "STAGE_D_SMOKE_PASSED_AWAITING_REVIEW",
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "global_iteration": iteration,
        "reflection_iteration": int(checkpoint["reflection_iteration"]),
        "transmittance_iteration": int(checkpoint["transmittance_iteration"]),
        "counts": counts, "finite_tensor_count": tensors,
        "finite_element_count": elements, "telemetry_rows": len(rows),
        "last_telemetry": rows[-1], "debug_directory": str(debug),
        "geometry_release": release,
        "required_debug_count": len(REQUIRED_DEBUG), "missing_debug": [],
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_output.with_suffix(audit_output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temporary, audit_output)
    print(result["verdict"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
