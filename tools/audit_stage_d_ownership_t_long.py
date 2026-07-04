#!/usr/bin/env python3
"""CPU-only audit for the bounded Stage D ownership transferred-T continuation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from plyfile import PlyData
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.cuboid_space import CuboidSpace, SUPPORT_STRICT_INSIDE
from geometry.geometry_release import sha256_file, validate_geometry_release
from stage_d_training import (
    CACHED_STEMS, FORMAL_RELEASE_SHA256, OWNERSHIP_T_LONG_BLACK_LIMIT,
    OWNERSHIP_T_LONG_CAPPED_FRACTION_LIMIT, OWNERSHIP_T_LONG_ENDPOINT,
    OWNERSHIP_T_LONG_MIN_SCALE_FACTOR, OWNERSHIP_T_LONG_NODES,
    OWNERSHIP_T_LONG_OUTPUT_NAME, OWNERSHIP_T_LONG_SATURATION_LIMIT,
    OWNERSHIP_T_LONG_SOURCE_SHA256,
)
from tools.audit_stage_d_ownership_ab import load_json, scan_tensors
from utils.stage_d_static_cache import state_sha256


REQUIRED_DEBUG = {
    "ground_truth.png", "final.png", "final_diff.png",
    "diffuse_contribution.png", "reflection_contribution.png",
    "transmittance_contribution.png", "inside_color.png", "inside_alpha.png",
    "inside_depth.png", "c_in_cond.png", "outside_color.png",
    "outside_alpha.png", "outside_depth.png", "transmittance_color.png",
    "transmittance_alpha.png", "front_position.png", "front_normal.png",
    "near_depth.png", "far_depth.png", "two_hit_valid.png",
    "cout_ownership_class_map.png", "t_support_legal.png",
    "final_t_off.png", "final_cout_off.png", "final_d_direct_off.png",
    "final_r_off.png", "inside_contribution.png", "cout_contribution.png",
    "transmittance_metadata.json",
}


def finite(value):
    if isinstance(value, dict): return all(finite(child) for child in value.values())
    if isinstance(value, (list, tuple)): return all(finite(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pilot-audit", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    errors, holds, warnings = [], [], []
    if output.name != OWNERSHIP_T_LONG_OUTPUT_NAME:
        errors.append("ownership T-long output name mismatch")
    if sha256_file(args.source) != OWNERSHIP_T_LONG_SOURCE_SHA256:
        errors.append("ownership T-long source hash mismatch")
    try:
        source_checkpoint = torch.load(args.source, map_location="cpu")
        source_branch_hashes = {
            branch: state_sha256(source_checkpoint[branch])
            for branch in ("diffuse", "reflection", "transmittance")
        }
    except Exception as exc:
        source_checkpoint = {}; source_branch_hashes = {}
        errors.append(f"ownership T-long source checkpoint unreadable: {exc}")
    pilot = load_json(args.pilot_audit, errors, "ownership A/B source audit")
    if pilot.get("technical_run_complete") is not True or pilot.get("errors") \
            or pilot.get("verdict") not in (
                "CUBOID_PATH_OWNERSHIP_PILOT_HOLD",
                "CUBOID_PATH_OWNERSHIP_PILOT_PASS",
            ):
        errors.append("ownership A/B source audit is not continuable")
    release = validate_geometry_release(args.geometry_manifest)
    if release["aggregate_sha256"] != FORMAL_RELEASE_SHA256:
        errors.append("Stage C release aggregate mismatch")
    manifest = json.loads(args.geometry_manifest.read_text(encoding="utf-8"))
    meta = json.loads(
        (Path(manifest["release_root"]) / "mesh_metadata.json").read_text(encoding="utf-8")
    )
    cuboid = CuboidSpace(
        axes=torch.tensor(meta["axes_columns"], dtype=torch.float64),
        lower=torch.tensor(meta["fitted_lower"], dtype=torch.float64),
        upper=torch.tensor(meta["fitted_upper"], dtype=torch.float64),
        interface_margin=0.05, epsilon=1e-6,
    )
    operator = load_json(
        output / "ownership_t_long_operator_record.json", errors, "operator record"
    )
    if operator.get("schema") != "rtgs_stage_d_ownership_t_long_operator_v1":
        errors.append("ownership T-long operator schema mismatch")
    for key, expected in (
        ("source_sha256_before", OWNERSHIP_T_LONG_SOURCE_SHA256),
        ("source_sha256_after", OWNERSHIP_T_LONG_SOURCE_SHA256),
        ("release_aggregate_before", FORMAL_RELEASE_SHA256),
        ("release_aggregate_after", FORMAL_RELEASE_SHA256),
    ):
        if operator.get(key) != expected:
            errors.append(f"operator {key} mismatch")
    cache = load_json(args.cache / "manifest.json", errors, "ownership v4 cache")
    if cache.get("identity", {}).get("cache_schema_version") \
            != "rtgs_stage_d_cuboid_front_cache_v4" \
            or len(cache.get("entries", [])) != 111:
        errors.append("ownership v4 cache identity/count mismatch")
    metadata = load_json(
        output / "ownership_t_long_metadata.json", errors, "ownership T-long metadata"
    )
    if metadata.get("schema") != "rtgs_stage_d_ownership_t_long_run_v1" \
            or metadata.get("start_checkpoint_sha256") != OWNERSHIP_T_LONG_SOURCE_SHA256:
        errors.append("ownership T-long metadata/source mismatch")
    before = metadata.get("phase_a_frozen_hash_before")
    after = metadata.get("phase_a_frozen_hash_after")
    if not before or before != after:
        errors.append("ownership T-long D/R frozen hash mismatch")

    checkpoints = {}
    for node in OWNERSHIP_T_LONG_NODES:
        path = output / f"chkpnt{node}.pth"
        try:
            checkpoint = torch.load(path, map_location="cpu")
            tensors, elements = scan_tensors(checkpoint, f"checkpoint.{node}")
            if checkpoint.get("global_iteration") != node \
                    or checkpoint.get("reflection_iteration") != 12000 \
                    or checkpoint.get("transmittance_iteration") != node - 15000:
                errors.append(f"checkpoint {node} iteration mismatch")
            for branch in ("diffuse", "reflection"):
                if before and state_sha256(checkpoint[branch]) != before.get(branch):
                    errors.append(f"checkpoint {node} changed frozen {branch}")
            if node == 15500:
                for branch in ("diffuse", "reflection", "transmittance"):
                    if source_branch_hashes and state_sha256(checkpoint[branch]) \
                            != source_branch_hashes[branch]:
                        errors.append(f"initial checkpoint changed source {branch}")
            t = checkpoint["transmittance"]
            if t.get("position_parameterization") != "cuboid_inside_support_sigmoid_v2" \
                    or t.get("scaling_parameterization") \
                    != "cuboid_support_uniform_cap_v1":
                errors.append(f"checkpoint {node} T parameterization mismatch")
            count = int(t["xyz"].shape[0])
            if count != 4096:
                errors.append(f"checkpoint {node} T count mismatch: {count}")
            rotation = torch.nn.functional.normalize(t["rotation"].double(), dim=-1)
            raw_scale = torch.exp(t["scaling_2d"].double())
            scaling = cuboid.constrain_support_scaling(rotation, raw_scale, sigma=3.0)
            world = cuboid.decode_inside_support_latent(
                t["xyz"].double(), rotation, scaling, sigma=3.0,
            )
            classes = cuboid.classify_support(world, rotation, scaling, sigma=3.0)
            if not bool((classes == SUPPORT_STRICT_INSIDE).all()):
                errors.append(f"checkpoint {node} T support is not strict-inside-safe")
            for branch in ("diffuse", "reflection", "transmittance"):
                ply = output / "point_cloud" / branch / f"iteration_{node}/point_cloud.ply"
                if not ply.is_file() \
                        or len(PlyData.read(ply)["vertex"].data) \
                        != int(checkpoint[branch]["xyz"].shape[0]):
                    errors.append(f"checkpoint {node} {branch} PLY missing/count mismatch")
            checkpoints[str(node)] = {
                "sha256": sha256_file(path), "tensors": tensors,
                "elements": elements, "t_count": count,
            }
        except Exception as exc:
            errors.append(f"checkpoint {node} unreadable: {exc}")

    telemetry_path = output / "stage_d_telemetry.jsonl"
    try:
        rows = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    except Exception as exc:
        rows = []; errors.append(f"telemetry unreadable: {exc}")
    if [row.get("global_iteration") for row in rows] \
            != list(range(15501, OWNERSHIP_T_LONG_ENDPOINT + 1)):
        errors.append("telemetry is not continuous for global 15,501--20,000")
    for row in rows:
        iteration = row.get("global_iteration")
        if row.get("schema") != "rtgs_stage_d_ownership_t_long_telemetry_v1" \
                or row.get("optimizer_updates_this_step") \
                != {"diffuse": 0, "reflection": 0, "transmittance": 1}:
            errors.append(f"telemetry identity/update mismatch at {iteration}"); break
        if row.get("counts", {}).get("transmittance") != 4096 \
                or row.get("topology_event", {}).get(
                    "transmittance_densify_prune_called"
                ) is not False:
            errors.append(f"T topology/count mismatch at {iteration}"); break
        if row.get("loss", {}).get("lambda_depth_enabled") is not False \
                or row.get("nonfinite_count") != 0 or not finite(row) \
                or row.get("ray_memory_policy", {}).get("oom_retry_count") != 0:
            errors.append(f"finite/depth/OOM contract failed at {iteration}"); break
        metrics = row.get("semantic_metrics", {})
        if metrics.get("t_spatial_counts") != {
            "strict_inside_safe": 4096, "interface_margin": 0,
            "strict_outside_safe": 0, "crossing_or_ambiguous": 0,
        }:
            errors.append(f"T support class drift at {iteration}"); break
        energy = metrics.get("contribution_energy", {})
        if energy.get("diffuse_contribution", 1.0) != 0.0 \
                or energy.get("reflection_contribution", 1.0) != 0.0:
            errors.append(f"transparent D/R ownership failed at {iteration}"); break
    if rows:
        tail = rows[-100:]
        saturation = sum(
            row["semantic_metrics"]["ain_saturation_fraction_ge_0_95"] for row in tail
        ) / len(tail)
        black = sum(
            row["semantic_metrics"]["high_ain_near_black_conditional_fraction"]
            for row in tail
        ) / len(tail)
        capped = sum(
            row["semantic_metrics"]["t_support_scale_cap"]["capped_count"] / 4096.0
            for row in tail
        ) / len(tail)
        minimum_factor = min(
            row["semantic_metrics"]["t_support_scale_cap"]["minimum_factor"]
            for row in tail
        )
        if saturation >= OWNERSHIP_T_LONG_SATURATION_LIMIT:
            errors.append("final 100-step Ain saturation guard failed")
        if black >= OWNERSHIP_T_LONG_BLACK_LIMIT:
            errors.append("final 100-step black-veil guard failed")
        if capped >= OWNERSHIP_T_LONG_CAPPED_FRACTION_LIMIT \
                or minimum_factor <= OWNERSHIP_T_LONG_MIN_SCALE_FACTOR:
            errors.append("final 100-step scale-cap guard failed")
        final_guard = {
            "saturation": saturation, "high_ain_near_black": black,
            "capped_fraction": capped, "minimum_scale_factor": minimum_factor,
        }
    else:
        final_guard = {}
    for node in OWNERSHIP_T_LONG_NODES:
        for stem in CACHED_STEMS:
            view = output / "debug" / f"iteration_{node:06d}" / stem
            missing = sorted(name for name in REQUIRED_DEBUG if not (view / name).is_file())
            if missing:
                errors.append(f"debug {node}/{stem} missing {missing}")
                continue
            data = load_json(
                view / "transmittance_metadata.json", errors,
                f"debug metadata {node}/{stem}",
            )
            if not finite(data) or "raw_float_statistics" not in data:
                errors.append(f"debug {node}/{stem} lacks finite raw statistics")
    holds.append("NO_INDEPENDENT_BIRD_ROI: bird-level semantic claims remain unavailable")
    warnings.append(
        "This T-only continuation retains strong D/R-off ownership isolation and is not final glass physics."
    )
    verdict = "OWNERSHIP_T_LONG_BLOCKED" if errors else "OWNERSHIP_T_LONG_HOLD"
    result = {
        "verdict": verdict, "technical_run_complete": not errors,
        "semantic_separation_claimed": False, "errors": errors,
        "holds": holds, "warnings": warnings, "checkpoints": checkpoints,
        "telemetry_rows": len(rows), "final_100_step_guard": final_guard,
        "source_sha256": OWNERSHIP_T_LONG_SOURCE_SHA256,
        "release_aggregate_sha256": FORMAL_RELEASE_SHA256,
        "final_metrics": rows[-1].get("semantic_metrics", {}) if rows else {},
    }
    (output / "ownership_t_long_cpu_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    (output / "ownership_t_long_report.md").write_text(
        "# Stage D ownership transferred-T long v1\n\n"
        f"Verdict: `{verdict}`\n\n"
        "This run updates only T from global 15,501 through 20,000 with D/R frozen, "
        "cuboid-front paths, strong D/R-off isolation, and support-safe Cout. "
        "It cannot establish final D/R/T separation or authorize Stage E.\n",
        encoding="utf-8",
    )
    print(verdict); print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
