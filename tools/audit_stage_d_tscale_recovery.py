#!/usr/bin/env python3
"""CPU-only audit for Stage D projected T-scale recovery runs."""

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
    CACHED_STEMS, FORMAL_RELEASE_SHA256, TSCALE_RECOVERY_LONG_NODES,
    TSCALE_RECOVERY_LONG_OUTPUT_NAME, TSCALE_RECOVERY_PREFLIGHT_NODES,
    TSCALE_RECOVERY_PREFLIGHT_OUTPUT_NAME, TSCALE_RECOVERY_RAW_ACTIVE_ATOL,
    TSCALE_RECOVERY_SCALE_FACTOR_LIMIT, TSCALE_RECOVERY_SOURCE_SHA256,
)
from tools.audit_stage_d_ownership_ab import load_json, scan_tensors
from utils.stage_d_static_cache import state_sha256


REQUIRED_DEBUG = {
    "ground_truth.png", "transparent_mask.png", "final.png", "final_diff.png",
    "diffuse_contribution.png", "reflection_contribution.png",
    "transmittance_contribution.png", "inside_color.png", "inside_alpha.png",
    "inside_depth.png", "c_in_cond.png", "outside_color.png", "outside_alpha.png",
    "outside_depth.png", "transmittance_color.png", "transmittance_alpha.png",
    "front_position.png", "front_normal.png", "near_depth.png", "far_depth.png",
    "two_hit_valid.png", "cout_ownership_class_map.png", "t_support_legal.png",
    "final_t_off.png", "final_cout_off.png", "final_d_direct_off.png",
    "final_r_off.png", "inside_contribution.png", "cout_contribution.png",
    "transmittance_metadata.json",
}


def finite(value):
    if isinstance(value, dict):
        return all(finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def node_metrics(output, node, errors):
    rows = []
    for stem in CACHED_STEMS:
        directory = output / "debug" / f"iteration_{node:06d}" / stem
        missing = sorted(name for name in REQUIRED_DEBUG if not (directory / name).is_file())
        if missing:
            errors.append(f"debug {node}/{stem} missing {missing}")
            continue
        data = load_json(
            directory / "transmittance_metadata.json", errors,
            f"debug metadata {node}/{stem}",
        )
        if not finite(data) or "raw_float_statistics" not in data:
            errors.append(f"debug {node}/{stem} lacks finite raw statistics")
            continue
        raw = data["raw_float_statistics"]
        rows.append({
            "camera_stem": stem,
            "ain_mean": raw["inside_alpha"]["mean"],
            "cin_energy": data["energy"]["inside_color_valid_mean"],
            "t_contribution": data["energy"]["transmittance_contribution_transparent_mean"],
            "transparent_l1": data["rgb_l1"]["transparent_hard"],
            "diffuse_energy": raw["diffuse_contribution"]["mean"],
            "reflection_energy": raw["reflection_contribution"]["mean"],
            "path": data.get("path_contract", {}),
        })
    return {
        "views": rows,
        "mean": {
            key: mean([row[key] for row in rows])
            for key in (
                "ain_mean", "cin_energy", "t_contribution", "transparent_l1",
                "diffuse_energy", "reflection_energy",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("preflight", "long"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    errors, warnings = [], []
    preflight = args.profile == "preflight"
    expected_name = (
        TSCALE_RECOVERY_PREFLIGHT_OUTPUT_NAME if preflight
        else TSCALE_RECOVERY_LONG_OUTPUT_NAME
    )
    nodes = TSCALE_RECOVERY_PREFLIGHT_NODES if preflight else TSCALE_RECOVERY_LONG_NODES
    start = 16000 if preflight else 16050
    endpoint = 16050 if preflight else 20000
    if output.name != expected_name:
        errors.append("T-scale recovery output name mismatch")
    source_hash = sha256_file(args.source)
    if preflight and source_hash != TSCALE_RECOVERY_SOURCE_SHA256:
        errors.append("legacy global-16000 source hash mismatch")
    try:
        source = torch.load(args.source, map_location="cpu")
        source_hashes = {
            branch: state_sha256(source[branch])
            for branch in ("diffuse", "reflection", "transmittance")
        }
    except Exception as exc:
        source = {}; source_hashes = {}
        errors.append(f"source checkpoint unreadable: {exc}")

    release = validate_geometry_release(args.geometry_manifest)
    if release["aggregate_sha256"] != FORMAL_RELEASE_SHA256:
        errors.append("Stage C release aggregate mismatch")
    manifest = json.loads(args.geometry_manifest.read_text(encoding="utf-8"))
    mesh_meta = json.loads(
        (Path(manifest["release_root"]) / "mesh_metadata.json").read_text(encoding="utf-8")
    )
    cuboid = CuboidSpace(
        axes=torch.tensor(mesh_meta["axes_columns"], dtype=torch.float64),
        lower=torch.tensor(mesh_meta["fitted_lower"], dtype=torch.float64),
        upper=torch.tensor(mesh_meta["fitted_upper"], dtype=torch.float64),
        interface_margin=0.05, epsilon=1e-6,
    )
    operator = load_json(
        output / "tscale_recovery_operator_record.json", errors, "operator record"
    )
    if operator.get("schema") != "rtgs_stage_d_tscale_recovery_operator_v1" \
            or operator.get("profile") != args.profile:
        errors.append("T-scale recovery operator identity mismatch")
    for key, expected in (
        ("source_sha256_before", source_hash),
        ("source_sha256_after", source_hash),
        ("release_aggregate_before", FORMAL_RELEASE_SHA256),
        ("release_aggregate_after", FORMAL_RELEASE_SHA256),
    ):
        if operator.get(key) != expected:
            errors.append(f"operator {key} mismatch")
    if operator.get("evidence_tree_sha256_before") != operator.get(
        "evidence_tree_sha256_after"
    ):
        errors.append("blocked long-run evidence tree changed")
    cache = load_json(args.cache / "manifest.json", errors, "ownership v4 cache")
    if cache.get("identity", {}).get("cache_schema_version") \
            != "rtgs_stage_d_cuboid_front_cache_v4" \
            or len(cache.get("entries", [])) != 111:
        errors.append("ownership v4 cache identity/count mismatch")
    metadata = load_json(
        output / "tscale_recovery_metadata.json", errors, "T-scale recovery metadata"
    )
    profile_name = "preflight50" if preflight else "long"
    if metadata.get("schema") != "rtgs_stage_d_tscale_recovery_run_v1" \
            or metadata.get("profile") != profile_name \
            or metadata.get("start_checkpoint_sha256") != source_hash:
        errors.append("T-scale recovery metadata/source mismatch")
    before = metadata.get("phase_a_frozen_hash_before")
    after = metadata.get("phase_a_frozen_hash_after")
    if not before or before != after:
        errors.append("T-scale recovery D/R frozen hash mismatch")
    parity = load_json(output / "cache_parity_report.json", errors, "cache parity")
    if parity.get("status") != "PASS" or parity.get("failures"):
        errors.append("cache parity failed")
    migration = None
    if preflight:
        migration = load_json(
            output / "tscale_migration_parity.json", errors, "scale migration parity"
        )
        if migration.get("status") != "PASS" or migration.get("failures"):
            errors.append("global-16000 scale migration parity failed")
        if migration.get("migration", {}).get("affected_count") != 24:
            errors.append("global-16000 migration affected-count mismatch")
        if not all(migration.get("unchanged_non_scaling_optimizer_groups", {}).values()):
            errors.append("non-scaling T optimizer state changed during migration")

    checkpoints = {}
    for node in nodes:
        node_telemetry = load_json(
            output / "review_node_telemetry" / f"iteration_{node:06d}.json",
            errors, f"review node telemetry {node}",
        )
        if node_telemetry.get("global_iteration") != node \
                or node_telemetry.get("formal_review_node") is not True:
            errors.append(f"review node telemetry {node} mismatch")
        path = output / f"chkpnt{node}.pth"
        try:
            checkpoint = torch.load(path, map_location="cpu")
            tensors, elements = scan_tensors(checkpoint, f"checkpoint.{node}")
            branch_hashes = {
                branch: state_sha256(checkpoint[branch])
                for branch in ("diffuse", "reflection", "transmittance")
            }
            if checkpoint.get("global_iteration") != node \
                    or checkpoint.get("reflection_iteration") != 12000 \
                    or checkpoint.get("transmittance_iteration") != node - 15000:
                errors.append(f"checkpoint {node} iteration mismatch")
            for branch in ("diffuse", "reflection"):
                if source_hashes and branch_hashes[branch] != source_hashes[branch]:
                    errors.append(f"checkpoint {node} changed frozen {branch}")
            t = checkpoint["transmittance"]
            if t.get("position_parameterization") != "cuboid_inside_support_sigmoid_v2" \
                    or t.get("scaling_parameterization") \
                    != "cuboid_support_projected_cap_v2":
                errors.append(f"checkpoint {node} T parameterization mismatch")
            count = int(t["xyz"].shape[0])
            rotation = torch.nn.functional.normalize(t["rotation"].double(), dim=-1)
            raw = torch.exp(t["scaling_2d"].double())
            projected = t.get("projected_active_scaling")
            if not torch.is_tensor(projected) or projected.shape != raw.shape:
                errors.append(f"checkpoint {node} exact active-scale state missing")
                projected = cuboid.constrain_support_scaling(rotation, raw, sigma=3.0)
            active = projected.double()
            bounded = cuboid.constrain_support_scaling(rotation, raw, sigma=3.0)
            world = cuboid.decode_inside_support_latent(
                t["xyz"].double(), rotation, active, sigma=3.0,
            )
            classes = cuboid.classify_support(world, rotation, active, sigma=3.0)
            difference = (raw - active).abs()
            factor = active / raw.clamp_min(torch.finfo(raw.dtype).tiny)
            if count != 4096 or not bool((classes == SUPPORT_STRICT_INSIDE).all()):
                errors.append(f"checkpoint {node} T support/count failure")
            if float(difference.max()) > TSCALE_RECOVERY_RAW_ACTIVE_ATOL \
                    or float((active - bounded).abs().max()) \
                    > TSCALE_RECOVERY_RAW_ACTIVE_ATOL \
                    or float(factor.min()) < 1.0 - TSCALE_RECOVERY_RAW_ACTIVE_ATOL:
                errors.append(f"checkpoint {node} raw/active scale mismatch")
            audit = load_json(
                output / "checkpoint_audits" / f"iteration_{node:06d}.json",
                errors, f"checkpoint hash audit {node}",
            )
            if audit.get("checkpoint_sha256") != sha256_file(path) \
                    or audit.get("branch_hashes") != branch_hashes \
                    or audit.get("diffuse_frozen") is not True \
                    or audit.get("reflection_frozen") is not True:
                errors.append(f"checkpoint {node} hash audit mismatch")
            for branch in ("diffuse", "reflection", "transmittance"):
                ply = output / "point_cloud" / branch / f"iteration_{node}/point_cloud.ply"
                if not ply.is_file() or len(PlyData.read(ply)["vertex"].data) \
                        != int(checkpoint[branch]["xyz"].shape[0]):
                    errors.append(f"checkpoint {node} {branch} PLY missing/count mismatch")
            checkpoints[str(node)] = {
                "sha256": sha256_file(path), "branch_hashes": branch_hashes,
                "tensors": tensors, "elements": elements, "t_count": count,
                "raw_scale_max": float(raw.max()),
                "active_scale_max": float(active.max()),
                "minimum_factor": float(factor.min()),
                "raw_active_max_abs": float(difference.max()),
            }
        except Exception as exc:
            errors.append(f"checkpoint {node} unreadable: {exc}")

    telemetry_path = output / "stage_d_telemetry.jsonl"
    try:
        rows = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    except Exception as exc:
        rows = []; errors.append(f"telemetry unreadable: {exc}")
    expected_iterations = list(range(start + 1, endpoint + 1))
    if [row.get("global_iteration") for row in rows] != expected_iterations:
        errors.append("T-scale recovery telemetry is not continuous")
    for row in rows:
        iteration = row.get("global_iteration")
        metrics = row.get("semantic_metrics", {})
        cap = metrics.get("t_support_scale_cap", {})
        projection = row.get("t_scale_projection", {})
        energy = metrics.get("contribution_energy", {})
        if row.get("schema") != "rtgs_stage_d_tscale_recovery_telemetry_v1" \
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
        if metrics.get("t_spatial_counts") != {
            "strict_inside_safe": 4096, "interface_margin": 0,
            "strict_outside_safe": 0, "crossing_or_ambiguous": 0,
        }:
            errors.append(f"T support class drift at {iteration}"); break
        if energy.get("diffuse_contribution", 1.0) != 0.0 \
                or energy.get("reflection_contribution", 1.0) != 0.0:
            errors.append(f"transparent D/R ownership failed at {iteration}"); break
        if metrics.get("transparent_ownership_modes") != {
            "path": "cuboid_front_v1", "direct": "off", "reflection": "off",
        } or metrics.get("cout_filter", {}).get("formal_mode") \
                != "support_safe_outside":
            errors.append(f"path/Cout ownership mismatch at {iteration}"); break
        if cap.get("schema") != "cuboid_support_projected_cap_v2" \
                or cap.get("capped_count") != 0 \
                or cap.get("minimum_factor", 0) < 1.0 - TSCALE_RECOVERY_RAW_ACTIVE_ATOL \
                or cap.get("raw_active_max_abs", 1) > TSCALE_RECOVERY_RAW_ACTIVE_ATOL:
            errors.append(f"forward raw/active scale mismatch at {iteration}"); break
        if projection.get("schema") != "cuboid_support_projected_cap_v2" \
                or projection.get("minimum_factor_before", 0) \
                < TSCALE_RECOVERY_SCALE_FACTOR_LIMIT \
                or projection.get("raw_active_max_abs_after", 1) \
                > TSCALE_RECOVERY_RAW_ACTIVE_ATOL:
            errors.append(f"post-update scale projection failed at {iteration}"); break
        if row.get("tscale_recovery_guard", {}).get("failures"):
            errors.append(f"runtime T-scale guard failed at {iteration}"); break

    first_window, last_window = rows[:10], rows[-10:]
    telemetry_summary = {}
    if rows:
        def summarize(window):
            return {
                "ain_saturation": mean([
                    row["semantic_metrics"]["ain_saturation_fraction_ge_0_95"]
                    for row in window
                ]),
                "high_ain_near_black": mean([
                    row["semantic_metrics"]["high_ain_near_black_conditional_fraction"]
                    for row in window
                ]),
                "ain_mean": mean([
                    row["semantic_metrics"]["ain"]["mean"] for row in window
                ]),
                "cin_energy": mean([
                    row["semantic_metrics"]["cin_energy"] for row in window
                ]),
                "t_contribution": mean([
                    row["semantic_metrics"]["contribution_energy"][
                        "transmittance_contribution"
                    ] for row in window
                ]),
                "transparent_l1": mean([
                    row["semantic_metrics"]["transparent_rgb_l1"] for row in window
                ]),
                "projection_affected_count": mean([
                    row["t_scale_projection"]["affected_count"] for row in window
                ]),
                "minimum_pre_projection_factor": min(
                    row["t_scale_projection"]["minimum_factor_before"] for row in window
                ),
                "maximum_post_raw_active_abs": max(
                    row["t_scale_projection"]["raw_active_max_abs_after"] for row in window
                ),
            }
        telemetry_summary = {
            "first_10": summarize(first_window), "last_10": summarize(last_window),
        }
        first, last = telemetry_summary["first_10"], telemetry_summary["last_10"]
        if last["ain_saturation"] > max(0.5, first["ain_saturation"] + 0.10):
            errors.append("Ain saturation materially worsened")
        if last["high_ain_near_black"] > max(0.10, first["high_ain_near_black"] + 0.05):
            errors.append("high-Ain near-black materially worsened")
        if last["cin_energy"] < 0.5 * first["cin_energy"]:
            errors.append("Cin energy collapsed")
        if last["t_contribution"] < 0.5 * first["t_contribution"]:
            errors.append("T contribution collapsed")

    review = {str(node): node_metrics(output, node, errors) for node in nodes}
    if str(start) in review and str(endpoint) in review:
        initial, final = review[str(start)]["mean"], review[str(endpoint)]["mean"]
        if final["cin_energy"] < 0.5 * initial["cin_energy"]:
            errors.append("fixed-nine Cin energy collapsed")
        if final["t_contribution"] < 0.5 * initial["t_contribution"]:
            errors.append("fixed-nine T contribution collapsed")
        if final["transparent_l1"] > initial["transparent_l1"] + 0.05:
            errors.append("fixed-nine transparent L1 materially worsened")
        if final["diffuse_energy"] != 0 or final["reflection_energy"] != 0:
            errors.append("fixed-nine transparent D/R energy is nonzero")

    verdict = (
        "TSCALE_RECOVERY_PREFLIGHT_BLOCKED" if errors and preflight else (
            "TSCALE_RECOVERY_LONG_BLOCKED" if errors else (
                "TSCALE_RECOVERY_PREFLIGHT_PASS" if preflight
                else "TSCALE_RECOVERY_LONG_HOLD"
            )
        )
    )
    warnings.append(
        "Projected-scale recovery preserves T-only ownership isolation and is not Stage D acceptance."
    )
    result = {
        "verdict": verdict, "technical_run_complete": not errors,
        "semantic_separation_claimed": False, "errors": errors,
        "warnings": warnings, "profile": args.profile,
        "source_sha256": source_hash,
        "release_aggregate_sha256": FORMAL_RELEASE_SHA256,
        "checkpoints": checkpoints, "telemetry_rows": len(rows),
        "telemetry_summary": telemetry_summary, "review_nodes": review,
        "migration": migration,
    }
    (output / "tscale_recovery_cpu_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    (output / "tscale_recovery_report.md").write_text(
        "# Stage D projected T-scale recovery\n\n"
        f"Verdict: `{verdict}`\n\n"
        "This run migrates or resumes a projected T scale, keeps D/R frozen, "
        "retains cuboid-front ownership isolation, and does not authorize Stage E.\n",
        encoding="utf-8",
    )
    print(verdict)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
