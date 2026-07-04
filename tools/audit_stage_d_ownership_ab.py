#!/usr/bin/env python3
"""CPU-only final/partial audit for cuboid-front Stage D ownership A/B v4."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.cuboid_space import CuboidSpace, SUPPORT_STRICT_INSIDE
from geometry.geometry_release import sha256_file, validate_geometry_release
from stage_d_training import (
    CACHED_STEMS, FORMAL_RELEASE_ID, FORMAL_RELEASE_SHA256,
    FORMAL_SOURCE_SHA256, OWNERSHIP_ARM_NAMES, OWNERSHIP_NODES,
    OWNERSHIP_OUTPUT_NAME,
)
from utils.stage_d_static_cache import (
    OWNERSHIP_CACHE_SCHEMA, OWNERSHIP_MANIFEST_SCHEMA,
    canonical_sha256, state_sha256,
)
from tools.run_stage_d_ownership_ab import common_training_contract, training_command


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


def load_json(path, errors, label):
    path = Path(path)
    if not path.is_file():
        errors.append(f"missing {label}: {path}"); return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"unreadable {label}: {exc}"); return {}


def finite(value):
    if isinstance(value, dict): return all(finite(child) for child in value.values())
    if isinstance(value, (list, tuple)): return all(finite(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def scan_tensors(value, path="value"):
    tensors = elements = 0
    if torch.is_tensor(value):
        tensors, elements = 1, value.numel()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite tensor at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            a, b = scan_tensors(child, f"{path}.{key}"); tensors += a; elements += b
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            a, b = scan_tensors(child, f"{path}[{index}]"); tensors += a; elements += b
    return tensors, elements


def image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def roi_metrics(output, manifest_path, errors):
    if not manifest_path:
        return {"status": "NO_INDEPENDENT_BIRD_ROI"}
    manifest = load_json(manifest_path, errors, "bird ROI manifest")
    provenance = manifest.get("provenance", {})
    if manifest.get("schema") != "rtgs_independent_bird_roi_v1" \
            or provenance.get("model_prediction") is not False \
            or provenance.get("source") not in ("manual", "human_annotation"):
        errors.append("bird ROI manifest is not independent/manual")
        return {"status": "INVALID"}
    root = Path(manifest_path).parent; rows = []
    entries = {row["stem"]: row for row in manifest.get("entries", [])}
    for arm_dir in OWNERSHIP_ARM_NAMES.values():
        for stem in CACHED_STEMS:
            entry = entries.get(stem)
            if not entry:
                errors.append(f"bird ROI lacks fixed view {stem}"); continue
            path = root / entry["relative_path"]
            if not path.is_file() or sha256_file(path) != entry.get("sha256"):
                errors.append(f"bird ROI hash mismatch: {stem}"); continue
            mask = np.asarray(Image.open(path).convert("L")) >= 128
            view = output / arm_dir / "debug/iteration_015500" / stem
            gt = image(view / "ground_truth.png")
            final = image(view / "final.png")
            t_off = image(view / "final_t_off.png")
            cout_off = image(view / "final_cout_off.png")
            rows.append({
                "arm": arm_dir, "stem": stem,
                "delta_l1_bird_t_off": float(np.abs(t_off[mask] - gt[mask]).mean()
                                             - np.abs(final[mask] - gt[mask]).mean()),
                "delta_l1_bird_cout_off": float(np.abs(cout_off[mask] - gt[mask]).mean()
                                                - np.abs(final[mask] - gt[mask]).mean()),
            })
    return {"status": "AVAILABLE", "manifest": str(Path(manifest_path).resolve()), "rows": rows}


def audit_arm(output, arm, cuboid, errors):
    arm_dir = output / OWNERSHIP_ARM_NAMES[arm]
    metadata = load_json(arm_dir / "ownership_arm_metadata.json", errors, f"{arm} metadata")
    if metadata.get("schema") != "rtgs_stage_d_cuboid_path_ownership_arm_v4":
        errors.append(f"{arm} metadata schema mismatch")
    if metadata.get("arm") != arm \
            or metadata.get("bird_roi_status") != "OPERATOR_EVALUATION_ONLY":
        errors.append(f"{arm} identity/bird ROI status mismatch")
    config = metadata.get("config", {})
    required_modes = {
        "transparent_path_mode": "cuboid_front_v1",
        "transparent_direct_mode": "off",
        "transparent_reflection_mode": "off",
        "cout_ownership_mode": "support_safe_outside",
        "stage_d_depth_start_iteration": 40000,
    }
    for key, expected in required_modes.items():
        if config.get(key) != expected:
            errors.append(f"{arm} config {key} mismatch")
    t_topology = config.get("ownership_handoff", {}).get("t_topology", {})
    if t_topology.get("scaling_parameterization") \
            != "cuboid_support_uniform_cap_v1":
        errors.append(f"{arm} config T scale parameterization mismatch")
    before = metadata.get("phase_a_frozen_hash_before")
    after = metadata.get("phase_a_frozen_hash_after")
    if not before or before != after:
        errors.append(f"{arm} D/R frozen-state hash mismatch")
    parity = load_json(arm_dir / "cache_parity_report.json", errors, f"{arm} parity")
    if parity.get("status") != "PASS" or parity.get("failures") or len(parity.get("rows", [])) != 10:
        errors.append(f"{arm} cache/path parity failed")
    checkpoints = {}
    for node in OWNERSHIP_NODES:
        path = arm_dir / f"chkpnt{node}.pth"
        try:
            checkpoint = torch.load(path, map_location="cpu")
            tensors, elements = scan_tensors(checkpoint, f"{arm}.{node}")
            if checkpoint.get("global_iteration") != node \
                    or checkpoint.get("reflection_iteration") != 12000 \
                    or checkpoint.get("transmittance_iteration") != node - 15000:
                errors.append(f"{arm} checkpoint {node} iteration mismatch")
            for branch in ("diffuse", "reflection"):
                if before and state_sha256(checkpoint[branch]) != before.get(branch):
                    errors.append(f"{arm} checkpoint {node} changed frozen {branch}")
            t = checkpoint["transmittance"]
            if t.get("position_parameterization") != "cuboid_inside_support_sigmoid_v2":
                errors.append(f"{arm} checkpoint {node} T parameterization mismatch")
            if t.get("scaling_parameterization") != "cuboid_support_uniform_cap_v1":
                errors.append(f"{arm} checkpoint {node} T scale parameterization mismatch")
            initialization = t.get("initialization", {})
            if initialization.get("mode") != arm:
                errors.append(f"{arm} checkpoint {node} initialization provenance mismatch")
            if arm == "transferred_d_inside":
                selection = initialization.get("selection", {})
                calibration = initialization.get("opacity_recalibration", {})
                if selection.get("minimum_distinct_views") != 3 \
                        or selection.get("minimum_total_alpha_weight") != 0.01 \
                        or selection.get("support_class") != "strict_inside_safe" \
                        or calibration.get("scale") != 0.25 \
                        or calibration.get("minimum") != 0.005 \
                        or calibration.get("maximum") != 0.05:
                    errors.append(f"{arm} checkpoint {node} transfer contract mismatch")
            count = int(t.get("xyz", torch.empty(0, 3)).shape[0])
            if count != 4096:
                errors.append(f"{arm} checkpoint {node} T count {count}")
            if count:
                rotation = torch.nn.functional.normalize(t["rotation"].double(), dim=-1)
                raw_scaling = torch.exp(t["scaling_2d"].double())
                scaling = cuboid.constrain_support_scaling(
                    rotation, raw_scaling, sigma=3.0,
                )
                world = cuboid.decode_inside_support_latent(
                    t["xyz"].double(), rotation, scaling, sigma=3.0,
                )
                classes = cuboid.classify_support(world, rotation, scaling, sigma=3.0)
                if not bool((classes == SUPPORT_STRICT_INSIDE).all()):
                    errors.append(f"{arm} checkpoint {node} T support is not strict-inside-safe")
            for branch in ("diffuse", "reflection", "transmittance"):
                ply = arm_dir / "point_cloud" / branch / f"iteration_{node}" / "point_cloud.ply"
                expected = int(checkpoint[branch]["xyz"].shape[0])
                if not ply.is_file() or len(PlyData.read(ply)["vertex"].data) != expected:
                    errors.append(f"{arm} {node} {branch} PLY missing/count mismatch")
            checkpoints[node] = {
                "sha256": sha256_file(path), "tensors": tensors,
                "elements": elements, "t_count": count,
            }
        except Exception as exc:
            errors.append(f"{arm} checkpoint {node} unreadable: {exc}")
    telemetry_path = arm_dir / "stage_d_telemetry.jsonl"
    try:
        rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
    except Exception as exc:
        rows = []; errors.append(f"{arm} telemetry unreadable: {exc}")
    if [row.get("global_iteration") for row in rows] != list(range(15001, 15501)):
        errors.append(f"{arm} telemetry is not continuous for 500 steps")
    for row in rows:
        iteration = row.get("global_iteration")
        if row.get("schema") != "rtgs_stage_d_cuboid_path_ownership_telemetry_v4":
            errors.append(f"{arm} telemetry schema mismatch at {iteration}"); break
        if row.get("optimizer_updates_this_step") != {"diffuse": 0, "reflection": 0, "transmittance": 1}:
            errors.append(f"{arm} optimizer update mismatch at {iteration}"); break
        if row.get("counts", {}).get("transmittance") != 4096 \
                or row.get("topology_event", {}).get("transmittance_densify_prune_called") is not False:
            errors.append(f"{arm} T topology/count mismatch at {iteration}"); break
        if row.get("loss", {}).get("lambda_depth_enabled") is not False \
                or row.get("nonfinite_count") != 0 or not finite(row):
            errors.append(f"{arm} loss/finite contract failed at {iteration}"); break
        if row.get("ray_memory_policy", {}).get("oom_retry_count") != 0:
            errors.append(f"{arm} encountered an OOM retry at {iteration}"); break
        metrics = row.get("semantic_metrics", {})
        spatial = metrics.get("t_spatial_counts", {})
        if spatial != {
            "strict_inside_safe": 4096, "interface_margin": 0,
            "strict_outside_safe": 0, "crossing_or_ambiguous": 0,
        }:
            errors.append(f"{arm} T support class drift at {iteration}"); break
        path = metrics.get("cuboid_front_path", {})
        if path.get("origin_tnear_max_abs", 1.0) > 1e-5 \
                or path.get("normal_faceforward_min_dot", -1.0) <= 0 \
                or path.get("front_plane_residual_max", 1.0) > 5e-4 \
                or path.get("back_tfar_residual_max", 1.0) > 5e-4:
            errors.append(f"{arm} cuboid-front path contract failed at {iteration}"); break
        energy = metrics.get("contribution_energy", {})
        if energy.get("diffuse_contribution", 1.0) != 0.0 \
                or energy.get("reflection_contribution", 1.0) != 0.0:
            errors.append(f"{arm} transparent D/R-off ownership failed at {iteration}"); break
        if metrics.get("cout_filter", {}).get("formal_mode") != "support_safe_outside":
            errors.append(f"{arm} Cout formal support mode failed at {iteration}"); break
    node_metrics = {}
    for node in OWNERSHIP_NODES:
        node_metrics[node] = {}
        for stem in CACHED_STEMS:
            view = arm_dir / "debug" / f"iteration_{node:06d}" / stem
            missing = sorted(name for name in REQUIRED_DEBUG if not (view / name).is_file())
            if missing:
                errors.append(f"{arm} debug {node}/{stem} missing {missing}"); continue
            data = load_json(view / "transmittance_metadata.json", errors, f"{arm} node {node}/{stem}")
            if not finite(data) or "raw_float_statistics" not in data:
                errors.append(f"{arm} node {node}/{stem} lacks finite raw statistics")
            raw = data.get("raw_float_statistics_transparent_hard", {})
            if raw.get("diffuse_contribution", {}).get("max", 1.0) != 0.0 \
                    or raw.get("reflection_contribution", {}).get("max", 1.0) != 0.0:
                errors.append(f"{arm} node {node}/{stem} D/R-off map is nonzero")
            node_metrics[node][stem] = data
    return {
        "metadata": metadata, "checkpoints": checkpoints, "telemetry": rows,
        "node_metrics": node_metrics, "frozen_hash": before,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--bird-roi-manifest", type=Path)
    args = parser.parse_args(); output = args.output.resolve()
    errors, holds, warnings = [], [], []
    if output.name != OWNERSHIP_OUTPUT_NAME:
        errors.append("ownership A/B output name mismatch")
    release = validate_geometry_release(args.geometry_manifest)
    if release["geometry_release_id"] != FORMAL_RELEASE_ID \
            or release["aggregate_sha256"] != FORMAL_RELEASE_SHA256:
        errors.append("geometry release identity mismatch")
    release_manifest = json.loads(args.geometry_manifest.read_text(encoding="utf-8"))
    release_root = Path(release_manifest["release_root"])
    meta = json.loads((release_root / "mesh_metadata.json").read_text(encoding="utf-8"))
    cuboid = CuboidSpace(
        axes=torch.tensor(meta["axes_columns"], dtype=torch.float64),
        lower=torch.tensor(meta["fitted_lower"], dtype=torch.float64),
        upper=torch.tensor(meta["fitted_upper"], dtype=torch.float64),
        interface_margin=0.05, epsilon=1e-6,
    )
    operator = load_json(output / "ownership_operator_record.json", errors, "operator record")
    if operator.get("schema") != "rtgs_stage_d_cuboid_path_ownership_operator_v4":
        errors.append("operator schema mismatch")
    for key in ("source_sha256_before", "source_sha256_after"):
        if operator.get(key) != FORMAL_SOURCE_SHA256:
            errors.append(f"operator {key} mismatch")
    for key in ("release_aggregate_before", "release_aggregate_after"):
        if operator.get(key) != FORMAL_RELEASE_SHA256:
            errors.append(f"operator {key} mismatch")
    recorded_commands = operator.get("commands", {})
    try:
        command_a = recorded_commands["random_strict_inside"]
        command_b = recorded_commands["transferred_d_inside"]
        if common_training_contract(command_a) != common_training_contract(command_b):
            errors.append("A/B operator commands differ beyond initialization/output/cache reuse")
        if command_a != training_command("random_strict_inside") \
                or command_b != training_command("transferred_d_inside"):
            errors.append("operator commands differ from the committed v4 contract")
    except Exception as exc:
        errors.append(f"operator A/B command contract unreadable: {exc}")
    cache = load_json(output / "cuboid_front_cache_v4/manifest.json", errors, "v4 cache")
    identity = cache.get("identity", {})
    if cache.get("schema") != OWNERSHIP_MANIFEST_SCHEMA \
            or identity.get("cache_schema_version") != OWNERSHIP_CACHE_SCHEMA:
        errors.append("old/incompatible static cache schema was used")
    if identity.get("source_checkpoint_sha256") != FORMAL_SOURCE_SHA256 \
            or identity.get("geometry_release_aggregate_sha256") != FORMAL_RELEASE_SHA256:
        errors.append("v4 cache source/release identity mismatch")
    renderer = identity.get("renderer_config", {})
    required_renderer = {
        "schema": "rtgs_stage_d_cuboid_front_renderer_contract_v4",
        "transparent_path_mode": "cuboid_front_v1",
        "transparent_direct_mode": "off",
        "transparent_reflection_mode": "off",
        "cout_ownership_mode": "support_safe_outside",
        "support_classification": "cuboid_local_finite_3sigma_v1",
    }
    for key, expected in required_renderer.items():
        if renderer.get(key) != expected:
            errors.append(f"v4 cache renderer contract mismatch: {key}")
    if identity.get("renderer_config_sha256") != canonical_sha256(renderer):
        errors.append("v4 cache renderer hash mismatch")
    entries = cache.get("entries", [])
    if len(entries) != 111:
        errors.append("v4 cache does not contain 111 views")
    aggregate_rows = []
    for row in entries:
        try:
            path = output / "cuboid_front_cache_v4" / row["relative_path"]
            if not path.is_file() or sha256_file(path) != row["sha256"] \
                    or path.stat().st_size != row["size_bytes"]:
                errors.append(f"v4 cache file identity mismatch: {row.get('camera_stem')}")
                continue
            aggregate_rows.append({
                key: row[key]
                for key in ("camera_stem", "camera_identity_sha256", "sha256")
            })
            if row["camera_stem"] in CACHED_STEMS:
                payload = torch.load(path, map_location="cpu")
                static = payload["static_inputs"]
                components = static.get("semantic_cout_components", {})
                formal = components.get("strict_outside_safe")
                if formal is None or not torch.equal(static["outside_raw"], formal["raw"]) \
                        or not torch.equal(static["outside_alpha"], formal["alpha"]) \
                        or not torch.equal(static["outside_relative_depth"], formal["depth"]) \
                        or not torch.equal(static["outside_hit"], formal["hit"]):
                    errors.append(
                        f"v4 cache Cout formal path leaks a non-strict-outside class: {row['camera_stem']}"
                    )
        except Exception as exc:
            errors.append(f"v4 cache entry unreadable: {row.get('camera_stem')}: {exc}")
    if len(aggregate_rows) == len(entries) \
            and canonical_sha256(sorted(aggregate_rows, key=lambda row: row["camera_stem"])) \
            != cache.get("aggregate_sha256"):
        errors.append("v4 cache aggregate hash mismatch")
    arms = {arm: audit_arm(output, arm, cuboid, errors) for arm in OWNERSHIP_ARM_NAMES}
    if all(arms[arm]["telemetry"] for arm in arms):
        decks = [[row["camera_stem"] for row in arms[arm]["telemetry"]] for arm in arms]
        if decks[0] != decks[1]:
            errors.append("A/B random camera sequences differ")
    if arms["random_strict_inside"]["frozen_hash"] != arms["transferred_d_inside"]["frozen_hash"]:
        errors.append("A/B D/R source state differs")
    roi = roi_metrics(output, args.bird_roi_manifest, errors)
    if roi["status"] == "NO_INDEPENDENT_BIRD_ROI":
        holds.append("NO_INDEPENDENT_BIRD_ROI: bird-level intervention claims are unavailable")
    warnings.append(
        "Even a healthy pilot proves only T handoff conditions under strong ownership isolation, not final D/R/T separation."
    )
    verdict = "CUBOID_PATH_OWNERSHIP_PILOT_BLOCKED" if errors else (
        "CUBOID_PATH_OWNERSHIP_PILOT_HOLD" if holds else "CUBOID_PATH_OWNERSHIP_PILOT_PASS"
    )
    final_metrics = {}
    for arm, value in arms.items():
        rows = value["telemetry"]
        final_metrics[arm] = rows[-1].get("semantic_metrics", {}) if rows else {}
    result = {
        "verdict": verdict, "technical_run_complete": not errors,
        "semantic_separation_claimed": False,
        "errors": errors, "holds": holds, "warnings": warnings,
        "source_sha256": FORMAL_SOURCE_SHA256,
        "release_aggregate_sha256": FORMAL_RELEASE_SHA256,
        "cache_schema": identity.get("cache_schema_version"),
        "cache_identity_sha256": identity.get("identity_sha256"),
        "arms": {
            arm: {"checkpoints": value["checkpoints"], "telemetry_rows": len(value["telemetry"])}
            for arm, value in arms.items()
        },
        "final_metrics": final_metrics,
        "bird_roi": roi,
        "ab_only_variable": "transmittance initialization: random_strict_inside vs transferred_d_inside",
    }
    (output / "ownership_ab_cpu_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    report = output / "ownership_ab_report.md"
    report.write_text(
        "# Stage D cuboid-front ownership A/B v4\n\n"
        f"Verdict: `{verdict}`\n\n"
        "Both 500-step arms freeze D/R, use the frozen cuboid front path, disable transparent D/R contributions, retain only support-safe outside Cout, and keep fixed support-safe T topology. The only training hypothesis changed between arms is T initialization.\n\n"
        "This pilot cannot establish final physical D/R/T separation or authorize Stage E.\n\n"
        + ("## Errors\n\n" + "".join(f"- {item}\n" for item in errors) if errors else "")
        + ("\n## Holds\n\n" + "".join(f"- {item}\n" for item in holds) if holds else ""),
        encoding="utf-8",
    )
    print(verdict); print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
