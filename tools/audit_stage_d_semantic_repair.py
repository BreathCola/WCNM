#!/usr/bin/env python3
"""CPU-only final/partial audit for the Stage D semantic-repair pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.cuboid_space import CuboidSpace, INSIDE
from geometry.geometry_release import sha256_file, validate_geometry_release
from stage_d_training import (
    CACHED_STEMS, FORMAL_RELEASE_ID, FORMAL_RELEASE_SHA256,
    FORMAL_SOURCE_SHA256, SEMANTIC_NODES, SEMANTIC_OUTPUT_NAME,
)
from utils.stage_d_static_cache import (
    CACHE_SCHEMA, FORBIDDEN_CACHE_KEYS, MANIFEST_SCHEMA, canonical_sha256,
    state_sha256,
)


MASK_SHA = "056da740a6bb20e7b888be890ac39b597734d5e0487003b36384b57a9cb66551"
REQUIRED_DEBUG = {
    "ground_truth.png", "final.png", "diffuse_contribution.png",
    "reflection_unfiltered.png", "reflection_inside.png",
    "reflection_interface.png", "reflection_outside.png",
    "reflection_final_filtered.png", "transmittance_contribution.png",
    "inside_color.png", "inside_alpha.png", "inside_depth.png",
    "c_in_cond.png", "cout_unfiltered.png", "cout_inside.png",
    "cout_interface.png", "cout_outside.png", "cout_final.png",
    "outside_alpha.png", "outside_depth.png", "transmittance_color.png",
    "transmittance_alpha.png", "near_depth.png", "far_depth.png",
    "two_hit_valid.png", "depth_violation.png", "din_vs_far_violation.png",
    "transparent_mask.png", "ks.png", "t_spatial_class_map.png",
    "transmittance_metadata.json",
}


def load_json(path, errors, label):
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"unreadable {label}: {exc}")
        return {}


def scan_finite(value, path="value"):
    tensors = elements = 0
    if torch.is_tensor(value):
        tensors, elements = 1, value.numel()
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite tensor at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            a, b = scan_finite(child, f"{path}.{key}")
            tensors += a; elements += b
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            a, b = scan_finite(child, f"{path}[{index}]")
            tensors += a; elements += b
    return tensors, elements


def finite_scalars(value):
    if isinstance(value, dict):
        return all(finite_scalars(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_scalars(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def has_forbidden_key(value):
    if isinstance(value, dict):
        return any(
            str(key).lower() in FORBIDDEN_CACHE_KEYS or has_forbidden_key(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(has_forbidden_key(child) for child in value)
    return False


def png_array(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)


def verify_png_equal(first, second):
    a, b = png_array(first), png_array(second)
    return a.shape == b.shape and int(np.abs(a - b).max()) <= 1


def plot_curves(rows, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["global_iteration"] for row in rows]
    series = {
        "total loss": [row["loss"]["total"] for row in rows],
        "Ain mean": [row["semantic_metrics"]["ain"]["mean"] for row in rows],
        "Ain sat.": [row["semantic_metrics"]["ain_saturation_fraction_ge_0_95"] for row in rows],
        "high Ain + black": [row["semantic_metrics"]["high_ain_near_black_conditional_fraction"] for row in rows],
        "T energy": [row["semantic_metrics"]["contribution_energy"].get("transmittance_contribution", 0.0) for row in rows],
        "Din <= far": [row["din_le_t_far_fraction"] for row in rows],
        "step ms": [row["whole_step_wall_ms"] for row in rows],
    }
    figure, axes = plt.subplots(4, 2, figsize=(13, 14))
    for axis, (label, values) in zip(axes.flat, series.items()):
        axis.plot(x, values, linewidth=1.0)
        axis.set_title(label); axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    figure.tight_layout(); figure.savefig(target, dpi=140); plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    errors, holds, warnings = [], [], []
    if output.name != SEMANTIC_OUTPUT_NAME:
        errors.append("semantic pilot output name mismatch")

    try:
        release = validate_geometry_release(args.geometry_manifest)
        if release["geometry_release_id"] != FORMAL_RELEASE_ID:
            errors.append("geometry release ID mismatch")
        if release["aggregate_sha256"] != FORMAL_RELEASE_SHA256:
            errors.append("geometry release aggregate mismatch")
        manifest = json.loads(args.geometry_manifest.read_text(encoding="utf-8"))
        camera_ids = {
            row["stem"]: row["camera_identity_sha256"]
            for row in manifest["cache_entries"]
        }
    except Exception as exc:
        camera_ids = {}
        errors.append(f"geometry release validation failed: {exc}")

    operator = load_json(output / "semantic_repair_operator_record.json", errors, "operator record")
    if operator.get("schema") != "rtgs_stage_d_semantic_repair_operator_v3":
        errors.append("operator schema mismatch")
    for key in ("source_sha256_before", "source_sha256_after"):
        if operator.get(key) != FORMAL_SOURCE_SHA256:
            errors.append(f"operator {key} mismatch or unavailable")
    for key in ("release_aggregate_before", "release_aggregate_after"):
        if operator.get(key) != FORMAL_RELEASE_SHA256:
            errors.append(f"operator {key} mismatch or unavailable")
    causal = load_json(
        ROOT / "output/stage_d_tihubird_c03r8_semantic_repair_v3_audit/semantic_failure_audit.json",
        errors, "v2 causal audit",
    )
    if causal.get("read_only") is not True:
        errors.append("v2 causal audit is absent or not read-only")

    metadata = load_json(output / "semantic_repair_run_metadata.json", errors, "run metadata")
    if metadata.get("schema") != "rtgs_stage_d_semantic_repair_pilot_v3":
        errors.append("run metadata schema mismatch")
    before_hash = metadata.get("phase_a_frozen_hash_before")
    after_hash = metadata.get("phase_a_frozen_hash_after")
    if not before_hash or before_hash != after_hash:
        errors.append("D/R frozen-state before/after hash mismatch")
    config = metadata.get("config", {})
    repair = config.get("semantic_repair", {})
    if repair.get("t_topology", {}).get("required_count") != 4096:
        errors.append("semantic T topology metadata mismatch")
    if config.get("stage_d_depth_start_iteration") != 40000:
        errors.append("future L_depth activation mismatch")
    if metadata.get("bird_roi_available") is not False:
        warnings.append("unexpected bird ROI metadata; audit does not trust it")
    cuboid_metadata = metadata.get("cuboid_space", {})
    try:
        cuboid = CuboidSpace(
            axes=torch.tensor(cuboid_metadata["axes_columns"], dtype=torch.float64),
            lower=torch.tensor(cuboid_metadata["lower"], dtype=torch.float64),
            upper=torch.tensor(cuboid_metadata["upper"], dtype=torch.float64),
            interface_margin=float(cuboid_metadata["interface_margin"]),
            epsilon=float(cuboid_metadata["epsilon"]),
            interface_margin_mode=cuboid_metadata["transparent_interface_margin_mode"],
        )
    except Exception as exc:
        cuboid = None
        errors.append(f"invalid cuboid metadata: {exc}")

    cache_manifest = load_json(output / "static_dr_cache/manifest.json", errors, "cache manifest")
    identity = cache_manifest.get("identity", {})
    if cache_manifest.get("schema") != MANIFEST_SCHEMA or identity.get("cache_schema_version") != CACHE_SCHEMA:
        errors.append("cache schema mismatch")
    if identity.get("source_checkpoint_sha256") != FORMAL_SOURCE_SHA256:
        errors.append("cache source identity mismatch")
    if identity.get("geometry_release_aggregate_sha256") != FORMAL_RELEASE_SHA256:
        errors.append("cache release identity mismatch")
    if identity.get("mask_manifest_sha256") != MASK_SHA:
        errors.append("cache mask identity mismatch")
    renderer_config = identity.get("renderer_config", {})
    if identity.get("renderer_config_sha256") != canonical_sha256(renderer_config):
        errors.append("cache renderer hash mismatch")
    digest_value = dict(identity); digest = digest_value.pop("identity_sha256", None)
    if digest != canonical_sha256(digest_value):
        errors.append("cache identity digest mismatch")
    aggregate_rows = []
    for row in cache_manifest.get("entries", []):
        stem = row.get("camera_stem")
        path = output / "static_dr_cache" / row.get("relative_path", "missing")
        if camera_ids.get(stem) != row.get("camera_identity_sha256"):
            errors.append(f"cache camera identity mismatch: {stem}")
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            errors.append(f"cache file mismatch: {stem}")
            continue
        try:
            payload = torch.load(path, map_location="cpu")
            scan_finite(payload, f"cache.{stem}")
            if payload.get("contains_target_rgb") is not False or has_forbidden_key(payload):
                errors.append(f"cache contains forbidden target data: {stem}")
        except Exception as exc:
            errors.append(f"cache unreadable/nonfinite {stem}: {exc}")
        aggregate_rows.append({key: row.get(key) for key in ("camera_stem", "camera_identity_sha256", "sha256")})
    if len(cache_manifest.get("entries", [])) != 111:
        errors.append("cache does not contain exactly 111 views")
    if canonical_sha256(sorted(aggregate_rows, key=lambda row: row["camera_stem"])) != cache_manifest.get("aggregate_sha256"):
        errors.append("cache aggregate mismatch")

    parity = load_json(output / "cache_parity_report.json", errors, "cache parity report")
    if parity.get("status") != "PASS" or parity.get("failures") or len(parity.get("rows", [])) != 10:
        errors.append("cached/uncached parity did not pass all ten views")
    if parity.get("tolerances") != {"max_absolute": 2e-5, "mean_absolute": 2e-6}:
        errors.append("cache parity tolerance metadata mismatch")
    probe = parity.get("anti_veil_gradient_probe") or {}
    if not (
        probe.get("black", {}).get("t_opacity_gradient_nonzero")
        and probe.get("black", {}).get("t_color_gradient_nonzero")
        and probe.get("saturation", {}).get("t_opacity_gradient_nonzero")
    ):
        errors.append("anti-veil T gradient probe failed")
    if any(
        item.get("diffuse_gradient_leak") or item.get("reflection_gradient_leak")
        for item in probe.values() if isinstance(item, dict)
    ):
        errors.append("anti-veil gradient leaked into frozen D/R")

    checkpoints, final_checkpoint = {}, None
    for node in SEMANTIC_NODES:
        path = output / f"chkpnt{node}.pth"
        try:
            checkpoint = torch.load(path, map_location="cpu")
            tensors, elements = scan_finite(checkpoint, f"checkpoint.{node}")
            if (
                checkpoint.get("global_iteration") != node
                or checkpoint.get("reflection_iteration") != 12000
                or checkpoint.get("transmittance_iteration") != node - 15000
            ):
                errors.append(f"checkpoint {node} iteration semantics mismatch")
            states = {name: checkpoint.get(name, {}) for name in ("diffuse", "reflection", "transmittance")}
            if any(not state for state in states.values()):
                errors.append(f"checkpoint {node} lacks independent D/R/T state")
            if before_hash:
                for branch in ("diffuse", "reflection"):
                    if state_sha256(states[branch]) != before_hash.get(branch):
                        errors.append(f"checkpoint {node} mutated frozen {branch}")
            t_state = states["transmittance"]
            if t_state.get("position_parameterization") != "cuboid_inside_sigmoid_v1":
                errors.append(f"checkpoint {node} T parameterization mismatch")
            if t_state.get("cuboid_space") != cuboid_metadata:
                errors.append(f"checkpoint {node} cuboid metadata mismatch")
            checkpoint_config = checkpoint.get("config", {})
            if checkpoint_config.get("source_stage_b_checkpoint_sha256") != FORMAL_SOURCE_SHA256:
                errors.append(f"checkpoint {node} source identity mismatch")
            if checkpoint_config.get("geometry_release_aggregate_sha256") != FORMAL_RELEASE_SHA256:
                errors.append(f"checkpoint {node} geometry identity mismatch")
            count = int(t_state.get("xyz", torch.empty(0, 3)).shape[0])
            if count != 4096:
                errors.append(f"checkpoint {node} T count is {count}, expected 4096")
            if cuboid is not None and count:
                world = cuboid.decode_inside_latent(t_state["xyz"].double())
                if not bool((cuboid.classify(world) == INSIDE).all()):
                    errors.append(f"checkpoint {node} T is outside strict inside-safe region")
            for branch, state in states.items():
                count_branch = int(state.get("xyz", torch.empty(0, 3)).shape[0])
                ply = output / "point_cloud" / branch / f"iteration_{node}" / "point_cloud.ply"
                if not ply.is_file() or len(PlyData.read(ply)["vertex"].data) != count_branch:
                    errors.append(f"{node} {branch} PLY missing/count mismatch")
            checkpoints[node] = {"sha256": sha256_file(path), "tensors": tensors, "elements": elements, "t_count": count}
            if node == 16000:
                final_checkpoint = checkpoint
        except Exception as exc:
            errors.append(f"checkpoint {node} unreadable/nonfinite: {exc}")

    telemetry_path = output / "stage_d_telemetry.jsonl"
    rows = []
    if telemetry_path.is_file():
        try:
            rows = [json.loads(line) for line in telemetry_path.read_text(encoding="utf-8").splitlines()]
        except Exception as exc:
            errors.append(f"telemetry unreadable: {exc}")
    else:
        errors.append("missing telemetry")
    if [row.get("global_iteration") for row in rows] != list(range(15001, 16001)):
        errors.append("telemetry is not continuous from 15001 through 16000")
    for row in rows:
        iteration = row.get("global_iteration")
        if row.get("schema") != "rtgs_stage_d_semantic_repair_telemetry_v3":
            errors.append(f"telemetry schema mismatch at {iteration}"); break
        if row.get("optimizer_updates_this_step") != {"diffuse": 0, "reflection": 0, "transmittance": 1}:
            errors.append(f"optimizer update mismatch at {iteration}"); break
        if row.get("static_dr_cache_enabled") is not True:
            errors.append(f"static cache disabled unexpectedly at {iteration}"); break
        if row.get("counts", {}).get("transmittance") != 4096:
            errors.append(f"T count drift at {iteration}"); break
        if row.get("topology_event", {}).get("transmittance_densify_prune_called") is not False:
            errors.append(f"T topology event at {iteration}"); break
        if row.get("loss", {}).get("lambda_depth_enabled") is not False:
            errors.append(f"L_depth enabled early at {iteration}"); break
        if row.get("nonfinite_count") != 0 or not finite_scalars(row):
            errors.append(f"nonfinite telemetry at {iteration}"); break
        spatial = row.get("semantic_metrics", {}).get("t_spatial_counts", {})
        if spatial != {"inside": 4096, "interface": 0, "outside": 0}:
            errors.append(f"T spatial contract failed at {iteration}"); break

    node_metrics = {}
    for node in SEMANTIC_NODES:
        root = output / "debug" / f"iteration_{node:06d}"
        sheet = root / "semantic_repair_contact_sheet.png"
        if not sheet.is_file():
            errors.append(f"missing semantic contact sheet at {node}")
        else:
            try: Image.open(sheet).verify()
            except Exception as exc: errors.append(f"invalid contact sheet {node}: {exc}")
        node_metrics[node] = {}
        for stem in CACHED_STEMS:
            view = root / stem
            missing = sorted(name for name in REQUIRED_DEBUG if not (view / name).is_file())
            if missing:
                errors.append(f"debug {node}/{stem} missing {missing}")
                continue
            if not verify_png_equal(view / "reflection_final_filtered.png", view / "reflection_contribution.png"):
                errors.append(f"formal R output differs from outside-only filtered output: {node}/{stem}")
            if not verify_png_equal(view / "cout_final.png", view / "outside_color.png"):
                errors.append(f"formal Cout differs from outside-only filtered output: {node}/{stem}")
            data = load_json(view / "transmittance_metadata.json", errors, f"node metadata {node}/{stem}")
            if not finite_scalars(data):
                errors.append(f"nonfinite node metadata {node}/{stem}")
                continue
            node_metrics[node][stem] = data

    if rows:
        curves = output / "semantic_repair_key_curves.png"
        try: plot_curves(rows, curves)
        except Exception as exc: errors.append(f"curve generation failed: {exc}")
        final_metrics = rows[-1].get("semantic_metrics", {})
        tail = rows[-100:]
        sat_tail = np.asarray([r["semantic_metrics"]["ain_saturation_fraction_ge_0_95"] for r in tail])
        black_first = np.mean([r["semantic_metrics"]["high_ain_near_black_conditional_fraction"] for r in rows[:100]])
        black_last = np.mean([r["semantic_metrics"]["high_ain_near_black_conditional_fraction"] for r in tail])
        if float(np.mean(sat_tail)) >= 0.50:
            errors.append("Ain saturation covers a majority of transparent pixels in the final 100 steps")
        if black_last >= 0.25 and black_last > black_first + 0.05:
            errors.append("high-Ain near-black conditional fraction is high and still increasing")
    else:
        curves, final_metrics = output / "semantic_repair_key_curves.png", {}

    holds.append("No independent versioned bird ROI exists; bird-level quantitative proof is unavailable.")
    holds.append("Pilot contribution maps require nine-view human semantic review; loss reduction is not semantic proof.")
    verdict = "SEMANTIC_REPAIR_PILOT_BLOCKED" if errors else (
        "SEMANTIC_REPAIR_PILOT_HOLD" if holds else "SEMANTIC_REPAIR_PILOT_PASS"
    )
    times = np.asarray([row.get("whole_step_wall_ms", 0.0) for row in rows], dtype=np.float64)
    memory_alloc = max((row.get("cuda_max_memory_allocated_bytes", 0) for row in rows), default=0)
    memory_reserved = max((row.get("cuda_max_memory_reserved_bytes", 0) for row in rows), default=0)
    result = {
        "verdict": verdict, "technical_run_complete": not errors,
        "semantic_separation_claimed": False,
        "errors": errors, "holds": holds, "warnings": warnings,
        "checkpoints": checkpoints,
        "final_checkpoint_sha256": checkpoints.get(16000, {}).get("sha256"),
        "telemetry_rows": len(rows), "cache_views": len(cache_manifest.get("entries", [])),
        "phase_a_frozen_hash_before": before_hash,
        "phase_a_frozen_hash_after": after_hash,
        "final_semantic_metrics": final_metrics,
        "step_time_ms": {
            "mean": float(times.mean()) if times.size else None,
            "p50": float(np.quantile(times, 0.50)) if times.size else None,
            "p95": float(np.quantile(times, 0.95)) if times.size else None,
        },
        "peak_memory_bytes": {"allocated": memory_alloc, "reserved": memory_reserved},
        "final_contact_sheet": str(output / "debug/iteration_016000/semantic_repair_contact_sheet.png"),
        "key_curves": str(curves),
        "v2_causal_audit": str(ROOT / "output/stage_d_tihubird_c03r8_semantic_repair_v3_audit/semantic_failure_audit.json"),
    }
    audit_path = output / "semantic_repair_cpu_audit.json"
    audit_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    report = output / "semantic_repair_pilot_report.md"
    report.write_text(
        "# Stage D semantic-repair v3 pilot report\n\n"
        f"Verdict: `{verdict}`\n\n"
        "This is a 1,000-step frozen-D/R cached-T semantic falsification pilot, not full joint training and not final Stage D success.\n\n"
        "R and second-bounce Cout use outside-only formal contributions in the transparent mask; their inside/interface traces remain diagnostic evidence and are excluded from formal composition. T topology is fixed at 4,096 and T positions use the cuboid inside-safe parameterization.\n\n"
        "Loss reduction is not claimed as bird/background semantic separation. No independent bird ROI exists, so bird-level quantitative proof is unavailable.\n\n"
        f"Final checkpoint SHA-256: `{result['final_checkpoint_sha256']}`.\n\n"
        + ("## Errors\n\n" + "".join(f"- {item}\n" for item in errors) if errors else "")
        + ("\n## Holds\n\n" + "".join(f"- {item}\n" for item in holds) if holds else ""),
        encoding="utf-8",
    )
    print(verdict)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
