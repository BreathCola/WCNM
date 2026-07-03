#!/usr/bin/env python3
"""CPU-only final/partial audit for cached-T warm-up plus exact joint Stage D."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_release import sha256_file, validate_geometry_release
from stage_d_training import (
    CACHED_NODES, CACHED_STEMS, FORMAL_RELEASE_ID, FORMAL_RELEASE_SHA256,
    FORMAL_SOURCE_SHA256,
)
from tools.audit_stage_d_formal import plot_curves
from utils.stage_d_static_cache import (
    CACHE_SCHEMA, FORBIDDEN_CACHE_KEYS, MANIFEST_SCHEMA, canonical_sha256,
    state_sha256,
)


REQUIRED_DEBUG = {
    "ground_truth.png", "final.png", "diffuse_contribution.png",
    "reflection_contribution.png", "transmittance_contribution.png",
    "inside_color.png", "inside_alpha.png", "inside_depth.png",
    "outside_color.png", "outside_alpha.png", "outside_depth.png",
    "transmittance_color.png", "transmittance_alpha.png",
    "depth_violation.png", "din_vs_far_violation.png", "near_depth.png",
    "far_depth.png", "two_hit_valid.png", "ks.png", "transparent_mask.png",
    "transmittance_metadata.json",
}


def scan_finite(value, path="value"):
    tensors = elements = 0
    if torch.is_tensor(value):
        tensors, elements = 1, value.numel()
        if not torch.isfinite(value).all():
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


def expected_r_local(node):
    return 12000 if node <= 18000 else 12000 + node - 18000


def load_json(path, errors, label):
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"unreadable {label}: {exc}")
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    errors, warnings = [], []

    try:
        release = validate_geometry_release(args.geometry_manifest)
        release_manifest = json.loads(args.geometry_manifest.read_text(encoding="utf-8"))
        release_camera_identities = {
            row["stem"]: row["camera_identity_sha256"]
            for row in release_manifest["cache_entries"]
        }
        if release["geometry_release_id"] != FORMAL_RELEASE_ID:
            errors.append("geometry release ID mismatch")
        if release["aggregate_sha256"] != FORMAL_RELEASE_SHA256:
            errors.append("geometry release aggregate mismatch")
    except Exception as exc:
        release = {}
        release_camera_identities = {}
        errors.append(f"geometry release validation failed: {exc}")

    operator = load_json(
        output / "cached_twarmup_operator_record.json", errors, "operator record"
    )
    if operator.get("schema") != "rtgs_stage_d_cached_twarmup_operator_v1":
        errors.append("operator schema mismatch")
    for key in ("source_sha256_before", "source_sha256_after"):
        if operator.get(key) != FORMAL_SOURCE_SHA256:
            errors.append(f"operator {key} mismatch or unavailable")
    for key in ("release_aggregate_before", "release_aggregate_after"):
        if operator.get(key) != FORMAL_RELEASE_SHA256:
            errors.append(f"operator {key} mismatch or unavailable")

    metadata = load_json(
        output / "cached_twarmup_run_metadata.json", errors, "run metadata"
    )
    if metadata.get("schema") != "rtgs_stage_d_cached_twarmup_then_joint_v1":
        errors.append("run metadata schema mismatch")
    before_hash = metadata.get("phase_a_frozen_hash_before")
    after_hash = metadata.get("phase_a_frozen_hash_after")
    if not before_hash or before_hash != after_hash:
        errors.append("Phase A D/R frozen-state before/after hash mismatch")
    if metadata.get("phase_b", {}).get("static_dr_cache_enabled") is not False:
        errors.append("metadata does not explicitly disable cache in Phase B")
    if metadata.get("future_depth_activation_global") != 40000:
        errors.append("future L_depth activation metadata mismatch")

    cache_manifest = load_json(
        output / "static_dr_cache/manifest.json", errors, "static cache manifest"
    )
    if cache_manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("static cache manifest schema mismatch")
    cache_identity = cache_manifest.get("identity", {})
    if cache_identity.get("cache_schema_version") != CACHE_SCHEMA:
        errors.append("static cache payload schema identity mismatch")
    if cache_identity.get("source_checkpoint_sha256") != FORMAL_SOURCE_SHA256:
        errors.append("static cache source identity mismatch")
    if cache_identity.get("geometry_release_aggregate_sha256") != FORMAL_RELEASE_SHA256:
        errors.append("static cache release identity mismatch")
    if cache_identity.get("mask_manifest_sha256") \
            != "056da740a6bb20e7b888be890ac39b597734d5e0487003b36384b57a9cb66551":
        errors.append("static cache mask-manifest identity mismatch")
    renderer_config = cache_identity.get("renderer_config", {})
    if cache_identity.get("renderer_config_sha256") != canonical_sha256(renderer_config):
        errors.append("static cache renderer-config hash mismatch")
    identity_without_digest = dict(cache_identity)
    identity_digest = identity_without_digest.pop("identity_sha256", None)
    if identity_digest != canonical_sha256(identity_without_digest):
        errors.append("static cache identity digest mismatch")
    if metadata.get("cache_aggregate_sha256") != cache_manifest.get("aggregate_sha256"):
        errors.append("run metadata/cache aggregate mismatch")
    cache_rows = cache_manifest.get("entries", [])
    cache_aggregate_rows = []
    for row in cache_rows:
        if release_camera_identities.get(row.get("camera_stem")) \
                != row.get("camera_identity_sha256"):
            errors.append(f"cache/release camera identity mismatch: {row.get('camera_stem')}")
        path = output / "static_dr_cache" / row.get("relative_path", "missing")
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            errors.append(f"static cache file mismatch: {row.get('camera_stem')}")
            continue
        try:
            payload = torch.load(path, map_location="cpu")
            scan_finite(payload, f"cache.{row.get('camera_stem')}")
            if payload.get("contains_target_rgb") is not False or has_forbidden_key(payload):
                errors.append(f"cache contains forbidden target data: {row.get('camera_stem')}")
            if payload.get("schema") != CACHE_SCHEMA:
                errors.append(f"cache schema mismatch: {row.get('camera_stem')}")
            if payload.get("identity_sha256") != cache_identity.get("identity_sha256"):
                errors.append(f"cache identity mismatch: {row.get('camera_stem')}")
            if payload.get("camera_stem") != row.get("camera_stem") \
                    or payload.get("camera_identity_sha256") != row.get("camera_identity_sha256"):
                errors.append(f"cache camera identity mismatch: {row.get('camera_stem')}")
        except Exception as exc:
            errors.append(f"static cache unreadable/nonfinite {row.get('camera_stem')}: {exc}")
        cache_aggregate_rows.append({
            key: row.get(key)
            for key in ("camera_stem", "camera_identity_sha256", "sha256")
        })
    if len(cache_rows) != 111:
        errors.append(f"static cache requires 111 views, found {len(cache_rows)}")
    if canonical_sha256(sorted(cache_aggregate_rows, key=lambda row: row["camera_stem"])) \
            != cache_manifest.get("aggregate_sha256"):
        errors.append("static cache aggregate mismatch")

    parity = load_json(output / "cache_parity_report.json", errors, "cache parity report")
    if parity.get("status") != "PASS" or parity.get("failures"):
        errors.append("cached/uncached output-loss-gradient parity did not pass")
    if len(parity.get("rows", [])) != 10:
        errors.append("cache parity does not contain nine fixed plus one random view")
    performance = load_json(
        output / "cache_performance_benchmark.json", errors, "cache performance benchmark"
    )
    if performance.get("optimizer_updates") != 0:
        errors.append("performance benchmark mutated optimizer-update semantics")

    checkpoints, final_checkpoint = {}, None
    for node in CACHED_NODES:
        path = output / f"chkpnt{node}.pth"
        if not path.is_file():
            errors.append(f"missing checkpoint {node}")
            continue
        try:
            checkpoint = torch.load(path, map_location="cpu")
            tensors, elements = scan_finite(checkpoint, f"checkpoint.{node}")
            actual = (
                checkpoint.get("global_iteration"),
                checkpoint.get("reflection_iteration"),
                checkpoint.get("transmittance_iteration"),
            )
            expected = (node, expected_r_local(node), node - 15000)
            if actual != expected:
                errors.append(f"checkpoint {node} iteration tuple {actual} != {expected}")
            states = {
                name: checkpoint.get(name, {})
                for name in ("diffuse", "reflection", "transmittance")
            }
            if any(state.get("optimizer") is None for state in states.values()):
                errors.append(f"checkpoint {node} has incomplete optimizer state")
            try:
                storage = [
                    states[branch]["xyz"].untyped_storage().data_ptr()
                    for branch in ("diffuse", "reflection", "transmittance")
                ]
                if len(set(storage)) != 3:
                    errors.append(f"checkpoint {node} D/R/T storage is not independent")
            except Exception as exc:
                errors.append(f"checkpoint {node} branch independence check failed: {exc}")
            if node <= 18000 and before_hash:
                for branch in ("diffuse", "reflection"):
                    if state_sha256(states[branch]) != before_hash.get(branch):
                        errors.append(f"Phase A checkpoint {node} mutated {branch} state")
            config = checkpoint.get("config", {})
            if config.get("source_stage_b_checkpoint_sha256") != FORMAL_SOURCE_SHA256:
                errors.append(f"checkpoint {node} source identity mismatch")
            if config.get("stage_d_depth_start_iteration") != 40000:
                errors.append(f"checkpoint {node} depth schedule mismatch")
            counts = {
                branch: int(states[branch]["xyz"].shape[0]) for branch in states
                if "xyz" in states[branch]
            }
            for branch, count in counts.items():
                ply = output / "point_cloud" / branch / f"iteration_{node}" / "point_cloud.ply"
                if not ply.is_file() or len(PlyData.read(ply)["vertex"].data) != count:
                    errors.append(f"{node} {branch} PLY missing/count mismatch")
            checkpoints[node] = {
                "path": str(path), "sha256": sha256_file(path), "counts": counts,
                "finite_tensor_count": tensors, "finite_element_count": elements,
            }
            if node == 20000:
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
    if [row.get("global_iteration") for row in rows] != list(range(15001, 20001)):
        errors.append("telemetry is not continuous from global 15001 through 20000")
    for row in rows:
        iteration = row.get("global_iteration", 0)
        phase_a = iteration <= 18000
        if row.get("schema") != "rtgs_stage_d_cached_twarmup_telemetry_v1":
            errors.append(f"telemetry schema mismatch at {iteration}"); break
        if row.get("nonfinite_count") != 0 or not finite_scalars(row):
            errors.append(f"nonfinite telemetry at {iteration}"); break
        expected_updates = {
            "diffuse": int(not phase_a), "reflection": int(not phase_a),
            "transmittance": 1,
        }
        if row.get("optimizer_updates_this_step") != expected_updates:
            errors.append(f"optimizer-update semantics mismatch at {iteration}"); break
        if row.get("static_dr_cache_enabled") is not phase_a:
            errors.append(f"cache phase switch mismatch at {iteration}"); break
        if not phase_a and row.get("static_dr_cache_aggregate_sha256") is not None:
            errors.append(f"Phase B cache was not fully disabled at {iteration}"); break
        if row.get("loss", {}).get("lambda_depth_enabled") is not False:
            errors.append(f"L_depth enabled early at {iteration}"); break

    node_metrics = {}
    for node in CACHED_NODES:
        root = output / "debug" / f"iteration_{node:06d}"
        sheet = root / "contact_sheet.png"
        if not sheet.is_file():
            errors.append(f"missing contact sheet at {node}")
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
            data = load_json(
                view / "transmittance_metadata.json", errors,
                f"debug metadata {node}/{stem}",
            )
            if not finite_scalars(data):
                errors.append(f"debug metadata {node}/{stem} is nonfinite")
                continue
            node_metrics[node][stem] = {
                "din_le_t_far_fraction": data["din_le_t_far_fraction"],
                "inside_alpha_mean": data["raw_stats"]["inside_alpha"]["mean"],
                "inside_energy": data["energy"]["inside_color_valid_mean"],
                "outside_energy": data["energy"]["outside_color_valid_mean"],
                "full_rgb_l1": data["rgb_l1"]["full_image"],
            }

    curves = output / "key_curves.png"
    if rows and all(len(node_metrics.get(node, {})) == len(CACHED_STEMS) for node in CACHED_NODES):
        semantic_curves = plot_curves(rows, node_metrics, curves)
    else:
        semantic_curves = output / "node_semantic_curves.png"
        errors.append("insufficient complete data for required curves")

    t_health = {}
    if final_checkpoint is not None:
        state = final_checkpoint["transmittance"]
        alpha = torch.sigmoid(state["opacity_raw"].float())
        color = torch.sigmoid(state["color_raw"].float())
        t_health = {
            "count": int(alpha.shape[0]), "alpha_mean": float(alpha.mean()),
            "alpha_std": float(alpha.std()), "color_mean": float(color.mean()),
            "color_std": float(color.std()),
        }
        if not (t_health["count"] > 0 and t_health["alpha_mean"] > 1e-8
                and t_health["alpha_mean"] < 1 - 1e-8 and t_health["color_std"] > 0):
            errors.append("T field is zero, saturated, or collapsed")

    verdict = "CACHED_T_WARMUP_BLOCKED" if errors else "HOLD_FOR_SEMANTIC_REVIEW"
    result = {
        "verdict": verdict, "technical_run_complete": not errors,
        "semantic_separation_claimed": False,
        "semantic_note": (
            "Loss reduction is not evidence of bird/background separation; inspect all "
            "nine-view nodes before any continuation."
        ),
        "errors": errors, "warnings": warnings, "checkpoints": checkpoints,
        "final_checkpoint_sha256": checkpoints.get(20000, {}).get("sha256"),
        "telemetry_rows": len(rows), "cache_views": len(cache_rows),
        "phase_a_frozen_hash_before": before_hash,
        "phase_a_frozen_hash_after": after_hash,
        "phase_b_static_cache_disabled": bool(rows) and all(
            row.get("static_dr_cache_enabled") is False
            for row in rows if row.get("global_iteration", 0) >= 18001
        ),
        "t_health": t_health,
        "contact_sheet": str(output / "debug/iteration_020000/contact_sheet.png"),
        "key_curves": str(curves), "node_semantic_curves": str(semantic_curves),
    }
    audit_path = output / "stage_d_cached_twarmup_cpu_audit.json"
    _text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    audit_path.write_text(_text, encoding="utf-8")
    report_path = output / "cached_twarmup_then_joint_report.md"
    report_path.write_text(
        "# Stage D cached T warm-up + exact joint report\n\n"
        f"Verdict: `{verdict}`\n\n"
        "Phase A (global 15001–18000) froze D/R and performed no D/R optimizer "
        "updates. Phase A is not full joint training.\n\n"
        "Phase B (global 18001–20000) disabled the static cache and restored the "
        "exact D/R/T joint path. R-local therefore reaches 14000, not 17000.\n\n"
        "`L_depth` stayed disabled; future activation remains global 40000 with "
        "`lambda_depth=0.2`.\n\n"
        f"Final checkpoint SHA-256: `{result['final_checkpoint_sha256']}`.\n\n"
        "Falling loss is not claimed as bird/background semantic separation.\n"
        + ("\nErrors:\n" + "".join(f"- {error}\n" for error in errors) if errors else ""),
        encoding="utf-8",
    )
    print(verdict)
    print(_text)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
