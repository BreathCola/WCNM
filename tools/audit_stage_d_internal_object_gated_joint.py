#!/usr/bin/env python3
"""CPU-only audit for the D-016h internal-object gated D/R/T joint pilot."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_release import sha256_file, validate_geometry_release
from stage_d_training import (
    FORMAL_RELEASE_ID,
    FORMAL_RELEASE_SHA256,
    FORMAL_STEMS,
    INTERNAL_OBJECT_GATED_JOINT_ENDPOINT,
    INTERNAL_OBJECT_GATED_JOINT_NODES,
    INTERNAL_OBJECT_GATED_JOINT_OUTPUT_NAME,
    INTERNAL_OBJECT_GATED_JOINT_SOURCE_SHA256,
)
from utils.internal_object_mask import (
    INTERNAL_OBJECT_SEMANTICS_VERSION,
    REVIEWED_ROLE,
    validate_internal_object_mask_set,
)


VERDICT_PASS = "D016_GATED_JOINT_PASS_AWAITING_USER_REVIEW"
VERDICT_HOLD = "D016_GATED_JOINT_HOLD"
VERDICT_BLOCKED = "D016_GATED_JOINT_BLOCKED"
PLAN_SCHEMA = "rtgs_stage_d_internal_object_gated_joint_operator_plan_v1"
METADATA_SCHEMA = "rtgs_stage_d_internal_object_gated_joint_v1"
TELEMETRY_SCHEMA = "rtgs_stage_d_internal_object_gated_joint_telemetry_v1"
SOURCE = (
    ROOT
    / "output/stage_d_tihubird_c03r8_internal_object_townership_15500_20000_v1"
    / "chkpnt16500.pth"
)
OUTPUT = ROOT / "output" / INTERNAL_OBJECT_GATED_JOINT_OUTPUT_NAME
EXPECTED_INTERNAL_AGGREGATE = (
    "c0e49503e5f5c30b1ab26b7cfd79332ac9c486f35656f1425c9cefea516d4052"
)
REQUIRED_DEBUG = {
    "ground_truth.png",
    "final.png",
    "final_diff.png",
    "final_t_off.png",
    "final_d_direct_off.png",
    "final_r_off.png",
    "inside_color.png",
    "inside_alpha.png",
    "outside_color.png",
    "transmittance_contribution.png",
    "diffuse_contribution.png",
    "reflection_contribution.png",
    "bird_mask.png",
    "internal_base_mask.png",
    "internal_object_union_mask.png",
    "mpos_mignore_mneg.png",
    "float_metrics.json",
}
GEOMETRY_KEYS = {
    "diffuse": ("xyz", "rotation", "scaling_2d"),
    "reflection": ("xyz", "rotation", "scaling_2d"),
    "transmittance": ("xyz", "rotation", "scaling_2d"),
}
APPEARANCE_KEYS = {
    "diffuse": ("base_color_raw", "opacity_raw", "roughness_raw", "f0_raw", "ks_raw"),
    "reflection": ("color_raw", "opacity_raw"),
    "transmittance": ("color_raw", "opacity_raw"),
}


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_scalars(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_scalars(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_scalars(child) for child in value)
    return not isinstance(value, float) or math.isfinite(value)


def _scan_finite(value, path="checkpoint") -> tuple[int, int]:
    tensors = elements = 0
    if torch.is_tensor(value):
        tensors = 1
        elements = value.numel()
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite tensor at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            a, b = _scan_finite(child, f"{path}.{key}")
            tensors += a
            elements += b
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            a, b = _scan_finite(child, f"{path}[{index}]")
            tensors += a
            elements += b
    return tensors, elements


def _load_checkpoint(path: Path, node: int, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append(f"missing checkpoint {node}")
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("format") != "rtgs_stage_d":
            errors.append(f"checkpoint {node} is not rtgs_stage_d")
        if int(checkpoint.get("global_iteration", -1)) != int(node):
            errors.append(f"checkpoint {node} global iteration mismatch")
        _scan_finite(checkpoint)
        return checkpoint
    except Exception as exc:
        errors.append(f"checkpoint {node} unreadable: {exc}")
        return None


def _load_telemetry(output: Path, errors: list[str]) -> list[dict]:
    path = output / "stage_d_telemetry.jsonl"
    if not path.is_file():
        errors.append("missing telemetry jsonl")
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except Exception as exc:
            errors.append(f"telemetry line {line_no} unreadable: {exc}")
            continue
        if row.get("schema") != TELEMETRY_SCHEMA:
            errors.append(f"telemetry line {line_no} schema mismatch")
        if not _finite_scalars(row):
            errors.append(f"telemetry line {line_no} has non-finite scalar")
        rows.append(row)
    iterations = [row.get("global_iteration") for row in rows]
    if len(rows) != 1000:
        errors.append(f"telemetry row count is {len(rows)}, expected 1000")
    if iterations and (iterations[0], iterations[-1]) != (16501, 17500):
        errors.append(
            f"telemetry range is {iterations[0]}--{iterations[-1]}, expected 16501--17500"
        )
    if iterations != list(range(16501, 17501)):
        errors.append("telemetry global iterations are not exactly 16501--17500")
    for row in rows:
        updates = row.get("optimizer_updates_this_step", {})
        if updates.get("diffuse") != 1 or updates.get("reflection") != 1:
            errors.append("telemetry does not record D/R optimizer updates")
            break
        if updates.get("transmittance") != 1:
            errors.append("telemetry does not record T optimizer update")
            break
        event = row.get("topology_event", {})
        if (
            event.get("transmittance_densify_prune_called") is not False
            or event.get("diffuse_delta") != 0
            or event.get("reflection_delta") != 0
            or event.get("transmittance_delta") != 0
        ):
            errors.append("telemetry records topology/count change")
            break
        contract = row.get("t_parameter_group_contract") or {}
        if contract.get("schema") != (
            "rtgs_stage_d_internal_object_gated_joint_parameter_groups_v1"
        ):
            errors.append("telemetry gated-joint parameter contract missing")
            break
    return rows


def _audit_debug_products(output: Path, errors: list[str]) -> None:
    for node in INTERNAL_OBJECT_GATED_JOINT_NODES:
        debug_root = output / "debug" / f"iteration_{node:06d}"
        if not (debug_root / "contact_sheet.png").is_file():
            errors.append(f"missing contact sheet at {node}")
        for stem in FORMAL_STEMS:
            view = debug_root / stem
            missing = sorted(name for name in REQUIRED_DEBUG if not (view / name).is_file())
            if missing:
                errors.append(f"debug {node}/{stem} missing {missing}")
                continue
            try:
                metrics = _json(view / "float_metrics.json")
                if metrics.get("schema") != "rtgs_stage_d_internal_object_gated_joint_float_metrics_v1":
                    errors.append(f"float metrics schema mismatch at {node}/{stem}")
                if not _finite_scalars(metrics):
                    errors.append(f"non-finite float metrics at {node}/{stem}")
            except Exception as exc:
                errors.append(f"float metrics unreadable at {node}/{stem}: {exc}")


def _tensor_delta(first: torch.Tensor, second: torch.Tensor) -> dict:
    delta = torch.abs(second.float() - first.float())
    return {
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
    }


def _compare_checkpoints(source: dict, final: dict, errors: list[str]) -> dict:
    deltas = {}
    for branch, keys in GEOMETRY_KEYS.items():
        for key in keys:
            if not torch.equal(source[branch][key], final[branch][key]):
                errors.append(f"{branch} geometry key changed: {key}")
            deltas[f"{branch}.{key}"] = _tensor_delta(
                source[branch][key], final[branch][key],
            )
    appearance_changed = {}
    for branch, keys in APPEARANCE_KEYS.items():
        branch_changed = False
        for key in keys:
            changed = not torch.equal(source[branch][key], final[branch][key])
            branch_changed = branch_changed or changed
            deltas[f"{branch}.{key}"] = _tensor_delta(
                source[branch][key], final[branch][key],
            )
        appearance_changed[branch] = branch_changed
    if final.get("global_iteration") != INTERNAL_OBJECT_GATED_JOINT_ENDPOINT:
        errors.append("final checkpoint global iteration mismatch")
    if final.get("transmittance_iteration") != 2500:
        errors.append("final checkpoint T-local iteration mismatch")
    if final.get("reflection_iteration") != 13000:
        errors.append("final checkpoint R-local iteration mismatch")
    return {
        "deltas": deltas,
        "appearance_changed": appearance_changed,
    }


def _audit(args: argparse.Namespace) -> tuple[dict, str]:
    output = args.output.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    holds: list[str] = []
    if output.name != INTERNAL_OBJECT_GATED_JOINT_OUTPUT_NAME:
        errors.append("output directory name is not the D-016h gated-joint output name")
    if not output.is_dir():
        errors.append(f"output is missing: {output}")

    plan = {}
    if args.operator_plan:
        try:
            plan = _json(args.operator_plan.resolve())
            if plan.get("schema") != PLAN_SCHEMA:
                errors.append("operator plan schema mismatch")
            if Path(plan.get("output", "")).resolve() != output:
                errors.append("operator plan output path mismatch")
            if plan.get("execute") is not True:
                warnings.append("operator plan was not captured from --execute invocation")
        except Exception as exc:
            errors.append(f"operator plan unreadable: {exc}")

    if sha256_file(SOURCE) != INTERNAL_OBJECT_GATED_JOINT_SOURCE_SHA256:
        errors.append("16,500 source checkpoint SHA-256 mismatch")

    release = {}
    try:
        release = validate_geometry_release(args.geometry_manifest)
        if release.get("geometry_release_id") != FORMAL_RELEASE_ID:
            errors.append("Stage-C release ID mismatch")
        if release.get("aggregate_sha256") != FORMAL_RELEASE_SHA256:
            errors.append("Stage-C aggregate mismatch")
    except Exception as exc:
        errors.append(f"Stage-C release validation failed: {exc}")

    internal = {}
    try:
        internal = validate_internal_object_mask_set(
            args.scene.resolve(), args.images, args.internal_object_manifest,
        )
        if internal.get("role") != REVIEWED_ROLE:
            errors.append("internal-object role mismatch")
        if internal.get("human_status") != "accepted":
            errors.append("internal-object human status mismatch")
        if internal.get("internal_object_semantics_version") != INTERNAL_OBJECT_SEMANTICS_VERSION:
            errors.append("internal-object semantic version mismatch")
        if internal.get("aggregate_sha256") != EXPECTED_INTERNAL_AGGREGATE:
            errors.append("internal-object aggregate mismatch")
    except Exception as exc:
        errors.append(f"internal-object release validation failed: {exc}")

    metadata = {}
    try:
        metadata = _json(output / "internal_object_gated_joint_metadata.json")
        if metadata.get("schema") != METADATA_SCHEMA:
            errors.append("gated-joint metadata schema mismatch")
        if metadata.get("global") != [16501, 17500]:
            errors.append("metadata global range mismatch")
        if metadata.get("transmittance_local") != [1501, 2500]:
            errors.append("metadata T-local range mismatch")
        if metadata.get("diffuse_optimizer_updates") != 1000:
            errors.append("metadata D update count mismatch")
        if metadata.get("reflection_optimizer_updates") != 1000:
            errors.append("metadata R update count mismatch")
        if metadata.get("transmittance_optimizer_updates") != 1000:
            errors.append("metadata T update count mismatch")
        if metadata.get("topology_updates_allowed") is not False:
            errors.append("metadata topology gate mismatch")
        for key in ("t_reinitialization", "transferred_d_selection_rerun", "random_fill_rerun"):
            if metadata.get(key) is not False:
                errors.append(f"metadata says {key} is not false")
    except Exception as exc:
        errors.append(f"metadata unreadable: {exc}")

    rows = _load_telemetry(output, errors) if output.is_dir() else []
    _audit_debug_products(output, errors) if output.is_dir() else None

    checkpoints = {}
    checkpoint_objs = {}
    for node in INTERNAL_OBJECT_GATED_JOINT_NODES:
        checkpoint = _load_checkpoint(output / f"chkpnt{node}.pth", node, errors)
        if checkpoint is not None:
            checkpoint_objs[node] = checkpoint
            checkpoints[str(node)] = {
                "sha256": sha256_file(output / f"chkpnt{node}.pth"),
                "diffuse_count": int(checkpoint["diffuse"]["xyz"].shape[0]),
                "reflection_count": int(checkpoint["reflection"]["xyz"].shape[0]),
                "transmittance_count": int(checkpoint["transmittance"]["xyz"].shape[0]),
            }
            if int(checkpoint["transmittance"]["xyz"].shape[0]) != 4096:
                errors.append(f"checkpoint {node} T count is not 4096")

    comparison = {}
    source = checkpoint_objs.get(16500)
    final = checkpoint_objs.get(INTERNAL_OBJECT_GATED_JOINT_ENDPOINT)
    if source and final:
        comparison = _compare_checkpoints(source, final, errors)
        changed = comparison.get("appearance_changed", {})
        for branch, did_change in changed.items():
            if not did_change:
                warnings.append(f"{branch} trainable appearance did not change")

    if not errors:
        holds.append("AWAITING_USER_VISUAL_REVIEW: engineering PASS does not accept visual quality")
        holds.append("NO_STAGE_E_AUTHORIZATION")
    verdict = VERDICT_BLOCKED if errors else VERDICT_PASS
    result = {
        "schema": "rtgs_stage_d_internal_object_gated_joint_cpu_audit_v1",
        "verdict": verdict,
        "technical_run_complete": not errors,
        "semantic_separation_claimed": False,
        "stage_d_final_acceptance": False,
        "stage_e_authorized": False,
        "errors": errors,
        "warnings": warnings,
        "holds": holds,
        "output": str(output),
        "source_checkpoint": str(SOURCE),
        "source_checkpoint_sha256": INTERNAL_OBJECT_GATED_JOINT_SOURCE_SHA256,
        "stage_c_release": release,
        "internal_object_release": {
            "manifest": str(Path(args.internal_object_manifest).resolve()),
            "manifest_sha256": internal.get("manifest_file_sha256"),
            "aggregate_sha256": internal.get("aggregate_sha256"),
            "role": internal.get("role"),
            "human_status": internal.get("human_status"),
            "semantic_version": internal.get("internal_object_semantics_version"),
        },
        "operator_plan": plan,
        "metadata": metadata,
        "telemetry_rows": len(rows),
        "telemetry_range": [16501, 17500],
        "review_nodes": list(INTERNAL_OBJECT_GATED_JOINT_NODES),
        "checkpoints": checkpoints,
        "checkpoint_comparison": comparison,
    }
    return result, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--operator-plan", type=Path)
    parser.add_argument(
        "--geometry-manifest",
        type=Path,
        default=ROOT / "geometry_releases/stage_c_geometry_release_v1.json",
    )
    parser.add_argument("--scene", type=Path, default=ROOT / "data/TiHuBird")
    parser.add_argument("--images", default="images")
    parser.add_argument(
        "--internal-object-manifest",
        type=Path,
        default=ROOT / "data/TiHuBird/internal_object_masks_reviewed_v3/manifest.json",
    )
    args = parser.parse_args()
    result, verdict = _audit(args)
    output = args.output.resolve()
    if output.is_dir():
        audit_path = output / "internal_object_gated_joint_cpu_audit.json"
        report_path = output / "internal_object_gated_joint_cpu_audit.md"
        audit_path.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        report_path.write_text(
            "# D-016h internal-object gated joint CPU audit\n\n"
            f"Verdict: `{verdict}`\n\n"
            "This verdict is an engineering post-run result only. It does not "
            "accept visual quality, accept Stage D, or authorize Stage E.\n\n"
            + ("\nErrors:\n" + "".join(f"- {error}\n" for error in result["errors"]) if result["errors"] else "")
            + ("\nWarnings:\n" + "".join(f"- {warning}\n" for warning in result["warnings"]) if result["warnings"] else "")
            + ("\nHolds:\n" + "".join(f"- {hold}\n" for hold in result["holds"]) if result["holds"] else ""),
            encoding="utf-8",
        )
    print(verdict)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 2 if verdict == VERDICT_BLOCKED else 0


if __name__ == "__main__":
    raise SystemExit(main())
