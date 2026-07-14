#!/usr/bin/env python3
"""CPU-only final audit for the D-016 internal-object T-ownership pilot."""

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
    FORMAL_SOURCE_SHA256,
    FORMAL_STEMS,
    INTERNAL_OBJECT_ENDPOINT,
    INTERNAL_OBJECT_NODES,
    INTERNAL_OBJECT_OUTPUT_NAME,
)
from utils.internal_object_mask import (
    INTERNAL_OBJECT_SEMANTICS_VERSION,
    REVIEWED_ROLE,
    validate_internal_object_mask_set,
)
from utils.stage_d_static_cache import state_sha256


VERDICT_PASS = "D016_PILOT_PASS_AWAITING_USER_REVIEW"
VERDICT_HOLD = "D016_PILOT_HOLD"
VERDICT_BLOCKED = "D016_PILOT_BLOCKED"
TELEMETRY_SCHEMA = "rtgs_stage_d_internal_object_townership_telemetry_v1"
METADATA_SCHEMA = "rtgs_stage_d_internal_object_townership_pilot_v1"
PLAN_SCHEMA = "rtgs_stage_d_internal_object_operator_plan_v1"
REQUIRED_DEBUG = {
    "ground_truth.png",
    "final.png",
    "inside_color.png",
    "inside_alpha.png",
    "outside_color.png",
    "transparent_mask.png",
    "final_r_off.png",
    "final_t_off.png",
    "transmittance_metadata.json",
}
REQUIRED_SPLIT_KEYS = {
    "bird_pixel_count",
    "internal_base_pixel_count",
    "union_pixel_count",
    "mneg_pixel_count",
    "bird_ain_mean",
    "internal_base_ain_mean",
    "union_ain_mean",
    "mneg_ain_mean",
    "bird_ain_p05",
    "internal_base_ain_p95",
    "union_ain_above_alpha_floor_ratio",
    "mneg_cout_energy",
    "outside_union_cout_energy",
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


def _branch_hash(checkpoint: dict, branch: str) -> str:
    if branch not in checkpoint:
        raise KeyError(f"checkpoint lacks {branch} branch")
    return state_sha256(checkpoint[branch])


def _optimizer_hash(checkpoint: dict, branch: str) -> str:
    state = checkpoint.get(branch, {}).get("optimizer")
    if state is None:
        raise KeyError(f"checkpoint lacks {branch} optimizer")
    return state_sha256(state)


def _load_checkpoint(path: Path, node: int, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append(f"missing checkpoint {node}")
        return None
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("format") != "rtgs_stage_d":
        errors.append(f"checkpoint {node} format mismatch")
    expected = (node, 12000, node - 15000)
    actual = (
        checkpoint.get("global_iteration"),
        checkpoint.get("reflection_iteration"),
        checkpoint.get("transmittance_iteration"),
    )
    if actual != expected:
        errors.append(f"checkpoint {node} iteration tuple {actual} != {expected}")
    if checkpoint.get("geometry_release_aggregate_sha256") != FORMAL_RELEASE_SHA256:
        errors.append(f"checkpoint {node} Stage-C aggregate mismatch")
    config = checkpoint.get("config", {})
    if config.get("source_stage_b_checkpoint_sha256") != FORMAL_SOURCE_SHA256:
        errors.append(f"checkpoint {node} source identity mismatch")
    if config.get("stage_d_depth_start_iteration") != 40000:
        errors.append(f"checkpoint {node} L_depth schedule mismatch")
    try:
        _scan_finite(checkpoint)
    except Exception as exc:
        errors.append(str(exc))
    for branch in ("diffuse", "reflection", "transmittance"):
        state = checkpoint.get(branch, {})
        count = state.get("xyz")
        if not torch.is_tensor(count):
            errors.append(f"checkpoint {node} lacks {branch}.xyz")
        elif branch == "transmittance" and int(count.shape[0]) != 4096:
            errors.append(f"checkpoint {node} T count is not 4096")
    return checkpoint


def _audit(args: argparse.Namespace) -> tuple[dict, str]:
    output = args.output.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if output.name != INTERNAL_OBJECT_OUTPUT_NAME:
        errors.append("output directory name is not the D-016 internal-object pilot name")
    if not output.is_dir():
        errors.append(f"pilot output is missing: {output}")

    plan = _json(args.operator_plan)
    if plan.get("schema") != PLAN_SCHEMA:
        errors.append("operator plan schema mismatch")
    if plan.get("execute") is not True:
        errors.append("operator plan is not an executed pilot plan")
    if plan.get("output") != str(output):
        errors.append("operator plan output path mismatch")
    if plan.get("source_sha256") != FORMAL_SOURCE_SHA256:
        errors.append("operator plan source hash mismatch")
    if plan.get("release_aggregate_sha256") != FORMAL_RELEASE_SHA256:
        errors.append("operator plan Stage-C aggregate mismatch")
    if "proposal" in str(plan.get("internal_object_manifest_sha256", "")).lower():
        errors.append("operator plan appears to reference a proposal")

    release = validate_geometry_release(args.geometry_manifest)
    if release.get("geometry_release_id") != FORMAL_RELEASE_ID:
        errors.append("Stage-C release ID mismatch")
    if release.get("aggregate_sha256") != FORMAL_RELEASE_SHA256:
        errors.append("Stage-C aggregate mismatch")

    internal = validate_internal_object_mask_set(
        args.scene.resolve(), args.images, args.internal_object_manifest,
    )
    if internal.get("role") != REVIEWED_ROLE:
        errors.append("internal-object role mismatch")
    if internal.get("human_status") != "accepted":
        errors.append("internal-object human status mismatch")
    if internal.get("internal_object_semantics_version") != INTERNAL_OBJECT_SEMANTICS_VERSION:
        errors.append("internal-object semantic version mismatch")
    if internal.get("aggregate_sha256") != args.expected_internal_aggregate:
        errors.append("internal-object aggregate mismatch")

    metadata_path = output / "internal_object_townership_metadata.json"
    metadata = _json(metadata_path) if metadata_path.is_file() else {}
    if not metadata:
        errors.append("missing internal_object_townership_metadata.json")
    elif metadata.get("schema") != METADATA_SCHEMA:
        errors.append("metadata schema mismatch")
    else:
        if metadata.get("pilot_global") != [15001, INTERNAL_OBJECT_ENDPOINT]:
            errors.append("metadata pilot global range mismatch")
        if metadata.get("diffuse_optimizer_updates") != 0:
            errors.append("metadata says Diffuse was updated")
        if metadata.get("reflection_optimizer_updates") != 0:
            errors.append("metadata says Reflection was updated")
        if metadata.get("transmittance_optimizer_updates") != 500:
            errors.append("metadata T update count mismatch")
        if metadata.get("expected_t_count") != 4096:
            errors.append("metadata T count mismatch")
        init = metadata.get("actual_transmittance_initialization", {})
        filter_meta = init.get("internal_object_filter", {})
        for key in (
            "pre_filter_D_indices_sha256",
            "selected_D_indices_sha256",
            "rejected_D_indices_sha256",
            "random_fill_count",
        ):
            if key not in filter_meta:
                errors.append(f"T initialization lacks {key}")

    telemetry_path = output / "stage_d_telemetry.jsonl"
    rows = []
    if not telemetry_path.is_file():
        errors.append("missing stage_d_telemetry.jsonl")
    else:
        rows = [
            json.loads(line)
            for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [row.get("global_iteration") for row in rows] != list(range(15001, 15501)):
            errors.append("telemetry is not exactly global 15001--15500")
        for row in rows:
            step = row.get("global_iteration")
            if row.get("schema") != TELEMETRY_SCHEMA:
                errors.append(f"telemetry schema mismatch at {step}")
                break
            if row.get("nonfinite_count") != 0 or not _finite_scalars(row):
                errors.append(f"non-finite telemetry at {step}")
                break
            if row.get("loss", {}).get("lambda_depth_enabled") is not False:
                errors.append(f"L_depth enabled at {step}")
                break
            if row.get("optimizer_updates_this_step") != {
                "diffuse": 0, "reflection": 0, "transmittance": 1,
            }:
                errors.append(f"optimizer update isolation mismatch at {step}")
                break
            if row.get("counts", {}).get("transmittance") != 4096:
                errors.append(f"T count changed at {step}")
                break
            topology = row.get("topology_event", {})
            if topology.get("transmittance_densify_prune_called") or any(
                topology.get(key) != 0
                for key in ("diffuse_delta", "reflection_delta", "transmittance_delta")
            ):
                errors.append(f"topology changed at {step}")
                break
            split = row.get("internal_object_metrics", {})
            missing_split = sorted(REQUIRED_SPLIT_KEYS - set(split))
            if missing_split:
                errors.append(f"missing split internal-object metrics at {step}: {missing_split}")
                break

    checkpoints = {}
    for node in INTERNAL_OBJECT_NODES:
        checkpoint = _load_checkpoint(output / f"chkpnt{node}.pth", node, errors)
        if checkpoint is not None:
            checkpoints[node] = checkpoint
        for branch in ("diffuse", "reflection", "transmittance"):
            ply = output / "point_cloud" / branch / f"iteration_{node}" / "point_cloud.ply"
            if not ply.is_file():
                errors.append(f"missing {branch} PLY at {node}")

    if INTERNAL_OBJECT_NODES[0] in checkpoints and INTERNAL_OBJECT_NODES[-1] in checkpoints:
        first = checkpoints[INTERNAL_OBJECT_NODES[0]]
        final = checkpoints[INTERNAL_OBJECT_NODES[-1]]
        for branch in ("diffuse", "reflection"):
            if _branch_hash(first, branch) != _branch_hash(final, branch):
                errors.append(f"{branch} parameters changed")
            if _optimizer_hash(first, branch) != _optimizer_hash(final, branch):
                errors.append(f"{branch} optimizer state changed")
        if _branch_hash(first, "transmittance") == _branch_hash(final, "transmittance"):
            errors.append("T parameters did not change")

    for node in INTERNAL_OBJECT_NODES:
        debug_root = output / "debug" / f"iteration_{node:06d}"
        if not (debug_root / "contact_sheet.png").is_file():
            errors.append(f"missing contact sheet at {node}")
        for stem in FORMAL_STEMS:
            view = debug_root / stem
            missing = sorted(name for name in REQUIRED_DEBUG if not (view / name).is_file())
            if missing:
                errors.append(f"debug {node}/{stem} missing {missing}")

    verdict = VERDICT_BLOCKED if errors else VERDICT_PASS
    result = {
        "schema": "rtgs_stage_d_internal_object_townership_cpu_audit_v1",
        "verdict": verdict,
        "technical_run_complete": not errors,
        "semantic_separation_claimed": False,
        "stage_d_accepted": False,
        "stage_e_authorized": False,
        "errors": errors,
        "warnings": warnings,
        "output": str(output),
        "operator_plan": str(args.operator_plan.resolve()),
        "source_sha256": FORMAL_SOURCE_SHA256,
        "stage_c_release": release,
        "internal_object_release": {
            "manifest": str(Path(args.internal_object_manifest).resolve()),
            "manifest_sha256": internal.get("manifest_file_sha256"),
            "aggregate_sha256": internal.get("aggregate_sha256"),
            "role": internal.get("role"),
            "human_status": internal.get("human_status"),
            "semantic_version": internal.get("internal_object_semantics_version"),
        },
        "telemetry_rows": len(rows),
        "telemetry_range": [15001, 15500],
        "required_nodes": list(INTERNAL_OBJECT_NODES),
        "review_stems": list(FORMAL_STEMS),
    }
    return result, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-plan", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, default=ROOT / "geometry_releases/stage_c_geometry_release_v1.json")
    parser.add_argument("--scene", type=Path, default=ROOT / "data/TiHuBird")
    parser.add_argument("--images", default="images")
    parser.add_argument("--internal-object-manifest", type=Path, default=ROOT / "data/TiHuBird/internal_object_masks_reviewed_v3/manifest.json")
    parser.add_argument("--expected-internal-aggregate", default="c0e49503e5f5c30b1ab26b7cfd79332ac9c486f35656f1425c9cefea516d4052")
    args = parser.parse_args()
    result, verdict = _audit(args)
    output = args.output.resolve()
    if output.is_dir():
        audit_path = output / "internal_object_townership_cpu_audit.json"
        report_path = output / "internal_object_townership_cpu_audit.md"
        audit_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        report_path.write_text(
            "# D-016 internal-object T-ownership CPU audit\n\n"
            f"Verdict: `{verdict}`\n\n"
            "This verdict is an engineering completeness result only. It does not "
            "claim semantic separation, Stage D acceptance, full training authorization, "
            "or Stage E authorization.\n\n"
            + ("\nErrors:\n" + "".join(f"- {error}\n" for error in result["errors"]) if result["errors"] else ""),
            encoding="utf-8",
        )
    print(verdict)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if verdict != VERDICT_BLOCKED else 2


if __name__ == "__main__":
    raise SystemExit(main())
