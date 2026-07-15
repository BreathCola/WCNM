#!/usr/bin/env python3
"""CPU-only audit for D-016 internal-object T-only continuation to 20,000."""

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
    INTERNAL_OBJECT_TO_20000_ENDPOINT,
    INTERNAL_OBJECT_TO_20000_NODES,
    INTERNAL_OBJECT_TO_20000_OUTPUT_NAME,
    INTERNAL_OBJECT_TO_20000_SOURCE_SHA256,
)
from utils.internal_object_mask import (
    INTERNAL_OBJECT_SEMANTICS_VERSION,
    REVIEWED_ROLE,
    validate_internal_object_mask_set,
)
from utils.stage_d_static_cache import state_sha256


VERDICT_PASS = "D016_TO_20000_PASS_AWAITING_USER_REVIEW"
VERDICT_HOLD = "D016_TO_20000_HOLD"
VERDICT_BLOCKED = "D016_TO_20000_BLOCKED"
TELEMETRY_SCHEMA = "rtgs_stage_d_internal_object_townership_to_20000_telemetry_v1"
METADATA_SCHEMA = "rtgs_stage_d_internal_object_townership_to_20000_v1"
POSTHOC_SCHEMA = "rtgs_stage_d_internal_object_townership_to_20000_posthoc_review_v1"
PLAN_SCHEMA = "rtgs_stage_d_internal_object_townership_to_20000_operator_plan_v1"
REVIEW_NODES = (15500,) + tuple(INTERNAL_OBJECT_TO_20000_NODES)
SOURCE = ROOT / "output/stage_d_tihubird_c03r8_internal_object_townership_pilot_v1/chkpnt15500.pth"
PILOT_OUTPUT = ROOT / "output/stage_d_tihubird_c03r8_internal_object_townership_pilot_v1"
PILOT_AUDIT = PILOT_OUTPUT / "internal_object_townership_cpu_audit.json"
EXPECTED_INTERNAL_AGGREGATE = "c0e49503e5f5c30b1ab26b7cfd79332ac9c486f35656f1425c9cefea516d4052"
REQUIRED_DEBUG = {
    "ground_truth.png",
    "final.png",
    "final_diff.png",
    "final_t_off.png",
    "final_r_off.png",
    "inside_color.png",
    "inside_alpha.png",
    "inside_depth.png",
    "outside_color.png",
    "outside_alpha.png",
    "outside_depth.png",
    "transmittance_contribution.png",
    "transparent_mask.png",
    "bird_mask_overlay.png",
    "internal_base_mask_overlay.png",
    "union_mask_overlay.png",
    "mpos_mignore_mneg.png",
    "float_metrics.json",
}
TREND_KEYS = (
    "loss.rgb",
    "loss.l_object",
    "internal_object_metrics.bird_ain_mean",
    "internal_object_metrics.bird_ain_p50",
    "internal_object_metrics.bird_ain_p95",
    "internal_object_metrics.bird_ain_above_alpha_floor_ratio",
    "internal_object_metrics.internal_base_ain_mean",
    "internal_object_metrics.internal_base_ain_p50",
    "internal_object_metrics.internal_base_ain_p95",
    "internal_object_metrics.internal_base_ain_above_alpha_floor_ratio",
    "internal_object_metrics.union_ain_mean",
    "internal_object_metrics.union_ain_p50",
    "internal_object_metrics.union_ain_p95",
    "internal_object_metrics.union_ain_above_alpha_floor_ratio",
    "internal_object_metrics.mneg_ain_mean",
    "internal_object_metrics.ain_negative_spill_ge_0_10",
    "internal_object_metrics.bird_cin_energy",
    "internal_object_metrics.internal_base_cin_energy",
    "internal_object_metrics.mneg_cout_energy",
)


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


def _checkpoint_path(output: Path, node: int) -> Path:
    if node == 15500:
        return SOURCE
    return output / f"chkpnt{node}.pth"


def _load_checkpoint(path: Path, node: int, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append(f"missing checkpoint {node}")
        return None
    try:
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
            errors.append(f"checkpoint {node} original Stage-B source mismatch")
        if config.get("stage_d_depth_start_iteration") != 40000:
            errors.append(f"checkpoint {node} L_depth schedule mismatch")
        tensors, elements = _scan_finite(checkpoint)
        if int(checkpoint.get("transmittance", {}).get("xyz").shape[0]) != 4096:
            errors.append(f"checkpoint {node} T count is not 4096")
        checkpoint["_audit_tensor_count"] = tensors
        checkpoint["_audit_element_count"] = elements
        return checkpoint
    except Exception as exc:
        errors.append(f"checkpoint {node} unreadable: {exc}")
        return None


def _load_telemetry(output: Path, errors: list[str]) -> list[dict]:
    path = output / "stage_d_telemetry.jsonl"
    if not path.is_file():
        errors.append("missing stage_d_telemetry.jsonl")
        return []
    rows = []
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        errors.append(f"telemetry unreadable: {exc}")
        return []
    expected = list(range(15501, INTERNAL_OBJECT_TO_20000_ENDPOINT + 1))
    if [row.get("global_iteration") for row in rows] != expected:
        errors.append("telemetry is not exactly continuous global 15501--20000")
    if [row.get("transmittance_local_iteration") for row in rows] != list(range(501, 5001)):
        errors.append("telemetry is not exactly continuous T-local 501--5000")
    for row in rows:
        step = row.get("global_iteration")
        if row.get("schema") != TELEMETRY_SCHEMA:
            errors.append(f"telemetry schema mismatch at {step}")
            break
        if row.get("training_phase") != "internal_object_townership_t_only":
            errors.append(f"training phase mismatch at {step}")
            break
        if row.get("optimizer_updates_this_step") != {"diffuse": 0, "reflection": 0, "transmittance": 1}:
            errors.append(f"optimizer update isolation mismatch at {step}")
            break
        if row.get("counts", {}).get("transmittance") != 4096:
            errors.append(f"T count changed at {step}")
            break
        topology = row.get("topology_event", {})
        if topology.get("transmittance_densify_prune_called") or any(
            topology.get(key) != 0 for key in ("diffuse_delta", "reflection_delta", "transmittance_delta")
        ):
            errors.append(f"topology changed at {step}")
            break
        if row.get("loss", {}).get("lambda_depth_enabled") is not False:
            errors.append(f"L_depth enabled at {step}")
            break
        if row.get("nonfinite_count") != 0 or not _finite_scalars(row):
            errors.append(f"non-finite telemetry at {step}")
            break
    return rows


def _get_nested(row: dict, dotted: str):
    value = row
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _trend_table(rows: list[dict]) -> list[dict]:
    by_iteration = {row.get("global_iteration"): row for row in rows}
    result = []
    for node in REVIEW_NODES:
        if node == 15500:
            continue
        row = by_iteration.get(node)
        if not row:
            continue
        item = {"global_iteration": node}
        for key in TREND_KEYS:
            item[key.replace(".", "_")] = _get_nested(row, key)
        result.append(item)
    return result


def _recommended_candidates(trends: list[dict]) -> dict:
    if not trends:
        return {}
    def best_min(key):
        valid = [row for row in trends if isinstance(row.get(key), (int, float))]
        return min(valid, key=lambda row: row[key])["global_iteration"] if valid else None
    def best_max(key):
        valid = [row for row in trends if isinstance(row.get(key), (int, float))]
        return max(valid, key=lambda row: row[key])["global_iteration"] if valid else None
    return {
        "best_rgb": best_min("loss_rgb"),
        "best_object_ownership": best_max("internal_object_metrics_union_ain_above_alpha_floor_ratio"),
        "best_leakage_tradeoff": best_min("internal_object_metrics_mneg_ain_mean"),
        "endpoint_20000": 20000,
        "note": "Engineering metrics only; final checkpoint selection requires human visual review.",
    }


def _audit_posthoc(output: Path, errors: list[str]) -> dict:
    posthoc = output / "posthoc_review"
    manifest_path = posthoc / "materialization_manifest.json"
    if not manifest_path.is_file():
        errors.append("missing posthoc review materialization manifest")
        return {}
    try:
        manifest = _json(manifest_path)
    except Exception as exc:
        errors.append(f"cannot read posthoc materialization manifest: {exc}")
        return {}
    if manifest.get("schema") != POSTHOC_SCHEMA:
        errors.append("posthoc materialization schema mismatch")
    source_contract = manifest.get("node_source_contract", {})
    if source_contract.get("deterministic_fresh_replay") is not False:
        errors.append("posthoc materializer must not use deterministic fresh replay")
    for key in ("t_reinitialization", "transferred_d_selection_rerun", "random_fill_rerun"):
        if source_contract.get(key) is not False:
            errors.append(f"posthoc materializer violated {key}=false")
    if manifest.get("no_optimizer_execution") is not True:
        errors.append("posthoc materialization must record no optimizer execution")
    if manifest.get("no_backward") is not True:
        errors.append("posthoc materialization must record no backward")
    if manifest.get("no_checkpoint_write") is not True:
        errors.append("posthoc materialization must record no checkpoint writes")
    if manifest.get("review_nodes") != list(REVIEW_NODES):
        errors.append("posthoc review nodes mismatch")
    for name in (
        "overview_final_15500_20000.png",
        "overview_cin_15500_20000.png",
        "overview_ain_15500_20000.png",
    ):
        if not (posthoc / name).is_file():
            errors.append(f"missing overview {name}")
    for node in REVIEW_NODES:
        debug_root = posthoc / "debug" / f"iteration_{node:06d}"
        if not (debug_root / "contact_sheet.png").is_file():
            errors.append(f"missing contact sheet at {node}")
        if not (debug_root / "semantic_repair_contact_sheet.png").is_file():
            errors.append(f"missing semantic repair contact sheet at {node}")
        for stem in FORMAL_STEMS:
            view = debug_root / stem
            missing = sorted(name for name in REQUIRED_DEBUG if not (view / name).is_file())
            if missing:
                errors.append(f"debug {node}/{stem} missing {missing}")
                continue
            try:
                metrics = _json(view / "float_metrics.json")
                if metrics.get("schema") != "rtgs_stage_d_internal_object_townership_to_20000_float_metrics_v1":
                    errors.append(f"float metrics schema mismatch at {node}/{stem}")
                if not _finite_scalars(metrics):
                    errors.append(f"non-finite float metrics at {node}/{stem}")
            except Exception as exc:
                errors.append(f"float metrics unreadable at {node}/{stem}: {exc}")
    return manifest


def _audit(args: argparse.Namespace) -> tuple[dict, str]:
    output = args.output.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    holds: list[str] = []
    if output.name != INTERNAL_OBJECT_TO_20000_OUTPUT_NAME:
        errors.append("output directory name is not the D-016 to-20000 output name")
    if not output.is_dir():
        errors.append(f"output is missing: {output}")

    plan = {}
    if args.operator_plan:
        try:
            plan = _json(args.operator_plan.resolve())
            if plan.get("schema") != PLAN_SCHEMA:
                errors.append("operator plan schema mismatch")
            if plan.get("output") != str(output):
                errors.append("operator plan output path mismatch")
        except Exception as exc:
            errors.append(f"operator plan unreadable: {exc}")

    if sha256_file(SOURCE) != INTERNAL_OBJECT_TO_20000_SOURCE_SHA256:
        errors.append("15,500 source checkpoint SHA-256 mismatch")
    pilot_audit = {}
    try:
        pilot_audit = _json(PILOT_AUDIT)
        if pilot_audit.get("verdict") != "D016_PILOT_PASS_AWAITING_USER_REVIEW":
            errors.append("pilot audit verdict mismatch")
        if pilot_audit.get("technical_run_complete") is not True:
            errors.append("pilot audit technical_run_complete mismatch")
        if pilot_audit.get("errors") != []:
            errors.append("pilot audit errors are not empty")
    except Exception as exc:
        errors.append(f"pilot audit unreadable: {exc}")

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
        metadata = _json(output / "internal_object_townership_to_20000_metadata.json")
        if metadata.get("schema") != METADATA_SCHEMA:
            errors.append("to-20000 metadata schema mismatch")
        if metadata.get("global") != [15501, 20000]:
            errors.append("metadata global range mismatch")
        if metadata.get("transmittance_local") != [501, 5000]:
            errors.append("metadata T-local range mismatch")
        if metadata.get("diffuse_optimizer_updates") != 0 or metadata.get("reflection_optimizer_updates") != 0:
            errors.append("metadata says D/R was updated")
        if metadata.get("transmittance_optimizer_updates") != 4500:
            errors.append("metadata T update count mismatch")
        if metadata.get("expected_t_count") != 4096:
            errors.append("metadata T count mismatch")
        if metadata.get("t_reinitialization") is not False:
            errors.append("metadata says T was reinitialized")
        if metadata.get("transferred_d_selection_rerun") is not False:
            errors.append("metadata says transferred-D selection reran")
        if metadata.get("random_fill_rerun") is not False:
            errors.append("metadata says random fill reran")
    except Exception as exc:
        errors.append(f"metadata unreadable: {exc}")

    rows = _load_telemetry(output, errors)
    trends = _trend_table(rows)
    posthoc = _audit_posthoc(output, errors)

    checkpoints = {}
    checkpoint_objs = {}
    for node in REVIEW_NODES:
        checkpoint = _load_checkpoint(_checkpoint_path(output, node), node, errors)
        if checkpoint is not None:
            checkpoint_objs[node] = checkpoint
            branch_hashes = {}
            optimizer_hashes = {}
            for branch in ("diffuse", "reflection", "transmittance"):
                try:
                    branch_hashes[branch] = _branch_hash(checkpoint, branch)
                    optimizer_hashes[branch] = _optimizer_hash(checkpoint, branch)
                except Exception as exc:
                    errors.append(f"checkpoint {node} {branch} hash failed: {exc}")
            checkpoints[str(node)] = {
                "sha256": sha256_file(_checkpoint_path(output, node)),
                "branch_hashes": branch_hashes,
                "optimizer_hashes": optimizer_hashes,
                "tensor_count": checkpoint.get("_audit_tensor_count"),
                "element_count": checkpoint.get("_audit_element_count"),
                "t_count": int(checkpoint["transmittance"]["xyz"].shape[0]),
            }
        if node != 15500:
            for branch in ("diffuse", "reflection", "transmittance"):
                ply = output / "point_cloud" / branch / f"iteration_{node}" / "point_cloud.ply"
                if not ply.is_file():
                    errors.append(f"missing {branch} PLY at {node}")

    source = checkpoint_objs.get(15500)
    final = checkpoint_objs.get(20000)
    if source and final:
        for branch in ("diffuse", "reflection"):
            if _branch_hash(source, branch) != _branch_hash(final, branch):
                errors.append(f"{branch} parameters changed between 15500 and 20000")
            if _optimizer_hash(source, branch) != _optimizer_hash(final, branch):
                errors.append(f"{branch} optimizer state changed between 15500 and 20000")
        if _branch_hash(source, "transmittance") == _branch_hash(final, "transmittance"):
            errors.append("T parameters did not change")
        try:
            source_t_opt = source["transmittance"]["optimizer"]
            final_t_opt = final["transmittance"]["optimizer"]
            if state_sha256(source_t_opt) == state_sha256(final_t_opt):
                errors.append("T optimizer state did not change")
            if final.get("transmittance_iteration") != 5000:
                errors.append("final T-local iteration is not 5000")
        except Exception as exc:
            errors.append(f"T optimizer continuity check failed: {exc}")

    manifest_hashes = posthoc.get("immutable_files_before_after", {}) if isinstance(posthoc, dict) else {}
    for path, pair in manifest_hashes.items():
        p = Path(path)
        if p.is_file() and pair.get("before") != sha256_file(p):
            errors.append(f"immutable file changed after materialization: {path}")
        if pair.get("before") != pair.get("after"):
            errors.append(f"materializer observed immutable file mutation: {path}")

    if not errors:
        holds.append("AWAITING_USER_VISUAL_REVIEW: engineering PASS does not select a final checkpoint")
        holds.append("NO_JOINT_TRAINING_AUTHORIZATION: D/R restoration remains unauthorized")
        holds.append("NO_STAGE_E_AUTHORIZATION")
    verdict = VERDICT_BLOCKED if errors else VERDICT_PASS
    result = {
        "schema": "rtgs_stage_d_internal_object_townership_to_20000_cpu_audit_v1",
        "verdict": verdict,
        "technical_run_complete": not errors,
        "semantic_separation_claimed": False,
        "stage_d_final_acceptance": False,
        "joint_training_authorized": False,
        "stage_e_authorized": False,
        "errors": errors,
        "warnings": warnings,
        "holds": holds,
        "output": str(output),
        "source_checkpoint": str(SOURCE),
        "source_checkpoint_sha256": INTERNAL_OBJECT_TO_20000_SOURCE_SHA256,
        "pilot_audit_identity": {
            "path": str(PILOT_AUDIT),
            "verdict": pilot_audit.get("verdict"),
            "technical_run_complete": pilot_audit.get("technical_run_complete"),
        },
        "stage_c_release": release,
        "internal_object_release": {
            "manifest": str(Path(args.internal_object_manifest).resolve()),
            "manifest_sha256": internal.get("manifest_file_sha256"),
            "aggregate_sha256": internal.get("aggregate_sha256"),
            "role": internal.get("role"),
            "human_status": internal.get("human_status"),
            "semantic_version": internal.get("internal_object_semantics_version"),
        },
        "git_execution_head": metadata.get("source", {}).get("git_head") or metadata.get("repository_head"),
        "output_name": output.name,
        "telemetry_rows": len(rows),
        "telemetry_range": [15501, 20000],
        "transmittance_local_range": [501, 5000],
        "review_nodes": list(REVIEW_NODES),
        "checkpoints": checkpoints,
        "trend_table": trends,
        "recommended_review_candidates": _recommended_candidates(trends),
        "posthoc_review_artifacts": posthoc.get("posthoc_review_artifacts", {}) if isinstance(posthoc, dict) else {},
        "no_optimizer_execution": posthoc.get("no_optimizer_execution") if isinstance(posthoc, dict) else None,
    }
    return result, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / INTERNAL_OBJECT_TO_20000_OUTPUT_NAME)
    parser.add_argument("--operator-plan", type=Path, default=Path("/tmp/d016_to_20000_plan.json"))
    parser.add_argument("--geometry-manifest", type=Path, default=ROOT / "geometry_releases/stage_c_geometry_release_v1.json")
    parser.add_argument("--scene", type=Path, default=ROOT / "data/TiHuBird")
    parser.add_argument("--images", default="images")
    parser.add_argument("--internal-object-manifest", type=Path, default=ROOT / "data/TiHuBird/internal_object_masks_reviewed_v3/manifest.json")
    args = parser.parse_args()
    result, verdict = _audit(args)
    output = args.output.resolve()
    if output.is_dir():
        audit_path = output / "internal_object_townership_to_20000_cpu_audit.json"
        report_path = output / "internal_object_townership_to_20000_cpu_audit.md"
        audit_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
        report_path.write_text(
            "# D-016 internal-object T-ownership 15,500->20,000 CPU audit\n\n"
            f"Verdict: `{verdict}`\n\n"
            "This verdict is an engineering post-run review result only. It does not "
            "select a final checkpoint, authorize joint training, accept Stage D, or "
            "authorize Stage E.\n\n"
            + ("\nErrors:\n" + "".join(f"- {error}\n" for error in result["errors"]) if result["errors"] else "")
            + ("\nHolds:\n" + "".join(f"- {hold}\n" for hold in result["holds"]) if result["holds"] else ""),
            encoding="utf-8",
        )
    print(verdict)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 2 if verdict == VERDICT_BLOCKED else 0


if __name__ == "__main__":
    raise SystemExit(main())
