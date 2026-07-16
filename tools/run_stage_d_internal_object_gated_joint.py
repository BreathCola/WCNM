#!/usr/bin/env python3
"""Plan or execute the D-016h internal-object gated D/R/T joint pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.geometry_release import validate_geometry_release
from stage_d_training import (
    FORMAL_RELEASE_ID,
    FORMAL_RELEASE_SHA256,
    FORMAL_SOURCE_SHA256,
    INTERNAL_OBJECT_GATED_JOINT_ENDPOINT,
    INTERNAL_OBJECT_GATED_JOINT_LR_OVERRIDES,
    INTERNAL_OBJECT_GATED_JOINT_NODES,
    INTERNAL_OBJECT_GATED_JOINT_OUTPUT_NAME,
    INTERNAL_OBJECT_GATED_JOINT_SOURCE_SHA256,
)
from utils.internal_object_mask import validate_internal_object_mask_set
from utils.specular_mask import validate_specular_mask_set


SOURCE = (
    ROOT
    / "output/stage_d_tihubird_c03r8_internal_object_townership_15500_20000_v1"
    / "chkpnt16500.pth"
)
TO_20000_AUDIT = (
    ROOT
    / "output/stage_d_tihubird_c03r8_internal_object_townership_15500_20000_v1"
    / "internal_object_townership_to_20000_cpu_audit.json"
)
COLOR_RECOVERY_AUDIT = (
    ROOT
    / "output/stage_d_tihubird_c03r8_internal_object_color_recovery_16500_17000_v1"
    / "internal_object_tcolor_recovery_cpu_audit.json"
)
RELEASE = ROOT / "geometry_releases/stage_c_geometry_release_v1.json"
GLASS_MASK = "specular_masks_reviewed_v1/manifest.json"
INTERNAL_MASK = "internal_object_masks_reviewed_v3/manifest.json"
OUTPUT = ROOT / "output" / INTERNAL_OBJECT_GATED_JOINT_OUTPUT_NAME
EXPECTED_INTERNAL_AGGREGATE = (
    "c0e49503e5f5c30b1ab26b7cfd79332ac9c486f35656f1425c9cefea516d4052"
)
PLAN_SCHEMA = "rtgs_stage_d_internal_object_gated_joint_operator_plan_v1"


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_identity() -> dict:
    def out(command: list[str]) -> str:
        return subprocess.check_output(command, cwd=ROOT, text=True).strip()

    identity = {
        "head": out(["git", "rev-parse", "HEAD"]),
        "branch": out(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
    }
    try:
        identity["ahead_behind"] = out([
            "git", "rev-list", "--left-right", "--count", "HEAD...@{u}",
        ])
    except subprocess.CalledProcessError:
        identity["ahead_behind"] = None
    status = out(["git", "status", "--porcelain"])
    identity["worktree_clean"] = status == ""
    identity["dirty_entries"] = status.splitlines()
    return identity


def _validate_git_for_execute() -> None:
    identity = _git_identity()
    if identity["branch"] != "feature/stage-d-transmittance":
        raise RuntimeError(f"wrong branch: {identity['branch']}")
    if identity["ahead_behind"] not in ("0\t0", "0 0"):
        raise RuntimeError(
            f"local/remote parity is not 0/0: {identity['ahead_behind']}"
        )


def _validate_no_conflicting_training_process() -> None:
    result = subprocess.run(
        ["pgrep", "-af", r"(train\.py|run_stage_d_.*--execute)"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    current = Path(__file__).name
    conflicts = [
        line for line in result.stdout.splitlines()
        if current not in line and "pgrep -af" not in line
    ]
    if conflicts:
        raise RuntimeError(
            "conflicting training process exists: " + " | ".join(conflicts)
        )


def _validate_source_checkpoint(path: Path) -> dict:
    if sha256_file(path) != INTERNAL_OBJECT_GATED_JOINT_SOURCE_SHA256:
        raise RuntimeError("source checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("format") != "rtgs_stage_d":
        raise RuntimeError("source checkpoint is not rtgs_stage_d")
    if (
        checkpoint.get("global_iteration"),
        checkpoint.get("reflection_iteration"),
        checkpoint.get("transmittance_iteration"),
    ) != (16500, 12000, 1500):
        raise RuntimeError("source checkpoint iteration identity mismatch")
    if int(checkpoint["transmittance"]["xyz"].shape[0]) != 4096:
        raise RuntimeError("source checkpoint T count is not 4096")
    if checkpoint.get("geometry_release_aggregate_sha256") != FORMAL_RELEASE_SHA256:
        raise RuntimeError("source checkpoint Stage-C aggregate mismatch")
    config = checkpoint.get("config", {})
    if config.get("internal_object_to_20000", {}).get("schema") != (
        "rtgs_stage_d_internal_object_townership_to_20000_v1"
    ):
        raise RuntimeError("source checkpoint is not from D-016 to-20000")
    if config.get("internal_object_ownership", {}).get("schema") != (
        "rtgs_stage_d_internal_object_townership_v1"
    ):
        raise RuntimeError("source checkpoint lacks D-016 ownership provenance")
    return checkpoint


def _validate_to_20000_audit(path: Path) -> dict:
    audit = _json(path)
    if audit.get("verdict") != "D016_TO_20000_PASS_AWAITING_USER_REVIEW":
        raise RuntimeError("to-20000 CPU audit verdict is not PASS")
    if audit.get("technical_run_complete") is not True:
        raise RuntimeError("to-20000 CPU audit technical_run_complete is not true")
    if audit.get("errors") != []:
        raise RuntimeError("to-20000 CPU audit errors are not empty")
    recommended = audit.get("recommended_review_candidates", {})
    if recommended.get("best_rgb") != 16500:
        raise RuntimeError("to-20000 CPU audit best_rgb is not 16500")
    if recommended.get("best_leakage_tradeoff") != 16500:
        raise RuntimeError("to-20000 CPU audit best_leakage_tradeoff is not 16500")
    return audit


def _color_recovery_evidence() -> dict:
    if not COLOR_RECOVERY_AUDIT.is_file():
        return {"available": False, "path": str(COLOR_RECOVERY_AUDIT)}
    audit = _json(COLOR_RECOVERY_AUDIT)
    return {
        "available": True,
        "path": str(COLOR_RECOVERY_AUDIT),
        "verdict": audit.get("verdict"),
        "errors": audit.get("errors", [])[:8],
        "warnings": audit.get("warnings", []),
        "interpretation": (
            "weak_or_blocked_review_evidence_only; not used as source checkpoint"
        ),
    }


def training_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, str(ROOT / "train.py"),
        "-s", str(ROOT / "data/TiHuBird"),
        "-m", str(OUTPUT),
        "--images", "images",
        "--model_type", "surfel",
        "--stage", "stage_d",
        "--experiment", "TiHuBird C03-r8 Stage D D-016h gated internal-object joint",
        "--resolution", "8",
        "--normal_priors", "diffrender_priors_candidates/axis_smoke/C03/normal",
        "--normal_prior_space", "camera",
        "--specular_masks", GLASS_MASK,
        "--internal_object_masks", args.internal_object_masks,
        "--geometry_release_manifest", str(RELEASE),
        "--transmittance_init_mode", "transferred_d_inside",
        "--transmittance_init_count", "4096",
        "--transmittance_init_seed", "20260703",
        "--transmittance_compose", "alpha_over",
        "--transparent_path_mode", "cuboid_front_v1",
        "--transparent_direct_mode", "interface_only",
        "--transparent_reflection_mode", "support_safe_outside",
        "--cout_ownership_mode", "support_safe_outside",
        "--ray_background", "scene",
        "--ray_chunk_size", str(args.ray_chunk_size),
        "--iterations", str(INTERNAL_OBJECT_GATED_JOINT_ENDPOINT),
        "--start_checkpoint", str(SOURCE),
        "--lambda_spec", "0.2",
        "--specular_k0", "0.9",
        "--lambda_depth", "0.2",
        "--stage_d_depth_start_iteration", "40000",
        "--stage_d_internal_object_gated_joint",
        "--stage_d_phase_a_end_iteration", str(INTERNAL_OBJECT_GATED_JOINT_ENDPOINT),
        "--transparent_interface_margin", "0.05",
        "--transparent_interface_margin_mode", "exclude",
        "--transfer_min_views", "3",
        "--transfer_min_total_weight", "0.01",
        "--transfer_depth_margin", "0.05",
        "--object_mask_min_views", str(args.object_mask_min_views),
        "--object_mask_min_support_ratio", str(args.object_mask_min_support_ratio),
        "--object_mask_boundary_ignore_px", str(args.object_mask_boundary_ignore_px),
        "--object_mask_erode_px", str(args.object_mask_erode_px),
        "--object_mask_dilate_px", str(args.object_mask_dilate_px),
        "--object_alpha_floor", str(args.object_alpha_floor),
        "--lambda_object_positive", "0.0",
        "--lambda_object_negative", str(args.lambda_object_negative),
        "--lambda_object_cin_color", str(args.lambda_object_cin_color),
        "--disable_viewer",
        "--quiet",
        "--save_iterations", *[str(node) for node in INTERNAL_OBJECT_GATED_JOINT_NODES],
        "--checkpoint_iterations", *[str(node) for node in INTERNAL_OBJECT_GATED_JOINT_NODES],
    ]


def build_plan(args: argparse.Namespace) -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing D-016h gated-joint output: {OUTPUT}")
    release = validate_geometry_release(RELEASE)
    glass = validate_specular_mask_set(ROOT / "data/TiHuBird", "images", GLASS_MASK)
    internal = validate_internal_object_mask_set(
        ROOT / "data/TiHuBird", "images", args.internal_object_masks,
    )
    source_checkpoint = _validate_source_checkpoint(SOURCE)
    to_20000 = _validate_to_20000_audit(TO_20000_AUDIT)
    if release.get("geometry_release_id") != FORMAL_RELEASE_ID:
        raise RuntimeError("Stage-C release ID mismatch")
    if release.get("aggregate_sha256") != FORMAL_RELEASE_SHA256:
        raise RuntimeError("Stage-C release hash mismatch")
    if internal.get("aggregate_sha256") != EXPECTED_INTERNAL_AGGREGATE:
        raise RuntimeError("formal internal-object aggregate mismatch")
    command = training_command(args)
    return {
        "schema": PLAN_SCHEMA,
        "execute": bool(args.execute),
        "output": str(OUTPUT),
        "source": str(SOURCE),
        "source_sha256": INTERNAL_OBJECT_GATED_JOINT_SOURCE_SHA256,
        "source_header": {
            "format": source_checkpoint.get("format"),
            "global_iteration": source_checkpoint.get("global_iteration"),
            "reflection_iteration": source_checkpoint.get("reflection_iteration"),
            "transmittance_iteration": source_checkpoint.get("transmittance_iteration"),
            "transmittance_count": int(source_checkpoint["transmittance"]["xyz"].shape[0]),
        },
        "source_selection_reason": {
            "best_rgb": to_20000["recommended_review_candidates"]["best_rgb"],
            "best_leakage_tradeoff": to_20000["recommended_review_candidates"]["best_leakage_tradeoff"],
            "color_recovery_was_weak": True,
            "endpoint_20000_is_not_auto_selected": True,
        },
        "release": str(RELEASE),
        "release_id": release.get("geometry_release_id"),
        "release_aggregate_sha256": release.get("aggregate_sha256"),
        "glass_mask_manifest_sha256": glass["manifest_file_sha256"],
        "glass_mask_aggregate_sha256": glass.get("aggregate_sha256"),
        "internal_object_manifest_sha256": internal["manifest_file_sha256"],
        "internal_object_aggregate_sha256": internal.get("aggregate_sha256"),
        "internal_object_role": internal.get("role"),
        "internal_object_human_status": internal.get("human_status"),
        "internal_object_semantics_version": internal.get("internal_object_semantics_version"),
        "color_recovery_evidence": _color_recovery_evidence(),
        "iterations": {
            "start": 16500,
            "first_update": 16501,
            "end_inclusive": INTERNAL_OBJECT_GATED_JOINT_ENDPOINT,
            "updates": 1000,
            "transmittance_local": [1501, 2500],
        },
        "nodes": list(INTERNAL_OBJECT_GATED_JOINT_NODES),
        "transmittance": {
            "count": 4096,
            "optimizer_resume": True,
            "reinitialization": False,
            "transferred_d_selection_rerun": False,
            "random_fill_rerun": False,
            "topology_change": False,
            "densify": False,
            "prune": False,
        },
        "parameter_group_contract": {
            "trainable": {
                branch: sorted(groups)
                for branch, groups in INTERNAL_OBJECT_GATED_JOINT_LR_OVERRIDES.items()
            },
            "frozen": {
                "diffuse": ["xyz", "scaling", "rotation", "exposure"],
                "reflection": ["xyz", "scaling", "rotation"],
                "transmittance": ["xyz", "scaling", "rotation"],
            },
            "lr_overrides": INTERNAL_OBJECT_GATED_JOINT_LR_OVERRIDES,
        },
        "losses": {
            "full_frame_rgb_loss": True,
            "reviewed_v3_object_masks": True,
            "lambda_object_positive": 0.0,
            "lambda_object_negative": float(args.lambda_object_negative),
            "lambda_object_cin_color": float(args.lambda_object_cin_color),
            "mignore_supervision": False,
        },
        "renderer": {
            "exact_uncached": True,
            "transparent_direct_mode": "interface_only",
            "transparent_reflection_mode": "support_safe_outside",
            "cout_ownership_mode": "support_safe_outside",
            "ray_domain_changed": False,
        },
        "git": _git_identity(),
        "command": command,
        "note": "Default mode is plan-only; --execute launches the D-016h gated D/R/T joint pilot.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--internal-object-masks", default=INTERNAL_MASK)
    parser.add_argument("--object-mask-min-views", type=int, default=3)
    parser.add_argument("--object-mask-min-support-ratio", type=float, default=0.6)
    parser.add_argument("--object-mask-boundary-ignore-px", type=int, default=5)
    parser.add_argument("--object-mask-erode-px", type=int, default=3)
    parser.add_argument("--object-mask-dilate-px", type=int, default=3)
    parser.add_argument("--object-alpha-floor", type=float, default=0.35)
    parser.add_argument("--lambda-object-negative", type=float, default=0.05)
    parser.add_argument("--lambda-object-cin-color", type=float, default=0.05)
    parser.add_argument("--ray-chunk-size", type=int, default=512)
    args = parser.parse_args()
    plan = build_plan(args)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    _validate_git_for_execute()
    _validate_no_conflicting_training_process()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.call(plan["command"], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
