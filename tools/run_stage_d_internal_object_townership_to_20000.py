#!/usr/bin/env python3
"""Plan or execute the D-016 internal-object T-only continuation to global 20,000."""

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
    INTERNAL_OBJECT_TO_20000_ENDPOINT,
    INTERNAL_OBJECT_TO_20000_NODES,
    INTERNAL_OBJECT_TO_20000_OUTPUT_NAME,
    INTERNAL_OBJECT_TO_20000_SOURCE_SHA256,
)
from utils.internal_object_mask import validate_internal_object_mask_set
from utils.specular_mask import validate_specular_mask_set
from utils.stage_d_static_cache import state_sha256


SOURCE = ROOT / "output/stage_d_tihubird_c03r8_internal_object_townership_pilot_v1/chkpnt15500.pth"
PILOT_OUTPUT = ROOT / "output/stage_d_tihubird_c03r8_internal_object_townership_pilot_v1"
PILOT_AUDIT = PILOT_OUTPUT / "internal_object_townership_cpu_audit.json"
CACHE = PILOT_OUTPUT / "static_dr_cache"
RELEASE = ROOT / "geometry_releases/stage_c_geometry_release_v1.json"
GLASS_MASK = "specular_masks_reviewed_v1/manifest.json"
INTERNAL_MASK = "internal_object_masks_reviewed_v3/manifest.json"
OUTPUT = ROOT / "output" / INTERNAL_OBJECT_TO_20000_OUTPUT_NAME


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_git_for_execute() -> None:
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if branch != "feature/stage-d-transmittance":
        raise RuntimeError(f"wrong branch: {branch}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("worktree is not clean")
    parity = subprocess.check_output(
        ["git", "rev-list", "--left-right", "--count", "HEAD...@{u}"],
        cwd=ROOT, text=True,
    ).strip()
    if parity != "0\t0":
        raise RuntimeError(f"local/remote parity is not 0/0: {parity}")


def _validate_no_conflicting_training_process() -> None:
    result = subprocess.run(
        ["pgrep", "-af", r"(train\.py|run_stage_d_.*--execute)"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    current = str(Path(__file__).name)
    conflicts = [
        line for line in result.stdout.splitlines()
        if current not in line and "pgrep -af" not in line
    ]
    if conflicts:
        raise RuntimeError("conflicting training process exists: " + " | ".join(conflicts))


def _validate_source_checkpoint(path: Path) -> dict:
    if sha256_file(path) != INTERNAL_OBJECT_TO_20000_SOURCE_SHA256:
        raise RuntimeError("source checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("format") != "rtgs_stage_d":
        raise RuntimeError("source checkpoint is not rtgs_stage_d")
    if (
        checkpoint.get("global_iteration"),
        checkpoint.get("reflection_iteration"),
        checkpoint.get("transmittance_iteration"),
    ) != (15500, 12000, 500):
        raise RuntimeError("source checkpoint iteration identity mismatch")
    transmittance = checkpoint.get("transmittance", {})
    if transmittance.get("optimizer") is None:
        raise RuntimeError("BLOCKED_BY_RESUME_CONTRACT: missing T optimizer state")
    if int(transmittance.get("xyz").shape[0]) != 4096:
        raise RuntimeError("source checkpoint T count is not 4096")
    if checkpoint.get("geometry_release_aggregate_sha256") != FORMAL_RELEASE_SHA256:
        raise RuntimeError("source checkpoint Stage-C aggregate mismatch")
    prior_cache = checkpoint.get("config", {}).get("cached_t_warmup", {})
    frozen_after = prior_cache.get("phase_a_frozen_hash_after", {})
    if state_sha256(checkpoint["diffuse"]) != frozen_after.get("diffuse"):
        raise RuntimeError("pilot static cache D identity does not match source D")
    if state_sha256(checkpoint["reflection"]) != frozen_after.get("reflection"):
        raise RuntimeError("pilot static cache R identity does not match source R")
    if Path(prior_cache.get("cache_path", "")).resolve() != CACHE.resolve():
        raise RuntimeError("source checkpoint does not identify the pilot static cache")
    return checkpoint


def _validate_pilot_audit(path: Path) -> dict:
    audit = _json(path)
    if audit.get("verdict") != "D016_PILOT_PASS_AWAITING_USER_REVIEW":
        raise RuntimeError("pilot CPU audit verdict is not PASS")
    if audit.get("technical_run_complete") is not True:
        raise RuntimeError("pilot CPU audit technical_run_complete is not true")
    if audit.get("errors") != []:
        raise RuntimeError("pilot CPU audit errors are not empty")
    return audit


def _validate_static_cache(source_checkpoint: dict) -> dict:
    manifest = _json(CACHE / "manifest.json")
    expected = source_checkpoint.get("config", {}).get("cached_t_warmup", {})
    if manifest.get("aggregate_sha256") != expected.get("cache_aggregate_sha256"):
        raise RuntimeError("pilot static cache aggregate mismatch")
    return manifest


def training_command(args) -> list[str]:
    return [
        sys.executable, str(ROOT / "train.py"),
        "-s", str(ROOT / "data/TiHuBird"),
        "-m", str(OUTPUT),
        "--images", "images",
        "--model_type", "surfel",
        "--stage", "stage_d",
        "--experiment", "TiHuBird C03-r8 Stage D D-016 internal-object T ownership to 20k",
        "--resolution", "8",
        "--normal_priors", "diffrender_priors_candidates/axis_smoke/C03/normal",
        "--normal_prior_space", "camera",
        "--specular_masks", GLASS_MASK,
        "--internal_object_masks", args.internal_object_masks,
        "--geometry_release_manifest", str(RELEASE),
        "--stage_d_static_cache_path", str(CACHE),
        "--stage_d_reuse_static_cache",
        "--transmittance_init_mode", "transferred_d_inside",
        "--transmittance_init_count", "4096",
        "--transmittance_init_seed", "20260703",
        "--transmittance_compose", "alpha_over",
        "--transparent_path_mode", "cuboid_front_v1",
        "--transparent_direct_mode", "off",
        "--transparent_reflection_mode", "off",
        "--cout_ownership_mode", "support_safe_outside",
        "--ray_background", "scene",
        "--ray_chunk_size", "2048",
        "--iterations", str(INTERNAL_OBJECT_TO_20000_ENDPOINT),
        "--start_checkpoint", str(SOURCE),
        "--lambda_spec", "0.2",
        "--specular_k0", "0.9",
        "--lambda_depth", "0.2",
        "--stage_d_depth_start_iteration", "40000",
        "--stage_d_internal_object_to_20000",
        "--stage_d_phase_a_end_iteration", str(INTERNAL_OBJECT_TO_20000_ENDPOINT),
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
        "--lambda_object_positive", str(args.lambda_object_positive),
        "--lambda_object_negative", str(args.lambda_object_negative),
        "--stage_d_cache_parity_atol", "2e-5",
        "--stage_d_cache_parity_mean_atol", "2e-6",
        "--disable_viewer",
        "--quiet",
        "--save_iterations", *[str(node) for node in INTERNAL_OBJECT_TO_20000_NODES],
        "--checkpoint_iterations", *[str(node) for node in INTERNAL_OBJECT_TO_20000_NODES],
    ]


def build_plan(args) -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing existing D-016 to-20000 output: {OUTPUT}")
    release = validate_geometry_release(RELEASE)
    glass = validate_specular_mask_set(ROOT / "data/TiHuBird", "images", GLASS_MASK)
    internal = validate_internal_object_mask_set(
        ROOT / "data/TiHuBird", "images", args.internal_object_masks,
    )
    source_checkpoint = _validate_source_checkpoint(SOURCE)
    audit = _validate_pilot_audit(PILOT_AUDIT)
    cache_manifest = _validate_static_cache(source_checkpoint)
    if release.get("geometry_release_id") != FORMAL_RELEASE_ID:
        raise RuntimeError("Stage-C release ID mismatch")
    if release.get("aggregate_sha256") != FORMAL_RELEASE_SHA256:
        raise RuntimeError("Stage-C release hash mismatch")
    if internal.get("aggregate_sha256") != "c0e49503e5f5c30b1ab26b7cfd79332ac9c486f35656f1425c9cefea516d4052":
        raise RuntimeError("formal internal-object aggregate mismatch")
    command = training_command(args)
    return {
        "schema": "rtgs_stage_d_internal_object_townership_to_20000_operator_plan_v1",
        "execute": bool(args.execute),
        "output": str(OUTPUT),
        "source": str(SOURCE),
        "source_sha256": INTERNAL_OBJECT_TO_20000_SOURCE_SHA256,
        "source_header": {
            "format": source_checkpoint.get("format"),
            "global_iteration": source_checkpoint.get("global_iteration"),
            "reflection_iteration": source_checkpoint.get("reflection_iteration"),
            "transmittance_iteration": source_checkpoint.get("transmittance_iteration"),
            "transmittance_count": int(source_checkpoint["transmittance"]["xyz"].shape[0]),
            "transmittance_optimizer_present": True,
        },
        "pilot_audit": {
            "path": str(PILOT_AUDIT),
            "verdict": audit.get("verdict"),
            "technical_run_complete": audit.get("technical_run_complete"),
            "errors": audit.get("errors"),
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
        "static_cache": {
            "path": str(CACHE),
            "aggregate_sha256": cache_manifest.get("aggregate_sha256"),
            "reused": True,
        },
        "iterations": {
            "start": 15500,
            "first_update": 15501,
            "end_inclusive": INTERNAL_OBJECT_TO_20000_ENDPOINT,
            "updates": 4500,
            "transmittance_local": [501, 5000],
        },
        "nodes": list(INTERNAL_OBJECT_TO_20000_NODES),
        "transmittance": {
            "count": 4096,
            "optimizer_resume": True,
            "reinitialization": False,
            "transferred_d_selection_rerun": False,
            "random_fill_rerun": False,
        },
        "frozen_branches": {"diffuse": True, "reflection": True, "transmittance": False},
        "topology": {"densify": False, "prune": False, "required_t_count": 4096},
        "losses": {
            "object_alpha_floor": float(args.object_alpha_floor),
            "lambda_object_positive": float(args.lambda_object_positive),
            "lambda_object_negative": float(args.lambda_object_negative),
            "depth_enabled_during_run": False,
            "future_depth_activation_global": 40000,
        },
        "command": command,
        "note": "Default mode is plan-only; --execute launches the D-016 T-only continuation.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--internal-object-masks", default=INTERNAL_MASK)
    parser.add_argument("--object-mask-min-views", type=int, default=3)
    parser.add_argument("--object-mask-min-support-ratio", type=float, default=0.6)
    parser.add_argument("--object-mask-boundary-ignore-px", type=int, default=5)
    parser.add_argument("--object-mask-erode-px", type=int, default=3)
    parser.add_argument("--object-mask-dilate-px", type=int, default=3)
    parser.add_argument("--object-alpha-floor", type=float, default=0.35)
    parser.add_argument("--lambda-object-positive", type=float, default=0.05)
    parser.add_argument("--lambda-object-negative", type=float, default=0.05)
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
